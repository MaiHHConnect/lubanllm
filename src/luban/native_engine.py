"""Verified, persistent JSON-lines bridge to the native Metal inference worker."""
from __future__ import annotations

from collections import deque
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import select
import subprocess
import sys
import threading
import time


class NativeEngineError(RuntimeError):
    pass


def _sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _token_ids(value, *, empty=False):
    return (isinstance(value, list) and (empty or bool(value))
            and all(type(token) is int and 0 <= token < 2**32 for token in value))


class NativeEngine:
    def __init__(self, binary, model_path, manifest_sha256, core=None, *, precision="bf16", calibration_profile=None,
                 expert=None,
                 direct_q4=None, mlx_model=None, vision=False,
                 skip_layer_mask=0, exit_after=24, layer_paging_path=None,
                 load_timeout=180.0, request_timeout=180.0, cancel_timeout=5.0):
        self.binary = Path(binary).absolute()
        self.model_path = Path(model_path).absolute()
        self.manifest_sha256 = manifest_sha256
        self.core = dict(core) if core is not None else None
        self.expert = dict(expert) if expert is not None else None
        self.direct_q4 = dict(direct_q4) if direct_q4 is not None else None
        self.mlx_model = dict(mlx_model) if mlx_model is not None else None
        if precision == "mlx-q4":
            if (self.mlx_model is None or not isinstance(self.mlx_model.get("path"), (str, Path))
                    or not Path(self.mlx_model["path"]).is_absolute()
                    or not _digest(self.mlx_model.get("sha256"))
                    or type(self.mlx_model.get("weight_bytes")) is not int or self.mlx_model["weight_bytes"] <= 0
                    or skip_layer_mask != 0 or exit_after != 24 or layer_paging_path is not None):
                raise ValueError("MLX requires a bound Q4 model and full-layer execution")
        elif self.mlx_model is not None:
            raise ValueError("An MLX model binding is only valid for MLX precision")
        if self.direct_q4 is not None:
            if (precision != "mixed-q4-g16" or skip_layer_mask != 0 or exit_after != 24
                    or layer_paging_path is not None
                    or not isinstance(self.direct_q4.get("file"), (str, Path))
                    or not Path(self.direct_q4["file"]).is_absolute()
                    or not _digest(self.direct_q4.get("sha256"))
                    or not _digest(self.direct_q4.get("quantizer_sha256"))
                    or type(self.direct_q4.get("weight_bytes")) is not int
                    or self.direct_q4["weight_bytes"] <= 0):
                raise ValueError("Invalid approved direct Q4 binding")
        if self.expert is not None:
            if (self.core is None or precision not in {"mixed-q4-g16", "mlx-q4"} or skip_layer_mask != 0
                    or exit_after != 24 or layer_paging_path is not None):
                raise ValueError("Native expert requires an admitted core and ordinary full-layer Q4")
            if (not isinstance(self.expert.get("file"), (str, Path))
                    or not Path(self.expert["file"]).is_absolute()
                    or not _digest(self.expert.get("sha256"))
                    or type(self.expert.get("width")) is not int or not 1 <= self.expert["width"] <= 6144
                    or self.expert.get("core_sha256") != self.core.get("sha256")):
                raise ValueError("Invalid approved native expert binding")
        from .settings import native_execution_policy
        self.skip_layer_mask, self.exit_after, _ = native_execution_policy(skip_layer_mask, exit_after, layer_paging_path is not None)
        if self.core is not None and (exit_after < 24 or skip_layer_mask & (1 << 23)):
            raise ValueError("Active native core requires execution of layer 23")
        if layer_paging_path is not None:
            if (not isinstance(layer_paging_path, (str, Path)) or not Path(layer_paging_path).is_absolute()
                    or os.path.lexists(layer_paging_path)):
                raise ValueError("Native layer paging requires an absolute new file path")
            self.layer_paging_path = Path(layer_paging_path)
        else:
            self.layer_paging_path = None
        if precision not in {"mlx-q4", "bf16", "mixed-q4", "mixed-q4-g16", "mixed-q4-g16-packed", "mixed-q4-g32-middle", "mixed-q4-g32-clip97", "mixed-q4-g32-calibrated", "mixed-q4-g32", "mixed-q4-g32-skip0-early8", "mixed-q4-g32-skip0", "mixed-q4-g32-early12", "mixed-q4-mlp", "mixed-q4-gateup", "mixed-q4-down", "mixed-q4-first1", "mixed-q4-first2", "mixed-q4-first4", "mixed-q4-first6", "mixed-q4-early", "mixed-q4-early8", "mixed-q4-early12", "mixed-q4-calibrated", "mixed-q8"}:
            raise ValueError("Invalid native inference precision")
        if precision in {"mixed-q4-calibrated", "mixed-q4-g32-calibrated", "mixed-q4-g16-packed"}:
            if calibration_profile is None or not Path(calibration_profile).is_absolute() or not Path(calibration_profile).is_file():
                raise ValueError("Calibrated or packed Q4 requires an absolute profile file")
            self.calibration_profile = Path(calibration_profile).absolute()
        elif calibration_profile is not None:
            raise ValueError("A calibration profile is only valid for calibrated or packed Q4")
        else:
            self.calibration_profile = None
        if type(vision) is not bool or (vision and (precision != "mlx-q4" or expert is not None)):
            raise ValueError("Vision requires MLX precision without an expert")
        self.vision = vision
        self.precision = precision
        for timeout in (load_timeout, request_timeout, cancel_timeout):
            if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Native timeouts must be finite positive seconds")
        self.load_timeout, self.request_timeout, self.cancel_timeout = load_timeout, request_timeout, cancel_timeout
        self.binary_sha256 = None
        self.receipt = None
        self.loaded = None
        self.process = None
        self._cancel_fd = None
        self._cancel_read_fd = None
        self._serial = threading.RLock()
        self._lifecycle = threading.RLock()
        self._events = queue.Queue(maxsize=256)
        self._stderr = deque(maxlen=32)
        self._readers_stop = threading.Event()
        self._readers = []
        self._request_id = 0
        self._prefix_tokens = []
        self._prefix_bytes = 0
        self._prefix_step = None

    def _verify(self):
        if os.name != "posix":
            raise NativeEngineError("The native inference worker requires POSIX file descriptors")
        if not _digest(self.manifest_sha256):
            raise NativeEngineError("Invalid approved native manifest digest")
        if _sha(self.model_path / "upstream-file-manifest.json") != self.manifest_sha256:
            raise NativeEngineError("Approved native model manifest changed")
        receipt = json.loads(Path(str(self.binary) + ".json").read_bytes())
        source = receipt.get("source_sha256")
        valid_source = _digest(source) or (isinstance(source, dict) and bool(source)
                                           and all(isinstance(k, str) and _digest(v) for k, v in source.items()))
        if not valid_source or not _digest(receipt.get("binary_sha256")):
            raise NativeEngineError("Invalid native binary build receipt")
        if receipt.get("official_manifest_sha256", self.manifest_sha256) != self.manifest_sha256:
            raise NativeEngineError("Native binary receipt model binding mismatch")
        actual = _sha(self.binary)
        if actual != receipt["binary_sha256"]:
            raise NativeEngineError("Native binary does not match its build receipt")
        if self.core is not None:
            if (not self.core.get("file") or not isinstance(self.core.get("tensor_key"), str)
                    or not self.core["tensor_key"] or not _digest(self.core.get("sha256"))):
                raise NativeEngineError("Invalid approved native core binding")
            if _sha(self.core["file"]) != self.core["sha256"]:
                raise NativeEngineError("Approved native core file changed")
        if self.expert is not None and _sha(self.expert["file"]) != self.expert["sha256"]:
            raise NativeEngineError("Approved native expert file changed")
        self.binary_sha256, self.receipt = actual, receipt

    @staticmethod
    def _read_stdout(stream, events, stopped):
        def publish(value):
            while not stopped.is_set():
                try:
                    events.put(value, timeout=0.05)
                    return
                except queue.Full:
                    pass

        try:
            while not stopped.is_set():
                line = stream.readline((4 << 20) + 1)
                if not line:
                    publish(NativeEngineError("Native worker stdout closed"))
                    return
                if len(line) > 4 << 20 or not line.endswith(b"\n"):
                    raise NativeEngineError("Invalid native worker event framing")
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("event"), str):
                    raise NativeEngineError("Invalid native worker event")
                publish(value)
        except Exception as exc:
            publish(NativeEngineError(f"Native worker protocol error: {exc}"))

    @staticmethod
    def _read_stderr(stream, chunks, stopped):
        try:
            while not stopped.is_set():
                chunk = stream.read(4096)
                if not chunk:
                    return
                chunks.append(chunk)
        except (OSError, ValueError):
            pass

    def _next_event(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise NativeEngineError("Native worker response timed out")
        try:
            event = self._events.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            process = self.process
            if process is None or process.poll() is not None:
                # Reader may still have the final event in flight; allow its queue once.
                try:
                    event = self._events.get(timeout=min(remaining, 0.05))
                except queue.Empty:
                    raise NativeEngineError("Native worker exited without a complete response") from None
            else:
                return None
        if isinstance(event, Exception):
            raise event
        if event["event"] == "error":
            raise NativeEngineError(f"Native worker error: {event.get('error', event.get('message', 'unknown'))}")
        return event

    def start(self):
        with self._serial:
            if self.process is not None:
                if self.process.poll() is not None:
                    self._fail("Native worker exited")
                return self.loaded
            try:
                self._verify()
                with self._lifecycle:
                    read_fd, write_fd = os.pipe()
                    os.set_blocking(read_fd, False)
                    os.set_blocking(write_fd, False)
                    command = [str(self.binary), str(self.model_path), self.manifest_sha256, str(read_fd)]
                    if self.core is not None:
                        command += [str(self.core["file"]), self.core["tensor_key"], self.core["sha256"]]
                    if self.precision != "bf16":
                        command.append("--" + self.precision + ("=" + str(self.calibration_profile) if self.calibration_profile is not None else ""))
                    if self.mlx_model is not None:
                        command += ["--mlx-model=" + str(self.mlx_model["path"]),
                                    "--mlx-model-sha256=" + self.mlx_model["sha256"]]
                    if self.vision:
                        command.append("--vision")
                    if self.direct_q4 is not None:
                        command += ["--direct-q4=" + str(self.direct_q4["file"]),
                                    "--direct-q4-sha256=" + self.direct_q4["sha256"],
                                    "--quantizer-sha256=" + self.direct_q4["quantizer_sha256"]]
                    if self.expert is not None:
                        command += ["--expert-file=" + str(self.expert["file"]), "--expert-sha=" + self.expert["sha256"]]
                    if self.skip_layer_mask:
                        command.append(f"--skip-layer-mask={self.skip_layer_mask}")
                    if self.exit_after != 24:
                        command.append(f"--exit-after={self.exit_after}")
                    if self.layer_paging_path is not None:
                        command.append("--layer-paging=" + str(self.layer_paging_path))
                    try:
                        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                   stderr=subprocess.PIPE, pass_fds=(read_fd,), bufsize=0)
                    except BaseException:
                        os.close(write_fd)
                        os.close(read_fd)
                        raise
                    self.process, self._cancel_fd = process, write_fd
                    self._cancel_read_fd = read_fd
                    os.set_blocking(process.stdin.fileno(), False)
                    self._events = queue.Queue(maxsize=256)
                    self._stderr = deque(maxlen=32)
                    self._readers_stop = threading.Event()
                    self._readers = [
                        threading.Thread(target=self._read_stdout,
                                         args=(process.stdout, self._events, self._readers_stop), daemon=True),
                        threading.Thread(target=self._read_stderr,
                                         args=(process.stderr, self._stderr, self._readers_stop), daemon=True),
                    ]
                    for reader in self._readers:
                        reader.start()
                deadline = time.monotonic() + self.load_timeout
                while (event := self._next_event(deadline)) is None:
                    pass
                if event["event"] != "loaded":
                    raise NativeEngineError("Native worker did not acknowledge model loading")
                if event.get("protocol") != "luban-worker-jsonl-v1":
                    raise NativeEngineError("Unsupported native worker protocol")
                if event.get("manifest_sha256") != self.manifest_sha256:
                    raise NativeEngineError("Native worker loaded a different model manifest")
                if event.get("core_loaded") is not (self.core is not None):
                    raise NativeEngineError("Native worker core loading state mismatch")
                if self.core is not None and event.get("core_file_sha256") != self.core["sha256"]:
                    raise NativeEngineError("Native worker loaded a different core file")
                if event.get("expert_loaded", False) is not (self.expert is not None):
                    raise NativeEngineError("Native worker expert loading state mismatch")
                if self.expert is not None and (event.get("expert_file_sha256") != self.expert["sha256"]
                        or type(event.get("expert_width")) is not int or event["expert_width"] != self.expert["width"]
                        or type(event.get("expert_layer")) is not int or event["expert_layer"] != 23):
                    raise NativeEngineError("Native worker loaded a different expert")
                if event.get("inference_precision", "bf16") != self.precision:
                    raise NativeEngineError("Native worker loaded a different inference precision")
                if self.mlx_model is not None and any(event.get(key) != value for key, value in (
                        ("mlx_model_path", str(self.mlx_model["path"])),
                        ("mlx_model_sha256", self.mlx_model["sha256"]),
                        ("mlx_model_weight_bytes", self.mlx_model["weight_bytes"]))):
                    raise NativeEngineError("Native worker loaded a different MLX model")
                if event.get("vision_loaded", False) is not self.vision:
                    raise NativeEngineError("Native worker vision loading state mismatch")
                if event.get("direct_q4_loaded", False) is not (self.direct_q4 is not None):
                    raise NativeEngineError("Native worker direct Q4 loading state mismatch")
                if self.direct_q4 is not None and any(event.get(key) != value for key, value in (
                        ("direct_q4_sha256", self.direct_q4["sha256"]),
                        ("quantizer_sha256", self.direct_q4["quantizer_sha256"]),
                        ("direct_q4_weight_bytes", self.direct_q4["weight_bytes"]))):
                    raise NativeEngineError("Native worker loaded a different direct Q4 carrier")
                for key, expected, default in (("execution_skip_mask", self.skip_layer_mask, 0),
                                                ("execution_exit_after", self.exit_after, 24),
                                                ("layer_paging_enabled", self.layer_paging_path is not None, False)):
                    value = event.get(key, default)
                    if type(value) is not type(expected) or value != expected:
                        raise NativeEngineError("Native worker loaded a different execution policy: " + key)
                self.loaded = event
                if self.precision == "mlx-q4" and event.get("mlx_cache_reuse") is True:
                    # Loading leaves temporary allocator buffers even without a
                    # retained prefix. Release them before runtime admission.
                    self.prepare_prompt_cache([], cache_reuse=False, force=True)
                return event
            except Exception as exc:
                self._fail(str(exc))

    def prepare_direct_q4(self, output, quantizer_sha256, stop):
        """Run the native CPU converter once; cancellation uses its existing FD."""
        output = Path(output)
        if not output.is_absolute() or output.exists() or output.is_symlink() or not _digest(quantizer_sha256):
            raise ValueError("Direct Q4 preparation requires a new absolute carrier path")
        if stop.is_set():
            raise NativeEngineError("生成已取消")
        try:
            self._verify()
            read_fd, write_fd = os.pipe()
            self._cancel_read_fd, self._cancel_fd = read_fd, write_fd
            os.set_blocking(read_fd, False)
            os.set_blocking(write_fd, False)
            command = [str(self.binary), "--prepare-direct-q4", str(self.model_path), self.manifest_sha256,
                       str(output), str(read_fd), "--quantizer-sha256=" + quantizer_sha256]
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, pass_fds=(read_fd,))
            deadline = time.monotonic() + self.load_timeout
            cancelled_at = None
            while True:
                now = time.monotonic()
                if stop.is_set() and cancelled_at is None:
                    os.write(write_fd, b"x")
                    cancelled_at = now
                if now >= deadline or cancelled_at is not None and now - cancelled_at >= self.cancel_timeout:
                    raise NativeEngineError("生成已取消" if cancelled_at is not None else "Direct Q4 preparation timed out")
                try:
                    stdout, stderr = self.process.communicate(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if cancelled_at is not None or stop.is_set():
                raise NativeEngineError("生成已取消")
            if self.process.returncode != 0:
                raise NativeEngineError("Direct Q4 preparation failed: " + stderr[-8192:].decode(errors="replace"))
            event = json.loads(stdout)
            if (event.get("event") != "prepared" or event.get("format") != "luban-direct-q4-g16-v1"
                    or event.get("manifest_sha256") != self.manifest_sha256
                    or event.get("quantizer_sha256") != quantizer_sha256
                    or not _digest(event.get("direct_q4_sha256"))
                    or type(event.get("direct_q4_weight_bytes")) is not int
                    or event["direct_q4_weight_bytes"] <= 0
                    or not output.is_file() or output.is_symlink()):
                raise NativeEngineError("Native converter did not approve the direct Q4 carrier")
            return event
        finally:
            self.close()

    def _submit(self, request, deadline, stop):
        # A stop can race with a terminal event already in the stdout queue.
        # Drain only between requests, before new cancellation can be sent.
        while True:
            try:
                if not os.read(self._cancel_read_fd, 4096):
                    break
            except BlockingIOError:
                break
        payload = memoryview((json.dumps(request, allow_nan=False) + "\n").encode())
        while payload:
            if stop.is_set():
                raise NativeEngineError("Native request cancelled during submission")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeEngineError("Native worker request submission timed out")
            fd = self.process.stdin.fileno()
            if select.select([], [fd], [], min(remaining, 0.05))[1]:
                try:
                    payload = payload[os.write(fd, payload):]
                except BlockingIOError:
                    pass

    def prepare_prompt_cache(self, tokens, *, cache_reuse, force=False):
        """Evict incompatible retained state before the runtime's memory admission."""
        with self._serial:
            if not self._prefix_tokens and not force:
                return
            if not force and cache_reuse and len(tokens) > len(self._prefix_tokens) and tokens[:len(self._prefix_tokens)] == self._prefix_tokens:
                return
            self._request_id += 1
            request = {"id": str(self._request_id), "op": "clear_cache"}
            deadline = time.monotonic() + self.request_timeout
            try:
                self._submit(request, deadline, threading.Event())
                while (event := self._next_event(deadline)) is None:
                    pass
                if event.get("id") != request["id"] or event.get("event") != "cache_cleared":
                    raise NativeEngineError("Native worker cache eviction acknowledgement mismatch")
                self._prefix_tokens = []
                self._prefix_bytes = 0
                self._prefix_step = None
            except Exception as exc:
                self._fail(str(exc))

    def prompt_cache(self, tokens, *, cache_reuse):
        with self._serial:
            if (cache_reuse and self._prefix_tokens and len(tokens) > len(self._prefix_tokens)
                    and tokens[:len(self._prefix_tokens)] == self._prefix_tokens):
                return {"tokens": len(self._prefix_tokens), "bytes": self._prefix_bytes,
                        "prefill_step": self._prefix_step}
        return None

    def generate(self, tokens, max_tokens, eos_ids, temperature, on_token, stop, *, seed=42, cache_reuse=False,
                 pixel_file=None, image_grid_thw=None, context_limit=32768, prefill_step=512):
        if (not _token_ids(tokens) or not _token_ids(eos_ids) or len(eos_ids) > 64
                or type(max_tokens) is not int or not 1 <= max_tokens <= 2048
                or type(context_limit) is not int or not 1 <= context_limit <= (
                    262144 if self.precision == "mlx-q4" and not self.vision else 32768)
                or len(tokens) + max_tokens > context_limit
                or type(prefill_step) is not int or prefill_step not in (32, 64, 128, 256, 512)
                or prefill_step != 512 and (self.precision != "mlx-q4" or self.vision)
                or type(seed) is not int or not 0 <= seed < 2**64
                or type(cache_reuse) is not bool
                or type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2
                or not callable(on_token)):
            raise ValueError("Invalid native generation request")
        if self.vision:
            if (not isinstance(pixel_file, (str, Path)) or not Path(pixel_file).is_absolute()
                    or not Path(pixel_file).is_file() or not isinstance(image_grid_thw, list)
                    or not 1 <= len(image_grid_thw) <= 4
                    or any(not isinstance(grid, list) or len(grid) != 3
                           or any(type(value) is not int or not 1 <= value <= 32768 for value in grid)
                           or math.prod(grid) > 32768 for grid in image_grid_thw)):
                raise ValueError("Invalid MLX image inputs")
        elif pixel_file is not None or image_grid_thw is not None:
            raise ValueError("Image inputs require a vision worker")
        with self._serial:
            if stop.is_set():
                return {"event": "result", "tokens": [], "finish_reason": "cancelled"}
            self.start()
            self._request_id += 1
            request = {"id": str(self._request_id), "tokens": tokens, "max_tokens": max_tokens,
                       "eos_ids": eos_ids, "temperature": temperature, "seed": seed, "cache_reuse": cache_reuse}
            if self.precision == "mlx-q4":
                request["context_limit"] = context_limit
                if not self.vision:
                    request["prefill_step"] = prefill_step
            if self.vision:
                request.update(op="image_generate", pixel_file=str(pixel_file), image_grid_thw=image_grid_thw)
            emitted = []
            cancelled = False
            deadline = time.monotonic() + self.request_timeout * max(1, len(tokens) / 32768)
            try:
                self._submit(request, deadline, stop)
                while True:
                    if stop.is_set() and not cancelled:
                        os.write(self._cancel_fd, b"\x01")
                        cancelled = True
                        deadline = min(deadline, time.monotonic() + self.cancel_timeout)
                    event = self._next_event(deadline)
                    if event is None:
                        continue
                    if event.get("id") != request["id"]:
                        raise NativeEngineError("Native worker response request identity mismatch")
                    if event["event"] == "token":
                        token = event.get("token")
                        if not _token_ids([token]) or len(emitted) >= max_tokens:
                            raise NativeEngineError("Invalid native worker output token")
                        emitted.append(token)
                        on_token(token)
                    elif event["event"] == "result":
                        if not _token_ids(event.get("tokens"), empty=True) or event["tokens"] != emitted:
                            raise NativeEngineError("Native worker result differs from streamed tokens")
                        if event.get("finish_reason") not in {"stop", "length", "cancelled"}:
                            raise NativeEngineError("Invalid native worker completion reason")
                        if "generated_tokens" in event and event["generated_tokens"] != len(emitted):
                            raise NativeEngineError("Native worker output token count mismatch")
                        for key in ("prefill_ns", "ttft_ns", "decode_ns"):
                            value = event.get(key)
                            if value is not None and (type(value) is not int or value < 0):
                                raise NativeEngineError("Invalid native worker timing")
                        cached = event.get("cached_prompt_tokens", 0)
                        if type(cached) is not int or not 0 <= cached <= len(tokens) or cached and not cache_reuse:
                            raise NativeEngineError("Invalid native worker cached prompt count")
                        if self.precision == "mlx-q4" and not self.vision and self.loaded.get("mlx_cache_reuse") is True:
                            capacity = event.get("cache_capacity", 0)
                            if type(capacity) is not int or not 0 <= capacity < len(tokens) or capacity and not cache_reuse:
                                raise NativeEngineError("Invalid native worker retained prefix count")
                            self._prefix_tokens = tokens[:capacity]
                            size = event.get("prefix_cache_bytes", 0)
                            if type(size) is not int or size < 0 or size and not capacity:
                                raise NativeEngineError("Invalid native worker retained prefix bytes")
                            self._prefix_bytes = size
                            self._prefix_step = prefill_step if capacity else None
                        return {**event, "tokens": emitted}
                    else:
                        raise NativeEngineError("Unexpected native worker response event")
            except Exception as exc:
                self._fail(str(exc))

    def route_features(self, tokens, stop):
        """Return the resident Q4 backbone's mean-pooled, pre-norm layer-4 features."""
        if (self.precision != "mlx-q4" or self.vision or not _token_ids(tokens)
                or len(tokens) > 32768 or any(token >= 248320 for token in tokens)):
            raise ValueError("Invalid MLX routing request")
        with self._serial:
            if stop.is_set():
                raise NativeEngineError("Routing cancelled")
            self.start()
            self._request_id += 1
            request = {"id": str(self._request_id), "op": "route_features", "tokens": tokens, "layers": 4}
            deadline = time.monotonic() + self.request_timeout
            try:
                self._submit(request, deadline, stop)
                while True:
                    # Prefix computation has no decoder cancellation loop. Terminate
                    # this worker on cancellation so no stale response reaches reuse.
                    if stop.is_set():
                        raise NativeEngineError("Routing cancelled")
                    event = self._next_event(deadline)
                    if event is None:
                        continue
                    features = event.get("features")
                    if (event.get("id") != request["id"] or event.get("event") != "route_features"
                            or event.get("source") != "mlx-q4-prefix" or event.get("layers") != 4
                            or event.get("legacy_bf16_loaded") is not False
                            or event.get("feature_dim") != 2048
                            or not isinstance(features, list) or len(features) != 2048
                            or any(type(value) not in (int, float) or not math.isfinite(value) for value in features)):
                        raise NativeEngineError("Invalid MLX routing feature response")
                    return features
            except Exception as exc:
                self._fail(str(exc))

    def generate_image(self, tokens, max_tokens, eos_ids, temperature, on_token, stop, *,
                       pixel_file, image_grid_thw, seed=42):
        if not self.vision:
            raise ValueError("Image generation requires a vision worker")
        return self.generate(tokens, max_tokens, eos_ids, temperature, on_token, stop,
                             seed=seed, pixel_file=pixel_file, image_grid_thw=image_grid_thw)

    def _fail(self, message):
        self.close()
        stderr = b"".join(tuple(self._stderr))[-8192:].decode("utf-8", errors="replace").strip()
        raise NativeEngineError(message + (f"; stderr: {stderr}" if stderr else ""))

    def status(self):
        process = self.process
        result = {"pid": process.pid if process else None,
                  "running": process is not None and process.poll() is None,
                  "binary_sha256": self.binary_sha256,
                  "source_sha256": self.receipt.get("source_sha256") if self.receipt else None,
                  "manifest_sha256": self.manifest_sha256}
        if self.loaded is not None:
            result["weight_load_mode"] = ("direct-mlx-q4" if self.mlx_model is not None else
                                          "direct-q4" if self.direct_q4 is not None else "bf16-convert")
            if self.mlx_model is not None:
                result.update({key: self.loaded.get(key) for key in
                               ("mlx_model_sha256", "mlx_language_weight_bytes", "core_loaded",
                                "expert_loaded", "expert_file_sha256", "expert_width")})
        if result["running"] and sys.platform == "darwin":
            # rusage_info_v4: UUID + 35 uint64 fields, same API used by the native probe.
            class Usage(ctypes.Structure):
                _fields_ = [("uuid", ctypes.c_ubyte * 16), ("values", ctypes.c_uint64 * 35)]

            try:
                lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
                lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                lib.proc_pid_rusage.restype = ctypes.c_int
                usage = Usage()
                if lib.proc_pid_rusage(process.pid, 4, ctypes.byref(usage)) == 0:
                    result.update(physical_footprint_bytes=usage.values[7],
                                  lifetime_peak_physical_footprint_bytes=usage.values[28],
                                  memory_method="Darwin proc_pid_rusage RUSAGE_INFO_V4, child process")
            except OSError:
                pass
        return result

    def close(self):
        with self._lifecycle:
            process, self.process = self.process, None
            self.loaded = None
            self._prefix_tokens = []
            self._prefix_bytes = 0
            self._prefix_step = None
            self._readers_stop.set()
            if self._cancel_fd is not None:
                os.close(self._cancel_fd)
                self._cancel_fd = None
            if self._cancel_read_fd is not None:
                os.close(self._cancel_read_fd)
                self._cancel_read_fd = None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                for reader in self._readers:
                    reader.join(timeout=0.2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            self._readers = []
