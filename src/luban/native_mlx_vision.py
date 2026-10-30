"""Bounded image inference using the Q4 VLM in an exclusive native worker.

PIL validation, the tokenizer and Qwen patch layout are shared with the existing
image route. No original model weights or text-qualified expert are loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import tempfile
import time

import torch
from safetensors.torch import save_file

from .context import available_memory_bytes, output_budget, window
from .native_engine import NativeEngine
from .native_mlx import model_binding
from .product_inference import close, native_binary


def generate_images(runtime, messages, emit, stop, *, thinking, max_tokens, temperature):
    from .runtime import EmitStreamer
    from .vision import image_bytes, image_inputs, image_processor, MAX_IMAGES, MAX_IMAGE_TOTAL_BYTES

    with runtime.lock:
        runtime._ready()
        urls = [part["image_url"]["url"] for message in messages if isinstance(message["content"], list)
                for part in message["content"] if part.get("type") == "image_url"]
        if not 1 <= len(urls) <= MAX_IMAGES:
            raise ValueError("单次对话最多支持 4 张图片")
        if sum(len(image_bytes(url)[0]) for url in urls) > MAX_IMAGE_TOTAL_BYTES:
            raise ValueError("图片总量不能超过 5 MiB")
        if stop.is_set():
            raise RuntimeError("请求已停止")
        started = time.perf_counter()
        effective = "off" if runtime.active_core is not None else thinking
        inputs = image_inputs(runtime.tokenizer, image_processor(runtime.loaded_path), messages, effective)
        pixels = inputs["pixel_values"].detach().cpu().to(torch.float32).contiguous()
        grids = inputs["image_grid_thw"].tolist()
        prompt = inputs["input_ids"][0].tolist()
        binding = model_binding(runtime)
        binary = native_binary(runtime)
        if binding is None or binary is None:
            raise RuntimeError("MLX 图片推理程序不可用")
        core = None
        if runtime.active_core is not None:
            from .evolution import _read_version, _target
            metadata, weight = _read_version(runtime, runtime.active_core)
            active = json.loads((runtime.store / "active-core.json").read_text())
            if active != {"version": runtime.active_core, "identity": runtime.identity, "sha256": metadata["sha256"]}:
                raise ValueError("活动核心指针与正式已加载版本不一致")
            name, _ = _target(runtime)
            core = {"file": str(runtime.store / (runtime.active_core + ".safetensors")),
                    "tensor_key": name, "sha256": metadata["sha256"]}
            del weight
        # One native graph at a time. The next text request reloads its ordinary
        # resident text worker lazily after this bounded image request finishes.
        close(runtime)
        pager = runtime.cold_backbone
        if pager.depth_count or pager.loaded:
            raise RuntimeError("图片推理前须释放正在使用的 HF 核心")
        pager.evict_resident()
        patches = inputs["image_grid_thw"].prod(-1)
        workspace = int((patches * patches).sum()) * runtime.model.config.vision_config.num_heads * 8
        context = window(runtime.settings.snapshot(), runtime.text_config, available_memory_bytes(),
                         device=runtime.device, weight_bytes=binding["weight_bytes"] + workspace, dtype_bytes=2)
        requested = runtime.settings.snapshot().get("max_new_tokens", 0) if max_tokens is None else max_tokens
        maximum = min(2048, output_budget(len(prompt), requested, context))
        if len(prompt) + maximum > 32768:
            raise ValueError("图片上下文超过当前原生模型的 32768 词元限制")
        manifest = Path(runtime.loaded_path) / "upstream-file-manifest.json"
        engine = NativeEngine(binary, runtime.loaded_path, hashlib.sha256(manifest.read_bytes()).hexdigest(),
                              core=core, precision="mlx-q4", mlx_model=binding, vision=True)
        streamer = EmitStreamer(runtime.tokenizer, emit) if emit is not None else None
        previous_state = runtime.state
        runtime.state = "generating"
        runtime.native_engine = engine
        eos = runtime.model.generation_config.eos_token_id
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        eos_ids = list(dict.fromkeys([*eos_ids, 248044, 248046]))
        try:
            with tempfile.TemporaryDirectory(prefix="luban-mlx-image-") as directory:
                path = Path(directory) / "pixels.safetensors"
                save_file({"pixel_values": pixels}, str(path))
                path.chmod(0o600)
                if streamer is not None:
                    streamer.put(inputs["input_ids"])
                result = engine.generate_image(prompt, max_tokens=maximum, eos_ids=list(eos_ids),
                    temperature=temperature, stop=stop,
                    seed=secrets.randbits(64) if temperature > 0 else 42,
                    on_token=(lambda token: streamer.put(torch.tensor([token]))) if streamer else (lambda _: None),
                    pixel_file=str(path), image_grid_thw=grids)
            count = len(result["tokens"])
            complete = bool(count and result["tokens"][-1] in eos_ids and result["finish_reason"] == "stop" and not stop.is_set())
            elapsed = time.perf_counter() - started
            decode_ns = result.get("decode_ns")
            rate = (count - 1) * 1e9 / decode_ns if count > 1 and isinstance(decode_ns, int) and decode_ns > 0 else None
            usage = {"prompt_tokens": len(prompt), "completion_tokens": count, "total_tokens": len(prompt) + count,
                     "elapsed_seconds": elapsed, "tokens_per_second": rate, "output_limit": maximum,
                     "local_action": "local_core", "local_complete": complete, "reference_used": False,
                     "core_version": runtime.active_core, "expert_id": None,
                     "requested_thinking": thinking, "effective_thinking": effective,
                     "image_count": len(grids), "inference_engine": "mlx-q4-vision",
                     "finish_reason": "stop" if stop.is_set() or result["finish_reason"] == "cancelled" or complete else "length"}
            runtime.last_generation_usage = usage
            return usage
        finally:
            try:
                if streamer is not None:
                    streamer.end()
            finally:
                close(runtime)
                runtime.state = previous_state
