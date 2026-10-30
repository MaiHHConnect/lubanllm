"""Context admission estimates; a model's advertised window is not allocated RAM."""
from __future__ import annotations

import ctypes
from functools import cache
import os
import sys

import psutil

PRESETS = (65536, 131072, 262144)
GIB = 1024 ** 3
MAX_OUTPUT_TOKENS = 32768


class ContextAdmissionError(ValueError):
    pass


class _VMStatistics64(ctypes.Structure):
    # mach/vm_statistics.h, revisions 1 and 2. Counters are not byte sizes.
    _fields_ = [
        ("free", ctypes.c_uint32), ("active", ctypes.c_uint32),
        ("inactive", ctypes.c_uint32), ("wired", ctypes.c_uint32),
        ("lifetime", ctypes.c_uint64 * 9),
        ("purgeable", ctypes.c_uint32), ("speculative", ctypes.c_uint32),
        ("compression", ctypes.c_uint64 * 4),
        ("compressor", ctypes.c_uint32), ("throttled", ctypes.c_uint32),
        ("external", ctypes.c_uint32), ("internal", ctypes.c_uint32),
        ("uncompressed", ctypes.c_uint64), ("swapped", ctypes.c_uint64),
    ]


@cache
def _mach_memory_api():
    api = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    api.mach_host_self.argtypes, api.mach_host_self.restype = [], ctypes.c_uint32
    api.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int,
                                     ctypes.POINTER(_VMStatistics64), ctypes.POINTER(ctypes.c_uint32)]
    api.host_statistics64.restype = ctypes.c_int
    api.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    api.mach_port_deallocate.restype = ctypes.c_int
    return api, ctypes.c_uint32.in_dll(api, "mach_task_self_").value


def _darwin_memory_bytes():
    api, task = _mach_memory_api()
    host = api.mach_host_self()
    if not host:
        raise OSError("Unable to read host memory")
    try:
        stats = _VMStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int))
        if api.host_statistics64(host, 4, ctypes.byref(stats), ctypes.byref(count)) or not 38 <= count.value <= 40:
            raise OSError("Invalid host memory statistics")
        page = os.sysconf("SC_PAGE_SIZE")
        if page <= 0 or stats.speculative > stats.free:
            raise OSError("Invalid host memory page counts")
        # Mach free already includes speculative pages. Bound the additional
        # inactive allowance by file-backed pages not already counted there.
        # Do not add purgeable separately: it can overlap inactive pages.
        # Global counters cannot identify exactly which file pages are clean.
        return (stats.free + min(stats.inactive, max(0, stats.external - stats.speculative))) * page
    finally:
        if api.mach_port_deallocate(task, host):
            raise OSError("Unable to release host memory port")


def available_memory_bytes():
    """Fresh admission estimate including OS-reported free swap.

    ``virtual_memory().available`` is the platform's current reclaimable RAM
    estimate. On macOS the Mach value remains a diagnostic lower bound, but it
    must not discard the larger OS estimate or available swap. This is still an
    admission estimate, not an OOM or latency guarantee.
    """
    memory = psutil.virtual_memory()
    available = max(0, int(memory.available))
    if sys.platform != "darwin":
        pass
    else:
        try:
            # Keep the OS estimate when it includes more reclaimable inactive
            # pages; use Mach telemetry only when it is the larger reading.
            available = max(available, int(_darwin_memory_bytes()))
        except (OSError, ValueError, AttributeError):
            pass
    try:
        swap_free = max(0, int(getattr(psutil.swap_memory(), "free", 0)))
    except (OSError, ValueError, AttributeError):
        swap_free = 0
    return available + swap_free


def window(settings, text_config, available_bytes, *, device, weight_bytes=0, dtype_bytes=2,
           prefill_step=None, shared_kv_reservation=False):
    mode = settings.get("context_mode", "manual")
    requested = settings["context_tokens"]
    if text_config is None:
        return {"mode": mode, "configured_limit": requested, "model_limit": None,
                "current_limit": None, "estimated": True}
    maximum = int(getattr(text_config, "max_position_embeddings", requested))
    ceiling = min(requested, maximum)
    layers = getattr(text_config, "layer_types", None)
    full = layers.count("full_attention") if layers else getattr(text_config, "num_hidden_layers", 1)
    hidden = int(getattr(text_config, "hidden_size", 2048))
    heads = int(getattr(text_config, "num_attention_heads", 8))
    kv_heads = int(getattr(text_config, "num_key_value_heads", heads))
    head_dim = int(getattr(text_config, "head_dim", hidden // heads))
    # KV is retained across full-attention layers; prefill activations are a
    # conservative engineering allowance, not a measured physical peak.
    kv_bytes_per_token = 2 * full * kv_heads * head_dim * dtype_bytes
    linear = kv_bytes_per_token + 16 * hidden * dtype_bytes
    # Torch2.8 MPS general SDPA creates sequence-squared attention tensors.
    # CPU/unknown kernels are treated conservatively as well. No flash claim.
    quadratic = heads * (2 * dtype_bytes + 4)
    reserve = max(GIB, available_bytes // 4)
    budget = max(0, available_bytes - reserve - weight_bytes)
    if prefill_step is not None:
        # MLX evaluates bounded prefill chunks, not a prompt-wide N x N
        # attention tensor. A worker that reserves shared KV storage needs one
        # full KV plus a layer-growth allowance; legacy workers keep double KV.
        # Retain a chunk x context workspace even when SDPA uses a fused path.
        linear = ((full + 1) * 2 * kv_heads * head_dim * dtype_bytes
                  if shared_kv_reservation else 2 * kv_bytes_per_token)
        # Separate allowances for recurrent state/workspace and the worker's
        # bounded 256MiB free-buffer cache; active KV is counted above.
        budget = max(0, budget - 16 * hidden * dtype_bytes * prefill_step - GIB // 2)

    def estimate(tokens):
        return linear * tokens + quadratic * tokens * min(tokens, prefill_step or tokens)

    low, high = 0, ceiling
    while low < high:
        middle = (low + high + 1) // 2
        if estimate(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    # Smaller windows are permitted in automatic mode on small devices.
    safe = low // 256 * 256
    return {"mode": mode, "configured_limit": requested, "model_limit": maximum,
            "current_limit": safe if mode == "auto" else ceiling,
            "memory_estimate_limit": safe, "available_memory_bytes": available_bytes,
            "additional_weight_bytes": weight_bytes, "memory_budget_bytes": budget,
            "linear_bytes_per_token": linear, "kv_bytes_per_token": kv_bytes_per_token,
            "quadratic_bytes_per_token_squared": quadratic,
            "prefill_step": prefill_step,
            "estimated": True, "device": device,
            "scope": "engineering admission estimate; not a measured peak or no-OOM guarantee"}


def check_input(tokens, output_tokens, status):
    limit = status["current_limit"]
    total = tokens + output_tokens
    if limit is None:
        raise RuntimeError("模型上下文配置尚未就绪")
    if status.get("memory_estimate_limit") == 0:
        raise ContextAdmissionError("当前可用内存不足以加载模型并生成回答；请关闭占用内存的应用后重试。上下文设置未被改为0")
    if total > limit:
        raise ContextAdmissionError(f"输入与预留输出共 {total} token，超过当前上下文 {limit} token；请缩短对话")
    estimated = (status["linear_bytes_per_token"] * total
                 + status["quadratic_bytes_per_token_squared"] * tokens
                 * min(tokens, status.get("prefill_step") or tokens))
    estimated = max(0, estimated - status.get("resident_prefix_bytes", 0))
    if estimated > status["memory_budget_bytes"]:
        raise ContextAdmissionError("本次上下文预计超过设备可用内存，请缩短对话或选择自动档")


def fit_context(status, tokens, output_tokens, *, prefix=None):
    """Shrink only MLX prefill workspace, retaining the same memory snapshot/budget."""
    step = status.get("prefill_step")
    total = tokens + output_tokens
    if step is None:
        return status
    cached_step = prefix.get("prefill_step") if prefix else None
    resident = (min(prefix["bytes"], prefix["tokens"] * status.get(
                    "kv_bytes_per_token", status["linear_bytes_per_token"] // 2))
                if prefix and cached_step in (32, 64, 128, 256, 512) and cached_step <= step else 0)
    if not resident and total <= min(status["current_limit"], status.get("memory_estimate_limit", status["current_limit"])):
        return status
    ceiling = min(status.get("configured_limit", status["current_limit"]),
                  status.get("model_limit") or status["current_limit"],
                  status.get("native_limit", status.get("configured_limit", status["current_limit"])))
    if total > ceiling:
        return status
    candidates = ([cached_step] if resident else []) + [step, 64, 32]
    for smaller in dict.fromkeys(candidates):
        if smaller > step:
            continue
        credit = resident if smaller == cached_step else 0
        low, high = 0, ceiling
        while low < high:
            middle = (low + high + 1) // 2
            required = (status["linear_bytes_per_token"] * middle
                        + status["quadratic_bytes_per_token_squared"] * middle * min(middle, smaller))
            if max(0, required - credit) <= status["memory_budget_bytes"]:
                low = middle
            else:
                high = middle - 1
        safe = low // 256 * 256
        status = {**status, "prefill_step": smaller, "memory_estimate_limit": safe,
                  "resident_prefix_bytes": credit,
                  "current_limit": safe if status.get("mode") == "auto" else ceiling}
        if total <= min(status["current_limit"], safe):
            break
    return status


def generation_plan(runtime, tokens, requested, *, cache_reuse=False):
    """Admit output and execution against one fitted snapshot, freeing cache once if needed."""
    for attempt in range(2):
        status = runtime.context_status()
        engine = getattr(runtime, "native_engine", None)
        prefix = engine.prompt_cache(tokens[0].tolist(), cache_reuse=cache_reuse) if cache_reuse and engine else None
        status = fit_context(status, tokens.shape[1], requested or status.get("native_output_limit", 8192) * 4, prefix=prefix)
        try:
            check_input(tokens.shape[1], requested or 1, status)
            return status, output_budget(tokens.shape[1], requested, status)
        except ContextAdmissionError:
            engine = getattr(runtime, "native_engine", None)
            if attempt or engine is None or not getattr(engine, "_prefix_tokens", None):
                raise
            engine.prepare_prompt_cache(tokens[0].tolist(), cache_reuse=False)


def output_budget(tokens, requested, status):
    """Automatic output uses a quarter of the admitted window, never the advertised one."""
    if requested is not None and (type(requested) is not int or not 0 <= requested <= MAX_OUTPUT_TOKENS):
        raise ValueError(f"输出词元数需要在 1 到 {MAX_OUTPUT_TOKENS} 之间，或用 0 自动设置")
    engine_limit = status.get("native_output_limit", MAX_OUTPUT_TOKENS)
    if requested:
        if requested > engine_limit:
            raise ValueError(f"当前执行器最多输出 {engine_limit} 词元，请使用自动输出或调低上限")
        maximum = requested
    else:
        limit = status["current_limit"]
        if limit is None:
            raise RuntimeError("模型上下文配置尚未就绪")
        if status.get("memory_estimate_limit") == 0:
            check_input(tokens, 1, status)
        admitted = min(limit, status.get("memory_estimate_limit", limit))
        # Native requests stay <=2048 each; one automatic answer may span at
        # most four requests. Explicit API limits keep their existing contract.
        maximum = min(max(1, admitted // 4), admitted - tokens, engine_limit * 4, MAX_OUTPUT_TOKENS)
        if maximum <= 0:
            raise ValueError("当前上下文已没有输出空间，请先压缩或缩短对话")
    check_input(tokens, maximum, status)
    return maximum
