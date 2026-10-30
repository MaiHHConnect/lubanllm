"""Formal native text inference, with the existing Torch graph for experts/training."""
from contextlib import contextmanager, nullcontext
import hashlib
import json
import time
from pathlib import Path
import threading
import secrets

import torch

from .coldpaging import full_backbone
from .native_expert import native_expert_binding
from .native_q4 import binding as direct_q4_binding
from .native_mlx import model_binding as mlx_model_binding, expert_binding as mlx_expert_binding

PRODUCT_NATIVE_PRECISION = "bf16"
PRODUCT_NATIVE_Q4_PROFILE = Path(__file__).with_name("native_q4_clip_profile.txt")


def native_binary(runtime):
    if (getattr(runtime, "research", False) or getattr(runtime, "device", None) != "mps"
            or getattr(runtime, "cold_backbone", None) is None
            or getattr(getattr(runtime, "model", None), "config", None) is None
            or runtime.model.config.model_type != "qwen3_5"
            or runtime.dtype != torch.bfloat16):
        return None
    config = getattr(runtime, "text_config", None)
    if config is None or (config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.vocab_size) != (2048, 6144, 24, 248320):
        return None
    root = Path(__file__).resolve().parents[2]
    name = "luban-mlx-worker" if getattr(runtime, "native_inference_precision", None) == "mlx-q4" else "luban-worker"
    for path in (root / "native" / name, root / "build/native" / name):
        if path.is_file():
            return path
    if name == "luban-mlx-worker":
        raise RuntimeError("MLX 原生推理程序未安装")
    return None


def expert_binding(runtime, binary):
    return (mlx_expert_binding(runtime, binary) if getattr(runtime, "native_inference_precision", None) == "mlx-q4"
            else native_expert_binding(runtime, binary))


def trial_expert_binding(runtime, binary):
    """Return a usable local-trial binding, or disable that trial for this request.

    A trial proof is tied to the exact inference worker that produced it.  A
    rebuilt worker can therefore invalidate an old trial proof without making
    the base model unusable.  Formal/admitted bindings still fail closed.
    """
    try:
        return expert_binding(runtime, binary)
    except ValueError as error:
        if getattr(runtime, "router_meta", {}).get("execution_scope") == "local-trial":
            runtime._native_trial_expert_error = str(error)
            return None
        raise


def identity(runtime, actual_action=None):
    path = native_binary(runtime)
    if (path is not None and (actual_action != "neural_expert" or trial_expert_binding(runtime, path) is not None)
            and not (actual_action is not None and runtime.cold_backbone.depth_count > 0)):
        receipt = json.loads(path.with_name(path.name + ".json").read_text())
        result = {"engine": "luban-native-" + getattr(runtime, "native_inference_precision", PRODUCT_NATIVE_PRECISION) + "-metal", "binary_sha256": receipt["binary_sha256"]}
        if getattr(runtime, "native_inference_precision", None) == "mlx-q4":
            model = mlx_model_binding(runtime)
            from .mlx_learning import accepted_report
            admitted = accepted_report(runtime, path) is not None
            result.update(mlx_model_sha256=model["sha256"], mlx_revision=model["revision"],
                          execution_scope="product" if admitted else "local-trial", quality_admitted=admitted)
        skip_mask, exit_after = getattr(runtime, "native_skip_layer_mask", 0), getattr(runtime, "native_exit_after", 24)
        if skip_mask or exit_after != 24:
            result.update(execution_skip_mask=skip_mask, execution_exit_after=exit_after)
        if getattr(runtime, "native_layer_paging", False):
            result["layer_paging_enabled"] = True
        return result
    return {"engine": "torch-hf", "torch_version": torch.__version__}


def close(runtime):
    engine = getattr(runtime, "native_engine", None)
    if engine is not None:
        runtime.native_engine = None
        engine.close()


def weight_bytes(runtime):
    """Full native language allocation, including its FP32 recurrent tensors.

    The worker owns another copy of the embedding and first four layers. Do
    not count only HF meta tensors, or count an injected expert as base weight.
    """
    path = native_binary(runtime)
    mlx = mlx_model_binding(runtime) if path is not None else None
    if mlx is not None:
        # M4 native cold-load/generation high-water was 2,237,450,304 bytes.
        # Round up to 64 MiB; the context allocator retains its separate system margin.
        return max(mlx["weight_bytes"] + 2 * 2048 * 6144 * 2, 34 * 64 * 1024**2)
    direct = direct_q4_binding(runtime, path) if path is not None else None
    if direct is not None:
        return direct["cold_weight_bytes"]
    return sum(value.numel() * max(value.element_size(), 4 if name.endswith(
        ("linear_attn.A_log", "linear_attn.norm.weight")) else 2)
        for name, value in runtime.decoder.named_parameters() if ".expert." not in name)


def prepare_native(runtime, stop, *, wait=True):
    """Release completed HF routing before admitting a cold native allocation."""
    pager = getattr(runtime, "cold_backbone", None)
    path = native_binary(runtime)
    if path is None or pager.depth_count or pager.loaded:
        return
    if (runtime.branch.expert is not None
            and getattr(runtime, "native_inference_precision", None) == "mlx-q4"
            and getattr(runtime, "router_meta", {}).get("execution_scope") == "local-trial"
            and trial_expert_binding(runtime, path) is None):
        runtime.branch.expert = None
        return
    before = torch.mps.driver_allocated_memory() if runtime.device == "mps" else 0
    pager.evict_resident()
    engine = getattr(runtime, "native_engine", None)
    process = getattr(engine, "process", None)
    if wait and (process is None or process.poll() is not None):
        direct_q4_binding(runtime, path, prepare=True, stop=stop)
    if (not wait or runtime.device != "mps" or torch.mps.driver_allocated_memory() >= before
            or process is not None and process.poll() is None):
        return
    # Darwin's system VM statistics can lag an actual Metal release by ~1 s.
    # Re-read the unchanged admission estimate only after a measured release;
    # never credit estimated freed bytes or allocate the worker before admission.
    deadline = time.monotonic() + 2.0
    while runtime.context_status()["memory_estimate_limit"] == 0:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return  # The original input/memory guard still rejects this request.
        if stop.wait(min(0.05, remaining)):
            raise RuntimeError("生成已取消")


def start(runtime):
    path = native_binary(runtime)
    if path is None:
        return None
    pager = runtime.cold_backbone
    if pager.loaded or pager.depth_count:
        raise RuntimeError("原生加载前必须释放完整 HF 核心")
    from .settings import native_execution_policy, require_storage
    skip_mask, exit_after, layer_paging = native_execution_policy(
        getattr(runtime, "native_skip_layer_mask", 0), getattr(runtime, "native_exit_after", 24),
        getattr(runtime, "native_layer_paging", False))
    if runtime.active_core is not None and (exit_after < 24 or skip_mask & (1 << 23)):
        raise ValueError("活动学习核心要求执行第23层，不能提前退出或跳过该层")
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
    expert = expert_binding(runtime, path) if runtime.branch.expert is not None else None
    if runtime.branch.expert is not None and expert is None:
        raise ValueError("已选专家没有当前生成引擎的有效绑定")
    from .native_engine import NativeEngine
    manifest = Path(runtime.loaded_path) / "upstream-file-manifest.json"
    precision = getattr(runtime, "native_inference_precision", PRODUCT_NATIVE_PRECISION)
    profile = getattr(runtime, "native_q4_profile", PRODUCT_NATIVE_Q4_PROFILE if precision == "mixed-q4-calibrated" else None)
    paging_path = None
    if layer_paging:
        directory = require_storage(runtime.loaded_storage) / "runtime" / "layer-paging"
        directory.mkdir(parents=True, exist_ok=True)
        paging_path = directory / (secrets.token_hex(16) + ".bin")
    engine = NativeEngine(path, runtime.loaded_path, hashlib.sha256(manifest.read_bytes()).hexdigest(), core=core,
                          expert=expert,
                          direct_q4=direct_q4_binding(runtime, path), mlx_model=mlx_model_binding(runtime),
                          precision=precision, calibration_profile=profile, skip_layer_mask=skip_mask,
                          exit_after=exit_after, layer_paging_path=paging_path)
    try:
        pager.evict_resident()
        engine.start()
    except BaseException:
        engine.close()
        raise
    runtime.native_engine = engine
    return engine


@contextmanager
def answer_backbone(runtime, messages, allow_reference=True, allow_expert=True, stop=None):
    stop = stop if stop is not None else threading.Event()
    if not allow_expert:
        context = full_backbone(runtime) if native_binary(runtime) is None else nullcontext()
    else:
        branch = runtime.branch
        has_expert = branch.expert is not None or getattr(runtime, "pointer", Path("/nonexistent")).exists()
        if has_expert:
            hint = runtime.external_hint(messages)
            if hint and hint.get("has_hint") and (allow_reference and hint["status"] != "mastered"
                         or hint["status"] == "mastered" and hint["local_observation"]["local_action"] == "local_core"):
                has_expert = False
            elif (branch.requires_gate or runtime.router_meta.get("gates_expert")):
                has_expert = runtime.choose_route(messages).get("action") == "neural_expert"
        path = native_binary(runtime)
        native_expert = trial_expert_binding(runtime, path) if has_expert else None
        if (has_expert and native_expert is None
                and getattr(runtime, "native_inference_precision", None) == "mlx-q4"
                and getattr(runtime, "router_meta", {}).get("execution_scope") == "local-trial"):
            has_expert = False
            if getattr(runtime.branch, "expert", None) is not None:
                runtime.branch.expert = None
        if native_expert is not None and branch.expert is None:
            # Hold only the small, immutable expert on CPU; the native worker loads it once.
            branch.activate(branch.expert_id, device="cpu")
        context = full_backbone(runtime) if has_expert and native_expert is None or path is None else nullcontext()
    try:
        with context:
            yield
    except BaseException as error:
        try:
            prepare_native(runtime, stop, wait=False)
        except BaseException as cleanup:
            error.add_note("HF退出清理失败：" + str(cleanup))
        raise
    else:
        # Nested HF/training scopes retain their residency guard. Native cleanup
        # releases only the HF route/head, leaving an attached layer-23 expert intact.
        prepare_native(runtime, stop, wait=not stop.is_set())


def eos_token_ids(runtime):
    eos = getattr(getattr(runtime.model, "generation_config", None), "eos_token_id", None)
    eos = runtime.tokenizer.eos_token_id if not isinstance(eos, (int, list, tuple)) else eos
    ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    # This Qwen MLX source has no generation_config.json: its model config only
    # lists endoftext, while the chat tokenizer and training targets use im_end.
    if getattr(runtime, "native_inference_precision", None) == "mlx-q4":
        chat_eos = runtime.tokenizer.eos_token_id
        if type(chat_eos) is int and chat_eos not in ids:
            ids.append(chat_eos)
    return ids


def generate_tokens(runtime, tokens, maximum, *, temperature=0.0, emit=None, stop=None, cache_reuse=False, context=None):
    if type(cache_reuse) is not bool:
        raise ValueError("Native cache reuse must be a boolean")
    from .runtime import EmitStreamer, StopRequested
    stop = stop if stop is not None else threading.Event()
    eos_ids = eos_token_ids(runtime)
    pager = getattr(runtime, "cold_backbone", None)
    path = native_binary(runtime)
    expert = expert_binding(runtime, path) if runtime.branch.expert is not None else None
    use_native = (path is not None and (runtime.branch.expert is None or expert is not None)
                  and pager.depth_count == 0)
    if use_native:
        if type(maximum) is not int or not 1 <= maximum <= 8192:
            raise ValueError("原生单次回答输出预算需要在 1–8192 词元之间")
        if pager.loaded:
            pager._evict()
        pager.evict_resident()
        engine = getattr(runtime, "native_engine", None)
        if engine is not None and getattr(engine, "expert", None) != expert:
            close(runtime)
            engine = None
        engine = engine or start(runtime)
        streamer = EmitStreamer(runtime.tokenizer, emit) if emit is not None else None
        if streamer is not None:
            streamer.put(tokens.detach().cpu())
        generated = []
        measurements = []
        prompt = tokens[0].tolist()
        context_options = {}
        if getattr(runtime, "native_inference_precision", None) == "mlx-q4":
            context = context if context is not None else runtime.context_status()
            context_options = {"context_limit": context["current_limit"],
                               "prefill_step": context["prefill_step"]}
        try:
            for _ in range(4):
                if stop.is_set() or len(generated) >= maximum:
                    break
                chunk_limit = min(2048, maximum - len(generated))
                result = engine.generate(prompt + generated, max_tokens=chunk_limit, eos_ids=eos_ids,
                    temperature=temperature, stop=stop, seed=secrets.randbits(64) if emit is not None and temperature > 0 else 42,
                    cache_reuse=cache_reuse,
                    **context_options,
                    on_token=(lambda token: streamer.put(torch.tensor([token]))) if streamer else (lambda _: None))
                segment = result["tokens"]
                measurements.append(result)
                generated.extend(segment)
                if (result["finish_reason"] != "length" or len(segment) != chunk_limit
                        or segment[-1] in eos_ids):
                    break
        except BaseException:
            close(runtime)
            raise
        finally:
            if streamer is not None:
                streamer.end()
        decode_tokens = sum(max(0, len(row["tokens"]) - 1) for row in measurements)
        decode_ns = [row.get("decode_ns") for row in measurements if len(row["tokens"]) > 1]
        valid_decode = decode_ns and all(type(value) is int and value > 0 for value in decode_ns)
        prefill_ns = [row.get("prefill_ns") for row in measurements]
        ttft_ns = measurements[0].get("ttft_ns") if measurements else None
        runtime._generation_timing = {
            "decode_tokens": decode_tokens,
            "decode_seconds": sum(decode_ns) / 1e9 if valid_decode else None,
            "decode_tokens_per_second": decode_tokens * 1e9 / sum(decode_ns) if valid_decode else None,
            "prefill_seconds": sum(prefill_ns) / 1e9 if prefill_ns and all(type(value) is int and value >= 0 for value in prefill_ns) else None,
            "time_to_first_token_seconds": ttft_ns / 1e9 if type(ttft_ns) is int and ttft_ns >= 0 else None,
            # Later output segments reuse tokens generated within this response;
            # API cache usage counts only the caller's original input prefix.
            "cached_prompt_tokens": measurements[0].get("cached_prompt_tokens", 0) if measurements else 0,
            "native_memory": [row["native_memory"] for row in measurements if "native_memory" in row],
        }
        return torch.cat((tokens, torch.tensor([generated], device=tokens.device, dtype=tokens.dtype)), dim=1)
    with full_backbone(runtime), torch.no_grad():
        kwargs = {"max_new_tokens": maximum, "attention_mask": torch.ones_like(tokens),
                  "do_sample": temperature > 0, "temperature": temperature if temperature > 0 else None,
                  "top_p": 0.95 if temperature > 0 else None, "pad_token_id": runtime.tokenizer.eos_token_id,
                  "stopping_criteria": [StopRequested(stop)]}
        timing = {}
        if getattr(runtime, "native_inference_precision", None) == "mlx-q4":
            kwargs["eos_token_id"] = eos_ids
        kwargs["streamer"] = EmitStreamer(runtime.tokenizer, emit, timing=timing)
        started = time.perf_counter()
        output = runtime.model.generate(tokens, **kwargs)
        count = output.shape[1] - tokens.shape[1]
        seconds = timing.get("last", 0) - timing.get("first", 0)
        valid = timing.get("tokens") == count and count > 1 and seconds > 0
        runtime._generation_timing = {
            "decode_tokens": max(0, count - 1), "decode_seconds": seconds if valid else None,
            "decode_tokens_per_second": (count - 1) / seconds if valid else None,
            "time_to_first_token_seconds": timing["first"] - started if "first" in timing else None,
        }
        return output


def recheck_core(runtime, maximum):
    """Two separate closed-book product executions, never reuse HF mastery."""
    from .external_brain import ExternalBrain
    from .retention import messages as sample_messages, provenance
    from .generation_checks import validate_generation_check
    binding = runtime.external_binding(actual_action="local_core")
    if binding is None or runtime.training_session is not None or runtime.state != "ready":
        return {"checked": 0, "mastered": 0}
    brain = ExternalBrain(runtime.memory)
    rows = brain.recheck_candidates(runtime.identity, binding, limit=maximum)
    mastered = 0
    runtime._external_probe_active = True
    try:
        for candidate in rows:
            sample = candidate["reference"]
            context = sample_messages(sample)
            if candidate["binding"] != binding or candidate["context"] != context:
                raise ValueError("外脑复测材料与原生执行路径不一致")
            check = validate_generation_check(provenance(sample).get("generation_check"), default_expected=sample["answer"])
            for _ in range(2):
                usage = runtime._generate_loaded(context, lambda _: None, threading.Event(), thinking="off",
                    max_tokens=check.get("max_new_tokens", 192), temperature=0, allow_reference=False)
                if usage["external_brain_status"] == "mastered":
                    mastered += 1
                    break
    finally:
        runtime._external_probe_active = False
    return {"checked": len(rows), "mastered": mastered}
