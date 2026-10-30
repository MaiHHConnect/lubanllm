"""Research-only frozen last-down cache audit; never called by product training.

CLI only constructs an offline, randomly initialized tiny official Qwen3.5 CPU
model. An actual native model audit requires separate caller authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from luban.backbone import text_backbone
from luban.dynamic import DynamicMLP
from luban.evolution import (_target, _cancel, _answer_log_probs, _capture_replay_subspace,
                             _project_core_delta, _teacher_replay)
from verify_training_operators import _comparison, TOLERANCES

PROTOCOL = "research-frozen-last-down-full-sequence-cache-v1"
ROOT = Path(__file__).resolve().parents[1]


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type != "cpu":
        raise ValueError("Cache audit supports CPU or separately authorized MPS only")


def _sources(model):
    paths = {Path(inspect.getfile(type(module))).resolve() for module in model.modules()}
    paths.update(ROOT / name for name in (
        "scripts/verify_cached_training.py", "scripts/verify_training_operators.py",
        "src/luban/evolution.py", "src/luban/backbone.py", "src/luban/dynamic.py"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def _binding(runtime, parameter):
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5MLP, Qwen3_5RMSNorm
    model, decoder = runtime.model, runtime.decoder
    native_decoder, _ = text_backbone(model)
    layer = decoder.layers[-1]
    if (type(model) is not Qwen3_5ForConditionalGeneration or native_decoder is not decoder
            or type(layer) is not Qwen3_5DecoderLayer or type(layer.mlp) is not DynamicMLP
            or layer.mlp is not runtime.branch or type(layer.mlp.base) is not Qwen3_5MLP
            or type(layer.mlp.base.down_proj) is not torch.nn.Linear
            or type(decoder.norm) is not Qwen3_5RMSNorm or type(model.lm_head) is not torch.nn.Linear
            or layer.mlp.base.down_proj.weight is not parameter):
        raise ValueError("Cache supports only this actual official Qwen3.5 last-down graph")
    if runtime.branch.expert is not None:
        raise ValueError("Cache requires branch OFF")
    if not isinstance(getattr(runtime, "identity", None), str) or not runtime.identity:
        raise ValueError("Cache requires a bound model identity")
    for module in model.modules():
        if module.training or module._forward_pre_hooks or module._forward_hooks:
            raise ValueError("Cache requires eval graph without external forward hooks")
        if getattr(module.forward, "__func__", None) is not type(module).forward:
            raise ValueError("Cache rejects replaced graph forward methods")
    parameters = list(model.named_parameters())
    if any(value.requires_grad != (value is parameter) for _, value in parameters):
        raise ValueError("Cache requires all parameters frozen except last down weight")
    pointer = Path(runtime.store) / "active-core.json"
    return {"identity": runtime.identity, "core_version": getattr(runtime, "active_core", None),
        "core_pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None,
        "model": id(model), "decoder": id(decoder), "config": model.config.to_json_string(),
        "graph": [(name, id(module), type(module).__qualname__) for name, module in model.named_modules()],
        "sources": _sources(model),
        "frozen_parameters": [(name, id(value), value._version, tuple(value.shape), str(value.dtype), str(value.device))
                              for name, value in parameters if value is not parameter],
        "buffers": [(name, id(value), value._version, tuple(value.shape)) for name, value in model.named_buffers()]}


def capture_tail(runtime, sample):
    """Ephemeral CPU same-dtype X/R/ids/labels; no cached vocabulary logits."""
    if sample.get("purpose") != "training":
        raise ValueError("Cache audit may only read fixed TRAIN rows")
    name, parameter = _target(runtime)
    binding = _binding(runtime, parameter)
    ids, labels = runtime._sample_tokens(sample)
    if (ids.ndim != 2 or labels.shape != ids.shape or ids.shape[1] < 2
            or ids.dtype != torch.long or labels.dtype != torch.long
            or not (labels[:, 1:] != -100).any()):
        raise ValueError("Cache requires complete causal IDs and labels with answer/EOS targets")
    layer = runtime.decoder.layers[-1]
    captured, handles = {}, []

    def capture(key):
        def hook(module, args):
            if key in captured:
                raise ValueError("Cache graph projection executed more than once")
            captured[key] = args[0].detach().clone()
        return hook

    _sync(parameter.device)
    began = time.perf_counter()
    try:
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(capture("residual")))
        handles.append(runtime.branch.base.down_proj.register_forward_pre_hook(capture("x")))
        with torch.no_grad():
            runtime.model(ids, use_cache=False)
        _sync(parameter.device)
        graph_seconds = time.perf_counter() - began
    finally:
        for handle in handles:
            handle.remove()
    _cancel(runtime)
    if (set(captured) != {"x", "residual"}
            or captured["x"].shape != (*ids.shape, parameter.shape[1])
            or captured["residual"].shape != (*ids.shape, parameter.shape[0])
            or any(value.dtype != parameter.dtype or not torch.isfinite(value).all() for value in captured.values())):
        raise ValueError("Cache did not capture complete native same-dtype X and direct residual")
    copy_started = time.perf_counter()
    cpu = {key: value.to("cpu", copy=True) for key, value in {**captured, "ids": ids, "labels": labels}.items()}
    _sync(parameter.device)
    copy_seconds = time.perf_counter() - copy_started
    if _binding(runtime, parameter) != binding:
        raise ValueError("Frozen prefix/source/version changed during capture")
    return {**cpu, "binding": binding, "sample_sha": hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest(),
        "tensor_sha": {key: tensor_sha(value) for key, value in cpu.items()}, "parameter": name,
        "timing": {"native_graph_seconds": graph_seconds, "to_cpu_seconds": copy_seconds},
        "bytes": {key: value.numel() * value.element_size() for key, value in cpu.items()}}


def cached_logits(runtime, sample, cache, weight):
    """Validate the current graph and full inputs before every tail evaluation."""
    _, parameter = _target(runtime)
    if _binding(runtime, parameter) != cache["binding"]:
        raise ValueError("Cached frozen prefix/source/version no longer matches")
    if hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest() != cache["sample_sha"]:
        raise ValueError("Cached TRAIN sample changed")
    ids, labels = runtime._sample_tokens(sample)
    for key, value in (("ids", ids), ("labels", labels)):
        if tensor_sha(value) != cache["tensor_sha"][key]:
            raise ValueError("Cached full input or labels changed")
    for key in ("x", "residual", "ids", "labels"):
        value = cache[key]
        if value.device.type != "cpu" or tensor_sha(value) != cache["tensor_sha"][key]:
            raise ValueError("Cached CPU tensor was altered")
    if weight.shape != parameter.shape or weight.dtype != parameter.dtype or weight.device != parameter.device:
        raise ValueError("Cached tail weight shape/dtype/device differs from native last down")
    _sync(parameter.device)
    began = time.perf_counter()
    x, residual = (cache[key].to(parameter.device, copy=True) for key in ("x", "residual"))
    _sync(parameter.device)
    transfer_seconds = time.perf_counter() - began
    hidden = F.linear(x, weight, runtime.branch.base.down_proj.bias)
    logits = runtime.model.lm_head(runtime.decoder.norm(hidden + residual))
    return logits, transfer_seconds


def _ce(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def _compare(actual, expected, tolerance, kind="logits"):
    return _comparison(actual, expected, atol=tolerance[kind + "_atol"], rtol=tolerance["rtol"],
                       relative_l2=tolerance["relative_l2"])


def audit_cached_training(runtime, samples):
    """Two independent CE->KL steps with native/tail FP32 Adam + replay SVD.

    The first two fixed TRAIN rows supply CE; the last supplies qualified replay
    TRAIN for KL/projection. Independent actual replay qualification belongs to
    the caller's frozen protocol; this audit checks role/purpose, not truth.
    Only clones are optimized. No generation or commit.
    """
    if len(samples) != 3 or len({sample.get("id") for sample in samples}) != 3:
        raise ValueError("Audit requires exactly three distinct fixed TRAIN rows")
    if any(sample.get("purpose") != "training" for sample in samples):
        raise ValueError("Audit cannot consume held-out validation rows")
    if json.loads(samples[-1].get("provenance", "{}" )).get("evolution_role") != "replay":
        raise ValueError("The third row must be fixed replay TRAIN")
    with runtime.lock, torch.enable_grad():
        _cancel(runtime)
        model = runtime.model
        name, parameter = _target(runtime)
        tolerance = TOLERANCES.get(str(parameter.dtype))
        if tolerance is None:
            raise ValueError("Cache audit supports FP32/BF16 only")
        _sync(parameter.device)
        original = parameter.detach().clone()
        flags = [(value, value.requires_grad, value.grad) for value in model.parameters()]
        modes = [(module, module.training) for module in model.modules()]
        expert = runtime.branch.expert
        if expert is not None:
            raise ValueError("Cache audit requires branch OFF before entry")
        pointer = Path(runtime.store) / "active-core.json"
        pointer_before = pointer.read_bytes() if pointer.exists() else None
        core_before = getattr(runtime, "active_core", None)
        rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone() if parameter.device.type == "mps" else None
        caches, handles, rows = [], [], []
        try:
            model.eval().requires_grad_(False)
            parameter.requires_grad_(True)
            for sample in samples:
                caches.append(capture_tail(runtime, sample))
            native_basis, basis_report = _capture_replay_subspace(runtime, parameter, [samples[-1]])
            replay_cache = caches[-1]
            selected = replay_cache["x"][:, :-1][replay_cache["labels"][:, 1:] != -100].float()
            _, singular, vh = torch.linalg.svd(selected, full_matrices=False)
            rank = int((singular > max(selected.shape) * torch.finfo(torch.float32).eps * float(singular[0])).sum())
            if not 0 < rank < parameter.shape[1]:
                raise ValueError("Replay SVD must leave a nontrivial trainable complement")
            cached_basis = vh[:rank].T.contiguous()
            basis_check = _compare(cached_basis, native_basis, tolerance)
            native_master = torch.nn.Parameter(original.float().clone())
            tail_master = torch.nn.Parameter(original.float().clone())
            native_optimizer = torch.optim.Adam([native_master], lr=1e-5, eps=1e-6)
            tail_optimizer = torch.optim.Adam([tail_master], lr=1e-5, eps=1e-6)
            basis_native = native_basis.to(parameter.device)
            basis_tail = cached_basis.to(parameter.device)
            transfer_seconds = 0.0
            for step, (sample, cache) in enumerate(zip(samples[:2], caches[:2]), 1):
                _cancel(runtime)
                with torch.no_grad():
                    parameter.copy_(native_master.to(parameter.dtype))
                ids, labels = runtime._sample_tokens(sample)
                replay_ids, replay_labels = runtime._sample_tokens(samples[-1])
                native_optimizer.zero_grad(set_to_none=True)
                tail_optimizer.zero_grad(set_to_none=True)
                parameter.grad = None
                teacher_native = _teacher_replay(runtime, parameter, original, parameter.detach().clone(), replay_ids, replay_labels)
                with torch.no_grad():
                    teacher_logits, copied = cached_logits(runtime, samples[-1], replay_cache, original)
                    teacher_tail = _answer_log_probs(teacher_logits, replay_labels)
                    teacher_check = _compare(teacher_tail, teacher_native, tolerance)
                    del teacher_logits
                transfer_seconds += copied
                native_grads = []
                handles.append(parameter.register_hook(lambda value: native_grads.append(value.detach().clone())))
                native = model(ids, labels=labels, use_cache=False)
                native_logits = native.logits.detach()
                native_loss = native.loss
                native_loss.backward()
                native_ce = native_loss.detach()
                del native, native_loss
                native_replay = model(replay_ids, use_cache=False).logits
                native_replay_logits = native_replay.detach()
                native_kl = F.kl_div(_answer_log_probs(native_replay, replay_labels), teacher_native,
                                     reduction="batchmean", log_target=True)
                (5 * native_kl).backward()
                native_total = parameter.grad.detach().clone()
                handles.pop().remove()
                del teacher_native, native_replay
                tail_weight = tail_master.detach().to(parameter.dtype).clone().requires_grad_(True)
                tail_grads = []
                handles.append(tail_weight.register_hook(lambda value: tail_grads.append(value.detach().clone())))
                tail_logits, copied = cached_logits(runtime, sample, cache, tail_weight)
                transfer_seconds += copied
                tail_loss = _ce(tail_logits, labels)
                tail_loss.backward()
                tail_ce = tail_loss.detach()
                del tail_loss
                tail_replay, copied = cached_logits(runtime, samples[-1], replay_cache, tail_weight)
                transfer_seconds += copied
                tail_kl = F.kl_div(_answer_log_probs(tail_replay, replay_labels), teacher_tail,
                                   reduction="batchmean", log_target=True)
                (5 * tail_kl).backward()
                handles.pop().remove()
                if len(native_grads) != 2 or len(tail_grads) != 2:
                    raise ValueError("Audit did not preserve two separate CE then KL backward calls")
                checks = {"logits": _compare(tail_logits, native_logits, tolerance),
                    "replay_logits": _compare(tail_replay, native_replay_logits, tolerance),
                    "ce": _compare(tail_ce, native_ce, tolerance, "loss"),
                    "kl": _compare(tail_kl.detach(), native_kl.detach(), tolerance, "loss"),
                    "ce_gradient": _compare(tail_grads[0], native_grads[0], tolerance, "gradient"),
                    "weighted_kl_gradient": _compare(tail_grads[1], native_grads[1], tolerance, "gradient"),
                    "accumulated_gradient": _compare(tail_weight.grad, native_total, tolerance, "gradient"),
                    "original_teacher_log_probs": teacher_check}
                native_master.grad = native_total.float().clone()
                tail_master.grad = tail_weight.grad.float().clone()
                native_norm = torch.nn.utils.clip_grad_norm_([native_master], 1, error_if_nonfinite=True)
                tail_norm = torch.nn.utils.clip_grad_norm_([tail_master], 1, error_if_nonfinite=True)
                native_optimizer.step()
                tail_optimizer.step()
                _project_core_delta(native_master, original.float(), basis_native)
                _project_core_delta(tail_master, original.float(), basis_tail)
                checks["fp32_master"] = _comparison(tail_master, native_master, atol=tolerance["adam_atol"], rtol=0,
                                                      relative_l2=tolerance["relative_l2"])
                checks["cumulative_update"] = _comparison(tail_master - original.float(), native_master - original.float(),
                    atol=tolerance["adam_atol"], rtol=tolerance["rtol"], relative_l2=tolerance["update_relative_l2"])
                checks["native_updated_weight"] = _compare(tail_master.to(parameter.dtype), native_master.to(parameter.dtype), tolerance)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    checks["adam_" + key] = _compare(tail_optimizer.state[tail_master][key], native_optimizer.state[native_master][key], tolerance, "adam")
                rows.append({"step": step, "train_id": sample["id"], "replay_id": samples[-1]["id"],
                    "native_ce": float(native_ce.cpu()), "native_kl": float(native_kl.detach().cpu()),
                    "native_preclip_norm": float(native_norm.cpu()), "tail_preclip_norm": float(tail_norm.cpu()),
                    "checks": checks, "passed": all(value["passed"] for value in checks.values())})
                del tail_logits, tail_replay, teacher_tail, native_kl, tail_kl, tail_weight
            _sync(parameter.device)
            rng_unchanged = torch.equal(rng, torch.get_rng_state()) and (mps_rng is None or torch.equal(mps_rng, torch.mps.get_rng_state()))
            report = {"protocol": PROTOCOL, "kind": "research_only_cached_tail_audit", "passed": basis_check["passed"] and all(row["passed"] for row in rows) and rng_unchanged,
                "identity": runtime.identity, "core_version": core_before, "parameter": name,
                "dtype": str(parameter.dtype), "device": str(parameter.device), "torch_version": torch.__version__, "hardware": platform.platform(),
                "teacher_calls": 0, "generation_calls": 0, "publication": False, "gate_enabled": False,
                "backward_order": ["CE.backward", "(5*KL).backward"], "steps": rows,
                "replay_svd": basis_report, "cached_basis_check": basis_check, "rng_unchanged": rng_unchanged,
                "cache": [{"id": sample["id"], "shapes": {key: list(cache[key].shape) for key in ("x", "residual", "ids", "labels")},
                           "bytes": cache["bytes"], "timing": cache["timing"], "tensor_sha": cache["tensor_sha"]} for sample, cache in zip(samples, caches)],
                "cache_bytes": sum(sum(cache["bytes"].values()) for cache in caches),
                "cache_to_native_transfer_seconds": transfer_seconds, "source_sha256": caches[0]["binding"]["sources"],
                "scope": "three fixed TRAIN rows/two clone optimizer steps; complete same-shape native tail only, no production cache, no measured real2B speed/memory or quality claim"}
        finally:
            for handle in handles:
                handle.remove()
            with torch.no_grad():
                parameter.copy_(original)
            for value, flag, grad in flags:
                value.requires_grad_(flag)
                value.grad = grad
            for module, training in modes:
                module.training = training
            torch.set_rng_state(rng)
            if mps_rng is not None:
                torch.mps.set_rng_state(mps_rng)
            for cache in caches:
                cache.clear()
            caches.clear()
            _sync(parameter.device)
            if (not torch.equal(parameter, original) or runtime.branch.expert is not expert
                    or getattr(runtime, "active_core", None) != core_before
                    or (pointer.read_bytes() if pointer.exists() else None) != pointer_before
                    or any(value.requires_grad != flag or value.grad is not grad for value, flag, grad in flags)):
                raise RuntimeError("Cached audit failed to restore original runtime state")
        report["restored"] = True
        report["retained_cache_entries"] = len(caches)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True,
                        help="Only runs dedicated offline tiny official Qwen3.5 tests; no native model loader")
    parser.parse_args()
    import sys
    import unittest
    sys.path.insert(0, str(ROOT / "tests"))
    import test_cached_training
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_cached_training)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"kind": "tiny_official_cpu_cached_training_self_test", "passed": result.wasSuccessful(),
                      "tests_run": result.testsRun, "seed": 42, "reports": test_cached_training.REPORTS,
                      "teacher_calls": 0, "generation_calls": 0, "real_native_model_loaded": False},
                     ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
