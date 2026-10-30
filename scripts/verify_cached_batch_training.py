"""Research-only padded cached-tail batch objective audit, no model loader.

Per-question token means preserve the existing per-question CE/KL weighting.
One batch Adam update is a different trajectory from sequential question Adam.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import verify_cached_training as cached

PROTOCOL = "research-cached-tail-question-mean-batch-v1"
TRAIN_EOS_PROTOCOL = "runtime-sample-tokens-tokenizer-eos-v1"


def _check(actual, expected, tolerance, kind="logits", *, update=False):
    """Keep the old strict gate; standard allclose is diagnostic only."""
    atol = tolerance[kind + "_atol"]
    relative = tolerance["update_relative_l2" if update else "relative_l2"]
    result = cached._comparison(actual, expected, atol=atol, rtol=tolerance["rtol"], relative_l2=relative)
    result["strict_relative_passed"] = result["passed"]
    actual, expected = actual.detach().float(), expected.detach().float()
    result["standard_allclose_diagnostic"] = bool(torch.allclose(actual, expected, atol=atol, rtol=tolerance["rtol"]))
    scale = actual.numel() ** .5 * atol + tolerance["rtol"] * float(expected.norm())
    result["absolute_floor_normalized_l2_diagnostic"] = float((actual - expected).norm()) / max(scale, 1e-30) if result["finite"] else None
    return result


def _eos(runtime):
    # Runtime._sample_tokens appends tokenizer EOS, independently of generation.
    value = runtime.tokenizer.eos_token_id
    if type(value) is not int or value < 0:
        raise ValueError("Batch TRAIN labels require the tokenizer EOS ID")
    return {value}


def _labels(ids, labels, eos):
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape != labels.shape
            or ids.dtype != torch.long or labels.dtype != torch.long or ids.shape[1] < 2):
        raise ValueError("Batch audit requires one complete TRAIN sequence per row")
    selected = labels[0].ne(-100).nonzero(as_tuple=False).flatten()
    if (not selected.numel() or selected[0] < 1 or selected[-1] != ids.shape[1] - 1
            or not labels[0, selected[0]:].ne(-100).all()
            or not torch.equal(labels[0, selected], ids[0, selected])
            or int(ids[0, -1]) not in eos
            or any(int(value) in eos for value in labels[0, selected[:-1]])):
        raise ValueError("Batch labels require masked prefix and uninterrupted complete gold+EOS")


def _pack(runtime, samples, caches, weight):
    """Validate existing full-sequence caches before padding only the tail."""
    if not samples or len(samples) != len(caches):
        raise ValueError("Batch audit cannot drop or invent rows")
    _, parameter = cached._target(runtime)
    binding = cached._binding(runtime, parameter)
    if (weight.shape != parameter.shape or weight.dtype != parameter.dtype
            or weight.device != parameter.device or not torch.isfinite(weight).all()):
        raise ValueError("Batch weight differs from the finite native last-down tensor")
    eos = _eos(runtime)
    for sample, cache in zip(samples, caches):
        cached._cancel(runtime)
        if binding != cache["binding"] or hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest() != cache["sample_sha"]:
            raise ValueError("Batch cache input/source/model binding changed")
        ids, labels = runtime._sample_tokens(sample)
        _labels(ids, labels, eos)
        for key, value in (("ids", ids), ("labels", labels)):
            if cached.tensor_sha(value) != cache["tensor_sha"][key]:
                raise ValueError("Batch full input or labels changed")
        for key in ("x", "residual", "ids", "labels"):
            value = cache[key]
            if value.device.type != "cpu" or cached.tensor_sha(value) != cache["tensor_sha"][key]:
                raise ValueError("Batch CPU cache changed")
        if any(not torch.isfinite(cache[key]).all() for key in ("x", "residual")):
            raise ValueError("Batch cache is nonfinite")
    # No decoder padding: each X/R was captured from its native complete graph.
    # The remaining linear, RMSNorm and head are independent per position.
    x = pad_sequence([entry["x"][0] for entry in caches], batch_first=True).to(parameter.device)
    residual = pad_sequence([entry["residual"][0] for entry in caches], batch_first=True).to(parameter.device)
    labels = pad_sequence([entry["labels"][0] for entry in caches], batch_first=True, padding_value=-100).to(parameter.device)
    hidden = F.linear(x, weight, runtime.branch.base.down_proj.bias)
    logits = runtime.model.lm_head(runtime.decoder.norm(hidden + residual))
    if not torch.isfinite(logits).all():
        raise ValueError("Batch tail logits are nonfinite")
    return logits, labels


def question_ce(logits, labels):
    if logits.shape[:2] != labels.shape or not torch.isfinite(logits).all():
        raise ValueError("Question logits must match finite full-sequence labels")
    targets = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), targets, ignore_index=-100, reduction="none")
    counts = targets.ne(-100).sum(dim=1)
    if not counts.gt(0).all() or not torch.isfinite(losses).all():
        raise ValueError("Each question needs finite answer/EOS CE")
    return losses.sum(dim=1) / counts


def question_kl(logits, labels, teacher):
    if len(teacher) != len(labels) or logits.shape[:2] != labels.shape or not torch.isfinite(logits).all():
        raise ValueError("Each replay question requires its original teacher")
    values = []
    for row, target, reference in zip(logits, labels, teacher):
        log_probs = cached._answer_log_probs(row.unsqueeze(0), target.unsqueeze(0))
        if not log_probs.shape[0] or reference.shape != log_probs.shape or not torch.isfinite(reference).all():
            raise ValueError("Replay teacher mask/shape or values differ")
        values.append(F.kl_div(log_probs, reference, reduction="batchmean", log_target=True))
    result = torch.stack(values)
    if not torch.isfinite(result).all():
        raise ValueError("Replay KL is nonfinite")
    return result


def _native_means(runtime, new, replay, original, candidate):
    """Independent native per-question gradients, all evaluated at the same W."""
    parameter = runtime.branch.base.down_proj.weight
    ce_losses, ce_grads, kl_losses, kl_grads, teachers = [], [], [], [], []
    for sample in new:
        cached._cancel(runtime)
        ids, labels = runtime._sample_tokens(sample)
        parameter.grad = None
        loss = runtime.model(ids, labels=labels, use_cache=False).loss
        loss.backward()
        ce_losses.append(loss.detach())
        ce_grads.append(parameter.grad.detach().float().clone())
    for sample in replay:
        cached._cancel(runtime)
        ids, labels = runtime._sample_tokens(sample)
        reference = cached._teacher_replay(runtime, parameter, original, candidate, ids, labels)
        parameter.grad = None
        logits = runtime.model(ids, use_cache=False).logits
        loss = F.kl_div(cached._answer_log_probs(logits, labels), reference, reduction="batchmean", log_target=True)
        (5 * loss).backward()
        kl_losses.append(loss.detach())
        kl_grads.append(parameter.grad.detach().float().clone())
        teachers.append(reference)
    return (torch.stack(ce_losses), torch.stack(ce_grads).mean(dim=0),
            torch.stack(kl_losses), torch.stack(kl_grads).mean(dim=0), teachers)


def audit_cached_batch(runtime, new_samples, replay_samples, batch_size):
    """One epoch, complete NEW coverage, explicit cyclic 1:1 REPLAY exposure.

    Same-W independent native per-question gradient means are the reference.
    Batch CE and KL still use two backward calls; clip/Adam/projection occur once
    per actual batch. Only private clone masters are updated; no publication.
    """
    if type(batch_size) is not int or batch_size not in (1, 2, 4):
        raise ValueError("Research batch size must be 1, 2 or4")
    samples = list(new_samples) + list(replay_samples)
    if (not new_samples or not replay_samples or len({row.get("id") for row in samples}) != len(samples)
            or any(row.get("purpose") != "training" for row in samples)):
        raise ValueError("Batch audit requires distinct fixed NEW/REPLAY TRAIN rows")
    for rows, role in ((new_samples, "new"), (replay_samples, "replay")):
        if any(json.loads(row.get("provenance", "{}")).get("evolution_role") != role for row in rows):
            raise ValueError("NEW/REPLAY roles cannot be silently mixed")
    with runtime.lock, torch.enable_grad():
        began = time.perf_counter()
        cached._cancel(runtime)
        model = runtime.model
        name, parameter = cached._target(runtime)
        tolerance = cached.TOLERANCES.get(str(parameter.dtype))
        if tolerance is None or runtime.branch.expert is not None:
            raise ValueError("Batch audit requires branchOFF and native FP32/BF16")
        original = parameter.detach().clone()
        flags = [(value, value.requires_grad, value.grad) for value in model.parameters()]
        modes = [(module, module.training) for module in model.modules()]
        rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone() if parameter.device.type == "mps" else None
        pointer = Path(runtime.store) / "active-core.json"
        pointer_before = pointer.read_bytes() if pointer.exists() else None
        core_before = getattr(runtime, "active_core", None)
        script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        caches, batches = [], []
        try:
            model.eval().requires_grad_(False)
            parameter.requires_grad_(True)
            eos = _eos(runtime)
            for sample in samples:
                _labels(*runtime._sample_tokens(sample), eos)
                caches.append(cached.capture_tail(runtime, sample))
            basis, basis_report = cached._capture_replay_subspace(runtime, parameter, replay_samples)
            if not 0 < basis.shape[1] < parameter.shape[1]:
                raise ValueError("Replay projection must leave a nontrivial complement")
            basis = basis.to(parameter.device)
            master = torch.nn.Parameter(original.float().clone())
            optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
            for start in range(0, len(new_samples), batch_size):
                cached._cancel(runtime)
                selected = list(new_samples[start:start + batch_size])
                replay_indices = [(start + offset) % len(replay_samples) for offset in range(len(selected))]
                replay = [replay_samples[index] for index in replay_indices]
                new_cache = caches[start:start + len(selected)]
                replay_cache = [caches[len(new_samples) + index] for index in replay_indices]
                candidate = master.detach().to(parameter.dtype).clone()
                with torch.no_grad():
                    parameter.copy_(candidate)
                native_ce, mean_ce_grad, native_kl, mean_kl_grad, teachers = _native_means(runtime, selected, replay, original, candidate)
                weight = candidate.clone().requires_grad_(True)
                logits, labels = _pack(runtime, selected, new_cache, weight)
                ce_values = question_ce(logits, labels)
                ce_values.mean().backward()
                ce_gradient = weight.grad.detach().float().clone()
                del logits
                with torch.no_grad():
                    teacher_logits, replay_labels = _pack(runtime, replay, replay_cache, original)
                    cached_teachers = [cached._answer_log_probs(row.unsqueeze(0), target.unsqueeze(0))
                                       for row, target in zip(teacher_logits, replay_labels)]
                    teacher_checks = [_check(value, reference, tolerance) for value, reference in zip(cached_teachers, teachers)]
                    del teacher_logits
                replay_logits, replay_labels = _pack(runtime, replay, replay_cache, weight)
                kl_values = question_kl(replay_logits, replay_labels, cached_teachers)
                (5 * kl_values.mean()).backward()
                total_gradient = weight.grad.detach().float().clone()
                checks = {"ce_per_question": _check(ce_values.detach(), native_ce, tolerance, "loss"),
                    "kl_per_question": _check(kl_values.detach(), native_kl, tolerance, "loss"),
                    "mean_ce_gradient": _check(ce_gradient, mean_ce_grad, tolerance, "gradient"),
                    "total_ce_5kl_gradient": _check(total_gradient, mean_ce_grad + mean_kl_grad, tolerance, "gradient")}
                # Separate KL-only contribution avoids subtracting rounded total
                # gradients: same-W replay-only graph is an audit, not an update.
                replay_weight = candidate.clone().requires_grad_(True)
                replay_only, labels_only = _pack(runtime, replay, replay_cache, replay_weight)
                (5 * question_kl(replay_only, labels_only, cached_teachers).mean()).backward()
                checks["mean_weighted_kl_gradient"] = _check(replay_weight.grad.float(), mean_kl_grad, tolerance, "gradient")
                # The reference update starts at exactly this FP32 master and
                # these existing moments, using same-W per-question mean grads.
                # It is one batch update, never a sequential-Adam reference.
                before = master.detach().clone()
                reference_master = torch.nn.Parameter(before.clone())
                reference_optimizer = torch.optim.Adam([reference_master], lr=1e-5, eps=1e-6)
                reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                reference_master.grad = (mean_ce_grad + mean_kl_grad).clone()
                optimizer.zero_grad(set_to_none=True)
                master.grad = total_gradient.clone()
                norm = torch.nn.utils.clip_grad_norm_([master], 1, error_if_nonfinite=True)
                reference_norm = torch.nn.utils.clip_grad_norm_([reference_master], 1, error_if_nonfinite=True)
                optimizer.step()
                reference_optimizer.step()
                cached._project_core_delta(master, original.float(), basis)
                cached._project_core_delta(reference_master, original.float(), basis)
                if not torch.isfinite(master).all():
                    raise ValueError("Batch projected master is nonfinite")
                checks["preclip_norm"] = _check(norm, reference_norm, tolerance, "gradient")
                checks["fp32_master"] = _check(master, reference_master, tolerance, "adam")
                checks["batch_update"] = _check(master - before, reference_master - before, tolerance, "adam", update=True)
                checks["cumulative_update"] = _check(master - original.float(), reference_master - original.float(), tolerance, "adam", update=True)
                checks["native_updated_weight"] = _check(master.to(parameter.dtype), reference_master.to(parameter.dtype), tolerance)
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    checks["adam_" + key] = _check(optimizer.state[master][key], reference_optimizer.state[reference_master][key], tolerance, "adam")
                batches.append({"optimizer_step": len(batches) + 1, "batch_size_actual": len(selected),
                    "new_ids": [row["id"] for row in selected], "replay_ids": [row["id"] for row in replay],
                    "new_answer_eos_tokens": [int(entry["labels"][:, 1:].ne(-100).sum()) for entry in new_cache],
                    "replay_answer_eos_tokens": [int(entry["labels"][:, 1:].ne(-100).sum()) for entry in replay_cache],
                    "ce": float(ce_values.detach().mean()), "kl": float(kl_values.detach().mean()),
                    "preclip_norm": float(norm), "checks": checks, "original_teacher_checks": teacher_checks,
                    "passed": all(check["passed"] for check in checks.values()) and all(check["passed"] for check in teacher_checks)})
                del replay_logits, replay_only, ce_values, kl_values, teachers, cached_teachers, weight, replay_weight
            cached._sync(parameter.device)
            actual_ids = [item for row in batches for item in row["new_ids"]]
            if actual_ids != [row["id"] for row in new_samples]:
                raise ValueError("Batch audit omitted or duplicated NEW rows")
            if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_sha:
                raise ValueError("Batch audit source changed")
            if cached._binding(runtime, parameter) != caches[0]["binding"]:
                raise ValueError("Batch graph/source/version changed before completion")
            report = {"protocol": PROTOCOL, "passed": all(row["passed"] for row in batches), "batch_size_requested": batch_size,
                "dtype": str(parameter.dtype), "device": str(parameter.device), "seed_scope": "caller-fixed synthetic TRAIN",
                "identity": runtime.identity, "core_version": core_before, "parameter": name,
                "torch_version": torch.__version__, "hardware": platform.platform(),
                "objective": "per-question answer/EOS tokenmean, then equal question mean; CE+5KL",
                "new_replay_exposure_ratio": "1:1 per update; explicit cyclic replay; final NEW remainder kept",
                "backward_order": ["CE questionmean backward", "5KL questionmean backward"],
                "optimizer": {"master_dtype": "float32", "kind": "Adam", "lr": 1e-5, "eps": 1e-6, "clip": 1,
                              "updates_per_batch": 1, "replay_projection_after_adam": True},
                "batches": batches, "cache_bytes": sum(sum(entry["bytes"].values()) for entry in caches),
                "sources": caches[0]["binding"]["sources"], "batch_source_sha256": script_sha,
                "audit_elapsed_seconds": time.perf_counter() - began,
                "tolerances": tolerance, "comparison_policy": "old-strict-gate-unchanged; standard-allclose-v1-diagnostic-only",
                "replay_svd": basis_report, "teacher_calls": 0, "generation_calls": 0,
                "gate_enabled": False, "publication": False,
                "scope": "Same-W native per-example gradient averages vs actual padded cached tail; not sequential Adam equivalence, 2B speed, memory or learning quality"}
        finally:
            with torch.no_grad():
                parameter.copy_(original)
            for value, flag, gradient in flags:
                value.requires_grad_(flag)
                value.grad = gradient
            for module, mode in modes:
                module.training = mode
            torch.set_rng_state(rng)
            if mps_rng is not None:
                torch.mps.set_rng_state(mps_rng)
            for entry in caches:
                entry.clear()
            caches.clear()
            cached._sync(parameter.device)
            if (not torch.equal(parameter, original) or runtime.branch.expert is not None
                    or getattr(runtime, "active_core", None) != core_before
                    or (pointer.read_bytes() if pointer.exists() else None) != pointer_before
                    or any(value.requires_grad != flag or value.grad is not gradient for value, flag, gradient in flags)):
                raise RuntimeError("Batch audit failed to restore original state")
        report["restored"] = True
        report["retained_cache_entries"] = len(caches)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    import sys
    import unittest
    sys.path.insert(0, str(cached.ROOT / "tests"))
    import test_cached_batch_training
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_cached_batch_training))
    print(json.dumps({"kind": "tiny_official_cached_batch_cpu", "passed": result.wasSuccessful(),
                      "tests_run": result.testsRun, "seed": 42, "reports": test_cached_batch_training.REPORTS,
                      "teacher_calls": 0, "generation_calls": 0, "real_2b_loaded": False}, indent=2, allow_nan=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
