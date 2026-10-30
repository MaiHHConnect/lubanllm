"""Research G1 native-B1 / cached same-role B1..4 step; no loader or publication.

Caller owns the frozen schedule, original reference, cache preparation, model
process exclusivity and checks. Clone master/Adam persist; Runtime is restored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import verify_cached_training as cached
from verify_cached_batch_training import _eos, _labels, question_ce, question_kl

PROTOCOL = "research-g1-alternating-ce-replay-kl-step-v1"


def _adam_state(optimizer, master):
    state = optimizer.state.get(master, {})
    if not state:
        return 0
    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("Method Adam state is incomplete or unexpected")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != master.shape
                or value.dtype != master.dtype or value.device != master.device or not torch.isfinite(value).all()):
            raise ValueError("Method Adam moments are malformed/nonfinite")
    step = state["step"]
    if (not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all()
            or float(step) < 1 or float(step) != int(float(step))):
        raise ValueError("Method Adam step is not a finite positive integer")
    return int(float(step))


def _tail(runtime, rows, caches, weight, timing):
    began = time.perf_counter()
    _, parameter = cached._target(runtime)
    binding, eos = cached._binding(runtime, parameter), _eos(runtime)
    selected = []
    for row in rows:
        cached._cancel(runtime)
        entry = caches[row["id"]]
        if (entry["binding"] != binding
                or entry["sample_sha"] != hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()):
            raise ValueError("Method cache graph/source/sample binding changed")
        ids, labels = runtime._sample_tokens(row)
        _labels(ids, labels, eos)
        for key, value in (("ids", ids), ("labels", labels)):
            if cached.tensor_sha(value) != entry["tensor_sha"][key]:
                raise ValueError("Method full input/labels changed")
        for key in ("x", "residual", "ids", "labels"):
            value = entry[key]
            if value.device.type != "cpu" or cached.tensor_sha(value) != entry["tensor_sha"][key]:
                raise ValueError("Method full CPU cache changed")
        if any(not torch.isfinite(entry[key]).all() for key in ("x", "residual")):
            raise ValueError("Method cache is nonfinite")
        selected.append(entry)
    timing["validation_seconds"] += time.perf_counter() - began
    cached._sync(parameter.device)
    began = time.perf_counter()
    x = pad_sequence([entry["x"][0] for entry in selected], batch_first=True).to(parameter.device)
    residual = pad_sequence([entry["residual"][0] for entry in selected], batch_first=True).to(parameter.device)
    labels = pad_sequence([entry["labels"][0] for entry in selected], batch_first=True, padding_value=-100).to(parameter.device)
    cached._sync(parameter.device)
    timing["transfer_seconds"] += time.perf_counter() - began
    logits = runtime.model.lm_head(runtime.decoder.norm(F.linear(x, weight, runtime.branch.base.down_proj.bias) + residual))
    if not torch.isfinite(logits).all():
        raise ValueError("Method full logits are nonfinite, including padding")
    return logits, labels


def train_method_step(runtime, samples, replays, master, optimizer, reference, basis, *, method, ce_role, caches=None):
    """One same-role CE batch + equally weighted REPLAY KL5; G1 only.

    Original product schedule is 0-based even NEW CE, odd REPLAY CE, each with
    additional REPLAY KL. Native supports only B1; cached supports actual1..4
    rows (3 is a B4 remainder). Caller must flush at exact check boundaries.
    On any error caller aborts the study: clone optimizer/master are not a
    transactional resumable product session. Runtime W/flags/grad/modes/RNG
    are restored even when a clone update has already occurred.
    """
    if (method not in ("native", "cache") or ce_role not in ("new", "replay")
            or not isinstance(samples, (list, tuple)) or not isinstance(replays, (list, tuple))
            or not 1 <= len(samples) <= 4 or len(replays) != len(samples)
            or method == "native" and len(samples) != 1
            or method == "cache" and not isinstance(caches, dict)):
        raise ValueError("Method requires nativeB1 or cached same-role actualB1..4 and equal replay count")
    for rows, role in ((samples, ce_role), (replays, "replay")):
        if any(row.get("purpose") != "training" or json.loads(row.get("provenance", "{}")).get("evolution_role") != role for row in rows):
            raise ValueError("Method must preserve fixed TRAIN CE/REPLAY roles")
    with runtime.lock, torch.enable_grad():
        cached._cancel(runtime)
        _, parameter = cached._target(runtime)
        if (parameter.dtype not in (torch.float32, torch.bfloat16) or runtime.branch.expert is not None
                or not isinstance(master, torch.nn.Parameter) or master.dtype != torch.float32
                or master.shape != parameter.shape or master.device != parameter.device
                or not isinstance(reference, torch.Tensor) or reference.requires_grad
                or reference.shape != parameter.shape or reference.dtype != torch.float32 or reference.device != parameter.device
                or not isinstance(basis, torch.Tensor) or basis.requires_grad or basis.ndim != 2 or basis.shape[0] != parameter.shape[1]
                or not 0 < basis.shape[1] < parameter.shape[1] or basis.dtype != torch.float32 or basis.device != parameter.device
                or type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1
                or len(optimizer.param_groups[0]["params"]) != 1 or optimizer.param_groups[0]["params"][0] is not master
                or any(not torch.isfinite(value).all() for value in (parameter, master, reference, basis))):
            raise ValueError("Method requires bound finite native tensor, FP32 clone/Adam/reference/replay basis and branchOFF")
        group = optimizer.param_groups[0]
        if any(group[key] != value for key, value in (("lr", 1e-5), ("eps", 1e-6), ("weight_decay", 0), ("betas", (.9, .999)), ("amsgrad", False))):
            raise ValueError("Method must preserve original G1 Adam settings")
        previous_step = _adam_state(optimizer, master)
        model = runtime.model
        original = parameter.detach().clone()
        flags = [(value, value.requires_grad, value.grad) for value in model.parameters()]
        modes = [(module, module.training) for module in model.modules()]
        rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone() if parameter.device.type == "mps" else None
        pointer = Path(runtime.store) / "active-core.json"
        pointer_before = pointer.read_bytes() if pointer.exists() else None
        core_before = getattr(runtime, "active_core", None)
        source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        began = time.perf_counter()
        timing = {key: 0.0 for key in ("validation_seconds", "transfer_seconds", "ce_seconds", "kl_seconds", "update_seconds")}
        try:
            model.eval().requires_grad_(False)
            parameter.requires_grad_(True)
            binding = cached._binding(runtime, parameter)
            eos = _eos(runtime)
            for row in list(samples) + list(replays):
                _labels(*runtime._sample_tokens(row), eos)
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = None
            with torch.no_grad():
                parameter.copy_(master)
            candidate = parameter.detach().clone()
            timing["preparation_seconds"] = time.perf_counter() - began
            phase = time.perf_counter()
            if method == "native":
                ids, labels = runtime._sample_tokens(samples[0])
                ce = model(ids, labels=labels, use_cache=False).loss
                ce.backward()
            else:
                weight = candidate.clone().requires_grad_(True)
                logits, labels = _tail(runtime, samples, caches, weight, timing)
                ce = cached._ce(logits, labels) if len(samples) == 1 else question_ce(logits, labels).mean()
                ce.backward()
                del logits
            if not torch.isfinite(ce):
                raise ValueError("Method CE is nonfinite")
            ce_value = float(ce.detach().cpu())
            cached._sync(parameter.device)
            timing["ce_seconds"] = time.perf_counter() - phase
            cached._cancel(runtime)
            phase = time.perf_counter()
            if method == "native":
                ids, labels = runtime._sample_tokens(replays[0])
                teacher = cached._teacher_replay(runtime, parameter, reference.to(parameter.dtype), candidate, ids, labels)
                logits = model(ids, use_cache=False).logits
                kl = F.kl_div(cached._answer_log_probs(logits, labels), teacher, reduction="batchmean", log_target=True)
            else:
                with torch.no_grad():
                    teacher_logits, labels = _tail(runtime, replays, caches, reference.to(parameter.dtype), timing)
                    teacher = [cached._answer_log_probs(row.unsqueeze(0), label.unsqueeze(0)) for row, label in zip(teacher_logits, labels)]
                    del teacher_logits
                logits, labels = _tail(runtime, replays, caches, weight, timing)
                kl = question_kl(logits, labels, teacher).mean()
            if not torch.isfinite(kl):
                raise ValueError("Method replay KL is nonfinite")
            (5 * kl).backward()
            kl_value = float(kl.detach().cpu())
            cached._sync(parameter.device)
            timing["kl_seconds"] = time.perf_counter() - phase
            cached._cancel(runtime)
            gradient = parameter.grad if method == "native" else weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                raise ValueError("Method gradient missing or nonfinite")
            phase = time.perf_counter()
            master.grad = gradient.detach().float().clone()
            norm = torch.nn.utils.clip_grad_norm_([master], 1, error_if_nonfinite=True)
            optimizer.step()
            cached._project_core_delta(master, reference, basis)
            with torch.no_grad():
                parameter.copy_(master)
            if not torch.isfinite(parameter).all() or _adam_state(optimizer, master) != previous_step + 1:
                raise ValueError("Method native weight/Adam state is invalid after update")
            cached._sync(parameter.device)
            timing["update_seconds"] = time.perf_counter() - phase
            phase = time.perf_counter()
            if cached._binding(runtime, parameter) != binding or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_sha:
                raise ValueError("Method frozen graph/source/version changed")
            timing["final_guard_seconds"] = time.perf_counter() - phase
            timing["compute_seconds"] = timing["ce_seconds"] + timing["kl_seconds"] + timing["update_seconds"] - timing["validation_seconds"] - timing["transfer_seconds"]
            report = {"protocol": PROTOCOL, "method": method, "ce_role": ce_role, "actual_batch_size": len(samples),
                "ce_ids": [row["id"] for row in samples], "replay_kl_ids": [row["id"] for row in replays],
                "optimizer_steps": 1, "new_ce_exposures": len(samples) if ce_role == "new" else 0,
                "replay_ce_exposures": len(samples) if ce_role == "replay" else 0, "replay_kl_exposures": len(replays),
                "ce": ce_value, "kl": kl_value, "loss": ce_value + 5 * kl_value, "preclip_norm": float(norm.cpu()),
                "timing": timing, "checks_seconds": 0.0, "checks_run": False, "sources": binding["sources"],
                "method_source_sha256": source_sha,
                "teacher_calls": 0, "generation_calls": 0, "publication": False, "gate_enabled": False,
                "scope": "one actual clone update; caller controls continuous Adam/schedule/checks; B>1 is not old sequentialAdam equivalence"}
        finally:
            restoring = time.perf_counter()
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
            cached._sync(parameter.device)
            if (not torch.equal(parameter, original) or runtime.branch.expert is not None
                    or getattr(runtime, "active_core", None) != core_before
                    or (pointer.read_bytes() if pointer.exists() else None) != pointer_before
                    or any(value.requires_grad != flag or value.grad is not gradient for value, flag, gradient in flags)):
                raise RuntimeError("Method failed to restore Runtime state; study must abort")
            timing["restore_seconds"] = time.perf_counter() - restoring
            timing["step_elapsed_seconds"] = time.perf_counter() - began
        report["runtime_restored"] = True
        return report
