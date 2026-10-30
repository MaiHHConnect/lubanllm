"""Isolated real-graph preparation for preservation-first identity experiments.

No product adoption, teacher API, source snapshots or per-row file hashing.
The resident frozen prefix is captured once; equivalence is checked on the real
graph before caches may be used for any subsequent optimizer experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

import torch
from torch.nn import functional as F

from identity_context_material import prepare_material
from train_luban_identity import material, runtime_for
from luban.evolution import _target, _generation_loaded, _answer_logits, _cancel
from luban.settings import private_json
from luban.retention import messages
from luban.core_update import apply_core_residual
from verify_cached_batch_training import _labels, _eos
from verify_training_operators import TOLERANCES, _comparison


def event(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def capture(runtime, row, *, generated=None):
    """One frozen full prefix; preserve all positions and native BF16 operators."""
    if row.get("purpose") != "training":
        raise ValueError("Only TRAIN may enter the optimization cache")
    _cancel(runtime)
    if generated is None:
        ids, labels = runtime._sample_tokens(row)
        _labels(ids, labels, _eos(runtime))
    else:
        # Preserve the observed generation tokens, including both turn and
        # generation terminators; do not decode/re-encode or call them gold.
        if (generated.get("id") != row["id"] or generated.get("complete") is not True
                or generated.get("branch_enabled") is not False or generated.get("reference_used") is not False):
            raise ValueError("Protection requires a complete closed-book trajectory")
        prefix = runtime._tokens(messages(row), "off", output_reserve=generated["max_new_tokens"])
        completion = generated["raw_completion_token_ids"]
        if not completion or any(type(token) is not int or token < 0 for token in completion):
            raise ValueError("Invalid generated protection tokens")
        end = runtime.model.generation_config.eos_token_id
        if completion[-1] not in (end if isinstance(end, (list, tuple)) else [end]):
            raise ValueError("Protection trajectory lacks the actual generation terminator")
        ids = torch.cat((prefix, torch.tensor([completion], device=prefix.device)), dim=1)
        labels = ids.clone()
        labels[:, :prefix.shape[1]] = -100
    values, handles = {}, []

    def take(key):
        def hook(module, arguments):
            if key in values:
                raise ValueError("Expected one last-layer execution")
            values[key] = arguments[0].detach().cpu().clone()
        return hook

    try:
        handles.append(runtime.decoder.layers[-1].post_attention_layernorm.register_forward_pre_hook(take("residual")))
        handles.append(runtime.branch.base.down_proj.register_forward_pre_hook(take("x")))
        with torch.no_grad():
            runtime.model(ids, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    if set(values) != {"x", "residual"}:
        raise ValueError("Incomplete frozen-prefix capture")
    width, inputs = _target(runtime)[1].shape
    if values["x"].shape != (*ids.shape, inputs) or values["residual"].shape != (*ids.shape, width):
        raise ValueError("Frozen-prefix dimensions differ")
    if not all(bool(torch.isfinite(v).all()) for v in values.values()):
        raise ValueError("Nonfinite frozen prefix")
    return {**values, "ids": ids.cpu(), "labels": labels.cpu()}


def tail_logits(runtime, cache, weight):
    x, residual = (cache[key].to(weight.device) for key in ("x", "residual"))
    hidden = F.linear(x, weight, runtime.branch.base.down_proj.bias) + residual
    return runtime.model.lm_head(runtime.decoder.norm(hidden))


def tail_loss(runtime, cache, weight):
    labels = cache["labels"].to(weight.device)
    logits = tail_logits(runtime, cache, weight)
    return F.cross_entropy(_answer_logits(logits, labels), labels[:, 1:][labels[:, 1:] != -100])


def residual_logits(runtime, cache, reference, delta):
    """Unmerged FP32 increment; the frozen BF16 projection remains unchanged."""
    x, residual = (cache[key].to(reference.device) for key in ("x", "residual"))
    original = F.linear(x, reference, runtime.branch.base.down_proj.bias)
    hidden = apply_core_residual(x, original, delta) + residual
    return runtime.model.lm_head(runtime.decoder.norm(hidden))


def equivalence(runtime, row, cache):
    parameter = _target(runtime)[1]
    parameter.grad = None
    ids, labels = runtime._sample_tokens(row)
    full = runtime.model(ids, labels=labels, use_cache=False)
    full.loss.backward()
    full_gradient = parameter.grad.detach().cpu().clone()
    full_loss = full.loss.detach().cpu()
    full_logits = _answer_logits(full.logits.detach(), labels).cpu()
    del full
    parameter.grad = None
    candidate = parameter.detach().clone().requires_grad_()
    cached = tail_logits(runtime, cache, candidate)
    selected = _answer_logits(cached, labels)
    loss = F.cross_entropy(selected, labels[:, 1:][labels[:, 1:] != -100])
    loss.backward()
    tolerance = TOLERANCES[str(parameter.dtype)]
    result = {}
    for key, actual, expected in (("logits", selected.detach().cpu(), full_logits),
                                  ("loss", loss.detach().cpu(), full_loss),
                                  ("gradient", candidate.grad.detach().cpu(), full_gradient)):
        result[key] = _comparison(actual, expected, atol=tolerance[key + "_atol"],
            rtol=tolerance["rtol"], relative_l2=tolerance["relative_l2"])
        result[key]["exact"] = bool(torch.equal(actual, expected))
    result["passed"] = all(v["passed"] for v in result.values())
    return result


def prepare(root):
    groups, migrations = prepare_material(material())
    private_json(root / "material.json", {"groups": groups, "migrations": migrations})
    began = time.perf_counter()
    report = {"protocol": "preservation-first-real-graph-preparation-v1", "seed": 42,
        "hardware": platform.platform(), "precision": "BF16 MPS / FP32 master planned",
        "temperature": 0, "thinking": False, "cache": "frozen prefix once; generation KV local to each answer",
        "training_steps": 0, "teacher_calls": 0, "production_weights_changed": False,
        "memory_measurement": "Darwin parent getrusage peak bytes and MPS driver bytes separately, overlapping",
        "scope": "same-task development and exposed regressions; no all-domain guarantee"}
    runtime = None
    try:
        torch.manual_seed(42)
        runtime = runtime_for(root)
        parameter = _target(runtime)[1]
        if runtime.active_core is not None or parameter.shape != (2048, 6144):
            raise ValueError("Study must start from original isolated 2B core")
        if any(p.requires_grad and p is not parameter for p in runtime.model.parameters()):
            raise ValueError("Only the final down projection may train")
        report.update(identity=runtime.identity, parameter=_target(runtime)[0], load_seconds=time.perf_counter()-began)
        audited = [groups["new_training"][0], groups["new_training"][31], groups["replay_training"][0]]
        caches = {}
        report["equivalence"] = []
        for row in audited:
            caches[row["id"]] = capture(runtime, row)
            check = equivalence(runtime, row, caches[row["id"]])
            report["equivalence"].append({"id": row["id"], **check})
            private_json(root / "preparation.json", report)
            event("equivalence", id=row["id"], passed=check["passed"])
            if not check["passed"]:
                raise RuntimeError("Real-graph cache equivalence failed; no optimization allowed")
        for index, row in enumerate(groups["new_training"] + groups["replay_training"]):
            if row["id"] not in caches:
                caches[row["id"]] = capture(runtime, row)
            if index % 10 == 0:
                event("capture", rows=index+1)
        matrix = torch.cat([caches[row["id"]]["x"][:, :-1][caches[row["id"]]["labels"][:, 1:] != -100].float()
                            for row in groups["replay_training"]])
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        tolerance = max(matrix.shape) * torch.finfo(torch.float32).eps * float(singular[0])
        rank = int((singular > tolerance).sum())
        if not 0 < rank < parameter.shape[1]:
            raise ValueError("No valid protected complement")
        basis = vh[:rank].T.contiguous()
        report["basis"] = {"rank": rank, "input_dimension": parameter.shape[1], "svd_tolerance": tolerance}
        torch.save({"caches": caches, "basis": basis, "reference": parameter.detach().cpu()}, root / "prepared.pt")
        observations = {}
        for key in ("new_training", "new_validation", "replay_training"):
            observations[key] = []
            for index, row in enumerate(groups[key]):
                observations[key].extend(_generation_loaded(runtime, [row]))
                if index % 10 == 0:
                    event("baseline", group=key, rows=index+1)
            private_json(root / "baseline.json", observations)
        report["status"] = "prepared_no_optimizer_updates"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-began,
                      parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if torch.backends.mps.is_available():
            report["mps_driver_bytes_before_release"] = torch.mps.driver_allocated_memory()
        if runtime is not None:
            runtime._release_model()
        private_json(root / "preparation.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库'):
        raise SystemExit("External research volume is not mounted")
    root = args.root.resolve()
    if not root.is_relative_to(Path('/Volumes/移动数据库/lubanllm/experiments')):
        raise SystemExit("Use a new directory on the research volume")
    root.mkdir(parents=True, exist_ok=False)
    prepare(root)
