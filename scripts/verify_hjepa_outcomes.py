"""Offline outcome-target ablation on hash-bound archived real 2B features.

This is supervised result prediction, NOT an H-JEPA reproduction or a fresh
generalization test. It never constructs Runtime or publishes model weights.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import torch
from torch.nn import functional as F

from luban.calibration import SPLITS, evaluate_router, split_route_samples
from luban.decision import FastDecisionHead
from luban.training_checkpoint import _encoded
from verify_router_budget_counts import labelled_rows, nested_routes

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 101, 202)
STEPS = 160
OUTCOMES = ("success", "complete_wrong", "incomplete")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_json(path, expected):
    if sha(path) != expected:
        raise ValueError("archived source hash mismatch: " + str(path))
    return json.loads(Path(path).read_text())


def outcome(answer):
    if any(type(answer.get(k)) is not bool for k in ("passed", "complete")):
        raise ValueError("outcome requires actual boolean pass/completion")
    if answer["passed"] and not answer["complete"]:
        raise ValueError("incomplete answer cannot be successful")
    return 0 if answer["passed"] else 1 if answer["complete"] else 2


def outcome_routes(logits):
    # Merge both failure probabilities BEFORE the existing route calibration.
    return torch.stack((logits[:, 0], torch.full_like(logits[:, 0], -torch.inf),
                        torch.logsumexp(logits[:, 1:], dim=1)), dim=1)


def binary_equivalence(features, state, targets):
    """Only two supported actions: BCE and two-class CE are the same objective."""
    weight = state["projection.weight"].double()
    bias = state["projection.bias"].double()
    score = F.linear(features.double(), (weight[2] - weight[0]).unsqueeze(0),
                     (bias[2] - bias[0]).reshape(1)).squeeze(1)
    score.requires_grad_(True)
    logits = torch.stack((torch.zeros_like(score), score), dim=1)
    labels = (targets == 2).long()
    ce = F.cross_entropy(logits, labels)
    bce = F.binary_cross_entropy_with_logits(score, labels.double())
    grad_ce = torch.autograd.grad(ce, score, retain_graph=True)[0]
    grad_bce = torch.autograd.grad(bce, score)[0]
    return {"loss_abs_error": abs(float(ce.detach() - bce.detach())),
            "gradient_max_abs_error": float((grad_ce - grad_bce).abs().max()),
            "probability_max_abs_error": float((logits.softmax(-1)[:, 1] - score.sigmoid()).abs().max().detach()),
            "same_decisions": bool(torch.equal(logits.argmax(-1), (score > 0).long()))}


def load_inputs():
    diagnosis_path = ROOT / "research/router_readonly_diagnosis_v1.json"
    descriptor_path = ROOT / "research/router_budget_counts_v2_result.json"
    diagnosis = json.loads(diagnosis_path.read_text())
    descriptor = json.loads(descriptor_path.read_text())
    raw_path = Path(descriptor["result_path"])
    raw = checked_json(raw_path, descriptor["result_sha256"])
    if raw["teacher_calls"] != 0 or raw["production_unchanged"] is not True:
        raise ValueError("historical isolation contract changed")
    rows = labelled_rows(raw["protocol"]["descriptor"]["pool"], raw["qualification"]["answers"])
    answers = {a["id"]: a for a in raw["qualification"]["answers"]}
    hashes = {str(p): sha(p) for p in (diagnosis_path, descriptor_path, raw_path)}
    trials = []
    for candidate in diagnosis["candidate_audit"]:
        path = Path(candidate["checkpoint_path"])
        if sha(path) != candidate["checkpoint_sha256"]:
            raise ValueError("checkpoint hash changed")
        hashes[str(path)] = sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        records = {"splits": nested_routes(rows, candidate["count"])}
        if hashlib.sha256(_encoded(records).encode()).hexdigest() != saved["binding"]["records_sha256"]:
            raise ValueError("checkpoint records binding changed")
        if (saved["binding"]["core_version"] is not None or saved["binding"]["scope"] != "research"
                or saved["binding"]["identity"] != raw["protocol"]["identity"]):
            raise ValueError("archived base/scope binding changed")
        splits = split_route_samples([r for name in SPLITS for r in records["splits"][name]], FastDecisionHead.actions)
        payload = saved["payload"]
        outcome_targets, audits = {}, {}
        for name in SPLITS:
            features, targets = payload["features"][name], payload["targets"][name]
            expected = torch.tensor([FastDecisionHead.actions.index(r["route_action"]) for r in splits[name]])
            if (features.shape != (len(expected), 2048) or features.dtype != torch.float32
                    or not torch.isfinite(features).all() or not torch.equal(targets, expected)):
                raise ValueError("feature/label order or shape mismatch")
            labels = [outcome(answers[r["id"]]) for r in splits[name]]
            outcome_targets[name] = torch.tensor(labels)
            audits[name] = {"ids": [r["id"] for r in splits[name]],
                           "outcome_counts": {key: labels.count(i) for i, key in enumerate(OUTCOMES)},
                           "features_sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest()}
        if set(outcome_targets["train"].tolist()) != {0, 1, 2}:
            raise ValueError("TRAIN lacks an observed outcome class; do not invent labels")
        trials.append({"count": candidate["count"], "payload": payload, "outcomes": outcome_targets, "audit": audits})
    if [t["count"] for t in trials] != [3, 6, 12]:
        raise ValueError("fixed archived comparison counts changed")
    return trials, hashes, raw["hardware"]


def fit(features, targets, initial):
    head = torch.nn.Linear(2048, 3)
    head.load_state_dict(initial)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.005)
    started = time.perf_counter()
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(features["train"]), targets["train"])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite research loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
        optimizer.step()
    with torch.no_grad():
        logits = {name: head(value) for name, value in features.items()}
    return logits, time.perf_counter() - started


def route_diagnostics(result):
    test = result["split_metrics"]["test"]
    effective = [d if c >= result["threshold"] else 0 for d, c in zip(test["decisions"], test["confidence"])]
    return {"test_route_accuracy": result["test_accuracy"],
            "candidate_gate_passed": result["calibrated"],
            "local_selection_precision": result["action_metrics"]["local_core"]["precision"],
            "teacher_failure_recall": result["action_metrics"]["teacher"]["recall"],
            "diagnostic_effective_unsafe_local_count": sum(d == 0 and y == 2 for d, y in zip(effective, test["targets"])),
            "diagnostic_effective_local_count": effective.count(0)}


def run(output):
    output = Path(output).absolute()
    external = Path("/Volumes/移动数据库")
    if not external.is_mount() or output.parent != external / "lubanllm/experiments" or output.exists():
        raise ValueError("requires mounted external volume and a fresh direct experiments directory")
    output.mkdir()
    protocol = {"schema_version": 1, "study": "hjepa-outcome-ablation-v1",
        "scope": "archived real 2B features; supervised outcome ablation; NOT H-JEPA reproduction",
        "counts_per_route_class": [3, 6, 12], "seeds": list(SEEDS), "steps": STEPS,
        "optimizer": "Adam lr0.005 clipnorm1; same initialization/6147 parameters per paired trial",
        "targets": list(OUTCOMES), "calibration": "CAL temperature; DEV threshold; unchanged existing gate",
        "heldout": "already observed historical TEST; no fresh generalization or adoption claim",
        "teacher_calls": 0, "backbone_loads": 0, "product_publication": False,
        "source_sha256": {str(ROOT / name): sha(ROOT / name) for name in (
            "scripts/verify_hjepa_outcomes.py", "scripts/verify_router_budget_counts.py",
            "scripts/verify_automatic_batches.py", "src/luban/calibration.py", "src/luban/decision.py",
            "src/luban/generation_checks.py", "src/luban/retention.py", "src/luban/training_checkpoint.py")}}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    started = time.perf_counter()
    torch.set_num_threads(1)
    trials, hashes, hardware = load_inputs()
    report = {"protocol": protocol, "input_sha256": hashes, "original_backbone_hardware": hardware,
        "current_hardware": {"platform": platform.platform(), "device": "cpu", "dtype": "FP32 heads; FP64 algebra check",
            "torch_version": torch.__version__, "memory_method": "resource.ru_maxrss process high-water RSS; Darwin bytes"},
        "trials": [], "equivalence": [], "data_audit": []}
    for trial in trials:
        payload, count = trial["payload"], trial["count"]
        report["data_audit"].append({"count": count, "splits": trial["audit"]})
        proof = binary_equivalence(payload["features"]["test"], payload["router"], payload["targets"]["test"])
        if any(proof[k] > 1e-10 for k in ("loss_abs_error", "gradient_max_abs_error", "probability_max_abs_error")) or not proof["same_decisions"]:
            raise ValueError("binary reparameterization proof failed")
        report["equivalence"].append({"count": count, **proof})
        for seed in SEEDS:
            torch.manual_seed(seed)
            initial = copy.deepcopy(torch.nn.Linear(2048, 3).state_dict())
            baseline, baseline_seconds = fit(payload["features"], payload["targets"], initial)
            predicted, outcome_seconds = fit(payload["features"], trial["outcomes"], initial)
            mapped = {name: outcome_routes(value) for name, value in predicted.items()}
            baseline_eval = evaluate_router(baseline, payload["targets"], FastDecisionHead.actions, ["local_core", "teacher"])
            outcome_eval = evaluate_router(mapped, payload["targets"], FastDecisionHead.actions, ["local_core", "teacher"])
            record = {"count": count, "seed": seed, "baseline": route_diagnostics(baseline_eval),
                "outcome": route_diagnostics(outcome_eval), "baseline_full": baseline_eval, "outcome_full": outcome_eval,
                "head_optimization_seconds": {"baseline": baseline_seconds, "outcome": outcome_seconds},
                "outcome_accuracy": {name: float((predicted[name].argmax(-1) == trial["outcomes"][name]).float().mean()) for name in SPLITS}}
            report["trials"].append(record)
            print(json.dumps({k: record[k] for k in ("count", "seed", "baseline", "outcome")}), flush=True)
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise ValueError("archived input changed during study")
    if any(sha(path) != expected for path, expected in protocol["source_sha256"].items()):
        raise ValueError("research execution source changed during study")
    report.update(elapsed_seconds=time.perf_counter() - started,
                  process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  archived_inputs_unchanged=True, product_accepted=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Research complete; no production model published.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
