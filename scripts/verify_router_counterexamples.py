"""Research-only paired test of two previously omitted real short TRAIN failures.

Freeze new operand-disjoint heldout before generation. Four-layer linear graph,
optimizer and adoption gate stay unchanged. No product samples or weights written.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import tempfile
import time

import torch
from torch import nn

from luban.calibration import SPLITS, evaluate_router
from luban.decision import FastDecisionHead
from luban.retention import messages, normalized_context, provenance, verified_source
from verify_automatic_batches import measured
from verify_hjepa_outcomes import load_inputs
from verify_router_budget_counts import integer_gold, labelled_rows
from verify_router_coverage_study import label_split, model_binding, sha
from diagnose_field_training import production_snapshot

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 101, 202)
DATA_SEED = 202610102
COUNTS = {"development": 24, "calibration": 24, "test": 48}
OPS = ("add", "subtract", "sum_three", "multiply")
BANDS = {"small": (1, 29), "medium": (31, 99), "long": (1001, 9999)}
SOURCES = (
    "scripts/verify_router_counterexamples.py", "scripts/verify_hjepa_outcomes.py",
    "scripts/verify_router_budget_counts.py", "scripts/verify_automatic_batches.py",
    "scripts/verify_router_coverage_study.py", "scripts/diagnose_field_training.py",
    "src/luban/runtime.py", "src/luban/evolution.py", "src/luban/decision.py",
    "src/luban/calibration.py", "src/luban/backbone.py", "src/luban/dynamic.py",
    "src/luban/coldpaging.py", "src/luban/context.py", "src/luban/process_lock.py",
    "src/luban/memory.py", "src/luban/settings.py", "src/luban/retention.py",
    "src/luban/training_checkpoint.py", "src/luban/generation_checks.py",
    "src/luban/core_publication.py", "src/luban/external_brain.py", "src/luban/backups.py")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def select_arms(train_rows, baseline_ids):
    """TRAIN-only selection; never accepts DEV/CAL/TEST evidence."""
    if any(provenance(r).get("split") != "train" or r.get("purpose") != "training"
           or not verified_source(r) for r in train_rows):
        raise ValueError("selector accepts only verified TRAIN")
    rows = {r["id"]: r for r in train_rows}
    if len(rows) != len(train_rows) or len(set(baseline_ids)) != 24:
        raise ValueError("unique TRAIN and fixed baseline24 required")
    def action(identifier):
        return provenance(rows[identifier])["route_action"]
    if Counter(action(i) for i in baseline_ids) != Counter(local_core=12, teacher=12):
        raise ValueError("baseline12/12 changed")
    short = sorted(r["id"] for r in train_rows if provenance(r)["route_action"] == "teacher"
                   and provenance(r)["family_group"].endswith("-short") and r["id"] not in baseline_ids)
    long = sorted(r["id"] for r in train_rows if provenance(r)["route_action"] == "teacher"
                  and provenance(r)["family_group"].endswith("-long") and r["id"] not in baseline_ids)
    if len(short) != 2 or len(long) < 2:
        raise ValueError("requires exactly two archived omitted short failures and two long controls")
    removed = sorted(i for i in baseline_ids if action(i) == "teacher")[-2:]
    arms = {"baseline24": sorted(baseline_ids),
        "replace_short24": sorted([i for i in baseline_ids if i not in removed] + short),
        "add_short26": sorted(list(baseline_ids) + short),
        "add_long26": sorted(list(baseline_ids) + long[:2])}
    return arms, {"short_failures": short, "long_failure_controls": long[:2], "replaced_long_failures": removed}


def operand_key(operation, args):
    return (operation, tuple(sorted(args) if operation in {"add", "sum_three", "multiply"} else args))


def fresh_pool(excluded_rows):
    """Fixed operators/number bands, unique contexts and numeric problems across splits."""
    rng, rows = random.Random(DATA_SEED), []
    contexts = {normalized_context(r) for r in excluded_rows}
    operands = set()
    for r in excluded_rows:
        check = provenance(r).get("checker", {})
        if check.get("operation") in OPS:
            operands.add(operand_key(check["operation"], check["args"]))
    for split, count in COUNTS.items():
        for operation in OPS:
            for band, (low, high) in BANDS.items():
                for index in range(count // (len(OPS) * len(BANDS))):
                    for attempt in range(10000):
                        args = [rng.randint(low, high) for _ in range(3 if operation == "sum_three" else 2)]
                        key = operand_key(operation, args)
                        symbol = {"add": "+", "subtract": "-", "sum_three": "+", "multiply": "×"}[operation]
                        prompt = "计算" + symbol.join(map(str, args)) + "。只输出结果整数，不要解释。"
                        row = {"id": f"counterexample-v1-{split}-{operation}-{band}-{index}",
                            "prompt": prompt, "source": "synthetic", "approved": 1, "training_allowed": 1,
                            "purpose": "validation", "provenance": {"messages": [{"role": "user", "content": prompt}],
                                "split": split, "operation": operation, "numeric_band": band,
                                "route_only": True, "program_verified": True, "human_verified": False,
                                "rights_basis": "project-authored exact finite integer arithmetic; no teacher",
                                "verification_evidence": "Python exact integer construction with archived args and gold",
                                "checker": {"operation": operation, "args": args}}}
                        if key not in operands and normalized_context(row) not in contexts:
                            break
                    else:
                        raise ValueError("fixed candidate generator exhausted; no adaptive heldout")
                    expected = integer_gold(row["provenance"]["checker"])
                    row["answer"] = str(expected)
                    row["provenance"]["checker"]["expected"] = expected
                    row["provenance"]["generation_check"] = {"type": "semantic", "validator": "integer",
                        "expected": expected, "max_new_tokens": 512}
                    rows.append(row); contexts.add(normalized_context(row)); operands.add(key)
    return rows


def diagnostics(result):
    test = result["split_metrics"]["test"]
    effective = [d if c >= result["threshold"] else 0 for d, c in zip(test["decisions"], test["confidence"])]
    return {"route_accuracy": result["test_accuracy"], "gate_passed": result["calibrated"],
        "raw_unsafe_local": sum(d == 0 and y == 2 for d, y in zip(test["decisions"], test["targets"])),
        "effective_unsafe_local": sum(d == 0 and y == 2 for d, y in zip(effective, test["targets"])),
        "effective_correct_local": sum(d == 0 and y == 0 for d, y in zip(effective, test["targets"])),
        "effective_local_count": effective.count(0), "effective_route_accuracy":
            sum(d == y for d, y in zip(effective, test["targets"])) / len(effective),
        "selected_local_precision": result["action_metrics"]["local_core"]["precision"]}


def beneficial(candidate, baseline):
    """Reject apparent safety gains that merely give up previously correct local answers."""
    return (candidate["route_accuracy"] > baseline["route_accuracy"]
        and candidate["effective_unsafe_local"] < baseline["effective_unsafe_local"]
        and candidate["effective_correct_local"] >= baseline["effective_correct_local"])


def prepare():
    trials, hashes, _ = load_inputs()
    descriptor = json.loads((ROOT / "research/router_budget_counts_v2_result.json").read_text())
    raw = json.loads(Path(descriptor["result_path"]).read_text())
    old = labelled_rows(raw["protocol"]["descriptor"]["pool"], raw["qualification"]["answers"])
    train = [r for r in old if provenance(r)["split"] == "train"]
    trial = trials[-1]
    arms, selection = select_arms(train, trial["audit"]["train"]["ids"])
    reserved = ROOT / "research/router_coverage_counts_v3_draft.json"
    hashes[str(reserved)] = sha(reserved)
    excluded = old + raw["protocol"]["descriptor"]["old44"] + json.loads(reserved.read_text())["pool"]
    return raw, trial, train, arms, selection, fresh_pool(excluded), hashes


def run(output, model_path, storage):
    from luban.evolution import _generation
    from luban.memory import Memory
    from luban.runtime import Runtime
    from luban.settings import Settings
    from luban.process_lock import ModelProcessLock

    output, model_path, storage = map(lambda p: Path(p).absolute(), (output, model_path, storage))
    if (not Path("/Volumes/移动数据库").is_mount() or not Path("/Volumes/Untitled").is_mount()
            or output.parent != storage / "experiments" or output.exists()
            or any(p.is_symlink() for p in (output, model_path, storage))):
        raise ValueError("requires mounted volumes and fresh direct experiment directory")
    started = time.perf_counter()
    torch.set_num_threads(1)
    raw, trial, train, arms, selection, pool, inputs = prepare()
    identity = raw["protocol"]["identity"]
    before = production_snapshot(storage)
    binding = model_binding(model_path, identity)
    protocol = {"study": "router-short-counterexamples-v1", "identity": identity, "core_version": None,
        "train_arms": arms, "selection": selection, "fresh_heldout": pool, "seeds": list(SEEDS),
        "architecture": "unchanged real first4 uniform mean + Linear2048→3,6147 parameters",
        "training": "same initialization; Adam lr.005 clip1 full-batch160; no core/expert training",
        "qualification": "new heldout greedy512/thinkingOFF/branchOFF/referenceFalse once per context",
        "budget": "1200s generation phase; deadline between examples, never alter token budget or retry",
        "primary": "replace_short24 vs baseline24; add_short26 vs add_long26 is quantity control",
        "benefit_rule": "all3 seeds: strictly higher raw route agreement, fewer effective unsafe local, no fewer effective correct local",
        "adoption_rule": "benefit rule plus all3 unchanged original gates; even a pass never publishes in this study",
        "family_scope": "new contexts and canonical operand tuples disjoint; arithmetic operator/template families overlap explicitly",
        "heldout_scope": "fresh96 frozen before model outputs, weights frozen before heldout generation; TEST48 once; seeds share the same48",
        "teacher_quality_measured": False, "teacher_calls": 0, "production_publication": False,
        "source_sha256": {n: sha(ROOT / n) for n in SOURCES}, "input_sha256": inputs,
        "model_binding": binding, "production_before": before}
    output.mkdir()
    (output / "protocol.json").write_text(encoded(protocol))
    report = {"protocol": protocol, "state": "running", "teacher_calls": 0, "product_accepted": False,
              "qualification": [], "trials": [], "publication": False}
    def save():
        (output / "report.json").write_text(encoded(report))
    save()
    runtime, models, old_features, fresh_features = None, {}, {}, {}
    try:
        with tempfile.TemporaryDirectory(prefix="luban-counterexamples-") as home:
            private = output / "private-runtime"
            settings = Settings(home)
            settings.data.update(storage_root=str(private), model_path=str(model_path), local_thinking="off")
            runtime = Runtime(settings, Memory(private), research=True)
            runtime.load(paging=False)
            if (runtime.identity != identity or runtime.active_core is not None or runtime.branch.expert is not None
                    or runtime.external_binding() is not None):
                raise ValueError("exact original core/OFF/noReference required")
            versions = {name: tensor._version for name, tensor in runtime.model.named_parameters()}
            report["hardware"] = {"platform": platform.platform(), "device": runtime.device,
                "precision": str(runtime.dtype), "torch": torch.__version__}
            rows_by_id = {r["id"]: r for r in train}
            baseline_ids = trial["audit"]["train"]["ids"]
            archived = {i: trial["payload"]["features"]["train"][j] for j, i in enumerate(baseline_ids)}
            with measured(runtime) as resources:
                for identifier in sorted({i for ids in arms.values() for i in ids}):
                    tokens = runtime._tokens(messages(rows_by_id[identifier]), thinking="off", shallow=True)
                    feature = runtime.shallow(tokens).float().mean(1).cpu()[0]
                    if feature.shape != (2048,) or not torch.isfinite(feature).all():
                        raise ValueError("invalid real graph TRAIN state")
                    old_features[identifier] = feature
            report["train_feature_measurement"] = resources
            equivalence = {i: float((old_features[i] - archived[i]).norm() / archived[i].norm().clamp_min(1e-8))
                           for i in baseline_ids}
            if max(equivalence.values()) > .01:
                raise ValueError("actual original graph differs from archived features beyond fixed BF16 tolerance")
            report["archived_train_feature_relative_l2"] = equivalence
            # Freeze every candidate BEFORE any fresh heldout answer or label exists.
            for seed in SEEDS:
                torch.manual_seed(seed)
                initial = copy.deepcopy(nn.Linear(2048, 3).state_dict())
                for name, identifiers in arms.items():
                    features = torch.stack([old_features[i] for i in identifiers])
                    labels = torch.tensor([FastDecisionHead.actions.index(provenance(rows_by_id[i])["route_action"])
                                           for i in identifiers])
                    # Keep weights before any heldout outputs; same original Linear/Adam recipe.
                    head = nn.Linear(2048, 3); head.load_state_dict(initial)
                    optimizer = torch.optim.Adam(head.parameters(), lr=.005)
                    fit_started = time.perf_counter()
                    for _ in range(160):
                        optimizer.zero_grad(set_to_none=True)
                        loss = nn.functional.cross_entropy(head(features), labels)
                        if not torch.isfinite(loss):
                            raise ValueError("nonfinite head training")
                        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1); optimizer.step()
                    models[(seed, name)] = head.eval()
                    weight_path = output / f"head-{seed}-{name}.pt"
                    torch.save(head.state_dict(), weight_path)
                    report.setdefault("frozen_before_heldout", []).append({"seed": seed, "arm": name,
                        "weights_sha256": sha(weight_path), "train_seconds": time.perf_counter() - fit_started,
                        "initial_state_sha256": hashlib.sha256(b"".join(x.numpy().tobytes() for x in initial.values())).hexdigest()})
            print("All12 same-graph heads frozen; starting once-only fresh qualification", flush=True); save()
            eos = runtime.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            generation_started = time.perf_counter()
            with measured(runtime) as resources:
                for index, row in enumerate(pool):
                    if time.perf_counter() - generation_started > 1200:
                        raise RuntimeError("fixed generation wall budget exhausted; no retry or partial improvement claim")
                    tokens = runtime._tokens(messages(row), thinking="off", shallow=True)
                    fresh_features[row["id"]] = runtime.shallow(tokens).float().mean(1).cpu()[0]
                    result = _generation(runtime, [row])[0]
                    label_split([row], [result], eos)
                    report["qualification"].append(result)
                    save()
                    if (index + 1) % 4 == 0:
                        print(f"fresh qualification {index + 1}/{len(pool)} elapsed{time.perf_counter() - generation_started:.1f}s", flush=True)
            report["generation_measurement"] = resources
            if versions != {n: p._version for n, p in runtime.model.named_parameters()}:
                raise ValueError("frozen backbone parameter versions changed")
            if runtime.branch.expert is not None or runtime.active_core is not None:
                raise ValueError("original core or branch state changed")
            runtime._release_model(); runtime = None
        labels = {}
        features = {}
        labelled = label_split(pool, report["qualification"], eos)
        for split in SPLITS[1:]:
            rows = [r for r in labelled if provenance(r)["split"] == split]
            features[split] = torch.stack([fresh_features[r["id"]] for r in rows])
            labels[split] = torch.tensor([FastDecisionHead.actions.index(provenance(r)["route_action"]) for r in rows])
        report["fresh_class_counts"] = {s: {a: int((labels[s] == FastDecisionHead.actions.index(a)).sum())
            for a in ("local_core", "teacher")} for s in labels}
        for seed in SEEDS:
            result = {"seed": seed, "arms": {}}
            for name in arms:
                with torch.no_grad():
                    logits = {s: models[(seed, name)](x) for s, x in features.items()}
                evaluated = evaluate_router(logits, labels, FastDecisionHead.actions, ["local_core", "teacher"])
                evaluated["scope"] = protocol["family_scope"] + "; actual local failure only, not measured teacher superiority"
                result["arms"][name] = {"diagnostics": diagnostics(evaluated), "evaluation": evaluated}
                print(encoded({"seed": seed, "arm": name, **result["arms"][name]["diagnostics"]}), flush=True)
            report["trials"].append(result)
        report["consistent_improvement"] = {name: all(beneficial(t["arms"][name]["diagnostics"],
            t["arms"]["baseline24"]["diagnostics"]) for t in report["trials"]) for name in arms if name != "baseline24"}
        report["all_original_gates_passed"] = {name: all(t["arms"][name]["diagnostics"]["gate_passed"]
            for t in report["trials"]) for name in arms}
        torch.save({"old_train_features": old_features, "fresh_features": fresh_features, "fresh_targets": labels},
                   output / "features.pt")
        if any(sha(ROOT / n) != v for n, v in protocol["source_sha256"].items()):
            raise ValueError("execution source changed during study")
        if any(sha(p) != v for p, v in inputs.items()) or model_binding(model_path, identity) != binding:
            raise ValueError("archived input or actual official model files changed")
        after = production_snapshot(storage)
        if after != before:
            raise ValueError("production state changed; no valid isolated-comparison claim")
        report.update(state="completed", elapsed_seconds=time.perf_counter() - started,
            source_inputs_unchanged=True, production_after=after, production_unchanged=True,
            feature_file_sha256=sha(output / "features.pt"))
        lock = ModelProcessLock(identity, "mps" if torch.backends.mps.is_available() else "cpu")
        lock.close(); report["model_lock_reacquired_and_released"] = True
        save()
    except BaseException as error:
        report.update(state="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        save(); raise
    finally:
        if runtime is not None:
            runtime._release_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--storage-root", default="/Volumes/移动数据库/lubanllm")
    args = parser.parse_args()
    run(args.output, args.model_path, args.storage_root)
