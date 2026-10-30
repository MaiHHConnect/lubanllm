"""Preregister bounded 3/6-per-class router and 8/16-row branch trials.

Default is a static plan. Execute only with an approved plan SHA and exclusive
model ownership. Working modules/database are private copies, never production.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import re
import sqlite3
import tempfile
import threading
import time
from contextlib import closing, contextmanager

from verify_router_v1 import cases, BUDGET, ROUTER_IDENTITY_CHECK, TARGET
from luban.calibration import SPLITS, split_route_samples, validate_route_splits
from luban.decision import FastDecisionHead
from luban.retention import (_validate_record, eligible_learning_sample, normalized_context,
                             provenance, protected_validation)
from luban.training_checkpoint import implementation_hashes

ROOT = Path(__file__).resolve().parents[1]
ROUTER_COUNTS = (3, 6)
BRANCH_COUNTS = (8, 16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def router_pool():
    rows = cases()
    evidence = "fixed Python integer construction: scripts/verify_router_v1.py sha256=" + sha(ROOT / "scripts/verify_router_v1.py")
    result = []
    for row in rows:
        identity = row.get("task_kind") == "identity"
        check = row.get("generation_check") or {"type": "semantic", "validator": "integer",
                    "expected": int(row["answer"]), "max_new_tokens": BUDGET}
        result.append({"id": row["case_id"], "prompt": row["prompt"], "answer": row["answer"],
            "source": "human" if identity else "synthetic", "approved": 1, "training_allowed": 1,
            "purpose": "training" if row["split"] == "train" else "validation", "provenance": {
                "messages": row.get("messages") or [{"role": "user", "content": row["prompt"]}],
                "family_group": row["family_group"], "split": row["split"], "route_only": True,
                "generation_check": check, "program_verified": True,
                "verification_evidence": "user-authored identity contract literal; not a general factual claim" if identity else evidence}})
    return result


def branch_rows(manifest, count):
    pool, valid = manifest["branch"]["train_pool"], manifest["branch"]["valid"]
    if len(pool) < max(BRANCH_COUNTS) or not valid:
        raise ValueError("branch manifest needs at least sixteen qualified TRAIN and fixed independent VALID")
    contexts, ids, families = set(), set(), {"training": set(), "validation": set()}
    for row in pool + valid:
        _validate_record(row)
        if not eligible_learning_sample(row):
            raise ValueError("unverified teacher/source cannot enter a batch trial")
        if row["purpose"] != ("training" if row in pool else "validation"):
            raise ValueError("branch purpose changed")
        context = normalized_context(row)
        family = provenance(row).get("split_family")
        if row["id"] in ids or context in contexts:
            raise ValueError("branch rows need unique full contexts and IDs")
        if family is not None and (not isinstance(family, str) or not family.strip()):
            raise ValueError("declared branch split family must be a nonempty string")
        ids.add(row["id"])
        contexts.add(context)
        if family is not None:
            families[row["purpose"]].add(family)
    if families["training"] & families["validation"]:
        raise ValueError("branch TRAIN/VALID family leakage")
    return sorted(copy.deepcopy(pool), key=lambda row: row["id"])[:count], copy.deepcopy(valid)


def plan(manifest=None):
    result = {"protocol": "automatic-batch-trials-v1", "seed": 42,
        "safety_amendment": "pre-allocation cross-root base/device POSIX process lock; explicit research loader and initial journal scope; no material/gold/split/budget changes",
        "router": {"train_per_class": list(ROUTER_COUNTS), "steps": 160, "generation_budget": BUDGET,
                   "pool": router_pool(), "selection": "branch OFF gold+EOS: passed=local_core, failed=teacher suggestion only; sort TRAIN IDs per class, nested first 3/6",
                   "insufficient_class": "blocked; do not resample or move held-out rows"},
        "branch": {"train_counts": list(BRANCH_COUNTS), "steps_per_candidate": 64,
                   "expected_round_robin_exposures_per_row": {"8": 8, "16": 4},
                   "validation": "same frozen VALID with distinct full contexts and all preregistered core/accepted-branch protection; family independence only where explicitly supported",
                   "knowledge_scope": "fixed NEW material in stated base/core configuration; a raw-base v7 comparison is not new knowledge after core adoption"},
        "teacher_calls": 0, "teacher_quality_claim": False, "production_publication": False,
        "selection_claim": "finite paired engineering comparison, no optimal batch-size claim",
        "evaluation_scope": "held-out rows reused across batch comparisons; not an untouched scientific final",
        "memory_measurement": "sampled RSS and MPS driver allocation every 0.1s; not physical peak",
        "source_sha256": {**{name: sha(ROOT / name) for name in (
            "scripts/verify_automatic_batches.py", "scripts/verify_router_v1.py")},
            **{"src/luban/" + name: digest for name, digest in implementation_hashes().items()}}}
    if manifest is not None:
        if (not isinstance(manifest.get("identity"), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["identity"])
                or (manifest.get("core_version") is not None and not re.fullmatch(r"core-[0-9a-f]{12}", manifest["core_version"]))
                or not isinstance(manifest.get("branch", {}).get("protection"), list)):
            raise ValueError("manifest needs exact base identity, core version and complete protection")
        for count in BRANCH_COUNTS:
            branch_rows(manifest, count)
        result["branch"]["missing_split_family_ids"] = [row["id"] for row in
            manifest["branch"]["train_pool"] + manifest["branch"]["valid"]
            if provenance(row).get("split_family") is None]
        result["branch"]["family_scope"] = "undeclared families remain unproven; do not invent family IDs or claim independent generalization"
        result["manifest"] = copy.deepcopy(manifest)
        result["manifest_sha256"] = hashlib.sha256(encoded(manifest).encode()).hexdigest()
    else:
        result["branch"]["state"] = "waiting_frozen_manifest"
    return result


def nested_routes(rows, count):
    full = split_route_samples(rows, FastDecisionHead.actions)
    actions = [action for action in FastDecisionHead.actions if any(r["route_action"] == action for r in full["train"])]
    result = copy.deepcopy(full)
    result["train"] = []
    for action in actions:
        selected = sorted([row for row in full["train"] if row["route_action"] == action], key=lambda row: row["id"])
        if len(selected) < count:
            raise ValueError(f"fixed TRAIN class {action} only has {len(selected)} rows, needs {count}")
        result["train"].extend(selected[:count])
    validate_route_splits(result, FastDecisionHead.actions)
    return result


def selected_files(store):
    """Copy only active selected modules plus graph proof, not old experiment files."""
    names = []
    for pointer, field, prefix in (("active-core.json", "version", "core"), ("active.json", "id", "expert"),
                                    ("active-router.json", "id", "router")):
        path = store / pointer
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        name = record[field]
        names.append(pointer)
        if prefix == "expert":
            matches = list(store.glob("layer-*/" + name + ".json"))
            if len(matches) != 1:
                raise ValueError("active expert does not have one complete module")
            name = matches[0].relative_to(store).as_posix()[:-5]
        names.extend((name + ".json", name + ".safetensors"))
        if prefix == "core" and (store / (name + ".retention.json")).exists():
            names.append(name + ".retention.json")
    if "active-router.json" not in names:
        names.extend(name for name in ("router.json", "router.safetensors") if (store / name).exists())
    if (store / "graph-verification.json").exists():
        names.append("graph-verification.json")
    from luban.backups import _file
    for name in names:
        if name != "graph-verification.json":
            _file(store, name)
        if (store / name).is_symlink():
            raise ValueError("working-state source cannot be a symlink")
    return names


@contextmanager
def measured(runtime):
    import psutil
    import torch
    stop, snapshots = threading.Event(), []
    process = psutil.Process()
    def sample():
        snapshots.append({"time": time.perf_counter(), "rss_bytes": process.memory_info().rss,
            "mps_driver_bytes": torch.mps.driver_allocated_memory() if runtime.device == "mps" else 0})
    def monitor():
        while not stop.wait(.1):
            sample()
    sample()
    started = time.perf_counter()
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    result = {}
    try:
        yield result
    finally:
        stop.set()
        monitor_thread.join()
        sample()
        result.update(elapsed_seconds=time.perf_counter() - started, sampled_peak_rss_bytes=max(r["rss_bytes"] for r in snapshots),
            sampled_peak_mps_driver_bytes=max(r["mps_driver_bytes"] for r in snapshots), snapshots=snapshots)


def execute(protocol, model_path, storage_root, output):
    import torch
    from luban.coldpaging import full_backbone
    from luban.evolution import _generation
    from luban.memory import Memory
    from luban.runtime import Runtime
    from luban.settings import Settings, require_storage
    from luban.training_checkpoint import TrainingSession
    production = require_storage(storage_root)
    manifest = protocol["manifest"]
    identity, core = manifest["identity"], manifest["core_version"]
    store = production / "modules/luban0-v1"
    names = selected_files(store)
    before = {name: sha(store / name) for name in names}
    source_db = production / "app/memory.sqlite3"
    database_before = sha(source_db)
    # Read-only preflight must not even create a journal table in production.
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as db:
        has_journal = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_runs'").fetchone()
        if has_journal and db.execute("SELECT 1 FROM training_runs WHERE identity=? AND state IN "
                "('queued','running','paused','committing')", (identity,)).fetchone():
            raise ValueError("another pending training run owns this base")
    output.mkdir(parents=True, exist_ok=False)
    baseline = output / "baseline"
    baseline_store = baseline / "modules/luban0-v1"
    for name in names:
        target = baseline_store / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(store / name, target)
    (baseline / "app").mkdir(parents=True)
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as source, closing(
            sqlite3.connect(baseline / "app/memory.sqlite3")) as destination:
        source.backup(destination)
    report = {"protocol": protocol, "teacher_calls": 0, "production_published": False,
              "production_module_sha_before": before, "production_database_sha_before": database_before,
              "trials": [], "route_evidence": []}
    (output / "protocol.json").write_text(encoded(protocol) + "\n")
    runtime = None
    try:
        with tempfile.TemporaryDirectory(prefix="luban-batch-settings-") as home:
            settings = Settings(home)  # Fresh empty teacher credentials, never production credentials.
            for kind, size in [("router", n) for n in ROUTER_COUNTS] + [("learning", n) for n in BRANCH_COUNTS]:
                trial_root = output / (kind + "-" + str(size))
                shutil.copytree(baseline, trial_root)
                if kind == "router":
                    # Router trial is explicitly a two-action branch-OFF study.
                    (trial_root / "modules/luban0-v1/active.json").unlink(missing_ok=True)
                if core is None:
                    (trial_root / "modules/luban0-v1/active-core.json").unlink(missing_ok=True)
                settings.data.update(storage_root=str(trial_root), model_path=str(model_path), local_thinking="off")
                runtime = runtime or Runtime(settings, Memory(trial_root), research=True)
                runtime.load()
                if runtime.identity != identity or runtime.active_core != core:
                    raise ValueError("working base/core differs from fixed manifest")
                trial = {"kind": kind, "batch_size": size, "seed": 42, "state": "preparing"}
                report["trials"].append(trial)
                try:
                    if kind == "router":
                        if not report["route_evidence"]:
                            with full_backbone(runtime):
                                if runtime.branch.expert is not None:
                                    raise ValueError("router study requires actual branch OFF")
                                off = _generation(runtime, protocol["router"]["pool"])
                            for row, old in zip(protocol["router"]["pool"], off):
                                action = "local_core" if old["passed"] else "teacher"
                                report["route_evidence"].append({"sample": row, "core": old, "route_action": action})
                            (output / "route-evidence.json").write_text(encoded(report["route_evidence"]) + "\n")
                        labelled = []
                        for item in report["route_evidence"]:
                            row = copy.deepcopy(item["sample"])
                            value = provenance(row)
                            value.update(route_action=item["route_action"], actual_core_version=core,
                                verification_evidence=value["verification_evidence"] + "; actual branch-OFF answer+EOS in route-evidence.json",
                                teacher_superiority_verified=False)
                            row["provenance"] = json.dumps(value, ensure_ascii=False)
                            labelled.append(row)
                        records = {"splits": nested_routes(labelled, size)}
                        arguments = {"steps": 160}
                        trial["split_ids"] = {name: [row["id"] for row in records["splits"][name]] for name in SPLITS}
                        trial["train_exposure"] = "160 full-batch updates per TRAIN row"
                    else:
                        train, valid = branch_rows(manifest, size)
                        protection = protected_validation(runtime, include_learning=True)
                        if protection != manifest["branch"]["protection"]:
                            raise ValueError("full protection changed from frozen manifest")
                        records, arguments = {"train": train, "valid": valid, "protection": protection}, {"steps": 64}
                        trial["train_exposure_per_candidate"] = {row["id"]: sum(step % size == index for step in range(64))
                                                                  for index, row in enumerate(train)}
                    session = TrainingSession(runtime, kind, arguments, records, scope="research")
                    runtime.training_session = session
                    with measured(runtime) as measurement:
                        result = runtime.calibrate_router() if kind == "router" else runtime.learn()
                    trial.update(state="complete", measurement=measurement, result=result, run_id=session.id,
                                 candidate_gate_passed=bool(result.get("promoted", result.get("calibrated"))), product_accepted=False)
                    if kind == "router":
                        with full_backbone(runtime):
                            test_rows = [row for row in protocol["router"]["pool"] if provenance(row)["split"] == "test"]
                            trial["actual_answers"] = _generation(runtime, test_rows)
                        trial["route_decisions"] = [runtime.choose_route(provenance(row)["messages"]) for row in test_rows]
                        trial["teacher_response_measured"] = False
                    else:
                        trial["actual_answers"] = result["candidates"]
                except ValueError as error:
                    trial.update(state="blocked", reason=str(error))
                finally:
                    runtime.training_session = None
                    (output / "report.json").write_text(encoded(report) + "\n")
    finally:
        after = {name: sha(store / name) for name in names}
        report["production_module_sha_after"] = after
        report["production_database_sha_after"] = sha(source_db)
        report["production_unchanged"] = before == after and database_before == report["production_database_sha_after"]
        (output / "report.json").write_text(encoded(report) + "\n")
        if not report["production_unchanged"]:
            raise RuntimeError("production module/database changed during isolated trial")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true", help="static only; no model or data writes")
    parser.add_argument("--manifest", type=Path, help="fixed identity/core + branch train_pool/valid/protection JSON")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execute and args.print_plan:
        parser.error("--print-plan and --execute are mutually exclusive")
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    protocol = plan(manifest)
    fingerprint = hashlib.sha256(encoded(protocol).encode()).hexdigest()
    if not args.execute:
        print(encoded({"plan_sha256": fingerprint, "prepared_without_model": True, "plan": protocol}))
        return
    if not manifest or args.plan_sha256 != fingerprint or not args.model_path or not args.storage_root or not args.output:
        parser.error("execution requires reviewed manifest, matching --plan-sha256, --model-path, --storage-root and new --output")
    execute(protocol, args.model_path, args.storage_root, args.output)


if __name__ == "__main__":
    main()
