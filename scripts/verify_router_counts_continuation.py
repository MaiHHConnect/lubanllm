"""Engineering continuation of the failed frozen 6/12 count fits, never requalify.

Default prints a static plan. Execute only with its exact separately approved SHA;
the original qualification, heldout observations and rejected3 remain immutable.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import tempfile

from verify_automatic_batches import encoded, measured, nested_routes, sha
from verify_router_budget_counts import labelled_rows, summary, validate_pool, _snapshot
from luban.training_checkpoint import implementation_hashes

ROOT = Path(__file__).resolve().parents[1]
OLD_PLAN = "995a43334719f66f27e4b664d2b9bce762788d7f9a37dd435de6c090977c0f7b"
OLD_RUN = "router-budget-counts-real-20261009-995a43334719"
OLD_EVIDENCE_SHA = "a37e5b54a5da58e8a7f72f3401f086c632c74d8a13a2e32df21543ee99424b92"
COUNTS = (6, 12)


def frozen_records(report):
    protocol = report["protocol"]
    if hashlib.sha256(encoded(protocol).encode()).hexdigest() != OLD_PLAN:
        raise ValueError("original frozen protocol changed")
    if report.get("production_unchanged") is not True or report.get("teacher_calls") != 0:
        raise ValueError("original attempt production/teacher guard failed")
    rows = protocol["descriptor"]["pool"]
    validate_pool(rows)
    actual = report["qualification"]
    if summary(rows, actual["answers"]) != {k: actual[k] for k in ("counts", "by_split")}:
        raise ValueError("original raw qualification summary changed")
    old3 = report["trials"][0]
    if (old3["train_per_actual_class"] != 3 or old3["state"] != "complete"
            or old3["candidate_gate_passed"] is not False or old3["product_accepted"] is not False):
        raise ValueError("original3 rejected observation changed")
    labelled = labelled_rows(rows, actual["answers"])
    return {str(count): {"splits": nested_routes(labelled, count)} for count in COUNTS}


def plan(storage_root):
    prior = Path(storage_root) / "experiments" / OLD_RUN
    if sha(prior / "evidence-sha256.json") != OLD_EVIDENCE_SHA:
        raise ValueError("original sealed evidence manifest changed")
    files = json.loads((prior / "evidence-sha256.json").read_text())["files"]
    for name in ("report.json", "qualification-evidence.json"):
        if sha(prior / name) != files[name]:
            raise ValueError("original sealed raw changed: " + name)
    baseline = {name: value for name, value in files.items() if name.startswith("baseline/")}
    actual_baseline = {str(p.relative_to(prior)): sha(p) for p in (prior / "baseline").rglob("*") if p.is_file()}
    if not baseline or actual_baseline != baseline:
        raise ValueError("original sealed baseline changed")
    report = json.loads((prior / "report.json").read_text())
    if encoded(report["qualification"]) != encoded(json.loads((prior / "qualification-evidence.json").read_text())):
        raise ValueError("qualification evidence differs from original report")
    records = frozen_records(report)
    return {"protocol": "router-counts-engineering-continuation-v1", "scope": "research",
        "original_plan_sha256": OLD_PLAN, "identity": report["protocol"]["identity"], "core_version": None,
        "prior_run": OLD_RUN, "prior_report_sha256": sha(prior / "report.json"),
        "prior_qualification_sha256": sha(prior / "qualification-evidence.json"),
        "prior_private_evidence_sha256": sha(prior / "evidence-sha256.json"),
        "baseline_sha256": _snapshot(prior / "baseline/modules/luban0-v1"),
        "baseline_file_sha256": baseline,
        "records": records, "train_per_actual_class": list(COUNTS), "steps": 160,
        "exposures_per_train_row": 160, "seed": 42, "teacher_calls": 0,
        "production_publication": False, "product_accepted": False, "qualification_generations": 0,
        "maintenance": "Root shared shallow admission guard corrected; full generation guard unchanged. New source binding; original failed attempt unchanged.",
        "heldout_scope": "Original DEV/CAL/TEST already observed in rejected3; fixed engineering comparison only, no new independent scientific test or optimal count claim.",
        "family_scope": report["protocol"]["family_scope"],
        "old3_reference": {"run_id": report["trials"][0]["run_id"], "gate_passed": False,
            "repeat_fit": False, "test_accuracy": report["trials"][0]["result"]["test_accuracy"]},
        "memory_measurement": "RSS/MPSdriver snapshots every0.1s; same original context_status return additionally records exact last admission estimate and contemporaneous driver bytes, no guard bypass",
        "source_sha256": {**{"src/luban/" + k: v for k, v in implementation_hashes().items()},
            **{n: sha(ROOT / n) for n in ("scripts/verify_automatic_batches.py",
                "scripts/verify_router_v1.py", "scripts/verify_router_budget_counts.py",
                "scripts/verify_router_counts_continuation.py")}}}


def execute(protocol, model_path, storage_root, output):
    import psutil
    import torch
    from luban.memory import Memory
    from luban.runtime import Runtime
    from luban.settings import Settings, require_storage
    from luban.training_checkpoint import TrainingSession
    if encoded(protocol) != encoded(plan(storage_root)):
        raise ValueError("continuation source/data plan changed")
    root = require_storage(storage_root)
    output = Path(output).absolute()
    if output.parent != root / "experiments" or output.exists() or output.is_symlink() or output.parent.is_symlink():
        raise ValueError("continuation requires fresh direct experiments directory")
    prior = root / "experiments" / protocol["prior_run"]
    store, source_db = root / "modules/luban0-v1", root / "app/memory.sqlite3"
    before, db_before = _snapshot(store), sha(source_db)
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='training_runs'").fetchone():
            if db.execute("SELECT 1 FROM training_runs WHERE identity=? AND state IN "
                    "('queued','running','paused','committing')", (protocol["identity"],)).fetchone():
                raise ValueError("production has a pending base task")
    output.mkdir()
    report = {"protocol": protocol, "teacher_calls": 0, "qualification_generations": 0,
        "production_publication": False, "production_module_sha_before": before,
        "production_database_sha_before": db_before, "trials": []}
    def save(): (output / "report.json").write_text(encoded(report) + "\n")
    (output / "protocol.json").write_text(encoded(protocol) + "\n")
    runtime = None
    try:
        with tempfile.TemporaryDirectory(prefix="luban-router-continuation-") as home:
            settings = Settings(home)
            settings.data.update(model_path=str(model_path), local_thinking="off")
            for count in COUNTS:
                trial = {"train_per_actual_class": count, "product_accepted": False, "state": "preparing"}
                report["trials"].append(trial); save()
                clone = output / ("router-" + str(count))
                shutil.copytree(prior / "baseline", clone)
                settings.data["storage_root"] = str(clone)
                runtime = Runtime(settings, Memory(clone), research=True)
                resources = {}
                try:
                    runtime.load()
                    if runtime.identity != protocol["identity"] or runtime.active_core is not None or runtime.branch.expert is not None:
                        raise ValueError("continuation requires original coreNone/branchOFF")
                    report["hardware"] = {"platform": platform.platform(),
                        "physical_memory_bytes": psutil.virtual_memory().total, "device": str(runtime.device),
                        "precision": str(next(runtime.model.parameters()).dtype), "torch_version": torch.__version__}
                    context_status = runtime.context_status
                    def observed_context(*args, **kwargs):
                        status = context_status(*args, **kwargs)
                        trial["last_admission"] = {"status": copy.deepcopy(status),
                            "mps_driver_bytes": torch.mps.driver_allocated_memory() if runtime.device == "mps" else 0,
                            "arguments": kwargs}
                        return status
                    runtime.context_status = observed_context
                    session = TrainingSession(runtime, "router", {"steps": 160},
                        copy.deepcopy(protocol["records"][str(count)]), scope="research")
                    runtime.training_session = session
                    with measured(runtime) as resources:
                        result = runtime.calibrate_router()
                    trial.update(state="complete", run_id=session.id, result=result,
                        measurement=resources, candidate_gate_passed=bool(result["calibrated"]),
                        trainable_parameters=sum(p.numel() for p in runtime.router.parameters()),
                        actual_test_answers_sha256=protocol["prior_qualification_sha256"],
                        route_decisions=[runtime.choose_route(row["provenance"]["messages"])
                            for row in protocol["records"][str(count)]["splits"]["test"]],
                        teacher_response_measured=False)
                except Exception as error:
                    trial.update(state="aborted", error=str(error))
                    trial["measurement"] = resources
                    raise
                finally:
                    runtime.training_session = None; runtime._release_model(); runtime = None; save()
    finally:
        if runtime is not None:
            runtime.training_session = None; runtime._release_model()
        report.update(production_module_sha_after=_snapshot(store), production_database_sha_after=sha(source_db))
        report["production_unchanged"] = (before == report["production_module_sha_after"]
            and db_before == report["production_database_sha_after"])
        save()
        if not report["production_unchanged"]: raise RuntimeError("production changed during continuation")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = plan(args.storage_root); fingerprint = hashlib.sha256(encoded(protocol).encode()).hexdigest()
    if not args.execute:
        print(encoded({"plan_sha256": fingerprint, "prepared_without_model": True, "plan": protocol})); return
    if args.plan_sha256 != fingerprint or not args.model_path or not args.output:
        parser.error("execution requires separately approved exact SHA/model/fresh output")
    execute(protocol, args.model_path, args.storage_root, args.output)


if __name__ == "__main__": main()
