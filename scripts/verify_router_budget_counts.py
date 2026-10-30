"""Fixed 128/512 answer-budget controls and 3/6/12 route-head trials; no teacher.

Default/--print-plan is static. Only --execute with the exact reviewed plan SHA
loads one research runtime. Private clones never publish product state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import tempfile

from verify_automatic_batches import encoded, measured, nested_routes, selected_files, sha
from luban.retention import normalized_context, provenance, verified_source
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.training_checkpoint import implementation_hashes

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "research/router_budget_counts_v2.json"
BUDGETS, COUNTS = (128, 512), (3, 6, 12)


def integer_gold(checker):
    op, args = checker["operation"], checker["args"]
    if not isinstance(args, list) or not args or any(type(a) is not int for a in args):
        raise ValueError("integer checker requires literal integer arguments")
    if op == "add": return args[0] + args[1]
    if op == "subtract": return args[0] - args[1]
    if op == "multiply": return args[0] * args[1]
    if op == "multiply_minus": return args[0] * args[1] - args[2]
    if op == "exact_divide":
        if not args[1] or args[0] % args[1]: raise ValueError("nonexact division")
        return args[0] // args[1]
    if op == "exact_divide_plus":
        if not args[1] or args[0] % args[1]: raise ValueError("nonexact division")
        return args[0] // args[1] + args[2]
    if op == "sum_three": return sum(args)
    if op == "squares_minus": return args[0] ** 2 + args[1] ** 2 - args[2]
    raise ValueError("unknown fixed integer checker")


def validate_pool(rows):
    ids, contexts, families, counts = set(), set(), {}, Counter()
    for row in rows:
        value = provenance(row)
        validate_generation_check(value.get("generation_check"))
        split, family = value.get("split"), value.get("family_group")
        if (row.get("source") != "synthetic" or value.get("program_verified") is not True
                or not verified_source(row) or not row.get("approved") or not row.get("training_allowed")):
            raise ValueError("source must be independently verified before local qualification")
        if split not in {"train", "development", "calibration", "test"} or not isinstance(family, str) or not family:
            raise ValueError("fixed source split/family required")
        if row["purpose"] != ("training" if split == "train" else "validation"):
            raise ValueError("source purpose differs from fixed split")
        if row["id"] in ids or normalized_context(row) in contexts:
            raise ValueError("duplicate full context or ID")
        if family in families and families[family] != split:
            raise ValueError("declared family leakage")
        if value.get("route_action") is not None or not value.get("route_only"):
            raise ValueError("actual route labels must be absent until measured")
        expected = integer_gold(value["checker"])
        if (str(expected) != row["answer"] or value["checker"]["expected"] != expected
                or value["generation_check"].get("expected") != expected
                or value["generation_check"].get("max_new_tokens") != 512):
            raise ValueError("fixed gold/checker changed")
        ids.add(row["id"]); contexts.add(normalized_context(row)); families[family] = split; counts[split] += 1
    if counts != Counter(train=72, development=24, calibration=24, test=24):
        raise ValueError("fixed 72/24/24/24 counts changed")
    if sum("train-short-" in row["id"] for row in rows) != 36:
        raise ValueError("36 independently authored short TRAIN required")
    return dict(counts)


def plan():
    descriptor = json.loads(DESCRIPTOR.read_text())
    counts = validate_pool(descriptor["pool"])
    if len(descriptor["old44"]) != 44:
        raise ValueError("historical control pool changed")
    return {"protocol": "router-budget-counts-v2", "seed": 42, "scope": "research",
        "identity": descriptor["base_identity"], "core_version": None,
        "descriptor_sha256": sha(DESCRIPTOR), "descriptor": descriptor,
        "budget_controls": list(BUDGETS), "qualification_budget": 512,
        "train_per_actual_class": list(COUNTS), "steps": 160, "split_counts": counts,
        "selection": "Actual branch-OFF gold+EOS; pass=local_core, fail=teacher suggestion only. Fixed TRAIN IDs sorted within actual class; nested first3/6/12; same DEV/CAL/TEST every trial.",
        "insufficient_class": "block that count before fit; no teacher, no heldout move, no adaptive pool/budget or retry",
        "historical32_limit": "retain old evidence; implementation/context maintenance differs, so historical32 comparison is not a pure causal budget result",
        "identity_scope": "Arithmetic original-core study excludes identity from count qualification. Historical44 includes two identity rows only as separate control. Product head needs accepted identity core and separate validation.",
        "family_scope": descriptor["family_scope"], "exposures_per_train_row": 160,
        "teacher_calls": 0, "teacher_quality_measured": False, "production_publication": False,
        "memory_measurement": "RSS/MPS driver snapshots every0.1s, not physical peak; no resource measurement for blocked fits",
        "source_sha256": {**{"src/luban/" + key: value for key, value in implementation_hashes().items()},
            **{name: sha(ROOT / name) for name in ("scripts/verify_router_budget_counts.py",
                "scripts/verify_automatic_batches.py", "scripts/verify_router_v1.py")}}}


def at_budget(rows, budget):
    rows = copy.deepcopy(rows)
    for row in rows:
        value = provenance(row)
        value["generation_check"]["max_new_tokens"] = budget
        row["provenance"] = value
    return rows


def summary(rows, answers):
    if len(rows) != len(answers) or [row["id"] for row in rows] != [a["id"] for a in answers]:
        raise ValueError("generation record IDs/length differ from fixed pool")
    for row, answer in zip(rows, answers):
        if type(answer.get("complete")) is not bool or type(answer.get("passed")) is not bool:
            raise ValueError("complete/pass must be actual booleans")
        checked = evaluate_generation(answer["answer"], provenance(row)["generation_check"], answer["complete"])
        if answer["passed"] != checked["passed"]:
            raise ValueError("route qualification contradicts fixed answer/EOS checker")
    counts = Counter("passed" if answer["passed"] else "incomplete" if not answer["complete"] else "complete_wrong"
                     for answer in answers)
    by_split = {}
    for split in ("train", "development", "calibration", "test"):
        selected = [(row, answer) for row, answer in zip(rows, answers) if provenance(row)["split"] == split]
        by_split[split] = dict(Counter("passed" if answer["passed"] else "incomplete" if not answer["complete"] else "complete_wrong"
                                     for _, answer in selected))
    return {"counts": dict(counts), "by_split": by_split}


def labelled_rows(rows, answers):
    summary(rows, answers)
    result = copy.deepcopy(rows)
    for row, answer in zip(result, answers):
        if answer["branch_enabled"] or answer["thinking"] != "off":
            raise ValueError("qualification must be actual branch-OFF/thinking-off")
        value = provenance(row)
        value.update(route_action="local_core" if answer["passed"] else "teacher", actual_core_version=None,
            verification_evidence=value["verification_evidence"] + "; fixed qualification-evidence.json actual gold/EOS",
            teacher_superiority_verified=False)
        row["provenance"] = value
    return result


def _snapshot(store):
    return {name: sha(store / name) for name in selected_files(store)}


def execute(protocol, model_path, storage_root, output):
    import psutil
    import torch
    from luban.evolution import _generation
    from luban.memory import Memory
    from luban.runtime import Runtime
    from luban.settings import Settings, require_storage
    from luban.training_checkpoint import TrainingSession
    if encoded(protocol) != encoded(plan()):
        raise ValueError("preregistered source/data plan changed before execution")
    production = require_storage(storage_root)
    output = Path(output).absolute()
    if (output.parent != production / "experiments" or output.parent.is_symlink() or output.is_symlink() or output.exists()):
        raise ValueError("execution needs a fresh direct external experiments directory")
    store, source_db = production / "modules/luban0-v1", production / "app/memory.sqlite3"
    before, database_before = _snapshot(store), sha(source_db)
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as db:
        has_journal = db.execute("SELECT 1 FROM sqlite_master WHERE name='training_runs'").fetchone()
        if has_journal and db.execute("SELECT 1 FROM training_runs WHERE identity=? AND state IN "
                "('queued','running','paused','committing')", (protocol["identity"],)).fetchone():
            raise ValueError("another pending training job owns the base")
    output.mkdir()
    baseline = output / "baseline"
    for name in before:
        path = baseline / "modules/luban0-v1" / name
        path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(store / name, path)
    # Explicit raw-base research, regardless of selected production modules; no legacy router/branch.
    for name in ("active-core.json", "active.json", "active-router.json", "router.json", "router.safetensors"):
        (baseline / "modules/luban0-v1" / name).unlink(missing_ok=True)
    (baseline / "app").mkdir(parents=True)
    # Empty private Memory excludes external-reference hints and product journal/materials.
    Memory(baseline)
    report = {"protocol": protocol, "production_module_sha_before": before, "production_database_sha_before": database_before,
              "teacher_calls": 0, "production_publication": False, "controls": [], "trials": []}
    runtime = None
    def save(): (output / "report.json").write_text(encoded(report) + "\n")
    (output / "protocol.json").write_text(encoded(protocol) + "\n")
    try:
        with tempfile.TemporaryDirectory(prefix="luban-router-budget-") as home:
            settings = Settings(home)  # Empty credentials; teacher is never used.
            settings.data.update(storage_root=str(baseline), model_path=str(model_path), local_thinking="off")
            runtime = Runtime(settings, Memory(baseline), research=True)
            runtime.load()
            if runtime.identity != protocol["identity"] or runtime.active_core is not None or runtime.branch.expert is not None:
                raise ValueError("fixed original base/core/OFF differs")
            chip = None
            if platform.system() == "Darwin":
                observed = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
                chip = observed.stdout.strip() if observed.returncode == 0 else None
            report["hardware"] = {"platform": platform.platform(), "chip": chip,
                "physical_memory_bytes": psutil.virtual_memory().total, "device": str(runtime.device),
                "precision": str(next(runtime.model.parameters()).dtype), "torch_version": torch.__version__}
            save()
            for budget in BUDGETS:
                rows = at_budget(protocol["descriptor"]["old44"], budget)
                with measured(runtime) as resources:
                    answers = _generation(runtime, rows)
                record = {"budget": budget, "answers": answers, "measurement": resources, **summary(rows, answers)}
                report["controls"].append(record); save()
            rows = protocol["descriptor"]["pool"]
            with measured(runtime) as resources:
                answers = _generation(runtime, rows)
            report["qualification"] = {"answers": answers, "measurement": resources, **summary(rows, answers)}
            (output / "qualification-evidence.json").write_text(encoded(report["qualification"]) + "\n"); save()
            labelled = labelled_rows(rows, answers)
            runtime._release_model(); runtime = None
            for count in COUNTS:
                trial = {"train_per_actual_class": count, "product_accepted": False}
                report["trials"].append(trial)
                try:
                    records = {"splits": nested_routes(labelled, count)}
                except ValueError as failure:
                    trial.update(state="blocked", reason=str(failure)); save(); continue
                clone = output / ("router-" + str(count)); shutil.copytree(baseline, clone)
                settings.data["storage_root"] = str(clone)
                runtime = Runtime(settings, Memory(clone), research=True)
                try:
                    runtime.load()
                    if runtime.identity != protocol["identity"] or runtime.active_core is not None or runtime.branch.expert is not None:
                        raise ValueError("trial original-core/OFF changed")
                    session = TrainingSession(runtime, "router", {"steps": 160}, records, scope="research")
                    runtime.training_session = session
                    with measured(runtime) as resources:
                        result = runtime.calibrate_router()
                    trial.update(state="complete", run_id=session.id, result=result, measurement=resources,
                        trainable_parameters=sum(p.numel() for p in runtime.router.parameters()),
                        split_ids={key: [r["id"] for r in group] for key, group in records["splits"].items()},
                        exposures_per_train_row=160, candidate_gate_passed=bool(result["calibrated"]),
                        route_decisions=[runtime.choose_route(provenance(row)["messages"]) for row in records["splits"]["test"]],
                        actual_test_answers=[a for a in answers if a["id"] in {r["id"] for r in records["splits"]["test"]}],
                        teacher_response_measured=False)
                finally:
                    runtime.training_session = None; runtime._release_model(); runtime = None; save()
    finally:
        if runtime is not None:
            runtime.training_session = None; runtime._release_model()
        report.update(production_module_sha_after=_snapshot(store), production_database_sha_after=sha(source_db))
        report["production_unchanged"] = (before == report["production_module_sha_after"]
                                          and database_before == report["production_database_sha_after"])
        save()
        if not report["production_unchanged"]:
            raise RuntimeError("production selected modules/database changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-sha256")
    for name in ("model-path", "storage-root", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.execute and args.print_plan: parser.error("static and execution are mutually exclusive")
    protocol = plan(); fingerprint = hashlib.sha256(encoded(protocol).encode()).hexdigest()
    if not args.execute:
        print(encoded({"plan_sha256": fingerprint, "prepared_without_model": True, "plan": protocol})); return
    if args.plan_sha256 != fingerprint or not all((args.model_path, args.storage_root, args.output)):
        parser.error("execution requires reviewed exact plan SHA, model path, storage root and fresh output")
    execute(protocol, args.model_path, args.storage_root, args.output)


if __name__ == "__main__": main()
