"""Two fixed NEW-corpus sizes, followed by one engineering acceptance of the winner.

Corpus size is not optimizer batch size. Exposed development cases select a
finite configuration; this is neither a universal optimum nor scientific final.
"""
import argparse
import copy
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import uuid

from run_luban0_v7 import CHECKPOINTS, exposure_schedule
from verify_luban0_final import branch_off_generation, digest, retire, tensor_digest, verify_recovery, write_new
from verify_luban0_product import content_digest, recovery_cases
from verify_training_operators import audit_training_operators
from luban.evolution import consolidate, consolidation_samples, restore_core, _target
from luban.memory import Memory
from luban.retention import protected_validation, read_core_retention
from luban.settings import Settings, private_json, require_storage
from luban.training_checkpoint import TrainingPaused, TrainingSession, training_runs, implementation_hashes

REPOSITORY = Path(__file__).resolve().parents[1]
FOUR_UPPERCASE = (0, 2, 4, 10)
ARGUMENTS = {"steps": 384, "learning_rate": 1e-5, "distillation_weight": 5, "preserve_replay_subspace": True}


def prepare():
    target = REPOSITORY / "research/luban0_batch_sweep_protocol.json"
    if target.exists():
        raise FileExistsError("The two-configuration protocol already exists; do not replace it")
    previous = json.loads((REPOSITORY / "research/luban0_v7_run.json").read_text())
    previous_directory = Path(previous["run_directory"])
    previous_protocol = previous_directory / "protocol-operator-order.json"
    v7 = json.loads(previous_protocol.read_text())
    root = require_storage(v7["storage_root"])
    if previous["passed"] or previous["generation"] or (root / "modules/luban0-v1/active-core.json").exists():
        raise ValueError("Preparation needs the retained rejected v7 and original core None")
    groups = json.loads((previous_directory / "seeded-groups.json").read_text())
    memory = Memory(root)
    memory.model_identity = v7["base_identity"]
    current = consolidation_samples(SimpleNamespace(memory=memory, active_core=None, identity=v7["base_identity"], store=root / "modules/luban0-v1"))
    if groups != current:
        raise ValueError("Current103 materials differ from the frozen v7 corpus")
    identity_ids = {row["id"] for row in v7["before_groups"]["new_training"]}
    uppercase = {json.loads(row["provenance"])["protocol_id"]: row for row in groups["new_training"] if row["id"] not in identity_ids}
    selected_ids = {uppercase[f"v7-uppercase-training-{index}"]["id"] for index in FOUR_UPPERCASE}
    trials = []
    failed_greeting = "790bf39e957c4b869494c1c8382995f9"
    for count in (26, 34):
        records = copy.deepcopy(groups)
        if count == 26:
            records["new_training"] = [row for row in records["new_training"] if row["id"] in identity_ids | selected_ids]
        if len(records["new_training"]) != count or not identity_ids <= {r["id"] for r in records["new_training"]}:
            raise ValueError("Fixed queue configurations must retain all22 identity/greeting TRAIN")
        index = [r["id"] for r in records["new_training"]].index(failed_greeting)
        exposures = [{"step": step, "updates": ((step + 1) // 2) // count + int(index < ((step + 1) // 2) % count)} for step in CHECKPOINTS]
        trials.append({"new_training_count": count, "session_id": uuid.uuid4().hex[:12],
            "groups": records, "groups_sha256": content_digest(records),
            "exposure_schedule": exposure_schedule(count, 44), "known_greeting_exposures": exposures})
    directory = root / "experiments" / ("luban0-batch-sweep-preregister-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    plan = {"scope": "two fixed composition/corpus-size engineering configurations; not a universal sweetspot or independent scientific final",
        "base_identity": v7["base_identity"], "original_tensor_sha256": previous["original_tensor_sha256"],
        "storage_root": str(root), "model_path": v7["model_path"], "groups": groups, "groups_sha256": content_digest(groups),
        "trials": trials, "arguments": ARGUMENTS, "training_attempts_per_configuration": 1,
        "optimizer_batch_size": 1, "gradient_scope": "one supervised sequence plus matched frozen-core replay KL per Adam step; no mini-batch accumulation",
        "corpus_counts": "26=22 unchanged identity/greetings plus uppercase0/2/4/10;34=22 plus all12; each uses identical NEW DEV8,REPLAY TRAIN44,REPLAY DEV17",
        "four_uppercase_sources": ["luban exposed old failure", "MaiHaoHuo mixed case", "Qwen mixed case", "core2 with arithmetic history"],
        "selection": "execute26 then34 once from exact same original tensor; TRAIN selects first two consecutive declared checks with everyNEW>=2exposures; development all gates select smallest corpus-count that promotes; no development-driven retraining",
        "engineering_cases": v7["engineering_cases"], "engineering_cases_sha256": content_digest(v7["engineering_cases"]),
        "engineering_policy": "only the chosen configuration generates fixed16 once; any failure or interruption retires toNone, never tries another configuration against those cases",
        "retention_policy": "both trial snapshots declare fixed16 generation-only obligations, never TRAIN/loss/DEV feedback; chosen must pass all16 and genuinebackup/original/fullnative reload; all41 protection records/hashes unchanged",
        "trial_publication_scope": "journal core promotions are validated DEV research candidates only; production/app remain idle until chosen engineering accepted; old/new success counts are not a product acceptance claim",
        "resume_policy": "same immutable manifest/session ID may resume an interrupted TRAIN from durable master/Adam/RNG without resetting384budget; completed/rejected trials never retrain; interrupted engineering rejects instead of repeating questions",
        "execution": "one full native Runtime.load(paging=False) instance; not productionpaging performance evidence",
        "operator_sample": v7["uppercase_data"][0], "teacher_calls": 0, "script_sha256": digest(__file__),
        "historical_reports": [*v7["old_reports"], {"path": str(REPOSITORY / "research/luban0_v7_run.json"), "sha256": digest(REPOSITORY / "research/luban0_v7_run.json")}],
        "v7_protocol_path": str(previous_protocol), "v7_protocol_sha256": digest(previous_protocol),
        "risks": ["composition changes with corpus count, so count alone is not a proven cause", "DEV was already observed in earlier rounds, not untouched", "seven/eight exposures may still fail identity or uppercase transfer", "strict finite protection does not guarantee all future knowledge"]}
    write_new(directory / "protocol.json", plan)
    descriptor = {"protocol_path": str(directory / "protocol.json"), "protocol_sha256": digest(directory / "protocol.json"), "prepared_without_model": True}
    write_new(target, descriptor)
    print(json.dumps(descriptor, ensure_ascii=False), flush=True)



def mark_product_acceptance(runtime, session_id, accepted):
    """Append outer engineering acceptance without rewriting a frozen commit report."""
    row = next(value for value in training_runs(runtime.memory, runtime.identity) if value["id"] == session_id)
    if row["state"] != "committed" or row.get("scope") != "research":
        raise ValueError("Engineering acceptance needs a committed research candidate")
    if accepted and row["commit"]["report"]["version"] != runtime.active_core:
        raise ValueError("Engineering acceptance belongs to a different active core")
    # The original constructor correctly rejects a terminal job for training.
    # This state-only update uses the existing writer, without resuming it.
    session = object.__new__(TrainingSession)
    session.runtime, session.memory, session.id = runtime, runtime.memory, session_id
    session.binding, session.body = row["binding"], row
    session.body["product_accepted"] = bool(accepted)
    session._write("committed")


def source_manifest(execution_script=None):
    manifest = {str(REPOSITORY / "src/luban" / name): sha for name, sha in implementation_hashes().items()}
    paths = [REPOSITORY / "src/luban" / (name + ".py") for name in
        ("backups", "memory", "settings")]
    paths += [Path(__file__), REPOSITORY / "scripts/run_luban0_v7.py", REPOSITORY / "scripts/run_luban0_v6.py",
              REPOSITORY / "scripts/verify_training_operators.py", REPOSITORY / "scripts/verify_luban0_final.py",
              REPOSITORY / "scripts/verify_luban0_product.py"]
    if execution_script is not None and Path(execution_script) not in paths:
        paths.append(Path(execution_script))
    manifest.update({str(path): digest(path) for path in paths})
    return manifest


def run(protocol_path, protocol_sha, *, expected_arguments=None, execution_script=None,
        summary_name="luban0_batch_sweep.json"):
    if digest(protocol_path) != protocol_sha:
        raise ValueError("Fixed batch experiment protocol SHA mismatch")
    plan = json.loads(protocol_path.read_text())
    arguments = expected_arguments or ARGUMENTS
    source = Path(execution_script or __file__)
    if digest(source) != plan["script_sha256"] or plan["arguments"] != arguments or content_digest(plan["engineering_cases"]) != plan["engineering_cases_sha256"]:
        raise ValueError("Fixed experiment source/policy/engineering changed")
    if "training_checkpoints" in plan:
        from luban.evolution import TRAIN_CHECKPOINTS
        if plan["training_checkpoints"] != [step for step in TRAIN_CHECKPOINTS if step <= arguments["steps"]]:
            raise ValueError("Fixed training checkpoints changed")
    directory = protocol_path.parent
    result_path = directory / "result.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {
        "protocol_sha256": protocol_sha, "scope": plan["scope"], "passed": False, "model_started": False,
        "trials": [], "generation": [], "teacher_calls": 0, "started": time.time()}
    if report.get("passed") or report.get("error"):
        raise ValueError("This experiment already completed; no further training or engineering reuse")
    manifest = source_manifest(source)
    if plan.get("execution_manifest") is not None and plan["execution_manifest"] != manifest:
        raise ValueError("Sealed experiment source implementations changed")
    manifest_path = directory / "execution-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen experiment implementations changed; no checkpoint reuse")
    else:
        write_new(manifest_path, manifest)
    runtime, accepted_session_id = None, None
    try:
        from luban.runtime import Runtime
        root = require_storage(plan["storage_root"])
        with tempfile.TemporaryDirectory(prefix="luban0-batch-settings-") as home:
            settings = Settings(home)
            settings.update({"storage_root": str(root), "model_path": plan["model_path"], "local_thinking": "off", "temperature": 0})
            runtime = Runtime(settings, Memory(root), research=True)
            runtime.load = partial(runtime.load, paging=False)
            report["model_started"] = True
            private_json(result_path, report)
            print(json.dumps({"model_started": True, "protocol_sha256": protocol_sha}), flush=True)
            runtime.load()
            restore_core(runtime, None)
            if (runtime.identity != plan["base_identity"] or tensor_digest(_target(runtime)[1]) != plan["original_tensor_sha256"]
                    or consolidation_samples(runtime) != plan["groups"]):
                raise ValueError("Original core tensor/base/full frozen corpus no longer matches")
            if (directory / "engineering-started.json").exists():
                raise ValueError("An engineering evaluation was interrupted; reject rather than repeat it")
            if "operator_audit" not in report:
                report["operator_audit"] = audit_training_operators(runtime, plan["operator_sample"])
                private_json(result_path, report)
            if report["operator_audit"]["passed"] is not True or report["operator_audit"]["restored"] is not True:
                raise ValueError("Operator audit failed/restoration unproven")
            report["trials"] = []
            for trial in plan["trials"]:
                if content_digest(trial["groups"]) != trial["groups_sha256"]:
                    raise ValueError("Fixed trial materials changed")
                restore_core(runtime, None)
                existing = next((r for r in training_runs(runtime.memory, runtime.identity) if r["id"] == trial["session_id"]), None)
                if existing and existing["state"] == "committed":
                    trained = existing["commit"]["report"]
                elif existing and existing["state"] == "rejected":
                    trained = json.loads((runtime.store / ("evolution-" + trial["session_id"] + ".json")).read_text())
                else:
                    records = {"groups": trial["groups"], "protection": protected_validation(runtime, include_learning=False),
                               "retention_regressions": plan["engineering_cases"]}
                    session = TrainingSession(runtime, "evolution", arguments, records,
                                              run_id=trial["session_id"], scope="research")
                    runtime.training_session = session
                    try:
                        trained = consolidate(runtime, **arguments)
                    except TrainingPaused:
                        session.pause()  # consolidation has already restored weights and exited its context.
                        report["paused"] = True
                        raise
                    finally:
                        runtime.training_session = None
                report["trials"].append({"new_training_count": trial["new_training_count"], "report": trained})
                private_json(result_path, report)
                print(json.dumps({"phase": "trial_complete", "new_training_count": trial["new_training_count"],
                    "promoted": trained["promoted"], "completed_steps": trained.get("completed_steps")}), flush=True)
            choices = [r for r in report["trials"] if r["report"]["promoted"]]
            if not choices:
                raise ValueError("No fixed configuration passed; untouched engineering not generated")
            chosen = min(choices, key=lambda r: r["new_training_count"])
            version = chosen["report"]["version"]
            restore_core(runtime, version)
            report.update(chosen_new_training_count=chosen["new_training_count"], candidate_version=version)
            write_new(directory / "engineering-started.json", {"candidate_version": version, "protocol_sha256": protocol_sha})
            for row in plan["engineering_cases"]:
                generated = branch_off_generation(runtime, [row])[0]
                report["generation"].append(generated)
                private_json(result_path, report)
                print(json.dumps({"phase": "engineering", "case": row["id"], "passed": generated["passed"]}), flush=True)
            if not all(row["passed"] for row in report["generation"]):
                raise ValueError("Chosen configuration failed fixed engineering; no fallback configuration or retuning")
            snapshot = read_core_retention(runtime)
            protected = protected_validation(runtime, include_learning=False)
            chosen_groups = next(t["groups"] for t in plan["trials"] if t["new_training_count"] == chosen["new_training_count"])
            if snapshot["groups"] != chosen_groups or snapshot["regression_validation"] != plan["engineering_cases"] or len(protected) != 41:
                raise ValueError("Full chosen-corpus and41protection snapshot differs")
            retention_path = runtime.store / (version + ".retention.json")
            report["retention"] = {"file_sha256": digest(retention_path), "content_sha256": snapshot["content_sha256"],
                "protected_validation_count": len(protected), "protected_validation_sha256": content_digest(protected)}
            verify_recovery(runtime, version, recovery_cases(plan["engineering_cases"], report["generation"]), report["generation"], report)
            if read_core_retention(runtime) != snapshot or content_digest(protected_validation(runtime, include_learning=False)) != content_digest(protected) or digest(retention_path) != report["retention"]["file_sha256"]:
                raise ValueError("True backup/reload changed complete41protection")
            if any(digest(path) != sha for path, sha in manifest.items()) or any(digest(r["path"]) != r["sha256"] for r in plan["historical_reports"]):
                raise ValueError("Execution sources or historical raw failure reports changed")
            report.update(passed=True, active_core=runtime.active_core, retention_restored_unchanged=True)
            private_json(result_path, report)
            accepted_session_id = chosen["report"]["id"]
            mark_product_acceptance(runtime, accepted_session_id, True)
            runtime.memory.event("batch_engineering_validation", {"passed": True, "version": version, "scope": plan["scope"], "protocol_sha256": protocol_sha})
    except TrainingPaused:
        raise
    except BaseException as error:
        report.update(passed=False, error=str(error) or type(error).__name__)
        if runtime is not None and accepted_session_id is not None:
            mark_product_acceptance(runtime, accepted_session_id, False)
        if runtime is not None:
            report["retirement"] = retire(runtime, None)
            if "method" not in report["retirement"]:
                restored_sha = tensor_digest(_target(runtime)[1])
                report["retirement"]["tensor_sha256"] = restored_sha
                if restored_sha != plan["original_tensor_sha256"]:
                    report["retirement"]["restored"] = False
                    raise RuntimeError("Retired original-core tensor SHA does not match") from error
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        private_json(result_path, report)
        private_json(REPOSITORY / "research" / summary_name, report)
        print(json.dumps({"passed": report["passed"], "protocol_sha256": protocol_sha, "error": report.get("error"), "paused": report.get("paused"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", type=Path)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif not args.protocol_sha256:
        parser.error("--run requires --protocol-sha256")
    else:
        run(args.run, args.protocol_sha256)


if __name__ == "__main__":
    main()
