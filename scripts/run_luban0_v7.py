"""One preregistered v7 correction attempt; v6 baseline failures remain failures.

Uppercase is NEW supervised correction. Only original-correct development rows
become replay; exposed baseline errors are development, never an independent final.
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

from run_luban0_v6 import CHECKPOINTS
from verify_luban0_final import branch_off_generation, digest, retire, tensor_digest, verify_recovery, write_new
from verify_luban0_product import content_digest, recovery_cases
from verify_training_operators import OPERATOR_AUDIT_PROTOCOL, TOLERANCES, audit_training_operators
from luban.memory import Memory
from luban.settings import Settings, private_json, require_storage

REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = "luban0-v7-uppercase-new-correction"


def exposure_schedule(new_count=34, replay_count=44):
    return [{"step": step, "new_updates": (step + 1) // 2,
        "minimum_new_exposures": ((step + 1) // 2) // new_count,
        "maximum_new_exposures": (((step + 1) // 2) + new_count - 1) // new_count,
        "minimum_replay_exposures": (step // 2) // replay_count,
        "maximum_replay_exposures": ((step // 2) + replay_count - 1) // replay_count} for step in CHECKPOINTS]


def corrected_rows(v6_plan, v6_result):
    observations = {row["id"]: row for row in v6_result["baseline_generation"]}
    rows = copy.deepcopy(v6_plan["uppercase_data"])
    for row in rows:
        previous_id = row["id"]
        row["id"] = previous_id.replace("v6-", "v7-", 1)
        provenance = json.loads(row["provenance"])
        role = "new"
        if row["purpose"] == "validation":
            observed = observations[previous_id]
            if observed["passed"]:
                role = "replay"
                provenance.update(baseline_preservation_eligible=True, original_baseline_verified_core=None,
                    original_baseline_verified_identity=v6_plan["base_identity"])
            provenance.update(original_baseline_result=copy.deepcopy(observed),
                development_scope="original baseline already observed in rejected v6; not an independent untouched test")
        provenance.update(evolution_role=role, dataset=DATASET, split_family=row["id"],
            inherited_v6_protocol_id=previous_id, classification_reason="NEW supervised uppercase guarantee/correction; frozen-original KL and projection capture only old replay TRAIN")
        row["provenance"] = json.dumps(provenance, ensure_ascii=False)
    return rows


def prepare():
    from luban.evolution import consolidation_samples
    descriptor_path = REPOSITORY / "research/luban0_v7_protocol.json"
    if descriptor_path.exists():
        raise FileExistsError("V7 is already preregistered; do not replace the fixed correction attempt")
    v6_descriptor = json.loads((REPOSITORY / "research/luban0_v6_protocol.json").read_text())
    v6_path = Path(v6_descriptor["protocol_path"])
    if digest(v6_path) != v6_descriptor["protocol_sha256"]:
        raise ValueError("Immutable v6 protocol SHA mismatch")
    prior = json.loads(v6_path.read_text())
    v6_result_path = REPOSITORY / "research/luban0_v6_run.json"
    result = json.loads(v6_result_path.read_text())
    original_results = result["baseline_generation"]
    if (result["passed"] or result["training_started"] or result["generation"]
        or len(original_results) != 4 or [row["passed"] for row in original_results] != [True, False, True, False]
        or not all(row["complete"] and row["branch_enabled"] is False for row in original_results)
        or result["operator_audit"]["restored"] is not True):
        raise ValueError("V7 requires the retained original v6 two-correct/two-wrong preflight with no training/engineering exposure")
    root = require_storage(prior["storage_root"])
    if (root / "modules/luban0-v1/active-core.json").exists():
        raise ValueError("V7 must start from original core None")
    memory = Memory(root)
    memory.model_identity = prior["base_identity"]
    before = consolidation_samples(SimpleNamespace(memory=memory, active_core=None,
        identity=prior["base_identity"], store=root / "modules/luban0-v1"))
    if before != prior["before_groups"]:
        raise ValueError("Original87 materials changed after rejected v6; no silent corpus change")
    rows = corrected_rows(prior, result)
    expected_counts = {"new_training": 34, "new_validation": 8, "replay_training": 44, "replay_validation": 17}
    run = root / "experiments" / ("luban0-v7-preregister-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    plan = copy.deepcopy(prior)
    plan.update(dataset=DATASET, uppercase_data=rows, expected_seeded_counts=expected_counts,
        scope="post-v6 baseline-informed supervised correction and once-only fixed engineering acceptance; NOT an independent scientific final",
        v6_original_baseline=original_results, v6_original_tensor_sha256=result["original_tensor_sha256"],
        v6_protocol_path=str(v6_path), v6_protocol_sha256=digest(v6_path),
        known_failure_use="old product-ordinary-7 is exposed and becomes NEW correction TRAIN; never called a fresh test",
        classification_policy="12 uppercase TRAIN become NEW; v6 original-correct gamma/PyTorch become REPLAY DEV; original-wrong tokyo/signal7 become NEW DEV with correct gold unchanged; no original wrong distribution in uppercase KL/projection",
        baseline_policy="reuse the frozen original-v6 baseline only after exact original tensor/base identity binding; do not rerun or silently substitute failed baseline gold",
        engineering_cases_sha256=content_digest(prior["engineering_cases"]),
        engineering_reuse_scope="same v6 fixed16 hashes; v6 never generated these; evaluate once only after v7 TRAIN and DEV pass; finite engineering, not scientific independent final",
        exposure_schedule=exposure_schedule(), script_sha256=digest(__file__))
    plan["operator_audit"]["sample"] = rows[0]
    plan["operator_audit"]["protocol_version"] = OPERATOR_AUDIT_PROTOCOL
    plan["operator_audit"]["tolerances"] = copy.deepcopy(TOLERANCES)
    plan["old_reports"].append({"path": str(v6_result_path), "sha256": digest(v6_result_path)})
    plan["policy"]["exposure_budget_scope"] = "supervised update counts only (replay KL is additional): 384 alternating steps give192NEW/192REPLAY updates; NEW34 min5/max6, REPLAY44 min4/max5; >=2 exposure first possible TRAIN160, two-pass earliest160/192; sufficient exposure is not guaranteed successful learning"
    write_new(run / "uppercase-data.json", rows)
    write_new(run / "engineering-cases.json", plan["engineering_cases"])
    write_new(run / "protocol.json", plan)
    descriptor = {"protocol_path": str(run / "protocol.json"), "protocol_sha256": digest(run / "protocol.json"),
        "run_directory": str(run), "prepared_without_model": True, "teacher_calls": 0,
        "expected_seeded_counts": expected_counts, "exposure_schedule": exposure_schedule()}
    write_new(descriptor_path, descriptor)
    print(json.dumps(descriptor, ensure_ascii=False), flush=True)


def run_once(protocol_path, expected_sha):
    from luban.evolution import consolidate, consolidation_samples, _target, _memory, TRAIN_CHECKPOINTS
    from luban.retention import read_core_retention, protected_validation
    from luban.runtime import Runtime
    import torch
    if digest(protocol_path) != expected_sha:
        raise ValueError("V7 fixed protocol SHA mismatch")
    plan = json.loads(protocol_path.read_text())
    if digest(__file__) != plan["script_sha256"] or list(TRAIN_CHECKPOINTS) != plan["policy"]["checkpoints"]:
        raise ValueError("Fixed v7 script or training-checkpoint policy changed")
    if content_digest(plan["engineering_cases"]) != plan["engineering_cases_sha256"]:
        raise ValueError("Fixed once-only engineering16 content SHA mismatch")
    root, directory = require_storage(plan["storage_root"]), protocol_path.parent
    if (root / "modules/luban0-v1/active-core.json").exists():
        raise ValueError("V7 must start from the original core None")
    with (directory / "run-started.json").open("x") as marker:
        json.dump({"protocol_sha256": expected_sha, "started": time.time(), "training_attempts": 1}, marker)
    files = [REPOSITORY / "src/luban" / (name + ".py") for name in
             ("runtime", "evolution", "retention", "backups", "coldpaging", "backbone", "dynamic", "memory", "settings", "generation_checks")]
    files += [Path(__file__), REPOSITORY / "scripts/verify_training_operators.py", REPOSITORY / "scripts/verify_luban0_final.py", REPOSITORY / "scripts/verify_luban0_product.py", REPOSITORY / "scripts/run_luban0_v6.py"]
    execution = {"protocol_sha256": expected_sha, "source_sha256": {str(path): digest(path) for path in files}}
    write_new(directory / "execution-manifest.json", execution)
    report = {"protocol_sha256": expected_sha, "execution_manifest_sha256": digest(directory / "execution-manifest.json"),
        "run_directory": str(directory), "scope": plan["scope"], "passed": False, "model_started": False,
        "training_started": False, "teacher_calls": 0, "baseline_generation": copy.deepcopy(plan["v6_original_baseline"]),
        "baseline_evidence_scope": "frozen original v6 preflight, not a new v7 generation call", "generation": []}
    started, runtime = time.perf_counter(), None
    try:
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(prefix="luban0-v7-settings-") as home:
            settings = Settings(home)
            settings.update({"storage_root": str(root), "model_path": plan["model_path"], "local_thinking": "off", "temperature": 0})
            runtime = Runtime(settings, Memory(root))
            runtime.load = partial(runtime.load, paging=False)
            report["model_started"] = True
            private_json(directory / "result.json", report)
            print(json.dumps({"phase": "model_loading", "protocol_sha256": expected_sha, "model_started": True}), flush=True)
            runtime.load()
            if runtime.active_core is not None or runtime.identity != plan["base_identity"] or consolidation_samples(runtime) != plan["before_groups"]:
                raise ValueError("Original core/base identity/full material corpus changed after preregistration")
            report.update(device=runtime.device, dtype=str(runtime.dtype), torch_version=torch.__version__,
                memory_samples=[_memory(runtime)], original_tensor_sha256=tensor_digest(_target(runtime)[1]))
            if report["original_tensor_sha256"] != plan["v6_original_tensor_sha256"]:
                raise ValueError("V6 original-baseline evidence belongs to a different core tensor")
            report["operator_audit"] = audit_training_operators(runtime, plan["operator_audit"]["sample"])
            if report["operator_audit"]["passed"] is not True or report["operator_audit"]["restored"] is not True:
                raise ValueError("Operator audit failed or did not restore original state")
            seeded = []
            for row in plan["uppercase_data"]:
                provenance = json.loads(row["provenance"])
                provenance.update(protocol_id=row["id"], protocol_sha256=expected_sha)
                if row["purpose"] == "validation" and provenance["evolution_role"] == "replay":
                    provenance.update(baseline_preservation_eligible=True, original_baseline_verified_core=None,
                        original_baseline_verified_identity=runtime.identity)
                identifier = runtime.memory.add_sample(row["prompt"], row["answer"], source="human", purpose=row["purpose"],
                    approved=True, training_allowed=True, provenance=provenance)
                seeded.append({"protocol_id": row["id"], "sample_id": identifier})
            groups = consolidation_samples(runtime)
            if {key: len(values) for key, values in groups.items()} != plan["expected_seeded_counts"]:
                raise ValueError("Fixed v7 seeded corpus counts do not match preregistration")
            write_new(directory / "seeded-groups.json", groups)
            report.update(seeded=seeded, training_dataset_sha256=content_digest(groups), training_started=True)
            private_json(directory / "result.json", report)
            report["evolution"] = consolidate(runtime, 384, learning_rate=1e-5, distillation_weight=5,
                preserve_replay_subspace=True, retention_regressions=plan["engineering_cases"])
            private_json(directory / "result.json", report)
            if not report["evolution"]["promoted"]:
                raise ValueError("One fixed-budget core attempt rejected; no engineering evaluation or additional training")
            report["candidate_version"] = runtime.active_core
            for row in plan["engineering_cases"]:
                generated = branch_off_generation(runtime, [row])[0]
                report["generation"].append(generated)
                private_json(directory / "result.json", report)
                print(json.dumps({"phase": "engineering", "case": row["id"], "passed": generated["passed"], "complete": generated["complete"]}), flush=True)
            if not all(row["passed"] for row in report["generation"]):
                raise ValueError("Fresh finite engineering acceptance failed; retire without retuning")
            snapshot = read_core_retention(runtime)
            protected = protected_validation(runtime)
            if snapshot["groups"] != groups or snapshot["regression_validation"] != plan["engineering_cases"]:
                raise ValueError("Published retention obligations differ from fixed complete corpus/new16")
            expected_protected = {row["id"] for key, values in groups.items() if key.endswith("validation") for row in values}
            expected_protected.update(row["id"] for row in plan["engineering_cases"])
            if len(protected) != len(expected_protected) or {row["id"] for row in protected} != expected_protected:
                raise ValueError("Retention does not expose the complete protected validation set")
            retention_path = runtime.store / (runtime.active_core + ".retention.json")
            report["retention"] = {"content_sha256": snapshot["content_sha256"],
                "file_sha256": digest(retention_path),
                "protected_validation_count": len(protected), "protected_validation_sha256": content_digest(protected)}
            verify_recovery(runtime, report["candidate_version"], recovery_cases(plan["engineering_cases"], report["generation"]), report["generation"], report)
            if (read_core_retention(runtime) != snapshot or content_digest(protected_validation(runtime)) != content_digest(protected)
                    or digest(retention_path) != report["retention"]["file_sha256"]):
                raise ValueError("True backup/full native reload changed complete retained materials/protection")
            if any(digest(path) != old_sha for path, old_sha in execution["source_sha256"].items()):
                raise ValueError("Frozen execution sources changed during the controlled run")
            for old in plan["old_reports"]:
                if digest(old["path"]) != old["sha256"]:
                    raise ValueError("An earlier scientific/product failure report changed")
            report.update(passed=True, active_core=runtime.active_core, retention_restored_unchanged=True,
                memory_samples=report["memory_samples"] + [_memory(runtime)])
            runtime.memory.event("core_v7_engineering_validation", {"passed": True, "version": runtime.active_core,
                "protocol_sha256": expected_sha, "scope": plan["scope"]})
    except BaseException as error:
        report.update(passed=False, error=str(error) or type(error).__name__)
        if runtime is not None:
            try:
                report["retirement"] = retire(runtime, None)
            except Exception as retirement_error:
                report["retirement_error"] = str(retirement_error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        private_json(directory / "result.json", report)
        private_json(REPOSITORY / "research/luban0_v7_run.json", report)
        print(json.dumps({"passed": report["passed"], "protocol_sha256": expected_sha, "run_directory": str(directory),
            "error": report.get("error"), "retirement": report.get("retirement"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


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
        parser.error("--run requires the explicit preregistered --protocol-sha256")
    else:
        run_once(args.run, args.protocol_sha256)


if __name__ == "__main__":
    main()
