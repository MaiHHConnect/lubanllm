"""One separate 34-record/512-step research attempt; no old-budget extension.

Prepare fixed data first, seal source hashes after coordinated safety patches,
then run once under an explicit GO. The shared runner retains every strict gate.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from run_luban0_batch_sweep import REPOSITORY, run, source_manifest
from verify_luban0_final import digest, write_new
from verify_luban0_product import content_digest
from luban.settings import require_storage

ARGUMENTS = {"steps": 512, "learning_rate": 1e-5, "distillation_weight": 5, "preserve_replay_subspace": True}
CHECKPOINTS = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]


def prepare():
    descriptor = REPOSITORY / "research/luban0_34_512_preparation.json"
    if descriptor.exists():
        raise FileExistsError("Separate512 preparation exists; do not replace its fixed materials")
    previous = json.loads((REPOSITORY / "research/luban0_batch_sweep_protocol.json").read_text())
    previous_path = Path(previous["protocol_path"])
    if digest(previous_path) != previous["protocol_sha256"]:
        raise ValueError("Retained384 protocol SHA mismatch")
    plan = json.loads(previous_path.read_text())
    result_path = previous_path.parent / "result.json"
    result = json.loads(result_path.read_text())
    if result["passed"] or result["generation"] or len(result["trials"]) != 2 or any(t["report"]["promoted"] for t in result["trials"]):
        raise ValueError("Separate512 requires both rejected384 trials and untouched engineering16")
    root = require_storage(plan["storage_root"])
    if (root / "modules/luban0-v1/active-core.json").exists():
        raise ValueError("Preparation needs original core None")
    trial = copy.deepcopy(next(t for t in plan["trials"] if t["new_training_count"] == 34))
    trial["session_id"] = uuid.uuid4().hex[:12]
    if content_digest(trial["groups"]) != trial["groups_sha256"] or content_digest(plan["engineering_cases"]) != plan["engineering_cases_sha256"]:
        raise ValueError("Frozen material SHA mismatch")
    greeting_index = [r["id"] for r in trial["groups"]["new_training"]].index("790bf39e957c4b869494c1c8382995f9")
    trial["exposure_schedule"] = [{"step": k, "new_updates": k // 2,
        "minimum_new_exposures": (k // 2) // 34, "maximum_new_exposures": (k // 2 + 33) // 34,
        "minimum_replay_exposures": (k // 2) // 44, "maximum_replay_exposures": (k // 2 + 43) // 44} for k in CHECKPOINTS]
    trial["known_greeting_exposures"] = [{"step": k, "updates": (k // 2) // 34 + int(greeting_index < (k // 2) % 34)} for k in CHECKPOINTS]
    plan.update(prepared_without_model=True, sealed=False, execution_manifest=None, script_sha256=None,
        scope="one separate bounded34/512 engineering correction; not extension of rejected384 run or independent scientific final",
        arguments=ARGUMENTS, training_checkpoints=CHECKPOINTS, trials=[trial],
        selection="first TWO CONSECUTIVE fixed TRAIN allgeneration/EOS passes plus unchanged replaylossbudget and everyNEW>=2exposures; then DEV8/17allpass; noDEV-drivenretraining",
        training_attempts_per_configuration=1,
        corpus_counts="unchanged22identity+12uppercase NEWTRAIN34, NEWDEV8, REPLAYTRAIN44, REPLAYDEV17; optimizerbatch1",
        reason="known desktop greeting failed at6exposures in384; historicalv5 firstpassed after7/8. Increasedexposure is a hypothesis;26trainingfit still failedtransfer/retention.",
        exposure_limit="512 provides minNEW7/max8; known greeting only7 atboth480/512, not8; no guarantee of success",
        previous_sweep_protocol_path=str(previous_path), previous_sweep_protocol_sha256=digest(previous_path),
        previous_sweep_result_path=str(result_path), previous_sweep_result_sha256=digest(result_path),
        implementation_binding="pending root/engine safety patch acceptance and source freeze; this draft cannot run",
        historical_reports=[*plan["historical_reports"], {"path": str(result_path), "sha256": digest(result_path)},
            {"path": str(REPOSITORY / "research/luban0_batch_sweep.json"), "sha256": digest(REPOSITORY / "research/luban0_batch_sweep.json")}])
    directory = root / "experiments" / ("luban0-34-512-preregister-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "draft-protocol.json", plan)
    record = {"draft_path": str(directory / "draft-protocol.json"), "draft_sha256": digest(directory / "draft-protocol.json"), "prepared_without_model": True, "sealed": False}
    write_new(descriptor, record)
    print(json.dumps(record, ensure_ascii=False))


def seal(draft_path):
    """Only invoke after root confirms all execution sources are frozen."""
    prepared = json.loads((REPOSITORY / "research/luban0_34_512_preparation.json").read_text())
    if str(draft_path) != prepared["draft_path"] or digest(draft_path) != prepared["draft_sha256"]:
        raise ValueError("Fixed512 draft changed")
    plan = json.loads(draft_path.read_text())
    from luban.evolution import TRAIN_CHECKPOINTS
    if list(TRAIN_CHECKPOINTS) != CHECKPOINTS or plan["arguments"] != ARGUMENTS:
        raise ValueError("512 policy implementation does not match fixed draft")
    plan.update(sealed=True, script_sha256=digest(__file__), execution_manifest=source_manifest(__file__),
                implementation_binding="execution source hashes sealed before model; same checkpoint package/source/record binding at actualrun")
    path = draft_path.parent / "protocol.json"
    write_new(path, plan)
    record = {"protocol_path": str(path), "protocol_sha256": digest(path), "prepared_without_model": True, "sealed": True}
    write_new(REPOSITORY / "research/luban0_34_512_protocol.json", record)
    print(json.dumps(record, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--seal", type=Path)
    action.add_argument("--print-plan", type=Path)
    action.add_argument("--run", type=Path)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.seal:
        seal(args.seal)
    elif args.print_plan:
        print(args.print_plan.read_text())
    else:
        plan = json.loads(args.run.read_text())
        if not plan.get("sealed") or plan["training_checkpoints"] != CHECKPOINTS or plan["arguments"] != ARGUMENTS or len(plan["trials"]) != 1:
            raise ValueError("512 needs the one fixed sealed protocol")
        if not args.protocol_sha256:
            parser.error("--run requires --protocol-sha256")
        run(args.run, args.protocol_sha256, expected_arguments=ARGUMENTS,
            execution_script=__file__, summary_name="luban0_34_512_run.json")


if __name__ == "__main__":
    main()
