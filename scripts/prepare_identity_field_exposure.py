"""One fixed57/71 mean1 exposure diagnostic; SOURCE ONLY, no model or seal."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from run_identity_coverage import FIELD_ARGUMENTS, FIELD_EXPOSURE_ARGUMENTS, check_original_production, validate_protocol_policy
from verify_luban0_final import digest, write_new
from luban.evolution import FIELD_EXPOSURE_CHECKPOINTS, field_first_token_policy
from luban.settings import require_storage

REPO = Path(__file__).resolve().parents[1]
PREPARATION = REPO / "research/core_identity_field_exposure_preparation.json"
PROPOSAL = REPO / "research/core_identity_field_exposure_draft.json"


def exposure_plan(old, result, weighted_result, *, directory, session_id, prepared):
    """Restore the mean1 control and change only its bounded exposure ceiling."""
    validate_protocol_policy(old)
    if old["arguments"] != FIELD_ARGUMENTS or "field_loss_policy" in old:
        raise ValueError("Exposure diagnostic requires the original native mean1 fixed57/832 control")
    for index, raw in enumerate((result, weighted_result)):
        training = raw["training"]
        if (raw["passed"] is not False or training["promoted"] is not False
                or training["steps"] != 832 or training["completed_steps"] != 832
                or training["previous_version"] is not None
                or training["training_selection"]["selected_step"] is not None
                or training.get("field_loss_policy") != (field_first_token_policy() if index else None)
                or raw["retirement"] != {"version": None, "previous_version": None, "restored": True,
                                         "tensor_sha256": old["original_tensor_sha256"]}
                or raw.get("engineering") != []
                or any(key in raw for key in ("known_regression", "recovery", "production_publication"))):
            raise ValueError("Both historical832 runs must be rejected and actually retired to originalNone")
    plan = copy.deepcopy(old)
    plan.update(status="prepared independent native mean1 exposure diagnostic; NOT sealed/run", sealed=False,
        prepared=prepared, session_id=session_id, research_root=str(directory / "research-storage"),
        arguments=copy.deepcopy(FIELD_EXPOSURE_ARGUMENTS), training_checkpoints=list(FIELD_EXPOSURE_CHECKPOINTS),
        reason="Same57/71, native mean1 and round-robin as original mean1/832. Change only the independent ceiling to1536, not the rejected weight4 objective or weights.",
        objective="Native full answer/EOS mean CE for every row; existing KL5 and cumulative replay-subspace projection unchanged.",
        selection="First two consecutive full57 NEW TRAIN / all current qualified REPLAY TRAIN generation / native replay-loss / minNEW2 passes; 47 fixed checks64..1536 every32. Then all20 DEV/16 REPLAYDEV/27 known/8 engineering and full71 recovery gates unchanged.",
        repeat_budget_scope="Independent originalNone new1536 with new master/Adam/RNG/session. Neither rejected832 run is resumed. Hard stop at1536; if rejected, stop blind budget extensions of this single-down_proj strategy.",
        exposure_limit="768 NEW optimizer slots /57 gives min13/max14; all nine field rows get exactly13 own updates in the fixed sorted order, not13 complete gradient batches. No success or optimum guarantee.",
        source_binding="SOURCE ONLY fixed57/1536 native mean1. New source/session review, native v1 audit,115 TRAIN-only prefix proof, formal seal and separate MODEL GO required.",
        run_descriptor_path=str(REPO / "research/core_identity_field_exposure_run.json"),
        script_sha256=None, execution_manifest=None)
    validate_protocol_policy(plan)
    for key in ("groups_before_original_baseline", "groups_sha256", "source_groups", "source_rows",
                "superseded_source_ids", "replay_training_candidates", "retention_regressions", "engineering_cases",
                "old_known_development_regressions", "seed", "optimizer_batch_size", "operator_audit",
                "original_tensor_sha256", "protection_count", "prefix_alignment_policy"):
        if plan[key] != old[key]:
            raise ValueError("Fixed57/71 material, rights, generation, optimizer order or native audit changed")
    return plan


def prepare():
    if PREPARATION.exists() or PROPOSAL.exists():
        raise FileExistsError("Never overwrite an existing exposure preparation")
    entries = [json.loads((REPO / ("research/core_identity_" + name + "_protocol.json")).read_text())
               for name in ("fields", "first_gold")]
    protocols = [Path(entry["protocol_path"]) for entry in entries]
    raws = [path.parent / "result.json" for path in protocols]
    closed = [json.loads((REPO / ("research/core_identity_" + name + "_result.json")).read_text())
              for name in ("fields", "first_gold")]
    for entry, protocol, raw, result_entry in zip(entries, protocols, raws, closed):
        if (digest(protocol) != entry["protocol_sha256"] or result_entry["protocol_path"] != str(protocol)
                or result_entry["protocol_sha256"] != entry["protocol_sha256"]
                or result_entry["result_path"] != str(raw) or digest(raw) != result_entry["result_sha256"]
                or json.loads(raw.read_text())["protocol_sha256"] != digest(protocol)):
            raise ValueError("Historical protocol/result binding changed")
    old = json.loads(protocols[0].read_text())
    check_original_production(old)
    directory = require_storage(old["production_root"]) / "experiments" / ("luban0-identity-field-exposure-preregister-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    plan = exposure_plan(old, *(json.loads(path.read_text()) for path in raws), directory=directory,
                         session_id=uuid.uuid4().hex[:12], prepared=time.time())
    history = [{"path": str(path), "sha256": digest(path)} for path in protocols + raws]
    write_new(PROPOSAL, {"scope": "SOURCE ONLY native mean1 finite exposure proposal; no seal/model",
        "historical_controls": history, "arguments": plan["arguments"], "checkpoints": plan["training_checkpoints"],
        "unchanged_material_sha256": plan["groups_sha256"],
        "unchanged_protection_sha256": plan["retention_regressions_sha256"],
        "stop_rule": "Reject and stop this single-down_proj budget ladder if any fixed gate fails"})
    plan["historical_files"] += history + [{"path": str(PROPOSAL), "sha256": digest(PROPOSAL)}]
    directory.mkdir(mode=0o700, exist_ok=False)
    draft = directory / "draft-protocol.json"
    write_new(draft, plan)
    record = {"draft_path": str(draft), "draft_sha256": digest(draft), "prepared_without_model": True,
        "sealed": False, "model_go": False, "prefix_count": 115, "protection_count": 71,
        "protocol_descriptor_path": str(REPO / "research/core_identity_field_exposure_protocol.json")}
    write_new(PREPARATION, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    prepare()
