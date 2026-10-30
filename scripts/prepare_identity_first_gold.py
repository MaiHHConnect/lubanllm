"""Independent fixed57/71/832 first-gold objective; preparation only, no model."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from run_identity_coverage import FIELD_WEIGHTED_ARGUMENTS, check_original_production, validate_protocol_policy
from verify_luban0_final import digest, write_new
from verify_training_operators import FIELD_OPERATOR_AUDIT_PROTOCOL, TOLERANCES
from luban.evolution import field_first_token_policy
from luban.settings import require_storage

REPO = Path(__file__).resolve().parents[1]
PREPARATION = REPO / "research/core_identity_first_gold_preparation.json"
PROPOSAL = REPO / "research/core_identity_first_gold_draft.json"


def weighted_plan(old, result, diagnostic, *, directory, session_id, prepared):
    """Change only the nine-row TRAIN objective, retaining every complete record."""
    validate_protocol_policy(old)
    training = result["training"]
    if (old["arguments"].get("field_first_token_weight", 1) != 1 or old["arguments"]["steps"] != 832
            or result["passed"] is not False or training["promoted"] is not False
            or training["completed_steps"] != 832 or training["training_selection"]["selected_step"] is not None
            or result["retirement"] != {"version": None, "previous_version": None, "restored": True,
                                       "tensor_sha256": old["original_tensor_sha256"]}
            or result.get("engineering") != [] or any(key in result for key in ("known_regression", "recovery", "production_publication"))
            or diagnostic["passed"] is not True or diagnostic["preserved"] is not True
            or diagnostic["optimizer_steps"] != 0 or diagnostic["generation_calls"] != 0
            or diagnostic["diagnostic"]["restoration"]["passed"] is not True
            or diagnostic["diagnostic"]["restoration"]["actual_original_sha256"] != old["original_tensor_sha256"]):
        raise ValueError("Requires strict rejected832 and genuine restored nine-TRAIN no-update diagnostic")
    plan = copy.deepcopy(old)
    policy = field_first_token_policy()
    rows = {row["id"]: row for row in old["groups_before_original_baseline"]["new_training"]}
    if any(set(row["id"] for row in state["rows"]) != set(policy["training_ids"])
           for state in diagnostic["diagnostic"]["states"]) or [state["state"] for state in diagnostic["diagnostic"]["states"]] != ["originalNone", "rejected832"]:
        raise ValueError("Diagnostic must cover exactly the original nine TRAIN rows in two states")
    plan.update(status="prepared independent first-gold objective; NOT sealed/run", sealed=False,
        prepared=prepared, session_id=session_id, research_root=str(directory / "research-storage"),
        arguments=copy.deepcopy(FIELD_WEIGHTED_ARGUMENTS), field_loss_policy=policy,
        field_loss_operator_audit={"protocol": FIELD_OPERATOR_AUDIT_PROTOCOL,
            "tolerances": copy.deepcopy(TOLERANCES),
            "samples": [copy.deepcopy(rows["field-context-train-name-multi"]), copy.deepcopy(rows["field-context-train-position-multi"])]},
        reason="Same57/71/832 and round-robin. Fixed nine TRAIN first causal gold weight4, remaining gold/EOS1, normalize n+3; no curriculum or new examples.",
        objective="Native CE except the exact nine NEW TRAIN rows use normalized first-gold weighting; KL5/replay/native quality losses unchanged. Different objective, not equivalence to old mean CE.",
        repeat_budget_scope="Independent originalNone new832; no rejected832 weight/master/Adam/RNG/session continuation.",
        source_binding="Unsealed first-gold objective; new same-dtype weighted operator audits, independent CPU review and explicit seal/MODEL GO required.",
        run_descriptor_path=str(REPO / "research/core_identity_first_gold_run.json"),
        script_sha256=None, execution_manifest=None)
    validate_protocol_policy(plan)
    for key in ("groups_before_original_baseline", "source_groups", "source_rows", "superseded_source_ids",
                "replay_training_candidates", "retention_regressions", "engineering_cases", "old_known_development_regressions",
                "training_checkpoints", "seed", "operator_audit", "original_tensor_sha256", "protection_count"):
        if plan[key] != old[key]:
            raise ValueError("Fixed data, rights, exposures/checkpoints or legacy native audit changed")
    return plan


def prepare():
    if PREPARATION.exists() or PROPOSAL.exists():
        raise FileExistsError("Never overwrite an existing first-gold preparation")
    entry = json.loads((REPO / "research/core_identity_fields_protocol.json").read_text())
    prior = Path(entry["protocol_path"])
    raw = prior.parent / "result.json"
    diag_entry = json.loads((REPO / "research/field_training_diagnostic_result.json").read_text())
    diagnostic_path = Path(diag_entry["raw_result"])
    if digest(prior) != entry["protocol_sha256"] or digest(diagnostic_path) != diag_entry["raw_result_sha256"]:
        raise ValueError("Historical original protocol or no-update diagnostic SHA changed")
    old, result, diagnostic = json.loads(prior.read_text()), json.loads(raw.read_text()), json.loads(diagnostic_path.read_text())
    if result["protocol_sha256"] != digest(prior):
        raise ValueError("Rejected832 result belongs to another protocol")
    check_original_production(old)
    directory = require_storage(old["production_root"]) / "experiments" / ("luban0-identity-first-gold-preregister-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    plan = weighted_plan(old, result, diagnostic, directory=directory, session_id=uuid.uuid4().hex[:12], prepared=time.time())
    plan["historical_files"] += [{"path": str(path), "sha256": digest(path)} for path in (prior, raw, diagnostic_path)]
    write_new(PROPOSAL, {"scope": "SOURCE ONLY static first-gold proposal; no seal/model",
        "previous_protocol": {"path": str(prior), "sha256": digest(prior)},
        "previous_result": {"path": str(raw), "sha256": digest(raw)},
        "diagnostic": {"path": str(diagnostic_path), "sha256": digest(diagnostic_path)},
        "field_loss_policy": plan["field_loss_policy"], "arguments": plan["arguments"],
        "unchanged_material_sha256": old["groups_sha256"], "unchanged_protection_sha256": old["retention_regressions_sha256"]})
    plan["historical_files"].append({"path": str(PROPOSAL), "sha256": digest(PROPOSAL)})
    directory.mkdir(mode=0o700, exist_ok=False)
    draft = directory / "draft-protocol.json"
    write_new(draft, plan)
    record = {"draft_path": str(draft), "draft_sha256": digest(draft), "prepared_without_model": True,
        "sealed": False, "model_go": False, "prefix_count": 115, "protection_count": 71,
        "protocol_descriptor_path": str(REPO / "research/core_identity_first_gold_protocol.json")}
    write_new(PREPARATION, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    prepare()
