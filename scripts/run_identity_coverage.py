"""One preregistered coverage correction, private research DB, atomic adoption.

Preparation and sealing allocate no model. Run only after the coordinator's GO;
original qualifications are TRAIN-only, and frozen engineering is used once.
"""
import argparse
from contextlib import closing
import copy
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import uuid

from prepare_identity_coverage import clone_research_database, qualify_original_replay_pool
from run_luban0_batch_sweep import source_manifest
from verify_luban0_final import branch_off_generation, digest, retire, tensor_digest, verify_recovery, write_new
from verify_luban0_product import content_digest, recovery_cases
from verify_training_operators import OPERATOR_AUDIT_PROTOCOL, FIELD_OPERATOR_AUDIT_PROTOCOL, TOLERANCES, audit_training_operators
from luban.core_publication import (COLUMNS, content_sha, publish_isolated_core, _copy_immutable,
                                    PRESERVATION_CORRECTIONS, RELATION_DATASET, RELATION_ENGINEERING_IDS,
                                    FIELD_DATASET, validate_field_teaching, validate_relation_teaching, validate_relation_pool,
                                    validate_source_core_commit)
from luban.evolution import (TRAIN_CHECKPOINTS, RELATION_TRAIN_CHECKPOINTS, FIELD_TRAIN_CHECKPOINTS, FIELD_EXPOSURE_CHECKPOINTS, REPLAY_GENERATION_POLICY, consolidate, consolidation_samples,
                             restore_core, _target, _training_replay_generation)
from luban.evolution import field_first_token_policy
from luban.memory import Memory
from luban.retention import (_validate_groups, _validate_record, messages, protected_validation, read_core_retention,
                             validate_retention_conflicts)
from luban.settings import Settings, private_json, require_storage
from luban.training_checkpoint import TrainingPaused, TrainingSession, active_hashes, recover_commits, training_runs
from luban.process_lock import ModelProcessLock

REPOSITORY = Path(__file__).resolve().parents[1]
ARGUMENTS = {"steps": 512, "learning_rate": 1e-5, "distillation_weight": 5, "preserve_replay_subspace": True}
RELATION_ARGUMENTS = {**ARGUMENTS, "steps": 640}
FIELD_ARGUMENTS = {**ARGUMENTS, "steps": 832}
FIELD_WEIGHTED_ARGUMENTS = {**FIELD_ARGUMENTS, "field_first_token_weight": 4}
FIELD_EXPOSURE_ARGUMENTS = {**FIELD_ARGUMENTS, "steps": 1536}


def validate_protocol_policy(plan):
    """Legacy512 stays fixed; exact44/57 separately authorize640/832."""
    field = plan.get("relation_dataset") == FIELD_DATASET
    relation = plan.get("relation_dataset") in {RELATION_DATASET, FIELD_DATASET}
    arguments, checkpoints = ((FIELD_ARGUMENTS, FIELD_TRAIN_CHECKPOINTS) if field else
        (RELATION_ARGUMENTS, RELATION_TRAIN_CHECKPOINTS) if relation else (ARGUMENTS, TRAIN_CHECKPOINTS))
    if plan["arguments"].get("steps") == 1536:
        if not field or "field_loss_policy" in plan or "field_loss_operator_audit" in plan:
            raise ValueError("1536 requires the exact fixed57 native mean1 exposure diagnostic")
        arguments, checkpoints = FIELD_EXPOSURE_ARGUMENTS, FIELD_EXPOSURE_CHECKPOINTS
    if "field_loss_policy" in plan:
        samples = {row["id"]: row for row in plan["groups_before_original_baseline"]["new_training"]}
        audit = {"protocol": FIELD_OPERATOR_AUDIT_PROTOCOL, "tolerances": TOLERANCES,
            "samples": [samples.get("field-context-train-name-multi"), samples.get("field-context-train-position-multi")]}
        if not field or plan["field_loss_policy"] != field_first_token_policy() or plan.get("field_loss_operator_audit") != audit:
            raise ValueError("Exact fixed-nine TRAIN weighted objective and two native audits required")
        arguments = FIELD_WEIGHTED_ARGUMENTS
    elif "field_loss_operator_audit" in plan:
        raise ValueError("Weighted audit requires the exact weighted loss policy")
    if plan["arguments"] != arguments or plan["training_checkpoints"] != list(checkpoints):
        raise ValueError("Fixed coverage budget/checkpoint policy changed")
    if relation:
        groups, regressions = plan["groups_before_original_baseline"], plan["retention_regressions"]
        (validate_field_teaching if field else validate_relation_teaching)(groups, regressions)
        validate_relation_pool(plan["replay_training_candidates"])
        readonly = {row["id"]: row for row in regressions}
        engineering = plan["engineering_cases"]
        known = plan["old_known_development_regressions"]
        if (plan.get("protection_count") != (71 if field else 63) or plan.get("maximum_attempts") != 1
                or plan.get("initial_core_version") is not None or plan.get("warm_start_candidate") is not False
                or {row["id"] for row in engineering} != RELATION_ENGINEERING_IDS or len(engineering) != 8
                or len(known) != 27 or {row["id"] for row in known} != set(readonly) - RELATION_ENGINEERING_IDS
                or any(row != readonly.get(row["id"]) for row in engineering + known)
                or plan.get("minimum_replay_candidates") != 4 or len(plan["replay_training_candidates"]) != 12):
            raise ValueError("Exact63/71 obligations, new engineering8 and originalNone single attempt are required")
    return arguments


def verify_relation_recovery(runtime, plan, protected, report):
    """After native reload, re-answer all fixed63/71 obligations before publication."""
    if plan.get("relation_dataset") not in {RELATION_DATASET, FIELD_DATASET}:
        return
    count = 71 if plan["relation_dataset"] == FIELD_DATASET else 63
    if plan.get("protection_count") != count or len(protected) != count or len({row["id"] for row in protected}) != count:
        raise ValueError("Relation recovery requires all" + str(count) + " unique full records")
    replay = branch_off_generation(runtime, protected)
    report["recovery"]["full_protection_replay"] = replay
    passed = ([row["id"] for row in replay] == [row["id"] for row in protected]
              and all(row.get("passed") is True and row.get("complete") is True
                      and row.get("branch_enabled") is False and row.get("reference_used", False) is False for row in replay))
    report["recovery"]["full_protection_replay_passed"] = passed
    if not passed:
        report["recovery"]["passed"] = False
        raise ValueError("Native full reload generation failed at least one fixed" + str(count) + " obligation; retire candidate")


def execution_manifest():
    manifest = source_manifest(__file__)
    for path in (REPOSITORY / "scripts/prepare_identity_coverage.py", REPOSITORY / "scripts/prepare_identity_coverage_english.py",
                 REPOSITORY / "scripts/prepare_identity_known_correction.py",
                 REPOSITORY / "scripts/prepare_identity_preservation.py",
                 REPOSITORY / "scripts/prepare_identity_relations.py",
                 REPOSITORY / "scripts/prepare_identity_fields.py",
                 REPOSITORY / "scripts/prepare_identity_first_gold.py",
                 REPOSITORY / "scripts/prepare_identity_field_exposure.py",
                 REPOSITORY / "src/luban/core_publication.py",
                 REPOSITORY / "src/luban/upgrades.py"):
        manifest[str(path)] = digest(path)
    return manifest


def production_context(plan):
    memory = Memory(require_storage(plan["production_root"]))
    memory.model_identity = plan["base_identity"]
    store = memory.root / "modules/luban0-v1"
    return SimpleNamespace(memory=memory, store=store, identity=plan["base_identity"], active_core=None, branch=None)


def check_original_production(plan):
    context = production_context(plan)
    if (context.store / "active-core.json").exists():
        raise ValueError("Coverage run requires production original core None")
    if active_hashes(context) != plan["production_active_hashes"]:
        raise ValueError("Frozen production active files changed")
    if consolidation_samples(context) != plan["source_groups"]:
        raise ValueError("Frozen production original corpus changed")
    return context


def prepare():
    source = json.loads((REPOSITORY / "research/core_identity_coverage_protocol_draft.json").read_text())
    path = Path(source["draft_path"])
    if digest(path) != source["draft_sha256"]:
        raise ValueError("Fixed coverage data draft SHA mismatch")
    plan = json.loads(path.read_text())
    previous = Path(plan["prior_run_result_path"])
    if digest(previous) != plan["prior_run_result_sha256"]:
        raise ValueError("Historical failure report changed")
    prior = json.loads((previous.parent / "protocol.json").read_text())
    old = prior["groups"]
    if {key: len(rows) for key, rows in old.items()} != {"new_training": 34, "new_validation": 8, "replay_training": 44, "replay_validation": 17}:
        raise ValueError("Coverage draft needs the unchanged original103 corpus")
    root = require_storage(prior["storage_root"])
    directory = root / "experiments" / ("luban0-identity-coverage-preregister-" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    context = SimpleNamespace(memory=Memory(root), store=root / "modules/luban0-v1", active_core=None,
                              identity=plan["base_identity"], branch=None)
    context.memory.model_identity = plan["base_identity"]
    if (context.store / "active-core.json").exists() or consolidation_samples(context) != old:
        raise ValueError("Production must remain original None and unchanged103 source materials")
    # The current finite experiment has no accepted branch/router. Do not invent
    # a path resolver or silently change a production branch binding.
    if (context.store / "active.json").exists():
        raise ValueError("Coverage preparation needs the declared original branch-free state")
    timestamp = time.time()
    groups = copy.deepcopy(plan["groups_before_original_baseline"])
    for rows in groups.values():
        for row in rows:
            row.setdefault("created", timestamp)
        rows.sort(key=lambda row: row["id"])
    _validate_groups(groups)
    regressions = copy.deepcopy(plan["old_known_development_regressions"] + plan["engineering_cases"])
    for row in regressions:
        _validate_record(row, generation_only=True)
    plan.update(status="prepared immutable one-attempt protocol; implementations not sealed", sealed=False,
        production_root=str(root), research_root=str(directory / "research-storage"), model_path=prior["model_path"],
        source_groups=old, source_rows=[copy.deepcopy(row) for rows in old.values() for row in rows],
        production_active_hashes=active_hashes(context), session_id=uuid.uuid4().hex[:12], prepared=timestamp,
        data_draft_path=str(path), data_draft_sha256=digest(path), groups_before_original_baseline=groups,
        groups_sha256=content_digest(groups), retention_regressions=regressions,
        retention_regressions_sha256=content_digest(regressions), arguments=ARGUMENTS,
        training_checkpoints=list(TRAIN_CHECKPOINTS), policy_version=REPLAY_GENERATION_POLICY,
        operator_audit={"protocol": OPERATOR_AUDIT_PROTOCOL, "tolerances": copy.deepcopy(TOLERANCES),
                        "sample": copy.deepcopy(old["replay_training"][0])},
        historical_files=[{"path": str(p), "sha256": digest(p)} for p in
            (previous, previous.parent / "protocol.json", REPOSITORY / "research/luban0_v1_final.json",
             REPOSITORY / "research/luban0_v1_product.json", REPOSITORY / "research/luban0_v6_run.json",
             REPOSITORY / "research/luban0_v7_run.json", REPOSITORY / "research/luban0_batch_sweep.json")],
        selection="one bounded512attempt; fixed12 originalTRAINpool once min4; two consecutive fixed TRAIN checks NEWall+qualifiedREPLAYgenerationall+sameoldloss+minNEW2; then freshDEV8/oldREPLAY17 once, knownOLD8 once; no DEV-driven training/retry",
        snapshot_if_all_pass="derive all49 protections =freshNEWDEV8+oldREPLAYDEV17+knownoldNEW8+engineering16; all complete preserved through real backup/original restore/native full reload",
        publication="private SQLite native backup, same stable session ID; only after all outer gates/reload commit immutable files+complete matchinggroups+archiveexact12rawrows+activepointer+oneevent in recoverable productionjournal; failedtrial never counts as accepted",
        execution="single explicit research Runtime, native full paging=False including recovery; not production paging performance proof",
        resume="same source/session/records/Adam/RNG; no budget reset; interrupted engineering rejects instead of repeating; prepared production publication rolls forward verified intent",
        teacher_calls=0, maximum_attempts=1, script_sha256=None, execution_manifest=None)
    plan["source_rows_sha256"] = content_sha(plan["source_rows"])
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "draft-protocol.json"
    write_new(path, plan)
    descriptor = {"draft_path": str(path), "draft_sha256": digest(path), "prepared_without_model": True, "sealed": False}
    write_new(REPOSITORY / "research/core_identity_coverage_preparation.json", descriptor)
    print(json.dumps(descriptor, ensure_ascii=False), flush=True)


def seal(path, preparation=None):
    record = json.loads((preparation or REPOSITORY / "research/core_identity_coverage_preparation.json").read_text())
    if str(path) != record["draft_path"] or digest(path) != record["draft_sha256"]:
        raise ValueError("Fixed prepared coverage protocol changed")
    plan = json.loads(path.read_text())
    check_original_production(plan)
    validate_protocol_policy(plan)
    plan.update(sealed=True, status="sealed finite one-attempt engineering coverage correction",
                execution_manifest=execution_manifest(), script_sha256=digest(__file__))
    final = path.parent / "protocol.json"
    write_new(final, plan)
    descriptor = {"protocol_path": str(final), "protocol_sha256": digest(final), "sealed": True,
                  "prepared_without_model": True, "source_count": len(plan["execution_manifest"])}
    write_new(Path(record.get("protocol_descriptor_path", REPOSITORY / "research/core_identity_coverage_protocol.json")), descriptor)
    print(json.dumps(descriptor, ensure_ascii=False), flush=True)


def verify_training_prefix_alignment(runtime, plan):
    """Compare actual inference/TRAIN tokens only, before qualification or Adam."""
    if plan.get("prefix_alignment_required") is not True:
        return None
    checked = []
    groups = plan["groups_before_original_baseline"]
    for name, rows in (("new_training", groups["new_training"]),
                       ("replay_training", groups["replay_training"]),
                       ("replay_training_candidates", plan["replay_training_candidates"])):
        for row in rows:
            prefix = runtime._tokens(messages(row), "off")
            ids, labels = runtime._sample_tokens(row)
            if (list(prefix.shape[:1]) != [1] or prefix.ndim != 2 or ids.ndim != 2
                    or labels.shape != ids.shape or list(ids.shape[:1]) != [1]):
                raise ValueError("Training prefix alignment needs one complete sequence")
            prefix_ids, full_ids, target_labels = (value.detach().cpu().tolist()[0] for value in (prefix, ids, labels))
            first = next((i for i, value in enumerate(target_labels) if value != -100), None)
            gold = runtime.tokenizer.encode(row["answer"], add_special_tokens=False) + [runtime.tokenizer.eos_token_id]
            if (first is None or prefix_ids != full_ids[:first] or target_labels[:first] != [-100] * first
                    or full_ids[first:] != gold or target_labels[first:] != gold):
                raise ValueError("Training/inference prefix or gold+EOS token alignment failed: " + row["id"])
            checked.append({"group": name, "id": row["id"], "prefix_tokens": len(prefix_ids),
                "gold_eos_tokens": len(gold), "prefix_sha256": content_digest(prefix_ids),
                "gold_eos_sha256": content_digest(gold), "passed": True})
    return {"passed": True, "rows": checked, "count": len(checked),
            "scope": "only NEW TRAIN/REPLAY TRAIN/fixed candidates; no DEV or engineering tokenization/generation"}


def qualify_required_replay(runtime, plan):
    """Two predeclared old tasks must actually pass ORIGINAL OFF before training."""
    expected = {key for key, (_, _, role) in PRESERVATION_CORRECTIONS.items() if role == "replay"}
    rows = [row for row in plan["groups_before_original_baseline"]["replay_training"]
            if row["id"] in expected]
    if (set(plan.get("required_replay_training_ids", [])) != expected or {row["id"] for row in rows} != expected
            or runtime.active_core is not None or runtime.identity != plan["base_identity"]
            or tensor_digest(_target(runtime)[1]) != plan["original_tensor_sha256"]):
        raise ValueError("Required old-task original baseline identity or complete two-row binding changed")
    observed = branch_off_generation(runtime, rows)
    if [row["id"] for row in observed] != [row["id"] for row in rows]:
        raise ValueError("Required original replay baseline changed complete row coverage")
    qualified = []
    for source, actual in zip(rows, observed):
        if not (actual["passed"] and actual["complete"] and actual["answer"] == source["answer"]
                and actual["branch_enabled"] is False):
            continue
        row = copy.deepcopy(source)
        value = json.loads(row["provenance"])
        value.update(baseline_preservation_eligible=True, original_baseline_verified_core=None,
            original_baseline_verified_identity=runtime.identity, original_baseline_generation=actual,
            training_generation_gate=True, qualification_scope="current run original core, predeclared TRAIN only; formerly observed development goal")
        row["provenance"] = json.dumps(value, ensure_ascii=False)
        qualified.append(row)
    return {"passed": len(qualified) == 2, "qualified": qualified, "generation": observed,
            "scope": "both current original-core TRAIN tasks mandatory; no historical ID rewriting or omission"}


def qualify_required_replay_once(runtime, plan, directory, protocol_sha):
    marker = directory / "required-replay-baseline-started.json"
    if marker.exists():
        raise ValueError("Mandatory original replay baseline interrupted; reject without repeat")
    write_new(marker, {"protocol_sha256": protocol_sha, "ids": plan["required_replay_training_ids"]})
    return qualify_required_replay(runtime, plan)


def qualified_groups(plan, qualification, required_replay=None):
    if not qualification["passed"] or not 4 <= qualification["qualified_count"] <= 12:
        raise ValueError("Fixed original uppercase replay pool failed min4; no seeding or optimization")
    groups = copy.deepcopy(plan["groups_before_original_baseline"])
    if plan.get("required_replay_training_ids"):
        expected = set(plan["required_replay_training_ids"])
        if (not required_replay or required_replay.get("passed") is not True
                or len(required_replay["qualified"]) != 2
                or {row["id"] for row in required_replay["qualified"]} != expected):
            raise ValueError("Both predeclared original-correct replay tasks must qualify; no seeding")
        current = {row["id"]: row for row in groups["replay_training"]}
        for row in required_replay["qualified"]:
            original = current.get(row["id"])
            if original is None or any(row[key] != original[key] for key in COLUMNS if key != "provenance"):
                raise ValueError("Required replay qualifier changed an anchored training row")
            added = {"baseline_preservation_eligible", "original_baseline_verified_core", "original_baseline_verified_identity",
                     "original_baseline_generation", "training_generation_gate", "qualification_scope"}
            before, after = json.loads(original["provenance"]), json.loads(row["provenance"])
            if {k: v for k, v in before.items() if k not in added} != {k: v for k, v in after.items() if k not in added}:
                raise ValueError("Required replay qualifier changed gold/context/check/provenance")
            current[row["id"]] = copy.deepcopy(row)
        groups["replay_training"] = list(current.values())
    for row in qualification["qualified"]:
        row = copy.deepcopy(row)
        row["created"] = plan["prepared"]
        groups["replay_training"].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["id"])
    _validate_groups(groups)
    return groups


def seed_private_groups(memory, groups, source_rows):
    """Edit only the isolated clone. Every gold/source/provenance stays intact."""
    incoming = {row["id"]: row for rows in groups.values() for row in rows}
    with memory.lock, memory.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        for row in source_rows:
            current = db.execute("SELECT * FROM samples WHERE id=?", (row["id"],)).fetchone()
            if current is None or dict(current) != row:
                raise ValueError("Private clone differs from the anchored source corpus")
        for row in incoming.values():
            current = db.execute("SELECT * FROM samples WHERE id=?", (row["id"],)).fetchone()
            if current is not None and dict(current) != row:
                raise ValueError("Private seed conflicts with an existing stable ID")
        for row in source_rows:
            if row["id"] not in incoming:
                db.execute("UPDATE samples SET approved=0,training_allowed=0 WHERE id=?", (row["id"],))
        for row in incoming.values():
            db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row[key] for key in COLUMNS))


def initialize_isolation(plan, directory):
    root = Path(plan["research_root"])
    marker = directory / "isolation.json"
    if marker.exists():
        return json.loads(marker.read_text())
    if root.exists():
        raise ValueError("Interrupted private clone initialization cannot be reused or silently overwritten")
    context = check_original_production(plan)
    root.mkdir(mode=0o700)
    record = clone_research_database(context.memory.root, root)
    store = root / "modules/luban0-v1"
    store.mkdir(parents=True)
    for name, sha in plan["production_active_hashes"].items():
        if sha is not None:
            _copy_immutable(context.store / name, store / name)
    _copy_immutable(context.store / "graph-verification.json", store / "graph-verification.json")
    record.update(production_active_hashes=plan["production_active_hashes"], source_rows_sha256=plan["source_rows_sha256"])
    write_new(marker, record)
    return record


def require_untouched_engineering(directory, report):
    if (directory / "engineering-started.json").exists() and not report.get("engineering_complete"):
        raise ValueError("Engineering was interrupted; reject without repeating its frozen questions")


def recover_prepared_publication(plan, directory, report):
    """Finish a verified outer acceptance after a crash, without model allocation."""
    descriptor = report.get("publication_prepared")
    if descriptor is None:
        return False
    acceptance_path = directory / ("coverage-acceptance-" + plan["session_id"] + ".json")
    if descriptor != {"name": acceptance_path.name, "sha256": digest(acceptance_path)}:
        raise ValueError("Prepared outer acceptance changed; cannot recover publication")
    acceptance = json.loads(acceptance_path.read_text())
    if (acceptance.get("protocol_sha256") != report["protocol_sha256"]
            or acceptance.get("session_id") != plan["session_id"]
            or any(value is not True for value in acceptance["outer_gates"].values())
            or set(acceptance["outer_gates"]) != {"fresh_development", "known_regression", "engineering", "recovery", "protection_snapshot"}):
        raise ValueError("Prepared outer acceptance is incomplete or from another attempt")
    context = production_context(plan)
    row = next((value for value in training_runs(context.memory, context.identity) if value["id"] == plan["session_id"]), None)
    if row is None or row["state"] not in {"committing", "committed"} or row.get("product_accepted") is not True:
        raise ValueError("No prepared production intent; reject without reusing engineering or restarting training")
    if row["commit"].get("material_publication", {}).get("acceptance", {}).get("report_sha256") != descriptor["sha256"]:
        raise ValueError("Production intent does not bind this fixed outer acceptance")
    with closing(ModelProcessLock(context.identity, row["binding"]["device"])):
        recover_commits(context.memory, context.store, context.identity)
        committed = next(value for value in training_runs(context.memory, context.identity) if value["id"] == plan["session_id"])
        version = "core-" + plan["session_id"]
        pointer = json.loads((context.store / "active-core.json").read_text()) if (context.store / "active-core.json").exists() else None
        if committed["state"] != "committed" or pointer != row["commit"]["pointers"][0]["after"]:
            raise ValueError("Prepared publication did not recover to its verified active core")
        for name, sha in committed["commit"]["artifacts"].items():
            if digest(context.store / name) != sha:
                raise ValueError("Recovered production immutable artifact SHA differs")
        context.active_core = version
        snapshot = read_core_retention(context)
        if (snapshot["groups"] != committed["records"]["groups"]
                or snapshot["regression_validation"] != committed["records"]["retention_regressions"]
                or content_digest(protected_validation(context, include_learning=False)) != acceptance["retention"]["protected_sha256"]):
            raise ValueError("Recovered production changed its complete protected obligations")
        from luban.upgrades import read_upgrade_records
        weights_sha = committed["commit"]["report"]["checkpoint"]["sha256"]
        if not any(record["weights_sha256"] == weights_sha for record in read_upgrade_records(context.memory.root)):
            raise ValueError("Recovered product acceptance is not a verified countable upgrade")
    report.update(passed=True, product_accepted=True, active_core=version, publication_pending=False,
        publication_recovered_without_model=True, error=None)
    return True


def run(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError("Fixed coverage protocol SHA mismatch")
    plan, directory = json.loads(path.read_text()), path.parent
    arguments = validate_protocol_policy(plan)
    if (plan.get("sealed") is not True or plan["execution_manifest"] != execution_manifest()
            or plan["script_sha256"] != digest(__file__)
            or plan["policy_version"] != REPLAY_GENERATION_POLICY):
        raise ValueError("Coverage execution implementations/policy are not the sealed versions")
    result_path = directory / "result.json"
    report = json.loads(result_path.read_text()) if result_path.exists() else {
        "protocol_sha256": expected_sha, "session_id": plan["session_id"], "passed": False,
        "scope": "finite engineering coverage correction; no independent scientific final or optimal-count claim",
        "teacher_calls": 0, "started": time.time(), "model_started": False, "engineering": []}
    if report.get("passed") or report.get("error") and not report.get("publication_pending"):
        raise ValueError("This attempt already ended; never initialize a second attempt")
    runtime = None
    publication_started = False
    production = production_context(plan)
    def persist():
        private_json(result_path, report)
        private_json(Path(plan.get("run_descriptor_path", REPOSITORY / "research/core_identity_coverage_run.json")), report)
    try:
        if recover_prepared_publication(plan, directory, report):
            persist()
            return
        require_untouched_engineering(directory, report)
        check_original_production(plan)
        report["isolation"] = initialize_isolation(plan, directory)
        manifest_path = directory / "execution-manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != plan["execution_manifest"]:
                raise ValueError("Actual execution source manifest changed")
        else:
            write_new(manifest_path, plan["execution_manifest"])
        with tempfile.TemporaryDirectory(prefix="luban0-coverage-settings-") as home:
            from luban.runtime import Runtime
            settings = Settings(home)
            settings.update({"storage_root": plan["research_root"], "model_path": plan["model_path"], "local_thinking": "off", "temperature": 0})
            runtime = Runtime(settings, Memory(plan["research_root"]), research=True)
            runtime.load = partial(runtime.load, paging=False)
            report["model_load_requested"] = True
            persist()
            print(json.dumps({"model_load_requested": True, "protocol_sha256": expected_sha, "session_id": plan["session_id"]}), flush=True)
            runtime.load()
            report["model_started"] = True
            persist()
            print(json.dumps({"model_started": True, "protocol_sha256": expected_sha, "session_id": plan["session_id"]}), flush=True)
            restore_core(runtime, None)
            if runtime.identity != plan["base_identity"] or tensor_digest(_target(runtime)[1]) != plan["original_tensor_sha256"]:
                raise ValueError("Original base identity/tensor SHA mismatch before audit or training")
            if plan.get("prefix_alignment_required") is True:
                aligned = verify_training_prefix_alignment(runtime, plan)
                if "prefix_alignment" in report and report["prefix_alignment"] != aligned:
                    raise ValueError("Training prefix alignment changed on resume")
                report["prefix_alignment"] = aligned
                persist()
            if "operator_audit" not in report:
                report["operator_audit"] = audit_training_operators(runtime, plan["operator_audit"]["sample"])
                persist()
            if report["operator_audit"]["passed"] is not True or report["operator_audit"]["restored"] is not True:
                raise ValueError("Native training-operator audit/restoration failed")
            if "field_loss_policy" in plan:
                if "field_loss_operator_audits" not in report:
                    report["field_loss_operator_audits"] = [audit_training_operators(runtime, sample,
                        field_first_token_weight=4) for sample in plan["field_loss_operator_audit"]["samples"]]
                    persist()
                if (len(report["field_loss_operator_audits"]) != 2 or any(row["passed"] is not True
                        or row["restored"] is not True or row.get("field_loss_policy") != plan["field_loss_policy"]
                        or row.get("protocol_version") != FIELD_OPERATOR_AUDIT_PROTOCOL
                        or row.get("sample_id") != sample["id"]
                        or row.get("tolerances") != TOLERANCES[str(runtime.dtype)]
                        for row, sample in zip(report["field_loss_operator_audits"], plan["field_loss_operator_audit"]["samples"]))):
                    raise ValueError("Weighted first-gold native objective audit/restoration failed")
            if "original_pool" not in report:
                marker = directory / "qualification-started.json"
                if marker.exists():
                    raise ValueError("Original TRAIN-only qualification was interrupted; no adaptive replay or repeat")
                write_new(marker, {"protocol_sha256": expected_sha, "pool_sha256": plan["replay_candidate_sha256"]})
                report["original_pool"] = qualify_original_replay_pool(runtime, plan)
                persist()
                for row in report["original_pool"]["generation"]:
                    print(json.dumps({"phase": "original_replay_train_qualification", "id": row["id"], "passed": row["passed"], "complete": row["complete"]}), flush=True)
            if plan.get("required_replay_training_ids") and "required_replay_baseline" not in report:
                report["required_replay_baseline"] = qualify_required_replay_once(runtime, plan, directory, expected_sha)
                persist()
                for row in report["required_replay_baseline"]["generation"]:
                    print(json.dumps({"phase": "mandatory_original_replay_baseline", "id": row["id"],
                                      "passed": row["passed"], "complete": row["complete"]}), flush=True)
            groups = qualified_groups(plan, report["original_pool"], report.get("required_replay_baseline"))
            _training_replay_generation(runtime, groups)
            frozen = directory / "qualified-groups.json"
            if frozen.exists():
                if json.loads(frozen.read_text()) != groups:
                    raise ValueError("Qualified frozen corpus changed on resume")
            else:
                write_new(frozen, groups)
            regressions = plan["retention_regressions"]
            session_row = next((row for row in training_runs(runtime.memory, runtime.identity) if row["id"] == plan["session_id"]), None)
            if session_row is None:
                seed_private_groups(runtime.memory, groups, plan["source_rows"])
                if consolidation_samples(runtime) != groups:
                    raise ValueError("Private seeded complete corpus differs from the fixed manifest")
            if session_row and session_row["state"] == "rejected":
                raise ValueError("The single training attempt was rejected; no restart")
            if session_row and session_row["state"] == "committed":
                trained = validate_source_core_commit(session_row, plan, groups, regressions, store=runtime.store)
                restore_core(runtime, trained["version"])
            else:
                session = TrainingSession(runtime, "evolution", arguments,
                    {"groups": groups, "protection": [], "retention_regressions": regressions},
                    run_id=plan["session_id"], scope="research")
                runtime.training_session = session
                try:
                    trained = consolidate(runtime, **arguments)
                except TrainingPaused:
                    session.pause()
                    report["paused"] = True
                    raise
                finally:
                    runtime.training_session = None
            report["training"] = trained
            persist()
            if not trained["promoted"]:
                raise ValueError("Bounded TRAIN/freshDEV/oldREPLAY gates failed; no engineering generated")
            version = trained["version"]
            if "known_regression" not in report:
                report["known_regression"] = branch_off_generation(runtime, plan["old_known_development_regressions"])
                persist()
            if not all(row["passed"] for row in report["known_regression"]):
                raise ValueError("Immutable known old NEW regressions failed; no engineering generated")
            if not report.get("engineering_complete"):
                write_new(directory / "engineering-started.json", {"version": version, "protocol_sha256": expected_sha})
                for row in plan["engineering_cases"]:
                    generated = branch_off_generation(runtime, [row])[0]
                    report["engineering"].append(generated)
                    persist()
                    print(json.dumps({"phase": "engineering", "id": row["id"], "passed": generated["passed"]}), flush=True)
                report["engineering_complete"] = True
                persist()
            if (not all(row["passed"] for row in report["engineering"])
                    or [row["id"] for row in report["engineering"]] != [row["id"] for row in plan["engineering_cases"]]):
                raise ValueError("Fixed one-use engineering failed; no fallback, variant additions or retuning")
            snapshot = read_core_retention(runtime)
            protected = protected_validation(runtime, include_learning=False)
            expected_protected = groups["new_validation"] + groups["replay_validation"] + regressions
            if plan.get("relation_dataset") in {RELATION_DATASET, FIELD_DATASET}:
                count = 71 if plan["relation_dataset"] == FIELD_DATASET else 63
                if len(expected_protected) != count or len({row["id"] for row in expected_protected}) != count:
                    raise ValueError("Relation teaching requires all" + str(count) + " unique complete protection records")
            if snapshot["groups"] != groups or snapshot["regression_validation"] != regressions or protected != expected_protected:
                raise ValueError("Complete retention obligations differ from frozen full records")
            report["retention"] = {"protected_count": len(expected_protected), "protected_sha256": content_digest(protected),
                "content_sha256": snapshot["content_sha256"], "file_sha256": digest(runtime.store / (version + ".retention.json")),
                "counts": {**{key: len(rows) for key, rows in groups.items()}, "regression_validation": len(regressions)}}
            verify_recovery(runtime, version, recovery_cases(plan["engineering_cases"], report["engineering"]), report["engineering"], report)
            if (read_core_retention(runtime) != snapshot or protected_validation(runtime, include_learning=False) != protected
                    or digest(runtime.store / (version + ".retention.json")) != report["retention"]["file_sha256"]):
                raise ValueError("Genuine backup/native full reload changed complete protected records or SHA")
            verify_relation_recovery(runtime, plan, protected, report)
            if execution_manifest() != plan["execution_manifest"] or any(digest(row["path"]) != row["sha256"] for row in plan["historical_files"]):
                raise ValueError("Frozen execution sources or historical raw failures changed")
            check_original_production(plan)
            report["outer_gates"] = {"fresh_development": all(trained["gates"].values()),
                "known_regression": True, "engineering": True, "recovery": report["recovery"]["passed"], "protection_snapshot": True}
            persist()
            acceptance_path = directory / ("coverage-acceptance-" + plan["session_id"] + ".json")
            if acceptance_path.exists():
                raise ValueError("Acceptance publication was interrupted; recover its journal instead of rewriting evidence")
            write_new(acceptance_path, report)
            report["publication_prepared"] = {"name": acceptance_path.name, "sha256": digest(acceptance_path)}
            persist()
            acceptance = {"protocol_sha256": expected_sha, "report_path": acceptance_path.name,
                "report_sha256": digest(acceptance_path), "gates": report["outer_gates"], "teacher_calls": 0}
            publication_started = True
            report["production_publication"] = publish_isolated_core(runtime, production.memory, production.store,
                plan, groups, regressions, plan["source_rows"], acceptance_path, acceptance)
            report.update(passed=True, active_core=version, product_accepted=True, paused=False)
            from luban.upgrades import read_upgrade_records
            if not any(record["weights_sha256"] == trained["checkpoint"]["sha256"] for record in read_upgrade_records(production.memory.root)):
                raise ValueError("Actual production publication is not countable through its artifact/journal reader")
            persist()
    except TrainingPaused:
        raise
    except BaseException as error:
        report.update(error=str(error) or type(error).__name__, passed=False)
        if publication_started:
            row = next((row for row in training_runs(production.memory, plan["base_identity"]) if row["id"] == plan["session_id"]), None)
            if row and row["state"] in {"committing", "committed"}:
                # All outer gates already passed; prepared publication owns this
                # intent. Never call a verified product a training rejection.
                report["publication_pending"] = row["state"] == "committing"
                report["passed"] = row["state"] == "committed"
                report["product_accepted"] = row["state"] == "committed"
        if runtime is not None and not report["passed"]:
            report["retirement"] = retire(runtime, None)
            if "method" not in report["retirement"]:
                report["retirement"]["tensor_sha256"] = tensor_digest(_target(runtime)[1])
                if report["retirement"]["tensor_sha256"] != plan["original_tensor_sha256"]:
                    raise RuntimeError("Retired original tensor SHA differs") from error
        elif runtime is None and not report["passed"]:
            pointer = Path(plan["research_root"]) / "modules/luban0-v1/active-core.json"
            if pointer.is_symlink():
                raise ValueError("Refusing a symlink research pointer") from error
            if pointer.exists():
                target = json.loads(pointer.read_text())
                if target.get("version") != "core-" + plan["session_id"]:
                    raise ValueError("Cannot retire another research candidate") from error
                pointer.unlink()
            report["retirement"] = {"version": None, "scope": "private on-disk pointer only; no live model or new tensor measurement"}
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        persist()
        if runtime is not None:
            runtime._release_model()
        print(json.dumps({"passed": report["passed"], "session_id": plan["session_id"], "error": report.get("error"),
            "publication_pending": report.get("publication_pending", False), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--seal", type=Path)
    action.add_argument("--print-plan", type=Path)
    action.add_argument("--run", type=Path)
    parser.add_argument("--preparation", type=Path, help="explicit fixed preparation descriptor used only by --seal")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.seal:
        seal(args.seal, args.preparation)
    elif args.print_plan:
        print(args.print_plan.read_text())
    elif not args.protocol_sha256:
        parser.error("--run requires --protocol-sha256")
    else:
        run(args.run, args.protocol_sha256)


if __name__ == "__main__":
    main()
