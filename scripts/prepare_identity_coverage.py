"""Freeze finite coverage data; qualify replay only from correct original output.

This CLI only prepares or inspects data. It never loads a model, seeds production,
seals execution implementations, or trains. Qualification is called by the future
single-attempt runner after its original-core SHA and exclusive-process checks.
"""
import argparse
import copy
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import uuid

from verify_luban0_final import branch_off_generation, digest, tensor_digest, write_new
from verify_luban0_product import content_digest
from luban.evolution import _target
from luban.retention import _validate_groups, _validate_record, normalized_context
from luban.settings import require_storage

REPOSITORY = Path(__file__).resolve().parents[1]


def prepare():
    descriptor = REPOSITORY / "research/core_identity_coverage_protocol_draft.json"
    if descriptor.exists():
        raise FileExistsError("Coverage draft exists; do not replace its fixed pool")
    previous = json.loads((REPOSITORY / "research/core_identity_repair_draft.json").read_text())
    previous_path = Path(previous["draft_path"])
    if digest(previous_path) != previous["draft_sha256"]:
        raise ValueError("Previous static draft changed")
    plan = json.loads(previous_path.read_text())
    pool = copy.deepcopy(plan["replay_training_candidates"])
    identity = plan["base_identity"]
    for index, word in enumerate(("banana", "monkey", "TableTop", "audio_7"), start=8):
        answer = word.upper()
        provenance = {"base_identity": identity, "evolution_role": "replay",
            "dataset": "identity-core-coverage-draft", "protocol_id": f"coverage-replay-candidate-{index}",
            "split_family": f"coverage-replay-candidate-{index}", "program_verified": True,
            "human_verified": False, "verification_evidence": f"Python str.upper: {word!r} -> {answer!r}",
            "authoring_scope": "locally self-authored under explicit user-directed teacher/review authorization",
            "baseline_preservation_eligible": None,
            "messages": [{"role": "system", "content": "请直接按字符处理规则完成当前任务。"},
                {"role": "user", "content": f"对「{word}」执行英文字母大写转换，其余字符保留。请只返回转换后的完整字符串，不加引号。"}],
            "generation_check": {"type": "exact", "expected": answer, "max_new_tokens": 192}}
        pool.append({"id": f"coverage-replay-candidate-{index}", "prompt": provenance["messages"][-1]["content"],
            "answer": answer, "source": "synthetic", "approved": 0, "training_allowed": True,
            "purpose": "training", "provenance": json.dumps(provenance, ensure_ascii=False)})
    groups = plan["groups_before_original_baseline"]
    _validate_groups(groups)
    contexts = {normalized_context(row) for rows in groups.values() for row in rows}
    for row in pool:
        checked = copy.deepcopy(row)
        checked["approved"] = 1
        _validate_record(checked)
        if normalized_context(row) in contexts or row["approved"]:
            raise ValueError("Unqualified pool overlaps a declared group or is prematurely approved")
        contexts.add(normalized_context(row))
    if len(pool) != 12:
        raise ValueError("Fixed replay pool must have exactly twelve candidates")
    plan.update(status="static amended data protocol; unsealed, no model or database writes",
        previous_static_draft_path=str(previous_path), previous_static_draft_sha256=digest(previous_path),
        replay_training_candidates=pool, replay_candidate_sha256=content_digest(pool), minimum_replay_candidates=4,
        baseline_policy="generate fixed12 original branchOFF greedily with EOS and unchanged exactgold; qualify every passing row in declared order; require>=4 otherwise abort before optimizer; failedgold/raw stay diagnostics, no adaptive replacement or reassignment",
        planned_counts_if_baseline_qualifies={"new_training": 34, "new_validation": 8,
            "replay_training": "44 plus 4..12 baseline-qualified uppercase rows", "replay_validation": 17},
        training_selection_change_proposed="two consecutive fixed TRAIN checks: allNEW34 generation/EOS, every qualified uppercaseREPLAYTRAIN exactgeneration/EOS, unchanged allREPLAYTRAIN lossbudget and minNEW>=2; nofreshDEVfeedback",
        isolation="SQLite native backup to new research root; training materials changed only in that clone; production samples/journal/corepointer untouched until complete acceptance and validated publication",
        publication="candidate in isolatedresearchroot may be DEVvalidated first; production remains originalNone until freshDEV8+allold25knownregressions+fixedunseen16+realbackup/reload+fullprotectionSHAallpass; production journal/materials/artifacts must match snapshot atomically; needs implemented reviewed publisher before sealing",
        source_binding="pending engine exit, protective TRAIN generation gate, root router safety patch, and reviewed publication; do not seal implementations now")
    directory = require_storage(Path(previous_path).parents[2]) / "experiments" / (
        "luban0-identity-coverage-pool-draft-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "draft-data-protocol.json"
    write_new(path, plan)
    record = {"draft_path": str(path), "draft_sha256": digest(path), "prepared_without_model": True,
              "sealed": False, "baseline_verified": False, "pool_count": 12, "minimum_qualified": 4}
    write_new(descriptor, record)
    print(json.dumps(record, ensure_ascii=False))


def qualify_original_replay_pool(runtime, plan):
    """No DEV, optimization, substitution, database write, or model allocation."""
    pool = plan["replay_training_candidates"]
    if (runtime.active_core is not None or runtime.identity != plan["base_identity"]
            or tensor_digest(_target(runtime)[1]) != plan["original_tensor_sha256"]
            or len(pool) != 12 or content_digest(pool) != plan["replay_candidate_sha256"]
            or plan["minimum_replay_candidates"] != 4):
        raise ValueError("Original replay qualification binding or fixed pool mismatch")
    observations = branch_off_generation(runtime, pool)
    if [row["id"] for row in observations] != [row["id"] for row in pool]:
        raise ValueError("Original replay qualification changed row order/coverage")
    qualified = []
    for source, observed in zip(pool, observations):
        if not observed["passed"]:
            continue
        if not observed["complete"] or observed["answer"] != source["answer"] or observed["branch_enabled"]:
            raise ValueError("Qualification cannot treat incorrect, truncated or branch output as original knowledge")
        row = copy.deepcopy(source)
        row["approved"] = 1
        value = json.loads(row["provenance"])
        value.update(baseline_preservation_eligible=True, original_baseline_verified_core=None,
            original_baseline_verified_identity=runtime.identity, original_baseline_generation=observed,
            training_generation_gate=True, qualification_scope="original TRAIN-only baseline, never heldout feedback")
        row["provenance"] = json.dumps(value, ensure_ascii=False)
        _validate_record(row)
        qualified.append(row)
    return {"passed": len(qualified) >= 4, "minimum_qualified": 4, "qualified_count": len(qualified),
            "qualified": qualified, "generation": observations,
            "scope": "fixed12 training-only original baseline; no adaptive word replacement or gold changes"}


def clone_research_database(source_root, research_root):
    """SQLite backup reads production and exclusively creates a private clone."""
    source = Path(source_root) / "app/memory.sqlite3"
    target = Path(research_root) / "app/memory.sqlite3"
    require_storage(source_root)
    require_storage(research_root)
    if source.is_symlink() or target.exists() or target.parent.is_symlink():
        raise ValueError("Research database clone needs a fresh safe target")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original, closing(sqlite3.connect(target)) as cloned:
        original.backup(cloned)
        if cloned.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Research database clone integrity failed")
    return {"source_path": str(source), "clone_path": str(target), "clone_sha256": digest(target),
            "scope": "private clone only; no production seeding or promotion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--inspect", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        plan = json.loads(args.inspect.read_text())
        print(json.dumps({"status": plan["status"], "groups": {k: len(v) for k, v in plan["groups_before_original_baseline"].items()},
            "fixed_pool": [{"id": r["id"], "answer": r["answer"]} for r in plan["replay_training_candidates"]],
            "minimum_qualified": plan["minimum_replay_candidates"], "engineering_sha256": plan["engineering_cases_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
