#!/usr/bin/env python3
"""Produce one bounded, executable contract for recursive expert admission."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from luban.memory import Memory
from luban.training_checkpoint import TrainingSession
from promote_identity_expert import transactional_replace, validate_original_protection


def rollback_probe():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        sources, targets = root / "sources", root / "targets"
        sources.mkdir()
        targets.mkdir()
        (sources / "one").write_bytes(b"new-one")
        (sources / "two").write_bytes(b"new-two")
        (targets / "one").write_bytes(b"old-one")
        (targets / "two").write_bytes(b"old-two")
        real_replace = os.replace
        calls = {"count": 0}

        def fail_once(source, target):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("contract rollback injection")
            return real_replace(source, target)

        try:
            with patch("promote_identity_expert.os.replace", side_effect=fail_once):
                transactional_replace(((sources / "one", targets / "one"),
                                       (sources / "two", targets / "two")))
        except OSError as error:
            return ((str(error) == "contract rollback injection"
                     and (targets / "one").read_bytes() == b"old-one"
                     and (targets / "two").read_bytes() == b"old-two"))
        return False


def checkpoint_probe():
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        store = root / "modules"
        store.mkdir()
        task = SimpleNamespace(memory=Memory(root), store=store,
                               identity="recursive-contract-test", active_core=None,
                               branch=SimpleNamespace(expert_id=None), device="cpu", dtype="torch.float32")
        baseline = store / "original-core-protection.json"
        baseline.write_text("baseline-v1")
        session = TrainingSession(task, "learning", {}, [])
        baseline.write_text("baseline-v2")
        try:
            TrainingSession(task, "learning", {}, [], run_id=session.id)
        except ValueError as error:
            return "检查点与当前模型" in str(error)
        return False


def parent_continuity_probe(candidate):
    """Require a later candidate to identify and replay its immediate parent."""
    round_index = candidate.get("recursive_round", 1)
    if round_index < 2:
        return True
    report = candidate.get("learning_report", {})
    parent = candidate.get("parent_expert_id")
    return bool(
        candidate.get("parent_identity_passed") is True
        and isinstance(parent, str) and bool(parent)
        and report.get("previous_expert") == parent
        and report.get("promoted") is True
        and report.get("active_expert") not in {None, parent}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--original-protection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("递归合同输出必须是新文件，拒绝覆盖")
    candidate = json.loads(Path(args.candidate).read_text())
    route = json.loads(Path(args.route).read_text())
    original = json.loads(Path(args.original_protection).read_text())
    identity = candidate.get("learning_report", {}).get("base_identity")
    protection = validate_original_protection(original, identity)
    route_result = route.get("route_calibration", {})
    expert_proof = route_result.get("expert_route_proof", {})
    scope = route_result.get("expert_scope", {})
    gates = {
        "original_core_protection": protection["passed"],
        "expert_closed_book": expert_proof.get("passed") is True and expert_proof.get("count", 0) >= 1,
        "route_classification": (route.get("status") == "passed" and route_result.get("calibrated") is True
                                  and route_result.get("threshold", 0) >= 0.8),
        "expert_scope": scope.get("passed") is True,
        "checkpoint_restore": checkpoint_probe(),
        "transactional_rollback": rollback_probe(),
    }
    recursive_round = candidate.get("recursive_round", 1)
    if recursive_round >= 2:
        gates["parent_continuity"] = parent_continuity_probe(candidate)
    protocol = "recursive-admission-v2" if recursive_round >= 2 else "recursive-admission-v1"
    payload = {
        "protocol": protocol,
        "status": "passed" if all(gates.values()) else "rejected",
        "candidate": str(Path(args.candidate)),
        "route": str(Path(args.route)),
        "original_protection": str(Path(args.original_protection)),
        "expert_id": route_result.get("expert_id"),
        "router_id": route_result.get("id"),
        "original_core_protection_sha256": protection["original_tensor_sha256"],
        **gates,
        "scope": "finite recursive publication contract; later rounds require immediate-parent continuity; no all-domain retention claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "expert_id": payload["expert_id"],
                      "router_id": payload["router_id"], "gates": gates}, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
