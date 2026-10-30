#!/usr/bin/env python3
"""Promote a fully verified isolated identity expert and router atomically."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
import re


MIN_THRESHOLD = 0.85


def validate_original_protection(record, identity):
    """Validate the immutable ordinary-core gate before any publication."""
    if (record.get("protocol") != "original-core-protection-v1"
            or record.get("passed") is not True
            or record.get("branch_enabled") is not False
            or record.get("teacher_calls") != 0
            or record.get("training_performed") is not False
            or record.get("identity") != identity
            or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("original_tensor_sha256", "")))):
        raise ValueError("原始核心保护基线缺失、未通过或与候选基座不绑定")
    generation = record.get("generation")
    cases = record.get("cases")
    if not isinstance(cases, list) or len(cases) != 8 or not isinstance(generation, list) or len(generation) != 8:
        raise ValueError("原始核心保护基线必须包含固定八条普通任务")
    if any(row.get("passed") is not True or row.get("complete") is not True
           or row.get("branch_enabled") is not False or row.get("reference_used") is not False
           for row in generation):
        raise ValueError("原始核心保护基线含未完成、分支开启或参考污染的样本")
    return {"passed": True, "case_count": 8, "identity": identity,
            "original_tensor_sha256": record["original_tensor_sha256"]}


def validate_recursive_proof(proof, *, expert_id, router_id, original_sha256, candidate_round=1):
    """Require one independently produced proof that all recursive gates agree."""
    expected_protocol = "recursive-admission-v2" if candidate_round >= 2 else "recursive-admission-v1"
    if (proof.get("protocol") != expected_protocol
            or proof.get("status") != "passed"
            or proof.get("expert_id") != expert_id
            or proof.get("router_id") != router_id
            or proof.get("original_core_protection_sha256") != original_sha256):
        raise ValueError("递归发布合同缺失、版本不一致或未通过")
    required = ("original_core_protection", "expert_closed_book", "route_classification",
                "expert_scope", "checkpoint_restore", "transactional_rollback")
    if candidate_round >= 2:
        required += ("parent_continuity",)
    if any(proof.get(key) is not True for key in required):
        raise ValueError("递归发布合同没有同时通过原始保护、专家、路由、断点和回滚门")
    return {key: True for key in required}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_parent_preservation_metadata(record, candidate_round):
    """Recursive candidates must carry a durable immediate-parent teacher."""
    if candidate_round < 2:
        return 0
    rows = record.get("preservation_rows", [])
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError("递归候选专家保护材料元数据无效")
    parent_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            provenance = json.loads(row.get("provenance", "{}"))
        except (TypeError, ValueError):
            provenance = {}
        if provenance.get("preservation_teacher") == "expert":
            parent_rows.append(row)
    if not parent_rows:
        raise ValueError("递归候选专家缺少持久化的立即父专家保护材料")
    return len(parent_rows)


def _copy_fsync(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"候选文件缺失或为符号链接：{source}")
    if target.is_symlink():
        raise ValueError(f"产品目标不能是符号链接：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    with target.open("rb") as stream:
        os.fsync(stream.fileno())


def transactional_replace(pairs):
    """Replace a complete expert/router bundle and restore every target on failure.

    Staging is outside the live names. Existing bytes are held only for this
    transaction; no persistent snapshot or pointer is created.
    """
    pairs = [(Path(source), Path(target)) for source, target in pairs]
    if not pairs:
        raise ValueError("晋升文件集合不能为空")
    stage = Path(tempfile.mkdtemp(prefix="identity-promotion-stage-",
                                  dir=str(pairs[0][1].parent)))
    originals = {}
    staged = []
    committed = []
    try:
        for index, (source, target) in enumerate(pairs):
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"候选文件缺失或为符号链接：{source}")
            if target.is_symlink():
                raise ValueError(f"产品目标不能是符号链接：{target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            stage_file = stage / f"{index:02d}-{target.name}"
            _copy_fsync(source, stage_file)
            staged.append((stage_file, target))
            originals[target] = target.read_bytes() if target.exists() else None
        for stage_file, target in staged:
            os.replace(stage_file, target)
            committed.append(target)
    except BaseException:
        for target in reversed(committed):
            old = originals[target]
            if old is None:
                target.unlink(missing_ok=True)
            else:
                rollback = stage / f"rollback-{len(committed)}-{target.name}"
                rollback.write_bytes(old)
                with rollback.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(rollback, target)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="candidate result.json")
    parser.add_argument("--route", required=True, help="route-proof.json")
    parser.add_argument("--candidate-storage", required=True)
    parser.add_argument("--production-storage", required=True)
    parser.add_argument("--original-protection", required=True,
                        help="independent original-core protection baseline")
    parser.add_argument("--recursive-proof", required=True,
                        help="combined recursive admission contract proof")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text())
    route = json.loads(Path(args.route).read_text())
    original = json.loads(Path(args.original_protection).read_text())
    recursive_proof = json.loads(Path(args.recursive_proof).read_text())
    candidate_identity = candidate.get("learning_report", {}).get("base_identity")
    protection = validate_original_protection(original, candidate_identity)
    route_result = route.get("route_calibration", {})
    if not (candidate.get("identity_passed") and candidate.get("ordinary_passed")
            and candidate.get("learning_report", {}).get("promoted")):
        raise ValueError("候选未同时通过身份、普通保护和专家学习晋升")
    if not (route.get("status") == "passed" and route_result.get("calibrated")
            and route_result.get("threshold", 0) >= MIN_THRESHOLD
            and route_result.get("expert_route_proof", {}).get("passed")
            and route_result.get("expert_scope", {}).get("passed")):
        raise ValueError("候选路由未通过独立闭卷证明或保守阈值")
    expert_id = candidate["learning_report"].get("active_expert")
    if expert_id != route_result.get("expert_id"):
        raise ValueError("候选专家和路由绑定不一致")
    recursive_gates = validate_recursive_proof(
        recursive_proof, expert_id=expert_id, router_id=route_result["id"],
        original_sha256=protection["original_tensor_sha256"],
        candidate_round=candidate.get("recursive_round", 1))

    candidate_module = Path(args.candidate_storage) / "modules" / "luban0-v1"
    production_module = Path(args.production_storage) / "modules" / "luban0-v1"
    expert_dir = candidate_module / "layer-23"
    production_expert_dir = production_module / "layer-23"
    expert_meta = expert_dir / f"{expert_id}.json"
    expert_weights = expert_dir / f"{expert_id}.safetensors"
    router_meta = candidate_module / f"{route_result['id']}.json"
    router_weights = candidate_module / f"{route_result['id']}.safetensors"
    active_expert = candidate_module / "active.json"
    active_router = candidate_module / "active-router.json"
    for path in (expert_meta, expert_weights, router_meta, router_weights, active_expert, active_router):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"候选发布文件缺失或为符号链接：{path}")
    if json.loads(expert_meta.read_text()).get("requires_gate") is not True:
        raise ValueError("候选专家没有强制路由门")
    expert_record = json.loads(expert_meta.read_text())
    validate_parent_preservation_metadata(expert_record, candidate.get("recursive_round", 1))
    router_record = json.loads(router_meta.read_text())
    if router_record.get("expert_id") != expert_id or router_record.get("threshold", 0) < MIN_THRESHOLD:
        raise ValueError("路由元数据没有绑定专家或阈值不足")
    active_router_record = json.loads(active_router.read_text())
    if active_router_record.get("id") != route_result["id"]:
        raise ValueError("活动路由指针与路由报告不一致")
    production_module.mkdir(parents=True, exist_ok=True)
    production_expert_dir.mkdir(parents=True, exist_ok=True)
    before = {}
    for path in (production_module / "active.json", production_module / "active-router.json"):
        before[path.name] = path.read_text() if path.exists() and not path.is_symlink() else None
    promotion_pairs = (
        (expert_weights, production_expert_dir / expert_weights.name),
        (expert_meta, production_expert_dir / expert_meta.name),
        (router_weights, production_module / router_weights.name),
        (router_meta, production_module / router_meta.name),
        (active_expert, production_module / "active.json"),
        (active_router, production_module / "active-router.json"),
    )
    transactional_replace(promotion_pairs)
    report = {
        "status": "promoted",
        "protocol": "identity-expert-promotion-v2-transactional",
        "candidate": str(Path(args.candidate)),
        "route": str(Path(args.route)),
        "recursive_proof": str(Path(args.recursive_proof)),
        "recursive_protocol": recursive_proof["protocol"],
        "expert_id": expert_id,
        "router_id": route_result["id"],
        "threshold": route_result["threshold"],
        "identity_passed": candidate["identity_passed"],
        "ordinary_passed": candidate["ordinary_passed"],
        "original_core_protection": protection,
        "recursive_admission_gates": recursive_gates,
        "expert_route_count": route_result["expert_route_proof"]["count"],
        "production_storage": args.production_storage,
        "production_mutations": [
            f"modules/luban0-v1/layer-23/{expert_id}.safetensors",
            f"modules/luban0-v1/layer-23/{expert_id}.json",
            f"modules/luban0-v1/{route_result['id']}.safetensors",
            f"modules/luban0-v1/{route_result['id']}.json",
            "modules/luban0-v1/active.json",
            "modules/luban0-v1/active-router.json",
        ],
        "previous_active_pointers": before,
        "expert_sha256": sha256(production_expert_dir / expert_weights.name),
        "router_sha256": sha256(production_module / router_weights.name),
        "transactional_bundle": True,
        "timestamp": time.time(),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "expert_id": expert_id,
                      "router_id": route_result["id"], "threshold": route_result["threshold"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
