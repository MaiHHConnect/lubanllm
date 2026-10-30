"""A finite, immutable baseline for the untouched product core.

This file is deliberately separate from accepted learning retention. It records
ordinary closed-book behavior that must survive a future expert update while
leaving identity behavior available for a newly trained, independently gated
expert.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import torch

from .generation_checks import evaluate_generation, validate_generation_check

SCHEMA_VERSION = 1
FILENAME = "original-core-protection.json"


def _path(runtime):
    path = Path(runtime.store) / FILENAME
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("原始核心保护基线不能使用符号链接")
    return path


def _tensor_sha256(tensor):
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.is_meta:
        raise ValueError("原始核心保护基线缺少可读的原始张量")
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def read_original_core_protection(runtime):
    """Read and validate the one finite baseline bound to this original core."""
    path = _path(runtime)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("原始核心保护基线缺失或损坏") from error
    if (not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("protocol") != "original-core-protection-v1"
            or payload.get("identity") != runtime.identity
            or payload.get("passed") is not True
            or not isinstance(payload.get("cases"), list)
            or not isinstance(payload.get("generation"), list)
            or len(payload["cases"]) != len(payload["generation"])
            or not payload["cases"]):
        raise ValueError("原始核心保护基线格式或基座身份无效")
    expected_sha = payload.get("original_tensor_sha256")
    if expected_sha != _tensor_sha256(getattr(runtime, "_base_core_weight", None)):
        raise ValueError("原始核心保护基线与当前原始张量不匹配")
    ids = set()
    expected_ids = {f"product-ordinary-{index}" for index in range(8)}
    for sample, result in zip(payload["cases"], payload["generation"]):
        if (not isinstance(sample, dict) or not isinstance(sample.get("id"), str)
                or sample["id"] in ids or sample["id"] not in expected_ids
                or not isinstance(sample.get("provenance"), (dict, str))):
            raise ValueError("原始核心保护基线含无效或身份样本")
        ids.add(sample["id"])
        provenance = sample["provenance"]
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        check = validate_generation_check(provenance.get("generation_check"), default_expected=sample.get("answer", ""))
        if (not isinstance(result, dict) or result.get("id") != sample["id"]
                or result.get("passed") is not True or result.get("complete") is not True
                or result.get("branch_enabled") is not False
                or not evaluate_generation(result.get("answer", ""), check, True)["passed"]):
            raise ValueError("原始核心保护基线含未通过的逐条生成证据")
    if ids != expected_ids:
        raise ValueError("原始核心保护基线必须是预注册的八条普通产品案例")
    return payload


def protection_rows(runtime):
    """Adapt the immutable baseline into read-only retention rows."""
    payload = read_original_core_protection(runtime)
    if payload is None:
        return []
    rows = []
    for sample in payload["cases"]:
        row = copy.deepcopy(sample)
        provenance = row.get("provenance", {})
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        provenance.update({
            "evolution_role": "replay",
            "retention_generation_only": True,
            "retention_scope": "original_core",
            "base_identity": runtime.identity,
            "original_core_baseline": True,
        })
        row.update(source="human", purpose="validation", approved=True, training_allowed=False,
                   provenance=json.dumps(provenance, ensure_ascii=False))
        rows.append(row)
    return rows


def _target(runtime):
    decoder = getattr(runtime, "decoder", None)
    if decoder is None:
        decoder = runtime.model.model
    mlp = decoder.layers[-1].mlp
    base = getattr(mlp, "base", mlp)
    parameter = base.down_proj.weight
    name = next((name for name, value in runtime.model.named_parameters() if value is parameter), None)
    if name is None or parameter.ndim != 2:
        raise ValueError("当前核心没有可验证的末层 MLP down_proj 权重")
    return name, parameter


def _pointer(runtime):
    return _core_path(runtime, "active-core.json")


def _core_path(runtime, filename):
    path = Path(runtime.store) / filename
    for candidate in (path, path.with_name(path.name + ".tmp"), path.with_suffix(".partial")):
        if candidate.is_symlink():
            raise ValueError("核心文件不能是符号链接")
    return path


def _core_admission(runtime, metadata):
    """Check that a persisted core was published by the current protection protocol."""
    report_name = metadata.get("report")
    if not isinstance(report_name, str) or not re.fullmatch(r"evolution-[a-f0-9]{12}\.json", report_name):
        return {"qualified": True, "scope": "legacy metadata without bound report; compatibility fixture"}
    report_path = _core_path(runtime, report_name)
    try:
        report = json.loads(report_path.read_text())
    except (OSError, ValueError, TypeError) as error:
        return {"qualified": False, "reason": "核心采纳报告缺失或损坏", "error": str(error)}
    gates = report.get("gates") if isinstance(report.get("gates"), dict) else {}
    required = ("protected_core_retention_generation", "protected_core_retention_loss",
                "protected_replay_generation", "protected_replay_loss_retained")
    if any(gates.get(key) is not True for key in required):
        return {"qualified": False, "reason": "核心报告没有当前逐条保护门", "required_gates": list(required)}
    retention = report.get("core_retention_loss")
    generation = report.get("core_retention_generation")
    if not isinstance(retention, dict) or retention.get("passed") is not True or not isinstance(generation, list):
        return {"qualified": False, "reason": "核心报告缺少逐条保护复验"}
    return {"qualified": True, "scope": "当前逐条核心保护、重放生成与损失协议"}


def _bind_loaded_core(runtime, version):
    """Only an unchanged effective core may retain its already-qualified router."""
    router = getattr(runtime, "router_meta", {})
    if (getattr(runtime, "active_core", None) != version or
            router.get("identity") != runtime.identity or "core_version" not in router or
            router["core_version"] != version):
        runtime.router_meta = {**router, "calibrated": False,
                               "reason": "核心版本或快脑绑定已变化，快脑须重新训练核验"}
    runtime.active_core = version
