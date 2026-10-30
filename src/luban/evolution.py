"""Reversible, branch-OFF updates to an existing decoder core parameter.

A checkpoint stores a cumulative replacement of one pretrained core tensor, not
an added adapter. Promotion requires actual held-out generation and old-task
retention; teacher-forcing loss alone cannot publish a core version.
"""
from __future__ import annotations

import gc
import copy
import hashlib
import json
import math
import os
import platform
import re
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import psutil
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file
from transformers import StoppingCriteria

from .settings import private_json
from .generation_checks import validate_generation_check, evaluate_generation, is_finite_generation_check
from .retention import (consolidation_groups, provenance as _provenance, messages as _messages,
                        normalized_context as _normalized_messages, read_core_retention,
                        protected_validation, save_core_retention, _validate_groups, _validate_record,
                        validate_retention_conflicts)
from .original_protection import (read_original_core_protection, _target, _pointer, _core_path,
                                  _core_admission, _bind_loaded_core)
from .coldpaging import full_backbone
from .training_checkpoint import TrainingPaused
from .retention import retention_loss_records
from .evolution_progress import (PARTIAL_POLICY, PhaseTimer, checkpoints as partial_checkpoints,
    unique_records, correct_ids, finish_partial, content_sha, progress as partial_progress, validate_product_records)

SEED = 42
LEARNING_RATE = 1e-4
TRAIN_CHECKPOINTS = (64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512)
RELATION_TRAIN_CHECKPOINTS = TRAIN_CHECKPOINTS + (544, 576, 608, 640)
FIELD_TRAIN_CHECKPOINTS = RELATION_TRAIN_CHECKPOINTS + (672, 704, 736, 768, 800, 832)
FIELD_EXPOSURE_CHECKPOINTS = FIELD_TRAIN_CHECKPOINTS + tuple(range(864, 1537, 32))
TRAIN_CHECKPOINT_POLICY = "v6-bounded-research-512"
REPLAY_GENERATION_POLICY = "v9-whole-string-qualified-replay-generation"
MINIMUM_REPLAY_GENERATION = 4
FIELD_FIRST_TOKEN_IDS = tuple(sorted(f"field-context-train-{attribute}-{form}"
    for attribute in ("name", "tuner", "position") for form in ("system", "single", "multi")))


def field_first_token_policy():
    return {"version": "field-first-gold-weight4-v1", "first_gold_weight": 4,
            "other_gold_eos_weight": 1, "normalization": "target_count+3",
            "training_ids": list(FIELD_FIRST_TOKEN_IDS),
            "scope": "only nine fixed NEW TRAIN rows; native quality losses and all other rows unchanged"}


def consolidation_samples(runtime, *, daily_only=False):
    """Merge immutable accepted protection with genuinely new qualified rows."""
    return consolidation_groups(runtime, daily_only=daily_only)


def _dataset_sha256(groups):
    return hashlib.sha256(json.dumps(groups, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _training_replay_generation(runtime, groups):
    """Only predeclared, original-correct TRAIN rows add decoder protection."""
    rows = []
    for key, samples in groups.items():
        for sample in samples:
            source = _provenance(sample)
            if source.get("training_generation_gate") is not True:
                continue
            observed = source.get("original_baseline_generation") or {}
            check = validate_generation_check(source.get("generation_check"), default_expected=sample["answer"])
            whole_string = (check["type"] == "exact" or
                            (check["type"] == "semantic" and check.get("validator") == "normalized_exact"))
            if (key != "replay_training" or not whole_string or check.get("expected") != sample["answer"]
                    or source.get("baseline_preservation_eligible") is not True
                    or source.get("original_baseline_verified_identity") != runtime.identity
                    or "original_baseline_verified_core" not in source
                    or source.get("original_baseline_verified_core") is not None
                    or observed.get("id") != sample["id"] or observed.get("passed") is not True
                    or observed.get("complete") is not True or observed.get("branch_enabled") is not False
                    or observed.get("reference_used", False) is not False
                    or observed.get("answer") != sample["answer"]):
                raise ValueError("重放 TRAIN 生成保护缺少固定原核心正确、完整且分支关闭的证据")
            rows.append(sample)
    if rows and len(rows) < MINIMUM_REPLAY_GENERATION:
        raise ValueError("重放 TRAIN 生成保护至少需要四条预先指定的原核心正确材料")
    return rows


def _read_version(runtime, version):
    if not isinstance(version, str) or not re.fullmatch(r"core-[a-f0-9]{12}", version):
        raise ValueError("核心版本名称无效")
    metadata = json.loads(_core_path(runtime, version + ".json").read_text())
    weights = _core_path(runtime, version + ".safetensors")
    if metadata.get("identity") != runtime.identity or metadata.get("version") != version:
        raise ValueError("核心版本与原始基座不兼容")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata.get("sha256"):
        raise ValueError("核心版本权重校验失败")
    if metadata.get("retention_sha256") is not None:
        read_core_retention(runtime, version)
        retention_path = Path(runtime.store) / (version + ".retention.json")
        if hashlib.sha256(retention_path.read_bytes()).hexdigest() != metadata["retention_sha256"]:
            raise ValueError("核心版本保护快照与元数据不匹配")
    name, parameter = _target(runtime)
    tensors = load_file(str(weights))
    if set(tensors) != {name} or tuple(tensors[name].shape) != tuple(parameter.shape):
        raise ValueError("核心版本真实模型图或权重尺寸不兼容")
    if tensors[name].dtype != parameter.dtype or not torch.isfinite(tensors[name]).all():
        raise ValueError("核心版本精度不兼容或权重含非有限值")
    return metadata, tensors[name]


def load_active_core(runtime):
    """Called after loading original weights and inserting the zero branch."""
    name, parameter = _target(runtime)
    metadata_only = bool(getattr(getattr(runtime, "cold_backbone", None), "metadata_only", False))
    # A native-only graph has no original HF tensor. Core files still pass the
    # same identity, tensor/shape/finite, retention and admission checks below.
    original = (None if metadata_only else
                runtime.cold_backbone.original_tensor(name, parameter) if parameter.is_meta
                else parameter.detach().cpu().clone())
    pointer = _pointer(runtime)
    if not pointer.exists():
        from .foundation import install_foundation
        foundation = install_foundation(runtime)
        if foundation is None:
            runtime._base_core_weight = original
            _bind_loaded_core(runtime, None)
            return None
        _read_version(runtime, foundation['version'])
        private_json(pointer, foundation)
    active = json.loads(pointer.read_text())
    if active.get("identity") != runtime.identity:
        raise ValueError("活动核心与原始基座不兼容")
    metadata, weights = _read_version(runtime, active.get("version"))
    if active.get("sha256") != metadata["sha256"]:
        raise ValueError("活动核心指针校验失败")
    runtime._base_core_weight = original
    admission = _core_admission(runtime, metadata)
    if admission.get("qualified") is not True:
        runtime.core_requires_review = {"version": metadata["version"], **admission}
        _bind_loaded_core(runtime, None)
        return None
    runtime.core_requires_review = None
    if not parameter.is_meta:
        with torch.no_grad():
            parameter.copy_(weights.to(parameter.device))
    _bind_loaded_core(runtime, metadata["version"])
    return metadata


def restore_core(runtime, version=None):
    """Restore a verified immutable core version or the originally loaded base."""
    native_only = getattr(getattr(runtime, "cold_backbone", None), "metadata_only", False)
    with runtime.lock, nullcontext() if native_only else full_backbone(runtime):
        if native_only:
            runtime.close_native()
        return _restore_core_loaded(runtime, version)


def _restore_core_loaded(runtime, version=None):
    with runtime.lock:
        runtime._ready()
        _, parameter = _target(runtime)
        pointer = _pointer(runtime)
        previous_pointer = pointer.read_bytes() if pointer.exists() else None
        previous_version = getattr(runtime, "active_core", None)
        native_only = getattr(getattr(runtime, "cold_backbone", None), "metadata_only", False)
        previous_weights = None if native_only else parameter.detach().cpu().clone()
        if version is None:
            weights = getattr(runtime, "_base_core_weight", None)
            if not native_only and (weights is None or tuple(weights.shape) != tuple(parameter.shape) or weights.dtype != parameter.dtype):
                raise ValueError("原核心恢复快照不可用，请重新加载模型")
            metadata = None
        else:
            metadata, weights = _read_version(runtime, version)
            if native_only and _core_admission(runtime, metadata).get("qualified") is not True:
                raise ValueError("核心版本未通过当前保护检查，不能恢复")
        try:
            if not native_only:
                with torch.no_grad():
                    parameter.copy_(weights.to(parameter.device))
            if metadata:
                private_json(pointer, {"version": version, "identity": runtime.identity, "sha256": metadata["sha256"]})
            else:
                pointer.unlink(missing_ok=True)
            runtime.active_core = version
            runtime.router_meta = {**getattr(runtime, "router_meta", {}), "calibrated": False,
                                   "reason": "核心版本已变化，快脑须重新训练核验"}
            return {"version": version, "previous_version": previous_version, "restored": True}
        except Exception:
            if not native_only:
                with torch.no_grad():
                    parameter.copy_(previous_weights.to(parameter.device))
            runtime.active_core = previous_version
            current = pointer.read_bytes() if pointer.exists() else None
            if current != previous_pointer:
                if previous_pointer is None:
                    pointer.unlink(missing_ok=True)
                else:
                    private_json(pointer, json.loads(previous_pointer))
            raise


def core_status(runtime, *, partial_policy=None, daily_only=False):
    result = {"version": getattr(runtime, "active_core", None), "available": False, "ready": False,
              "counts": {}, "blocked_reason": "模型尚未就绪"}
    if getattr(runtime, "model", None) is None:
        return result
    try:
        if getattr(getattr(runtime, "cold_backbone", None), "metadata_only", False):
            result["available"] = True
            result["blocked_reason"] = "当前MLX-Q4仅保留量化模型，旧核心进化需要原始权重"
            return result
        _target(runtime)
        result["available"] = True
        from .native_mlx import require_training_source
        require_training_source(runtime)
        samples = consolidation_samples(runtime, daily_only=daily_only)
        result["counts"] = {key: len(rows) for key, rows in samples.items()}
        active = getattr(runtime, "active_core", None)
        original_protection = None
        if active is None and getattr(getattr(runtime, "model", None), "config", None) is not None:
            original_protection = read_original_core_protection(runtime)
        if (active is None and not getattr(runtime, "research", False)
                and getattr(getattr(runtime, "model", None), "config", None) is not None
                and getattr(runtime.model.config, "model_type", None) == "qwen3_5"):
            if original_protection is None:
                result["blocked_reason"] = "当前原始核心缺少已绑定的独立保护基线，暂不启动核心进化"
                return result
        already_consolidated = False
        if isinstance(active, str) and re.fullmatch(r"core-[a-f0-9]{12}", active):
            metadata = json.loads(_core_path(runtime, active + ".json").read_text())
            already_consolidated = (metadata.get("identity") == runtime.identity
                and metadata.get("version") == active and metadata.get("dataset_sha256") == _dataset_sha256(samples))
        if active and not samples["new_training"] and not samples["new_validation"]:
            result["blocked_reason"] = "当前材料已巩固到活动核心，请添加真正新的已确认知识及独立检验"
        elif (not all(samples.values()) if partial_policy != PARTIAL_POLICY else
              not all(samples[key] for key in ("new_training", "replay_training", "replay_validation"))):
            result["blocked_reason"] = "需要已确认的新知识训练/独立检验，以及旧能力重放训练/保护检验材料"
        elif already_consolidated:
            result["blocked_reason"] = "当前材料已巩固到活动核心，请添加新的已确认知识"
        elif ((getattr(runtime, "core_graph_verification", None) or {}).get("passed") is not True
              or (getattr(runtime, "core_graph_verification", None) or {}).get("identity") != runtime.identity):
            result["blocked_reason"] = "当前真实核心模型图尚未通过等价验证"
        else:
            result.update(ready=True, blocked_reason="")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result["blocked_reason"] = str(error)
    return result


class _CancelGeneration(StoppingCriteria):
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, input_ids, scores, **kwargs):
        session = getattr(self.runtime, "training_session", None)
        return bool((self.runtime.job and self.runtime.job.get("cancel_requested"))
                    or (session and session.pause_requested))


def _cancel(runtime):
    session = getattr(runtime, "training_session", None)
    if session and session.pause_requested:
        raise TrainingPaused("核心训练已在有效检查点暂停")
    if runtime.job and runtime.job.get("cancel_requested"):
        raise RuntimeError("进化已取消，核心版本保持不变")


def _session_material(runtime, session, arguments, regressions):
    """The durable manifest, rather than subsequently edited Memory, owns this run."""
    if (session.body.get("kind") != "evolution" or session.binding.get("identity") != runtime.identity
            or session.binding.get("core_version") != getattr(runtime, "active_core", None)
            or any(session.body["arguments"].get(key) != value for key, value in arguments.items())):
        raise ValueError("核心训练会话与当前版本或训练参数不兼容")
    records = copy.deepcopy(session.body["records"])
    frozen_sha = hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if frozen_sha != session.binding.get("records_sha256"):
        raise ValueError("冻结训练或保护材料校验失败")
    if arguments.get("partial_policy") == PARTIAL_POLICY and arguments["steps"] == 450:
        validate_product_records(records)
    groups = records["groups"]
    _validate_groups(groups)
    if (not all(groups.values()) if arguments.get("partial_policy") != PARTIAL_POLICY else
            not all(groups[key] for key in ("new_training", "replay_training", "replay_validation"))):
        raise ValueError("冻结核心材料需要完整的新知识与旧能力训练/检验")
    proof = getattr(runtime, "core_graph_verification", None) or {}
    if proof.get("passed") is not True or proof.get("identity") != runtime.identity:
        raise ValueError("当前真实核心模型图尚未通过等价验证")
    protection = records["protection"]
    if protection != protected_validation(runtime, include_learning=False):
        raise ValueError("冻结保护材料与当前已提交核心不一致")
    previous = read_core_retention(runtime)
    if previous:
        old_ids = {row["id"] for rows in previous["groups"].values() for row in rows}
        replay_ids = {row["id"] for key, rows in groups.items() if key.startswith("replay_") for row in rows}
        if not old_ids <= replay_ids:
            raise ValueError("冻结核心材料遗漏已采纳知识的重放保护")
        validate_retention_conflicts(previous, [row for rows in groups.values() for row in rows])
    frozen_regressions = records["retention_regressions"]
    if regressions and list(regressions) != frozen_regressions:
        raise ValueError("额外保护检验与冻结训练会话不一致")
    for row in frozen_regressions:
        _validate_record(row, generation_only=True)
    return groups, protection, frozen_regressions


def _resume_core(session, saved, report, parameter, previous, groups, steps, master, optimizer):
    """Check step coverage before restoring a master tensor and its exact Adam state."""
    step = saved.get("step")
    expected = {row["id"]: ((step + 1) // 2) // len(groups["new_training"])
                + int(index < ((step + 1) // 2) % len(groups["new_training"]))
                for index, row in enumerate(groups["new_training"])} if isinstance(step, int) else {}
    if saved["report"].get("goal_sequence", {}).get("protocol") == "evolution-goal-new-data-fixed450-v1":
        from .goal_fresh import phase_exposures
        expected = phase_exposures(saved["report"]["goal_sequence_sources"], groups, step)
    original_sha = hashlib.sha256(previous.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if (saved.get("format") != 1 or saved.get("id") != session.id or isinstance(step, bool)
            or not isinstance(step, int) or not 0 <= step <= steps
            or saved.get("dataset_sha256") != report["dataset_sha256"]
            or saved.get("original_sha256") != original_sha
            or saved.get("parameter") != report["parameter"]
            or saved.get("new_exposures") != expected
            or saved["report"].get("id") != report["id"]
            or saved["report"].get("version") != report["version"]
            or saved["report"].get("steps") != report.get("steps")
            or saved["report"].get("split_ids") != report["split_ids"]
            or saved["report"].get("field_loss_policy") != report.get("field_loss_policy")
            or saved["master"].shape != parameter.shape or saved["master"].dtype != torch.float32):
        raise ValueError("核心训练检查点与权重、材料或曝光次数不兼容")
    old_engine, new_engine = saved["report"].get("training_engine", {}), report.get("training_engine", {})
    if any(old_engine.get(key) != new_engine.get(key) for key in ("engine", "binary_sha256")):
        raise ValueError("恢复检查点的训练引擎或二进制不同")
    history = saved["report"]["training_selection"]
    if saved["report"].get("goal_sequence", {}).get("protocol") in {"evolution-goal-prospective-fixed450-v1", "evolution-goal-prospective-fixed450-v2", "evolution-goal-prospective-fixed450-v3", "evolution-goal-prospective-fixed450-v4", "evolution-goal-new-data-fixed450-v1"}:
        from .goal_fresh import validate_resume_history
        validate_resume_history(saved, report, groups)
        history = None
    if saved["report"].get("goal_sequence", {}).get("protocol") in {"evolution-goal-fixed450-rounds-v2", "evolution-goal-fixed450-rounds-v3"}:
        from .goal_sequence import validate_resume_history
        validate_resume_history(saved, report, groups)
        history = None
    if history is not None:
        declared = report["training_selection"]
        if any(history.get(key) != declared[key] for key in
               ("checkpoints", "policy_version", "required_consecutive_passes", "required_new_exposures",
                "replay_generation_ids", "minimum_replay_generation")):
            raise ValueError("核心训练检查点选择政策不兼容")
        completed = [value for value in declared["checkpoints"] if value <= step]
        results = history["results"]
        streak = 0
        previous_correct = set()
        before = {p["id"] for p in saved["report"].get("partial_training_baseline", []) if p["passed"]}
        for row in results:
            minimum_exposures = ((row["step"] + 1) // 2) // len(groups["new_training"])
            replay_budget = saved["report"]["baseline_training_replay_loss"] * 1.05 + .01
            accepted = (all(value["passed"] for value in row["new_generation"])
                and all(value["passed"] for value in row["replay_training_generation"])
                and row["replay_loss"] <= replay_budget and minimum_exposures >= 2)
            if report.get("partial_policy") == PARTIAL_POLICY:
                now = correct_ids(row["new_generation"], groups["new_training"])
                accepted = (bool((now & previous_correct) - before) and minimum_exposures >= 2
                    and all(value["passed"] for value in row["replay_training_generation"])
                    and row["replay_loss"] <= replay_budget)
                previous_correct = now
            if (row["minimum_new_exposures"] != minimum_exposures or row["accepted"] != accepted
                    or [value["id"] for value in row["new_generation"]] != list(expected)
                    or [value["id"] for value in row["replay_training_generation"]] != declared["replay_generation_ids"]
                    or any(value["passed"] and (not value["complete"] or value["branch_enabled"])
                           for value in row["replay_training_generation"])):
                raise ValueError("核心训练检查点生成或曝光门槛历史不一致")
            streak = streak + 1 if row["accepted"] else 0
            if row["consecutive_passes"] != streak:
                raise ValueError("核心训练检查点连续通过历史不一致")
        selected_invalid = ((history["selected_step"] is not None and (streak < 2 or step != steps or history["selected_step"] != step))
            or (history["selected_step"] is None and step == steps and streak >= 2)) if report.get("partial_policy") == PARTIAL_POLICY else (
            (history["selected_step"] is not None and (streak != 2 or history["selected_step"] != step))
            or (history["selected_step"] is None and streak >= 2))
        if ([row["step"] for row in results] != completed or saved["consecutive_passes"] != streak or selected_invalid):
            raise ValueError("核心训练检查点缺少完整检查历史")
    with torch.no_grad():
        master.copy_(saved["master"].to(master.device))
        parameter.copy_(master)
    optimizer.load_state_dict(saved["optimizer"])
    return copy.deepcopy(saved["report"])


def _save_core_step(session, runtime, report, parameter, previous, master, optimizer, basis,
                    new_exposures, kl_total, kl_maximum, consecutive_passes):
    step = runtime.job["step"]
    payload = {"format": 1, "id": session.id, "parameter": report["parameter"],
        "dataset_sha256": report["dataset_sha256"],
        "original_sha256": hashlib.sha256(previous.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        "step": step, "master": master, "optimizer": optimizer.state_dict(), "basis": basis,
        "new_exposures": new_exposures, "kl_total": kl_total, "kl_maximum": kl_maximum,
        "consecutive_passes": consecutive_passes, "report": report}
    session.save(payload, {"step": step, "phase": "training", "steps": report["steps"]})
    _cancel(runtime)


def _memory(runtime):
    result = {"rss_bytes": psutil.Process().memory_info().rss}
    if str(runtime.device) == "mps":
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    return result


def _loss(runtime, samples):
    values = []
    with torch.no_grad():
        for sample in samples:
            _cancel(runtime)
            ids, labels = runtime._sample_tokens(sample)
            value = float(runtime.model(ids, labels=labels, use_cache=False).loss.detach().float().cpu())
            if not math.isfinite(value):
                raise RuntimeError("核心检验损失异常")
            values.append(value)
    return sum(values) / len(values)


def _loss_by_sample(runtime, samples):
    """Return finite per-record CE for the final protection decision."""
    return [{'id': sample['id'], 'loss': float(_loss(runtime, [sample]))} for sample in samples]


def _retention_loss_guard(baseline, candidate):
    """Compare each retained record with one fixed FP32 measurement bound."""
    if not isinstance(baseline, list) or not isinstance(candidate, list):
        raise ValueError('核心保护损失记录必须是列表')
    old = {row.get('id'): row.get('loss') for row in baseline}
    current = {row.get('id'): row.get('loss') for row in candidate}
    if not old and not current:
        return {'passed': True, 'rows': [],
                'scope': '没有已保护材料；没有能力保护证据'}
    if (set(old) != set(current)
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                   for value in (*old.values(), *current.values()))):
        raise ValueError('核心保护损失记录不完整或含非有限值')
    rows = []
    passed = True
    for identifier, bound in old.items():
        value = current[identifier]
        tolerance = 64 * torch.finfo(torch.float32).eps * max(1., abs(float(bound)))
        excess = value - bound - tolerance
        okay = excess <= 0
        passed = passed and okay
        rows.append({'id': identifier, 'baseline_loss': bound, 'candidate_loss': value,
                     'tolerance': tolerance, 'excess_over_tolerance': excess,
                     'nonincreasing': okay})
    return {'passed': passed, 'rows': rows,
            'scope': '每条已保护材料；固定FP32测量上界，不是平均loss或能力全域证明'}


def _generation(runtime, samples):
    with full_backbone(runtime):
        return _generation_loaded(runtime, samples)


def _generation_loaded(runtime, samples):
    results = []
    for sample in samples:
        _cancel(runtime)
        check = validate_generation_check(_provenance(sample).get("generation_check"), default_expected=sample["answer"])
        maximum = check.get("max_new_tokens", 192)
        tokens = runtime._tokens(_messages(sample), "off", output_reserve=maximum)
        with torch.no_grad():
            out = runtime.model.generate(tokens, attention_mask=torch.ones_like(tokens),
                max_new_tokens=maximum, do_sample=False, use_cache=True,
                pad_token_id=runtime.tokenizer.eos_token_id, stopping_criteria=[_CancelGeneration(runtime)])
        _cancel(runtime)
        answer_ids = out[0, tokens.shape[1]:]
        decoded = runtime.tokenizer.decode(answer_ids, skip_special_tokens=True)
        answer = decoded.strip()
        eos = getattr(runtime.model.generation_config, "eos_token_id", runtime.tokenizer.eos_token_id)
        eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
        complete = bool(answer_ids.numel() and int(answer_ids[-1]) in eos_ids)
        results.append({"id": sample["id"], "answer": answer, **evaluate_generation(answer, check, complete),
                        "raw_completion_token_ids": [int(value) for value in answer_ids.tolist()],
                        "decoded_completion_unstripped": decoded,
                        "completion_tokens": int(answer_ids.numel()), "max_new_tokens": maximum,
                        "branch_enabled": runtime.branch.expert is not None, "reference_used": False, "thinking": "off", "do_sample": False})
    return results


def _answer_logits(logits, labels):
    """Float32 full-vocabulary logits at causal answer/EOS positions."""
    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("核心保留训练需要完整序列 logits 与对应标签")
    selected = logits[:, :-1][labels[:, 1:] != -100].float()
    if not selected.shape[0] or not torch.isfinite(selected).all():
        raise RuntimeError("核心保留训练缺少回答词元或 logits 异常")
    return selected


def _answer_log_probs(logits, labels):
    return F.log_softmax(_answer_logits(logits, labels), dim=-1)


def _field_weighted_loss(logits, labels):
    """Weight4 at first causal gold, weight1 through EOS, normalized by n+3.

    This changes the objective. Dataset/stage authorization is checked before
    candidate allocation; the backbone graph and decoding remain native.
    """
    if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape[1] < 2:
        raise ValueError("字段损失需要一个完整因果序列")
    targets = labels[:, 1:]
    mask = targets != -100
    positions = mask[0].nonzero(as_tuple=False).flatten()
    if (positions.numel() < 2 or positions[0] < 1
            or not mask[0, positions[0]:].all() or positions[-1] != targets.shape[1] - 1
            or not labels[0, :positions[0] + 1].eq(-100).all()):
        raise ValueError("字段损失首gold/前缀mask/连续gold+EOS位置无效")
    ce = F.cross_entropy(_answer_logits(logits, labels), targets[mask], reduction="none")
    return (ce.sum() + 3 * ce[0]) / (ce.numel() + 3)


def _teacher_replay(runtime, parameter, previous, candidate, ids, labels, *, return_logits=False):
    """Use the same core's pre-run tensor; restore before student autograd."""
    with torch.no_grad():
        try:
            parameter.copy_(previous.to(parameter.device))
            _cancel(runtime)
            logits = runtime.model(ids, use_cache=False).logits
            reference = (_answer_logits(logits, labels) if return_logits else _answer_log_probs(logits, labels)).detach()
        finally:
            parameter.copy_(candidate)
    _cancel(runtime)
    return reference


def _capture_replay_subspace(runtime, parameter, samples, *, protection_capture=None):
    from .evolution_progress import capture_replay_subspace
    return capture_replay_subspace(runtime, parameter, samples, cancel=_cancel, protection_capture=protection_capture)


def _project_core_delta(master, reference, basis):
    """Constrain the whole post-Adam update, including optimizer momentum."""
    with torch.no_grad():
        delta = master - reference
        master.copy_(reference + delta - (delta @ basis) @ basis.T)
        if not torch.isfinite(master).all():
            raise RuntimeError("核心累计投影产生非有限值")


def _constraint_residual(parameter, master, reference, basis):
    with torch.no_grad():
        quantized_delta = parameter.detach().float() - reference
        residual = quantized_delta @ basis
        master_residual = (master.detach() - reference) @ basis
        if not torch.isfinite(residual).all() or not torch.isfinite(master_residual).all():
            raise RuntimeError("核心量化约束残差异常")
        return {"quantized_frobenius": float(residual.norm().cpu()),
                "quantized_max_abs": float(residual.abs().max().cpu()),
                "quantized_relative_to_delta": float((residual.norm() / quantized_delta.norm().clamp_min(1e-12)).cpu()),
                "master_frobenius": float(master_residual.norm().cpu()), "weight_dtype": str(parameter.dtype)}


def _replay_logit_drift(runtime, parameter, previous, candidate, samples):
    values, positions, absolute_sum, squared_sum, maximum = 0, 0, 0.0, 0.0, 0.0
    records = []
    with torch.no_grad():
        for sample in samples:
            _cancel(runtime)
            ids, labels = runtime._sample_tokens(sample)
            teacher = _teacher_replay(runtime, parameter, previous, candidate, ids, labels, return_logits=True)
            student = _answer_logits(runtime.model(ids, use_cache=False).logits, labels)
            difference = student - teacher
            if not torch.isfinite(difference).all():
                raise RuntimeError("核心重放 logits 漂移含非有限值")
            count, absolute, squared = difference.numel(), float(difference.abs().sum().cpu()), float(difference.square().sum().cpu())
            local_max = float(difference.abs().max().cpu())
            if not all(math.isfinite(value) for value in (absolute, squared, local_max)):
                raise RuntimeError("核心重放 logits 漂移统计含非有限值")
            records.append({"id": sample["id"], "positions": difference.shape[0], "max_abs": local_max,
                            "mean_abs": absolute / count})
            values += count
            positions += difference.shape[0]
            absolute_sum += absolute
            squared_sum += squared
            maximum = max(maximum, local_max)
            del teacher, student, difference
    return {"finite": True, "values": values, "positions": positions, "max_abs": maximum,
            "mean_abs": absolute_sum / values, "rms": math.sqrt(squared_sum / values), "samples": records,
            "scope": "full-vocabulary raw logits at replay TRAIN answer/EOS predictor positions; both cores branch OFF"}


def _baseline_validation(runtime, parameter, previous, candidate, groups, report):
    """Run the pre-update comparison after TRAIN selection, then restore it."""
    with torch.no_grad():
        try:
            parameter.copy_(previous.to(parameter.device))
            report["baseline_new_loss"] = _loss(runtime, groups["new_validation"])
            report["baseline_replay_loss"] = _loss(runtime, groups["replay_validation"])
            report["baseline_new_generation"] = _generation(runtime, groups["new_validation"])
            report["baseline_replay_generation"] = _generation(runtime, groups["replay_validation"])
        finally:
            parameter.copy_(candidate)


def consolidate(runtime, steps=None, *, learning_rate=None, distillation_weight=None,
                preserve_replay_subspace=None, retention_regressions=(), field_first_token_weight=1, partial_policy=None, goal_sequence=None):
    from .native_core_training import defaults
    from .native_mlx import require_training_source
    require_training_source(runtime)
    steps, learning_rate, distillation_weight, preserve_replay_subspace = defaults(
        runtime, steps, learning_rate, distillation_weight, preserve_replay_subspace)
    with runtime.lock, full_backbone(runtime):
        return _consolidate_loaded(runtime, steps, learning_rate=learning_rate,
            distillation_weight=distillation_weight, preserve_replay_subspace=preserve_replay_subspace,
            retention_regressions=retention_regressions, field_first_token_weight=field_first_token_weight, partial_policy=partial_policy, goal_sequence=goal_sequence)


def _consolidate_loaded(runtime, steps=64, *, learning_rate=LEARNING_RATE, distillation_weight=0,
                        preserve_replay_subspace=False, retention_regressions=(), field_first_token_weight=1, partial_policy=None, goal_sequence=None):
    """Update the real core in place; restore it unless all final gates pass."""
    from .native_mlx import require_training_source
    require_training_source(runtime)
    if partial_policy not in {None, PARTIAL_POLICY}:
        raise ValueError("未知部分进化政策")
    partial = partial_policy == PARTIAL_POLICY
    if isinstance(steps, bool) or not isinstance(steps, int) or not (1 <= steps <= 4096 or steps in {5400, 6750, 9450} and goal_sequence is not None):
        raise ValueError("进化步数须在 1–4096 之间")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or not 1e-8 <= learning_rate <= 0.01:
        raise ValueError("核心学习率须在 1e-8–0.01 之间")
    if isinstance(distillation_weight, bool) or not isinstance(distillation_weight, (int, float)) or not math.isfinite(distillation_weight) or not 0 <= distillation_weight <= 100:
        raise ValueError("核心保留训练权重须在 0–100 之间")
    if isinstance(field_first_token_weight, bool) or not isinstance(field_first_token_weight, (int, float)) or field_first_token_weight not in {1, 4}:
        raise ValueError("字段首gold权重只支持原行为1或固定研究4")
    model_config = getattr(getattr(runtime, "model", None), "config", None)
    if (getattr(model_config, "model_type", None) == "qwen3_5"
            and getattr(runtime, "research", False) is not True
            and (not preserve_replay_subspace or distillation_weight < 5)):
        raise ValueError("生产 Qwen3.5 核心进化必须启用 replay 子空间保护和至少5.0蒸馏权重")
    from . import goal_sequence as goal
    if goal_sequence is not None and goal.read_json(goal_sequence).get("protocol") in {"evolution-goal-prospective-fixed450-v1", "evolution-goal-prospective-fixed450-v2", "evolution-goal-prospective-fixed450-v3", "evolution-goal-prospective-fixed450-v4", "evolution-goal-new-data-fixed450-v1"}:
        from . import goal_fresh as goal
    episode = goal.check_call(runtime, goal_sequence, steps, learning_rate, distillation_weight, preserve_replay_subspace, field_first_token_weight, partial_policy)
    if not isinstance(preserve_replay_subspace, bool) or (preserve_replay_subspace and not (episode or 64 <= steps <= 512 or steps in {640, 832, 1536} or partial and steps == 704)):
        raise ValueError("核心子空间保护须为布尔值，启用时训练上限须在 64–512 步之间；固定研究另限 640/832/1536")
    with runtime.lock:
        runtime._ready()
        session = getattr(runtime, "training_session", None)
        body_weight = session.body.get("arguments", {}).get("field_first_token_weight", 1) if session else 1
        if (isinstance(body_weight, bool) or not isinstance(body_weight, (int, float))
                or body_weight not in {1, 4} or (session and body_weight != field_first_token_weight)):
            raise ValueError("调用字段权重与冻结会话不一致")
        product_partial = (steps == 450 and session is not None and session.body.get("scope") == "product"
                           and getattr(runtime, "research", False) is not True)
        if partial and (not product_partial and (steps not in {352, 704} or session is None
                or session.body.get("scope") != "research" or getattr(runtime, "research", False) is not True)
                or field_first_token_weight != 1
                or not preserve_replay_subspace or learning_rate != 1e-5 or distillation_weight != 5):
            raise ValueError("部分进化需要研究352/704或产品450、已绑定会话、原mean1/LR/KL和子空间保护")
        if (session and session.body.get("arguments", {}).get("partial_policy") != partial_policy):
            raise ValueError("部分进化调用与冻结会话政策不一致")
        if steps == 1536 and (session is None or session.body.get("scope") != "research"
                or getattr(runtime, "research", False) is not True or getattr(runtime, "active_core", None) is not None
                or field_first_token_weight != 1 or not preserve_replay_subspace
                or learning_rate != 1e-5 or distillation_weight != 5):
            raise ValueError("1536 步仅允许固定57题 mean1 原始核心 research session")
        arguments = {"steps": steps, "learning_rate": learning_rate, "distillation_weight": distillation_weight,
                     "preserve_replay_subspace": preserve_replay_subspace}
        if steps == 1536 and (session.body.get("arguments") != arguments
                or session.binding.get("arguments") != arguments or session.binding.get("scope") != "research"):
            raise ValueError("1536 调用、冻结正文及research binding参数必须完全一致")
        if partial:
            arguments["partial_policy"] = partial_policy
            if session.body["arguments"] != arguments or session.binding["arguments"] != arguments:
                raise ValueError("部分进化正文与binding参数必须完全一致")
        if field_first_token_weight == 4:
            if (steps != 832 or session is None or session.body.get("scope") != "research"
                    or getattr(runtime, "research", False) is not True
                    or getattr(runtime, "active_core", None) is not None or not preserve_replay_subspace
                    or learning_rate != 1e-5 or distillation_weight != 5):
                raise ValueError("字段weight4仅允许固定57/832原始核心research session")
            arguments["field_first_token_weight"] = 4
        if episode:
            arguments = goal.arguments(episode)
        if session:
            groups, core_protection, retention_regressions = _session_material(runtime, session,
                arguments, retention_regressions)
        else:
            readiness = core_status(runtime, partial_policy=partial_policy)
            if not readiness["ready"]:
                raise ValueError(readiness["blocked_reason"])
            groups = consolidation_samples(runtime)
            core_protection = protected_validation(runtime, include_learning=False)
        checkpoints = goal.checkpoints(episode) if episode else TRAIN_CHECKPOINTS
        if steps in {832, 1536} or (preserve_replay_subspace and steps == 640):
            from .core_publication import validate_relation_teaching, validate_field_teaching
            if (not preserve_replay_subspace or session is None or session.body.get("scope") != "research" or getattr(runtime, "active_core", None) is not None
                    or learning_rate != 1e-5 or distillation_weight != 5):
                raise ValueError("640/832 步仅允许从原始核心起步的固定关系教学研究 session")
            validator = validate_field_teaching if steps in {832, 1536} else validate_relation_teaching
            validator(groups, retention_regressions, qualified=True)
            checkpoints = (FIELD_EXPOSURE_CHECKPOINTS if steps == 1536 else
                           FIELD_TRAIN_CHECKPOINTS if steps == 832 else RELATION_TRAIN_CHECKPOINTS)
        if partial:
            checkpoints = partial_checkpoints(steps)
        prior_retention = read_core_retention(runtime)
        replay_generation_rows = _training_replay_generation(runtime, groups)
        if replay_generation_rows and not preserve_replay_subspace:
            raise ValueError("固定重放 TRAIN 生成保护需要启用完整训练检查点选择政策")
        # Validate every tensorization before allocating/training a candidate.
        for key, rows in groups.items():
            if preserve_replay_subspace and key.endswith("validation"):
                continue
            for sample in rows:
                runtime._sample_tokens(sample)
        from .native_core_training import select as select_engine, steps as native_steps
        binary, training_engine = select_engine(runtime, preserve_replay_subspace=preserve_replay_subspace,
            distillation_weight=distillation_weight, field_first_token_weight=field_first_token_weight, partial_policy=partial_policy)
        name, parameter = _target(runtime)
        previous = parameter.detach().cpu().clone()
        original_expert = runtime.branch.expert
        requires_grad = [(value, value.requires_grad) for value in runtime.model.parameters()]
        previous_version = getattr(runtime, "active_core", None)
        old_pointer = _pointer(runtime).read_bytes() if _pointer(runtime).exists() else None
        run_id = session.id if session else uuid.uuid4().hex[:12]
        version = "core-" + run_id
        started = time.time()
        report = {"id": run_id, "version": version, "identity": runtime.identity, "previous_version": previous_version,
                  "promoted": False, "seed": SEED, "steps": steps, "started": started, "training_engine": training_engine,
                  "dataset_sha256": _dataset_sha256(groups),
                  "replay_subspace": {"enabled": preserve_replay_subspace},
                  "training_selection": {"checkpoints": [step for step in checkpoints if step <= steps],
                      "policy_version": REPLAY_GENERATION_POLICY if replay_generation_rows else TRAIN_CHECKPOINT_POLICY,
                      "replay_generation_ids": [row["id"] for row in replay_generation_rows],
                      "minimum_replay_generation": MINIMUM_REPLAY_GENERATION if replay_generation_rows else 0,
                      "required_consecutive_passes": 2,
                      "required_new_exposures": 2,
                      "results": [], "selected_step": None,
                      "scope": "at least two exposures per new TRAIN sample; first two consecutive NEW TRAIN generation, predeclared original-correct REPLAY TRAIN generation and replay-loss passes; no heldout feedback"} if preserve_replay_subspace else None,
                  "parameter": name, "trained_parameters": parameter.numel(), "added_parameters": 0,
                  "dtype": str(parameter.dtype), "optimizer_dtype": "torch.float32", "learning_rate": learning_rate,
                  "distillation": {"weight": distillation_weight, "teacher_version": previous_version,
                      "teacher_identity": runtime.identity, "teacher_scope": "frozen pre-run core; branch OFF; no external teacher",
                      "teacher_tensor_sha256": hashlib.sha256(previous.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() if distillation_weight else None,
                      "position_scope": "causal answer and EOS positions, prompt labels masked; full vocabulary; float32",
                      "normalization": "KL(teacher || student), mean per predicted position, temperature 1",
                      "statistic_scope": "mean/max of per-step normalized KL; target_tokens counts repeated exposures",
                      "steps": 0, "target_tokens": 0, "kl_mean": None, "kl_max": None},
                  "promotion_thresholds": {"new_loss_ratio_below": 0.995, "replay_loss_ratio_max": 1.05,
                      "replay_loss_additive_tolerance": 0.01, "all_heldout_generation_required": True,
                      "new_generation_gain_required": True, "scope": "finite curated cases, not broad capability guarantee"},
                  "device": str(runtime.device), "hardware": platform.platform(), "torch_version": str(torch.__version__),
                  "branch_enabled": False, "graph_verification": runtime.core_graph_verification,
                  "split_ids": {key: [row["id"] for row in rows] for key, rows in groups.items()},
                  "sample_sha256": {row["id"]: hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                                    for rows in groups.values() for row in rows},
                  "split_scope": "distinct normalized full contexts; declared split_family disjoint; semantic independence requires curated review",
                  "evaluation_scope": "curated user knowledge and protected replay; not a scientific generalization benchmark",
                  "memory_measurement": "process RSS and MPS allocator snapshots; sampled maxima, not physical peak",
                  "memory_samples": []}
        if episode:
            report["goal_sequence"] = goal.report_metadata(episode)
            report["goal_sequence_sources"] = goal.source_metadata(episode)
            report["training_selection"].update(policy_version=episode["policy_version"],
                required_consecutive_passes=0, required_same_weight_passes=2)
        runtime.state = "evolving"
        pending_cancel = bool(runtime.job and runtime.job.get("cancel_requested"))
        runtime.job = {"id": run_id, "kind": "evolution", "state": "running", "step": 0, "steps": steps,
                       "phase": "preparing", "cancel_requested": pending_cancel}
        timer = PhaseTimer(runtime, report) if partial else None
        if partial:
            report["partial_policy"] = PARTIAL_POLICY
            report["promotion_thresholds"] = {"scope": PARTIAL_POLICY,
                "two_frozen_complete_finite_observations": True, "positive_baseline_to_candidate_growth": True,
                "all_original_correct_and_prior_mastery_retained": True,
                "replay_loss_ratio_at_most": 1.05, "replay_loss_additive_tolerance": .01,
                "all_new_generation_required": False, "new_loss_is_advisory": True}
            report["training_selection"].update(policy_version=PARTIAL_POLICY,
                scope="fixed-cap 50-step growth curve; frozen candidate assessed twice after optimization; safety failures stop and roll back")
            partial_records = unique_records(groups, core_protection, retention_regressions)
            report["partial_records_sha256"] = content_sha(partial_records)
            report["partial_required_ids"] = sorted({r["id"] for r in core_protection}
                | {r["id"] for rows in (prior_retention or {}).get("groups", {}).values() for r in rows})
            timer.switch("preparation")
        optimizer = master = basis = reference = None
        committed = commit_prepared = False
        runtime.branch.expert = None
        if field_first_token_weight == 4:
            report["field_loss_policy"] = field_first_token_policy()
        try:
            if session:
                session.running()
            _cancel(runtime)
            saved = session.load() if session else None
            if episode and saved is None:
                saved = goal.initial_saved(runtime, episode, session, parameter, previous, groups, _generation, _loss)
            if saved and (saved["report"].get("partial_policy") != report.get("partial_policy")
                    or saved["report"].get("field_loss_policy") != report.get("field_loss_policy")
                    or saved["report"].get("steps") != report["steps"]
                    or (saved["report"].get("training_selection") or {}).get("checkpoints") !=
                       (report["training_selection"] or {}).get("checkpoints")):
                raise ValueError("恢复检查点预算或字段目标政策不同，不能分配训练候选")
            resume_rng = torch.get_rng_state().clone() if saved else None
            resume_mps_rng = torch.mps.get_rng_state() if saved and str(runtime.device) == "mps" else None
            runtime.model.requires_grad_(False)
            report["memory_samples"].append(_memory(runtime))
            allow_training = True
            if partial:
                training_records = unique_records({k: groups[k] for k in ("new_training", "replay_training")}, core_protection, [])
                report["partial_training_baseline"] = (saved["report"]["partial_training_baseline"] if saved
                    else _generation(runtime, training_records))
                before = correct_ids(report["partial_training_baseline"], training_records, finite=False)
                report["effective_new_training_ids"] = [r["id"] for r in groups["new_training"] if r["id"] not in before
                    and is_finite_generation_check(validate_generation_check(_provenance(r).get("generation_check"), default_expected=r["answer"]))]
                replay_generation_rows = [r for r in groups["replay_training"] if r["id"] in before]
                report["training_selection"]["replay_generation_ids"] = [r["id"] for r in replay_generation_rows]
                report["training_selection"]["minimum_replay_generation"] = 0
            report["baseline_core_retention_generation"] = _generation(runtime, core_protection)
            if not all(row["passed"] for row in report["baseline_core_retention_generation"]):
                report["rejection_reason"] = "当前核心未通过已有知识完整保护检验，不能继续进化"
                allow_training = False
            retention_materials = unique_records({}, core_protection, retention_regressions)
            baseline_targets = list(report["baseline_core_retention_generation"])
            observed_ids = {row["id"] for row in baseline_targets}
            missing_targets = [row for row in retention_materials if row["answer"] == "" and row["id"] not in observed_ids]
            if missing_targets:
                baseline_targets.extend(_generation(runtime, missing_targets))
            retention_records = retention_loss_records(retention_materials, baseline_targets)
            report["core_retention_loss_targets"] = [{"id": row["id"], "answer": row["answer"]} for row in retention_records]
            if saved and any(row["answer"] == "" for row in retention_materials) and saved["report"].get(
                    "core_retention_loss_targets") != report["core_retention_loss_targets"]:
                raise ValueError("恢复检查点的保护损失目标不同，不能混用空答案结束符损失与完整答案损失")
            if "baseline_core_retention_loss" not in report:
                report["baseline_core_retention_loss"] = _loss_by_sample(runtime, retention_records)
            report["baseline_training_replay_generation"] = _generation(runtime, replay_generation_rows)
            if not all(row["passed"] for row in report["baseline_training_replay_generation"]):
                report["rejection_reason"] = "当前核心未通过预先指定的重放 TRAIN 生成保护，不能开始训练"
                allow_training = False
            if allow_training and preserve_replay_subspace:
                if saved:
                    basis_cpu, report["replay_subspace"] = saved["basis"], saved["report"]["replay_subspace"]
                else:
                    protection_capture = goal.capture_basis(runtime, episode, report, parameter, previous, groups, _generation, _loss) if episode and episode["protocol"] in {"evolution-goal-prospective-fixed450-v1", "evolution-goal-prospective-fixed450-v2", "evolution-goal-prospective-fixed450-v3", "evolution-goal-prospective-fixed450-v4", "evolution-goal-new-data-fixed450-v1"} else None
                    basis_cpu, report["replay_subspace"] = _capture_replay_subspace(runtime, parameter, groups["replay_training"], protection_capture=protection_capture)
                rank = report["replay_subspace"]["rank"]
                if (not 0 < rank < parameter.shape[1] or tuple(basis_cpu.shape) != (parameter.shape[1], rank)
                        or basis_cpu.dtype != torch.float32):
                    raise ValueError("重放子空间没有有效保护方向或已占满核心输入，不能安全更新")
                basis = basis_cpu.to(parameter.device)
                del basis_cpu
                reference = previous.to(device=parameter.device, dtype=torch.float32)
                report["baseline_training_replay_loss"] = _loss(runtime, groups["replay_training"])
            elif allow_training:
                _baseline_validation(runtime, parameter, previous, previous.to(parameter.device), groups, report)
                if not all(row["passed"] for row in report["baseline_replay_generation"]):
                    report["rejection_reason"] = "原核心未通过旧能力保护检验，不能据此宣称保持旧能力"
                    allow_training = False
                elif all(row["passed"] for row in report["baseline_new_generation"]):
                    report["rejection_reason"] = "原核心已通过这些新知识检验，没有可证明的新增能力需要巩固"
                    allow_training = False
            if allow_training:
                runtime.job["phase"] = "training"
                parameter.requires_grad_(True)
                master = torch.nn.Parameter(parameter.detach().float().clone())
                optimizer = torch.optim.Adam([master], lr=learning_rate, eps=1e-6)
                # Equal new/replay exposure, independent of corpus size. This is
                # a first-version policy, not a claimed optimal replay ratio.
                training = [groups["new_training"], groups["replay_training"]]
                new_exposures = {row["id"]: 0 for row in groups["new_training"]}
                kl_total, kl_maximum, consecutive_passes = 0.0, 0.0, 0
                resume_step = 0
                if saved:
                    report = _resume_core(session, saved, report, parameter, previous, groups, steps, master, optimizer)
                    report.setdefault("resumed_from_steps", []).append(saved["step"])
                    report["elapsed_scope"] = "wall time including pauses and process restarts"
                    started = report["started"]
                    report["memory_samples"].append(_memory(runtime))
                    resume_step = saved["step"]
                    new_exposures = saved["new_exposures"]
                    kl_total, kl_maximum = saved["kl_total"], saved["kl_maximum"]
                    consecutive_passes = saved["consecutive_passes"]
                    runtime.job["step"] = resume_step
                if timer:
                    timer.switch(None)
                    if saved:
                        fresh = timer.report["timing"]
                        for phase in ("preparation", "evaluation", "check", "checkpoint", "optimization"):
                            report["timing"][phase + "_seconds"] += fresh[phase + "_seconds"]
                    timer.report = report
                    timer.switch("optimization")
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(SEED)
                    if saved:
                        torch.set_rng_state(resume_rng)
                        if resume_mps_rng is not None:
                            torch.mps.set_rng_state(resume_mps_rng)
                    selected = bool(report["training_selection"] and (report["training_selection"]["selected_step"] is not None or episode and goal.stopped(report)))
                    selected |= partial and report.get("partial_safety_failed") is True
                    native = (native_steps(runtime, binary, report, groups, parameter, previous, master, optimizer,
                        reference, basis, resume_step, lambda: (new_exposures, kl_total, kl_maximum, consecutive_passes))
                        if binary is not None else None)
                    for step in range(steps if selected else resume_step, steps):
                        _cancel(runtime)
                        step_started = time.perf_counter()
                        rows = training[step % 2]
                        sample = rows[(step // 2) % len(rows)]
                        if native is not None:
                            actual_step, native_row, target_tokens = next(native)
                            if actual_step != step:
                                raise ValueError("原生核心恢复步数与训练循环不匹配")
                            ce_value, kl_value = native_row["ce"], native_row["kl"]
                            kl_total += kl_value
                            kl_maximum = max(kl_maximum, kl_value)
                            report["distillation"].update(steps=step + 1,
                                target_tokens=report["distillation"]["target_tokens"] + target_tokens,
                                kl_mean=kl_total / (step + 1), kl_max=kl_maximum)
                        else:
                            ids, labels = runtime._sample_tokens(sample)
                            optimizer.zero_grad(set_to_none=True)
                            parameter.grad = None
                            if distillation_weight:
                                replay = groups["replay_training"][(step // 2) % len(groups["replay_training"])]
                                replay_ids, replay_labels = (ids, labels) if step % 2 else runtime._sample_tokens(replay)
                                teacher = _teacher_replay(runtime, parameter, previous, master, replay_ids, replay_labels)
                            output = runtime.model(ids, labels=labels, use_cache=False)
                            weighted = field_first_token_weight == 4 and step % 2 == 0 and sample["id"] in FIELD_FIRST_TOKEN_IDS
                            loss = _field_weighted_loss(output.logits, labels) if weighted else output.loss
                            native_ce_value = float(output.loss.detach().float().cpu()) if field_first_token_weight == 4 else None
                            del output
                            if not torch.isfinite(loss):
                                raise RuntimeError("核心训练损失异常，已恢复旧版本")
                            loss.backward()
                            ce_value = float(loss.detach().float().cpu())
                            del loss
                            kl_value = None
                            if distillation_weight:
                                _cancel(runtime)
                                student = _answer_log_probs(runtime.model(replay_ids, use_cache=False).logits, replay_labels)
                                kl = F.kl_div(student, teacher, reduction="batchmean", log_target=True)
                                if not torch.isfinite(kl):
                                    raise RuntimeError("核心保留训练 KL 异常，已恢复旧版本")
                                (distillation_weight * kl).backward()
                                kl_value = float(kl.detach().cpu())
                                kl_total += kl_value
                                kl_maximum = max(kl_maximum, kl_value)
                                report["distillation"].update(steps=step + 1,
                                    target_tokens=report["distillation"]["target_tokens"] + teacher.shape[0],
                                    kl_mean=kl_total / (step + 1), kl_max=kl_maximum)
                                del student, teacher, kl, replay_ids, replay_labels
                            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                                raise RuntimeError("核心梯度缺失或异常，已恢复旧版本")
                            master.grad = parameter.grad.detach().float().clone()
                            torch.nn.utils.clip_grad_norm_([master], 1.0)
                            optimizer.step()
                            if preserve_replay_subspace:
                                _project_core_delta(master, reference, basis)
                            with torch.no_grad():
                                parameter.copy_(master)
                        if step % 2 == 0:
                            new_exposures[sample["id"]] += 1
                        runtime.job.update(step=step + 1, loss=ce_value + distillation_weight * (kl_value or 0),
                                           supervised_loss=ce_value, replay_kl=kl_value)
                        if timer:
                            duration = time.perf_counter() - step_started
                            report["timing"]["step_times"].append({"step": step + 1, "wall_seconds": duration, "sample_role": "new" if step % 2 == 0 else "replay"})
                            runtime.job.update(step_wall_seconds=duration, timing=report["timing"])
                            report["timing"]["step_wall_mean_seconds"] = sum(p["wall_seconds"] for p in report["timing"]["step_times"]) / (step + 1)
                        if field_first_token_weight == 4:
                            runtime.job.update(native_supervised_loss=native_ce_value, field_first_weight_applied=weighted)
                        if step % 8 == 0 or step + 1 == steps:
                            report["memory_samples"].append(_memory(runtime))
                        if preserve_replay_subspace and step + 1 in checkpoints:
                            runtime.job["phase"] = "training_check"
                            if timer:
                                timer.switch("check")
                            parameter.grad = None
                            if episode:
                                goal.observe_boundary(runtime, episode, report, groups, parameter, step + 1, _generation, _loss)
                            else:
                                train_generation = _generation(runtime, groups["new_training"])
                                replay_generation = _generation(runtime, replay_generation_rows)
                                replay_loss = _loss(runtime, groups["replay_training"])
                                replay_budget = report["baseline_training_replay_loss"] * 1.05 + 0.01
                                minimum_exposures = min(new_exposures.values())
                                accepted = (all(row["passed"] for row in train_generation)
                                    and all(row["passed"] for row in replay_generation)
                                    and replay_loss <= replay_budget and minimum_exposures >= 2)
                                if partial:
                                    before = {p["id"] for p in report["partial_training_baseline"] if p["passed"]}
                                    now = correct_ids(train_generation, groups["new_training"])
                                    prior = report["training_selection"]["results"]
                                    stable = now & (correct_ids(prior[-1]["new_generation"], groups["new_training"]) if prior else set())
                                    accepted = (bool(stable - before) and minimum_exposures >= 2
                                        and all(p["passed"] for p in replay_generation) and replay_loss <= replay_budget)
                                    forgotten = (before & {r["id"] for r in groups["new_training"]}) - correct_ids(train_generation, groups["new_training"], finite=False)
                                    if forgotten or any(not p["passed"] for p in replay_generation) or replay_loss > replay_budget:
                                        report["partial_safety_failed"] = True
                                    runtime.job["learning_progress"] = {"last_check_step": step + 1,
                                        "candidate_newly_mastered": len(stable - before), "pending": len(groups["new_training"]) - len(stable),
                                        "forgotten_ids": sorted(forgotten),
                                        "scope": "last TRAIN check only; candidate, not adopted mastery"}
                                consecutive_passes = consecutive_passes + 1 if accepted else 0
                                report["training_selection"]["results"].append({"step": step + 1,
                                    "new_generation": train_generation, "replay_training_generation": replay_generation,
                                    "replay_loss": replay_loss, "accepted": accepted,
                                    "consecutive_passes": consecutive_passes,
                                    "minimum_new_exposures": minimum_exposures,
                                    "constraint_residual": _constraint_residual(parameter, master, reference, basis)})
                                print(json.dumps({"kind": "core_train_checkpoint", "policy_version": report["training_selection"]["policy_version"],
                                    "step": step + 1, "new_passed": sum(row["passed"] for row in train_generation),
                                    "new_total": len(train_generation), "replay_loss": replay_loss,
                                    "protected_replay_train_passed": sum(row["passed"] for row in replay_generation),
                                    "protected_replay_train_total": len(replay_generation),
                                    "replay_loss_budget": replay_budget, "accepted": accepted,
                                    "minimum_new_exposures": minimum_exposures,
                                    "consecutive_passes": consecutive_passes}, ensure_ascii=False), flush=True)
                                if consecutive_passes >= 2 and (not partial or step + 1 == steps):
                                    report["training_selection"]["selected_step"] = step + 1
                            runtime.job["phase"] = "training"
                            if timer:
                                timer.switch("optimization")
                        # A check-step is saved only AFTER its whole generation,
                        # replay loss and streak update. Interrupted checks redo
                        # at most 16 steps from the preceding valid generation.
                        if session and ((step + 1) % 16 == 0 or step + 1 == steps or (partial or episode) and step + 1 in checkpoints):
                            if timer:
                                timer.switch("checkpoint")
                            _save_core_step(session, runtime, report, parameter, previous, master, optimizer, basis,
                                new_exposures, kl_total, kl_maximum, consecutive_passes)
                            if timer:
                                timer.switch("optimization")
                        if partial and report.get("partial_safety_failed"):
                            break
                        if preserve_replay_subspace and not partial and (report["training_selection"]["selected_step"] is not None or episode and goal.stopped(report)):
                            break
                parameter.grad = None
                parameter.requires_grad_(False)
                report["completed_steps"] = runtime.job["step"]
                if episode:
                    goal.finish_metadata(episode, report)
                evaluate_final = True
                runtime.job["phase"] = "validation"
                if timer:
                    timer.switch("evaluation")
                if preserve_replay_subspace:
                    report["replay_subspace"]["constraint_residual"] = _constraint_residual(parameter, master, reference, basis)
                    report["replay_subspace"]["replay_training_logit_drift"] = _replay_logit_drift(
                        runtime, parameter, previous, master, groups["replay_training"])
                    if partial:
                        report["candidate_core_retention_loss"] = _loss_by_sample(runtime, retention_records)
                        snapshot_groups = finish_partial(runtime, report, partial_records, groups, _generation, _loss, parameter, previous)
                        evaluate_final = False
                        runtime.job["learning_progress"] = report["learning_progress"]
                    elif report["training_selection"]["selected_step"] is None:
                        report["rejection_reason"] = ("完整450轮尚有待学知识或保护失败；未采纳候选，不读取检验集选参" if episode else "预先指定的训练检查点未连续两次通过新知识生成与旧能力保护，不使用检验集继续选参")
                        evaluate_final = False
                    else:
                        validation_reference = goal.validation_reference(episode, previous) if episode else previous
                        _baseline_validation(runtime, parameter, validation_reference, master, groups, report)
                        report["validation_scope"] = "pre-run reference and selected candidate evaluated once after TRAIN selection; no subsequent optimization"
                if evaluate_final:
                    report["validation_new_loss"] = _loss(runtime, groups["new_validation"])
                    report["validation_replay_loss"] = _loss(runtime, groups["replay_validation"])
                    report["new_generation"] = _generation(runtime, groups["new_validation"])
                    report["replay_generation"] = _generation(runtime, groups["replay_validation"])
                    report["core_retention_generation"] = _generation(runtime, core_protection)
                    report["core_retention_loss"] = _retention_loss_guard(
                        report.get("baseline_core_retention_loss", []),
                        _loss_by_sample(runtime, retention_records))
                    gates = {"heldout_new_loss_improved": report["validation_new_loss"] < report["baseline_new_loss"] * 0.995,
                             "protected_replay_loss_retained": report["validation_replay_loss"] <= report["baseline_replay_loss"] * 1.05 + 0.01,
                             "heldout_new_generation": all(row["passed"] for row in report["new_generation"]),
                             "heldout_new_generation_improved": sum(row["passed"] for row in report["new_generation"]) > sum(row["passed"] for row in report["baseline_new_generation"]),
                             "baseline_protected_replay_generation": all(row["passed"] for row in report["baseline_replay_generation"]),
                             "protected_replay_generation": all(row["passed"] for row in report["replay_generation"]),
                             "protected_core_retention_generation": all(row["passed"] for row in report["core_retention_generation"]),
                             "protected_core_retention_loss": report["core_retention_loss"]["passed"],
                             "core_tensor_changed": not torch.equal(validation_reference if episode and preserve_replay_subspace else previous, parameter.detach().cpu())}
                    report["gates"] = gates
                    report["promoted"] = all(gates.values())
                    if not report["promoted"]:
                        report["rejection_reason"] = "核心独立新知识或旧能力保护检验未通过"
            _cancel(runtime)
            runtime.job["phase"] = "commit" if report["promoted"] else "rejected"
            if timer:
                timer.switch("preparation")
                runtime.job["timing"] = report["timing"]
            report["elapsed_seconds"] = time.time() - started
            report["memory_samples"].append(_memory(runtime))
            report["sampled_memory_maxima"] = {key: max(sample.get(key, 0) for sample in report["memory_samples"])
                                              for key in report["memory_samples"][0]}
            if report["promoted"]:
                store = Path(runtime.store)
                weights = _core_path(runtime, version + ".safetensors")
                partial_path = weights.with_suffix(".partial")
                save_file({name: parameter.detach().cpu().contiguous()}, str(partial_path))
                os.chmod(partial_path, 0o600)
                with partial_path.open("rb") as checkpoint:
                    os.fsync(checkpoint.fileno())
                os.replace(partial_path, weights)
                if partial:
                    binding = {"core_version": version, "core_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                               "expert_id": None, "expert_sha256": None}
                    report["learning_progress"] = partial_progress(partial_records, report["partial_baseline"],
                        report["partial_candidate_first"], report["partial_candidate_second"], binding, accepted=True)
                    mastered = set(report["learning_progress"]["mastered_ids"]) | set(report["learning_progress"]["retained_ids"])
                    retention_regressions = [r for r in retention_regressions if r["id"] in mastered]
                    # Every mastered TRAIN context becomes mandatory generation protection as well as replay.
                    for row in (r for rows in snapshot_groups.values() for r in rows if r["purpose"] == "training"):
                        protected = copy.deepcopy(row)
                        protected.update(id="mastered-" + row["id"], purpose="validation", training_allowed=False)
                        value = _provenance(protected)
                        value.update(retention_generation_only=True, evolution_role="replay", original_sample_id=row["id"])
                        protected["provenance"] = json.dumps(value, ensure_ascii=False)
                        retention_regressions.append(protected)
                retention = save_core_retention(runtime, version, snapshot_groups if partial else groups,
                    regression_samples=list(prior_retention["regression_validation"] if prior_retention else ())
                        + list(retention_regressions))
                if partial:
                    report["accepted_retention_groups"] = snapshot_groups
                    report["accepted_retention_regressions"] = (list(prior_retention["regression_validation"] if prior_retention else ())
                        + list(retention_regressions))
                metadata = {"version": version, "identity": runtime.identity, "previous_version": previous_version,
                            "retention_sha256": retention["file_sha256"],
                            "dataset_sha256": report["dataset_sha256"],
                            "parameter": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "report": "evolution-" + run_id + ".json",
                            "format": "cumulative core tensor replacement", "trained_parameters": parameter.numel()}
                private_json(_core_path(runtime, version + ".json"), metadata)
                # Read saved tensors back before committing the only active pointer.
                _read_version(runtime, version)
                report["checkpoint"] = metadata
                report["retention"] = retention
            if timer:
                timer.switch(None)
            report["publication_state"] = "validated_candidate" if report["promoted"] else "rejected"
            _cancel(runtime)
            report_path = _core_path(runtime, "evolution-" + run_id + ".json")
            if session and report["promoted"]:
                report["publication_state"] = "published"
                private_json(report_path, report)
                if json.loads(report_path.read_text()) != json.loads(json.dumps(report, allow_nan=False)):
                    raise ValueError("核心候选报告回读不一致")
                active = {"version": version, "identity": runtime.identity, "sha256": report["checkpoint"]["sha256"]}
                artifacts = [version + suffix for suffix in (".safetensors", ".retention.json", ".json")]
                artifacts.append(report_path.name)
                session.prepare_commit(copy.deepcopy(report), {"active-core.json": active}, artifacts, event_kind="evolution")
                commit_prepared = True
                session.finish_commit()
                committed = True
            else:
                private_json(report_path, report)
                if report["promoted"]:
                    private_json(_pointer(runtime), {"version": version, "identity": runtime.identity,
                                                      "sha256": report["checkpoint"]["sha256"]})
                    report["publication_state"] = "published"
                    private_json(report_path, report)
                runtime.memory.event("evolution", report)
                if session:
                    session.rejected(report.get("rejection_reason", "核心候选未通过完整检验"))
            if report["promoted"]:
                runtime.active_core = version
                runtime.router_meta = {**getattr(runtime, "router_meta", {}), "calibrated": False,
                                       "reason": "核心版本已变化，快脑须重新训练核验"}
                committed = True
            runtime.job.update(state="complete", phase="complete" if report["promoted"] else "rejected", result=report)
            return report
        except TrainingPaused:
            if timer:
                timer.switch(None)
            runtime.job.update(state="paused", phase="paused")
            raise
        except Exception as error:
            if timer:
                timer.switch(None)
            if session and session.body.get("state") == "committed":
                # An acknowledgement error cannot undo an already durable
                # publication. Its exact report/event remain the authority.
                committed = True
                runtime.active_core = version
                runtime.router_meta = {**getattr(runtime, "router_meta", {}), "calibrated": False,
                                       "reason": "核心版本已变化，快脑须重新训练核验"}
                runtime.job.update(state="complete", phase="complete", result=report,
                                   commit_acknowledgement_error=str(error))
                return report
            if session and (commit_prepared or session.body.get("state") in {"committing", "committed"}):
                # The journal owns a verified immutable report now. Recovery
                # must finish that intent, not observe a rewritten failure hash.
                runtime.job.update(state="error", phase="committing", error=str(error))
                raise
            report.update(promoted=False, publication_state="failed", error=str(error), elapsed_seconds=time.time() - started)
            runtime.job.update(state="error", phase="error", error=str(error), result=report)
            # A failure report must never replace the original training failure.
            try:
                private_json(_core_path(runtime, "evolution-" + run_id + ".json"), report)
                runtime.memory.event("evolution_failure", report)
            except Exception:
                pass
            raise
        finally:
            if not committed:
                with torch.no_grad():
                    parameter.copy_(previous.to(parameter.device))
                runtime.active_core = previous_version
                # private_json commits atomically, so failure leaves the old
                # pointer. Restore only if a failing write changed it anyway.
                pointer = _pointer(runtime)
                current = pointer.read_bytes() if pointer.exists() else None
                if current != old_pointer:
                    if old_pointer is None:
                        pointer.unlink(missing_ok=True)
                    else:
                        private_json(pointer, json.loads(old_pointer))
            parameter.grad = None
            for value, original in requires_grad:
                value.requires_grad_(original)
            runtime.branch.expert = original_expert
            runtime.state = "ready"
            optimizer = master = basis = reference = None
            gc.collect()
            if str(runtime.device) == "mps":
                torch.mps.empty_cache()
