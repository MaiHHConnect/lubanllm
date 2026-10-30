"""Finite closed-book growth receipts and measured phase costs for partial evolution."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time

import torch

from .generation_checks import evaluate_generation, validate_generation_check, is_finite_generation_check
from .retention import messages, provenance

PARTIAL_POLICY = "closed-book-growth-v1"
PARTIAL_BUDGETS = (352, 450, 704)
PRODUCT_EXPERIENCE = 350
PRODUCT_BATCH = 57
DIRECTED_TEACHING = "explicit-model-identity-v1"
PARTIAL_GATES = {"closed_book_growth", "original_correct_retained", "all_previous_mastery_retained",
    "two_complete_closed_book_proofs", "minimum_new_exposures", "training_safety_retained",
    "protected_replay_loss_retained", "core_tensor_changed", "material_binding_verified",
    "protected_core_retention_generation", "protected_core_retention_loss", "protected_replay_generation"}


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def checkpoints(steps):
    if steps not in PARTIAL_BUDGETS:
        raise ValueError("部分进化仅支持研究352/704或产品固定450步")
    return list(range(50, steps, 50)) + [steps]


def unique_records(groups, protection, regressions):
    result, seen = [], {}
    for row in [r for rows in groups.values() for r in rows] + list(protection) + list(regressions):
        if row["id"] in seen:
            prior = seen[row["id"]]
            if (messages(prior) != messages(row) or prior["answer"] != row["answer"]
                    or provenance(prior).get("generation_check") != provenance(row).get("generation_check")):
                raise ValueError("部分进化同一ID有冲突材料")
            continue
        seen[row["id"]] = row
        result.append(copy.deepcopy(row))
    return result


def validate_product_records(records):
    """Bind bounded batches to earned experience or explicit model teaching."""
    from .retention import _validate_record, _validate_groups, normalized_context
    common = {"groups", "protection", "retention_regressions"}
    if not isinstance(records, dict) or set(records) not in (common | {"experience"}, common | {"directed_teaching"}):
        raise ValueError("产品450轮需要互斥的冻结350经验或明确授权定向教学")
    groups = records["groups"]
    _validate_groups(groups)
    if not 1 <= len(groups["new_training"]) <= PRODUCT_BATCH:
        raise ValueError("产品固定450轮只训练至多57条待学新内容")
    if "directed_teaching" in records:
        _validate_directed_teaching(records["directed_teaching"], groups)
        return
    pool = records["experience"]
    if not isinstance(pool, list) or len(pool) != PRODUCT_EXPERIENCE:
        raise ValueError("产品进化需要350条有效去重新内容")
    contexts, ids, allowed = set(), set(), set()
    for row in pool:
        _validate_record(row)
        context = normalized_context(row)
        if context in contexts or row["id"] in ids or provenance(row).get("evolution_role") != "new":
            raise ValueError("经验材料重复或不是新知识")
        contexts.add(context)
        ids.add(row["id"])
        allowed.add(content_sha(row))
    if any(content_sha(row) not in allowed
           for key in ("new_training", "new_validation") for row in groups[key]):
        raise ValueError("本轮新知识不属于已冻结经验材料")


def _validate_directed_teaching(teaching, groups):
    """A narrow authorization receipt, not permission to invent daily experience."""
    if (not isinstance(teaching, dict) or set(teaching) != {"protocol", "authorization", "facts"}
            or teaching["protocol"] != DIRECTED_TEACHING):
        raise ValueError("定向教学协议无效")
    auth = teaching["authorization"]
    if (not isinstance(auth, dict) or set(auth) != {"source", "explicit", "training_allowed", "reference"}
            or auth["source"] != "human" or auth["explicit"] is not True or auth["training_allowed"] is not True
            or not isinstance(auth["reference"], str) or not auth["reference"].strip()):
        raise ValueError("定向教学缺少真实明确的人类训练授权引用")
    facts = teaching["facts"]
    expected = {"model_name": "luban0", "model_fine_tuner": "麦好火"}
    # User-authorized positioning, not a measured claim of unlimited capability.
    if isinstance(facts, dict) and "model_positioning" in facts:
        expected["model_positioning"] = "无限进化"
    if not isinstance(facts, dict) or set(facts) != set(expected):
        raise ValueError("定向教学仅授权模型名称、微调方及可选的无限进化定位，不包含用户身份")
    covered = {purpose: set() for purpose in ("training", "validation")}
    rows = {purpose: {row["id"]: row for row in groups["new_" + purpose] + [
        row for row in groups["replay_" + purpose]
        if isinstance(provenance(row).get("retention_from_version"), str)
        and re.fullmatch(r"core-[a-f0-9]{12}", provenance(row)["retention_from_version"])
    ]} for purpose in covered}
    for key, value in expected.items():
        fact = facts[key]
        if (not isinstance(fact, dict) or set(fact) != {"subject", "value", "training_ids", "validation_ids"}
                or fact["subject"] != "model" or fact["value"] != value):
            raise ValueError("定向教学事实必须匹配授权的模型身份与定位")
        for purpose in covered:
            ids = fact[purpose + "_ids"]
            if (not isinstance(ids, list) or not ids or any(not isinstance(identifier, str) for identifier in ids)
                    or len(set(ids)) != len(ids) or set(ids) & covered[purpose]
                    or not set(ids) <= set(rows[purpose])):
                raise ValueError("每个教学事实需要独立完整且不重复的训练与验证ID")
            covered[purpose].update(ids)
            for identifier in ids:
                row = rows[purpose][identifier]
                source = provenance(row)
                check = validate_generation_check(source.get("generation_check"), default_expected=row["answer"])
                if (row.get("source") != "human" or source.get("protocol_id") != DIRECTED_TEACHING
                        or source.get("authorization_reference") != auth["reference"]
                        or source.get("teaching_fact") != key or not isinstance(source.get("split_family"), str)
                        or not source["split_family"].strip() or row["answer"] != value
                        or check["type"] != "exact" or check["expected"] != value):
                    raise ValueError("教学材料必须绑定授权事实、独立问法家族与准确短答案")
    if any(not {row["id"] for row in groups["new_" + purpose]} <= covered[purpose] for purpose in covered):
        raise ValueError("全部NEW材料必须属于本次明确授权的模型身份与定位")


def correct_ids(observations, records, *, finite=True):
    if [p.get("id") for p in observations] != [r["id"] for r in records]:
        raise ValueError("部分进化缺少完整且有序的逐项观察")
    result = set()
    for row, proof in zip(records, observations):
        if any(type(proof.get(key)) is not bool for key in ("passed", "complete", "branch_enabled", "reference_used")):
            raise ValueError("部分进化观察标志必须为真实布尔值")
        if proof["branch_enabled"] or proof["reference_used"]:
            raise ValueError("部分进化失败基线与候选也必须实际闭卷分支OFF")
        check = validate_generation_check(provenance(row).get("generation_check"), default_expected=row["answer"])
        actual = bool(proof.get("complete") is True and evaluate_generation(proof.get("answer", ""), check, True)["passed"])
        if proof.get("passed") != actual:
            raise ValueError("部分进化passed与实际完整回答不一致")
        if (proof.get("passed") is True and proof.get("complete") is True
                and proof.get("branch_enabled") is False and proof.get("reference_used", False) is False
                and evaluate_generation(proof.get("answer", ""), check, True)["passed"]):
            if not finite or is_finite_generation_check(check):
                result.add(row["id"])
        elif proof.get("passed") is True:
            raise ValueError("部分进化的通过观察不是实际闭卷完整正确回答")
    return result


def progress(records, baseline, first, second, binding, *, accepted=False):
    before = correct_ids(baseline, records, finite=False)
    after = correct_ids(first, records, finite=False) & correct_ids(second, records, finite=False)
    mastered = correct_ids(first, records) & correct_ids(second, records)
    gained, forgotten = mastered - before, before - after
    proofs = []
    for row, one, two in zip(records, first, second):
        if row["id"] in mastered:
            proofs.append({"id": row["id"], "context_sha256": content_sha(messages(row)),
                "sample_sha256": content_sha(row), "check": validate_generation_check(
                    provenance(row).get("generation_check"), default_expected=row["answer"]),
                "binding": copy.deepcopy(binding), "observations": [copy.deepcopy(one), copy.deepcopy(two)]})
    ids = [r["id"] for r in records]
    return {"policy": PARTIAL_POLICY, "attempted_ids": ids, "baseline_correct_ids": sorted(before),
        "mastered_ids": sorted(mastered), "candidate_learned_ids": sorted(gained),
        "learned_ids": sorted(gained) if accepted else [], "pending_ids": sorted(set(ids) - mastered),
        "retained_ids": sorted(before & after), "forgotten_ids": sorted(forgotten), "attempted": len(ids), "candidate_newly_mastered": len(gained),
        "newly_mastered": len(gained) if accepted else 0, "pending": len(set(ids) - mastered),
        "retained": len(before & after), "forgotten": len(forgotten), "mastery_proofs": proofs,
        "scope": "two complete closed-book observations of each exact full context on one frozen core; not transfer"}


def _protection_evidence(report, records, frozen=None):
    """Reuse both final generations; compare measured per-record CE with the original bound."""
    from .evolution import _retention_loss_guard
    by_id = {row["id"]: row for row in records}
    core_ids = [row["id"] for row in report["baseline_core_retention_generation"]]
    replay_ids = report["split_ids"]["replay_validation"]
    baseline = report["baseline_core_retention_loss"]
    candidate = report["candidate_core_retention_loss"]
    loss_ids = [row["id"] for row in baseline]
    for identifiers in (core_ids, replay_ids, loss_ids):
        if len(set(identifiers)) != len(identifiers) or not set(identifiers) <= set(by_id):
            raise ValueError("部分进化保护ID重复或不属于冻结材料")
    if frozen is not None and (core_ids != [r["id"] for r in frozen["protection"]]
            or replay_ids != [r["id"] for r in frozen["groups"]["replay_validation"]]
            or loss_ids != [r["id"] for r in unique_records({}, frozen["protection"], frozen["retention_regressions"])]):
        raise ValueError("部分进化保护证据遗漏冻结材料")
    if ([row["id"] for row in candidate] != loss_ids or any(type(row.get("loss")) not in (int, float)
            for row in baseline + candidate)):
        raise ValueError("部分进化逐条保护损失缺失或无效")
    core_rows = [by_id[key] for key in core_ids]
    if correct_ids(report["baseline_core_retention_generation"], core_rows, finite=False) != set(core_ids):
        raise ValueError("部分进化原核心保护基线未通过")
    before = correct_ids(report["partial_baseline"], records, finite=False)
    replay_ids = [key for key in replay_ids if key in before]
    observations = [{row["id"]: row for row in report[key]} for key in
                    ("partial_baseline", "partial_candidate_first", "partial_candidate_second")]
    evidence = {"protected_replay_ids": replay_ids,
        "replay_generation_scope": "baseline-correct replay_validation only; both frozen candidate observations required",
        "core_retention_generation": [observations[1][key] for key in core_ids],
        "core_retention_generation_second": [observations[2][key] for key in core_ids],
        "baseline_replay_generation": [observations[0][key] for key in replay_ids],
        "replay_generation": [observations[1][key] for key in replay_ids],
        "replay_generation_second": [observations[2][key] for key in replay_ids],
        "core_retention_loss": _retention_loss_guard(baseline, candidate)}
    def retained(keys, identifiers):
        selected = [by_id[key] for key in identifiers]
        return all(correct_ids(evidence[key], selected, finite=False) == set(identifiers) for key in keys)
    gates = {"protected_core_retention_generation": retained(
        ("core_retention_generation", "core_retention_generation_second"), core_ids),
        "protected_replay_generation": retained(("replay_generation", "replay_generation_second"), replay_ids),
        "protected_core_retention_loss": evidence["core_retention_loss"]["passed"]}
    return evidence, gates


def validate_partial_progress(report, records, identity, version, weights_sha):
    frozen = records if isinstance(records, dict) else None
    if frozen is not None:
        if report.get("steps") == 450:
            validate_product_records(frozen)
        records = unique_records(frozen["groups"], frozen["protection"], frozen["retention_regressions"])
    value = report.get("learning_progress") or {}
    if report.get("partial_policy") != PARTIAL_POLICY or value.get("policy") != PARTIAL_POLICY:
        raise ValueError("部分进化政策不兼容")
    binding = {"core_version": version, "core_sha256": weights_sha, "expert_id": None, "expert_sha256": None}
    recomputed = progress(records, report["partial_baseline"], report["partial_candidate_first"],
                          report["partial_candidate_second"], binding, accepted=report.get("promoted") is True)
    if value != recomputed or report.get("identity") != identity:
        raise ValueError("部分进化计数或实际绑定与逐题原始证明不一致")
    if content_sha(records) != report.get("partial_records_sha256"):
        raise ValueError("部分进化完整材料SHA不一致")
    if report.get("promoted") is True:
        try:
            evidence, gates = _protection_evidence(report, records, frozen)
            if any(report.get(key) != item for key, item in evidence.items()) or any(
                    report.get("gates", {}).get(key) is not item for key, item in gates.items()):
                raise ValueError("部分进化保护门与实际逐条证据不一致")
        except (KeyError, TypeError) as error:
            raise ValueError("部分进化缺少逐条保护证据") from error
    if report.get("promoted") is True and (report.get("partial_safety_failed") is True
            or not isinstance(report.get("gates"), dict) or set(report["gates"]) != PARTIAL_GATES
            or any(value is not True for value in report["gates"].values())):
        raise ValueError("部分进化完整安全门禁未通过")
    if report.get("steps") == 450 and report.get("promoted") and report.get("completed_steps") != 450:
        raise ValueError("产品进化未完成固定450次实际更新")
    if frozen is not None and report.get("promoted"):
        required = {r["id"] for r in frozen["protection"]}
        required |= {r["id"] for r in records if provenance(r).get("retention_from_version")}
        if not required <= (set(value["mastered_ids"]) | set(value["retained_ids"])):
            raise ValueError("部分进化遗漏既有掌握保护")
        if report.get("accepted_retention_groups") != retained_groups(frozen["groups"], set(value["mastered_ids"]) | set(value["retained_ids"])):
            raise ValueError("待学习项被错误加入已采纳保护")
    if report.get("promoted") and (not value["learned_ids"] or value["forgotten_ids"]):
        raise ValueError("部分进化没有新增掌握或丢失原会能力")
    return value


def retained_groups(groups, mastered):
    """Attempted/pending records stay in the journal/Memory, never become accepted replay."""
    return {key: [copy.deepcopy(row) for row in rows if row["id"] in mastered]
            for key, rows in groups.items()}


class PhaseTimer:
    """Synchronize only at phase boundaries; step timings add no device barriers."""
    def __init__(self, runtime, report):
        self.runtime, self.report = runtime, report
        self.phase, self.mark = None, time.perf_counter()
        value = report.setdefault("timing", {})
        for phase in ("optimization", "check", "evaluation", "preparation", "checkpoint"):
            value.setdefault(phase + "_seconds", 0.0)
        value.setdefault("pause_seconds", 0.0)
        value.setdefault("step_times", [])
        value["measurement"] = "perf_counter wall; MPS synchronization at phase boundaries only; per-step dispatch wall includes existing loss readback, not an extra device barrier"

    def switch(self, phase):
        if str(self.runtime.device) == "mps":
            torch.mps.synchronize()
        now = time.perf_counter()
        if self.phase:
            self.report["timing"][self.phase + "_seconds"] += now - self.mark
        self.phase, self.mark = phase, now
        self.update()

    def update(self):
        value = self.report["timing"]
        session = getattr(self.runtime, "training_session", None)
        value["pause_seconds"] = session.body.get("pause_elapsed_seconds", 0.0) if session else 0.0
        value["active_elapsed_seconds"] = sum(value[p + "_seconds"] for p in
            ("optimization", "check", "evaluation", "preparation", "checkpoint"))
        value["cumulative_elapsed_seconds"] = max(0.0, time.time() - self.report["started"])
        value["unmeasured_seconds"] = max(0.0, value["cumulative_elapsed_seconds"]
            - value["active_elapsed_seconds"] - value["pause_seconds"])
        gained = (self.report.get("learning_progress") or {}).get("newly_mastered", 0)
        value["growth_per_active_minute"] = gained * 60 / value["active_elapsed_seconds"] if value["active_elapsed_seconds"] else None
        if any(not math.isfinite(value[k]) or value[k] < 0 for k in value if k.endswith("_seconds")):
            raise ValueError("进化计时不完整或异常")


def finish_partial(runtime, report, records, groups, generation, loss, parameter, previous):
    """Freeze one candidate, assess every declared target twice, and forbid old loss."""
    cached = {p["id"]: p for p in report["partial_training_baseline"]}
    deferred = [r for r in records if r["id"] not in cached]
    candidate = parameter.detach().clone()
    with torch.no_grad():
        try:
            parameter.copy_(previous.to(parameter.device))
            cached.update({p["id"]: p for p in generation(runtime, deferred)})
            report["baseline_replay_loss"] = loss(runtime, groups["replay_validation"])
            report["baseline_new_loss"] = loss(runtime, groups["new_validation"]) if groups["new_validation"] else None
        finally:
            parameter.copy_(candidate)
    report["partial_baseline"] = [cached[r["id"]] for r in records]
    first, second = generation(runtime, records), generation(runtime, records)
    weights_sha = hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    binding = {"core_version": report["version"], "core_sha256": weights_sha,
               "expert_id": None, "expert_sha256": None}
    value = progress(records, report["partial_baseline"], first, second, binding)
    report.update(partial_candidate_first=first, partial_candidate_second=second)
    report["validation_replay_loss"] = loss(runtime, groups["replay_validation"])
    report["validation_new_loss"] = loss(runtime, groups["new_validation"]) if groups["new_validation"] else None
    evidence, protection_gates = _protection_evidence(report, records)
    report.update(evidence)
    gates = {"closed_book_growth": bool(value["candidate_learned_ids"]),
        "original_correct_retained": not value["forgotten_ids"],
        "all_previous_mastery_retained": set(report["partial_required_ids"]) <= (set(value["mastered_ids"]) | set(value["retained_ids"])),
        "two_complete_closed_book_proofs": bool(value["mastered_ids"]),
        "minimum_new_exposures": (report["completed_steps"] // 2) // len(groups["new_training"]) >= 2,
        "training_safety_retained": report.get("partial_safety_failed") is not True,
        "protected_replay_loss_retained": report["validation_replay_loss"] <= report["baseline_replay_loss"] * 1.05 + .01,
        "core_tensor_changed": not torch.equal(parameter.detach().cpu(), previous),
        "material_binding_verified": content_sha(records) == report["partial_records_sha256"]}
    gates.update(protection_gates)
    report["gates"], report["promoted"] = gates, all(gates.values())
    report["learning_progress"] = progress(records, report["partial_baseline"], first, second, binding,
                                           accepted=report["promoted"])
    if not report["promoted"]:
        report["rejection_reason"] = "本轮无新增闭卷掌握或原会/已掌握能力保护未通过，全部候选回滚"
    return retained_groups(groups, set(value["mastered_ids"]) | set(value["retained_ids"]))


@torch.no_grad()
def extend_protected_basis(basis, generated):
    """Extend a CPU FP32 span using the existing replay-capture algorithm.

    Empty and full old spans are valid inputs here. A caller that needs a
    trainable complement must separately call validate_basis on the result.
    """
    def _finite(value, name):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    if (not isinstance(basis, torch.Tensor) or basis.dtype != torch.float32
            or basis.device.type != "cpu" or basis.ndim != 2 or basis.shape[0] < 1
            or not 0 <= basis.shape[1] <= basis.shape[0]
            or not isinstance(generated, torch.Tensor) or generated.dtype != torch.float32
            or generated.device.type != "cpu" or generated.ndim != 2
            or generated.shape[0] < 1 or generated.shape[1] != basis.shape[0]):
        raise ValueError("basis and generated must be compatible CPU FP32 matrices")
    _finite(basis, "basis")
    _finite(generated, "generated protection inputs")
    error = basis.T @ basis - torch.eye(basis.shape[1])
    if float(error.norm()) > 1e-4:
        raise ValueError("basis must have orthonormal columns")
    original_basis = basis.detach()
    residual = generated - (generated @ basis) @ basis.T
    residual -= (residual @ basis) @ basis.T
    _finite(residual, "generated protection residual")
    _, extra_singular, extra_vh = torch.linalg.svd(residual, full_matrices=False)
    _finite(extra_singular, "generated protection singular values")
    extra_tolerance = max(residual.shape) * torch.finfo(torch.float32).eps * float(extra_singular[0])
    added_rank = int((extra_singular > extra_tolerance).sum())
    extra = extra_vh[:added_rank].T.contiguous()
    extra -= basis @ (basis.T @ extra)
    extra -= basis @ (basis.T @ extra)
    # Numerical copies of existing directions must not enlarge the rank.
    extra_u, stable_singular, _ = torch.linalg.svd(extra, full_matrices=False)
    _finite(stable_singular, "stable protection singular values")
    added_rank = int((stable_singular > 1e-5).sum())
    if added_rank:
        basis = torch.linalg.qr(torch.cat((basis, extra_u[:, :added_rank]), dim=1), mode='reduced').Q.contiguous()
    else:
        basis = original_basis
    # An empty original span has no directions that could be weakened.
    old_span_residual = (float((original_basis - basis @ (basis.T @ original_basis)).abs().max())
                         if original_basis.numel() else 0.)
    if old_span_residual > 2e-5:
        raise ValueError('追加旧生成路径不能削弱原TRAIN保护方向')
    return basis, {"added_rank": added_rank, "extra_tolerance": extra_tolerance,
                   "old_span_residual": old_span_residual}


def capture_replay_subspace(runtime, parameter, samples, *, cancel, protection_capture=None):
    """CPU rank-revealing SVD of frozen last-projection TRAIN inputs only."""
    if protection_capture is not None and (
            hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != protection_capture.get("original_tensor_sha256")
            or runtime.model.training or runtime.branch.expert is not None):
        raise ValueError("capture must execute on original eval branch-OFF B45")
    decoder = getattr(runtime, "decoder", runtime.model.model)
    mlp = decoder.layers[-1].mlp
    projection = getattr(mlp, "base", mlp).down_proj
    rows, counts = [], []
    current_labels = None

    def capture(module, args):
        inputs = args[0]
        if inputs.ndim != 3 or tuple(inputs.shape[:2]) != tuple(current_labels.shape):
            raise ValueError("核心子空间需要完整序列的末层投影输入")
        selected = inputs[:, :-1][current_labels[:, 1:] != -100].detach().cpu().float()
        if not selected.shape[0] or selected.shape[1] != parameter.shape[1] or not torch.isfinite(selected).all():
            raise ValueError("核心子空间输入缺失、尺寸不兼容或非有限")
        rows.append(selected)
        counts.append(selected.shape[0])

    if protection_capture is not None:
        from .goal_fresh import validated_protection_tokens
        protected = validated_protection_tokens(runtime, protection_capture)
    else:
        protected = []
    handle = projection.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for sample in samples:
                if sample.get("purpose") != "training" or provenance(sample).get("evolution_role") != "replay":
                    raise ValueError("核心子空间只允许捕获重放训练材料，不能读取检验集")
                cancel(runtime)
                ids, current_labels = runtime._sample_tokens(sample)
                count = len(rows)
                # Only the final LM projection is shortened; every decoder
                # input position still executes and reaches the capture hook.
                runtime.model(ids, use_cache=False, logits_to_keep=1)
                if len(rows) != count + 1:
                    raise RuntimeError("真实末层投影没有执行一次完整输入捕获")
            for sample_id, ids, current_labels in protected:
                cancel(runtime)
                count = len(rows)
                runtime.model(ids, use_cache=False, logits_to_keep=1)
                if len(rows) != count + 1:
                    raise RuntimeError("旧保护生成路径没有执行一次完整输入捕获")
    finally:
        handle.remove()
    matrix = torch.cat(rows[:len(samples)], dim=0)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    cancel(runtime)
    tolerance = max(matrix.shape) * torch.finfo(torch.float32).eps * float(singular[0])
    rank = int((singular > tolerance).sum())
    basis = vh[:rank].T.contiguous()
    training_rank, added_rank, old_span_residual = rank, 0, 0.0
    if protected:
        generated = torch.cat(rows[len(samples):], dim=0)
        basis, extension = extend_protected_basis(basis, generated)
        added_rank = extension['added_rank']
        extra_tolerance = extension['extra_tolerance']
        old_span_residual = extension['old_span_residual']
        rank = basis.shape[1]
    orthogonality = float((basis.T @ basis - torch.eye(rank)).abs().max()) if rank else 0.0
    metadata = {"enabled": True, "rank": rank, "input_dimension": parameter.shape[1],
        "captured_positions": sum(counts), "training_ids": [sample["id"] for sample in samples],
        "positions_per_sample": counts, "svd_dtype": "torch.float32", "svd_device": "cpu",
        "rank_tolerance": tolerance, "rank_tolerance_rule": "max(rows, columns) * float32 epsilon * largest singular value",
        "largest_singular_value": float(singular[0]), "smallest_retained_singular_value": float(singular[training_rank - 1]) if training_rank else None,
        "orthogonality_max_abs": orthogonality,
        "scope": "frozen replay training answer/EOS predictor inputs only; finite span, not all future tasks"}
    if protection_capture is not None:
        metadata.update(basis_sha256=hashlib.sha256(basis.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(), training_rank=training_rank, added_protection_rank=added_rank, old_span_max_abs=old_span_residual, protection_rank_tolerance=extra_tolerance if protected else None, protection_ids=[entry[0] for entry in protected],
            protection_positions_per_sample=counts[len(samples):],
            positions_per_sample=counts[:len(samples)],
            protection_capture_sha256=content_sha(protection_capture),
            baseline_tensor_sha256=protection_capture['baseline_tensor_sha256'],
            scope='original-correct known OLD generated raw answer/EOS predictor states plus unchanged REPLAY TRAIN gold states; no CE/KL on protected validation; no unseen-validation claim')
    return basis, metadata
