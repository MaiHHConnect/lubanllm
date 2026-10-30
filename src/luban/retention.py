"""Immutable accepted-core materials; database deletion never removes protection."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

from .generation_checks import evaluate_generation, validate_generation_check

RETENTION_SCHEMA_VERSION = 1
GROUP_KEYS = ("new_training", "new_validation", "replay_training", "replay_validation")
_VERSION = re.compile(r"core-[a-f0-9]{12}")


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def provenance(sample):
    value = sample.get("provenance", "{}")
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("材料来源记录必须为对象")
    return value


def retention_loss_records(records, baseline):
    """Freeze a separate loss view; generated validation answers never enter TRAIN."""
    observed = {row["id"]: row for row in baseline}
    if len(observed) != len(baseline):
        raise ValueError("Duplicate retention baseline observations")
    result = []
    for row in records:
        target = copy.deepcopy(row)
        if not isinstance(row["answer"], str):
            raise ValueError("Retention loss target must be a string")
        if row["answer"] == "":
            value = observed.get(row["id"], {})
            if (value.get("passed") is not True or value.get("complete") is not True
                    or value.get("branch_enabled") is not False or value.get("reference_used") is not False
                    or not isinstance(value.get("answer"), str) or not value["answer"].strip()):
                raise ValueError("Empty retention target requires a correct complete core-only baseline answer")
            check = validate_generation_check(provenance(row).get("generation_check"), default_expected=row["answer"])
            if not evaluate_generation(value["answer"], check, True)["passed"]:
                raise ValueError("Retention baseline answer failed its original check")
            target["answer"] = value["answer"]
        result.append(target)
    return result


def messages(sample):
    rows = provenance(sample).get("messages") or [{"role": "user", "content": sample.get("prompt")}]
    if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict)
            or row.get("role") not in {"system", "user", "assistant"}
            or not isinstance(row.get("content"), str) or not row["content"].strip() for row in rows)):
        raise ValueError("保护材料需要完整的角色与非空上下文")
    return rows


def normalized_context(sample):
    return _encoded([{**row, "content": "".join(unicodedata.normalize("NFKC", row["content"]).split())}
                     for row in messages(sample)]).decode()


def verified_source(sample):
    value = provenance(sample)
    return sample.get("source") == "human" or (
        (value.get("program_verified") is True or value.get("human_verified") is True)
        and isinstance(value.get("verification_evidence"), str)
        and bool(value["verification_evidence"].strip()))


def eligible_learning_sample(sample):
    value = provenance(sample)
    return (bool(sample.get("approved")) and bool(sample.get("training_allowed"))
            and not value.get("route_only") and not value.get("retention_generation_only")
            and verified_source(sample))


def daily_learning_source(sample):
    """Product teaching origins only; historical research remains usable as protection."""
    value = provenance(sample)
    if (sample.get("source") not in {"human", "teacher"}
            or any(value.get(key) for key in ("dataset", "protocol_id", "benchmark_run",
                                              "route_only", "retention_generation_only"))):
        return False
    return (value.get("verified_by") == "local requester"
            or bool(value.get("feeding_batch_id") and value.get("human_verified") is True
                    and value.get("verification_evidence"))
            or bool(sample.get("source") == "teacher" and value.get("request_id")
                    and value.get("messages")))


def _validate_record(sample, *, generation_only=False):
    if (not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not sample["id"]
            or not isinstance(sample.get("answer"), str) or sample.get("purpose") not in {"training", "validation"}):
        raise ValueError("核心保护材料缺少完整记录或有效拆分")
    normalized_context(sample)
    value = provenance(sample)
    if generation_only:
        if (sample["purpose"] != "validation" or not sample.get("approved")
                or sample.get("training_allowed") or value.get("retention_generation_only") is not True
                or not verified_source(sample)):
            raise ValueError("额外保护必须是已确认的只读生成检验，不能允许训练")
    elif not eligible_learning_sample(sample) or not sample["answer"].strip():
        raise ValueError("核心保护训练/检验材料未确认正确或不允许训练")
    validate_generation_check(value.get("generation_check"), default_expected=sample["answer"])


def _validate_groups(groups):
    if not isinstance(groups, dict) or set(groups) != set(GROUP_KEYS):
        raise ValueError("核心保护快照需要完整四组材料")
    contexts, ids, families = {}, {}, {}
    for key, records in groups.items():
        if not isinstance(records, list):
            raise ValueError("核心保护材料组必须为列表")
        role, purpose = key.split("_")
        for sample in records:
            _validate_record(sample)
            value = provenance(sample)
            if sample["purpose"] != purpose or value.get("evolution_role") != role:
                raise ValueError("核心保护材料角色或拆分不匹配")
            context = normalized_context(sample)
            if sample["id"] in ids or context in contexts:
                raise ValueError("核心保护快照存在重复ID或上下文")
            ids[sample["id"]], contexts[context] = sample, sample
            family = value.get("split_family")
            if family:
                if not isinstance(family, str) or (family in families and families[family] != purpose):
                    raise ValueError("核心训练与独立检验出现相同模板家族，不能进化")
                families[family] = purpose


def validate_core_retention(payload, identity, version):
    if (not isinstance(version, str) or not _VERSION.fullmatch(version)
            or not isinstance(payload, dict) or set(payload) != {
                "schema_version", "version", "identity", "groups", "regression_validation", "content_sha256"}
            or type(payload.get("schema_version")) is not int or payload.get("schema_version") != RETENTION_SCHEMA_VERSION
            or payload.get("version") != version or payload.get("identity") != identity):
        raise ValueError("核心保护快照格式、版本或基座身份不兼容")
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    if hashlib.sha256(_encoded(content)).hexdigest() != payload.get("content_sha256"):
        raise ValueError("核心保护快照内容校验失败")
    _validate_groups(payload["groups"])
    regressions = payload["regression_validation"]
    if not isinstance(regressions, list):
        raise ValueError("核心生成保护材料必须为列表")
    ids = {sample["id"] for records in payload["groups"].values() for sample in records}
    seen = {}
    for sample in regressions:
        _validate_record(sample, generation_only=True)
        if sample["id"] in ids:
            raise ValueError("核心生成保护材料ID重复")
        ids.add(sample["id"])
        context = normalized_context(sample)
        check = validate_generation_check(provenance(sample).get("generation_check"), default_expected=sample["answer"])
        if context in seen and seen[context] != check:
            raise ValueError("核心生成保护规则对同一上下文有冲突")
        seen[context] = check
    return payload


def read_retention_file(path, identity, version):
    path = Path(path)
    if any(candidate.is_symlink() for candidate in (path, path.parent)):
        raise ValueError("核心保护快照不能使用符号链接")
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("活动核心缺少完整保护快照；不能继续学习或进化") from error
    return validate_core_retention(payload, identity, version)


def read_core_retention(runtime, version=None):
    version = version if version is not None else getattr(runtime, "active_core", None)
    if version is None:
        return None
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("核心保护版本名称无效")
    return read_retention_file(Path(runtime.store) / (version + ".retention.json"), runtime.identity, version)


def save_core_retention(runtime, version, groups, *, regression_samples=()):
    payload = {"schema_version": RETENTION_SCHEMA_VERSION, "version": version, "identity": runtime.identity,
               "groups": copy.deepcopy(groups), "regression_validation": copy.deepcopy(list(regression_samples))}
    payload["content_sha256"] = hashlib.sha256(_encoded(payload)).hexdigest()
    validate_core_retention(payload, runtime.identity, version)
    path = Path(runtime.store) / (version + ".retention.json")
    if any(candidate.is_symlink() for candidate in (path, path.parent)):
        raise ValueError("核心保护快照不能使用符号链接")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if read_retention_file(path, runtime.identity, version) != payload:
            raise ValueError("已有核心保护快照不可覆盖；纠错替代协议尚未实现")
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encoded(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    if read_retention_file(path, runtime.identity, version) != payload:
        raise ValueError("核心保护快照回读不一致")
    return {"path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_sha256": payload["content_sha256"]}


def protected_validation(runtime, *, include_learning=False):
    """Core gold by default; opt-in committed branch gold remains branch-scoped."""
    snapshot = read_core_retention(runtime)
    result = ([] if snapshot is None else copy.deepcopy(snapshot["groups"]["new_validation"]
              + snapshot["groups"]["replay_validation"] + snapshot["regression_validation"]))
    if snapshot is None:
        # The original-core contract is read-only and excludes identity cases.
        # It is the protection source while no accepted core version exists.
        from .original_protection import protection_rows
        result.extend(protection_rows(runtime))
    if not include_learning:
        return result
    from .memory import Memory
    from .bun_bridge import RemoteMemory
    memory = getattr(runtime, "memory", None)
    if not isinstance(memory, (Memory, RemoteMemory)):
        return result  # Pure core/runtime fixtures have no persistent job ledger.
    from .training_checkpoint import training_runs, is_native_mlx_run
    contexts, ids = {}, {}
    for row in result:
        contexts.setdefault(normalized_context(row), []).append(row)
        ids[row["id"]] = normalized_context(row)
    def append(row, branch=False):
        row = copy.deepcopy(row)
        value = provenance(row)
        _validate_record(row, generation_only=value.get("retention_generation_only") is True)
        if row["purpose"] != "validation":
            raise ValueError("已采纳学习保护必须保持独立检验拆分")
        context = normalized_context(row)
        if row["id"] in ids and ids[row["id"]] != context:
            raise ValueError("已采纳学习保护同一ID上下文冲突")
        prior = contexts.get(context, [])
        check = validate_generation_check(value.get("generation_check"), default_expected=row["answer"])
        for original in prior:
            old_check = validate_generation_check(provenance(original).get("generation_check"),
                                                  default_expected=original["answer"])
            if original["answer"] and row["answer"] and original["answer"] != row["answer"]:
                raise ValueError("已采纳学习保护同一上下文答案冲突；纠错替代协议尚未实现")
            if row["answer"] and original["answer"]:
                compatible = (evaluate_generation(row["answer"], old_check, True)["passed"] and
                              evaluate_generation(original["answer"], check, True)["passed"])
            else:
                compatible = {k: v for k, v in old_check.items() if k != "max_new_tokens"} == {
                    k: v for k, v in check.items() if k != "max_new_tokens"}
            if not compatible:
                raise ValueError("已采纳学习保护同一上下文检查规则冲突")
        if prior:
            ids[row["id"]] = context
            return
        if branch:
            value["retention_scope"] = "accepted_branch"
            row["provenance"] = json.dumps(value, ensure_ascii=False)
        result.append(row)
        contexts[context], ids[row["id"]] = [row], context
    if snapshot is None:
        # These are labelled old capabilities, and still must pass a real OFF baseline.
        session = getattr(runtime, "training_session", None)
        replay_rows = (session.body["records"]["protection"] if session and session.body["kind"] == "learning"
                       else memory.samples("validation"))
        for row in replay_rows:
            value = provenance(row)
            if (eligible_learning_sample(row) and value.get("evolution_role") == "replay"
                    and value.get("retention_scope") != "accepted_branch"
                    and value.get("baseline_preservation_eligible") is not False):
                append(row)
    for run in reversed(training_runs(memory, runtime.identity)):
        if (is_native_mlx_run(run) or run["state"] != "committed" or run["kind"] != "learning" or
                (run.get("scope") == "research" and run.get("product_accepted") is not True)):
            continue
        records, binding = run.get("records"), run.get("binding", {})
        if (not isinstance(records, dict) or set(records) != {"train", "valid", "protection"}
                or binding.get("identity") != runtime.identity or
                hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True,
                    allow_nan=False).encode()).hexdigest() != binding.get("records_sha256")
                or run.get("commit", {}).get("report", {}).get("promoted") is not True):
            raise ValueError("已采纳学习保护清单不完整或内容SHA不符")
        for row in records["valid"] + records["protection"]:
            append(row, branch=True)
    return result


def validate_retention_conflicts(snapshot, records):
    if snapshot is None:
        return
    protected = [row for rows in snapshot["groups"].values() for row in rows]
    by_id = {row["id"]: row for row in protected}
    by_context = {normalized_context(row): row for row in protected}
    regressions = {}
    for row in snapshot["regression_validation"]:
        regressions.setdefault(normalized_context(row), []).append(row)
    for row in records:
        context = normalized_context(row)
        prior = by_id.get(row.get("id")) or by_context.get(context)
        if prior and (normalized_context(prior) != context or prior["answer"] != row.get("answer")
                      or prior["purpose"] != row.get("purpose")):
            raise ValueError("当前材料与已采纳知识保护的答案或拆分冲突；纠错替代协议尚未实现")
        for regression in regressions.get(context, []):
            check = validate_generation_check(provenance(regression).get("generation_check"),
                                              default_expected=regression["answer"])
            if not evaluate_generation(row.get("answer", ""), check, True)["passed"]:
                raise ValueError("当前材料与已采纳知识保护的生成保护规则冲突；纠错替代协议尚未实现")


def consolidation_groups(runtime, *, daily_only=False):
    snapshot = read_core_retention(runtime)
    current = [row for row in runtime.memory.samples()
               if eligible_learning_sample(row) and provenance(row).get("evolution_role") in {"new", "replay"}
               and row.get("purpose") in {"training", "validation"}
               and not (provenance(row).get("evolution_role") == "replay" and row["purpose"] == "validation"
                        and provenance(row).get("baseline_preservation_eligible") is False)]
    if daily_only:
        current = [row for row in current if provenance(row).get("evolution_role") == "replay"
                   or daily_learning_source(row)]
    validate_retention_conflicts(snapshot, current)
    groups = {key: [] for key in GROUP_KEYS}
    seen, ids, families = {}, {}, {}
    protected_ids, protected_contexts = set(), set()
    retained = []
    if snapshot:
        for records in snapshot["groups"].values():
            for original in records:
                row = copy.deepcopy(original)
                value = provenance(row)
                value.update(evolution_role="replay", retention_from_version=snapshot["version"])
                row["provenance"] = json.dumps(value, ensure_ascii=False)
                retained.append(row)
                protected_ids.add(row["id"])
                protected_contexts.add(normalized_context(row))
    for is_retained, row in [(True, row) for row in retained] + [(False, row) for row in current]:
        value = provenance(row)
        role, purpose = value.get("evolution_role"), row.get("purpose")
        if (role not in {"new", "replay"} or purpose not in {"training", "validation"}
                or not eligible_learning_sample(row)
                or (role == "replay" and purpose == "validation" and value.get("baseline_preservation_eligible") is False)):
            continue
        context = normalized_context(row)
        if not is_retained and (row["id"] in protected_ids or context in protected_contexts):
            continue  # Accepted old NEW is replay; repeating it cannot unlock a new evolution.
        if row["id"] in ids and normalized_context(ids[row["id"]]) != context:
            raise ValueError("核心材料同一ID对应不同上下文")
        if context in seen:
            prior = seen[context]
            if prior["purpose"] != purpose:
                raise ValueError("核心训练与独立检验出现相同上下文，不能进化")
            if prior["answer"] != row["answer"] or provenance(prior).get("evolution_role") != role:
                raise ValueError("核心材料对同一上下文有冲突答案或角色")
            continue
        _validate_record(row)
        seen[context], ids[row["id"]] = row, row
        family = value.get("split_family")
        if family:
            if not isinstance(family, str) or (family in families and families[family] != purpose):
                raise ValueError("核心训练与独立检验出现相同模板家族，不能进化")
            families[family] = purpose
        groups[role + "_" + purpose].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["id"])
    return groups
