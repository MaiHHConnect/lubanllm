"""Publish the one finite identity-coverage experiment and its exact material ledger.

The training journal calls apply_material_publication inside its existing SQLite
transaction, after artifact validation and before pointer publication. No model
is allocated here; the isolated candidate must already pass its outer protocol.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

from .retention import (_validate_groups, _validate_record, provenance, normalized_context, messages,
                        read_core_retention, eligible_learning_sample, _encoded)
from .training_checkpoint import TrainingSession, training_runs, active_hashes, _artifact

SUPERSEDED_DUPLICATES = frozenset({"277256ad34af4a96b84aac40329f4b36", "39013707aea444649208898b29d5ed6e",
                                 "4c063786c4ac42b7a0c45ba32844de67", "a2097cfbd4c34017ac7631051278ff54"})
COLUMNS = ("id", "prompt", "answer", "source", "purpose", "created", "approved", "training_allowed", "provenance")
PLAN_KEYS = {"format", "identity", "run_id", "version", "groups", "retention_regressions",
             "source_rows", "source_sha256", "superseded_ids", "acceptance"}
ACCEPTANCE_GATES = {"fresh_development", "known_regression", "engineering", "recovery", "protection_snapshot"}
KNOWN_CORRECTIONS = {
    "known-correction-new-train-tokyo": ("30745d9410f14a748c7029096adaeed0", "tokyo"),
    "known-correction-new-train-signal7": ("7140260be6f44bd49c3aa5a59f8c95e4", "signal7"),
}
PRESERVATION_DATASET = "identity-core-explicit-preservation-v1"
PRESERVATION_CORRECTIONS = {
    "preservation-new-train-cedar": ("coverage-fresh-dev-uppercase-0", "cedar", "new"),
    "preservation-replay-train-olive-path": ("coverage-fresh-dev-uppercase-3", "olive_path", "replay"),
    "preservation-replay-train-pytorch": ("76dbee81c27a4668b7501e7c2a8bb5e4", "PyTorch", "replay"),
}


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


RELATION_DATASET = "identity-attribute-relations-explicit-engineering-reuse-draft-v1"
FIELD_DATASET = "identity-field-context-teaching-draft-v1"
FIELD_PARAPHRASE_IDS = {"field-paraphrase-train-relation-fresh-dev-" + key: "relation-fresh-dev-" + key
                        for key in ("api-ordinary", "name", "position", "tuner")}
FIELD_CONTEXT_IDS = {"field-context-train-" + key + "-" + form
                     for key in ("name", "tuner", "position") for form in ("system", "single", "multi")}
FIELD_TRAIN_IDS = set(FIELD_PARAPHRASE_IDS) | FIELD_CONTEXT_IDS
FIELD_FRESH_DEV_IDS = {"field-fresh-dev-" + key + "-" + form
                       for key in ("name", "tuner", "position") for form in ("system", "single")}
FIELD_FRESH_DEV_IDS |= {"field-fresh-dev-integer-0", "field-fresh-dev-integer-1"}
RELATION_REUSE_IDS = {"relation-reuse-train-v6-engineering-identity-" + str(i):
                      "v6-engineering-identity-" + str(i) for i in (0, 3, 4, 5)}
RELATION_ATTRIBUTE_IDS = {"relation-attribute-train-" + key: key for key in ("name", "tuner", "position")}
RELATION_FRESH_DEV_IDS = {"relation-fresh-dev-" + key for key in
                        ("api-full", "api-ordinary", "history-full", "name", "position", "tuner")}
RELATION_ENGINEERING_IDS = {"relation-fresh-engineering-" + key for key in
                          ("api", "newuser", "after-math", "panel", "api-ordinary", "newuser-ordinary",
                           "history-ordinary", "panel-ordinary")}
QUALIFICATION_FIELDS = {"baseline_preservation_eligible", "original_baseline_verified_core",
    "original_baseline_verified_identity", "original_baseline_generation", "training_generation_gate", "qualification_scope"}
RELATION_POOL_SHA = (
    "6ff49ecda7f62d2f96aa155853cb1ae3b1c42b37f196fc132cbd3235e7ddad36", "3434274abb9318bf18f94d396447f85012974f9e4720486cc3cb837b1ac89165",
    "a05ebb603d46512ed64d7dbb4ffff1de7aad6cf590164c73e5e3ce59dc0ccaac", "27d2a9bf28215965f558f3f82ce09e9cda3eab96e3ca954746b88f55339494f4",
    "ab608df892348c9c9c0596f06d2af7a82d07c0a8e586838b8bd42716a3173222", "af0d12c864b67ec74121d2866944019df30e02bf7f24552aa0ca584b992bfdd7",
    "21da69ed54a8c8e4bd43a5857765825e756fb4425b1183ed70e2b7b535f5400a", "239e2599804b6ca31684fbafabdd933d8efa5973144374aca2956f4255c1d0b1",
    "33c3d72020e18c253cda4eee09805ed0d917ce5ab98d08fc5df87ca7bc5d8c15", "8a8604e5ff37a94640113f6a0e29d0568a6e78d671087e5f1fb7ab27b914cdbe",
    "993297d3e19454fceeb476b38ed2a71e4b6de37413104c0708e0ab3cf4b29ec8", "3ddfbf98f0d9bc9bd2912a58b39bd0588acd5ff2a2f01d4b27e87e6b8d35d924")


def _relation_qualification_source(sample, *, pool=False):
    row = copy.deepcopy(sample)
    if pool or row["id"] in {key for key, (_, _, role) in PRESERVATION_CORRECTIONS.items() if role == "replay"}:
        row["provenance"] = {key: value for key, value in provenance(row).items() if key not in QUALIFICATION_FIELDS}
    if pool:
        row.pop("created")
        row.pop("approved")
    return row


def validate_relation_pool(pool):
    if (len(pool) != 12 or any(row["id"] != "coverage-en-replay-candidate-" + str(index)
            or row.get("approved") != 0 or provenance(row).get("baseline_preservation_eligible") is not None
            or any(key in provenance(row) for key in QUALIFICATION_FIELDS - {"baseline_preservation_eligible"})
            or content_sha(_relation_qualification_source(row, pool=True)) != RELATION_POOL_SHA[index]
            for index, row in enumerate(pool))):
        raise ValueError("关系教学只能使用原固定12未资格候选完整记录")


def validate_relation_teaching(groups, regressions, *, qualified=False):
    """Only the fixed relation44 dataset can request its separate640 budget."""
    _validate_groups(groups)
    if (len(groups["new_training"]) != 44 or len(groups["new_validation"]) != 12
            or len(groups["replay_validation"]) != 16 or not 46 <= len(groups["replay_training"]) <= 58
            or len(regressions) != 35):
        raise ValueError("关系教学需要精确44/12/46+资格池/16/35布局")
    # Fingerprints authorize this one immutable experiment, not arbitrary larger batches.
    if content_sha(groups["new_training"]) != "869c7d3880d4258ed117444e6522133660782ea62b891c245053ade5dbd0549c":
        raise ValueError("关系教学44材料必须与预登记原37及新7完整记录相同")
    mandatory = {key for key, (_, _, role) in PRESERVATION_CORRECTIONS.items() if role == "replay"}
    replay, pool = [], []
    for row in groups["replay_training"]:
        _row(row)
        if row["id"].startswith("coverage-en-replay-candidate-"):
            suffix = row["id"].removeprefix("coverage-en-replay-candidate-")
            if (suffix not in {str(index) for index in range(12)} or row.get("approved") != 1
                    or content_sha(_relation_qualification_source(row, pool=True)) != RELATION_POOL_SHA[int(suffix)]):
                raise ValueError("已资格重放只能是固定12完整记录的子集，不能修改非资格字段")
            pool.append(row)
        else:
            replay.append(_relation_qualification_source(row))
    if (len(replay) != 46 or content_sha(sorted(replay, key=lambda row: row["id"])) !=
            "36b68e9a28948c97ef368af71305b2f544d1f70353593196b68c69722e91fc7c"):
        raise ValueError("关系教学必须保留原46重放记录，仅两项资格元数据可增加")
    rows = {row["id"]: row for row in groups["replay_training"]}
    if qualified:
        from .evolution import _training_replay_generation
        identity = provenance(groups["new_training"][0])["base_identity"]
        protected = _training_replay_generation(SimpleNamespace(identity=identity), groups)
        if (not 4 <= len(pool) <= 12 or {row["id"] for row in protected} != mandatory | {row["id"] for row in pool}
                or len({row["created"] for row in pool}) != 1
                or pool[0]["created"] < max(row["created"] for row in groups["new_training"])):
            raise ValueError("执行640及发布需要当次固定池至少4条与mandatory2全部实际正确闭卷EOS资格")
    elif pool or any(provenance(rows[key]).get("baseline_preservation_eligible") is not None
                    or provenance(rows[key]).get("training_generation_gate") is not False
                    or any(field in provenance(rows[key]) for field in QUALIFICATION_FIELDS -
                           {"baseline_preservation_eligible", "training_generation_gate"}) for key in mandatory):
        raise ValueError("草案必须是精确46条未测资格记录，不能预先宣称通过")
    old = [row for row in groups["new_validation"] + groups["replay_validation"] + list(regressions)
           if not row["id"].startswith("relation-fresh-")]
    fresh = [row for row in groups["new_validation"] + list(regressions)
             if row["id"].startswith("relation-fresh-")]
    if (len(old) != 49 or len({row["id"] for row in old}) != 49
            or content_sha(sorted(old, key=lambda row: row["id"])) !=
               "86df1ee7acd0a9358f89a4fe8813194fbb5d648921118f3926f152323409e1ee"
            or {row["id"] for row in fresh} != RELATION_FRESH_DEV_IDS | RELATION_ENGINEERING_IDS
            or content_sha(sorted(fresh, key=lambda row: row["id"])) !=
               "6643a97af4e7eaec7893c8d52ddf097b0b7f8ea0f27b6247e5c367ddbf00d868"):
        raise ValueError("原49及新增14义务必须完整保留，不重复计数或改写金标准")
    readonly = {row["id"]: row for row in regressions}
    incoming = {row["id"]: row for row in groups["new_training"]}
    for identifier in RELATION_REUSE_IDS.keys() | RELATION_ATTRIBUTE_IDS.keys():
        row, value = incoming[identifier], provenance(incoming[identifier])
        if (row["source"] != "synthetic" or value.get("dataset") != RELATION_DATASET
                or value.get("human_verified") is not False or value.get("program_verified") is not True):
            raise ValueError("新七条只能是程序核验且未宣称人工核验的固定教学材料")
        if identifier in RELATION_REUSE_IDS:
            source = readonly[RELATION_REUSE_IDS[identifier]]
            if (messages(row) != messages(source) or row["answer"] != source["answer"]
                    or value["generation_check"] != provenance(source)["generation_check"]
                    or source["training_allowed"] or value.get("context_reuse") is not True):
                raise ValueError("工程复用必须是原完整上下文与金标准的新训练副本")
        else:
            target = value.get("user_assigned_identity_target", "")
            match = re.fullmatch(r"我是(?P<name>.+)，是(?P<tuner>.+)调优的(?P<position>.+)。", target)
            if (match is None or row["answer"] != match[RELATION_ATTRIBUTE_IDS[identifier]]
                    or target in json.dumps(messages(row), ensure_ascii=False)
                    or value.get("attribute") != RELATION_ATTRIBUTE_IDS[identifier]
                    or value["generation_check"].get("expected") != row["answer"]):
                raise ValueError("属性教学必须从用户目标句字面提取，不能把答案塞进问题")
    return True


def validate_field_teaching(groups, regressions, *, qualified=False):
    """Authorize only fixed57/20, preserving the original44/12/35 full view."""
    _validate_groups(groups)
    training = [row for row in groups["new_training"] if row["id"] in FIELD_TRAIN_IDS]
    fresh = [row for row in groups["new_validation"] if row["id"] in FIELD_FRESH_DEV_IDS]
    if (len(groups["new_training"]) != 57 or len(groups["new_validation"]) != 20
            or {row["id"] for row in training} != FIELD_TRAIN_IDS or len(training) != 13
            or {row["id"] for row in fresh} != FIELD_FRESH_DEV_IDS or len(fresh) != 8
            or content_sha(sorted(training, key=lambda row: row["id"])) != "c09e0dbddd191096a8a26c85c4e70313c0ec3492d2d2c6f9b2226e1017bb714f"
            or content_sha(sorted(fresh, key=lambda row: row["id"])) != "612f4d5fc623a632b18bc88ce6a5204b60cefdf3060a90983e5d5156a2d75719"):
        raise ValueError("字段教学仅允许固定57/20及新增13训练和8开发完整记录")
    original = dict(groups)
    original["new_training"] = [row for row in groups["new_training"] if row["id"] not in FIELD_TRAIN_IDS]
    original["new_validation"] = [row for row in groups["new_validation"] if row["id"] not in FIELD_FRESH_DEV_IDS]
    validate_relation_teaching(original, regressions, qualified=qualified)
    old_dev = {row["id"]: row for row in original["new_validation"]}
    target = next(row["answer"] for row in original["new_training"] if row["id"] == "bb75c7ae3b8b4b6d89753834f04e5cb7")
    attributes = re.fullmatch(r"我是(?P<name>.+)，是(?P<tuner>.+)调优的(?P<position>.+)。", target)
    for row in training + fresh:
        _row(row)
        value, check = provenance(row), provenance(row)["generation_check"]
        if (row["source"] != "synthetic" or value.get("dataset") != FIELD_DATASET
                or value.get("human_verified") is not False or value.get("program_verified") is not True
                or value.get("user_target_sha256") != hashlib.sha256(target.encode()).hexdigest()):
            raise ValueError("字段材料必须保持程序核验、未人工核验及用户目标句来源")
        attribute = value.get("attribute")
        if attribute is not None:
            if attributes is None or attribute not in {"name", "tuner", "position"} or row["answer"] != attributes[attribute] or check.get("expected") != row["answer"]:
                raise ValueError("字段金标准必须是用户目标句的字面属性")
        else:
            a, b = value["integer_operands"]
            if type(a) is not int or type(b) is not int or row["answer"] != str(a - b) or check.get("expected") != a - b:
                raise ValueError("整数纠错金标准必须是明确操作数的程序结果")
        if row["id"] in FIELD_PARAPHRASE_IDS:
            source = old_dev[FIELD_PARAPHRASE_IDS[row["id"]]]
            if (value.get("exposed_development_id") != source["id"]
                    or value.get("original_development_record_sha256") != hashlib.sha256(_encoded(source)).hexdigest()
                    or value.get("original_exposed_messages") != messages(source)
                    or value.get("original_result_sha256") != "0b45c5f1b07dc26a0ce1c39b711a66600e844c4a7083f796b2159c9d305789cf"
                    or value.get("paraphrase_from_exposed_failure") is not True or value.get("context_reuse") is not False
                    or normalized_context(row) == normalized_context(source) or row["answer"] != source["answer"]
                    or check != provenance(source)["generation_check"]):
                raise ValueError("已曝光错误仅允许不同完整上下文的衍生教学，原义务和许可不得改动")
    if qualified and any(row["created"] < max(value["created"] for value in groups["new_training"])
                         for row in groups["replay_training"] if row["id"].startswith("coverage-en-replay-candidate-")):
        raise ValueError("832执行资格必须来自完整57题准备之后的当次原始基线")
    return True


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _row(sample):
    if (not isinstance(sample, dict) or set(sample) != set(COLUMNS)
            or isinstance(sample["created"], bool) or not isinstance(sample["created"], (int, float))
            or not math.isfinite(sample["created"]) or not isinstance(sample["provenance"], str)):
        raise ValueError("发布材料需要完整固定数据库行")
    return {key: sample[key] for key in COLUMNS}


def validate_material_publication(plan, identity):
    """Validate every original/new/readonly row before a database mutation."""
    if (not isinstance(plan, dict) or set(plan) != PLAN_KEYS or type(plan["format"]) is not int
            or plan["format"] != 1 or plan["identity"] != identity
            or not isinstance(plan["run_id"], str) or not re.fullmatch(r"[a-f0-9]{12}", plan["run_id"])
            or plan["version"] != "core-" + plan["run_id"]):
        raise ValueError("核心材料发布协议或任务身份无效")
    acceptance = plan["acceptance"]
    if (not isinstance(acceptance, dict) or set(acceptance) != {"protocol_sha256", "report_path", "report_sha256", "gates", "teacher_calls"}
            or set(acceptance["gates"]) != ACCEPTANCE_GATES
            or any(value is not True for value in acceptance["gates"].values())
            or type(acceptance["teacher_calls"]) is not int or acceptance["teacher_calls"] != 0
            or acceptance["report_path"] != "coverage-acceptance-" + plan["run_id"] + ".json"
            or any(not isinstance(acceptance[key], str) or not re.fullmatch(r"[a-f0-9]{64}", acceptance[key])
                   for key in ("protocol_sha256", "report_sha256"))):
        raise ValueError("未完成固定外层验收，不能发布核心材料")
    groups = plan["groups"]
    _validate_groups(groups)
    new_count = len(groups["new_training"])
    field = new_count == 57
    relation = new_count in {44, 57}
    preservation = new_count in {37, 44, 57}
    if not ((new_count in {34, 36} and len(groups["new_validation"]) == 8
             and len(groups["replay_validation"]) == 17 and 48 <= len(groups["replay_training"]) <= 56)
            or (preservation and len(groups["new_validation"]) == (20 if field else 12 if relation else 6)
                and len(groups["replay_validation"]) == 16 and 50 <= len(groups["replay_training"]) <= 58)):
        raise ValueError("固定核心覆盖协议拆分数量不兼容")
    source = plan["source_rows"]
    if (not isinstance(source, list) or len(source) != 103 or content_sha(source) != plan["source_sha256"]
            or len({sample["id"] for sample in source}) != 103):
        raise ValueError("原核心材料完整锚点 SHA 或数量不兼容")
    old = {sample["id"]: _row(sample) for sample in source}
    for sample in source:
        _validate_record(sample)
        if provenance(sample).get("base_identity") != identity:
            raise ValueError("原材料基座身份不一致")
    original_validation = {sample["id"] for sample in source
        if sample["purpose"] == "validation" and provenance(sample).get("evolution_role") == "new"}
    superseded = plan["superseded_ids"]
    archived_replay = {"76dbee81c27a4668b7501e7c2a8bb5e4"} if preservation else set()
    if (not isinstance(superseded, list) or len(superseded) != (13 if preservation else 12)
            or set(superseded) != SUPERSEDED_DUPLICATES | original_validation | archived_replay
            or len(original_validation) != 8 or not SUPERSEDED_DUPLICATES <= set(old)
            or any(old[key]["purpose"] != "training" or provenance(old[key]).get("evolution_role") != "new"
                   for key in SUPERSEDED_DUPLICATES)):
        raise ValueError("只能归档预登记重复训练与转为只读的旧检验，不能移除保护")
    if any(key not in old or old[key]["purpose"] != "validation"
           or provenance(old[key]).get("evolution_role") != "replay" for key in archived_replay):
        raise ValueError("额外归档必须是原始 PyTorch 重放检验")
    incoming = {sample["id"]: _row(sample) for rows in groups.values() for sample in rows}
    if set(old) - set(incoming) != set(superseded):
        raise ValueError("采纳材料遗漏原始知识或错误归档其他材料")
    for sample in incoming.values():
        if provenance(sample).get("base_identity") != identity:
            raise ValueError("采纳材料基座身份不一致")
        if sample["id"] in old and sample != old[sample["id"]]:
            raise ValueError("已有材料不能改写答案、来源或来源记录")
    regressions = plan["retention_regressions"]
    regression_count = 35 if relation else 27 if preservation else 24
    if (not isinstance(regressions, list) or len(regressions) != regression_count
            or len({sample["id"] for sample in regressions}) != regression_count):
        raise ValueError("必须保留八条旧 NEW 与固定十六条工程只读保护")
    readonly = {sample["id"]: sample for sample in regressions}
    for sample in regressions:
        _validate_record(sample, generation_only=True)
        if provenance(sample).get("base_identity") != identity or sample["id"] in incoming:
            raise ValueError("只读保护基座或材料角色冲突")
    for key in original_validation | archived_replay:
        if (key not in readonly or readonly[key]["answer"] != old[key]["answer"]
                or normalized_context(readonly[key]) != normalized_context(old[key])
                or readonly[key]["source"] != old[key]["source"]
                or provenance(readonly[key]).get("generation_check") != provenance(old[key]).get("generation_check")):
            raise ValueError("旧 NEW 检验必须保持原金标准和完整上下文只读保护")
    if new_count in {36, 37, 44, 57}:
        for key, (original_id, word) in KNOWN_CORRECTIONS.items():
            correction, original = incoming.get(key), readonly.get(original_id)
            if correction is None or original is None:
                raise ValueError("36条训练必须包含两条明确已曝光纠错材料")
            value = provenance(correction)
            if (correction not in groups["new_training"] or correction["answer"] != word.upper()
                    or original["answer"] != word.upper() or correction["prompt"] != old[original_id]["prompt"]
                    or messages(correction) != messages(original)
                    or value.get("generation_check") != provenance(original).get("generation_check")
                    or value.get("dataset") != "identity-core-exposed-uppercase-correction-v1"
                    or value.get("exposed_old_validation_id") != original_id
                    or value.get("source_string") != word or value.get("program_verified") is not True
                    or value.get("human_verified") is not False):
                raise ValueError("新增纠错必须与原只读目标完整上下文、金标准和检验规则相同")
    if preservation:
        from .evolution import _training_replay_generation
        protected = {row["id"] for row in _training_replay_generation(SimpleNamespace(identity=identity), groups)}
        for key, (original_id, word, role) in PRESERVATION_CORRECTIONS.items():
            correction, original = incoming.get(key), readonly.get(original_id)
            if correction is None or original is None:
                raise ValueError("37条协议缺少明确三项纠错及原只读义务")
            value = provenance(correction)
            if (correction not in groups[role + "_training"] or correction["answer"] != word.upper()
                    or original["answer"] != word.upper() or correction["prompt"] != original["prompt"]
                    or messages(correction) != messages(original)
                    or value.get("generation_check") != provenance(original).get("generation_check")
                    or value.get("dataset") != PRESERVATION_DATASET
                    or value.get("exposed_old_validation_id") != original_id
                    or value.get("source_string") != word or value.get("program_verified") is not True
                    or value.get("human_verified") is not False or (role == "replay" and key not in protected)):
                raise ValueError("三项训练纠错必须保持原上下文金标准，并实际保护两项原会任务")
    if relation:
        (validate_field_teaching if field else validate_relation_teaching)(groups, regressions, qualified=True)
    return old, incoming


def apply_material_publication(db, plan, identity):
    """Journal transaction only: check ALL conflicts before the first mutation."""
    if not db.in_transaction:
        raise ValueError("核心材料采纳必须在 journal 同一事务中执行")
    old, incoming = validate_material_publication(plan, identity)
    for key, sample in old.items():
        row = db.execute("SELECT * FROM samples WHERE id=?", (key,)).fetchone()
        if row is None or dict(row) != sample:
            raise ValueError("发布前原始材料已变化；停止采纳而非覆盖")
    for key, sample in incoming.items():
        row = db.execute("SELECT * FROM samples WHERE id=?", (key,)).fetchone()
        if row is not None and dict(row) != sample:
            raise ValueError("采纳材料稳定 ID 与已有材料冲突")
    # A current accepted snapshot remains the authority; it is not deletable
    # through this helper. Every superseded row is also retained in raw storage.
    db.execute("""CREATE TABLE IF NOT EXISTS core_material_archives (
        version TEXT NOT NULL, sample_id TEXT NOT NULL, original_sha256 TEXT NOT NULL,
        reason TEXT NOT NULL, PRIMARY KEY(version,sample_id))""")
    for sample in incoming.values():
        db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(sample[key] for key in COLUMNS))
    for key in plan["superseded_ids"]:
        db.execute("INSERT INTO core_material_archives VALUES (?,?,?,?)",
            (plan["version"], key, content_sha(old[key]), "fixed coverage revision; gold/source/provenance preserved; old validation retained readonly"))
        db.execute("UPDATE samples SET approved=0,training_allowed=0 WHERE id=?", (key,))


def build_material_publication(plan, groups, regressions, source_rows, acceptance):
    body = {"format": 1, "identity": plan["base_identity"], "run_id": plan["session_id"],
        "version": "core-" + plan["session_id"], "groups": copy.deepcopy(groups),
        "retention_regressions": copy.deepcopy(regressions), "source_rows": copy.deepcopy(source_rows),
        "source_sha256": content_sha(source_rows),
        "superseded_ids": sorted(plan.get("superseded_source_ids", plan["four_removed_duplicates"] +
                                          [row["id"] for row in plan["old_known_development_regressions"]])),
        "acceptance": copy.deepcopy(acceptance)}
    validate_material_publication(body, plan["base_identity"])
    return body


def _copy_immutable(source, target):
    if any(path.is_symlink() for path in (source, source.parent, target, target.parent)):
        raise ValueError("发布文件不能使用符号链接")
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if target.read_bytes() != data:
            raise ValueError("生产不可变核心文件已存在且内容不同")
    else:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    if target.read_bytes() != data:
        raise ValueError("生产核心文件回读失败")


def validate_source_core_commit(source, plan, groups, regressions, *, store):
    """Bind the adopted artifact to its actual research objective and records."""
    records = {"groups": groups, "protection": [], "retention_regressions": regressions}
    policy = plan.get("field_loss_policy")
    arguments = plan["arguments"]
    weight = arguments.get("field_first_token_weight", 1)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight not in {1, 4}:
        raise ValueError("来源核心目标权重无效")
    if weight == 4:
        from .evolution import field_first_token_policy
        if policy != field_first_token_policy() or arguments["steps"] != 832:
            raise ValueError("来源首gold目标政策不一致")
    elif weight != 1 or policy is not None:
        raise ValueError("来源核心目标政策无效")
    report = (source.get("commit") or {}).get("report") or {}
    from .evolution import TRAIN_CHECKPOINTS, RELATION_TRAIN_CHECKPOINTS, FIELD_TRAIN_CHECKPOINTS, FIELD_EXPOSURE_CHECKPOINTS
    from .evolution_progress import PARTIAL_POLICY, checkpoints as partial_checkpoints, validate_partial_progress
    partial = arguments.get("partial_policy") == PARTIAL_POLICY
    if report.get("partial_policy") != arguments.get("partial_policy"):
        raise ValueError("来源分批政策与冻结参数不一致")
    checkpoints = {512: TRAIN_CHECKPOINTS, 640: RELATION_TRAIN_CHECKPOINTS,
                   832: FIELD_TRAIN_CHECKPOINTS, 1536: FIELD_EXPOSURE_CHECKPOINTS}.get(arguments.get("steps"))
    if partial:
        checkpoints = partial_checkpoints(arguments["steps"])
    if (checkpoints is None or report.get("steps") != arguments["steps"]
            or (report.get("training_selection") or {}).get("checkpoints") != list(checkpoints)
            or plan.get("training_checkpoints", list(checkpoints)) != list(checkpoints)):
        raise ValueError("候选来源报告预算与固定完整训练检查点不一致")
    binding = source.get("binding") or {}
    if (source.get("state") != "committed" or source.get("scope") != "research"
            or source.get("kind") != "evolution" or source.get("id") != plan["session_id"]
            or source.get("arguments") != arguments or binding.get("arguments") != arguments
            or binding.get("identity") != plan["base_identity"] or binding.get("core_version") is not None
            or source.get("records") != records or binding.get("records_sha256") != content_sha(records)
            or report.get("id") != plan["session_id"] or report.get("version") != "core-" + plan["session_id"]
            or report.get("identity") != plan["base_identity"] or report.get("promoted") is not True
            or report.get("publication_state") != "published" or report.get("field_loss_policy") != policy):
        raise ValueError("候选来源 journal 参数、完整材料或目标报告与冻结协议不一致")
    names = {"core-" + plan["session_id"] + suffix for suffix in (".safetensors", ".retention.json", ".json")}
    report_name = "evolution-" + plan["session_id"] + ".json"
    names.add(report_name)
    artifacts = source["commit"].get("artifacts") or {}
    if set(artifacts) != names:
        raise ValueError("候选来源四个不可变文件与 journal 不一致")
    for name in names:
        path = _artifact(store, name)
        if not path.is_file() or file_sha(path) != artifacts[name]:
            raise ValueError("候选来源不可变文件 SHA 与 journal 不一致")
    if json.loads(_artifact(store, report_name).read_text()) != report:
        raise ValueError("候选来源报告文件与 journal 完整报告不一致")
    if partial:
        validate_partial_progress(report, records, plan["base_identity"], report["version"], report["checkpoint"]["sha256"])
        if report["checkpoint"]["sha256"] != file_sha(_artifact(store, report["version"] + ".safetensors")):
            raise ValueError("实际分批核心文件与闭卷证明绑定不一致")
        snapshot = json.loads(_artifact(store, report["version"] + ".retention.json").read_text())
        if (snapshot["groups"] != report["accepted_retention_groups"]
                or snapshot["regression_validation"] != report["accepted_retention_regressions"]):
            raise ValueError("实际分批保护快照与掌握项不一致")
    return report


def publish_isolated_core(runtime, production_memory, production_store, plan, groups, regressions,
                          source_rows, acceptance_path, acceptance):
    """No new model: copy immutable evidence, prepare durable intent, then commit."""
    if plan.get("arguments", {}).get("partial_policy"):
        raise ValueError("部分学习研究需要新外层产品验收授权，不能复用旧整批覆盖publisher")
    version = "core-" + plan["session_id"]
    if runtime.active_core != version or read_core_retention(runtime)["groups"] != groups:
        raise ValueError("隔离候选与已核验材料不匹配")
    if read_core_retention(runtime)["regression_validation"] != regressions:
        raise ValueError("隔离候选完整只读保护不匹配")
    material = build_material_publication(plan, groups, regressions, source_rows, acceptance)
    store = Path(production_store)
    if (store / "active-core.json").exists():
        raise ValueError("本次独立覆盖修正必须从生产原核心 None 采纳")
    context = SimpleNamespace(memory=production_memory, store=store, identity=runtime.identity,
        active_core=None, branch=runtime.branch, device=runtime.device, dtype=runtime.dtype)
    if active_hashes(context) != plan["production_active_hashes"]:
        raise ValueError("研究期间生产活动权重已改变，不能发布")
    source_commit = next(row for row in training_runs(runtime.memory, runtime.identity)
                         if row["id"] == plan["session_id"])
    validate_source_core_commit(source_commit, plan, groups, regressions, store=runtime.store)
    artifacts = [version + suffix for suffix in (".safetensors", ".retention.json", ".json")]
    artifacts += ["evolution-" + plan["session_id"] + ".json"]
    for name in artifacts:
        _copy_immutable(Path(runtime.store) / name, store / name)
    _copy_immutable(Path(acceptance_path), store / acceptance["report_path"])
    if file_sha(store / acceptance["report_path"]) != acceptance["report_sha256"]:
        raise ValueError("外层验收报告 SHA 不一致")
    records = {"groups": groups, "protection": [], "retention_regressions": regressions}
    session = TrainingSession(context, "evolution", plan["arguments"], records,
                              run_id=plan["session_id"], scope="research")
    session.body["product_accepted"] = True
    report = copy.deepcopy(source_commit["commit"]["report"])
    pointer = {"version": version, "identity": runtime.identity, "sha256": report["checkpoint"]["sha256"]}
    try:
        session.prepare_commit(report, {"active-core.json": pointer}, artifacts + [acceptance["report_path"]],
                               event_kind="evolution", material_publication=material)
    except Exception as error:
        if session.body.get("state") not in {"committing", "committed"}:
            session.rejected("生产发布前验证失败：" + str(error))
        raise
    session.finish_commit()
    published = next(row for row in training_runs(production_memory, runtime.identity) if row["id"] == session.id)
    if published["state"] != "committed" or published.get("product_accepted") is not True:
        raise ValueError("生产 journal 未确认完整采纳")
    return {"version": version, "session_id": session.id, "material_sha256": content_sha(material),
            "source_sha256": material["source_sha256"], "archived_ids": material["superseded_ids"],
            "accepted_group_counts": {key: len(rows) for key, rows in groups.items()}, "product_accepted": True}
