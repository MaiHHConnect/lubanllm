"""Continue the accepted core in an isolated Bun-owned store; never adopt it.

prepare is CPU/source-only. run requires the exact prepared SHA and an isolated
Bun RPC owner; only root executes it after releasing the product model lock.
Ordinary450 retains its original gates; explicit closed-parent protocols add complete450 rounds.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, closing
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import secrets
import select
import subprocess
import tempfile
import threading
import time


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = "evolution-goal-current-core-native-v1"
CONTINUATION_PROTOCOL = "evolution-goal-fixed450-rounds-v2"
CONTINUATION_PROTOCOL_V3 = "evolution-goal-fixed450-rounds-v3"
CONTINUATION_PROTOCOLS = {CONTINUATION_PROTOCOL, CONTINUATION_PROTOCOL_V3}
FRESH_PROTOCOL = "evolution-goal-prospective-fixed450-v1"
FRESH_RESUME_PROTOCOL = "evolution-goal-prospective-fixed450-v2"
FRESH_RESUME_PROTOCOL_V3 = "evolution-goal-prospective-fixed450-v3"
FRESH_RESUME_PROTOCOL_V4 = "evolution-goal-prospective-fixed450-v4"
DATA_PHASE = "evolution-goal-new-data-fixed450-v1"
FRESH_RESUME_PROTOCOLS = {FRESH_RESUME_PROTOCOL, FRESH_RESUME_PROTOCOL_V3, FRESH_RESUME_PROTOCOL_V4, DATA_PHASE}
FRESH_PROTOCOLS = {FRESH_PROTOCOL, *FRESH_RESUME_PROTOCOLS}
ARGUMENTS = {"steps": 450, "learning_rate": 1e-5,
             "distillation_weight": 5, "preserve_replay_subspace": True}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plain(path):
    path = Path(path).absolute()
    need(not any(p.is_symlink() for p in (path, *path.parents)), "路径不能包含符号链接")
    return path


def read(path):
    path = plain(path)
    need(path.is_file() and path.stat().st_size <= 32 * 1024**2, "需要有界普通JSON文件")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "JSON字段重复")
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        need(math.isfinite(result), "JSON不能包含非有限数字")
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=lambda _: need(False, "JSON不能包含非有限数字"))


def write_new(path, value):
    path = plain(path)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def source_hashes(*, continuation=False, fresh=False):
    names = ["scripts/train_evolution_goal.py", "scripts/goal_data_service.ts", "native/Training.cpp",
             "src/luban/bun_bridge.py", "backend/memory.ts", "backend/settings.ts",
             "backend/server.ts", "backend/wire.ts", "backend/home.ts", "backend/agreement.ts",
             "src/luban/upgrades.py", "src/luban/model_packages.py"]
    names += ["src/luban/" + name + ".py" for name in (
        "runtime", "evolution", "retention", "generation_checks", "training_checkpoint",
        "native_core_training", "luban_training", "foundation", "product_inference",
        "backbone", "coldpaging", "dynamic", "process_lock", "context")]
    names += [name for name in ("native/luban-training", "build/luban-training", "build/native/luban-training",
                               "native/luban-worker", "native/luban-worker.json",
                               "build/native/luban-worker", "build/native/luban-worker.json")
              if (REPO / name).is_file()]
    if continuation:
        from luban.training_checkpoint import implementation_hashes
        names.extend("src/luban/" + name for name in implementation_hashes())
        names.append("src/luban/goal_sequence.py")
    if fresh:
        from luban.training_checkpoint import implementation_hashes
        names.extend("src/luban/" + name for name in implementation_hashes())
        names.extend(("src/luban/goal_fresh.py", "src/luban/goal_topics.py"))
    return {name: sha(REPO / name) for name in names}


def materials(payload, snapshot, *, fresh=False, data_phase=False):
    from luban.retention import (_validate_groups, eligible_learning_sample,
                                provenance, validate_retention_conflicts)
    rows = payload.get("samples")
    need(isinstance(rows, list) and len(rows) == (68 if data_phase else 32 if fresh else 20), "冻结新材料计数不同")
    train = [copy.deepcopy(row) for row in rows if row.get("purpose") == "training"]
    valid = [copy.deepcopy(row) for row in rows if row.get("purpose") == "validation"]
    need((len(train), len(valid)) == ((56, 12) if data_phase else (24, 8) if fresh else (16, 4)), "冻结TRAIN/VALID拆分不同")
    for row in train + valid:
        value = provenance(row)
        need(row.get("approved") is True and row.get("training_allowed") is True
             and eligible_learning_sample(row) and value.get("draft") is not True
             and value.get("teacher_generated") is False and value.get("teacher_calls") == 0
             and value.get("evolution_role") == "new", "材料仍是草稿或缺少真实核验/训练许可")
        check = value.get("generation_check", {})
        valid_rule = (check.get("type") == "semantic" and check.get("validator") == "goal_topics"
                      and check.get("expected") in {"framework", "fast_slow", "own_mastery", "policy"}) if fresh else (check.get("type") == "exact" and check.get("expected") == row["answer"])
        need(valid_rule and isinstance(value.get("split_family"), str) and value["split_family"],
             "本次材料须预先冻结完整答案及独立模板家族")
    validate_retention_conflicts(snapshot, train + valid)
    groups = {"new_training": train, "new_validation": valid,
              "replay_training": [], "replay_validation": []}
    for old_rows in snapshot["groups"].values():
        for original in old_rows:
            row = copy.deepcopy(original)
            value = provenance(row)
            value.update(evolution_role="replay", retention_from_version=snapshot["version"])
            row["provenance"] = json.dumps(value, ensure_ascii=False)
            # Keep historical original-core qualification unchanged. The shared
            # consolidate gate also reruns these rows on the actual current core.
            groups["replay_" + row["purpose"]].append(row)
    _validate_groups(groups)
    need(all(groups.values()), "需要全部旧核心训练和独立检验材料")
    return groups


def prepared(args):
    from luban.retention import read_retention_file, _validate_record, provenance, normalized_context
    formal_path = plain(args.protocol)
    formal = read(formal_path)
    fresh = formal.get("protocol") == FRESH_PROTOCOL
    fresh_checked = None
    if fresh:
        from luban import goal_fresh
        fresh_checked = goal_fresh.load_verified(formal_path)
    formal_root = formal_path.parent
    need(formal.get("status") == "PRE_REGISTERED_NOT_TRAINED"
         and formal.get("weights_updated") is False and formal.get("production_written") is False
         and formal.get("gpu_used") is False and formal.get("teacher_calls") == 0,
         "需要未训练的完整正式预登记协议")
    formal_files = {}
    for key in ("new_material", "old_retention_complete", "old_training_original", "old_protection", "acceptance"):
        filename = formal[key]
        need(isinstance(filename, str) and Path(filename).name == filename, "协议成员路径无效")
        formal_files[key] = {"path": str(formal_root / filename), "sha256": sha(plain(formal_root / filename))}
    store, model = plain(args.source_store), plain(args.model_root)
    output = plain(args.output)
    need(store.name == "luban0-v1" and store.parent.name == "modules", "来源需要正式核心store")
    need(output.parent.is_dir() and not output.exists()
         and not output.is_relative_to(store) and not store.is_relative_to(output)
         and not output.is_relative_to(model) and not model.is_relative_to(output), "输出必须是模型外独立新目录")
    need(not (store / "active.json").exists() and not (store / "active-router.json").exists(),
         "本次默认核心续训不能忽略现有分支或活动快脑；请先制定它们的完整保护协议")
    pointer = read(store / "active-core.json")
    version, identity = pointer.get("version"), pointer.get("identity")
    need(isinstance(version, str) and re.fullmatch(r"core-[a-f0-9]{12}", version)
         and isinstance(identity, str) and re.fullmatch(r"[a-f0-9]{64}", identity), "正式核心身份无效")
    metadata = read(store / (version + ".json"))
    report_name = "evolution-" + version.removeprefix("core-") + ".json"
    names = ["active-core.json", "graph-verification.json", version + ".json",
             version + ".safetensors", version + ".retention.json", report_name]
    hashes = {name: sha(plain(store / name)) for name in names}
    need(metadata.get("sha256") == hashes[version + ".safetensors"] == pointer.get("sha256")
         and metadata.get("identity") == identity and metadata.get("version") == version
         and metadata.get("report") == report_name
         and metadata.get("retention_sha256") == hashes[version + ".retention.json"], "来源核心成员绑定不一致")
    proof = read(store / "graph-verification.json")
    need(proof.get("passed") is True and proof.get("identity") == identity, "来源真实模型图未通过等价验证")
    snapshot = read_retention_file(store / (version + ".retention.json"), identity, version)
    need(pointer == formal["source_core"] and str(store.parent.parent) == formal["storage_root"]
         and hashes[version + ".retention.json"] == formal["source_retention_sha256"]
         == formal_files["old_retention_complete"]["sha256"], "正式协议未绑定当前完整来源核心")
    material_path = formal_files["new_material"]["path"]
    source_material = read(material_path)
    payload = {"samples": source_material["TRAIN"] + source_material["VALID"]}
    groups = materials(payload, snapshot, fresh=fresh)
    old_train = [row for rows in snapshot["groups"].values() for row in rows if row["purpose"] == "training"]
    protection = snapshot["groups"]["new_validation"] + snapshot["groups"]["replay_validation"] + snapshot["regression_validation"]
    need(old_train == read(formal_files["old_training_original"]["path"])
         and protection == read(formal_files["old_protection"]["path"])
         and len(old_train) == 83 and len(protection) == 89, "全部83旧TRAIN与89保护记录必须原文保持")
    acceptance = read(formal_files["acceptance"]["path"])
    need(isinstance(acceptance, list) and len(acceptance) == 8, "需要完整8条只读最终验收")
    contexts = {normalized_context(row) for rows in groups.values() for row in rows}
    for row in acceptance:
        _validate_record(row, generation_only=True)
        value = provenance(row)
        check = value["generation_check"]
        valid_rule = (check["type"] == "semantic" and check.get("validator") == "goal_topics"
                      and check.get("expected") in {"framework", "fast_slow", "own_mastery", "policy"}) if fresh else (check["type"] == "exact" and check["expected"] == row["answer"])
        need(normalized_context(row) not in contexts and valid_rule, "最终验收上下文泄漏或完整答案规则变化")
        contexts.add(normalized_context(row))
    upstream = read(model / "upstream-file-manifest.json")
    records = sorted((f["Path"], f["Sha256"]) for f in upstream["Data"]["Files"]
                     if f["Path"].endswith(".safetensors") or f["Path"] == "config.json")
    need(hashlib.sha256(json.dumps(records).encode()).hexdigest() == identity, "模型目录与当前核心基座不同")
    plan = {"protocol": PROTOCOL, "arguments": ARGUMENTS, "seed": 42,
        "source_store": str(store), "source_files": hashes, "identity": identity,
        "starting_core": version, "model_root": str(model),
        "model_manifest_sha256": sha(model / "upstream-file-manifest.json"),
        "model_config_sha256": sha(model / "config.json"),
        "formal_protocol": str(formal_path), "formal_protocol_sha256": sha(formal_path),
        "formal_files": formal_files, "materials": material_path, "materials_sha256": sha(material_path),
        "groups": groups, "protection": protection, "old_training_original": old_train,
        "acceptance": acceptance, "retention_regressions": acceptance, "source_sha256": source_hashes(fresh=fresh),
        "output": str(output), "home": str(output / "isolated-home"),
        "storage": str(output / "isolated-storage"),
        "scope": "isolated product-graph core training; no production adoption or model-directory writes",
        "budget_policy": "450 maximum; unchanged first-two-consecutive TRAIN checks; VALID never used to select another checkpoint",
        "teacher_api_calls": 0, "model_started": False}
    if fresh:
        plan.update(protocol=FRESH_PROTOCOL, arguments=goal_fresh.arguments(fresh_checked),
            goal_sequence_protocol={"path": str(formal_path), "sha256": sha(formal_path)},
            goal_sequence=goal_fresh.metadata(fresh_checked),
            budget_policy="fresh B45 max2700; complete450 TRAIN-only same-W double rounds; once fresh heldout then stop")
        need(plan["source_sha256"] == fresh_checked["new_source_sha256"], "fresh源码闭包不同")
    output.mkdir(mode=0o700)
    if fresh:
        snapshot = output / "source-snapshot"
        snapshot.mkdir(mode=0o700)
        for name, checksum in plan["source_sha256"].items():
            member = snapshot / name
            member.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(plain(REPO / name), member)
            need(sha(member) == checksum, "fresh完整源码快照SHA不同")
        write_new(snapshot / "index.json", plan["source_sha256"])
    write_new(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"), "sha256": sha(output / "plan.json"),
            "model_started": False, "production_written": False}


def continuation_data(path):
    if read(plain(path)).get("protocol") in FRESH_RESUME_PROTOCOLS:
        from luban import goal_fresh
        checked = goal_fresh.load_verified(path)
        return (checked, read(checked["files"]["parent_plan"]["path"]),
                read(checked["files"]["baseline_old_training"]["path"]), goal_fresh.arguments(checked), goal_fresh.metadata(checked))
    """Pure evidence checks before creating a new product training episode."""
    from luban.goal_sequence import load_verified, arguments, metadata, content_sha, VALIDATION_FIELDS
    checked = load_verified(plain(path))
    files = checked["files"]
    parent = read(files["parent_plan"]["path"])
    result = read(files["parent_result"]["path"])
    baseline = read(files["baseline_old_training"]["path"])
    need(parent.get("protocol") == (PROTOCOL if checked["parent_steps"] == 450 else CONTINUATION_PROTOCOL)
         and parent.get("arguments") == checked["parent_arguments"]
         and result.get("protocol") == parent.get("protocol") and result.get("plan_sha256") == files["parent_plan"]["sha256"]
         and result.get("passed") is False and result.get("released") is True
         and result.get("production_written") is False and result.get("adopted") is False
         and type(result.get("teacher_api_calls")) is int and result["teacher_api_calls"] == 0,
         "必须是完整闭合、未采纳且已释放的父会话")
    rejected = result.get("training") or {}
    need(rejected.get("completed_steps") == checked["parent_steps"] and type(rejected.get("completed_steps")) is int
         and rejected.get("promoted") is False and rejected.get("gates") is None
         and not (VALIDATION_FIELDS - {"gates"}) & set(rejected)
         and rejected.get("id") == checked["parent_run_id"]
         and rejected.get("previous_version") == parent["starting_core"] == checked["starting_core"]
         and parent["identity"] == checked["identity"] and result.get("source_core") == parent["starting_core"],
         "父会话没有完整结束、已访问VALID或来源身份不同")
    records = {key: copy.deepcopy(parent[key]) for key in ("groups", "protection", "retention_regressions")}
    need(records == read(files["records"]["path"]) and content_sha(records) == checked["records_sha256"],
         "父会话完整训练/保护材料不同")
    need(result.get("baseline_old_training") == baseline
         and [row.get("id") for row in baseline] == [row["id"] for row in parent["old_training_original"]]
         and len(baseline) == 83, "原83TRAIN基线观察缺失或被筛选")
    aliases = {"new_material": "materials", "old_retention_complete": "old_retention",
               "old_training_original": "old_training", "old_protection": "old_protection", "acceptance": "acceptance"}
    need(parent["formal_protocol_sha256"] == files["formal_protocol"]["sha256"]
         and sha(parent["formal_protocol"]) == parent["formal_protocol_sha256"]
         and all(parent["formal_files"][key]["sha256"] == files[alias]["sha256"]
                 for key, alias in aliases.items()), "父会话与续训协议的正式材料不一致")
    return checked, parent, baseline, arguments(checked), metadata(checked)


def prepared_continuation(args):
    checked, parent, baseline, arguments, meta = continuation_data(args.goal_sequence_protocol)
    output = plain(args.output)
    need(output.parent.is_dir() and not output.exists(), "续训输出必须是全新独立目录")
    for source in (parent["source_store"], parent["model_root"], parent["output"]):
        source = plain(source)
        need(not output.is_relative_to(source) and not source.is_relative_to(output), "续训目录不能与父会话/模型/生产重叠")
    need(source_hashes(continuation=checked["protocol"] not in FRESH_RESUME_PROTOCOLS, fresh=checked["protocol"] in FRESH_RESUME_PROTOCOLS) == checked["new_source_sha256"], "续训新实现完整指纹不同")
    plan = copy.deepcopy(parent)
    plan.update(protocol=checked["protocol"], arguments=arguments, source_sha256=source_hashes(continuation=checked["protocol"] not in FRESH_RESUME_PROTOCOLS, fresh=checked["protocol"] in FRESH_RESUME_PROTOCOLS),
        goal_sequence_protocol={"path": checked["_protocol_path"], "sha256": checked["_protocol_sha256"]},
        goal_sequence=meta, parent_plan=checked["files"]["parent_plan"], parent_acceptance=checked["files"]["parent_result"],
        inherited_old_baseline=checked["files"]["baseline_old_training"],
        output=str(output), home=str(output / "isolated-home"), storage=str(output / "isolated-storage"),
        budget_policy=f"closed{checked['parent_steps']} plus at most1350 new updates, cumulative{meta['total_budget']}; complete450 rounds, TRAIN-only selection then one heldout evaluation",
        model_started=False, teacher_api_calls=0)
    if checked["protocol"] == DATA_PHASE:
        from luban.goal_fresh import phase_plan_fields
        plan.update(phase_plan_fields(checked))
    output.mkdir(mode=0o700)
    if checked["protocol"] in FRESH_RESUME_PROTOCOLS:
        snapshot = output / "source-snapshot"
        snapshot.mkdir(mode=0o700)
        for name, checksum in plan["source_sha256"].items():
            member = snapshot / name
            member.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, member)
            need(sha(member) == checksum, "续训完整38源码快照SHA不同")
        write_new(snapshot / "index.json", plan["source_sha256"])
    write_new(output / "plan.json", plan)
    return {"plan": str(output / "plan.json"), "sha256": sha(output / "plan.json"),
            "model_started": False, "production_written": False, "goal_sequence": meta}


def verify_plan(path, approved_sha):
    need(re.fullmatch(r"[a-f0-9]{64}", approved_sha or "") and sha(path) == approved_sha,
         "需要已核验的精确plan SHA")
    plan = read(path)
    continuation = plan.get("protocol") in CONTINUATION_PROTOCOLS | FRESH_RESUME_PROTOCOLS
    fresh = plan.get("protocol") in FRESH_PROTOCOLS
    need(plan.get("protocol") in {PROTOCOL, *FRESH_PROTOCOLS, *CONTINUATION_PROTOCOLS}, "未知训练协议")
    expected_arguments = ARGUMENTS
    if fresh and not continuation:
        from luban import goal_fresh
        descriptor = plan["goal_sequence_protocol"]
        need(descriptor == {"path": plan["formal_protocol"], "sha256": plan["formal_protocol_sha256"]}, "fresh协议绑定不同")
        checked = goal_fresh.load_verified(descriptor["path"])
        expected_arguments = goal_fresh.arguments(checked)
        need(plan["goal_sequence"] == goal_fresh.metadata(checked), "fresh预算不同")
    if continuation:
        descriptor = plan["goal_sequence_protocol"]
        need(sha(descriptor["path"]) == descriptor["sha256"], "续训协议SHA不同")
        checked, parent, baseline, expected_arguments, meta = continuation_data(descriptor["path"])
        changed = {"protocol", "arguments", "source_sha256", "goal_sequence_protocol", "goal_sequence", "parent_plan",
                   "parent_acceptance", "inherited_old_baseline", "output", "home", "storage", "budget_policy"}
        if checked["protocol"] == DATA_PHASE:
            from luban.goal_fresh import phase_plan_fields
            fields = phase_plan_fields(checked)
            need(all(plan.get(k) == v for k,v in fields.items()), "new data phase fields changed")
            changed.update(fields)
        need({key: value for key, value in plan.items() if key not in changed}
             == {key: value for key, value in parent.items() if key not in changed}
             and plan.get("goal_sequence") == meta
             and plan.get("parent_plan") == checked["files"]["parent_plan"]
             and plan.get("parent_acceptance") == checked["files"]["parent_result"]
             and plan.get("inherited_old_baseline") == checked["files"]["baseline_old_training"], "续训改变了原冻结内容或父来源")
    need(plan.get("arguments") == expected_arguments
         and plan.get("source_sha256") == source_hashes(continuation=continuation, fresh=fresh), "计划或实现源码已变")
    root = plain(plan["output"])
    need(plain(path) == root / "plan.json" and plan["home"] == str(root / "isolated-home")
         and plan["storage"] == str(root / "isolated-storage"), "隔离目录绑定已变")
    store = plain(plan["source_store"])
    need(all(sha(plain(store / name)) == digest for name, digest in plan["source_files"].items()), "当前正式核心已经改变")
    need(sha(plan["formal_protocol"]) == plan["formal_protocol_sha256"]
         and all(sha(entry["path"]) == entry["sha256"] for entry in plan["formal_files"].values())
         and sha(plan["materials"]) == plan["materials_sha256"]
         and sha(Path(plan["model_root"]) / "upstream-file-manifest.json") == plan["model_manifest_sha256"]
         and sha(Path(plan["model_root"]) / "config.json") == plan["model_config_sha256"], "材料或模型清单已变")
    from luban.retention import read_retention_file
    snapshot = read_retention_file(store / (plan["starting_core"] + ".retention.json"),
                                   plan["identity"], plan["starting_core"])
    payload = read(plan["materials"])
    need(materials({"samples": payload["TRAIN"] + payload["VALID"]}, snapshot, fresh=fresh, data_phase=plan["protocol"] == DATA_PHASE) == plan["groups"], "冻结材料已变")
    return plan


def clone_receipt(plan, memory):
    """Read only the bound accepted receipt through Bun; write isolated ledger."""
    from luban.bun_bridge import readonly_connection
    from luban.training_checkpoint import _schema, is_product_run
    source_root = Path(plan["source_store"]).parent.parent
    run_id = plan["starting_core"].removeprefix("core-")
    with closing(readonly_connection(source_root / "app/memory.sqlite3")) as db:
        pending = db.execute("SELECT id FROM training_runs WHERE identity=? AND state IN ('queued','running','paused','committing')",
                             (plan["identity"],)).fetchall()
        rows = db.execute("SELECT * FROM training_runs WHERE id=? AND identity=?",
                          (run_id, plan["identity"])).fetchall()
        event = db.execute("SELECT * FROM events WHERE id=?", ("training:" + run_id,)).fetchone()
    need(not pending and len(rows) == 1 and rows[0]["state"] == "committed" and event is not None,
         "来源存在待执行任务或缺少唯一正式提交/事件")
    row = rows[0]
    body = json.loads(row["body"])
    commit = body.get("commit") or {}
    original_report = read(Path(plan["source_store"]) / ("evolution-" + run_id + ".json"))
    installed_report = commit.get("report") or {}
    if installed_report != original_report:
        extra = {"foundation_install", "source_report_sha256", "training_updates_executed", "installation_scope"}
        need(set(installed_report) == set(original_report) | extra
             and {key: value for key, value in installed_report.items() if key not in extra} == original_report
             and installed_report.get("foundation_install") is True
             and type(installed_report.get("training_updates_executed")) is int
             and installed_report["training_updates_executed"] == 0
             and installed_report.get("source_report_sha256") == plan["source_files"]["evolution-" + run_id + ".json"],
             "来源安装回执与不可变源报告不一致")
    need(is_product_run(body) and body.get("kind") == "evolution"
         and body.get("binding", {}).get("identity") == plan["identity"]
         and event["kind"] == "evolution" and json.loads(event["body"]) == commit.get("report"),
         "来源正式回执或事件不一致")
    required = {plan["starting_core"] + ext for ext in (".json", ".safetensors", ".retention.json")}
    required.add("evolution-" + run_id + ".json")
    need(set(commit.get("artifacts", {})) == required
         and all(commit["artifacts"][name] == plan["source_files"][name] for name in required)
         and len(commit.get("pointers", [])) == 1
         and commit["pointers"][0].get("name") == "active-core.json"
         and commit["pointers"][0].get("after") == read(Path(plan["source_store"]) / "active-core.json"),
         "来源提交未绑定完整核心文件与活动指针")
    with memory.lock, memory.connect() as db:
        _schema(db)
        db.execute("INSERT INTO training_runs VALUES (?,?,?,?,?,?)", tuple(row))
        db.execute("INSERT INTO events (id,kind,body,created) VALUES (?,?,?,?)", tuple(event))


def verify_parent_journal(sequence, plan):
    """Compare the frozen rejected parent receipt to the actual Bun-owned ledger."""
    from luban.bun_bridge import readonly_connection
    parent = read(sequence["files"]["parent_plan"]["path"])
    expected = read(sequence["files"]["parent_journal"]["path"])
    database = Path(parent["storage"]) / "app/memory.sqlite3"
    with closing(readonly_connection(database)) as db:
        row = db.execute("SELECT * FROM training_runs WHERE id=? AND identity=?",
                         (sequence["parent_run_id"], plan["identity"])).fetchone()
    need(row is not None and row["state"] == "rejected", "父会话实际Bun回执不是闭合rejected")
    actual = {**json.loads(row["body"]), "state": row["state"]}
    need(actual == expected, "父会话实际Bun回执与封存来源不同")


def all_passed(rows, expected, *, native=False):
    need(len(rows) == len(expected) and [r["id"] for r in rows] == [r["id"] for r in expected]
         and all(r.get("passed") is True and r.get("complete") is True
                 and r.get("reference_used") is False and r.get("branch_enabled") is False
                 and isinstance(r.get("raw_completion_token_ids"), list)
                 and r["raw_completion_token_ids"] for r in rows), "真实闭卷答案/EOS或完整行绑定未通过")
    if native:
        need(all(row.get("finish_reason") == "stop" for row in rows), "原生生成未自然结束")


def goal_observation_binding(plan, report, sequence):
    """Validate immutable TRAIN observations without model allocation or current-source hashes."""
    if plan.get("protocol") == DATA_PHASE:
        from luban.goal_fresh import phase_observation_binding
        return phase_observation_binding(plan, report, sequence)
    if plan.get("protocol") in FRESH_PROTOCOLS:
        from verify_goal_foundation import fresh_observation_binding
        return fresh_observation_binding(plan, report, sequence)
    meta = report['goal_sequence']
    base = {'protocol_sha256': sha(sequence['_protocol_path']), 'parent_run_id': sequence['parent_run_id'],
            'parent_steps': sequence['parent_steps'], 'total_budget': sequence['max_steps'], 'new_steps_budget': 1350,
            'protocol': sequence['protocol'], 'new_steps_executed': report['completed_steps'] - sequence['parent_steps'],
            'cumulative_steps': report['completed_steps'], 'selected_segment': (report['completed_steps'] + 449) // 450,
            'training_budget_scope': 'bounded goal-specific episode, ordinary product512 limit unchanged'}
    extra = {'entry_parent'} if sequence['protocol'] == CONTINUATION_PROTOCOL_V3 else set()
    need(set(meta) == set(base) | {'reference_baseline_after_parent_closed', 'entry450', 'stage_observations'} | extra
         and all(meta[key] == value for key, value in base.items()), 'Goal observation metadata schema differs')
    training = plan['groups']['new_training']
    replay = {row['id']: row for row in plan['groups']['replay_training']}
    replay_ids = sequence['replay_generation_ids']
    need(len(training) == 16 and len(replay_ids) == 13 and set(replay_ids) <= replay.keys(), 'TRAIN16/old13 inventory differs')
    def observation(proof, step, label, reference, prior):
        need(type(proof.get('step')) is int and proof.get('step') == step and proof.get('label') == label
             and re.fullmatch('[a-f0-9]{64}', proof.get('tensor_sha256', ''))
             and proof.get('branch_enabled') is False and proof.get('reference_used') is False
             and proof.get('validation_rows_accessed') == 0
             and type(proof.get('observed_at')) in {int, float} and math.isfinite(proof['observed_at'])
             and type(proof.get('elapsed_seconds')) in {int, float} and math.isfinite(proof['elapsed_seconds'])
             and proof['elapsed_seconds'] >= 0 and len(proof.get('observations', [])) == 2,
             'Need two immutable same-W TRAIN observations with actual timing')
        sets = []
        for values in proof['observations']:
            need(set(values) == {'new', 'old13', 'replay_training_loss'}
                 and type(values['replay_training_loss']) in {int, float} and math.isfinite(values['replay_training_loss']), 'Invalid TRAIN loss observation')
            for key, samples in (('new', training), ('old13', [replay[i] for i in replay_ids])):
                rows = values[key]
                need([row['id'] for row in rows] == [sample['id'] for sample in samples], 'Double observation row order differs')
                for row in rows:
                    need(type(row.get('passed')) is bool and row.get('branch_enabled') is False
                         and row.get('reference_used') is False and (not row['passed'] or row.get('complete') is True), 'Observation lacks local complete output')
            sets.append({row['id'] for row in values['new'] if row['passed']})
        confirmed = set.intersection(*sets)
        expected = {'single_observation_correct': [len(ids) for ids in sets], 'confirmed_ids': sorted(confirmed),
            'confirmed_count': len(confirmed), 'pending_ids': sorted({sample['id'] for sample in training} - confirmed),
            'pending_count': len(training) - len(confirmed), 'newly_confirmed_from_reference': sorted(confirmed - reference),
            'newly_confirmed_since_prior': sorted(confirmed - prior), 'forgotten_from_reference': sorted(reference - confirmed),
            'forgotten_since_prior': sorted(prior - confirmed),
            'old13_both_passed': all(row['passed'] for values in proof['observations'] for row in values['old13'])}
        need(all(proof.get(key) == value for key, value in expected.items()), 'Incorrect confirmed/added/forgotten TRAIN accounting')
        return confirmed
    baseline = meta['reference_baseline_after_parent_closed']
    reference = observation(baseline, 0, 'reference_baseline_after_parent_closed', set(), set())
    need(baseline['tensor_sha256'] == sequence['original_tensor_sha256'], 'Later reference baseline is not original b45 tensor')
    prior = observation(meta['entry450'], 450, 'entry450_after_parent_closed', reference, reference)
    entry450_sha = (report['goal_sequence_sources']['parent_final_tensor_sha256']
                    if extra else sequence['parent_final_tensor_sha256'])
    need(meta['entry450']['tensor_sha256'] == entry450_sha, 'Entry450 observation is not actual parent FINAL tensor')
    if extra:
        parent_report = read(sequence['files']['parent_report']['path'])
        inherited = {key: parent_report['goal_sequence'][key] for key in ('reference_baseline_after_parent_closed', 'entry450', 'stage_observations')}
        actual = {key: meta[key] if key != 'stage_observations' else meta[key][:3] for key in inherited}
        need(actual == inherited and hashlib.sha256(json.dumps(actual, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest() == sequence['inherited_observations_sha256'], 'Inherited observation proof was rewritten')
        old_prior = set(meta['entry450']['confirmed_ids']) | set().union(*(set(proof['confirmed_ids']) for proof in meta['stage_observations'][:3]))
        observation(meta['entry_parent'], 1800, 'entry1800_after_parent_closed', reference, old_prior)
        need(meta['entry_parent']['tensor_sha256'] == sequence['parent_final_tensor_sha256'], 'Entry1800 tensor differs')
    selection = report['training_selection']
    need(selection.get('policy_version') == 'goal-fixed450-same-weight-v2'
         and type(selection.get('required_consecutive_passes')) is int and selection.get('required_consecutive_passes') == 0
         and type(selection.get('required_same_weight_passes')) is int and selection.get('required_same_weight_passes') == 2,
         'Fixed-round selection must use two same-W observations, not cross-W consecutive checks')
    stages = meta['stage_observations']
    need([proof['step'] for proof in stages] == [step for step in sequence.get('checkpoints', list(range(64, 449, 32)) + [900, 1350, 1800]) if step >= 900 and step <= report['completed_steps']], 'Stage boundary inventory differs')
    def round_status(proof):
        budget = report['baseline_training_replay_loss'] * 1.05 + .01
        losses_passed = all(values['replay_training_loss'] <= budget for values in proof['observations'])
        failed = bool(proof['forgotten_from_reference'] or proof['forgotten_since_prior'] or not proof['old13_both_passed'] or not losses_passed)
        expected = {'replay_loss_budget': budget, 'replay_loss_both_passed': losses_passed, 'safety_failed': failed,
            'round_result': 'protection_failed' if failed else 'satisfied' if proof['confirmed_count'] == 16 else 'partial' if proof['confirmed_count'] else 'pending',
            'actual_updates': 450}
        need(all(proof.get(key) == value for key, value in expected.items()), 'Fixed450 actual update/mastery/pending/old safety accounting differs')
        return not failed and proof['confirmed_count'] == 16
    round_status(meta['entry450'])
    need(meta['entry450']['safety_failed'] is False, 'Unsafe entry450 cannot begin a continuation episode')
    if extra:
        round_status(meta['entry_parent'])
        need(meta['entry_parent']['safety_failed'] is False, 'Unsafe actual1800 entry cannot begin another episode')
    selected = None
    history = selection['results'][13:]
    need(len(history) == len(stages), 'Fixed450 proof/selection row inventory differs')
    for index, proof in enumerate(stages):
        if extra and proof['step'] == sequence['parent_steps'] + 450:
            entry = meta['entry_parent']
            prior |= set(entry['confirmed_ids'])
        confirmed = observation(proof, proof['step'], 'segment-boundary', reference, prior)
        satisfied = round_status(proof)
        first = proof['observations'][0]
        expected = {'step': proof['step'], 'new_generation': first['new'], 'replay_training_generation': first['old13'],
            'replay_loss': first['replay_training_loss'], 'accepted': satisfied, 'consecutive_passes': 0,
            'same_weight_passes': 2 if satisfied else 0, 'minimum_new_exposures': (proof['step'] // 2) // 16,
            'tensor_sha256': proof['tensor_sha256']}
        need(history[index] == expected, 'Fixed450 selection row is not actual same-W double proof')
        if satisfied or proof['safety_failed']:
            need(index == len(stages) - 1 and proof['step'] == report['completed_steps'], 'Optimization continued after satisfaction or protection failure')
        if satisfied:
            selected = proof['step']
        prior |= confirmed
    need(selection.get('selected_step') == selected, 'Only fully satisfied complete450 same-W proof can select')
    if selected is not None:
        need(selected == report['completed_steps'] and selected in set(range(900, sequence['max_steps'] + 1, 450)) and selected > sequence['parent_steps']
             and stages[-1]['step'] == selected and stages[-1]['confirmed_count'] == 16
             and stages[-1]['pending_count'] == 0 and stages[-1]['old13_both_passed'] is True
             and not stages[-1]['forgotten_from_reference'] and not stages[-1]['forgotten_since_prior']
             and all(values['replay_training_loss'] <= report['baseline_training_replay_loss'] * 1.05 + .01
                     for values in stages[-1]['observations']), 'Final selection requires same-W double fullTRAIN and old safety')
    return reference, prior


def terminal_selected_confirmation(plan, report, rows, tensor_sha256):
    """Account for the actual W after a complete450-step round."""
    need(type(report['completed_steps']) is int
         and report['completed_steps'] in (set(range(7200 if plan.get('protocol') == DATA_PHASE else 450, 9451 if plan.get('protocol') == DATA_PHASE else 6751 if plan.get('protocol') == FRESH_RESUME_PROTOCOL_V4 else 5401 if plan.get('protocol') == FRESH_RESUME_PROTOCOL_V3 else 4051 if plan.get('protocol') == FRESH_RESUME_PROTOCOL else 2701, 450)) if plan.get('protocol') in FRESH_PROTOCOLS else ({2250, 2700, 3150} if plan.get('protocol') == CONTINUATION_PROTOCOL_V3 else {900, 1350, 1800})),
         'Final goal selection requires a complete450-step round boundary')
    samples = plan['groups']['new_training']
    wanted = {sample['id'] for sample in samples}
    sources = {}
    for engine in ('hf', 'native'):
        for turn in ('first', 'second'):
            phase = engine + '_new_' + turn
            values = [row for row in rows[phase] if row['id'] in wanted]
            all_passed(values, samples, native=engine == 'native')
            sources[phase] = {'rows_sha256': hashlib.sha256(json.dumps(values, sort_keys=True,
                ensure_ascii=False, allow_nan=False).encode()).hexdigest(), 'confirmed_ids': sorted(wanted)}
    meta = report['goal_sequence']
    reference = set(meta['reference_baseline_before_training' if plan.get('protocol') in FRESH_PROTOCOLS else 'reference_baseline_after_parent_closed']['confirmed_ids'])
    prior = set(meta['entry450']['confirmed_ids']) if 'entry450' in meta else set()
    if 'entry_parent' in meta:
        prior.update(meta['entry_parent']['confirmed_ids'])
    for proof in meta['stage_observations']:
        prior.update(proof['confirmed_ids'])
    return {'step': report['completed_steps'], 'segment': (report['completed_steps'] + 449) // 450,
            'tensor_sha256': tensor_sha256, 'confirmed_ids': sorted(wanted), 'confirmed_count': len(wanted), 'pending_ids': [], 'pending_count': 0,
            'newly_confirmed_from_reference': sorted(wanted - reference), 'newly_confirmed_since_prior': sorted(wanted - prior),
            'forgotten_from_reference': sorted(reference - wanted), 'forgotten_since_prior': sorted(prior - wanted),
            'sources': sources, 'scope': f'Actual selected-W final HF/native double TRAIN{len(wanted)} proof; original C report/journal unchanged'}


def native_rows(runtime, samples, engine):
    from luban.retention import messages, provenance
    from luban.generation_checks import evaluate_generation
    eos = runtime.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    rows = []
    for sample in samples:
        check = provenance(sample)["generation_check"]
        maximum = check.get("max_new_tokens", 192)
        ids = runtime._tokens(messages(sample), "off", output_reserve=maximum)
        value = engine.generate(ids[0].tolist(), max_tokens=maximum, eos_ids=eos,
            temperature=0, seed=42, on_token=lambda _: None, stop=threading.Event())
        tokens = value["tokens"]
        decoded = runtime.tokenizer.decode(tokens, skip_special_tokens=True)
        complete = bool(value.get("finish_reason") == "stop" and tokens and tokens[-1] in eos)
        rows.append({"id": sample["id"], "answer": decoded.strip(),
            **evaluate_generation(decoded.strip(), check, complete), "raw_completion_token_ids": tokens,
            "decoded_completion_unstripped": decoded, "completion_tokens": len(tokens),
            "max_new_tokens": maximum, "finish_reason": value.get("finish_reason"),
            "reference_used": False, "branch_enabled": False})
    return rows


@contextmanager
def data_service(home, receipt_path):
    """One bounded OS child; secrets remain in parent/child environment only."""
    receipt = {"started": False, "released": False, "pid": None}
    previous = {key: os.environ.get(key) for key in ("LUBAN_BUN_RPC_URL", "LUBAN_BUN_RPC_TOKEN")}
    child = None
    with tempfile.TemporaryDirectory(prefix="luban-goal-rpc-") as stage_name:
        stage = Path(stage_name)
        (stage / "backend").mkdir(); (stage / "scripts").mkdir()
        for name in ("server", "memory", "settings", "wire", "agreement", "home"):
            shutil.copyfile(REPO / "backend" / (name + ".ts"), stage / "backend" / (name + ".ts"))
        shutil.copyfile(REPO / "scripts/goal_data_service.ts", stage / "scripts/goal_data_service.ts")
        token = secrets.token_urlsafe(32)
        environment = {**os.environ, "LUBAN_BUN_RPC_TOKEN": token,
                       "LUBAN_GOAL_PARENT_PID": str(os.getpid())}
        try:
            bun = shutil.which("bun")
            need(bun is not None, "找不到Bun数据owner")
            child = subprocess.Popen([bun, str(stage / "scripts/goal_data_service.ts"), str(home)],
                cwd=stage, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True)
            receipt["pid"] = child.pid
            output = b""
            deadline = time.monotonic() + 10
            while b"\n" not in output:
                need(child.poll() is None and time.monotonic() < deadline, "隔离Bun owner启动失败或超时")
                readable, _, _ = select.select([child.stdout], [], [], min(0.2, max(0, deadline - time.monotonic())))
                if readable:
                    chunk = os.read(child.stdout.fileno(), 1024)
                    need(chunk and len(output) + len(chunk) <= 1024, "隔离Bun启动回执无效")
                    output += chunk
            url = output.split(b"\n", 1)[0].decode("ascii")
            need(re.fullmatch(r"http://127\.0\.0\.1:[0-9]+/internal/rpc", url), "隔离Bun地址无效")
            os.environ.update(LUBAN_BUN_RPC_URL=url, LUBAN_BUN_RPC_TOKEN=token)
            receipt.update(started=True, url=url)
            yield receipt
        except BaseException as error:
            receipt.update(error_type=type(error).__name__, error=str(error))
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait(timeout=5)
                receipt.update(exit_code=child.returncode, released=child.poll() is not None)
                receipt["stderr"] = child.stderr.read(8192).decode("utf-8", errors="replace").replace(token, "[redacted]")
                child.stdout.close(); child.stderr.close()
            write_new(receipt_path, receipt)


def run(args):
    plan = verify_plan(args.run, args.approved_sha)
    root = Path(plan["output"])
    need(not (root / "data-service.json").exists() and not (root / "execution-started.json").exists(),
         "本次计划已执行；保留历史失败另立新协议")
    try:
        with data_service(plan["home"], root / "data-service.json"):
            return execute(plan, args.approved_sha)
    except BaseException as error:
        if not (root / "acceptance.json").exists():
            write_new(root / "setup-failure.json", {"passed": False, "model_started": False,
                "production_written": False, "error_type": type(error).__name__, "error": str(error)})
        raise


def execute(plan, approved_sha):
    from luban.bun_bridge import enabled
    need(enabled(), "本次实验必须由独立Bun owner持有业务数据")
    root, storage = plain(plan["output"]), plain(plan["storage"])
    need(not storage.exists(), "训练store必须为独立新目录；不能隐式接续或覆盖失败记录")
    need(not (root / "execution-started.json").exists(), "本次计划已经执行；保留失败后另立协议")
    from luban.settings import Settings, require_storage, private_json
    settings = Settings(plan["home"])
    # RPC enforces that this home belongs to its Bun owner. A product owner's
    # endpoint cannot be reused to mutate this isolated home's configuration.
    settings.update({"model_path": plan["model_root"], "storage_root": str(storage),
        "temperature": 0, "max_new_tokens": 512, "local_thinking": "off",
        "automatic": {"mode": "manual", "scheduled_time": "02:00"},
        "teacher": {"base_url": "", "model": ""}})
    require_storage(root.parent)
    write_new(root / "execution-started.json", {"plan_sha256": approved_sha,
                                               "pid": os.getpid(), "started": time.time()})
    store = storage / "modules/luban0-v1"
    store.mkdir(parents=True, mode=0o700)
    for name, digest in plan["source_files"].items():
        shutil.copyfile(Path(plan["source_store"]) / name, store / name)
        need(sha(store / name) == digest, "隔离复制文件SHA不同")
    import torch
    from luban.memory import Memory
    from luban.runtime import Runtime
    from luban.training_checkpoint import TrainingSession, training_runs
    from luban.evolution import _generation, _target, consolidate
    from luban.coldpaging import full_backbone
    from luban.retention import protected_validation
    from luban import luban_training as engine, product_inference
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    memory = Memory(storage)
    memory.model_identity = plan["identity"]
    clone_receipt(plan, memory)
    runtime = Runtime(settings, memory, research=False)
    result = {"protocol": plan["protocol"], "plan_sha256": approved_sha, "passed": False,
        "teacher_api_calls": 0, "production_written": False, "adopted": False,
        "source_core": plan["starting_core"], "rows": {}, "started": time.time()}
    try:
        runtime.load()
        runtime.close_native()
        need(runtime.identity == plan["identity"] and runtime.active_core == plan["starting_core"]
             and runtime.branch.expert is None, "实际加载的起始核心/分支与冻结计划不同")
        need(protected_validation(runtime) == plan["protection"], "完整旧保护清单不同")
        records = {key: copy.deepcopy(plan[key]) for key in ("groups", "protection", "retention_regressions")}
        # Validate real token budgets before allocating any optimizer candidate.
        with full_backbone(runtime):
            for rows in plan["groups"].values():
                for row in rows:
                    runtime._sample_tokens(row)
            from luban.native_core_training import select
            binary, selected = select(runtime, preserve_replay_subspace=True,
                                     distillation_weight=5, field_first_token_weight=1, partial_policy=None)
            need(binary is not None and selected["engine"] == engine.ENGINE_ID,
                 "本次必须执行正式CE-release原生优化器，不能退回HF训练")
            result.update(starting_tensor_sha256=engine.tensor_sha(_target(runtime)[1]),
                          training_binary_sha256=selected["binary_sha256"])
            result["baseline_old_training"] = (read(plan["inherited_old_baseline"]["path"])
                if plan["protocol"] in CONTINUATION_PROTOCOLS | FRESH_RESUME_PROTOCOLS else _generation(runtime, plan["old_training_original"]))
            private_json(root / "baseline-old-training.json", result["baseline_old_training"])
        sequence_path = plan.get("goal_sequence_protocol", {}).get("path")
        sequence = None
        if sequence_path:
            from luban import goal_fresh, goal_sequence
            selected_goal = goal_fresh if plan["protocol"] in FRESH_PROTOCOLS else goal_sequence
            sequence = selected_goal.load_verified(sequence_path)
            if plan["protocol"] in CONTINUATION_PROTOCOLS | FRESH_RESUME_PROTOCOLS:
                verify_parent_journal(sequence, plan)
        session = TrainingSession(runtime, "evolution", plan["arguments"], records, scope="product",
                                  run_id=sequence["new_run_id"] if sequence else None)
        runtime.training_session = session
        call_arguments = {key: value for key, value in plan["arguments"].items() if key != "goal_sequence"}
        report = consolidate(runtime, **call_arguments, **({"goal_sequence": sequence_path} if sequence_path else {}))
        result["training"] = report
        if sequence:
            goal_observation_binding(plan, report, sequence)
            result["goal_sequence"] = copy.deepcopy(report["goal_sequence"])
        private_json(root / "quality-progress.json", result)
        need(report.get("promoted") is True and all(value is True for value in report.get("gates", {}).values())
             and report.get("training_engine", {}).get("engine") == engine.ENGINE_ID
             and report.get("previous_version") == plan["starting_core"], "隔离候选没有通过正式核心训练门禁")
        runtime.training_session = None
        runtime.branch.expert = None
        wanted = plan["groups"]["new_training"] + plan["groups"]["new_validation"] + plan["acceptance"]
        protection = plan["protection"]
        with full_backbone(runtime):
            candidate_sha = engine.tensor_sha(_target(runtime)[1])
            for phase, samples in (("hf_new_first", wanted), ("hf_new_second", wanted), ("hf_old", protection)):
                rows = _generation(runtime, samples)
                result["rows"][phase] = rows
                private_json(root / "quality-progress.json", result)
                all_passed(rows, samples)
                need(engine.tensor_sha(_target(runtime)[1]) == candidate_sha, "闭卷检查改变了候选权重")
        native = product_inference.start(runtime)
        checkpoint = report["checkpoint"]
        need(native is not None and native.loaded.get("core_loaded") is True
             and native.loaded.get("core_file_sha256") == checkpoint["sha256"]
             and native.loaded.get("core_tensor_sha256") == candidate_sha
             and native.loaded.get("base_identity") == plan["identity"], "原生worker未加载同一真实候选")
        result["native_loaded"] = native.loaded
        for phase, samples in (("native_new_first", wanted), ("native_new_second", wanted), ("native_old", protection)):
            rows = native_rows(runtime, samples, native)
            result["rows"][phase] = rows
            private_json(root / "quality-progress.json", result)
            all_passed(rows, samples, native=True)
        if sequence:
            result["terminal_selected_training_confirmation"] = terminal_selected_confirmation(plan, report, result["rows"], candidate_sha)
        current = next(row for row in training_runs(runtime.memory, runtime.identity) if row["id"] == session.id)
        need(current["state"] == "committed" and current["commit"]["report"] == report, "隔离提交回执不同")
        if sequence:
            with memory.lock, memory.connect() as db:
                event = db.execute("SELECT kind,body FROM events WHERE id=?", ("training:" + session.id,)).fetchone()
            result["training_journal"] = current
            result["training_event"] = {"kind": event["kind"], "body": json.loads(event["body"])} if event else None
            need(result["training_event"] == {"kind": "evolution", "body": report}, "续训真实采纳事件不同")
        need(all(sha(store / name) == digest for name, digest in current["commit"]["artifacts"].items()),
             "隔离候选四文件SHA已变")
        need(all(sha(Path(plan["source_store"]) / name) == digest
                 for name, digest in plan["source_files"].items()), "训练期间来源正式核心改变")
        result.update(passed=True, candidate_tensor_sha256=candidate_sha, candidate_version=runtime.active_core,
            artifacts=current["commit"]["artifacts"], completed_steps=report["completed_steps"],
            acceptance_scope=f"HF+native closed-book new{len(plan['groups']['new_training']) + len(plan['groups']['new_validation'])}+acceptance{len(plan['acceptance'])} twice and every immutable old{len(protection)} validation; isolated only")
    except BaseException as error:
        result.update(passed=False, error=str(error), error_type=type(error).__name__)
        raise
    finally:
        runtime._release_model()
        result.update(released=runtime.model is None and runtime.native_engine is None
                      and runtime._model_process_lock is None, elapsed_seconds=time.time() - result["started"])
        private_json(root / "acceptance.json", result)
    return {"acceptance": str(root / "acceptance.json"), "passed": result["passed"], "adopted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", type=Path)
    actions.add_argument("--prepare-continuation", action="store_true")
    parser.add_argument("--goal-sequence-protocol", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--source-store", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approved-sha")
    args = parser.parse_args()
    if args.prepare_continuation:
        if not args.goal_sequence_protocol or not args.output:
            parser.error("prepare-continuation需要goal-sequence-protocol和output")
        result = prepared_continuation(args)
    elif args.prepare:
        if not all((args.protocol, args.source_store, args.model_root, args.output)):
            parser.error("prepare需要protocol/source-store/model-root/output")
        result = prepared(args)
    else:
        if not args.approved_sha:
            parser.error("run需要精确approved-sha")
        result = run(args)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
