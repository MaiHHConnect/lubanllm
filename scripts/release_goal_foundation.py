"""Prepare/build a new local foundation directory from completed goal evidence.

No training, model allocation, adoption, network calls or public upload occur.
The prepared plan freezes inputs and a fresh F identity. Build requires its exact
approved SHA; it copies the 13 pinned official files and reseals four F members.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import uuid
import zlib

from luban.model_packages import TRUSTED, TRUSTED_IDENTITY, _plain, _record
from luban.retention import (_encoded, provenance, read_retention_file,
                             validate_core_retention, normalized_context)
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.upgrades import CORE_GATES, _core_records
from train_evolution_goal import (ARGUMENTS, PROTOCOL, FRESH_PROTOCOL, FRESH_PROTOCOLS, CONTINUATION_PROTOCOL, CONTINUATION_PROTOCOLS, continuation_data,
                                  need, read, sha, write_new, source_hashes, data_service, goal_observation_binding, terminal_selected_confirmation)

REPO = Path(__file__).resolve().parents[1]
RELEASE_PROTOCOL = "evolution-goal-local-foundation-v1"
DATA_PHASE = "evolution-goal-new-data-fixed450-v1"
FRESH_PROTOCOLS = FRESH_PROTOCOLS | {"evolution-goal-prospective-fixed450-v4", DATA_PHASE}


def mounted(path):
    path = _plain(path, directory=True)
    need(path.is_dir(), "发布父目录必须已经存在")
    if path.parts[:2] == ("/", "Volumes"):
        need((Path("/Volumes") / path.parts[2]).is_mount(), "外置发布盘未挂载")
    return path


def check_base(root, identity):
    root = _plain(root, directory=True)
    need(root.is_dir() and identity == TRUSTED_IDENTITY, "必须保留已固定官方2B基座身份")
    need(len(TRUSTED["base"]) == 13, "官方13文件白名单发生变化")
    for name, expected in TRUSTED["base"].items():
        need(_record(root / name) == expected, "官方文件SHA/大小不一致: " + name)
    config = read(root / "config.json")
    eos = config.get("text_config", config).get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    need(eos and all(type(token) is int and token >= 0 for token in eos), "官方EOS定义缺失")
    return eos


def tensor_sha(path, metadata):
    """Hash saved BF16 bytes without importing Torch or allocating a model."""
    path = _plain(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        need(len(prefix) == 8, "权重头缺失")
        length = struct.unpack("<Q", prefix)[0]
        need(2 <= length <= 1024**2, "权重头越界")
        from luban.model_packages import _json
        header = _json(stream.read(length))
        names = set(header) - {"__metadata__"}
        need(names == {metadata["parameter"]}, "必须是实际唯一末层核心张量")
        record = header[metadata["parameter"]]
        need(record == {"dtype": "BF16", "shape": [2048, 6144], "data_offsets": [0, 25165824]}
             and path.stat().st_size == 8 + length + 25165824, "核心张量形状/精度/范围不一致")
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            need(all((word & 0x7f80) != 0x7f80 for (word,) in struct.iter_unpack("<H", block)),
                 "核心张量含非有限BF16")
            digest.update(block)
        return digest.hexdigest()


def check_rows(rows, samples, eos, *, native):
    need(isinstance(rows, list) and [row.get("id") for row in rows] == [row["id"] for row in samples],
         "闭卷证据行清单/顺序不同")
    for row, sample in zip(rows, samples):
        tokens = row.get("raw_completion_token_ids")
        check = validate_generation_check(provenance(sample).get("generation_check"), default_expected=sample["answer"])
        maximum = check["max_new_tokens"]
        need(isinstance(tokens, list) and 0 < len(tokens) <= maximum
             and all(type(token) is int and token >= 0 for token in tokens) and tokens[-1] in eos
             and row.get("completion_tokens") == len(tokens) and type(row.get("completion_tokens")) is int
             and row.get("max_new_tokens") == maximum and type(row.get("max_new_tokens")) is int
             and row.get("passed") is True and row.get("complete") is True
             and row.get("reference_used") is False and row.get("branch_enabled") is False
             and isinstance(row.get("decoded_completion_unstripped"), str)
             and row["decoded_completion_unstripped"].strip() == row.get("answer"), "原始闭卷输出/EOS不完整")
        actual = evaluate_generation(row["answer"], check, True)
        need(actual.get("passed") is True and all(type(row.get(key)) is type(value) and row.get(key) == value
                                                  for key, value in actual.items()),
             "原始完整答案不满足冻结规则")
        if native:
            need(row.get("finish_reason") == "stop", "原生输出没有自然EOS结束")
        else:
            need(row.get("thinking") == "off" and row.get("do_sample") is False, "HF闭卷参数变化")


def release_contract(plan):
    """450 stays exact; only the separately frozen closed-parent goal episode can extend it."""
    if plan.get("protocol") == PROTOCOL:
        need(plan.get("arguments") == ARGUMENTS, "450参数不能重写为累计预算")
        return ARGUMENTS, plan["formal_protocol_sha256"], False
    if plan.get("protocol") in FRESH_PROTOCOLS:
        from luban import goal_fresh
        checked = goal_fresh.load_verified(plan["goal_sequence_protocol"]["path"])
        need(plan["arguments"] == goal_fresh.arguments(checked) and plan["goal_sequence"] == goal_fresh.metadata(checked)
             and plan["source_sha256"] == checked["new_source_sha256"], "fresh预算/源码不同")
        return plan["arguments"], checked["_protocol_sha256"], True
    need(plan.get("protocol") in CONTINUATION_PROTOCOLS, "不是已授权目标训练协议")
    descriptor = plan["goal_sequence_protocol"]
    need(sha(descriptor["path"]) == descriptor["sha256"], "续训协议SHA不同")
    checked, parent, baseline, arguments, meta = continuation_data(descriptor["path"])
    need(plan.get("arguments") == arguments and plan.get("goal_sequence") == meta
         and plan.get("source_sha256") == checked["new_source_sha256"]
         and all(plan[key] == parent[key] for key in ("groups", "protection", "retention_regressions", "acceptance",
             "old_training_original", "starting_core", "source_store", "source_files", "identity", "formal_files",
             "formal_protocol", "formal_protocol_sha256", "model_root", "model_config_sha256", "model_manifest_sha256")),
         "续训原完整内容/新实现冻结不同")
    return arguments, checked["_protocol_sha256"], True


def sequence_report(plan, accepted, report):
    if plan.get("protocol") in FRESH_PROTOCOLS:
        from verify_goal_foundation import fresh_training_binding
        fresh_training_binding(Path(plan["output"]) / "acceptance.json", Path(plan["formal_protocol"]))
        journal = accepted["training_journal"]
    else:
        journal = accepted.get("training_journal", {})
    if plan.get("protocol") not in FRESH_PROTOCOLS:
        _closed_sequence_report(plan, accepted, report)
    _actual_sequence_journal(plan, accepted, report, journal)


def _closed_sequence_report(plan, accepted, report):
    from luban.goal_sequence import load_verified, report_metadata, validate_protocol, source_metadata
    checked = load_verified(plan["goal_sequence_protocol"]["path"])
    completed = report["completed_steps"]
    need(report.get("goal_sequence_sources") == source_metadata(checked), "原450历史/张量/源码来源链不同")
    goal_observation_binding(plan, report, checked)
    need(report.get("goal_sequence") == accepted.get("goal_sequence") and completed in validate_protocol(checked)[13:]
         and report["training_selection"]["checkpoints"] == validate_protocol(checked)
         and report["training_selection"]["selected_step"] == completed,
         "续训本段/累计执行或选择门槛不同")
    need(accepted.get("terminal_selected_training_confirmation") == terminal_selected_confirmation(
        plan, report, accepted["rows"], accepted["candidate_tensor_sha256"]), "最终选中W的HF/native双TRAIN确认不同")
    parent_report = read(checked["files"]["parent_report"]["path"])
    history = report["training_selection"]["results"]
    need(history[:len(parent_report["training_selection"]["results"])] == parent_report["training_selection"]["results"]
         and [row["step"] for row in history] == [step for step in validate_protocol(checked) if step <= completed],
         "丢失/重写原450或续训完整选择历史")
    journal = accepted.get("training_journal", {})
    commit = journal.get("commit", {})
    records = {key: plan[key] for key in ("groups", "protection", "retention_regressions")}
    need(journal.get("id") == report["id"] and journal.get("state") == "committed"
         and journal.get("scope") == journal.get("binding", {}).get("scope") == "product"
         and journal.get("arguments") == journal["binding"].get("arguments") == plan["arguments"]
         and journal.get("records") == records and commit.get("report") == report
         and commit.get("artifacts") == accepted["artifacts"]
         and commit.get("event_kind") == "evolution"
         and accepted.get("training_event") == {"kind": "evolution", "body": report},
         "续训源C实际product提交/事件不完整")


def _actual_sequence_journal(plan, accepted, report, journal):
    with tempfile.TemporaryDirectory(prefix="luban-goal-publish-ledger-") as name:
        from contextlib import closing
        from luban.bun_bridge import readonly_connection
        root = Path(name).resolve()
        with data_service(root / "home", root / "owner.json"):
            with closing(readonly_connection(Path(plan["storage"]) / "app/memory.sqlite3")) as db:
                row = db.execute("SELECT * FROM training_runs WHERE id=? AND identity=?", (report["id"], plan["identity"])).fetchone()
                event = db.execute("SELECT kind,body FROM events WHERE id=?", ("training:" + report["id"],)).fetchone()
            need(row is not None and {**json.loads(row["body"]), "state": row["state"]} == journal
                 and event is not None and {"kind": event["kind"], "body": json.loads(event["body"])} == accepted["training_event"],
                 "续训源C实际Bun账本在封存后不同")


def inputs(plan_path, acceptance_path, native_path):
    plan, accepted, native = read(plan_path), read(acceptance_path), read(native_path)
    arguments, training_protocol_sha, continuation = release_contract(plan)
    need(type(plan.get("seed")) is int and plan["seed"] == 42, "不是冻结的正式目标训练计划")
    need(isinstance(plan.get("starting_core"), str) and re.fullmatch(r"core-[a-f0-9]{12}", plan["starting_core"]),
         "起始正式核心版本无效")
    need(_plain(plan_path) == _plain(Path(plan["output"]) / "plan.json")
         and plan["storage"] == str(Path(plan["output"]) / "isolated-storage"), "runner隔离目录绑定不同")
    need(accepted.get("protocol") == plan["protocol"] and accepted.get("plan_sha256") == sha(plan_path)
         and accepted.get("passed") is True and accepted.get("released") is True
         and accepted.get("production_written") is False and accepted.get("adopted") is False
         and type(accepted.get("teacher_api_calls")) is int and accepted["teacher_api_calls"] == 0
         and accepted.get("source_core") == plan.get("starting_core"), "真实runner验收未完整通过/释放")
    need(native.get("passed") is True and native.get("native_loaded") == accepted.get("native_loaded")
         and isinstance(native.get("rows"), dict)
         and all(native["rows"].get(key) == accepted.get("rows", {}).get(key)
                 for key in ("native_new_first", "native_new_second", "native_old")), "原生proof与真实验收不同")
    need(plan.get("identity") == TRUSTED_IDENTITY and plan.get("source_sha256") == source_hashes(continuation=plan.get("protocol") in CONTINUATION_PROTOCOLS, fresh=plan.get("protocol") in FRESH_PROTOCOLS),
         "续训基座或完整训练实现指纹不同")
    for name, digest in plan["source_sha256"].items():
        need(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
             and sha(_plain(REPO / name)) == digest, "训练实现源码已变化: " + str(name))
    protocol = _plain(plan["formal_protocol"])
    need(sha(protocol) == plan["formal_protocol_sha256"], "正式预登记协议已变化")
    formal = read(protocol)
    phase = plan["protocol"] == DATA_PHASE
    need(formal.get("status") == ("FROZEN_NEW_DATA_PHASE" if phase else "PRE_REGISTERED_NOT_TRAINED")
         and formal.get("weights_updated") is False and formal.get("production_written") is False
         and formal.get("gpu_used") is False and type(formal.get("teacher_calls")) is int
         and formal["teacher_calls"] == 0, "正式预登记状态不一致")
    need(set(plan["formal_files"]) == {"new_material", "old_retention_complete", "old_training_original", "old_protection", "acceptance"},
         "正式冻结材料成员不完整")
    for key, entry in plan["formal_files"].items():
        need(isinstance(formal[key], str) and Path(formal[key]).name == formal[key]
             and (entry == formal["new_files"].get(key, formal["original_formal_files"].get(key)) if phase else Path(entry["path"]) == protocol.parent / formal[key]), "材料不属于正式预登记协议")
        need(sha(_plain(entry["path"])) == entry["sha256"], "冻结材料文件已变化")
    source = _plain(plan["source_store"], directory=True)
    previous_names = {plan["starting_core"] + ext for ext in (".json", ".safetensors", ".retention.json")}
    previous_names |= {"active-core.json", "graph-verification.json", "evolution-" + plan["starting_core"].removeprefix("core-") + ".json"}
    need(set(plan["source_files"]) == previous_names
         and all(sha(_plain(source / name)) == digest for name, digest in plan["source_files"].items()),
         "原正式核心在训练后变化")
    need(formal["source_core"] == read(source / "active-core.json")
         and formal["source_core"]["version"] == plan["starting_core"]
         and formal["source_core"]["identity"] == plan["identity"]
         and formal["source_retention_sha256"] == plan["source_files"][plan["starting_core"] + ".retention.json"]
         == plan["formal_files"]["old_retention_complete"]["sha256"], "原正式核心与预登记协议不同")
    previous = read_retention_file(source / (plan["starting_core"] + ".retention.json"),
                                   plan["identity"], plan["starting_core"])
    need(plan["protection"] == previous["groups"]["new_validation"] + previous["groups"]["replay_validation"]
         + previous["regression_validation"] and len(plan["protection"]) == 89
         and len(plan["old_training_original"]) == 83
         and plan["old_training_original"] == read(plan["formal_files"]["old_training_original"]["path"])
         and plan["protection"] == read(plan["formal_files"]["old_protection"]["path"])
         and plan["old_training_original"] == [row for rows in previous["groups"].values()
                                               for row in rows if row["purpose"] == "training"],
         "全部旧83训练与89保护记录不一致")
    from train_evolution_goal import materials
    material = read(plan["materials"])
    need(sha(plan["materials"]) == plan["materials_sha256"]
         and materials({"samples": material["TRAIN"] + material["VALID"]}, previous, fresh=plan.get("protocol") in FRESH_PROTOCOLS, data_phase=phase) == plan["groups"]
         and len(plan["acceptance"]) == (12 if phase else 8) and plan["retention_regressions"] == plan["acceptance"]
         and plan["acceptance"] == read(plan["formal_files"]["acceptance"]["path"]),
         "完整冻结材料/只读验收不同")
    version = accepted.get("candidate_version")
    need(isinstance(version, str) and re.fullmatch(r"core-[a-f0-9]{12}", version)
         and version != plan["starting_core"], "候选版本无效")
    store = _plain(Path(plan["storage"]) / "modules/luban0-v1", directory=True)
    names = {version + ext for ext in (".json", ".safetensors", ".retention.json")}
    report_name = "evolution-" + version.removeprefix("core-") + ".json"
    names.add(report_name)
    need(set(accepted.get("artifacts", {})) == names
         and all(sha(_plain(store / name)) == accepted["artifacts"][name] for name in names), "候选四成员SHA不同")
    report, metadata = read(store / report_name), read(store / (version + ".json"))
    snapshot = read_retention_file(store / (version + ".retention.json"), plan["identity"], version)
    need(report == accepted.get("training") and report.get("checkpoint") == metadata
         and report.get("id") == version.removeprefix("core-") and report.get("version") == version
         and report.get("identity") == plan["identity"] and report.get("partial_policy") is None
         and not {"native_identity_verified", "native_identity_sha256", "source_training_version", "foundation_install"} & set(report)
         and report.get("previous_version") == plan["starting_core"]
         and report.get("gates") == dict.fromkeys(CORE_GATES, True)
         and all(type(value) is bool for value in report["gates"].values())
         and snapshot["groups"] == plan["groups"]
         and snapshot["regression_validation"] == previous["regression_validation"] + plan["acceptance"]
         and report.get("completed_steps") == accepted.get("completed_steps"), "报告/保护快照不是同一次完整普通训练")
    from luban.foundation import _distillation_weight
    need({"steps": report.get("steps"), "learning_rate": report.get("learning_rate"),
          "distillation_weight": _distillation_weight(report),
          "preserve_replay_subspace": report.get("replay_subspace", {}).get("enabled")} == {key: value for key, value in arguments.items() if key != "goal_sequence"},
         "训练参数与冻结计划不同")
    if continuation:
        sequence_report(plan, accepted, report)
    need(metadata.get("identity") == plan["identity"] and metadata.get("sha256") == accepted["artifacts"][version + ".safetensors"]
         and metadata.get("version") == version and metadata.get("report") == report_name
         and metadata.get("retention_sha256") == accepted["artifacts"][version + ".retention.json"]
         and report.get("retention", {}).get("file_sha256") == metadata["retention_sha256"]
         and report.get("retention", {}).get("content_sha256") == snapshot["content_sha256"], "候选元数据不同")
    dataset_sha = hashlib.sha256(json.dumps(snapshot["groups"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    need(report.get("dataset_sha256") == metadata.get("dataset_sha256") == dataset_sha
         and report.get("split_ids") == {key: [row["id"] for row in rows] for key, rows in snapshot["groups"].items()}
         and report.get("sample_sha256") == {row["id"]: hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                                             for rows in snapshot["groups"].values() for row in rows}, "来源C完整材料指纹不同")
    journal, events = receipt(report, snapshot, accepted["artifacts"])
    need(len(_core_records(Path(plan["storage"]), journal, events)) == 1, "来源C完整普通损失/保护门禁未通过")
    raw_sha = tensor_sha(store / (version + ".safetensors"), metadata)
    loaded = accepted.get("native_loaded", {})
    need(raw_sha == accepted.get("candidate_tensor_sha256") == loaded.get("core_tensor_sha256")
         and loaded.get("core_loaded") is True and loaded.get("base_identity") == plan["identity"]
         and loaded.get("core_file_sha256") == metadata["sha256"], "原生实际加载张量与候选不同")
    eos = check_base(plan["model_root"], plan["identity"])
    need(sha(Path(plan["model_root"]) / "upstream-file-manifest.json") == plan["model_manifest_sha256"]
         and sha(Path(plan["model_root"]) / "config.json") == plan["model_config_sha256"], "冻结官方模型清单不同")
    wanted = plan["groups"]["new_training"] + plan["groups"]["new_validation"] + plan["acceptance"]
    contexts = [normalized_context(row) for row in wanted]
    training_contexts = {normalized_context(row) for rows in plan["groups"].values() for row in rows}
    need(len(contexts) == len(set(contexts))
         and not training_contexts & {normalized_context(row) for row in plan["acceptance"]}, "最终验收上下文重复或泄漏")
    expected_phases = {"hf_new_first", "hf_new_second", "hf_old", "native_new_first", "native_new_second", "native_old"}
    need(set(accepted["rows"]) == expected_phases, "六轮真实验收清单不同")
    for phase in sorted(expected_phases):
        check_rows(accepted["rows"][phase], plan["protection"] if phase.endswith("old") else wanted,
                   eos, native=phase.startswith("native"))
    return plan, accepted, store, report, metadata, snapshot


def receipt(report, snapshot, artifacts):
    """A proposed fresh-install receipt for shared guards, never a live journal."""
    from luban.core_publication import content_sha
    records = {"groups": snapshot["groups"], "protection": snapshot["regression_validation"], "retention_regressions": []}
    body = {"scope": "product", "binding": {"identity": report["identity"], "arguments": {"steps": report["steps"], "learning_rate": report["learning_rate"],
                "distillation_weight": report.get("distillation_weight", report.get("distillation", {}).get("weight")),
                "preserve_replay_subspace": True},
            "records_sha256": content_sha(records)}, "records": records,
        "commit": {"report": report, "artifacts": artifacts, "event_kind": "evolution",
            "pointers": [{"name": "active-core.json", "before": None, "after": {
                "version": report["version"], "identity": report["identity"], "sha256": report["checkpoint"]["sha256"]}}]}}
    return {report["id"]: {"kind": "evolution", "state": "committed", "body": body}}, {
        "training:" + report["id"]: {"kind": "evolution", "body": report}}


def portable_fresh_proof(plan, accepted, store, acceptance_path, plan_path):
    """Seal actual admitted C inputs after the unchanged complete preflight."""
    from verify_goal_foundation import fresh_training_binding
    fresh_training_binding(Path(acceptance_path), Path(plan['formal_protocol']))
    formal = read(plan['formal_protocol'])
    paths = {'candidate': acceptance_path, 'plan': plan_path, 'protocol': plan['formal_protocol'],
        'source_report': Path(store) / ('evolution-' + accepted['candidate_version'][5:] + '.json'),
        'capture': formal['protection_capture']['path'], 'config': Path(plan['model_root']) / 'config.json',
        'tokenizer': Path(plan['model_root']) / 'tokenizer.json',
        'binary_receipt': Path(plan['output']) / 'source-snapshot/build/native/luban-worker.json',
        **{key: value['path'] for key, value in plan['formal_files'].items()}}
    if plan.get('protocol') in {'evolution-goal-prospective-fixed450-v2','evolution-goal-prospective-fixed450-v3','evolution-goal-prospective-fixed450-v4'}:
        paths['sequence_protocol'] = plan['goal_sequence_protocol']['path']
    phase = plan.get('protocol') == DATA_PHASE
    if phase:
        paths.update(original_protocol=formal['original_formal_protocol']['path'],
            original_new_material=formal['original_formal_files']['new_material']['path'],
            original_acceptance=formal['original_formal_files']['acceptance']['path'],
            parent_report=formal['files']['parent_report']['path'], parent_result=formal['files']['parent_result']['path'])
        need(len(paths) == 18, 'Phase exact18 portable documents required')
    documents = {}
    for key, path in paths.items():
        read(path)  # Reject duplicates, nonfinite values, links and oversized input first.
        documents[key] = Path(path).read_bytes().decode('utf-8')
    need(sha(paths['tokenizer']) == TRUSTED['base']['tokenizer.json']['sha256']
         and sha(paths['config']) == TRUSTED['base']['config.json']['sha256'], 'portable official decoder/config SHA differs')
    need(read(paths['source_report']) == accepted['training'], 'portable source report is not actual C')
    resumed = plan.get('protocol') in {'evolution-goal-prospective-fixed450-v2','evolution-goal-prospective-fixed450-v3','evolution-goal-prospective-fixed450-v4'}
    if resumed or phase:
        need(all(0 < len(value.encode('utf-8')) <= 32*1024**2 for value in documents.values())
             and sum(len(value.encode('utf-8')) for value in documents.values()) <= 64*1024**2,
             'portable decoded evidence exceeds bounded publication budget')
        documents = {key: {'raw_sha256': hashlib.sha256(value.encode('utf-8')).hexdigest(),
            'raw_bytes': len(value.encode('utf-8')), 'data': base64.b64encode(zlib.compress(value.encode('utf-8'),9)).decode('ascii')}
            for key,value in documents.items()}
        need(len(json.dumps(documents,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2).encode()) <= 24*1024**2,
             'portable compressed fresh evidence exceeds24MiB')
    else:
        need(sum(len(value.encode('utf-8')) for value in documents.values()) <= 24 * 1024**2,
             'portable fresh evidence exceeds publication budget')
    snapshot = Path(plan['output']) / 'source-snapshot'
    need(all(sha(_plain(snapshot / name)) == checksum for name, checksum in plan['source_sha256'].items()),
         'portable frozen source snapshot changed')
    return {'policy': 'fresh-b45-data-phase-core8-v1' if phase else 'fresh-b45-portable-core8-v2' if resumed else 'fresh-b45-portable-core8-v1',
            **({'encoding':'zlib-base64-utf8-v1'} if resumed or phase else {}), 'documents': documents,
            'source_snapshot_sha256': copy.deepcopy(plan['source_sha256'])}


def _bounded_report(report):
    data = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
    need(len(data.encode('utf-8')) <= 32 * 1024**2, "foundation report with portable proof exceeds32MiB")


def prepare(args):
    plan_path, acceptance_path, native_path = (_plain(path) for path in (args.plan, args.acceptance, args.native_proof))
    plan, accepted, store, report, metadata, snapshot = inputs(plan_path, acceptance_path, native_path)
    target = _plain(args.target, directory=True)
    mounted(target.parent)
    need(target.name == "luban0-v2" and not target.exists(), "仅允许全新luban0-v2目录")
    for source in (store, Path(plan["source_store"]), Path(plan["model_root"]), Path(plan["output"])):
        need(not target.is_relative_to(source) and not source.is_relative_to(target), "目标与来源目录不得重叠")
    uid = uuid.uuid4().hex[:12]
    need("core-" + uid not in (report["version"], report["previous_version"]), "新发布版本冲突")
    output = _plain(args.output_plan)
    mounted(output.parent)
    need(not output.is_relative_to(target), "发布计划必须保存在模型目录外")
    frozen = {"protocol": RELEASE_PROTOCOL, "version": "core-" + uid, "target": str(target),
        "plan": {"path": str(plan_path), "sha256": sha(plan_path)},
        "acceptance": {"path": str(acceptance_path), "sha256": sha(acceptance_path)},
        "native_proof": {"path": str(native_path), "sha256": sha(native_path)},
        "source_training_version": report["version"], "source_previous_version": report["previous_version"],
        "source_quality_sha256": accepted["artifacts"]["evolution-" + report["id"] + ".json"],
        "source_protocol_sha256": release_contract(plan)[1],
        "upstream_files": TRUSTED["base"], "implementation_sha256": sha(Path(__file__)),
        "scope": "local published directory only; not public upload, production adoption or new training"}
    write_new(output, frozen)
    return {"plan": str(output), "sha256": sha(output), "version": frozen["version"],
            "model_started": False, "production_written": False, "uploaded": False}


def copy_new(source, target, expected):
    _plain(source)
    _plain(target)
    with Path(source).open("rb") as incoming, Path(target).open("xb") as outgoing:
        os.chmod(target, 0o600)
        shutil.copyfileobj(incoming, outgoing, 1024**2)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    need(_record(target) == expected, "发布复制后SHA/大小不一致: " + target.name)


def build(path, approved_sha):
    path = _plain(path)
    need(isinstance(approved_sha, str) and re.fullmatch(r"[a-f0-9]{64}", approved_sha)
         and sha(path) == approved_sha, "需要精确批准的发布plan SHA")
    frozen = read(path)
    need(frozen.get("protocol") == RELEASE_PROTOCOL and frozen.get("implementation_sha256") == sha(Path(__file__))
         and frozen.get("upstream_files") == TRUSTED["base"], "发布实现/官方白名单已变化")
    for key in ("plan", "acceptance", "native_proof"):
        need(sha(_plain(frozen[key]["path"])) == frozen[key]["sha256"], "发布输入证据已变化: " + key)
    plan, accepted, store, original, metadata, snapshot = inputs(*(frozen[key]["path"] for key in ("plan", "acceptance", "native_proof")))
    need(frozen.get("source_training_version") == original["version"]
         and frozen.get("source_previous_version") == original["previous_version"]
         and frozen.get("source_quality_sha256") == accepted["artifacts"]["evolution-" + original["id"] + ".json"]
         and frozen.get("source_protocol_sha256") == release_contract(plan)[1], "发布来源绑定不同")
    version = frozen["version"]
    need(isinstance(version, str) and re.fullmatch(r"core-[a-f0-9]{12}", version)
         and version not in (original["version"], original["previous_version"]), "新发布版本不独立")
    target = _plain(frozen["target"], directory=True)
    parent = mounted(target.parent)
    need(target.name == "luban0-v2" and not target.exists(), "目标已存在；禁止覆写")
    for source in (store, Path(plan["source_store"]), Path(plan["model_root"]), Path(plan["output"])):
        need(not target.is_relative_to(source) and not source.is_relative_to(target), "目标与来源目录重叠")
    stage = Path(tempfile.mkdtemp(prefix=".goal-foundation-", dir=parent))
    try:
        members = stage / "modules/luban0-v1"
        members.mkdir(parents=True, mode=0o700)
        report_name = "evolution-" + version.removeprefix("core-") + ".json"
        retention = copy.deepcopy(snapshot)
        retention["version"] = version
        retention["content_sha256"] = hashlib.sha256(_encoded({key: value for key, value in retention.items()
                                                                if key != "content_sha256"})).hexdigest()
        validate_core_retention(retention, original["identity"], version)
        write_new(members / (version + ".retention.json"), retention)
        copy_new(store / (original["version"] + ".safetensors"), members / (version + ".safetensors"),
                 _record(store / (original["version"] + ".safetensors")))
        metadata = copy.deepcopy(metadata)
        metadata.update(version=version, report=report_name, retention_sha256=sha(members / (version + ".retention.json")))
        write_new(members / (version + ".json"), metadata)
        report = copy.deepcopy(original)
        report.update(id=version.removeprefix("core-"), version=version, checkpoint=metadata,
            native_identity_verified=True, native_identity_sha256=frozen["native_proof"]["sha256"],
            source_quality_sha256=frozen["source_quality_sha256"], source_protocol_sha256=frozen["source_protocol_sha256"],
            source_training_version=original["version"], source_previous_version=original["previous_version"])
        report["retention"] = {"path": str(target / "luban-foundation" / (version + ".retention.json")),
            "file_sha256": metadata["retention_sha256"], "content_sha256": retention["content_sha256"]}
        if plan.get("protocol") in FRESH_PROTOCOLS:
            need(frozen["native_proof"]["sha256"] == frozen["acceptance"]["sha256"], "fresh native identity must seal full actual candidate acceptance")
            report["fresh_source_proof"] = portable_fresh_proof(plan, accepted, store,
                frozen["acceptance"]["path"], frozen["plan"]["path"])
            _bounded_report(report)
        write_new(members / report_name, report)
        artifacts = {file.name: sha(file) for file in members.iterdir()}
        journal, events = receipt(report, retention, artifacts)
        need(_core_records(stage, journal, events) == [], "正式基础发布没有通过共享普通八门禁")
        need(tensor_sha(members / (version + ".safetensors"), metadata) == accepted["candidate_tensor_sha256"],
             "正式F张量不是原候选C")
        members.rename(stage / "luban-foundation")
        (stage / "modules").rmdir()
        for name, expected in TRUSTED["base"].items():
            copy_new(Path(plan["model_root"]) / name, stage / name, expected)
        manifest = {"identity": original["identity"], "version": version, "quality_passed": True,
                    "report": report_name, "files": artifacts}
        write_new(stage / "luban-foundation.json", manifest)
        need(set(file.name for file in stage.iterdir()) == set(TRUSTED["base"]) | {"luban-foundation", "luban-foundation.json"},
             "发布目录出现额外载荷")
        # Verify every source member again after copying. The original C journal/report never change.
        need(all(sha(store / name) == digest for name, digest in accepted["artifacts"].items()), "复制期间来源C变化")
        need(all(sha(frozen[key]["path"]) == frozen[key]["sha256"] for key in ("plan", "acceptance", "native_proof"))
             and sha(Path(__file__)) == frozen["implementation_sha256"], "复制期间发布证据/实现变化")
        need(not target.exists(), "目标在发布期间出现；禁止替换")
        # mkdir reserves the final pathname without replacing an existing empty directory.
        target.mkdir(mode=0o700)
        try:
            for file in sorted(stage.iterdir(), key=lambda entry: entry.name == "luban-foundation.json"):
                file.rename(target / file.name)
        except BaseException:
            # An incomplete final directory is retained, never mistaken for a successful release.
            raise
        stage.rmdir()
        return {"passed": True, "version": version, "target": str(target), "source_training_version": original["version"],
            "manifest_sha256": sha(target / "luban-foundation.json"), "files": artifacts,
            "plan_sha256": approved_sha, "model_started": False, "production_written": False,
            "training_updates_executed": 0, "uploaded": False, "scope": frozen["scope"]}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--build", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--native-proof", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--approved-sha")
    args = parser.parse_args()
    if args.prepare:
        if not all((args.plan, args.acceptance, args.native_proof, args.target, args.output_plan)):
            parser.error("prepare需要plan/acceptance/native-proof/target/output-plan")
        result = prepare(args)
    else:
        result = build(args.build, args.approved_sha)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
