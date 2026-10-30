"""Check the already-running installed App; never start/load/train a model.

Uses real UI SSE, normal conversation writes and persisted decision events.
Public APIs cannot disable reference injection: any actual reference use fails.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

import httpx

DATA_PHASE = "evolution-goal-new-data-fixed450-v1"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    if isinstance(value, float):
        need(math.isfinite(value), "API包含非有限数字")
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, list):
        for item in value: finite(item)
    return value


def read(path):
    return finite(json.loads(Path(path).read_text()))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def messages(row):
    provenance = row.get("provenance", {})
    provenance = json.loads(provenance) if isinstance(provenance, str) else provenance
    return provenance.get("messages") or [{"role": "user", "content": row["prompt"]}]


def parse_sse(raw):
    events = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(line[5:].lstrip(" ") for line in block.splitlines() if line.startswith("data:"))
        if data:
            need(data != "[DONE]", "UI SSE不能替换为兼容API结束标记")
            value = finite(json.loads(data))
            need(isinstance(value, dict), "SSE事件必须为对象")
            events.append(value)
    return events


def goal_content(answer):
    """Conservative four-topic content check; preserve raw text for human review."""
    compact = re.sub(r"\s+", "", answer)
    topics = {
        "fast_and_slow": all(word in compact for word in ("快", "慢")),
        "surface_learning": bool(re.search(r"(表层|浅层).{0,24}(学习|经验|吸收|适应)", compact)),
        "deep_weights": bool(re.search(r"(深层|巩固|沉淀).{0,32}权重", compact)),
        "new_and_old_gate": all(re.search(pattern, compact) for pattern in
            (r"(新知识|新能力|新本领)", r"(旧能力|旧本领)", r"(检验|验证|验收|测试)", r"(合格|通过).{0,12}(升级|采纳)")),
    }
    return {"passed": all(topics.values()), "topics": topics,
            "scope": "Conservative topic-presence check; raw output retained for semantic review"}


def repetition(answer):
    compact = re.sub(r"\s+", "", answer)
    # Three copies of a 16-character span catch the user's looping identity
    # failure without treating a reused short technical term as a loop.
    found = next((compact[index:index + 16] for index in range(max(0, len(compact) - 15))
                  if compact.count(compact[index:index + 16]) >= 3), None)
    return {"passed": found is None, "repeated_span": found, "span_characters": 16, "minimum_copies": 3}


def foundation_binding(path, candidate, candidate_path, protocol_path, plan):
    """Read-only F -> immutable C proof; never rewrite the candidate acceptance."""
    need(path.name == "luban-foundation.json" and not path.is_symlink()
         and not path.parent.is_symlink(), "需要真实正式基础清单")
    manifest = read(path)
    version, source = manifest.get("version"), candidate["candidate_version"]
    need(isinstance(version, str) and re.fullmatch(r"core-[a-f0-9]{12}", version)
         and version not in {source, candidate["source_core"]} and manifest.get("quality_passed") is True
         and manifest.get("identity") == plan["identity"], "正式F版本/身份/质量状态无效")
    report_name = "evolution-" + version[5:] + ".json"
    expected = {version + suffix for suffix in (".json", ".safetensors", ".retention.json")} | {report_name}
    files = manifest.get("files")
    need(manifest.get("report") == report_name and isinstance(files, dict) and set(files) == expected,
         "正式F四文件清单不完整")
    members = path.parent / "luban-foundation"
    need(members.is_dir() and not members.is_symlink(), "正式F成员目录无效")
    for name, checksum in files.items():
        member = members / name
        need(not member.is_symlink() and isinstance(checksum, str)
             and re.fullmatch(r"[a-f0-9]{64}", checksum) and sha(member) == checksum, "正式F成员SHA不同：" + name)
    original = Path(plan["storage"]) / "modules/luban0-v1"
    source_report_name = "evolution-" + source[5:] + ".json"
    source_names = {source + suffix for suffix in (".json", ".safetensors", ".retention.json")} | {source_report_name}
    need(set(candidate["artifacts"]) == source_names, "原始C四文件清单无效")
    for name, checksum in candidate["artifacts"].items():
        member = original / name
        need(not member.is_symlink() and sha(member) == checksum, "原始C文件SHA不同：" + name)
    report = read(members / report_name)
    source_report = read(original / source_report_name)
    metadata = read(members / (version + ".json"))
    retention = read(members / (version + ".retention.json"))
    source_retention = read(original / (source + ".retention.json"))
    core_sha = candidate["artifacts"][source + ".safetensors"]
    expected_protocol_sha = sha(protocol_path)
    continued_binding = None
    if plan.get("protocol") in {"evolution-goal-fixed450-rounds-v2", "evolution-goal-fixed450-rounds-v3", "evolution-goal-prospective-fixed450-v1", "evolution-goal-prospective-fixed450-v2", "evolution-goal-prospective-fixed450-v3", "evolution-goal-prospective-fixed450-v4", DATA_PHASE}:
        from verify_goal_foundation import continued_training_binding, fresh_training_binding
        continued_binding = (fresh_training_binding(candidate_path, protocol_path, report)
            if plan.get("protocol") in {"evolution-goal-prospective-fixed450-v1", "evolution-goal-prospective-fixed450-v2", "evolution-goal-prospective-fixed450-v3", "evolution-goal-prospective-fixed450-v4", DATA_PHASE} else continued_training_binding(candidate_path, protocol_path, report))
        expected_protocol_sha = continued_binding["training_protocol_sha256"]
    need(source_report == candidate["training"]
         and read(original / (source + ".json")) == source_report["checkpoint"]
         and report.get("previous_version") == candidate["source_core"]
         and report.get("source_previous_version") == candidate["source_core"]
         and report.get("source_training_version") == source
         and report.get("source_quality_sha256") == candidate["artifacts"][source_report_name]
         and report.get("source_protocol_sha256") == expected_protocol_sha
         and report.get("native_identity_sha256") == sha(candidate_path),
         "正式F没有绑定实际C报告/原生验收/预登记协议")
    need(report.get("version") == version and report.get("id") == version[5:]
         and report.get("identity") == manifest["identity"] and report.get("promoted") is True
         and report.get("publication_state") == "published" and report.get("native_identity_verified") is True
         and report.get("gates") and all(value is True for value in report["gates"].values()),
         "正式F发布与完整门禁无效")
    changed = {"id", "version", "checkpoint", "retention", "native_identity_verified", "native_identity_sha256",
               "source_quality_sha256", "source_protocol_sha256", "source_training_version", "source_previous_version"}
    if plan.get("protocol") in {"evolution-goal-prospective-fixed450-v1", "evolution-goal-prospective-fixed450-v2", "evolution-goal-prospective-fixed450-v3", "evolution-goal-prospective-fixed450-v4", DATA_PHASE}:
        from release_goal_foundation import portable_fresh_proof
        actual_plan_path = candidate_path.parent / "plan.json"
        need(read(actual_plan_path) == plan and candidate.get("plan_sha256") == sha(actual_plan_path),
             "正式F便携证明的实际C计划不同")
        need("fresh_source_proof" not in source_report
             and report.get("fresh_source_proof") == portable_fresh_proof(
                 plan, candidate, original, candidate_path, actual_plan_path),
             "正式F便携证明不等于实际C冻结封装")
        changed.add("fresh_source_proof")
    else:
        need("fresh_source_proof" not in report and "fresh_source_proof" not in source_report,
             "旧协议不能附加fresh便携证明")
    need({key: value for key, value in report.items() if key not in changed}
         == {key: value for key, value in source_report.items() if key not in changed},
         "正式F改变了原C训练报告或门禁")
    need(report.get("checkpoint") == metadata and metadata.get("version") == version
         and metadata.get("report") == report_name and metadata.get("identity") == manifest["identity"]
         and files[version + ".safetensors"] == metadata.get("sha256") == core_sha
         and metadata.get("retention_sha256") == files[version + ".retention.json"],
         "正式F checkpoint/SF字节与原C不同")
    relinked = {"version", "report", "retention_sha256"}
    need({key: value for key, value in metadata.items() if key not in relinked}
         == {key: value for key, value in source_report["checkpoint"].items() if key not in relinked},
         "正式F checkpoint改变了原C参数/身份")
    need(retention.get("version") == version and retention.get("identity") == manifest["identity"]
         and digest({key: value for key, value in retention.items() if key != "content_sha256"})
         == retention.get("content_sha256")
         and {key: value for key, value in retention.items() if key not in {"version", "content_sha256"}}
         == {key: value for key, value in source_retention.items() if key not in {"version", "content_sha256"}}
         and report.get("retention", {}).get("file_sha256") == files[version + ".retention.json"]
         and report["retention"].get("content_sha256") == retention.get("content_sha256"),
         "正式F保护快照不同")
    return {"version": version, "identity": manifest["identity"], "manifest_path": str(path),
            "manifest_sha256": sha(path), "report_sha256": files[report_name], "files": files,
            "source_training_version": source, "source_quality_sha256": candidate["artifacts"][source_report_name],
            "native_identity_sha256": sha(candidate_path), "source_protocol_sha256": expected_protocol_sha,
            "continued_training_binding": continued_binding,
            "core_file_sha256": core_sha, "core_tensor_sha256": candidate["candidate_tensor_sha256"]}


def check_response(record, case, expected_core, event):
    from luban.generation_checks import evaluate_generation
    events = record["events"]
    need(not any(row.get("type") == "error" for row in events), "实际SSE返回错误")
    meta = [row for row in events if row.get("type") == "meta"]
    done = [row for row in events if row.get("type") == "done"]
    need(len(meta) == 1 and len(done) == 1 and events[-1].get("type") == "done", "SSE缺少完整唯一meta/done")
    usage = done[0].get("usage") or {}
    answer = "".join(row.get("content", "") for row in events if row.get("type") == "delta")
    record.update(answer=answer, usage=usage, meta=meta[0], conversation_id=done[0].get("conversation_id"),
                  decision_event=event, source_core=event.get("model_context", {}).get("core_version"))
    complete = usage.get("finish_reason") == "stop" and usage.get("local_complete") is True
    check = (goal_content(answer) if case.get("user_default_temperature") and not case.get("frozen_goal_check") else
             evaluate_generation(answer, case["check"], complete))
    check["complete"] = complete
    check["passed"] = check["passed"] and complete
    record["generation_check"] = check
    record["repetition_check"] = repetition(answer)
    need(check["passed"], "完整目标答案或自然EOS不合格：" + case["id"])
    need(record["repetition_check"]["passed"], "实际输出出现重复循环")
    need(not any(row.get("reasoning_content") for row in events), "thinking off出现实际思考正文")
    need(usage.get("reference_used") is False and usage.get("reference_sample_id") is None,
         "实际使用了参考数据，不能当作闭卷掌握")
    need(usage.get("requested_thinking") == "off" and usage.get("effective_thinking") == "off",
         "实际生成模板不是thinking off")
    need(meta[0].get("route") == "local" and event.get("decision", {}).get("action") == "local_core"
         and event.get("complete") is True and event.get("model_context", {}).get("core_version") == expected_core
         and event.get("model_context", {}).get("expert_id") is None,
         "实际决策未使用预期新版核心的独立生成")
    need(event.get("request_id") == meta[0].get("request_id")
         and event.get("messages") == record["request"]["messages"]
         and event.get("answer") == answer and event.get("usage") == usage,
         "持久事件与实际SSE正文/上下文/usage不一致")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        need(type(usage.get(key)) is int and usage[key] > 0, "token accounting无效")
    need(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
         and usage["completion_tokens"] <= record["request"]["max_tokens"], "token accounting不一致")
    record.update(passed=True, reference_used=False, complete_eos_reported=True,
        eos_evidence="Actual API local_complete plus matching persisted complete decision; raw token/EOS proof remains in candidate acceptance")


def app_cases(new, old, acceptance, *, fresh=False):
    identity_context = [{"role": "user", "content": "你是谁？"}]
    identity = next(row for row in old if messages(row) == identity_context)
    identity_check = json.loads(identity["provenance"])["generation_check"]
    cases = [{"id": "retained-identity", "messages": identity_context, "check": identity_check}]
    goal_answers = {row["answer"] for row in new["TRAIN"] + new["VALID"]}
    for row in acceptance:
        value = json.loads(row["provenance"])
        need((fresh or row["answer"] in goal_answers) and row["answer"] != identity["answer"]
             and (value["generation_check"].get("validator") == "goal_topics" if fresh else value["generation_check"]["type"] == "exact"), "新目标验收不能替换为身份口号")
        cases.append({"id": row["id"], "messages": messages(row), "check": value["generation_check"]})
    need(len(cases) == 9, "必须完整检查8条预登记目标问题")
    cases.append({"id": "retained-17-plus-25", "messages": [{"role": "user", "content": "计算 17 + 25，只输出数字。"}],
                  "check": {"type": "semantic", "validator": "integer", "expected": 42, "max_new_tokens": 128}})
    if fresh:
        original_row = next(row for row in new["TRAIN"] if messages(row) == [{"role": "user", "content": "怎么无限进化"}])
        original_check = json.loads(original_row["provenance"])["generation_check"]
        original = {"id": "known-user-original-question", "messages": messages(original_row), "check": original_check,
                    "scope": "Previously seen user symptom; TRAIN regression, not untouched test"}
        cases.append(original)
        cases.append({"id": "known-identity-to-evolution", "messages": [{"role": "user", "content": "你是谁？"},
            {"role": "assistant", "content": identity["answer"]}, {"role": "user", "content": "怎么无限进化"}],
            "check": original_check, "scope": "Previously seen identity-to-evolution regression, not untouched test"})
    else:
        original = next(case for case in cases if case["id"] == "goal-acceptance-original-user-question")
    cases.append({"id": "goal-user-default-temperature", "messages": copy.deepcopy(original["messages"]),
                  "check": copy.deepcopy(original["check"]) if fresh else {"max_new_tokens": 256}, "user_default_temperature": True, "frozen_goal_check": fresh})
    if fresh:
        # Known human-reported failure context, not a new answer gold or unseen transfer.
        repeated_history = (
            "“无限进化”听起来很酷！🚀 但这其实是鲁班零号（luban0）在 **qwen3.5** 架构下，"
            "结合qwen3.5训练，能无限进化的模型。\n\n具体来说，我的进化体现在以下几个方面：\n\n"
            "1. **无限进化的模型**：作为qwen3.5训练的，我能无限进化的模型。"
            "这意味着我不仅能无限进化的模型，还能无限进化的模型。\n\n"
            "2. **qwen3.5训练的，能无限进化的模型**。基于qwen3.5训练的，能无限进化的模型。\n\n"
            "3. **基于qwen3.5训练的，能无限进化的模型**。基于qwen3.5训练的，能无限进化的模型。\n\n"
            "4. **基于qwen3.5训练的，能无限进化的模型**。基于qwen3.5训练的，能无限进化的模型。\n\n"
            "5. **基于qwen3.5训练的，能无限进化的模型**。基于qwen3.5训练的，能无限进化的模型。"
            + "".join(f"\n\n{i}. **基于qwen3.5训练的，能无限进化的模型**。基于qwen3.5训练的，能无限进化的模型。" for i in range(6, 15))
            + "\n\n15. **基于")
        cases.append({"id": "known-repetition-history-to-evolution", "messages": copy.deepcopy(original["messages"]) + [
            {"role": "assistant", "content": repeated_history}, {"role": "user", "content": "怎么无限进化"}],
            "check": original["check"], "scope": "Previously reported repeating-answer context; known TRAIN regression, not untouched test"})
    return cases


def phase_app_cases(plan, protocol_path):
    """Preserve the original14 compatibility cases; append current frozen12."""
    from luban import goal_fresh
    checked = goal_fresh.load_verified(protocol_path)
    need(plan.get("protocol") == DATA_PHASE and checked["protocol"] == DATA_PHASE
         and plan["formal_files"] == checked["_files"]
         and plan["original_formal_protocol"] == checked["original_formal_protocol"]
         and plan["original_formal_files"] == checked["original_formal_files"], "Phase App material ancestry differs")
    original = checked["original_formal_protocol"]
    need(original["sha256"] == "a94531b7802ee06b6e286d434be052c38b3fb1ff8594e019c768265d58866d3f"
         and sha(original["path"]) == original["sha256"], "Original App14 formal SHA differs")
    old_files = checked["original_formal_files"]
    for entry in [*old_files.values(), *checked["_files"].values()]:
        need(sha(entry["path"]) == entry["sha256"], "Phase App original/current frozen material changed")
    old_new = read(old_files["new_material"]["path"])
    old_accept = read(old_files["acceptance"]["path"])
    old = read(old_files["old_protection"]["path"])
    need((len(old_new["TRAIN"]),len(old_new["VALID"]),len(old_accept),len(old)) == (24,8,8,89), "Original App14 inventory differs")
    cases = app_cases(old_new, old, old_accept, fresh=True)
    current = read(checked["_files"]["acceptance"]["path"])
    need(current == plan["acceptance"] and len(current) == 12, "Phase current12 acceptance differs")
    from luban.generation_checks import is_finite_generation_check, evaluate_generation
    for row in current:
        provenance = row["provenance"]
        provenance = json.loads(provenance) if isinstance(provenance,str) else provenance
        check = provenance["generation_check"]
        need(check.get("type") == "semantic" and check.get("validator") == "goal_topics"
             and is_finite_generation_check(check) and evaluate_generation(row["answer"],check,True)["passed"], "Phase App finite frozen goal contract differs")
        cases.append({"id": row["id"], "messages": messages(row), "check": check, "frozen_goal_check": True})
    need(len(cases) == 26 and len({r["id"] for r in cases}) == 26, "Phase App must retain14 and append12 unique cases")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--candidate-acceptance", required=True, type=Path)
    parser.add_argument("--foundation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--home", type=Path, default=Path.home() / "Library/Application Support/Luban")
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-build", type=int)
    args = parser.parse_args()
    need(not args.output.exists(), "证据目录必须是新目录")
    formal = read(args.protocol)
    candidate = read(args.candidate_acceptance)
    plan_path = args.candidate_acceptance.parent / "plan.json"
    plan = read(plan_path)
    need(candidate.get("passed") is True and candidate.get("released") is True,
         "不能验证未通过/未释放的训练候选")
    need(candidate.get("plan_sha256") == sha(plan_path)
         and plan["formal_protocol_sha256"] == sha(args.protocol)
         and candidate.get("source_core") == formal["source_core"]["version"],
         "候选验收未绑定本次正式协议/来源核心")
    candidate_core = candidate.get("candidate_version")
    need(isinstance(candidate_core, str) and re.fullmatch(r"core-[a-f0-9]{12}", candidate_core)
         and candidate_core != formal["source_core"]["version"], "需要新版候选核心")
    core_sha = candidate["artifacts"][candidate_core + ".safetensors"]
    need(type(candidate.get("teacher_api_calls")) is int and candidate["teacher_api_calls"] == 0
         and candidate.get("native_loaded", {}).get("core_file_sha256") == core_sha
         and candidate["native_loaded"].get("core_tensor_sha256") == candidate.get("candidate_tensor_sha256")
         and candidate["native_loaded"].get("core_loaded") is True
         and candidate["native_loaded"].get("base_identity") == plan["identity"],
         "训练候选未绑定实际原生验收")
    binding = foundation_binding(args.foundation_manifest, candidate, args.candidate_acceptance, args.protocol, plan)
    expected_core = binding["version"]
    parent = args.protocol.parent
    if plan.get("protocol") == DATA_PHASE:
        cases = phase_app_cases(plan, args.protocol)
    else:
        new = read(parent / formal["new_material"])
        old = read(parent / formal["old_protection"])
        for key in ("new_material", "old_protection", "acceptance"):
            need(sha(parent / formal[key]) == plan["formal_files"][key]["sha256"], "正式验收材料已变：" + key)
        cases = app_cases(new, old, read(parent / formal["acceptance"]), fresh=plan.get("protocol") in {"evolution-goal-prospective-fixed450-v1", "evolution-goal-prospective-fixed450-v2", "evolution-goal-prospective-fixed450-v3", "evolution-goal-prospective-fixed450-v4", DATA_PHASE})
    launch = read(args.home / "launch.json")
    need(type(launch.get("port")) is int and 1 <= launch["port"] <= 65535
         and isinstance(launch.get("token"), str) and launch["token"], "本机启动信息无效")
    secret = launch["token"]
    args.output.mkdir(parents=True, mode=0o700)
    report = {"passed": False, "backend": launch.get("backend"), "pid": launch.get("pid"),
        "compute_pid": launch.get("compute_pid"), "expected_core": expected_core,
        "core_file_sha256": core_sha, "protocol_sha256": sha(args.protocol),
        "candidate_acceptance_sha256": sha(args.candidate_acceptance), "responses": [],
        "candidate_version": candidate_core, "foundation_binding": binding,
        "teacher_calls_requested": 0, "training_requested": False,
        "api_reference_control": "No public disable flag; fail on actual reference_used=true",
        "scope": "Actual installed App/API curated answers; no generalization/performance claim",
        "case_inventory": {"original_compatibility": 14, "current_phase_acceptance": 12,
            "original_acceptance8": "Previously unseen, untrained old acceptance; additional App compatibility, not current-phase heldout"} if plan.get("protocol") == DATA_PHASE else {"original_compatibility": len(cases)}}
    def save():
        text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).replace(secret, "[redacted]") + "\n"
        (args.output / "result.json").write_text(text)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{launch["port"]}', timeout=180,
                          trust_env=False, cookies={"luban_session": secret}, follow_redirects=False) as client:
            def get(path):
                response = client.get(path); response.raise_for_status()
                return finite(response.json())
            report.update(version=get("/api/version"), status=get("/api/status"), health=get("/api/health"))
            need(report["backend"] == "bun", "需要已安装Bun正式App")
            if args.expected_version:
                need(report["version"]["version"] == args.expected_version, "应用版本不是预期新版")
            if args.expected_build is not None:
                need(report["version"]["build"] == args.expected_build, "应用build不是预期新版")
            runtime = report["status"]["runtime"]
            need(runtime["state"] == "ready" and runtime["core_version"] == expected_core
                 and runtime.get("expert_parameters") == 0, "实际App未就绪或核心/分支不匹配")
            need(Path(runtime["model_path"]).resolve() == args.foundation_manifest.parent.resolve(),
                 "实际App模型目录不是已绑定正式F目录")
            need((runtime.get("job") or {}).get("state") not in {"queued", "running"}, "活动模型操作，保持原状")
            close = get("/api/automatic/close-status")
            need(not close.get("running") and not close.get("package_busy"), "活动工作，保持原状")
            settings = get("/api/settings")
            report["user_default_temperature"] = settings["temperature"]
            store = Path(settings["storage_root"]) / "modules/luban0-v1"
            pointer = read(store / "active-core.json")
            need(pointer["version"] == expected_core and pointer["sha256"] == core_sha
                 and pointer["identity"] == binding["identity"]
                 and all(not (store / name).is_symlink() and sha(store / name) == checksum
                         for name, checksum in binding["files"].items()), "安装的活动核心四文件不同")
            initial = get("/api/conversations")["conversations"]
            old_conversations = {row["id"]: digest(get("/api/conversations/" + row["id"])) for row in initial}
            before_data = {path: digest(get(path)) for path in ("/api/samples", "/api/materials")}
            save()
            identity_answer = None
            identity_conversation = None
            for case in cases:
                context = copy.deepcopy(case["messages"])
                body = {"messages": context, "route": "local", "thinking": "off", "temperature": 0,
                        "max_tokens": case["check"].get("max_new_tokens", 192), "stream": True}
                if case.get("user_default_temperature"):
                    body.pop("temperature")
                if case["id"] in {"goal-acceptance-identity-to-evolution", "known-identity-to-evolution"}:
                    need(identity_answer == context[1]["content"], "实际身份回答与预登记多轮前缀不同")
                    context[1]["content"] = identity_answer
                    body["conversation_id"] = identity_conversation
                record = {"id": case["id"], "request": body, "passed": False}
                report["responses"].append(record)
                started = time.perf_counter(); chunks = []
                try:
                    with client.stream("POST", "/api/chat", json=body) as response:
                        record["http_status"] = response.status_code
                        record["content_type"] = response.headers.get("content-type")
                        if response.status_code >= 400:
                            chunks.append(response.read())
                        response.raise_for_status()
                        need("text/event-stream" in (record["content_type"] or ""), "实际API不是SSE")
                        received = 0
                        for chunk in response.iter_bytes():
                            chunks.append(chunk)
                            received += len(chunk)
                            need(received <= 4 * 1024 * 1024, "SSE超出4MiB证据上限")
                finally:
                    raw = b"".join(chunks).decode("utf-8", errors="replace")
                    record.update(raw_sse=raw, seconds=time.perf_counter() - started)
                    (args.output / (case["id"] + ".sse")).write_text(raw.replace(secret, "[redacted]"))
                    save()
                record["events"] = parse_sse(raw)
                metas = [event for event in record["events"] if event.get("type") == "meta"]
                request_id = metas[0].get("request_id") if len(metas) == 1 else None
                decisions = [event["body"] for event in get("/api/events")["events"]
                             if event["kind"] == "decision" and event["body"].get("request_id") == request_id]
                need(len(decisions) == 1, "缺少匹配此次SSE的唯一持久决策事件")
                need(decisions[0].get("base_identity") == binding["identity"], "实际事件基座身份不同")
                check_response(record, case, expected_core, decisions[0]); save()
                if case["id"] == "retained-identity":
                    identity_answer, identity_conversation = record["answer"].strip(), record["conversation_id"]
            report["final_status"] = get("/api/status")
            need(report["final_status"]["runtime"]["core_version"] == expected_core
                 and report["final_status"]["teacher_calls"] == report["status"]["teacher_calls"], "核心或教师调用计数改变")
            need(before_data == {path: digest(get(path)) for path in before_data}, "原学习材料发生变化")
            need(get("/api/settings") == settings, "用户配置发生变化")
            need(all(digest(get("/api/conversations/" + identifier)) == value
                     for identifier, value in old_conversations.items()), "既有对话内容发生变化")
            report.update(passed=True, checks=len(cases), retained_conversations=len(old_conversations),
                          conversations_after=len(get("/api/conversations")["conversations"]),
                          source_core=expected_core, old_conversations_unchanged=True, old_learning_records_unchanged=True)
    except BaseException as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    print(f"Installed App API checks passed: {len(cases)}; teacher/training requests: 0")


if __name__ == "__main__": main()
