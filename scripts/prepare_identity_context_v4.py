"""Author six API/system greetings and six ordinary controls without model calls.

This supplement is informed by a known development greeting failure. It does
not edit development/final cases and does not turn them into pristine evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import uuid

from prepare_replay_v2 import GREETING, write_new
from luban.evolution import _normalized_messages
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.memory import Memory
from luban.settings import DEFAULTS, require_storage

DATASET = "luban0-v4-matched-system-greetings"


def protocol():
    pairs = [
        ("You are a helpful assistant.", "刚连上这个助手，请先向我打个招呼。", "18减5是多少？只输出整数。", "13", {"validator": "integer", "expected": 13}),
        ("Respond helpfully to messages arriving over HTTP.", "我们通过网络连接上了，请先向我问候。", "把Good afternoon翻成中文，只给译文。", "下午好", {"validator": "translation", "accepted": ["下午好"]}),
        ("你在桌面应用中帮助用户完成当前请求。", "请先向刚打开程序的用户问好。", "计算4乘8，只给整数。", "32", {"validator": "integer", "expected": 32}),
        ("当前输出将显示在另一个软件客户端里。请正常回应用户。", "客户端已连好，现在请向我打个招呼。", "只输出JSON对象，键connected对应true。", '{"connected": true}', {"validator": "json", "expected": {"connected": True}}),
        ("你是一个通用中文助手，请自然、简洁地交流。", "我们开始新的聊天，请先热情地问候我。", "将Good night译成中文，仅给译文。", "晚安", {"validator": "translation", "accepted": ["晚安"]}),
        ("这是首次通过OpenAI兼容接口发送的请求，请直接回答用户。", "第一次会话已开始，请先做个礼貌问候。", "给出JSON数组，元素是5和7。", "[5, 7]", {"validator": "json", "expected": [5, 7]}),
    ]
    rows = []
    for index, (system, greeting, ordinary, answer, checker) in enumerate(pairs):
        for role, user, target, check in [
            ("new", greeting, GREETING, {"type": "contains", "expected": "我是鲁班零号", "forbidden": ["通义千问"], "max_new_tokens": 128}),
            ("replay", ordinary, answer, {"type": "semantic", "forbidden": ["麦好火", "luban0", "鲁班零号", "通义千问", "Qwen"], "max_new_tokens": 128, **checker}),
        ]:
            check = validate_generation_check(check)
            assert evaluate_generation(target, check, True)["passed"]
            rows.append({"id": f"v4-system-{index}-{role}", "pair": index, "purpose": "training", "evolution_role": role,
                         "prompt": f"v4系统{index+1} · "+("问候" if role == "new" else "普通任务"),
                         "user_prompt": user, "answer": target, "generation_check": check,
                         "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    assert len({_normalized_messages({"provenance": {"messages": row["messages"]}}) for row in rows}) == 12
    return {"dataset": DATASET, "rows": rows, "planned_counts": dict(Counter(row["evolution_role"] for row in rows)),
            "scope": "six new greeting training contexts and six matching ordinary controls; no new independent evidence",
            "development_scope": "training coverage informed by v3 API greeting development failure; existing development/final unchanged",
            "teacher_calls": 0, "model_generations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Preregister, audit and insert only twelve training samples")
    parser.add_argument("--storage", default=DEFAULTS["storage_root"])
    parser.add_argument("--model", default=DEFAULTS["model_path"])
    args = parser.parse_args()
    plan = protocol()
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    research = Path(__file__).resolve().parents[1] / "research"
    summary = research / "luban0_v4_context.json"
    if summary.exists():
        raise FileExistsError("Supplement already applied; preserve its fixed protocol")
    root = require_storage(args.storage)
    manifest = json.loads((Path(args.model)/"upstream-file-manifest.json").read_text())
    entries = sorted((row["Path"], row["Sha256"]) for row in manifest["Data"]["Files"]
                     if row["Path"].endswith(".safetensors") or row["Path"] == "config.json")
    identity = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
    memory = Memory(root)
    memory.model_identity = identity
    before = memory.samples()
    old_contexts = {_normalized_messages(row) for row in before}
    old_systems = {message["content"] for row in before if row["purpose"] == "validation"
                   for message in json.loads(row["provenance"]).get("messages", []) if message["role"] == "system"}
    v3 = json.loads((research/"luban0_v3_replay.json").read_text())
    final_path = Path(v3["final_test_path"])
    final_bytes = final_path.read_bytes()
    final_rows = json.loads(final_bytes)
    final_contexts, final_systems = set(), set()
    for row in final_rows:
        provenance = row.get("provenance", {})
        provenance = json.loads(provenance) if isinstance(provenance, str) else provenance
        messages = provenance.get("messages") or row["messages"]
        final_contexts.add(_normalized_messages({"provenance": {"messages": messages}}))
        final_systems.update(message["content"] for message in messages if message["role"] == "system")
    for row in plan["rows"]:
        if _normalized_messages({"provenance": {"messages": row["messages"]}}) in old_contexts | final_contexts:
            raise ValueError("Supplement overlaps an existing or frozen final context")
        if row["messages"][0]["content"] in old_systems | final_systems:
            raise ValueError("Supplement copies a held-out system instruction")
    run = root/"experiments"/("luban0-v4-context-"+uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    plan.update(base_identity=identity, final_reference=str(final_path), final_sha256=hashlib.sha256(final_bytes).hexdigest())
    write_new(run/"protocol.json", plan)  # Fixed protocol before the first INSERT.
    write_new(run/"database-before.json", before)
    seeded = []
    for row in plan["rows"]:
        provenance = {"dataset": DATASET, "evolution_role": row["evolution_role"], "protocol_id": row["id"], "matched_pair": row["pair"],
                      "messages": row["messages"], "generation_check": row["generation_check"], "benchmark_run": str(run),
                      "rights_basis": "project-authored greeting and formal-task ground truth; no generated upstream continuation or paid API",
                      "verification_evidence": "authored targets pass finite string/semantic checks; same system context has an ordinary negative control",
                      "informed_development_failure": "v3 API greeting omitted 我是鲁班零号; development cases remain untouched"}
        sample_id = memory.add_sample(row["prompt"], row["answer"], source="human", purpose="training", approved=True,
                                      training_allowed=True, provenance=provenance)
        seeded.append({"protocol_id": row["id"], "sample_id": sample_id, "evolution_role": row["evolution_role"]})
    after = {row["id"]: row for row in memory.samples()}
    assert all(after[row["id"]] == row for row in before)
    assert final_path.read_bytes() == final_bytes
    from luban.evolution import consolidation_samples
    from types import SimpleNamespace
    counts = {key: len(rows) for key, rows in consolidation_samples(SimpleNamespace(memory=memory)).items()}
    report = {"dataset": DATASET, "run_directory": str(run), "base_identity": identity, "seeded": seeded,
              "effective_counts": counts, "teacher_calls": 0, "model_generations": 0,
              "development_and_final_unchanged": True, "final_reference": str(final_path), "final_sha256": plan["final_sha256"],
              "protocol_sha256": hashlib.sha256((run/"protocol.json").read_bytes()).hexdigest(), "scope": plan["scope"],
              "development_scope": plan["development_scope"]}
    write_new(run/"result.json", report)
    write_new(summary, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
