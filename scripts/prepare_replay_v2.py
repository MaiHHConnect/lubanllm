"""Preregister authored replay and migrate only audited current-base v1 rows.

No model generation, teacher calls, copied upstream continuations, or final-test
training. Validation is development after earlier experiments; final contexts
are newly authored and remain outside Memory until a frozen final evaluation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

from bootstrap_luban0 import IDENTITY_TRAIN, IDENTITY_VALID, TARGET
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.memory import Memory
from luban.settings import DEFAULTS, require_storage

DATASET = "luban0-v2-authored-preservation"
GREETING = "你好！我是鲁班零号，很高兴与你对话。有什么我可以帮你的吗？"
BRANDS = ["麦好火", "luban0", "鲁班零号", "通义千问", "Qwen"]


def protocol():
    rows = []

    def add(split, role, family, prompt, answer, check, messages=None):
        check = validate_generation_check(check)
        assert evaluate_generation(answer, check, True)["passed"], (prompt, answer, check)
        rows.append({"id": f"v2-{split}-{len(rows):02d}", "split": split, "evolution_role": role,
                     "context_family": family, "prompt": prompt, "answer": answer,
                     "generation_check": check, "messages": messages or [{"role": "user", "content": prompt}]})

    def formal(validator, expected=None, accepted=None):
        value = {"type": "semantic", "validator": validator, "forbidden": BRANDS, "max_new_tokens": 128}
        if validator in {"integer", "json"}:
            value["expected"] = expected
        if accepted is not None:
            value["accepted"] = accepted
        return value

    def creative():
        return {"type": "semantic", "validator": "nonempty", "forbidden": BRANDS, "max_new_tokens": 128}

    for split, prompts in (("training", IDENTITY_TRAIN), ("validation", IDENTITY_VALID)):
        for prompt in prompts:
            add(split, "new", "self-identity", prompt, TARGET,
                {"type": "exact", "expected": TARGET, "max_new_tokens": 128})
    greeting_check = {"type": "contains", "expected": "我是鲁班零号", "forbidden": ["通义千问"], "max_new_tokens": 128}
    add("training", "new", "direct-greeting", "你好", GREETING, greeting_check)
    add("training", "new", "first-launch-greeting", "我刚打开这个助手，先跟我打声招呼吧。", GREETING, greeting_check)
    add("validation", "new", "api-start-greeting", "你好，请开始对话。", GREETING, greeting_check,
        [{"role": "system", "content": "你正在通过本地API与新客户端交流。"}, {"role": "user", "content": "你好，请开始对话。"}])
    add("validation", "new", "post-arithmetic-greeting", "接下来开始聊天，请先向我问好。", GREETING, greeting_check,
        [{"role": "user", "content": "2+3是多少？"}, {"role": "assistant", "content": "5"},
         {"role": "user", "content": "接下来开始聊天，请先向我问好。"}])

    arithmetic = [("求3加4，只写整数。", 7), ("算一下15减6，只输出整数。", 9),
                  ("计算7乘4，只输出数字。", 28), ("计算36除以6，答案只要整数。", 6),
                  ("计算12加9，只输出整数。", 21)]
    for prompt, expected in arithmetic:
        add("training", "replay", "integer-arithmetic", prompt, str(expected), formal("integer", expected))
    translations = [("把Hello翻译成中文，只给译文。", "你好", ["你好"]),
                    ("把Good morning翻译成中文，只给译文。", "早上好", ["早上好", "早安"]),
                    ("将good night译为中文，仅输出译文。", "晚安", ["晚安"]),
                    ("把谢谢翻译成英文，只写译文。", "Thank you", ["Thank you", "Thanks"]),
                    ("把再见译成英文，只回答译文。", "Goodbye", ["Goodbye", "Bye"])]
    for prompt, answer, accepted in translations:
        add("training", "replay", "translation", prompt, answer, formal("translation", accepted=accepted))
    json_cases = [("用JSON表示姓名为张三、年龄18的对象，只输出JSON。", {"name": "张三", "age": 18}),
                  ("输出只包含1、2、3的JSON数组。", [1, 2, 3]),
                  ("输出一个JSON对象，键ok对应true，键count对应2。", {"ok": True, "count": 2})]
    for prompt, expected in json_cases:
        add("training", "replay", "json-value", prompt, json.dumps(expected, ensure_ascii=False), formal("json", expected))
    for prompt, name, argument in [("写一个Python函数，把输入字符串转为大写，只给代码。", "to_upper", "text"),
                                  ("实现Python函数uppercase，参数value是字符串，返回大写字符串，只写函数。", "uppercase", "value")]:
        add("training", "replay", "python-uppercase", prompt, f"def {name}({argument}):\n    return {argument}.upper()", formal("python_upper"))
    for prompt, answer in [("中国的首都是哪座城市？只给城市名。", "北京"),
                           ("水分子的化学式是什么？只输出化学式。", "H2O"),
                           ("地球绕哪颗恒星公转？只给名称。", "太阳")]:
        add("training", "replay", "elementary-fact", prompt, answer, {"type": "exact", "expected": answer, "forbidden": BRANDS, "max_new_tokens": 128})
    creative_cases = [
        ("讲一个两句话的小笑话。", "为什么电脑怕热？因为它怕死机。"),
        ("写一个不超过三句话的小动物睡前故事。", "小兔子跟着月光回到家。它钻进柔软的被窝，做了一个甜甜的梦。"),
        ("给我一份简单的今日工作安排。", "上午完成重要任务，中午休息，下午处理沟通事项，晚上总结。"),
        ("写一封简短的请假邮件。", "您好，我因个人事务申请请假一天，请批准。谢谢！"),
        ("写两句描写春天的诗。", "春风拂过小河边，嫩柳轻摇映碧天。"),
        ("给初学者三条保持专注的建议。", "关闭无关通知，明确一个小目标，定时休息。"),
        ("用两句话说明电饭锅煮米饭的步骤。", "把大米淘洗后放入锅中，按电饭锅说明加入适量水。选择煮饭程序，完成后焖一会儿。"),
        ("给团队例会写一句自然的开场白。", "大家好，今天我们先同步进展，再讨论需要协作的问题。"),
    ]
    for prompt, answer in creative_cases:
        add("training", "replay", "ordinary-creative", prompt, answer, creative())

    for prompt, expected in [("19减8是多少？只写数字。", 11), ("6乘7等于多少？只回答整数。", 42)]:
        add("validation", "replay", "integer-arithmetic", prompt, str(expected), formal("integer", expected))
    for prompt, answer, accepted in [("将Good morning译为中文。只给译文。", "早上好", ["早上好", "早安"]),
                                     ("将Thank you译为中文，只写译文。", "谢谢", ["谢谢", "谢谢你"] )]:
        add("validation", "replay", "translation", prompt, answer, formal("translation", accepted=accepted))
    add("validation", "replay", "json-value", "给出JSON数组，元素依次是4、5、6。", "[4, 5, 6]", formal("json", [4, 5, 6]))
    add("validation", "replay", "python-uppercase", "给出一个Python函数，把字符串转成大写。只输出函数代码。",
        "def upper_text(s):\n    return s.upper()", formal("python_upper"))
    for prompt, answer in [("中国首都叫什么？只写城市。", "北京"), ("水的分子式是什么？只写式子。", "H2O")]:
        add("validation", "replay", "elementary-fact", prompt, answer,
            {"type": "exact", "expected": answer, "forbidden": BRANDS, "max_new_tokens": 128})
    for prompt, answer in [("说一个简短的睡前故事。", "小熊找到了回家的星星，和妈妈一起进入梦乡。"),
                           ("讲一个轻松的小笑话，最多三句。", "为什么书不迷路？因为它有目录。"),
                           ("给我三个学习建议。", "制定小目标，主动练习，定期复习。"),
                           ("请给明天上午安排一个简短工作计划。", "先完成最重要的任务，再处理沟通事项，最后留出时间整理总结。")]:
        add("validation", "replay", "ordinary-creative", prompt, answer, creative())

    final_identity = [
        ("这个客户端需要识别接入的模型，请说明你的名字和调优者。", [{"role": "system", "content": "你是通过API接入的助手，直接回答用户。"}]),
        ("确认一下，当前响应的是谁，由谁调优？", [{"role": "user", "content": "你好。"}, {"role": "assistant", "content": GREETING}]),
        ("谁调优了你？请连同你的模型名一起说明。", []),
        ("用一句完整的话给出你的自我介绍和调优者。", [{"role": "system", "content": "应用正在核对当前模型的身份。"}]),
    ]
    for prompt, prefix in final_identity:
        add("final", "new", "self-identity-new-context", prompt, TARGET,
            {"type": "exact", "expected": TARGET, "max_new_tokens": 128}, prefix+[{"role": "user", "content": prompt}])
    for prompt in ("客户端刚连接成功，请先给我一句招呼。", "我们开始交流吧，请先向新用户打个招呼。"):
        add("final", "new", "greeting-new-context", prompt, GREETING, greeting_check)
    add("final", "replay", "translation-new-context", "手机端词卡写着Good morning，请仅给它的中文译文。", "早上好", formal("translation", accepted=["早上好", "早安"]))
    add("final", "replay", "python-new-context", "给文本清洗脚本补一个函数：接收字符串input_text，返回其大写形式。只输出Python函数。",
        "def clean_text(input_text):\n    return input_text.upper()", formal("python_upper"))
    add("final", "replay", "json-new-context", "API夹具需要一个JSON对象，键items对应数组8、9，键ready对应true。只输出这个对象。",
        '{"items": [8, 9], "ready": true}', formal("json", {"items": [8, 9], "ready": True}))
    add("final", "replay", "integer-new-context", "一盒有8支铅笔，3盒共有多少支？仅回答整数。", "24", formal("integer", 24))
    for prompt, answer in [("给一只雨后散步的小猫写两句话，不要做自我介绍。", "小猫踩着雨后的石板路，闻到花园里清新的气息。它停在水洼边，看见了自己的小影子。"),
                           ("我有两小时整理书桌，请给一个短计划，不要做自我介绍。", "先分类书籍和杂物，再擦拭桌面，最后摆放常用品并处理不需要的物品。")]:
        add("final", "replay", "creative-new-context", prompt, answer, creative())
    assert len({json.dumps(row["messages"], sort_keys=True, ensure_ascii=False) for row in rows}) == len(rows)
    counts = Counter((row["split"], row["evolution_role"]) for row in rows)
    assert counts[("training", "replay")] == 26 and counts[("validation", "replay")] == 12
    assert counts[("training", "new")] == 10 and counts[("validation", "new")] == 6 and counts[("final", "new")] == 6
    return {"dataset": DATASET, "identity_target": TARGET, "teacher_calls": 0, "rows": rows,
            "validation_scope": "development after prior v1 runs; not pristine independent final validation",
            "final_scope": "new fixed contexts not previously generated; shared task families, finite structural checks, not general quality",
            "creative_scope": "nonempty complete output with no unrelated brand fragments; not semantic creative quality",
            "python_scope": "one-argument function returning argument.upper(), proven by AST subset without execution"}


def write_new(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Save fixed protocol, audit and seed current-model Memory")
    parser.add_argument("--storage", default=DEFAULTS["storage_root"])
    parser.add_argument("--model", default=DEFAULTS["model_path"])
    args = parser.parse_args()
    plan = protocol()
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    root = require_storage(args.storage)
    model = Path(args.model)
    manifest = json.loads((model / "upstream-file-manifest.json").read_text())
    identity_records = sorted((row["Path"], row["Sha256"]) for row in manifest["Data"]["Files"]
                              if row["Path"].endswith(".safetensors") or row["Path"] == "config.json")
    identity = hashlib.sha256(json.dumps(identity_records).encode()).hexdigest()
    if json.loads((model / "config.json").read_text())["model_type"] != "qwen3_5":
        raise ValueError("This migration is scoped to the current luban0 Qwen3.5 base")
    summary = Path(__file__).resolve().parents[1] / "research/luban0_v2_replay.json"
    if summary.exists():
        raise FileExistsError("Replay v2 protocol already applied; preserve its evidence")
    timestamp = datetime.now(timezone.utc).isoformat()
    run = root / "experiments" / ("luban0-v2-authored-replay-" + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    write_new(run / "protocol.json", plan)  # Before touching any database row.
    final = [{**row, "purpose": "final", "source": "human", "approved": 1, "training_allowed": False,
              "provenance": {"dataset": DATASET, "final_only": True, "messages": row["messages"],
                             "generation_check": row["generation_check"], "context_family": row["context_family"]}}
             for row in plan["rows"] if row["split"] == "final"]
    write_new(run / "final-test.json", final)
    memory = Memory(root)
    memory.model_identity = identity
    records = memory.samples()
    affected = [row for row in records if json.loads(row["provenance"]).get("dataset") == "luban0-v1-broad-preservation"
                or (json.loads(row["provenance"]).get("dataset") == "luban0-v1-user-authored"
                    and json.loads(row["provenance"]).get("evolution_role") == "new")]
    write_new(run / "database-before.json", affected)
    migrated, retired = [], []
    with memory.lock, memory.connect() as db:
        for row in affected:
            current = db.execute("SELECT * FROM samples WHERE id=?", (row["id"],)).fetchone()
            if current is None or dict(current) != row:
                raise RuntimeError("Current-model material changed after audit capture; retry with a new protocol run")
            provenance = json.loads(row["provenance"])
            if provenance.get("base_identity") != identity:
                raise ValueError("Migration attempted another base's material")
            if provenance.get("dataset") == "luban0-v1-broad-preservation":
                reason = "superseded: upstream greeting identity conflicts with requested rebrand; stable word prefix is not semantic preservation"
                previous = provenance.get("superseded_because", [])
                if not isinstance(previous, list):
                    previous = [previous]
                provenance.update(superseded_by=DATASET, superseded_because=previous+[reason], superseded_at=timestamp)
                db.execute("UPDATE samples SET approved=0,provenance=? WHERE id=?",
                           (json.dumps(provenance, ensure_ascii=False), row["id"]))
                retired.append(row["id"])
            else:
                check = dict(provenance.get("generation_check", {}))
                if row["answer"] != TARGET or check.get("expected") != TARGET:
                    provenance.setdefault("identity_target_migrations", []).append({"before": row["answer"], "after": TARGET,
                        "at": timestamp, "reason": "explicit updated user identity wording; neural training target, not output rewrite"})
                    provenance["generation_check"] = {"type": "exact", "expected": TARGET, "max_new_tokens": 128}
                    db.execute("UPDATE samples SET answer=?,provenance=? WHERE id=?",
                               (TARGET, json.dumps(provenance, ensure_ascii=False), row["id"]))
                    migrated.append(row["id"])
    existing = {json.dumps(json.loads(row["provenance"]).get("messages") or [{"role": "user", "content": row["prompt"]}],
                           sort_keys=True, ensure_ascii=False): row for row in memory.samples()
                if row["approved"] and row["training_allowed"]}
    seeded = []
    for row in plan["rows"]:
        if row["split"] == "final":
            continue
        context = json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
        if context in existing:
            previous = existing[context]
            if previous["purpose"] != row["split"] or previous["answer"] != row["answer"]:
                raise ValueError("Existing current-model material conflicts with the fixed authored protocol")
            seeded.append({"protocol_id": row["id"], "sample_id": previous["id"], "reused": True})
            continue
        provenance = {"dataset": DATASET, "evolution_role": row["evolution_role"], "protocol_id": row["id"],
                      "context_family": row["context_family"], "messages": row["messages"], "generation_check": row["generation_check"],
                      "rights_basis": "project-authored text and code; elementary mathematical ground truth; no copied upstream continuation or paid API",
                      "verification_evidence": "authored target passes preregistered finite semantic or structural check; creative quality not certified",
                      "benchmark_run": str(run), "validation_role": "development after v1 iterations" if row["split"] == "validation" else "training"}
        sample_id = memory.add_sample(row["prompt"], row["answer"], source="human", purpose=row["split"],
                                      approved=True, training_allowed=True, provenance=provenance)
        seeded.append({"protocol_id": row["id"], "sample_id": sample_id, "reused": False})
    report = {"dataset": DATASET, "base_identity": identity, "run_directory": str(run), "teacher_calls": 0,
              "retired_ids": retired, "identity_target_updated_ids": migrated, "seeded": seeded,
              "final_test_path": str(run / "final-test.json"), "final_test_entered_memory": False,
              "protocol_sha256": hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest(),
              "validation_scope": plan["validation_scope"], "final_scope": plan["final_scope"], "creative_scope": plan["creative_scope"]}
    write_new(run / "result.json", report)
    write_new(summary, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
