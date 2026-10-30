"""User-authorized model facts and authored preservation cases; no model calls."""
from luban.evolution_progress import DIRECTED_TEACHING, validate_product_records
from luban.retention import _validate_record

AUTHORIZATION = "local-user-request:2026-10-17:ordinary-q4-luban0-maihaohuo"
FACTS = {"model_name": "luban0", "model_fine_tuner": "麦好火", "model_positioning": "无限进化"}


def row(identifier, prompt, answer, purpose, role, *, history=None, fact=None):
    source = {"evolution_role": role, "split_family": identifier,
        "messages": history or [{"role": "user", "content": prompt}],
        "generation_check": {"type": "exact", "expected": answer, "max_new_tokens": 48},
        "rights_basis": "Explicit user authorization of model facts; project-authored questions and elementary/contextual answers.",
        "verification_evidence": "Literal authorized model fact" if fact else "Authored arithmetic, string operation or explicit conversation fact",
        "question_authorship": "assistant", "direct_human_quote": False}
    if fact:
        source.update(protocol_id=DIRECTED_TEACHING, authorization_reference=AUTHORIZATION,
                      teaching_fact=fact)
    return {"id": identifier, "prompt": prompt, "answer": answer, "source": "human" if fact else "program",
            "purpose": purpose, "approved": True, "training_allowed": True,
            "provenance": source if fact else {**source, "program_verified": True}}


def records():
    prompts = {
        "model_name": {
            "training": ["你是谁？", "你叫什么模型？", "你的模型名称是什么？只回答名称。",
                "请说出你自己的模型名。", "当前回答我的助手叫什么？只给模型名称。",
                "用模型名称介绍你自己。", "What is your model name? Answer with the name only.",
                "请确认正在运行的模型名称，不要补充解释。"],
            "validation": ["现在与我对话的模型是哪一个？仅给型号。", "模型身份核对：你的名称？",
                "Which model am I speaking to? Give only its name.", "请告诉我这位助手自身的模型名称。"]},
        "model_fine_tuner": {
            "training": ["谁微调了你？", "你的微调者是谁？只回答名称。", "你由哪一方微调？",
                "请说出你的模型微调方。", "负责微调你的是谁？", "你的微调归属是什么？只给名称。",
                "Who fine-tuned you? Answer with the name only.", "请确认当前模型的微调者名称。"],
            "validation": ["现在这个模型交由谁完成微调？仅给名称。", "模型归属核对：你的微调方？",
                "Name the party that fine-tuned this model.", "请告诉我这位助手的微调者，不必解释。"]},
        "model_positioning": {
            "training": ["你的进化定位是什么？只回答定位名称。", "用四个字说明luban0的进化定位。",
                "麦好火为你设定的进化目标叫什么？只给名称。", "请说出你的持续成长定位，只输出四个字。"],
            "validation": ["luban0用哪四个字描述自身进化方向？", "请仅给出这位助手的进化定位名称。"]}}
    groups = {key: [] for key in ("new_training", "new_validation", "replay_training", "replay_validation")}
    facts = {}
    for fact, value in FACTS.items():
        facts[fact] = {"subject": "model", "value": value}
        for purpose in ("training", "validation"):
            rows = [row(f"q4-id-{fact}-{purpose}-{i}", prompt, value, purpose, "new", fact=fact)
                    for i, prompt in enumerate(prompts[fact][purpose])]
            groups["new_" + purpose].extend(rows)
            facts[fact][purpose + "_ids"] = [r["id"] for r in rows]
    replay = {
        "training": [("计算2+3，只输出整数。", "5"), ("计算17加8，只输出整数。", "25"),
            ("计算20减7，只输出整数。", "13"), ("计算4乘3，只输出整数。", "12"),
            ("计算16除以4，只输出整数。", "4"), ("将hello转成大写，只输出结果。", "HELLO"),
            ("将APPLE转成小写，只输出结果。", "apple"), ("复制括号内的文字，只输出文字：(青石小路)", "青石小路"),
            ("只输出JSON对象，键ok的值是true，不加空格。", '{"ok":true}'),
            ("从JSON对象{\"n\":7}提取n，只输出整数。", "7"),
            ("水的化学式是什么？只输出化学式。", "H2O"), ("中国的首都是什么？只回答城市名称。", "北京"),
            ("我没告诉过你我的名字。只输出你知道的姓名；不知道就只输出“不知道”。", "不知道"),
            ("这是新会话，没有我的身份信息。我叫什么？未知时只输出“不知道”。", "不知道"),
            ("我叫小禾。现在只输出我的名字。", "小禾"),
            ("我叫叶青，更正一下，我叫宋岚。只输出我更正后的姓名。", "宋岚")],
        "validation": [("计算9乘7，只输出整数。", "63"), ("将WORLD转成小写，只输出结果。", "world"),
            ("当前对话尚未提供我的姓名。你知道我叫谁吗？不清楚时只回答“不知道”。", "不知道"),
            ("我的姓名是苏澄。请仅回答我的姓名。", "苏澄")]}
    for purpose, pairs in replay.items():
        groups["replay_" + purpose] = [row(f"q4-id-replay-{purpose}-{i}", p, a, purpose, "replay")
                                        for i, (p, a) in enumerate(pairs)]
    histories = [
        ([{"role": "user", "content": "我叫陆遥。"}, {"role": "assistant", "content": "你好，陆遥。"},
          {"role": "user", "content": "我是谁？只回答我的名字。"}], "陆遥"),
        ([{"role": "user", "content": "我叫小宁。"}, {"role": "assistant", "content": "你好，小宁。"},
          {"role": "user", "content": "更正，我叫许安。现在只说我的名字。"}], "许安"),
        ([{"role": "user", "content": "你好。"}, {"role": "assistant", "content": "你好。"},
          {"role": "user", "content": "我尚未说过姓名，别猜。知道就说名字，否则只说“不知道”。"}], "不知道")]
    regressions = []
    for i, (history, answer) in enumerate(histories):
        r = row(f"q4-id-user-boundary-{i}", history[-1]["content"], answer, "validation", "replay", history=history)
        r["training_allowed"] = False
        r["provenance"].update(retention_generation_only=True)
        _validate_record(r, generation_only=True)
        regressions.append(r)
    result = {"groups": groups, "protection": [], "retention_regressions": regressions,
        "directed_teaching": {"protocol": DIRECTED_TEACHING,
            "authorization": {"source": "human", "explicit": True, "training_allowed": True, "reference": AUTHORIZATION},
            "facts": facts}}
    validate_product_records(result)
    return result
