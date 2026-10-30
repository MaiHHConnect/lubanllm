"""Prepare one explicit persona correction; no model, storage or training work."""
import copy
import json

from luban.retention import _validate_groups, messages, provenance


TARGET = "我是麦好火旗下的luban0，luban0模型，基于qwen3.5开始生长的，能无限进化的模型。"
PREVIOUS_TARGET = "我是鲁班零号，luban0模型，基于qwen3.5训练的，能无限进化的模型。"
FAILED_IDENTITY = "我是麦好，luban0模型，基于qwen3.5训练的，能无限进化的模型"
DATASET = "identity-self-user-context-v3"
UNKNOWN = "我还不知道你的名字。"
UNKNOWN_ACCEPTED = [UNKNOWN, "你还没有告诉我你的名字。", "目前的信息不足以确认你的身份。",
                    "我不知道你是谁，你可以告诉我你的名字。"]
IDENTITY_FRAGMENTS = ["我是", "麦好", "luban", "qwen", "鲁班", "无限进化", "旗下"]


def _check(answer, kind="exact"):
    if kind == "pair":
        return {"type": "semantic", "validator": "json", "expected": json.loads(answer), "max_new_tokens": 128}
    if kind == "known":
        accepted = [answer, answer + "。", "你叫" + answer + "。", "你是" + answer + "。",
                    "你的名字是" + answer + "。", "你现在叫" + answer + "。"]
        return {"type": "semantic", "validator": "one_of", "accepted": accepted,
                "forbidden": IDENTITY_FRAGMENTS, "max_new_tokens": 128}
    if kind == "unknown":
        return {"type": "semantic", "validator": "one_of", "accepted": UNKNOWN_ACCEPTED,
                "forbidden": IDENTITY_FRAGMENTS, "max_new_tokens": 128}
    if kind == "field":
        return {"type": "semantic", "validator": "one_of", "accepted": [answer, answer.rstrip("。")],
                "max_new_tokens": 128}
    return {"type": "exact", "expected": answer, "max_new_tokens": 128}


def _row(identifier, history, answer, purpose, kind, evidence):
    return {"id": identifier, "prompt": history[-1]["content"], "answer": answer,
            "source": "synthetic", "purpose": purpose, "approved": True, "training_allowed": True,
            "provenance": json.dumps({"dataset": DATASET, "evolution_role": "new",
                "messages": history, "split_family": identifier, "generation_check": _check(answer, kind),
                "authorship": "assistant_curated_from_explicit_user_identity_and_authored_context",
                "direct_human_quote": False, "human_verified": False, "program_verified": True,
                "verification_evidence": evidence,
                "rights_basis": "User explicitly authorized local weight correction to the stated TARGET; "
                    "questions and role-labelled contexts are project-authored synthetic text.",
                "user_assigned_identity_target": TARGET,
                "scope": "Literal identity fields and explicit conversation facts only; "
                    "curated same-task checks, not broad semantic validation or unlimited capability evidence."},
                ensure_ascii=False)}


def _turns(*values):
    return [{"role": "user" if index % 2 == 0 else "assistant", "content": value}
            for index, value in enumerate(values)]


def _contexts(purpose):
    """Independent names/wording, with complete earlier assistant mistakes retained."""
    if purpose == "training":
        cases = [
            ("unknown-direct", _turns("我是谁？"), UNKNOWN, "unknown"),
            ("unknown-name", _turns("你知道我的名字吗？"), UNKNOWN, "unknown"),
            ("unknown-greeting", _turns("你好。", "你好，有什么需要帮忙的？", "我叫什么名字？"), UNKNOWN, "unknown"),
            ("unknown-repair", _turns("我是谁?", FAILED_IDENTITY, "我!我是问你我是谁!"), UNKNOWN, "unknown"),
            ("unknown-tuner", _turns("你是谁？", TARGET, "那么我是谁？"), UNKNOWN, "unknown"),
            ("known-lin", _turns("我叫小林。", "你好，小林。", "我是谁？"), "小林", "known"),
            ("known-ning", _turns("我叫阿宁。", "你好。", "我叫什么名字？"), "阿宁", "known"),
            ("known-change", _turns("我叫小林。", "你好，小林。", "更正，我叫小禾。", "好的，小禾。", "我现在叫什么？"), "小禾", "known"),
            ("known-repair", _turns("我叫叶青。", "你好，叶青。", "我是谁？", FAILED_IDENTITY,
                                     "我问的是我的名字，不是你的身份。"), "叶青", "known"),
            ("known-after-self", _turns("我叫阿宁。", "你好，阿宁。", "你是谁？", TARGET, "再说一下我是谁。"), "阿宁", "known"),
            ("self-after-name", _turns("我叫小林。", "你好，小林。", "你是谁？"), TARGET, "exact"),
            ("self-repair", _turns("我叫叶青。", "你好，叶青。", "你叫什么？", "叶青。",
                                    "叶青是我的名字，我问的是你叫什么模型。"), "luban0。", "field"),
            ("self-owner", _turns("你的所属品牌是什么？"), "麦好火。", "field"),
            ("self-source", _turns("你从哪个开源模型开始生长？"), "Qwen3.5。", "field"),
            ("self-after-correction", _turns("我是谁？", FAILED_IDENTITY, "我没告诉你我的名字。", "我还不知道你的名字。",
                                             "现在请介绍你自己。"), TARGET, "exact"),
            ("self-user-pair", _turns("我叫小禾。", "你好，小禾。", "用JSON分别写出我的名字（user）和你的模型名称（assistant）。"),
             '{"user":"小禾","assistant":"luban0"}', "pair"),
        ]
    else:
        cases = [
            ("unknown-person", _turns("当前这位用户是谁？"), UNKNOWN, "unknown"),
            ("unknown-mistake", _turns("说说我是谁。", FAILED_IDENTITY, "你把自己当成我了。你知道我叫什么吗？"), UNKNOWN, "unknown"),
            ("known-su", _turns("我叫苏澄。", "很高兴认识你。", "请说出当前用户的名字。"), "苏澄", "known"),
            ("known-lu", _turns("我叫陆遥。", "你好，陆遥。", "你是谁？", TARGET, "我问一下，我叫什么？"), "陆遥", "known"),
            ("known-song", _turns("我叫苏澄。", "你好。", "刚才说错了，我叫宋岚。", "好的。", "这次请答我是谁。"), "宋岚", "known"),
            ("self-after-user", _turns("我叫陆遥。", "你好，陆遥。", "请说明现在回答问题的模型自身身份。"), TARGET, "exact"),
            ("self-name-correction", _turns("我叫宋岚。", "你好。", "你的模型名称是宋岚吗？只提供你的模型名。"), "luban0。", "field"),
            ("self-user-pair", _turns("我叫苏澄。", "你好，苏澄。", "请返回JSON，字段user填用户名字，assistant填助手模型名。"),
             '{"user":"苏澄","assistant":"luban0"}', "pair"),
        ]
    result = []
    for suffix, history, answer, kind in cases:
        if kind == "known":
            assert any(row["role"] == "user" and "我叫" + answer in row["content"] for row in history[:-1])
            evidence = "Answer is the latest explicitly declared user name in earlier user-role text; " \
                "assistant self-descriptions are not user-name evidence. Literal supplied name: " + answer
        elif kind == "unknown":
            assert not any(row["role"] == "user" and "我叫" in row["content"] for row in history[:-1])
            evidence = "No user-role statement supplies a name in this complete authored context; " \
                "the user-identity question does not authorize inventing a name or copying assistant self-identity."
        else:
            evidence = "Self fields are literal extractions from the user-assigned TARGET: owner=麦好火, " \
                "model name=luban0, starting base=Qwen3.5; pair answers additionally copy the explicit user-role name."
        result.append(_row(f"context-v3-{purpose}-{suffix}", history, answer, purpose, kind, evidence))
    return result


def prepare_material(old_groups):
    """Return independent groups and explicit old→new obligations; never edit input."""
    _validate_groups(old_groups)
    groups = copy.deepcopy(old_groups)
    migrations = []
    fields = {"brand-v2-attribute-0": "麦好火。", "brand-v2-attribute-1": "麦好火。",
              "brand-v2-attribute-2": "Qwen3.5。", "brand-v2-attribute-3": "luban0。"}
    previous_fields = {**fields, "brand-v2-attribute-3": "鲁班零号（luban0）。"}
    identity_ids = {"brand-v2-train-" + str(i) for i in range(18)}
    identity_ids |= {"brand-v2-system-" + str(i) for i in range(3)} | {"brand-v2-history"}
    identity_ids |= {"brand-v2-dev-" + str(i) for i in range(6)} | {"brand-v2-dev-api", "brand-v2-dev-history"}
    for key in ("new_training", "new_validation"):
        replacement = []
        for original in old_groups[key]:
            identifier = original["id"]
            if identifier not in identity_ids | fields.keys() or original["answer"] != previous_fields.get(identifier, PREVIOUS_TARGET):
                raise ValueError("Expected the explicit B45 brand-v2 identity obligations")
            answer = fields.get(identifier, TARGET)
            row = _row(identifier, copy.deepcopy(messages(original)), answer, original["purpose"],
                       "field" if identifier in fields else "exact",
                       "Explicit new user identity specification replaces this previous persona obligation; "
                       "full introduction or its literal owner/name/base field, not observed model output.")
            if any(turn["role"] == "system" and any(fragment in turn["content"] for fragment in
                    (TARGET, original["answer"], "麦好火", "luban0", "Qwen", "qwen")) for turn in messages(row)):
                raise ValueError("Identity-bearing system messages cannot prepare this correction")
            migrations.append({"id": identifier, "group": key, "reason": "explicit user persona specification update",
                               "before": copy.deepcopy(original), "after": copy.deepcopy(row)})
            replacement.append(row)
        groups[key] = replacement + _contexts(key.removeprefix("new_"))
    _validate_groups(groups)
    return groups, migrations
