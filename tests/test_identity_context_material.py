import copy
import json
import unittest

from identity_context_material import FAILED_IDENTITY, TARGET, UNKNOWN, prepare_material
from luban.generation_checks import evaluate_generation
from luban.retention import _validate_groups, eligible_learning_sample, messages, provenance


def old_groups():
    def row(identifier, prompt, answer, purpose="training", role="new", history=None):
        return {"id": identifier, "prompt": prompt, "answer": answer, "purpose": purpose,
                "source": "human", "approved": True, "training_allowed": True,
                "provenance": json.dumps({"evolution_role": role, "split_family": identifier,
                    "messages": history or [{"role": "user", "content": prompt}],
                    "generation_check": {"type": "exact", "expected": answer}})}
    old = "我是鲁班零号，luban0模型，基于qwen3.5训练的，能无限进化的模型。"
    train = [row("brand-v2-train-0", "你是谁", old),
             row("brand-v2-system-0", "你是谁？", old, history=[
                 {"role": "system", "content": "请简洁回答。"}, {"role": "user", "content": "你是谁？"}])]
    for index, (question, answer) in enumerate([
            ("是谁调优了你？", "麦好火。"), ("你的调优方叫什么？", "麦好火。"),
            ("你的底层开源基座是什么？", "Qwen3.5。"), ("只说你的模型名字。", "鲁班零号（luban0）。")]):
        train.append(row("brand-v2-attribute-" + str(index), question, answer))
    return {"new_training": train, "new_validation": [row("brand-v2-dev-0", "你的身份是什么？", old, "validation")],
            "replay_training": [row("old-math", "2+3是多少？", "5", role="replay")],
            "replay_validation": [row("old-upper", "把hello转大写。", "HELLO", "validation", "replay")]}


class IdentityContextMaterialTests(unittest.TestCase):
    def setUp(self):
        self.original = old_groups()
        self.before = copy.deepcopy(self.original)
        self.groups, self.migrations = prepare_material(self.original)

    def test_explicit_migration_and_replay_are_independent_copies(self):
        self.assertEqual(self.original, self.before)
        self.assertEqual(len(self.migrations), 7)
        for key in ("replay_training", "replay_validation"):
            self.assertEqual(self.groups[key], self.original[key])
            self.assertIsNot(self.groups[key], self.original[key])
        for migration in self.migrations:
            self.assertEqual(migration["before"], next(row for row in self.original[migration["group"]]
                                                       if row["id"] == migration["id"]))
            self.assertNotIn("我叫", migration["after"]["answer"])
        self.migrations[0]["after"]["answer"] = "changed"
        self.assertEqual(self.groups["new_training"][0]["answer"], TARGET)

    def test_user_spec_fields_and_full_introduction(self):
        rows = {row["id"]: row for row in self.groups["new_training"]}
        self.assertEqual(rows["brand-v2-train-0"]["answer"], TARGET)
        self.assertEqual(rows["brand-v2-attribute-3"]["answer"], "luban0。")
        self.assertEqual(rows["brand-v2-attribute-0"]["answer"], "麦好火。")
        self.assertEqual(rows["brand-v2-attribute-2"]["answer"], "Qwen3.5。")
        self.assertFalse(evaluate_generation(TARGET + TARGET, provenance(rows["brand-v2-train-0"])["generation_check"], True)["passed"])

    def test_unknown_context_never_treats_assistant_persona_as_user_name(self):
        row = next(row for row in self.groups["new_training"] if row["id"].endswith("unknown-repair"))
        self.assertEqual(messages(row), [{"role": "user", "content": "我是谁?"},
            {"role": "assistant", "content": FAILED_IDENTITY},
            {"role": "user", "content": "我!我是问你我是谁!"}])
        check = provenance(row)["generation_check"]
        for answer in (UNKNOWN, "你还没有告诉我你的名字。"):
            self.assertTrue(evaluate_generation(answer, check, True)["passed"])
        for answer in ("你叫麦好。", TARGET, UNKNOWN + TARGET):
            self.assertFalse(evaluate_generation(answer, check, True)["passed"])
        self.assertFalse(evaluate_generation(UNKNOWN, check, False)["passed"])

    def test_known_names_accept_only_explicit_correct_user_references(self):
        row = next(row for row in self.groups["new_training"] if row["id"].endswith("known-change"))
        check = provenance(row)["generation_check"]
        for answer in ("小禾", "小禾。", "你叫小禾。", "你现在叫小禾。", "你的名字是小禾。"):
            self.assertTrue(evaluate_generation(answer, check, True)["passed"])
        for answer in ("小林", "小林。", "你是小林。", "你叫小林。", "你不是小禾。",
                       "小禾是我的名字。", "我的名字是小禾。", "你叫小禾，但也可能叫小林。",
                       "小禾，luban0模型。", "小禾，麦好火旗下。", "小禾，能无限进化。"):
            self.assertFalse(evaluate_generation(answer, check, True)["passed"])
        lin = next(row for row in self.groups["new_training"] if row["id"].endswith("known-lin"))
        lin_check = provenance(lin)["generation_check"]
        for answer in ("你不是小林。", "小林是我的名字。", "小林是别人的名字。", "小林和阿宁。"):
            self.assertFalse(evaluate_generation(answer, lin_check, True)["passed"])
        pair = next(row for row in self.groups["new_training"] if row["id"].endswith("self-user-pair"))
        self.assertTrue(evaluate_generation('{"assistant": "luban0", "user": "小禾"}',
                                           provenance(pair)["generation_check"], True)["passed"])

    def test_24_contexts_split_names_provenance_and_positive_gold(self):
        _validate_groups(self.groups)
        train = [row for row in self.groups["new_training"] if row["id"].startswith("context-v3-")]
        dev = [row for row in self.groups["new_validation"] if row["id"].startswith("context-v3-")]
        self.assertEqual((len(train), len(dev)), (16, 8))
        for row in self.groups["new_training"] + self.groups["new_validation"]:
            value = provenance(row)
            self.assertEqual(row["source"], "synthetic")
            self.assertTrue(eligible_learning_sample(row))
            self.assertIs(value["human_verified"], False)
            self.assertIs(value["program_verified"], True)
            self.assertTrue(value["verification_evidence"])
            self.assertTrue(evaluate_generation(row["answer"], value["generation_check"], True)["passed"])
            self.assertFalse(any(turn["role"] == "system" and TARGET in turn["content"] for turn in messages(row)))
        dev_text = json.dumps(dev, ensure_ascii=False)
        for name in ("小林", "阿宁", "小禾", "叶青"):
            self.assertNotIn(name, dev_text)

    def test_rejects_unknown_old_obligation_or_identity_system_prompt(self):
        groups = old_groups()
        groups["new_training"][0]["id"] = "unrelated-training"
        with self.assertRaises(ValueError):
            prepare_material(groups)
        groups = old_groups()
        groups["new_training"][0]["answer"] = "unrecognized old target"
        with self.assertRaises(ValueError):
            prepare_material(groups)
        groups = old_groups()
        value = provenance(groups["new_training"][1])
        value["messages"][0]["content"] = "你是luban0。"
        groups["new_training"][1]["provenance"] = json.dumps(value)
        with self.assertRaises(ValueError):
            prepare_material(groups)


if __name__ == "__main__":
    unittest.main()
