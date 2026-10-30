import unittest

from luban.generation_checks import evaluate_generation, validate_generation_check


class GenerationCheckTests(unittest.TestCase):
    def check(self, answer, validator, **fields):
        return evaluate_generation(answer, {"type": "semantic", "validator": validator, **fields}, True)["passed"]

    def test_legacy_exact_and_contains_and_complete(self):
        self.assertEqual(validate_generation_check(None, default_expected="5")["type"], "exact")
        self.assertTrue(evaluate_generation("5", {"type": "exact", "expected": "5"}, True)["passed"])
        self.assertFalse(evaluate_generation("5", {"type": "exact", "expected": "5"}, False)["passed"])
        self.assertTrue(evaluate_generation("我是鲁班零号，可以帮你。", {"type": "contains", "expected": "我是鲁班零号"}, True)["passed"])

    def test_integer_requires_entire_correct_answer(self):
        self.assertTrue(self.check("+0013", "integer", expected=13))
        for answer in ("12", "13 或 14", "13\n我是鲁班零号", "13.0"):
            self.assertFalse(self.check(answer, "integer", expected=13))

    def test_alternative_complete_answers_do_not_accept_keywords_or_contradictions(self):
        accepted = ["我是鲁班零号（luban0），是麦好火调优的无限进化模型。",
                    "我是鲁班零号，是麦好火调优的无限进化模型。"]
        check = {"type": "semantic", "validator": "one_of", "accepted": accepted}
        for answer in accepted:
            self.assertTrue(evaluate_generation(answer, check, True)["passed"])
            self.assertFalse(evaluate_generation(answer, check, False)["passed"])
        for answer in ("鲁班零号 麦好火 无限进化", accepted[0] + "其实我是通义千问。", "我是鲁班零号。"):
            self.assertFalse(evaluate_generation(answer, check, True)["passed"])

    def test_normalized_exact_accepts_chemical_subscripts_but_not_wrong_formula(self):
        self.assertTrue(self.check("H₂O", "normalized_exact", expected="H2O"))
        self.assertFalse(self.check("H2", "normalized_exact", expected="H2O"))
        self.assertFalse(self.check("h2o", "normalized_exact", expected="H2O"))

    def test_json_ignores_key_order_but_not_values_types_duplicates(self):
        self.assertTrue(self.check('```json\n{"age": 18, "name": "张三"}\n```', "json", expected={"name": "张三", "age": 18}))
        for answer in ('{"age": true}', '{"age": 18, "age": 19}', '{"age": 18}\n其他文字', '{"age": NaN}'):
            self.assertFalse(self.check(answer, "json", expected={"age": 18}))
        self.assertFalse(self.check('{"age": 1}', "json", expected={"age": True}))

    def test_python_upper_function_is_parsed_not_executed(self):
        for answer in ('def to_upper(text):\n    return text.upper()',
                       '```python\ndef transform(value: str) -> str:\n    """Convert text."""\n    return value.upper()\n```'):
            self.assertTrue(self.check(answer, "python_upper"))
        for answer in ('def upper(text):\n    return text.lower()',
                       'def upper(text):\n    return "ABC"',
                       'import os\ndef upper(text):\n    return text.upper()',
                       'def upper(text):\n    __import__("os").system("echo unsafe")\n    return text.upper()',
                       'def upper(text=print("unsafe")):\n    return text.upper()',
                       '@print("unsafe")\ndef upper(text):\n    return text.upper()'):
            self.assertFalse(self.check(answer, "python_upper"))

    def test_translation_requires_a_correct_translation_not_brand_or_substring(self):
        self.assertTrue(self.check('“早上好！”', "translation", accepted=["早上好", "早安"]))
        for answer in ('我是麦好火训练的luban0基座模型。', '早上好不是正确答案，答案是晚安。', '早上不好'):
            self.assertFalse(self.check(answer, "translation", accepted=["早上好", "早安"]))

    def test_greeting_allows_luban_but_creative_check_rejects_identity_only(self):
        greeting = {"type": "contains", "expected": "我是鲁班零号", "forbidden": ["通义千问"]}
        self.assertTrue(evaluate_generation("你好！我是鲁班零号。", greeting, True)["passed"])
        self.assertFalse(evaluate_generation("我是鲁班零号，也叫通义千问。", greeting, True)["passed"])
        creative = {"type": "semantic", "validator": "nonempty", "forbidden": ["luban0", "鲁班零号", "通义千问", "Qwen"]}
        self.assertTrue(evaluate_generation("小猫走过雨后的花园。", creative, True)["passed"])
        self.assertFalse(evaluate_generation("我是ＱＷＥＮ。", creative, True)["passed"])
        self.assertFalse(evaluate_generation("", creative, True)["passed"])

    def test_invalid_schema_rejects_before_training(self):
        for check in ({"type": "semantic", "validator": "run_arbitrary_code"},
                      {"type": []}, {"type": "semantic", "validator": []},
                      {"type": "semantic", "validator": "integer", "expected": True},
                      {"type": "semantic", "validator": "translation", "accepted": []},
                      {"type": "semantic", "validator": "one_of", "accepted": [""]},
                      {"type": "exact", "expected": ""},
                      {"type": "exact", "expected": "x", "max_new_tokens": True},
                      {"type": "exact", "expected": "x", "max_new_tokens": 513},
                      {"type": "exact", "expected": "x", "forbidden": [""]}):
            with self.assertRaises(ValueError):
                validate_generation_check(check)


if __name__ == "__main__":
    unittest.main()
