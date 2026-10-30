import copy
import json
import unittest
from types import SimpleNamespace

from luban.retention import retention_loss_records
from luban.runtime import Runtime


class RetentionLossTargetsTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "ordinary", "prompt": "2+2?", "answer": "", "training_allowed": False,
                    "provenance": json.dumps({"generation_check": {"type": "semantic", "validator": "integer", "expected": 4}})}
        self.observed = {"id": "ordinary", "answer": "4", "passed": True, "complete": True,
                         "branch_enabled": False, "reference_used": False}

    def test_actual_tokenizer_consumer_has_answer_and_eos_for_both_observations(self):
        original = copy.deepcopy(self.row)
        targets = retention_loss_records([self.row], [self.observed])
        tokenizer = SimpleNamespace(eos_token_id=99, apply_chat_template=lambda *a, **kw: "prompt",
                                    encode=lambda text, **kw: [7, 8] if text == "prompt" else [4] if text == "4" else [])
        runtime = SimpleNamespace(tokenizer=tokenizer, device="cpu")
        _, old_labels = Runtime._sample_tokens(runtime, self.row)
        self.assertEqual(old_labels[old_labels != -100].tolist(), [99])
        for _ in ("baseline", "candidate"):
            _, labels = Runtime._sample_tokens(runtime, targets[0])
            self.assertEqual(labels[labels != -100].tolist(), [4, 99])
        self.assertEqual(self.row, original)
        self.assertFalse(targets[0]["training_allowed"])

    def test_rejects_bad_or_missing_observations(self):
        for changes in ({"answer": "5"}, {"answer": ""}, {"answer": " "}, {"complete": False},
                        {"passed": False}, {"branch_enabled": True}, {"reference_used": True}, {"reference_used": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                retention_loss_records([self.row], [{**self.observed, **changes}])
        with self.assertRaises(ValueError):
            retention_loss_records([self.row], [])
        with self.assertRaises(ValueError):
            retention_loss_records([self.row], [self.observed, self.observed])

    def test_authored_answer_not_replaced_and_result_detached(self):
        row = {**self.row, "answer": " 4 "}
        result = retention_loss_records([row], [])
        self.assertEqual(result[0]["answer"], " 4 ")
        result[0]["answer"] = "changed"
        self.assertEqual(row["answer"], " 4 ")


if __name__ == "__main__":
    unittest.main()
