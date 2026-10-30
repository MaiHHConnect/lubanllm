import json
import unittest
from unittest.mock import patch

from luban.runtime import (_freeze_preservation_rows, _merge_preservation_rows,
                           _preservation_generation, _validate_preservation_row)


def protected_row(identifier="ordinary"):
    return {
        "id": identifier,
        "answer": "原始答案",
        "provenance": json.dumps({
            "preservation_only": True,
            "preservation_baseline": {"answer": "原始答案", "complete": True},
            "generation_check": {"type": "exact", "expected": "原始答案"},
        }, ensure_ascii=False),
    }


class ExpertPreservationTests(unittest.TestCase):
    def test_automatic_rows_cannot_be_replaced_by_explicit_rows(self):
        required = [protected_row("core-ordinary"), protected_row("parent-ordinary")]
        extra = protected_row("new-material")
        merged = _merge_preservation_rows(required, [extra])
        self.assertEqual([row["id"] for row in merged],
                         ["core-ordinary", "parent-ordinary", "new-material"])
        self.assertEqual(len(_merge_preservation_rows(required, [protected_row("core-ordinary")])), 2)
        conflicting = protected_row("core-ordinary")
        conflicting["answer"] = "篡改答案"
        with self.assertRaises(ValueError):
            _merge_preservation_rows(required, [conflicting])

    def test_branch_off_generation_is_frozen_as_exact_teacher(self):
        row = {"id": "ordinary", "answer": "", "purpose": "validation",
               "approved": True, "training_allowed": False,
               "provenance": json.dumps({"generation_check": {"type": "contains", "expected": "原"}})}
        frozen = _freeze_preservation_rows(
            [row], [{"id": "ordinary", "answer": "原始答案", "complete": True}],
            evidence="test")
        self.assertEqual(frozen[0]["answer"], "原始答案")
        value = json.loads(frozen[0]["provenance"])
        self.assertTrue(value["preservation_only"])
        self.assertEqual(value["preservation_teacher"], "core")
        self.assertEqual(value["generation_check"], {"type": "exact", "expected": "原始答案"})

        parent = _freeze_preservation_rows(
            [row], [{"id": "ordinary", "answer": "父专家答案", "complete": True}],
            evidence="test", teacher_mode="expert")
        self.assertEqual(json.loads(parent[0]["provenance"])["preservation_teacher"], "expert")

    @patch("luban.evolution._generation")
    def test_preservation_requires_exact_answer_and_completion_state(self, generation):
        generation.return_value = [{"id": "ordinary", "answer": "原始答案",
                                    "complete": True, "passed": False}]
        result = _preservation_generation(object(), [protected_row()])
        self.assertTrue(result[0]["passed"])
        self.assertTrue(result[0]["preserved"])

        generation.return_value = [{"id": "ordinary", "answer": "原始答案",
                                    "complete": False, "passed": False}]
        result = _preservation_generation(object(), [protected_row()])
        self.assertFalse(result[0]["passed"])

    @patch("luban.evolution._generation")
    def test_preservation_rejects_changed_text(self, generation):
        generation.return_value = [{"id": "ordinary", "answer": "身份回答",
                                    "complete": True, "passed": True}]
        result = _preservation_generation(object(), [protected_row()])
        self.assertFalse(result[0]["passed"])
        self.assertFalse(result[0]["preserved"])

    def test_persisted_parent_teacher_requires_immutable_fields(self):
        row = protected_row("parent")
        row.update(purpose="validation", approved=True, training_allowed=False)
        value = json.loads(row["provenance"])
        value["preservation_teacher"] = "expert"
        row["provenance"] = json.dumps(value, ensure_ascii=False)
        self.assertEqual(_validate_preservation_row(row, teacher="expert")["preservation_teacher"], "expert")
        value["preservation_teacher"] = "core"
        row["provenance"] = json.dumps(value, ensure_ascii=False)
        with self.assertRaises(ValueError):
            _validate_preservation_row(row, teacher="expert")
        value["preservation_teacher"] = "expert"
        row["answer"] = "被改写的训练标签"
        row["provenance"] = json.dumps(value, ensure_ascii=False)
        with self.assertRaises(ValueError):
            _validate_preservation_row(row, teacher="expert")


if __name__ == "__main__":
    unittest.main()
