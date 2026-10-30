"""Sample deduplication and split history belong to the active base identity."""
import hashlib
import json
import tempfile
import unittest

from luban.memory import Memory


class MemoryIdentityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.memory = Memory(temp.name)

    def add(self, prompt="同一个问题", answer="同一个答案", purpose="training", **kwargs):
        return self.memory.add_sample(prompt, answer, purpose=purpose, approved=True,
                                      training_allowed=True, **kwargs)

    def capture(self, prompt="同一个问题", answer="同一个答案"):
        material_id = self.memory.capture_teacher([{"role": "user", "content": prompt}], answer, {}, True)
        return next(row for row in self.memory.teacher_materials() if row["id"] == material_id)

    def test_identical_prompt_on_another_base_can_have_an_independent_split(self):
        self.memory.model_identity = "old-base"
        old = self.add(purpose="validation")
        self.memory.model_identity = "new-base"
        new = self.add(purpose="training", provenance={"base_identity": "forged-old-base"})
        self.assertNotEqual(old, new)
        self.assertEqual([row["id"] for row in self.memory.samples()], [new])
        self.assertEqual(json.loads(self.memory.samples()[0]["provenance"])["base_identity"], "new-base")
        self.memory.model_identity = "old-base"
        self.assertEqual([row["id"] for row in self.memory.samples()], [old])

    def test_same_base_cross_split_guard_remains_in_both_directions(self):
        self.memory.model_identity = "current-base"
        for original, other in (("training", "validation"), ("validation", "training")):
            self.add(prompt=original, purpose=original)
            with self.assertRaisesRegex(ValueError, "训练和检验"):
                self.add(prompt=" " + original + " ", purpose=other)

    def test_unscoped_legacy_sample_does_not_block_a_scoped_sample(self):
        old = self.add(purpose="validation")
        self.memory.model_identity = "current-base"
        current = self.add(purpose="training")
        self.assertNotEqual(old, current)
        self.assertEqual([row["id"] for row in self.memory.samples()], [current])

    def test_teacher_capture_does_not_reuse_other_base_id_or_split(self):
        prompt = "教师去重测试"
        split = "validation" if int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 5 == 0 else "training"
        other_split = "training" if split == "validation" else "validation"
        self.memory.model_identity = "old-base"
        old = self.add(prompt=prompt, purpose=other_split)
        self.memory.model_identity = "new-base"
        captured = self.capture(prompt=prompt)
        self.assertNotEqual(captured["sample_id"], old)
        self.assertEqual(captured["purpose"], split)
        sample = self.memory.samples()[0]
        self.assertEqual(sample["id"], captured["sample_id"])
        self.assertEqual(json.loads(sample["provenance"])["base_identity"], "new-base")
        self.memory.model_identity = "old-base"
        self.assertEqual(self.memory.samples()[0]["id"], old)
        self.assertEqual(self.memory.samples()[0]["purpose"], other_split)

    def test_teacher_capture_reuses_same_base_id_and_manual_split(self):
        self.memory.model_identity = "current-base"
        existing = self.add(purpose="validation")
        first = self.capture()
        second = self.capture(answer="不同教师答案")
        self.assertEqual(first["sample_id"], existing)
        self.assertNotEqual(second["sample_id"], existing)
        self.assertEqual(first["purpose"], "validation")
        self.assertEqual(second["purpose"], "validation")
        self.assertEqual(len(self.memory.samples()), 2)

    def test_retired_scoped_record_is_preserved_but_not_reused(self):
        self.memory.model_identity = "current-base"
        existing = self.add(purpose="validation")
        with self.memory.lock, self.memory.connect() as db:
            db.execute("UPDATE samples SET approved=0 WHERE id=?", (existing,))
        captured = self.capture()
        self.assertNotEqual(captured["sample_id"], existing)
        self.assertEqual(captured["purpose"], "validation")
        rows = {row["id"]: row for row in self.memory.samples()}
        self.assertEqual(rows[existing]["approved"], 0)
        self.assertEqual(rows[captured["sample_id"]]["approved"], 1)
        with self.assertRaisesRegex(ValueError, "训练和检验"):
            self.add(purpose="training")

    def test_missing_identity_retains_legacy_global_semantics(self):
        self.memory.model_identity = "some-base"
        old = self.add(purpose="validation")
        self.memory.model_identity = None
        with self.assertRaisesRegex(ValueError, "训练和检验"):
            self.add(purpose="training")
        captured = self.capture()
        self.assertEqual(captured["sample_id"], old)
        self.assertEqual(captured["purpose"], "validation")


if __name__ == "__main__":
    unittest.main()
