"""Isolated CPU data qualification and SQLite clone checks; no model download."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_identity_coverage import clone_research_database, qualify_original_replay_pool
from verify_luban0_final import tensor_digest
from verify_luban0_product import content_digest
from luban.memory import Memory


class IdentityCoverageTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.tensor([[1., 2.], [3., 4.]])
        self.runtime = SimpleNamespace(active_core=None, identity="fixture-base")
        self.pool = [{"id": f"pool-{i}", "prompt": f"Uppercase word{i}", "answer": f"WORD{i}",
            "source": "synthetic", "approved": 0, "training_allowed": True, "purpose": "training",
            "provenance": json.dumps({"evolution_role": "replay", "program_verified": True,
                "verification_evidence": f"Python str.upper('word{i}') = 'WORD{i}'",
                "messages": [{"role": "user", "content": f"Uppercase word{i}"}],
                "generation_check": {"type": "exact", "expected": f"WORD{i}"}})} for i in range(12)]
        self.plan = {"base_identity": self.runtime.identity, "original_tensor_sha256": tensor_digest(self.weight),
            "replay_training_candidates": self.pool, "replay_candidate_sha256": content_digest(self.pool),
            "minimum_replay_candidates": 4}

    def observations(self, passes):
        return [{"id": row["id"], "answer": row["answer"] if i < passes else "wrong",
                 "passed": i < passes, "complete": True, "branch_enabled": False}
                for i, row in enumerate(self.pool)]

    def qualify(self, observations):
        with patch("prepare_identity_coverage._target", return_value=("fixture", self.weight)), \
             patch("prepare_identity_coverage.branch_off_generation", return_value=observations):
            return qualify_original_replay_pool(self.runtime, self.plan)

    def test_pool_below_minimum_preserves_wrong_gold_and_never_approves_source(self):
        before = copy.deepcopy(self.pool)
        observations = self.observations(3)
        result = self.qualify(observations)
        self.assertFalse(result["passed"])
        self.assertEqual(result["qualified_count"], 3)
        self.assertEqual(result["generation"], observations)
        self.assertEqual(self.pool, before)
        self.assertEqual(result["generation"][3]["answer"], "wrong")
        self.assertEqual(self.pool[3]["answer"], "WORD3")

    def test_all_passing_rows_qualify_in_fixed_order_with_actual_generation_guard(self):
        result = self.qualify(self.observations(7))
        self.assertTrue(result["passed"])
        self.assertEqual([row["id"] for row in result["qualified"]], [row["id"] for row in self.pool[:7]])
        for row in result["qualified"]:
            value = json.loads(row["provenance"])
            self.assertTrue(row["approved"])
            self.assertTrue(value["training_generation_gate"])
            self.assertTrue(value["baseline_preservation_eligible"])
            self.assertTrue(value["original_baseline_generation"]["complete"])
        self.assertTrue(all(row["approved"] == 0 for row in self.pool))

    def test_forged_pass_truncation_branch_or_binding_cannot_qualify(self):
        for mutation in ({"complete": False}, {"answer": "not-gold"}, {"branch_enabled": True}):
            with self.subTest(mutation=mutation):
                observations = self.observations(12)
                observations[0].update(mutation)
                with self.assertRaisesRegex(ValueError, "incorrect, truncated or branch"):
                    self.qualify(observations)
        self.runtime.active_core = "core-123456789abc"
        with patch("prepare_identity_coverage.branch_off_generation") as generate, self.assertRaises(ValueError):
            qualify_original_replay_pool(self.runtime, self.plan)
        generate.assert_not_called()

    def test_native_sqlite_clone_does_not_seed_or_overwrite_production(self):
        with tempfile.TemporaryDirectory() as directory:
            source, research = Path(directory) / "production", Path(directory) / "research"
            source.mkdir()
            research.mkdir()
            memory = Memory(source)
            memory.add_sample("known question", "known answer", approved=True, training_allowed=True)
            before = hashlib.sha256(memory.path.read_bytes()).hexdigest()
            result = clone_research_database(source, research)
            self.assertEqual(hashlib.sha256(memory.path.read_bytes()).hexdigest(), before)
            self.assertEqual(Memory(research).samples(), memory.samples())
            self.assertEqual(result["clone_path"], str(research / "app/memory.sqlite3"))
            with self.assertRaises(ValueError):
                clone_research_database(source, research)
            self.assertEqual(hashlib.sha256(memory.path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
