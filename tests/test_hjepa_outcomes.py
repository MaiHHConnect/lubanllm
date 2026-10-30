"""Checks target integrity and probability merging, not model capability."""
import sys
import unittest
from pathlib import Path
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_hjepa_outcomes import binary_equivalence, checked_json, outcome, outcome_routes


class OutcomeResearchTests(unittest.TestCase):
    def test_outcomes_require_complete_verified_success(self):
        self.assertEqual(outcome({"passed": True, "complete": True}), 0)
        self.assertEqual(outcome({"passed": False, "complete": True}), 1)
        self.assertEqual(outcome({"passed": False, "complete": False}), 2)
        for answer in ({"passed": 1, "complete": True}, {"passed": True, "complete": False}, {}):
            with self.assertRaises(ValueError):
                outcome(answer)

    def test_merge_sums_failure_probabilities_not_argmax(self):
        z = torch.tensor([[1., .9, .9], [-10., 10., -10.]], dtype=torch.float64)
        merged = outcome_routes(z).softmax(-1)
        original = z.softmax(-1)
        self.assertTrue(torch.allclose(merged[:, 0], original[:, 0]))
        self.assertTrue(torch.allclose(merged[:, 2], original[:, 1:].sum(-1)))
        self.assertEqual(merged[0].argmax().item(), 2)
        self.assertTrue(torch.equal(merged[:, 1], torch.zeros(2)))

    def test_binary_equivalence_includes_gradient_and_probabilities(self):
        features = torch.tensor([[1., 2.], [-2., 1.], [.5, -.5]])
        state = {"projection.weight": torch.tensor([[.1, .2], [.7, -.1], [-.3, .9]]),
                 "projection.bias": torch.tensor([.1, .2, .3])}
        result = binary_equivalence(features, state, torch.tensor([0, 2, 0]))
        self.assertTrue(result["same_decisions"])
        for key in ("loss_abs_error", "gradient_max_abs_error", "probability_max_abs_error"):
            self.assertLess(result[key], 1e-12)

    def test_archived_hash_checked_before_json_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.json"
            path.write_text("invalid JSON")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                checked_json(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
