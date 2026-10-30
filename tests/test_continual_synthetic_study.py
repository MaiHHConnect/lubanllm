"""Finite task-time metric mathematics; never execute a model or application."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from run_continual_synthetic import matrix_metrics


class ContinualSyntheticStudyTests(unittest.TestCase):
    def test_never_acquired_tasks_remain_visible_at_zero(self):
        metrics = matrix_metrics([[0, 0, 0] for _ in range(4)])
        self.assertEqual(metrics["final_average_accuracy"], 0)
        self.assertEqual(metrics["immediate_acquisition"], [0, 0, 0])
        self.assertEqual(metrics["backward_transfer"], 0)
        self.assertEqual(metrics["forgetting"], 0)
        self.assertEqual(metrics["forward_transfer"], 0)

    def test_regressions_produce_negative_backward_transfer(self):
        # A: .75 -> 1 -> .25, B: .5 -> .25; C is learned only at its stage.
        metrics = matrix_metrics([[0, 0, 0], [.75, 0, 0], [1, .5, 0], [.25, .25, 1]])
        self.assertEqual(metrics["immediate_acquisition"], [.75, .5, 1])
        self.assertEqual(metrics["backward_transfer"], -.375)
        self.assertEqual(metrics["forgetting"], .5)
        self.assertEqual(metrics["final_average_accuracy"], .5)

    def test_successor_improvement_can_have_negative_forgetting(self):
        # Do not clip forgetting at zero or turn continued improvement into loss.
        metrics = matrix_metrics([[0, 0, 0], [.25, 0, 0], [.5, .25, 0], [.75, .75, 1]])
        self.assertEqual(metrics["backward_transfer"], .5)
        self.assertEqual(metrics["forgetting"], -.375)
        self.assertEqual(metrics["immediate_acquisition"], [.25, .25, 1])

    def test_future_task_changes_are_relative_to_frozen_anchor(self):
        metrics = matrix_metrics([[.25, .25, .5], [.5, .75, .5], [.75, 1, .25], [.75, 1, 1]])
        self.assertEqual(metrics["forward_transfer"], .125)

    def test_matrix_must_include_anchor_and_all_three_task_columns(self):
        malformed = (
            [], [[0, 0, 0]] * 3, [[0, 0, 0]] * 5,
            [[0, 0]] * 4, [[0, 0, 0, 0]] * 4,
            [[0, 0, 0], [0, 0, 0], [0, 0], [0, 0, 0]],
        )
        for matrix in malformed:
            with self.subTest(matrix=matrix), self.assertRaises(AssertionError):
                matrix_metrics(matrix)


if __name__ == "__main__":
    unittest.main()
