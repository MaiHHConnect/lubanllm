"""Family partition guards and a frozen, genuinely independent route test."""
import copy
import json
import math
import unittest

import torch

from luban.calibration import (SPLITS, evaluate_router, fit_temperature,
                               probability_metrics, split_route_samples, validate_route_splits,
                               calibrate_expert_scope)

ACTIONS = ("local_core", "neural_expert", "teacher")
SUPPORTED = ["local_core", "neural_expert"]


def sample(index, action="local_core", split="train", family=None):
    provenance = {"route_action": action, "family_group": family or f"family-{index}"}
    if split:
        provenance["split"] = split
    return {"id": str(index), "prompt": f"问题 {index}", "answer": "已核验答案",
            "purpose": "training" if split == "train" else "validation",
            "source": "human", "approved": 1, "training_allowed": 1, "provenance": json.dumps(provenance)}


class SplitTests(unittest.TestCase):
    def test_partition_is_order_independent_and_does_not_mutate_records(self):
        rows = [sample(i, split=None) for i in range(40)]
        original = copy.deepcopy(rows)
        first = split_route_samples(rows, ACTIONS)
        second = split_route_samples(list(reversed(rows)), ACTIONS)
        self.assertEqual(first, second)
        self.assertEqual(rows, original)
        self.assertFalse(first["train"])
        self.assertEqual(sum(map(len, first.values())), 40)
        self.assertTrue(all(first[s] for s in SPLITS[1:]))

    def test_family_cross_split_and_training_to_test_are_rejected(self):
        rows = [sample(1, family="one-template"), sample(2, split="test", family="one-template")]
        with self.assertRaisesRegex(ValueError, "family_group"):
            split_route_samples(rows, ACTIONS)
        row = sample(3, split="test")
        row["purpose"] = "training"
        with self.assertRaisesRegex(ValueError, "已用于训练"):
            split_route_samples([row], ACTIONS)

    def test_same_family_with_missing_split_is_not_resampled(self):
        training = sample(1, family="generator-a")
        validation = sample(2, split=None, family="generator-a")
        with self.assertRaisesRegex(ValueError, "family_group"):
            split_route_samples([training, validation], ACTIONS)

    def test_duplicate_normalization_conflicts_and_deduplication(self):
        first = sample(1)
        duplicate = {**first, "id": "2", "prompt": "  问题   １  "}
        self.assertEqual(len(split_route_samples([first, duplicate], ACTIONS)["train"]), 1)
        conflicting = {**duplicate, "provenance": json.dumps({"route_action": "teacher",
                                                               "family_group": "family-1", "split": "train"})}
        with self.assertRaisesRegex(ValueError, "重复问题"):
            split_route_samples([first, conflicting], ACTIONS)
        with self.assertRaisesRegex(ValueError, "重复问题"):
            split_route_samples([first, {**duplicate, "answer": "另一个答案"}], ACTIONS)
        with self.assertRaisesRegex(ValueError, "重复问题"):
            split_route_samples([{**first, "answer": "True"}, {**duplicate, "answer": "TRUE"}], ACTIONS)

    def test_legacy_data_is_auditable_but_not_independent_evidence(self):
        row = sample(1)
        row["provenance"] = json.dumps({"route_action": "local_core"})
        with self.assertRaisesRegex(ValueError, "family_group"):
            split_route_samples([row], ACTIONS)
        legacy = split_route_samples([row], ACTIONS, require_family=False)
        self.assertFalse(legacy["train"][0]["family_verified"])

    def test_class_count_gate_and_unapproved_exclusion(self):
        rows = [sample(f"{split}-{action}-{i}", action, split)
                for split in SPLITS for action in SUPPORTED for i in range(3 if split == "train" else 2)]
        rows.append({**sample("not-approved"), "approved": 0})
        splits = split_route_samples(rows, ACTIONS)
        self.assertEqual(validate_route_splits(splits, ACTIONS), SUPPORTED)
        splits["test"].pop()
        with self.assertRaisesRegex(ValueError, "test"):
            validate_route_splits(splits, ACTIONS)

    def test_route_only_requires_independent_verified_source(self):
        human = sample("human")
        provenance = json.loads(human["provenance"])
        provenance["route_only"] = True
        human["provenance"] = json.dumps(provenance)
        teacher = {**human, "id": "teacher", "prompt": "teacher question", "source": "teacher"}
        self.assertEqual(len(split_route_samples([human, teacher], ACTIONS)["train"]), 1)
        provenance.update(program_verified=True, verification_evidence="integer arithmetic checker")
        teacher["provenance"] = json.dumps({**provenance, "family_group": "teacher-family"})
        self.assertEqual(len(split_route_samples([human, teacher], ACTIONS)["train"]), 2)
        provenance["verification_evidence"] = ""
        teacher["provenance"] = json.dumps(provenance)
        self.assertEqual(len(split_route_samples([teacher], ACTIONS)["train"]), 0)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.targets = torch.tensor([0, 0, 1, 1])
        self.good = torch.tensor([[4., 0., -math.inf], [3., 0., -math.inf],
                                  [0., 4., -math.inf], [0., 3., -math.inf]])

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_expert_scope_rejects_non_expert_representations(self):
        features = {
            "train": torch.tensor([[1., 0.], [1., .1], [.9, 0.], [.95, .05]]),
            "development": torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]]),
            "calibration": torch.tensor([[1., 0.], [0., 1.]]),
            "test": torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]]),
        }
        targets = {
            "train": torch.tensor([1, 1, 1, 0]),
            "development": torch.tensor([1, 1, 0, 0]),
            "calibration": torch.tensor([1, 0]),
            "test": torch.tensor([1, 1, 0, 0]),
        }
        result = calibrate_expert_scope(features, targets, ACTIONS, SUPPORTED)
        self.assertTrue(result["passed"])
        self.assertGreater(result["threshold"], 0)
        self.assertEqual(result["development"]["non_expert_selected"], 0)
        self.assertEqual(result["test"]["non_expert_selected"], 0)

    def test_metrics_have_known_nll_brier_ece_and_finite_masked_fitting(self):
        result = probability_metrics(torch.zeros(2, 2), torch.tensor([0, 1]))
        self.assertAlmostEqual(result["nll"], math.log(2))
        self.assertAlmostEqual(result["brier"], .5)
        self.assertAlmostEqual(result["ece"], 0)
        temperature = fit_temperature(self.good, self.targets)
        self.assertTrue(.05 <= temperature <= 20)
        self.assertLessEqual(probability_metrics(self.good, self.targets, temperature)["nll"],
                             probability_metrics(self.good, self.targets)["nll"])

    def test_temperature_reduces_overconfidence_nll(self):
        logits = torch.tensor([[10., 0.], [10., 0.], [10., 0.], [10., 0.]])
        targets = torch.tensor([0, 0, 0, 1])
        temperature = fit_temperature(logits, targets)
        self.assertGreater(temperature, 1)
        self.assertLess(probability_metrics(logits, targets, temperature)["nll"],
                        probability_metrics(logits, targets)["nll"])

    def test_test_failures_reject_without_changing_temperature_or_threshold(self):
        logits = {s: self.good.clone() for s in SPLITS[1:]}
        targets = {s: self.targets.clone() for s in SPLITS[1:]}
        passed = evaluate_router(logits, targets, ACTIONS, SUPPORTED)
        self.assertTrue(passed["calibrated"])
        targets["test"] = torch.tensor([1, 1, 0, 0])
        failed = evaluate_router(logits, targets, ACTIONS, SUPPORTED)
        self.assertFalse(failed["calibrated"])
        self.assertEqual(failed["temperature"], passed["temperature"])
        self.assertEqual(failed["threshold"], passed["threshold"])
        self.assertEqual(failed["test_accuracy"], 0)

    def test_development_selects_measured_threshold_and_reports_coverage(self):
        logits = {s: self.good.clone() for s in SPLITS[1:]}
        targets = {s: self.targets.clone() for s in SPLITS[1:]}
        logits["development"] = torch.cat([self.good * 10,
                                                torch.tensor([[0.1, 0., -math.inf]])])
        targets["development"] = torch.tensor([0, 0, 1, 1, 1])
        # A deliberately uncertain calibration set keeps confidence distinctions visible.
        logits["calibration"] = torch.tensor([[1., 0., -math.inf]] * 2 + [[0., 1., -math.inf]] * 2)
        result = evaluate_router(logits, targets, ACTIONS, SUPPORTED)
        selected = result["development_selection"]
        self.assertEqual(selected["selected"], 4)
        self.assertEqual(selected["coverage"], .8)
        self.assertEqual(selected["accuracy"], 1)
        self.assertGreaterEqual(result["threshold"], .8)
        self.assertTrue(result["threshold_candidates"])

    def test_insufficient_test_class_evidence_refuses_promotion(self):
        logits = {s: self.good.clone() for s in SPLITS[1:]}
        targets = {s: self.targets.clone() for s in SPLITS[1:]}
        logits["test"], targets["test"] = self.good[:3], self.targets[:3]
        result = evaluate_router(logits, targets, ACTIONS, SUPPORTED)
        self.assertFalse(result["calibrated"])
        self.assertFalse(result["minimum_counts_passed"])

    def test_invalid_scores_are_rejected(self):
        for logits, targets in [(torch.tensor([[math.nan, 0.]]), torch.tensor([0])),
                                (torch.tensor([[-math.inf, -math.inf]]), torch.tensor([0])),
                                (torch.zeros(1, 2), torch.tensor([2])),
                                (torch.zeros(1, 2), torch.tensor([1.])),
                                (torch.zeros(0, 2), torch.tensor([], dtype=torch.long))]:
            with self.assertRaises(ValueError):
                probability_metrics(logits, targets)


if __name__ == "__main__":
    unittest.main()
