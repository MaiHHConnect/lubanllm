"""Synthetic protocol checks only: no model or external teacher execution."""
from collections import Counter
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from continual_synthetic_data import build_dataset, schedule, LABELS, TEMPLATES
from luban.generation_checks import evaluate_generation, validate_generation_check


class ContinualSyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.dataset = build_dataset()
        self.train = self.dataset["groups"]["TRAIN"]

    def test_fixed_data_are_deterministic_and_fresh_objects(self):
        self.assertEqual(self.dataset, build_dataset())
        self.dataset["groups"]["TRAIN"][0]["messages"][0]["content"] = "changed"
        self.assertNotEqual(self.dataset, build_dataset())
        self.assertEqual(build_dataset()["metadata"]["task_ids"], ["A", "B", "C"])

    def test_splits_are_disjoint_wordings_of_same_facts(self):
        groups = self.dataset["groups"]
        self.assertEqual({key: len(rows) for key, rows in groups.items()}, {"TRAIN": 24, "DEV": 12, "TEST": 24})
        self.assertEqual(len({template for values in TEMPLATES.values() for template in values}), 5)
        all_rows = [row for rows in groups.values() for row in rows]
        self.assertEqual(len({row["id"] for row in all_rows}), 60)
        self.assertEqual(len({"".join(row["prompt"].split()).casefold() for row in all_rows}), 60)
        fact_sets = [{row["fact_id"] for row in rows} for rows in groups.values()]
        self.assertEqual(fact_sets[0], fact_sets[1])
        self.assertEqual(fact_sets[0], fact_sets[2])
        for key, rows in groups.items():
            self.assertEqual(set(Counter(row["fact_id"] for row in rows).values()), {len(TEMPLATES[key])})

    def test_random_bijections_and_no_answer_label_in_prompts(self):
        facts = self.dataset["facts"]
        self.assertEqual(len({fact["entity"] for fact in facts}), 12)
        for task in ("A", "B", "C"):
            selected = [fact for fact in facts if fact["task"] == task]
            self.assertEqual({fact["answer"] for fact in selected}, set(LABELS))
            self.assertNotEqual([fact["answer"] for fact in selected], list(LABELS))
        for fact in facts:
            self.assertEqual(len(fact["entity"]), 6)
            self.assertTrue(set(fact["entity"]).isdisjoint(LABELS))
        for rows in self.dataset["groups"].values():
            for row in rows:
                self.assertTrue(set(row["prompt"]).isdisjoint(LABELS))
                self.assertEqual(row["messages"], [{"role": "user", "content": row["prompt"]}])
                self.assertEqual(validate_generation_check(row["check"])["expected"], row["answer"])

    def test_protection_checks_are_objective_and_evaluation_only(self):
        self.assertEqual(len(self.dataset["protection"]), 8)
        for row in self.dataset["protection"]:
            self.assertEqual(row["task"], "P")
            self.assertFalse(row["training_allowed"])
            self.assertTrue(evaluate_generation(row["answer"], row["check"], True)["passed"])
            self.assertFalse(evaluate_generation(row["answer"], row["check"], False)["passed"])
            self.assertFalse(evaluate_generation("unrelated", row["check"], True)["passed"])

    def test_schedule_counts_and_fair_row_repeats(self):
        for seed in (7, 42, 123):
            for stage in (1, 2, 3):
                for arm in ("current_only", "replay"):
                    with self.subTest(seed=seed, stage=stage, arm=arm):
                        rows = schedule(self.train, stage, arm, seed)
                        self.assertEqual(len(rows), 64)
                        expected = {"ABC"[stage - 1]: 64}
                        if arm == "replay" and stage > 1:
                            expected = {"ABC"[stage - 1]: 32, **{task: 32 // (stage - 1) for task in "ABC"[:stage - 1]}}
                        self.assertEqual(Counter(row["task"] for row in rows), expected)
                        for task, count in expected.items():
                            self.assertEqual(set(Counter(row["id"] for row in rows if row["task"] == task).values()), {count // 8})

    def test_paired_stage_one_and_current_order_match(self):
        before = copy.deepcopy(self.train)
        for seed in (7, 42, 123):
            self.assertEqual(schedule(self.train, 1, "current_only", seed), schedule(self.train, 1, "replay", seed))
            for stage in (2, 3):
                current = schedule(self.train, stage, "current_only", seed)
                replay = schedule(self.train, stage, "replay", seed)
                self.assertEqual(current[:32], replay[::2])
                self.assertEqual(current, schedule(list(reversed(self.train)), stage, "current_only", seed))
        self.assertEqual(before, self.train)
        self.assertNotEqual(schedule(self.train, 1, "current_only", 7), schedule(self.train, 1, "current_only", 42))

    def test_non_training_and_invalid_inputs_rejected(self):
        for group in ("DEV", "TEST"):
            with self.assertRaises(ValueError):
                schedule(self.train + self.dataset["groups"][group], 2, "replay", 7)
        with self.assertRaises(ValueError):
            schedule(self.train + self.dataset["protection"], 2, "replay", 7)
        for field, value in (("split", "test"), ("purpose", "validation"), ("training_allowed", False)):
            rows = copy.deepcopy(self.train)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                schedule(rows, 1, "current_only", 7)
        for stage, arm, seed in ((0, "replay", 7), (4, "replay", 7), (True, "replay", 7),
                                 (1, "unknown", 7), (1, "replay", True)):
            with self.assertRaises(ValueError):
                schedule(self.train, stage, arm, seed)
        for rows in ([], self.train[:-1], self.train + [self.train[0]]):
            with self.assertRaises(ValueError):
                schedule(rows, 1, "current_only", 7)


if __name__ == "__main__":
    unittest.main()
