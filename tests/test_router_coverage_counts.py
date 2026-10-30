"""Fabricated CPU labels test selection only; no actual qualification or model."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_router_coverage_counts as runner
from luban.retention import provenance


def fixture_qualification():
    """Every answer/core SHA here is a fixture, never a real current baseline."""
    data = runner.descriptor()
    rows = runner._train(data)
    answers = []
    for row in rows:
        value = provenance(row)
        correct = (value.get("origin_sample_id") is not None
                   or (value["numeric_digit_band"] == "short" and value["operation_family"] in {"add", "sum_three"})
                   or (value["numeric_digit_band"] == "long" and value["operation_family"] == "sum_three"))
        answers.append({"id": row["id"], "answer": row["answer"] if correct else "not_an_integer",
            "passed": correct, "complete": True, "branch_enabled": False, "thinking": "off",
            "do_sample": False, "max_new_tokens": 512, "raw_completion_token_ids": [123, 999]})
    return {"evidence_scope": "fabricated_cpu_fixture", "binding": {
        "identity": data["base_identity"], "core_version": "core-abcdef123456", "product_accepted": True,
        "core_tensor_sha256": "a" * 64, "core_metadata_sha256": "b" * 64, "core_weights_sha256": "c" * 64,
        "active_core_pointer_sha256": "d" * 64, "train_records_sha256": runner.digest(rows),
        "reference_used": False, "teacher_calls": 0, "eos_token_ids": [999]}, "answers": answers}


def relabel(qualification, predicate):
    rows = {row["id"]: row for row in runner._train(runner.descriptor())}
    for answer in qualification["answers"]:
        row = rows[answer["id"]]
        correct = predicate(row)
        answer.update(answer=row["answer"] if correct else "not_an_integer", passed=correct)


class RouterCoverageCountsTests(unittest.TestCase):
    def test_static_default_without_site_imports_has_no_torch_model_or_paid_path(self):
        command = "import runpy,sys;runpy.run_path('scripts/verify_router_coverage_counts.py',run_name='__main__');assert not any(n in sys.modules for n in ['torch','transformers','luban.runtime','luban.speech','luban.teacher'])"
        result = subprocess.run([str(ROOT / ".venv/bin/python"), "-S", "-c", command],
            cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["counts"], {"train": 98, "development": 32, "calibration": 32, "test": 32})
        self.assertEqual(plan["state"], "waiting_accepted_core")
        self.assertFalse(plan["sealed"])
        self.assertFalse(plan["execution_ready"])
        self.assertFalse(plan["model_go"])
        self.assertEqual(plan["teacher_calls"], 0)
        self.assertEqual(plan["model_allocations"], 0)

    def test_nested_coverage_priorities_histograms_and_immutable_input(self):
        value = fixture_qualification()
        before = copy.deepcopy(value)
        result = runner.select_training(value)
        self.assertEqual(value, before)
        self.assertEqual(result["state"], "fixture_ready")
        self.assertFalse(result["execution_ready"])
        self.assertFalse(result["product_adoption"])
        self.assertFalse(result["heldout_inspected"])
        self.assertFalse(result["teacher_quality_measured"])
        self.assertEqual(len(result["train_labels"]), 98)
        self.assertEqual(len({row["id"] for row in result["train_labels"]}), 98)
        rows = {row["id"]: row for row in runner._train(runner.descriptor())}
        ids = result["nested_ids"]
        local = [rows[i] for i in ids["3"]["local_core"]]
        self.assertTrue(all(provenance(r).get("origin_sample_id") for r in local[:2]))
        self.assertEqual(runner.cell(local[2]), ("add", "short", "verbose"))
        teacher = [rows[i] for i in ids["3"]["teacher"]]
        self.assertEqual([runner.cell(r) for r in teacher], [
            ("subtract", "short", "verbose"), ("add", "long", "brief"), ("multiply", "long", "verbose")])
        for action in runner.ACTIONS:
            self.assertEqual(ids["3"][action], ids["6"][action][:3])
            self.assertEqual(ids["6"][action], ids["12"][action][:6])
            self.assertEqual(len(set(ids["12"][action])), 12)
            self.assertEqual(sum(result["class_cell_histogram"][action].values()), result["class_counts"][action])
            for count in runner.COUNTS:
                self.assertEqual(sum(result["selected_class_cell_histogram"][str(count)][action].values()), count)
        self.assertIn("change", result["distribution_caveat"])

    def test_train_order_does_not_change_predeclared_selection(self):
        value = fixture_qualification()
        first = runner.select_training(value)
        value["answers"].reverse()
        second = runner.select_training(value)
        self.assertEqual(first["nested_ids"], second["nested_ids"])
        self.assertEqual(first["selected_class_cell_histogram"], second["selected_class_cell_histogram"])

    def test_no_current_binding_never_reuses_original_none_or_legacy_success(self):
        self.assertEqual(runner.select_training(None)["state"], "waiting")
        for key, replacement in [("core_version", None), ("product_accepted", False)]:
            value = fixture_qualification()
            value["binding"][key] = replacement
            with self.subTest(key=key):
                result = runner.select_training(value)
                self.assertEqual(result["state"], "waiting")
                self.assertEqual(result["nested_ids"], {})

    def test_anchor_current_wrong_or_truncated_remains_waiting(self):
        for truncated in (False, True):
            value = fixture_qualification()
            anchor = next(a for a in value["answers"] if "anchor" in a["id"])
            anchor["passed"] = False
            if truncated:
                anchor.update(complete=False, raw_completion_token_ids=[123])
            else:
                anchor["answer"] = "wrong"
            with self.subTest(truncated=truncated):
                result = runner.select_training(value)
                self.assertEqual(result["state"], "waiting")
                self.assertIn("anchors", result["reason"])
                self.assertEqual(result["nested_ids"], {})

    def test_missing_class_or_specific_coverage_is_waiting_without_substitution(self):
        value = fixture_qualification()
        relabel(value, lambda row: True)
        self.assertIn("needs12", runner.select_training(value)["reason"])
        for absent in ("short_failure", "long_add_sub_failure", "long_multiply_failure", "short_success"):
            value = fixture_qualification()
            def correct(row):
                op, band, _ = runner.cell(row)
                if provenance(row).get("origin_sample_id"): return True
                if absent == "short_failure": return band == "short" or (band == "long" and op == "sum_three")
                if absent == "long_add_sub_failure": return (band == "long" and op in {"add", "subtract", "sum_three"}) or (band == "short" and op in {"add", "sum_three"})
                if absent == "long_multiply_failure": return (band == "long" and op in {"multiply", "sum_three"}) or (band == "short" and op in {"add", "sum_three"})
                return band == "long" and op == "sum_three"
            relabel(value, correct)
            result = runner.select_training(value)
            with self.subTest(absent=absent):
                self.assertTrue(all(count >= 12 for count in result["class_counts"].values()))
                self.assertEqual(result["state"], "waiting")
                self.assertIn("coverage", result["reason"])
                self.assertEqual(result["nested_ids"], {})

    def test_heldout_and_unknown_or_duplicate_ids_are_rejected_not_selected(self):
        for problem in ("heldout_id", "extra_heldout_payload", "duplicate", "unknown"):
            value = fixture_qualification()
            if problem == "extra_heldout_payload": value["heldout_answers"] = [{"id": "test-leak", "passed": True}]
            elif problem == "duplicate": value["answers"].append(copy.deepcopy(value["answers"][0]))
            elif problem == "unknown": value["answers"][0]["id"] = "unknown"
            else:
                heldout = next(r for r in runner.descriptor()["pool"] if provenance(r)["split"] == "test")
                value["answers"][0]["id"] = heldout["id"]
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                runner.select_training(value)
        value = fixture_qualification(); value["answers"].pop()
        self.assertEqual(runner.select_training(value)["state"], "waiting")

    def test_fixed_raw_eos_budget_pass_and_route_binding_are_not_advisory_booleans(self):
        for key, replacement in [("passed", False), ("complete", False), ("branch_enabled", True),
                ("do_sample", True), ("thinking", "on"), ("max_new_tokens", 128),
                ("raw_completion_token_ids", [123]), ("raw_completion_token_ids", [True, 999]),
                ("passed", 1), ("answer", "wrong")]:
            value = fixture_qualification()
            answer = next(a for a in value["answers"] if a["passed"])
            answer[key] = replacement
            with self.subTest(key=key, replacement=replacement), self.assertRaises(ValueError):
                runner.select_training(value)

    def test_core_dataset_reference_and_teacher_binding_must_match(self):
        for key, replacement in [("identity", "other-base"), ("core_version", "legacyNone"),
                ("core_tensor_sha256", ""), ("train_records_sha256", "e" * 64),
                ("reference_used", True), ("teacher_calls", 1), ("teacher_calls", False), ("eos_token_ids", [True])]:
            value = fixture_qualification(); value["binding"][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.select_training(value)

    def test_changed_static_data_does_not_silently_refresh_the_selector_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            data = runner.descriptor()
            data["pool"][0]["answer"] = "changed"
            path.write_text(json.dumps(data))
            with patch.object(runner, "DESCRIPTOR", path), self.assertRaisesRegex(ValueError, "Fixed coverage data changed"):
                runner.plan()


if __name__ == "__main__":
    unittest.main()
