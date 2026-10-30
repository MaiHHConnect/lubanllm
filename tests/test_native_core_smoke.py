"""Stdlib-only CLI/material tests; these never execute the actual model."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke_native_core_training.py"
spec = importlib.util.spec_from_file_location("native_smoke_source", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def protocol():
    return {"groups": {key: [{"id": role + str(i), "prompt": "question" + str(i),
        "answer": "full answer", "purpose": "training", "approved": True,
        "training_allowed": True, "provenance": json.dumps({"evolution_role": role})}
        for i in range(3)] for key, role in (("new_training", "new"), ("replay_training", "replay"))}}


class NativeCoreSmokeTests(unittest.TestCase):
    def test_extracts_original_two_rows_without_mutating_protocol(self):
        value = protocol()
        original = copy.deepcopy(value)
        selected = module.selected_rows(value)
        self.assertEqual(selected, {key: rows[:2] for key, rows in original["groups"].items()})
        self.assertEqual(value, original)

    def test_unverified_disallowed_wrong_role_and_validation_rows_rejected(self):
        for change in ({"approved": False}, {"training_allowed": False}, {"approved": 1.0},
                       {"approved": "1"}, {"approved": 2}, {"training_allowed": "true"},
                       {"purpose": "validation"}, {"provenance": '{"evolution_role":"replay"}'}, {"id": ""}):
            value = protocol()
            value["groups"]["new_training"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.selected_rows(value)

    def test_sqlite_integer_permissions_preserve_the_original_rows(self):
        value = protocol()
        for sample in value["groups"]["replay_training"]:
            sample.update(approved=1, training_allowed=1)
        original = copy.deepcopy(value)
        selected = module.selected_rows(value)
        self.assertEqual(selected["replay_training"], original["groups"]["replay_training"][:2])
        self.assertEqual(value, original)

    def test_not_enough_rows_or_duplicate_id_rejected(self):
        value = protocol()
        value["groups"]["replay_training"] = value["groups"]["replay_training"][:1]
        with self.assertRaises(ValueError):
            module.selected_rows(value)
        value = protocol()
        value["groups"]["replay_training"][0]["id"] = value["groups"]["new_training"][0]["id"]
        with self.assertRaises(ValueError):
            module.selected_rows(value)

    def test_help_does_not_import_torch_or_load_a_model(self):
        completed = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--model-root", completed.stdout)
        self.assertIn("--protocol", completed.stdout)


if __name__ == "__main__":
    unittest.main()
