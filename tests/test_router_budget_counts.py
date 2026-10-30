"""Static arithmetic and CPU control-flow fixtures only, no model or HTTP."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("router_budget_counts", ROOT / "scripts/verify_router_budget_counts.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixtures(rows):
    # Artificial labels prove split selection only, not neural qualification.
    return [{"id": row["id"], "answer": row["answer"] if "-short-" in row["id"] else "wrong",
             "passed": "-short-" in row["id"], "complete": True, "thinking": "off", "branch_enabled": False}
            for row in rows]


class RouterBudgetCountsTests(unittest.TestCase):
    def setUp(self):
        self.descriptor = json.loads(runner.DESCRIPTOR.read_text())
        self.rows = self.descriptor["pool"]

    def test_fixed_plan_source_guard_counts_and_no_model(self):
        with patch("luban.runtime.Runtime.load", side_effect=AssertionError("no model")):
            protocol = runner.plan()
        self.assertEqual(protocol["split_counts"], {"train": 72, "development": 24, "calibration": 24, "test": 24})
        self.assertEqual(protocol["budget_controls"], [128, 512])
        self.assertEqual(protocol["train_per_actual_class"], [3, 6, 12])
        self.assertEqual(protocol["teacher_calls"], 0)
        self.assertFalse(protocol["production_publication"])
        self.assertEqual(protocol["source_sha256"]["src/luban/runtime.py"], runner.sha(ROOT / "src/luban/runtime.py"))
        self.assertIn("unproven", protocol["family_scope"])
        self.assertEqual(len(protocol["descriptor"]["old44"]), 44)
        heldout = [row for row in self.rows if row["provenance"]["split"] != "train"]
        self.assertEqual(hashlib.sha256(runner.encoded(heldout).encode()).hexdigest(),
                         "274d0db2ec4a3c8d8882e06929b318289a6ce31c20ee2ce5ce139f2884c46cb2")

    def test_budget_change_preserves_all_gold_context_split_and_input(self):
        old = self.descriptor["old44"]
        before = copy.deepcopy(old)
        changed = runner.at_budget(old, 128)
        self.assertEqual(old, before)
        for prior, current in zip(old, changed):
            self.assertEqual(prior["answer"], current["answer"])
            self.assertEqual(prior["provenance"]["messages"], current["provenance"]["messages"])
            self.assertEqual(prior["provenance"]["split"], current["provenance"]["split"])
            current["provenance"]["generation_check"]["max_new_tokens"] = 32
            self.assertEqual(prior, current)

    def test_program_gold_recomputed_and_bad_source_gold_family_rejected(self):
        runner.validate_pool(self.rows)
        for mutation in ("gold", "verification", "evidence", "family", "duplicate", "label"):
            rows = copy.deepcopy(self.rows)
            if mutation == "gold": rows[0]["answer"] = "111"
            if mutation == "verification": rows[0]["provenance"]["program_verified"] = False
            if mutation == "evidence": rows[0]["provenance"]["verification_evidence"] = ""
            if mutation == "family": rows[-1]["provenance"]["family_group"] = rows[48]["provenance"]["family_group"]
            if mutation == "duplicate": rows[-1]["id"] = rows[0]["id"]
            if mutation == "label": rows[0]["provenance"]["route_action"] = "local_core"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): runner.validate_pool(rows)
        with self.assertRaises(ValueError): runner.integer_gold({"operation": "exact_divide", "args": [3, 2]})
        with self.assertRaises(ValueError): runner.integer_gold({"operation": "eval", "args": [2]})

    def test_nested3_6_12_all_heldout_unchanged_no_fixture_source_mutation(self):
        before = copy.deepcopy(self.rows)
        labelled = runner.labelled_rows(self.rows, fixtures(self.rows))
        chosen = [runner.nested_routes(labelled, count) for count in (3, 6, 12)]
        self.assertEqual(self.rows, before)
        for action in ("local_core", "teacher"):
            ids = [[r["id"] for r in split["train"] if r["route_action"] == action] for split in chosen]
            self.assertEqual(ids[0], ids[2][:3]); self.assertEqual(ids[1], ids[2][:6])
        for name in ("development", "calibration", "test"):
            self.assertEqual(chosen[0][name], chosen[2][name])
            self.assertEqual(chosen[1][name], chosen[2][name])
        self.assertTrue(all(not row["provenance"]["teacher_superiority_verified"] for row in labelled))

    def test_insufficient_actual_class_blocks_without_heldout_move(self):
        labelled = runner.labelled_rows(self.rows, fixtures(self.rows))
        hard = [r for r in labelled if r["provenance"]["split"] == "train" and r["provenance"]["route_action"] == "teacher"]
        for row in hard[7:]: row["provenance"]["route_action"] = "local_core"
        before = copy.deepcopy(labelled)
        runner.nested_routes(labelled, 6)
        with self.assertRaisesRegex(ValueError, "only has 7"): runner.nested_routes(labelled, 12)
        self.assertEqual(before, labelled)

    def test_incomplete_is_teacher_even_if_literal_answer_matches(self):
        answers = fixtures(self.rows)
        answers[0].update(complete=False, passed=False)
        labelled = runner.labelled_rows(self.rows, answers)
        self.assertEqual(labelled[0]["provenance"]["route_action"], "teacher")
        measured = runner.summary(self.rows, answers)
        self.assertEqual(measured["counts"]["incomplete"], 1)
        answers[0]["passed"] = True
        with self.assertRaisesRegex(ValueError, "contradicts"): runner.labelled_rows(self.rows, answers)

    def test_changed_order_or_branch_on_evidence_rejected(self):
        answers = fixtures(self.rows)
        answers[0]["branch_enabled"] = True
        with self.assertRaises(ValueError): runner.labelled_rows(self.rows, answers)
        with self.assertRaises(ValueError): runner.summary(self.rows, answers[::-1])

    def test_execute_requires_exact_preregistered_plan_without_loading(self):
        protocol = runner.plan(); protocol["steps"] = 161
        with patch("luban.runtime.Runtime.load", side_effect=AssertionError("no model")), self.assertRaisesRegex(ValueError, "plan changed"):
            runner.execute(protocol, ROOT, ROOT, ROOT / "unused")

    def test_cli_default_static_does_not_allocate_model(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_router_budget_counts.py"), "--print-plan"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        body = json.loads(result.stdout)
        self.assertTrue(body["prepared_without_model"])
        self.assertEqual(body["plan"]["steps"], 160)
        self.assertEqual(len(body["plan_sha256"]), 64)

    def test_isolated_execution_failure_releases_owner_and_keeps_production_sha(self):
        from luban.memory import Memory
        import torch
        protocol = runner.plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "experiments").mkdir()
            store = root / "modules/luban0-v1"; store.mkdir(parents=True)
            (store / "graph-verification.json").write_text("{}")
            memory = Memory(root)
            with memory.connect(): pass
            db_sha, graph_sha = runner.sha(memory.path), runner.sha(store / "graph-verification.json")
            output = root / "experiments/isolated-failure"
            owners = []
            class FakeRuntime:
                def __init__(self, settings, memory, research=False):
                    self.research, self.settings, self.memory = research, settings, memory
                    self.identity = protocol["identity"]; self.active_core = None
                    self.branch = SimpleNamespace(expert=None); self.device = "cpu"
                    self.model = torch.nn.Linear(2, 2); self.released = False
                    owners.append(self)
                def load(self):
                    assert self.research is True
                def _release_model(self): self.released = True
            @contextmanager
            def resources(runtime): yield {}
            with patch.object(runner, "plan", return_value=protocol), patch.object(runner, "measured", resources), \
                    patch("luban.runtime.Runtime", FakeRuntime), \
                    patch("luban.evolution._generation", side_effect=RuntimeError("isolated fixture failure")), \
                    self.assertRaisesRegex(RuntimeError, "fixture failure"):
                runner.execute(protocol, root / "dummy-model", root, output)
            self.assertTrue(owners[0].released)
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["production_unchanged"])
            self.assertEqual(runner.sha(memory.path), db_sha)
            self.assertEqual(runner.sha(store / "graph-verification.json"), graph_sha)
            self.assertFalse((output / "baseline/modules/luban0-v1/active-core.json").exists())


if __name__ == "__main__": unittest.main()
