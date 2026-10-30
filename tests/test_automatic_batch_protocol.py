"""Preregistration and qualification only; never load a model or call HTTP."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("automatic_batch_protocol", SCRIPTS / "verify_automatic_batches.py")
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)

from luban.calibration import SPLITS
from luban.retention import eligible_learning_sample, provenance, verified_source


def row(index, purpose="training"):
    return {"id": str(index), "prompt": "question " + str(index), "answer": "checked",
            "purpose": purpose, "approved": 1, "training_allowed": 1, "source": "synthetic",
            "provenance": {"program_verified": True, "verification_evidence": "isolated fixture checker",
                           "split_family": "family-" + str(index)}}


def manifest():
    return {"identity": "a" * 64, "core_version": None, "branch": {
        "train_pool": [row(i) for i in range(16)], "valid": [row(30, "validation")], "protection": []}}


class BatchProtocolTests(unittest.TestCase):
    def test_static_plan_has_fixed_gold_sources_and_no_model_calls(self):
        with patch("luban.runtime.Runtime.load", side_effect=AssertionError("no model")):
            plan = batch.plan(manifest())
        self.assertEqual(plan["teacher_calls"], 0)
        self.assertFalse(plan["production_publication"])
        self.assertEqual(plan["router"]["steps"], 160)
        self.assertEqual(plan["branch"]["steps_per_candidate"], 64)
        self.assertEqual(len(plan["router"]["pool"]), 44)
        self.assertTrue(all(verified_source(record) for record in plan["router"]["pool"]))
        self.assertTrue(all(provenance(record)["program_verified"] is True for record in plan["router"]["pool"]))
        self.assertTrue(all(provenance(record)["verification_evidence"] for record in plan["router"]["pool"]))
        self.assertEqual(plan["manifest"], manifest())
        self.assertEqual(plan["source_sha256"]["src/luban/evolution.py"], batch.sha(SCRIPTS.parent / "src/luban/evolution.py"))
        self.assertEqual(plan["source_sha256"]["src/luban/process_lock.py"], batch.sha(SCRIPTS.parent / "src/luban/process_lock.py"))

    def test_branch_nested_sizes_keep_valid_and_manifest_unmodified(self):
        fixed = manifest()
        before = copy.deepcopy(fixed)
        smaller, valid = batch.branch_rows(fixed, 8)
        larger, same_valid = batch.branch_rows(fixed, 16)
        self.assertEqual(smaller, larger[:8])
        self.assertEqual(valid, same_valid)
        self.assertEqual(fixed, before)
        self.assertTrue(all(eligible_learning_sample(record) for record in smaller + valid))

    def test_inherited_missing_families_are_reported_without_inventing_evidence(self):
        fixed = manifest()
        del fixed["branch"]["train_pool"][0]["provenance"]["split_family"]
        before = copy.deepcopy(fixed)
        protocol = batch.plan(fixed)
        self.assertEqual(protocol["branch"]["missing_split_family_ids"], ["0"])
        self.assertIn("unproven", protocol["branch"]["family_scope"])
        self.assertEqual(fixed, before)

    def test_teacher_opinions_and_family_leaks_cannot_train(self):
        bad = manifest()
        bad["branch"]["train_pool"][0]["provenance"]["program_verified"] = False
        with self.assertRaises(ValueError):
            batch.plan(bad)
        leaked = manifest()
        leaked["branch"]["valid"][0]["provenance"]["split_family"] = "family-0"
        with self.assertRaisesRegex(ValueError, "family leakage"):
            batch.plan(leaked)

    def test_router_nested_classes_preserve_every_heldout_row(self):
        labelled = batch.router_pool()
        for record in labelled:
            value = record["provenance"]
            # Fixture labels only test deterministic selection, not neural accuracy.
            value["route_action"] = "teacher" if len(record["answer"]) > 14 else "local_core"
        smaller = batch.nested_routes(labelled, 3)
        larger = batch.nested_routes(labelled, 6)
        self.assertEqual({action: sum(r["route_action"] == action for r in smaller["train"])
                          for action in ("local_core", "teacher")}, {"local_core": 3, "teacher": 3})
        for split in SPLITS[1:]:
            self.assertEqual(smaller[split], larger[split])
        for action in ("local_core", "teacher"):
            small_ids = [row["id"] for row in smaller["train"] if row["route_action"] == action]
            large_ids = [row["id"] for row in larger["train"] if row["route_action"] == action]
            self.assertEqual(small_ids, large_ids[:3])

    def test_insufficient_class_blocks_without_resampling(self):
        labelled = batch.router_pool()
        for record in labelled:
            record["provenance"]["route_action"] = "local_core"
        before = copy.deepcopy(labelled)
        with self.assertRaises(ValueError):
            batch.nested_routes(labelled, 3)
        self.assertEqual(before, labelled)


if __name__ == "__main__":
    unittest.main()
