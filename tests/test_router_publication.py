"""Verified labels, isolated journal and all-or-nothing outer route adoption."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_router_v1 import accept_router, cases, restore_router, route_records
from luban.calibration import split_route_samples, validate_route_splits
from luban.decision import FastDecisionHead
from luban.memory import Memory
from luban.training_checkpoint import TrainingSession, training_runs


class RouterPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.runtime = SimpleNamespace(memory=Memory(root), store=root / "modules", identity="router-base",
            active_core="core-123456abcdef", branch=SimpleNamespace(expert_id=None), device="cpu", dtype=torch.float32,
            router=torch.nn.Linear(2, 2), router_meta={"calibrated": False})
        self.runtime.store.mkdir()
        measured = [{**row, "route_action": "teacher" if row["intended_difficulty"] == "hard" else "local_core",
                     "label_basis": "fixed fixture"} for row in cases()]
        self.records = route_records(measured, self.runtime.identity, self.runtime.active_core, root, 128)
        splits = split_route_samples(self.records, FastDecisionHead.actions)
        validate_route_splits(splits, FastDecisionHead.actions)
        self.session = TrainingSession(self.runtime, "router", {"steps": 160},
            {"splits": splits, "publication_records": self.records}, scope="research")
        artifact = self.runtime.store / "candidate.safetensors"
        artifact.write_bytes(b"CPU bounded fixture")
        self.session.prepare_commit({"calibrated": True}, {"active-router.json": {"id": "router-" + self.session.id}},
            [artifact.name], event_kind="router")
        self.session.finish_commit()
        self.guards = [{"case_id": str(i), "passed": True, "decision": {"route": "local", "action": "local_core"}} for i in range(12)]

    def test_complete_records_qualify_but_do_not_enter_product_before_outer_guard(self):
        self.assertEqual(self.runtime.memory.samples(), [])
        row = training_runs(self.runtime.memory, self.runtime.identity)[0]
        self.assertEqual(row["scope"], "research")
        self.assertNotEqual(row.get("product_accepted"), True)
        accept_router(self.runtime, self.session, self.records, self.guards)
        self.assertEqual(len(self.runtime.memory.samples()), 44)
        self.assertTrue(training_runs(self.runtime.memory, self.runtime.identity)[0]["product_accepted"])
        accept_router(self.runtime, self.session, self.records, self.guards)
        self.assertEqual(len(self.runtime.memory.samples()), 44)

    def test_failed_guard_corrupt_artifact_or_last_sample_conflict_has_no_partial_adoption(self):
        guards = copy.deepcopy(self.guards)
        guards[-1]["passed"] = False
        with self.assertRaisesRegex(ValueError, "guard"):
            accept_router(self.runtime, self.session, self.records, guards)
        sample = self.records[-1] | {"answer": "conflict"}
        with self.runtime.memory.connect() as db:
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(sample.values()))
        with self.assertRaisesRegex(ValueError, "conflict"):
            accept_router(self.runtime, self.session, self.records, self.guards)
        self.assertEqual(len(self.runtime.memory.samples()), 1)
        self.assertNotEqual(training_runs(self.runtime.memory, self.runtime.identity)[0].get("product_accepted"), True)
        (self.runtime.store / "candidate.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "artifact"):
            accept_router(self.runtime, self.session, self.records, self.guards)

    def test_rollback_restores_active_head_pointer_and_preserves_core_pointer(self):
        core = self.runtime.store / "active-core.json"
        core.write_text('{"version":"preserved"}')
        state = copy.deepcopy(self.runtime.router.state_dict())
        with torch.no_grad():
            self.runtime.router.weight.fill_(99)
        restore_router(self.runtime, state, {"calibrated": False}, {"active-router.json": None})
        self.assertFalse((self.runtime.store / "active-router.json").exists())
        self.assertEqual(json.loads(core.read_text()), {"version": "preserved"})
        self.assertTrue(torch.equal(self.runtime.router.weight, state["weight"]))


if __name__ == "__main__":
    unittest.main()
