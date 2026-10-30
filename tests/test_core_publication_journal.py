"""Real SQLite journal hooks: publish complete material and pointer exactly once."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from luban.memory import Memory
from luban.training_checkpoint import TrainingSession, recover_commits, training_runs
from tests.test_core_publication import fixture_plan


class PublicationJournalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.runtime = SimpleNamespace(memory=Memory(root), store=root / "modules",
            identity="fixture-base", active_core=None, branch=SimpleNamespace(expert_id=None),
            device="cpu", dtype=torch.float32)
        self.runtime.store.mkdir()
        self.plan = fixture_plan()
        with self.runtime.memory.connect() as db:
            for row in self.plan["source_rows"]:
                db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))
        acceptance = self.plan["acceptance"]
        path = self.runtime.store / acceptance["report_path"]
        path.write_bytes(b"fixed outer acceptance fixture")
        acceptance["report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.session = TrainingSession(self.runtime, "evolution", {},
            {"groups": self.plan["groups"], "retention_regressions": self.plan["retention_regressions"]},
            run_id=self.plan["run_id"], scope="research")
        self.session.body["product_accepted"] = True

    def prepare(self, plan=None):
        self.session.prepare_commit({"promoted": True, "version": self.plan["version"]},
            {"active-core.json": {"version": self.plan["version"], "identity": self.runtime.identity}},
            [self.plan["acceptance"]["report_path"]], event_kind="evolution",
            material_publication=plan or self.plan)

    def test_prepared_recovery_publishes_material_and_pointer_once(self):
        self.prepare()
        self.assertEqual(len(self.runtime.memory.samples()), 103)
        recover_commits(self.runtime.memory, self.runtime.store, self.runtime.identity)
        self.assertEqual(len(self.runtime.memory.samples()), 119)
        self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity)[0]["state"], "committed")
        pointer = self.runtime.store / "active-core.json"
        self.assertTrue(pointer.exists())
        pointer.unlink()  # A later explicit restore must survive repeated acknowledgment.
        self.session.finish_commit()
        self.assertFalse(pointer.exists())
        with self.runtime.memory.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM core_material_archives").fetchone()[0], 12)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_source_conflict_stops_before_material_or_pointer_publication(self):
        self.prepare()
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET answer='changed' WHERE id=?", (self.plan["source_rows"][-1]["id"],))
        with self.assertRaisesRegex(ValueError, "材料已变化"):
            self.session.finish_commit()
        self.assertEqual(len(self.runtime.memory.samples()), 103)
        self.assertFalse((self.runtime.store / "active-core.json").exists())

    def test_changed_complete_records_or_acceptance_hash_cannot_prepare(self):
        for change in ("groups", "hash", "id"):
            plan = copy.deepcopy(self.plan)
            if change == "groups":
                plan["groups"]["new_training"][0]["answer"] = "tampered"
            elif change == "hash":
                plan["acceptance"]["report_sha256"] = "f" * 64
            else:
                plan["run_id"] = "abcdef123456"
                plan["version"] = "core-abcdef123456"
                plan["acceptance"]["report_path"] = "coverage-acceptance-abcdef123456.json"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare(plan)
        self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity)[0]["state"], "queued")


if __name__ == "__main__":
    unittest.main()
