import json
import hashlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from luban.memory import Memory
from luban.training_checkpoint import TrainingSession, reconcile_commit, recover_commits, training_runs, ensure_product_state


def runtime(root):
    return SimpleNamespace(memory=Memory(root), store=Path(root) / "modules", identity="checkpoint-test-base",
                           active_core=None, branch=SimpleNamespace(expert_id=None), device="cpu", dtype=torch.float32)


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()


class TrainingCheckpointTests(unittest.TestCase):
    def test_native_mlx_journal_is_not_mutated_or_reconciled_by_python(self):
        for state in ("running", "committing"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as root:
                task = runtime(root)
                session = TrainingSession(task, "learning", {}, [])
                session.body.update(arguments={"protocol": "native-mlx-expert-preservation-v1"},
                                    owner={"pid": 999999999}, commit={"native_owner": True})
                session._write(state)
                before = training_runs(task.memory, task.identity)
                recover_commits(task.memory, task.store, task.identity)
                self.assertEqual(training_runs(task.memory, task.identity), before)
                with self.assertRaisesRegex(ValueError, "原生MLX"):
                    reconcile_commit(task.memory, task.store, before[0])

    def test_python_training_owner_refuses_mlx_before_writing(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            with self.assertRaisesRegex(ValueError, "原生MLX"):
                TrainingSession(task, "learning", {"protocol": "native-mlx-expert-preservation-v1"}, [])
            self.assertFalse(task.store.exists())
            task.native_inference_precision = "mlx-q4"
            with self.assertRaisesRegex(ValueError, "原生MLX"):
                TrainingSession(task, "learning", {}, [])
            self.assertFalse(task.store.exists())

    def test_training_binding_rejects_changed_original_core_protection(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            baseline = task.store / "original-core-protection.json"
            baseline.write_text("baseline-v1")
            session = TrainingSession(task, "learning", {}, [])
            self.assertEqual(
                session.binding["original_core_protection_sha256"],
                hashlib.sha256(b"baseline-v1").hexdigest())
            baseline.write_text("baseline-v2")
            with self.assertRaisesRegex(ValueError, "检查点与当前模型"):
                TrainingSession(task, "learning", {}, [], run_id=session.id)

    def test_measured_pause_accumulates_once_across_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "learning", {}, [])
            with patch("luban.training_checkpoint.time.time", return_value=1000):
                session.pause()
            with patch("luban.training_checkpoint.time.time", return_value=1010):
                session.pause()
            self.assertEqual(session.body["paused_at"], 1000)
            with patch("luban.training_checkpoint.time.time", return_value=1020):
                resumed = TrainingSession(task, "learning", {}, [], run_id=session.id)
            self.assertEqual(resumed.body["pause_elapsed_seconds"], 20)
            self.assertIsNone(resumed.body["paused_at"])
            again = TrainingSession(task, "learning", {}, [], run_id=session.id)
            self.assertEqual(again.body["pause_elapsed_seconds"], 20)
            with patch("luban.training_checkpoint.time.time", return_value=1030):
                again.pause()
            with patch("luban.training_checkpoint.time.time", return_value=1045):
                twice = TrainingSession(task, "learning", {}, [], run_id=session.id)
            self.assertEqual(twice.body["pause_elapsed_seconds"], 35)

    def test_dead_owner_recovery_does_not_invent_pause_duration(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "learning", {}, [])
            session.body["owner"] = {"pid": 999999999, "started": 0}
            session._write("running")
            recover_commits(task.memory, task.store, task.identity)
            paused = training_runs(task.memory, task.identity)[0]
            self.assertEqual(paused["state"], "paused")
            self.assertNotIn("paused_at", paused)
            resumed = TrainingSession(task, "learning", {}, [], run_id=session.id)
            self.assertNotIn("pause_elapsed_seconds", resumed.body)

    def test_backwards_pause_clock_is_rejected_before_lease_change(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "learning", {}, [])
            with patch("luban.training_checkpoint.time.time", return_value=2000):
                session.pause()
            with patch("luban.training_checkpoint.time.time", return_value=1900):
                with self.assertRaisesRegex(ValueError, "暂停计时无效"):
                    TrainingSession(task, "learning", {}, [], run_id=session.id)
            row = training_runs(task.memory, task.identity)[0]
            self.assertEqual(row["state"], "paused")
            self.assertIsNone(row["owner"])

    def test_research_scope_persisted_on_first_insert_and_not_product_recovered(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            session = TrainingSession(task, "router", {}, [], scope="research")
            row = training_runs(task.memory, task.identity)[0]
            self.assertEqual(row["scope"], "research")
            self.assertEqual(row["binding"]["scope"], "research")
            artifact = task.store / "candidate.safetensors"
            artifact.write_bytes(b"validated fixture")
            session.prepare_commit({"calibrated": True}, {"active-router.json": {"id": session.id}},
                                   [artifact.name], event_kind="router")
            with self.assertRaisesRegex(ValueError, "研究候选"):
                recover_commits(task.memory, task.store, task.identity)
            self.assertFalse((task.store / "active-router.json").exists())
            self.assertEqual(training_runs(task.memory, task.identity)[0]["state"], "committing")
            recover_commits(task.memory, task.store, task.identity, research=True)
            for accepted in (False, 1, "true"):
                session.body["product_accepted"] = accepted
                session._write("committed")
                with self.assertRaisesRegex(ValueError, "禁止加载"):
                    ensure_product_state(task.memory, task.store, task.identity)
            session.body["product_accepted"] = True
            session._write("committed")
            ensure_product_state(task.memory, task.store, task.identity)

    def test_conflicting_commit_event_does_not_write_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            session = TrainingSession(task, "router", {}, [])
            artifact = task.store / "candidate.safetensors"
            artifact.write_bytes(b"validated fixture")
            session.prepare_commit({"calibrated": True}, {"active-router.json": {"id": session.id}},
                                   [artifact.name], event_kind="router")
            with task.memory.lock, task.memory.connect() as db:
                db.execute("INSERT INTO events VALUES (?,?,?,?)", ("training:" + session.id, "router", "{}", 1))
            with self.assertRaisesRegex(ValueError, "冲突"):
                session.finish_commit()
            self.assertFalse((task.store / "active-router.json").exists())

    def test_duplicate_commit_cannot_undo_later_user_restore(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            session = TrainingSession(task, "router", {}, [])
            artifact = task.store / "candidate.safetensors"
            artifact.write_bytes(b"validated fixture")
            session.prepare_commit({"calibrated": True}, {"active-router.json": {"id": session.id}},
                                   [artifact.name], event_kind="router")
            session.finish_commit()
            (task.store / "active-router.json").unlink()
            session.finish_commit()
            self.assertFalse((task.store / "active-router.json").exists())

    def test_adam_and_rng_resume_exact_after_real_sigkill(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "learning", {"steps": 12}, [{"id": "sample-1"}], run_id="abcdef123456")
            session.pause()  # Parent has released its lease before the child continues it.
            torch.manual_seed(42)
            expected = torch.nn.Linear(3, 2)
            optimizer = torch.optim.Adam(expected.parameters(), lr=.01)
            for _ in range(12):
                update(expected, optimizer)
            child = r'''
import os, signal, sys, torch
from pathlib import Path
from types import SimpleNamespace
from luban.memory import Memory
from luban.training_checkpoint import TrainingSession
root = sys.argv[1]
runtime = SimpleNamespace(memory=Memory(root),store=Path(root)/'modules',identity='checkpoint-test-base',active_core=None,branch=SimpleNamespace(expert_id=None),device='cpu',dtype=torch.float32)
session = TrainingSession(runtime,'learning',{'steps':12},[{'id':'sample-1'}],run_id='abcdef123456')
torch.manual_seed(42)
model=torch.nn.Linear(3,2)
optimizer=torch.optim.Adam(model.parameters(),lr=.01)
for step in range(6):
    optimizer.zero_grad(set_to_none=True)
    loss=model(torch.randn(4,3)).square().mean()
    loss.backward()
    optimizer.step()
session.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':6},{'step':6,'phase':'training'})
os.kill(os.getpid(), signal.SIGKILL)
'''
            process = subprocess.run([sys.executable, "-c", child, root], capture_output=True, timeout=30)
            self.assertEqual(process.returncode, -9, process.stderr.decode())
            resumed = TrainingSession(task, "learning", {"steps": 12}, [{"id": "sample-1"}], run_id=session.id)
            actual = torch.nn.Linear(3, 2)
            optimizer = torch.optim.Adam(actual.parameters(), lr=.01)
            saved = resumed.load()
            actual.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            for _ in range(saved["step"], 12):
                update(actual, optimizer)
            for name, value in expected.state_dict().items():
                self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)
            self.assertEqual(resumed.body["progress"]["step"], 6)

    def test_partial_unpublished_generation_keeps_last_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "router", {}, [])
            session.save({"weight": torch.tensor([1.]), "step": 1}, {"step": 1})
            (session.directory / ("f" * 32 + ".pt")).write_bytes(b"interrupted save")
            resumed = TrainingSession(task, "router", {}, [], run_id=session.id)
            self.assertEqual(resumed.load()["step"], 1)
            for step in range(2, 5):
                resumed.save({"weight": torch.tensor([float(step)]), "step": step}, {"step": step})
            self.assertEqual(len(list(session.directory.glob("*.pt"))), 2)

    def test_corrupt_checkpoint_and_changed_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "evolution", {"steps": 64}, [{"answer": "verified"}])
            session.save({"master": torch.ones(2, 2)}, {"step": 8})
            with self.assertRaises(ValueError):
                TrainingSession(task, "evolution", {"steps": 65}, [{"answer": "verified"}], run_id=session.id)
            with self.assertRaises(ValueError):
                TrainingSession(task, "evolution", {"steps": 64}, [{"answer": "changed"}], run_id=session.id)
            path = session.directory / session.body["checkpoint"]["name"]
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "SHA"):
                session.load()
            self.assertFalse((task.store / "active-core.json").exists())

    def test_corrupt_latest_falls_back_to_previous_valid_generation(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "learning", {}, [])
            session.save({"step": 8, "weight": torch.ones(1)}, {"step": 8})
            session.save({"step": 16, "weight": torch.ones(1) * 2}, {"step": 16})
            (session.directory / session.body["checkpoint"]["name"]).write_bytes(b"corrupt latest")
            self.assertEqual(session.load()["step"], 8)
            self.assertEqual(session.body["progress"]["step"], 8)
            self.assertIn("前一代", session.body["recovery_note"])

    def test_safe_load_normalizes_primitive_subclass_and_commit_rechecks_binding(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "router", {}, [])
            session.save({"torch": torch.__version__}, {"step": 1})
            self.assertIs(type(session.load()["torch"]), str)
            (task.store / "router.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "活动版本"):
                session.prepare_commit({"calibrated": True}, {}, [], event_kind="router")

    def test_one_pending_job_and_invalid_tensor_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            session = TrainingSession(task, "router", {}, [])
            with self.assertRaises(sqlite3.IntegrityError):
                TrainingSession(task, "learning", {}, [])
            with self.assertRaises(ValueError):
                session.save({"weight": torch.tensor([float("nan")])}, {})
            self.assertIsNone(session.body["checkpoint"])

    def test_crash_after_pointer_rolls_forward_commit_exactly_once(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            session = TrainingSession(task, "evolution", {}, [])
            artifact = task.store / "core-proof.safetensors"
            artifact.write_bytes(b"validated candidate fixture")
            target = {"version": "core-test", "identity": task.identity}
            report = {"id": session.id, "promoted": True, "publication_state": "published"}
            session.prepare_commit(report, {"active-core.json": target}, [artifact.name], event_kind="evolution")
            (task.store / "active-core.json").write_text(json.dumps(target))
            row = training_runs(task.memory, task.identity)[0]
            self.assertEqual(row["state"], "committing")
            reconcile_commit(task.memory, task.store, row)
            reconcile_commit(task.memory, task.store, row)
            self.assertEqual(training_runs(task.memory, task.identity)[0]["state"], "committed")
            with task.memory.lock, task.memory.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE kind='evolution'").fetchone()[0], 1)

    def test_bad_commit_artifact_cannot_change_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            session = TrainingSession(task, "router", {}, [])
            weights = task.store / "router-proof.safetensors"
            weights.write_bytes(b"valid")
            session.prepare_commit({"calibrated": True}, {"active-router.json": {"id": session.id}},
                                   [weights.name], event_kind="router")
            weights.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                reconcile_commit(task.memory, task.store, session.body)
            self.assertFalse((task.store / "active-router.json").exists())
            with self.assertRaises(ValueError):
                session.prepare_commit({"calibrated": True}, {"settings.json": {}}, [], event_kind="router")

    def test_native_rollback_intent_survives_crash_before_or_after_pointer_restore(self):
        for previous in (None, {"version": "core-old"}):
            for restored in (False, True):
                with self.subTest(previous=previous, restored=restored), tempfile.TemporaryDirectory() as root:
                    task = runtime(root)
                    task.store.mkdir()
                    pointer = task.store / "active-core.json"
                    if previous:
                        pointer.write_text(json.dumps(previous))
                        (task.store / "core-old.json").write_text("{}")
                        (task.store / "core-old.safetensors").write_bytes(b"accepted fixture")
                    session = TrainingSession(task, "evolution", {}, [])
                    candidate = task.store / "candidate.safetensors"
                    candidate.write_bytes(b"valid")
                    target = {"version": "candidate"}
                    session.prepare_commit({"promoted": True}, {pointer.name: target},
                                           [candidate.name], event_kind="evolution")
                    session.body["rollback_reason"] = "native reload failed"
                    session._write("committing")
                    selected = previous if restored else target
                    if selected is None:
                        pointer.unlink(missing_ok=True)
                    else:
                        pointer.write_text(json.dumps(selected))
                    recover_commits(task.memory, task.store, task.identity)
                    recover_commits(task.memory, task.store, task.identity)
                    self.assertEqual(json.loads(pointer.read_text()) if pointer.exists() else None, previous)
                    row = training_runs(task.memory, task.identity)[0]
                    self.assertEqual(row["state"], "rejected")
                    self.assertEqual(row["rollback_reason"], "native reload failed")
                    self.assertNotIn("actual_reload_confirmed", row)
                    self.assertEqual(task.memory.events(), [])

    def test_native_rollback_intent_rejects_rival_pointer_without_finishing(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            session = TrainingSession(task, "evolution", {}, [])
            session.prepare_commit({"promoted": True}, {"active-core.json": {"version": "candidate"}},
                                   [], event_kind="evolution")
            session.body["rollback_reason"] = "native reload failed"
            session._write("committing")
            pointer = task.store / "active-core.json"
            rival = {"version": "another-owner"}
            pointer.write_text(json.dumps(rival))
            with self.assertRaisesRegex(ValueError, "其他任务"):
                recover_commits(task.memory, task.store, task.identity)
            self.assertEqual(json.loads(pointer.read_text()), rival)
            self.assertEqual(training_runs(task.memory, task.identity)[0]["state"], "committing")
            self.assertEqual(task.memory.events(), [])

    def test_persisted_rollback_intent_overrides_stale_session_finish(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            session = TrainingSession(task, "evolution", {}, [])
            session.prepare_commit({"promoted": True}, {"active-core.json": {"version": "candidate"}},
                                   [], event_kind="evolution")
            body = {**session.body, "rollback_reason": "native reload failed"}
            body.pop("state", None)
            with task.memory.lock, task.memory.connect() as db:
                db.execute("UPDATE training_runs SET body=? WHERE id=?", (json.dumps(body), session.id))
            with self.assertRaisesRegex(ValueError, "回滚意图拒绝"):
                session.finish_commit()
            self.assertEqual(session.body["state"], "rejected")
            self.assertFalse((task.store / "active-core.json").exists())
            self.assertEqual(task.memory.events(), [])

    def test_startup_reverts_corrupt_prepared_commit_and_preserves_old_version(self):
        with tempfile.TemporaryDirectory() as root:
            task = runtime(root)
            task.store.mkdir()
            old = {"version": "previous-good"}
            (task.store / "active-core.json").write_text(json.dumps(old))
            (task.store / "previous-good.json").write_text("{}")
            (task.store / "previous-good.safetensors").write_bytes(b"accepted fixture")
            candidate = task.store / "candidate.safetensors"
            candidate.write_bytes(b"valid")
            session = TrainingSession(task, "evolution", {}, [])
            target = {"version": "candidate"}
            session.prepare_commit({"promoted": True}, {"active-core.json": target},
                                   [candidate.name], event_kind="evolution")
            (task.store / "active-core.json").write_text(json.dumps(target))
            candidate.write_bytes(b"damaged")
            recover_commits(task.memory, task.store, task.identity)
            self.assertEqual(json.loads((task.store / "active-core.json").read_text()), old)
            self.assertEqual(training_runs(task.memory, task.identity)[0]["state"], "rejected")
            with task.memory.lock, task.memory.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
