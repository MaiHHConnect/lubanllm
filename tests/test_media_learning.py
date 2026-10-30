"""Temporary private SQLite/files and synchronous fake native rounds; no model."""
import base64
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from luban.media_learning import MediaLearning
from luban.memory import Memory
from luban.operation_learning import prepare_demonstrations
from luban.settings import Settings
from test_operation_learning import demonstration


class ImmediateThread:
    def __init__(self, target, args=(), **kwargs): self.target, self.args = target, args
    def start(self): self.target(*self.args)
    def is_alive(self): return False
    def join(self, timeout=None): pass


class MediaLearningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.settings = Settings(self.root / "home")
        self.settings.update({"storage_root": str(self.root / "storage")})
        self.runtime = SimpleNamespace(memory=Memory(self.root / "storage"), lock=threading.RLock(),
            active_core=None, job={}, state="ready", model=object())
        self.stops = []
        def stop():
            self.stops.append(True); self.runtime.model = None; self.runtime.state = "stopped"
        self.runtime.stop = stop
        self.service = MediaLearning(self.runtime, self.settings)
        self.calls = []
        self.runner = self.round
        self.module = SimpleNamespace(ROUND_STEPS=5, capability=lambda runtime: {"available": True},
            train_round=lambda *args, **kwargs: self.runner(*args, **kwargs), prepare_demonstrations=prepare_demonstrations)
        def kind(value):
            if value not in {"image", "sound", "operation"}: raise ValueError("invalid kind")
            return self.module
        self.service._kind = kind
        self.addCleanup(self.service.close)

    def round(self, owner, samples, saved_candidate=None, cancel=None, progress=None):
        self.calls.append((owner, samples, saved_candidate))
        self.assertTrue(callable(cancel)); self.assertFalse(cancel())
        progress({"phase": "training", "step": 1, "steps": 5})
        return {"state": "awaiting_quality", "round_budget": 5, "completed_steps": 5,
                "total_adam_step": 5, "candidate_path": "a" * 12, "adopted": False, "quality_checked": False}

    def body(self, kind="image"):
        return {"kind": kind, "training_allowed": True, "answers_reviewed": True,
            "files": [{"name": f"{n}.png" if kind == "image" else f"{n}.wav", "question": "What?", "answer": "fixture",
                       "family": f"source{n}", "content": base64.b64encode(f"CPU fixture {n}".encode()).decode()}
                      for n in range(2)]}

    def create(self, kind="image"):
        return self.service.create(self.body(kind))["id"]

    def run_lesson(self, lesson_id):
        with patch("luban.media_learning.threading.Thread", ImmediateThread):
            self.service.start(lesson_id)
        return self.service._load(lesson_id)

    def test_upload_permission_and_private_saved_assets_without_public_paths(self):
        for field in ("training_allowed", "answers_reviewed"):
            body = self.body(); body[field] = False
            with self.assertRaises(ValueError): self.service.create(body)
        public = self.service.create(self.body())
        saved = self.service._load(public["id"])
        self.assertEqual([x["split"] for x in saved["samples"]], ["training", "validation"])
        for row in saved["samples"]:
            path = Path(row["asset_path"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertTrue(path.read_bytes().startswith(b"CPU fixture"))
        self.assertNotIn("asset_path", json.dumps(public)); self.assertNotIn("content", json.dumps(public))
        self.assertFalse(public["adopted"])

    def test_leakage_duplicate_file_invalid_name_encoding_and_batch_bound_reject(self):
        changes = [lambda b: b["files"][1].update(family="source0"),
                   lambda b: b["files"][1].update(content=b["files"][0]["content"]),
                   lambda b: b["files"][0].update(name="../escape.png"),
                   lambda b: b["files"][0].update(content="%bad-base64"),
                   lambda b: b["files"][0].update(answer=""),
                   lambda b: b["files"][0].update(family=""),
                   lambda b: b["files"][0].update(content="a" * (12 * 1024 * 1024))]
        for change in changes:
            body = self.body(); change(body)
            with self.subTest(change=change), self.assertRaises(ValueError): self.service.create(body)

    def test_fixed_round_retains_unadopted_candidate_and_next_explicit_round(self):
        lesson = self.create()
        first = self.run_lesson(lesson)
        self.assertEqual((first["round"], first["step"], first["total_adam_step"]), (1, 5, 5))
        self.assertFalse(first["adopted"]); self.assertFalse(first["protection_checked"])
        self.assertEqual(len(self.calls), 1)
        def second(owner, samples, saved_candidate=None, cancel=None, progress=None):
            self.assertEqual(saved_candidate, "a" * 12)
            progress({"phase": "training", "step": 1, "steps": 5})
            return {"state": "awaiting_quality", "round_budget": 5, "completed_steps": 5,
                    "total_adam_step": 10, "candidate_path": "b" * 12, "adopted": False}
        self.runner = second
        final = self.run_lesson(lesson)
        self.assertEqual((final["state"], final["round"], final["steps"], final["total_adam_step"]), ("awaiting_quality", 2, 5, 10))
        self.assertEqual(final["candidate"], "b" * 12)

    def test_cancel_callable_and_paused_partial_round_resumes_remainder(self):
        lesson = self.create()
        def paused(owner, samples, saved_candidate=None, cancel=None, progress=None):
            self.service.pause(); self.assertTrue(cancel())
            return {"state": "paused", "round_budget": 5, "completed_steps": 2,
                    "total_adam_step": 2, "candidate_path": "c" * 12, "adopted": False}
        self.runner = paused
        first = self.run_lesson(lesson)
        self.assertEqual((first["state"], first["round"], first["step"]), ("paused", 0, 2))
        def resumed(owner, samples, saved_candidate=None, cancel=None, progress=None):
            self.assertEqual(saved_candidate, "c" * 12); self.assertFalse(cancel())
            progress({"phase": "training", "step": 3, "steps": 5})
            return {"state": "awaiting_quality", "round_budget": 5, "completed_steps": 5,
                    "total_adam_step": 5, "candidate_path": "d" * 12, "adopted": False}
        self.runner = resumed
        after = self.run_lesson(lesson)
        self.assertEqual((after["round"], after["total_adam_step"]), (1, 5))

    def test_claimed_adoption_and_fabricated_result_progress_are_not_saved(self):
        for corrupt in ({"adopted": True}, {"completed_steps": True}, {"completed_steps": 6},
                        {"total_adam_step": 999}, {"candidate_path": "/tmp/foreign.pt"}, {"round_budget": 450}):
            lesson = self.create()
            def invalid(*args, **kwargs):
                return {"state": "awaiting_quality", "round_budget": 5, "completed_steps": 5,
                    "total_adam_step": 5, "candidate_path": "a" * 12, "adopted": False, **corrupt}
            self.runner = invalid
            result = self.run_lesson(lesson)
            with self.subTest(corrupt=corrupt):
                self.assertEqual(result["state"], "error")
                self.assertIsNone(result["candidate"])
                self.assertEqual(result["total_adam_step"], 0)

    def test_partial_exception_preserves_only_valid_complete_checkpoint(self):
        for invalid in (False, True):
            lesson = self.create()
            def failed(*args, **kwargs):
                error = RuntimeError("fixture interruption")
                error.candidate = {"state": "paused", "round_budget": 5, "completed_steps": 999 if invalid else 2,
                    "total_adam_step": 999 if invalid else 2, "candidate_path": "e" * 12, "adopted": False}
                raise error
            self.runner = failed
            result = self.run_lesson(lesson)
            self.assertEqual(result["state"], "error")
            self.assertEqual(result["candidate"], None if invalid else "e" * 12)
            self.assertEqual(result["step"], 0 if invalid else 2)

    def test_saved_checkpoint_event_survives_later_unsaved_error(self):
        lesson = self.create()
        def failed(owner, samples, saved_candidate=None, cancel=None, progress=None):
            candidate = {"state": "paused", "round_budget": 5, "completed_steps": 2,
                "total_adam_step": 2, "candidate_path": "1" * 12, "adopted": False}
            progress({"phase": "training", "step": 2, "steps": 5, "candidate": candidate})
            durable = self.service._load(lesson)
            self.assertEqual((durable["candidate"], durable["checkpoint_step"], durable["checkpoint_total"]),
                             ("1" * 12, 2, 2))
            progress({"phase": "training", "step": 3, "steps": 5})
            raise RuntimeError("unsaved next update")
        self.runner = failed
        final = self.run_lesson(lesson)
        self.assertEqual((final["state"], final["candidate"], final["step"], final["total_adam_step"]),
                         ("error", "1" * 12, 2, 2))

    def test_forged_display_progress_and_checkpoint_event_reject(self):
        for value in ({"step": 999}, {"step": True}, {"steps": 450},
                      {"candidate": {"state": "paused", "round_budget": 5, "completed_steps": 2,
                       "total_adam_step": 7, "candidate_path": "2" * 12, "adopted": False}}):
            lesson = self.create()
            def invalid(owner, samples, saved_candidate=None, cancel=None, progress=None):
                progress({"phase": "training", **value})
                return {"state": "awaiting_quality", "round_budget": 5, "completed_steps": 5,
                        "total_adam_step": 5, "candidate_path": "2" * 12, "adopted": False}
            self.runner = invalid
            final = self.run_lesson(lesson)
            with self.subTest(value=value):
                self.assertEqual((final["state"], final["candidate"], final["step"]), ("error", None, 0))

    def test_operation_saved_provenance_rewrite_rejected_before_training(self):
        directory = self.settings.home / "operation-demonstrations"
        directory.mkdir(mode=0o700)
        selections = []
        for index in range(2):
            file_id = str(uuid.uuid4())
            path = directory / (file_id + ".json")
            path.write_text(json.dumps(demonstration(index))); path.chmod(0o600)
            selections.append({"file_id": file_id, "split": "TRAIN" if index == 0 else "HOLDOUT"})
        lesson = self.service.create({"kind": "operation", "answers_reviewed": True,
            "training_allowed": True, "demonstrations": selections})["id"]
        stored = self.service._load(lesson)
        self.assertTrue(all("image" not in row and "rgb" not in row for row in stored["samples"]))
        changed = demonstration(0); changed["steps"][0]["observation"]["rgb"][0] = 89
        (directory / (selections[0]["file_id"] + ".json")).write_text(json.dumps(changed))
        final = self.run_lesson(lesson)
        self.assertEqual(final["state"], "error"); self.assertEqual(self.calls, [])
        self.assertIsNone(final["candidate"])

    def test_dead_owner_restart_keeps_checkpoint_live_owner_cannot_restart(self):
        lesson = self.create()
        saved = self.service._load(lesson)
        saved.update(state="running", owner={"pid": os.getpid(), "started": 1}, candidate="f" * 12, step=4, total_adam_step=4, checkpoint_step=2, checkpoint_total=2)
        self.service._write(saved)
        with patch("luban.media_learning._owner_alive", return_value=False):
            recovered = self.service._load(lesson)
        self.assertEqual((recovered["state"], recovered["candidate"], recovered["step"]), ("interrupted", "f" * 12, 2))
        recovered.update(state="running", owner={"pid": os.getpid() + 10000, "started": 1})
        self.service._write(recovered)
        with patch("luban.media_learning._owner_alive", return_value=True), self.assertRaises((RuntimeError, ValueError)):
            self.service.start(lesson)

    def test_start_failure_clears_owner_and_restores_runtime(self):
        for failure in ("thread", "write"):
            lesson = self.create()
            previous_job = {"state": "complete", "kind": "previous"}
            self.runtime.job = previous_job
            original_write = self.service._write
            writes = []
            def write(body):
                writes.append(body["state"])
                if failure == "write" and len(writes) == 1:
                    raise OSError("fixture database write failed")
                original_write(body)
            with patch.object(self.service, "_write", side_effect=write), patch(
                    "luban.media_learning.threading.Thread.start", side_effect=RuntimeError("fixture thread failed")):
                with self.assertRaises((RuntimeError, OSError)):
                    self.service.start(lesson)
            self.assertEqual(self.runtime.state, "ready")
            self.assertIs(self.runtime.job, previous_job)
            self.assertIsNone(self.service.worker)
            saved = self.service._load(lesson)
            self.assertEqual(saved["state"], "error")
            self.assertIsNone(saved["owner"])
            self.assertIsNone(saved["candidate"])

    def test_changed_model_or_core_blocks_old_lesson_and_sound_releases_owner(self):
        lesson = self.create()
        self.runtime.active_core = "core-changed"
        with self.assertRaises(ValueError): self.service.start(lesson)
        self.runtime.active_core = None
        saved = self.service._load(lesson); saved["binding"]["model_path"] = "/different"
        self.service._write(saved)
        with self.assertRaises(ValueError): self.service.start(lesson)
        sound = self.create("sound")
        self.run_lesson(sound)
        self.assertEqual(len(self.stops), 1)
        self.assertIs(self.calls[-1][0], self.settings)
        self.assertEqual(self.runtime.state, "stopped")


if __name__ == "__main__": unittest.main()
