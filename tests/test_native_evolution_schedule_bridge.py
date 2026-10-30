"""Shared idle priorities remain; core scheduling never enters the Python trainer."""
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from luban.automatic import AutomaticLearning
from luban.memory import Memory
from luban.settings import Settings


class NativeEvolutionScheduleBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.runtime = SimpleNamespace(memory=Memory(root), identity="a" * 64, store=root / "modules",
                                       state="ready", active_core=None, branch=SimpleNamespace(expert_id=None))
        self.settings = Settings(root / "home")
        self.busy = False
        self.auto = AutomaticLearning(self.runtime, self.settings, threading.Lock(), lambda: self.busy)
        self.auto.last_activity = time.monotonic() - 100
        self.native = self.auto.native_evolution = Mock(return_value={"job_id": "native-job", "state": "running"})

    def test_public_evolution_actions_do_not_select_materials_or_create_python_sessions(self):
        with patch.object(self.auto, "_manifest", side_effect=AssertionError("Python core material selection")), \
                patch("luban.automatic.TrainingSession", side_effect=AssertionError("Python training session")):
            self.assertEqual(self.auto.start_manual("evolution"), {"job_id": "native-job", "state": "running"})
            self.auto.evolve("signature")
            self.auto.defer("signature")
            with self.assertRaisesRegex(ValueError, "学会后提前结束"):
                self.auto.start_manual("evolution", 451)
        self.assertEqual([call.args[0] for call in self.native.call_args_list], ["start", "evolve", "defer"])
        self.assertIsNone(self.auto.worker)

    def test_idle_priority_is_preserved_and_native_tick_runs_only_after_other_work(self):
        checked = []
        def missing(kind):
            checked.append(kind)
            raise ValueError("not ready")
        with patch.object(self.auto, "_manifest", side_effect=missing), \
                patch("luban.automatic.knowledge_keys", return_value=(set(), set(), None)):
            self.auto.on_idle = lambda: True
            self.auto.tick(datetime(2026, 10, 17, 2))
            self.native.assert_not_called()
            self.auto.on_idle = lambda: False
            self.auto.tick(datetime(2026, 10, 17, 2))
        self.assertEqual(checked, ["router"])
        self.native.assert_called_once_with("tick", {"now": "2026-10-17T02:00:00"})
        self.assertIsNone(self.auto.worker)
        self.assertFalse(self.auto.close_status()["running"])

    def test_model_busy_or_idle_cooldown_does_not_dispatch_native(self):
        self.busy = True
        self.auto.tick()
        self.busy = False
        self.auto.last_activity = time.monotonic()
        self.auto.tick()
        self.native.assert_not_called()

    def test_native_checkpoint_is_forwarded_without_python_claim_or_rejection(self):
        pending = {"id": "abcdef123456", "kind": "evolution", "scope": "product", "state": "paused"}
        with patch("luban.automatic.training_runs", return_value=[pending]), \
                patch.object(self.auto, "_check_resume_rights", side_effect=AssertionError("Python resume")), \
                patch("luban.automatic.TrainingSession", side_effect=AssertionError("Python checkpoint claim")):
            self.auto.tick()
        self.native.assert_called_once_with("resume", {"run_id": "abcdef123456"})
        self.assertEqual(pending["state"], "paused")

    def test_status_uses_native_experience_and_reminder_without_python_evolution_manifest(self):
        self.native.return_value = {"pending_signature": "native-signature", "experience": {"current": 350, "target": 350}, "reminder_due": True}
        with patch.object(self.auto, "_evolution_materials", side_effect=AssertionError("Python selection")), \
                patch.object(self.auto, "_manifest", side_effect=AssertionError("Python manifest")), \
                patch("luban.automatic.knowledge_keys", return_value=(set(), set(), None)):
            status = self.auto.status()
        self.assertEqual(status["experience"]["current"], 350)
        self.assertTrue(status["reminder_due"])
        self.native.assert_called_once_with("status", {})


if __name__ == "__main__":
    unittest.main()
