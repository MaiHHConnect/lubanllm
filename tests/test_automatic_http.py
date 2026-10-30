"""Exercise actual authenticated HTTP while an unverified candidate owns RAM."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

from luban.memory import Memory
from luban.server import create_app
from luban.settings import Settings
from luban.training_checkpoint import TrainingPaused, training_runs
from test_app import FakeRuntime


class AutomaticHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(root / "settings")
        self.runtime = FakeRuntime(Memory(root))
        self.runtime.identity, self.runtime.store = "http-automatic-base", root / "modules"
        self.runtime.device, self.runtime.dtype = "cpu", torch.float32
        self.runtime.branch = SimpleNamespace(expert_id=None)
        self.runtime.active_core = None
        self.runtime.training_session = None
        self.app = create_app(self.settings, runtime=self.runtime, autoload=False)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/session", json={"token": self.settings.secret("ui_token")})
        self.auto = self.app.state.automatic

    def start_candidate(self):
        started = threading.Event()
        self.runtime.weight = "accepted"
        def learn(steps):
            with self.runtime.lock:
                self.runtime.weight, self.runtime.state = "candidate", "learning"
                started.set()
                while not self.runtime.training_session.pause_requested:
                    time.sleep(.001)
                self.runtime.weight, self.runtime.state = "accepted", "ready"
                raise TrainingPaused()
        self.runtime.learn = learn
        self.auto._launch("learning", {"steps": 64}, {"train": [], "valid": [], "protection": []})
        self.assertTrue(started.wait(2))

    def test_chat_preempts_training_and_only_observes_accepted_weights(self):
        self.start_candidate()
        generate = self.runtime.generate
        observed = []
        def wrapped(*args, **kwargs):
            observed.append(self.runtime.weight)
            return generate(*args, **kwargs)
        self.runtime.generate = wrapped
        response = self.client.post("/api/chat", json={"messages": [{"role": "user", "content": "你好"}], "route": "local"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('"type":"error"', response.text)
        self.assertEqual(observed, ["accepted"])
        self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity)[0]["state"], "paused")

    def test_close_and_update_admission_share_worker_guard_and_cookie_boundary(self):
        self.start_candidate()
        self.assertTrue(self.client.get("/api/automatic/close-status").json()["running"])
        self.assertEqual(self.client.post("/api/update/prepare").status_code, 409)
        self.assertEqual(self.client.patch("/api/settings", json={"storage_root": "/tmp/changed-not-used"}).status_code, 409)
        self.assertTrue(self.client.post("/api/automatic/pause").json()["ok"])
        self.assertEqual(self.runtime.weight, "accepted")
        state = self.client.get("/api/automatic/close-status").json()
        self.assertFalse(state["running"])
        self.assertTrue(state["resumable"])
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/api/automatic/pause").status_code, 401)

    def test_learning_switch_saves_while_candidate_runs_without_pause_or_weight_change(self):
        self.start_candidate()
        job = self.runtime.job["id"]
        response = self.client.patch("/api/settings", json={"automatic": {"learning_enabled": False}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(Settings(self.settings.home).snapshot()["automatic"]["learning_enabled"])
        self.assertEqual(self.runtime.job["id"], job)
        self.assertEqual(self.runtime.weight, "candidate")
        self.assertFalse(self.runtime.training_session.pause_requested)
        self.assertTrue(self.auto.worker.is_alive())
