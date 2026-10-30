"""Model lifecycle and HTTP admission on CPU fixtures, without upstream weights."""
import json
from pathlib import Path
import secrets
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import torch

from luban.memory import Memory
from luban.automatic import AutomaticLearning
from luban.process_lock import ModelProcessLock
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings


class ModelStopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = Settings(self.root / "private")
        self.settings.update({"storage_root": str(self.root / "storage")})
        self.runtime = Runtime(self.settings, Memory(self.root / "storage"))
        self.runtime.device, self.runtime.dtype = "cpu", torch.float32
        self.addCleanup(self.runtime._release_model)

    def resident(self):
        runtime = self.runtime
        runtime.model = runtime.decoder = torch.nn.Linear(4, 4)
        runtime.tokenizer = object()
        runtime.branch = SimpleNamespace(expert_id=None)
        runtime.router = torch.nn.Linear(4, 3)
        runtime.text_config = SimpleNamespace(hidden_size=4)
        runtime.active_core = "fixture-core"
        runtime.base_count = runtime.language_count = 20
        runtime.loaded_path = runtime.loaded_storage = "fixture"
        runtime.state = "ready"

    def client(self):
        client = TestClient(create_app(self.settings, self.runtime, autoload=False),
                            raise_server_exceptions=False)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def login(self, client):
        self.assertEqual(client.post("/api/session", json={
            "token": self.settings.secret("ui_token")}).status_code, 200)

    def test_manual_model_start_keeps_idle_monitor_without_autoloading_weights(self):
        self.runtime.state = "stopped"
        app = create_app(self.settings, self.runtime, autoload=False)
        automatic = app.state.automatic
        observed = threading.Event()
        with patch.object(self.runtime, "load") as load, patch.object(automatic, "tick", side_effect=observed.set):
            with TestClient(app):
                self.assertTrue(automatic.monitor.is_alive())
                self.assertEqual(self.runtime.state, "stopped")
                self.assertFalse(observed.wait(.1))
                load.assert_not_called()
                self.runtime.state = "ready"
                self.assertTrue(observed.wait(3), "the real background monitor must tick after manual readiness")
                load.assert_not_called()
            self.assertTrue(automatic.stop_event.is_set())

    def test_stop_releases_native_hf_and_process_lock_without_writing_storage(self):
        self.resident()
        runtime = self.runtime
        native = runtime.native_engine = Mock()
        identity = secrets.token_hex(32)
        lease = runtime._model_process_lock = ModelProcessLock(identity, "cpu")
        stored = self.root / "storage" / "saved-weights"
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(b"keep accepted weights")
        self.assertEqual(runtime.stop(), {"ok": True})

        native.close.assert_called_once_with()
        self.assertIsNone(lease.fd)
        reopened = ModelProcessLock(identity, "cpu")
        reopened.close()
        for name in ("model", "decoder", "tokenizer", "branch", "router", "native_engine",
                     "_model_process_lock", "active_core", "text_config"):
            self.assertIsNone(getattr(runtime, name), name)
        self.assertEqual(runtime.status()["state"], "stopped")
        self.assertEqual(runtime.status()["language_parameters"], 0)
        self.assertEqual(stored.read_bytes(), b"keep accepted weights")
        self.assertEqual(runtime.stop(), {"ok": True})

    def test_stopped_generation_reports_not_ready_without_loading(self):
        self.resident()
        self.runtime.stop()
        with self.assertRaisesRegex(RuntimeError, "模型尚未就绪"):
            self.runtime.generate([{"role": "user", "content": "你好"}], Mock(), threading.Event())

    def test_saved_generation_speed_survives_restart_and_model_release(self):
        usage = {"decode_tokens_per_second": 51.5}
        (self.settings.home / "last-generation-usage.json").write_text(json.dumps(usage))
        restarted = Runtime(self.settings, self.runtime.memory)
        self.addCleanup(restarted._release_model)
        self.assertEqual(restarted.status()["last_generation_usage"], usage)
        restarted._release_model()  # Also called while loading a model again.
        self.assertEqual(restarted.status()["last_generation_usage"], usage)

    def test_stopped_status_keeps_dashboard_history_and_model_metadata(self):
        self.resident()
        self.runtime.memory.record_usage("local", {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.runtime.stop()
        client = self.client()
        self.login(client)
        status = client.get("/api/status").json()
        self.assertEqual(status["runtime"]["state"], "stopped")
        self.assertEqual(status["runtime"]["language_parameters"], 20)
        self.assertEqual(status["usage"]["local"]["total_tokens"], 5)

    def test_runtime_rejects_busy_states_jobs_and_training_sessions(self):
        self.resident()
        runtime = self.runtime
        with patch.object(runtime, "_release_model") as release:
            for state in ("loading", "generating", "learning", "evolving", "evaluating", "stopping"):
                runtime.state = state
                with self.subTest(state=state), self.assertRaises(RuntimeError):
                    runtime.stop()
            runtime.state = "ready"
            for state in ("queued", "running"):
                runtime.job = {"state": state}
                with self.subTest(job=state), self.assertRaises(RuntimeError):
                    runtime.stop()
            runtime.job = None
            runtime.training_session = object()
            with self.assertRaises(RuntimeError):
                runtime.stop()
            runtime.training_session = None
            runtime._external_probe_active = True
            with self.assertRaises(RuntimeError):
                runtime.stop()
            release.assert_not_called()

    def test_locked_runtime_refuses_immediately_without_releasing_weights(self):
        self.resident()
        entered, finish = threading.Event(), threading.Event()

        def hold():
            with self.runtime.lock:
                entered.set()
                finish.wait(5)

        with ThreadPoolExecutor(max_workers=1) as pool:
            held = pool.submit(hold)
            self.assertTrue(entered.wait(5))
            try:
                with self.assertRaises(RuntimeError):
                    self.runtime.stop()
                self.assertIsNotNone(self.runtime.model)
            finally:
                finish.set()
            held.result()
        self.assertEqual(self.runtime.stop(), {"ok": True})

    def test_stop_failure_reports_error_then_allows_retry(self):
        self.resident()
        with patch.object(self.runtime, "_release_model", side_effect=RuntimeError("close failed")):
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                self.runtime.stop()
        self.assertEqual(self.runtime.state, "error")
        self.assertEqual(self.runtime.error, "close failed")
        self.assertIsNotNone(self.runtime.model)
        self.runtime.stop()
        self.assertEqual((self.runtime.state, self.runtime.error), ("stopped", ""))

    def test_native_close_failure_is_reported_without_claiming_release(self):
        self.resident()
        native = self.runtime.native_engine = Mock()
        native.close.side_effect = RuntimeError("native close failed")
        with self.assertRaisesRegex(RuntimeError, "native close failed"):
            self.runtime.stop()
        self.assertEqual(self.runtime.state, "error")
        self.assertEqual(self.runtime.error, "native close failed")
        self.assertIsNotNone(self.runtime.model)
        # Existing close() detaches the engine before closing it; failure is
        # surfaced, rather than pretending that the process was released.
        self.assertIsNone(self.runtime.native_engine)
        native.close.assert_called_once_with()

    def test_idle_automatic_ticks_leave_stopped_model_unloaded(self):
        self.resident()
        self.runtime.identity = "a" * 64
        self.runtime.store = self.root / "storage" / "modules"
        automatic = AutomaticLearning(self.runtime, self.settings, threading.Lock(), lambda: False)
        automatic.last_activity = 0
        automatic.on_idle = Mock()
        self.runtime.stop()
        with patch.object(self.runtime, "load") as load, \
                patch.object(automatic, "_manifest") as manifest, \
                patch.object(automatic, "_launch") as launch:
            for _ in range(3):
                automatic.tick()
            load.assert_not_called()
            manifest.assert_not_called()
            launch.assert_not_called()
            automatic.on_idle.assert_not_called()
        self.assertEqual(self.runtime.state, "stopped")
        self.assertIsNone(self.runtime.model)
        self.assertIsNone(automatic.worker)

    def test_stop_requires_session_and_respects_update_lease(self):
        self.resident()
        client = self.client()
        self.assertEqual(client.post("/api/model/stop").status_code, 401)
        self.login(client)
        self.assertEqual(client.post("/api/update/prepare").status_code, 200)
        self.assertEqual(client.post("/api/model/stop").status_code, 409)
        self.assertIsNotNone(self.runtime.model)
        self.assertEqual(client.post("/api/update/cancel").status_code, 200)
        self.assertEqual(client.post("/api/model/stop").status_code, 200)
        self.assertEqual(client.get("/api/health").json()["state"], "stopped")

    def test_stop_api_preserves_training_and_active_automatic_worker(self):
        self.resident()
        client = self.client()
        self.login(client)
        for state in ("loading", "generating", "learning", "evolving", "evaluating"):
            self.runtime.state = state
            self.assertEqual(client.post("/api/model/stop").status_code, 409)
        self.runtime.state = "ready"
        self.runtime.job = {"state": "queued"}
        self.assertEqual(client.post("/api/model/stop").status_code, 409)
        self.runtime.job = None
        with patch("luban.server.AutomaticLearning.close_status", return_value={"running": True}):
            self.assertEqual(client.post("/api/model/stop").status_code, 409)
        self.assertIsNotNone(self.runtime.model)

    def test_stop_api_failure_returns_error_and_releases_command_admission(self):
        self.resident()
        client = self.client()
        self.login(client)
        with patch.object(self.runtime, "_release_model", side_effect=RuntimeError("close failed")):
            self.assertEqual(client.post("/api/model/stop").status_code, 409)
            self.assertEqual(client.get("/api/health").json()["state"], "error")
        self.assertEqual(client.post("/api/model/stop").status_code, 200)

    def test_inflight_request_blocks_stop_before_runtime_state_changes(self):
        self.resident()
        client = self.client()
        self.login(client)
        entered, finish = threading.Event(), threading.Event()

        def generate(messages, emit, stop, **kwargs):
            entered.set()
            finish.wait(5)
            emit({"content": "fixture answer"})
            return {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                    "finish_reason": "stop"}

        with patch.object(self.runtime, "generate", side_effect=generate), \
                ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(client.post, "/api/chat", json={
                "messages": [{"role": "user", "content": "你好"}], "route": "local"})
            self.assertTrue(entered.wait(5))
            try:
                self.assertEqual(self.runtime.state, "ready")
                self.assertEqual(client.post("/api/model/stop").status_code, 409)
                self.assertIsNotNone(self.runtime.model)
            finally:
                finish.set()
            self.assertEqual(request.result().status_code, 200)
        self.assertEqual(client.post("/api/model/stop").status_code, 200)

    def test_stop_holds_command_admission_until_release_finishes(self):
        self.resident()
        client = self.client()
        self.login(client)
        entered, finish = threading.Event(), threading.Event()
        original = self.runtime._release_model

        def release():
            entered.set()
            finish.wait(5)
            original()

        with patch.object(self.runtime, "_release_model", side_effect=release), \
                ThreadPoolExecutor(max_workers=1) as pool:
            stopping = pool.submit(client.post, "/api/model/stop")
            self.assertTrue(entered.wait(5))
            try:
                self.assertEqual(client.post("/api/model/reload").status_code, 409)
                self.assertEqual(client.post("/api/model/stop").status_code, 409)
            finally:
                finish.set()
            self.assertEqual(stopping.result().status_code, 200)

    def test_learning_versions_include_file_creation_time(self):
        store = self.root / "storage" / "layer-27"
        store.mkdir(parents=True)
        (store / "learned-fixture.json").write_text(json.dumps({"hidden": 4, "width": 8}))
        self.runtime.branch = SimpleNamespace(store=store)
        versions = self.runtime.versions()
        self.assertEqual(versions[0]["id"], "learned-fixture")
        self.assertEqual(versions[0]["parameters"], 96)
        self.assertRegex(versions[0]["created_at"], r"^20\d\d-\d\d-\d\dT")

    def test_learning_history_is_available_before_model_start(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 24}}))
        self.settings.update({"model_path": str(model)})
        store = self.root / "storage/modules/luban0-v1/layer-23"
        store.mkdir(parents=True)
        (store / "learned-history.json").write_text(json.dumps({"hidden": 4, "width": 8}))
        self.assertIsNone(self.runtime.branch)
        with patch.object(self.runtime, "load", side_effect=AssertionError("history must not start model")):
            versions = self.runtime.versions()
        self.assertEqual(versions[0]["id"], "learned-history")
        self.assertEqual(versions[0]["parameters"], 96)
        self.assertRegex(versions[0]["created_at"], r"^20\d\d-\d\d-\d\dT")
        self.assertEqual(self.runtime.state, "stopped")
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 28}))
        self.assertEqual(self.runtime.versions(), [])  # Do not show another model family's records.

    def test_learning_attempt_history_reports_rejection_without_loading_or_replacing_versions(self):
        from luban.learning_history import latest_learning_attempt
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 24}}))
        self.settings.update({"model_path": str(model)})
        store = self.root / "storage/modules/luban0-v1"
        (store / "layer-23").mkdir(parents=True)
        (store / "active.json").write_text(json.dumps({"identity": "history-model"}))
        (store / "layer-23/learned-old.json").write_text(json.dumps({"hidden": 4, "width": 8}))
        with self.runtime.memory.connect() as db:
            db.execute("CREATE TABLE training_runs (id TEXT,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)")
            for identifier, identity, kind, state, started, updated, scope in (
                    ("old-success", "history-model", "learning", "committed", 1000, 9000, "product"),
                    ("latest-failure", "history-model", "learning", "rejected", 2000, 2001, "product"),
                    ("experiment", "history-model", "learning", "running", 3000, 3000, "research"),
                    ("other-model", "other", "learning", "running", 4000, 4000, "product"),
                    ("router", "history-model", "router", "committed", 5000, 5000, "product")):
                db.execute("INSERT INTO training_runs VALUES (?,?,?,?,?,?)", (identifier, identity, kind, state,
                    json.dumps({"started": started, "scope": scope, "progress": {"step": 0}}), updated))
        with patch.object(self.runtime, "load", side_effect=AssertionError("history must not start model")):
            attempt = latest_learning_attempt(self.settings, self.runtime.memory)
            self.assertEqual(attempt, {"id": "latest-failure", "started_at": "1970-01-01T00:33:20Z", "state": "rejected"})
            self.assertEqual(self.runtime.versions()[0]["id"], "learned-old")
            self.assertEqual(latest_learning_attempt(self.settings, self.runtime.memory, "empty-model"), None)
            self.assertEqual(self.runtime.state, "stopped")
            client = self.client()
            self.login(client)
            response = client.get("/api/experts")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["last_attempt"], attempt)
            self.assertEqual(response.json()["versions"][0]["id"], "learned-old")

    def test_learning_attempt_history_without_a_journal_has_no_invented_date(self):
        from luban.learning_history import latest_learning_attempt
        self.assertIsNone(latest_learning_attempt(self.settings, self.runtime.memory, "model"))

    def test_factory_baseline_is_not_personal_history_or_last_attempt(self):
        from luban.learning_history import latest_learning_attempt, saved_learning_versions
        from luban.training_checkpoint import _schema
        store = self.root / 'storage/modules/luban0-v1/layer-23'
        store.mkdir(parents=True)
        for identifier in ('factory', 'personal'):
            (store / ('learned-' + identifier + '.json')).write_text(json.dumps(dict(hidden=4, width=8)))
        report = dict(active_expert='learned-factory', promoted=True)
        factory = dict(factory_baseline=True, started=3000, scope='product',
                       commit=dict(event_kind='learning', report=report))
        with self.runtime.memory.connect() as db:
            _schema(db)
            db.execute('INSERT INTO training_runs VALUES(?,?,?,?,?,?)',
                       ('factory', 'model', 'learning', 'committed', json.dumps(factory), 3001))
            db.execute('INSERT INTO events VALUES(?,?,?,?)',
                       ('training:factory', 'factory:learning', json.dumps(report), 3000))
        self.assertIsNone(latest_learning_attempt(self.settings, self.runtime.memory, 'model'))
        self.assertEqual([r['id'] for r in saved_learning_versions(self.settings, store)], ['learned-personal'])
        with self.runtime.memory.connect() as db:
            db.execute('INSERT INTO training_runs VALUES(?,?,?,?,?,?)',
                       ('personal', 'model', 'learning', 'rejected', json.dumps(dict(started=2000, scope='product')), 2001))
        self.assertEqual(latest_learning_attempt(self.settings, self.runtime.memory, 'model')['id'], 'personal')
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE events SET body='{}' WHERE id='training:factory'")
        with self.assertRaisesRegex(ValueError, '出厂学习基线'):
            saved_learning_versions(self.settings, store)
        with self.assertRaisesRegex(ValueError, '出厂学习基线'):
            latest_learning_attempt(self.settings, self.runtime.memory, 'model')

    def test_stop_then_reload_uses_real_cpu_load_lifecycle(self):
        model_path = self.root / "model"
        model_path.mkdir()
        (model_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        (model_path / "upstream-file-manifest.json").write_text('{"Data":{"Files":[]}}')
        self.settings.update({"model_path": str(model_path)})
        runtime = self.runtime
        client = self.client()
        self.login(client)
        ready = threading.Event()
        original_load = runtime.load

        def load():
            try:
                return original_load(paging=False)
            finally:
                ready.set()

        def backbone(*args):
            model = torch.nn.Linear(4, 4)
            model.config = SimpleNamespace(model_type="qwen3")
            return model

        with patch.object(runtime, "load", side_effect=load), \
                patch("luban.runtime.load_backbone", side_effect=backbone) as allocate, \
                patch("luban.runtime.text_backbone", side_effect=lambda model: (
                    model, SimpleNamespace(hidden_size=4, num_hidden_layers=1))), \
                patch("luban.runtime.AutoTokenizer.from_pretrained", return_value=Mock()), \
                patch("luban.runtime.insert_experts", side_effect=lambda *a, **k: [SimpleNamespace(expert_id=None)]), \
                patch("luban.evolution.load_active_core"), patch.object(runtime, "_load_router"):
            for _ in range(2):
                ready.clear()
                self.assertEqual(client.post("/api/model/reload").status_code, 200)
                self.assertTrue(ready.wait(5))
                self.assertEqual(runtime.state, "ready", runtime.error)
                self.assertIsNotNone(runtime.model)
                self.assertEqual(client.post("/api/model/stop").status_code, 200)
                self.assertIsNone(runtime.model)
            self.assertEqual(allocate.call_count, 2)
