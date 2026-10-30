"""Actual lock handoff and concurrent cancellation, without allocating a model."""
import threading
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from luban.native_evolution_lease import NativeEvolutionLease


class NativeEvolutionLeaseTests(unittest.TestCase):
    def setUp(self):
        self.commands = threading.Lock()
        self.runtime = SimpleNamespace(lock=threading.RLock(), state="ready", identity="original",
            active_core=None, store="/model-store", memory=SimpleNamespace(root="/data"),
            settings=SimpleNamespace(snapshot=lambda: {"model_path": "/model"}))
        self.stops = 0

        def stop():
            with self.runtime.lock:
                self.stops += 1
                self.runtime.state = "stopped"
        self.runtime.stop = stop
        self.lease = NativeEvolutionLease(self.runtime, self.commands, lambda: None)
        self.identifier = str(uuid.uuid4())
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.lease.active:
            self.lease.release(self.identifier)

    def test_same_owner_holds_both_locks_until_release_and_acquire_is_idempotent(self):
        result = self.lease.acquire(self.identifier)
        self.assertEqual(result, self.lease.acquire(self.identifier))
        self.assertTrue(result["was_ready"])
        self.assertEqual(result["model"], "/model")
        self.assertEqual(self.stops, 1)
        self.assertFalse(self.commands.acquire(blocking=False))
        self.assertFalse(self.runtime.lock.acquire(blocking=False))
        released = self.lease.release(self.identifier)
        self.assertEqual(released, {"ok": True, "released": True, "was_ready": True})
        self.assertEqual(released, self.lease.release(self.identifier))
        self.assertTrue(self.commands.acquire(blocking=False))
        self.commands.release()
        self.assertTrue(self.runtime.lock.acquire(blocking=False))
        self.runtime.lock.release()
        self.assertEqual(self.runtime.state, "stopped")  # Bun chooses when to reload.
        with self.assertRaises(ValueError):
            self.lease.acquire(self.identifier)

    def test_other_lease_cannot_start_or_release_active_owner(self):
        self.lease.acquire(self.identifier)
        with self.assertRaises(RuntimeError):
            self.lease.acquire(str(uuid.uuid4()))
        with self.assertRaises(ValueError):
            self.lease.release(str(uuid.uuid4()))
        self.assertTrue(self.lease.active)
        self.assertEqual(self.stops, 1)

    def test_preparation_requires_current_owner_and_release_waits_for_cleanup(self):
        with self.assertRaises(ValueError):
            self.lease.prepare(self.identifier, lambda: None)
        self.lease.acquire(self.identifier)
        with self.assertRaises(ValueError):
            self.lease.prepare(str(uuid.uuid4()), lambda: None)
        entered, finish = threading.Event(), threading.Event()
        def operation():
            entered.set()
            self.assertTrue(finish.wait(3))
            return "closed"
        result, released = [], []
        first = threading.Thread(target=lambda: result.append(self.lease.prepare(self.identifier, operation)))
        first.start()
        self.assertTrue(entered.wait(3))
        second = threading.Thread(target=lambda: released.append(self.lease.release(self.identifier)))
        second.start()
        self.assertFalse(self.commands.acquire(blocking=False))
        finish.set()
        first.join(3); second.join(3)
        self.assertEqual(result, ["closed"])
        self.assertTrue(released[0]["released"])
        with self.assertRaises(ValueError):
            self.lease.prepare(self.identifier, lambda: None)

    def test_busy_command_and_runtime_fail_before_unload(self):
        for lock in (self.commands, self.runtime.lock):
            with lock:
                with self.assertRaises(RuntimeError):
                    self.lease.acquire(self.identifier)
            self.assertFalse(self.lease.active)
            self.assertEqual(self.stops, 0)
        self.assertEqual(self.lease.release(self.identifier), {"ok": True, "released": False, "was_ready": False})

    def test_unload_failure_releases_both_locks_without_claiming_success(self):
        self.runtime.stop = lambda: (_ for _ in ()).throw(RuntimeError("unload failed"))
        with self.assertRaisesRegex(RuntimeError, "unload failed"):
            self.lease.acquire(self.identifier)
        self.assertFalse(self.lease.active)
        self.assertTrue(self.commands.acquire(blocking=False))
        self.commands.release()
        self.assertTrue(self.runtime.lock.acquire(blocking=False))
        self.runtime.lock.release()

    def test_release_during_unload_waits_for_actual_unload_before_unlock(self):
        entered, proceed = threading.Event(), threading.Event()

        def stop():
            entered.set()
            if not proceed.wait(3):
                raise RuntimeError("test unload timed out")
            self.runtime.state = "stopped"
        self.runtime.stop = stop
        acquired, released = [], []
        first = threading.Thread(target=lambda: acquired.append(self.lease.acquire(self.identifier)))
        first.start()
        self.assertTrue(entered.wait(3))
        second = threading.Thread(target=lambda: released.append(self.lease.release(self.identifier)))
        second.start()
        self.assertFalse(self.commands.acquire(blocking=False))
        self.assertFalse(self.runtime.lock.acquire(blocking=False))
        proceed.set()
        first.join(3)
        second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(acquired[0]["lease_id"], self.identifier)
        self.assertTrue(released[0]["was_ready"])
        self.assertFalse(self.lease.active)

    def test_stopped_model_and_invalid_identifiers(self):
        for invalid in (None, "", "../file", 1):
            with self.assertRaises(ValueError):
                self.lease.acquire(invalid)
        self.runtime.state = "stopped"
        self.assertFalse(self.lease.acquire(self.identifier)["was_ready"])
        self.assertFalse(self.lease.release(self.identifier)["was_ready"])

    def test_private_http_handoff_authentication_and_mutation_block(self):
        from fastapi.testclient import TestClient
        from luban.memory import Memory
        from luban.server import create_app
        from luban.settings import Settings
        from test_app import FakeRuntime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(root / "home")
            settings.update({"storage_root": str(root / "data")})
            runtime = FakeRuntime(Memory(root / "data"))
            runtime.settings, runtime.store = settings, root / "data/modules/luban0-v1"
            runtime.identity, runtime.active_core = "a" * 64, None
            runtime.stop = lambda: setattr(runtime, "state", "stopped")
            with patch("luban.bun_bridge.enabled", return_value=True), patch.dict(os.environ, {
                    "LUBAN_COMPUTE_TOKEN": "lease-test-token", "LUBAN_PUBLIC_PORT": "8765"}):
                app = create_app(settings, runtime=runtime, autoload=False)
                with TestClient(app, client=("127.0.0.1", 50000)) as client:
                    path = "/internal/native-evolution/acquire"
                    body = {"lease_id": self.identifier}
                    self.assertEqual(client.post(path, json=body).status_code, 401)
                    client.headers["Authorization"] = "Bearer lease-test-token"
                    response = client.post(path, json=body)
                    self.assertEqual(response.status_code, 200, response.text)
                    try:
                        self.assertEqual(client.post(path, json=body).json(), response.json())
                        self.assertEqual(client.post("/api/model/reload", json={}).status_code, 409)
                        self.assertEqual(client.post("/v1/chat/completions", json={}).status_code, 409)
                        self.assertEqual(client.patch("/api/settings", json={}).status_code, 409)
                        for enabled in (False, True, False):
                            changed = client.patch("/api/settings", json={"automatic": {"learning_enabled": enabled}})
                            self.assertEqual(changed.status_code, 200, changed.text)
                            with patch("luban.bun_bridge.enabled", return_value=False):
                                self.assertEqual(Settings(root / "home").snapshot()["automatic"]["learning_enabled"], enabled)
                            self.assertTrue(app.state.native_evolution_lease.active)
                        for change in ({"automatic": {"learning_enabled": False}, "temperature": .5},
                                       {"automatic": {"learning_enabled": False, "mode": "auto", "scheduled_time": "02:00"}},
                                       {"automatic": {"learning_enabled": 0}}, {"automatic": {"learning_enabled": "false"}}):
                            self.assertEqual(client.patch("/api/settings", json=change).status_code, 409)
                        self.assertTrue(app.state.native_evolution_lease.active)
                        self.assertEqual(client.get("/api/health").status_code, 200)
                    finally:
                        released = client.post("/internal/native-evolution/release", json=body)
                        self.assertEqual(released.status_code, 200, released.text)
                        self.assertTrue(released.json()["was_ready"])
                        self.assertFalse(app.state.native_evolution_lease.active)


if __name__ == "__main__":
    unittest.main()
