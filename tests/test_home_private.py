"""Private homepage reads with isolated CPU state; no model, teacher, or network."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from luban.memory import Memory
from luban.server import create_app
from luban.settings import Settings
from tests.test_app import FakeRuntime


class HomePrivateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = Settings(self.root / "private")
        self.settings.update({"storage_root": str(self.root / "data")})
        self.memory = Memory(self.root / "data")
        self.memory.model_identity = "a" * 64
        self.runtime = FakeRuntime(self.memory)
        self.runtime.generate = Mock(side_effect=AssertionError("Homepage must never generate"))
        speech = SimpleNamespace(status=lambda: {"state": "unloaded"})
        with patch("luban.bun_bridge.enabled", return_value=True), patch.dict(os.environ, {"LUBAN_COMPUTE_TOKEN": "offline-fixture-compute"}):
            self.app = create_app(self.settings, runtime=self.runtime, autoload=False, speech=speech)
        self.client = TestClient(self.app, client=("127.0.0.1", 12345), raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.headers = {"authorization": "Bearer offline-fixture-compute"}

    def test_private_auth_and_scope_does_not_fetch_growth(self):
        with patch.object(self.app.state.growth, "profile", side_effect=AssertionError("ordinary scope must stay cheap")):
            for path in ("/internal/home-profile", "/internal/home-evolution", "/internal/data-scope"):
                self.assertEqual(self.client.get(path).status_code, 401)
            value = self.client.get("/internal/data-scope", headers=self.headers)
            self.assertEqual(value.status_code, 200)
            self.assertEqual(value.json(), {"root": str(self.memory.root), "identity": "a" * 64})

    def test_home_profile_reuses_locally_verified_engine_bound_profile(self):
        profile = [{"domain": "mathematics", "growth": None}]
        with patch.object(self.app.state.growth, "profile", return_value={"growth_profile": profile}) as read:
            response = self.client.get("/internal/home-profile", headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["growth_profile"], profile)
            read.assert_called_once_with()
        self.runtime.generate.assert_not_called()

    def test_verified_rename_count_filters_branch_and_foreign_base_and_fails_closed(self):
        # Existing verifier sees no source reports in this genuinely empty fixture.
        response = self.client.get("/internal/home-evolution", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["evolution_count"], 0)
        records = [{"kind": "core", "base_sha256": "a" * 64},
                   {"kind": "core", "base_sha256": "b" * 64},
                   {"base_sha256": "a" * 64}]
        with patch("luban.upgrades.read_upgrade_records", return_value=records) as verify:
            response = self.client.get("/internal/home-evolution", headers=self.headers)
            self.assertEqual(response.json()["evolution_count"], 1)
            verify.assert_called_once_with(self.memory.root)
        with patch("luban.upgrades.read_upgrade_records", side_effect=RuntimeError("bad persisted weights")):
            self.assertEqual(self.client.get("/internal/home-evolution", headers=self.headers).status_code, 409)
        self.runtime.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
