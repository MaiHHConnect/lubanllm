"""One product evolution method; migrate old budgets without loading a model."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import test_app
from luban.settings import Settings


class EvolutionSingleModeSettingsTests(unittest.TestCase):
    def test_default_and_legacy_budget_migrate_preserving_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(Settings(directory).snapshot()["automatic"], {"mode": "manual", "scheduled_time": "02:00", "learning_enabled": True})
            path = Path(directory) / "settings.json"
            saved = json.loads(path.read_text())
            for budget in ("recommended", "full", "unknown", None):
                with self.subTest(budget=budget):
                    saved["automatic"] = {"mode": "auto", "scheduled_time": "23:14", "evolution_budget": budget}
                    path.write_text(json.dumps(saved))
                    settings = Settings(directory)
                    expected = {"mode": "auto", "scheduled_time": "23:14", "learning_enabled": True}
                    self.assertEqual(settings.public()["automatic"], expected)
                    self.assertEqual(json.loads(path.read_text())["automatic"], expected)

    def test_manual_auto_time_remain_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(directory)
            settings.update({"automatic": {"mode": "auto", "scheduled_time": "03:21"}})
            settings = Settings(directory)
            self.assertEqual(settings.snapshot()["automatic"], {"mode": "auto", "scheduled_time": "03:21", "learning_enabled": True})
            settings.update({"automatic": {"mode": "manual", "scheduled_time": "04:22"}})
            self.assertEqual(Settings(directory).snapshot()["automatic"], {"mode": "manual", "scheduled_time": "04:22", "learning_enabled": True})

    def test_learning_switch_persists_and_old_schedule_updates_do_not_reenable_it(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(directory)
            settings.update({"automatic": {"learning_enabled": False}})
            settings = Settings(directory)
            self.assertFalse(settings.snapshot()["automatic"]["learning_enabled"])
            settings.update({"automatic": {"mode": "auto", "scheduled_time": "04:22"}})
            self.assertFalse(Settings(directory).snapshot()["automatic"]["learning_enabled"])
            for value in (None, 0, 1, "false", [], {}):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    settings.update({"automatic": {"learning_enabled": value}})
            settings.update({"automatic": {"learning_enabled": True}})
            self.assertTrue(Settings(directory).snapshot()["automatic"]["learning_enabled"])

    def test_budget_request_rejects_entire_settings_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(directory)
            before = settings.path.read_bytes()
            for budget in ("recommended", "full", 352, True, None, [], {}):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    settings.update({"temperature": .2, "automatic": {
                        "mode": "manual", "scheduled_time": "02:00", "evolution_budget": budget}})
                self.assertEqual(settings.path.read_bytes(), before)


class EvolutionSingleModeApiTests(unittest.TestCase):
    setUp = test_app.AppTests.setUp
    teacher_response = test_app.AppTests.teacher_response
    login = test_app.AppTests.login

    def test_no_user_budget_reaches_shared_manifest(self):
        self.login()
        automatic = self.client.app.state.automatic
        launch = Mock(return_value={"job_id": "cpu-fixture"})
        with patch.object(automatic, "_available", return_value=True), patch.object(automatic, "start_manual", launch):
            response = self.client.post("/api/automatic/settings", json={"mode": "auto", "scheduled_time": "23:14"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.client.post("/api/evolution/start", json={}).status_code, 200)
            launch.assert_called_once_with("evolution")
            launch.reset_mock()
            for body in ({"steps": 352}, {"steps": 704}, {"steps": 832}, {"evolution_budget": "full"}, [], True):
                self.assertEqual(self.client.post("/api/evolution/start", json=body).status_code, 400)
            launch.assert_not_called()

    def test_no_unbound_legacy_thread_when_runtime_unavailable(self):
        self.login()
        automatic = self.client.app.state.automatic
        with patch.object(automatic, "_available", return_value=False), patch.object(automatic, "start_manual") as launch:
            self.assertEqual(self.client.post("/api/evolution/start", json={}).status_code, 400)
            launch.assert_not_called()
            self.assertIsNone(self.runtime.job)
