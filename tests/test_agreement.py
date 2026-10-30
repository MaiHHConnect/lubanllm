"""Agreement acceptance and material admission, isolated from real credentials/API."""
import json
import stat
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from luban.agreement import AGREEMENT_VERSION, agreement_status
from luban.memory import Memory
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings
from luban.teacher import Teacher


class AgreementSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(self.temp.name)

    def test_fresh_read_never_accepts_and_patch_cannot_accept(self):
        before = self.settings.path.read_bytes()
        self.assertFalse(self.settings.agreement()["accepted"])
        self.assertEqual(self.settings.path.read_bytes(), before)
        self.settings.update({"agreement": {"version": AGREEMENT_VERSION,
                              "accepted_at": "2026-10-09T00:00:00+00:00"}})
        self.assertFalse(self.settings.agreement()["accepted"])

    def test_acceptance_survives_restart_without_changing_timestamp(self):
        self.assertTrue(self.settings.accept_agreement()["accepted"])
        record = self.settings.snapshot()["agreement"]
        self.assertEqual(record["version"], AGREEMENT_VERSION)
        self.assertIsNotNone(datetime.fromisoformat(record["accepted_at"]).tzinfo)
        self.assertEqual(stat.S_IMODE(self.settings.path.stat().st_mode), 0o600)
        reopened = Settings(self.temp.name)
        self.assertTrue(reopened.agreement()["accepted"])
        reopened.accept_agreement()
        self.assertEqual(reopened.snapshot()["agreement"], record)
        with patch("luban.agreement.AGREEMENT_VERSION", "next-version"):
            self.assertFalse(reopened.agreement()["accepted"])

    def test_invalid_or_old_record_is_not_acceptance(self):
        for record in (None, {}, {"version": "old", "accepted_at": "2026-10-09T00:00:00+00:00"},
                       {"version": AGREEMENT_VERSION, "accepted_at": "invalid"},
                       {"version": AGREEMENT_VERSION, "accepted_at": "2026-10-09T00:00:00"}):
            with self.subTest(record=record):
                self.assertFalse(agreement_status({"agreement": record})["accepted"])

    def test_accept_write_failure_does_not_accept_in_memory(self):
        before = self.settings.path.read_bytes()
        with patch("luban.settings.private_json", side_effect=OSError("offline disk")):
            with self.assertRaises(OSError):
                self.settings.accept_agreement()
        self.assertFalse(self.settings.agreement()["accepted"])
        self.assertEqual(self.settings.path.read_bytes(), before)

    def test_legacy_provider_fields_removed_without_touching_key(self):
        self.settings.update({"teacher": {"api_key": "offline-private-key",
                              "base_url": "http://teacher.invalid/v1", "model": "offline"}})
        saved = self.settings.snapshot()
        saved["teacher"].update(training_allowed=True, terms_reference="legacy terms")
        self.settings.path.write_text(json.dumps(saved))
        reopened = Settings(self.temp.name)
        reopened.update({"teacher": {"training_allowed": True, "terms_reference": "ignored"}})
        public = reopened.public()
        self.assertNotIn("training_allowed", public["teacher"])
        self.assertNotIn("terms_reference", public["teacher"])
        self.assertNotIn("offline-private-key", json.dumps(public))
        self.assertEqual(reopened.secret("teacher_key"), "offline-private-key")
        self.assertFalse(reopened.agreement()["accepted"])
        persisted = json.loads(reopened.path.read_text())
        self.assertNotIn("training_allowed", persisted["teacher"])
        self.assertNotIn("terms_reference", persisted["teacher"])

    def test_validation_is_reversible_and_invalid_update_changes_nothing(self):
        before = self.settings.snapshot()
        disk = self.settings.path.read_bytes()
        data, credentials = self.settings.validated({"teacher": {"model": "draft-model"}})
        self.assertEqual(data["teacher"]["model"], "draft-model")
        self.assertEqual(self.settings.snapshot(), before)
        self.assertEqual(self.settings.path.read_bytes(), disk)
        with self.assertRaises(ValueError):
            self.settings.update({"teacher": {"api_key": "bad\nkey", "model": "invalid"}})
        self.assertEqual(self.settings.snapshot(), before)
        self.assertEqual(self.settings.path.read_bytes(), disk)


class AgreementAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(root / "private")
        self.settings.update({"storage_root": str(root / "materials"), "teacher": {
            "base_url": "http://teacher.invalid/v1", "api_key": "offline-key", "model": "offline"}})
        self.runtime = Runtime.__new__(Runtime)
        self.runtime.memory = Memory(root / "materials")
        self.runtime.lock = threading.RLock()
        self.runtime.state = "ready"
        self.runtime.job = None
        self.runtime.choose_route = lambda messages, requested: {"route": "teacher", "source": "user"}
        self.runtime.teacher_usage = lambda *args: {"prompt_tokens": 2, "completion_tokens": 2,
                                                   "total_tokens": 4, "estimated": True}
        self.runtime._sample_tokens = lambda sample: ([], [])
        self.runtime.tokenizer = object()
        self.teacher = Teacher(self.settings, httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "text/event-stream"},
            text='data: {"choices":[{"delta":{"content":"离线教师回答"}}]}\n\ndata: [DONE]\n\n')))
        self.client = TestClient(create_app(self.settings, self.runtime, self.teacher, autoload=False),
                                 raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def login(self):
        self.assertEqual(self.client.post("/api/session", json={
            "token": self.settings.secret("ui_token")}).status_code, 200)

    def test_endpoints_require_ui_session_and_get_never_changes_acceptance(self):
        self.assertEqual(self.client.get("/api/agreement").status_code, 401)
        self.assertEqual(self.client.post("/api/agreement/accept", json={}).status_code, 401)
        self.login()
        response = self.client.get("/api/agreement")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"version", "accepted", "text"})
        self.assertFalse(response.json()["accepted"])
        self.assertIn("第三方", response.json()["text"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        accepted = self.client.post("/api/agreement/accept", json={})
        self.assertEqual(accepted.status_code, 200)
        self.assertTrue(accepted.json()["accepted"])
        self.assertTrue(self.client.get("/api/agreement").json()["accepted"])
        self.assertTrue(Settings(self.settings.home).agreement()["accepted"])

    def test_teacher_manual_sample_requires_agreement_but_no_provider_permission_fields(self):
        self.login()
        sample = {"prompt": "独立问题", "answer": "核验答案", "source": "teacher", "approved": True}
        denied = self.client.post("/api/samples", json=sample)
        self.assertEqual(denied.status_code, 400)
        self.assertIn("使用协议", denied.text)
        self.assertEqual(self.runtime.memory.samples(), [])
        self.client.post("/api/agreement/accept", json={})
        self.assertEqual(self.client.post("/api/samples", json=sample).status_code, 200)
        provenance = json.loads(self.runtime.memory.samples()[0]["provenance"])
        self.assertEqual(provenance["agreement_version"], AGREEMENT_VERSION)
        self.assertTrue(provenance["user_responsibility"])
        self.assertEqual(provenance["rights_basis"], "user responsibility under agreement")
        self.assertNotIn("terms", provenance)
        self.assertNotIn("offline-key", json.dumps(provenance))

    def test_acceptance_does_not_replace_human_or_synthetic_sample_permission(self):
        self.login()
        self.client.post("/api/agreement/accept", json={})
        for source in ("human", "synthetic"):
            with self.subTest(source=source):
                sample = {"prompt": source, "answer": "答案", "source": source, "approved": True}
                self.assertEqual(self.client.post("/api/samples", json=sample).status_code, 400)
                sample["training_allowed"] = True
                self.assertEqual(self.client.post("/api/samples", json=sample).status_code, 200)

    def test_existing_teacher_requests_archive_before_acceptance_and_admit_afterward(self):
        body = {"messages": [{"role": "user", "content": "离线问题"}], "route": "teacher"}
        bearer = {"Authorization": "Bearer " + self.settings.secret("access_key")}
        self.assertEqual(self.client.post("/v1/chat/completions", headers=bearer, json=body).status_code, 200)
        self.assertEqual(self.runtime.memory.material_counts(), {"total": 1, "pending": 1, "eligible": 0})
        self.assertEqual(self.runtime.memory.samples(), [])
        provenance = json.loads(self.runtime.memory.teacher_materials()[0]["provenance"])
        self.assertFalse(provenance["user_responsibility"])
        self.login()
        self.client.post("/api/agreement/accept", json={})
        body["messages"][0]["content"] = "新离线问题"
        self.assertEqual(self.client.post("/v1/chat/completions", headers=bearer, json=body).status_code, 200)
        self.assertEqual(self.runtime.memory.material_counts(), {"total": 2, "pending": 1, "eligible": 1})
        provenance = json.loads(self.runtime.memory.samples()[0]["provenance"])
        self.assertTrue(provenance["user_responsibility"])
        self.assertEqual(provenance["agreement_version"], AGREEMENT_VERSION)
        self.assertEqual(provenance["rights_basis"], "user responsibility under agreement")


if __name__ == "__main__":
    unittest.main()
