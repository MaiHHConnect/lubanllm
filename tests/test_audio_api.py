"""Text-only API scope: audio rejects before ASR, state changes or paid calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from luban.memory import Memory
from luban.server import create_app
from luban.settings import Settings
from luban.speech import MAX_AUDIO_BYTES, SpeechRuntime
from test_app import FakeRuntime


class AudioApiTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.settings = Settings(root / "private")
        self.settings.update({"storage_root": str(root / "storage")})
        self.runtime = FakeRuntime(Memory(root / "storage"))
        self.speech = SpeechRuntime("/nonexistent")
        self.speech.transcribe_bytes = Mock(side_effect=AssertionError("ASR must not run"))
        self.speech._load = Mock(side_effect=AssertionError("ASR must not load"))
        self.app = create_app(self.settings, self.runtime, autoload=False, speech=self.speech)
        self.app.state.teacher.stream = Mock(side_effect=AssertionError("teacher must not run"))
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.bearer = {"Authorization": "Bearer " + self.settings.secret("access_key")}

    def login(self):
        self.assertEqual(self.client.post("/api/session", json={"token": self.settings.secret("ui_token")}).status_code, 200)

    def audio(self, path="/api/audio/transcriptions", data=b"not even WAV", headers=None):
        return self.client.post(path, content=data, headers={"content-type": "audio/wav", **(headers or {})})

    def assert_unused(self):
        self.speech.transcribe_bytes.assert_not_called()
        self.speech._load.assert_not_called()
        self.app.state.teacher.stream.assert_not_called()
        self.assertEqual(self.runtime.calls, [])
        self.assertIsNone(self.speech.model)
        self.assertIsNone(self.speech.processor)

    def test_cookie_and_bearer_auth_remain_distinct(self):
        for path in ("/api/audio/transcriptions", "/v1/audio/transcriptions"):
            self.assertEqual(self.audio(path).status_code, 401)
        self.assertEqual(self.audio(headers=self.bearer).status_code, 401)
        self.login()
        self.assertEqual(self.audio("/v1/audio/transcriptions").status_code, 401)
        self.assertEqual(self.audio().status_code, 400)
        self.client.cookies.clear()
        self.assertEqual(self.audio("/v1/audio/transcriptions", headers=self.bearer).status_code, 400)
        self.assert_unused()

    def test_audio_is_explicitly_closed_before_pause_queue_or_asr_for_all_types(self):
        self.login()
        with patch.object(self.app.state.automatic, "pause", side_effect=AssertionError("must not pause")), \
                patch.object(self.app.state.benchmark, "pause", side_effect=AssertionError("must not pause")):
            for content_type in ("audio/wav", "audio/mpeg", "application/json", "multipart/form-data", ""):
                response = self.audio(headers={"content-type": content_type})
                self.assertEqual(response.status_code, 400)
                self.assertIn("音频接口暂不开放", response.json()["error"]["message"])
                self.assertEqual(response.headers["cache-control"], "no-store")
            for state in ("ready", "stopped", "loading", "generating", "learning", "evolving"):
                self.runtime.state = state
                self.assertEqual(self.audio().status_code, 400)
                self.assertEqual(self.runtime.state, state)
        self.assert_unused()

    def test_existing_body_limits_still_reject_before_any_speech_call(self):
        self.login()
        for length in (str(MAX_AUDIO_BYTES + 1), "-1", "invalid"):
            self.assertEqual(self.audio(headers={"content-length": length}).status_code, 413)
        self.assertEqual(self.audio(data=b"x" * (MAX_AUDIO_BYTES + 1), headers={"content-length": "1"}).status_code, 413)
        chunks = iter([b"x" * (1024 * 1024)] * 8 + [b"x"])
        self.assertEqual(self.audio(data=chunks).status_code, 413)
        self.assert_unused()


if __name__ == "__main__":
    unittest.main()
