"""Real authenticated HTTP admission, with no model or teacher calls."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
from PIL import Image
from luban.memory import Memory
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings


class MediaHTTPTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.settings = Settings(root / "private")
        self.settings.update({"storage_root": str(root / "storage")})
        self.runtime = Runtime(self.settings, Memory(root / "storage"))
        self.app = create_app(self.settings, self.runtime, autoload=False)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.module = SimpleNamespace(ROUND_STEPS=5, capability=lambda _: {"available": True, "reason": "",
            "required_available_bytes": 7_968_468_672})
        self.kind = patch.object(self.app.state.media_learning, "_kind", return_value=self.module)
        self.kind.start()
        self.addCleanup(self.kind.stop)
        self.materials = {"kind": "image", "training_allowed": True, "answers_reviewed": True,
                          "files": [{"name": f"{i}.png", "content": base64.b64encode(bytes([i + 1])).decode(),
                                     "answer": "fixture", "question": "fixture?", "family": str(i)} for i in range(2)]}

    def login(self):
        self.assertEqual(self.client.post("/api/session", json={"token": self.settings.secret("ui_token")}).status_code, 200)

    def test_auth_origin_limits_and_metadata_only_status(self):
        self.assertEqual(self.client.get("/api/learning/media").status_code, 401)
        self.login()
        self.assertEqual(self.client.post("/api/learning/media", json=self.materials,
                                        headers={"origin": "https://elsewhere.invalid"}).status_code, 403)
        response = self.client.post("/api/learning/media", content=b"{}", headers={"content-length": str(12 * 1024 * 1024 + 1)})
        self.assertEqual(response.status_code, 413)
        lesson = self.client.post("/api/learning/media", json=self.materials)
        self.assertEqual(lesson.status_code, 200, lesson.text)
        response = self.client.get("/api/learning/media?kind=image")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("asset_path", response.text)
        self.assertNotIn("content", response.text)
        self.assertFalse(response.json()["lesson"]["adopted"])

    def test_running_model_job_rejects_media_start_but_cancel_is_available(self):
        self.login()
        lesson = self.client.post("/api/learning/media", json=self.materials).json()
        self.runtime.state, self.runtime.job = "learning", {"state": "running", "kind": "evolution"}
        response = self.client.post("/api/learning/media/start", json={"id": lesson["id"]})
        self.assertNotEqual(response.status_code, 200)
        self.assertIsNone(self.app.state.media_learning.worker)
        self.assertEqual(self.client.post("/api/learning/media/cancel", json={}).status_code, 200)
        self.assertTrue(self.app.state.media_learning.cancel.is_set())
        self.runtime.state, self.runtime.job = "stopped", None

    def test_explicit_start_uses_one_worker_and_no_teacher(self):
        self.login()
        entered, finish = threading.Event(), threading.Event()
        def train(owner, records, saved_candidate, cancel, progress):
            entered.set()
            self.assertTrue(finish.wait(3))
            return {"state": "awaiting_quality", "candidate_path": "abcdef123456", "completed_steps": 5,
                    "total_adam_step": 5, "round_budget": 5, "adopted": False}
        self.module.train_round = train
        lesson = self.client.post("/api/learning/media", json=self.materials).json()
        with patch.object(self.app.state.teacher, "complete", create=True) as teacher:
            response = self.client.post("/api/learning/media/start", json={"id": lesson["id"]})
            try:
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(entered.wait(3))
                self.assertNotEqual(self.client.post("/api/learning/media/start", json={"id": lesson["id"]}).status_code, 200)
            finally:
                finish.set()
                self.app.state.media_learning.worker.join(3)
            teacher.assert_not_called()
        current = self.client.get("/api/learning/media?kind=image").json()["lesson"]
        self.assertEqual(current["state"], "awaiting_quality")
        self.assertEqual(current["total_adam_step"], 5)
        self.assertFalse(current["adopted"])

    def test_teacher_queue_authentic_media_and_confirm_http_before_training(self):
        self.login()
        self.settings.accept_agreement()
        self.settings.update({"teacher": {"base_url": "http://offline.invalid/v1", "api_key": "fixture-key",
                                          "model": "fixture", "teaching_modalities": ["image"]}})
        sent = []
        def reply(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "鲁班"}, "finish_reason": "stop"}]})
        self.app.state.teacher.transport = httpx.MockTransport(reply)
        body = dict(self.materials, teacher_requested=True, teacher_confirmed=True, answers_reviewed=False)
        for index, row in enumerate(body["files"]):
            stream = io.BytesIO(); Image.new("RGB", (2, 2), (index, 3, 4)).save(stream, format="PNG")
            row.update(content=base64.b64encode(stream.getvalue()).decode(), answer="")
        response = self.client.post("/api/learning/media", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        identifier = response.json()["id"]
        self.assertEqual(response.json()["state"], "teacher_queued"); self.assertEqual(sent, [])
        self.assertNotEqual(self.client.post("/api/learning/media/start", json={"id": identifier}).status_code, 200)
        for _ in range(2): self.app.state.media_learning.run_teacher_one(threading.Event())
        current = self.client.get("/api/learning/media?kind=image").json()["lesson"]
        self.assertEqual(current["state"], "teacher_review"); self.assertEqual(len(sent), 2)
        confirm = {"id": identifier, "answers_reviewed": True,
                   "samples": [{key: row[key] for key in ("id", "question", "answer")} for row in current["samples"]]}
        self.assertEqual(self.client.post("/api/learning/media/confirm", json=confirm,
                                         headers={"origin": "https://elsewhere.invalid"}).status_code, 403)
        response = self.client.post("/api/learning/media/confirm", json=confirm)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "ready"); self.assertFalse(response.json()["adopted"])

    def test_video_self_upload_keeps_sequence_through_authenticated_worker(self):
        self.login()
        import copy
        body = copy.deepcopy(self.materials)
        body['kind'] = 'video'
        for row in body['files']: row['name'] = row['name'].replace('.png', '.mp4')
        def frames(source, directory):
            directory.mkdir(mode=0o700)
            result = []
            for index in range(2):
                path = directory / f'{index}.jpg'
                Image.new('RGB', (4, 4), (index * 100, 0, 0)).save(path)
                path.chmod(0o600)
                result.append({'path': str(path), 'timestamp_seconds': index + .25})
            return result
        with patch('luban.video_learning.extract_frames', frames):
            response = self.client.post('/api/learning/media', json=body)
        self.assertEqual(response.status_code, 200, response.text)
        value = response.json()
        self.assertEqual((value['kind'], value['state'], value['frame_count']), ('video', 'ready', 4))
        self.assertNotIn('asset_path', response.text)
        received = []
        def train(owner, records, saved_candidate, cancel, progress):
            received.extend(records)
            self.assertEqual([row['kind'] for row in records], ['video', 'video'])
            self.assertEqual([frame['timestamp_seconds'] for frame in records[0]['video_frames']], [.25, 1.25])
            return {'state': 'awaiting_quality', 'candidate_path': 'abcdef123456', 'completed_steps': 5,
                    'total_adam_step': 5, 'round_budget': 5, 'adopted': False}
        self.module.train_round = train
        response = self.client.post('/api/learning/media/start', json={'id': value['id']})
        self.assertEqual(response.status_code, 200, response.text)
        self.app.state.media_learning.worker.join(3)
        self.assertEqual(len(received), 2)
        current = self.client.get('/api/learning/media?kind=video').json()['lesson']
        self.assertEqual((current['kind'], current['state'], current['total_adam_step']), ('video', 'awaiting_quality', 5))
        self.assertFalse(current['adopted'])


if __name__ == "__main__":
    unittest.main()
