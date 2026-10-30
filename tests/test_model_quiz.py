import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

import test_growth as fixtures
from luban.growth_service import GrowthService
from luban.model_quiz import _questions, _work
from luban.server import create_app
from luban.settings import Settings
from luban.teacher import Teacher


class ModelQuizTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.GrowthTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runtime = self.fixture.runtime
        self.board = SimpleNamespace(_request=Mock(side_effect=AssertionError("network used")))
        self.service = GrowthService(self.runtime, self.board, threading.Lock(), lambda: False)
        self.addCleanup(self.service.close)

    def test_no_materials_no_work_and_busy_protection(self):
        self.assertEqual(self.service.start_quiz()["state"], "waiting_materials")
        self.assertEqual(self.runtime.state, "ready")
        self.runtime.state = "generating"
        with self.assertRaises(RuntimeError):
            self.service.start_quiz()
        self.board._request.assert_not_called()

    def test_prior_answer_context_is_never_given_as_a_reference(self):
        self.fixture.source("Compute 3 + 4.", "7", provenance={"messages": [
            {"role": "assistant", "content": "The answer is 7."},
            {"role": "user", "content": "Compute 3 + 4."}]})
        self.assertEqual(_questions(self.service), [])

    def test_real_api_requires_session_and_respects_update_lease(self):
        settings = Settings(self.fixture.store / "quiz-home")
        settings.update({"storage_root": str(self.runtime.memory.root)})
        teacher = Teacher(settings, httpx.MockTransport(lambda _: (_ for _ in ()).throw(AssertionError("teacher called"))))
        self.runtime.job = None
        app = create_app(settings, self.runtime, teacher, autoload=False)
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/model-quiz").status_code, 401)
            self.assertEqual(client.post("/api/model-quiz/start", json={}).status_code, 401)
            self.assertEqual(client.post("/api/session", json={"token": settings.secret("ui_token")}).status_code, 200)
            self.assertEqual(client.get("/api/model-quiz").json()["state"], "idle")
            self.assertEqual(client.post("/api/model-quiz/start", json={}).json()["state"], "waiting_materials")
            self.assertEqual(client.post("/api/update/prepare", json={}).status_code, 200)
            response = client.post("/api/model-quiz/start", json={})
            self.assertEqual(response.status_code, 409)
            self.assertIn("准备更新", response.text)
            self.assertEqual(client.post("/api/model-quiz/cancel", json={}).status_code, 200)

    def test_fresh_tasks_are_reserved_and_only_actual_closed_book_failure_is_marked(self):
        identifier = self.fixture.source("Compute 3 + 4.", "7")
        data = self.service._read()
        data["history"] = [{"original_before": {"cases": [{"source_id": identifier, "passed": False}]}}]
        self.service._save(data)
        questions = _questions(self.service)
        self.assertEqual(len(questions), 6)
        self.assertFalse(questions[0]["previously_wrong"])
        data = self.service._read()
        data["history"][0]["original_before"]["metadata"] = {"teacher_calls": 0, "reference": False}
        self.service._save(data)
        questions = _questions(self.service)
        self.assertTrue(questions[0]["previously_wrong"])
        self.assertEqual(questions, _questions(self.service))
        for question in questions[1:]:
            with self.assertRaises(ValueError):
                self.runtime.memory.add_sample(question["prompt"], "7", approved=True, training_allowed=True)
        self.board._request.assert_not_called()

    def test_real_tiny_runner_forces_teacher_route_to_local_and_never_uses_references(self):
        self.fixture.source("Compute 3 + 4.", "7")
        self.runtime.choose_route = lambda _: {"route": "teacher", "action": "teacher"}
        self.service.start_quiz()
        self.service.worker.join(10)
        state = self.service.quiz_status()
        self.assertEqual(state["state"], "complete", state)
        self.assertEqual(state["result"]["question_count"], 6)
        self.assertEqual(state["result"]["correct"], 1)
        self.assertEqual(state["result"]["teacher_calls"], 0)
        self.assertIsNone(state["result"]["newly_correct"])
        self.assertEqual(self.runtime.state, "ready")
        self.assertEqual(len(self.runtime.model.calls), 6)
        self.assertTrue(all(not item["expert_present"] for item in self.runtime.model.calls))
        self.assertNotIn('expected', json.dumps([call[0] for call in self.fixture.token_calls]))
        self.board._request.assert_not_called()

    def test_growth_counts_require_same_suite_and_different_model_version(self):
        self.fixture.source("Compute 3 + 4.", "7")
        questions = _questions(self.service)
        metadata = {"elapsed_seconds": .1, "items": [], "teacher_calls": 0, "reference": False}
        version = "a" * 64
        correct = False
        def run(*args, **kwargs):
            self.assertTrue(kwargs["force_local"])
            return [{"text": "7" if correct and q["kind"] == "original" else "wrong", "finish_reason": "stop"}
                    for q in questions], metadata
        with patch('luban.model_quiz._run', side_effect=run), patch('luban.model_quiz.personal_model_version', side_effect=lambda _:version):
            _work(self.service, questions, None)
            from luban.model_quiz import _read
            previous = _read(self.service)
            correct = True
            _work(self.service, questions, previous)
            self.assertIsNone(self.service.quiz_status()["result"]["newly_correct"])
            version = "b" * 64
            _work(self.service, questions, previous)
            self.assertEqual(self.service.quiz_status()["result"]["newly_correct"], 1)
            self.assertEqual(self.service.quiz_status()["result"]["regressed"], 0)
            changed = [dict(q) for q in questions]
            changed[0]["prompt"] = "Compute 4 + 3."
            _work(self.service, changed, previous)
            self.assertIsNone(self.service.quiz_status()["result"]["newly_correct"])

    def test_cancel_and_failed_run_preserve_last_completed_score(self):
        self.fixture.source("Compute 3 + 4.", "7")
        self.service.start_quiz(); self.service.worker.join(10)
        saved = (self.service.path.parent / "quiz.json").read_bytes()
        with patch('luban.model_quiz._run', side_effect=InterruptedError):
            self.service.start_quiz(); self.service.worker.join(10)
        self.assertEqual(self.service.quiz_status()["state"], "cancelled")
        self.assertEqual((self.service.path.parent / "quiz.json").read_bytes(), saved)
        self.assertEqual(self.runtime.state, "ready")
        with patch('luban.model_quiz._run', side_effect=ValueError("fixture failure")):
            self.service.start_quiz(); self.service.worker.join(10)
        self.assertEqual(self.service.quiz_status()["state"], "error")
        self.assertEqual((self.service.path.parent / "quiz.json").read_bytes(), saved)
