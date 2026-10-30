"""Offline teaching mode, durable practice and paid-boundary tests; no model loading."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from luban.automatic import AutomaticLearning
from luban.memory import Memory
from luban.runtime import Runtime
from luban.retention import eligible_learning_sample
from luban.server import create_app
from luban.settings import Settings
from luban.teacher import Teacher
from luban.teacher_practice import TeacherPractice
from test_app import FakeRuntime, events


class PracticeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(self.root / "private")
        self.settings.update({"storage_root": str(self.root / "data"), "teacher": {
            "base_url": "http://offline.invalid/v1", "api_key": "fixture-secret", "model": "fixture",
            "answer_mode": "teacher_first", "session_call_limit": 20}})
        self.settings.accept_agreement()
        self.runtime = FakeRuntime(Memory(self.root / "data"))
        self.runtime.identity = "practice-base"
        self.runtime.store = self.root / "modules"
        self.runtime.active_core = None
        self.runtime.branch = SimpleNamespace(expert_id=None)
        self.runtime.tokenizer = None
        self.paid = []
        def respond(request):
            payload = json.loads(request.content)
            self.paid.append(payload)
            text = "教师即时答案"
            if payload["messages"][0]["content"].startswith("快速批改"):
                text = "[true]"
            return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                          "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})
        self.teacher = Teacher(self.settings, httpx.MockTransport(respond))
        self.practice = TeacherPractice(self.runtime, self.settings, self.teacher)
        self.messages = [{"role": "user", "content": "同一道问题"}]

    def queue(self, identifier="request-one"):
        self.practice.enqueue(identifier, self.messages, "private-material")

    def rows(self):
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM teacher_practice ORDER BY request_id")]

    def complete_generate(self, messages, emit, stopped, **options):
        self.assertEqual(messages, self.messages)
        self.assertFalse(options["allow_reference"])
        self.assertFalse(options["observe_external"])
        emit({"content": "本地独立答案"})
        return {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4,
                "finish_reason": "stop", "local_complete": True, "reference_used": False,
                "local_action": "local_core", "core_version": "core-before"}

    def test_mode_defaults_persistence_and_strict_validation(self):
        fresh = Settings(self.root / "fresh")
        self.assertEqual(fresh.public()["teacher"]["answer_mode"], "local_first")
        self.assertEqual(Settings(self.settings.home).public()["teacher"]["answer_mode"], "teacher_first")
        for value in ["", None, True, 2, [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.settings.update({"teacher": {"answer_mode": value}})
        self.assertEqual(self.settings.snapshot()["teacher"]["answer_mode"], "teacher_first")
        self.settings.update({"teacher": {"answer_mode": "auto"}})
        self.assertEqual(self.settings.snapshot()["teacher"]["answer_mode"], "auto")

    def test_teacher_first_overrides_every_request_route_without_local_generation_or_compaction(self):
        with TestClient(create_app(self.settings, self.runtime, self.teacher, autoload=False)) as client:
            client.post("/api/session", json={"token": self.settings.secret("ui_token")}).raise_for_status()
            with patch.object(self.runtime, "choose_route", side_effect=AssertionError("must not route locally")), \
                    patch("luban.server.compact_messages", side_effect=AssertionError("must not compact locally")):
                for route in ["auto", "local", "teacher"]:
                    response = client.post("/api/chat", json={"messages": self.messages, "route": route, "auto_compact": True})
                    self.assertEqual(response.status_code, 200)
                    stream = events(response)
                    self.assertFalse(any(item["type"] == "error" for item in stream), stream)
                    self.assertEqual(next(item for item in stream if item["type"] == "meta")["source"], "teacher_first")
                    self.assertEqual("".join(item.get("content", "") for item in stream), "教师即时答案")
                response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + self.settings.secret("access_key")},
                                       json={"messages": self.messages, "route": "local"})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["choices"][0]["message"]["content"], "教师即时答案")
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(len(self.paid), 4)
        self.assertEqual(len(self.rows()), 4)

    def test_missing_teacher_errors_without_silent_local_fallback(self):
        self.settings.update({"teacher": {"base_url": ""}})
        with TestClient(create_app(self.settings, self.runtime, self.teacher, autoload=False)) as client:
            client.post("/api/session", json={"token": self.settings.secret("ui_token")})
            response = client.post("/api/chat", json={"messages": self.messages, "route": "auto"})
            self.assertTrue(any(item["type"] == "error" for item in events(response)))
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(self.paid, [])

    def test_queue_is_idempotent_and_independent_answer_is_reviewed_without_training(self):
        self.queue(); self.queue()
        self.runtime.generate = self.complete_generate
        self.practice.run_one()
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]["state"], "complete")
        self.assertEqual(len(self.paid), 1)
        self.assertIn("本地独立答案", self.paid[0]["messages"][1]["content"])
        self.assertNotIn("教师即时答案", self.paid[0]["messages"][1]["content"])
        self.assertEqual(self.runtime.memory.samples(), [])
        TeacherPractice(self.runtime, self.settings, self.teacher).run_one()
        self.assertEqual(len(self.paid), 1)

    def test_restart_retries_local_work_but_never_ambiguous_paid_work(self):
        self.queue("one"); self.queue("two")
        self.practice._state("one", "generating")
        self.practice._state("two", "review_started", "decision-two")
        reopened = TeacherPractice(self.runtime, self.settings, self.teacher)
        self.assertTrue(reopened.pending())
        self.assertEqual([row["state"] for row in self.rows()], ["queued", "interrupted"])
        self.runtime.generate = self.complete_generate
        reopened.run_one()
        self.assertEqual([row["state"] for row in self.rows()], ["complete", "interrupted"])
        self.assertEqual(len(self.paid), 1)

    def test_budget_and_permission_wait_without_regenerating_answer(self):
        self.queue()
        generate = Mock(side_effect=self.complete_generate)
        self.runtime.generate = generate
        self.teacher.calls = self.settings.snapshot()["teacher"]["session_call_limit"]
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "answer_ready")
        self.assertFalse(self.practice.pending())  # Waiting for funds cannot starve other idle learning.
        self.practice.run_one()
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(self.paid, [])
        self.teacher.calls = 0
        with patch.object(self.settings, "agreement", return_value={"accepted": False}):
            self.practice.run_one()
        self.assertEqual(self.paid, [])
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "complete")
        self.assertEqual(generate.call_count, 1)

    def test_failed_paid_review_is_not_automatically_retried(self):
        self.queue()
        self.runtime.generate = self.complete_generate
        with patch("luban.teacher_practice.review_candidates", side_effect=RuntimeError("private answer fixture-secret")) as review:
            self.practice.run_one(); self.practice.run_one()
            review.assert_called_once()
        self.assertEqual(self.rows()[0]["state"], "review_failed")
        self.assertNotIn("fixture-secret", json.dumps(self.rows()))

    def test_cancellation_and_incomplete_generation_never_call_teacher(self):
        self.queue()
        def cancel(messages, emit, stopped, **options):
            stopped.set()
            return self.complete_generate(messages, emit, stopped, **options)
        self.runtime.generate = cancel
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "queued")
        self.assertEqual(self.paid, [])
        self.practice.stop_event.clear()
        self.runtime.generate = lambda *args, **kwargs: {"finish_reason": "length", "local_complete": False,
            "reference_used": False, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "incomplete")
        self.assertEqual(self.paid, [])

    def test_wrong_answer_gets_original_detailed_review_unknown_stays_unverified(self):
        for verdict in (False, None):
            with self.subTest(verdict=verdict):
                identifier = "wrong" if verdict is False else "unknown"
                self.queue(identifier)
                self.runtime.generate = self.complete_generate
                def respond(request):
                    payload = json.loads(request.content)
                    self.paid.append(payload)
                    quick = payload["messages"][0]["content"].startswith("快速批改")
                    content = json.dumps([verdict] if quick else {"correct": False, "route_action": "teacher", "reason": "该答案不满足已给出的计算标准"})
                    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})
                self.teacher.transport = httpx.MockTransport(respond)
                before = len(self.paid)
                self.practice.run_one()
                row = next(row for row in self.rows() if row["request_id"] == identifier)
                self.assertEqual(row["state"], "complete" if verdict is False else "unknown")
                self.assertEqual(len(self.paid) - before, 2 if verdict is False else 1)
                self.assertEqual(self.runtime.memory.samples(), [])
                self.practice.run_one()
                self.assertEqual(len(self.paid) - before, 2 if verdict is False else 1)

    def test_foreground_pause_interrupts_active_local_practice_before_teacher_call(self):
        import time
        self.queue()
        entered = threading.Event()
        def generate(messages, emit, stopped, **options):
            entered.set()
            self.assertTrue(stopped.wait(2))
            return self.complete_generate(messages, emit, stopped, **options)
        self.runtime.generate = generate
        automatic = AutomaticLearning(self.runtime, self.settings, threading.Lock(), lambda: False)
        automatic.teacher_practice = self.practice
        automatic.last_activity = time.monotonic() - 100
        with patch("luban.automatic.training_runs", return_value=[]), patch("luban.automatic.knowledge_keys", return_value=(set(),set(),None)):
            automatic.tick()
            self.assertTrue(entered.wait(2))
            automatic.pause(timeout=2)
        self.assertFalse(automatic.worker.is_alive())
        self.assertEqual(self.rows()[0]["state"], "queued")
        self.assertEqual(self.paid, [])

    def test_mode_and_model_identity_isolate_pending_practice(self):
        self.queue()
        self.settings.update({"teacher": {"answer_mode": "local_first"}})
        self.assertTrue(self.practice.pending())  # Mode changes only affect new foreground answers.
        self.settings.update({"teacher": {"answer_mode": "teacher_first"}})
        self.runtime.identity = "another-base"
        self.assertFalse(self.practice.pending())
        self.assertEqual(self.rows()[0]["state"], "queued")

    def test_local_first_answers_locally_then_reviews_exact_record_without_second_generation(self):
        self.settings.update({"teacher": {"answer_mode": "local_first"}})
        with TestClient(create_app(self.settings, self.runtime, self.teacher, autoload=False)) as client:
            client.post("/api/session", json={"token": self.settings.secret("ui_token")})
            with patch.object(self.runtime, "choose_route", side_effect=AssertionError("auto must be local first")):
                response = client.post("/api/chat", json={"messages": self.messages, "route": "auto"})
            stream = events(response)
            self.assertEqual("".join(item.get("content", "") for item in stream), "本地回答")
            self.assertEqual(len(self.runtime.calls), 1)
            self.assertEqual(self.paid, [])
            self.assertEqual(self.rows()[0]["state"], "answer_ready")
            practice = client.app.state.automatic.teacher_practice
            practice.stop_event.clear()
            practice.run_one()
            self.assertEqual(self.rows()[0]["state"], "complete")
            self.assertEqual(len(self.runtime.calls), 1)
            self.assertEqual(len(self.paid), 1)
            self.assertIn("本地回答", self.paid[0]["messages"][1]["content"])
            # Explicit teacher is still a deliberate per-request override.
            response = client.post("/api/chat", json={"messages": self.messages, "route": "teacher"})
            self.assertEqual("".join(item.get("content", "") for item in events(response)), "教师即时答案")
            self.assertEqual(len(self.runtime.calls), 1)
            self.assertEqual(len(self.paid), 2)

    def test_mode_change_preserves_and_completes_preexisting_teacher_first_practice(self):
        self.queue()
        self.settings.update({"teacher": {"answer_mode": "local_first"}})
        self.runtime.generate = self.complete_generate
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "complete")
        self.assertEqual(len(self.paid), 1)

    def test_storage_and_memory_instance_change_initialize_independent_queue(self):
        self.queue("old-storage")
        old_memory = self.runtime.memory
        self.runtime.memory = Memory(self.root / "new-data")
        self.assertFalse(self.practice.pending())
        self.queue("new-storage")
        self.assertEqual([row["request_id"] for row in self.rows()], ["new-storage"])
        self.runtime.generate = self.complete_generate
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "complete")
        self.runtime.memory = old_memory
        self.assertTrue(self.practice.pending())
        self.assertEqual([row["request_id"] for row in self.rows()], ["old-storage"])
        # The same root with a replacement Memory object also remains supported.
        self.runtime.memory = Memory(old_memory.root)
        self.assertTrue(self.practice.pending())

    def test_question_without_loaded_model_binds_first_practice_but_never_rebinds_old_answer(self):
        del self.runtime.identity
        self.queue("without-loaded-base")
        self.assertEqual(self.rows()[0]["state"], "queued")
        self.assertIsNone(self.rows()[0]["identity"])
        self.runtime.identity = "first-practice-model"
        self.runtime.generate = self.complete_generate
        self.teacher.calls = self.settings.snapshot()["teacher"]["session_call_limit"]
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "answer_ready")
        self.assertEqual(self.rows()[0]["identity"], "first-practice-model")
        self.runtime.identity = "unrelated-new-model"
        self.teacher.calls = 0
        self.assertFalse(self.practice.pending())
        self.practice.run_one()
        self.assertEqual(self.paid, [])
        self.assertEqual(self.rows()[0]["identity"], "first-practice-model")
        # Legacy/local actual answers without original base evidence must never
        # be claimed by a later model, unlike an as-yet-unanswered question.
        del self.runtime.identity
        self.practice.enqueue("unbound-local-answer", self.messages, decision_id="old-decision")
        self.runtime.identity = "unrelated-new-model"
        self.assertFalse(self.practice.pending())
        row = next(row for row in self.rows() if row["request_id"] == "unbound-local-answer")
        self.assertEqual(row["state"], "unbound")
        self.assertIsNone(row["identity"])

    def test_existing_reliable_local_proof_is_complete_without_paid_review(self):
        self.runtime.memory.add_sample("同一道问题", "本地独立答案", source="human", approved=True,
                                       training_allowed=True, provenance={"messages": self.messages})
        self.queue()
        self.runtime.generate = self.complete_generate
        self.practice.run_one()
        self.assertEqual(self.rows()[0]["state"], "verified_local")
        self.assertEqual(self.paid, [])
        self.assertFalse(self.practice.pending())

    def use_real_teacher_helpers_without_model(self):
        for name in ("teacher_usage", "teacher_eligibility", "_sample_tokens"):
            setattr(self.runtime, name, getattr(Runtime, name).__get__(self.runtime))
        self.runtime.device = "cpu"
        self.runtime.tokenizer = None

    def load_tiny_tokenizer(self):
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, messages, tokenize=False, **kwargs):
                text = "".join(message["content"] for message in messages)
                return list(range(len(text))) if tokenize else text
            def encode(self, text, **kwargs):
                return list(range(len(text)))
        self.runtime.tokenizer = Tokenizer()

    def test_real_runtime_unknown_usage_keeps_teacher_answer_and_pending_material_then_rechecks_after_start(self):
        self.use_real_teacher_helpers_without_model()
        del self.runtime.identity
        def respond(request):
            self.paid.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "教师即时答案"}, "finish_reason": "stop"}]})
        self.teacher.transport = httpx.MockTransport(respond)
        with TestClient(create_app(self.settings, self.runtime, self.teacher, autoload=False)) as client:
            client.post("/api/session", json={"token": self.settings.secret("ui_token")})
            response = client.post("/api/chat", json={"messages": self.messages})
            stream = events(response)
            self.assertFalse(any(item["type"] == "error" for item in stream), stream)
            self.assertEqual("".join(item.get("content", "") for item in stream), "教师即时答案")
            self.assertTrue(stream[-1]["usage"]["unknown"])
            self.assertIsNone(stream[-1]["usage"]["total_tokens"])
            self.assertEqual(self.rows()[0]["state"], "queued")
            original = self.runtime.memory.teacher_materials()[0]
            self.assertEqual(original["status"], "pending")
            self.assertIsNone(original["sample_id"])
            summary = client.get("/api/status").json()["usage"]["teacher"]
            self.assertEqual(summary["calls"], 1)
            self.assertEqual(summary["unknown_calls"], 1)
            self.assertIsNone(summary["total_tokens"])
        self.runtime.identity = "first-real-practice-model"
        self.runtime.memory.model_identity = self.runtime.identity
        self.load_tiny_tokenizer()
        self.runtime.generate = self.complete_generate
        # Return only valid review data after the original answer is already stored.
        self.teacher.transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
            "choices": [{"message": {"content": "[true]"}, "finish_reason": "stop"}]}))
        self.practice.run_one()
        material = self.runtime.memory.teacher_materials()[0]
        self.assertEqual(material["id"], original["id"])
        self.assertEqual(material["status"], "eligible")
        self.assertIsNotNone(material["sample_id"])
        samples = self.runtime.memory.samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["answer"], "教师即时答案")
        self.assertFalse(eligible_learning_sample(samples[0]))  # Structural review never proves truth.
        self.assertEqual(self.rows()[0]["state"], "complete")

    def test_unknown_usage_never_fabricates_totals_or_accepts_partially_known_payloads(self):
        memory = self.runtime.memory
        memory.record_usage("teacher", {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6})
        unknown = {"unknown": True, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        memory.record_usage("teacher", unknown)
        self.assertEqual(memory.usage()["teacher"], {"calls": 2, "prompt_tokens": None, "completion_tokens": None,
                         "total_tokens": None, "estimated_calls": 0, "unknown_calls": 1})
        for payload in [{"unknown": True}, {**unknown, "total_tokens": 0}, {**unknown, "prompt_tokens": 1}]:
            with self.assertRaises(ValueError):
                memory.record_usage("teacher", payload)
        self.assertEqual(memory.usage()["teacher"]["calls"], 2)

    def test_pause_after_quick_wrong_verdict_prevents_second_paid_detail_request(self):
        self.queue()
        self.runtime.generate = self.complete_generate
        def respond(request):
            self.paid.append(json.loads(request.content))
            self.practice.stop_event.set()
            return httpx.Response(200, json={"choices": [{"message": {"content": "[false]"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})
        self.teacher.transport = httpx.MockTransport(respond)
        self.practice.run_one()
        self.assertEqual(len(self.paid), 1)
        self.assertEqual(self.teacher.calls, 1)
        self.assertEqual(self.rows()[0]["state"], "review_failed")
        self.practice.stop_event.clear()
        self.practice.run_one()
        self.assertEqual(len(self.paid), 1)

    def test_practice_thinking_tags_are_separated_even_across_delta_boundaries(self):
        self.queue()
        def generate(messages, emit, stopped, **options):
            self.assertEqual(options["thinking"], "high")
            for part in ["<th", "ink>这段思考不能作答", "</thi", "nk>本地", "独立答案"]:
                emit({"content": part})
            emit({"reasoning_content": "独立的思考字段也不作答"})
            return {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4,
                    "finish_reason": "stop", "local_complete": True, "reference_used": False,
                    "local_action": "local_core"}
        self.settings.update({"local_thinking": "high"})
        self.runtime.generate = generate
        self.practice.run_one()
        self.assertIn("本地独立答案", self.paid[0]["messages"][1]["content"])
        self.assertNotIn("这段思考", self.paid[0]["messages"][1]["content"])
        self.assertNotIn("<think>", self.paid[0]["messages"][1]["content"])

    def test_pending_material_recheck_preserves_original_rights_and_512_token_limit(self):
        self.use_real_teacher_helpers_without_model()
        self.load_tiny_tokenizer()
        self.runtime.generate = self.complete_generate
        agreement = self.settings.agreement()
        for identifier, answer, responsibility in [("no-rights", "短答案", False), ("too-long", "长" * 600, True)]:
            material_id = self.runtime.memory.capture_teacher(self.messages, answer,
                {"user_responsibility": responsibility, "agreement_version": agreement["version"]}, False, "词表未加载")
            self.practice.enqueue(identifier, self.messages, material_id)
            self.practice.run_one()
            material = next(row for row in self.runtime.memory.teacher_materials() if row["id"] == material_id)
            self.assertEqual(material["status"], "pending")
            self.assertIsNone(material["sample_id"])
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_material_recheck_never_nests_ordinary_rpc_inside_database_lease(self):
        from contextlib import contextmanager
        agreement = self.settings.agreement()
        material_id = self.runtime.memory.capture_teacher(self.messages, "教师答案",
            {"user_responsibility": True, "agreement_version": agreement["version"]}, False, "词表未加载")
        memory = self.runtime.memory
        original_connect = memory.connect
        original_reserved = memory.growth_reserved
        original_agreement = self.settings.agreement
        active = False
        @contextmanager
        def leased_connect():
            nonlocal active
            self.assertFalse(active)
            with original_connect() as db:
                active = True
                try:
                    yield db
                finally:
                    active = False
        def reserved(messages):
            self.assertFalse(active, "ordinary memory RPC would wait on its own active lease")
            return original_reserved(messages)
        def current_agreement():
            self.assertFalse(active, "settings lookup should also run outside the DB lease")
            return original_agreement()
        with patch.object(memory, "connect", side_effect=leased_connect), \
                patch.object(memory, "growth_reserved", side_effect=reserved), \
                patch.object(self.settings, "agreement", side_effect=current_agreement):
            self.practice._recheck_material({"material_id": material_id}, self.messages)
        self.assertEqual(memory.teacher_materials()[0]["status"], "eligible")

    def test_pending_material_recheck_reuses_nonempty_and_12000_character_gate(self):
        agreement = self.settings.agreement()
        self.runtime.teacher_eligibility = Mock(return_value=(True, "假词表压缩后少于512词元"))
        for index, (messages, answer) in enumerate([(self.messages, "A" * 12000),
                ([{"role": "user", "content": "   "}], "answer"), (self.messages, "   ")]):
            with self.subTest(index=index):
                material_id = self.runtime.memory.capture_teacher(messages, answer,
                    {"user_responsibility": True, "agreement_version": agreement["version"]}, False, "尚未检查")
                self.practice._recheck_material({"material_id": material_id}, messages)
                material = next(row for row in self.runtime.memory.teacher_materials() if row["id"] == material_id)
                self.assertEqual(material["status"], "pending")
                self.assertIsNone(material["sample_id"])
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_scheduler_respects_foreground_busy_cooldown_and_pause(self):
        import time
        self.queue()
        self.runtime.generate = self.complete_generate
        busy = True
        automatic = AutomaticLearning(self.runtime, self.settings, threading.Lock(), lambda: busy)
        automatic.teacher_practice = self.practice
        automatic.last_activity = time.monotonic() - 100
        automatic.tick()
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(self.rows()[0]["state"], "queued")
        busy = False
        automatic.last_activity = time.monotonic()
        automatic.tick()
        self.assertIsNone(automatic.worker)
        automatic.last_activity -= 100
        with patch("luban.automatic.training_runs", return_value=[]), patch("luban.automatic.knowledge_keys", return_value=(set(),set(),None)):
            automatic.tick()
            automatic.worker.join(2)
        self.assertEqual(self.rows()[0]["state"], "complete")
        automatic.pause()
        self.assertTrue(self.practice.stop_event.is_set())
