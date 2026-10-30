"""Real temporary SQLite and isolated HTTP fixtures; no model or paid network."""
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import httpx

from luban.feeding import FeedingService, MAX_RESPONSE_CHARS
from luban.memory import Memory
from luban.retention import eligible_learning_sample, provenance
from luban.runtime import Runtime
from luban.settings import Settings
from luban.teacher import Teacher


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(json.dumps(messages, ensure_ascii=False))))

    def encode(self, text, **kwargs):
        return list(range(len(text)))


class FeedingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(self.root / "private")
        self.settings.update({"storage_root": str(self.root / "storage"), "teacher": {
            "base_url": "http://offline.invalid/v1", "api_key": "offline-secret", "model": "fixture"}})
        self.settings.accept_agreement()
        self.runtime = SimpleNamespace(memory=Memory(self.root / "storage"), identity="a" * 64,
            active_core=None, tokenizer=Tokenizer(), lock=threading.RLock(), training_session=None,
            store=self.root / "storage/modules/luban0-v1")
        self.runtime.memory.model_identity = self.runtime.identity
        self.runtime.teacher_usage = lambda *args: Runtime.teacher_usage(self.runtime, *args)
        self.requests = []
        self.reply = None
        self.finish, self.reported = "stop", True
        self.teacher = Teacher(self.settings, httpx.MockTransport(self.respond))
        self.service = FeedingService(self.runtime, self.settings, self.teacher)

    def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        source = json.loads(payload["messages"][1]["content"])["source"]
        row = {"source_id": source["id"], "question": "模型叫什么？", "answer": "鲁班零号", "quote": source["text"]}
        value = self.reply(row) if callable(self.reply) else self.reply if self.reply is not None else [row]
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        data = {"choices": [{"message": {"content": raw}, "finish_reason": self.finish}]}
        if self.reported:
            data["usage"] = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        return httpx.Response(200, json=data)

    def scan(self, text="模型名：鲁班零号。", name="notes.txt"):
        return self.service.scan([{"name": name, "content": text}])

    def cleaned(self):
        batch = self.scan()
        return self.service.clean(batch["id"])

    def test_downloadable_templates_match_accepted_formats_without_teacher(self):
        root = Path(__file__).resolve().parents[1] / "frontend/assets/templates"
        result = self.service.scan([{"name": name, "content": (root / name).read_text(encoding="utf-8")}
                                    for name in ("source.txt", "qa.csv", "qa.jsonl")])
        self.assertTrue(all(file["supported"] for file in result["files"]))
        self.assertEqual(len(result["files"]), 3)
        self.assertEqual(self.requests, [])

    def self_scan(self, pairs=None, name="qa.json"):
        return self.service.scan([{"name": name, "content": json.dumps(pairs or [
            {"question": "模型叫什么？", "answer": "鲁班零号"}], ensure_ascii=False)}], mode="self")

    def test_self_qa_preview_and_confirmation_never_call_or_bind_teacher(self):
        self.settings.credentials["teacher_key"] = ""
        pending = self.self_scan()
        self.assertEqual(pending["mode"], "self")
        self.assertEqual(pending["state"], "pending_confirmation")
        self.assertEqual(pending["max_teacher_calls"], 0)
        self.assertFalse(pending["preview"][0]["human_verified"])
        self.assertEqual(self.runtime.memory.samples(), [])
        self.settings.update({"teacher": {"base_url": "http://different.invalid/v1", "model": "another"}})
        self.assertEqual(self.service.clean(pending["id"])["state"], "pending_confirmation")
        self.assertEqual(self.service.confirm(pending["id"])["added"], 1)
        sample = self.runtime.memory.samples()[0]
        self.assertEqual(sample["source"], "human")
        self.assertTrue(eligible_learning_sample(sample))
        self.assertTrue(provenance(sample)["human_verified"])
        self.assertNotIn("teacher_provider", provenance(sample))
        self.assertNotIn("teacher_model", provenance(sample))
        self.assertIn("question: 模型叫什么？\nanswer: 鲁班零号", provenance(sample)["source_quote"])
        self.assertEqual(self.service.confirm(pending["id"])["already_confirmed"], 1)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.teacher.calls, 0)

    def test_self_formats_keep_whole_qa_and_same_origin_across_conversion(self):
        pairs = [{"question": "问题甲", "answer": "甲" * 1100}, {"question": "问题乙", "answer": "乙" * 1100}]
        batch = self.self_scan(pairs)
        self.assertEqual(batch["chunk_count"], 2)
        self.assertEqual([item["answer"] for item in batch["preview"]], [pair["answer"] for pair in pairs])
        self.assertTrue(all(item["answer"] in item["quote"] for item in batch["preview"]))
        self.assertEqual(self.service.confirm(batch["id"])["pending"], 2)
        self.assertEqual(self.runtime.memory.samples(), [])
        for name, text in (("qa.csv", "question,answer\n模型叫什么？,鲁班零号\n"),
                           ("qa.jsonl", '{"question":"模型叫什么？","answer":"鲁班零号"}')):
            converted = self.service.scan([{"name": name, "content": text}], mode="self")
            original = self.self_scan()
            self.assertEqual(converted["files"][0]["origin_family"], original["files"][0]["origin_family"])
            self.assertEqual(converted["files"][0]["purpose"], original["files"][0]["purpose"])
            self.assertEqual(len(converted["preview"]), 1)
        self.assertEqual(self.requests, [])

    def test_self_rejects_raw_text_malformed_records_and_existing_bounds(self):
        for mode in ("manual", None, []):
            with self.assertRaises(ValueError): self.service.scan([], mode=mode)
        for name, text in (("note.txt", "question: 问题\nanswer: 答案"), ("note.md", "# 问答"),
                           ("bad.json", '[{"question":"q","answer":"a","extra":1}]')):
            batch = self.service.scan([{"name": name, "content": text}], mode="self")
            self.assertFalse(batch["files"][0]["supported"])
            self.assertEqual(batch["preview"], [])
            self.assertEqual(self.service.clean(batch["id"])["teacher_calls_used"], 0)
            with self.assertRaises(ValueError): self.service.confirm(batch["id"])
        with self.assertRaises(ValueError):
            self.self_scan([{"question": str(i), "answer": "a"} for i in range(17)])
        too_long = self.self_scan([{"question": "q", "answer": "a" * 2000}])
        self.assertFalse(too_long["files"][0]["supported"])
        self.assertEqual(self.requests, [])

    def test_self_preserves_agreement_identity_source_and_tokenizer_guards(self):
        pending = self.self_scan()
        self.settings.data["agreement"] = {}
        with self.assertRaises(ValueError): self.service.confirm(pending["id"])
        self.settings.accept_agreement()
        self.runtime.tokenizer = None
        self.assertEqual(self.service.confirm(pending["id"])["pending"], 1)
        self.runtime.tokenizer = Tokenizer()
        self.runtime.identity = "b" * 64
        with self.assertRaises(ValueError): self.service.confirm(pending["id"])
        self.runtime.identity = "a" * 64
        source = self.root / "storage/app/feeding" / pending["id"] / (pending["files"][0]["sha256"] + ".txt")
        source.write_text("changed")
        with self.assertRaises(ValueError): self.service.confirm(pending["id"])
        self.runtime.identity = None
        unready = self.self_scan()
        with self.assertRaisesRegex(ValueError, "身份"): self.service.confirm(unready["id"])
        self.assertEqual(self.requests, [])
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_self_batch_conflict_rolls_back_all_and_split_does_not_change(self):
        pending = self.self_scan([{"question": "甲？", "answer": "甲"}, {"question": "乙？", "answer": "乙"}])
        first, second = pending["preview"]
        self.assertEqual(first["origin_family"], second["origin_family"])
        self.assertEqual(first["purpose"], second["purpose"])
        self.runtime.memory.add_sample(second["question"], "不同金标准", purpose=second["purpose"],
            approved=True, training_allowed=True, provenance={"base_identity": self.runtime.identity})
        with self.assertRaises(ValueError): self.service.confirm(pending["id"])
        self.assertEqual(len(self.runtime.memory.samples()), 1)
        self.assertTrue(all(item["state"] == "pending" for item in self.service.status(pending["id"])["preview"]))
        self.assertEqual(self.requests, [])

    def test_legacy_teacher_batches_without_mode_keep_original_binding(self):
        pending = self.cleaned()
        body = self.service._load(pending["id"])
        body.pop("mode")
        self.service._write(body)
        self.assertEqual(self.service.status(pending["id"])["mode"], "teacher")
        self.settings.update({"teacher": {"model": "another"}})
        with self.assertRaises(ValueError): self.service.confirm(pending["id"])

    def test_scan_supports_strict_formats_without_teacher_or_samples(self):
        pair = {"question": "模型叫什么？", "answer": "鲁班零号"}
        result = self.service.scan([
            {"name": "folder/note.txt", "content": "模型名：鲁班零号。"},
            {"name": "note.md", "content": "模型名：鲁班零号。"},
            {"name": "pair.csv", "content": "question,answer\n模型叫什么？,鲁班零号\n"},
            {"name": "pair.json", "content": json.dumps([pair], ensure_ascii=False)},
            {"name": "pair.jsonl", "content": json.dumps(pair, ensure_ascii=False)},
            {"name": "image.png", "content": "not a supported text source"}])
        self.assertEqual(sum(f["supported"] for f in result["files"]), 5)
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(result["preview"], [])
        self.assertEqual(self.requests, [])
        self.assertEqual(self.runtime.memory.samples(), [])
        self.assertIn("可能收费", result["fee_notice"])

    def test_path_private_internal_secret_and_bounds_fail_before_calls(self):
        for name in ("../secret.txt", "/absolute.txt", "folder/../doc.txt", "internal/doc.txt",
                     "credentials.json", ".env", "a\\b.txt", "a//b.txt", "a:doc.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError): self.scan(name=name)
        for text in ("LUBAN_INTERNAL_ONLY_MINDMAP", "offline-secret", "a\x00b", "x" * 65537):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError): self.scan(text)
        with self.assertRaises(ValueError): self.service.scan([{"name": "x.txt", "content": "x"}] * 21)
        with self.assertRaises(ValueError): self.scan("x" * 33000)
        self.assertEqual(self.requests, [])

    def test_teacher_accepts_unstructured_tables_and_json_but_self_requires_qa(self):
        for name, text in (("bad.csv", "question,answer,extra\na,b,c"), ("bad.json", '{"question":"a","answer":"b"}'),
                           ("duplicate.json", '[{"question":"a","question":"b","answer":"c"}]'),
                           ("bad.jsonl", '{"question":"a","answer":2}')):
            result = self.scan(text, name)
            self.assertTrue(result["files"][0]["supported"])
            self.assertEqual(self.service._load(result["id"])["chunks"][0]["text"], text)
            strict = self.service.scan([{"name": name, "content": text}], mode="self")
            self.assertFalse(strict["files"][0]["supported"])
        self.assertEqual(self.requests, [])

    def test_cleanup_pending_no_verified_sample_and_idempotent_paid_call(self):
        result = self.cleaned()
        self.assertEqual(result["state"], "pending_confirmation")
        self.assertEqual(result["teacher_calls_used"], 1)
        self.assertEqual(self.runtime.memory.usage()["teacher"]["calls"], 1)
        self.assertFalse(result["preview"][0]["human_verified"])
        self.assertFalse(result["preview"][0]["training_allowed"])
        self.assertEqual(self.runtime.memory.samples(), [])
        repeated = FeedingService(self.runtime, self.settings, self.teacher).clean(result["id"])
        self.assertEqual(repeated["preview"], result["preview"])
        self.assertEqual(len(self.requests), 1)

    def test_explicit_confirmation_is_qualified_and_atomic_idempotent(self):
        pending = self.cleaned()
        result = self.service.confirm(pending["id"])
        self.assertEqual(result["added"], 1)
        sample = self.runtime.memory.samples()[0]
        value = provenance(sample)
        self.assertEqual(sample["source"], "teacher")
        self.assertTrue(eligible_learning_sample(sample))
        self.assertTrue(value["human_verified"])
        self.assertIn("explicitly confirmed", value["verification_evidence"])
        self.assertEqual(value["source_quote"], "模型名：鲁班零号。")
        self.assertEqual(value["evolution_role"], "new")
        again = FeedingService(self.runtime, self.settings, self.teacher).confirm(pending["id"])
        self.assertEqual(again["added"], 0)
        self.assertEqual(again["already_confirmed"], 1)
        self.assertEqual(len(self.runtime.memory.samples()), 1)
        self.assertEqual(self.teacher.calls, 1)

    def test_same_origin_renamed_or_converted_stays_one_split(self):
        pair = {"question": "模型叫什么？", "answer": "鲁班零号"}
        first = self.scan(json.dumps([pair], ensure_ascii=False), "first.json")
        second = self.scan("question,answer\n模型叫什么？,鲁班零号\n", "renamed.csv")
        a, b = first["files"][0], second["files"][0]
        self.assertEqual(a["origin_family"], b["origin_family"])
        self.assertEqual(a["purpose"], b["purpose"])

    def test_quota_or_explicit_budget_failure_makes_zero_calls(self):
        self.settings.update({"teacher": {"session_call_limit": 1}})
        batch = self.scan("模型名：鲁班零号。" * 300)
        with self.assertRaises(RuntimeError): self.service.clean(batch["id"], max_calls=1)
        self.assertEqual(self.teacher.calls, 0)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.service.status(batch["id"])["state"], "scanned")

    def test_agreement_and_missing_provider_zero_calls(self):
        self.settings.data["agreement"] = {}
        batch = self.scan()
        with self.assertRaises(ValueError): self.service.clean(batch["id"])
        self.settings.accept_agreement()
        self.settings.credentials["teacher_key"] = ""
        with self.assertRaises(ValueError): self.service.clean(batch["id"])
        self.assertEqual(self.teacher.calls, 0)

    def test_configuration_core_or_source_mutation_rejected(self):
        batch = self.scan()
        self.runtime.active_core = "core-" + "b" * 12
        with self.assertRaises(ValueError): self.service.clean(batch["id"])
        self.runtime.active_core = None
        source = self.root / "storage/app/feeding" / batch["id"] / (batch["files"][0]["sha256"] + ".txt")
        source.write_text("modified")
        with self.assertRaises(ValueError): self.service.clean(batch["id"])
        self.assertEqual(self.teacher.calls, 0)
        batch = self.scan()
        self.settings.update({"teacher": {"model": "another"}})
        with self.assertRaises(ValueError): self.service.clean(batch["id"])
        self.assertEqual(self.teacher.calls, 0)

    def test_unknown_quote_or_source_empty_duplicate_json_never_approved(self):
        invalid = [lambda row: [row | {"quote": "not in source"}], lambda row: [row | {"source_id": "other"}],
                   lambda row: [row | {"answer": "invented"}], [],
                   '[{"source_id":"a","source_id":"b","question":"q","answer":"a","quote":"a"}]']
        for reply in invalid:
            self.reply = reply
            batch = self.cleaned()
            self.assertEqual(batch["state"], "pending")
            self.assertEqual(batch["preview"], [])
            with self.assertRaises(ValueError): self.service.confirm(batch["id"])
        self.assertEqual(self.runtime.memory.samples(), [])
        self.assertEqual(self.runtime.memory.usage()["teacher"]["calls"], len(invalid))

    def test_incomplete_response_preserves_raw_usage_pending_without_retry(self):
        self.finish = "length"
        result = self.cleaned()
        self.assertEqual(result["state"], "pending")
        self.assertTrue(result["calls"][0]["raw"])
        self.assertEqual(result["calls"][0]["usage"]["total_tokens"], 70)
        self.assertEqual(result["calls"][0]["usage_status"], "reported")
        self.service.clean(result["id"])
        self.assertEqual(self.teacher.calls, 1)
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_missing_usage_and_tokenizer_never_imply_zero_cost_or_approval(self):
        self.reported = False
        self.runtime.tokenizer = None
        result = self.cleaned()
        self.assertEqual(result["calls"][0]["usage_status"], "unknown")
        self.assertEqual(result["teacher_calls_used"], 1)
        self.assertEqual(self.runtime.memory.usage()["teacher"]["estimated_calls"], 1)
        confirmed = self.service.confirm(result["id"])
        self.assertEqual(confirmed["added"], 0)
        self.assertEqual(confirmed["pending"], 1)
        self.runtime.tokenizer = Tokenizer()
        self.assertEqual(self.service.confirm(result["id"])["added"], 1)
        self.assertEqual(self.teacher.calls, 1)

    def test_long_complete_qa_stays_pending_not_truncated(self):
        answer = "鲁" * 600
        self.reply = lambda row: [row | {"answer": answer}]
        batch = self.scan("记录：" + answer)
        result = self.service.clean(batch["id"])
        confirmation = self.service.confirm(batch["id"])
        self.assertEqual(confirmation["added"], 0)
        self.assertIn("512", confirmation["preview"][0]["reason"])
        self.assertEqual(result["preview"][0]["answer"], answer)
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_invalid_confirm_ids_and_preview_mutation_rejected(self):
        batch = self.cleaned()
        item = batch["preview"][0]["id"]
        for ids in ([item, item], ["not-in-batch"], [{}], [], "all"):
            with self.assertRaises(ValueError): self.service.confirm(batch["id"], ids)
        body = self.service._load(batch["id"])
        body["preview"][0]["answer"] = "invented"
        self.service._write(body)
        with self.assertRaises(ValueError): self.service.confirm(batch["id"])
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_bad_http_or_partial_stream_counts_attempt_preserves_partial(self):
        def failure(request): return httpx.Response(429, text="offline-secret")
        self.teacher.transport = httpx.MockTransport(failure)
        result = self.cleaned()
        self.assertEqual(result["state"], "pending")
        self.assertEqual(result["teacher_calls_used"], 1)
        self.assertNotIn("offline-secret", json.dumps(result))
        original = self.teacher.stream
        def interrupted(*args, **kwargs):
            self.teacher.calls += 1
            yield {"content": "partial JSON offline-secret"}
            raise RuntimeError("interrupted offline-secret")
        self.teacher.stream = interrupted
        result = self.cleaned()
        self.teacher.stream = original
        self.assertIn("partial JSON", result["calls"][0]["raw"])
        self.assertNotIn("offline-secret", json.dumps(result))
        self.assertEqual(result["state"], "pending")

    def test_unknown_interrupted_call_marker_cannot_be_reissued(self):
        batch = self.scan()
        body = self.service._load(batch["id"])
        body["state"] = "cleaning"
        body["calls"] = [{"state": "requesting", "call_attempted": None, "source_id": "frozen",
                           "raw": "", "usage_status": "unknown_if_interrupted"}]
        self.service._write(body)
        recovered = FeedingService(self.runtime, self.settings, self.teacher).status()["batches"][0]
        self.assertEqual(recovered["state"], "pending")
        self.assertEqual(recovered["unknown_call_count"], 1)
        result = FeedingService(self.runtime, self.settings, self.teacher).clean(batch["id"])
        self.assertEqual(result["unknown_call_count"], 1)
        self.assertEqual(result["state"], "pending")
        self.assertEqual(self.teacher.calls, 0)
        self.assertEqual(self.requests, [])

    def test_full_batch_confirmation_conflict_rolls_back_every_insert(self):
        batch = self.cleaned()
        item = batch["preview"][0]
        self.runtime.memory.add_sample(item["question"], "已有不同金标准", purpose=item["purpose"], approved=True,
            training_allowed=True, provenance={"base_identity": self.runtime.identity})
        with self.assertRaises(ValueError): self.service.confirm(batch["id"])
        self.assertEqual(len(self.runtime.memory.samples()), 1)
        self.assertEqual(self.service.status(batch["id"])["preview"][0]["state"], "pending")

    def test_frozen_credentials_redact_even_escaped_secret(self):
        secret = 'offline"key\\tail'
        self.settings.update({"teacher": {"api_key": secret}})
        self.reply = lambda row: json.dumps({"echo": secret})
        result = self.cleaned()
        self.assertNotIn(secret, result["calls"][0]["raw"])
        self.assertNotIn(json.dumps(secret)[1:-1], result["calls"][0]["raw"])
        self.assertIn("[redacted]", result["calls"][0]["raw"])

    def test_multiple_insert_confirmation_is_atomic_and_source_split_is_fixed(self):
        self.reply = lambda row: [row, {**row, "question": "另一个模型问题？"}]
        batch = self.cleaned()
        self.assertEqual(len(batch["preview"]), 2)
        first, second = batch["preview"]
        self.assertEqual(first["origin_family"], second["origin_family"])
        self.assertEqual(first["purpose"], second["purpose"])
        self.runtime.memory.add_sample(second["question"], "不同金标准", purpose=second["purpose"],
            approved=True, training_allowed=True, provenance={"base_identity": self.runtime.identity})
        with self.assertRaises(ValueError):
            self.service.confirm(batch["id"])
        self.assertEqual(len(self.runtime.memory.samples()), 1)
        self.assertTrue(all(item["state"] == "pending" for item in self.service.status(batch["id"])["preview"]))

    def test_mid_batch_config_change_does_not_change_frozen_teacher_credentials(self):
        requests = []
        def changed(request):
            requests.append((request.headers["Authorization"], json.loads(request.content)["model"]))
            response = self.respond(request)
            if len(requests) == 1:
                self.settings.update({"teacher": {"api_key": "different-key", "model": "different-model"}})
            return response
        self.teacher = Teacher(self.settings, httpx.MockTransport(changed))
        self.service = FeedingService(self.runtime, self.settings, self.teacher)
        batch = self.service.scan([{ "name": "a.txt", "content": "模型名：鲁班零号。"},
                                   {"name": "b.txt", "content": "第二个资料的模型名：鲁班零号。"}])
        result = self.service.clean(batch["id"])
        self.assertEqual(requests, [("Bearer offline-secret", "fixture")] * 2)
        self.assertEqual(result["teacher_calls_used"], 2)
        self.assertNotIn("offline-secret", json.dumps(result))
        with self.assertRaises(ValueError):
            self.service.confirm(batch["id"])

    def test_response_overflow_is_pending_not_successful_truncation(self):
        self.reply = "x" * (MAX_RESPONSE_CHARS + 1)
        result = self.cleaned()
        self.assertEqual(result["state"], "pending")
        self.assertEqual(len(result["calls"][0]["raw"]), MAX_RESPONSE_CHARS)
        self.assertEqual(self.runtime.memory.samples(), [])


if __name__ == "__main__":
    unittest.main()
