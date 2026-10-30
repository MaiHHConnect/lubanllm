"""CPU-only checks for installed-App evidence validation; no App/model startup."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/verify_goal_app.py"
spec = importlib.util.spec_from_file_location("verify_goal_app", SOURCE)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

CORE = "core-123456abcdef"
GOLD = ("我的目标，是让模型像人一样持续学习、积累和成长。"
        "设计借鉴人脑的快慢协作和记忆巩固：快脑先判断，慢脑深入处理；"
        "表层学习快速吸收新经验，深层学习将成熟知识逐步沉淀到模型权重，"
        "让请教后会答变成自己真正会。"
        "每次进化都检验新知识、保护旧能力，合格才升级，失败就回退。")


def fixture():
    context = [{"role": "user", "content": "怎么无限进化"}]
    case = {"id": "goal", "messages": context,
            "check": {"type": "exact", "expected": GOLD, "max_new_tokens": 256}}
    usage = {"finish_reason": "stop", "local_complete": True, "reference_used": False,
             "reference_sample_id": None, "requested_thinking": "off", "effective_thinking": "off",
             "prompt_tokens": 11, "completion_tokens": 80, "total_tokens": 91}
    record = {"request": {"messages": context, "max_tokens": 256}, "events": [
        {"type": "meta", "route": "local", "request_id": "req-1"},
        {"type": "delta", "content": GOLD[:30]},
        {"type": "delta", "content": GOLD[30:]},
        {"type": "done", "usage": usage, "conversation_id": "new-conversation"}]}
    event = {"request_id": "req-1", "messages": context, "answer": GOLD, "usage": usage,
             "decision": {"action": "local_core"}, "complete": True,
             "model_context": {"core_version": CORE, "expert_id": None}}
    return record, case, event


class GoalAppVerificationTests(unittest.TestCase):
    def test_real_ui_sse_shape_crlf_and_unicode(self):
        record, case, event = fixture()
        raw = ": keepalive\r\n\r\n" + "".join(
            "data: " + json.dumps(row, ensure_ascii=False) + "\r\n\r\n" for row in record["events"])
        record["events"] = app.parse_sse(raw)
        app.check_response(record, case, CORE, event)
        self.assertTrue(record["passed"])
        self.assertEqual(record["answer"], GOLD)
        self.assertEqual(record["source_core"], CORE)
        self.assertEqual(record["conversation_id"], "new-conversation")

    def test_default_temperature_preserves_request_and_checks_four_topics(self):
        record, case, event = fixture()
        case.update(user_default_temperature=True, check={"max_new_tokens": 256})
        app.check_response(record, case, CORE, event)
        self.assertNotIn("temperature", record["request"])
        self.assertTrue(all(record["generation_check"]["topics"].values()))
        self.assertTrue(record["complete_eos_reported"])

    def test_reference_stop_limit_cancel_thinking_and_accounting_rejected(self):
        changes = [{"reference_used": True}, {"reference_sample_id": "sample"},
                   {"finish_reason": "length"}, {"local_complete": False},
                   {"effective_thinking": "low"}, {"completion_tokens": True},
                   {"total_tokens": 92}, {"completion_tokens": 400, "total_tokens": 411}]
        for change in changes:
            with self.subTest(change=change):
                record, case, event = fixture()
                record["events"][-1]["usage"].update(change)
                with self.assertRaises(ValueError):
                    app.check_response(record, case, CORE, event)

    def test_wrong_source_or_event_cannot_pass(self):
        for mutate in [
            lambda event: event["model_context"].update(core_version="core-old"),
            lambda event: event["model_context"].update(expert_id="branch"),
            lambda event: event["decision"].update(action="local_expert"),
            lambda event: event.update(complete=False),
            lambda event: event.update(request_id="other"),
            lambda event: event.update(answer="different"),
            lambda event: event.update(messages=[{"role": "user", "content": "different"}]),
            lambda event: event.update(usage={}),
        ]:
            with self.subTest(mutate=mutate):
                record, case, event = fixture()
                mutate(event)
                with self.assertRaises(ValueError):
                    app.check_response(record, case, CORE, event)

    def test_slogan_and_repetition_cannot_pass_default_sampling(self):
        slogan = "我是鲁班零号，luban0模型，基于qwen3.5训练的，能无限进化的模型。"
        self.assertFalse(app.goal_content(slogan)["passed"])
        self.assertTrue(app.repetition(GOLD)["passed"])
        self.assertFalse(app.repetition(slogan * 3)["passed"])
        self.assertFalse(app.repetition(GOLD * 3)["passed"])
        self.assertFalse(app.goal_content(GOLD.replace("快", "").replace("慢", ""))["passed"])
        for missing in ["表层学习快速吸收新经验，", "深层学习将成熟知识逐步沉淀到模型权重，",
                        "每次进化都检验新知识、保护旧能力，合格才升级，失败就回退。"]:
            self.assertFalse(app.goal_content(GOLD.replace(missing, ""))["passed"], missing)

    def test_incomplete_error_duplicate_done_and_reasoning_rejected(self):
        for transform in [
            lambda events: events.pop(),
            lambda events: events.append(copy.deepcopy(events[-1])),
            lambda events: events.insert(1, {"type": "error", "message": "failure"}),
            lambda events: events[1].update(reasoning_content="thinking"),
        ]:
            record, case, event = fixture()
            transform(record["events"])
            with self.assertRaises(ValueError):
                app.check_response(record, case, CORE, event)

    def test_nonfinite_json_and_compatibility_sse_rejected(self):
        for raw in ["data: [DONE]\n\n", "data: []\n\n", "data: {bad}\n\n",
                    'data: {"number": NaN}\n\n', 'data: {"number": Infinity}\n\n']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                app.parse_sse(raw)
        with self.assertRaises(ValueError):
            app.finite({"nested": [float("inf")]})


class FoundationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source, self.previous, self.version = CORE, "core-b45d965728bb", "core-fedcba654321"
        self.identity = "a" * 64
        self.store = self.root / "storage/modules/luban0-v1"
        self.store.mkdir(parents=True)
        self.members = self.root / "model/luban-foundation"
        self.members.mkdir(parents=True)
        self.manifest_path = self.members.parent / "luban-foundation.json"
        self.protocol = self.root / "protocol.json"
        self.acceptance = self.root / "acceptance.json"
        self.plan = {"identity": self.identity, "storage": str(self.root / "storage")}
        self.save(self.protocol, {"source_core": {"version": self.previous}})
        # Tiny bytes represent immutable SF copying in this CPU binding test;
        # real raw BF16/EOS validation is a separate runner/native proof.
        (self.store / (self.source + ".safetensors")).write_bytes(b"immutable CPU SF fixture")
        sf_sha = app.sha(self.store / (self.source + ".safetensors"))
        retention = {"schema_version": 1, "version": self.source, "identity": self.identity,
                     "groups": {}, "regression_validation": []}
        retention["content_sha256"] = app.digest(retention)
        self.save(self.store / (self.source + ".retention.json"), retention)
        source_report_name = "evolution-" + self.source[5:] + ".json"
        checkpoint = {"version": self.source, "identity": self.identity, "report": source_report_name,
                      "sha256": sf_sha, "retention_sha256": app.sha(self.store / (self.source + ".retention.json"))}
        source_report = {"version": self.source, "id": self.source[5:], "previous_version": self.previous,
                         "identity": self.identity, "publication_state": "published", "promoted": True,
                         "checkpoint": checkpoint, "gates": {"real_original_gate": True}, "steps": 450,
                         "retention": {"file_sha256": checkpoint["retention_sha256"],
                                       "content_sha256": retention["content_sha256"]}}
        self.save(self.store / (self.source + ".json"), checkpoint)
        self.save(self.store / source_report_name, source_report)
        self.candidate = {"passed": True, "candidate_version": self.source, "source_core": self.previous,
                          "candidate_tensor_sha256": "c" * 64, "training": source_report,
                          "artifacts": {member.name: app.sha(member) for member in self.store.iterdir()}}
        self.save(self.acceptance, self.candidate)
        (self.members / (self.version + ".safetensors")).write_bytes(
            (self.store / (self.source + ".safetensors")).read_bytes())
        retention = copy.deepcopy(retention)
        retention["version"] = self.version
        retention["content_sha256"] = app.digest({key: value for key, value in retention.items() if key != "content_sha256"})
        self.save(self.members / (self.version + ".retention.json"), retention)
        self.report_name = "evolution-" + self.version[5:] + ".json"
        checkpoint = {**checkpoint, "version": self.version, "report": self.report_name,
                      "retention_sha256": app.sha(self.members / (self.version + ".retention.json"))}
        self.save(self.members / (self.version + ".json"), checkpoint)
        self.report = {**source_report, "version": self.version, "id": self.version[5:], "checkpoint": checkpoint,
                       "source_previous_version": self.previous, "source_training_version": self.source,
                       "source_quality_sha256": self.candidate["artifacts"][source_report_name],
                       "source_protocol_sha256": app.sha(self.protocol),
                       "native_identity_sha256": app.sha(self.acceptance), "native_identity_verified": True,
                       "retention": {"file_sha256": checkpoint["retention_sha256"],
                                     "content_sha256": retention["content_sha256"]}}
        self.reseal()

    def save(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def reseal(self):
        self.save(self.members / self.report_name, self.report)
        self.save(self.manifest_path, {"version": self.version, "identity": self.identity, "quality_passed": True,
                                      "report": self.report_name,
                                      "files": {member.name: app.sha(member) for member in self.members.iterdir()}})

    def binding(self):
        return app.foundation_binding(self.manifest_path, self.candidate, self.acceptance, self.protocol, self.plan)

    def test_real_F_files_link_to_unchanged_C_proof(self):
        before = self.acceptance.read_bytes()
        binding = self.binding()
        self.assertEqual(binding["version"], self.version)
        self.assertEqual(binding["source_training_version"], self.source)
        self.assertEqual(binding["manifest_sha256"], app.sha(self.manifest_path))
        self.assertEqual(binding["native_identity_sha256"], app.sha(self.acceptance))
        self.assertEqual(binding["core_file_sha256"], self.candidate["artifacts"][self.source + ".safetensors"])
        self.assertEqual(before, self.acceptance.read_bytes())

    def test_resealed_wrong_source_proof_or_original_training_gate_rejected(self):
        original = copy.deepcopy(self.report)
        for change in [{"source_training_version": "core-000000000000"}, {"previous_version": "core-000000000000"},
                       {"source_quality_sha256": "f" * 64}, {"source_protocol_sha256": "f" * 64},
                       {"native_identity_sha256": "f" * 64}, {"steps": 1},
                       {"gates": {"real_original_gate": False}}]:
            with self.subTest(change=change):
                self.report = {**copy.deepcopy(original), **change}
                self.reseal()
                with self.assertRaises(ValueError):
                    self.binding()

    def test_resealed_different_SF_bytes_rejected(self):
        (self.members / (self.version + ".safetensors")).write_bytes(b"different actual bytes")
        self.report["checkpoint"]["sha256"] = app.sha(self.members / (self.version + ".safetensors"))
        self.save(self.members / (self.version + ".json"), self.report["checkpoint"])
        self.reseal()
        with self.assertRaises(ValueError):
            self.binding()

    def test_changed_immutable_C_and_missing_F_member_rejected(self):
        source_member = self.store / (self.source + ".json")
        original = source_member.read_bytes()
        source_member.write_bytes(original + b" ")
        with self.assertRaises(ValueError):
            self.binding()
        source_member.write_bytes(original)
        (self.members / (self.version + ".retention.json")).unlink()
        self.reseal()
        with self.assertRaises(ValueError):
            self.binding()


if __name__ == "__main__":
    unittest.main()
