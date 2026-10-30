"""Application contracts with temporary storage and an offline HTTP teacher."""
import json
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from luban.agreement import AGREEMENT_VERSION
from luban.memory import Memory
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import DEFAULTS, Settings
from luban.teacher import Teacher


MESSAGES = [{"role": "user", "content": "你好"}]


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


class FakeRuntime:
    """Exercise the actual HTTP adapters without loading pretrained weights."""
    def __init__(self, memory):
        self.memory = memory
        self.state = "ready"
        self.job = None
        self.lock = threading.RLock()
        self.calls = []
        self.failure = None
        self.restore_ids = []

    def status(self):
        return {"state": self.state, "router": {"calibrated": False}}

    def choose_route(self, messages, requested="auto"):
        return {"route": "local" if requested == "auto" else requested,
                "source": "local_default" if requested == "auto" else "user",
                "confidence": None}

    def generate(self, messages, emit, stop, **kwargs):
        if self.failure:
            raise RuntimeError(self.failure)
        self.calls.append((messages, kwargs))
        emit({"reasoning_content": "思考"})
        emit({"content": "本地"})
        emit({"content": "回答"})
        return {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4,
                "finish_reason": "stop"}

    def versions(self):
        return []

    def training_backups(self):
        return [{"id": "fixture-backup", "kind": "learn"}]

    def restore_training(self, backup_id):
        self.restore_ids.append(backup_id)
        if backup_id != "fixture-backup":
            raise ValueError("训练备份编号无效")
        return {"id": backup_id, "kind": "learn"}

    def teacher_usage(self, messages, answer, reasoning, reported=None):
        if reported is not None:
            return {**reported, "estimated": False}
        return {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
                "estimated": True}

    def teacher_eligibility(self, messages, answer, config):
        allowed = config.get("agreement_accepted") is True
        return allowed, "测试替身：仅核验协议接受状态，真实词表由 Runtime 单独验证"


class SettingsTests(unittest.TestCase):
    def test_native_execution_policy_is_opt_in_strict_and_persistent(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home)
            self.assertEqual([settings.snapshot()[key] for key in (
                "native_skip_layer_mask", "native_exit_after", "native_layer_paging")], [0, 24, False])
            for key, invalid_values in {
                "native_skip_layer_mask": (True, "2", 2.0, -1, 1, 1 << 24, (1 << 24) - 1),
                "native_exit_after": (True, "12", 12.0, 0, 25),
                "native_layer_paging": (None, "true", 0, 1),
            }.items():
                for value in invalid_values:
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        settings.update({key: value})
            policy = {"native_skip_layer_mask": 2, "native_exit_after": 12, "native_layer_paging": True}
            settings.update(policy)
            saved = Settings(home).snapshot()
            self.assertTrue(all(saved[key] == value for key, value in policy.items()))
            for changes in ({"native_skip_layer_mask": 1 << 12}, {"native_exit_after": 1},
                            {"native_skip_layer_mask": 1 << 23, "native_exit_after": 23}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "退出位置"):
                    settings.update(changes)
            self.assertEqual(settings.snapshot()["native_skip_layer_mask"], 2)

    def test_packed_q4_requires_file_and_clears_profile_on_precision_change(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home)
            profile = Path(home, "profile.bin")
            profile.write_bytes(b"fixture")
            for invalid in ("", "relative.bin", str(profile.parent), str(profile.with_name("missing.bin")), None):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    settings.update({"native_inference_precision": "mixed-q4-g16-packed", "native_q4_profile": invalid})
            self.assertEqual(settings.snapshot()["native_inference_precision"], "mlx-q4")
            with self.assertRaises(ValueError):
                settings.update({"native_q4_profile": str(profile)})
            settings.update({"native_inference_precision": "mixed-q4-g16-packed", "native_q4_profile": str(profile)})
            self.assertEqual(Settings(home).snapshot()["native_q4_profile"], str(profile))
            settings.update({"native_inference_precision": "bf16"})
            self.assertEqual(settings.snapshot()["native_q4_profile"], "")
            with self.assertRaises(ValueError):
                settings.update({"native_inference_precision": "mixed-q4-g16-packed"})

    def test_shipped_native_precisions_roundtrip_and_default(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home)
            self.assertEqual(settings.snapshot()["native_inference_precision"], "mlx-q4")
            for mode in ("mlx-q4", "bf16", "mixed-q4-g32", "mixed-q4-g16", "mixed-q8"):
                settings.update({"native_inference_precision": mode})
                self.assertEqual(Settings(home).snapshot()["native_inference_precision"], mode)
            with self.assertRaises(ValueError):
                settings.update({"native_inference_precision": "mixed-q4-unknown"})

    def test_optional_teacher_name_persists_clears_and_preserves_credentials(self):
        from luban.feeding import _config
        with tempfile.TemporaryDirectory() as home:
            path = Path(home, "settings.json")
            path.write_text(json.dumps({"teacher": {"model": "offline"}}))
            settings = Settings(home)
            self.assertEqual(settings.public()["teacher"]["name"], "")
            original_training_config = _config(settings)
            settings.update({"teacher": {"name": "  千问老师  ", "api_key": "offline-only-key"}})
            reopened = Settings(home)
            self.assertEqual(reopened.public()["teacher"]["name"], "千问老师")
            self.assertEqual(_config(reopened), original_training_config)
            for name in (None, True, 12, [], {}, "师" * 61, "老\x00师", "老\x85师", "\n老师"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    reopened.update({"teacher": {"name": name}})
            self.assertEqual(reopened.public()["teacher"]["name"], "千问老师")
            reopened.update({"teacher": {"name": ""}})
            self.assertEqual(Settings(home).public()["teacher"]["name"], "")
            self.assertEqual(reopened.snapshot()["teacher"]["model"], "offline")
            self.assertEqual(reopened.secret("teacher_key"), "offline-only-key")

    def test_fresh_settings_use_packaged_mlx_model_directories(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home).snapshot()
            self.assertEqual(Path(settings["model_path"]), Path(home) / "data/models/metadata")
            self.assertEqual(Path(settings["native_mlx_model_path"]), Path(home) / "data/models/mlx")

    def test_only_installed_legacy_default_is_migrated_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy, alias = root / "upstream", root / "luban0"
            legacy.mkdir()
            (legacy / "config.json").write_text("{}")
            alias.symlink_to(legacy, target_is_directory=True)
            with patch("luban.settings.LEGACY_MODEL_PATH", str(legacy)), patch.dict(DEFAULTS, model_path=str(alias)):
                for previous in (str(legacy), str(root / "custom")):
                    with self.subTest(previous=previous):
                        home = root / ("private-" + Path(previous).name)
                        home.mkdir(exist_ok=True)
                        path = home / "settings.json"
                        path.write_text(json.dumps({"model_path": previous, "temperature": 0.3}))
                        settings = Settings(home)
                        expected = str(alias) if previous == str(legacy) else previous
                        self.assertEqual(settings.snapshot()["model_path"], expected)
                        self.assertEqual(json.loads(path.read_text())["model_path"], expected)
                        self.assertEqual(settings.snapshot()["temperature"], 0.3)

    def test_missing_installation_preserves_existing_model_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = str(root / "upstream")
            (root / "settings.json").write_text(json.dumps({"model_path": previous}))
            with patch("luban.settings.LEGACY_MODEL_PATH", previous), patch.dict(DEFAULTS, model_path=str(root / "luban0")):
                self.assertEqual(Settings(root).snapshot()["model_path"], previous)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = Settings(root / "private")
        self.settings.update({"storage_root": str(root / "storage")})
        self.runtime = FakeRuntime(Memory(root / "storage"))
        self.requests = []
        self.teacher = Teacher(self.settings, httpx.MockTransport(self.teacher_response))
        self.client = TestClient(create_app(self.settings, self.runtime, self.teacher,
                                            autoload=False), raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def teacher_response(self, request):
        self.requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "offline-model"}]})
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text='data: {"choices":[{"delta":{"content":"教师回答"}}]}\n\ndata: [DONE]\n\n')

    def login(self):
        result = self.client.post("/api/session", json={"token": self.settings.secret("ui_token")})
        self.assertEqual(result.status_code, 200)
        return result

    def bearer(self):
        return {"Authorization": "Bearer " + self.settings.secret("access_key")}

    def configure_teacher(self, **changes):
        self.settings.update({"teacher": {"base_url": "https://teacher.invalid/v1",
            "api_key": "teacher-private-key", "model": "offline-model", **changes}})

    def sample(self, **changes):
        return {"prompt": "题目", "answer": "答案", "approved": True,
                "training_allowed": True, **changes}

    def test_update_prepare_requires_session_and_busy_runtime_cannot_install(self):
        self.assertEqual(self.client.post("/api/update/prepare").status_code, 401)
        self.login()
        for state in ("loading", "generating", "learning"):
            self.runtime.state = state
            self.assertEqual(self.client.post("/api/update/prepare").status_code, 409)
        self.runtime.state = "ready"
        self.runtime.job = {"state": "queued"}
        self.assertEqual(self.client.post("/api/update/prepare").status_code, 409)

    def test_update_lease_blocks_new_chat_and_mutations_until_cancelled(self):
        self.login()
        self.assertEqual(self.client.post("/api/update/prepare").status_code, 200)
        self.assertEqual(self.client.post("/api/chat", json={"messages": MESSAGES}).status_code, 409)
        self.assertEqual(self.client.post("/api/model/reload").status_code, 409)
        self.assertEqual(self.client.patch("/api/settings", json={"temperature": 0.2}).status_code, 409)
        self.assertEqual(self.client.post("/api/update/cancel").status_code, 200)
        self.assertEqual(self.client.post("/api/chat", json={"messages": MESSAGES}).status_code, 200)

    def test_update_prepare_refuses_inflight_request_even_before_runtime_state_changes(self):
        self.login()
        entered, release = threading.Event(), threading.Event()
        original = self.runtime.generate
        def held(*args, **kwargs):
            entered.set()
            release.wait(5)
            return original(*args, **kwargs)
        self.runtime.generate = held
        with ThreadPoolExecutor(max_workers=1) as pool:
            request = pool.submit(self.client.post, "/api/chat", json={"messages": MESSAGES})
            self.assertTrue(entered.wait(5))
            try:
                self.assertEqual(self.client.post("/api/update/prepare").status_code, 409)
            finally:
                release.set()
            self.assertEqual(request.result().status_code, 200)

    def test_update_lease_expires_after_abandoned_install(self):
        self.login()
        with patch("luban.server.time.monotonic", return_value=100):
            self.assertEqual(self.client.post("/api/update/prepare").status_code, 200)
        with patch("luban.server.time.monotonic", return_value=161):
            self.assertEqual(self.client.post("/api/chat", json={"messages": MESSAGES}).status_code, 200)
        self.assertEqual(self.client.get("/api/version").json(),
                         {k: self.client.get("/api/health").json()[k] for k in ("version", "build")})

    def test_session_and_bearer_are_distinct_capabilities(self):
        for path in ("/api/status", "/api/settings", "/api/access-key", "/api/samples"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path, headers=self.bearer()).status_code, 401)
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        self.assertEqual(self.client.post("/api/session", json={"token": "wrong"}).status_code, 401)
        cookie = self.login().headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        self.assertEqual(self.client.get("/v1/models", headers=self.bearer()).status_code, 200)
        self.assertEqual(self.client.get("/api/status").headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/api/status").headers["x-content-type-options"], "nosniff")

    def test_cross_site_and_untrusted_host_are_rejected(self):
        self.login()
        self.assertEqual(self.client.get("/api/status", headers={"host": "attacker.invalid"}).status_code, 403)
        self.assertEqual(self.client.get("/api/status", headers={"origin": "https://attacker.invalid"}).status_code, 403)
        self.assertEqual(self.client.post("/api/session", json={"token": self.settings.secret("ui_token")},
                                          headers={"origin": "https://attacker.invalid"}).status_code, 403)
        self.assertEqual(self.client.get("/api/status", headers={"origin": "http://127.0.0.1:8765"}).status_code, 200)

    def test_key_rotation_revokes_previous_bearer_and_public_views_hide_secrets(self):
        self.login()
        self.configure_teacher()
        old = self.bearer()
        rotated = self.client.post("/api/access-key/rotate")
        self.assertEqual(rotated.status_code, 200)
        self.assertNotEqual(rotated.json()["key"], old["Authorization"][7:])
        self.assertEqual(self.client.get("/v1/models", headers=old).status_code, 401)
        self.assertEqual(self.client.get("/v1/models", headers=self.bearer()).status_code, 200)
        for path in ("/api/health", "/api/status", "/api/settings", "/api/samples", "/api/events"):
            with self.subTest(path=path):
                body = self.client.get(path).text
                for name in ("access_key", "ui_token", "teacher_key"):
                    self.assertNotIn(self.settings.secret(name), body)
        restarted = Settings(self.settings.home)
        self.assertEqual(restarted.secret("access_key"), rotated.json()["key"])
        for path in (self.settings.path, self.settings.secret_path, self.settings.home / "launch.json"):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.settings.home.stat().st_mode), 0o700)

    def test_teacher_key_blank_preserves_and_clear_removes(self):
        self.login()
        self.configure_teacher()
        result = self.client.patch("/api/settings", json={"teacher": {"api_key": ""}})
        self.assertTrue(result.json()["teacher"]["key_configured"])
        self.assertEqual(self.settings.secret("teacher_key"), "teacher-private-key")
        result = self.client.patch("/api/settings", json={"teacher": {"clear_key": True}})
        self.assertFalse(result.json()["teacher"]["key_configured"])
        self.assertEqual(self.settings.secret("teacher_key"), "")

    def test_local_json_and_compatible_stream_have_usage(self):
        answer = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                  json={"messages": MESSAGES, "max_tokens": 12, "temperature": 0})
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.json()["choices"][0]["message"]["content"], "本地回答")
        self.assertEqual(answer.json()["usage"]["total_tokens"], 4)
        self.assertEqual(self.runtime.calls[-1][1]["max_tokens"], 12)
        stream = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                  json={"messages": MESSAGES, "stream": True})
        parsed = events(stream)
        self.assertEqual(parsed[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(parsed[-1]["usage"]["total_tokens"], 4)
        self.assertTrue(stream.text.rstrip().endswith("data: [DONE]"))
        self.assertEqual(self.runtime.memory.conversations(), [])

    def test_long_tool_history_reaches_runtime_without_message_truncation(self):
        messages = [{"role": "system", "content": "保留完整历史。"}]
        for index in range(40):
            messages.extend([
                {"role": "user", "content": f"查询第{index}项"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"call_{index}", "type": "function", "function": {
                        "name": "lookup", "arguments": json.dumps({"index": index})}}]},
                {"role": "tool", "tool_call_id": f"call_{index}", "content": str(index)},
                {"role": "assistant", "content": f"第{index}项结果为{index}"},
            ])
        messages.append({"role": "user", "content": "根据全部结果回答。"})
        self.assertEqual(len(messages), 162)
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                response = self.client.post("/v1/chat/completions", headers=self.bearer(),
                    json={"messages": messages, "max_tokens": 12, "stream": streaming})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(self.runtime.calls[-1][0]), len(messages))
                self.assertTrue(self.runtime.calls[-1][0][0]["content"].startswith(messages[0]["content"]))
                for original, actual in zip(messages[1:], self.runtime.calls[-1][0][1:]):
                    expected = json.loads(json.dumps(original))
                    expected["content"] = expected["content"] or ""
                    for call in expected.get("tool_calls", []):
                        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                    self.assertEqual(actual, expected)
        invalid = [*messages, {"role": "tool", "tool_call_id": "unknown", "content": "x"}]
        response = self.client.post("/v1/chat/completions", headers=self.bearer(), json={"messages": invalid})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.runtime.calls), 2)

    def test_local_first_auto_stays_local_without_asking_calibrated_teacher_router(self):
        self.runtime.choose_route = lambda *_: {"route": "teacher", "action": "teacher",
            "source": "calibrated_head", "confidence": 0.95}
        self.login()
        response = self.client.post("/api/chat", json={"messages": MESSAGES, "route": "auto"})
        values = events(response)
        meta = next(v for v in values if v["type"] == "meta")
        self.assertEqual(meta["source"], "local_first")
        self.assertEqual(meta["route"], "local")
        self.assertNotIn("suggested_action", meta)
        self.assertTrue(self.runtime.calls)
        self.assertFalse(self.requests)

    def test_waiting_streams_send_comments_without_inventing_content_or_usage(self):
        generate = self.runtime.generate
        def delayed(*args, **kwargs):
            threading.Event().wait(0.06)
            return generate(*args, **kwargs)
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        self.login()
        with patch("luban.server.SSE_HEARTBEAT_SECONDS", 0.01), patch.object(self.runtime, "generate", side_effect=delayed):
            for path, body in [("/api/chat", {}), ("/v1/chat/completions", {"tools": tools, "stream": True})]:
                response = self.client.post(path, headers=self.bearer(), json={"messages": MESSAGES, **body})
                self.assertEqual(response.status_code, 200)
                self.assertIn(": keepalive\n\n", response.text)
                parsed = events(response)
                self.assertNotIn("error", parsed[-1])
                self.assertEqual(parsed[-1]["usage"]["total_tokens"], 4)
                if path == "/api/chat":
                    self.assertEqual("".join(v.get("content", "") for v in parsed), "本地回答")
                else:
                    self.assertEqual("".join(v["choices"][0]["delta"].get("content", "") for v in parsed), "本地回答")
                    self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))

    def test_automatic_answer_mode_uses_router_and_falls_back_when_teacher_is_unavailable(self):
        self.configure_teacher(answer_mode="auto")
        self.runtime.choose_route = lambda *_: {"route": "teacher", "action": "teacher",
            "source": "calibrated_head", "confidence": 0.95}
        self.login()
        response = self.client.post("/api/chat", json={"messages": MESSAGES, "route": "auto"})
        values = events(response)
        meta = next(v for v in values if v["type"] == "meta")
        self.assertEqual(meta["route"], "teacher")
        self.assertEqual("".join(v.get("content", "") for v in values), "教师回答")
        self.settings.update({"teacher": {"base_url": ""}})
        response = self.client.post("/api/chat", json={"messages": MESSAGES, "route": "auto"})
        values = events(response)
        meta = next(v for v in values if v["type"] == "meta")
        self.assertEqual(meta["source"], "auto_teacher_unavailable")
        self.assertEqual(meta["route"], "local")

    def test_teacher_draft_http_list_and_generation_do_not_save_settings(self):
        self.login()
        before, secrets_before = self.settings.path.read_bytes(), self.settings.secret_path.read_bytes()
        draft = {"base_url": "http://remote.invalid/v1", "api_key": "draft-test-key", "model": "draft-model"}
        models = self.client.post("/api/teacher/models", json={"teacher": draft})
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["models"], ["offline-model"])
        tested = self.client.post("/api/teacher/test", json={"teacher": draft})
        self.assertEqual(tested.status_code, 200)
        self.assertEqual(tested.json()["reply"], "教师回答")
        self.assertEqual(tested.json()["model"], "draft-model")
        self.assertEqual(json.loads(self.requests[-1].content)["max_tokens"], 16)
        self.assertEqual(self.requests[-1].headers["authorization"], "Bearer draft-test-key")
        self.assertEqual(self.settings.path.read_bytes(), before)
        self.assertEqual(self.settings.secret_path.read_bytes(), secrets_before)
        self.assertEqual(self.runtime.memory.usage()["teacher"]["calls"], 1)
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_draft_new_address_does_not_receive_saved_key(self):
        self.login()
        self.configure_teacher()
        for endpoint in ("/api/teacher/models", "/api/teacher/test"):
            with self.subTest(endpoint=endpoint):
                response = self.client.post(endpoint, json={"teacher": {"base_url": "http://different.invalid/v1", "model": "new-model"}})
                self.assertEqual(response.status_code, 400)
                self.assertIn("重新填写密钥", response.json()["error"]["message"])
        self.assertFalse(self.requests)
        same = self.client.post("/api/teacher/test", json={"teacher": {"base_url": "https://teacher.invalid/v1", "model": "new-model"}})
        self.assertEqual(same.status_code, 200)
        self.assertEqual(self.requests[-1].headers["authorization"], "Bearer teacher-private-key")

    def test_teacher_test_reports_http_network_and_empty_reply_without_secret(self):
        self.login()
        draft = {"base_url": "http://test.invalid/v1", "api_key": "private-draft-secret", "model": "test-model"}
        for status in (401, 404, 429):
            self.teacher.transport = httpx.MockTransport(lambda r: httpx.Response(status, text="private-draft-secret"))
            response = self.client.post("/api/teacher/test", json={"teacher": draft})
            self.assertEqual(response.status_code, 409)
            self.assertIn(str(status), response.json()["error"]["message"])
            self.assertNotIn(draft["api_key"], response.text)
        def disconnected(request):
            raise httpx.ConnectError("private-draft-secret", request=request)
        self.teacher.transport = httpx.MockTransport(disconnected)
        response = self.client.post("/api/teacher/test", json={"teacher": draft})
        self.assertEqual(response.status_code, 409)
        self.assertNotIn(draft["api_key"], response.text)
        self.teacher.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}))
        response = self.client.post("/api/teacher/test", json={"teacher": draft})
        self.assertEqual(response.status_code, 409)
        self.assertIn("未返回可见回答", response.json()["error"]["message"])

    def test_remote_http_teacher_setting_is_allowed(self):
        self.login()
        response = self.client.patch("/api/settings", json={"teacher": {"base_url": "http://remote.invalid/v1"}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.settings.snapshot()["teacher"]["base_url"], "http://remote.invalid/v1")

    def test_teacher_success_responses_redact_the_effective_draft_key(self):
        self.login()
        key = "draft-private-secret"
        draft = {"base_url": "http://remote.invalid/v1", "api_key": key, "model": key}
        def echo(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": key}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK " + key, "reasoning_content": key}, "finish_reason": "stop"}]})
        self.teacher.transport = httpx.MockTransport(echo)
        for endpoint in ("/api/teacher/models", "/api/teacher/test"):
            response = self.client.post(endpoint, json={"teacher": draft})
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(key, response.text)
            self.assertIn("[redacted]", response.text)
        self.teacher.transport = httpx.MockTransport(lambda r: httpx.Response(200,
            headers={"content-type": "text/event-stream"}, text="".join(
                "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
                for piece in (key[:8], key[8:])) + "data: [DONE]\n\n"))
        response = self.client.post("/api/teacher/test", json={"teacher": draft})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(key, response.text)
        self.assertIn("[redacted]", response.text)

    def test_ui_chat_persists_history_but_not_training_materials(self):
        self.login()
        result = self.client.post("/api/chat", json={"messages": MESSAGES})
        parsed = events(result)
        self.assertEqual(parsed[0]["source"], "local_first")
        history = self.client.get("/api/conversations/" + parsed[-1]["conversation_id"]).json()
        self.assertEqual(history["messages"][-1]["content"], "本地回答")
        self.assertEqual(self.runtime.memory.samples(), [])
        self.assertEqual(self.client.get("/api/conversations/missing").status_code, 404)

    def test_conversation_delete_is_authenticated_idle_only_and_preserves_learning(self):
        conversation_id = self.runtime.memory.save_conversation(None, MESSAGES)
        endpoint = "/api/conversations/" + conversation_id
        self.assertEqual(self.client.delete(endpoint).status_code, 401)
        self.login()
        sample = self.runtime.memory.add_sample("问题", "答案", approved=True, training_allowed=True)
        self.runtime.state = "generating"
        self.assertEqual(self.client.delete(endpoint).status_code, 409)
        self.assertIsNotNone(self.runtime.memory.conversation(conversation_id))
        self.runtime.state = "ready"
        self.assertEqual(self.client.delete(endpoint).json(), {"ok": True})
        self.assertEqual(self.client.get(endpoint).status_code, 404)
        self.assertEqual(self.client.delete(endpoint).json(), {"ok": True})
        self.assertEqual([item["id"] for item in self.runtime.memory.samples()], [sample])

    def test_ui_automatic_output_ignores_legacy_setting_and_keeps_explicit_api_limit(self):
        self.login()
        self.settings.update({"max_new_tokens": 512})
        result = self.client.post("/api/chat", json={"messages": MESSAGES})
        self.assertEqual(events(result)[-1]["type"], "done")
        self.assertEqual(self.runtime.calls[-1][1]["max_tokens"], 0)
        self.client.post("/api/chat", json={"messages": MESSAGES, "max_tokens": 24})
        self.assertEqual(self.runtime.calls[-1][1]["max_tokens"], 24)
        self.client.post("/v1/chat/completions", headers=self.bearer(), json={"messages": MESSAGES})
        self.assertEqual(self.runtime.calls[-1][1]["max_tokens"], 512)
        self.assertEqual(self.settings.snapshot()["max_new_tokens"], 512)

    def test_ui_teacher_length_calls_provider_once_with_bounded_auto_budget(self):
        self.login()
        self.configure_teacher()
        self.settings.update({"max_new_tokens": 512})
        def truncated(request):
            self.requests.append(request)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=
                'data: {"choices":[{"delta":{"content":"未完"},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')
        self.teacher.transport = httpx.MockTransport(truncated)
        result = self.client.post("/api/chat", json={"messages": MESSAGES, "route": "teacher"})
        self.assertEqual(events(result)[-1]["usage"]["finish_reason"], "length")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(json.loads(self.requests[0].content)["max_tokens"], 2048)
        self.assertEqual(self.runtime.memory.usage()["teacher"]["calls"], 1)
        self.assertEqual(self.runtime.calls, [])

    def test_errors_redact_credentials_in_json_and_stream(self):
        self.configure_teacher()
        self.runtime.failure = " ".join(self.settings.secret(name) for name in ("teacher_key", "access_key", "ui_token"))
        for streaming in (False, True):
            result = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                      json={"messages": MESSAGES, "stream": streaming})
            self.assertEqual(result.status_code, 200 if streaming else 409)
            self.assertIn("[redacted]", result.text)
            for name in ("teacher_key", "access_key", "ui_token"):
                self.assertNotIn(self.settings.secret(name), result.text)

    def test_parameter_validation_prevents_any_generation(self):
        cases = [{"messages": []}, {"messages": [{"role": "tool", "content": "x"}]},
                 {"messages": [{"role": "user", "content": " "}]}, {"model": "other"},
                 {"route": "other"}, {"thinking": "other"}, {"max_tokens": 0},
                 {"max_tokens": 32769}, {"temperature": -1}, {"temperature": "0.5"},
                 {"messages": [{"role": "user", "content": "x" * 1048577}]}]
        for changes in cases:
            with self.subTest(changes=list(changes)):
                result = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                          json={"messages": MESSAGES, **changes})
                self.assertEqual(result.status_code, 400)
        self.assertEqual(self.runtime.calls, [])

    def test_wrong_json_types_are_client_errors(self):
        self.login()
        cases = [("/api/session", []), ("/api/settings", {"teacher": None}),
                 ("/api/settings", {"teacher": {"base_url": None}}),
                 ("/api/settings", {"access_key": None}),
                 ("/api/samples", self.sample(prompt=None)),
                 ("/api/samples", self.sample(source=[])),
                 ("/api/learn", {"steps": None}),
                 ("/v1/chat/completions", []),
                 ("/v1/chat/completions", {"messages": MESSAGES, "max_tokens": None})]
        for path, body in cases:
            with self.subTest(path=path, body=body):
                result = self.client.request("PATCH" if path == "/api/settings" else "POST", path,
                                              json=body, headers=self.bearer())
                self.assertEqual(result.status_code, 400)
        self.assertEqual(self.runtime.calls, [])

    def test_numeric_and_permission_types_cannot_silently_coerce(self):
        self.login()
        for body in ({"max_new_tokens": True}, {"max_new_tokens": 32.5},
                     {"teacher": {"send_reasoning_effort": "false"}},
                     {"teacher": {"session_call_limit": 3.5}}):
            with self.subTest(body=body):
                self.assertEqual(self.client.patch("/api/settings", json=body).status_code, 400)
        for body in ({"max_tokens": True}, {"max_tokens": 2.5}, {"temperature": True}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/v1/chat/completions", headers=self.bearer(),
                                                  json={"messages": MESSAGES, **body}).status_code, 400)

    def test_invalid_settings_leave_previous_config_and_secrets_unchanged(self):
        self.login()
        before = self.settings.snapshot()
        for teacher_url in ("ftp://remote.invalid/v1", "https://user:pass@teacher.invalid/v1",
                            "https://teacher.invalid/v1?key=secret", "https://teacher.invalid/#secret"):
            with self.subTest(url=teacher_url):
                result = self.client.patch("/api/settings", json={"temperature": 1,
                    "teacher": {"base_url": teacher_url, "api_key": "must-not-save"}})
                self.assertEqual(result.status_code, 400)
                self.assertEqual(self.settings.snapshot(), before)
                self.assertEqual(self.settings.secret("teacher_key"), "")
        self.runtime.state = "generating"
        self.assertEqual(self.client.patch("/api/settings", json={"storage_root": self.temp.name}).status_code, 409)

    def test_credentials_must_be_valid_http_header_text(self):
        self.login()
        for body in ({"access_key": "密" * 20}, {"access_key": "a" * 20 + "\x00"},
                     {"teacher": {"api_key": "教师密钥"}},
                     {"teacher": {"api_key": "secret\nheader"}},
                     {"teacher": {"api_key": "secret\x00header"}}):
            with self.subTest(body=body):
                self.assertEqual(self.client.patch("/api/settings", json=body).status_code, 400)

    def test_body_limit_and_malformed_json(self):
        self.login()
        result = self.client.post("/api/chat", content=b"{}", headers={"content-length": "8388609"})
        self.assertEqual(result.status_code, 413)
        result = self.client.post("/api/chat", content="{", headers={"content-type": "application/json"})
        self.assertEqual(result.status_code, 400)
        result = self.client.post("/api/chat", content=iter([b"x" * 4194305, b"x" * 4194305]))
        self.assertEqual(result.status_code, 413)

    def test_explicit_sample_permissions_and_disjoint_validation(self):
        self.login()
        for changes in ({"approved": False}, {"training_allowed": False},
                        {"approved": "true"}, {"training_allowed": 1}, {"purpose": "other"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.client.post("/api/samples", json=self.sample(**changes)).status_code, 400)
        training = self.client.post("/api/samples", json=self.sample())
        self.assertEqual(training.status_code, 200)
        duplicate = self.client.post("/api/samples", json=self.sample(prompt=" 题目 ", purpose="validation"))
        self.assertEqual(duplicate.status_code, 400)
        valid = self.client.post("/api/samples", json=self.sample(prompt="独立题目", purpose="validation"))
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(len(self.runtime.memory.samples("training")), 1)
        self.assertEqual(len(self.runtime.memory.samples("validation")), 1)
        self.assertEqual(self.client.delete("/api/samples/" + training.json()["id"]).status_code, 200)
        self.assertEqual(self.runtime.memory.samples("training"), [])
        self.assertEqual(self.client.post("/api/learn", json={"steps": 4}).status_code, 400)

    def test_learning_content_delete_only_accepts_pending_items(self):
        self.login()
        self.runtime.identity = "fixture-identity"
        self.runtime.store = str(self.settings.home)
        sample = self.client.post("/api/samples", json=self.sample(prompt="不想学习的题目")).json()["id"]
        contents = self.client.get("/api/learning/contents").json()
        self.assertIn(sample, {item["id"] for item in contents["pending"]["items"]})
        self.assertEqual(next(item for item in contents["pending"]["items"] if item["id"] == sample)["answer"], "答案")
        deleted = self.client.delete("/api/learning/contents/" + sample)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json(), {"ok": True, "deleted": sample})
        self.assertEqual(self.runtime.memory.samples(), [])
        self.assertEqual(self.client.delete("/api/learning/contents/missing").status_code, 400)

    def test_teacher_material_requires_agreement_and_keeps_user_responsibility_provenance(self):
        self.login()
        self.configure_teacher()
        sample = self.sample(source="teacher")
        del sample["training_allowed"]
        self.assertEqual(self.client.post("/api/samples", json=sample).status_code, 400)
        self.settings.accept_agreement()
        self.assertEqual(self.client.post("/api/samples", json=sample).status_code, 200)
        provenance = json.loads(self.runtime.memory.samples()[0]["provenance"])
        self.assertEqual(provenance["model"], "offline-model")
        self.assertEqual(provenance["agreement_version"], AGREEMENT_VERSION)
        self.assertTrue(provenance["user_responsibility"])
        self.assertEqual(provenance["rights_basis"], "user responsibility under agreement")
        self.assertNotIn("terms", provenance)
        self.assertNotIn(self.settings.secret("teacher_key"), json.dumps(provenance))

    def test_teacher_route_uses_teacher_effort_not_local_thinking(self):
        self.login()
        self.configure_teacher(thinking="high", send_reasoning_effort=True, session_call_limit=1)
        self.assertEqual(self.client.post("/api/teacher/models").json(), {"models": ["offline-model"]})
        self.assertEqual(self.teacher.calls, 0)
        result = self.client.post("/v1/chat/completions", headers=self.bearer(),
            json={"messages": MESSAGES, "route": "teacher", "thinking": "off", "max_tokens": 10})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["choices"][0]["message"]["content"], "教师回答")
        payload = json.loads(self.requests[-1].content)
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["max_tokens"], 10)
        self.assertEqual(self.requests[-1].headers["authorization"], "Bearer teacher-private-key")
        self.assertEqual(self.runtime.calls, [])
        result = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                  json={"messages": MESSAGES, "route": "teacher"})
        self.assertEqual(result.status_code, 409)
        self.assertEqual(len(self.requests), 2)

    def test_teacher_api_auto_archives_material_before_agreement_without_training(self):
        self.configure_teacher()
        result = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                  json={"messages": MESSAGES, "route": "teacher"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["choices"][0]["message"]["content"], "教师回答")
        self.assertEqual(self.runtime.memory.material_counts(), {"total": 1, "pending": 1, "eligible": 0})
        material = self.runtime.memory.teacher_materials()[0]
        self.assertEqual(material["answer"], "教师回答")
        self.assertEqual(material["status"], "pending")
        self.assertEqual(self.runtime.memory.samples(), [])
        self.assertEqual(self.runtime.memory.usage()["teacher"], dict(calls=1,
            prompt_tokens=3, completion_tokens=2, total_tokens=5, estimated_calls=1))

    def test_teacher_reported_usage_is_counted_once_and_not_response_content(self):
        self.configure_teacher()
        self.settings.accept_agreement()
        reported = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        def respond(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=
                'data: {"choices":[{"delta":{"content":"教师回答"}}]}\n\n'
                'data: ' + json.dumps({"choices": [], "usage": reported}) + '\n\ndata: [DONE]\n\n')
        self.teacher.transport = httpx.MockTransport(respond)
        result = self.client.post("/v1/chat/completions", headers=self.bearer(),
            json={"messages": MESSAGES, "route": "teacher", "stream": True})
        self.assertEqual(result.status_code, 200)
        parsed = events(result)
        content = "".join(event["choices"][0]["delta"].get("content", "") for event in parsed)
        self.assertEqual(content, "教师回答")
        self.assertEqual({key: parsed[-1]["usage"][key] for key in reported}, reported)
        self.assertEqual(self.runtime.memory.usage()["teacher"], {**reported, "calls": 1, "estimated_calls": 0})
        material = self.runtime.memory.teacher_materials()[0]
        self.assertEqual(material["status"], "eligible")
        self.assertEqual(material["answer"], "教师回答")
        sample = self.runtime.memory.samples()[0]
        self.assertEqual(sample["source"], "teacher")
        provenance = json.loads(sample["provenance"])
        self.assertIn("structural", provenance["verification_scope"])
        self.assertNotIn("verified_by", provenance)
        self.assertNotIn("route_action", provenance)
        self.assertNotIn(self.settings.secret("teacher_key"), json.dumps(provenance))
        self.login()
        status = self.client.get("/api/status").json()
        self.assertEqual(status["usage"]["teacher"]["total_tokens"], 18)
        self.assertEqual(status["teacher_materials"]["eligible"], 1)

    def test_local_usage_is_separate_and_failed_generation_does_not_count(self):
        result = self.client.post("/v1/chat/completions", headers=self.bearer(), json={"messages": MESSAGES})
        self.assertEqual(result.status_code, 200)
        before = self.runtime.memory.usage()
        self.assertEqual(before["local"]["calls"], 1)
        self.assertEqual(before["local"]["total_tokens"], 4)
        self.assertEqual(before["local"]["estimated_calls"], 0)
        self.assertEqual(before["teacher"]["calls"], 0)
        self.runtime.failure = "模拟模型失败"
        self.assertEqual(self.client.post("/v1/chat/completions", headers=self.bearer(),
                                         json={"messages": MESSAGES}).status_code, 409)
        self.assertEqual(self.runtime.memory.usage(), before)

    def check_partial_teacher_response(self, text):
        self.configure_teacher()
        self.settings.accept_agreement()
        self.teacher.transport = httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "text/event-stream"}, text=text))
        result = self.client.post("/v1/chat/completions", headers=self.bearer(),
            json={"messages": MESSAGES, "route": "teacher"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["choices"][0]["message"]["content"], "部分回答")
        self.assertNotEqual(result.json()["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.runtime.memory.teacher_materials()[0]["status"], "pending")
        self.assertEqual(self.runtime.memory.samples(), [])

    def test_teacher_truncated_stream_is_archive_only(self):
        self.check_partial_teacher_response('data: {"choices":[{"delta":{"content":"部分回答"}}]}\n\n')

    def test_teacher_length_limited_answer_is_archive_only(self):
        self.check_partial_teacher_response('data: {"choices":[{"delta":{"content":"部分回答"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\ndata: [DONE]\n\n')

    def test_local_thinking_tags_split_across_chunks_are_separated_in_json_and_ui_history(self):
        def generate(messages, emit, stop, **kwargs):
            for content in ("<thi", "nk>真实思考", "</thi", "nk>答案"):
                emit({"content": content})
            return {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6,
                    "finish_reason": "stop"}

        with patch.object(self.runtime, "generate", generate):
            response = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                        json={"messages": MESSAGES, "route": "local"})
            self.assertEqual(response.status_code, 200)
            message = response.json()["choices"][0]["message"]
            self.assertEqual(message["content"], "答案")
            self.assertEqual(message["reasoning_content"], "真实思考")
            self.login()
            response = self.client.post("/api/chat", json={"messages": MESSAGES, "route": "local"})
        self.assertEqual(response.status_code, 200)
        parsed = events(response)
        self.assertEqual("".join(row.get("content", "") for row in parsed), "答案")
        self.assertEqual("".join(row.get("reasoning_content", "") for row in parsed), "真实思考")
        conversation_id = next(row["conversation_id"] for row in parsed if row["type"] == "done")
        saved = self.client.get("/api/conversations/" + conversation_id).json()["messages"][-1]
        self.assertEqual(saved["content"], "答案")
        self.assertNotIn("reasoning_content", saved)

    def test_teacher_content_thinking_tags_do_not_enter_material_answers(self):
        self.configure_teacher()
        self.settings.accept_agreement()  # Isolated temporary home only.
        chunks = ("<thi", "nk>教师思考", "</thi", "nk>教师正文")
        stream = "".join("data: " + json.dumps({"choices": [{"delta": {"content": content}}]})
                         + "\n\n" for content in chunks) + "data: [DONE]\n\n"
        self.teacher.transport = httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "text/event-stream"}, text=stream))
        response = self.client.post("/v1/chat/completions", headers=self.bearer(),
                                    json={"messages": MESSAGES, "route": "teacher"})
        self.assertEqual(response.status_code, 200)
        message = response.json()["choices"][0]["message"]
        self.assertEqual(message["content"], "教师正文")
        self.assertEqual(message["reasoning_content"], "教师思考")
        material = self.runtime.memory.teacher_materials()[0]
        self.assertEqual(material["answer"], "教师正文")
        self.assertEqual(material["status"], "eligible")
        self.assertEqual(self.runtime.memory.samples()[0]["answer"], "教师正文")

    def test_training_backup_restore_requires_session_valid_id_and_exclusive_idle_runtime(self):
        path = "/api/training/backups/restore"
        self.assertEqual(self.client.get("/api/training/backups").status_code, 401)
        self.assertEqual(self.client.post(path, json={"id": "fixture-backup"}).status_code, 401)
        self.assertEqual(self.runtime.restore_ids, [])
        self.login()
        self.assertEqual(self.client.get("/api/training/backups").json()["backups"][0]["id"], "fixture-backup")
        for state, job in (("generating", None), ("ready", {"state": "queued"})):
            with self.subTest(state=state, job=job):
                self.runtime.state, self.runtime.job = state, job
                self.assertEqual(self.client.post(path, json={"id": "fixture-backup"}).status_code, 409)
                self.assertEqual(self.runtime.restore_ids, [])
        self.runtime.state, self.runtime.job = "ready", None
        for body in ({}, {"id": "../outside"}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post(path, json=body).status_code, 400)
        self.assertEqual(self.runtime.restore_ids, ["", "../outside"])
        started, release = threading.Event(), threading.Event()
        original_restore = self.runtime.restore_training

        def restore(backup_id):
            started.set()
            if not release.wait(5):
                raise RuntimeError("隔离测试恢复未释放")
            return original_restore(backup_id)

        with patch.object(self.runtime, "restore_training", restore), ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.client.post, path, json={"id": "fixture-backup"})
            try:
                self.assertTrue(started.wait(5))
                self.assertEqual(self.client.post(path, json={"id": "fixture-backup"}).status_code, 409)
                self.assertEqual(self.runtime.restore_ids, ["", "../outside"])
            finally:
                release.set()
            response = pending.result(timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True, "id": "fixture-backup", "kind": "learn"})
        self.assertEqual(self.runtime.restore_ids, ["", "../outside", "fixture-backup"])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = Memory(self.temp.name)

    def test_usage_accumulates_by_route_and_survives_reopen(self):
        self.memory.record_usage("local", {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        self.memory.record_usage("teacher", {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12, "estimated": False})
        self.memory.record_usage("teacher", {"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24, "estimated": True})
        result = Memory(self.temp.name).usage()
        self.assertEqual(result["local"], dict(calls=1, prompt_tokens=2, completion_tokens=3, total_tokens=5, estimated_calls=0))
        self.assertEqual(result["teacher"], dict(calls=2, prompt_tokens=16, completion_tokens=20, total_tokens=36, estimated_calls=1))
        for invalid in (-1, True, 1.5, "2"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.memory.record_usage("teacher", {"prompt_tokens": invalid})
        with self.assertRaises(ValueError):
            self.memory.record_usage("other", {})
        self.assertEqual(self.memory.usage(), result)

    def test_pending_teacher_material_preserves_full_context_without_sample(self):
        messages = [{"role": "system", "content": "测试上下文"}, *MESSAGES]
        material_id = self.memory.capture_teacher(messages, "教师回答", {"provider": "test", "model": "teacher"},
                                                  False, "未许可")
        material = self.memory.teacher_materials()[0]
        self.assertEqual(material["id"], material_id)
        self.assertEqual(json.loads(material["messages"]), messages)
        self.assertEqual(material["reason"], "未许可")
        self.assertEqual(self.memory.material_counts(), {"total": 1, "pending": 1, "eligible": 0})
        self.assertEqual(self.memory.samples(), [])

    def test_auto_material_duplicates_keep_one_sample_and_never_cross_groups(self):
        messages = [{"role": "user", "content": "前一个问题"}, {"role": "assistant", "content": "先前回复"}, *MESSAGES]
        first = self.memory.capture_teacher(messages, "答案一", {}, True)
        self.memory.capture_teacher(messages, "答案一", {}, True)
        self.memory.capture_teacher(messages, "答案二", {}, True)
        samples = self.memory.samples()
        self.assertEqual(len(samples), 2)
        self.assertEqual({sample["prompt"] for sample in samples}, {MESSAGES[0]["content"]})
        self.assertEqual(len({sample["purpose"] for sample in samples}), 1)
        for sample in samples:
            provenance = json.loads(sample["provenance"])
            self.assertEqual(provenance["messages"], messages)
            self.assertIn("structural", provenance["verification_scope"])
            self.assertNotIn("verified_by", provenance)
        self.assertEqual(self.memory.material_counts(), {"total": 3, "pending": 0, "eligible": 3})
        before = next(row for row in self.memory.teacher_materials() if row["id"] == first)
        self.memory.delete_sample(before["sample_id"])
        self.assertEqual(self.memory.material_counts(), {"total": 3, "pending": 2, "eligible": 1})

    def test_auto_material_respects_existing_manual_split_and_fixed_partition(self):
        self.memory.add_sample("人工题目", "人工答案", purpose="validation", approved=True, training_allowed=True)
        self.memory.capture_teacher([{"role": "user", "content": "人工题目"}], "新教师答案", {}, True)
        self.assertEqual({s["purpose"] for s in self.memory.samples()}, {"validation"})
        split = {}
        for index in range(30):
            prompt = f"独立问题{index}"
            self.memory.capture_teacher([{"role": "user", "content": prompt}], "答案", {}, True)
        for sample in self.memory.samples():
            split[sample["prompt"]] = sample["purpose"]
        self.assertEqual(set(split.values()), {"training", "validation"})
        reopened = Memory(self.temp.name)
        for index in range(30):
            prompt = f"独立问题{index}"
            reopened.capture_teacher([{"role": "user", "content": prompt}], "更新答案", {}, True)
        for sample in reopened.samples():
            self.assertEqual(sample["purpose"], split[sample["prompt"]])


class AccountingRuntimeTests(unittest.TestCase):
    """Real accounting/eligibility methods with a tiny deterministic tokenizer."""
    def setUp(self):
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, messages, *, tokenize, **kwargs):
                return [1, 2, 3] if tokenize else "prefix"
            def encode(self, text, **kwargs):
                return list(range(len(text)))
        self.runtime = Runtime.__new__(Runtime)
        self.runtime.tokenizer = Tokenizer()
        self.runtime.device = "cpu"

    def test_provider_usage_is_exact_and_invalid_provider_usage_is_estimated(self):
        reported = {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17}
        self.assertEqual(self.runtime.teacher_usage(MESSAGES, "答", "思考", reported), {**reported, "estimated": False})
        for reported in (None, {"prompt_tokens": True, "completion_tokens": 7, "total_tokens": 8},
                         {"prompt_tokens": -1, "completion_tokens": 7, "total_tokens": 6}):
            with self.subTest(reported=reported):
                self.assertEqual(self.runtime.teacher_usage(MESSAGES, "答", "思考", reported),
                    {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6, "estimated": True})
        self.runtime.tokenizer = None
        with self.assertRaises(RuntimeError):
            self.runtime.teacher_usage(MESSAGES, "答", "", None)

    def test_teacher_eligibility_requires_agreement_and_512_token_boundary(self):
        config = {"agreement_accepted": False, "agreement_version": AGREEMENT_VERSION}
        self.assertFalse(self.runtime.teacher_eligibility(MESSAGES, "答案", config)[0])
        config = {"agreement_accepted": True, "agreement_version": AGREEMENT_VERSION}
        self.assertTrue(self.runtime.teacher_eligibility(MESSAGES, "x" * 505, config)[0])
        self.assertFalse(self.runtime.teacher_eligibility(MESSAGES, "x" * 506, config)[0])
        self.assertFalse(self.runtime.teacher_eligibility(MESSAGES, " ", config)[0])
        self.assertFalse(self.runtime.teacher_eligibility([{"role": "system", "content": "规则"}], "答案", config)[0])
        self.runtime.tokenizer = None
        self.assertFalse(self.runtime.teacher_eligibility(MESSAGES, "答案", config)[0])


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(self.temp.name)
        self.settings.update({"teacher": {"base_url": "https://teacher.invalid/v1",
            "api_key": "secret-never-in-errors", "model": "test-model", "session_call_limit": 1}})

    def test_unconfigured_teacher_does_not_make_network_request(self):
        self.settings.update({"teacher": {"clear_key": True}})
        requests = []
        teacher = Teacher(self.settings, httpx.MockTransport(lambda r: requests.append(r)))
        with self.assertRaises(ValueError):
            teacher.models()
        with self.assertRaises(ValueError):
            list(teacher.stream(MESSAGES, 10))
        self.assertEqual(requests, [])
        self.assertEqual(teacher.calls, 0)

    def test_stream_parses_reasoning_content_keepalives_and_done(self):
        payloads = []
        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, text=
                ': keepalive\n\nevent: message\ndata: {"choices":[{"delta":{"reasoning_content":"想"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"答"}}]}\n\ndata: [DONE]\n\n'
                'data: {"choices":[{"delta":{"content":"不应出现"}}]}\n\n')
        teacher = Teacher(self.settings, httpx.MockTransport(respond))
        self.assertEqual(list(teacher.stream(MESSAGES, 10)), [{"reasoning_content": "想"}, {"content": "答"}])
        self.assertNotIn("reasoning_effort", payloads[0])
        self.assertEqual(payloads[0]["model"], "test-model")
        self.assertEqual(payloads[0]["stream_options"], {"include_usage": True})

    def test_teacher_extracts_usage_from_stream_and_json_fallback(self):
        reported = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                if streaming:
                    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, text=
                        'data: {"choices":[{"delta":{"content":"答案"}}]}\n\n'
                        'data: ' + json.dumps({"choices": [], "usage": reported}) + '\n\ndata: [DONE]\n\n')
                else:
                    response = httpx.Response(200, json={"choices": [{"message": {"content": "答案"}}], "usage": reported})
                teacher = Teacher(self.settings, httpx.MockTransport(lambda request: response))
                self.assertEqual(list(teacher.stream(MESSAGES, 10)), [{"content": "答案"}, {"usage": reported}])

    def test_reasoning_effort_is_opt_in_and_default_is_omitted(self):
        payloads = []
        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "答案", "reasoning_content": "思考"}}]})
        teacher = Teacher(self.settings, httpx.MockTransport(respond))
        self.settings.update({"teacher": {"send_reasoning_effort": True, "thinking": "default", "session_call_limit": 3}})
        self.assertEqual(list(teacher.stream(MESSAGES, 10)), [{"reasoning_content": "思考"}, {"content": "答案"}])
        self.settings.update({"teacher": {"thinking": "high"}})
        list(teacher.stream(MESSAGES, 10))
        list(teacher.stream(MESSAGES, 10, thinking="low"))
        self.assertNotIn("reasoning_effort", payloads[0])
        self.assertEqual(payloads[1]["reasoning_effort"], "high")
        self.assertEqual(payloads[2]["reasoning_effort"], "low")

    def test_errors_and_redirects_are_sanitized_and_attempts_use_quota(self):
        for status in (302, 401, 429, 500):
            with self.subTest(status=status):
                requests = []
                def respond(request):
                    requests.append(request)
                    return httpx.Response(status, headers={"location": "https://attacker.invalid/"},
                                          text="secret-never-in-errors private-prompt")
                teacher = Teacher(self.settings, httpx.MockTransport(respond))
                with self.assertRaisesRegex(RuntimeError, f"HTTP {status}") as raised:
                    list(teacher.stream(MESSAGES, 10))
                self.assertNotIn("secret-never-in-errors", str(raised.exception))
                self.assertNotIn("private-prompt", str(raised.exception))
                with self.assertRaisesRegex(RuntimeError, "额度"):
                    list(teacher.stream(MESSAGES, 10))
                self.assertEqual(len(requests), 1)
                self.assertEqual(teacher.calls, 1)

    def test_provider_sse_error_body_is_not_disclosed(self):
        teacher = Teacher(self.settings, httpx.MockTransport(lambda request: httpx.Response(200,
            headers={"content-type": "text/event-stream"},
            text='data: {"error":{"message":"secret-never-in-errors private-prompt"}}\n\n')))
        with self.assertRaisesRegex(RuntimeError, "教师流式响应报告错误") as raised:
            list(teacher.stream(MESSAGES, 10))
        self.assertNotIn("private-prompt", str(raised.exception))

    def test_concurrent_requests_cannot_exceed_teacher_quota(self):
        self.settings.update({"teacher": {"session_call_limit": 3}})
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "答案"}}]})
        teacher = Teacher(self.settings, httpx.MockTransport(respond))
        def attempt(_):
            try:
                list(teacher.stream(MESSAGES, 10))
                return "sent"
            except RuntimeError as error:
                self.assertIn("额度", str(error))
                return "blocked"
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(attempt, range(12)))
        self.assertEqual(results.count("sent"), 3)
        self.assertEqual(results.count("blocked"), 9)
        self.assertEqual(len(requests), 3)
        self.assertEqual(teacher.calls, 3)


if __name__ == "__main__":
    unittest.main()
