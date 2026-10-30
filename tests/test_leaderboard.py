"""Opt-in ranking tests use offline HTTP transport, never the production board."""
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from luban.leaderboard import Leaderboard, SITE, COUNTRY_TRACE, COUNTRY_CODES, detect_country, validated_profile
from luban.server import create_app
from luban.settings import Settings
from tests.test_upgrades import upgrade_fixture


class LeaderboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        geo_patch = patch("luban.leaderboard.detect_country", return_value=None)
        self.geo_detection = geo_patch.start()
        self.addCleanup(geo_patch.stop)
        self.calls = []
        self.entries = {}

        def handler(request):
            self.calls.append(request)
            self.assertEqual(str(request.url).split("/api/")[0], SITE)
            if request.method == "GET":
                return httpx.Response(200, json={"entries": list(self.entries.values()),
                                                "total": len(self.entries), "updated_at": None, "protocol_version": 2, "measurement": "accepted_branch_upgrade_records"})
            token = request.headers["authorization"]
            if request.method == "PUT":
                body = json.loads(request.content)
                old = self.entries.get(token, {"evolution_count": 0})
                proofs = getattr(self, "proofs", {})
                proofs.setdefault(token, set()).update((record["base_sha256"], record["weights_sha256"]) for record in body["upgrades"])
                self.proofs = proofs
                entry = {"username": body["username"], "evolution_count": max(old["evolution_count"], len(proofs[token])), "rank": 1, "country_code": body.get("country_code")}
                self.entries[token] = entry
                return httpx.Response(200, json={"entry": entry, "protocol_version": 2})
            self.entries.pop(token, None)
            return httpx.Response(200, json={"ok": True})

        self.transport = httpx.MockTransport(handler)
        self.board = Leaderboard(Path(self.temp.name), self.transport)

    def test_save_and_public_fetch_never_publish_or_send_credentials(self):
        self.board.save_profile({"username": "  Ｖinson  "})
        self.assertEqual(self.calls, [])
        public = self.board.read()
        self.assertEqual(public["profile"]["username"], "Vinson")
        self.assertEqual(public["entries"], [])
        self.assertNotIn("token", public["profile"])
        self.assertNotIn("authorization", self.calls[0].headers)
        self.assertEqual(stat.S_IMODE(self.board.path.stat().st_mode), 0o600)

    def test_publish_update_restart_and_remove_share_one_private_installation(self):
        self.board.save_profile({"username": "鲁班用户"})
        published = self.board.publish()
        self.assertEqual(published["entry"]["rank"], 1)
        token = self.calls[-1].headers["authorization"]
        self.assertRegex(token, r"^Bearer [0-9a-f]{64}$")
        self.assertNotIn(token.removeprefix("Bearer "), json.dumps(published))
        self.assertEqual(self.board.read()["total"], 1)
        reopened = Leaderboard(Path(self.temp.name), self.transport)
        reopened.save_profile({"username": "更新名称"})
        self.assertEqual(reopened.profile()["published_profile"]["username"], "鲁班用户")
        reopened.publish()
        self.assertEqual(self.calls[-1].headers["authorization"], token)
        self.assertEqual(len(self.entries), 1)
        removed = reopened.unpublish()
        self.assertEqual(self.entries, {})
        self.assertIsNone(removed["profile"]["published_profile"])
        self.assertEqual(removed["profile"]["evolution_count"], 0)
        self.assertEqual(self.calls[-1].method, "DELETE")

    def test_history_count_is_automatic_and_publish_batches_all_records(self):
        for i in range(113):
            upgrade_fixture(self.temp.name, i)
        self.board.save_profile({"username": "automatic"})
        self.assertEqual(self.board.profile()["evolution_count"], 113)
        result = self.board.publish()
        self.assertEqual(result["entry"]["evolution_count"], 113)
        bodies = [json.loads(request.content) for request in self.calls]
        self.assertEqual([len(body["upgrades"]) for body in bodies], [50, 50, 13])
        self.assertTrue(all(set(body) == {"username", "upgrades", "country_code"} for body in bodies))
        self.board.publish()
        self.assertEqual(self.board.profile()["published_profile"]["evolution_count"], 113)
        for path in (Path(self.temp.name) / "modules/app").glob("learning-*.json"):
            path.unlink()
        # A restored local history does not erase the server's accepted history.
        result = self.board.publish()
        self.assertEqual(result["profile"]["evolution_count"], 0)
        self.assertEqual(result["entry"]["evolution_count"], 113)

    def test_legacy_counts_and_snapshots_are_discarded_but_identity_preserved(self):
        old = {"username": "legacy", "evolution_count": 999, "token": "c" * 64,
               "published_profile": {"username": "legacy", "evolution_count": 999}}
        self.board.path.write_text(json.dumps(old))
        upgrade_fixture(self.temp.name)
        migrated = Leaderboard(Path(self.temp.name), self.transport)
        self.assertEqual(migrated.profile()["evolution_count"], 1)
        self.assertIsNone(migrated.profile()["published_profile"])
        self.assertEqual(migrated.data["token"], old["token"])
        self.assertEqual(json.loads(migrated.path.read_text())["protocol_version"], 2)
        self.assertNotIn("evolution_count", json.loads(migrated.path.read_text()))
        migrated.publish()
        self.assertEqual(self.calls[-1].headers["authorization"], "Bearer " + old["token"])

    def test_migration_disk_error_preserves_legacy_file(self):
        original = json.dumps({"username": "legacy", "evolution_count": 999, "token": "d" * 64})
        self.board.path.write_text(original)
        with patch("luban.leaderboard.private_json", side_effect=OSError("disk unavailable")):
            migrated = Leaderboard(Path(self.temp.name), self.transport)
        self.assertEqual(migrated.path.read_text(), original)
        self.assertIn("保存失败", migrated.profile()["count_error"])
        with self.assertRaises(RuntimeError):
            migrated.publish()

    def test_corrupt_history_blocks_reporting_but_allows_public_read_and_exit(self):
        self.board.save_profile({"username": "broken"})
        self.board.publish()
        path, _, _, _ = upgrade_fixture(self.temp.name)
        path.write_text("broken")
        before = len(self.calls)
        with self.assertRaisesRegex(RuntimeError, "无法完整核验"):
            self.board.publish()
        self.assertEqual(len(self.calls), before)
        result = self.board.read()
        self.assertIsNone(result["profile"]["evolution_count"])
        self.assertIn("无法完整核验", result["profile"]["count_error"])
        self.assertEqual(result["total"], 1)
        self.board.unpublish()
        self.assertEqual(self.calls[-1].method, "DELETE")

    def test_old_or_unidentified_public_protocol_does_not_display_self_reports(self):
        base = {"entries": [{"username": "old", "evolution_count": 999, "rank": 1}], "total": 1}
        policies = ({}, {"protocol_version": 1}, {"protocol_version": 2},
                    {"protocol_version": 2, "measurement": "self_reported"},
                    {"protocol_version": 2.0, "measurement": "accepted_branch_upgrade_records"})
        for policy in policies:
            self.board.transport = httpx.MockTransport(lambda request: httpx.Response(200, json={**base, **policy}))
            with self.subTest(policy=policy):
                result = self.board.read()
                self.assertEqual(result["entries"], [])
                self.assertEqual(result["total"], 0)
                self.assertIn("协议不兼容", result["error"])

    def test_publish_requires_explicit_v2_response_before_claiming_success(self):
        self.board.save_profile({"username": "new"})
        for version in (None, 1, "2", 2.0, True):
            reply = {"entry": {"username": "new", "evolution_count": 999, "rank": 1}}
            if version is not None:
                reply["protocol_version"] = version
            self.board.transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))
            with self.subTest(version=version), self.assertRaisesRegex(RuntimeError, "上报结果格式异常"):
                self.board.publish()
            self.assertIsNone(self.board.profile()["published_profile"])

    def test_country_preselection_runs_once_and_manual_selection_wins(self):
        self.geo_detection.return_value = "CN"
        result = self.board.read()
        self.assertEqual(result["profile"]["country_code"], "CN")
        self.board.read()
        self.assertEqual(self.geo_detection.call_count, 1)
        self.board.save_profile({"username": "country", "country_code": "GB"})
        self.board.publish()
        self.assertEqual(json.loads(self.calls[-1].content)["country_code"], "GB")
        self.assertEqual(self.board.profile()["published_profile"]["country_code"], "GB")
        reopened = Leaderboard(Path(self.temp.name), self.transport)
        reopened.read()
        self.assertEqual(self.geo_detection.call_count, 1)
        self.assertEqual(reopened.profile()["country_code"], "GB")
        # A v0.2.3 username-only save retains the selected region.
        reopened.save_profile({"username": "renamed"})
        self.assertEqual(reopened.profile()["country_code"], "GB")
        reopened.save_profile({"username": "renamed", "country_code": None})
        reopened.read()
        self.assertEqual(self.geo_detection.call_count, 1)
        self.assertIsNone(reopened.profile()["country_code"])

    def test_country_detection_failure_is_optional_and_retries_after_cooldown(self):
        with patch("luban.leaderboard.time.monotonic", return_value=100):
            self.board.read()
            self.board.read()
        self.assertEqual(self.geo_detection.call_count, 1)
        self.assertFalse(self.board.data["country_detected"])
        self.geo_detection.return_value = "CN"
        with patch("luban.leaderboard.time.monotonic", return_value=159):
            self.assertIsNone(self.board.read()["profile"]["country_code"])
        with patch("luban.leaderboard.time.monotonic", return_value=160):
            self.assertEqual(self.board.read()["profile"]["country_code"], "CN")
        self.assertEqual(self.geo_detection.call_count, 2)
        self.board.save_profile({"username": "unknown"})
        self.assertEqual(self.board.publish()["entry"]["evolution_count"], 0)
        self.assertEqual(self.board.profile()["country_code"], "CN")
        with patch("luban.leaderboard.private_json", side_effect=OSError("disk failure")):
            other = Leaderboard(Path(self.temp.name) / "optional", self.transport)
            self.assertIsNone(other.read()["error"])

    def test_legacy_failed_lookup_is_retried_but_explicit_blank_survives_restart(self):
        legacy = {key: value for key, value in self.board.data.items() if key != "country_manual"}
        self.board.path.write_text(json.dumps({**legacy, "country_detected": True,
                                              "country_code": None, "username": "legacy"}))
        self.geo_detection.return_value = "CN"
        reopened = Leaderboard(Path(self.temp.name), self.transport)
        self.assertEqual(reopened.read()["profile"]["country_code"], "CN")
        reopened.save_profile({"username": "legacy", "country_code": None})
        explicit_blank = Leaderboard(Path(self.temp.name), self.transport)
        self.assertIsNone(explicit_blank.read()["profile"]["country_code"])
        self.assertEqual(self.geo_detection.call_count, 1)

    def test_country_save_failure_does_not_disable_future_detection(self):
        self.geo_detection.return_value = "CN"
        with patch("luban.leaderboard.time.monotonic", return_value=100):
            with patch("luban.leaderboard.private_json", side_effect=OSError("disk failure")):
                self.assertIsNone(self.board.read()["profile"]["country_code"])
            self.board.read()
        self.assertEqual(self.geo_detection.call_count, 1)
        with patch("luban.leaderboard.time.monotonic", return_value=160):
            self.assertEqual(self.board.read()["profile"]["country_code"], "CN")
        self.assertEqual(self.geo_detection.call_count, 2)

    def test_country_whitelist_and_legacy_public_entries(self):
        for code in ("US", "CN", "HK", "TW", None):
            self.assertEqual(validated_profile({"username": "country", "country_code": code})["country_code"], code)
        for code in ("XX", "ZZ", "EU", "UK", "cn", " CN ", "", 1, True, [], {}):
            with self.subTest(code=code), self.assertRaises(ValueError):
                self.board.save_profile({"username": "country", "country_code": code})
        self.entries["legacy"] = {"username": "legacy", "evolution_count": 0, "rank": 1}
        self.assertIsNone(self.board.read()["entries"][0]["country_code"])

    def test_page_fetch_is_bounded_and_only_requested_page_is_fetched(self):
        rows = [{"username": f"user-{i}", "evolution_count": 101 - i, "rank": i + 1} for i in range(101)]
        pages = []
        def handler(request):
            page = int(request.url.params["page"])
            pages.append(page)
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, json={"entries": rows[(page - 1) * 100:page * 100], "total": 101,
                                            "page": page, "page_size": 100, "protocol_version": 2,
                                            "measurement": "accepted_branch_upgrade_records"})
        self.board.transport = httpx.MockTransport(handler)
        first = self.board.read()
        self.assertEqual(len(first["entries"]), 100)
        self.assertEqual(pages, [1])
        second = self.board.read(2)
        self.assertEqual([row["rank"] for row in second["entries"]], [101])
        self.assertEqual(second["page"], 2)
        self.assertEqual(second["page_size"], 100)
        self.assertEqual(pages, [1, 2])
        for invalid in (True, 0, -1, 1.0, "2", 1000000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.board.read(invalid)
        self.assertEqual(pages, [1, 2])

    def test_new_names_use_sixteen_codepoints_and_legacy_names_remain_editable(self):
        for username in ("中" * 16, "a" * 16, "😀" * 16):
            self.assertEqual(validated_profile({"username": username})["username"], username)
        for username in ("中" * 17, "a" * 17, "😀" * 17):
            with self.subTest(username=username), self.assertRaises(ValueError):
                validated_profile({"username": username})
        saved = {"protocol_version": 2, "username": "a" * 24, "token": "d" * 64, "published_profile": None}
        self.board.path.write_text(json.dumps(saved))
        reopened = Leaderboard(Path(self.temp.name), self.transport)
        self.assertIsNone(reopened.storage_error)
        self.assertEqual(reopened.profile()["username"], "a" * 24)
        reopened.save_profile({"username": "short"})
        self.assertEqual(reopened.data["token"], saved["token"])
        self.entries["legacy"] = {"username": "a" * 24, "evolution_count": 0, "rank": 1}
        self.assertEqual(reopened.read()["entries"][0]["username"], "a" * 24)

    def test_legacy_short_word_mask_expansion_never_locks_the_installation(self):
        saved = {"protocol_version": 2, "username": "x" * 24, "token": "e" * 64,
                 "published_profile": {"username": "x" * 24, "evolution_count": 1}}
        self.board.path.write_text(json.dumps(saved))
        reopened = Leaderboard(Path(self.temp.name), self.transport)
        self.assertIsNone(reopened.storage_error)
        self.assertEqual(reopened.profile()["username"], "***")
        self.assertEqual(reopened.profile()["published_profile"]["username"], "***")
        self.assertIsNone(Leaderboard(Path(self.temp.name), self.transport).storage_error)
        reopened.save_profile({"username": "normal"})
        self.assertEqual(reopened.data["token"], saved["token"])
        self.entries["legacy"] = {"username": "x" * 24, "evolution_count": 1, "rank": 1}
        self.assertEqual(reopened.read()["entries"][0]["username"], "***")

    def test_shared_sensitive_words_are_masked_before_local_save(self):
        for username, expected in (("最强玩家", "***玩家"), ("傻-逼用户", "***用户"), ("ＦＵＣＫ", "***"), ("***玩家", "***玩家")):
            with self.subTest(username=username):
                self.board.save_profile({"username": username})
                self.assertEqual(self.board.profile()["username"], expected)
                self.assertEqual(self.board.publish()["entry"]["username"], expected)

    def test_masked_name_is_saved_and_published_without_response_mismatch(self):
        with patch("luban.leaderboard.mask_name", side_effect=lambda name: name.replace("敏感词", "***")):
            self.board.save_profile({"username": "用户敏感词"})
            self.assertEqual(self.board.profile()["username"], "用户***")
            result = self.board.publish()
            self.assertEqual(result["entry"]["username"], "用户***")
            self.assertEqual(json.loads(self.calls[-1].content)["username"], "用户***")
        with patch("luban.leaderboard.mask_name", side_effect=lambda name: name.replace("词", "***")):
            with self.assertRaisesRegex(ValueError, "屏蔽后超过"):
                self.board.save_profile({"username": "词" * 6})

    def test_strict_validation_preserves_old_profile(self):
        self.board.save_profile({"username": "valid"})
        before = self.board.path.read_bytes()
        for username in ("", "   ", "x" * 17, "a\nb", "a\u200bb", 4, None):
            with self.subTest(username=username), self.assertRaises(ValueError):
                self.board.save_profile({"username": username})
        for count in (True, 0, 1, -1, 1.0, "1", 1_000_000_001, None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.board.save_profile({"username": "valid", "evolution_count": count})
        with self.assertRaises(ValueError):
            self.board.save_profile({"username": "valid", "unexpected": 1})
        self.assertEqual(self.board.path.read_bytes(), before)
        self.assertEqual(validated_profile({"username": "😀" * 16})["username"], "😀" * 16)

    def test_network_failures_keep_profile_and_public_get_returns_clear_error(self):
        self.board.save_profile({"username": "offline"})
        def fail(request):
            raise httpx.ConnectError("private request details", request=request)
        self.board.transport = httpx.MockTransport(fail)
        with self.assertRaisesRegex(RuntimeError, "本地设置已保留"):
            self.board.publish()
        profile = self.board.read()
        self.assertEqual(profile["profile"]["username"], "offline")
        self.assertIn("无法连接", profile["error"])
        self.assertNotIn("private", json.dumps(profile))
        token = self.board.data["token"]
        self.assertEqual(len(token), 64)
        self.assertNotIn(token, json.dumps(profile))
        with self.assertRaises(RuntimeError):
            self.board.unpublish()
        self.assertEqual(self.board.data["token"], token)

    def test_delete_failure_keeps_record_of_last_publication(self):
        self.board.save_profile({"username": "published"})
        self.board.publish()
        before = self.board.path.read_bytes()
        self.board.transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "secret"}))
        with self.assertRaises(RuntimeError):
            self.board.unpublish()
        self.assertEqual(self.board.path.read_bytes(), before)
        self.assertIsNotNone(self.board.profile()["published_profile"])

    def test_conflict_and_malformed_remote_reply_do_not_claim_publication(self):
        self.board.save_profile({"username": "occupied"})
        self.board.transport = httpx.MockTransport(lambda request: httpx.Response(409, json={"error": "credential leak"}))
        with self.assertRaisesRegex(RuntimeError, "已被其他安装使用"):
            self.board.publish()
        self.assertIsNone(self.board.profile()["published_profile"])
        self.board.transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"entry": {"username": "occupied", "evolution_count": 6, "rank": True}}))
        with self.assertRaisesRegex(RuntimeError, "上报结果格式异常"):
            self.board.publish()
        self.board.transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"entries": {}, "total": 0, "protocol_version": 2, "measurement": "accepted_branch_upgrade_records"}))
        self.assertIn("格式异常", self.board.read()["error"])

    def test_malformed_delete_reply_keeps_last_publication(self):
        self.board.save_profile({"username": "published"})
        self.board.publish()
        before = self.board.path.read_bytes()
        for reply in ({}, {"ok": False}, {"ok": 1}, []):
            self.board.transport = httpx.MockTransport(lambda request: httpx.Response(200, json=reply))
            with self.subTest(reply=reply), self.assertRaisesRegex(RuntimeError, "退出结果格式异常"):
                self.board.unpublish()
            self.assertEqual(self.board.path.read_bytes(), before)

    def test_private_save_failure_and_corruption_never_overwrite_old_data(self):
        self.board.save_profile({"username": "before"})
        before = self.board.path.read_bytes()
        with patch("luban.leaderboard.private_json", side_effect=OSError("broken disk")):
            with self.assertRaises(RuntimeError):
                self.board.save_profile({"username": "after"})
        self.assertEqual(self.board.path.read_bytes(), before)
        self.assertEqual(self.board.profile()["username"], "before")
        self.board.path.write_text("broken json")
        reopened = Leaderboard(Path(self.temp.name), self.transport)
        self.assertIn("损坏", reopened.read()["profile"]["count_error"])
        for action in (lambda: reopened.save_profile({"username": "after"}), reopened.publish, reopened.unpublish):
            with self.assertRaises(RuntimeError):
                action()
        self.assertEqual(self.board.path.read_text(), "broken json")


class LeaderboardAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(self.temp.name)
        self.settings.update({"storage_root": str(Path(self.temp.name) / "storage")})
        self.original_settings = self.settings.path.read_bytes()
        self.app = create_app(self.settings, SimpleNamespace(state="ready"), autoload=False)
        geo_patch = patch("luban.leaderboard.detect_country", return_value=None)
        self.geo_detection = geo_patch.start()
        self.addCleanup(geo_patch.stop)
        self.calls = []
        def handler(request):
            self.calls.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"entries": [], "total": 0, "updated_at": None, "protocol_version": 2, "measurement": "accepted_branch_upgrade_records"})
            if request.method == "PUT":
                return httpx.Response(200, json={"entry": {"username": json.loads(request.content)["username"], "evolution_count": 0, "rank": 1}, "protocol_version": 2})
            return httpx.Response(200, json={"ok": True})
        self.app.state.leaderboard.transport = httpx.MockTransport(handler)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_all_ranking_operations_require_local_session_and_block_foreign_origin(self):
        for method, path in (("GET", "/api/leaderboard"), ("PUT", "/api/leaderboard/profile"),
                             ("POST", "/api/leaderboard/publish"), ("DELETE", "/api/leaderboard/publish")):
            self.assertEqual(self.client.request(method, path).status_code, 401)
        self.client.post("/api/session", json={"token": self.settings.secret("ui_token")})
        self.assertEqual(self.client.put("/api/leaderboard/profile", headers={"origin": "https://foreign.invalid"},
                                        json={"username": "user"}).status_code, 403)
        self.assertEqual(self.calls, [])

    def test_page_api_validates_query_before_upstream_request(self):
        self.client.post("/api/session", json={"token": self.settings.secret("ui_token")})
        for raw in ("0", "-1", "1.5", "true", "1000000", "01", ""):
            with self.subTest(raw=raw):
                self.assertEqual(self.client.get("/api/leaderboard?page=" + raw).status_code, 400)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.client.get("/api/leaderboard?page=2").json()["page"], 2)
        self.assertEqual(self.calls[-1].url.params["page"], "2")

    def test_local_lifecycle_validation_and_private_response(self):
        self.client.post("/api/session", json={"token": self.settings.secret("ui_token")})
        invalid = self.client.put("/api/leaderboard/profile", json={"username": "valid", "evolution_count": True})
        self.assertEqual(invalid.status_code, 400)
        saved = self.client.put("/api/leaderboard/profile", json={"username": " local "})
        self.assertEqual(saved.json()["profile"]["username"], "local")
        self.assertEqual(self.calls, [])
        published = self.client.post("/api/leaderboard/publish", json={})
        self.assertEqual(published.status_code, 200)
        token = self.app.state.leaderboard.data["token"]
        response = self.client.get("/api/leaderboard")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn(token, response.text + published.text + saved.text)
        self.assertNotIn("authorization", self.calls[-1].headers)
        self.assertEqual(self.client.delete("/api/leaderboard/publish").status_code, 200)
        self.assertIsNone(self.app.state.leaderboard.profile()["published_profile"])
        self.assertEqual(self.settings.path.read_bytes(), self.original_settings)


class CountryDetectionTests(unittest.TestCase):
    def test_trace_only_extracts_iso_region_and_never_sends_credentials(self):
        requests = []
        def handler(request):
            requests.append(request)
            self.assertEqual(str(request.url), COUNTRY_TRACE)
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("cookie", request.headers)
            self.assertEqual(request.content, b"")
            return httpx.Response(200, text="ip=192.0.2.1\nloc=CN\ntls=TLSv1.3\n")
        self.assertEqual(detect_country(httpx.MockTransport(handler)), "CN")
        self.assertEqual(len(requests), 1)

    def test_trace_invalid_unknown_redirect_and_network_failure_return_none(self):
        for code, text in ((200, "loc=XX"), (200, "loc=EU"), (200, "loc=CN\nloc=US"),
                           (200, "ip=192.0.2.1"), (302, "loc=CN"), (500, "loc=CN"),
                           (200, "x" * 4097 + "\nloc=CN")):
            with self.subTest(code=code, text=text[:20]):
                self.assertIsNone(detect_country(httpx.MockTransport(lambda request: httpx.Response(code, text=text))))
        def fail(request):
            raise httpx.ConnectError("private trace", request=request)
        self.assertIsNone(detect_country(httpx.MockTransport(fail)))


if __name__ == "__main__":
    unittest.main()
