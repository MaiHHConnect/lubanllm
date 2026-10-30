"""Private CPU HTTP boundaries with real SQLite/ZIP; no model or live network."""
import copy
from contextlib import ExitStack
import inspect
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import httpx
from fastapi.testclient import TestClient
from starlette.requests import Request

from luban import model_packages as packages
from luban.memory import Memory
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings, private_json
from luban.teacher import Teacher
from tests.test_feeding import Tokenizer
from tests.test_model_packages import tensor


class ModelToolsHTTPTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()  # macOS /var alias is not an uploaded symlink.
        self.source = self.root / "production"
        self.settings = Settings(self.root / "private")
        trusted = {}
        for key in ("base", "asr"):
            directory = self.source / "models" / key
            directory.mkdir(parents=True)
            private_json(directory / "config.json", {"model_type": "qwen3_5" if key == "base" else "qwen3_asr"})
            tensor(directory / "model.safetensors", {"fixture.weight": [2, 2]})
            (directory / "LICENSE").write_text("CPU fixture, no real model distribution")
            trusted[key] = {path.name: packages._record(path) for path in directory.iterdir()}
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(packages, "TRUSTED", trusted))
        self.network = self.patches.enter_context(patch("urllib.request.urlopen", side_effect=AssertionError("no live network")))
        self.live_http = self.patches.enter_context(patch("httpx.HTTPTransport.handle_request", side_effect=AssertionError("no live HTTP")))
        self.settings.update({"storage_root": str(self.source), "model_path": str(self.source / "models/base"),
            "speech_model_path": str(self.source / "models/asr"), "native_inference_precision": "bf16", "teacher": {
                "base_url": "http://offline.invalid/v1", "api_key": "fixture-only-secret", "model": "fixture"}})
        self.settings.accept_agreement()
        self.memory = Memory(self.source)
        self.memory.model_identity = packages.TRUSTED_IDENTITY
        self.memory.add_sample("private pending user knowledge", "private gold", approved=True, training_allowed=True)
        with self.memory.connect() as db:
            db.execute("INSERT INTO conversations VALUES (?,?,?,?)", ("secret-chat", "DO NOT EXPORT CHAT", "[]", 1))
        store = self.source / "modules/luban0-v1"; store.mkdir(parents=True)
        self.runtime = Runtime.__new__(Runtime)
        self.runtime.__dict__.update(settings=self.settings, memory=self.memory, identity=packages.TRUSTED_IDENTITY,
            active_core=None, store=store, state="ready", job=None, training_session=None, lock=threading.RLock(),
            tokenizer=Tokenizer(), device="cpu", model=None, branch=None, router=None, decoder=None,
            cold_backbone=None, _model_process_lock=None, last_generation_usage=None,
            _last_generation_usage=None, status=lambda: {"state": "ready"})
        self.runtime._release_model = Mock(spec=Runtime._release_model, wraps=self.runtime._release_model)
        self.runtime.load = Mock(spec=Runtime.load, side_effect=self.fixture_load)
        self.load_observations = []
        self.runtime.teacher_usage = lambda *args: Runtime.teacher_usage(self.runtime, *args)
        self.requests = []
        self.teacher = Teacher(self.settings, httpx.MockTransport(self.respond))
        self.speech = SimpleNamespace(model=None, model_path=Path(self.settings.snapshot()["speech_model_path"]),
                                      status=lambda: {"state": "idle"})
        self.app = create_app(self.settings, self.runtime, self.teacher, autoload=False, speech=self.speech)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/session", json={"token": self.settings.secret("ui_token")}).raise_for_status()
        # Inspect the single actual route's shared task only to observe streaming order.
        route = next(route for route in self.app.routes if getattr(route, "path", None) == "/api/model-packages/import")
        self.task = inspect.getclosurevars(route.endpoint).nonlocals["task"]
        self.app.state.automatic.pause = Mock(wraps=self.app.state.automatic.pause)

    def fixture_load(self):
        # Read real tiny tensors, but never instantiate or execute a language model.
        from safetensors.torch import load_file
        config = self.settings.snapshot()
        self.load_observations.append({key: config[key] for key in packages._PATH_KEYS})
        tensors = load_file(str(Path(config["model_path"]) / "model.safetensors"), device="cpu")
        self.runtime.model = SimpleNamespace(tensors=tensors)
        self.runtime.tokenizer = Tokenizer()
        self.runtime.memory = Memory(config["storage_root"])
        self.runtime.memory.model_identity = packages.TRUSTED_IDENTITY
        self.runtime.store = Path(config["storage_root"]) / "modules/luban0-v1"
        self.runtime.state = "ready"

    def package_bytes(self):
        target = self.root / "tiny-valid-import.zip"
        packages.export_package(self.runtime, target)
        return target.read_bytes()

    def respond(self, request):
        body = json.loads(request.content); self.requests.append(body)
        source = json.loads(body["messages"][1]["content"])["source"]
        answer = [{"source_id": source["id"], "question": "名称是什么？", "answer": "鲁班零号", "quote": source["text"]}]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer, ensure_ascii=False)},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})

    def task_status(self):
        response = self.client.get("/api/model-packages"); response.raise_for_status()
        return response.json()["task"]

    def wait_task(self, state="complete"):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            value = self.task_status()
            if value["state"] == state: return value
            if value["state"] == "error" and state != "error": self.fail(value)
            time.sleep(.005)
        self.fail("package fixture did not finish: " + str(self.task_status()))

    def scan(self):
        response = self.client.post("/api/feeding/scan", json={"files": [{"name": "notes.txt", "content": "名称：鲁班零号。"}]})
        response.raise_for_status(); return response.json()

    def test_authentication_precedes_body_read_validation_consents_and_pause(self):
        self.client.cookies.clear()
        paths = ("/api/model-packages/import", "/api/model-packages/export", "/api/model-packages/reset",
                 "/api/feeding/scan", "/api/feeding/clean", "/api/feeding/confirm")
        # ASGI may drain a rejected request to finish sending its response;
        # application's JSON/body validation must still follow authentication.
        with patch.object(Request, "body", side_effect=AssertionError("unauth body must not be buffered")), \
                patch.object(Request, "json", side_effect=AssertionError("unauth JSON must not be parsed")):
            for path in paths:
                with self.subTest(path=path):
                    response = self.client.post(path, content=b"not-json", headers={"content-type": "application/zip", "content-length": "999999999999"})
                    self.assertEqual(response.status_code, 401, response.text)
        self.app.state.automatic.pause.assert_not_called()
        self.assertEqual(self.task, {"state": "idle"})
        self.assertEqual(self.requests, [])
        self.runtime._release_model.assert_not_called(); self.runtime.load.assert_not_called()

    def test_first_install_rebinds_memory_and_speech_without_starting_model(self):
        self.runtime.state = 'stopped'
        destination = self.root / 'first-install'
        speech_path = destination / 'models/asr'
        def install(settings, progress):
            speech_path.mkdir(parents=True)
            memory = Memory(destination)
            memory.add_sample('installed factory material', 'preserved answer', approved=True, training_allowed=True)
            settings.update(dict(storage_root=str(destination), speech_model_path=str(speech_path)))
            progress(dict(phase='complete'))
            return dict(installed=True, root=str(destination))
        with patch('luban.package_api.install_models', side_effect=install) as installer:
            response = self.client.post('/api/model-packages/install')
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIs(response.json()['started'], True)
            result = self.wait_task()
        installer.assert_called_once()
        self.assertIs(result['result']['installed'], True)
        self.assertIsNot(self.runtime.memory, self.memory)
        self.assertEqual(self.runtime.memory.path, destination / 'app/memory.sqlite3')
        self.assertEqual([row['prompt'] for row in self.runtime.memory.samples()], ['installed factory material'])
        self.assertEqual(self.speech.model_path, speech_path)
        self.assertEqual(self.runtime.state, 'stopped')
        self.runtime.load.assert_not_called()
        self.runtime._release_model.assert_not_called()

    def test_fee_consent_and_answer_truth_consent_are_distinct_exact_booleans(self):
        baseline = len(self.memory.samples())
        batch = self.scan()
        for value in (None, False, "true", 1):
            body = {"id": batch["id"], "confirmed_answers": True}
            if value is not None: body["confirmed_api_use"] = value
            response = self.client.post("/api/feeding/clean", json=body)
            self.assertEqual(response.status_code, 400)
        self.app.state.automatic.pause.assert_not_called(); self.assertEqual(self.teacher.calls, 0)
        cleaned = self.client.post("/api/feeding/clean", json={"id": batch["id"], "confirmed_api_use": True})
        self.assertEqual(cleaned.status_code, 200, cleaned.text)
        self.assertEqual(cleaned.json()["counts"], {"pending": 1, "confirmed": 0})
        self.assertEqual(self.teacher.calls, 1); self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.memory.samples()), baseline)
        pauses = self.app.state.automatic.pause.call_count
        for value in (None, False, "true", 1):
            body = {"id": batch["id"], "confirmed_api_use": True}
            if value is not None: body["confirmed_answers"] = value
            self.assertEqual(self.client.post("/api/feeding/confirm", json=body).status_code, 400)
        self.assertEqual(self.app.state.automatic.pause.call_count, pauses)
        confirmed = self.client.post("/api/feeding/confirm", json={"id": batch["id"], "confirmed_answers": True})
        self.assertEqual(confirmed.status_code, 200, confirmed.text)
        self.assertEqual(confirmed.json()["added"], 1)
        row = next(row for row in self.memory.samples() if row["source"] == "teacher")
        value = json.loads(row["provenance"])
        self.assertIs(value["human_verified"], True); self.assertTrue(value["verification_evidence"])
        self.assertEqual(self.teacher.calls, 1)
        again = self.client.post("/api/feeding/confirm", json={"id": batch["id"], "confirmed_answers": True})
        self.assertEqual(again.json()["already_confirmed"], 1)
        self.assertEqual(len(self.memory.samples()), baseline + 1)
        self.runtime.load.assert_not_called(); self.runtime._release_model.assert_not_called()

    def test_import_is_non_json_stream_not_body_buffer_and_staging_gets_actual_bytes(self):
        payload = b"PK\x03\x04" + b"\xff" * 200100  # Larger than normal JSON limit, deliberately not JSON.
        stages, stream_states = [], []
        original = Request.stream
        async def checked_stream(request):
            if request.url.path == "/api/model-packages/import":
                stream_states.append(self.task["state"])
                self.assertEqual(self.task["state"], "uploading")
            async for chunk in original(request): yield chunk
        def stage(path, parent, **kwargs):
            stages.append((Path(path).read_bytes(), Path(path)))
            raise ValueError("fixture refuses opaque package after streamed upload")
        with patch.object(Request, "body", side_effect=AssertionError("ZIP must not use Request.body")), \
                patch.object(Request, "stream", checked_stream), patch("luban.package_api.stage_import", side_effect=stage):
            response = self.client.post("/api/model-packages/import", content=payload, headers={"content-type": "application/zip"})
            self.assertEqual(response.status_code, 200, response.text)
            status = self.wait_task("error")
        self.assertTrue(stream_states)
        self.assertEqual(stages[0][0], payload)
        self.assertFalse(stages[0][1].exists())
        self.assertEqual(status["bytes"], len(payload))
        self.runtime._release_model.assert_not_called(); self.runtime.load.assert_not_called()

    def test_import_empty_and_content_length_mismatch_clean_upload_without_stage(self):
        for payload, declared in ((b"", "0"), (b"PKsmall", "99")):
            with self.subTest(payload=payload), patch("luban.package_api.stage_import") as stage:
                response = self.client.post("/api/model-packages/import", content=payload,
                                            headers={"content-type": "application/zip", "content-length": declared})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(self.task_status()["state"], "error")
                self.assertEqual(list((self.source / "app/model-packages").glob("upload-*.zip")), [])
                stage.assert_not_called()
        self.runtime._release_model.assert_not_called(); self.runtime.load.assert_not_called()

    def test_wrong_import_type_and_declared_limit_reject_before_pause_or_upload(self):
        for headers, expected in (({"content-type": "application/json"}, 400),
                                  ({"content-type": "application/zip", "content-length": "999999999999"}, 413)):
            response = self.client.post("/api/model-packages/import", content=b"{}", headers=headers)
            self.assertEqual(response.status_code, expected, response.text)
        self.app.state.automatic.pause.assert_not_called()
        self.assertEqual(self.task, {"state": "idle"})
        self.assertFalse((self.source / "app/model-packages").exists())

    def test_existing_running_task_failed_import_does_not_replace_status(self):
        started, finish = threading.Event(), threading.Event()
        self.addCleanup(finish.set)
        def export(runtime, target, progress=None):
            progress({"phase": "fixture-export", "bytes": 7, "total": 20})
            started.set()
            if not finish.wait(5): raise RuntimeError("fixture worker timeout")
            Path(target).write_bytes(b"fixture")
            return {"path": str(target), "package_id": "fixture", "bytes": 7}
        with patch("luban.package_api.export_package", side_effect=export):
            response = self.client.post("/api/model-packages/export")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(started.wait(3))
            before = copy.deepcopy(self.task_status()); self.assertEqual(before["state"], "running")
            rejected = self.client.post("/api/model-packages/import", content=b"", headers={"content-type": "application/zip"})
            self.assertEqual(rejected.status_code, 409, rejected.text)
            self.assertEqual(self.task_status(), before)
            self.assertEqual(list((self.source / "app/model-packages").glob("upload-*.zip")), [])
            finish.set(); self.wait_task()

    def test_unpublished_reset_exact_phrase_never_network_pause_release_or_change_data(self):
        settings_before, records_before = self.settings.snapshot(), self.memory.samples()
        db_before = packages._sha(self.memory.path)
        with patch("luban.package_api.FOUNDATION_SHA256", None), patch("luban.package_api.stage_reset") as stage:
            for phrase in ("我同意重修 ", "wrong", packages.RESET_PHRASE):
                response = self.client.post("/api/model-packages/reset", json={"phrase": phrase})
                self.assertEqual(response.status_code, 400, response.text)
            stage.assert_not_called()
        self.app.state.automatic.pause.assert_not_called(); self.runtime._release_model.assert_not_called()
        self.runtime.load.assert_not_called(); self.network.assert_not_called(); self.live_http.assert_not_called()
        self.assertEqual(self.settings.snapshot(), settings_before)
        self.assertEqual(self.memory.samples(), records_before)
        self.assertEqual(packages._sha(self.memory.path), db_before)
        self.assertEqual(self.task, {"state": "idle"})

    def test_real_small_export_download_file_response_hides_private_paths_and_data(self):
        response = self.client.post("/api/model-packages/export")
        self.assertEqual(response.status_code, 200, response.text)
        status = self.wait_task()
        self.assertNotIn("path", status["result"])
        self.assertNotIn(str(self.root), json.dumps(status))
        url = status["result"]["download"]
        # No read_bytes of the archive: FileResponse serves via opened chunk stream.
        with patch.object(Path, "read_bytes", side_effect=AssertionError("download must not buffer archive")):
            with self.client.stream("GET", url) as download:
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.headers["content-type"], "application/zip")
                self.assertNotIn(str(self.root), json.dumps(dict(download.headers)))
                payload = b"".join(download.iter_bytes())
        self.assertEqual(len(payload), status["result"]["bytes"])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            accepted = archive.read("accepted.json").decode()
            self.assertNotIn("private pending user knowledge", accepted)
            self.assertNotIn("DO NOT EXPORT CHAT", accepted)
            for name in archive.namelist():
                self.assertNotIn("fixture-only-secret", archive.read(name).decode("latin1"))
        self.assertEqual(self.client.get("/api/model-packages/download/not-current").status_code, 400)
        self.runtime._release_model.assert_not_called(); self.runtime.load.assert_not_called()
        self.network.assert_not_called(); self.live_http.assert_not_called()

    def test_prepared_activation_recovers_real_settings_journal_before_loader_start(self):
        old = {key: self.settings.snapshot()[key] for key in packages._PATH_KEYS}
        alternate = self.root / "interrupted-new"
        alternate.mkdir()
        new = {"storage_root": str(alternate), "model_path": str(alternate / "models/base"),
               "speech_model_path": str(alternate / "models/asr")}
        self.settings.update(new)
        journal = self.settings.home / "package-activation.json"
        private_json(journal, {"schema": packages.SCHEMA, "state": "prepared", "old": old, "new": new})
        loaded = threading.Event(); seen = []
        def load():
            seen.append(({key: self.settings.snapshot()[key] for key in packages._PATH_KEYS}, json.loads(journal.read_text())["state"]))
            self.runtime.state = "ready"; loaded.set()
        self.runtime.load = Mock(side_effect=load)
        with patch("luban.automatic.AutomaticLearning.start") as start:
            app = create_app(self.settings, self.runtime, self.teacher, autoload=True, speech=self.speech)
            with TestClient(app, raise_server_exceptions=False):
                self.assertTrue(loaded.wait(3))
                start.assert_called_once()
        self.assertEqual(seen, [(old, "rolled_back")])
        self.assertEqual({key: self.settings.snapshot()[key] for key in packages._PATH_KEYS}, old)
        self.assertEqual(json.loads(journal.read_text())["state"], "rolled_back")
        self.runtime._release_model.assert_not_called(); self.network.assert_not_called()

    def test_real_small_zip_import_activates_only_after_new_model_load(self):
        old, records = self.settings.snapshot(), self.memory.samples()
        old_tensor = packages._sha(Path(old["model_path"]) / "model.safetensors")
        payload = self.package_bytes()
        response = self.client.post("/api/model-packages/import", content=payload,
                                    headers={"content-type": "application/zip"})
        self.assertEqual(response.status_code, 200, response.text)
        finished = self.wait_task()
        self.assertTrue(finished["result"]["activated"])
        current = self.settings.snapshot()
        self.assertNotEqual(current["storage_root"], old["storage_root"])
        self.assertEqual(self.load_observations, [{key: current[key] for key in packages._PATH_KEYS}])
        self.assertEqual(self.runtime._release_model.call_count, 1)
        self.assertEqual(self.runtime.state, "ready")
        journal = json.loads((self.settings.home / "package-activation.json").read_text())
        self.assertEqual(journal["state"], "committed")
        self.assertEqual(journal["old"], {key: old[key] for key in
                         (*packages._PATH_KEYS, "native_inference_precision", "native_mlx_model_path")})
        self.assertEqual(self.memory.samples(), records)
        self.assertEqual(packages._sha(Path(old["model_path"]) / "model.safetensors"), old_tensor)
        self.assertTrue((Path(current["storage_root"]) / "package-owner.json").is_file())
        self.assertEqual(list((self.source / "app/model-packages").glob("upload-*.zip")), [])
        self.network.assert_not_called(); self.live_http.assert_not_called()

    def test_new_load_failure_recovers_real_prepared_journal_and_old_model_paths(self):
        old, records = self.settings.snapshot(), self.memory.samples()
        old_paths = {key: old[key] for key in packages._PATH_KEYS}
        old_tensor = packages._sha(Path(old["model_path"]) / "model.safetensors")
        payload = self.package_bytes()
        attempts = []
        def fail_new():
            paths = {key: self.settings.snapshot()[key] for key in packages._PATH_KEYS}
            attempts.append(paths)
            if paths != old_paths:
                journal = json.loads((self.settings.home / "package-activation.json").read_text())
                self.assertEqual(journal["state"], "prepared")
                raise ValueError("fixture new-model computation fails")
            self.fixture_load()
        self.runtime.load.side_effect = fail_new
        response = self.client.post("/api/model-packages/import", content=payload,
                                    headers={"content-type": "application/zip"})
        self.assertEqual(response.status_code, 200, response.text)
        failed = self.wait_task("error")
        self.assertIn("fixture new-model", failed["error"])
        self.assertEqual(len(attempts), 2); self.assertNotEqual(attempts[0], old_paths)
        self.assertEqual(attempts[1], old_paths)
        self.assertEqual(self.settings.snapshot(), old)
        self.assertEqual(self.runtime._release_model.call_count, 2)
        self.assertEqual(self.runtime.state, "ready")
        self.assertEqual(self.load_observations, [old_paths])
        self.assertEqual(json.loads((self.settings.home / "package-activation.json").read_text())["state"], "rolled_back")
        self.assertEqual(self.runtime.memory.samples(), records)
        self.assertEqual(packages._sha(Path(old["model_path"]) / "model.safetensors"), old_tensor)
        self.assertEqual(list((self.source / "app/model-packages").glob("upload-*.zip")), [])

    def test_actual_prepare_rejection_restores_old_model_without_settings_mutation(self):
        old, records = self.settings.snapshot(), self.memory.samples()
        payload = self.package_bytes()
        def invalid_prepare(settings, staged):
            # Actual prepare validates the staged ownership record before any switch.
            invalid = {**staged, "journal_sha256": "0" * 64}
            return packages.prepare_activation(settings, invalid)
        with patch("luban.package_api.prepare_activation", side_effect=invalid_prepare):
            response = self.client.post("/api/model-packages/import", content=payload,
                                        headers={"content-type": "application/zip"})
            self.assertEqual(response.status_code, 200, response.text)
            failed = self.wait_task("error")
        self.assertIn("未经完整验证", failed["error"])
        self.assertEqual(self.settings.snapshot(), old)
        self.assertEqual(self.runtime._release_model.call_count, 2)
        self.runtime.load.assert_called_once()
        self.assertEqual(self.load_observations, [{key: old[key] for key in packages._PATH_KEYS}])
        self.assertEqual(self.runtime.state, "ready")
        self.assertEqual(self.runtime.memory.samples(), records)
        self.assertFalse((self.settings.home / "package-activation.json").exists())

    def test_committed_reset_cleanup_failure_reports_enabled_model_without_rollback(self):
        old = self.settings.snapshot()
        payload = self.package_bytes()
        foundation = self.root / "fixture-foundation.zip"
        with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(foundation, "w") as target:
            for info in source.infolist():
                content = source.read(info)
                if info.filename == "manifest.json":
                    manifest = json.loads(content)
                    manifest.pop("package_id"); manifest["kind"] = "foundation"
                    manifest["package_id"] = hashlib.sha256(packages._encoded(manifest)).hexdigest()
                    content = packages._encoded(manifest)
                target.writestr(info, content)
        digest = packages._sha(foundation)
        def staged_without_network(phrase, parent, progress=None):
            self.assertEqual(phrase, packages.RESET_PHRASE)
            staged = packages.stage_import(foundation, parent, progress=progress, require_foundation=True)
            staged["activation_intent"] = "reset"
            owner_path = Path(staged["root"]) / "package-owner.json"
            owner = json.loads(owner_path.read_text()); owner["result"] = staged
            private_json(owner_path, owner)
            return staged
        with patch("luban.package_api.FOUNDATION_SHA256", digest), patch.object(packages, "FOUNDATION_SHA256", digest), \
                patch("luban.package_api.stage_reset", side_effect=staged_without_network), \
                patch("luban.package_api.cleanup_retired_package", side_effect=OSError("fixture cleanup fails")):
            response = self.client.post("/api/model-packages/reset", json={"phrase": packages.RESET_PHRASE})
            self.assertEqual(response.status_code, 200, response.text)
            completed = self.wait_task()
        self.assertTrue(completed["result"]["activated"])
        self.assertFalse(completed["result"]["cleanup"]["removed"])
        self.assertIn("已启用", completed["result"]["cleanup"]["reason"])
        self.assertNotEqual(self.settings.snapshot()["storage_root"], old["storage_root"])
        self.assertEqual(json.loads((self.settings.home / "package-activation.json").read_text())["state"], "committed")
        self.assertEqual(self.runtime._release_model.call_count, 1); self.runtime.load.assert_called_once()
        self.assertTrue(Path(old["model_path"]).is_dir())
        self.network.assert_not_called(); self.live_http.assert_not_called()


if __name__ == "__main__": unittest.main()
