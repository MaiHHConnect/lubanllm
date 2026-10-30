"""Loopback application UI and bearer-protected compatible inference API."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import queue
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .memory import Memory
from .runtime import Runtime
from .settings import Settings, private_json, integer
from .teacher import Teacher
from .teacher_practice import TeacherPractice
from .tool_protocol import (validate_tools, validate_tool_options, normalize_tool_messages,
                            ToolOutputBuffer, prepare_tool_prompt)
from .thinking import ThinkingStream
from .review import evolution_status, review_candidates
from .leaderboard import Leaderboard
from .speech import SpeechRuntime, MAX_AUDIO_BYTES
from .evolution import restore_core
from .automatic import AutomaticLearning
from .benchmark_service import BenchmarkService
from .growth_service import GrowthService
from .package_api import install_package_routes, UPLOAD_LIMIT
from .model_packages import recover_activation
from .feeding import FeedingService
from .media_learning import MediaLearning
from .external_brain import ExternalBrain, MAX_CONTEXT_BYTES
from .vision import extract_document, MAX_DOCUMENT_BYTES, MAX_TEXT_CHARS
from .documents import extract_source
from .chat_context import compact_messages

SSE_HEARTBEAT_SECONDS = 15
from .native_evolution_lease import NativeEvolutionLease
from .learning_history import latest_learning_attempt


def validated_messages(body):
    if not isinstance(body, dict):
        raise ValueError("请求需要 JSON 对象")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages 需要非空对话消息数组")
    media_keys = {"attachments", "image", "images", "image_url", "input_image", "audio", "audio_url",
                  "input_audio", "video", "video_url", "input_video", "file", "files", "input_file"}
    if media_keys.intersection(body) or body.get("modalities", ["text"]) != ["text"]:
        raise ValueError("当前对话接口仅支持文字，不接收图片或音频")
    messages = normalize_tool_messages(messages)
    result, text_length = [], 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("消息 role 应为 system、user 或 assistant")
        if media_keys.intersection(message):
            raise ValueError("当前对话接口仅支持文字，不接收图片或音频")
        content = message.get("content")
        if isinstance(content, list):
            if not 1 <= len(content) <= 32:
                raise ValueError("消息内容需要 1–32 个文本部分")
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("消息内容部分需要是对象")
                if part.get("type") == "text" and isinstance(part.get("text"), str) and not media_keys.intersection(part):
                    text_length += len(part["text"])
                    parts.append(part["text"])
                else:
                    raise ValueError("当前对话接口仅支持文字，不接收图片或音频")
            content = "".join(parts)
        elif isinstance(content, str):
            text_length += len(content)
        empty_allowed = message["role"] == "tool" or bool(message.get("tool_calls"))
        if not (isinstance(content, str) and (content.strip() or empty_allowed)):
            raise ValueError("消息 content 需要非空文本")
        item = {"role": message["role"], "content": content}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message:
                item[key] = message[key]
        text_length += len(json.dumps(message.get("tool_calls", []), ensure_ascii=False))
        result.append(item)
    if text_length > MAX_TEXT_CHARS:
        raise ValueError("对话文本过长")
    return result


def create_app(settings=None, runtime=None, teacher=None, port=8765, autoload=True, speech=None):
    from .bun_bridge import enabled as bun_enabled
    private_compute = bun_enabled()
    compute_token = os.environ.get("LUBAN_COMPUTE_TOKEN", "") if private_compute else ""
    if private_compute and (not compute_token or not compute_token.isascii()
                            or any(not 33 <= ord(c) <= 126 for c in compute_token)):
        raise RuntimeError("Private compute token is invalid")
    public_port = int(os.environ.get("LUBAN_PUBLIC_PORT", str(port))) if private_compute else port
    settings = settings or Settings()
    recover_activation(settings)
    runtime = runtime or Runtime(settings, Memory(settings.snapshot()["storage_root"]))
    teacher = teacher or Teacher(settings)
    speech = speech or SpeechRuntime(settings.snapshot()["speech_model_path"])
    def cumulative_tokens():
        usage = runtime.memory.usage()
        totals = [usage.get(route, {}).get("total_tokens", 0) for route in ("local", "teacher")]
        return None if any(value is None for value in totals) else sum(totals)
    leaderboard = Leaderboard(settings.home, storage_root=lambda: settings.snapshot()["storage_root"],
                              token_provider=cumulative_tokens)
    requests = {}
    tasks_lock = threading.Lock()
    command_lock = threading.Lock()
    update_until = 0.0
    package_task = {"state": "idle"}
    feeding = FeedingService(runtime, settings, teacher)
    media_learning = MediaLearning(runtime, settings, teacher)
    def package_busy():
        return package_task["state"] in {"queued", "uploading", "running"}
    automatic = AutomaticLearning(runtime, settings, command_lock,
                                  lambda: bool(requests) or time.monotonic() < update_until or package_busy())
    if private_compute and os.environ.get("LUBAN_NATIVE_EVOLUTION") == "1":
        from .bun_bridge import Bridge
        evolution_bridge = Bridge()
        automatic.native_evolution = lambda method, payload: evolution_bridge.call(
            "native_evolution", method, kwargs=payload, root=str(runtime.memory.root),
            store=str(runtime.store), identity=runtime.identity)
    benchmark = BenchmarkService(runtime, leaderboard, settings.home, command_lock,
        lambda: bool(requests) or time.monotonic() < update_until or package_busy() or automatic.close_status()["running"]
        or bool(runtime.job and runtime.job.get("state") in {"queued", "running"}))
    growth = GrowthService(runtime, leaderboard, command_lock,
        lambda: bool(requests) or time.monotonic() < update_until or package_busy() or automatic.close_status()["running"]
        or bool(runtime.job and runtime.job.get("state") in {"queued", "running"}))
    leaderboard.evaluation_provider = lambda: {**benchmark.profile(), **growth.profile()}
    leaderboard.growth_provider = growth.publication
    practice = TeacherPractice(runtime, settings, teacher)
    practice.media_learning = media_learning
    automatic.teacher_practice = practice
    automatic.before_training = growth.before_training
    automatic.on_neural_upgrade = growth.after_upgrade
    automatic.on_idle = growth.idle
    automatic.on_pause = growth.pause
    version_path = Path(__file__).resolve().parents[2] / "release.json"
    metadata = json.loads(version_path.read_text()) if version_path.exists() else {"version": "0.1.0", "build": 1}

    def ensure_not_updating():
        if package_busy():
            raise RuntimeError("正在处理模型包，请完成后再操作")
        if time.monotonic() < update_until:
            raise RuntimeError("客户端正在准备更新，请稍后重试")

    def load():
        for attempt in range(3):
            try:
                runtime.load()
                return
            except RuntimeError as error:
                # Bun starts the public and private listeners immediately before
                # spawning Python. A short startup race must not permanently mark
                # the model as failed; persistent errors still remain visible.
                if "Private Bun RPC unavailable" not in str(error) or attempt == 2:
                    break
                time.sleep(0.5 * (attempt + 1))
            except Exception:
                break
        # Runtime exposes a useful error; keys and prompt bodies are not logged.

    def native_evolution_idle():
        ensure_not_updating()
        if requests or automatic.close_status()["running"]:
            raise RuntimeError("模型正在运行当前任务，请稍后重试")

    native_evolution_lease = NativeEvolutionLease(runtime, command_lock, native_evolution_idle)

    @asynccontextmanager
    async def lifespan(app):
        if autoload:
            runtime.state = "loading"
            threading.Thread(target=load, daemon=True, name="luban-loader").start()
        # Watching idle work is independent of opt-in model loading; tick gates on ready.
        automatic.start()
        if not private_compute:
            private_json(settings.home / "launch.json", {"port": port, "pid": os.getpid(),
                                                        "token": settings.secret("ui_token")})
        yield
        for event in list(requests.values()):
            event.set()
        await asyncio.to_thread(benchmark.close)
        await asyncio.to_thread(growth.close)
        await asyncio.to_thread(automatic.stop)
        await asyncio.to_thread(media_learning.close)
        if hasattr(runtime, "close_native"):
            await asyncio.to_thread(runtime.close_native)

    app = FastAPI(title="Luban Local", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings, app.state.runtime, app.state.teacher = settings, runtime, teacher
    app.state.leaderboard = leaderboard
    app.state.speech = speech
    app.state.automatic = automatic
    app.state.benchmark = benchmark
    app.state.growth = growth
    app.state.feeding = feeding
    app.state.media_learning = media_learning
    app.state.native_evolution_lease = native_evolution_lease

    if private_compute:
        @app.post("/internal/native-evolution/acquire")
        async def native_evolution_acquire(request: Request):
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"lease_id"}:
                raise ValueError("原生进化租约请求无效")
            return await asyncio.to_thread(native_evolution_lease.acquire, body["lease_id"])

        @app.post("/internal/native-evolution/release")
        async def native_evolution_release(request: Request):
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"lease_id"}:
                raise ValueError("原生进化租约请求无效")
            return await asyncio.to_thread(native_evolution_lease.release, body["lease_id"])

        @app.post("/internal/native-evolution/mlx-routes")
        async def native_mlx_routes(request: Request):
            from .mlx_learning import check_routes
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"lease_id", "records"}:
                raise ValueError("MLX学习路由请求无效")
            return await asyncio.to_thread(native_evolution_lease.prepare, body["lease_id"],
                                           lambda: check_routes(runtime, body["records"]))

        @app.get("/internal/data-scope")
        def data_scope():
            return {"root": str(runtime.memory.root), "identity": runtime.memory.model_identity,
                    "store": str(runtime.store)}

        @app.get("/internal/home-profile")
        def home_profile():
            return {"root": str(runtime.memory.root), "identity": runtime.memory.model_identity,
                    "growth_profile": growth.profile()["growth_profile"]}

        @app.get("/internal/home-evolution")
        def home_evolution():
            from .upgrades import read_upgrade_records
            identity = runtime.memory.model_identity
            records = read_upgrade_records(runtime.memory.root)
            return {"identity": identity, "root": str(runtime.memory.root),
                    "evolution_count": sum(row.get("kind") == "core" and row["base_sha256"] == identity
                                           for row in records)}

    def learning_toggle(body):
        return (isinstance(body, dict) and set(body) == {"automatic"}
                and isinstance(body["automatic"], dict) and set(body["automatic"]) == {"learning_enabled"}
                and type(body["automatic"]["learning_enabled"]) is bool)

    @app.middleware("http")
    async def boundary(request, call_next):
        if private_compute:
            peer = request.client.host if request.client else ""
            supplied = request.headers.get("authorization", "")
            if (peer not in {"127.0.0.1", "::1"} or not supplied.isascii()
                    or not secrets.compare_digest(supplied, "Bearer " + compute_token)):
                return JSONResponse({"error": {"message": "Invalid private compute key"}}, status_code=401)
        host = request.url.hostname
        if host not in {"127.0.0.1", "localhost", "::1", "testserver"}:
            return JSONResponse({"error": {"message": "Host is not allowed"}}, status_code=403)
        origin = request.headers.get("origin")
        allowed_origins = {f"http://127.0.0.1:{public_port}", f"http://localhost:{public_port}", "http://testserver"}
        if request.url.path.startswith("/api/") and origin and origin not in allowed_origins:
            return JSONResponse({"error": {"message": "Origin is not allowed"}}, status_code=403)
        if not private_compute and request.url.path.startswith("/api/") and request.url.path not in {"/api/session", "/api/health"}:
            cookie = request.cookies.get("luban_session", "")
            if not cookie or not cookie.isascii() or not secrets.compare_digest(cookie, settings.secret("ui_token")):
                return JSONResponse({"error": {"message": "请从鲁班应用启动页面"}}, status_code=401)
        if not private_compute and request.url.path.startswith("/v1/"):
            expected = "Bearer " + settings.secret("access_key")
            supplied = request.headers.get("authorization", "")
            if not supplied.isascii() or not secrets.compare_digest(supplied, expected):
                return JSONResponse({"error": {"message": "Invalid local API key", "type": "authentication_error"}}, status_code=401)
        decoded = None
        if request.method in {"POST", "PATCH", "PUT"}:
            audio_request = request.url.path in {"/api/audio/transcriptions", "/v1/audio/transcriptions"}
            chat_request = request.url.path in {"/api/chat", "/v1/chat/completions"}
            document_request = request.url.path in {"/api/chat/attachments/extract", "/api/feeding/extract"}
            package_upload = request.url.path == "/api/model-packages/import"
            feed_scan = request.url.path == "/api/feeding/scan"
            media_upload = request.url.path == "/api/learning/media"
            body_limit = UPLOAD_LIMIT if package_upload else 12 * 1024 * 1024 if media_upload else 2 * 1024 * 1024 if feed_scan else MAX_DOCUMENT_BYTES if document_request else MAX_AUDIO_BYTES if audio_request else 8 * 1024 * 1024 if chat_request else 200000
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > body_limit):
                return JSONResponse({"error": {"message": "Request body too large"}}, status_code=413)
            if not package_upload:
                raw = bytearray()
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > body_limit:
                        return JSONResponse({"error": {"message": "Request body too large"}}, status_code=413)
                request._body = bytes(raw)
                if raw and not (audio_request or document_request):
                    try:
                        decoded = json.loads(raw)
                    except (ValueError, UnicodeError):
                        return JSONResponse({"error": {"message": "Invalid JSON"}}, status_code=400)
                    if not isinstance(decoded, dict):
                        return JSONResponse({"error": {"message": "请求需要 JSON 对象"}}, status_code=400)
        if (native_evolution_lease.active and request.method in {"POST", "PATCH", "PUT", "DELETE"}
                and request.url.path not in {"/internal/native-evolution/acquire", "/internal/native-evolution/release",
                                             "/internal/native-evolution/mlx-routes"}
                and not (request.method == "PATCH" and request.url.path == "/api/settings" and learning_toggle(decoded))):
            return JSONResponse({"error": {"message": "原生进化正在占用模型，请先完成或取消进化"}}, status_code=409)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: https://ai.maihh.net; connect-src 'self' https://ai.maihh.net; frame-ancestors 'none'"
        return response

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    async def invalid(request, error):
        return JSONResponse({"error": {"message": str(error)}}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request, error):
        return JSONResponse({"error": {"message": str(error)}}, status_code=409)

    @app.exception_handler(httpx.RequestError)
    async def teacher_connection_error(request, error):
        message = "教师连接超时，请检查服务地址或稍后重试" if isinstance(error, httpx.TimeoutException) else "无法连接教师服务，请检查地址、网络与代理设置"
        return JSONResponse({"error": {"message": message}}, status_code=409)

    @app.get("/api/health")
    def health():
        return {"app": "luban", "state": runtime.state, "version": metadata["version"], "build": metadata["build"]}

    @app.get("/api/version")
    def client_version():
        return {"version": metadata["version"], "build": metadata["build"]}

    @app.get("/api/chat/capabilities")
    def chat_features():
        return {"text": True, "images": False, "audio": False,
                "image_reason": "当前对话仅支持文字，图片和音频暂不开放",
                "image_routes": [], "image_model_verified": False,
                "documents": {"docx": True, "pdf": False},
                "limits": {"max_document_bytes": MAX_DOCUMENT_BYTES, "max_text_chars": MAX_TEXT_CHARS}}

    @app.post("/api/feeding/extract")
    async def extract_feeding_source(request: Request):
        return await asyncio.to_thread(extract_source, await request.body(), request.query_params.get("name"))

    @app.post("/api/chat/attachments/extract")
    async def extract_attachment(request: Request):
        return await asyncio.to_thread(extract_document, await request.body(), request.query_params.get("name"))

    @app.post("/api/update/prepare")
    def prepare_update():
        nonlocal update_until
        def prepare():
            nonlocal update_until
            update_until = time.monotonic() + 60
            return {"ok": True}
        return exclusive_command(prepare)

    @app.post("/api/update/cancel")
    def cancel_update():
        nonlocal update_until
        with command_lock:
            update_until = 0.0
        return {"ok": True}

    @app.post("/api/session")
    async def session(request: Request):
        body = await request.json()
        token = body.get("token", "")
        if not isinstance(token, str) or not token.isascii() or not secrets.compare_digest(token, settings.secret("ui_token")):
            raise HTTPException(401, "Invalid launch session")
        response = JSONResponse({"ok": True})
        response.set_cookie("luban_session", token, httponly=True, samesite="strict", path="/")
        return response

    @app.get("/api/status")
    def status():
        model_ready = runtime.state in {"ready", "generating", "learning", "evolving", "evaluating"}
        dashboard_data = model_ready or runtime.state == "stopped"
        if dashboard_data:
            try:
                records = runtime.memory.samples()
                samples = {p: sum(s["purpose"] == p and s["approved"] and s["training_allowed"]
                                 and not json.loads(s["provenance"]).get("route_only") for s in records)
                           for p in ("training", "validation")}
            except (OSError, RuntimeError):
                samples = {"training": 0, "validation": 0}
            try:
                usage = runtime.memory.usage()
                materials = runtime.memory.material_counts()
            except (OSError, RuntimeError):
                usage, materials = {}, {"total": 0, "pending": 0, "eligible": 0}
        else:
            samples, usage, materials = {"training": 0, "validation": 0}, {}, {"total": 0, "pending": 0, "eligible": 0}
        if model_ready:
            runtime_status = runtime.status()
        else:
            # Stopping releases the model tensors, not the historical usage
            # ledger. Keep the dashboard useful while clearly reporting that
            # inference is unavailable until the user starts the model.
            base_parameters = (getattr(runtime, "_last_loaded_base_count", 0)
                               if runtime.state == "stopped" else 0)
            language_parameters = (getattr(runtime, "_last_loaded_language_count", 0)
                                   if runtime.state == "stopped" else 0)
            runtime_status = {
                "state": runtime.state, "error": runtime.error, "device": runtime.device,
                "model": "luban0", "model_path": "", "base_parameters": base_parameters,
                "language_parameters": language_parameters, "language_layers": 0, "routing_layers": 0,
                "core_training_layer": None, "core_version": None, "core_requires_review": None,
                "active_expert": None, "expert_parameters": 0,
                "paging": {"enabled": False}, "context": None,
                "inference": {"engine": "loading"},
                "training_engine": {"name": "Luban 训练引擎", "core": "loading", "routing_and_experts": "loading"},
                "native_process": None, "last_generation_usage": copy.deepcopy(getattr(runtime, "_last_generation_usage", None)),
                "router": {"supported_actions": [], "calibrated": False, "gates_expert": False},
                "job": None, "rss_bytes": 0, "metal_bytes": 0,
            }
        return {"runtime": runtime_status, "speech": speech.status(), "teacher_calls": teacher.calls,
                "samples": samples, "usage": usage, "teacher_materials": materials,
                "automatic": automatic.status(), "benchmark": benchmark.status(), "growth": growth.status(),
                "api_base": f"http://127.0.0.1:{public_port}/v1"}

    @app.post("/api/automatic/settings")
    async def automatic_settings(request: Request):
        return settings.update({"automatic": await request.json()})

    @app.post("/api/automatic/evolve")
    async def automatic_evolve(request: Request):
        body = await request.json()
        return await asyncio.to_thread(automatic.evolve, body.get("signature"))

    @app.post("/api/automatic/defer")
    async def automatic_defer(request: Request):
        body = await request.json()
        return automatic.defer(body.get("signature"))

    @app.get("/api/automatic/close-status")
    def close_status():
        return {**automatic.close_status(), "package_busy": package_busy()}

    @app.get("/api/learning/contents")
    def get_learning_contents():
        return automatic.learning_contents()

    @app.delete("/api/learning/contents/{sample_id}")
    async def delete_learning_content(sample_id: str):
        contents = automatic.learning_contents()
        pending = {item.get("id") for item in contents.get("pending", {}).get("items", [])}
        if sample_id not in pending:
            raise ValueError("只能删除尚未学会的学习条目")
        await asyncio.to_thread(benchmark.pause)
        await asyncio.to_thread(automatic.pause)
        with command_lock:
            ensure_not_updating()
            runtime.memory.delete_sample(sample_id)
        return {"ok": True, "deleted": sample_id}

    @app.post("/api/automatic/pause")
    async def pause_automatic():
        if package_busy():
            raise RuntimeError("正在处理模型包，请完成后再退出")
        await asyncio.to_thread(benchmark.pause)
        return await asyncio.to_thread(automatic.pause, closing=True)

    @app.post("/api/automatic/resume")
    def resume_automatic():
        automatic.closing = False
        automatic.last_activity = time.monotonic()
        return {"ok": True}

    @app.get("/api/settings")
    def get_settings():
        return settings.public()

    @app.get("/api/agreement")
    def get_agreement():
        return settings.agreement()

    @app.get("/api/leaderboard")
    def get_leaderboard(request: Request):
        page = request.query_params.get("page", "1")
        if not re.fullmatch(r"[1-9][0-9]{0,5}", page):
            raise ValueError("页码需要是 1–999999 的整数")
        return leaderboard.read(int(page), request.query_params.get("sort", "growth"), request.query_params.get("domain", "mathematics"))

    @app.post("/api/growth/start")
    def start_growth():
        return exclusive_command(growth.start)

    @app.post("/api/growth/cancel")
    async def cancel_growth():
        return await asyncio.to_thread(growth.pause)

    @app.get("/api/model-quiz")
    def get_model_quiz():
        return growth.quiz_status()

    @app.post("/api/model-quiz/start")
    def start_model_quiz():
        return exclusive_command(growth.start_quiz)

    @app.post("/api/model-quiz/cancel")
    async def cancel_model_quiz():
        return await asyncio.to_thread(growth.pause)

    @app.post("/api/benchmark/start")
    def start_benchmark():
        return exclusive_command(benchmark.start)

    @app.post("/api/benchmark/cancel")
    async def cancel_benchmark():
        return await asyncio.to_thread(benchmark.pause)

    @app.put("/api/leaderboard/profile")
    async def save_leaderboard_profile(request: Request):
        return leaderboard.save_profile(await request.json())

    @app.post("/api/leaderboard/publish")
    def publish_leaderboard():
        return leaderboard.publish()

    @app.delete("/api/leaderboard/publish")
    def unpublish_leaderboard():
        return leaderboard.unpublish()

    @app.post("/api/agreement/accept")
    def accept_agreement():
        return settings.accept_agreement()

    @app.patch("/api/settings")
    async def set_settings(request: Request):
        body = await request.json()
        if learning_toggle(body):
            # This scheduling preference does not touch the resident model or its lease.
            ensure_not_updating()
            return settings.update(body)
        with command_lock:
            ensure_not_updating()
            if any(k in body for k in ("model_path", "speech_model_path", "storage_root")) and (runtime.state in {"loading", "generating", "learning", "evolving", "evaluating"} or requests or (runtime.job and runtime.job.get("state") == "queued") or automatic.close_status()["running"]):
                raise RuntimeError("模型忙碌时不能修改模型或存储目录")
            return settings.update(body)

    @app.get("/api/access-key")
    def access_key():
        return {"key": settings.secret("access_key")}

    @app.post("/api/access-key/rotate")
    def rotate_access_key():
        key = "lb-" + secrets.token_urlsafe(32)
        settings.update({"access_key": key})
        return {"key": key}

    @app.post("/api/teacher/models")
    async def teacher_models(request: Request):
        body = await request.json() if await request.body() else {}
        return await asyncio.to_thread(lambda: {"models": teacher.models(body.get("teacher"))})

    @app.post("/api/teacher/test")
    async def test_teacher(request: Request):
        body = await request.json()

        def work():
            started = time.perf_counter()
            config, headers = teacher._config(body.get("teacher"))
            messages = [{"role": "user", "content": "这是连接测试，请只回复 OK。"}]
            text, reasoning, reported = [], [], None
            for delta in teacher.stream(messages, 16, draft=body.get("teacher")):
                if delta.get("content"):
                    text.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("usage"):
                    reported = delta["usage"]
            reply = teacher.redact("".join(text), headers)
            if not reply.strip():
                raise RuntimeError("教师未返回可见回答，请检查模型或降低思考程度后重试")
            with runtime.lock:
                try:
                    usage = runtime.teacher_usage(messages, reply, "".join(reasoning), reported)
                except RuntimeError:
                    usage = None
                if usage:
                    runtime.memory.record_usage("teacher", usage)
            return {"ok": True, "model": teacher.redact(config["model"], headers), "reply": reply,
                    "elapsed_seconds": time.perf_counter() - started, "usage": usage}

        return await asyncio.to_thread(exclusive_command, work)

    @app.get("/api/conversations")
    def conversations():
        return {"conversations": runtime.memory.conversations()}

    @app.get("/api/conversations/{conversation_id}")
    def conversation(conversation_id: str):
        result = runtime.memory.conversation(conversation_id)
        if result is None:
            raise HTTPException(404, "Conversation not found")
        return result

    @app.delete("/api/conversations/{conversation_id}")
    def delete_conversation(conversation_id: str):
        exclusive_command(lambda: runtime.memory.delete_conversation(conversation_id))
        return {"ok": True}

    @app.get("/api/samples")
    def samples():
        return {"samples": [row for row in runtime.memory.samples() if row["approved"] and row["training_allowed"]]}

    @app.get("/api/materials")
    def materials():
        return {"materials": runtime.memory.teacher_materials()}

    @app.post("/api/samples")
    async def add_sample(request: Request):
        body = await request.json()
        source = body.get("source", "human")
        provenance = {"verified_by": "local requester", "created": time.time()}
        if source == "synthetic":
            provenance["scope"] = "manually supplied synthetic research material; not a teacher response"
        for key in ("family_group", "split", "split_family", "evolution_role"):
            if key in body:
                if not isinstance(body[key], str) or not body[key].strip():
                    raise ValueError(key + " 需要非空文本")
                provenance[key] = body[key]
        if source == "human":
            provenance.setdefault("evolution_role", "new")
        action = body.get("route_action")
        if action:
            if action not in {"local_core", "neural_expert", "teacher"}:
                raise ValueError("路由标注应为 local_core、neural_expert 或 teacher")
            provenance["route_action"] = action
        if source == "teacher":
            config = settings.snapshot()["teacher"]
            agreement = settings.agreement()
            if not agreement["accepted"]:
                raise ValueError("请先阅读并接受使用协议，再提交教师学习材料；您需自行处理对应服务条款与第三方权利")
            provenance.update(provider=config["base_url"], model=config["model"],
                              rights_basis="user responsibility under agreement",
                              agreement_version=agreement["version"], user_responsibility=True)
        sample_id = runtime.memory.add_sample(body.get("prompt", ""), body.get("answer", ""), source,
                            body.get("purpose", "training"), body.get("approved") is True,
                            source == "teacher" or body.get("training_allowed") is True, provenance)
        return {"id": sample_id}

    @app.delete("/api/samples/{sample_id}")
    async def delete_sample(sample_id: str):
        await asyncio.to_thread(benchmark.pause)
        await asyncio.to_thread(automatic.pause)
        with command_lock:
            ensure_not_updating()
            runtime.memory.delete_sample(sample_id)
        return {"ok": True}

    @app.get("/api/events")
    def events():
        return {"events": runtime.memory.events()}

    @app.get("/api/experts")
    def experts():
        return {"versions": runtime.versions(),
                "last_attempt": latest_learning_attempt(settings, runtime.memory, getattr(runtime, "identity", None))}

    @app.get("/api/training/backups")
    def training_backups():
        return {"backups": runtime.training_backups()}

    def exclusive_command(function):
        if not command_lock.acquire(blocking=False):
            raise RuntimeError("另一个模型操作正在进行，请稍后重试")
        try:
            ensure_not_updating()
            if runtime.state in {"loading", "generating", "learning", "evolving", "evaluating"} or (runtime.job and runtime.job.get("state") == "queued") or requests or automatic.close_status()["running"]:
                raise RuntimeError("模型正在运行当前任务，请等待或停止后重试")
            return function()
        finally:
            command_lock.release()

    @app.post("/api/model/reload")
    def reload_model():
        def start():
            runtime.state = "loading"
            threading.Thread(target=load, daemon=True).start()
            return {"ok": True}
        return exclusive_command(start)

    @app.post("/api/model/stop")
    def stop_model():
        return exclusive_command(runtime.stop)

    install_package_routes(app, settings, runtime, speech, automatic, command_lock, exclusive_command, package_task)

    @app.get("/api/learning/media")
    def media_status(request: Request):
        return media_learning.status(request.query_params.get("kind", "image"))

    @app.post("/api/learning/media")
    async def media_create(request: Request):
        body = await request.json()
        return await asyncio.to_thread(exclusive_command, lambda: media_learning.create(body))

    @app.post("/api/learning/media/start")
    async def media_start(request: Request):
        body = await request.json()
        return await asyncio.to_thread(exclusive_command, lambda: media_learning.start(body.get("id")))

    @app.post("/api/learning/media/confirm")
    async def media_confirm(request: Request):
        body = await request.json()
        return await asyncio.to_thread(exclusive_command, lambda: media_learning.confirm_teacher(body))

    @app.post("/api/learning/media/cancel")
    def media_cancel():
        return media_learning.pause()

    @app.get("/api/feeding")
    def feeding_status(request: Request):
        return feeding.status(request.query_params.get("id"))

    @app.post("/api/feeding/scan")
    async def feeding_scan(request: Request):
        body = await request.json()
        return await asyncio.to_thread(exclusive_command, lambda: feeding.scan(body.get("files"), mode=body.get("mode", "teacher")))

    @app.post("/api/feeding/clean")
    async def feeding_clean(request: Request):
        body = await request.json()
        if body.get("confirmed_api_use") is not True:
            raise ValueError("清洗会调用已配置教师 API，请先确认调用及费用")
        await asyncio.to_thread(automatic.pause)
        def clean():
            with runtime.lock:
                return feeding.clean(body.get("id"), max_calls=body.get("max_calls", 10))
        return await asyncio.to_thread(exclusive_command, clean)

    @app.post("/api/feeding/confirm")
    async def feeding_confirm(request: Request):
        body = await request.json()
        if body.get("confirmed_answers") is not True:
            raise ValueError("请核验清洗后的问答与引用，再确认加入学习")
        await asyncio.to_thread(automatic.pause)
        def confirm():
            with runtime.lock:
                return feeding.confirm(body.get("id"), body.get("ids"))
        return await asyncio.to_thread(exclusive_command, confirm)

    @app.get("/api/external-brain")
    def external_brain_status():
        if runtime.state != "ready":
            return {"available": False, "reason": "当前模型正在处理任务"}
        binding = runtime.external_binding() if hasattr(runtime, "external_binding") else None
        return ExternalBrain(runtime.memory).stats(runtime.identity, binding) if binding else {"available": False}

    @app.post("/api/expert/hibernate")
    def hibernate():
        exclusive_command(runtime.hibernate)
        return {"ok": True}

    @app.post("/api/expert/wake")
    def wake():
        exclusive_command(runtime.wake)
        return {"ok": True}

    @app.post("/api/expert/rollback")
    async def rollback(request: Request):
        body = await request.json()
        await asyncio.to_thread(exclusive_command, lambda: runtime.rollback(body.get("id", "")))
        return {"ok": True}

    @app.post("/api/training/backups/restore")
    async def restore_training(request: Request):
        body = await request.json()
        result = await asyncio.to_thread(exclusive_command, lambda: runtime.restore_training(body.get("id", "")))
        return {"ok": True, **result}

    @app.get("/api/evolution")
    @app.get("/api/evolution/status")
    def evolution():
        result = evolution_status(runtime, settings, teacher)
        result["media_materials"] = media_learning.evolution_materials()
        parameters = getattr(runtime, "language_count", getattr(runtime, "base_count", 0))
        result["capacity"] = {"current_parameters": parameters, "after_parameters": parameters,
                              "additional_weight_bytes": 0, "will_grow": False, "eligible": result["ready"]}
        return result

    @app.post("/api/evolution/review")
    async def review_evolution(request: Request):
        body = await request.json()
        return await asyncio.to_thread(exclusive_command,
                                       lambda: review_candidates(runtime, settings, teacher, body.get("ids")))

    @app.post("/api/evolution/start")
    async def evolve(request: Request):
        body = await request.json()
        if not isinstance(body, dict) or body:
            raise ValueError("进化参数由已验证方法自动决定")
        if not automatic._available():
            raise ValueError("模型尚未就绪，请等待可恢复的进化任务")
        return await asyncio.to_thread(automatic.start_manual, "evolution")

    @app.post("/api/evolution/restore")
    async def restore_evolution(request: Request):
        body = await request.json()
        return await asyncio.to_thread(exclusive_command, lambda: restore_core(runtime, body.get("version")))

    @app.post("/api/audio/transcriptions")
    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        raise ValueError("当前 API 仅支持文字，音频接口暂不开放")

    @app.post("/api/learn")
    async def learn(request: Request):
        body = await request.json()
        steps = integer(body.get("steps", 16))
        if not 4 <= steps <= 64:
            raise ValueError("学习步数应为 4–64")
        if automatic._available():
            return await asyncio.to_thread(automatic.start_manual, "learning", steps)
        if not runtime.memory.samples("training") or not runtime.memory.samples("validation"):
            raise ValueError("请先添加学习材料及独立检验材料")

        def work():
            try:
                runtime.learn(steps)
            except Exception as error:
                runtime.job = {"state": "error", "error": str(error)}

        def start():
            runtime.job = {"state": "queued", "step": 0, "steps": steps}
            threading.Thread(target=work, daemon=True, name="luban-learner").start()
            return {"job_id": "pending"}
        return exclusive_command(start)

    @app.post("/api/learn/cancel")
    def cancel_learn():
        if runtime.job:
            runtime.job["cancel_requested"] = True
        return {"ok": True}

    @app.post("/api/router/calibrate")
    def calibrate_router():
        if automatic._available():
            return automatic.start_manual("router", 160)
        def work():
            try:
                runtime.calibrate_router()
            except Exception as error:
                runtime.job = {"kind": "router", "state": "error", "error": str(error)}

        def start():
            runtime.job = {"kind": "router", "state": "queued", "step": 0, "steps": 160}
            threading.Thread(target=work, daemon=True, name="luban-router").start()
            return {"job_id": "pending"}
        return exclusive_command(start)

    @app.post("/api/stop")
    async def stop(request: Request):
        body = await request.json()
        with tasks_lock:
            event = requests.get(body.get("request_id"))
            if event:
                event.set()
        return {"ok": True}

    async def chat(body, ui=False):
        messages = validated_messages(body)
        tools = validate_tools(body)
        tool_choice, parallel = validate_tool_options(body, tools)
        tool_mode = bool(tools or any(m.get("tool_calls") or m["role"] == "tool" for m in messages))
        if tool_mode and (ui or body.get("route") == "teacher"):
            raise ValueError("工具调用仅支持本地兼容接口，由客户端执行工具")
        if body.get("model", "luban0") != "luban0":
            raise ValueError("模型名称应为 luban0")
        route = body.get("route", "auto")
        if route not in {"auto", "local", "teacher"}:
            raise ValueError("route 应为 auto、local 或 teacher")
        maximum = integer(body.get("max_tokens", 0 if ui else settings.snapshot()["max_new_tokens"]))
        if not (1 <= maximum <= 32768 or maximum == 0 and "max_tokens" not in body):
            raise ValueError("max_tokens 应为 1–32768")
        auto_compact = ui and body.get("auto_compact", False)
        compact_only = ui and body.get("compact_only", False)
        if not isinstance(auto_compact, bool) or not isinstance(compact_only, bool):
            raise ValueError("上下文压缩选项需要布尔值")
        thinking = body.get("thinking", settings.snapshot()["local_thinking"])
        if thinking not in {"off", "low", "medium", "high"}:
            raise ValueError("thinking 应为 off、low、medium 或 high")
        temperature = body.get("temperature", settings.snapshot()["temperature"])
        if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not 0 <= temperature <= 2:
            raise ValueError("temperature 应为 0–2")
        request_id = "chatcmpl-" + uuid.uuid4().hex
        stopped = threading.Event()
        output = queue.Queue()
        await asyncio.to_thread(benchmark.pause)
        await asyncio.to_thread(automatic.pause)
        if not command_lock.acquire(blocking=False):
            raise RuntimeError("模型操作正在完成，请稍后重试")
        try:
            with tasks_lock:
                ensure_not_updating()
                if len(requests) >= 4:
                    raise RuntimeError("请求队列已满，请稍后重试")
                requests[request_id] = stopped
        finally:
            command_lock.release()

        def worker():
            nonlocal messages
            parts, reasoning, calls = [], [], []
            tool_buffer = ToolOutputBuffer(tools, tool_choice, parallel) if tool_mode else None
            usage = None
            decision = None
            teacher_config = None

            def forward(delta):
                if delta.get("content"):
                    parts.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                output.put({"type": "delta", **delta})

            thinking_stream = ThinkingStream(forward)

            def emit(delta):
                if delta.get("content"):
                    if tool_buffer is not None:
                        tool_buffer.feed(delta["content"])
                    else:
                        thinking_stream.feed(delta["content"])
                if delta.get("reasoning_content"):
                    forward({"reasoning_content": delta["reasoning_content"]})

            try:
                automatic.pause()
                with runtime.lock:
                    if runtime.state in {"loading", "learning", "evolving"} or (runtime.job and runtime.job.get("state") == "queued"):
                        raise RuntimeError("模型正在加载或学习，请稍后重试")
                    if stopped.is_set():
                        raise RuntimeError("请求已停止")
                    teacher_config = settings.snapshot()["teacher"]
                    teacher_first = (teacher_config.get("answer_mode", "local_first") == "teacher_first"
                                     and "text" in teacher_config.get("teaching_modalities", ["text"]))
                    automatic_answer = teacher_config.get("answer_mode", "local_first") == "auto"
                    teacher_available = (bool(settings.secret("teacher_key"))
                                         and bool(teacher_config.get("base_url"))
                                         and bool(teacher_config.get("model"))
                                         and "text" in teacher_config.get("teaching_modalities", ["text"]))
                    if compact_only or auto_compact and not teacher_first:
                        output.put({"type": "meta", "route": "local", "action": "local_core",
                                    "source": "context_check", "request_id": request_id})
                        messages, context, compaction = compact_messages(runtime, messages, stopped, thinking, compact_only)
                        if compaction:
                            output.put(compaction)
                        output.put({"type": "context", **context})
                        if compact_only:
                            output.put({"type": "done", "compact_only": True})
                            return
                    if tool_mode:
                        teacher_first = False
                        decision = {"route": "local", "action": "local_core", "source": "tool_protocol"}
                    elif teacher_first:
                        decision = {"route": "teacher", "action": "teacher", "source": "teacher_first"}
                    elif route == "auto" and automatic_answer:
                        decision = runtime.choose_route(messages, "auto")
                        if decision["route"] == "teacher" and not teacher_available:
                            decision = {"route": "local", "action": "local_core", "source": "auto_teacher_unavailable",
                                        "confidence": decision.get("confidence")}
                    elif route == "auto":
                        decision = {"route": "local", "action": "local_core", "source": "local_first"}
                    else:
                        decision = runtime.choose_route(messages, route)
                    teacher_config = settings.snapshot()["teacher"]
                    output.put({"type": "meta", **decision, "request_id": request_id})
                    if decision["route"] == "teacher":
                        teacher_config = settings.snapshot()["teacher"]
                        agreement = settings.agreement()
                        teacher_config.update(agreement_accepted=agreement["accepted"],
                                              agreement_version=agreement["version"])
                        reported_usage = None
                        teacher_finish = "stop"
                        # Teacher output stays bounded when local settings use auto.
                        for delta in teacher.stream(messages, maximum or 2048):
                            if stopped.is_set():
                                break
                            if delta.get("usage"):
                                reported_usage = delta["usage"]
                            elif delta.get("finish_reason"):
                                teacher_finish = delta["finish_reason"]
                            else:
                                emit(delta)
                        thinking_stream.finish()
                        try:
                            usage = runtime.teacher_usage(messages, "".join(parts), "".join(reasoning), reported_usage)
                        except RuntimeError as failure:
                            if str(failure) != "教师未提供词元统计，本地词表也尚未就绪" or runtime.tokenizer is not None:
                                raise
                            usage = {"unknown": True, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
                        usage["finish_reason"] = teacher_finish
                    else:
                        generation_options = {}
                        local_messages = messages
                        if tool_mode:
                            local_messages = prepare_tool_prompt(messages, tool_choice, parallel)
                            generation_options = {"tools": tools, "allow_reference": False, "observe_external": False}
                        usage = runtime.generate(local_messages, emit, stopped, thinking=thinking,
                                                 max_tokens=maximum, temperature=temperature, cache_reuse=True, **generation_options)
                        thinking_stream.finish()
                        if tool_buffer is not None and not stopped.is_set() and usage["finish_reason"] == "stop":
                            text, calls = tool_buffer.finish()
                            if tool_buffer.reasoning:
                                forward({"reasoning_content": tool_buffer.reasoning})
                            if text:
                                parts.append(text)
                                output.put({"type": "delta", "content": text})
                            if calls:
                                usage["finish_reason"] = "tool_calls"
                                output.put({"type": "delta", "tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]})
                actual_action = usage.pop("local_action", "teacher" if decision["route"] == "teacher" else decision.get("action"))
                model_context = {key: usage.pop(key, None) for key in
                    ("core_version", "expert_id", "expert_core_version", "module_requires_review")}
                runtime.memory.record_usage(decision["route"], usage)
                decision_id = runtime.memory.event("tool_decision" if tool_mode else "decision", {"request_id": request_id, "messages": messages,
                    "answer": "".join(parts), "decision": {**decision, "action": actual_action},
                    "base_identity": getattr(runtime, "identity", None),
                    "model_context": model_context,
                    "usage": usage, "complete": not stopped.is_set() and usage["finish_reason"] == "stop"})
                if (not tool_mode and not teacher_first and decision["route"] == "local" and parts and not stopped.is_set()
                        and usage["finish_reason"] == "stop"):
                    practice.enqueue(request_id, messages, decision_id=decision_id)
                if decision["route"] == "teacher" and parts:
                    answer = "".join(parts)
                    eligible, reason = runtime.teacher_eligibility(messages, answer, teacher_config)
                    if stopped.is_set():
                        eligible, reason = False, "请求被中止，回答可能不完整"
                    elif usage["finish_reason"] != "stop":
                        eligible, reason = False, "教师未完整结束回答：" + str(usage["finish_reason"])
                    provenance = {"provider": teacher_config["base_url"], "model": teacher_config["model"],
                                  "rights_basis": ("user responsibility under agreement"
                                                   if teacher_config["agreement_accepted"] else "pending agreement"),
                                  "agreement_version": teacher_config["agreement_version"],
                                  "user_responsibility": teacher_config["agreement_accepted"],
                                  "request_id": request_id, "created": time.time()}
                    material_id = runtime.memory.capture_teacher(messages, answer, provenance, eligible, reason)
                    if teacher_first and not stopped.is_set() and usage["finish_reason"] == "stop":
                        practice.enqueue(request_id, messages, material_id)
                    if (not stopped.is_set() and usage["finish_reason"] == "stop" and not getattr(runtime, "research", False)
                            and isinstance(getattr(runtime, "identity", None), str) and re.fullmatch(r"[a-f0-9]{64}", runtime.identity)
                            and len(json.dumps(messages, ensure_ascii=False).encode()) <= MAX_CONTEXT_BYTES and len(answer) <= 12000):
                        ExternalBrain(runtime.memory).record_teacher(messages, runtime.identity, answer, provenance, material_id=material_id)
                conversation_id = None
                if ui:
                    previous = runtime.memory.conversation(body["conversation_id"]) if body.get("conversation_id") else None
                    if previous:
                        for old, current in zip(previous["messages"], messages):
                            if old["role"] != current["role"] or old["content"] != current["content"]:
                                break
                            if current["role"] == "assistant" and old.get("source") in {"local", "teacher"}:
                                current["source"] = old["source"]
                    full = messages + [{"role": "assistant", "content": "".join(parts), "source": decision["route"]}]
                    conversation_id = runtime.memory.save_conversation(body.get("conversation_id"), full)
                output.put({"type": "done", "usage": usage, "conversation_id": conversation_id})
            except Exception as error:
                thinking_stream.finish()
                # HTTP adapter errors have already been sanitized; never include credentials.
                message = str(error)
                for name in ("teacher_key", "access_key", "ui_token"):
                    key = settings.secret(name)
                    if key:
                        message = message.replace(key, "[redacted]")
                output.put({"type": "error", "message": message})
            finally:
                output.put(None)
                with tasks_lock:
                    requests.pop(request_id, None)

        threading.Thread(target=worker, daemon=True, name="luban-chat").start()

        async def stream_events():
            try:
                while True:
                    try:
                        event = await asyncio.to_thread(output.get, True, SSE_HEARTBEAT_SECONDS)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    if event is None:
                        if not ui:
                            yield "data: [DONE]\n\n"
                        break
                    if ui:
                        data = event
                    elif event["type"] == "error":
                        data = {"error": {"message": event["message"], "type": "runtime_error"}}
                    else:
                        delta, finish = {}, None
                        if event["type"] == "delta":
                            delta = {k: event[k] for k in ("content", "reasoning_content", "tool_calls") if k in event}
                        elif event["type"] == "meta":
                            delta = {"role": "assistant"}
                        elif event["type"] == "done":
                            finish = event["usage"]["finish_reason"]
                        data = {"id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                                "model": "luban0", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        if event["type"] == "done":
                            data["usage"] = event["usage"]
                    yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
            finally:
                stopped.set()

        if ui or body.get("stream", False):
            return StreamingResponse(stream_events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        content, reasoning, calls, usage = [], [], [], None
        try:
            while True:
                event = await asyncio.to_thread(output.get)
                if event is None:
                    break
                if event["type"] == "error":
                    raise RuntimeError(event["message"])
                if event["type"] == "delta":
                    content.append(event.get("content", ""))
                    reasoning.append(event.get("reasoning_content", ""))
                    calls.extend({k: v for k, v in call.items() if k != "index"} for call in event.get("tool_calls", []))
                if event["type"] == "done":
                    usage = event["usage"]
        finally:
            stopped.set()
        return {"id": request_id, "object": "chat.completion", "created": int(time.time()),
                "model": "luban0", "choices": [{"index": 0, "message": {"role": "assistant",
                    "content": "".join(content) or (None if calls else ""), "reasoning_content": "".join(reasoning),
                    **({"tool_calls": calls} if calls else {})},
                    "finish_reason": usage["finish_reason"] if usage else "stop"}], "usage": usage}

    @app.post("/api/chat")
    async def ui_chat(request: Request):
        return await chat(await request.json(), ui=True)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "luban0", "object": "model", "owned_by": "maihaohuo"}]}

    @app.post("/v1/chat/completions")
    async def compatible_chat(request: Request):
        return await chat(await request.json())

    web = Path(__file__).with_name("web")
    if web.exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    else:
        @app.get("/")
        def missing_ui():
            return JSONResponse({"message": "前端尚未构建，请执行 bun run build"}, status_code=503)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--home", default=None)
    args = parser.parse_args()
    app = create_app(Settings(args.home), port=args.port,
                     autoload=os.environ.get("LUBAN_COMPUTE_AUTOLOAD", "0") != "0")
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
