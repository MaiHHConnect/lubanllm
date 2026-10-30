"""Authenticated model-package routes with streamed uploads and bounded jobs."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading
import time
import shutil
import uuid

from fastapi import Request
from fastapi.responses import FileResponse

from .model_packages import (FOUNDATION_URL, FOUNDATION_SHA256, MAX_BYTES, MAX_TEXT,
    export_package, stage_import, stage_reset, prepare_activation, finish_activation,
    recover_activation, cleanup_retired_package)
from .settings import require_storage
from .foundation_install import install_models, installation_status


UPLOAD_LIMIT = MAX_BYTES + MAX_TEXT


def install_package_routes(app, settings, runtime, speech, automatic, command_lock, exclusive_command, task):
    """One package job blocks model mutations; settings switch only after staging."""
    def directory():
        parent = require_storage(settings.snapshot()["storage_root"]) / "app/model-packages"
        if parent.is_symlink():
            raise ValueError("模型包目录不能是符号链接")
        parent.mkdir(parents=True, exist_ok=True)
        return parent

    def progress(value):
        task.update(value)

    def resume():
        automatic.last_activity = time.monotonic()

    def reserve(kind):
        if task["state"] in {"queued", "running", "uploading"}:
            raise RuntimeError("正在处理模型包，请稍后重试")
        task.clear()
        task.update(state="queued", kind=kind, id=uuid.uuid4().hex, bytes=0, total=None)

    def activate(staged):
        with runtime.lock:
            if getattr(speech, "model", None) is not None:
                raise RuntimeError("语音模型正在使用，请稍后重试")
            runtime._release_model()
            runtime.state = "stopped"
            try:
                token = prepare_activation(settings, staged)
                runtime.load()
                speech.model_path = Path(settings.snapshot()["speech_model_path"])
                finish_activation(settings, token)
            except BaseException:
                runtime._release_model()
                recover_activation(settings)
                runtime.load()
                speech.model_path = Path(settings.snapshot()["speech_model_path"])
                raise
            cleanup = None
            if staged["activation_intent"] == "reset":
                try:
                    cleanup = cleanup_retired_package(settings, token)
                except Exception:
                    cleanup = {"removed": False, "reason": "新基础模型已启用，旧目录清理未完成，原文件仍保留"}
            return {"package_id": staged["package_id"], "kind": staged["kind"], "activated": True,
                    "cleanup": cleanup}

    def work(kind, uploaded=None, phrase=None):
        try:
            with command_lock:
                task.update(state="running")
                if kind == "install":
                    if runtime.state != "stopped":
                        raise RuntimeError("请先停止模型再安装基础模型")
                    result = install_models(settings, progress)
                    from .memory import Memory
                    runtime.memory = Memory(settings.snapshot()["storage_root"])
                    speech.model_path = Path(settings.snapshot()["speech_model_path"])
                elif kind == "export":
                    path = directory() / ("luban0-" + task["id"] + ".zip")
                    result = export_package(runtime, path, progress=progress)
                    result["download"] = "/api/model-packages/download/" + task["id"]
                else:
                    staged = (stage_import(uploaded, directory(), progress=progress) if kind == "import"
                              else stage_reset(phrase, directory(), progress=progress))
                    result = activate(staged)
                task.update(state="complete", result=result)
        except Exception as error:
            message = str(error)
            for key in ("teacher_key", "access_key", "ui_token"):
                secret = settings.secret(key)
                if secret:
                    message = message.replace(secret, "[redacted]")
            task.update(state="error", error=message)
        finally:
            if uploaded is not None:
                uploaded.unlink(missing_ok=True)
            resume()

    @app.get("/api/model-packages")
    def status():
        value = {key: item for key, item in task.items() if key != "result"}
        if task.get("result"):
            value["result"] = {key: item for key, item in task["result"].items() if key != "path"}
        return {"task": value, "foundation_url": FOUNDATION_URL, "foundation_available": FOUNDATION_SHA256 is not None,
                "installation": installation_status(settings),
                "export_contains": "基础模型、声音模型、已采纳权重及保护材料；保护材料可能含用户知识内容"}

    @app.post("/api/model-packages/install")
    async def first_install():
        if runtime.state != "stopped":
            raise RuntimeError("请先停止模型再安装基础模型")
        await asyncio.to_thread(automatic.pause)
        try:
            exclusive_command(lambda: reserve("install"))
        except BaseException:
            resume()
            raise
        threading.Thread(target=work, args=("install",), daemon=True, name="luban-first-install").start()
        return {"started": True, "id": task["id"]}

    @app.post("/api/model-packages/export")
    async def export():
        await asyncio.to_thread(automatic.pause)
        try:
            exclusive_command(lambda: reserve("export"))
        except BaseException:
            resume()
            raise
        threading.Thread(target=work, args=("export",), daemon=True, name="luban-package-export").start()
        return {"started": True, "id": task["id"]}

    @app.get("/api/model-packages/download/{identifier}")
    def download(identifier: str):
        if identifier != task.get("id") or task.get("kind") != "export" or task.get("state") != "complete":
            raise ValueError("导出包尚未完成或已经失效")
        path = Path(task["result"]["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError("导出包文件不可用")
        return FileResponse(path, media_type="application/zip", filename="luban0-我的模型.zip")

    @app.post("/api/model-packages/import")
    async def import_package(request: Request):
        if request.headers.get("content-type", "").split(";")[0] != "application/zip":
            raise ValueError("请上传 ZIP 模型包")
        await asyncio.to_thread(automatic.pause)
        uploaded = None
        started = False
        reserved = False
        try:
            exclusive_command(lambda: reserve("import"))
            reserved = True
            task.update(state="uploading", phase="upload")
            uploaded = directory() / ("upload-" + task["id"] + ".zip")
            declared = request.headers.get("content-length")
            total = int(declared) if declared and declared.isdecimal() else None
            if total is not None and shutil.disk_usage(uploaded.parent).free < total + 256 * 1024**2:
                raise ValueError("磁盘空间不足，无法接收模型包")
            task["total"] = total
            with uploaded.open("xb") as output:
                os.chmod(uploaded, 0o600)
                done = 0
                async for chunk in request.stream():
                    done += len(chunk)
                    if done > UPLOAD_LIMIT:
                        raise ValueError("模型包上传尺寸超限")
                    output.write(chunk)
                    task["bytes"] = done
                if not done or total is not None and done != total:
                    raise ValueError("模型包上传不完整")
                output.flush()
                os.fsync(output.fileno())
            threading.Thread(target=work, args=("import", uploaded), daemon=True, name="luban-package-import").start()
            started = True
            return {"started": True, "id": task["id"]}
        except BaseException as error:
            if reserved:
                task.update(state="error", error=str(error))
            raise
        finally:
            if not started:
                if uploaded is not None:
                    uploaded.unlink(missing_ok=True)
                resume()

    @app.post("/api/model-packages/reset")
    async def reset(request: Request):
        from .model_packages import RESET_PHRASE
        body = await request.json()
        if body.get("phrase") != RESET_PHRASE:
            raise ValueError("请输入“我同意重修”")
        # An unpublished URL must fail before pausing, releasing or deleting.
        if FOUNDATION_SHA256 is None:
            raise ValueError("基础鲁班零号模型包尚未发布，当前模型和学习资料保持不变")
        await asyncio.to_thread(automatic.pause)
        try:
            exclusive_command(lambda: reserve("reset"))
        except BaseException:
            resume()
            raise
        threading.Thread(target=work, kwargs={"kind": "reset", "phrase": body["phrase"]}, daemon=True).start()
        return {"started": True, "id": task["id"]}
