"""Explicit local media lessons; candidate rounds never count as evolution."""
from __future__ import annotations

import base64
import binascii
import copy
import importlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

import psutil

from .settings import require_storage
from .training_checkpoint import _owner_alive, TrainingPaused

MAX_MEDIA_BYTES = 8 * 1024 * 1024
KINDS = {"image": "image_learning", "sound": "speech_learning", "operation": "operation_learning", "video": "image_learning"}


class MediaLearning:
    def __init__(self, runtime, settings, teacher=None):
        self.runtime, self.settings, self.teacher = runtime, settings, teacher
        self.lock = threading.RLock()
        self.worker = None
        self.cancel = threading.Event()

    def _kind(self, kind):
        if kind not in KINDS:
            raise ValueError("请选择图片、声音或操作学习")
        return importlib.import_module("." + KINDS[kind], __package__)

    def _schema(self, db):
        db.execute("CREATE TABLE IF NOT EXISTS media_lessons (id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, updated REAL NOT NULL)")

    def _write(self, body):
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            self._schema(db)
            db.execute("INSERT OR REPLACE INTO media_lessons VALUES (?,?,?,?)",
                       (body["id"], body["kind"], json.dumps(body, ensure_ascii=False, allow_nan=False), time.time()))

    def _load(self, lesson_id):
        if not isinstance(lesson_id, str) or not re.fullmatch(r"[a-f0-9]{32}", lesson_id):
            raise ValueError("学习资料编号无效")
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            self._schema(db)
            row = db.execute("SELECT body FROM media_lessons WHERE id=?", (lesson_id,)).fetchone()
        if row is None:
            raise ValueError("找不到学习资料")
        body = json.loads(row["body"])
        local_finished = ((body.get("owner") or {}).get("pid") == os.getpid()
                          and not (self.worker and self.worker.is_alive()))
        if body["state"] == "running" and (not _owner_alive(body) or local_finished):
            body.update(state="interrupted", error="上次学习中断，保留最近完整候选；请核对后继续", owner=None)
            body.update(step=body.get("checkpoint_step", 0), total_adam_step=body.get("checkpoint_total", 0))
            self._write(body)
        if body["state"] == "teacher_running" and not _owner_alive(body):
            body.update(state="teacher_interrupted", owner=None,
                        error="教师请求中断，费用或结果可能未知；不会自动重复调用")
            self._write(body)
        return body

    @staticmethod
    def public(body):
        keys = ("id", "kind", "state", "round", "step", "steps", "total_adam_step", "phase", "error", "adopted", "teacher_modality", "video_count", "frame_count", "learning_mode")
        value = {key: copy.deepcopy(body.get(key)) for key in keys}
        value["samples"] = [{key: row[key] for key in ("id", "name", "split", "family", "question", "answer", "timestamp_seconds", "source_name") if key in row}
                            for row in body["samples"]]
        for saved, public in zip(body["samples"], value["samples"]):
            if body["kind"] == "video":
                public.update(frame_count=len(saved["video_frames"]),
                              timestamps=[frame["timestamp_seconds"] for frame in saved["video_frames"]])
        value["has_checkpoint"] = bool(body.get("candidate"))
        value["quality_checked"] = body.get("quality_checked", False)
        value["protection_checked"] = body.get("protection_checked", False)
        return value

    def status(self, kind):
        module = self._kind(kind)
        with self.lock, self.runtime.memory.lock, self.runtime.memory.connect() as db:
            self._schema(db)
            row = db.execute("SELECT id FROM media_lessons WHERE kind=? ORDER BY updated DESC LIMIT 1", (kind,)).fetchone()
        value = {"kind": kind, "round_budget": module.ROUND_STEPS,
                 "lesson": self.public(self._load(row["id"])) if row else None,
                 "teacher_enabled": {"sound": "audio"}.get(kind, kind) in
                    self.settings.snapshot()["teacher"].get("teaching_modalities", ["text"])}
        value["video_teacher_enabled"] = "video" in self.settings.snapshot()["teacher"].get("teaching_modalities", ["text"])
        if hasattr(module, "capability"):
            value["capability"] = module.capability(self.runtime)
            if kind == "video":
                from .native_video_training import EXTRA_AVAILABLE_BYTES
                value["capability"]["required_available_bytes"] += EXTRA_AVAILABLE_BYTES
        return value

    def evolution_materials(self):
        """Include local media in the existing evolution view, never as text labels.

        Current candidate writers cannot publish a complete multimodal core.
        Material counts therefore grant neither readiness nor adopted experience.
        """
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            self._schema(db)
            rows = db.execute("""SELECT kind,COUNT(*) AS lessons,
                SUM(json_array_length(body,'$.samples')) AS samples,
                SUM(CASE WHEN json_extract(body,'$.candidate') IS NOT NULL THEN 1 ELSE 0 END) AS candidates
                FROM media_lessons GROUP BY kind""").fetchall()
        counts = {row["kind"]: dict(row) for row in rows}
        return [{"kind": kind, "lessons": counts.get(kind, {}).get("lessons", 0),
                 "samples": counts.get(kind, {}).get("samples", 0),
                 "candidates": counts.get(kind, {}).get("candidates", 0),
                 "ready": False, "adopted": False,
                 "reason": "等待对应核心的完整质量、防遗忘与成套采纳恢复验收"}
                for kind in KINDS]

    def create(self, body):
        kind = body.get("kind")
        self._kind(kind)
        teaching = body.get("teacher_requested") is True
        modality = body.get("teacher_modality", "audio" if kind == "sound" else kind)
        if modality != ("audio" if kind == "sound" else kind):
            raise ValueError("教师资料类型与学习方式不一致")
        if teaching:
            if kind not in {"image", "sound", "video"} or body.get("teacher_confirmed") is not True:
                raise ValueError("请确认将图片或录音发送给已配置的教师，可能产生费用")
            self._teacher_config(modality)
        if body.get("training_allowed") is not True or not teaching and body.get("answers_reviewed") is not True:
            raise ValueError("请核对答案与来源分组，并确认这些资料允许本地训练")
        files = body.get("files")
        if kind == "operation":
            from .operation_learning import prepare_demonstrations
            records = prepare_demonstrations(self.settings.home, body.get("demonstrations"))
            records = [{key: row[key] for key in ("id", "kind", "split", "family", "question", "answer", "provenance")}
                       for row in records]
        else:
            records = self._files(kind, files, reviewed=not teaching, video=modality == "video")
        lesson_id = uuid.uuid4().hex
        binding = {key: self.settings.snapshot()[key] for key in ("model_path", "speech_model_path", "storage_root")}
        saved = {"id": lesson_id, "kind": kind, "samples": records, "binding": binding,
                 "core_version": getattr(self.runtime, "active_core", None), "state": "ready", "round": 0,
                 "step": 0, "steps": self._kind(kind).ROUND_STEPS, "phase": "ready", "candidate": None,
                 "checkpoint_step": 0, "checkpoint_total": 0,
                 "total_adam_step": 0, "adopted": False, "created": time.time(), "owner": None}
        if teaching:
            saved.update(state="teacher_queued", phase="teacher_queued", teacher_config=self._teacher_binding(),
                         teacher_next=0, teacher_modality=modality, teacher_agreement=self.settings.agreement()["version"])
        if modality == "video":
            saved.update(video_count=len(files), frame_count=sum(len(row["video_frames"]) for row in records), learning_mode="joint_video")
        if kind == "operation":
            saved["demonstrations"] = copy.deepcopy(body["demonstrations"])
        self._write(saved)
        return self.public(saved)

    def _files(self, kind, files, *, reviewed=True, video=False):
        if not isinstance(files, list) or not 2 <= len(files) <= 20:
            raise ValueError("请选择至少两个独立来源的资料，最多20个文件")
        if video and len(files) > 5:
            raise ValueError("每批视频最多5个，每段不超过30秒，合计不超过8MiB")
        parsed, families, seen, total = [], [], [], 0
        for row in files:
            if not isinstance(row, dict):
                raise ValueError("学习文件格式无效")
            name, answer, family = row.get("name"), row.get("answer"), row.get("family")
            question = row.get("question", "")
            if (not isinstance(name, str) or not name or len(name) > 180 or Path(name).name != name
                    or re.search(r"[\\\x00-\x1f]", name)):
                raise ValueError("学习文件名无效")
            if (not isinstance(answer, str) or reviewed and not answer.strip() or len(answer) > 4096
                    or not isinstance(family, str) or not family.strip() or len(family) > 160
                    or not isinstance(question, str) or len(question) > 2048 or kind in {"image", "video"} and not question.strip()):
                raise ValueError("请填写完整答案和来源分组；图片还需要问题")
            suffix = Path(name).suffix.lower()
            allowed = {".mp4", ".mov", ".m4v", ".webm"} if video else ({".png", ".jpg", ".jpeg", ".webp"} if kind == "image" else {".wav"})
            if suffix not in allowed:
                raise ValueError("图片支持PNG/JPEG/WebP，声音支持完整WAV录音")
            try:
                encoded = row.get("content")
                if not isinstance(encoded, str) or len(encoded) > MAX_MEDIA_BYTES * 4 // 3 + 4:
                    raise ValueError("学习文件过大")
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, UnicodeError) as error:
                raise ValueError("学习文件编码无效") from error
            total += len(raw)
            if not raw or total > MAX_MEDIA_BYTES:
                raise ValueError("每批图片或声音合计不能超过8MiB")
            if raw in seen:
                raise ValueError("同一原文件不能重复用于训练与独立检验")
            seen.append(raw)
            family = family.strip()
            if family not in families:
                families.append(family)
            parsed.append((name, suffix, question.strip(), answer.strip(), family, raw))
        if len(families) < 2:
            raise ValueError("同一来源不能同时训练与检验，请提供至少两个独立来源")
        root = require_storage(self.runtime.memory.root).resolve()
        directory = root / "app" / "media-lessons"
        for path in (root / "app", directory):
            if path.is_symlink():
                raise ValueError("学习资料目录不能是链接")
            path.mkdir(mode=0o700, exist_ok=True)
        directory = directory / uuid.uuid4().hex
        directory.mkdir(mode=0o700)
        records = []
        for name, suffix, question, answer, family, raw in parsed:
            key = uuid.uuid4().hex
            path = directory / (key + suffix)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            record = {"id": key, "name": name, "kind": kind, "question": question, "answer": answer,
                      "family": "source-" + str(families.index(family)), "source_label": family,
                      "split": "validation" if family == families[-1] else "training",
                      "asset_path": str(path), "training_allowed": reviewed,
                      "transcript_reviewed": reviewed, "answers_reviewed": reviewed, "approved": reviewed}
            if video:
                from .video_learning import extract_frames
                frames = extract_frames(path, directory / (key + "-frames"))
                if len(frames) < 2:
                    raise ValueError("视频联合学习至少需要两张有序画面")
                record["video_frames"] = [{"asset_path": frame["path"], "timestamp_seconds": frame["timestamp_seconds"]}
                                          for frame in frames]
            records.append(record)
        return records

    def _teacher_binding(self):
        config = self.settings.snapshot()["teacher"]
        return {key: config[key] for key in ("base_url", "model", "thinking", "send_reasoning_effort")}

    def _teacher_config(self, kind):
        modality = "audio" if kind == "sound" else kind
        if modality not in self.settings.snapshot()["teacher"].get("teaching_modalities", ["text"]):
            raise ValueError("请先在教师设置的“教什么”中启用对应类型")
        if not self.settings.agreement()["accepted"]:
            raise ValueError("请先确认使用协议和资料发送许可")
        if self.teacher is None:
            raise ValueError("教师尚未配置")
        config, _ = self.teacher._config()
        if not config["model"] or (config["session_call_limit"] and self.teacher.calls >= config["session_call_limit"]):
            raise ValueError("教师模型未配置或本次调用额度已用完")

    def teacher_pending(self):
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            self._schema(db)
            ids = [row["id"] for row in db.execute("SELECT id FROM media_lessons WHERE "
                "json_extract(body,'$.state') IN ('teacher_queued','teacher_running') ORDER BY updated,id")]
        for identifier in ids:
            body = self._load(identifier)
            if body["state"] != "teacher_queued":
                continue
            try:
                self._teacher_config(body.get("teacher_modality", body["kind"]))
            except (ValueError, RuntimeError):
                continue
            return identifier
        return None

    def run_teacher_one(self, stopped):
        """One saved file per idle turn; never replay an uncertain paid request."""
        identifier = self.teacher_pending()
        if not identifier or stopped.is_set():
            return
        body = self._load(identifier)
        try:
            self._teacher_config(body.get("teacher_modality", body["kind"]))
            if (body["teacher_config"] != self._teacher_binding()
                    or body["teacher_agreement"] != self.settings.agreement()["version"]
                    or body["binding"]["storage_root"] != self.settings.snapshot()["storage_root"]):
                raise ValueError("教师或协议已改变，请重新确认资料发送")
            row = body["samples"][body["teacher_next"]]
            from .speech_learning import _read
            from .thinking import ThinkingStream
            import io
            if body["kind"] == "video":
                from .video_learning import teacher_content
                content = teacher_content(row, self.runtime.memory.root)
            else:
                path = Path(row["asset_path"])
                directory = Path(self.runtime.memory.root) / "app/media-lessons"
                if not path.is_relative_to(directory):
                    directory = Path(self.runtime.memory.root).resolve() / "app/media-lessons"
                if (not path.is_relative_to(directory)
                        or path.resolve() != Path(self.runtime.memory.root).resolve() / "app/media-lessons" / path.relative_to(directory)):
                    raise ValueError("教师资料不在本批私有保存目录")
                raw = _read(row["asset_path"], MAX_MEDIA_BYTES)
                encoded = base64.b64encode(raw).decode("ascii")
                if body["kind"] == "image":
                    from PIL import Image
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.format not in {"PNG", "JPEG", "WEBP"} or image.width * image.height > 40_000_000:
                            raise ValueError("图片格式或尺寸无效")
                        mime = Image.MIME[image.format]
                        image.verify()
                    part = {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + encoded}}
                    prompt = "依据图片回答问题；看不清或无法确定时明确说明，不猜测。问题：" + row["question"]
                else:
                    import wave
                    with wave.open(io.BytesIO(raw)) as wav:
                        if wav.getnchannels() not in (1, 2) or wav.getsampwidth() not in (1, 2, 3, 4) or wav.getnframes() <= 0:
                            raise ValueError("录音格式无效")
                        if len(wav.readframes(wav.getnframes())) != wav.getnframes() * wav.getnchannels() * wav.getsampwidth():
                            raise ValueError("录音内容不完整")
                    part = {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}
                    prompt = "只逐字转写这段录音，不添加说明；听不清时明确标注，不猜测。"
                if body.get("teacher_modality") == "video":
                    prompt = ("这是历史逐帧视频在" + str(row["timestamp_seconds"]) + "秒的抽样画面，只依据本帧回答；未提供音轨或完整连续动作。" + prompt)
                content = [{"type": "text", "text": prompt}, part]
            if stopped.is_set():
                return
            # Durable boundary before the first network attempt, not after it.
            body.update(state="teacher_running", phase="teacher_running",
                        owner={"pid": os.getpid(), "started": psutil.Process().create_time()})
            self._write(body)
            parts, usage, finish, count = [], None, None, 0
            calls_before = self.teacher.calls
            parser = ThinkingStream(lambda delta: parts.append(delta["content"]) if delta.get("content") else None)
            try:
                for delta in self.teacher.stream([{"role": "user", "content": content}],
                        2048, modality=body.get("teacher_modality", "audio" if body["kind"] == "sound" else "image")):
                    if stopped.is_set():
                        raise RuntimeError("教师请求中断")
                    text = delta.get("content", "")
                    count += len(text)
                    if count > 16384:
                        raise ValueError("教师标注过长")
                    parser.feed(text)
                    usage = delta.get("usage", usage)
                    finish = delta.get("finish_reason", finish)
                parser.finish()
            finally:
                if self.teacher.calls > calls_before:
                    self.runtime.memory.record_usage("teacher", usage or {
                        "unknown": True, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None})
            answer = "".join(parts).strip()
            if finish != "stop" or not answer or len(answer) > 4096:
                raise ValueError("教师未返回完整有效标注")
            row.update(answer=answer, teacher_generated=True, answers_reviewed=False, transcript_reviewed=False, approved=False)
            body["teacher_next"] += 1
            state = "teacher_review" if body["teacher_next"] == len(body["samples"]) else "teacher_queued"
            body.update(state=state, phase=state, owner=None, error=None)
            self._write(body)
        except Exception:
            # No provider text, private paths, credentials or raw media in public errors.
            body.update(state="teacher_interrupted" if stopped.is_set() else "teacher_failed", owner=None,
                        error="教师标注未完成；请检查设置或服务。可能已有费用，不会自动重复调用")
            self._write(body)

    def confirm_teacher(self, value):
        with self.lock:
            body = self._load(value.get("id"))
            if body["state"] != "teacher_review" or value.get("answers_reviewed") is not True:
                raise ValueError("请等待老师完成，并核对全部标注后确认")
            rows = value.get("samples")
            if not isinstance(rows, list) or len(rows) != len(body["samples"]):
                raise ValueError("标注资料不完整")
            updated = copy.deepcopy(body)
            for saved, row in zip(updated["samples"], rows):
                if (not isinstance(row, dict) or row.get("id") != saved["id"]
                        or not isinstance(row.get("answer"), str) or not row["answer"].strip() or len(row["answer"]) > 4096
                        or not isinstance(row.get("question"), str) or len(row["question"]) > 2048
                        or body["kind"] in {"image", "video"} and not row["question"].strip()):
                    raise ValueError("请核对完整答案和图片问题")
                saved.update(answer=row["answer"].strip(), question=row["question"].strip(),
                             training_allowed=True, answers_reviewed=True, transcript_reviewed=True, approved=True)
            updated.update(state="ready", phase="ready", error=None)
            self._write(updated)
            return self.public(updated)

    def start(self, lesson_id):
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("本轮学习仍在进行")
            body = self._load(lesson_id)
            if body["state"].startswith("teacher_"):
                raise ValueError("教师标注须全部完成并由用户核对后才能训练")
            if body["state"] == "running" and _owner_alive(body):
                raise RuntimeError("这批资料正在另一项真实学习任务中使用")
            config = self.settings.snapshot()
            if any(config[key] != value for key, value in body["binding"].items()):
                raise ValueError("模型或资料存储位置已改变，请重新准备本轮资料")
            if body["core_version"] != getattr(self.runtime, "active_core", None) and body["kind"] != "sound":
                raise ValueError("文字核心版本已改变，请重新准备本轮资料")
            module = self._kind(body["kind"])
            if hasattr(module, "capability"):
                capability = module.capability(self.runtime)
                if not capability["available"]:
                    raise ValueError(capability["reason"])
            # The ASR model has its own weights. Release the idle text owner first.
            if body["kind"] == "sound":
                self.runtime.stop()
            self.cancel.clear()
            previous_state, previous_job = self.runtime.state, self.runtime.job
            body.update(state="running", phase="preparing", error=None,
                        owner={"pid": os.getpid(), "started": psutil.Process().create_time()})
            self.runtime.job = {"id": body["id"], "kind": "media_learning", "media_kind": body["kind"],
                                "state": "queued", "phase": "preparing", "step": 0, "steps": body["steps"]}
            self.runtime.state = "learning"
            try:
                self._write(body)
                self.worker = threading.Thread(target=self._run, args=(body,), daemon=True, name="luban-media-learner")
                self.worker.start()
            except Exception as error:
                self.worker = None
                self.runtime.state, self.runtime.job = previous_state, previous_job
                body.update(state="error", phase="error", error=str(error), owner=None)
                self._write(body)
                raise
            return self.public(body)

    def _run(self, body):
        previous_step, previous_total = body["step"], body["total_adam_step"]
        last_saved = {"candidate_path": body.get("candidate"), "completed_steps": previous_step,
                      "total_adam_step": previous_total}
        def valid_progress(result):
            start = previous_total - (previous_step if previous_step < body["steps"] else 0)
            reference = result.get("candidate_path") if isinstance(result, dict) else None
            if not isinstance(reference, str) or not reference:
                return False
            if body["kind"] == "sound":
                path = Path(reference)
                if path.name != "candidate.json" or path.resolve() != path or not path.is_relative_to(
                        Path(self.runtime.memory.root) / "modules" / "speech-learning"):
                    return False
            elif not re.fullmatch(r"[a-f0-9]{12}", reference):
                return False
            return (isinstance(result, dict) and result.get("state") in {"awaiting_quality", "paused"}
                    and result.get("round_budget") == body["steps"]
                    and isinstance(result.get("candidate_path"), str) and bool(result["candidate_path"])
                    and type(result.get("completed_steps")) is int
                    and 0 <= result["completed_steps"] <= body["steps"]
                    and type(result.get("total_adam_step")) is int
                    and result["total_adam_step"] == start + result["completed_steps"]
                    and result["total_adam_step"] >= previous_total
                    and not result.get("adopted")
                    and (result["state"] != "awaiting_quality" or result["completed_steps"] == body["steps"]))
        def progress(value):
            nonlocal last_saved
            with self.lock:
                if self.runtime.job.get("cancel_requested"):
                    self.cancel.set()
                if "steps" in value and value["steps"] != body["steps"]:
                    raise RuntimeError("本轮训练预算不能在执行中改变")
                if "step" in value and (type(value["step"]) is not int or not 0 <= value["step"] <= body["steps"]):
                    raise RuntimeError("本轮实际更新进度无效")
                saved = value.get("candidate")
                if saved is not None:
                    if not valid_progress(saved):
                        raise RuntimeError("原生检查点进度无效")
                    last_saved = saved
                    body.update(candidate=saved["candidate_path"], checkpoint_step=saved["completed_steps"],
                                checkpoint_total=saved["total_adam_step"])
                shown = {key: value[key] for key in ("phase", "step") if key in value}
                body.update(shown)
                self.runtime.job.update(shown, state="running")
                self._write(body)
        try:
            with self.runtime.lock:
                module = self._kind(body["kind"])
                owner = self.settings if body["kind"] == "sound" else self.runtime
                records = (module.prepare_demonstrations(self.settings.home, body["demonstrations"])
                           if body["kind"] == "operation" else body["samples"])
                if body["kind"] == "operation":
                    saved = [{key: row[key] for key in ("id", "kind", "split", "family", "question", "answer", "provenance")}
                             for row in records]
                    if saved != body["samples"]:
                        raise ValueError("原操作示范已改变，请重新核对资料；旧候选保留")
                result = module.train_round(owner, records, saved_candidate=body.get("candidate"),
                                            cancel=self.cancel.is_set, progress=progress)
                # A trained candidate is neither mastery nor a published upgrade.
                if result.get("adopted"):
                    raise RuntimeError("局部学习入口不能自行发布进化版本")
                if not valid_progress(result):
                    raise RuntimeError("原生学习返回的轮次或检查点进度无效")
                body.update(state=result["state"], candidate=result.get("candidate_path"),
                            step=result["completed_steps"], total_adam_step=result["total_adam_step"],
                            checkpoint_step=result["completed_steps"], checkpoint_total=result["total_adam_step"],
                            quality_checked=result.get("quality_checked", False), protection_checked=False)
                if body["step"] == body["steps"]:
                    body["round"] += 1
                body["phase"] = "paused" if body["state"] == "paused" else "awaiting_quality"
                self.runtime.job.update(state="paused" if body["state"] == "paused" else "complete",
                                        phase=body["phase"], step=body["step"], result={"promoted": False})
        except Exception as error:
            paused = isinstance(error, TrainingPaused)
            body.update(state="paused" if paused else "error", error=None if paused else str(error),
                        phase="paused" if paused else "error")
            saved = getattr(error, "candidate", None)
            if not valid_progress(saved):
                saved = last_saved
            if isinstance(saved, dict):
                body.update(candidate=saved["candidate_path"], step=saved["completed_steps"],
                            total_adam_step=saved["total_adam_step"], checkpoint_step=saved["completed_steps"],
                            checkpoint_total=saved["total_adam_step"])
            self.runtime.job.update(state=body["state"], error=body["error"], step=body["step"])
        finally:
            with self.lock:
                body["owner"] = None
                try:
                    self._write(body)
                except Exception as error:
                    self.runtime.state, self.runtime.error = "error", "学习状态保存失败：" + str(error)
                    self.runtime.job.update(state="error", error=self.runtime.error)
                finally:
                    if self.runtime.state != "error":
                        self.runtime.state = "ready" if getattr(self.runtime, "model", None) is not None else "stopped"

    def pause(self):
        self.cancel.set()
        return {"ok": True}

    def close(self):
        self.cancel.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=30)
            if self.worker.is_alive():
                raise RuntimeError("原生学习尚未关闭，保留候选后再退出")
