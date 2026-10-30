"""Private local application settings. Secrets never travel in public status."""
from __future__ import annotations

import copy
import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from .agreement import AGREEMENT_VERSION, agreement_status
from .context import PRESETS, MAX_OUTPUT_TOKENS

LEGACY_MODEL_PATH = "/Volumes/移动数据库/lubanllm/models/Qwen3-1.7B"
TEACHING_MODALITIES = ("text", "image", "video", "audio")

DEFAULTS = {
    "model_path": "/Volumes/移动数据库/lubanllm/models/luban0",
    "speech_model_path": "/Volumes/移动数据库/lubanllm/models/Qwen3-ASR-0.6B-luban0-v1",
    "storage_root": "/Volumes/移动数据库/lubanllm",
    "local_thinking": "off",
    "native_inference_precision": "bf16",
    "native_q4_profile": "",
    "native_mlx_model_path": "",
    "native_skip_layer_mask": 0,
    "native_exit_after": 24,
    "native_layer_paging": False,
    "max_new_tokens": 0,
    "context_tokens": 262144,
    "context_mode": "auto",
    "temperature": 0.6,
    "agreement": {"version": "", "accepted_at": None},
    "automatic": {"mode": "manual", "scheduled_time": "02:00", "learning_enabled": True},
    "teacher": {"name": "", "base_url": "", "model": "", "thinking": "default", "answer_mode": "local_first",
                "send_reasoning_effort": False, "timeout_seconds": 120,
                "session_call_limit": 0, "teaching_modalities": ["text"]},
}


def teaching_modalities(value):
    if (not isinstance(value, list) or any(not isinstance(item, str) or item not in TEACHING_MODALITIES for item in value)
            or len(set(value)) != len(value)):
        raise ValueError("教师教学范围需要不重复的文字、图片、视频或音频列表")
    return [item for item in TEACHING_MODALITIES if item in value]


def native_q4_profile(precision, value):
    if not isinstance(value, str):
        raise ValueError("Q4 转换文件路径需要字符串")
    if precision == "mixed-q4-g16-packed":
        if not Path(value).is_absolute() or not Path(value).is_file():
            raise ValueError("Q4 转换文件必须是已存在文件的绝对路径")
    elif value:
        raise ValueError("Q4 转换文件仅用于 packed Q4 精度")
    return value


def native_execution_policy(skip_mask, exit_after, layer_paging):
    if type(skip_mask) is not int or not 0 <= skip_mask < 1 << 24 or skip_mask & 1:
        raise ValueError("原生跳层掩码必须是24位非负整数且保留第0层")
    if type(exit_after) is not int or not 1 <= exit_after <= 24:
        raise ValueError("原生退出层数必须是1至24的整数")
    if skip_mask >> exit_after:
        raise ValueError("原生跳层掩码不能包含退出位置及之后的层")
    if type(layer_paging) is not bool:
        raise ValueError("原生逐层读取必须是布尔值")
    return skip_mask, exit_after, layer_paging


def default_home():
    if os.uname().sysname == "Darwin":
        return Path.home() / "Library/Application Support/Luban"
    return Path.home() / ".local/share/luban"


def integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("需要整数")
    return int(value)


def valid_key(value):
    return isinstance(value, str) and bool(value) and all(33 <= ord(c) <= 126 for c in value)


def private_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.chmod(temp, 0o600)
    os.replace(temp, path)


class Settings:
    def __new__(cls, home=None):
        from .bun_bridge import enabled, RemoteSettings
        if cls is Settings and enabled():
            return RemoteSettings(home)
        return super().__new__(cls)

    def __init__(self, home=None):
        self.home = Path(home or default_home())
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.home, 0o700)
        self.path = self.home / "settings.json"
        self.secret_path = self.home / "credentials.json"
        self.lock = threading.RLock()
        self.data = copy.deepcopy(DEFAULTS)
        if not self.path.exists():
            root = self.home / "data"
            self.data.update(storage_root=str(root), model_path=str(root / "models/metadata"),
                             speech_model_path=str(root / "models/speech"),
                             native_mlx_model_path=str(root / "models/mlx"), native_inference_precision="mlx-q4")
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            self.data.update({k: v for k, v in saved.items() if k != "teacher"})
            automatic = saved.get("automatic", {})
            self.data["automatic"] = {**DEFAULTS["automatic"], **({k: v for k, v in automatic.items()
                if k in DEFAULTS["automatic"]} if isinstance(automatic, dict) else {})}
            if type(self.data["automatic"]["learning_enabled"]) is not bool:
                self.data["automatic"]["learning_enabled"] = True
            self.data["teacher"].update({k: v for k, v in saved.get("teacher", {}).items()
                                         if k in DEFAULTS["teacher"]})
            if (saved.get("model_path") == LEGACY_MODEL_PATH
                    and (Path(DEFAULTS["model_path"]) / "config.json").is_file()):
                self.data["model_path"] = DEFAULTS["model_path"]
            if (saved.get("context_mode") not in {"auto", "manual"}
                    or type(saved.get("context_tokens")) is not int
                    or saved["context_tokens"] not in PRESETS):
                self.data.update(context_mode="auto", context_tokens=262144)
            elif self.data["context_mode"] == "auto":
                self.data["context_tokens"] = 262144
        self.data["teacher"]["teaching_modalities"] = teaching_modalities(self.data["teacher"]["teaching_modalities"])
        self.credentials = (json.loads(self.secret_path.read_text()) if self.secret_path.exists()
                            else {"access_key": "lb-" + secrets.token_urlsafe(32),
                                  "ui_token": secrets.token_urlsafe(32), "teacher_key": ""})
        self._persist()

    def _persist(self):
        private_json(self.path, self.data)
        private_json(self.secret_path, self.credentials)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.data)

    def public(self):
        result = self.snapshot()
        result["teacher"]["key_configured"] = bool(self.secret("teacher_key"))
        result["access_key_configured"] = bool(self.secret("access_key"))
        return result

    def secret(self, name):
        with self.lock:
            return self.credentials.get(name, "")

    def agreement(self):
        return agreement_status(self.snapshot())

    def accept_agreement(self):
        with self.lock:
            if not self.agreement()["accepted"]:
                data = self.snapshot()
                data["agreement"] = {"version": AGREEMENT_VERSION,
                                     "accepted_at": datetime.now(timezone.utc).isoformat()}
                private_json(self.path, data)
                self.data = data
            return self.agreement()

    def validated(self, changes):
        if not isinstance(changes, dict):
            raise ValueError("设置需要 JSON 对象")
        with self.lock:
            data = self.snapshot()
            credentials = self.credentials.copy()
            for key in ["model_path", "speech_model_path", "storage_root"]:
                if key in changes:
                    if not isinstance(changes[key], str) or not Path(changes[key]).is_absolute():
                        raise ValueError("模型和存储目录必须是绝对路径")
                    data[key] = changes[key]
            for key, choices in {"local_thinking": {"off", "low", "medium", "high"}}.items():
                if key in changes:
                    if changes[key] not in choices:
                        raise ValueError("无效的思考程度")
                    data[key] = changes[key]
            if "native_inference_precision" in changes:
                if changes["native_inference_precision"] not in {"mlx-q4", "bf16", "mixed-q4-g32", "mixed-q4-g16", "mixed-q4-g16-packed", "mixed-q8"}:
                    raise ValueError("原生推理精度无效")
                data["native_inference_precision"] = changes["native_inference_precision"]
                if data["native_inference_precision"] != "mixed-q4-g16-packed":
                    data["native_q4_profile"] = ""
            if "native_mlx_model_path" in changes:
                value = changes["native_mlx_model_path"]
                if not isinstance(value, str) or value and not Path(value).is_absolute():
                    raise ValueError("MLX 模型目录必须是绝对路径")
                data["native_mlx_model_path"] = value
            if data["native_inference_precision"] == "mlx-q4" and not data["native_mlx_model_path"]:
                raise ValueError("MLX 推理需要指定模型目录")
            if "native_q4_profile" in changes:
                data["native_q4_profile"] = changes["native_q4_profile"]
            if "native_inference_precision" in changes or "native_q4_profile" in changes:
                data["native_q4_profile"] = native_q4_profile(data["native_inference_precision"], data["native_q4_profile"])
            execution_keys = ("native_skip_layer_mask", "native_exit_after", "native_layer_paging")
            if any(key in changes for key in execution_keys):
                for key in execution_keys:
                    if key in changes:
                        data[key] = changes[key]
                native_execution_policy(*(data[key] for key in execution_keys))
            for key, bounds in {"max_new_tokens": (16, MAX_OUTPUT_TOKENS)}.items():
                if key in changes:
                    value = integer(changes[key])
                    if value != 0 and not bounds[0] <= value <= bounds[1]:
                        raise ValueError(f"{key} 超出范围 {bounds}")
                    data[key] = value
            if "context_mode" in changes:
                if changes["context_mode"] not in {"auto", "manual"}:
                    raise ValueError("上下文请选择自动或手动档")
                data["context_mode"] = changes["context_mode"]
            if "context_tokens" in changes:
                value = integer(changes["context_tokens"])
                if value not in PRESETS:
                    raise ValueError("手动上下文请选择64K、128K或256K")
                data["context_tokens"] = value
                if "context_mode" not in changes:
                    data["context_mode"] = "manual"
            if data["context_mode"] == "auto":
                data["context_tokens"] = 262144
            if "temperature" in changes:
                if isinstance(changes["temperature"], bool):
                    raise ValueError("温度需要数值")
                value = float(changes["temperature"])
                if not 0 <= value <= 2:
                    raise ValueError("温度需要在 0 到 2 之间")
                data["temperature"] = value
            if "automatic" in changes:
                change = changes["automatic"]
                if (not isinstance(change, dict) or set(change) not in ({"learning_enabled"},
                        {"mode", "scheduled_time"}, {"mode", "scheduled_time", "learning_enabled"})):
                    raise ValueError("进化安排需要手动/自动及有效的24小时时间")
                value = {**data["automatic"], **change}
                if (type(value["learning_enabled"]) is not bool or value["mode"] not in {"manual", "auto"}
                        or not isinstance(value["scheduled_time"], str)
                        or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value["scheduled_time"])):
                    raise ValueError("进化安排需要手动/自动及有效的24小时时间")
                data["automatic"] = value.copy()
            if "teacher" in changes:
                teacher = changes["teacher"]
                if not isinstance(teacher, dict):
                    raise ValueError("教师设置需要 JSON 对象")
                for key in data["teacher"]:
                    if key in teacher:
                        data["teacher"][key] = teacher[key]
                for key in ("name", "base_url", "model", "thinking"):
                    if not isinstance(data["teacher"][key], str):
                        raise ValueError(f"教师 {key} 需要文本")
                if re.search(r"[\x00-\x1f\x7f-\x9f]", data["teacher"]["name"]):
                    raise ValueError("教师名称不能包含控制字符")
                data["teacher"]["name"] = re.sub(r"^[\s\ufeff]+|[\s\ufeff]+$", "", data["teacher"]["name"])
                if len(data["teacher"]["name"]) > 60:
                    raise ValueError("教师名称不能超过60个字符")
                for key in ("send_reasoning_effort",):
                    if not isinstance(data["teacher"][key], bool):
                        raise ValueError(f"教师 {key} 需要布尔值")
                url = data["teacher"]["base_url"].strip().rstrip("/")
                if url:
                    parsed = urlsplit(url)
                    if parsed.username or parsed.password or parsed.query or parsed.fragment:
                        raise ValueError("教师地址不能包含密码、查询参数或片段")
                    if parsed.scheme not in {"http", "https"}:
                        raise ValueError("教师地址需要使用 HTTP 或 HTTPS")
                    if not parsed.hostname:
                        raise ValueError("教师地址缺少域名")
                data["teacher"]["base_url"] = url
                if not isinstance(data["teacher"]["answer_mode"], str) or data["teacher"]["answer_mode"] not in {"local_first", "teacher_first", "auto"}:
                    raise ValueError("教师回答模式无效")
                data["teacher"]["teaching_modalities"] = teaching_modalities(data["teacher"]["teaching_modalities"])
                if data["teacher"]["thinking"] not in {"default", "none", "minimal", "low", "medium", "high"}:
                    raise ValueError("教师思考程度无效")
                if not 0 <= integer(data["teacher"]["session_call_limit"]) <= 1000:
                    raise ValueError("单次启动的教师调用上限应为0（不限）或1–1000")
                if not 5 <= integer(data["teacher"]["timeout_seconds"]) <= 600:
                    raise ValueError("教师超时应为 5–600 秒")
                for key in ("session_call_limit", "timeout_seconds"):
                    data["teacher"][key] = int(data["teacher"][key])
                if "api_key" in teacher and teacher["api_key"]:
                    if not valid_key(teacher["api_key"]):
                        raise ValueError("教师密钥需要不含换行的文本")
                    credentials["teacher_key"] = teacher["api_key"]
                if teacher.get("clear_key"):
                    credentials["teacher_key"] = ""
            if "access_key" in changes:
                if not valid_key(changes["access_key"]) or len(changes["access_key"]) < 16:
                    raise ValueError("本地接口密钥至少 16 个字符")
                credentials["access_key"] = changes["access_key"]
            return data, credentials

    def update(self, changes):
        with self.lock:
            self.data, self.credentials = self.validated(changes)
            self._persist()
            return self.public()


def require_storage(path):
    root = Path(path)
    if str(root).startswith("/Volumes/") and not (Path("/Volumes") / root.parts[2]).is_mount():
        raise RuntimeError("存储盘未连接，请连接后重试")
    root.mkdir(parents=True, exist_ok=True)
    return root
