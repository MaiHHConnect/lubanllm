"""OpenAI-compatible external teacher; no calls until explicitly configured."""
from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import httpx

MAX_MEDIA_BYTES = 8 * 1024 * 1024
MAX_MEDIA_PARTS = 4
IMAGE_DATA = re.compile(r"data:image/(?:png|jpeg|webp);base64,([A-Za-z0-9+/]*={0,2})\Z")


def _validate_teaching(messages, modality, allowed):
    """Check the actual wire content before reserving a paid attempt."""
    if not isinstance(modality, str) or modality not in {"text", "image", "audio", "video"}:
        raise ValueError("教师教学类型无效")
    if modality not in allowed:
        raise ValueError("此类型的教师教学已关闭")
    if not isinstance(messages, list) or not messages:
        raise ValueError("教师消息无效")
    found, media_count, media_bytes = set(), 0, 0
    for message in messages:
        if (not isinstance(message, dict) or set(message) - {"role", "content", "name"}
                or not isinstance(message.get("role"), str)
                or message["role"] not in {"system", "user", "assistant", "developer"}):
            raise ValueError("教师消息格式无效")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not content:
            raise ValueError("教师消息内容无效")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("教师内容项无效")
            kind = part.get("type")
            if not isinstance(kind, str):
                raise ValueError("教师内容项类型无效")
            if kind == "text":
                if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                    raise ValueError("教师文字内容无效")
                continue
            if kind in {"video", "video_url", "input_video"}:
                raise ValueError("视频教师仅接收带时间戳说明的本地画面帧，不接收视频 URL 或音轨")
            if message["role"] != "user":
                raise ValueError("媒体只能放在用户教师消息中")
            if kind == "image_url" and set(part) == {"type", "image_url"}:
                value = part["image_url"]
                if (not isinstance(value, dict) or set(value) - {"url", "detail"}
                        or not isinstance(value.get("detail", "auto"), str)
                        or value.get("detail", "auto") not in {"auto", "low", "high"}):
                    raise ValueError("教师图片内容无效")
                match = IMAGE_DATA.fullmatch(value.get("url", "")) if isinstance(value.get("url"), str) else None
                if match is None:
                    raise ValueError("教师图片仅支持本地 PNG、JPEG、WebP base64 内容")
                encoded, actual = match[1], "image"
            elif kind == "input_audio" and set(part) == {"type", "input_audio"}:
                value = part["input_audio"]
                if (not isinstance(value, dict) or set(value) != {"data", "format"}
                        or not isinstance(value["format"], str)
                        or value["format"] not in {"wav", "mp3"} or not isinstance(value["data"], str)):
                    raise ValueError("教师录音仅支持 WAV 或 MP3 base64 内容")
                encoded, actual = value["data"], "audio"
            else:
                raise ValueError("教师内容项类型无效")
            # Video is a bounded frame sequence; its text timestamps and no-audio
            # explanation are supplied by the media lesson entry, not custom API fields.
            if modality == "video" and actual == "image":
                actual = "video"
            if actual != modality or actual not in allowed:
                raise ValueError("教师内容与已启用的教学类型不一致")
            media_count += 1
            if media_count > MAX_MEDIA_PARTS or len(encoded) > 4 * ((MAX_MEDIA_BYTES + 2) // 3):
                raise ValueError("教师媒体内容超过限制")
            try:
                size = len(base64.b64decode(encoded, validate=True))
            except (ValueError, binascii.Error):
                raise ValueError("教师媒体 base64 无效") from None
            media_bytes += size
            if not size or media_bytes > MAX_MEDIA_BYTES:
                raise ValueError("教师媒体为空或合计超过 8 MiB")
            found.add(actual)
    if modality != "text" and found != {modality}:
        raise ValueError("教师请求缺少对应媒体内容")


class Teacher:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport
        self.calls = 0
        self.lock = threading.Lock()

    def _config(self, draft=None):
        config = self.settings.snapshot()["teacher"]
        key = self.settings.secret("teacher_key")
        if draft is not None:
            data, credentials = self.settings.validated({"teacher": draft})
            if data["teacher"]["base_url"] != config["base_url"] and not draft.get("api_key"):
                raise ValueError("测试新教师地址时请重新填写密钥")
            config, key = data["teacher"], credentials["teacher_key"]
        if not config["base_url"] or not key:
            raise ValueError("请先设置教师 API 地址与密钥")
        return config, {"Authorization": "Bearer " + key}

    def models(self, draft=None):
        config, headers = self._config(draft)
        with httpx.Client(timeout=config["timeout_seconds"], transport=self.transport, follow_redirects=False) as client:
            response = client.get(config["base_url"] + "/models", headers=headers)
            self._check(response)
            try:
                models = [str(item["id"]) for item in response.json()["data"] if isinstance(item, dict) and "id" in item]
            except (ValueError, KeyError, TypeError):
                raise RuntimeError("教师未返回兼容的模型列表，请手动填写模型名称后测试") from None
            if not models:
                raise RuntimeError("教师模型列表为空，请手动填写模型名称后测试")
            return [self.redact(name, headers) for name in models]

    @staticmethod
    def redact(text, headers):
        if not isinstance(text, str):
            raise RuntimeError("教师返回了无效的文本内容")
        return text.replace(headers["Authorization"][7:], "[redacted]")

    @staticmethod
    def _check(response):
        if response.status_code >= 400 or 300 <= response.status_code < 400:
            # Do not surface provider bodies: they may contain submitted secrets/text.
            hints = {401: "密钥无效或已失效", 403: "密钥没有访问权限", 404: "路径不存在，请检查兼容 API 地址是否需要 /v1；也可手动填写模型后测试",
                     429: "请求受限，请检查额度或稍后重试"}
            raise RuntimeError(f"教师接口返回 HTTP {response.status_code}：" + hints.get(response.status_code, "请检查地址、服务状态及模型设置"))

    def stream(self, messages, max_tokens, thinking=None, draft=None, modality="text"):
        config, headers = self._config(draft)
        _validate_teaching(messages, modality, config.get("teaching_modalities", ["text"]))
        if not config["model"]:
            raise ValueError("请选择教师模型")
        with self.lock:
            if config["session_call_limit"] and self.calls >= config["session_call_limit"]:
                raise RuntimeError("本次启动的教师调用额度已用完，可在设置中调整")
            self.calls += 1
        payload = {"model": config["model"], "messages": messages,
                   "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens}
        effort = thinking or config["thinking"]
        if config["send_reasoning_effort"] and effort != "default":
            payload["reasoning_effort"] = effort
        with httpx.Client(timeout=config["timeout_seconds"], transport=self.transport, follow_redirects=False) as client:
            with client.stream("POST", config["base_url"] + "/chat/completions", headers=headers, json=payload) as response:
                self._check(response)
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    data = json.loads(response.read())
                    message = data["choices"][0]["message"]
                    if message.get("reasoning_content"):
                        yield {"reasoning_content": self.redact(message["reasoning_content"], headers)}
                    yield {"content": self.redact(message.get("content") or "", headers)}
                    if data["choices"][0].get("finish_reason"):
                        yield {"finish_reason": data["choices"][0]["finish_reason"]}
                    if data.get("usage"):
                        yield {"usage": data["usage"]}
                    return
                ended = False
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        ended = True
                        break
                    data = json.loads(raw)
                    if data.get("error"):
                        raise RuntimeError("教师流式响应报告错误")
                    if data.get("usage"):
                        yield {"usage": data["usage"]}
                    for choice in data.get("choices", []):
                        delta = choice.get("delta", {})
                        yield {k: self.redact(delta[k], headers) for k in ("content", "reasoning_content") if delta.get(k)}
                        if choice.get("finish_reason"):
                            ended = True
                            yield {"finish_reason": choice["finish_reason"]}
                if not ended:
                    yield {"finish_reason": "incomplete"}
