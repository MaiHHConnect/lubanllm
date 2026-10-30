"""Local, bounded WAV transcription; ASR weights are resident for one request only."""
from __future__ import annotations

import gc
import struct
import threading
import time
from pathlib import Path

import torch
from .settings import require_storage

DEFAULT_SPEECH_MODEL_PATH = "/Volumes/移动数据库/lubanllm/models/Qwen3-ASR-0.6B-luban0-v1"
MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
SAMPLE_RATE = 16000


def decode_wav(data: bytes):
    """Decode mono integer PCM/float32 RIFF WAV and resample to 16 kHz in memory."""
    if not isinstance(data, bytes) or not 44 <= len(data) <= MAX_AUDIO_BYTES:
        raise ValueError("请上传不超过8 MiB的单声道WAV音频")
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE" or struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("WAV头或文件长度无效")
    fmt = samples = None
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("WAV块头不完整")
        tag, size = struct.unpack_from("<4sI", data, offset)
        offset += 8
        end = offset + size
        if end + (size % 2) > len(data):
            raise ValueError("WAV音频块不完整")
        if tag == b"fmt ":
            if fmt is not None or size < 16:
                raise ValueError("WAV格式块无效")
            fmt = struct.unpack_from("<HHIIHH", data, offset)
        elif tag == b"data":
            if samples is not None:
                raise ValueError("WAV只能包含一个音频块")
            samples = memoryview(data)[offset:end]
        offset = end + (size % 2)
    if fmt is None or samples is None:
        raise ValueError("WAV缺少格式或音频块")
    encoding, channels, rate, byte_rate, block_align, bits = fmt
    width = bits // 8
    if channels != 1 or not 8000 <= rate <= 48000:
        raise ValueError("仅支持8–48 kHz单声道WAV")
    if not ((encoding == 1 and bits in (8, 16, 24, 32)) or (encoding == 3 and bits == 32)):
        raise ValueError("仅支持PCM 8/16/24/32位或32位浮点WAV")
    if block_align != width or byte_rate != rate * width or not samples or len(samples) % width:
        raise ValueError("WAV样本长度或采样格式无效")
    frames = len(samples) // width
    duration = frames / rate
    if not 0.05 <= duration <= MAX_AUDIO_SECONDS:
        raise ValueError("音频时长应在0.05到30秒之间")
    # Copy the read-only incoming buffer before exposing it to PyTorch.
    buffer = bytearray(samples)
    if bits == 24:
        parts = torch.frombuffer(buffer, dtype=torch.uint8).reshape(-1, 3).to(torch.int32)
        audio = parts[:, 0] | (parts[:, 1] << 8) | (parts[:, 2] << 16)
        audio = ((audio ^ 0x800000) - 0x800000).to(torch.float32) / 8388608
    else:
        dtype = torch.float32 if encoding == 3 else {8: torch.uint8, 16: torch.int16, 32: torch.int32}[bits]
        audio = torch.frombuffer(buffer, dtype=dtype).to(torch.float32)
        if encoding == 1:
            audio = (audio - 128) / 128 if bits == 8 else audio / (2 ** (bits - 1))
    if not torch.isfinite(audio).all() or (audio.abs() > 1).any():
        raise ValueError("音频样本必须是有限值且处于[-1,1]范围")
    if rate != SAMPLE_RATE:
        audio = torch.nn.functional.interpolate(audio[None, None], size=round(duration * SAMPLE_RATE),
                                                mode="linear", align_corners=False)[0, 0]
    return audio.contiguous().numpy(), duration


class SpeechRuntime:
    def __init__(self, model_path=DEFAULT_SPEECH_MODEL_PATH):
        self.model_path = Path(model_path)
        self.lock = threading.Lock()
        self.model = self.processor = None
        self.state, self.error = "unloaded", ""

    def status(self):
        return {"state": self.state, "error": self.error, "model": "Qwen3-ASR-0.6B",
                "available": (self.model_path / "config.json").is_file()
                and (self.model_path / "model.safetensors").is_file(),
                "device": "cpu", "precision": "float32", "max_seconds": MAX_AUDIO_SECONDS}

    def _load(self):
        if str(self.model_path).startswith("/Volumes/"):
            require_storage(Path("/Volumes") / self.model_path.parts[2])
        if not self.status()["available"]:
            raise RuntimeError("找不到本地语音识别权重，请检查模型目录和存储盘")
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(str(self.model_path), local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            str(self.model_path), dtype=torch.float32, local_files_only=True,
            trust_remote_code=False, attn_implementation="sdpa").eval().requires_grad_(False)

    def transcribe_bytes(self, data: bytes, language: str | None = None):
        audio, duration = decode_wav(data)
        # Use the model's native language validator before loading large weights.
        from transformers.models.qwen3_asr.processing_qwen3_asr import resolve_language
        language = resolve_language(language) if language is None or isinstance(language, str) else ""
        if language == "":
            raise ValueError("语种必须是语言代码或名称")
        with self.lock:
            started = time.monotonic()
            inputs = output = generated = None
            self.state, self.error = "loading", ""
            try:
                self._load()
                self.state = "transcribing"
                with torch.inference_mode():
                    inputs = self.processor.apply_transcription_request(
                        audio=audio, language=language, processor_kwargs={"sampling_rate": SAMPLE_RATE})
                    output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
                    generated = output[0, inputs["input_ids"].shape[1]:]
                    eos = self.model.generation_config.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    if generated.numel() >= 512 and int(generated[-1]) not in eos:
                        raise RuntimeError("语音转写达到输出上限，请缩短音频后重试")
                    parsed = self.processor.decode(generated, return_format="parsed")
                return {"text": parsed["transcription"], "language": parsed["language"],
                        "duration_seconds": duration, "elapsed_seconds": time.monotonic() - started,
                        "model": "Qwen3-ASR-0.6B", "device": "cpu", "precision": "float32"}
            except Exception as error:
                self.error = str(error)
                raise
            finally:
                self.model = self.processor = inputs = output = generated = None
                gc.collect()
                self.state = "unloaded"
