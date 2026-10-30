"""Local image-only Qwen3.5 inference; never routes image material into learning."""
from __future__ import annotations

import base64
from contextlib import ExitStack
import io
import json
from pathlib import Path
import re
import time
import warnings
import zipfile
import xml.etree.ElementTree as ET

import torch
from PIL import Image, ImageOps
from safetensors import safe_open

from .coldpaging import full_backbone
from .context import output_budget, window as context_window, available_memory_bytes

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_TOTAL_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16 * 1024 * 1024
MAX_IMAGES = 4
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 1048576
IMAGE_URL = re.compile(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/]*={0,2})\Z")


def has_images(messages):
    return any(isinstance(message["content"], list) and any(
        part.get("type") == "image_url" for part in message["content"]) for message in messages)


def image_bytes(url):
    match = IMAGE_URL.fullmatch(url) if isinstance(url, str) else None
    if match is None:
        raise ValueError("图片仅支持本地 PNG、JPEG、WebP 的 base64 data URL")
    if len(match[2]) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("单张图片不能超过 4 MiB")
    try:
        raw = base64.b64decode(match[2], validate=True)
    except ValueError as error:
        raise ValueError("图片 base64 无效") from error
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("图片为空或超过 4 MiB")
    return raw, {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[match[1]]


def open_image(url):
    raw, expected_format = image_bytes(url)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != expected_format or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("图片格式不符或解码后超过 16 百万像素")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("请使用静态图片，当前不支持动画或视频")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("图片损坏或解码尺寸过大") from error


def image_processor(path):
    # The installed AutoProcessor also requires a video processor / torchvision.
    # Use the installed official PIL implementation for this image-only route.
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    return Qwen2VLImageProcessorPil.from_pretrained(str(path), local_files_only=True)


def capabilities(runtime, settings):
    path = Path(getattr(runtime, "loaded_path", None) or settings.snapshot()["model_path"])
    reason = ""
    try:
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "qwen3_5" or not config.get("vision_config"):
            raise ValueError("当前基座不支持本地图片")
        image_processor(path)
    except (OSError, ValueError, ImportError, TypeError) as error:
        reason = "本地图片处理暂不可用：" + str(error)
    return {"images": not reason, "image_reason": reason,
            "image_engine": ("mlx-q4-vision" if getattr(runtime, "native_inference_precision", None) == "mlx-q4"
                             else "torch-hf-qwen3.5-image-only"), "image_model_verified": False,
            "image_routes": ["auto", "local"],
            "limits": {"max_images": MAX_IMAGES, "max_image_bytes": MAX_IMAGE_BYTES,
                       "max_image_total_bytes": MAX_IMAGE_TOTAL_BYTES,
                       "max_image_pixels": MAX_IMAGE_PIXELS, "max_image_tokens": 1024,
                       "max_document_bytes": MAX_DOCUMENT_BYTES, "max_text_chars": MAX_TEXT_CHARS},
            "documents": {"docx": True, "pdf": False}}


def image_inputs(tokenizer, processor, messages, thinking):
    template_messages, images = [], []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "image_url":
                    images.append(open_image(part["image_url"]["url"]))
                    parts.append({"type": "image"})
                else:
                    parts.append(part)
            content = parts
        template_messages.append({"role": message["role"], "content": content})
    text = tokenizer.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking != "off")
    visual = processor(images, return_tensors="pt", min_pixels=65536, max_pixels=1048576)
    marker = "<|image_pad|>"
    pieces = text.split(marker)
    if len(pieces) != len(images) + 1:
        raise ValueError("图片占位符与实际图片不一致，请勿输入内部视觉标记")
    # This is Qwen3VLProcessor.replace_image_token's upstream grid / merge rule.
    counts = visual["image_grid_thw"].prod(-1) // processor.merge_size ** 2
    text = pieces[0] + "".join(marker * int(count) + suffix for count, suffix in zip(counts, pieces[1:]))
    tokens = tokenizer([text], return_tensors="pt", return_token_type_ids=False)
    image_id = tokenizer.convert_tokens_to_ids(marker)
    if int((tokens["input_ids"] == image_id).sum()) != int(counts.sum()):
        raise ValueError("图片 token 与预处理网格不一致")
    tokens["mm_token_type_ids"] = (tokens["input_ids"] == image_id).to(torch.long)
    return {**tokens, **visual}


def load_visual(runtime):
    visual = runtime.model.model.visual
    if not any(value.is_meta for value in visual.parameters()):
        return
    pager = runtime.cold_backbone
    if pager is None:
        raise RuntimeError("视觉权重缺失，请重新加载模型")
    if any(pager._stat(path) != pager.file_stats[name] for name, path in pager.files.items()):
        raise ValueError("冷视觉权重已变化，请重新加载模型")
    prefix = next(name for name, module in runtime.model.named_modules() if module is visual)
    with ExitStack() as stack:
        readers = {}
        state = {}
        for name, reference in visual.state_dict().items():
            key = prefix + "." + name
            filename = pager.weight_map.get(key)
            if filename not in pager.files:
                raise ValueError("冷视觉权重缺少模型图张量：" + key)
            if filename not in readers:
                readers[filename] = stack.enter_context(safe_open(pager.files[filename], framework="pt", device="cpu"))
            value = readers[filename].get_tensor(key)
            if value.shape != reference.shape:
                raise ValueError("冷视觉权重尺寸不符：" + key)
            state[name] = value.to(device=runtime.device, dtype=reference.dtype)
        visual.load_state_dict(state, strict=True, assign=True)
    for name, value in pager.vision_buffers.items():
        owner, _, attribute = name.rpartition(".")
        module = visual.get_submodule(owner) if owner else visual
        module._buffers[attribute] = value.to(runtime.device)
    visual.eval().requires_grad_(False)


def generate_images(runtime, messages, emit, stop, *, thinking, max_tokens, temperature):
    if getattr(runtime, "native_inference_precision", None) == "mlx-q4":
        from .native_mlx_vision import generate_images as generate_mlx_images
        return generate_mlx_images(runtime, messages, emit, stop, thinking=thinking,
                                   max_tokens=max_tokens, temperature=temperature)
    from .runtime import EmitStreamer, StopRequested
    with runtime.lock:
        runtime._ready()
        if runtime.model.config.model_type != "qwen3_5":
            raise RuntimeError("当前基座不支持图片")
        started = time.perf_counter()
        effective_thinking = "off" if runtime.active_core is not None else thinking
        inputs = image_inputs(runtime.tokenizer, image_processor(runtime.loaded_path), messages, effective_thinking)
        missing_weights = sum(value.numel() * value.element_size() for value in runtime.model.parameters() if value.is_meta)
        # Include the vision attention workspace; the existing language admission
        # estimate alone cannot admit image inference. This is not a peak measurement.
        patches = inputs["image_grid_thw"].prod(-1)
        vision_workspace = int((patches * patches).sum()) * runtime.model.config.vision_config.num_heads * 8
        status = context_window(runtime.settings.snapshot(), runtime.text_config, available_memory_bytes(),
            device=runtime.device, weight_bytes=missing_weights + vision_workspace,
            dtype_bytes=2 if runtime.dtype == torch.bfloat16 else 4)
        requested = runtime.settings.snapshot().get("max_new_tokens", 0) if max_tokens is None else max_tokens
        maximum = output_budget(inputs["input_ids"].shape[1], requested, status)
        if stop.is_set():
            raise RuntimeError("请求已停止")
        previous_state, expert = runtime.state, runtime.branch.expert
        runtime.state = "generating"
        runtime.branch.expert = None  # Text-trained route / experts have no image qualification.
        try:
            with full_backbone(runtime), torch.no_grad():
                load_visual(runtime)
                inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
                output = runtime.model.generate(**inputs, max_new_tokens=maximum,
                    do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                    top_p=0.95 if temperature > 0 else None, pad_token_id=runtime.tokenizer.eos_token_id,
                    stopping_criteria=[StopRequested(stop)], streamer=EmitStreamer(runtime.tokenizer, emit))
                count = output.shape[1] - inputs["input_ids"].shape[1]
                eos = runtime.model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, (list, tuple)) else [eos]
                complete = bool(count and int(output[0, -1]) in eos_ids and not stop.is_set())
                elapsed = time.perf_counter() - started
                usage = {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": count,
                        "total_tokens": output.shape[1], "elapsed_seconds": elapsed,
                        "tokens_per_second": count / elapsed if elapsed > 0 else None,
                        "output_limit": maximum,
                        "local_action": "local_core", "local_complete": complete, "reference_used": False,
                        "core_version": runtime.active_core, "expert_id": None,
                        "requested_thinking": thinking, "effective_thinking": effective_thinking,
                        "image_count": int(inputs["image_grid_thw"].shape[0]), "inference_engine": "torch-hf-vision",
                        "finish_reason": "stop" if stop.is_set() or complete else "length"}
                runtime.last_generation_usage = usage
                return usage
        finally:
            runtime.branch.expert = expert
            if runtime.cold_backbone is not None:
                runtime._unload_saved_expert()
            runtime.state = previous_state


def extract_document(raw, name):
    if not isinstance(name, str) or not name or len(name) > 255 or "\x00" in name:
        raise ValueError("附件文件名无效")
    if Path(name).suffix.lower() != ".docx":
        raise ValueError("当前只支持 DOCX 文字提取；PDF 请先转为文本或图片")
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("文档为空或超过 8 MiB")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if len(archive.infolist()) > 1024:
                raise ValueError("DOCX 内容条目过多")
            entries = [item for item in archive.infolist() if item.filename == "word/document.xml"]
            if len(entries) != 1 or entries[0].file_size > MAX_DOCUMENT_BYTES or entries[0].flag_bits & 1:
                raise ValueError("DOCX 主文档缺失、过大或被加密")
            with archive.open(entries[0]) as stream:
                xml = stream.read(MAX_DOCUMENT_BYTES + 1)
            if len(xml) > MAX_DOCUMENT_BYTES or b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
                raise ValueError("DOCX XML 不安全或过大")
            root = ET.fromstring(xml.decode("utf-8-sig"))
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, UnicodeError, ET.ParseError) as error:
        raise ValueError("DOCX 文件损坏，无法提取文字") from error
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    text = "\n".join("".join(node.text or "" if node.tag == ns + "t" else "\t" if node.tag == ns + "tab" else "\n"
                              for node in paragraph.iter() if node.tag in {ns + "t", ns + "tab", ns + "br"})
                     for paragraph in root.iter(ns + "p"))
    if not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError("文档没有可读取文字或文字超过限制")
    return {"text": text, "format": "docx", "scope": "document text only; embedded images are not extracted"}
