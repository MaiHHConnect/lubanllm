"""Image and document boundaries, official small graph and offline HTTP isolation."""
import base64
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image
import torch
from fastapi.testclient import TestClient
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

from luban.coldpaging import full_backbone
from luban.memory import Memory
from luban.server import create_app, validated_messages
from luban.settings import Settings
from luban.vision import (capabilities, extract_document, generate_images, image_inputs, load_visual,
                          open_image, MAX_DOCUMENT_BYTES)
from test_app import FakeRuntime, events
import test_coldpaging


def png(color="red", size=(32, 32)):
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def message(url=None):
    return {"role": "user", "content": [{"type": "text", "text": "看看图片"},
        {"type": "image_url", "image_url": {"url": url or png()}}]}


def docx(xml):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
    return raw.getvalue()


DOC = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>第一行</w:t><w:tab/><w:t>第二项</w:t></w:r></w:p><w:p><w:r><w:t>下一行</w:t></w:r></w:p></w:body></w:document>'


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    vocab = {f"v{i}": i for i in range(64)}
    for token, index in [("[UNK]", 0), ("<|image_pad|>", 60), ("<|vision_start|>", 62), ("<|vision_end|>", 63)]:
        vocab.pop(f"v{index}")
        vocab[token] = index
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="[UNK]")),
        additional_special_tokens=["<|image_pad|>", "<|vision_start|>", "<|vision_end|>"])
    tokenizer.chat_template = "{% for m in messages %}{% if m.content is string %}{{m.content}}{% else %}{% for p in m.content %}{% if p.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{p.text}}{% endif %}{% endfor %}{% endif %}{% endfor %}"
    tokenizer.eos_token = "v2"
    tokenizer.pad_token = "[UNK]"
    return tokenizer


class ImageBoundaryTests(unittest.TestCase):
    def test_text_arrays_normalize_and_chat_images_reject_without_decoding(self):
        original = [{"role": "user", "content": "文字"}, {"role": "assistant", "content": "回答"},
                    {"role": "user", "content": [{"type": "text", "text": "再看看"}]}]
        result = validated_messages({"messages": original})
        self.assertEqual(result[0], original[0])
        self.assertEqual(result[2]["content"], "再看看")
        with patch("luban.vision.open_image", side_effect=AssertionError("must reject before image decoding")), self.assertRaisesRegex(ValueError, "仅支持文字"):
            validated_messages({"messages": [message()]})

    def test_urls_invalid_base64_format_animation_and_dimensions_reject(self):
        for url in ["https://example.com/image.png", "file:///etc/passwd", "data:image/svg+xml;base64,AA==",
                    "data:image/png;base64,!", "data:image/png;base64,AAAA", png().replace("image/png", "image/jpeg")]:
            with self.subTest(url=url[:45]), self.assertRaises(ValueError):
                open_image(url)
        with patch("luban.vision.MAX_IMAGE_PIXELS", 16), self.assertRaisesRegex(ValueError, "像素"):
            open_image(png())
        data = io.BytesIO()
        Image.new("RGB", (32, 32), "red").save(data, "PNG", save_all=True,
            append_images=[Image.new("RGB", (32, 32), "blue")], duration=100)
        with self.assertRaisesRegex(ValueError, "动画"):
            open_image("data:image/png;base64," + base64.b64encode(data.getvalue()).decode())

    def test_unknown_parts_roles_and_limits_reject(self):
        for value in [[], [{"type": "text", "text": " "}], [{"type": "audio", "text": "x"}], [None]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validated_messages({"messages": [{"role": "user", "content": value}]})
        for role in ["assistant", "system"]:
            with self.assertRaises(ValueError):
                validated_messages({"messages": [{**message(), "role": role}]})
        with self.assertRaisesRegex(ValueError, "仅支持文字"):
            validated_messages({"messages": [message()] * 5})
        with self.assertRaisesRegex(ValueError, "过长"):
            validated_messages({"messages": [{"role": "user", "content": "x" * 1048577}]})

    def test_docx_text_tables_and_security_boundaries(self):
        self.assertEqual(extract_document(docx(DOC), "材料.docx")["text"], "第一行\t第二项\n下一行")
        table = DOC.replace("<w:p><w:r><w:t>下一行", "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>下一行").replace("</w:body>", "</w:tc></w:tr></w:tbl></w:body>")
        self.assertIn("下一行", extract_document(docx(table), "表格.DOCX")["text"])
        bad = ['<!DOCTYPE a [<!ENTITY x "hidden">]><a>&x;</a>', '<w:document>', '<a/>']
        for xml in bad:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                extract_document(docx(xml), "bad.docx")
        for raw, name in [(b"broken", "bad.docx"), (docx(DOC), "bad.pdf"), (docx(DOC), None), (b"", "empty.docx")]:
            with self.assertRaises(ValueError):
                extract_document(raw, name)
        with patch("luban.vision.MAX_DOCUMENT_BYTES", 16), self.assertRaises(ValueError):
            extract_document(docx(DOC), "big.docx")

    def test_official_pil_preprocessing_grid_tokens_and_real_pixels(self):
        tokenizer = tiny_tokenizer()
        processor = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2)
        first = image_inputs(tokenizer, processor, [message()], "off")
        second = image_inputs(tokenizer, processor, [message(png("blue"))], "off")
        self.assertEqual(first["image_grid_thw"].tolist(), [[1, 16, 16]])
        self.assertEqual(first["pixel_values"].shape, (256, 1536))
        self.assertEqual(int(first["mm_token_type_ids"].sum()), 64)
        self.assertTrue(torch.equal(first["input_ids"], second["input_ids"]))
        self.assertFalse(torch.equal(first["pixel_values"], second["pixel_values"]))
        two = image_inputs(tokenizer, processor, [message(), {"role": "assistant", "content": "红色"}, message(png("blue"))], "off")
        self.assertEqual(two["image_grid_thw"].shape[0], 2)
        self.assertEqual(int(two["mm_token_type_ids"].sum()), 128)


class VisionPagingTests(unittest.TestCase):
    setUp = test_coldpaging.ColdPagingTests.setUp
    assert_cold = test_coldpaging.ColdPagingTests.assert_cold

    def inputs(self):
        return {"input_ids": torch.tensor([[62, 60, 63, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 0, 0]]), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "pixel_values": torch.randn(4, 24)}

    def test_real_upstream_cpu_vision_logits_exact_after_cold_restore(self):
        inputs = self.inputs()
        with torch.no_grad():
            before = self.runtime.model(**inputs, use_cache=False).logits
            changed = self.runtime.model(**{**inputs, "pixel_values": -inputs["pixel_values"]}, use_cache=False).logits
        self.assertGreater(float((before - changed).abs().max()), 0)
        buffer = self.pager.vision_buffers["rotary_pos_emb.inv_freq"]
        self.pager._evict()
        self.assertTrue(self.pager.vision.rotary_pos_emb.inv_freq.is_meta)
        with full_backbone(self.runtime):
            load_visual(self.runtime)
            self.assertTrue(torch.equal(self.pager.vision.rotary_pos_emb.inv_freq, buffer))
            with torch.no_grad():
                after = self.runtime.model(**inputs, use_cache=False).logits
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assert_cold()
        self.assertTrue(all(value.is_meta for value in self.pager.vision.parameters()))

    def test_real_cpu_generate_from_picture_through_template_and_pixels(self):
        self.runtime._ready = lambda: None
        self.runtime.settings = SimpleNamespace(snapshot=lambda: {"context_tokens": 512, "context_mode": "manual"})
        self.runtime.text_config = self.runtime.model.config.text_config
        # Context admission uses 256-token bins; this tiny graph has no learned position table.
        self.runtime.text_config.max_position_embeddings = 512
        self.runtime.loaded_path = "fixture"
        self.runtime.state = "ready"
        self.runtime.tokenizer = tiny_tokenizer()
        self.runtime._unload_saved_expert = lambda: None
        processor = Qwen2VLImageProcessorPil(patch_size=2, temporal_patch_size=2, merge_size=2)
        # Scale pixel budget to the existing tiny graph; production uses patch16
        # and the documented 1024 image-token ceiling. Model.generate is real.
        tiny_processor = Mock(side_effect=lambda images, **kwargs: processor(images, return_tensors="pt", min_pixels=16, max_pixels=16))
        tiny_processor.merge_size = 2
        self.pager._evict()
        emitted = []
        with patch("luban.vision.image_processor", return_value=tiny_processor):
            usage = generate_images(self.runtime, [message()], emitted.append, threading.Event(),
                                    thinking="off", max_tokens=2, temperature=0)
        self.assertEqual(usage["image_count"], 1)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(usage["inference_engine"], "torch-hf-vision")
        self.assertEqual(usage["output_limit"], 2)
        self.assertGreater(usage["elapsed_seconds"], 0)
        self.assertEqual(usage["tokens_per_second"], usage["completion_tokens"] / usage["elapsed_seconds"])
        self.assertIs(self.runtime.last_generation_usage, usage)
        self.assertEqual(self.runtime.state, "ready")
        self.assert_cold()
        self.assertTrue(all(value.is_meta for value in self.pager.vision.parameters()))

    def test_memory_refusal_precedes_loading_visual_or_language_weights(self):
        self.runtime._ready = lambda: None
        self.runtime.settings = SimpleNamespace(snapshot=lambda: {"context_tokens": 512, "context_mode": "manual"})
        self.runtime.text_config = self.runtime.model.config.text_config
        # Context admission uses 256-token bins; this tiny graph has no learned position table.
        self.runtime.text_config.max_position_embeddings = 512
        self.runtime.loaded_path = "fixture"
        self.runtime.state = "ready"
        self.runtime.tokenizer = SimpleNamespace(eos_token_id=2)
        self.pager._evict()
        with patch("luban.vision.image_processor"), patch("luban.vision.image_inputs", return_value=self.inputs()), \
                patch("luban.vision.available_memory_bytes", return_value=0), \
                patch("luban.vision.full_backbone", side_effect=AssertionError("must not load weights")):
            with self.assertRaisesRegex(ValueError, "可用内存"):
                generate_images(self.runtime, [message()], lambda _: None, threading.Event(),
                                thinking="off", max_tokens=4, temperature=0)
        self.assertEqual(self.runtime.state, "ready")
        self.assert_cold()

    def test_missing_visual_weight_rejects_and_unloads(self):
        self.pager._evict()
        key = next(key for key in self.pager.weight_map if key.startswith("model.visual."))
        self.pager.weight_map.pop(key)
        with self.assertRaisesRegex(ValueError, "缺少"):
            with full_backbone(self.runtime):
                load_visual(self.runtime)
        self.assert_cold()
        self.assertTrue(all(value.is_meta for value in self.pager.vision.parameters()))

    def test_generate_failure_and_cancellation_restore_expert_and_state(self):
        self.runtime._ready = lambda: None
        self.runtime.settings = SimpleNamespace(snapshot=lambda: {"context_tokens": 512, "context_mode": "manual"})
        self.runtime.text_config = self.runtime.model.config.text_config
        # Context admission uses 256-token bins; this tiny graph has no learned position table.
        self.runtime.text_config.max_position_embeddings = 512
        self.runtime.loaded_path = "fixture"
        self.runtime.state = "ready"
        self.runtime.tokenizer = SimpleNamespace(eos_token_id=2)
        self.runtime._unload_saved_expert = lambda: None
        expert = self.runtime.branch.expert
        self.pager._evict()
        with patch("luban.vision.image_processor"), patch("luban.vision.image_inputs", return_value=self.inputs()), \
                patch.object(self.runtime.model, "generate", side_effect=RuntimeError("injected vision failure")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                generate_images(self.runtime, [message()], lambda _: None, threading.Event(),
                                thinking="off", max_tokens=4, temperature=0)
        self.assertEqual(self.runtime.state, "ready")
        self.assertIs(self.runtime.branch.expert, expert)
        self.assert_cold()
        stopped = threading.Event()
        stopped.set()
        with patch("luban.vision.image_processor"), patch("luban.vision.image_inputs", return_value=self.inputs()):
            with self.assertRaisesRegex(RuntimeError, "停止"):
                generate_images(self.runtime, [message()], lambda _: None, stopped,
                                thinking="off", max_tokens=4, temperature=0)
        self.assert_cold()


class ImageHttpTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.settings = Settings(root / "home")
        self.settings.update({"storage_root": str(root / "storage")})
        self.runtime = FakeRuntime(Memory(root / "storage"))
        self.client = TestClient(create_app(self.settings, self.runtime, autoload=False))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.post("/api/session", json={"token": self.settings.secret("ui_token")})

    def test_media_rejects_before_queue_router_teacher_or_model_in_all_chat_modes(self):
        self.runtime.choose_route = Mock(side_effect=AssertionError("text router must not run"))
        self.runtime.generate = Mock(side_effect=AssertionError("text generation must not run"))
        app = self.client.app
        bearer = {"Authorization": "Bearer " + self.settings.secret("access_key")}
        for path in ("/api/chat", "/v1/chat/completions"):
            for route in ("auto", "local", "teacher"):
                for parts in (message()["content"], [{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}]):
                    for mode in ({}, {"stream": True}, {"compact_only": True, "auto_compact": True}):
                        with self.subTest(path=path, route=route, parts=parts[0]["type"], mode=mode), \
                                patch.object(app.state.automatic, "pause", side_effect=AssertionError("must reject before enqueue/pause")), \
                                patch.object(app.state.benchmark, "pause", side_effect=AssertionError("must reject before benchmark pause")), \
                                patch.object(app.state.teacher, "stream", side_effect=AssertionError("must not call teacher")):
                            response = self.client.post(path, headers=bearer if path.startswith("/v1") else {},
                                json={"messages": [{"role": "user", "content": parts}], "route": route, **mode})
                            self.assertEqual(response.status_code, 400)
                            self.assertIn("仅支持文字", response.json()["error"]["message"])
        self.assertEqual(self.runtime.memory.samples(), [])
        self.assertEqual(self.runtime.memory.conversations(), [])
        with self.runtime.memory.connect() as db:
            records = db.execute("SELECT kind,body FROM events").fetchall()
        self.assertEqual(records, [])
        self.runtime.choose_route.assert_not_called()
        self.runtime.generate.assert_not_called()

    def test_media_side_fields_are_not_silently_dropped_from_text(self):
        for key in ("attachments", "image", "images", "image_url", "audio", "audio_url", "input_audio", "input_image", "video", "video_url", "input_video", "file", "files", "input_file"):
            for body in ({"messages": [{"role": "user", "content": "文字", key: ["附件"]}]},
                         {"messages": [{"role": "user", "content": "文字"}], key: ["附件"]},
                         {"messages": [{"role": "user", "content": [{"type": "text", "text": "文字", key: "附件"}]}]}):
                with self.subTest(key=key, body=body):
                    response = self.client.post("/api/chat", json=body)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("仅支持文字", response.json()["error"]["message"])
        self.assertEqual(self.runtime.calls, [])
        for modalities in (["audio"], ["text", "audio"], "text", None):
            response = self.client.post("/api/chat", json={"messages": [{"role": "user", "content": "文字"}], "modalities": modalities})
            self.assertEqual(response.status_code, 400)

    def test_document_raw_endpoint_limit_auth_and_cpu_extraction(self):
        response = self.client.post("/api/chat/attachments/extract?name=test.docx", content=docx(DOC),
                                    headers={"content-type": "application/octet-stream"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "第一行\t第二项\n下一行")
        self.assertEqual(self.runtime.calls, [])
        response = self.client.post("/api/chat/attachments/extract?name=test.docx", content=b"x",
                                    headers={"content-length": str(MAX_DOCUMENT_BYTES + 1)})
        self.assertEqual(response.status_code, 413)
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/api/chat/capabilities").status_code, 401)
        self.assertEqual(self.client.post("/api/chat/attachments/extract?name=test.docx", content=docx(DOC)).status_code, 401)

    def test_capabilities_are_text_only_without_loading_visual_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "vision_config": {"depth": 1}}))
            self.runtime.loaded_path = directory
            with patch("luban.vision.image_processor") as processor:
                data = self.client.get("/api/chat/capabilities").json()
                self.assertTrue(data["text"])
                self.assertFalse(data["images"])
                self.assertFalse(data["audio"])
                self.assertFalse(data["image_model_verified"])
                self.assertEqual(data["image_routes"], [])
                self.assertEqual(data["documents"], {"docx": True, "pdf": False})
                processor.assert_not_called()
            with patch("luban.vision.image_processor", side_effect=ImportError("missing PIL")):
                data = self.client.get("/api/chat/capabilities").json()
                self.assertFalse(data["images"])
                self.assertIn("仅支持文字", data["image_reason"])
        self.assertEqual(self.runtime.calls, [])


if __name__ == "__main__":
    unittest.main()
