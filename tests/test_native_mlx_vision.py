"""Image worker ownership, core-only binding, and existing decoder boundaries."""
import base64
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch
from safetensors.torch import load_file
from luban.native_mlx_vision import generate_images


def image_message():
    raw = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(raw, format="PNG")
    return {"role": "user", "content": [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(raw.getvalue()).decode()}}]}


class NativeMLXVisionTests(unittest.TestCase):
    def test_image_dispatch_does_not_use_hf_backbone(self):
        from luban.vision import generate_images as dispatch
        expected = {"inference_engine": "mlx-q4-vision"}
        with patch("luban.native_mlx_vision.generate_images", return_value=expected) as native, \
             patch("luban.vision.full_backbone", side_effect=AssertionError("BF16 must not load")):
            result = dispatch(SimpleNamespace(native_inference_precision="mlx-q4"), [], None, threading.Event(),
                              thinking="off", max_tokens=16, temperature=0)
        self.assertIs(result, expected)
        native.assert_called_once()

    def test_exclusive_worker_pixels_usage_and_cleanup(self):
        for tokens, reason, complete in [([7, 248044], "stop", True), ([7, 248046], "stop", True),
                                         ([7], "cancelled", False)]:
            with self.subTest(tokens=tokens, reason=reason):
                self.exercise_worker(tokens, reason, complete)

    def exercise_worker(self, tokens, reason, complete):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "upstream-file-manifest.json").write_text("{}")
            old = Mock()
            pager = SimpleNamespace(depth_count=0, loaded=False, evict_resident=Mock())
            runtime = SimpleNamespace(lock=threading.RLock(), _ready=Mock(), active_core=None, tokenizer=Mock(),
                loaded_path=directory, state="ready", native_engine=old, cold_backbone=pager,
                settings=SimpleNamespace(snapshot=lambda: {"max_new_tokens": 16}), text_config=Mock(), device="mps",
                model=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(num_heads=1)),
                                      generation_config=SimpleNamespace(eos_token_id=[2])),
                branch=SimpleNamespace(expert=object()))
            if tokens[-1] == 248046:
                runtime.active_core = "core-v1"
                runtime.store, runtime.identity = root, "fixture-model"
                (root / "active-core.json").write_text(
                    '{"version":"core-v1","identity":"fixture-model","sha256":"core-sha"}')
            expert = runtime.branch.expert
            inputs = {"input_ids": torch.tensor([[62, 60, 63]]), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                      "pixel_values": torch.ones(4, 1536)}
            engine = Mock()
            captured = {}
            def run(prompt, **kwargs):
                self.assertTrue(old.close.called)
                self.assertIs(runtime.native_engine, engine)
                self.assertEqual(runtime.state, "generating")
                self.assertEqual(kwargs["image_grid_thw"], [[1, 2, 2]])
                captured["path"] = kwargs["pixel_file"]
                self.assertTrue(torch.equal(load_file(captured["path"])["pixel_values"], inputs["pixel_values"]))
                self.assertEqual(kwargs["eos_ids"], [2, 248044, 248046])
                return {"tokens": tokens, "finish_reason": reason, "decode_ns": 100_000_000}
            engine.generate_image.side_effect = run
            with patch("luban.vision.image_inputs", return_value=inputs), patch("luban.vision.image_processor"), \
                 patch("luban.native_mlx_vision.model_binding", return_value={"weight_bytes": 100}), \
                 patch("luban.native_mlx_vision.native_binary", return_value=root / "worker"), \
                 patch("luban.native_mlx_vision.window", return_value={}), \
                 patch("luban.native_mlx_vision.output_budget", return_value=16), \
                 patch("luban.native_mlx_vision.NativeEngine", return_value=engine) as factory, \
                 patch("luban.evolution._read_version", return_value=({"sha256": "core-sha"}, None)), \
                 patch("luban.evolution._target", return_value=("model.language_model.layers.23.mlp.base.down_proj.weight", None)):
                usage = generate_images(runtime, [image_message()], None, threading.Event(),
                                        thinking="off", max_tokens=16, temperature=0)
            self.assertTrue(factory.call_args.kwargs["vision"])
            if runtime.active_core:
                self.assertEqual(factory.call_args.kwargs["core"]["file"], str(root / "core-v1.safetensors"))
                self.assertEqual(factory.call_args.kwargs["core"]["sha256"], "core-sha")
            else:
                self.assertIsNone(factory.call_args.kwargs["core"])
            self.assertNotIn("expert", factory.call_args.kwargs)
            self.assertEqual(usage["tokens_per_second"], 10 if len(tokens) > 1 else None)
            self.assertEqual(usage["local_complete"], complete)
            self.assertEqual(usage["finish_reason"], "stop")
            self.assertEqual(usage["image_count"], 1)
            self.assertFalse(Path(captured["path"]).exists())
            self.assertIs(runtime.branch.expert, expert)
            self.assertEqual(runtime.state, "ready")
            self.assertIsNone(runtime.native_engine)
            engine.close.assert_called_once()

    def test_cancelled_before_preprocessing_and_image_count_limit(self):
        runtime = SimpleNamespace(lock=threading.RLock(), _ready=Mock())
        stop = threading.Event(); stop.set()
        with patch("luban.vision.image_inputs", side_effect=AssertionError("must not preprocess")):
            with self.assertRaisesRegex(RuntimeError, "停止"):
                generate_images(runtime, [image_message()], None, stop, thinking="off", max_tokens=16, temperature=0)
            with self.assertRaisesRegex(ValueError, "4 张"):
                generate_images(runtime, [image_message()] * 5, None, threading.Event(),
                                thinking="off", max_tokens=16, temperature=0)


if __name__ == "__main__":
    unittest.main()
