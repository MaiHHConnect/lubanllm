"""Real upstream graph paging equivalence and recovery; offline CPU only."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from luban.backbone import load_backbone, text_backbone
from luban.coldpaging import ColdBackbone, full_backbone
from luban.dynamic import insert_experts
from luban.evolution import load_active_core, _target
from luban.runtime import Runtime


class ColdPagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        torch.manual_seed(42)
        root = Path(self.temp.name)
        config = Qwen3_5Config(text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 128,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "full_attention"],
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        }, vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                          "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                          "spatial_merge_size": 2, "num_position_embeddings": 16},
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
            tie_word_embeddings=True)
        model = Qwen3_5ForConditionalGeneration(config).eval().requires_grad_(False)
        model.save_pretrained(root)
        decoder, _ = text_backbone(model)
        store = root / "modules"
        branch = insert_experts(model, [5], 8, store, "paging-fixture")[0]
        self.runtime = SimpleNamespace(model=model, decoder=decoder, branch=branch, store=store,
                                       lock=threading.RLock(), device="cpu", dtype=torch.float32,
                                       identity="paging-fixture", active_core=None)
        self.runtime.shallow = Runtime.shallow.__get__(self.runtime)
        load_active_core(self.runtime)
        self.ids = torch.tensor([[3, 7, 13, 19, 23]])
        with torch.no_grad():
            self.baseline_logits = model(self.ids, use_cache=False).logits
            self.baseline_features = self.runtime.shallow(self.ids)
        self.pager = ColdBackbone(self.runtime, root)
        self.runtime.cold_backbone = self.pager

    def assert_cold(self):
        self.assertFalse(self.pager.loaded)
        self.assertTrue(all(value.is_meta for layer in self.runtime.decoder.layers[4:] for value in layer.parameters()))
        self.assertTrue(all(not value.is_meta for layer in self.runtime.decoder.layers[:4] for value in layer.parameters()))
        self.assertIs(self.runtime.model.lm_head.weight, self.runtime.decoder.embed_tokens.weight)
        self.assertFalse(self.runtime.model.lm_head.weight.is_meta)
        self.assertEqual(self.runtime.model.device.type, "cpu")

    def start_cold(self, dtype=torch.float32):
        root = Path(self.temp.name)
        with patch("transformers.AutoModelForMultimodalLM.from_pretrained",
                   side_effect=AssertionError("cold startup must not use the full loader")):
            model = load_backbone(root, dtype, "cpu", cold=True)
        self.assertTrue(all(value.is_meta for value in model.parameters()))
        self.assertTrue(all(not value.is_meta for value in model.buffers()))
        self.runtime.model = model
        self.runtime.decoder, _ = text_backbone(model)
        self.runtime.dtype = dtype
        self.runtime.branch = insert_experts(model, [5], 8, self.runtime.store, "paging-fixture")[0]
        self.pager = ColdBackbone(self.runtime, root)
        self.runtime.cold_backbone = self.pager
        with patch.object(self.pager, "_tensor", wraps=self.pager._tensor) as reads:
            self.pager.load_resident()
        keys = [call.args[0] for call in reads.call_args_list]
        self.assertTrue(keys)
        self.assertFalse(any("visual" in key or ".layers.4." in key or ".layers.5." in key for key in keys))
        load_active_core(self.runtime)
        self.assert_cold()
        self.assertFalse(self.pager.status()["vision_loaded"])

    def test_direct_cold_start_features_logits_gradient_and_generation_match_full(self):
        parameter = _target(self.runtime)[1]
        parameter.requires_grad_(True)
        expected_gradient, = torch.autograd.grad(self.runtime.model(self.ids, labels=self.ids, use_cache=False).loss, parameter)
        with torch.no_grad():
            expected_tokens = self.runtime.model.generate(self.ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
        self.start_cold()
        torch.testing.assert_close(self.runtime.shallow(self.ids), self.baseline_features, atol=0, rtol=0)
        self.assertEqual(self.pager.loads, 0)
        with full_backbone(self.runtime):
            parameter = _target(self.runtime)[1]
            parameter.requires_grad_(True)
            output = self.runtime.model(self.ids, labels=self.ids, use_cache=False)
            gradient, = torch.autograd.grad(output.loss, parameter)
            torch.testing.assert_close(output.logits, self.baseline_logits, atol=0, rtol=0)
            torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)
            parameter.requires_grad_(False)
            with torch.no_grad():
                actual = self.runtime.model.generate(self.ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
            self.assertTrue(torch.equal(actual, expected_tokens))
        self.assert_cold()

    def test_direct_cold_start_bf16_matches_official_full_loader(self):
        full = load_backbone(self.temp.name, torch.bfloat16, "cpu")
        with torch.no_grad():
            expected = full(self.ids, use_cache=False).logits
        self.start_cold(torch.bfloat16)
        with full_backbone(self.runtime), torch.no_grad():
            actual = self.runtime.model(self.ids, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assert_cold()

    def test_unused_shallow_weights_can_sleep_and_restore_exact_features_and_logits(self):
        self.start_cold()
        self.pager.evict_resident()
        self.assertTrue(self.runtime.model.lm_head.weight.is_meta)
        self.assertIs(self.runtime.model.lm_head.weight, self.runtime.decoder.embed_tokens.weight)
        self.assertEqual(self.pager.status()["resident_language_layers"], 0)
        self.assertEqual(self.pager.status()["cold_language_layers"], 6)
        with patch.object(self.pager, "load_resident", wraps=self.pager.load_resident) as load:
            torch.testing.assert_close(self.runtime.shallow(self.ids), self.baseline_features, atol=0, rtol=0)
            torch.testing.assert_close(self.runtime.shallow(self.ids), self.baseline_features, atol=0, rtol=0)
        self.assertEqual(load.call_count, 1)
        self.pager.evict_resident()
        with full_backbone(self.runtime), torch.no_grad():
            torch.testing.assert_close(self.runtime.model(self.ids, use_cache=False).logits,
                                       self.baseline_logits, atol=0, rtol=0)
            with self.assertRaisesRegex(RuntimeError, "使用中"):
                self.pager.evict_resident()
        self.assert_cold()

    def test_shallow_restore_failure_clears_partial_weights_and_retries_cleanly(self):
        self.start_cold()
        self.pager.evict_resident()
        key = self.pager.prefix + ".layers.1.mlp.up_proj.weight"
        shard = self.pager.weight_map.pop(key)
        with self.assertRaisesRegex(ValueError, "缺少"):
            self.runtime.shallow(self.ids)
        self.assertTrue(all(value.is_meta for module in self.pager.resident_modules() for value in module.parameters()))
        self.pager.weight_map[key] = shard
        torch.testing.assert_close(self.runtime.shallow(self.ids), self.baseline_features, atol=0, rtol=0)

    def test_direct_cold_start_restores_visual_logits_and_buffers(self):
        from luban.vision import load_visual
        inputs = {"input_ids": torch.tensor([[62, 60, 63, 3]]), "attention_mask": torch.ones(1, 4, dtype=torch.long),
                  "mm_token_type_ids": torch.tensor([[0, 1, 0, 0]]), "image_grid_thw": torch.tensor([[1, 2, 2]]),
                  "pixel_values": torch.randn(4, 24)}
        with torch.no_grad():
            expected = self.runtime.model(**inputs, use_cache=False).logits
        self.start_cold()
        with full_backbone(self.runtime), torch.no_grad():
            load_visual(self.runtime)
            actual = self.runtime.model(**inputs, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assert_cold()
        self.assertFalse(self.pager.status()["vision_loaded"])

    def test_direct_cold_start_sharded_weights_and_missing_resident_reject(self):
        # Exercise the existing official shard index, without touching old files.
        full = load_backbone(self.temp.name, torch.float32, "cpu")
        full.save_pretrained(self.temp.name, max_shard_size="10KB")
        self.start_cold()
        with full_backbone(self.runtime), torch.no_grad():
            torch.testing.assert_close(self.runtime.model(self.ids, use_cache=False).logits, self.baseline_logits, atol=0, rtol=0)
        self.pager.weight_map.pop(self.pager.prefix + ".embed_tokens.weight")
        with self.assertRaisesRegex(ValueError, "缺少"):
            self.pager.load_resident()

    def test_direct_cold_start_validates_selected_core_without_materializing_last_layer(self):
        name, weight = _target(self.runtime)
        original = weight.detach().clone()
        version = "core-222222222222"
        candidate = original + 0.125
        path = self.runtime.store / (version + ".safetensors")
        save_file({name: candidate}, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.runtime.store / (version + ".json")).write_text(json.dumps({"version": version,
            "identity": self.runtime.identity, "parameter": name, "sha256": digest}))
        pointer = self.runtime.store / "active-core.json"
        pointer.write_text(json.dumps({"version": version, "identity": self.runtime.identity, "sha256": digest}))
        self.start_cold()
        self.assertEqual(self.runtime.active_core, version)
        torch.testing.assert_close(self.runtime._base_core_weight, original, atol=0, rtol=0)
        self.assertTrue(_target(self.runtime)[1].is_meta)
        with full_backbone(self.runtime):
            torch.testing.assert_close(_target(self.runtime)[1], candidate, atol=0, rtol=0)
        self.assert_cold()
        pointer.write_text(json.dumps({"version": version, "identity": "wrong", "sha256": digest}))
        with self.assertRaisesRegex(ValueError, "不兼容"):
            load_active_core(self.runtime)

    def test_runtime_load_stop_reload_uses_direct_cold_path_and_cleans_failed_load(self):
        from luban.memory import Memory
        from test_vision import tiny_tokenizer
        root = Path(self.temp.name)
        (root / "upstream-file-manifest.json").write_text(json.dumps({"Data": {"Files": [
            {"Path": "config.json", "Sha256": "0" * 64}]}}))
        storage = root / "runtime-storage"
        storage.mkdir()
        settings = SimpleNamespace(snapshot=lambda: {"storage_root": str(storage), "model_path": str(root),
                                                      "context_tokens": 128, "context_mode": "manual"})
        runtime = Runtime(settings, Memory(storage), research=True)
        runtime.device, runtime.dtype = "cpu", torch.float32
        self.addCleanup(runtime._release_model)
        with patch("luban.runtime.AutoTokenizer.from_pretrained", return_value=tiny_tokenizer()), \
                patch("transformers.AutoModelForMultimodalLM.from_pretrained",
                      side_effect=AssertionError("full loader called")), \
                patch.object(ColdBackbone, "_evict", side_effect=AssertionError("startup evicted weights")):
            runtime.load()
            self.assertEqual(runtime.state, "ready")
            self.assertFalse(runtime.cold_backbone.loaded)
            self.assertFalse(runtime.status()["paging"]["vision_loaded"])
            torch.testing.assert_close(runtime.shallow(self.ids), self.baseline_features, atol=0, rtol=0)
            runtime.stop()
            self.assertEqual(runtime.state, "stopped")
            runtime.load()
            self.assertEqual(runtime.state, "ready")
            runtime.stop()
            with patch.object(ColdBackbone, "load_resident", side_effect=ValueError("bad resident weight")):
                with self.assertRaisesRegex(ValueError, "bad resident"):
                    runtime.load()
            self.assertEqual(runtime.state, "error")
            self.assertIsNone(runtime.model)
            self.assertIsNone(runtime._model_process_lock)

    def test_native_capable_startup_defers_both_engines_until_needed(self):
        from luban.memory import Memory
        from test_vision import tiny_tokenizer
        root = Path(self.temp.name)
        (root / "upstream-file-manifest.json").write_text(json.dumps({"Data": {"Files": [
            {"Path": "config.json", "Sha256": "0" * 64}]}}))
        storage = root / "lazy-runtime-storage"
        storage.mkdir()
        settings = SimpleNamespace(snapshot=lambda: {"storage_root": str(storage), "model_path": str(root),
                                                      "context_tokens": 128, "context_mode": "manual"})
        runtime = Runtime(settings, Memory(storage), research=True)
        runtime.device, runtime.dtype = "cpu", torch.float32
        self.addCleanup(runtime._release_model)
        with patch("luban.runtime.AutoTokenizer.from_pretrained", return_value=tiny_tokenizer()), \
                patch("luban.product_inference.native_binary", return_value=root / "worker"), \
                patch("luban.product_inference.start", side_effect=AssertionError("eager native start")), \
                patch.object(ColdBackbone, "load_resident", side_effect=AssertionError("eager HF weights")):
            runtime.load()
        self.assertEqual(runtime.state, "ready")
        self.assertIsNone(runtime.native_engine)
        self.assertEqual(runtime.cold_backbone.status()["resident_language_layers"], 0)
        self.assertTrue(all(value.is_meta for value in runtime.decoder.parameters()))
        torch.testing.assert_close(runtime.shallow(self.ids), self.baseline_features, atol=0, rtol=0)
        self.assertEqual(runtime.cold_backbone.status()["resident_language_layers"], 4)

    def test_shallow_does_not_load_cold_layers_and_logits_and_gradient_are_exact(self):
        parameter = _target(self.runtime)[1]
        parameter.requires_grad_(True)
        native_loss = self.runtime.model(self.ids, labels=self.ids, use_cache=False).loss
        native_gradient, = torch.autograd.grad(native_loss, parameter)
        parameter.requires_grad_(False)
        self.pager._evict()
        self.assert_cold()
        torch.testing.assert_close(self.runtime.shallow(self.ids), self.baseline_features, atol=0, rtol=0)
        self.assertEqual(self.pager.loads, 0)
        with full_backbone(self.runtime):
            parameter = _target(self.runtime)[1]
            parameter.requires_grad_(True)
            output = self.runtime.model(self.ids, labels=self.ids, use_cache=False)
            gradient, = torch.autograd.grad(output.loss, parameter)
            torch.testing.assert_close(output.logits, self.baseline_logits, atol=0, rtol=0)
            torch.testing.assert_close(gradient, native_gradient, atol=0, rtol=0)
            parameter.requires_grad_(False)
        self.assert_cold()

    def test_two_cached_generations_match_full_and_nested_scope_releases_once(self):
        prompts = (self.ids, self.ids[:, :3])
        with torch.no_grad():
            expected = [self.runtime.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0) for ids in prompts]
        self.pager._evict()
        for ids, answer in zip(prompts, expected):
            with full_backbone(self.runtime):
                with full_backbone(self.runtime):
                    actual = self.runtime.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
                self.assertTrue(self.pager.loaded)
                self.assertEqual(self.pager.depth_count, 1)
                self.assertTrue(torch.equal(actual, answer))
            self.assert_cold()
        self.assertEqual(self.pager.loads, 2)

    def test_current_core_is_reapplied_and_original_snapshot_is_preserved(self):
        name, weight = _target(self.runtime)
        original = weight.detach().clone()
        version = "core-111111111111"
        candidate = original + 0.125
        path = self.runtime.store / (version + ".safetensors")
        save_file({name: candidate}, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.runtime.store / (version + ".json")).write_text(json.dumps({"version": version,
            "identity": self.runtime.identity, "parameter": name, "sha256": digest}))
        (self.runtime.store / "active-core.json").write_text(json.dumps({"version": version,
            "identity": self.runtime.identity, "sha256": digest}))
        self.pager._evict()
        for _ in range(2):
            with full_backbone(self.runtime):
                torch.testing.assert_close(_target(self.runtime)[1], candidate, atol=0, rtol=0)
                torch.testing.assert_close(self.runtime._base_core_weight, original, atol=0, rtol=0)
            self.assert_cold()

    def test_missing_tensor_and_forward_failure_return_to_cold_without_pointer_changes(self):
        self.pager._evict()
        key = next(key for key in self.pager.weight_map if ".layers.5." in key)
        saved = self.pager.weight_map.pop(key)
        with self.assertRaisesRegex(ValueError, "缺少"):
            with full_backbone(self.runtime):
                self.fail("Missing weight must not reach execution")
        self.assertEqual(self.pager.depth_count, 0)
        self.assert_cold()
        self.pager.weight_map[key] = saved
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with full_backbone(self.runtime):
                raise RuntimeError("injected forward failure")
        self.assert_cold()
        self.assertFalse((self.runtime.store / "active-core.json").exists())


if __name__ == "__main__":
    unittest.main()
