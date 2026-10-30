"""MLX-Q4 startup only needs the model graph metadata, not BF16 shards."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config

from luban.coldpaging import ColdBackbone, full_backbone
from luban.evolution import _target, load_active_core, restore_core
from luban.memory import Memory
from luban.runtime import Runtime


class MlxMetadataStartupTests(unittest.TestCase):
    def test_mlx_q4_loads_without_upstream_weight_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Qwen3_5Config(
                text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                             "num_hidden_layers": 6, "num_attention_heads": 4,
                             "num_key_value_heads": 2, "head_dim": 8, "linear_key_head_dim": 8,
                             "linear_value_head_dim": 8, "linear_num_key_heads": 2,
                             "linear_num_value_heads": 4, "linear_conv_kernel_dim": 4,
                             "max_position_embeddings": 128,
                             "layer_types": ["linear_attention"] * 6,
                             "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                                 "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
                             "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0},
                vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                               "num_heads": 2, "out_hidden_size": 32, "patch_size": 2,
                               "temporal_patch_size": 2, "spatial_merge_size": 2,
                               "num_position_embeddings": 16}, image_token_id=60,
                video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
                tie_word_embeddings=True)
            config.save_pretrained(root)
            (root / "upstream-file-manifest.json").write_text(
                json.dumps({"Data": {"Files": [{"Path": "config.json", "Sha256": "0" * 64}]}}))
            storage = root / "storage"
            storage.mkdir()
            settings = SimpleNamespace(
                home=storage,
                snapshot=lambda: {"storage_root": str(storage), "model_path": str(root),
                                  "context_tokens": 128, "context_mode": "manual",
                                  "native_inference_precision": "mlx-q4",
                                  "native_mlx_model_path": str(root)})
            runtime = Runtime(settings, Memory(storage), research=True)
            runtime.device, runtime.dtype = "cpu", torch.float32
            self.addCleanup(runtime._release_model)
            with patch("luban.runtime.AutoTokenizer.from_pretrained"), \
                 patch("luban.product_inference.native_binary", return_value=root / "worker"):
                runtime.load(paging=False)
            self.assertEqual(runtime.state, "ready")
            self.assertTrue(runtime.cold_backbone.metadata_only)
            self.assertFalse(any(root.glob("model*.safetensors")))
            # Presence of an original shard must never switch MLX back to HF.
            (root / "model.safetensors").write_bytes(b"intentionally unreadable BF16 placeholder")
            with patch("luban.runtime.AutoTokenizer.from_pretrained"), \
                 patch("luban.product_inference.native_binary", return_value=root / "worker"), \
                 patch.object(ColdBackbone, "original_tensor", side_effect=AssertionError("BF16 read")):
                runtime.load()
            self.assertTrue(runtime.cold_backbone.metadata_only)
            name, parameter = _target(runtime)
            version = "core-111111111111"
            weights = runtime.store / (version + ".safetensors")
            save_file({name: torch.ones(parameter.shape, dtype=parameter.dtype)}, str(weights))
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            metadata = {"identity": runtime.identity, "version": version, "sha256": digest}
            (runtime.store / (version + ".json")).write_text(json.dumps(metadata))
            pointer = runtime.store / "active-core.json"
            pointer.write_text(json.dumps(metadata))
            self.assertEqual(load_active_core(runtime)["version"], version)
            self.assertIsNone(runtime._base_core_weight)
            self.assertTrue(parameter.is_meta)
            runtime.close_native = Mock()
            self.assertTrue(restore_core(runtime)["restored"])
            self.assertIsNone(runtime.active_core)
            self.assertFalse(pointer.exists())
            self.assertTrue(restore_core(runtime, version)["restored"])
            self.assertEqual(runtime.active_core, version)
            runtime.close_native.assert_called()
            # A malformed saved core must remain rejected without BF16 present.
            weights.write_bytes(b"bad checkpoint")
            with self.assertRaisesRegex(ValueError, "校验失败"):
                load_active_core(runtime)
            with self.assertRaisesRegex(RuntimeError, "元数据模式"):
                with full_backbone(runtime):
                    pass


if __name__ == "__main__":
    unittest.main()
