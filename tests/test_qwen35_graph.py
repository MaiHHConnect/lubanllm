"""Offline CPU graph invariants on the official hybrid Qwen3.5 implementation."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from luban.backbone import text_backbone
from luban.dynamic import DynamicMLP, insert_experts
from luban.runtime import Runtime


class Qwen35GraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Path(self.temp.name)
        config = Qwen3_5Config(
            text_config={
                "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 6, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "linear_conv_kernel_dim": 4, "max_position_embeddings": 128,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                                "full_attention", "linear_attention", "full_attention"],
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                    "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
                "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
            },
            vision_config={
                "depth": 1, "hidden_size": 16, "intermediate_size": 32,
                "num_heads": 2, "out_hidden_size": 32, "patch_size": 2,
                "temporal_patch_size": 2, "spatial_merge_size": 2,
                "num_position_embeddings": 16,
            },
            image_token_id=60, video_token_id=61,
            vision_start_token_id=62, vision_end_token_id=63,
            tie_word_embeddings=True,
        )
        self.model = Qwen3_5ForConditionalGeneration(config).cpu().eval()
        self.decoder, self.text_config = text_backbone(self.model)
        self.tokens = torch.tensor([[3, 7, 13, 19, 23, 29, 31, 37]])

    def logits(self, keep):
        with torch.no_grad():
            return self.model(self.tokens, use_cache=False, logits_to_keep=keep).logits.detach().clone()

    def test_official_hybrid_zero_branch_and_disk_round_trip_preserve_logits(self):
        self.assertEqual(self.model.config.model_type, "qwen3_5")
        self.assertIs(self.decoder, self.model.model.language_model)
        self.assertEqual(self.text_config.layer_types[:4],
                         ["linear_attention"] * 3 + ["full_attention"])
        baselines = {keep: self.logits(keep) for keep in (0, 1)}
        original_mlp = self.decoder.layers[-1].mlp
        original_attention = self.decoder.layers[-1].self_attn
        branch = insert_experts(self.model, [5], 8, self.store, "tiny-qwen35-official")[0]
        self.assertIsInstance(self.decoder.layers[-1].mlp, DynamicMLP)
        self.assertIs(branch.base, original_mlp)
        self.assertIs(self.decoder.layers[-1].self_attn, original_attention)
        self.assertTrue(all(not parameter.requires_grad for parameter in branch.base.parameters()))
        for keep, baseline in baselines.items():
            with self.subTest(state="inserted-without-expert", logits_to_keep=keep):
                torch.testing.assert_close(self.logits(keep), baseline, rtol=0, atol=0)
        branch.activate("zero", create=True)
        self.assertEqual(branch.resident_expert_params, 3 * 32 * 8)
        self.assertEqual(torch.count_nonzero(branch.expert.down.weight).item(), 0)
        for keep, baseline in baselines.items():
            with self.subTest(state="active-zero", logits_to_keep=keep):
                torch.testing.assert_close(self.logits(keep), baseline, rtol=0, atol=0)
        branch.hibernate()
        self.assertIsNone(branch.expert)
        self.assertEqual(branch.resident_expert_params, 0)
        self.assertTrue((self.store / "layer-5/zero.safetensors").is_file())
        for keep, baseline in baselines.items():
            with self.subTest(state="hibernated", logits_to_keep=keep):
                torch.testing.assert_close(self.logits(keep), baseline, rtol=0, atol=0)
        branch.activate("zero")
        for keep, baseline in baselines.items():
            with self.subTest(state="reloaded-zero", logits_to_keep=keep):
                torch.testing.assert_close(self.logits(keep), baseline, rtol=0, atol=0)

    def test_shallow_executes_exactly_first_four_hybrid_layers_and_preserves_state(self):
        runtime = Runtime.__new__(Runtime)
        runtime.model, runtime.decoder = self.model, self.decoder
        executed, captured = [], []
        handles = []
        try:
            for index, layer in enumerate(self.decoder.layers):
                def record(module, args, layer_index=index):
                    executed.append(layer_index)
                handles.append(layer.register_forward_pre_hook(record))

            def capture(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                captured.append(value.detach().clone())
            handles.append(self.decoder.layers[3].register_forward_hook(capture))
            self.logits(1)
            self.assertEqual(executed, list(range(6)))
            reference = captured[-1]
            executed.clear()
            before_hooks = len(self.decoder.layers[3]._forward_hooks)
            shallow = runtime.shallow(self.tokens)
            self.assertEqual(executed, [0, 1, 2, 3])
            self.assertEqual(len(self.decoder.layers[3]._forward_hooks), before_hooks)
            self.assertFalse(shallow.requires_grad)
            torch.testing.assert_close(shallow, reference, rtol=0, atol=0)
            torch.testing.assert_close(captured[-1], reference, rtol=0, atol=0)
            # The intentional early exit must not leak a hook or poison a later
            # full model call; all six layers should still execute normally.
            executed.clear()
            self.logits(1)
            self.assertEqual(executed, list(range(6)))
            torch.testing.assert_close(captured[-1], reference, rtol=0, atol=0)
        finally:
            for handle in handles:
                handle.remove()

    def test_unsupported_decoder_rejected_before_modifying_parameters_or_storage(self):
        model = nn.Linear(2, 2)
        model.config = SimpleNamespace(model_type="unsupported")
        weights = model.weight.detach().clone()
        storage = self.store / "must-not-be-created"
        with self.assertRaisesRegex(ValueError, "supported"):
            insert_experts(model, [0], 8, storage, "unsupported")
        torch.testing.assert_close(model.weight, weights, rtol=0, atol=0)
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(storage.exists())


if __name__ == "__main__":
    unittest.main()
