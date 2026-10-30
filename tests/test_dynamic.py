import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from luban.dynamic import DynamicMLP, insert_experts


class DynamicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.branch = DynamicMLP(nn.Linear(8, 8, bias=False), 8, 12,
                                 Path(self.tmp.name), "test-base:layer-0", 288)

    def test_eval_base_keeps_wrapper_and_activated_expert_in_eval(self):
        base = nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.9)).eval()
        x = torch.randn(16, 8)
        expected = base(x).detach()
        branch = DynamicMLP(base, 8, 12, Path(self.tmp.name), "eval-base", 288)
        self.assertTrue(all(not module.training for module in branch.modules()))
        torch.testing.assert_close(branch(x), expected, rtol=0, atol=0)
        branch.activate("eval-expert", create=True)
        self.assertTrue(all(not module.training for module in branch.modules()))
        torch.testing.assert_close(branch(x), expected, rtol=0, atol=0)
        branch.hibernate()
        branch.activate("eval-expert")
        self.assertTrue(all(not module.training for module in branch.modules()))
        torch.testing.assert_close(branch(x), expected, rtol=0, atol=0)

    def test_training_base_keeps_dropout_and_expert_training(self):
        base = nn.Sequential(nn.Linear(8, 8), nn.Dropout(1)).train()
        branch = DynamicMLP(base, 8, 12, Path(self.tmp.name), "train-base", 288)
        x = torch.randn(16, 8)
        self.assertTrue(all(module.training for module in branch.modules()))
        torch.testing.assert_close(branch(x), torch.zeros_like(x), rtol=0, atol=0)
        branch.activate("train-expert", create=True)
        self.assertTrue(all(module.training for module in branch.modules()))
        branch.hibernate()
        branch.eval()
        branch.activate("train-expert")
        self.assertTrue(all(not module.training for module in branch.modules()))
        branch.train()
        self.assertTrue(all(module.training for module in branch.modules()))

    def test_identity_train_hibernate_reload(self):
        x = torch.randn(32, 8)
        target = torch.randn(32, 8)
        original = self.branch.base.weight.detach().clone()
        baseline = self.branch(x).detach()
        expert = self.branch.activate("task-a", create=True)
        torch.testing.assert_close(self.branch(x), baseline, rtol=0, atol=0)
        optimizer = torch.optim.Adam(expert.parameters(), lr=0.03)
        start = ((self.branch(x) - target) ** 2).mean().item()
        for _ in range(60):
            optimizer.zero_grad()
            loss = ((self.branch(x) - target) ** 2).mean()
            loss.backward()
            optimizer.step()
        end = ((self.branch(x) - target) ** 2).mean().item()
        self.assertLess(end, start * 0.7)
        torch.testing.assert_close(original, self.branch.base.weight, rtol=0, atol=0)
        learned = self.branch(x).detach()
        del optimizer, expert
        self.branch.hibernate()
        self.assertEqual(self.branch.resident_expert_params, 0)
        self.branch.activate("task-a")
        torch.testing.assert_close(learned, self.branch(x), rtol=0, atol=0)

    def test_incompatible_or_corrupt_checkpoint_rejected(self):
        self.branch.activate("a", create=True)
        self.branch.hibernate()
        self.branch.identity = "different-base"
        with self.assertRaises(ValueError):
            self.branch.activate("a")
        self.branch.identity = "test-base:layer-0"
        path = Path(self.tmp.name) / "a.safetensors"
        path.write_bytes(path.read_bytes() + b"corrupted")
        with self.assertRaises(ValueError):
            self.branch.activate("a")

    def test_budget_and_path_guard(self):
        with self.assertRaises(ValueError):
            DynamicMLP(nn.Linear(8, 8), 8, 12, Path(self.tmp.name), "x", 287)
        with self.assertRaises(ValueError):
            self.branch.activate("../escape", create=True)
        self.branch.activate("a", create=True)
        with self.assertRaises(RuntimeError):
            self.branch.activate("b", create=True)

    def test_core_binding_survives_hibernation_and_loading(self):
        self.branch.activate("versioned", create=True)
        self.branch.core_version = "core-0123456789ab"
        self.branch.hibernate()
        self.branch.core_version = None
        self.branch.activate("versioned")
        self.assertEqual(self.branch.core_version, "core-0123456789ab")

    def test_preservation_rows_survive_hibernation_and_loading(self):
        self.branch.activate("protected", create=True)
        self.branch.preservation_rows = [{"id": "parent-identity", "answer": "父专家答案"}]
        self.branch.hibernate()
        self.branch.activate("protected")
        self.assertEqual(self.branch.preservation_rows,
                         [{"id": "parent-identity", "answer": "父专家答案"}])

    def test_real_qwen3_graph_zero_branch_preserves_logits(self):
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
        model = Qwen3ForCausalLM(config).eval()
        tokens = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            baseline = model(tokens).logits
        branches = insert_experts(model, [1], 8, Path(self.tmp.name), "tiny-random")
        branches[0].activate("new", create=True)
        self.assertTrue(all(not module.training for module in model.modules()))
        with torch.no_grad():
            torch.testing.assert_close(model(tokens).logits, baseline, rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in branches[0].base.parameters()))


if __name__ == "__main__":
    unittest.main()
