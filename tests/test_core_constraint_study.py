"""Protection trajectories keep observed special tokens, separate from gold EOS."""
import copy
import json
import tempfile
import unittest

import torch

from test_cached_training import fixture
from study_core_constraint import capture, tail_loss, residual_logits
from luban.core_update import apply_core_residual


class ConstraintStudyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.runtime, rows, _ = fixture(self.folder.name, torch.bfloat16)
        self.runtime.model.eval().requires_grad_(False)
        self.runtime.tokenizer.eos_token_id = 63
        self.runtime.model.generation_config.eos_token_id = 2
        self.runtime._tokens = lambda *args, **kwargs: torch.tensor([[3, 7, 13]])
        self.row = {**rows[0], "prompt": "Authored probe", "provenance": json.dumps({
            "messages": [{"role": "user", "content": "Authored probe"}]})}
        self.generated = {"id": self.row["id"], "complete": True, "branch_enabled": False,
            "reference_used": False, "max_new_tokens": 8, "raw_completion_token_ids": [19, 63, 29, 2]}

    def test_exact_observed_tokens_keep_internal_turn_eos_and_generation_terminal(self):
        old = self.runtime.branch.base.down_proj.weight.detach().clone()
        cache = capture(self.runtime, self.row, generated=self.generated)
        self.assertEqual(cache["ids"].tolist(), [[3, 7, 13, 19, 63, 29, 2]])
        self.assertEqual(cache["labels"].tolist(), [[-100, -100, -100, 19, 63, 29, 2]])
        weight = old.clone().requires_grad_()
        loss = tail_loss(self.runtime, cache, weight)
        loss.backward()
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertTrue(torch.equal(old, self.runtime.branch.base.down_proj.weight))
        self.assertFalse(self.runtime.branch.base.down_proj._forward_pre_hooks)

    def test_incomplete_wrong_identity_or_terminal_trajectories_rejected(self):
        for update in ({"complete": False}, {"id": "different"}, {"branch_enabled": True},
                       {"reference_used": True}, {"raw_completion_token_ids": [19, 63]},
                       {"raw_completion_token_ids": []}, {"raw_completion_token_ids": [False, 2]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                capture(self.runtime, self.row, generated={**self.generated, **update})

    def test_dev_never_enters_protection_cache(self):
        row = copy.deepcopy(self.row)
        row["purpose"] = "validation"
        with self.assertRaises(ValueError):
            capture(self.runtime, row, generated=self.generated)

    def test_unmerged_residual_matches_actual_tiny_graph_output_and_delta_gradient(self):
        cache = capture(self.runtime, self.row, generated=self.generated)
        weight = self.runtime.branch.base.down_proj.weight
        original = self.runtime.model(cache["ids"], use_cache=False).logits.detach()
        for amount in (0., .001):
            delta = torch.full(weight.shape, amount, dtype=torch.float32, requires_grad=True)
            handle = self.runtime.branch.base.down_proj.register_forward_hook(
                lambda module, inputs, output: apply_core_residual(inputs[0], output, delta))
            try:
                full = self.runtime.model(cache["ids"], labels=cache["labels"], use_cache=False)
                full.loss.backward()
                full_grad = delta.grad.clone()
            finally:
                handle.remove()
            if amount == 0:
                self.assertTrue(torch.equal(full.logits.detach(), original))
            delta.grad = None
            logits = residual_logits(self.runtime, cache, weight, delta)
            labels = cache["labels"]
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            self.assertTrue(torch.equal(logits.detach(), full.logits.detach()))
            self.assertTrue(torch.equal(delta.grad, full_grad))


if __name__ == "__main__":
    unittest.main()
