"""CPU full-graph versus recomputed VJP, including residual and temporal paths."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from q4_layerwise_sensitivity import layerwise_gradients


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(4, 6), nn.Linear(4, 6), nn.Linear(6, 4)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix, self.mlp = nn.Linear(4, 4), ToyMLP()
        self.grad_only_change = False
        self.fail = False

    def forward(self, hidden_states, *, position_embeddings, attention_mask,
                position_ids, past_key_values, use_cache):
        assert past_key_values is None and use_cache is False
        if self.fail and torch.is_grad_enabled():
            raise RuntimeError('deliberate replay failure')
        # Future losses depend on all earlier tokens; no tokenwise shortcut is valid.
        context = hidden_states.float().cumsum(1).to(hidden_states.dtype)
        hidden = hidden_states+self.mix(context)*position_embeddings[0]+attention_mask
        hidden = hidden+position_ids[..., None].to(hidden.dtype)*.001
        output = hidden+self.mlp(hidden)
        return output+.25 if self.grad_only_change and torch.is_grad_enabled() else output


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation='eager')
        self.embed_tokens = nn.Embedding(11, 4)
        self.layers = nn.ModuleList([ToyLayer() for _ in range(3)])
        self.norm = nn.LayerNorm(4)

    def forward(self, ids, *, use_cache=False, past_key_values=None):
        hidden = self.embed_tokens(ids)
        position_ids = torch.arange(ids.shape[1], device=ids.device)[None, :]
        position_embeddings = (torch.full_like(hidden, .25), torch.zeros_like(hidden))
        mask = torch.full_like(hidden, .125)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=mask,
                           position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder, self.lm_head = ToyDecoder(), nn.Linear(4, 11)


class LayerwiseSensitivityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ToyModel().to(torch.bfloat16).eval().requires_grad_(False)
        self.decoder = self.model.decoder
        self.ids = torch.tensor([[1, 2, 3, 4, 5]])
        self.positions, self.targets = torch.tensor([2, 3, 4]), torch.tensor([4, 5, 6])
        # Deliberately omit the middle layer and request a nonexecution key order.
        self.projections = [(2, 'up'), (0, 'gate'), (2, 'down'), (0, 'up')]

    def reference(self, reduction):
        captured, handles = {}, []
        def capture(name):
            def hook(_module, _args, value):
                captured[name] = value
            return hook
        handles.append(self.decoder.embed_tokens.register_forward_hook(
            lambda _module, _args, value: value.requires_grad_(True)))
        for layer, kind in self.projections:
            handles.append(getattr(self.decoder.layers[layer].mlp, kind+'_proj').register_forward_hook(
                capture(f'layer{layer:02d}-{kind}')))
        try:
            hidden = self.decoder(self.ids).last_hidden_state
            logits = self.model.lm_head(hidden[:, self.positions, :])[0]
            loss = F.cross_entropy(logits.float(), self.targets, reduction=reduction)
            keys = [f'layer{i:02d}-{kind}' for i, kind in self.projections]
            gradients = torch.autograd.grad(loss, [captured[key] for key in keys])
            return logits.detach(), float(loss.detach()), dict(zip(keys, gradients, strict=True))
        finally:
            for handle in handles:
                handle.remove()

    def run_helper(self, **kwargs):
        return layerwise_gradients(self.model, self.decoder, self.ids, self.positions,
                                   self.targets, self.projections, **kwargs)

    def assert_hooks_clean(self):
        for module in self.model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))

    def test_sum_and_mean_exact_full_graph_with_temporal_residual_paths(self):
        for reduction in ('sum', 'mean'):
            with self.subTest(reduction=reduction):
                logits, loss, expected = self.reference(reduction)
                events = []
                actual_logits, actual_loss, actual, controls = self.run_helper(reduction=reduction, progress=events.append)
                self.assertTrue(torch.equal(actual_logits, logits))
                self.assertEqual(actual_loss, loss)
                self.assertEqual(list(actual), list(expected))
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]), name)
                    self.assertEqual(actual[name].dtype, torch.bfloat16)
                    self.assertEqual(actual[name].device.type, 'cpu')
                    self.assertFalse(actual[name].requires_grad)
                self.assertTrue(torch.any(actual['layer00-gate'][:, 0]))
                self.assertEqual(controls['layer_order'], [2, 1, 0])
                self.assertEqual(controls['captured_boundaries'], 4)
                self.assertEqual(controls['boundary_bytes'], 4*5*4*2)
                self.assertFalse(controls['full_graph_gradient_equivalence_tested'])
                self.assertEqual([e['layer'] for e in events if e['stage'] == 'layer_vjp_complete'], [2, 1, 0])
                self.assert_hooks_clean()

    def test_grad_only_forward_change_rejected_and_hooks_removed(self):
        self.decoder.layers[1].grad_only_change = True
        with self.assertRaisesRegex(ValueError, 'layer 1 recomputed forward'):
            self.run_helper()
        self.assert_hooks_clean()

    def test_replay_exception_and_progress_exception_cleanup(self):
        self.decoder.layers[1].fail = True
        with self.assertRaisesRegex(RuntimeError, 'deliberate replay'):
            self.run_helper()
        self.assert_hooks_clean()
        self.decoder.layers[1].fail = False
        def cancel(event):
            if event['stage'] == 'layer_forward_exact':
                raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            self.run_helper(progress=cancel)
        self.assert_hooks_clean()

    def test_scope_and_frozen_guards(self):
        for projections in ([], [(0, 'gate'), (0, 'gate')], [(3, 'gate')], [(0, 'bad')]):
            with self.subTest(projections=projections), self.assertRaises(ValueError):
                layerwise_gradients(self.model, self.decoder, self.ids, self.positions, self.targets, projections)
        self.decoder.config._attn_implementation = 'sdpa'
        with self.assertRaisesRegex(ValueError, 'eager'):
            self.run_helper()
        self.decoder.config._attn_implementation = 'eager'
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            self.run_helper()
        self.model.eval()
        self.model.lm_head.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            self.run_helper()
        self.model.lm_head.weight.requires_grad_(False)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, 'autograd'):
            self.run_helper()
        self.assert_hooks_clean()

    def test_nonfinite_boundary_rejected_and_hooks_removed(self):
        with torch.no_grad():
            self.decoder.embed_tokens.weight[1, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite layer boundary'):
            self.run_helper()
        self.assert_hooks_clean()


if __name__ == '__main__':
    unittest.main()
