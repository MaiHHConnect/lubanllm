"""Tiny real hybrid Qwen CPU diagnostics, without MPS or training updates."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import torch

from test_cached_training import fixture
import diagnose_residual_generation as diagnostic
from luban.core_update import apply_core_residual
from study_core_constraint import capture


class ResidualGenerationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        fork = torch.random.fork_rng(devices=[])
        fork.__enter__()
        self.addCleanup(fork.__exit__, None, None, None)
        torch.manual_seed(42)
        self.runtime, rows, self.tokens = fixture(self.folder.name, torch.bfloat16)
        self.runtime.model.eval().requires_grad_(False)
        self.runtime.tokenizer.eos_token_id = 2
        self.rows = rows[:1]
        self.weight = self.runtime.branch.base.down_proj.weight
        self.delta = (torch.randn(self.weight.shape) * .02).requires_grad_()
        self.caches = {row['id']: capture(self.runtime, row) for row in self.rows}

    def observed(self):
        row = self.rows[0]
        ids, labels = self.tokens[row['id']]
        start = int(labels[0].ne(-100).nonzero()[0])
        handle = self.runtime.branch.base.down_proj.register_forward_hook(
            lambda module, args, output: apply_core_residual(args[0], output, self.delta))
        try:
            with torch.no_grad():
                output = self.runtime.model.generate(ids[:, :start], max_new_tokens=8,
                    do_sample=False, pad_token_id=2, use_cache=True)
        finally:
            handle.remove()
        return {row['id']: {'raw_completion_token_ids': output[0, start:].tolist()}}

    def test_nonzero_full_cached_math_gold_eos_and_actual_generation_alignment(self):
        before = self.weight.clone()
        delta_before = self.delta.detach().clone()
        result = diagnostic.diagnose(self.runtime, self.rows, self.caches, self.delta, self.observed())[0]
        self.assertTrue(result['math']['exact_equivalence'])
        self.assertTrue(result['observed_greedy_reproduced'])
        self.assertEqual(result['gold_token_ids_including_training_eos'], [23, 2])
        self.assertEqual(result['gold_forward_lengths'], [4, 1])
        self.assertEqual([row['target_token_id'] for row in result['trained_gold_kv']], [23, 2])
        self.assertEqual(len(result['trained_same_prefix_controls']), 3)
        self.assertEqual(len(result['zero_same_prefix_controls']), 3)
        self.assertEqual(result['trained_same_prefix_controls'][0]['prefix_token_ids'], [3, 7, 13, 19])
        self.assertIn('gold_at_matching_prefix', result['trained_same_prefix_controls'][0])
        self.assertTrue(result['original_parameter_objects_versions_and_delta_state_unchanged'])
        self.assertTrue(torch.isfinite(torch.tensor(result['zero_full_sequence_ce'])))
        self.assertTrue(torch.equal(self.weight, before))
        self.assertTrue(torch.equal(self.delta, delta_before))
        self.assertIsNone(self.delta.grad)
        self.assertFalse(self.runtime.branch.base.down_proj._forward_hooks)

    def test_math_gate_also_checks_unsupervised_prefix_logits(self):
        original = diagnostic.residual_logits

        def changed(*args):
            values = original(*args).clone()
            values[0, 0, 0] += 1
            return values

        with patch.object(diagnostic, 'residual_logits', side_effect=changed):
            result = diagnostic.diagnose(self.runtime, self.rows, self.caches, self.delta, self.observed())[0]
        self.assertFalse(result['math']['exact_equivalence'])
        self.assertFalse(result['math']['logits']['exact'])
        self.assertTrue(result['math']['loss']['exact'])
        self.assertTrue(result['math']['gradient']['exact'])

    def test_run_rejects_different_generation_prefix_and_records_release_failure(self):
        folder = Path(self.folder.name) / 'round'
        folder.mkdir()
        row = self.rows[0]
        row['prompt'] = 'A fixed tiny fixture question'
        self.runtime.identity = 'original'
        self.runtime.active_core = None
        self.runtime._model_process_lock = object()
        self.runtime._tokens = lambda *args, **kwargs: torch.tensor([[99]])
        self.runtime._release_model = lambda: (_ for _ in ()).throw(RuntimeError('release failed'))
        loaded = ({}, [], {'candidate_qualified': False}, {'delta': self.delta.detach()},
                  {'reference': self.weight.detach().clone(), 'caches': self.caches}, None, None,
                  [{'id': row['id'], 'max_new_tokens': 8}])
        with patch.object(Path, 'is_mount', return_value=True), \
                patch.object(Path, 'is_relative_to', return_value=True), \
                patch.object(diagnostic, 'model_identity', return_value='original'), \
                patch.object(diagnostic, 'load_round', return_value=loaded), \
                patch.object(diagnostic, 'select_train', return_value=self.rows), \
                patch.object(diagnostic, 'runtime_for', return_value=self.runtime), \
                patch.object(torch.backends.mps, 'is_available', return_value=True), \
                patch.object(torch.mps, 'driver_allocated_memory', return_value=0), \
                patch.object(diagnostic, 'diagnose') as execute:
            with self.assertRaisesRegex(RuntimeError, 'release failed'):
                diagnostic.run(folder)
        execute.assert_not_called()
        report = json.loads(next(Path(self.folder.name).glob('residual-generation-diagnostic-*/result.json')).read_text())
        self.assertEqual(report['status'], 'failed_release')
        self.assertIn('generation prompt', report['error'])
        self.assertIn('release failed', report['release_error'])

    def test_failure_removes_hook_and_rejects_dev_and_zero_delta(self):
        with patch.object(diagnostic, 'residual_logits', side_effect=RuntimeError('authored cache failure')):
            with self.assertRaisesRegex(RuntimeError, 'authored cache failure'):
                diagnostic.diagnose(self.runtime, self.rows, self.caches, self.delta, self.observed())
        self.assertFalse(self.runtime.branch.base.down_proj._forward_hooks)
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            diagnostic.diagnose(self.runtime, [{**self.rows[0], 'purpose': 'validation'}], self.caches, self.delta, {})
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            diagnostic.diagnose(self.runtime, self.rows, self.caches, torch.zeros_like(self.delta, requires_grad=True), {})

    def test_metrics_distinguish_low_average_ce_from_sequence_argmax(self):
        stats = [diagnostic.token_stats(torch.tensor([5., 0., 0.]), 0),
                 diagnostic.token_stats(torch.tensor([1., 1.01, -8.]), 0)]
        records = [{**value, 'prediction_index': index, 'tf_top1_token_id': value['top1_token_id'],
                    'full_sequence_tf_vs_kv': {'passed': True}} for index, value in enumerate(stats)]
        summary = diagnostic.summarize_gold(records)
        self.assertLess(summary['mean_ce'], .5)
        self.assertFalse(summary['every_gold_argmax_correct_including_eos'])
        self.assertEqual(summary['first_gold_argmax_error'], 1)
        self.assertEqual(stats[1]['target_rank'], 2)
        self.assertLess(stats[1]['target_minus_highest_other_logit'], 0)

    def test_controls_use_first_divergence_and_prefix_after_repeat(self):
        raw, gold = [3, 4, 8, 9, 8, 9, 7, 2], [3, 4, 5, 2]
        divergence, repeat, positions = diagnostic.prefix_positions(raw, gold)
        self.assertEqual(divergence, 2)
        self.assertEqual(repeat['after_prefix_tokens'], 6)
        self.assertIn(2, positions)
        self.assertIn(6, positions)
        self.assertEqual(repeat['token_ids'], [8, 9])


if __name__ == '__main__':
    unittest.main()
