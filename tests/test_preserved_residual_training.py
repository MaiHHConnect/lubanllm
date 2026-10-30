"""The write decision retains fixed baselines and optimizer rejection semantics."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch

from train_preserved_residual import EXPANDED_SCALES, checkpoint, select_update
import train_preserved_residual as training


class PreservedResidualTrainingTests(unittest.TestCase):
    def setUp(self):
        self.current = torch.zeros(1, 2)
        self.proposed = torch.tensor([[0., 1.]])
        self.basis = torch.tensor([[1.], [0.]])
        self.baseline = torch.tensor([1., 2.])

    def choose(self, new, old, current=None, proposed=None, before=1.):
        return select_update(self.current if current is None else current,
            self.proposed if proposed is None else proposed, self.basis, self.baseline,
            before, new, old)

    def test_conflict_cannot_be_averaged_away_and_smaller_feasible_step_can_write(self):
        selected, trials = self.choose(lambda x: 1. - float(x[0, 1]),
            lambda x: torch.tensor([1.1, .1]) if x[0, 1] > .1 else self.baseline.clone())
        self.assertEqual(selected[0, 1], .0625)
        self.assertEqual(trials[0]['reason'], 'fixed_baseline_protection')
        self.assertTrue(trials[-1]['accepted'])
        self.assertTrue(torch.equal(self.baseline, torch.tensor([1., 2.])))

    def test_fixed_baseline_never_becomes_previous_candidate_loss(self):
        selected, _ = self.choose(lambda x: .5, lambda x: torch.tensor([.9, 1.9]))
        second, trials = self.choose(lambda x: .4, lambda x: torch.tensor([1.001, 1.8]),
            current=selected, before=.5)
        self.assertTrue(torch.equal(second, selected))
        self.assertTrue(all(t['reason'] == 'fixed_baseline_protection' for t in trials))
        self.assertTrue(torch.equal(self.baseline, torch.tensor([1., 2.])))

    def test_expanded_recipe_full_step_still_requires_every_fixed_constraint(self):
        selected, trials = select_update(self.current, self.proposed, self.basis, self.baseline, 2.,
            lambda x: 2. - float(x[0, 1]), lambda x: self.baseline.clone(), scales=EXPANDED_SCALES)
        self.assertTrue(torch.equal(selected, self.proposed))
        self.assertEqual(trials[0]['scale'], 1.)
        selected, trials = select_update(self.current, self.proposed, self.basis, self.baseline, 2.,
            lambda x: 2. - float(x[0, 1]),
            lambda x: torch.tensor([1.001, 1.]) if x[0, 1] > .3 else self.baseline.clone(),
            scales=EXPANDED_SCALES)
        self.assertEqual(float(selected[0, 1]), .25)
        self.assertEqual(trials[0]['reason'], 'fixed_baseline_protection')

    def test_no_improvement_never_counts_as_learning_or_checks_old_unnecessarily(self):
        def unexpected(_):
            self.fail('No-benefit candidate must not be selected')
        selected, trials = self.choose(lambda x: 1., unexpected)
        self.assertTrue(torch.equal(selected, self.current))
        self.assertTrue(all(t['reason'] == 'no_supervised_improvement' for t in trials))

    def test_actual_scaled_subspace_violation_is_rejected(self):
        selected, trials = self.choose(lambda x: .5, lambda x: self.baseline,
                                      proposed=torch.tensor([[1., 1.]]))
        self.assertTrue(torch.equal(selected, self.current))
        self.assertTrue(all(t['reason'] == 'scaled_subspace_rounding' for t in trials))

    def test_nonfinite_losses_abort_instead_of_accepting(self):
        with self.assertRaises(ValueError):
            self.choose(lambda x: float('nan'), lambda x: self.baseline)
        with self.assertRaises(ValueError):
            self.choose(lambda x: .5, lambda x: torch.tensor([float('nan'), 2.]))

    def test_adam_rejection_restores_parameter_but_keeps_real_momentum_and_checkpoint(self):
        delta = torch.nn.Parameter(self.current.clone())
        optimizer = torch.optim.Adam([delta], lr=.001)
        delta.grad = torch.ones_like(delta)
        optimizer.step()
        old_moment = optimizer.state[delta]['exp_avg']
        before = old_moment.clone()
        with torch.no_grad():
            delta.copy_(self.current)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'recovery.pt'
            # CPU-only test simulates a device transfer allocating a new
            # tensor, so a shallow state_dict mutation cannot pass unnoticed.
            with patch.object(torch.Tensor, 'cpu', lambda value: value.clone()):
                checkpoint(path, delta, optimizer, 1, 0, 'original', self.baseline)
            saved = torch.load(path, weights_only=True)
        self.assertIs(optimizer.state[delta]['exp_avg'], old_moment)
        self.assertTrue(torch.equal(old_moment, before))
        self.assertTrue(torch.equal(saved['delta'], self.current))
        self.assertEqual(saved['steps'], 1)
        self.assertEqual(saved['accepted_updates'], 0)
        self.assertEqual(float(saved['optimizer']['state'][0]['step']), 1.)
        delta.grad = torch.zeros_like(delta)
        optimizer.step()
        self.assertTrue(bool((delta != self.current).any()))

    def test_successor_shared_loop_450_uses_frozen_parent_and_only_E_optimizer(self):
        from test_cached_training import fixture
        from study_core_constraint import capture
        from luban.core_update import apply_core_residual
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime, rows, tokens = fixture(folder, torch.bfloat16)
            runtime.model.eval().requires_grad_(False)
            runtime.device = torch.device('cpu')
            runtime.tokenizer.eos_token_id = 2
            runtime._tokens = lambda *a, **k: tokens[rows[0]['id']][0][:, :4]
            runtime._release_model = unittest.mock.Mock()
            for row in rows:
                row['provenance'] = json.dumps({'messages': [{'role': 'user', 'content': 'fixture TRAIN'}]})
            new, probe, replay = rows
            dev = {**new, 'id': 'dev-only', 'purpose': 'validation'}
            groups = {'new_training': [new], 'new_validation': [dev], 'replay_training': [replay], 'replay_validation': []}
            def record(row):
                return {'id': row['id'], 'raw_completion_token_ids': [2], 'complete': True,
                        'passed': True, 'correct': True, 'forbidden_found': [], 'max_new_tokens': 8,
                        'branch_enabled': False, 'reference_used': False, 'thinking': 'off', 'do_sample': False}
            final = [record(row) for row in [new, dev, replay, probe]]
            final[0]['passed'] = False
            final[1]['passed'] = False
            reference = runtime.branch.base.down_proj.weight.detach().clone()
            parent_delta = torch.zeros_like(reference, dtype=torch.float32)
            parent_delta[0, 0], parent_delta[0, -1] = .3, .2
            parent_before = parent_delta.clone()
            basis = torch.eye(parent_delta.shape[1])[:, :3]
            binding = {'round_directory': 'parent', 'verification_directory': 'verified',
                       'protocol': 'parent', 'identity': runtime.identity}
            parent = {'groups': groups, 'data': {'reference': reference,
                'caches': {new['id']: capture(runtime, new)}}, 'basis': basis,
                'checkpoint': {'delta': parent_delta}, 'final': final, 'binding': binding}
            contract = {'parent_binding': binding, 'mastered_train_ids': [], 'round_index': 1,
                'arithmetic': 'one_fp32_parent_plus_increment', 'protected_raw': {probe['id']: final[-1]},
                'original_correct_replay_ids': [replay['id']], 'parent_correct_replay_ids': [replay['id']],
                'protection_entries': [{'id': probe['id']}, {'id': replay['id']}],
                'inherited_bounds': {probe['id']: 2., replay['id']: 2.}}
            (root / 'preparation.json').write_text(json.dumps({'status': 'prepared_no_optimizer_updates',
                                                              'identity': runtime.identity}))
            (root / 'material.json').write_text(json.dumps({'groups': groups}))
            def objective(runtime, cache, original, deployed):
                if cache['ids'].shape[1] == tokens[new['id']][0].shape[1]:
                    return (deployed[0, -1] - .4).square()
                return deployed.sum() * 0 + 2.
            def logits(runtime, cache, original, deployed):
                return (deployed[0, -1] * 0 + torch.zeros(1, 2))
            observed_totals = []
            def residual(inputs, output, total):
                observed_totals.append(total.detach().clone())
                return apply_core_residual(inputs, output, total)
            def generate(runtime, all_rows):
                self.assertEqual([row['id'] for row in all_rows], [row['id'] for row in [new, dev, replay, probe]])
                with torch.no_grad():
                    runtime.model(tokens[new['id']][0], use_cache=False)
                return final
            with patch('residual_successor.load_parent', return_value=parent), patch(
                    'residual_successor.prepare_successor', return_value=(basis, {}, contract)), patch.object(
                    training, 'runtime_for', return_value=runtime), patch.object(training, 'probes', return_value=[probe]), patch.object(
                    training, 'loss', side_effect=objective), patch.object(training, 'answer_logits', side_effect=logits), patch.object(
                    training, '_generation_loaded', side_effect=generate), patch.object(training, 'event'), patch.object(
                    training, 'apply_core_residual', side_effect=residual), patch('torch.backends.mps.is_available', return_value=False):
                training.run(root, parent_round=root / 'parent', parent_verification=root / 'verified')
            destination = next(root.glob('fixed450-residual-v4-successor-*'))
            saved = torch.load(destination / 'recovery.pt', weights_only=True)
            report = json.loads((destination / 'result.json').read_text())
            trace = [json.loads(line) for line in (destination / 'steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(trace), 450)
            self.assertEqual(report['optimizer_steps'], 450)
            self.assertEqual(report['accepted_updates'], 225)
            self.assertFalse(report['candidate_qualified'])  # DEV remains a final gate.
            self.assertTrue(report['inherited_raw_unchanged'])
            self.assertEqual(float(saved['optimizer']['state'][0]['step']), 450.)
            self.assertTrue(torch.equal(saved['delta'], parent_delta + saved['increment']))
            self.assertTrue(torch.equal(observed_totals[0], saved['delta']))
            self.assertGreater(float((saved['delta'] @ basis).norm()), .2)
            self.assertLess(float((saved['increment'] @ basis).norm()), 1e-6)
            self.assertTrue(torch.equal(parent_delta, parent_before))
            self.assertTrue(torch.equal(runtime.branch.base.down_proj.weight, reference))
            self.assertFalse(runtime.branch.base.down_proj._forward_hooks)
            runtime._release_model.assert_called_once()

    def test_terminal_tensor_check_failure_still_releases_and_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weight = torch.nn.Parameter(torch.zeros(1, 2, dtype=torch.bfloat16))
            runtime = SimpleNamespace(active_core=None, identity='fixture', device=torch.device('cpu'),
                                      _release_model=unittest.mock.Mock())
            groups = {'new_training': [{'id': 'new'}], 'new_validation': [], 'replay_training': []}
            parent = {'groups': groups, 'data': {'reference': weight.detach().clone()},
                      'basis': self.basis, 'binding': {'round_directory': 'parent'}}
            (root / 'preparation.json').write_text(json.dumps({'status': 'prepared_no_optimizer_updates', 'identity': 'fixture'}))
            (root / 'material.json').write_text(json.dumps({'groups': groups}))
            with patch('residual_successor.load_parent', return_value=parent), patch.object(
                    training, 'runtime_for', return_value=runtime), patch.object(training, '_target',
                    side_effect=[('weight', weight), RuntimeError('authored terminal tensor readback failure')]), patch.object(
                    training, 'capture', side_effect=RuntimeError('authored prepare failure')), patch.object(training, 'event'):
                with self.assertRaisesRegex(RuntimeError, 'terminal tensor readback'):
                    training.run(root, parent_round=root / 'parent', parent_verification=root / 'verified')
            runtime._release_model.assert_called_once()
            destination = next(root.glob('fixed450-residual-v4-successor-*'))
            receipt = json.loads((destination / 'result.json').read_text())
            self.assertEqual(receipt['status'], 'failed_release')
            self.assertIn('terminal tensor readback', receipt['release_error'])
            self.assertEqual(receipt['optimizer_steps'], 0)


if __name__ == '__main__':
    unittest.main()
