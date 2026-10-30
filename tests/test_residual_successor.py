"""CPU contracts and official tiny BF16 graph for frozen D / constrained E."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_cached_training import fixture
from study_core_constraint import capture
from train_preserved_residual import checkpoint, loss
import residual_successor as successor


class ResidualSuccessorTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.directory = Path(self.folder.name).resolve() / 'child'
        self.directory.mkdir()
        fork = torch.random.fork_rng(devices=[])
        fork.__enter__()
        self.addCleanup(fork.__exit__, None, None, None)
        torch.manual_seed(42)
        self.runtime, rows, tokens = fixture(self.folder.name, torch.bfloat16)
        for row in rows:
            provenance = json.loads(row['provenance'])
            provenance['messages'] = [{'role': 'user', 'content': 'synthetic TRAIN context'}]
            row['provenance'] = json.dumps(provenance)
        self.runtime.model.eval().requires_grad_(False)
        self.runtime.tokenizer.eos_token_id = 2
        self.runtime._tokens = lambda *a, **k: tokens[rows[1]['id']][0][:, :7]
        self.parent_delta = torch.randn(self.runtime.branch.base.down_proj.weight.shape) * .02
        self.probes = [{**rows[0], 'id': 'probe-' + str(i)} for i in range(4)]
        self.runtime._sample_tokens = lambda row: tokens[rows[0]['id'] if row['id'].startswith('probe-') else row['id']]
        self.new, self.replay = rows[1], rows[2]
        groups = {'new_training': [self.new], 'new_validation': [{**rows[0], 'purpose': 'validation'}],
                  'replay_training': [self.replay], 'replay_validation': []}
        self.caches = {row['id']: capture(self.runtime, row) for row in self.probes + [self.new, self.replay]}
        short_x = self.caches[self.probes[0]['id']]['x'].float().reshape(-1, 64)
        basis, _ = successor.extend_protected_basis(torch.empty(64, 0), short_x)
        def observed(row, raw):
            return {'id': row['id'], 'raw_completion_token_ids': raw, 'passed': True,
                    'correct': True, 'complete': True, 'forbidden_found': [], 'max_new_tokens': 16,
                    'branch_enabled': False, 'reference_used': False, 'thinking': 'off', 'do_sample': False}
        original_ids = [row['id'] for row in self.probes + [self.replay]]
        baseline = [observed(row, [23, 2]) for row in self.probes + [self.replay]]
        final = baseline + [observed(self.new, [11, 12, 2])]
        fixed = {'ids': original_ids, 'losses': [10.] * len(original_ids)}
        self.parent = dict(groups=groups, rows=self.probes + [self.new, self.replay],
            result={'protocol': 'parent', 'candidate_qualified': False}, checkpoint={'delta': self.parent_delta},
            data={'reference': self.runtime.branch.base.down_proj.weight.detach().clone(), 'caches': self.caches},
            fixed=fixed, baseline=baseline, final=final, basis=basis, successor=None,
            binding={'round_directory': str(self.directory.parent / 'parent'),
                     'verification_directory': str(self.directory.parent / 'verified'),
                     'protocol': 'parent', 'identity': 'fixture'})

    def prepare(self):
        # Contract test supplies fixed actual-graph predictor inputs; greedy KV
        # mechanics are tested in test_residual_round, not simulated as evidence.
        def trajectory(model, tokenizer, row, observed):
            value = capture(self.runtime, row, generated=observed)
            inputs = value['x'][:, :-1][value['labels'][:, 1:] != -100]
            return {'inputs': inputs, 'raw_completion_token_ids': observed['raw_completion_token_ids']}
        with patch('probe_core_constraint.probes', return_value=self.probes), patch(
                'verify_residual_round.capture_greedy_trajectory', side_effect=trajectory):
            return successor.prepare_successor(self.runtime, self.parent, self.caches,
                                               self.parent_delta + torch.zeros_like(self.parent_delta))

    def contract(self):
        basis, extension, contract = self.prepare()
        values = [float(loss(self.runtime, self.caches[entry['id']],
                            self.parent['data']['reference'], self.parent_delta).detach())
                  for entry in contract['protection_entries']]
        contract['parent_losses'] = values
        fixed = {'ids': [row['id'] for row in contract['protection_entries']],
                 'losses': [min(contract['inherited_bounds'].get(row['id'], value), value)
                            for row, value in zip(contract['protection_entries'], values)], 'basis_rank': basis.shape[1]}
        e = torch.randn_like(self.parent_delta) * .001
        e -= (e @ basis) @ basis.T
        e -= (e @ basis) @ basis.T
        saved = {'increment': e, 'delta': self.parent_delta + e,
                 'fixed_protection_losses': torch.tensor(fixed['losses']), 'parent_binding': self.parent['binding']}
        result = {'protocol': successor.PROTOCOL, 'parent_binding': self.parent['binding'],
                  'mastered_train_ids': contract['mastered_train_ids'], 'basis': {'rank': basis.shape[1]}, 'extension': extension}
        torch.save(contract, self.directory / 'successor.pt')
        return result, saved, fixed, {'basis': basis, 'extension': extension}, contract

    def validate(self, result, saved, fixed, artifact):
        with patch.object(successor, 'load_parent', return_value=self.parent), patch(
                'probe_core_constraint.probes', return_value=self.probes):
            return successor.validate_successor_contract(self.directory, result, saved, fixed, artifact, 'fixture')

    def test_nonzero_parent_full_cached_math_zero_E_and_failure_cleanup(self):
        before, old = self.parent_delta.clone(), self.runtime.branch.base.down_proj.weight.clone()
        audit = successor.parent_equivalence(self.runtime, self.caches[self.new['id']], self.parent_delta)
        self.assertTrue(audit['zero_E_parent_logits_exact'])
        self.assertTrue(audit['nonzero_parent_math']['exact_equivalence'])
        self.assertTrue(torch.equal(self.parent_delta, before))
        self.assertTrue(torch.equal(self.runtime.branch.base.down_proj.weight, old))
        self.assertFalse(self.runtime.branch.base.down_proj._forward_hooks)
        with patch('diagnose_residual_generation.full_cached_math', side_effect=RuntimeError('math failure')):
            with self.assertRaisesRegex(RuntimeError, 'math failure'):
                successor.parent_equivalence(self.runtime, self.caches[self.new['id']], self.parent_delta)
        self.assertFalse(self.runtime.branch.base.down_proj._forward_hooks)

    def test_actual_total_can_leave_span_while_only_E_is_constrained_and_raw_differs_from_gold(self):
        result, saved, fixed, artifact, contract = self.contract()
        checked = self.validate(result, saved, fixed, artifact)
        self.assertGreater(float((saved['delta'] @ artifact['basis']).norm()), .01)
        self.assertLess(checked['basis_validation']['increment_subspace_residual'], 1e-6)
        gold, raw = contract['protection_entries'][-2:]
        self.assertFalse(torch.equal(gold['ids'], raw['ids']))
        self.assertEqual(raw['ids'][raw['labels'] != -100].tolist(), [11, 12, 2])
        self.assertGreater(artifact['extension']['added_rank'], 0)
        self.assertLess(float((self.parent['basis'] - artifact['basis'] @ (
            artifact['basis'].T @ self.parent['basis'])).abs().max()), 2e-5)

    def test_contract_rejects_changed_total_threshold_dev_raw_and_geometry(self):
        result, saved, fixed, artifact, contract = self.contract()
        for kind in ('total', 'threshold', 'dev', 'raw', 'increment', 'counts'):
            r, s, f, a, c = copy.deepcopy((result, saved, fixed, artifact, contract))
            if kind == 'total': s['delta'][0, 0] += .1
            if kind == 'threshold': f['losses'][0] += .1
            if kind == 'dev': c['mastered_train_ids'].append(self.parent['groups']['new_validation'][0]['id'])
            if kind == 'raw': c['protected_raw'][self.new['id']]['raw_completion_token_ids'][0] += 1
            if kind == 'increment':
                s['increment'] += .1 * torch.ones(s['increment'].shape[0], 1) @ a['basis'][:, :1].T
                s['delta'] = self.parent_delta + s['increment']
            if kind == 'counts':
                a['extension']['actual_kv_positions'] += 1
                r['extension'] = a['extension']
            torch.save(c, self.directory / 'successor.pt')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(r, s, f, a)

    def test_next_successor_retains_existing_raw_and_does_not_reexpand_or_read_dev(self):
        _, _, _, _, contract = self.contract()
        self.parent['successor'] = contract
        self.parent['fixed'] = {'ids': [row['id'] for row in contract['protection_entries']],
                                'losses': contract['parent_losses']}
        self.parent['basis'], _, _ = self.prepare()
        with patch('probe_core_constraint.probes', return_value=self.probes), patch(
                'verify_residual_round.capture_greedy_trajectory', side_effect=AssertionError('old S must not recapture KV')):
            basis, extension, next_contract = successor.prepare_successor(
                self.runtime, self.parent, self.caches, self.parent_delta)
        self.assertTrue(torch.equal(basis, self.parent['basis']))
        self.assertEqual(extension['added_rank'], 0)
        self.assertEqual(extension['actual_kv_positions'], 0)
        self.assertEqual(next_contract['protected_raw'], contract['protected_raw'])
        self.assertEqual(next_contract['round_index'], 2)
        self.assertEqual(next_contract['mastered_train_ids'], [self.new['id']])

    def test_failed_KV_capture_removes_parent_hook_and_full_span_refuses_learning(self):
        with patch('probe_core_constraint.probes', return_value=self.probes), patch(
                'verify_residual_round.capture_greedy_trajectory', side_effect=RuntimeError('bad KV')):
            with self.assertRaisesRegex(RuntimeError, 'bad KV'):
                successor.prepare_successor(self.runtime, self.parent, self.caches, self.parent_delta)
        self.assertFalse(self.runtime.branch.base.down_proj._forward_hooks)
        with patch.object(successor, 'extend_protected_basis', return_value=(torch.eye(64), {})):
            with self.assertRaisesRegex(ValueError, 'nonempty trainable complement'):
                self.prepare()

    def test_checkpoint_keeps_E_optimizer_and_actual_FP32_total_separate(self):
        increment = torch.nn.Parameter(torch.zeros_like(self.parent_delta))
        optimizer = torch.optim.Adam([increment], lr=1e-5, eps=1e-6)
        increment.grad = torch.ones_like(increment)
        optimizer.step()
        with torch.no_grad():
            increment.zero_()  # Rejection consumes the real Adam proposal.
        binding = {'parent_binding': self.parent['binding']}
        path = self.directory / 'recovery.pt'
        checkpoint(path, increment, optimizer, 1, 0, 'fixture', torch.ones(2),
                   successor=binding, parent_delta=self.parent_delta)
        state = torch.load(path, weights_only=True)
        self.assertTrue(torch.equal(state['delta'], self.parent_delta))
        self.assertFalse(bool(state['increment'].count_nonzero()))
        self.assertEqual(float(state['optimizer']['state'][0]['step']), 1.)
        self.assertTrue(bool(state['optimizer']['state'][0]['exp_avg'].count_nonzero()))
        self.assertEqual(state['parent_binding'], self.parent['binding'])

    def test_cycle_is_rejected_before_reading_parent(self):
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            successor.validate_successor_contract(self.directory, {}, {}, {}, {}, 'fixture',
                                                  ancestry=(str(self.directory),))

    def test_later_correct_replay_becomes_a_permanent_parent_obligation(self):
        _, _, _, _, contract = self.contract()
        added = {**self.replay, 'id': 'formerly-wrong-replay'}
        self.parent['groups']['replay_training'].append(added)
        self.parent['final'].append({**self.parent['final'][-1], 'id': added['id'], 'passed': True})
        self.parent['successor'] = contract
        original, current = successor.replay_correct(self.parent)
        self.assertEqual(original, [self.replay['id']])
        self.assertEqual(current, [self.replay['id'], added['id']])
        contract['parent_correct_replay_ids'] = current
        self.parent['final'][-1]['passed'] = False
        with self.assertRaisesRegex(ValueError, 'subsequently correct replay'):
            successor.replay_correct(self.parent)

    def test_independent_audit_releases_runtime_even_if_memory_query_fails(self):
        output = self.directory / 'audit'
        output.mkdir()
        self.runtime._model_process_lock = object()
        def release():
            self.runtime._model_process_lock = None
        with patch.object(Path, 'is_mount', return_value=True), patch.object(
                Path, 'is_relative_to', return_value=True), patch('tempfile.mkdtemp', return_value=str(output)), patch(
                'train_luban_identity.runtime_for', return_value=self.runtime), patch(
                'verify_local_review.model_identity', return_value=self.runtime.identity), patch.object(
                successor, 'load_parent', return_value=self.parent), patch.object(
                successor, 'parent_equivalence', return_value={'zero_E_parent_logits_exact': True}), patch(
                'torch.backends.mps.is_available', return_value=True), patch(
                'torch.mps.driver_allocated_memory', side_effect=RuntimeError('authored memory query failure')), patch.object(
                self.runtime, '_release_model', side_effect=release, create=True) as released, patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'authored memory query failure'):
                successor.audit_parent(Path(self.parent['binding']['round_directory']),
                                       Path(self.parent['binding']['verification_directory']))
        released.assert_called_once()
        report = json.loads((output / 'result.json').read_text())
        self.assertEqual(report['status'], 'failed_release')
        self.assertTrue(report['model_lock_released'])

    def test_parent_admission_matches_actual_independent_delta_raw_and_fixed_losses(self):
        from safetensors.torch import save_file
        parent_dir = Path(self.parent['binding']['round_directory'])
        verification = Path(self.parent['binding']['verification_directory'])
        verification.mkdir()
        self.parent['data']['basis'] = self.parent['basis']
        self.parent['result'].update(probe_tokens_unchanged=True, baseline_correct_replay=1,
                                     retained_correct_replay=1)
        loaded = tuple(self.parent[key] for key in
                       ('groups', 'rows', 'result', 'checkpoint', 'data', 'fixed', 'baseline', 'final'))
        artifact = verification / 'diagnostic-residual.safetensors'
        save_file({'delta': self.parent_delta}, artifact, metadata={'base_identity': 'fixture'})
        report = {'status': 'diagnostic_verification_passed', 'identity': 'fixture',
                  'round_directory': str(parent_dir), 'source_protocol': 'parent',
                  'source_candidate_qualified': False, 'production_adoption': False,
                  'training_performed': False, 'teacher_calls': 0, 'diagnostic_delta_artifact': str(artifact)}
        report.update({key: True for key in ('carrier_consistent', 'generation_tokens_exact',
            'generation_validators_exact', 'full_model_loss_guard_passed',
            'original_parameter_versions_unchanged', 'model_lock_released')})
        (verification / 'result.json').write_text(json.dumps(report))
        (verification / 'generation.json').write_text(json.dumps(self.parent['final']))
        losses = [{'id': key, 'fixed_baseline': value, 'full_model_loss': value - 1, 'nonincreasing': True}
                  for key, value in zip(self.parent['fixed']['ids'], self.parent['fixed']['losses'])]
        (verification / 'protection.json').write_text(json.dumps(losses))
        with patch('verify_residual_round.load_round', return_value=loaded) as loader:
            accepted = successor.load_parent(parent_dir, verification, self.directory.parent, 'fixture')
            loader.assert_called_once_with(parent_dir, 'fixture', ancestry=())
            self.assertEqual(accepted['binding'], self.parent['binding'])
            wrong = copy.deepcopy(self.parent['final'])
            wrong[-1]['raw_completion_token_ids'][0] += 1
            (verification / 'generation.json').write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, 'Independent raw'):
                successor.load_parent(parent_dir, verification, self.directory.parent, 'fixture')
            (verification / 'generation.json').write_text(json.dumps(self.parent['final']))
            save_file({'delta': self.parent_delta + .1}, artifact, metadata={'base_identity': 'fixture'})
            with self.assertRaisesRegex(ValueError, 'Independently loaded residual'):
                successor.load_parent(parent_dir, verification, self.directory.parent, 'fixture')


if __name__ == '__main__':
    unittest.main()
