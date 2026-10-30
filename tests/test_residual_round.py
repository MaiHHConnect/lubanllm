"""CPU protocol counterexamples and tiny official graph; never execute 2B."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_cached_training import fixture
import verify_residual_round as verify


IDENTITY = 'a' * 64
CONTEXT = {'current_limit': 512, 'linear_bytes_per_token': 1,
           'quadratic_bytes_per_token_squared': 1, 'memory_budget_bytes': 1024 ** 3}


class ProtectionToleranceTests(unittest.TestCase):
    def test_only_new_inherited_entries_get_fixed_fp32_measurement_bound(self):
        successor = {'original_fixed': {'ids': ['old']}}
        self.assertEqual(verify.inherited_loss_tolerance('old', 1., successor), 0.)
        tolerance = verify.inherited_loss_tolerance('new', 1., successor)
        self.assertGreater(tolerance, 0.)
        self.assertLessEqual(1. + tolerance - 1., tolerance)
        self.assertEqual(verify.inherited_loss_tolerance('new', 1., None), 0.)


def checkpoint_fixture():
    reference = torch.zeros(2, 3, dtype=torch.bfloat16)
    result = {'status': 'fixed450_complete_research_only', 'optimizer_steps': 450,
        'steps_required': 450, 'seed': 42, 'original_runtime_weight_unchanged': True,
        'production_adoption': False, 'production_weights_changed': False,
        'final_loss_guard': {'rejected': False}, 'candidate_qualified': False,
        'protocol': 'preservation-first-residual-fixed450-v1',
        'precision': 'BF16 frozen model and deployed tail; FP32 unmerged delta/Adam',
        'identity': IDENTITY, 'accepted_updates': 18}
    fixed = {'ids': ['probe', 'replay'], 'losses': [1., 2.]}
    checkpoint = {'steps': 450, 'identity': IDENTITY, 'accepted_updates': 18,
        'production_adoption': False, 'delta': torch.zeros(2, 3),
        'fixed_protection_losses': torch.tensor([1., 2.]),
        'optimizer': {'param_groups': torch.optim.Adam([torch.nn.Parameter(torch.zeros(2, 3))],
            lr=1e-5, eps=1e-6).state_dict()['param_groups'],
            'state': {0: {'step': torch.tensor(450.), 'exp_avg': torch.zeros(2, 3),
                          'exp_avg_sq': torch.zeros(2, 3)}}}}
    return result, checkpoint, reference, fixed


class Tokenizer:
    eos_token_id = 63

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, context, **kwargs):
        self.calls.append((context, kwargs))
        return torch.tensor([[3, 7, 13, 19]]) if kwargs['tokenize'] else 'prefix'

    def encode(self, text, **kwargs):
        return [3, 7, 13, 19] if text == 'prefix' else [23]

    def decode(self, ids, **kwargs):
        return 'fixture'


def row():
    return {'id': 'authored', 'purpose': 'training', 'answer': 'fixture', 'provenance': json.dumps({
        'messages': [{'role': 'user', 'content': 'Authored context'}],
        'generation_check': {'type': 'exact', 'expected': 'fixture', 'max_new_tokens': 8}})}


class ResidualRoundTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def test_completed_checkpoint_allows_unmastered_candidate_without_adoption(self):
        verify.validate_checkpoint(*checkpoint_fixture(), IDENTITY)

    def test_new_recipes_bind_expansion_scales_and_stable_solver(self):
        base, *_ = checkpoint_fixture()
        for version, expanded in [('v2-expanded', True), ('v3-expanded-stable', True), ('v3-stable', False),
                                  ('v4-successor', True)]:
            result = {**base, 'protocol': 'preservation-first-residual-fixed450-' + version,
                'expanded_span': expanded, 'scales': ([1.] if expanded else []) + [.25, .0625, .015625, .00390625],
                'constraint_solver': 'fp32-feasible-first-double-projection-v1' if version.startswith(('v3', 'v4')) else None}
            verify.completed(result)
            for changed in ({'expanded_span': not expanded}, {'scales': [.25]}, {'protocol': result['protocol'] + '-unknown'}):
                with self.subTest(version=version, changed=changed), self.assertRaises(ValueError):
                    verify.completed({**result, **changed})
            if version.startswith(('v3', 'v4')):
                with self.assertRaisesRegex(ValueError, 'solver'):
                    verify.completed({**result, 'constraint_solver': 'legacy'})
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            verify.completed({**base, 'expanded_span': True})
        with self.assertRaisesRegex(ValueError, 'solver'):
            verify.completed({**base, 'constraint_solver': 'fp32-feasible-first-double-projection-v1'})

    def test_expanded_span_binds_old_directions_delta_and_metadata(self):
        original = torch.eye(3)[:, :1]
        extension = {'added_rank': 1, 'actual_kv_positions': 4, 'teacher_forced_positions': 4}
        result = {'basis': {'rank': 2}, 'extension': extension}
        artifact = {'basis': torch.eye(3)[:, :2], 'extension': extension}
        checkpoint = {'delta': torch.tensor([[0., 0., .1]])}
        fixed = {'basis_rank': 2}
        verify.validate_expanded_span(result, checkpoint, original, fixed, artifact)
        for changed_artifact, changed_checkpoint in (
                ({**artifact, 'basis': torch.eye(3)[:, 1:]}, checkpoint),
                (artifact, {'delta': torch.tensor([[.1, 0., 0.]])}),
                ({**artifact, 'extension': {**extension, 'added_rank': 0}}, checkpoint),
                ({**artifact, 'basis': torch.full((3, 2), float('nan'))}, checkpoint)):
            with self.assertRaises(ValueError):
                verify.validate_expanded_span(result, changed_checkpoint, original, fixed, changed_artifact)

    def test_round_rejects_incomplete_or_unprotected_or_changed_or_wrong_precision(self):
        for change in ({'status': 'running'}, {'optimizer_steps': 449}, {'steps_required': 900},
                {'original_runtime_weight_unchanged': False}, {'production_adoption': True},
                {'production_weights_changed': True}, {'final_loss_guard': {'rejected': True}},
                {'seed': 43}, {'precision': 'Q8'}, {'identity': 'b' * 64}):
            result, checkpoint, reference, fixed = checkpoint_fixture()
            result.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify.validate_checkpoint(result, checkpoint, reference, fixed, IDENTITY)

    def test_checkpoint_rejects_adam_step_baseline_shape_dtype_and_nonfinite(self):
        for mutation in (
            lambda c: c.update(steps=449), lambda c: c.update(identity='b' * 64),
            lambda c: c.update(accepted_updates=17), lambda c: c.update(production_adoption=True),
            lambda c: c.update(delta=torch.zeros(2, 3, dtype=torch.bfloat16)),
            lambda c: c.update(delta=torch.zeros(1, 3)),
            lambda c: c.update(delta=torch.full((2, 3), float('nan'))),
            lambda c: c.update(fixed_protection_losses=torch.tensor([1., 1.999])),
            lambda c: c['optimizer']['state'][0].update(step=torch.tensor(449.)),
            lambda c: c['optimizer']['state'][0].update(exp_avg=torch.full((2, 3), float('inf'))),
            lambda c: c['optimizer']['state'][0].update(exp_avg_sq=-torch.ones(2, 3)),
            lambda c: c['optimizer']['param_groups'][0].update(params=[0, 1])):
            result, checkpoint, reference, fixed = checkpoint_fixture()
            mutation(checkpoint)
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                verify.validate_checkpoint(result, checkpoint, reference, fixed, IDENTITY)

    def test_unfinished_round_rejected_before_tensor_or_model_loading(self):
        (self.root / 'result.json').write_text(json.dumps({'status': 'running'}))
        with patch.object(verify.torch, 'load') as load, self.assertRaises(ValueError):
            verify.load_round(self.root, IDENTITY)
        load.assert_not_called()

    def test_successor_checkpoint_is_explicit_and_legacy_cannot_claim_parent(self):
        result, checkpoint, reference, fixed = checkpoint_fixture()
        checkpoint.update(increment=torch.zeros_like(checkpoint['delta']), parent_binding={'identity': IDENTITY})
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            verify.validate_checkpoint(result, checkpoint, reference, fixed, IDENTITY)
        result.update(protocol=verify.SUCCESSOR_PROTOCOL, expanded_span=True,
            scales=[1., .25, .0625, .015625, .00390625],
            constraint_solver='fp32-feasible-first-double-projection-v1')
        verify.validate_checkpoint(result, checkpoint, reference, fixed, IDENTITY)
        for change in ({'increment': checkpoint['increment'].bfloat16()}, {'parent_binding': None},
                       {'increment': torch.full_like(checkpoint['increment'], float('inf'))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify.validate_checkpoint(result, {**checkpoint, **change}, reference, fixed, IDENTITY)

    def test_cyclic_parent_rejected_before_reading_round_files(self):
        with patch.object(verify, 'read') as read, self.assertRaisesRegex(ValueError, 'Cyclic'):
            verify.load_round(self.root, IDENTITY, ancestry=(str(self.root / 'nested' / '..'),))
        read.assert_not_called()

    def test_successor_load_delegates_geometry_and_binds_complete_parent_baseline(self):
        result, checkpoint, _, fixed = checkpoint_fixture()
        result.update(protocol=verify.SUCCESSOR_PROTOCOL, expanded_span=True,
            scales=[1., .25, .0625, .015625, .00390625],
            constraint_solver='fp32-feasible-first-double-projection-v1')
        groups = {'new_training': [{**row(), 'id': f'train-{index}'} for index in range(32)],
            'new_validation': [{**row(), 'id': 'dev'}], 'replay_training': [{**row(), 'id': 'replay'}],
            'replay_validation': []}
        records = [{'id': item['id'], 'raw_completion_token_ids': [2], 'branch_enabled': False,
                    'reference_used': False, 'thinking': 'off', 'do_sample': False}
                   for item in groups['new_training'] + groups['new_validation'] + groups['replay_training'] + verify.probes()]
        mapping = {'result.json': result, 'settings.json': verify.GENERATION_SETTINGS,
            'preparation.json': {'status': 'prepared_no_optimizer_updates', 'identity': IDENTITY},
            'material.json': {'groups': groups}, 'fixed_protection.json': fixed,
            'baseline_generation.json': records, 'final_generation.json': records}
        tensors = {'recovery.pt': checkpoint, 'prepared.pt': {'reference': SimpleNamespace(shape=(2048, 6144)),
            'basis': torch.eye(3)[:, :1], 'caches': {}},
            'expanded-basis.pt': {'basis': torch.eye(3)[:, :2]}}
        bound = {'parent_delta': torch.ones(2, 3), 'successor': {'format': 'residual-successor-v1'},
            'parent_bundle': {'final': records}, 'basis_validation': {'increment_subspace_residual': 0.}}
        with patch.object(verify, 'read', side_effect=lambda path: mapping[path.name]), \
                patch.object(verify.torch, 'load', side_effect=lambda path, **kwargs: tensors[path.name]), \
                patch.object(verify, 'validate_checkpoint'), \
                patch.object(verify, 'validate_expanded_span') as legacy, \
                patch('residual_successor.validate_successor_contract', return_value=bound) as validate:
            loaded = verify.load_round(self.root, IDENTITY)
            self.assertIs(loaded[4]['parent_delta'], bound['parent_delta'])
            self.assertIs(loaded[4]['successor'], bound['successor'])
            self.assertEqual(validate.call_args.kwargs, {'ancestry': ()})
            legacy.assert_not_called()
            mapping['baseline_generation.json'] = [{**item, 'raw_completion_token_ids': [3]} for item in records]
            with self.assertRaisesRegex(ValueError, 'parent generation'):
                verify.load_round(self.root, IDENTITY)

    def test_checkpoint_rejects_changed_optimizer_dynamics_despite_valid_moments(self):
        for key, value in [('betas', (0.5, 0.999)), ('weight_decay', 0.01), ('maximize', True),
                ('amsgrad', True), ('decoupled_weight_decay', True), ('fused', True),
                ('foreach', True), ('capturable', True), ('differentiable', True)]:
            result, checkpoint, reference, fixed = checkpoint_fixture()
            checkpoint['optimizer']['param_groups'][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Adam'):
                verify.validate_checkpoint(result, checkpoint, reference, fixed, IDENTITY)
        result, checkpoint, reference, fixed = checkpoint_fixture()
        del checkpoint['optimizer']['param_groups'][0]['betas']
        with self.assertRaisesRegex(ValueError, 'Adam'):
            verify.validate_checkpoint(result, checkpoint, reference, fixed, IDENTITY)

    def test_prompt_budget_and_gold_binding_preserve_template_and_roles(self):
        tokenizer = Tokenizer()
        ids = verify.prompt_tokens(tokenizer, row(), 8, CONTEXT)
        self.assertEqual(ids.tolist(), [[3, 7, 13, 19]])
        self.assertFalse(tokenizer.calls[-1][1]['enable_thinking'])
        self.assertEqual(tokenizer.calls[-1][0], [{'role': 'user', 'content': 'Authored context'}])
        with self.assertRaisesRegex(ValueError, '512'):
            verify.prompt_tokens(tokenizer, row(), 509, CONTEXT)
        cache = {'ids': torch.tensor([[3, 7, 13, 19, 23, 63]]),
                 'labels': torch.tensor([[-100, -100, -100, -100, 23, 63]])}
        actual, labels = verify.gold_sequence(tokenizer, row(), cache)
        self.assertIs(actual, cache['ids'])
        self.assertIs(labels, cache['labels'])
        cache['ids'][0, 0] = 9
        with self.assertRaisesRegex(ValueError, 'differ'):
            verify.gold_sequence(tokenizer, row(), cache)

    def test_probe_uses_observed_raw_tokens_including_internal_tokenizer_eos(self):
        context = patch.object(verify, 'generation_context', return_value=CONTEXT)
        context.start()
        self.addCleanup(context.stop)
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2]))
        observed = {'complete': True, 'max_new_tokens': 8, 'raw_completion_token_ids': [23, 63, 29, 2]}
        ids, labels = verify.probe_sequence(Tokenizer(), model, row(), observed)
        self.assertEqual(ids.tolist(), [[3, 7, 13, 19, 23, 63, 29, 2]])
        self.assertEqual(labels.tolist(), [[-100, -100, -100, -100, 23, 63, 29, 2]])
        for change in ({'complete': False}, {'max_new_tokens': 9}, {'raw_completion_token_ids': [23, 63]}):
            with self.assertRaises(ValueError):
                verify.probe_sequence(Tokenizer(), model, row(), {**observed, **change})

    def test_generation_context_is_not_the_512_token_training_gold_limit(self):
        tokenizer = Tokenizer()
        with patch.object(tokenizer, 'apply_chat_template', return_value=torch.ones((1, 530), dtype=torch.long)):
            ids = verify.prompt_tokens(tokenizer, row(), 64, {**CONTEXT, 'current_limit': 1024})
            self.assertEqual(ids.shape[1], 530)
            with self.assertRaisesRegex(ValueError, '512'):
                verify.prompt_tokens(tokenizer, row(), 64, CONTEXT)
            with self.assertRaises(ValueError):
                verify.prompt_tokens(tokenizer, row(), 64,
                    {**CONTEXT, 'current_limit': 1024, 'memory_budget_bytes': 100})

    def tiny(self):
        runtime, rows, tensors = fixture(self.folder.name, torch.bfloat16)
        # The real admission policy rounds automatic limits in 256-token blocks.
        runtime.model.config.text_config.max_position_embeddings = 512
        # Exercise an ordinary standalone official model, without Runtime routes.
        runtime.decoder.layers[-1].mlp = runtime.branch.base
        runtime.model.eval().requires_grad_(False)
        return runtime.model, rows, tensors

    def test_three_tiny_full_graph_audits_save_load_exact_and_restore_original(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model, rows, tensors = self.tiny()
            projection = model.model.language_model.layers[-1].mlp.down_proj
            before = projection.weight.clone()
            delta = torch.randn(before.shape) * .02
            with patch.object(verify, 'load_residual_core', wraps=verify.load_residual_core) as load:
                checks = verify.audit_carrier(model, delta, IDENTITY,
                    [(sample['id'], tensors[sample['id']][0]) for sample in rows], self.root / 'delta.safetensors')
            self.assertEqual(load.call_count, 1)
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(item['zero_exact'] and item['research_hook_logits_exact'] and item['serialized_delta_exact'] for item in checks))
        self.assertIsInstance(model.model.language_model.layers[-1].mlp.down_proj, verify.ResidualProjection)
        verify.uninstall_residual_core(model)
        self.assertIs(model.model.language_model.layers[-1].mlp.down_proj, projection)
        self.assertTrue(torch.equal(projection.weight, before))
        self.assertFalse(projection._forward_hooks)

    def test_successor_parent_plus_zero_and_full_graph_parent_losses(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model, rows, tensors = self.tiny()
            projection = model.model.language_model.layers[-1].mlp.down_proj
            before = projection.weight.clone()
            parent = torch.randn(before.shape) * .02
            parent_before = parent.clone()
            sequences = [(row['id'], tensors[row['id']][0]) for row in rows]
            protection = [(row['id'], *tensors[row['id']]) for row in rows]
            checks = verify.audit_parent_zero(model, parent, sequences, protection)
            self.assertEqual(len(checks['logits']), 3)
            self.assertTrue(all(row['parent_plus_zero_logits_exact'] for row in checks['logits']))
            self.assertEqual(len(checks['protection_losses']), 3)
            verify.install_residual_core(model, parent, base_identity=IDENTITY, base_core=None)
            try:
                expected = [verify.protection_loss(model, ids, labels) for _, ids, labels in protection]
            finally:
                verify.uninstall_residual_core(model)
            self.assertEqual(checks['protection_losses'], expected)
            self.assertTrue(torch.equal(parent, parent_before))
            self.assertTrue(torch.equal(projection.weight, before))
            self.assertFalse(projection._forward_hooks)
            with patch.object(verify, 'logits', side_effect=RuntimeError('parent audit failed')), \
                    self.assertRaisesRegex(RuntimeError, 'parent audit failed'):
                verify.audit_parent_zero(model, parent, sequences)
            self.assertFalse(projection._forward_hooks)

    def test_successor_gold_and_parent_raw_have_separate_tokenizer_bound_obligations(self):
        tokenizer, material = Tokenizer(), row()
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2]))
        gold = {'ids': torch.tensor([[3, 7, 13, 19, 23, 63]]),
                'labels': torch.tensor([[-100, -100, -100, -100, 23, 63]])}
        raw = {'complete': True, 'max_new_tokens': 8, 'raw_completion_token_ids': [29, 31, 2]}
        with patch.object(verify, 'generation_context', return_value=CONTEXT):
            ids, labels = verify.probe_sequence(tokenizer, model, material, raw)
            successor = {'protected_raw': {material['id']: raw}, 'protection_entries': [
                {'id': 'successor-gold:authored', 'row_id': material['id'], 'kind': 'mastered_gold', **gold},
                {'id': 'successor-raw:authored', 'row_id': material['id'], 'kind': 'mastered_raw',
                 'ids': ids, 'labels': labels}]}
            bound = verify.successor_sequences(tokenizer, model, [material], {material['id']: gold}, successor)
            self.assertEqual(bound[0][1][0, 4:].tolist(), [23, 63])
            self.assertEqual(bound[1][1][0, 4:].tolist(), [29, 31, 2])
            successor['protection_entries'][1]['ids'] = ids.clone()
            successor['protection_entries'][1]['ids'][0, 0] = 9
            with self.assertRaisesRegex(ValueError, 'tokenizer-bound'):
                verify.successor_sequences(tokenizer, model, [material], {material['id']: gold}, successor)
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                verify.successor_sequences(tokenizer, model, [{**material, 'purpose': 'validation'}],
                                           {material['id']: gold}, successor)

    def test_successor_rejects_losing_replay_that_only_the_parent_learned(self):
        successor = {'protected_raw': {}, 'mastered_train_ids': [],
            'original_correct_replay_ids': ['original-correct'],
            'parent_correct_replay_ids': ['original-correct', 'parent-newly-correct']}
        child = {'original-correct': {'passed': True}, 'parent-newly-correct': {'passed': False}}
        checks = verify.successor_retention(successor, child)
        self.assertTrue(checks['original_correct_replay_retained'])
        self.assertFalse(checks['parent_correct_replay_retained'])
        self.assertFalse(checks['successor_retention_passed'])
        child['parent-newly-correct']['passed'] = True
        self.assertTrue(verify.successor_retention(successor, child)['successor_retention_passed'])

    def test_hook_cleanup_when_full_forward_fails(self):
        model, rows, tensors = self.tiny()
        projection = model.model.language_model.layers[-1].mlp.down_proj
        calls = [torch.zeros(1), torch.zeros(1), RuntimeError('authored forward failure')]
        with patch.object(verify, 'logits', side_effect=calls), self.assertRaisesRegex(RuntimeError, 'authored'):
            verify.audit_carrier(model, torch.zeros_like(projection.weight, dtype=torch.float32), IDENTITY,
                [(rows[0]['id'], tensors[rows[0]['id']][0])], self.root / 'failure.safetensors')
        self.assertIs(model.model.language_model.layers[-1].mlp.down_proj, projection)
        self.assertFalse(projection._forward_hooks)

    def test_protection_loss_detects_changed_distribution_despite_same_argmax(self):
        model = torch.nn.Module()
        model.register_parameter('weight', torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)))
        ids, labels = torch.tensor([[1, 0]]), torch.tensor([[-100, 0]])
        before, after = torch.tensor([[[3., 0.], [0., 0.]]]), torch.tensor([[[2., 0.], [0., 0.]]])
        model.forward = lambda *args, **kwargs: SimpleNamespace(logits=before)
        baseline = verify.protection_loss(model, ids, labels)
        model.forward = lambda *args, **kwargs: SimpleNamespace(logits=after)
        self.assertTrue(torch.equal(before.argmax(-1), after.argmax(-1)))
        self.assertGreater(verify.protection_loss(model, ids, labels), baseline)

    def test_direct_tiny_generation_compares_raw_tokens_and_same_validator(self):
        model, _, _ = self.tiny()
        tokenizer = Tokenizer()
        material = row()
        expected = {'max_new_tokens': 8, 'raw_completion_token_ids': []}
        first = verify.generate(model, tokenizer, material, expected)
        expected.update(first)
        repeated = verify.generate(model, tokenizer, material, expected)
        self.assertTrue(repeated['tokens_match_round'])
        self.assertTrue(repeated['validator_matches_round'])

    def test_actual_kv_capture_aligns_prefill_decode_and_raw_tokens(self):
        model, _, _ = self.tiny()
        model.generation_config.eos_token_id = 2
        model.generation_config.forced_eos_token_id = 2
        model.generation_config.min_new_tokens = 3
        tokenizer, material = Tokenizer(), row()
        expected = {'max_new_tokens': 8, 'raw_completion_token_ids': []}
        expected.update(verify.generate(model, tokenizer, material, expected))
        projection = model.model.language_model.layers[-1].mlp.down_proj
        before = projection.weight.clone()
        observed = []
        handle = projection.register_forward_hook(lambda module, args, output: observed.append(output[0, -1].clone()))
        try:
            captured = verify.capture_greedy_trajectory(model, tokenizer, material, expected)
        finally:
            handle.remove()
        count = len(expected['raw_completion_token_ids'])
        self.assertGreater(count, 3)
        self.assertEqual(captured['forward_lengths'], [4] + [1] * (count - 1))
        self.assertEqual(captured['raw_completion_token_ids'], expected['raw_completion_token_ids'])
        self.assertEqual(tuple(captured['inputs'].shape), (count, projection.in_features))
        self.assertEqual(captured['inputs'].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(captured['outputs'], torch.stack(observed)))
        self.assertTrue(torch.equal(before, projection.weight))
        self.assertFalse(projection._forward_hooks)

    def test_kv_capture_rejects_dev_incomplete_mismatch_and_cleans_up_failure(self):
        model, _, _ = self.tiny()
        tokenizer, material = Tokenizer(), row()
        projection = model.model.language_model.layers[-1].mlp.down_proj
        with patch.object(verify, 'generate') as generate, self.assertRaisesRegex(ValueError, 'TRAIN'):
            verify.capture_greedy_trajectory(model, tokenizer, {**material, 'purpose': 'validation'}, {})
        generate.assert_not_called()
        def failing(*args):
            model(torch.tensor([[3, 7, 13, 19]]), use_cache=False)
            raise RuntimeError('injected generation failure')
        with patch.object(verify, 'generate', side_effect=failing), self.assertRaisesRegex(RuntimeError, 'injected'):
            verify.capture_greedy_trajectory(model, tokenizer, material, {})
        self.assertFalse(projection._forward_hooks)
        def invalid(**changes):
            model(torch.tensor([[3, 7, 13, 19]]), use_cache=False)
            return {'tokens_match_round': True, 'complete': True, 'validator_matches_round': True,
                    'raw_completion_token_ids': [2], **changes}
        for change in ({'tokens_match_round': False}, {'complete': False},
                       {'validator_matches_round': False}, {'raw_completion_token_ids': [3, 2]}):
            with self.subTest(change=change), patch.object(verify, 'generate', side_effect=lambda *args: invalid(**change)), \
                    self.assertRaisesRegex(ValueError, 'trajectory'):
                verify.capture_greedy_trajectory(model, tokenizer, material, {})
            self.assertFalse(projection._forward_hooks)

    def test_kv_capture_supports_inactive_dynamic_branch_and_rejects_active_expert(self):
        runtime, _, _ = fixture(self.folder.name, torch.bfloat16)
        model = runtime.model.eval().requires_grad_(False)
        model.config.text_config.max_position_embeddings = 512
        model.generation_config.eos_token_id = 2
        model.generation_config.forced_eos_token_id = 2
        model.generation_config.min_new_tokens = 3
        runtime.branch.expert = None
        tokenizer, material = Tokenizer(), row()
        expected = {'max_new_tokens': 8, 'raw_completion_token_ids': []}
        expected.update(verify.generate(model, tokenizer, material, expected))
        captured = verify.capture_greedy_trajectory(model, tokenizer, material, expected)
        self.assertEqual(len(captured['inputs']), len(expected['raw_completion_token_ids']))
        runtime.branch.expert = torch.nn.Identity()
        with patch.object(verify, 'generate') as generate, self.assertRaisesRegex(ValueError, 'active expert'):
            verify.capture_greedy_trajectory(model, tokenizer, material, expected)
        generate.assert_not_called()
        self.assertFalse(runtime.branch.base.down_proj._forward_hooks)

    def test_failed_model_lock_refuses_load_and_writes_failure(self):
        output = self.root / 'report'
        output.mkdir()
        groups = {'new_training': [], 'new_validation': [], 'replay_training': []}
        bundle = (groups, [], {'candidate_qualified': False, 'protocol': 'preservation-first-residual-fixed450-v1'}, {}, {}, {'ids': []}, [], [])
        with patch.object(verify, 'new_output', return_value=output), patch.object(verify, 'model_identity', return_value=IDENTITY), \
                patch.object(verify, 'load_round', return_value=bundle), \
                patch.object(verify, 'ModelProcessLock', side_effect=RuntimeError('model already resident')), \
                patch.object(verify, 'load_backbone') as load, patch.object(verify.torch.backends.mps, 'is_available', return_value=False), \
                self.assertRaisesRegex(RuntimeError, 'already resident'):
            verify.run(self.root, self.root / 'model')
        load.assert_not_called()
        report = verify.read(output / 'result.json')
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['model_lock_acquired'])
        self.assertFalse(report['production_adoption'])

    def test_acquired_lock_released_when_mps_unavailable_without_loading(self):
        output = self.root / 'report'
        output.mkdir()
        lock = SimpleNamespace(fd=123)
        lock.close = lambda: setattr(lock, 'fd', None)
        groups = {'new_training': [], 'new_validation': [], 'replay_training': []}
        bundle = (groups, [], {'candidate_qualified': False, 'protocol': 'preservation-first-residual-fixed450-v1'}, {}, {}, {'ids': []}, [], [])
        with patch.object(verify, 'new_output', return_value=output), patch.object(verify, 'model_identity', return_value=IDENTITY), \
                patch.object(verify, 'load_round', return_value=bundle), patch.object(verify, 'ModelProcessLock', return_value=lock), \
                patch.object(verify, 'load_backbone') as load, patch.object(verify.torch.backends.mps, 'is_available', return_value=False), \
                self.assertRaisesRegex(RuntimeError, 'requires MPS'):
            verify.run(self.root, self.root / 'model')
        load.assert_not_called()
        self.assertTrue(verify.read(output / 'result.json')['model_lock_released'])


if __name__ == '__main__':
    unittest.main()
