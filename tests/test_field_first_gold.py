"""Random CPU graphs and journal fixtures; no real model or held-out tokens."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import luban.evolution as evolution
from prepare_identity_first_gold import weighted_plan
from run_identity_coverage import FIELD_ARGUMENTS, FIELD_WEIGHTED_ARGUMENTS, validate_protocol_policy
from verify_training_operators import audit_training_operators, FIELD_OPERATOR_AUDIT_PROTOCOL, OPERATOR_AUDIT_PROTOCOL, TOLERANCES
import test_evolution
from test_field_training_diagnostic import tiny_runtime
from test_identity_fields import field_fixture, field_material
import test_core_publication


def plan_fixture():
    old, _, _, _ = field_fixture()
    old['seed'] = 42
    result = {'passed': False, 'training': {'promoted': False, 'completed_steps': 832,
        'training_selection': {'selected_step': None}}, 'engineering': [],
        'retirement': {'version': None, 'previous_version': None, 'restored': True,
                       'tensor_sha256': old['original_tensor_sha256']}}
    diagnostic = {'passed': True, 'preserved': True, 'optimizer_steps': 0, 'generation_calls': 0,
        'diagnostic': {'restoration': {'passed': True, 'actual_original_sha256': old['original_tensor_sha256']},
            'states': [{'state': state, 'rows': [{'id': value} for value in evolution.FIELD_FIRST_TOKEN_IDS]}
                       for state in ('originalNone', 'rejected832')]}}
    return weighted_plan(old, result, diagnostic, directory=Path('/unused'),
                         session_id='012345abcdef', prepared=old['prepared'] + 1), old, result, diagnostic


class FieldFirstGoldTests(unittest.TestCase):
    def fixture_runtime(self):
        fixture = test_evolution.EvolutionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture.runtime

    def test_literal_causal_gradient_weights_and_eos_fp32_bf16_no_rng(self):
        for dtype in (torch.float32, torch.bfloat16):
            for n in (4, 9):
                with self.subTest(dtype=dtype, targets=n):
                    logits = torch.randn(1, n + 4, 17).to(dtype).requires_grad_()
                    labels = torch.tensor([[-100] * 4 + list(range(n - 1)) + [16]])
                    rng = torch.get_rng_state().clone()
                    loss = evolution._field_weighted_loss(logits, labels)
                    gradient, = torch.autograd.grad(loss, logits)
                    selected = logits.detach().float()[0, 3:-1]
                    probabilities = selected.softmax(-1)
                    expected = probabilities - F.one_hot(labels[0, 4:], 17)
                    scales = torch.ones(n) / (n + 3)
                    scales[0] *= 4
                    expected *= scales[:, None]
                    reference = -selected.log_softmax(-1).gather(1, labels[0, 4:, None]).flatten()
                    self.assertAlmostEqual(float(loss.detach()), float((reference.sum() + 3 * reference[0]) / (n + 3)), places=6)
                    self.assertTrue(torch.allclose(gradient[0, 3:-1].float(), expected.to(dtype).float(), atol=1e-7, rtol=1e-6))
                    self.assertFalse(gradient[0, :3].any())
                    self.assertFalse(gradient[0, -1].any())
                    self.assertTrue(gradient[0, -2].any(), 'EOS predictor must stay supervised')
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    native = F.cross_entropy(logits[:, :-1].float().reshape(-1, 17), labels[:, 1:].reshape(-1))
                    self.assertNotEqual(float(loss.detach()), float(native.detach()))

    def test_noncontiguous_missing_eos_or_wrong_causal_prefix_reject(self):
        logits = torch.zeros(1, 6, 7)
        for labels in ([[0, 1, 2, 3, 4, 5]], [[-100, -100, 1, -100, 2, 3]],
                       [[-100, -100, 1, 2, 3, -100]], [[-100] * 5 + [1]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evolution._field_weighted_loss(logits, torch.tensor([labels]))

    def test_native_and_independent_weighted_tail_audit_both_dtypes_restore_state(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as folder:
                runtime, sample, _, labels = tiny_runtime(folder, dtype)
                runtime.lock, runtime.job, runtime.identity = threading.RLock(), None, 'CPU-random-graph'
                runtime.model.train()
                runtime.decoder.layers[0].eval()
                runtime.branch.expert = torch.nn.Linear(32, 32, bias=False).to(dtype)
                expert = runtime.branch.expert
                parameter = runtime.branch.base.down_proj.weight
                weight = parameter.detach().clone()
                parameter.grad = torch.ones_like(parameter)
                grad = parameter.grad
                flags = [value.requires_grad for value in runtime.model.parameters()]
                modes = [module.training for module in runtime.model.modules()]
                rng = torch.get_rng_state().clone()
                legacy = audit_training_operators(runtime, sample)
                weighted = audit_training_operators(runtime, sample, field_first_token_weight=4)
                self.assertEqual(legacy['protocol_version'], OPERATOR_AUDIT_PROTOCOL)
                self.assertNotIn('field_loss_policy', legacy)
                self.assertEqual(weighted['protocol_version'], FIELD_OPERATOR_AUDIT_PROTOCOL)
                self.assertTrue(weighted['passed'] and weighted['restored'])
                self.assertEqual(weighted['tolerances'], TOLERANCES[str(dtype)])
                self.assertEqual(weighted['target_tokens'], 4)
                self.assertEqual(weighted['cross_entropy']['abs_diff'], 0)
                self.assertTrue(weighted['logits']['exact'] and weighted['gradient']['exact'])
                self.assertTrue(torch.equal(weight, parameter))
                self.assertIs(parameter.grad, grad)
                self.assertIs(runtime.branch.expert, expert)
                self.assertEqual(flags, [value.requires_grad for value in runtime.model.parameters()])
                self.assertEqual(modes, [module.training for module in runtime.model.modules()])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                labels[0, -1] = 1
                with self.assertRaisesRegex(ValueError, r'gold\+EOS'):
                    audit_training_operators(runtime, sample, field_first_token_weight=4)
                self.assertTrue(torch.equal(weight, parameter))
                self.assertIs(runtime.branch.expert, expert)

    def test_same57_71_records_native_quality_and_legacy_arguments_unchanged(self):
        plan, old, result, diagnostic = plan_fixture()
        self.assertEqual(validate_protocol_policy(plan), FIELD_WEIGHTED_ARGUMENTS)
        self.assertEqual(FIELD_ARGUMENTS.get('field_first_token_weight', 1), 1)
        for key in ('groups_before_original_baseline', 'source_rows', 'source_groups', 'retention_regressions',
                    'engineering_cases', 'old_known_development_regressions', 'training_checkpoints', 'operator_audit'):
            self.assertEqual(plan[key], old[key])
        self.assertEqual(len(plan['field_loss_policy']['training_ids']), 9)
        self.assertEqual(len(plan['groups_before_original_baseline']['new_training']), 57)
        self.assertEqual(plan['protection_count'], 71)
        for mutation in ('weight', 'policy', 'ids', 'audit', 'audit_rows', 'budget', 'quality_gold'):
            value = copy.deepcopy(plan)
            if mutation == 'weight': value['arguments']['field_first_token_weight'] = 1
            elif mutation == 'policy': value['field_loss_policy']['normalization'] = 'target_count'
            elif mutation == 'ids': value['field_loss_policy']['training_ids'].pop()
            elif mutation == 'audit': value['field_loss_operator_audit']['tolerances']['torch.bfloat16']['loss_atol'] = 1
            elif mutation == 'audit_rows': value['field_loss_operator_audit']['samples'].reverse()
            elif mutation == 'budget': value['arguments']['steps'] = 640
            else: value['groups_before_original_baseline']['replay_validation'][0]['answer'] += 'wrong'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_protocol_policy(value)
        diagnostic['optimizer_steps'] = 1
        with self.assertRaises(ValueError):
            weighted_plan(old, result, diagnostic, directory=Path('/unused'), session_id='new', prepared=1)
        runtime = self.fixture_runtime()
        samples = evolution.consolidation_samples(runtime)['new_training']
        native_loss = evolution._loss(runtime, samples)
        with patch.object(evolution, '_field_weighted_loss', side_effect=AssertionError('quality must be native')):
            self.assertEqual(evolution._loss(runtime, samples), native_loss)

    def test_bad_weight_or_session_call_mismatch_before_material_or_candidate(self):
        runtime = self.fixture_runtime()
        for weight in (True, 0, 2, 4.1, float('nan'), float('inf'), '4'):
            with self.subTest(weight=weight), patch.object(evolution, '_target') as target, self.assertRaises(ValueError):
                evolution._consolidate_loaded(runtime, field_first_token_weight=weight)
            target.assert_not_called()
        for call, body in ((1, 4), (4, 1), (4, True), (1, 2)):
            runtime.training_session = SimpleNamespace(body={'arguments': {'field_first_token_weight': body}})
            with self.subTest(call=call, body=body), patch.object(evolution, '_session_material') as material, self.assertRaises(ValueError):
                evolution._consolidate_loaded(runtime, **FIELD_ARGUMENTS, field_first_token_weight=call)
            material.assert_not_called()

    def test_weight4_requires_research_original_fixed832_material_before_target(self):
        runtime = self.fixture_runtime()
        material = field_material()
        arguments = FIELD_WEIGHTED_ARGUMENTS.copy()
        body = {'scope': 'research', 'arguments': arguments}
        runtime.training_session = SimpleNamespace(body=body)
        runtime.identity = material['identity']
        runtime.research = True
        groups = material['groups']
        for mutation in ('scope', 'runtime_scope', 'core', 'steps', 'learning_rate', 'distillation_weight', 'preserve', 'arbitrary_gold'):
            changed = arguments.copy()
            body['scope'] = 'product' if mutation == 'scope' else 'research'
            runtime.research = mutation != 'runtime_scope'
            runtime.active_core = 'core-012345abcdef' if mutation == 'core' else None
            if mutation == 'steps': changed['steps'] = 640
            elif mutation == 'learning_rate': changed['learning_rate'] = 2e-5
            elif mutation == 'distillation_weight': changed['distillation_weight'] = 4
            elif mutation == 'preserve': changed['preserve_replay_subspace'] = False
            rows = copy.deepcopy(groups)
            if mutation == 'arbitrary_gold': rows['new_training'][0]['answer'] += 'wrong'
            with self.subTest(mutation=mutation), patch.object(evolution, '_session_material', return_value=(rows, [], material['retention_regressions'])), \
                    patch.object(evolution, '_target') as target, self.assertRaises(ValueError):
                evolution._consolidate_loaded(runtime, **changed)
            target.assert_not_called()

    def test_resume_rejects_old_or_mutated_policy_before_loading_adam(self):
        runtime = self.fixture_runtime()
        parameter = runtime.branch.base.down_proj.weight
        previous = parameter.detach().clone()
        master = torch.nn.Parameter(previous.clone())
        optimizer = torch.optim.Adam([master])
        report = {'id': 'run', 'version': 'core-012345abcdef', 'dataset_sha256': 'fixed', 'parameter': 'weight',
                  'split_ids': {}, 'field_loss_policy': evolution.field_first_token_policy()}
        saved = {'format': 1, 'id': 'run', 'step': 0, 'dataset_sha256': 'fixed', 'parameter': 'weight',
            'original_sha256': hashlib.sha256(previous.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'master': master.detach().clone(), 'new_exposures': {'new': 0}, 'report': copy.deepcopy(report)}
        groups = {'new_training': [{'id': 'new'}]}
        for policy in (None, evolution.field_first_token_policy() | {'first_gold_weight': 1}):
            saved['report']['field_loss_policy'] = policy
            with self.subTest(policy=policy), patch.object(optimizer, 'load_state_dict') as load, self.assertRaises(ValueError):
                evolution._resume_core(SimpleNamespace(id='run'), saved, report, parameter, previous, groups, 832, master, optimizer)
            load.assert_not_called()
            self.assertTrue(torch.equal(parameter, previous))

    def test_checkpoint_old_objective_rejects_before_basis_or_master_allocation(self):
        from luban.training_checkpoint import TrainingSession
        runtime = self.fixture_runtime()
        material = field_material()
        runtime.identity, runtime.research = material['identity'], True
        runtime.core_graph_verification['identity'] = runtime.identity
        records = {'groups': material['groups'], 'protection': [], 'retention_regressions': material['retention_regressions']}
        session = TrainingSession(runtime, 'evolution', FIELD_WEIGHTED_ARGUMENTS, records, scope='research')
        runtime.training_session = session
        runtime._sample_tokens = lambda row: (torch.tensor([[0, 0, 0, 1]]), torch.tensor([[-100, -100, 0, 1]]))
        original = runtime.branch.base.down_proj.weight.detach().clone()
        with patch.object(session, 'load', return_value={'report': {}}), \
                patch.object(evolution, '_capture_replay_subspace') as capture, \
                patch.object(torch.optim, 'Adam') as adam, self.assertRaisesRegex(ValueError, '不能分配'):
            evolution.consolidate(runtime, **FIELD_WEIGHTED_ARGUMENTS)
        capture.assert_not_called()
        adam.assert_not_called()
        self.assertTrue(torch.equal(original, runtime.branch.base.down_proj.weight))

    def test_actual_sqlite_journal_publication_keeps_weighted_report_and_count(self):
        fixture = test_core_publication.CorePublicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.plan = field_material()
        fixture.plan['field_loss_policy'] = evolution.field_first_token_policy()
        fixture.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it()

    def test_actual_old_mean_candidate_cannot_publish_as_weight4_before_any_copy(self):
        import luban.core_publication as publication
        native = publication.publish_isolated_core
        for mutation in ('1_to4', '4_to1', 'missing_args', 'missing_report', 'report_policy', 'binding_args', 'records',
                         'artifact_policy', 'missing_artifact', 'artifact_sha', 'artifact_report_rehashed'):
            fixture = test_core_publication.CorePublicationTests()
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            fixture.plan = field_material()
            if mutation != '1_to4': fixture.plan['field_loss_policy'] = evolution.field_first_token_policy()
            def mismatched(runtime, memory, store, plan, *args):
                changed = copy.deepcopy(plan)
                if mutation == '1_to4':
                    changed['arguments']['field_first_token_weight'] = 4
                    changed['field_loss_policy'] = evolution.field_first_token_policy()
                elif mutation == '4_to1':
                    changed['arguments'].pop('field_first_token_weight')
                    changed.pop('field_loss_policy')
                elif mutation.startswith('artifact') or mutation == 'missing_artifact':
                    from luban.training_checkpoint import training_runs
                    source = next(row for row in training_runs(runtime.memory, runtime.identity) if row['id'] == plan['session_id'])
                    name = 'evolution-' + plan['session_id'] + '.json'
                    path = Path(runtime.store) / name
                    if mutation == 'missing_artifact': path.unlink()
                    elif mutation == 'artifact_sha':
                        source['commit']['artifacts'][name] = '0' * 64
                    else:
                        actual = json.loads(path.read_text())
                        actual.pop('field_loss_policy')
                        path.write_text(json.dumps(actual))
                        if mutation == 'artifact_report_rehashed':
                            source['commit']['artifacts'][name] = publication.file_sha(path)
                    if mutation in {'artifact_sha', 'artifact_report_rehashed'}:
                        with runtime.memory.connect() as db:
                            db.execute('UPDATE training_runs SET body=? WHERE id=?', (json.dumps(source), plan['session_id']))
                else:
                    # Real temporary source journal corruption, never production data.
                    with runtime.memory.connect() as db:
                        body = json.loads(db.execute('SELECT body FROM training_runs WHERE id=?', (plan['session_id'],)).fetchone()[0])
                        if mutation == 'missing_args': body.pop('arguments')
                        elif mutation == 'missing_report': body['commit'].pop('report')
                        elif mutation == 'report_policy': body['commit']['report']['field_loss_policy'] = None
                        elif mutation == 'binding_args': body['binding']['arguments'] = FIELD_ARGUMENTS.copy()
                        else: body['records']['retention_regressions'].pop()
                        db.execute('UPDATE training_runs SET body=? WHERE id=?', (json.dumps(body), plan['session_id']))
                with memory.connect() as db: before = tuple(db.iterdump())
                files = {str(path): path.read_bytes() for path in Path(store).rglob('*') if path.is_file()}
                with patch.object(publication, '_copy_immutable') as copied, self.assertRaises(ValueError):
                    native(runtime, memory, store, changed, *args)
                copied.assert_not_called()
                self.assertFalse((Path(store) / 'active-core.json').exists())
                with memory.connect() as db: self.assertEqual(tuple(db.iterdump()), before)
                self.assertEqual({str(path): path.read_bytes() for path in Path(store).rglob('*') if path.is_file()}, files)
                raise RuntimeError('CPU mismatch correctly rejected before production mutation')
            with self.subTest(mutation=mutation), patch.object(test_core_publication, 'publish_isolated_core', side_effect=mismatched), \
                    self.assertRaisesRegex(RuntimeError, 'correctly rejected'):
                fixture.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it()

    def test_source_commit_cannot_restore_or_publish_mutated_policy_arguments_or_records(self):
        from luban.core_publication import validate_source_core_commit, content_sha, file_sha
        material = field_material()
        records = {'groups': material['groups'], 'protection': [], 'retention_regressions': material['retention_regressions']}
        policy = evolution.field_first_token_policy()
        plan = {'session_id': material['run_id'], 'base_identity': material['identity'],
                'arguments': FIELD_WEIGHTED_ARGUMENTS, 'field_loss_policy': policy}
        report = {'id': material['run_id'], 'identity': material['identity'], 'version': material['version'],
                  'promoted': True, 'publication_state': 'published', 'field_loss_policy': policy,
                  'steps': 832, 'training_selection': {'checkpoints': list(evolution.FIELD_TRAIN_CHECKPOINTS)}}
        source = {'id': material['run_id'], 'kind': 'evolution', 'state': 'committed', 'scope': 'research',
            'arguments': FIELD_WEIGHTED_ARGUMENTS, 'records': records, 'binding': {'arguments': FIELD_WEIGHTED_ARGUMENTS,
                'identity': material['identity'], 'core_version': None, 'records_sha256': content_sha(records)},
            'commit': {'report': report}}
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        store = Path(folder.name)
        names = [material['version'] + suffix for suffix in ('.safetensors', '.retention.json', '.json')]
        report_name = 'evolution-' + material['run_id'] + '.json'
        for name in names: (store / name).write_bytes(b'CPU integrity fixture, not model weights')
        (store / report_name).write_text(json.dumps(report))
        source['commit']['artifacts'] = {name: file_sha(store / name) for name in names + [report_name]}
        self.assertEqual(validate_source_core_commit(source, plan, material['groups'], material['retention_regressions'], store=store), report)
        for mutation in ('arguments', 'binding_args', 'records', 'records_sha', 'policy', 'stage', 'report_version', 'core'):
            changed = copy.deepcopy(source)
            if mutation == 'arguments': changed['arguments'] = FIELD_ARGUMENTS.copy()
            elif mutation == 'binding_args': changed['binding']['arguments'] = FIELD_ARGUMENTS.copy()
            elif mutation == 'records': changed['records']['retention_regressions'].pop()
            elif mutation == 'records_sha': changed['binding']['records_sha256'] = '0' * 64
            elif mutation == 'policy': changed['commit']['report']['field_loss_policy'] = None
            elif mutation == 'stage': changed['state'] = 'running'
            elif mutation == 'report_version': changed['commit']['report']['version'] = 'core-000000000000'
            else: changed['binding']['core_version'] = 'core-000000000000'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_source_core_commit(changed, plan, material['groups'], material['retention_regressions'], store=store)

    def test_weighted_832_pause_resume_master_adam_rng_exposures_and_all25_checks(self):
        """Real two-logit optimizer; generation is an explicit always-reject CPU fixture."""
        from luban.training_checkpoint import TrainingSession, TrainingPaused
        runtime = self.fixture_runtime()
        material = field_material()
        runtime.identity = material['identity']
        runtime.research = True
        runtime.core_graph_verification['identity'] = runtime.identity
        groups, regressions = material['groups'], material['retention_regressions']
        original = runtime.branch.base.down_proj.weight.detach().clone()
        def tokens(row):
            if row['purpose'] != 'training':
                raise AssertionError('Rejected CPU run must not tensorize DEV/regressions')
            target = 0 if json.loads(row['provenance'])['evolution_role'] == 'new' else 1
            return torch.tensor([[target, target, target, 1]]), torch.tensor([[-100, -100, target, 1]])
        runtime._sample_tokens = tokens
        native = runtime.model.forward
        def stochastic(ids, labels=None, **kwargs):
            result = native(ids, labels=None, **kwargs)
            if torch.is_grad_enabled():
                result.logits = result.logits * (.75 + .5 * torch.rand(()))
            if labels is not None:
                result.loss = F.cross_entropy(result.logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1))
            return result
        runtime.model.forward = stochastic
        records = {'groups': groups, 'protection': [], 'retention_regressions': regressions}
        arguments = FIELD_WEIGHTED_ARGUMENTS.copy()
        def generation(_, rows):
            return [{'id': row['id'], 'passed': json.loads(row['provenance'])['evolution_role'] == 'replay',
                     'complete': True, 'branch_enabled': False} for row in rows]
        control = TrainingSession(runtime, 'evolution', arguments, records, scope='research')
        runtime.training_session = control
        with patch.object(evolution, '_generation', side_effect=generation), patch('sys.stdout', io.StringIO()):
            report = evolution.consolidate(runtime, **arguments)
        self.assertFalse(report['promoted'])
        expected = control.load()
        self.assertTrue(torch.equal(original, runtime.branch.base.down_proj.weight))
        session = TrainingSession(runtime, 'evolution', arguments, records, scope='research')
        runtime.training_session = session
        save = session.save
        def pause(payload, progress):
            save(payload, progress)
            if progress['step'] == 816:
                session.pause_requested = True
        with patch.object(evolution, '_generation', side_effect=generation), patch('sys.stdout', io.StringIO()), \
                patch.object(session, 'save', side_effect=pause), self.assertRaises(TrainingPaused):
            evolution.consolidate(runtime, **arguments)
        durable = session.load()
        self.assertEqual(durable['step'], 816)
        self.assertEqual(durable['report']['field_loss_policy'], evolution.field_first_token_policy())
        self.assertEqual([row['step'] for row in durable['report']['training_selection']['results']],
                         list(evolution.FIELD_TRAIN_CHECKPOINTS[:-1]))
        self.assertEqual(min(durable['new_exposures'].values()), 7)
        self.assertTrue(torch.equal(original, runtime.branch.base.down_proj.weight))
        session.pause()
        resumed = TrainingSession(runtime, 'evolution', arguments, records, run_id=session.id, scope='research')
        runtime.training_session = resumed
        with patch.object(evolution, '_generation', side_effect=generation), patch('sys.stdout', io.StringIO()):
            report = evolution.consolidate(runtime, **arguments)
        actual = resumed.load()
        self.assertEqual(report['resumed_from_steps'], [816])
        self.assertEqual([row['step'] for row in report['training_selection']['results']], list(evolution.FIELD_TRAIN_CHECKPOINTS))
        self.assertEqual(report['training_selection']['selected_step'], None)
        self.assertEqual(expected['new_exposures'], actual['new_exposures'])
        self.assertEqual(expected['kl_total'], actual['kl_total'])
        self.assertTrue(torch.equal(expected['master'], actual['master']))
        for key, value in expected['optimizer']['state'][0].items():
            self.assertTrue(torch.equal(value, actual['optimizer']['state'][0][key]), key)
        self.assertTrue(torch.equal(original, runtime.branch.base.down_proj.weight))


if __name__ == '__main__':
    unittest.main()
