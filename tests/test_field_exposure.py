"""Finite1536 guard/real tiny Adam/SQLite evidence; never a2B ability claim."""
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import luban.evolution as evolution
import luban.core_publication as publication
from luban.training_checkpoint import TrainingSession, TrainingPaused, training_runs
from prepare_identity_field_exposure import exposure_plan
from run_identity_coverage import (ARGUMENTS, RELATION_ARGUMENTS, FIELD_ARGUMENTS, FIELD_WEIGHTED_ARGUMENTS,
                                  FIELD_EXPOSURE_ARGUMENTS, validate_protocol_policy)
from test_identity_fields import field_fixture, field_material
import test_core_publication
import test_evolution


def plan_fixture():
    old, _, _, _ = field_fixture()
    old['seed'], old['optimizer_batch_size'] = 42, 1
    result = {'passed': False, 'engineering': [], 'training': {'promoted': False, 'steps': 832,
        'completed_steps': 832, 'previous_version': None, 'training_selection': {'selected_step': None}},
        'retirement': {'version': None, 'previous_version': None, 'restored': True,
                       'tensor_sha256': old['original_tensor_sha256']}}
    weighted = copy.deepcopy(result)
    weighted['training']['field_loss_policy'] = evolution.field_first_token_policy()
    plan = exposure_plan(old, result, weighted, directory=Path('/unused'),
                         session_id='012345abcdef', prepared=old['prepared'] + 1)
    return plan, old, result, weighted


class FieldExposureTests(unittest.TestCase):
    def runtime(self):
        fixture = test_evolution.EvolutionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        runtime = fixture.runtime
        material = field_material()
        runtime.identity, runtime.research = material['identity'], True
        runtime.core_graph_verification['identity'] = runtime.identity
        runtime._sample_tokens = lambda row: (
            torch.tensor([[0 if json.loads(row['provenance'])['evolution_role'] == 'new' else 1, 1]]),
            torch.tensor([[-100, 0 if json.loads(row['provenance'])['evolution_role'] == 'new' else 1]]))
        records = {'groups': material['groups'], 'protection': [], 'retention_regressions': material['retention_regressions']}
        return runtime, records

    def test_preparation_only_changes_native_control_ceiling_and_keeps_all57_71_115_inputs(self):
        plan, old, result, weighted = plan_fixture()
        self.assertEqual(validate_protocol_policy(plan), FIELD_EXPOSURE_ARGUMENTS)
        self.assertEqual([ARGUMENTS['steps'], RELATION_ARGUMENTS['steps'], FIELD_ARGUMENTS['steps'], FIELD_WEIGHTED_ARGUMENTS['steps']],
                         [512, 640, 832, 832])
        self.assertNotIn('field_loss_policy', plan)
        self.assertNotIn('field_loss_operator_audit', plan)
        self.assertFalse(plan['sealed'])
        self.assertIsNone(plan['script_sha256'])
        self.assertIsNone(plan['execution_manifest'])
        self.assertEqual(len(plan['training_checkpoints']), 47)
        self.assertEqual(plan['training_checkpoints'], list(range(64, 1537, 32)))
        for key in ('groups_before_original_baseline', 'source_groups', 'source_rows', 'retention_regressions',
                    'engineering_cases', 'old_known_development_regressions', 'replay_training_candidates',
                    'superseded_source_ids', 'operator_audit', 'seed', 'optimizer_batch_size'):
            self.assertEqual(plan[key], old[key], key)
        rows = plan['groups_before_original_baseline']
        self.assertEqual(len(rows['new_training']) + len(rows['replay_training']) + len(plan['replay_training_candidates']), 115)
        self.assertEqual(len(rows['new_validation']) + len(rows['replay_validation']) + len(plan['retention_regressions']), 71)
        for mutation in ('weight', 'checkpoints', 'budget', 'gold', 'rights', 'warmstart', 'retry', 'count'):
            changed = copy.deepcopy(plan)
            if mutation == 'weight': changed['arguments']['field_first_token_weight'] = 4
            elif mutation == 'checkpoints': changed['training_checkpoints'].pop()
            elif mutation == 'budget': changed['arguments']['steps'] = 1537
            elif mutation in {'gold', 'rights'}:
                row = changed['groups_before_original_baseline']['new_training'][-1]
                row['answer' if mutation == 'gold' else 'training_allowed'] = 'wrong' if mutation == 'gold' else False
            elif mutation == 'warmstart': changed['warm_start_candidate'] = True
            elif mutation == 'retry': changed['maximum_attempts'] = 2
            else: changed['protection_count'] = 70
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_protocol_policy(changed)
        weighted['training']['field_loss_policy'] = None
        with self.assertRaises(ValueError):
            exposure_plan(old, result, weighted, directory=Path('/unused'), session_id='new', prepared=1)

    def test_wrong_scope_body_weight_material_budget_or_original_refuses_before_candidate(self):
        for mutation in ('direct', 'product', 'runtime', 'version', 'weight', 'body_weight', 'body832', 'body_extra',
                         'binding832', 'binding_scope', 'lr', 'kl', 'projection', 'gold'):
            runtime, records = self.runtime()
            arguments = copy.deepcopy(FIELD_EXPOSURE_ARGUMENTS)
            changed = copy.deepcopy(records)
            runtime.research, runtime.active_core = True, None
            if mutation == 'gold': changed['groups']['replay_training'][0]['answer'] = 'wrong'
            session = TrainingSession(runtime, 'evolution', copy.deepcopy(FIELD_ARGUMENTS if mutation == 'body832' else arguments),
                changed, scope='product' if mutation == 'product' else 'research')
            runtime.training_session = None if mutation == 'direct' else session
            if mutation == 'runtime': runtime.research = False
            elif mutation == 'version': runtime.active_core = 'core-111111111111'
            elif mutation == 'weight': arguments['field_first_token_weight'] = 4
            elif mutation == 'body_weight': session.body['arguments']['field_first_token_weight'] = 4
            elif mutation == 'body_extra': session.body['arguments']['unused_objective'] = True
            elif mutation == 'binding832': session.binding['arguments'] = FIELD_ARGUMENTS.copy()
            elif mutation == 'binding_scope': session.binding['scope'] = 'product'
            elif mutation == 'lr': arguments['learning_rate'] = 2e-5
            elif mutation == 'kl': arguments['distillation_weight'] = 4
            elif mutation == 'projection': arguments['preserve_replay_subspace'] = False
            with self.subTest(mutation=mutation), patch.object(evolution, '_target') as target, self.assertRaises(ValueError):
                evolution.consolidate(runtime, **arguments)
            target.assert_not_called()

    def test_832_1536_payload_and_history_mismatch_refuses_before_master_basis_and_adam(self):
        for arguments, wrong_steps, wrong_checkpoints in (
            (FIELD_EXPOSURE_ARGUMENTS, 832, evolution.FIELD_TRAIN_CHECKPOINTS),
            (FIELD_ARGUMENTS, 1536, evolution.FIELD_EXPOSURE_CHECKPOINTS),
            (FIELD_EXPOSURE_ARGUMENTS, 1536, evolution.FIELD_TRAIN_CHECKPOINTS)):
            runtime, records = self.runtime()
            original = runtime.branch.base.down_proj.weight.detach().clone()
            session = TrainingSession(runtime, 'evolution', arguments, records, scope='research')
            runtime.training_session = session
            session.save({'report': {'steps': wrong_steps, 'training_selection': {'checkpoints': list(wrong_checkpoints)}}},
                         {'step': 0, 'phase': 'training'})
            with self.subTest(steps=arguments['steps'], wrong=wrong_steps), \
                    patch.object(evolution, '_capture_replay_subspace') as basis, patch.object(torch.optim, 'Adam') as adam, \
                    self.assertRaisesRegex(ValueError, '不能分配'):
                evolution.consolidate(runtime, **arguments)
            basis.assert_not_called()
            adam.assert_not_called()
            self.assertTrue(torch.equal(original, runtime.branch.base.down_proj.weight))
            session.pause()
            other = FIELD_ARGUMENTS if arguments == FIELD_EXPOSURE_ARGUMENTS else FIELD_EXPOSURE_ARGUMENTS
            with self.assertRaises(ValueError):
                TrainingSession(runtime, 'evolution', other, records, run_id=session.id, scope='research')

    def test_real_1536_adam_pause_resume_exact_rng_exposure_history_and_native_quality(self):
        """Two-logit CPU optimizer; generation is deliberately always-reject, not ability evidence."""
        runtime, records = self.runtime()
        original = runtime.branch.base.down_proj.weight.detach().clone()
        native = runtime.model.forward
        def stochastic(ids, labels=None, **kwargs):
            out = native(ids, labels=None, **kwargs)
            if torch.is_grad_enabled(): out.logits = out.logits * (.75 + .5 * torch.rand(()))
            if labels is not None: out.loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1))
            return out
        runtime.model.forward = stochastic
        def generation(_, rows):
            return [{'id': row['id'], 'passed': json.loads(row['provenance'])['evolution_role'] == 'replay',
                     'complete': True, 'branch_enabled': False} for row in rows]
        def tokens(row):
            self.assertEqual(row['purpose'], 'training', 'Rejecting fixture must not peek DEV or regressions')
            target = 0 if json.loads(row['provenance'])['evolution_role'] == 'new' else 1
            return torch.tensor([[target, 1]]), torch.tensor([[-100, target]])
        runtime._sample_tokens = tokens
        arguments = FIELD_EXPOSURE_ARGUMENTS.copy()
        session = TrainingSession(runtime, 'evolution', arguments, records, scope='research')
        runtime.training_session = session
        with patch.object(evolution, '_generation', side_effect=generation), patch('sys.stdout', io.StringIO()), \
                patch.object(evolution, '_field_weighted_loss', side_effect=AssertionError('mean1 stays native')):
            report = evolution.consolidate(runtime, **arguments)
        expected = session.load()
        self.assertFalse(report['promoted'])
        self.assertNotIn('field_loss_policy', report)
        session = TrainingSession(runtime, 'evolution', arguments, records, scope='research')
        runtime.training_session = session
        save = session.save
        def pause(payload, progress):
            save(payload, progress)
            if progress['step'] == 816: session.pause_requested = True
        with patch.object(evolution, '_generation', side_effect=generation), patch('sys.stdout', io.StringIO()), \
                patch.object(session, 'save', side_effect=pause), self.assertRaises(TrainingPaused):
            evolution.consolidate(runtime, **arguments)
        durable = session.load()
        self.assertEqual(durable['step'], 816)
        self.assertEqual([row['step'] for row in durable['report']['training_selection']['results']],
                         [step for step in evolution.FIELD_EXPOSURE_CHECKPOINTS if step <= 816])
        self.assertTrue(torch.equal(original, runtime.branch.base.down_proj.weight))
        session.pause()
        resumed = TrainingSession(runtime, 'evolution', arguments, records, run_id=session.id, scope='research')
        runtime.training_session = resumed
        with patch.object(evolution, '_generation', side_effect=generation), patch('sys.stdout', io.StringIO()):
            report = evolution.consolidate(runtime, **arguments)
        actual = resumed.load()
        self.assertEqual(report['resumed_from_steps'], [816])
        self.assertEqual([row['step'] for row in report['training_selection']['results']], list(evolution.FIELD_EXPOSURE_CHECKPOINTS))
        self.assertEqual(report['completed_steps'], 1536)
        self.assertIsNone(report['training_selection']['selected_step'])
        self.assertEqual(min(actual['new_exposures'].values()), 13)
        self.assertEqual([actual['new_exposures'][identifier] for identifier in evolution.FIELD_FIRST_TOKEN_IDS], [13] * 9)
        self.assertEqual(expected['new_exposures'], actual['new_exposures'])
        self.assertEqual(expected['kl_total'], actual['kl_total'])
        self.assertTrue(torch.equal(expected['master'], actual['master']))
        for key, value in expected['optimizer']['state'][0].items():
            self.assertTrue(torch.equal(value, actual['optimizer']['state'][0][key]), key)
        self.assertTrue(torch.equal(original, runtime.branch.base.down_proj.weight))
        self.assertFalse((runtime.store / 'active-core.json').exists())

    def test_actual1536_sqlite_publication_and_upgrade_count(self):
        fixture = test_core_publication.CorePublicationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.plan = field_material()
        fixture.arguments = FIELD_EXPOSURE_ARGUMENTS
        fixture.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it()

    def test_actual_publication_wrong_budget_history_or_artifact_refuses_before_copy(self):
        native = publication.publish_isolated_core
        for mutation in ('report832', 'history25', 'args832', 'weighted', 'artifact'):
            fixture = test_core_publication.CorePublicationTests()
            fixture.setUp()
            self.addCleanup(fixture.doCleanups)
            fixture.plan, fixture.arguments = field_material(), FIELD_EXPOSURE_ARGUMENTS
            def mismatched(runtime, memory, store, plan, *args):
                source = next(row for row in training_runs(runtime.memory, runtime.identity) if row['id'] == plan['session_id'])
                if mutation == 'args832': plan = copy.deepcopy(plan) | {'arguments': FIELD_ARGUMENTS}
                else:
                    body = {key: value for key, value in source.items() if key != 'state'}
                    report = body['commit']['report']
                    if mutation == 'report832': report['steps'] = 832
                    elif mutation == 'history25': report['training_selection']['checkpoints'] = list(evolution.FIELD_TRAIN_CHECKPOINTS)
                    elif mutation == 'weighted': report['field_loss_policy'] = evolution.field_first_token_policy()
                    name = 'evolution-' + source['id'] + '.json'
                    path = Path(runtime.store) / name
                    path.write_text(json.dumps(report))
                    if mutation != 'artifact': body['commit']['artifacts'][name] = publication.file_sha(path)
                    else: path.write_text('corrupt')
                    with runtime.memory.connect() as db:
                        db.execute('UPDATE training_runs SET body=? WHERE id=?', (json.dumps(body), source['id']))
                with memory.connect() as db: before = tuple(db.iterdump())
                files = {str(path): path.read_bytes() for path in Path(store).rglob('*') if path.is_file()}
                with patch.object(publication, '_copy_immutable') as copied, self.assertRaises(ValueError):
                    native(runtime, memory, store, plan, *args)
                copied.assert_not_called()
                with memory.connect() as db: self.assertEqual(tuple(db.iterdump()), before)
                self.assertEqual({str(path): path.read_bytes() for path in Path(store).rglob('*') if path.is_file()}, files)
                self.assertFalse((Path(store) / 'active-core.json').exists())
                raise RuntimeError('CPU invalid budget correctly refused')
            with self.subTest(mutation=mutation), patch.object(test_core_publication, 'publish_isolated_core', side_effect=mismatched), \
                    self.assertRaisesRegex(RuntimeError, 'correctly refused'):
                fixture.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it()


if __name__ == '__main__':
    unittest.main()
