"""Actual tiny CPU core updates and finite receipts, never a 2B capability claim."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import torch

import luban.evolution as evolution
from luban.evolution_progress import (PARTIAL_POLICY, checkpoints, unique_records, correct_ids,
    progress, validate_partial_progress, PhaseTimer, content_sha)
from luban.retention import read_core_retention, protected_validation
from luban.training_checkpoint import TrainingSession, TrainingPaused, training_runs
from luban.automatic import receipt_keys, record_keys
import test_evolution

ARGS = {"steps": 352, "learning_rate": 1e-5, "distillation_weight": 5,
        "preserve_replay_subspace": True, "partial_policy": PARTIAL_POLICY}


class PartialEvolutionTests(unittest.TestCase):
    def product_fixture(self):
        case, rt, records = self.fixture()
        rt.training_session.rejected('research fixture replaced by isolated product fixture')
        rt.research = False
        pool = [copy.deepcopy(row) for key in ('new_training', 'new_validation') for row in records['groups'][key]]
        for index in range(350 - len(pool)):
            pool.append({'id': f'allowance-{index}', 'prompt': f'untrained allowance {index}', 'answer': 'verified',
                'source': 'human', 'purpose': 'training', 'approved': 1, 'training_allowed': 1,
                'provenance': json.dumps({'evolution_role': 'new'})})
        records['experience'] = pool
        args = {**ARGS, 'steps': 450}
        rt.training_session = TrainingSession(rt, 'evolution', args, records)
        return case, rt, records, args

    def test_product450_real_cpu_adam_partial_growth_commits_exact_budget_and_leaves_pending(self):
        _, rt, records, args = self.product_fixture()
        report = evolution.consolidate(rt, **args)
        self.assertTrue(report['promoted'], report)
        self.assertEqual(report['completed_steps'], 450)
        self.assertEqual(len(report['timing']['step_times']), 450)
        self.assertEqual(report['training_selection']['checkpoints'], list(range(50, 451, 50)))
        self.assertGreater(report['learning_progress']['newly_mastered'], 0)
        self.assertEqual(report['learning_progress']['pending'], 1)
        self.assertEqual(rt.training_session.body['state'], 'committed')
        self.assertEqual(rt.training_session.load()['optimizer']['state'][0]['step'].item(), 450)
        validate_partial_progress(report, records, rt.identity, report['version'], report['checkpoint']['sha256'])
        saved = rt.branch.base.down_proj.weight.detach().clone()
        with torch.no_grad():
            rt.branch.base.down_proj.weight.copy_(rt._base_core_weight)
        loaded = evolution.load_active_core(rt)
        self.assertEqual(loaded['version'], report['version'])
        self.assertTrue(torch.equal(saved, rt.branch.base.down_proj.weight))
        self.assertIsNone(rt.core_requires_review)

    def protected_fixture(self):
        case, rt, records, args = self.product_fixture()
        rt.training_session.rejected('replaced with explicit protected CPU fixture')
        records['protection'] = [{'id': 'protected-' + str(index), 'prompt': 'old protected ' + str(index),
            'answer': 'oldability', 'source': 'human', 'purpose': 'validation', 'approved': True,
            'training_allowed': False, 'provenance': json.dumps({'evolution_role': 'replay',
                'retention_generation_only': True})} for index in range(2)]
        protection = patch.object(evolution, 'protected_validation', return_value=records['protection'])
        protection.start()
        self.addCleanup(protection.stop)
        rt.training_session = TrainingSession(rt, 'evolution', args, records)
        return case, rt, records, args

    def test_partial_protection_reuses_two_generations_and_reloads_nonempty_evidence(self):
        _, rt, records, args = self.protected_fixture()
        generate = evolution._generation
        calls = []
        def observed(runtime, rows):
            if runtime.job['phase'] == 'validation':
                calls.append([r['id'] for r in rows])
            return generate(runtime, rows)
        with patch.object(evolution, '_generation', side_effect=observed):
            report = evolution.consolidate(rt, **args)
        self.assertTrue(report['promoted'], report)
        full = unique_records(records['groups'], records['protection'], [])
        self.assertEqual(calls[-2:], [[r['id'] for r in full]] * 2)
        self.assertEqual(len(calls), 3)  # Deferred original baseline, then exactly two candidate passes.
        self.assertEqual([r['id'] for r in report['core_retention_loss']['rows']],
                         [r['id'] for r in records['protection']])
        self.assertTrue(all(r['nonincreasing'] for r in report['core_retention_loss']['rows']))
        validate_partial_progress(report, records, rt.identity, report['version'], report['checkpoint']['sha256'])
        saved = rt.branch.base.down_proj.weight.detach().clone()
        with torch.no_grad():
            rt.branch.base.down_proj.weight.copy_(rt._base_core_weight)
        self.assertEqual(evolution.load_active_core(rt)['version'], report['version'])
        self.assertTrue(torch.equal(saved, rt.branch.base.down_proj.weight))
        for mutation in ('loss', 'loss_missing', 'core_missing', 'second_incomplete', 'gate', 'scope'):
            bad = copy.deepcopy(report)
            if mutation == 'loss': bad['candidate_core_retention_loss'][0]['loss'] += .01
            elif mutation == 'loss_missing':
                bad['candidate_core_retention_loss'].pop()
                bad['baseline_core_retention_loss'].pop()
            elif mutation == 'core_missing': bad['core_retention_generation'].pop()
            elif mutation == 'second_incomplete': bad['core_retention_generation_second'][0]['complete'] = False
            elif mutation == 'gate': bad['gates']['protected_core_retention_loss'] = False
            else: bad['protected_replay_ids'] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_partial_progress(bad, records, rt.identity, report['version'], report['checkpoint']['sha256'])

    def test_partial_single_retention_ce_increase_rejects_despite_generation_and_mean(self):
        _, rt, records, args = self.protected_fixture()
        original = rt.branch.base.down_proj.weight.detach().clone()
        measure = evolution._loss_by_sample
        calls = []
        def candidate_loss_fault(runtime, rows):
            values = measure(runtime, rows)
            calls.append(runtime.job['phase'])
            if runtime.job['phase'] == 'validation':
                # Fault injection after real tiny measurements: the mean is unchanged.
                values[0]['loss'] += .01
                values[1]['loss'] -= .01
            return values
        with patch.object(evolution, '_loss_by_sample', side_effect=candidate_loss_fault):
            report = evolution.consolidate(rt, **args)
        self.assertEqual(calls, ['preparing', 'validation'])
        self.assertEqual(report['completed_steps'], 450)
        self.assertFalse(report['promoted'])
        self.assertFalse(report['gates']['protected_core_retention_loss'])
        self.assertTrue(report['gates']['protected_core_retention_generation'])
        self.assertTrue(report['gates']['protected_replay_generation'])
        self.assertTrue(report['gates']['protected_replay_loss_retained'])
        self.assertAlmostEqual(sum(r['loss'] for r in report['baseline_core_retention_loss']),
                               sum(r['loss'] for r in report['candidate_core_retention_loss']))
        self.assertEqual(report['learning_progress']['newly_mastered'], 0)
        self.assertTrue(torch.equal(original, rt.branch.base.down_proj.weight))
        self.assertIsNone(rt.active_core)
        self.assertFalse((rt.store / 'active-core.json').exists())
        self.assertEqual(rt.training_session.body['state'], 'rejected')

    def test_partial_second_observation_missing_or_incomplete_cannot_commit(self):
        for fault in ('missing', 'incomplete', 'wrong'):
            with self.subTest(fault=fault):
                _, rt, records, args = self.protected_fixture()
                original = rt.branch.base.down_proj.weight.detach().clone()
                generate = evolution._generation
                observed = 0
                def damaged_second(runtime, rows):
                    nonlocal observed
                    result = generate(runtime, rows)
                    if runtime.job['phase'] == 'validation':
                        observed += 1
                        if observed == 3:
                            index = next(i for i, p in enumerate(result) if p['id'] == records['protection'][0]['id'])
                            if fault == 'missing': result.pop(index)
                            elif fault == 'incomplete': result[index].update(complete=False, passed=False)
                            else: result[index].update(answer='wrong', passed=False)
                    return result
                with patch.object(evolution, '_generation', side_effect=damaged_second):
                    if fault == 'missing':
                        with self.assertRaises(ValueError): evolution.consolidate(rt, **args)
                    else:
                        report = evolution.consolidate(rt, **args)
                        self.assertFalse(report['promoted'])
                        self.assertFalse(report['gates']['protected_core_retention_generation'])
                        self.assertEqual(report['learning_progress']['newly_mastered'], 0)
                self.assertTrue(torch.equal(original, rt.branch.base.down_proj.weight))
                self.assertIsNone(rt.active_core)
                self.assertFalse((rt.store / 'active-core.json').exists())

    def test_partial_replay_protection_scope_keeps_originally_wrong_rows_pending(self):
        from luban.evolution_progress import _protection_evidence
        _, rt, records, args = self.protected_fixture()
        report = evolution.consolidate(rt, **args)
        rows = unique_records(records['groups'], records['protection'], [])
        wrong = copy.deepcopy(records['groups']['replay_validation'][0])
        wrong['id'] = 'originally-wrong-replay'
        rows.append(wrong)
        report['split_ids']['replay_validation'].append(wrong['id'])
        for key in ('partial_baseline', 'partial_candidate_first', 'partial_candidate_second'):
            observation = copy.deepcopy(report[key][0])
            observation.update(id=wrong['id'], answer='wrong', passed=False, complete=True)
            report[key].append(observation)
        evidence, gates = _protection_evidence(report, rows)
        self.assertNotIn(wrong['id'], evidence['protected_replay_ids'])
        self.assertTrue(gates['protected_replay_generation'])
        protected_id = records['protection'][0]['id']
        item = next(p for p in report['partial_candidate_second'] if p['id'] == protected_id)
        item.update(answer='wrong', passed=False, complete=True)
        self.assertFalse(_protection_evidence(report, rows)[1]['protected_core_retention_generation'])

    def test_product450_pause_resume_retains_adam_and_baseline_exactly(self):
        _, control, _, args = self.product_fixture()
        evolution.consolidate(control, **args)
        expected = control.branch.base.down_proj.weight.detach().clone()
        expected_state = control.training_session.load()
        _, rt, records, args = self.product_fixture()
        save = evolution._save_core_step
        def pause(session, *values, **kwargs):
            save(session, *values, **kwargs)
            if rt.job['step'] == 64:
                session.pause_requested = True
                raise TrainingPaused('isolated product safe pause')
        with patch.object(evolution, '_save_core_step', pause), self.assertRaises(TrainingPaused):
            evolution.consolidate(rt, **args)
        rt.training_session.pause()
        run_id = rt.training_session.id
        rt.training_session = TrainingSession(rt, 'evolution', args, records, run_id=run_id)
        baseline = copy.deepcopy(rt.training_session.load()['report']['partial_training_baseline'])
        report = evolution.consolidate(rt, **args)
        self.assertTrue(report['promoted'], report)
        self.assertEqual(report['completed_steps'], 450)
        self.assertEqual(report['resumed_from_steps'], [64])
        self.assertEqual(report['partial_training_baseline'], baseline)
        self.assertTrue(torch.equal(expected, rt.branch.base.down_proj.weight))
        actual = rt.training_session.load()
        for key in ('exp_avg', 'exp_avg_sq', 'step'):
            self.assertTrue(torch.equal(expected_state['optimizer']['state'][0][key], actual['optimizer']['state'][0][key]))

    def test_product450_without_earned_experience_refuses_before_candidate(self):
        _, rt, records, args = self.product_fixture()
        records['experience'].pop()
        rt.training_session.body['records'] = records
        with patch.object(evolution, '_capture_replay_subspace') as basis, self.assertRaises(ValueError):
            evolution.consolidate(rt, **args)
        basis.assert_not_called()

    def fixture(self):
        case = test_evolution.EvolutionTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        rt = case.runtime
        rt.research = True
        rt.identity = "a" * 64
        rt.core_graph_verification["identity"] = rt.identity
        rt.store = rt.store / "app"
        with torch.no_grad():
            rt.branch.base.down_proj.weight.copy_(torch.tensor([[0., 0.], [.0001, 1.]]))
        rt._sample_tokens = lambda r: (torch.tensor([[0 if 'new' in r['prompt'] else 1, 1]]),
            torch.tensor([[-100, 0 if 'new' in r['prompt'] else 1]]))
        # One second NEW target is deliberately unlearnable in this tiny graph.
        rt.memory.add_sample('new pending training', 'different target', source='human',
            purpose='training', approved=True, training_allowed=True,
            provenance={'evolution_role': 'new', 'split_family': 'pending-only'})
        groups = evolution.consolidation_samples(rt)
        records = {'groups': groups, 'protection': [], 'retention_regressions': []}
        rt.training_session = TrainingSession(rt, 'evolution', copy.deepcopy(ARGS), records, scope='research')
        return case, rt, records

    def test_partial_actual_core_growth_pending_snapshot_and_proofs_file_binding(self):
        case, rt, records = self.fixture()
        original = rt.branch.base.down_proj.weight.clone()
        result = evolution.consolidate(rt, **ARGS)
        self.assertTrue(result['promoted'], result)
        self.assertLessEqual(result['completed_steps'], 352)
        value = result['learning_progress']
        self.assertEqual(value['newly_mastered'], 2)  # train+independent context, not a transfer claim
        self.assertEqual(value['pending'], 1)
        self.assertEqual(value['forgotten'], 0)
        self.assertNotIn(value['pending_ids'][0], [r['id'] for rows in read_core_retention(rt)['groups'].values() for r in rows])
        self.assertTrue(set(value['mastered_ids']) <= {r['id'] for r in unique_records(records['groups'], [], [])})
        protected = protected_validation(rt)
        mastered_train = [r['id'] for r in records['groups']['new_training'] if r['id'] in value['mastered_ids']]
        self.assertTrue(set('mastered-' + i for i in mastered_train) <= {r['id'] for r in protected})
        metadata = result['checkpoint']
        self.assertEqual(metadata['sha256'], hashlib.sha256((rt.store / (result['version'] + '.safetensors')).read_bytes()).hexdigest())
        validate_partial_progress(result, unique_records(records['groups'], [], []), rt.identity, result['version'], metadata['sha256'])
        run = training_runs(rt.memory, rt.identity)[0]
        self.assertNotIn(record_keys([r for r in records['groups']['new_training'] if r['id'] in value['pending_ids']]).pop(), receipt_keys(run))
        self.assertFalse(torch.equal(original, rt.branch.base.down_proj.weight))
        self.assertEqual([v['step'] for v in result['training_selection']['results']],
                         [v for v in checkpoints(352) if v <= result['completed_steps']])
        self.assertEqual(len(result['timing']['step_times']), result['completed_steps'])
        self.assertGreater(result['timing']['optimization_seconds'], 0)
        self.assertGreater(result['timing']['check_seconds'], 0)
        self.assertGreater(result['timing']['growth_per_active_minute'], 0)
        self.assertTrue(all(rt.model.generated))
        pending = evolution.consolidation_samples(rt)
        self.assertEqual({r["id"] for r in pending["new_training"]}, set(value["pending_ids"]))
        self.assertTrue(evolution.core_status(rt, partial_policy=PARTIAL_POLICY)["ready"])

    def test_old_answer_regression_refuses_whole_upgrade_even_new_growth(self):
        _, rt, _ = self.fixture()
        original = rt.branch.base.down_proj.weight.clone()
        native = rt.model.forward
        def corrupt_after_training(*args, **kwargs):
            output = native(*args, **kwargs)
            if torch.is_grad_enabled() and rt.branch.base.down_proj.weight.requires_grad:
                rt.model.replay_corruption = True
            return output
        rt.model.forward = corrupt_after_training
        result = evolution.consolidate(rt, **ARGS)
        self.assertFalse(result['promoted'])
        self.assertEqual(result['learning_progress']['newly_mastered'], 0)
        self.assertGreater(result['learning_progress']['forgotten'], 0)
        self.assertTrue(torch.equal(original, rt.branch.base.down_proj.weight))
        self.assertIsNone(rt.active_core)
        self.assertFalse((rt.store / 'active-core.json').exists())


    def test_pause_resume_keeps_growth_baseline_history_exposure_and_master_exact(self):
        _, control, _ = self.fixture()
        uninterrupted = evolution.consolidate(control, **ARGS)
        target = control.branch.base.down_proj.weight.detach().clone()
        _, rt, records = self.fixture()
        original_save = evolution._save_core_step
        def pause_once(session, *args, **kwargs):
            original_save(session, *args, **kwargs)
            if rt.job['step'] == 64:
                session.pause_requested = True
                raise TrainingPaused('CPU safe pause')
        with patch.object(evolution, '_save_core_step', pause_once), self.assertRaises(TrainingPaused):
            evolution.consolidate(rt, **ARGS)
        rt.training_session.pause()
        run_id = rt.training_session.id
        rt.training_session = TrainingSession(rt, 'evolution', copy.deepcopy(ARGS), records, run_id=run_id, scope='research')
        saved = rt.training_session.load()
        before = copy.deepcopy(saved['report']['partial_training_baseline'])
        self.assertEqual(saved['step'], 64)
        self.assertEqual([v['step'] for v in saved['report']['training_selection']['results']], [50])
        resumed = evolution.consolidate(rt, **ARGS)
        self.assertTrue(resumed['promoted'])
        self.assertTrue(torch.equal(target, rt.branch.base.down_proj.weight))
        self.assertEqual(resumed['partial_training_baseline'], before)
        self.assertEqual([v['step'] for v in resumed['training_selection']['results']], checkpoints(352))
        self.assertEqual(resumed['learning_progress']['newly_mastered'], uninterrupted['learning_progress']['newly_mastered'])
        self.assertEqual([v['step'] for v in resumed['timing']['step_times']], list(range(1, 353)))
        self.assertGreaterEqual(resumed['timing']['unmeasured_seconds'], 0)

    def test_real_upgrade_reader_and_tampered_mastery_or_pending_retention_fail_closed(self):
        from luban.upgrades import read_upgrade_records
        case, rt, records = self.fixture()
        report = evolution.consolidate(rt, **ARGS)
        with rt.memory.lock, rt.memory.connect() as db:
            body = json.loads(db.execute('SELECT body FROM training_runs WHERE id=?', (report['id'],)).fetchone()[0])
            body['product_accepted'] = True  # CPU publication contract fixture, no actual product authorization
            db.execute('UPDATE training_runs SET body=? WHERE id=?', (json.dumps(body), report['id']))
        value = read_upgrade_records(case.temp.name)
        self.assertTrue(value, value)
        core = next(v for v in value if v['kind'] == 'core')
        self.assertEqual(core['newly_mastered'], 2)
        self.assertEqual(core['pending'], 1)
        changed = copy.deepcopy(report)
        for mutation in ('hash', 'gate', 'safety', 'integer_bool', 'missing_reference'):
            bad = copy.deepcopy(report)
            if mutation == 'hash': bad.pop('partial_records_sha256')
            elif mutation == 'gate': bad['gates']['closed_book_growth'] = False
            elif mutation == 'safety': bad['partial_safety_failed'] = True
            elif mutation == 'integer_bool': bad['partial_baseline'][0]['passed'] = int(bad['partial_baseline'][0]['passed'])
            else: bad['partial_baseline'][0].pop('reference_used')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_partial_progress(bad, records, rt.identity, report['version'], report['checkpoint']['sha256'])
        changed['accepted_retention_groups']['new_training'].append(
            next(r for r in records['groups']['new_training'] if r['id'] in changed['learning_progress']['pending_ids']))
        with self.assertRaises(ValueError):
            validate_partial_progress(changed, records, rt.identity, report['version'], report['checkpoint']['sha256'])
        report['learning_progress']['newly_mastered'] = 99
        with self.assertRaises(ValueError):
            validate_partial_progress(report, records, rt.identity, report['version'], report['checkpoint']['sha256'])


    def test_train_selection_never_reads_dev_and_baseline_poison_refuses_before_basis(self):
        _, rt, _ = self.fixture()
        native = evolution._generation
        seen = []
        def observe(runtime, rows):
            phase = runtime.job.get('phase')
            if phase in {'preparing', 'training', 'training_check'}:
                self.assertTrue(all(r['purpose'] == 'training' for r in rows), 'NEW DEV must stay untouched during TRAIN selection')
            seen.append((phase, [r['id'] for r in rows]))
            return native(runtime, rows)
        with patch.object(evolution, '_generation', side_effect=observe):
            result = evolution.consolidate(rt, **ARGS)
        self.assertTrue(result['promoted'])
        self.assertFalse(result['promotion_thresholds']['all_new_generation_required'])
        self.assertTrue(result['promotion_thresholds']['new_loss_is_advisory'])
        self.assertEqual(len(result['effective_new_training_ids']), 2)
        self.assertTrue(any(phase == 'validation' and ids for phase, ids in seen))
        # Saved pre-run baseline cannot falsify a gold-correct answer as failed to mint growth.
        _, fresh, records = self.fixture()
        original_save = evolution._save_core_step
        def pause(session, *args, **kwargs):
            original_save(session, *args, **kwargs)
            if fresh.job['step'] == 64:
                session.pause_requested = True
                raise TrainingPaused('CPU baseline fixture')
        with patch.object(evolution, '_save_core_step', pause), self.assertRaises(TrainingPaused):
            evolution.consolidate(fresh, **ARGS)
        session = fresh.training_session
        session.pause_requested = False
        saved = session.load()
        proof = next(p for p in saved['report']['partial_training_baseline'] if p['passed'])
        proof['passed'] = False
        session.save(saved, {'step': 64, 'phase': 'training'})
        with patch.object(evolution, '_capture_replay_subspace') as basis, self.assertRaises(ValueError):
            evolution.consolidate(fresh, **ARGS)
        basis.assert_not_called()

    def test_timer_measures_known_pause_separately_and_does_not_invent_crash_pause(self):
        rt = type('Fixture', (), {'device': 'cpu', 'training_session': type('Session', (), {'body': {'pause_elapsed_seconds': 7.}})()})()
        report = {'started': 90.}
        with patch('luban.evolution_progress.time.time', return_value=100.), patch('luban.evolution_progress.time.perf_counter', side_effect=[0., 0., 2.]):
            timer = PhaseTimer(rt, report)
            timer.switch('optimization')
            timer.switch(None)
        self.assertEqual(report['timing']['optimization_seconds'], 2.)
        self.assertEqual(report['timing']['pause_seconds'], 7.)
        self.assertEqual(report['timing']['unmeasured_seconds'], 1.)

    def test_policy_budget_body_scope_refuses_before_candidate(self):
        for change in ('352strict', '704strict', 'weight', 'body', 'product', 'runtime', 'lr', 'budget'):
            _, rt, _ = self.fixture()
            args = copy.deepcopy(ARGS)
            if change == '352strict': args.pop('partial_policy')
            elif change == '704strict': args['steps'] = 704; args.pop('partial_policy')
            elif change == 'weight': args['field_first_token_weight'] = 4
            elif change == 'body': rt.training_session.body['arguments']['partial_policy'] = None
            elif change == 'product': rt.training_session.body['scope'] = 'product'
            elif change == 'runtime': rt.research = False
            elif change == 'lr': args['learning_rate'] = 2e-5
            else: args['steps'] = 351
            with self.subTest(change=change), patch.object(evolution, '_target') as target, self.assertRaises(ValueError):
                evolution.consolidate(rt, **args)
            target.assert_not_called()

    def test_reference_incomplete_or_single_pass_never_mastery_and_bad_receipt_refuses(self):
        _, rt, records = self.fixture()
        rows = unique_records(records['groups'], [], [])
        def observed(answer_new):
            return [{'id': r['id'], 'answer': answer_new if 'new' in r['prompt'] else r['answer'],
                     'passed': answer_new == r['answer'] if 'new' in r['prompt'] else True,
                     'complete': True, 'branch_enabled': False, 'reference_used': False} for r in rows]
        before, one, two = observed('wrong'), observed('newknowledge'), observed('wrong')
        for flag in ('branch_enabled', 'reference_used'):
            bad = copy.deepcopy(before)
            next(p for p in bad if not p['passed'])[flag] = True
            with self.subTest(failed_baseline=flag), self.assertRaises(ValueError):
                correct_ids(bad, rows)
        binding = {'core_version': 'core-012345abcdef', 'core_sha256': 'a' * 64, 'expert_id': None, 'expert_sha256': None}
        value = progress(rows, before, one, two, binding)
        self.assertEqual(value['candidate_newly_mastered'], 0)
        contains_rows = copy.deepcopy(rows)
        for row in contains_rows:
            source = json.loads(row['provenance'])
            source['generation_check'] = {'type': 'contains', 'expected': row['answer']}
            row['provenance'] = json.dumps(source)
        advisory = progress(contains_rows, observed('wrong'), observed('newknowledge'), observed('newknowledge'), binding)
        self.assertEqual(advisory['candidate_newly_mastered'], 0)
        self.assertEqual(advisory['mastered_ids'], [])
        for flag in ('complete', 'reference_used', 'branch_enabled'):
            bad = copy.deepcopy(one)
            item = next(v for v in bad if v['passed'] and 'new' in next(r['prompt'] for r in rows if r['id'] == v['id']))
            item[flag] = flag != 'complete'
            with self.subTest(flag=flag), self.assertRaises(ValueError): correct_ids(bad, rows)
        report = {'partial_policy': PARTIAL_POLICY, 'identity': rt.identity, 'promoted': False,
            'partial_baseline': before, 'partial_candidate_first': one, 'partial_candidate_second': two,
            'learning_progress': value, 'partial_records_sha256': content_sha(rows)}
        validate_partial_progress(report, rows, rt.identity, binding['core_version'], binding['core_sha256'])
        report['learning_progress']['learned_ids'] = [rows[0]['id']]
        with self.assertRaises(ValueError):
            validate_partial_progress(report, rows, rt.identity, binding['core_version'], binding['core_sha256'])


if __name__ == '__main__':
    unittest.main()
