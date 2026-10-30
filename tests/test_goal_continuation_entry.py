"""Temporary CPU checks for a distinct closed450 -> bounded1800 product episode."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import train_evolution_goal as goal
import release_goal_foundation as release
from luban import goal_sequence as sequence


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    return {'path': str(path), 'sha256': goal.sha(path)}

def observed_fixture(checked, plan, report):
    training = plan['groups']['new_training']
    checked['replay_generation_ids'] = [row['id'] for row in plan['groups']['replay_training'][:13]]
    checked['original_tensor_sha256'] = 'a' * 64
    checked['parent_final_tensor_sha256'] = 'b' * 64
    for key in ('parent_checkpoint', 'native_output_manifest', 'parent_journal'):
        checked['files'].setdefault(key, {'path': 'unused CPU mock', 'sha256': 'c' * 64})
    for key in ('parent_binding_sha256', 'parent_report_sha256'):
        checked.setdefault(key, 'd' * 64)
    for key in ('parent_implementation_sha256', 'new_implementation_sha256'):
        checked.setdefault(key, {})
    report['goal_sequence_sources'] = sequence.source_metadata(checked, {'results': report['training_selection']['results'][:13]})
    def row(identity, passed=True):
        return {'id': identity, 'passed': passed, 'complete': True, 'branch_enabled': False,
                'reference_used': False, 'raw_completion_token_ids': [1], 'finish_reason': 'stop'}
    def proof(step, label, count, reference, prior):
        ids = {sample['id'] for sample in training[:count]}
        values = {'new': [row(sample['id'], sample['id'] in ids) for sample in training],
                  'old13': [row(identity) for identity in checked['replay_generation_ids']], 'replay_training_loss': 1.0}
        return {'step': step, 'label': label, 'tensor_sha256': 'a' * 64 if step == 0 else 'b' * 64,
            'branch_enabled': False, 'reference_used': False, 'validation_rows_accessed': 0,
            'observed_at': 1.0, 'elapsed_seconds': 0.1, 'observations': [copy.deepcopy(values), copy.deepcopy(values)],
            'single_observation_correct': [count, count], 'confirmed_ids': sorted(ids), 'confirmed_count': count,
            'pending_ids': sorted({sample['id'] for sample in training} - ids), 'pending_count': 16 - count,
            'newly_confirmed_from_reference': sorted(ids - reference), 'newly_confirmed_since_prior': sorted(ids - prior),
            'forgotten_from_reference': sorted(reference - ids), 'forgotten_since_prior': sorted(prior - ids), 'old13_both_passed': True}
    reference = {sample['id'] for sample in training[:2]}
    entry = {sample['id'] for sample in training[:3]}
    stages = [proof(step, 'segment-boundary', 16, reference, entry) for step in (900, 1350, 1800) if step <= report['completed_steps']]
    # Each later boundary compares the union of earlier confirmed states.
    for stage in stages[1:]:
        stage['newly_confirmed_since_prior'] = []
    for stage in stages:
        stage.update(replay_loss_budget=1.06, replay_loss_both_passed=True, safety_failed=False, round_result='satisfied', actual_updates=450)
    report['training_selection']['results'] = report['training_selection']['results'][:13] + [
        {'step': stage['step'], 'new_generation': stage['observations'][0]['new'],
         'replay_training_generation': stage['observations'][0]['old13'], 'replay_loss': 1.0,
         'accepted': True, 'consecutive_passes': 0, 'same_weight_passes': 2,
         'minimum_new_exposures': (stage['step'] // 2) // 16, 'tensor_sha256': stage['tensor_sha256']} for stage in stages]
    report['training_selection'].update(policy_version='goal-fixed450-same-weight-v2', required_consecutive_passes=0, required_same_weight_passes=2)
    report['goal_sequence'].update(reference_baseline_after_parent_closed=proof(0, 'reference_baseline_after_parent_closed', 2, set(), set()),
                                  entry450={**proof(450, 'entry450_after_parent_closed', 3, reference, reference),
                                     'actual_updates':450, 'replay_loss_budget':1.06, 'replay_loss_both_passed':True,
                                     'safety_failed':False, 'round_result':'partial'}, stage_observations=stages)
    rows = {phase: [row(sample['id']) for sample in training] for phase in ('hf_new_first', 'hf_new_second', 'native_new_first', 'native_new_second')}
    return rows


class ContinuationEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.parent_dir = self.root / 'parent'
        self.parent_dir.mkdir()
        self.source = self.root / 'source/modules/luban0-v1'
        self.source.mkdir(parents=True)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.baseline = [{'id': 'old-' + str(i), 'answer': 'full original observation', 'passed': bool(i % 2)} for i in range(83)]
        self.groups = {'new_training': [{'id': 'new-' + str(i)} for i in range(16)],
                       'new_validation': [{'id': 'valid-' + str(i)} for i in range(4)],
                       'replay_training': [{'id': 'old-' + str(i)} for i in range(83)],
                       'replay_validation': [{'id': 'oldvalid-' + str(i)} for i in range(8)]}
        self.protection = [{'id': 'protect-' + str(i)} for i in range(89)]
        self.extra = [{'id': 'accept-' + str(i)} for i in range(8)]
        self.files = {}
        aliases = {'new_material': 'materials', 'old_retention_complete': 'old_retention',
                   'old_training_original': 'old_training', 'old_protection': 'old_protection', 'acceptance': 'acceptance'}
        for name in aliases.values():
            self.files[name] = save(self.root / (name + '.json'), {})
        self.files['formal_protocol'] = save(self.root / 'formal.json', {})
        formal_files = {key: self.files[alias] for key, alias in aliases.items()}
        self.parent = {'protocol': goal.PROTOCOL, 'arguments': copy.deepcopy(goal.ARGUMENTS), 'seed': 42,
                       'starting_core': 'core-b45d965728bb', 'identity': 'a' * 64,
                       'source_store': str(self.source), 'source_files': {}, 'source_sha256': {'old': 'b' * 64},
                       'model_root': str(self.model), 'model_config_sha256': 'c' * 64, 'model_manifest_sha256': 'd' * 64,
                       'formal_protocol': self.files['formal_protocol']['path'],
                       'formal_protocol_sha256': self.files['formal_protocol']['sha256'], 'formal_files': formal_files,
                       'groups': self.groups, 'protection': self.protection, 'retention_regressions': self.extra,
                       'acceptance': self.extra, 'old_training_original': copy.deepcopy(self.groups['replay_training']),
                       'output': str(self.parent_dir), 'home': str(self.parent_dir / 'isolated-home'),
                       'storage': str(self.parent_dir / 'isolated-storage'), 'budget_policy': 'original450',
                       'teacher_api_calls': 0, 'model_started': False}
        self.files['parent_plan'] = save(self.parent_dir / 'plan.json', self.parent)
        self.report = {'id': '123456abcdef', 'previous_version': self.parent['starting_core'],
                       'completed_steps': 450, 'promoted': False}
        self.result = {'protocol': goal.PROTOCOL, 'plan_sha256': self.files['parent_plan']['sha256'],
                       'passed': False, 'released': True, 'production_written': False, 'adopted': False,
                       'teacher_api_calls': 0, 'source_core': self.parent['starting_core'], 'training': self.report,
                       'baseline_old_training': self.baseline}
        self.files['parent_result'] = save(self.parent_dir / 'acceptance.json', self.result)
        self.files['baseline_old_training'] = save(self.parent_dir / 'baseline.json', self.baseline)
        self.records = {'groups': self.groups, 'protection': self.protection, 'retention_regressions': self.extra}
        self.files['records'] = save(self.parent_dir / 'records.json', self.records)
        self.checked = {'protocol': sequence.PROTOCOL, 'status': 'FROZEN_CLOSED_PARENT', 'parent_steps': 450,
                        'max_steps': 1800, 'segment_steps': 450, 'max_segments': 4, 'checkpoints': sequence.CHECKPOINTS,
                        'parent_arguments': copy.deepcopy(goal.ARGUMENTS), 'seed': 42, 'batch_size': 1,
                        'gradient_accumulation': 1, 'parent_run_id': self.report['id'], 'new_run_id': 'fedcba654321',
                        'policy_version': 'goal-fixed450-same-weight-v2', 'required_consecutive_passes': 0, 'required_same_weight_passes': 2,
                        'required_new_exposures': 2, 'reference_policy': 'original-accepted-core',
                        'basis_policy': 'exact-parent-basis', 'validation_policy': 'first-complete450-same-W-double-pass-once-then-stop',
                        'starting_core': self.parent['starting_core'], 'identity': self.parent['identity'],
                        'records_sha256': sequence.content_sha(self.records), 'files': self.files,
                        'new_source_sha256': {'new-full-bound-source': 'e' * 64}}
        self.sequence_path = self.root / 'sequence.json'
        save(self.sequence_path, self.checked)
        self.checked.update(_protocol_path=str(self.sequence_path), _protocol_sha256=goal.sha(self.sequence_path))
        self.load = patch.object(sequence, 'load_verified', return_value=self.checked).start()
        self.hashes = patch.object(goal, 'source_hashes', return_value=self.checked['new_source_sha256']).start()
        self.addCleanup(patch.stopall)

    def prepare(self, output=None):
        return goal.prepared_continuation(SimpleNamespace(goal_sequence_protocol=self.sequence_path,
                                                         output=output or self.root / 'new'))

    def test_prepare_distinct_budget_preserves_originals_and_baseline_without_model(self):
        before = {path: path.read_bytes() for path in self.parent_dir.iterdir()}
        answer = self.prepare()
        plan = goal.read(answer['plan'])
        self.assertEqual(plan['protocol'], sequence.PROTOCOL)
        self.assertEqual(plan['arguments']['steps'], 1800)
        self.assertEqual(plan['goal_sequence']['parent_steps'], 450)
        self.assertEqual(plan['goal_sequence']['new_steps_budget'], 1350)
        self.assertEqual(plan['groups'], self.parent['groups'])
        self.assertEqual(plan['protection'], self.parent['protection'])
        self.assertEqual(plan['inherited_old_baseline'], self.files['baseline_old_training'])
        self.assertFalse(plan['model_started'])
        self.assertEqual(before, {path: path.read_bytes() for path in self.parent_dir.iterdir()})
        self.assertNotEqual(Path(plan['output']), self.parent_dir)

    def test_live_passed_unreleased_or_heldout_parent_rejected_before_new_directory(self):
        original = copy.deepcopy(self.result)
        for change in ({'released': False}, {'passed': True}, {'adopted': True},
                       {'training': {**self.report, 'completed_steps': 448}},
                       {'training': {**self.report, 'validation_new_loss': .1}}):
            with self.subTest(change=change):
                save(Path(self.files['parent_result']['path']), {**original, **change})
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse((self.root / 'new').exists())

    def test_filtering_original83_baseline_rejected(self):
        save(Path(self.files['baseline_old_training']['path']), self.baseline[:-1])
        with self.assertRaises(ValueError):
            self.prepare()

    def test_parent_production_and_model_directory_overlap_rejected(self):
        for output in (self.parent_dir / 'another', self.model / 'another', self.source / 'another'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.prepare(output)

    def test_changed_new_source_rejected_before_plan_creation(self):
        self.hashes.return_value = {'wrong': 'f' * 64}
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse((self.root / 'new').exists())

    def test_publisher_explicit_new_protocol_preserves_full_old_materials(self):
        plan = goal.read(self.prepare()['plan'])
        args, checksum, continued = release.release_contract(plan)
        self.assertTrue(continued)
        self.assertEqual(args['steps'], 1800)
        self.assertEqual(checksum, self.checked['_protocol_sha256'])
        for field in ('protection', 'old_training_original', 'acceptance'):
            bad = copy.deepcopy(plan)
            bad[field] = bad[field][:-1]
            with self.subTest(field=field), self.assertRaises(ValueError):
                release.release_contract(bad)

    def test_original450_cannot_be_relabelled1800_or_unknown_protocol(self):
        old = {'protocol': goal.PROTOCOL, 'arguments': copy.deepcopy(goal.ARGUMENTS),
               'formal_protocol_sha256': 'a' * 64}
        self.assertEqual(release.release_contract(old), (goal.ARGUMENTS, 'a' * 64, False))
        for change in ({'protocol': 'invented'}, {'protocol': 'evolution-goal-continuation-native-v1'}, {'arguments': {**goal.ARGUMENTS, 'steps': 1800}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                release.release_contract({**old, **change})


class ActualCandidateJournalTests(ContinuationEntryTests):
    def test_actual_Bun_C_commit_and_event_are_read_only_verified(self):
        from contextlib import closing
        from luban.bun_bridge import RemoteMemory, readonly_connection
        plan = goal.read(self.prepare()['plan'])
        self.files['parent_report'] = save(self.parent_dir / 'checkpoint-report.json', {
            'training_selection': {'results': [{'step': step} for step in sequence.HISTORY]}})
        history = [{'step': step} for step in sequence.HISTORY + [900]]
        report = {'id': self.checked['new_run_id'], 'completed_steps': 900,
                  'goal_sequence': sequence.report_metadata(self.checked, 900),
                  'training_selection': {'checkpoints': sequence.CHECKPOINTS, 'results': history, 'selected_step': 900}}
        report['baseline_training_replay_loss'] = 1.0
        rows = observed_fixture(self.checked, plan, report)
        records = {key: plan[key] for key in ('groups', 'protection', 'retention_regressions')}
        journal = {'id': report['id'], 'identity': plan['identity'], 'kind': 'evolution', 'scope': 'product',
                   'state': 'committed', 'binding': {'scope': 'product', 'arguments': plan['arguments']},
                   'arguments': plan['arguments'], 'records': records,
                   'commit': {'report': report, 'artifacts': {'cpu-fixture': 'b' * 64}, 'event_kind': 'evolution'}}
        accepted = {'goal_sequence': report['goal_sequence'], 'training_journal': journal,
                    'training_event': {'kind': 'evolution', 'body': report}, 'artifacts': journal['commit']['artifacts'],
                    'rows': rows, 'candidate_tensor_sha256': 'b' * 64,
                    'terminal_selected_training_confirmation': goal.terminal_selected_confirmation(plan, report, rows, 'b' * 64)}
        with goal.data_service(self.root / 'ledger-owner-home', self.root / 'ledger-owner.json'):
            with patch('sqlite3.connect', side_effect=AssertionError('Python SQLite forbidden')):
                memory = RemoteMemory(plan['storage'])
                with memory.connect() as db:
                    db.execute('CREATE TABLE training_runs(id TEXT PRIMARY KEY,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)')
                    body = {key: value for key, value in journal.items() if key != 'state'}
                    db.execute('INSERT INTO training_runs VALUES(?,?,?,?,?,?)',
                               (report['id'], plan['identity'], 'evolution', 'committed', json.dumps(body), 1.0))
                    db.execute('INSERT INTO events VALUES(?,?,?,?)',
                               ('training:' + report['id'], 'evolution', json.dumps(report), 1.0))
                release.sequence_report(plan, accepted, report)
                with closing(readonly_connection(memory.path)) as db:
                    self.assertEqual(db.execute('SELECT COUNT(*) FROM training_runs').fetchone()[0], 1)
                with memory.connect() as db:
                    db.execute('UPDATE training_runs SET state=? WHERE id=?', ('rejected', report['id']))
                with self.assertRaises(ValueError):
                    release.sequence_report(plan, accepted, report)


import tests.test_verify_goal_app as foundation_tests
import verify_goal_foundation as fresh
import verify_goal_app as app
from luban.upgrades import CORE_GATES


class ContinuedFoundationProofTests(foundation_tests.FoundationBindingTests):
    def setUp(self):
        super().setUp()
        self.parent_dir = self.root / 'parent450'
        self.parent_dir.mkdir()
        names = {'new_material': 'materials.json', 'old_retention_complete': 'old-retention.json',
                 'old_training_original': 'old-training.json', 'old_protection': 'old-protection.json',
                 'acceptance': 'questions.json'}
        save(self.protocol, {**names, 'source_core': {'version': self.previous, 'identity': self.identity}})
        aliases = {'new_material': 'materials', 'old_retention_complete': 'old_retention',
                   'old_training_original': 'old_training', 'old_protection': 'old_protection', 'acceptance': 'acceptance'}
        files = {alias: save(self.root / names[key], {'CPU fixture': key}) for key, alias in aliases.items()}
        files['formal_protocol'] = {'path': str(self.protocol), 'sha256': app.sha(self.protocol)}
        implementation = ('training_checkpoint.py', 'runtime.py', 'evolution.py', 'evolution_progress.py', 'calibration.py',
                          'retention.py', 'dynamic.py', 'decision.py', 'generation_checks.py', 'coldpaging.py', 'backbone.py',
                          'automatic.py', 'process_lock.py', 'context.py', 'core_publication.py', 'external_brain.py',
                          'luban_training.py', 'native_core_training.py', 'foundation.py')
        snapshots = {}
        for name in implementation:
            path = self.parent_dir / name
            path.write_text('original CPU source fixture ' + name)
            snapshots[name] = {'path': str(path), 'sha256': app.sha(path)}
        old_impl = {name: entry['sha256'] for name, entry in snapshots.items()}
        old_source = {'src/luban/' + name: value for name, value in old_impl.items()}
        old_source['scripts/train_evolution_goal.py'] = old_impl['runtime.py']
        source_snapshot = {'src/luban/' + name: entry for name, entry in snapshots.items()}
        source_snapshot['scripts/train_evolution_goal.py'] = snapshots['runtime.py']
        new_source = {**old_source, 'src/luban/evolution.py': 'b' * 64,
                      'scripts/train_evolution_goal.py': 'b' * 64, 'src/luban/goal_sequence.py': 'c' * 64}
        groups = {'replay_training': [{'id': 'old-' + str(i)} for i in range(83)],
                  'new_training': [{'id': 'new-' + str(i)} for i in range(16)],
                  'new_validation': [{'id': 'valid-' + str(i)} for i in range(4)], 'replay_validation': []}
        records = {'groups': groups, 'protection': [], 'retention_regressions': []}
        baseline = [{'id': row['id'], 'answer': 'unchanged observation'} for row in groups['replay_training']]
        parent_plan = {'protocol': goal.PROTOCOL, 'arguments': goal.ARGUMENTS, 'source_sha256': old_source,
                       'formal_protocol_sha256': app.sha(self.protocol), 'identity': self.identity, 'starting_core': self.previous}
        files['parent_plan'] = save(self.parent_dir / 'plan.json', parent_plan)
        files['records'] = save(self.parent_dir / 'records.json', records)
        files['baseline_old_training'] = save(self.parent_dir / 'baseline.json', baseline)
        parent_binding = {'implementation_sha256': old_impl, 'records_sha256': fresh.digest(records)}
        files['parent_binding'] = save(self.parent_dir / 'binding.json', parent_binding)
        files['parent_checkpoint'] = save(self.parent_dir / 'checkpoint.json', {'actual CPU fixture checkpoint': 450})
        parent_report = {'id': '123456abcdef', 'training_selection': {'results': [{'step': step} for step in sequence.HISTORY]}}
        files['parent_report'] = save(self.parent_dir / 'report.json', parent_report)
        files['parent_journal'] = save(self.parent_dir / 'journal.json', {'id': parent_report['id'], 'state': 'rejected',
            'binding': parent_binding, 'records': records, 'checkpoint': {'sha256': files['parent_checkpoint']['sha256']}})
        files['parent_result'] = save(self.parent_dir / 'acceptance.json', {
            'plan_sha256': files['parent_plan']['sha256'], 'passed': False, 'released': True,
            'production_written': False, 'adopted': False, 'baseline_old_training': baseline,
            'training': {'completed_steps': 450, 'promoted': False}})
        for name in ('formal_manifest', 'native_manifest', 'native_output_manifest', 'native_trace'):
            files[name] = save(self.parent_dir / (name + '.json'), {'CPU evidence fixture': name})
        self.sequence_path = self.root / 'sequence.json'
        frozen = {'protocol': goal.CONTINUATION_PROTOCOL, 'status': 'FROZEN_CLOSED_PARENT', 'parent_steps': 450,
                  'max_steps': 1800, 'segment_steps': 450, 'max_segments': 4, 'parent_run_id': '123456abcdef',
                  'new_run_id': self.source[5:], 'identity': self.identity, 'starting_core': self.previous,
                  'files': files, 'parent_source_sha256': old_source, 'new_source_sha256': new_source,
                  'parent_source_snapshot': source_snapshot, 'parent_implementation_snapshot': snapshots,
                  'parent_implementation_sha256': old_impl,
                  'parent_binding_sha256': fresh.digest(parent_binding), 'parent_report_sha256': fresh.digest(parent_report),
                  'records_sha256': fresh.digest(records), 'native_files': {files['native_trace']['path']: files['native_trace']['sha256']},
                  'new_implementation_sha256': {name: new_source['src/luban/' + name] for name in (*implementation, 'goal_sequence.py')},
                  'closure': {'exit_code': 0, 'validation_reads': 0, 'final_step': 450, 'released': True,
                              'adopted': False, 'production_written': False, 'parent_journal_state': 'rejected', 'active_core': self.previous}}
        frozen.update(original_tensor_sha256='a' * 64, parent_final_tensor_sha256='b' * 64,
                      replay_generation_ids=[row['id'] for row in groups['replay_training'][:13]])
        save(self.sequence_path, frozen)
        checksum = app.sha(self.sequence_path)
        meta = {'protocol_sha256': checksum, 'parent_run_id': frozen['parent_run_id'],
                'parent_steps': 450, 'total_budget': 1800, 'new_steps_budget': 1350}
        self.plan.update(protocol=goal.CONTINUATION_PROTOCOL,
            arguments={**goal.ARGUMENTS, 'steps': 1800, 'goal_sequence': meta}, goal_sequence=meta,
            goal_sequence_protocol={'path': str(self.sequence_path), 'sha256': checksum},
            source_sha256=new_source, identity=self.identity, starting_core=self.previous,
            formal_protocol=str(self.protocol), formal_protocol_sha256=app.sha(self.protocol),
            formal_files={key: files[alias] for key, alias in aliases.items()}, **records)
        save(self.acceptance.parent / 'plan.json', self.plan)
        source_report_name = 'evolution-' + self.source[5:] + '.json'
        source_report = app.read(self.store / source_report_name)
        execution = {**meta, 'protocol': goal.CONTINUATION_PROTOCOL, 'new_steps_executed': 450, 'cumulative_steps': 900,
                     'selected_segment': 2, 'training_budget_scope': 'bounded goal-specific episode, ordinary product512 limit unchanged'}
        source_report.update(steps=1800, completed_steps=900, baseline_training_replay_loss=1.0, goal_sequence=execution, gates=dict.fromkeys(CORE_GATES, True),
            training_selection={'checkpoints': sequence.CHECKPOINTS, 'selected_step': 900,
                                'results': parent_report['training_selection']['results'] + [{'step': 900}]})
        frozen['_protocol_path'] = str(self.sequence_path)
        rows = observed_fixture(frozen, self.plan, source_report)
        save(self.store / source_report_name, source_report)
        self.candidate.update(protocol=goal.CONTINUATION_PROTOCOL, plan_sha256=app.sha(self.acceptance.parent / 'plan.json'),
            released=True, production_written=False, adopted=False, teacher_api_calls=0, training=source_report,
            goal_sequence=source_report['goal_sequence'], rows=rows,
            terminal_selected_training_confirmation=goal.terminal_selected_confirmation(self.plan, source_report, rows, self.candidate['candidate_tensor_sha256']),
            native_loaded={'core_loaded': True, 'core_tensor_sha256': self.candidate['candidate_tensor_sha256'],
                'core_file_sha256': self.candidate['artifacts'][self.source + '.safetensors']},
            training_journal={'id': frozen['new_run_id'], 'state': 'committed', 'scope': 'product',
                'binding': {'scope': 'product', 'arguments': self.plan['arguments']}, 'arguments': self.plan['arguments'],
                'records': records, 'commit': {'report': source_report, 'artifacts': self.candidate['artifacts'], 'event_kind': 'evolution'}},
            training_event={'kind': 'evolution', 'body': source_report})
        self.candidate['artifacts'][source_report_name] = app.sha(self.store / source_report_name)
        self.candidate['training_journal']['commit']['artifacts'] = self.candidate['artifacts']
        save(self.acceptance, self.candidate)
        self.report = {**source_report, 'version': self.version, 'id': self.version[5:],
                       'checkpoint': self.report['checkpoint'], 'retention': self.report['retention'],
                       'source_previous_version': self.previous, 'source_training_version': self.source,
                       'source_quality_sha256': self.candidate['artifacts'][source_report_name],
                       'source_protocol_sha256': checksum, 'native_identity_sha256': app.sha(self.acceptance),
                       'native_identity_verified': True}
        self.reseal()

    def test_std_only_actual_F_C_plan_sequence_formal_chain(self):
        result = fresh.continued_training_binding(self.acceptance, self.protocol, self.report)
        self.assertEqual(result['training_protocol_sha256'], app.sha(self.sequence_path))
        self.assertEqual(result['goal_sequence']['new_steps_executed'], 450)
        self.assertEqual(self.candidate['terminal_selected_training_confirmation']['confirmed_count'], 16)
        self.assertEqual(self.binding()['continued_training_binding'], result)

    def test_false_relabelled_tampered_or_wrong_new_plan_proof_rejected(self):
        for change in ({'passed': False}, {'protocol': goal.PROTOCOL}, {'plan_sha256': 'f' * 64},
                       {'goal_sequence': {**self.candidate['goal_sequence'], 'new_steps_executed': 900}}):
            value = {**copy.deepcopy(self.candidate), **change}
            save(self.acceptance, value)
            report = {**self.report, 'native_identity_sha256': app.sha(self.acceptance)}
            with self.subTest(change=change), self.assertRaises(ValueError):
                fresh.continued_training_binding(self.acceptance, self.protocol, report)

    def test_fresh_binding_in_real_CPU_process_forbids_torch_and_SQLite_writes(self):
        import subprocess
        import sys
        code = """import builtins,json,sys,sqlite3
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Model/Torch import forbidden in immutable STD proof')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sqlite3.connect=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('SQLite writes forbidden'))
from pathlib import Path
import verify_goal_foundation as verifier
result=verifier.continued_training_binding(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()))
print(result['candidate_version'])
"""
        report_path = self.root / 'F-report-input.json'
        save(report_path, self.report)
        process = subprocess.run([sys.executable, '-c', code, str(self.acceptance), str(self.protocol), str(report_path)],
            text=True, capture_output=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), self.source)

    def test_partial_same_W_round_records_pending_without_false_final_selection(self):
        frozen = app.read(self.sequence_path)
        frozen['_protocol_path'] = str(self.sequence_path)
        partial = copy.deepcopy(self.candidate['training'])
        stage = partial['goal_sequence']['stage_observations'][-1]
        stage['observations'][1]['new'][-1]['passed'] = False
        confirmed = {row['id'] for row in stage['observations'][1]['new'] if row['passed']}
        reference = set(partial['goal_sequence']['reference_baseline_after_parent_closed']['confirmed_ids'])
        prior = set(partial['goal_sequence']['entry450']['confirmed_ids'])
        all_ids = {row['id'] for row in self.plan['groups']['new_training']}
        stage.update(single_observation_correct=[16, 15], confirmed_ids=sorted(confirmed), confirmed_count=15,
            pending_ids=sorted(all_ids - confirmed), pending_count=1, round_result='partial',
            newly_confirmed_from_reference=sorted(confirmed - reference), newly_confirmed_since_prior=sorted(confirmed - prior),
            forgotten_from_reference=sorted(reference - confirmed), forgotten_since_prior=sorted(prior - confirmed))
        partial['training_selection']['selected_step'] = None
        partial['training_selection']['results'][-1].update(accepted=False, same_weight_passes=0)
        goal.goal_observation_binding(self.plan, partial, frozen)
        self.assertEqual(stage['pending_count'], 1)
        partial['training_selection']['selected_step'] = 900
        with self.assertRaises(ValueError):
            goal.goal_observation_binding(self.plan, partial, frozen)
        # Zero learned is still an executed round record, never a fully mastered final C.
        partial['training_selection']['selected_step'] = None
        for values in stage['observations']:
            for row in values['new']:
                row['passed'] = False
        stage.update(single_observation_correct=[0, 0], confirmed_ids=[], confirmed_count=0,
            pending_ids=sorted(all_ids), pending_count=16, round_result='protection_failed', safety_failed=True, newly_confirmed_from_reference=[], newly_confirmed_since_prior=[],
            forgotten_from_reference=sorted(reference), forgotten_since_prior=sorted(prior))
        partial['training_selection']['results'][-1].update(new_generation=stage['observations'][0]['new'])
        goal.goal_observation_binding(self.plan, partial, frozen)
        self.assertEqual(stage['confirmed_count'], 0)
        self.assertEqual(stage['pending_count'], 16)

    def test_same_W_double_observation_added_forgotten_and_final_selected_binding_reject_tamper(self):
        original = copy.deepcopy(self.candidate['training'])
        frozen = app.read(self.sequence_path)
        frozen['_protocol_path'] = str(self.sequence_path)
        goal.goal_observation_binding(self.plan, original, frozen)
        for field, change in (
            ('entry450', {'tensor_sha256': 'f' * 64}),
            ('reference_baseline_after_parent_closed', {'label': 'pretend_pre_training_baseline'}),
            ('entry450', {'confirmed_count': 16}),
            ('entry450', {'pending_count': 0}),
            ('entry450', {'pending_ids': []}),
            ('entry450', {'forgotten_from_reference': ['fabricated']}),
            ('entry450', {'observations': []}),
        ):
            bad = copy.deepcopy(original)
            bad['goal_sequence'][field].update(change)
            with self.subTest(field=field, change=change), self.assertRaises(ValueError):
                goal.goal_observation_binding(self.plan, bad, frozen)
        terminal = goal.terminal_selected_confirmation(self.plan, original, self.candidate['rows'], 'b' * 64)
        self.assertEqual(terminal['step'], 900)
        for forbidden in (868, 1318, 1768):
            early = copy.deepcopy(original)
            early['completed_steps'] = forbidden
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                goal.terminal_selected_confirmation(self.plan, early, self.candidate['rows'], 'c' * 64)
        bad_rows = copy.deepcopy(self.candidate['rows'])
        bad_rows['native_new_second'][0]['complete'] = False
        with self.assertRaises(ValueError):
            goal.terminal_selected_confirmation(self.plan, original, bad_rows, 'b' * 64)

    def test_mutated_closed_parent_or_formal_file_rejected(self):
        frozen = app.read(self.sequence_path)
        parent = Path(frozen['files']['parent_journal']['path'])
        parent.write_text('{"state":"running"}\n')
        with self.assertRaises(ValueError):
            fresh.continued_training_binding(self.acceptance, self.protocol, self.report)


if __name__ == '__main__':
    unittest.main()
