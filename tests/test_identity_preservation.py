"""Actual SQLite publication, frozen obligations and baseline guards; CPU only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_identity_known_correction import corrected_plan
from prepare_identity_preservation import preservation_plan, obligations
from run_identity_coverage import qualify_required_replay, qualify_required_replay_once, qualified_groups, tensor_digest
import test_core_publication
from test_core_publication import sample
from test_identity_known_correction import correction_fixture
from luban.core_publication import (PRESERVATION_CORRECTIONS, content_sha,
                                    validate_material_publication, apply_material_publication)
from luban.evolution import _training_replay_generation, TRAIN_CHECKPOINTS
from luban.generation_checks import evaluate_generation
from luban.memory import Memory
from luban.retention import provenance, save_core_retention, read_core_retention, protected_validation


def preservation_fixture():
    original, result, material = correction_fixture()
    old = corrected_plan(original, result, prepared=2., session_id='abcdef123456', directory=Path('/unused'))
    old['engineering_scope'] = 'untouched16 once only'
    old['training_checkpoints'] = list(TRAIN_CHECKPOINTS)
    old['four_removed_duplicates'] = material['superseded_ids'][:4]
    baseline_new, baseline_replay = [], []
    for index, (_, (old_id, word, role)) in enumerate(PRESERVATION_CORRECTIONS.items()):
        key = 'replay_validation' if role == 'replay' and word == 'PyTorch' else 'new_validation'
        row = old['groups_before_original_baseline'][key][1 if word == 'olive_path' else 0]
        row.update(id=old_id, prompt=f'把字符串“{word}”转为大写，不添加解释。', answer=word.upper())
        value = provenance(row)
        value.update(messages=[{'role': 'system', 'content': '直接完成当前任务。'},
                               {'role': 'user', 'content': row['prompt']}],
                     generation_check={'type': 'semantic', 'validator': 'normalized_exact',
                                       'expected': word.upper(), 'max_new_tokens': 192})
        row['provenance'] = json.dumps(value)
        if word == 'PyTorch':
            source = old['source_rows'][-17]
            source.clear()
            source.update(copy.deepcopy(row))
        observed = {'id': old_id, 'answer': word.upper() if role == 'replay' else 'Cedar',
                    'passed': role == 'replay', 'complete': True, 'branch_enabled': False}
        (baseline_replay if key == 'replay_validation' else baseline_new).append(observed)
    result = {'passed': False, 'engineering': [], 'training': {'promoted': False,
        'baseline_new_generation': baseline_new, 'baseline_replay_generation': baseline_replay},
        'retirement': {'version': None, 'restored': True, 'tensor_sha256': 'a' * 64}}
    return old, result, material


def qualified_row(row, identity='fixture-base'):
    row = copy.deepcopy(row)
    value = provenance(row)
    value.update(training_generation_gate=True, baseline_preservation_eligible=True,
        original_baseline_verified_identity=identity, original_baseline_verified_core=None,
        original_baseline_generation={'id': row['id'], 'answer': row['answer'], 'passed': True,
                                      'complete': True, 'branch_enabled': False},
        qualification_scope='current original core')
    row['provenance'] = json.dumps(value)
    return row


class IdentityPreservationTests(unittest.TestCase):
    def build(self):
        old, result, material = preservation_fixture()
        return preservation_plan(old, result, prepared=3., session_id='123456abcdef', directory=Path('/unused')), old, material

    def pool(self, plan):
        rows = [qualified_row(sample(f'pool-qualified-{i}', 'replay', 'training')) for i in range(4)]
        return {'passed': True, 'qualified_count': 4, 'qualified': rows}

    def qualified(self, plan):
        rows = [qualified_row(row) for row in plan['groups_before_original_baseline']['replay_training']
                if row['id'] in plan['required_replay_training_ids']]
        return {'passed': True, 'qualified': rows}

    def material(self):
        plan, old, material = self.build()
        material.update(groups=qualified_groups(plan, self.pool(plan), self.qualified(plan)),
            source_rows=old['source_rows'], source_sha256=content_sha(old['source_rows']),
            retention_regressions=plan['retention_regressions'], superseded_ids=plan['superseded_source_ids'])
        return material

    def test_exact_reclassification_preserves_103_all49_gold_checks_and_kept_training(self):
        plan, old, _ = self.build()
        self.assertEqual({key: len(rows) for key, rows in plan['groups_before_original_baseline'].items()},
                         {'new_training': 37, 'new_validation': 6, 'replay_training': 46, 'replay_validation': 16})
        self.assertEqual(plan['source_rows'], old['source_rows'])
        self.assertEqual(plan['source_groups'], old['source_groups'])
        self.assertEqual(plan['replay_training_candidates'], old['replay_training_candidates'])
        self.assertEqual(plan['engineering_cases'], old['engineering_cases'])
        self.assertEqual(plan['arguments'], old['arguments'])
        self.assertEqual(obligations(plan['groups_before_original_baseline'], plan['retention_regressions']),
                         obligations(old['groups_before_original_baseline'], old['retention_regressions']))
        self.assertEqual(len(plan['superseded_source_ids']), 13)
        self.assertEqual(len(plan['retention_regressions']), 27)
        for row in plan['preservation_training_rows']:
            value = provenance(row)
            self.assertIs(value['human_verified'], False)
            self.assertIs(value['program_verified'], True)
            self.assertEqual(row['answer'], value['source_string'].upper())
            self.assertNotIn('original_baseline_generation', value)
            if value['evolution_role'] == 'replay':
                self.assertIs(value['training_generation_gate'], False)
                self.assertIsNone(value['baseline_preservation_eligible'])
        self.assertFalse(plan['sealed'])

    def test_literal_source_mismatch_historical_non_eos_or_false_old_capability_refuses_draft(self):
        for change in ('literal', 'gold', 'eos', 'baseline'):
            old, result, _ = preservation_fixture()
            if change in {'literal', 'gold'}:
                row = old['groups_before_original_baseline']['new_validation'][0]
                if change == 'gold':
                    row['answer'] = 'Olive_path'
                else:
                    value = provenance(row)
                    value['messages'][-1]['content'] = '把字符串“wrong”转换。'
                    row['provenance'] = json.dumps(value)
            else:
                row = result['training']['baseline_replay_generation'][0]
                row['complete' if change == 'eos' else 'passed'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                preservation_plan(old, result, prepared=3., session_id='123456abcdef', directory=Path('/unused'))

    def test_current_original_baseline_uses_new_ids_complete_context_and_both_required(self):
        plan, _, _ = self.build()
        weight = torch.eye(2)
        plan['original_tensor_sha256'] = tensor_digest(weight)
        runtime = SimpleNamespace(identity=plan['base_identity'], active_core=None)
        seen = []
        def generate(runtime, rows):
            seen.extend(copy.deepcopy(rows))
            return [provenance(qualified_row(row))['original_baseline_generation'] for row in rows]
        with patch('run_identity_coverage._target', return_value=('weight', weight)), \
                patch('run_identity_coverage.branch_off_generation', side_effect=generate):
            result = qualify_required_replay(runtime, plan)
        self.assertTrue(result['passed'])
        self.assertEqual({row['id'] for row in seen}, set(plan['required_replay_training_ids']))
        self.assertTrue(all(row['purpose'] == 'training' for row in seen))
        self.assertTrue(all('original_baseline_generation' not in provenance(row) for row in seen))
        groups = qualified_groups(plan, self.pool(plan), result)
        self.assertEqual(len(_training_replay_generation(runtime, groups)), 6)
        self.assertEqual(len(groups['replay_training']), 50)
        bad = copy.deepcopy(result)
        bad['passed'] = False
        with self.assertRaisesRegex(ValueError, 'Both predeclared'):
            qualified_groups(plan, self.pool(plan), bad)
        with self.assertRaisesRegex(ValueError, 'min4'):
            qualified_groups(plan, {'passed': False, 'qualified_count': 3, 'qualified': []}, result)
        runtime.active_core = 'core-candidate'
        with patch('run_identity_coverage._target', return_value=('weight', weight)), \
                patch('run_identity_coverage.branch_off_generation') as model, self.assertRaises(ValueError):
            qualify_required_replay(runtime, plan)
        model.assert_not_called()

    def test_interrupted_mandatory_baseline_is_consumed_without_a_second_generation(self):
        plan, _, _ = self.build()
        with tempfile.TemporaryDirectory() as directory:
            with patch('run_identity_coverage.qualify_required_replay', side_effect=KeyboardInterrupt) as generation:
                with self.assertRaises(KeyboardInterrupt):
                    qualify_required_replay_once(None, plan, Path(directory), 'a' * 64)
                marker = Path(directory) / 'required-replay-baseline-started.json'
                before = marker.read_bytes()
                with self.assertRaisesRegex(ValueError, 'interrupted'):
                    qualify_required_replay_once(None, plan, Path(directory), 'a' * 64)
                self.assertEqual(marker.read_bytes(), before)
                self.assertEqual(generation.call_count, 1)

    def test_actual_baseline_wrong_case_truncation_branch_on_or_historical_ids_reject(self):
        plan, _, _ = self.build()
        weight = torch.eye(2)
        plan['original_tensor_sha256'] = tensor_digest(weight)
        runtime = SimpleNamespace(identity=plan['base_identity'], active_core=None)
        rows = [row for row in plan['groups_before_original_baseline']['replay_training']
                if row['id'] in plan['required_replay_training_ids']]
        for change in ('case', 'eos', 'branch', 'id'):
            observed = [provenance(qualified_row(row))['original_baseline_generation'] for row in rows]
            first = observed[0]
            first[{'case': 'answer', 'eos': 'complete', 'branch': 'branch_enabled', 'id': 'id'}[change]] = {
                'case': 'Olive_path', 'eos': False, 'branch': True, 'id': 'historical-validation'}[change]
            with patch('run_identity_coverage._target', return_value=('weight', weight)), \
                    patch('run_identity_coverage.branch_off_generation', return_value=observed):
                if change == 'id':
                    with self.assertRaisesRegex(ValueError, 'coverage'):
                        qualify_required_replay(runtime, plan)
                else:
                    result = qualify_required_replay(runtime, plan)
                    self.assertFalse(result['passed'])
                    with self.assertRaises(ValueError):
                        qualified_groups(plan, self.pool(plan), result)

    def test_normalized_whole_string_flag_still_rejects_case_expected_type_binding_or_eos_errors(self):
        material = self.material()
        groups = material['groups']
        runtime = SimpleNamespace(identity='fixture-base')
        self.assertEqual(len(_training_replay_generation(runtime, groups)), 6)
        self.assertTrue(evaluate_generation('ＰＹＴＯＲＣＨ\n', {'type': 'semantic',
            'validator': 'normalized_exact', 'expected': 'PYTORCH'}, True)['passed'])
        self.assertFalse(evaluate_generation('PYTorch', {'type': 'semantic',
            'validator': 'normalized_exact', 'expected': 'PYTORCH'}, True)['passed'])
        for change in ('case', 'expected', 'contains', 'eos', 'identity'):
            broken = copy.deepcopy(groups)
            row = next(row for row in broken['replay_training'] if row['id'] == 'preservation-replay-train-pytorch')
            value = provenance(row)
            if change == 'case': value['original_baseline_generation']['answer'] = 'PYTorch'
            elif change == 'expected': value['generation_check']['expected'] = 'PYTorch'
            elif change == 'contains': value['generation_check'] = {'type': 'contains', 'expected': row['answer']}
            elif change == 'eos': value['original_baseline_generation']['complete'] = False
            else: value['original_baseline_verified_identity'] = 'other-base'
            row['provenance'] = json.dumps(value)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, '正确、完整'):
                _training_replay_generation(runtime, broken)

    def test_exact37_publisher_rejects_changed_gold_context_check_false_human_claim_or_extra_archive(self):
        validate_material_publication(self.material(), 'fixture-base')
        for change in ('gold', 'context', 'check', 'human', 'extra', 'archive', 'baseline'):
            material = self.material()
            row = next(row for row in material['groups']['replay_training'] if row['id'] == 'preservation-replay-train-pytorch')
            value = provenance(row)
            if change == 'gold': row['answer'] = 'PYTorch'
            elif change == 'context': value['messages'][-1]['content'] += '改写'
            elif change == 'check': value['generation_check']['max_new_tokens'] = 193
            elif change == 'human': value['human_verified'] = True
            elif change == 'extra': row['id'] = 'unregistered'
            elif change == 'archive': material['superseded_ids'].append('old-replay-train-0')
            else: value['original_baseline_generation']['id'] = 'historical-dev-id'
            row['provenance'] = json.dumps(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_material_publication(material, 'fixture-base')

    def test_real_37_publication_journal_upgrade_count_and_recovery_idempotence(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = Memory(Path(self.temp.name))
        self.plan = self.material()
        test_core_publication.CorePublicationTests.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it(self)
        with self.memory.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM core_material_archives').fetchone()[0], 13)

    def test_all49_readback_protection_and_late_sql_failure_preserves_original103(self):
        material = self.material()
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimpleNamespace(store=Path(directory), identity='fixture-base', active_core=material['version'])
            save_core_retention(runtime, material['version'], material['groups'], regression_samples=material['retention_regressions'])
            self.assertEqual(len(protected_validation(runtime)), 49)
            self.assertEqual(read_core_retention(runtime)['groups'], material['groups'])
            memory = Memory(Path(directory))
            with memory.connect() as db:
                for row in material['source_rows']:
                    db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)', tuple(row.values()))
            before = memory.samples()
            with self.assertRaises(RuntimeError):
                with memory.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    apply_material_publication(db, material, 'fixture-base')
                    raise RuntimeError('late pointer failure')
            self.assertEqual(memory.samples(), before)
            with memory.connect() as db:
                self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='core_material_archives'").fetchone())


if __name__ == '__main__':
    unittest.main()
