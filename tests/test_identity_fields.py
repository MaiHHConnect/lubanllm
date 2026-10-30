"""Fixed57/832 and all71 protection; tiny CPU/SQLite fixtures, no real model."""
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
from prepare_identity_fields import PROPOSAL, PROPOSAL_SHA, field_plan
from run_identity_coverage import (ARGUMENTS, RELATION_ARGUMENTS, FIELD_ARGUMENTS, qualified_groups,
                                  validate_protocol_policy, verify_relation_recovery)
from verify_luban0_final import digest
import test_backups
import test_core_publication
from test_identity_relations import relation_fixture
from test_identity_preservation import qualified_row
from luban.backups import save_training_backup, restore_training_backup
from luban.core_publication import (FIELD_DATASET, FIELD_TRAIN_IDS, FIELD_FRESH_DEV_IDS, content_sha,
                                    validate_field_teaching, validate_material_publication, apply_material_publication)
from luban.evolution import FIELD_TRAIN_CHECKPOINTS
from luban.memory import Memory
from luban.retention import provenance, save_core_retention, read_core_retention, protected_validation


def field_fixture():
    proposal = json.loads(PROPOSAL.read_text())
    old, _, _, _ = relation_fixture()
    old['source_rows'] = copy.deepcopy(proposal['original_source_rows'])
    failed = {'relation-fresh-dev-' + key for key in ('api-ordinary', 'name', 'position', 'tuner')}
    result = {'passed': False, 'engineering': [], 'training': {'promoted': False,
        'new_generation': [{'id': row['id'], 'passed': row['id'] not in failed}
                           for row in old['groups_before_original_baseline']['new_validation']]},
        'retirement': {'version': None, 'restored': True, 'tensor_sha256': old['original_tensor_sha256']}}
    plan = field_plan(old, result, proposal, prepared=max(row['created'] for row in proposal['groups']['new_training']) + 1,
                      session_id='123456abcdef', directory=Path('/unused'))
    return plan, old, result, proposal


def field_material():
    plan, old, _, _ = field_fixture()
    identity = plan['base_identity']
    required = [qualified_row(row, identity) for row in plan['groups_before_original_baseline']['replay_training']
                if row['id'] in plan['required_replay_training_ids']]
    pool = []
    for source in plan['replay_training_candidates'][:4]:
        row = copy.deepcopy(source)
        row['approved'] = 1
        pool.append(qualified_row(row, identity))
    groups = qualified_groups(plan, {'passed': True, 'qualified_count': 4, 'qualified': pool},
                              {'passed': True, 'qualified': required})
    return {'format': 1, 'identity': identity, 'run_id': plan['session_id'], 'version': 'core-' + plan['session_id'],
        'groups': groups, 'retention_regressions': plan['retention_regressions'], 'source_rows': old['source_rows'],
        'source_sha256': content_sha(old['source_rows']), 'superseded_ids': plan['superseded_source_ids'],
        'acceptance': {'protocol_sha256': 'a' * 64, 'report_path': 'coverage-acceptance-' + plan['session_id'] + '.json',
            'report_sha256': 'b' * 64, 'gates': dict.fromkeys(test_core_publication.ACCEPTANCE_GATES, True), 'teacher_calls': 0}}


class IdentityFieldsTests(unittest.TestCase):
    def test_preparation_keeps_all_original_records_rights_and_only_exact13_plus8(self):
        plan, old, _, proposal = field_fixture()
        self.assertEqual(digest(PROPOSAL), PROPOSAL_SHA)
        groups = plan['groups_before_original_baseline']
        self.assertTrue(validate_field_teaching(groups, plan['retention_regressions']))
        self.assertEqual(plan['arguments'], FIELD_ARGUMENTS)
        self.assertEqual(ARGUMENTS['steps'], 512)
        self.assertEqual(RELATION_ARGUMENTS['steps'], 640)
        self.assertEqual(plan['training_checkpoints'], list(FIELD_TRAIN_CHECKPOINTS))
        for key in groups:
            self.assertEqual([row for row in groups[key] if not row['id'].startswith('field-')],
                             old['groups_before_original_baseline'][key])
        for key in ('source_rows', 'source_groups', 'retention_regressions', 'engineering_cases', 'superseded_source_ids'):
            self.assertEqual(plan[key], old[key])
        self.assertEqual(len(plan['source_rows']), 103)
        self.assertEqual(len(plan['superseded_source_ids']), 13)
        self.assertEqual(len(groups['new_validation']) + len(groups['replay_validation']) + len(plan['retention_regressions']), 71)
        self.assertTrue(all(provenance(row)['human_verified'] is False and provenance(row)['program_verified'] is True
                            for row in groups['new_training'] + groups['new_validation'] if row['id'].startswith('field-')))
        self.assertFalse(plan['sealed'])

    def test_832_and640_are_not_interchangeable_and_explicit71_once_original_budget_is_fixed(self):
        plan, old, _, _ = field_fixture()
        self.assertEqual(validate_protocol_policy(plan), FIELD_ARGUMENTS)
        for mutation in ('dataset', 'budget', 'checkpoints', 'count', 'engineering', 'known', 'warmstart', 'retry'):
            changed = copy.deepcopy(plan)
            if mutation == 'dataset': changed['relation_dataset'] = 'arbitrary57'
            elif mutation == 'budget': changed['arguments'] = RELATION_ARGUMENTS
            elif mutation == 'checkpoints': changed['training_checkpoints'].pop()
            elif mutation == 'count': changed['protection_count'] = 63
            elif mutation == 'engineering': changed['engineering_cases'].pop()
            elif mutation == 'known': changed['old_known_development_regressions'].pop()
            elif mutation == 'warmstart': changed['warm_start_candidate'] = True
            else: changed['maximum_attempts'] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_protocol_policy(changed)
        old.update(relation_dataset=FIELD_DATASET, arguments=FIELD_ARGUMENTS, training_checkpoints=list(FIELD_TRAIN_CHECKPOINTS))
        with self.assertRaises(ValueError):
            validate_protocol_policy(old)

    def test_shared_guard_and_publisher_reject_source_gold_rights_replay_or_archive_mutations_before_sql(self):
        material = field_material()
        self.assertTrue(validate_field_teaching(material['groups'], material['retention_regressions'], qualified=True))
        validate_material_publication(material, material['identity'])
        for mutation in ('gold', 'context', 'human', 'program', 'source', 'extra', 'missing_dev', 'old_rights',
                         'replay_gold', 'pool_min', 'pool_created', 'mandatory_flag', 'mandatory_eos', 'reference', 'archive'):
            changed = copy.deepcopy(material)
            row = next(row for row in changed['groups']['new_training'] if row['id'] in FIELD_TRAIN_IDS)
            flags = provenance(row)
            if mutation == 'gold': row['answer'] += 'wrong'
            elif mutation == 'context': flags['messages'][-1]['content'] += 'wrong'
            elif mutation == 'human': flags['human_verified'] = True
            elif mutation == 'program': flags['program_verified'] = False
            elif mutation == 'source': flags['original_development_record_sha256'] = 'b' * 64
            elif mutation == 'extra': row['id'] = 'arbitrary-teaching'
            elif mutation == 'missing_dev': changed['groups']['new_validation'].pop()
            elif mutation == 'old_rights': changed['retention_regressions'][0]['training_allowed'] = True
            elif mutation == 'replay_gold': changed['groups']['replay_training'][0]['answer'] = 'wrong'
            elif mutation in {'pool_min', 'pool_created'}:
                pool = next(r for r in changed['groups']['replay_training'] if r['id'].startswith('coverage-en-replay-candidate-'))
                if mutation == 'pool_min': changed['groups']['replay_training'].remove(pool)
                else: pool['created'] = max(r['created'] for r in material['groups']['new_training']) - 1
            elif mutation == 'archive': changed['superseded_ids'].append('arbitrary')
            else:
                mandatory = next(r for r in changed['groups']['replay_training'] if r['id'] == 'preservation-replay-train-olive-path')
                value = provenance(mandatory)
                if mutation == 'mandatory_flag': value['training_generation_gate'] = False
                elif mutation == 'mandatory_eos': value['original_baseline_generation']['complete'] = False
                else: value['original_baseline_generation']['reference_used'] = True
                mandatory['provenance'] = json.dumps(value, ensure_ascii=False)
            row['provenance'] = json.dumps(flags, ensure_ascii=False)
            with tempfile.TemporaryDirectory() as directory:
                memory = Memory(Path(directory))
                with memory.connect() as db:
                    for source in material['source_rows']:
                        db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)', tuple(source.values()))
                before = memory.samples()
                with self.subTest(mutation=mutation), self.assertRaises(ValueError), memory.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    apply_material_publication(db, changed, changed['identity'])
                self.assertEqual(memory.samples(), before)

    def test_native_reload_actually_calls_all71_and_any_wrong_eos_reference_or_missing_refuses_publication(self):
        plan, _, _, _ = field_fixture()
        groups = plan['groups_before_original_baseline']
        rows = groups['new_validation'] + groups['replay_validation'] + plan['retention_regressions']
        fixture = [{'id': row['id'], 'answer': row['answer'], 'passed': True, 'complete': True,
                    'branch_enabled': False, 'reference_used': False} for row in rows]
        for mutation in ('pass', 'gold', 'eos', 'branch', 'reference', 'missing'):
            generated = copy.deepcopy(fixture)
            report = {'recovery': {'passed': True}}
            if mutation == 'missing': generated.pop()
            elif mutation != 'pass': generated[-1][{'gold': 'passed', 'eos': 'complete', 'branch': 'branch_enabled',
                'reference': 'reference_used'}[mutation]] = mutation in {'branch', 'reference'}
            with patch('run_identity_coverage.branch_off_generation', return_value=generated) as actual:
                if mutation == 'pass': verify_relation_recovery(None, plan, rows, report)
                else:
                    with self.assertRaisesRegex(ValueError, 'fixed71'):
                        verify_relation_recovery(None, plan, rows, report)
                    self.assertFalse(report['recovery']['passed'])
                actual.assert_called_once_with(None, rows)
                self.assertEqual(len(report['recovery']['full_protection_replay']), len(generated))

    def test_actual57_sqlite_publication_matches_upgrade_count_and_idempotent_recovery(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = Memory(Path(self.temp.name))
        self.plan = field_material()
        test_core_publication.CorePublicationTests.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it(self)
        with self.memory.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM core_material_archives').fetchone()[0], 13)

    def test_actual_backup_restores_all71_records_and_tiny_tensor_before_generation(self):
        material = field_material()
        fixture = test_backups.BackupTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.identity = material['identity']
        expected_tensor = fixture.core_state()
        version = 'core-111111111111'
        runtime = SimpleNamespace(store=fixture.store, identity=fixture.identity, active_core=version)
        saved = save_core_retention(runtime, version, material['groups'], regression_samples=material['retention_regressions'])
        metadata = fixture.store / (version + '.json')
        record = json.loads(metadata.read_text())
        record['retention_sha256'] = saved['file_sha256']
        metadata.write_text(json.dumps(record))
        before, protection = read_core_retention(runtime), protected_validation(runtime)
        self.assertEqual(len(protection), 71)
        backup = save_training_backup(fixture.store, fixture.identity, 'learn')
        (fixture.store / (version + '.retention.json')).write_bytes(b'corrupt')
        fixture.core_state('core-222222222222', marker=100)
        restore_training_backup(fixture.store, fixture.identity, backup)
        actual_runtime, actual_tensor = fixture.reload_core()
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
        self.assertEqual(read_core_retention(actual_runtime), before)
        self.assertEqual(protected_validation(actual_runtime), protection)
        self.assertEqual(digest(fixture.store / (version + '.retention.json')), saved['file_sha256'])


if __name__ == '__main__':
    unittest.main()
