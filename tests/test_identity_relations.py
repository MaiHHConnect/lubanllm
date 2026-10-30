"""Exact relation44, journal/material rollback and full63 recovery; CPU only."""
import copy
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_identity_relations import PROPOSAL, PROPOSAL_SHA, relation_plan
from run_identity_coverage import (ARGUMENTS, RELATION_ARGUMENTS, qualified_groups, validate_protocol_policy,
                                  verify_relation_recovery)
from unittest.mock import patch
from verify_luban0_final import digest
import test_core_publication
from test_core_publication import sample, ACCEPTANCE_GATES
from test_identity_preservation import qualified_row
import test_backups
from luban.backups import save_training_backup, restore_training_backup
from luban.core_publication import (RELATION_DATASET, RELATION_REUSE_IDS, RELATION_ATTRIBUTE_IDS,
    RELATION_ENGINEERING_IDS, SUPERSEDED_DUPLICATES, content_sha,
    validate_relation_teaching, validate_material_publication, apply_material_publication)
from luban.evolution import TRAIN_CHECKPOINTS, RELATION_TRAIN_CHECKPOINTS
from luban.memory import Memory
from luban.retention import provenance, save_core_retention, read_core_retention, protected_validation


def relation_fixture():
    """Repo-fixed records; original103 ledger is a local SQLite test fixture."""
    proposal = json.loads(PROPOSAL.read_text())
    groups, readonly = copy.deepcopy(proposal['groups']), copy.deepcopy(proposal['regression_validation'])
    identity = proposal['base_identity']
    old_groups = copy.deepcopy(groups)
    old_groups['new_training'] = [row for row in groups['new_training'] if not row['id'].startswith('relation-')]
    old_groups['new_validation'] = [row for row in groups['new_validation'] if not row['id'].startswith('relation-')]
    old_readonly = [row for row in readonly if row['id'] not in RELATION_ENGINEERING_IDS]
    engineering = [row for row in old_readonly if row['id'].startswith('v6-engineering-')]
    source = [copy.deepcopy(row) for rows in old_groups.values() for row in rows
              if re.fullmatch('[a-f0-9]{32}', row['id'])]
    for row in old_readonly:
        if re.fullmatch('[a-f0-9]{32}', row['id']):
            row = copy.deepcopy(row)
            row['training_allowed'] = True
            flags = provenance(row)
            flags['evolution_role'] = 'replay' if row['id'] == '76dbee81c27a4668b7501e7c2a8bb5e4' else 'new'
            flags.pop('retention_generation_only', None)
            row['provenance'] = json.dumps(flags, ensure_ascii=False)
            source.append(row)
    for identifier in sorted(SUPERSEDED_DUPLICATES):
        row = sample(identifier, 'new', 'training')
        flags = provenance(row)
        flags['base_identity'] = identity
        row['provenance'] = json.dumps(flags)
        source.append(row)
    archived = sorted(SUPERSEDED_DUPLICATES | {row['id'] for row in source if row['purpose'] == 'validation'
        and provenance(row)['evolution_role'] == 'new'} | {'76dbee81c27a4668b7501e7c2a8bb5e4'})
    old = {'base_identity': identity, 'source_rows': source, 'source_groups': copy.deepcopy(old_groups),
        'groups_before_original_baseline': old_groups, 'retention_regressions': old_readonly,
        'engineering_cases': engineering, 'replay_training_candidates': copy.deepcopy(proposal['fixed_replay_candidates']),
        'superseded_source_ids': archived, 'required_replay_training_ids': proposal['required_replay_training_ids'],
        'original_tensor_sha256': 'a' * 64, 'operator_audit': {'sample': source[0]}, 'arguments': ARGUMENTS,
        'training_checkpoints': list(TRAIN_CHECKPOINTS), 'minimum_replay_candidates': 4, 'maximum_attempts': 1}
    result = {'passed': False, 'engineering_complete': True,
        'engineering': [{'id': row['id'], 'passed': row['id'] not in RELATION_REUSE_IDS.values()} for row in engineering],
        'retirement': {'version': None, 'restored': True, 'tensor_sha256': 'a' * 64}}
    plan = relation_plan(old, result, proposal, prepared=max(row['created'] for row in groups['new_training']) + 1,
                         session_id='123456abcdef', directory=Path('/unused'))
    return plan, old, result, proposal


def relation_material():
    plan, old, _, _ = relation_fixture()
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
        'groups': groups, 'retention_regressions': plan['retention_regressions'],
        'source_rows': old['source_rows'], 'source_sha256': content_sha(old['source_rows']),
        'superseded_ids': plan['superseded_source_ids'],
        'acceptance': {'protocol_sha256': 'a' * 64, 'report_path': 'coverage-acceptance-' + plan['session_id'] + '.json',
            'report_sha256': 'b' * 64, 'gates': dict.fromkeys(ACCEPTANCE_GATES, True), 'teacher_calls': 0}}


class IdentityRelationsTests(unittest.TestCase):
    def test_fixed44_literal7_preserves_original37_old49_permissions_and_only_new14_obligations(self):
        plan, old, _, proposal = relation_fixture()
        self.assertEqual(digest(PROPOSAL), PROPOSAL_SHA)
        self.assertTrue(validate_relation_teaching(plan['groups_before_original_baseline'], plan['retention_regressions']))
        self.assertEqual(plan['source_rows'], old['source_rows'])
        self.assertEqual(len(plan['source_rows']), 103)
        self.assertEqual(len(plan['superseded_source_ids']), 13)
        self.assertEqual(plan['replay_training_candidates'], old['replay_training_candidates'])
        self.assertEqual(plan['arguments'], RELATION_ARGUMENTS)
        self.assertEqual(ARGUMENTS['steps'], 512)
        self.assertEqual(len(plan['engineering_cases']), 8)
        self.assertEqual(len(plan['old_known_development_regressions']), 27)
        self.assertEqual(sum(len(plan['groups_before_original_baseline'][k]) for k in ('new_validation', 'replay_validation'))
                         + len(plan['retention_regressions']), 63)
        self.assertTrue(all(not row['training_allowed'] for row in old['engineering_cases']))
        new = {row['id']: row for row in proposal['groups']['new_training']}
        for key in RELATION_ATTRIBUTE_IDS.keys() | RELATION_REUSE_IDS.keys():
            self.assertIs(provenance(new[key])['human_verified'], False)
            self.assertIs(provenance(new[key])['program_verified'], True)
        self.assertFalse(plan['sealed'])
        self.assertIsNone(plan['execution_manifest'])

    def test_only_exact_relation_protocol_can_request640_or_change_engineering_count(self):
        plan, _, _, _ = relation_fixture()
        self.assertEqual(validate_protocol_policy(plan), RELATION_ARGUMENTS)
        for mutation in ('dataset', 'budget', 'checkpoints', 'count', 'engineering', 'known', 'warmstart', 'retry'):
            changed = copy.deepcopy(plan)
            if mutation == 'dataset': changed['relation_dataset'] = 'arbitrary larger dataset'
            elif mutation == 'budget': changed['arguments']['steps'] = 672
            elif mutation == 'checkpoints': changed['training_checkpoints'].pop()
            elif mutation == 'count': changed['protection_count'] = 65
            elif mutation == 'engineering': changed['engineering_cases'].pop()
            elif mutation == 'known': changed['old_known_development_regressions'].pop()
            elif mutation == 'warmstart': changed['warm_start_candidate'] = True
            else: changed['maximum_attempts'] = 2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_protocol_policy(changed)

    def test_exact44_publisher_rejects_gold_context_human_old_permissions_missing_obligation_or_archive_before_sql(self):
        material = relation_material()
        validate_material_publication(material, material['identity'])
        for mutation in ('gold', 'context', 'human', 'program', 'old', 'permissions', 'missing', 'archive', 'extra'):
            changed = copy.deepcopy(material)
            row = next(row for row in changed['groups']['new_training'] if row['id'] in RELATION_ATTRIBUTE_IDS)
            flags = provenance(row)
            if mutation == 'gold': row['answer'] += 'wrong'
            elif mutation == 'context': flags['messages'][-1]['content'] += ' altered'
            elif mutation == 'human': flags['human_verified'] = True
            elif mutation == 'program': flags['program_verified'] = False
            elif mutation == 'old': changed['retention_regressions'][0]['answer'] += 'wrong'
            elif mutation == 'permissions': changed['retention_regressions'][8]['training_allowed'] = True
            elif mutation == 'missing': changed['retention_regressions'].pop()
            elif mutation == 'archive': changed['superseded_ids'].append('arbitrary')
            else: row['id'] = 'unapproved-extra-teaching'
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

    def test_shared640_requires_original46_fixed_pool_subset_and_every_mandatory_current_off_eos_proof(self):
        material = relation_material()
        self.assertTrue(validate_relation_teaching(material['groups'], material['retention_regressions'], qualified=True))
        for mutation in ('replay_gold', 'pool_id', 'pool_gold', 'pool_context', 'minimum', 'created',
                         'mandatory_flag', 'mandatory_eos', 'reference', 'baseline_identity'):
            changed = copy.deepcopy(material)
            replay = changed['groups']['replay_training']
            mandatory = next(row for row in replay if row['id'] == 'preservation-replay-train-olive-path')
            pool = next(row for row in replay if row['id'].startswith('coverage-en-replay-candidate-'))
            flags = provenance(mandatory)
            if mutation == 'replay_gold': next(row for row in replay if re.fullmatch('[a-f0-9]{32}', row['id']))['answer'] = 'ALTERED'
            elif mutation == 'pool_id': pool['id'] = 'arbitrary-candidate'
            elif mutation == 'pool_gold': pool['answer'] = 'wrong'
            elif mutation == 'pool_context':
                value = provenance(pool)
                value['messages'][-1]['content'] += ' altered'
                pool['provenance'] = json.dumps(value)
            elif mutation == 'minimum': replay.remove(pool)
            elif mutation == 'created': pool['created'] = 1.
            elif mutation == 'mandatory_flag': flags['training_generation_gate'] = False
            elif mutation == 'mandatory_eos': flags['original_baseline_generation']['complete'] = False
            elif mutation == 'reference': flags['original_baseline_generation']['reference_used'] = True
            else: flags['original_baseline_verified_identity'] = 'other-base'
            mandatory['provenance'] = json.dumps(flags, ensure_ascii=False)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_relation_teaching(changed['groups'], changed['retention_regressions'], qualified=True)
        draft = copy.deepcopy(material['groups'])
        with self.assertRaisesRegex(ValueError, '草案'):
            validate_relation_teaching(draft, material['retention_regressions'])

    def test_native_reload_reanswers_all63_and_one_failed_or_reference_case_refuses_publication(self):
        plan, _, _, _ = relation_fixture()
        rows = plan['groups_before_original_baseline']['new_validation'] + plan['groups_before_original_baseline']['replay_validation'] + plan['retention_regressions']
        baseline = [{'id': row['id'], 'answer': row['answer'], 'passed': True, 'complete': True,
                     'branch_enabled': False, 'reference_used': False} for row in rows]
        for mutation in ('pass', 'gold', 'eos', 'branch', 'reference', 'missing'):
            report = {'recovery': {'passed': True}}
            generated = copy.deepcopy(baseline)
            if mutation == 'missing': generated.pop()
            elif mutation != 'pass': generated[-1][{'gold': 'passed', 'eos': 'complete',
                'branch': 'branch_enabled', 'reference': 'reference_used'}[mutation]] = mutation in {'branch', 'reference'}
            with patch('run_identity_coverage.branch_off_generation', return_value=generated) as actual:
                if mutation == 'pass':
                    verify_relation_recovery(None, plan, rows, report)
                    self.assertTrue(report['recovery']['full_protection_replay_passed'])
                else:
                    with self.assertRaisesRegex(ValueError, 'fixed63'):
                        verify_relation_recovery(None, plan, rows, report)
                    self.assertFalse(report['recovery']['passed'])
                actual.assert_called_once_with(None, rows)
                self.assertEqual(len(report['recovery']['full_protection_replay']), len(generated))

    def test_actual44_publication_journal_report_matches_upgrade_counts_and_recover_is_idempotent(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = Memory(Path(self.temp.name))
        self.plan = relation_material()
        test_core_publication.CorePublicationTests.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it(self)
        with self.memory.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM core_material_archives').fetchone()[0], 13)

    def test_actual_backup_restores_all63_full_records_and_tensor_native_reload(self):
        material = relation_material()
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
        before = read_core_retention(runtime)
        protection = protected_validation(runtime)
        self.assertEqual(len(protection), 63)
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
