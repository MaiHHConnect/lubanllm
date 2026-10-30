"""Exposed targets become honest TRAIN copies, with original readonly obligations."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_identity_known_correction import CORRECTIONS, corrected_plan
from run_identity_coverage import ARGUMENTS
from test_core_publication import fixture_plan, sample
from luban.core_publication import content_sha, validate_material_publication
from luban.memory import Memory
from luban.retention import (messages, provenance, protected_validation, read_core_retention,
                            save_core_retention, validate_retention_conflicts)


def correction_fixture():
    material = fixture_plan()
    groups = copy.deepcopy(material['groups'])
    groups['replay_training'] = groups['replay_training'][:44]
    old_rows, regressions = material['source_rows'], material['retention_regressions']
    for index, (old_id, word, _) in enumerate(CORRECTIONS):
        for row in (old_rows[34 + index], regressions[index]):
            row['id'] = old_id
            row['prompt'] = 'Uppercase ' + word
            row['answer'] = word.upper()
            value = provenance(row)
            value['messages'] = [{'role': 'system', 'content': 'Complete the user task'},
                                 {'role': 'user', 'content': row['prompt']}]
            value['generation_check'] = {'type': 'semantic', 'validator': 'normalized_exact',
                                         'expected': word.upper(), 'max_new_tokens': 192}
            row['provenance'] = json.dumps(value)
    plan = {'groups_before_original_baseline': groups, 'source_rows': old_rows,
        'source_groups': {}, 'base_identity': 'fixture-base', 'arguments': ARGUMENTS,
        'original_tensor_sha256': 'a' * 64, 'minimum_replay_candidates': 4,
        'replay_training_candidates': [sample('pool-' + str(i), 'replay', 'training') for i in range(12)],
        'retention_regressions': regressions, 'old_known_development_regressions': regressions[:8],
        'engineering_cases': regressions[8:], 'replay_candidate_sha256': 'b' * 64}
    result = {'passed': False, 'engineering': [], 'retirement': {'version': None, 'restored': True,
        'tensor_sha256': 'a' * 64}, 'known_regression': [
            {'id': old_id, 'answer': 'wrong', 'passed': False, 'complete': True} for old_id, _, _ in CORRECTIONS]}
    return plan, result, material


class KnownCorrectionTests(unittest.TestCase):
    def build(self, old, result):
        return corrected_plan(old, result, prepared=2., session_id='abcdef123456', directory=Path('/unused'))

    def test_only_two_known_train_copies_added_original_anchors_and_all_obligations_unchanged(self):
        old, result, _ = correction_fixture()
        before = copy.deepcopy(old)
        plan = self.build(old, result)
        self.assertEqual(old, before)
        self.assertEqual({key: len(rows) for key, rows in plan['groups_before_original_baseline'].items()},
                         {'new_training': 36, 'new_validation': 8, 'replay_training': 44, 'replay_validation': 17})
        self.assertEqual(plan['source_rows'], old['source_rows'])
        self.assertEqual(plan['retention_regressions'], old['retention_regressions'])
        self.assertEqual(plan['replay_training_candidates'], old['replay_training_candidates'])
        originals = {row['id']: row for row in old['source_rows']}
        for row, (old_id, word, new_id) in zip(plan['correction_training_rows'], CORRECTIONS):
            self.assertEqual(row['id'], new_id)
            self.assertEqual(messages(row), messages(originals[old_id]))
            self.assertEqual(row['answer'], word.upper())
            self.assertEqual(provenance(row)['generation_check'], provenance(originals[old_id])['generation_check'])
            self.assertEqual(provenance(row)['exposed_old_validation_id'], old_id)
            self.assertEqual(row['purpose'], 'training')
            self.assertTrue(provenance(row)['program_verified'])
            self.assertIs(provenance(row)['human_verified'], False)
        self.assertFalse(plan['warm_start_candidate'])
        self.assertIsNone(plan['initial_core_version'])

    def test_wrong_gold_context_or_nonretired_failure_cannot_authorize_correction(self):
        for change in ('gold', 'context', 'retirement'):
            old, result, _ = correction_fixture()
            if change == 'gold':
                old['source_rows'][34]['answer'] = 'TOKYO_WRONG'
            elif change == 'context':
                value = provenance(old['old_known_development_regressions'][0])
                value['messages'][1]['content'] = 'Different question'
                old['old_known_development_regressions'][0]['provenance'] = json.dumps(value)
            else:
                result['retirement']['restored'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build(old, result)

    def test_new_stable_id_conflict_rejects_before_any_old_plan_mutation(self):
        old, result, _ = correction_fixture()
        old['replay_training_candidates'][0]['id'] = CORRECTIONS[0][2]
        before = copy.deepcopy(old)
        with self.assertRaisesRegex(ValueError, 'stable ID conflicts'):
            self.build(old, result)
        self.assertEqual(old, before)

    def test_real_retention_readback_preserves_49_and_compatible_train_readonly_context(self):
        old, result, _ = correction_fixture()
        plan = self.build(old, result)
        with tempfile.TemporaryDirectory() as directory:
            runtime = SimpleNamespace(store=Path(directory), identity='fixture-base', active_core='core-abcdef123456')
            save_core_retention(runtime, runtime.active_core, plan['groups_before_original_baseline'],
                                regression_samples=plan['retention_regressions'])
            snapshot = read_core_retention(runtime)
            self.assertEqual(snapshot['regression_validation'], old['retention_regressions'])
            self.assertEqual(len(protected_validation(runtime)), 49)
            validate_retention_conflicts(snapshot, plan['correction_training_rows'])
            wrong = copy.deepcopy(plan['correction_training_rows'])
            wrong[0]['answer'] = 'TOKYO_WRONG'
            with self.assertRaises(ValueError):
                validate_retention_conflicts(snapshot, wrong)

    def publication_fixture(self):
        old, result, material = correction_fixture()
        plan = self.build(old, result)
        material['groups'] = plan['groups_before_original_baseline']
        material['groups']['replay_training'] += [sample('qualified-' + str(i), 'replay', 'training') for i in range(4)]
        material['source_rows'] = old['source_rows']
        material['source_sha256'] = content_sha(material['source_rows'])
        material['retention_regressions'] = old['retention_regressions']
        material['superseded_ids'] = sorted(set(material['superseded_ids']) - {'old-new-valid-0', 'old-new-valid-1'}
                                          | {row[0] for row in CORRECTIONS})
        return material

    def test_actual_publication_validator_accepts_36_only_with_exact_two_exposed_targets(self):
        material = self.publication_fixture()
        _, incoming = validate_material_publication(material, 'fixture-base')
        self.assertTrue({row[2] for row in CORRECTIONS} <= incoming.keys())

    def test_36_publisher_rejects_changed_gold_context_check_or_arbitrary_extra(self):
        for change in ('gold', 'context', 'check', 'extra', 'unverified_human_claim'):
            material = self.publication_fixture()
            row = next(row for row in material['groups']['new_training'] if row['id'] == CORRECTIONS[0][2])
            value = provenance(row)
            if change == 'gold':
                row['answer'] = 'WRONG'
            elif change == 'context':
                value['messages'][1]['content'] += ' modified'
            elif change == 'check':
                value['generation_check']['max_new_tokens'] = 193
            elif change == 'extra':
                row['id'] = 'arbitrary-extra'
            else:
                value['human_verified'] = True
            row['provenance'] = json.dumps(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_material_publication(material, 'fixture-base')

    def test_real_36_publisher_journal_upgrade_count_and_recovery_idempotence(self):
        from test_core_publication import CorePublicationTests
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = Memory(Path(self.temp.name))
        self.plan = self.publication_fixture()
        # Reuse the actual SQLite/artifact/pointer publication and upgrade-reader
        # integration, with the36 corpus and both same-context readonly targets.
        CorePublicationTests.test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it(self)


if __name__ == '__main__':
    unittest.main()
