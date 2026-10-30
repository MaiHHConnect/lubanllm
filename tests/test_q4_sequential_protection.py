"""CPU checks of frozen TRAIN coverage and per-case sequential Q4 protection."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from q4_sequential_protection import (GENERATION, GROUPS, check_proposal,
                                      establish_baseline, protection_cases)


class SequentialProtectionTests(unittest.TestCase):
    def setUp(self):
        self.cases = protection_cases()
        self.good = [{'id': case['id'], 'text': (case['check']['expected']
                     if isinstance(case['check']['expected'], str) else
                     json.dumps(case['check']['expected'], ensure_ascii=False)),
                     'complete': True, 'tokens': [100+i, 248044]}
                     for i, case in enumerate(self.cases)]

    def test_frozen_sixteen_train_cases_and_finite_gold_checks(self):
        self.assertEqual(len(self.cases), 16)
        self.assertEqual(len({c['id'] for c in self.cases}), 16)
        self.assertEqual(Counter(c['group'] for c in self.cases), {group: 4 for group in GROUPS})
        self.assertTrue(all(c['split'] == 'train' for c in self.cases))
        baseline = establish_baseline(self.cases, self.good)
        self.assertEqual(len(baseline['correct_ids']), 16)
        self.assertTrue(all(v == {'cases': 4, 'correct': 4, 'incomplete': 0} for v in baseline['coverage'].values()))
        self.assertEqual(baseline['generation'], GENERATION)
        # Material generation is deterministic and every call owns its nested values.
        self.cases[8]['check']['expected']['count'] = 99
        self.assertEqual(protection_cases()[8]['check']['expected']['count'], 4)

    def test_baseline_errors_retained_and_equal_total_does_not_offset_loss(self):
        before = deepcopy(self.good); before[9]['text'] = 'invalid'
        baseline = establish_baseline(self.cases, before)
        self.assertEqual(len(baseline['generations']), 16)
        self.assertEqual(len(baseline['scored']), 16)
        self.assertEqual(len(baseline['correct_ids']), 15)
        candidate = deepcopy(self.good); candidate[0]['text'] = 'wrong'
        report = check_proposal(self.cases, baseline, before, candidate)
        self.assertFalse(report['accepted'])
        self.assertEqual(len(report['candidate_correct_ids']), 15)
        self.assertEqual(report['lost_original_correct_ids'], [self.good[0]['id']])
        self.assertEqual(report['gained_vs_original_ids'], [self.good[9]['id']])
        self.assertEqual(report['rows'][0]['candidate']['text'], 'wrong')
        self.assertEqual(report['decision'], 'reject_restore_proposed_layer_bf16')

    def test_new_correct_in_accepted_state_is_protected_next_time(self):
        before = deepcopy(self.good); before[9]['text'] = 'invalid'
        baseline = establish_baseline(self.cases, before)
        first = check_proposal(self.cases, baseline, before, self.good)
        self.assertTrue(first['accepted'])
        next_report = check_proposal(self.cases, baseline, self.good, before)
        self.assertFalse(next_report['accepted'])
        self.assertEqual(next_report['lost_original_correct_ids'], [])
        self.assertEqual(next_report['lost_accepted_correct_ids'], [self.good[9]['id']])
        self.assertEqual(next_report['required_correct_ids'], sorted(c['id'] for c in self.cases))

    def test_incomplete_exact_text_fails_and_does_not_filter_case(self):
        baseline = establish_baseline(self.cases, self.good)
        candidate = deepcopy(self.good); candidate[3]['complete'] = False
        report = check_proposal(self.cases, baseline, self.good, candidate)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['lost_required_correct_ids'], [self.good[3]['id']])
        self.assertEqual(report['coverage']['candidate']['characters']['incomplete'], 1)
        self.assertEqual(len(report['rows']), 16)

    def test_json_uses_existing_strict_parser(self):
        baseline = establish_baseline(self.cases, self.good)
        candidate = deepcopy(self.good)
        candidate[8]['text'] = '{"count":4,"enabled":false,"tag":"saffron"}'
        self.assertTrue(check_proposal(self.cases, baseline, self.good, candidate)['accepted'])
        for text in ('{"tag":"saffron","enabled":false,"count":4,"count":4}',
                     '{"tag":"saffron","enabled":0,"count":4}',
                     '{"tag":"saffron","enabled":false,"count":NaN}'):
            candidate[8]['text'] = text
            with self.subTest(text=text):
                self.assertFalse(check_proposal(self.cases, baseline, self.good, candidate)['accepted'])

    def test_missing_duplicate_foreign_or_malformed_outputs_rejected(self):
        for variant in ('missing', 'duplicate', 'foreign', 'complete_type', 'text_type', 'empty_tokens',
                        'bool_token', 'negative_token', 'float_token', 'too_many_tokens'):
            rows = deepcopy(self.good)
            if variant == 'missing': rows.pop()
            elif variant == 'duplicate': rows[-1]['id'] = rows[0]['id']
            elif variant == 'foreign': rows[-1]['id'] = 'external-holdout'
            elif variant == 'complete_type': rows[0]['complete'] = 1
            elif variant == 'text_type': rows[0]['text'] = 42
            else:
                rows[0]['tokens'] = {'empty_tokens': [], 'bool_token': [True], 'negative_token': [-1],
                    'float_token': [1.5], 'too_many_tokens': [1]*193}[variant]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                establish_baseline(self.cases, rows)
        reordered = establish_baseline(self.cases, list(reversed(self.good)))
        self.assertEqual([r['id'] for r in reordered['scored']], [c['id'] for c in self.cases])

    def test_empty_baseline_corrupted_accepted_and_tampered_protocol_rejected(self):
        empty = [{**row, 'text': 'invalid'} for row in self.good]
        with self.assertRaises(ValueError): establish_baseline(self.cases, empty)
        baseline = establish_baseline(self.cases, self.good)
        bad_accepted = deepcopy(self.good); bad_accepted[2]['text'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'accepted state already lost'):
            check_proposal(self.cases, baseline, bad_accepted, self.good)
        for change in ('correct_ids', 'generation', 'cases'):
            broken = deepcopy(baseline)
            if change == 'correct_ids': broken['correct_ids'].pop()
            elif change == 'generation': broken['generation']['temperature'] = .6
            else: broken['cases'][0]['messages'][0]['content'] = 'changed'
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_proposal(self.cases, broken, self.good, self.good)
        with self.assertRaises(ValueError): establish_baseline(self.cases[:12], self.good[:12])

    def test_scores_recomputed_no_mutation_and_no_token_equality_gate(self):
        baseline = establish_baseline(self.cases, self.good)
        candidate = deepcopy(self.good)
        for row in candidate:
            row['score'] = {'passed': False}
            row['tokens'] = [700, 701, 248044]
        saved = deepcopy((self.cases, baseline, self.good, candidate))
        report = check_proposal(self.cases, baseline, self.good, candidate)
        self.assertTrue(report['accepted'])
        self.assertEqual((self.cases, baseline, self.good, candidate), saved)
        report['rows'][8]['candidate']['score']['expected']['count'] = 1000
        self.assertEqual(self.cases[8]['check']['expected']['count'], 4)
        candidate[0]['text'] = 'incorrect'
        candidate[0]['score'] = {'passed': True}
        self.assertFalse(check_proposal(self.cases, baseline, self.good, candidate)['accepted'])


if __name__ == '__main__':
    unittest.main()
