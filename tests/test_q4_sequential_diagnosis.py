import copy
import unittest

from scripts.diagnose_q4_sequential import diagnostic_cases


class SequentialDiagnosisTests(unittest.TestCase):
    def material(self):
        result = {'completed': True, 'results': {'gptq-sequential': {'quality': {
            'holdout-t0': {'lost_baseline_correct': ['case']}}}}}
        fixture = {'id': 'case', 'check': {'type': 'exact', 'expected': 'yes'}, 'max_tokens': 16}
        rows = [{'label': label, 'phase': 'holdout', 'case_id': 'case', 'temperature': 0,
                 'prompt_tokens': [1, 2], 'tokens': tokens, 'complete': True,
                 'execution_action': 'local_core', 'text': text}
                for label, tokens, text in [('bf16', [3, 4, 9], 'yes'), ('gptq-sequential', [3, 5, 9], 'no')]]
        fixtures = [{'phase': 'holdout', 'fixture': fixture, 'temperature': 0}]
        return result, rows, fixtures

    def test_shared_prefix_and_nominated_tokens_are_from_raw_outputs(self):
        args = self.material()
        before = copy.deepcopy(args)
        specs = diagnostic_cases(*args, [9])
        self.assertEqual(specs[0]['forced_tokens'], [3])
        self.assertEqual(specs[0]['probe_tokens'], [4, 5])
        self.assertEqual(specs[0]['case']['tokens'], [1, 2])
        self.assertEqual(args, before)

    def test_rejects_duplicate_changed_prompt_or_incomplete_output(self):
        for mutation in ('duplicate', 'prompt', 'incomplete', 'expert', 'score'):
            result, rows, fixtures = self.material()
            if mutation == 'duplicate': rows.append(rows[0])
            elif mutation == 'prompt': rows[1]['prompt_tokens'] = [2, 1]
            elif mutation == 'incomplete': rows[1]['complete'] = False
            elif mutation == 'expert': rows[1]['execution_action'] = 'neural_expert'
            else: rows[1]['text'] = 'yes'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                diagnostic_cases(result, rows, fixtures, [9])

    def test_rejects_stochastic_or_unfinished_evidence(self):
        for mutation in ('stochastic', 'unfinished'):
            result, rows, fixtures = self.material()
            if mutation == 'unfinished': result['completed'] = False
            else:
                group = result['results']['gptq-sequential']['quality']
                group['holdout-t0.6'] = group.pop('holdout-t0')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                diagnostic_cases(result, rows, fixtures, [9])


if __name__ == '__main__':
    unittest.main()
