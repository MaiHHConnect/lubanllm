"""Loss-position and scope guards for the frozen gradient feasibility probe."""
import sys
from pathlib import Path
import unittest

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts'),
               str(Path(__file__).resolve().parents[1]/'src')]
from probe_q4_output_sensitivity import PROJECTIONS, gradient_summary, probe_case


class SensitivityProbeTests(unittest.TestCase):
    def record(self):
        return {'id': 'cal-00', 'split': 'calibration', 'prompt_tokens': [1, 2],
                'generated_tokens': [3, 4, 5, 6], 'finish_reason': 'length'}

    def test_targets_include_last_prediction_without_forwarding_target(self):
        case = probe_case([self.record()], 10)
        self.assertEqual(case['tokens'], [1, 2, 3, 4])
        self.assertEqual(case['targets'], [3, 4, 5])
        self.assertEqual(case['target_positions'], [1, 2, 3])

    def test_reject_other_split_invalid_and_short_sequences(self):
        for changed in ({'split': 'observation'}, {'id': 'cal-01'},
                        {'generated_tokens': [3, 4]}, {'prompt_tokens': []},
                        {'prompt_tokens': [True]}, {'generated_tokens': [3, 4, 10]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                probe_case([{**self.record(), **changed}], 10)

    def test_full_continuation_supervises_last_emission_without_forwarding_it(self):
        case = probe_case([self.record()], 10, full_continuation=True)
        self.assertEqual(case['tokens'], [1, 2, 3, 4, 5])
        self.assertEqual(case['targets'], [3, 4, 5, 6])
        self.assertEqual(case['target_positions'], [1, 2, 3, 4])
        self.assertEqual(case['source_finish_reason'], 'length')

    def test_gradient_energy_keeps_token_dependence_and_allows_zero_elements(self):
        result = gradient_summary(torch.tensor([[[0., 2.], [1., 0.], [0., 0.]]]))
        self.assertEqual(result['token_squared_norms'], [4., 1., 0.])
        self.assertEqual(result['squared_norm'], 5.)
        self.assertTrue(result['token_energy_varies'])
        for value in (torch.zeros(1, 3, 2), torch.full((1, 3, 2), float('nan')),
                      torch.ones(2, 3, 2), torch.ones(3, 2)):
            with self.assertRaises(ValueError):
                gradient_summary(value)

    def test_projection_scope_retains_final_down_for_evolution(self):
        self.assertEqual(len(PROJECTIONS), 32)
        self.assertEqual(len(set(PROJECTIONS)), 32)
        self.assertNotIn((23, 'down'), PROJECTIONS)


if __name__ == '__main__':
    unittest.main()
