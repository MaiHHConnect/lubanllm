"""Gradient target leakage, phase mapping and BF16 evidence validation."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts'),
               str(Path(__file__).resolve().parents[1]/'src')]
from capture_q4_output_sensitivity import calibration_case, case_rows, raw_summary
from capture_q4_task_reference import layout


class CalibrationSensitivityTests(unittest.TestCase):
    def source(self, emitted=None):
        generated = [3, 4, 5] if emitted is None else emitted
        record = {'id': 'a', 'split': 'calibration', 'prompt_tokens': [1, 2],
                  'generated_tokens': generated, 'finish_reason': 'stop'}
        planned = {'id': 'a', 'split': 'calibration', 'tokens': [1, 2],
                   'forced_tokens': generated[:-1], 'positions': list(range(1+len(generated)))}
        return record, planned

    def test_last_target_is_not_forwarded_and_single_eos_works(self):
        for emitted in ([3, 4, 5], [5]):
            record, planned = self.source(emitted)
            case = calibration_case(record, planned, 10)
            self.assertEqual(case['tokens'], [1, 2]+emitted[:-1])
            self.assertEqual(case['targets'], emitted)
            self.assertEqual(case['target_positions'], list(range(1, 1+len(emitted))))

    def test_reject_observation_mismatched_source_and_bad_tokens(self):
        record, planned = self.source()
        for change in ({'split': 'observation'}, {'id': 'other'}, {'prompt_tokens': [True, 2]},
                       {'generated_tokens': []}, {'generated_tokens': [3, 4, 10]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                calibration_case({**record, **change}, planned, 10)
        for change in ({'split': 'observation'}, {'forced_tokens': [4, 3]},
                       {'positions': [0, 4]}, {'positions': [1, 0]}, {'positions': [0, 0]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                calibration_case(record, {**planned, **change}, 10)

    def test_phase_order_scatter_is_independent_of_run_order(self):
        record, a = self.source()
        b = {**a, 'id': 'b', 'positions': [1, 3]}
        obs = {**a, 'id': 'obs', 'split': 'observation'}
        mapping = layout({'cases': [a, obs, b]})
        ca = calibration_case(record, a, 10)
        cb = calibration_case({**record, 'id': 'b'}, b, 10)
        output = np.full(6, -1)
        # Longer/capacity-first order must not become on-disk case-first ordering.
        output[case_rows(mapping, cb)] = [21, 23]
        output[case_rows(mapping, ca)] = [10, 11, 12, 13]
        np.testing.assert_array_equal(output, [10, 11, 21, 12, 13, 23])
        self.assertEqual(mapping['rows_per_split'], {'calibration': 6, 'observation': 4})

    def test_reject_changed_selected_position_mapping(self):
        record, planned = self.source()
        mapping = layout({'cases': [planned]})
        case = calibration_case(record, planned, 10)
        with self.assertRaises(ValueError):
            case_rows(mapping, {**case, 'selected_positions': [0, 1]})

    def test_bf16_energy_allows_zero_channel_and_preserves_token_variation(self):
        x = np.array([[0, 2, -1], [0, 0, 0], [0, 1, 0]], np.float32)
        raw = (x.view(np.uint32) >> 16).astype(np.uint16)
        summary = raw_summary(raw)
        self.assertEqual(summary['nonzero'], 3)
        self.assertEqual(summary['squared_norm'], 6.)
        self.assertEqual(summary['token_squared_norms'], [5., 0., 1.])
        self.assertEqual(raw_summary(np.zeros((2, 3), np.uint16))['nonzero'], 0)

    def test_reject_nonfinite_or_malformed_carriers(self):
        for raw in (np.array([[0x7f80]], np.uint16), np.array([[0xff80]], np.uint16),
                    np.array([[0x7fc0]], np.uint16), np.zeros((0, 2), np.uint16),
                    np.zeros((2,), np.uint16), np.zeros((2, 3), np.float32)):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                raw_summary(raw)


if __name__ == '__main__':
    unittest.main()
