"""Small CPU mathematics and prerequisite guards for diagnostic output proxies."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import study_q4_corrected_outputs as outputs
from diagnose_native_q4 import bf16
from study_q4_column_protection import output_errors


class CorrectedOutputTests(unittest.TestCase):
    def test_explicit_dense_formula_and_full_old_report(self):
        rng = np.random.default_rng(42)
        w = bf16(rng.normal(size=(3, 4)).astype(np.float32))
        q = bf16(w*.9)
        fp = bf16(rng.normal(size=(7, 4)).astype(np.float32))
        old_x = bf16(fp+.03)
        actual_x = bf16(old_x-.07)
        indices = np.array([2, 0], dtype=np.int64)
        u = bf16(np.array([[.03, 0], [-.02, 0], [.01, 0]], np.float32))
        report = outputs.projection_proxy(w, q, fp, old_x, actual_x, indices, u)
        target = bf16(fp@w.T).astype(np.float64)
        baseline = bf16(old_x@q.T).astype(np.float64)
        old = bf16(bf16(old_x@q.T)+old_x[:, indices]@u.T).astype(np.float64)
        actual = bf16(bf16(actual_x@q.T)+actual_x[:, indices]@u.T).astype(np.float64)
        for name, value in [('v2_sse', baseline), ('fit_old_sse', old), ('fit_actual_sse', actual)]:
            self.assertAlmostEqual(report[name], float(np.sum((value-target)**2)), places=12)
        e, d = old-target, actual-old
        self.assertAlmostEqual(report['propagation']['twice_error_drift_inner_product'], float(2*np.sum(e*d)), places=12)
        expected = output_errors(fp@w.T, old_x@q.T, old_x, indices, u)
        self.assertTrue(outputs.reproduce_report(report['old_report'], expected)['exact'])
        bad = deepcopy(expected); bad['before_output_round']['ideal_additive_sse'] += .01
        with self.assertRaisesRegex(ValueError, 'did not reproduce'):
            outputs.reproduce_report(report['old_report'], bad)
        bad = deepcopy(expected); del bad['before_final_round']
        with self.assertRaises(ValueError):
            outputs.reproduce_report(report['old_report'], bad)

    def test_propagation_cross_term_can_reverse_or_improve_old_gain(self):
        target = np.zeros((1, 1), np.float32)
        baseline, old = np.array([[2]], np.float32), np.array([[1]], np.float32)
        reversed_result = outputs.error_decomposition(target, baseline, old, np.array([[3]], np.float32))
        self.assertEqual(reversed_result['propagation']['twice_error_drift_inner_product'], 4)
        self.assertEqual(reversed_result['propagation']['drift_energy'], 4)
        self.assertEqual(reversed_result['propagation']['actual_minus_old_sse'], 8)
        self.assertTrue(reversed_result['old_gain_strictly_reversed'])
        tied = outputs.error_decomposition(target, baseline, old, baseline)
        self.assertTrue(tied['old_gain_erased']); self.assertFalse(tied['old_gain_strictly_reversed'])
        better = outputs.error_decomposition(target, baseline, old, target)
        self.assertEqual(better['propagation']['twice_error_drift_inner_product'], -2)
        self.assertEqual(better['propagation']['drift_energy'], 1)
        self.assertFalse(better['old_gain_erased'])

    def test_same_inputs_zero_drift_and_zero_update(self):
        w = np.array([[2, -1]], np.float32); q = w*.5
        x = np.array([[1, 2], [3, 4]], np.float32)
        report = outputs.projection_proxy(w, q, x, x, x, np.array([0, 0]), np.zeros((1, 2), np.float32))
        self.assertEqual(report['v2_sse'], report['fit_old_sse'])
        self.assertEqual(report['fit_old_sse'], report['fit_actual_sse'])
        self.assertEqual(report['propagation']['drift_energy'], 0)
        self.assertEqual(report['propagation']['identity_error'], 0)

    def test_round_q4_before_adding_and_reject_nonfinite_or_unstored_factor(self):
        # 1.003 rounds to1 before the .003 residual; rounding only after addition would differ.
        w = q = np.array([[1.003]], np.float32)
        x = np.ones((1, 1), np.float32)
        u = bf16(np.array([[.003]], np.float32))
        report = outputs.projection_proxy(w, q, x, x, x, np.array([0]), u)
        self.assertEqual(report['fit_old_sse'], 0)
        self.assertNotEqual(float(bf16(q+u)[0, 0]), float(bf16(bf16(q)+u)[0, 0]))
        for bad in (np.array([[np.nan]], np.float32), np.array([[.003]], np.float32)):
            with self.assertRaises(ValueError):
                outputs.projection_proxy(w, q, x, x, x, np.array([0]), bad)
        with self.assertRaises(ValueError):
            outputs.projection_proxy(w, q, x, x, x, np.array([1]), u)

    def test_preflight_requires_completed_audited_capture_before_tensor_access(self):
        for capture, audit in [({}, {}), ({'schema': 'q4-corrected-inputs-diagnostic-v1', 'completed': True},
                                         {'audit_passed': False})]:
            with patch.object(outputs, 'read', side_effect=[{}, {}, capture, audit]), \
                    patch.object(outputs, 'frozen_scope', return_value=[]), \
                    patch.object(outputs, 'load_reference') as reference:
                with self.assertRaisesRegex(ValueError, 'independently audited'):
                    outputs.preflight()
                reference.assert_not_called()


if __name__ == '__main__':
    unittest.main()
