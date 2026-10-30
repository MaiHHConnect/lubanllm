"""CPU tests for one original-support ridge refit and retain-old calibration gate."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import study_q4_fixed_support_refit as refit
from diagnose_native_q4 import bf16


class FixedSupportRefitTests(unittest.TestCase):
    def test_joint_ridge_matches_independent_augmented_least_squares(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(13, 5)).astype(np.float32)
        residual = rng.normal(size=(13, 3)).astype(np.float32)
        indices = np.array([4, 1, 3, 0, 0, 0, 0, 0])
        old_scales = np.array([.7, 2., .3], np.float64)
        update, solve = refit.refit_fixed_support(x, residual, indices, old_scales, 3)
        z = x[:, indices[:3]].astype(np.float64)/old_scales
        augmented = np.vstack([z/np.sqrt(len(x)), np.sqrt(.01)*np.eye(3)])
        target = np.vstack([residual.astype(np.float64)/np.sqrt(len(x)), np.zeros((3, 3))])
        expected = np.linalg.lstsq(augmented, target, rcond=None)[0]
        np.testing.assert_allclose(solve['coefficients'], expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_array_equal(update[:, :3], bf16((expected/old_scales[:, None]).T.astype(np.float32)))
        np.testing.assert_array_equal(update[:, 3:], 0)
        np.testing.assert_array_equal(solve['scales'], old_scales)
        new_scales = np.sqrt(np.mean(x[:, indices[:3]].astype(np.float64)**2, axis=0))
        changed, _ = refit.refit_fixed_support(x, residual, indices, new_scales, 3)
        self.assertFalse(np.array_equal(changed, update))  # Must not renormalize from new Xfit.

    def test_fixed_support_order_correlated_and_zero_new_column(self):
        x = np.array([[1, 2, 0], [2, 4, 0], [-1, -2, 0], [3, 6, 0]], np.float32)
        r = np.array([[1, 2], [2, 1], [-1, -2], [3, 0]], np.float32)
        indices = np.array([1, 0, 2, 0, 0, 0, 0, 0])
        scales = np.array([2, 1, 5], np.float64)
        update, solve = refit.refit_fixed_support(x, r, indices, scales, 3)
        self.assertTrue(np.isfinite(update).all())
        np.testing.assert_array_equal(update[:, 2], 0)
        permuted, second = refit.refit_fixed_support(x, r, np.array([0, 1, 2, 0, 0, 0, 0, 0]), scales[[1, 0, 2]], 3)
        np.testing.assert_allclose(second['coefficients'], solve['coefficients'][[1, 0, 2]], atol=1e-12)
        np.testing.assert_array_equal(permuted[:, [1, 0, 2]], update[:, :3])

    def test_gate_strictly_beats_both_and_fallback_preserves_original(self):
        old = np.full((2, 8), 1.5, np.float32)
        new = np.full((2, 8), -.5, np.float32)
        def report(base, candidate):
            return {'after_output_round': {'baseline_sse': base, 'candidate_sse': candidate}}
        for baseline, old_sse, new_sse, accepted in [(10, 8, 7, True), (10, 12, 9, True),
                (10, 12, 11, False), (10, 8, 9, False), (10, 8, 8, False), (10, 12, 10, False)]:
            applied, gate = refit.replacement_gate(old, new, report(baseline, old_sse), report(baseline, new_sse))
            self.assertIs(gate['accepted'], accepted)
            np.testing.assert_array_equal(applied, new if accepted else old)
            self.assertEqual(gate['action'], 'replace_coefficients' if accepted else 'retain_original_coefficients')
        np.testing.assert_array_equal(old, 1.5)
        with self.assertRaises(ValueError):
            refit.replacement_gate(old, new, report(10, 8), report(11, 7))

    def test_original_scale_binding_and_zero_padding_guards(self):
        indices = np.array([2, 0, 0, 0, 0, 0, 0, 0])
        row = {'active_columns': 1, 'selection': {'selected_count': 1,
               'selected': [{'slot': 0, 'column': 2, 'rms': 3.5}]}}
        np.testing.assert_array_equal(refit.original_scales(row, indices, 1), [3.5])
        for bad in ([{'slot': 0, 'column': 1, 'rms': 3.5}], [{'slot': 0, 'column': 2, 'rms': 0}],
                    [{'slot': 0, 'column': 2, 'rms': float('nan')}]):
            with self.assertRaises(ValueError):
                refit.original_scales({**row, 'selection': {'selected_count': 1, 'selected': bad}}, indices, 1)
        x, r = np.ones((4, 3), np.float32), np.ones((4, 2), np.float32)
        for idx, scales, count in [(np.array([2, 2, 0, 0, 0, 0, 0, 0]), [1, 1], 2),
                (np.array([2, 1, 0, 0, 0, 0, 0, 0]), [1], 1), (indices, [0], 1),
                (indices, [float('inf')], 1)]:
            with self.assertRaises(ValueError):
                refit.refit_fixed_support(x, r, idx, scales, count)
        with self.assertRaises(ValueError):
            refit.refit_fixed_support(x*np.nan, r, indices, [1], 1)

    def test_empty_support_and_replacement_not_additive_delta(self):
        x, r = np.ones((3, 2), np.float32), np.ones((3, 1), np.float32)
        empty, solve = refit.refit_fixed_support(x, r, np.zeros(8, np.int64), [], 0)
        np.testing.assert_array_equal(empty, 0)
        self.assertEqual(solve['coefficients'].shape, (0, 1))
        actual, _ = refit.refit_fixed_support(x, r, np.zeros(8, np.int64), [1], 1)
        expected = bf16(np.array([[1/1.01]], np.float32))[0, 0]
        self.assertEqual(actual[0, 0], expected)
        self.assertNotEqual(actual[0, 0], expected+2.)  # No old U is added by the solve.


if __name__ == '__main__':
    unittest.main()
