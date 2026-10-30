"""Independent CPU ridge-objective checks, without real model or GPU execution."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from study_q4_column_fit import RIDGE, bf16, fit_columns


def dense_reference(z, residual, selected):
    residual = residual.astype(np.float64)
    if not selected:
        return np.empty((0, residual.shape[1])), float(np.square(residual).sum()/len(z))
    # Independent augmented least squares; does not use the production Schur/Cholesky path.
    matrix = np.vstack((z[:, selected]/np.sqrt(len(z)), np.sqrt(RIDGE)*np.eye(len(selected))))
    target = np.vstack((residual/np.sqrt(len(z)), np.zeros((len(selected), residual.shape[1]))))
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    objective = float(np.square(residual-z[:, selected]@coefficients).sum()/len(z)
                      +RIDGE*np.square(coefficients).sum())
    return coefficients, objective


class ColumnFitTests(unittest.TestCase):
    def test_each_choice_and_gain_matches_exhaustive_joint_ridge_objective(self):
        rng = np.random.default_rng(42)
        x = bf16(rng.normal(size=(27, 9)).astype(np.float32))
        residual = bf16(rng.normal(size=(27, 4)).astype(np.float32))
        indices, update, info = fit_columns(x, residual, rank=6)
        scales = np.sqrt(np.mean(x.astype(np.float64)**2, axis=0)); z = x/scales
        selected, old = [], dense_reference(z, residual, [])[1]
        for step in info['selected']:
            objectives = {j: dense_reference(z, residual, selected+[j])[1] for j in range(9) if j not in selected}
            best = min(objectives, key=lambda j: (objectives[j], j))
            self.assertEqual(step['column'], best)
            self.assertAlmostEqual(step['regularized_objective_decrease_per_row'], old-objectives[best], places=11)
            self.assertAlmostEqual(step['joint_regularized_objective_per_row'], objectives[best], places=11)
            selected.append(best); old = objectives[best]
        coefficients, objective = dense_reference(z, residual, selected)
        expected = bf16((coefficients/scales[selected, None]).T.astype(np.float32))
        np.testing.assert_array_equal(update[:, :len(selected)], expected)
        np.testing.assert_array_equal(indices[:len(selected)], selected)
        self.assertEqual(info['gram_rows_computed'], len(selected))
        self.assertNotIn('predicted_sse_decrease', info['selected'][0])

    def test_small_dense_solve_and_bf16_export_after_joint_refit(self):
        x = np.array([[1, 0], [0, 2], [1, 2]], np.float32)
        residual = np.array([[1, 2], [3, -1], [4, 1]], np.float32)
        indices, update, info = fit_columns(x, residual, rank=2)
        s = np.sqrt(np.mean(x.astype(np.float64)**2, axis=0)); z = x/s
        selected = indices[:info['selected_count']].tolist()
        c = np.linalg.solve(z[:, selected].T@z[:, selected]/len(x)+RIDGE*np.eye(2),
                            z[:, selected].T@residual/len(x))
        np.testing.assert_array_equal(update, bf16((c/s[selected, None]).T.astype(np.float32)))
        self.assertTrue(np.all((update.view(np.uint32) & 0xffff) == 0))
        self.assertEqual(info['selection_policy'], 'normalized_column_ridge_fit')
        self.assertIn('not W-Q', info['factor_semantics'])

    def test_zero_duplicate_and_strongly_correlated_columns_are_stable(self):
        a = np.array([1, -2, 3, -4, 2, 1], np.float32)
        x = np.stack((np.zeros_like(a), a, a, a+np.array([0, 0, 0, 0, 0, .0001], np.float32)), axis=1)
        residual = a[:, None]
        indices, update, info = fit_columns(x, residual)
        self.assertEqual(info['zero_rms_columns'], 1)
        self.assertNotIn(0, [s['column'] for s in info['selected']])
        self.assertTrue(np.isfinite(update).all())
        self.assertTrue(all(s['regularized_objective_decrease_per_row'] > 0 for s in info['selected']))
        self.assertEqual(len(set(indices[:info['selected_count']])), info['selected_count'])
        np.testing.assert_array_equal(update[:, info['selected_count']:], 0)
        repeated = np.stack((a, a), axis=1)
        self.assertEqual(fit_columns(repeated, residual, rank=1)[0][0], 0)
        z = x.astype(np.float64)/np.where(np.std(x, axis=0) > 0, np.sqrt(np.mean(x.astype(np.float64)**2, axis=0)), 1)
        selected = indices[:info['selected_count']].tolist()
        self.assertAlmostEqual(info['selected'][-1]['joint_regularized_objective_per_row'],
                               dense_reference(z, residual, selected)[1], places=10)

    def test_normalization_is_invariant_to_input_column_power_two_scaling(self):
        rng = np.random.default_rng(7)
        x = bf16(rng.normal(size=(24, 5)).astype(np.float32))
        residual = bf16(rng.normal(size=(24, 3)).astype(np.float32))
        multipliers = np.array([.25, 2, 8, .5, 4], np.float32)
        original = fit_columns(x, residual, rank=4)
        scaled = fit_columns(x*multipliers, residual, rank=4)
        np.testing.assert_array_equal(original[0], scaled[0])
        np.testing.assert_allclose(original[1], scaled[1]*multipliers[scaled[0]][None, :], rtol=0, atol=0)
        np.testing.assert_allclose(x[:, original[0]]@original[1].T,
                                   (x*multipliers)[:, scaled[0]]@scaled[1].T, rtol=1e-6, atol=1e-6)

    def test_zero_residual_zero_inputs_and_finite_shape_rank_guards(self):
        x = np.eye(3, dtype=np.float32); r = np.ones((3, 2), np.float32)
        for inputs, residual in ((x, np.zeros_like(r)), (np.zeros_like(x), r)):
            indices, update, info = fit_columns(inputs, residual)
            self.assertEqual(info['selected_count'], 0)
            np.testing.assert_array_equal(indices, 0); np.testing.assert_array_equal(update, 0)
        for inputs, residual in ((x*np.nan, r), (x, r*np.inf), (x[:2], r), (x[:0], r[:0])):
            with self.assertRaises(ValueError): fit_columns(inputs, residual)
        for rank in (0, 9, True, 1.5):
            with self.assertRaises(ValueError): fit_columns(x, r, rank=rank)


if __name__ == '__main__':
    unittest.main()
