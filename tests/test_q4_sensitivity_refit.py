"""Small CPU checks of token-dependent weighted ridge and its rounded-only gate."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_native_q4 import bf16
from study_q4_fixed_support_refit import refit_fixed_support
from study_q4_sensitivity_refit import (normalized_sensitivity, refit_sensitive_support,
                                      weighted_output_errors, weighted_replacement_gate)


class SensitivityRefitTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(17, 5)).astype(np.float32)
        self.r = rng.normal(size=(17, 3)).astype(np.float32)
        self.indices = np.array([3, 1, 4, 0, 0, 0, 0, 0], np.int64)
        self.scales = np.array([2., .7, 1.3])
        self.old = np.zeros((3, 8), np.float32)
        self.old[:, :3] = bf16(rng.normal(size=(3, 3)).astype(np.float32))
        self.g = rng.normal(size=self.r.shape).astype(np.float32)

    def fit(self, gradient=None, **changes):
        d, means, stats = normalized_sensitivity(self.g if gradient is None else gradient)
        args = dict(inputs=self.x, residual=self.r, indices=self.indices, scales=self.scales,
                    active_count=3, old=self.old, weights=d, mean_square=means)
        args.update(changes)
        result, solve = refit_sensitive_support(**args)
        return result, solve, stats, d, means

    def test_weighted_solve_matches_independent_augmented_lstsq(self):
        actual, solve, stats, d, _ = self.fit()
        z = self.x[:, self.indices[:3]].astype(np.float64)/self.scales
        for output in range(3):
            root = np.sqrt(d[:, output]/len(z))
            matrix = np.vstack([root[:, None]*z, np.sqrt(.01)*np.eye(3)])
            target = np.concatenate([root*self.r[:, output].astype(np.float64), np.zeros(3)])
            expected = np.linalg.lstsq(matrix, target, rcond=None)[0]
            np.testing.assert_allclose(solve['coefficients'][:, output], expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(actual[output, :3], bf16((expected/self.scales).astype(np.float32)))
        np.testing.assert_array_equal(actual[:, 3:], 0)
        self.assertEqual(stats['active_channels'], 3)
        self.assertEqual(solve['systems'].shape, (3, 3, 3))
        self.assertTrue(np.all(solve['active_channels']))
        plain, _ = refit_fixed_support(self.x, self.r, self.indices, self.scales, 3)
        self.assertFalse(np.array_equal(actual, plain))

    def test_constant_token_weights_cancel_and_per_channel_rescaling_cancels(self):
        g = np.tile(np.array([.25, 2., -8.], np.float32), (17, 1))
        actual, solve, _, d, _ = self.fit(g)
        plain, unweighted = refit_fixed_support(self.x, self.r, self.indices, self.scales, 3)
        np.testing.assert_array_equal(d, 1.)
        np.testing.assert_array_equal(actual, plain)
        np.testing.assert_allclose(solve['coefficients'], unweighted['coefficients'], atol=1e-13)
        first = self.fit()[0]
        scaled = self.fit(self.g*np.array([2., .25, -4.], np.float32))[0]
        np.testing.assert_array_equal(first, scaled)

    def test_zero_channels_preserve_full_old_row_and_do_not_add_coefficients(self):
        g = self.g.copy();g[:, 1] = 0
        actual, solve, stats, d, _ = self.fit(g)
        np.testing.assert_array_equal(actual[1], self.old[1])
        np.testing.assert_array_equal(solve['systems'][1], 0.)
        np.testing.assert_array_equal(solve['coefficients'][:, 1], self.old[1, :3]*self.scales)
        np.testing.assert_array_equal(d[:, 1], 0.)
        self.assertEqual(stats['zero_channels'], 1)
        self.assertFalse(solve['active_channels'][1])
        other_old = self.old.copy();other_old[0, :3] = 8
        changed = self.fit(g, old=other_old)[0]
        np.testing.assert_array_equal(changed[0], actual[0])  # Replacement is independent of old active U.
        empty, _, empty_stats, _, _ = self.fit(np.zeros_like(g))
        np.testing.assert_array_equal(empty, self.old)
        self.assertEqual(empty_stats['active_channels'], 0)

    def test_weighted_two_round_error_matches_scalar_reference(self):
        rng = np.random.default_rng(51)
        target = rng.normal(size=self.r.shape).astype(np.float32)
        baseline = rng.normal(size=self.r.shape).astype(np.float32)
        _, _, _, d, means = self.fit()
        actual = weighted_output_errors(target, baseline, self.x, self.indices, self.old, d, means)
        correction = self.x[:, self.indices]@self.old.T
        rounded_base, rounded_target = bf16(baseline), bf16(target)
        candidate = bf16(rounded_base+correction)
        for label, value in (('baseline', rounded_base), ('candidate', candidate)):
            weighted = raw = 0.
            for t in range(len(value)):
                for o in range(value.shape[1]):
                    error = float(value[t, o])-float(rounded_target[t, o])
                    weighted += d[t, o]*error**2
                    raw += float(self.g[t, o])**2*error**2
            self.assertAlmostEqual(actual['after_output_round'][label+'_sse'], weighted, places=11)
            self.assertAlmostEqual(actual['raw_gradient_squared_sse_report_only'][label+'_sse'], raw, places=11)
            self.assertAlmostEqual(actual['after_output_round'][label+'_mse'], weighted/(17*3), places=12)

    def test_gate_uses_rounded_normalized_error_only_and_strict_fallback(self):
        x = np.ones((2, 1), np.float32)
        idx = np.zeros(8, np.int64)
        target = np.full((2, 1), 1.0078125, np.float32)
        baseline = np.full((2, 1), 1.003, np.float32)
        old = np.zeros((1, 8), np.float32)
        new = old.copy();new[0, 0] = .001953125
        d, means, _ = normalized_sensitivity(np.ones((2, 1), np.float32))
        before = weighted_output_errors(target, baseline, x, idx, old, d, means)
        after = weighted_output_errors(target, baseline, x, idx, new, d, means)
        # Ignoring the first BF16 round would produce an apparent strict improvement.
        self.assertTrue(np.sum((bf16(baseline+x[:, idx]@new.T)-bf16(target))**2)
                        < before['after_output_round']['candidate_sse'])
        applied, gate = weighted_replacement_gate(old, new, before, after)
        self.assertFalse(gate['accepted'])
        np.testing.assert_array_equal(applied, old)
        new[0, 0] = .0078125
        after = weighted_output_errors(target, baseline, x, idx, new, d, means)
        # These report-only values must not become a second gate.
        after['raw_gradient_squared_sse_report_only']['candidate_sse'] = 1e30
        applied, gate = weighted_replacement_gate(old, new, before, after)
        self.assertTrue(gate['accepted'])
        np.testing.assert_array_equal(applied, new)
        zero_d, zero_means, _ = normalized_sensitivity(np.zeros((2, 1), np.float32))
        before = weighted_output_errors(target, baseline, x, idx, old, zero_d, zero_means)
        after = weighted_output_errors(target, baseline, x, idx, new, zero_d, zero_means)
        self.assertEqual(before['after_output_round']['candidate_mse'], 0)
        self.assertFalse(weighted_replacement_gate(old, new, before, after)[1]['accepted'])
        def report(base, candidate):
            return {'rows': 2, 'outputs': 1, 'active_channels': 1,
                    'after_output_round': {'baseline_sse': base, 'candidate_sse': candidate}}
        for zero, previous, candidate, expected in ((10, 12, 11, False), (10, 12, 9, True),
                                                    (10, 8, 8, False), (10, 8, 7, True)):
            applied, gate = weighted_replacement_gate(old, new, report(zero, previous), report(zero, candidate))
            self.assertIs(gate['accepted'], expected)
            np.testing.assert_array_equal(applied, new if expected else old)

    def test_invalid_support_weights_gradient_and_precision_rejected(self):
        for g in (np.full((17, 3), np.nan), np.empty((0, 3)), np.ones(3)):
            with self.subTest(shape=g.shape), self.assertRaises(ValueError):
                normalized_sensitivity(g)
        d, means, _ = normalized_sensitivity(self.g)
        bad_old = self.old.copy();bad_old[0, 0] = np.float32(.123456)
        for changes in ({'indices': np.array([3, 3, 4, 0, 0, 0, 0, 0])},
                        {'scales': [2., 0., 1.]}, {'old': bad_old}, {'weights': d*2},
                        {'weights': -d}, {'mean_square': np.zeros(3)}, {'residual': self.r[:-1]}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.fit(**changes)


if __name__ == '__main__':
    unittest.main()
