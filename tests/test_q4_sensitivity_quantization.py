"""Small CPU mathematical controls; no model, native worker or GPU execution."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_q4_packed import asymmetric_factor, bf16, gptq_quantize, inverse_factor, metadata
from q4_sensitivity_quantization import normalized_token_weights, quantize_sensitive, weighted_inputs


class SensitivityQuantizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng = np.random.default_rng(42)
        cls.x = rng.normal(size=(80, 32)).astype(np.float32)
        cls.fp = (cls.x+.3*rng.normal(size=cls.x.shape)).astype(np.float32)
        cls.w = bf16(rng.normal(0, .2, size=(5, 32)).astype(np.float32))
        cls.d = np.linspace(.1, 1.9, 80)

    def test_raw_square_pooling_precedes_normalization(self):
        g = np.array([[1., -1., 10.], [2., -2., 0.], [0., 0., 0.]])
        d, stats = normalized_token_weights(g)
        raw = np.array([102., 8., 0.])/3
        np.testing.assert_allclose(d, raw/raw.mean(), rtol=1e-15)
        np.testing.assert_array_equal(normalized_token_weights(-g)[0], d)
        np.testing.assert_array_equal(normalized_token_weights(g*8)[0], d)
        channel_normalized = (g*g/np.mean(g*g, axis=0)).mean(axis=1)
        self.assertFalse(np.allclose(d, channel_normalized))
        self.assertEqual(stats['groups'], 1)
        self.assertEqual(stats['zero_weight_rows'], 1)
        self.assertEqual(stats['raw_mean_square'], raw.mean())
        zero, stats = normalized_token_weights(np.zeros_like(g))
        np.testing.assert_array_equal(zero, 0.)
        self.assertTrue(stats['all_zero'])
        self.assertEqual(stats['effective_rows'], 0.)

    def test_constant_weights_exactly_recover_original_conversion(self):
        d, _ = normalized_token_weights(np.tile([1., -3., 8.], (80, 1)))
        np.testing.assert_array_equal(d, 1.)
        fp, x = weighted_inputs(self.fp, self.x, d)
        np.testing.assert_array_equal(fp, self.fp)
        np.testing.assert_array_equal(x, self.x)
        upper, hessian = inverse_factor(self.x)
        correction = asymmetric_factor(self.fp, self.x, upper)
        _, _, scales, biases = metadata(self.w)
        expected = gptq_quantize(self.w, scales, biases, upper,
                                 refit_grid=True, correction=correction)
        actual = quantize_sensitive(self.w, self.fp, self.x, d)
        for got, want in zip(actual[:3], expected, strict=True):
            np.testing.assert_array_equal(got, want)
        self.assertEqual(actual[3]['hessian'], hessian)

    def test_weighted_gram_cross_and_correction_match_direct_formulas(self):
        fp, x = weighted_inputs(self.fp, self.x, self.d)
        reference, current = self.fp.astype(np.float64), self.x.astype(np.float64)
        gram = current.T@(self.d[:, None]*current)/len(current)
        cross = (reference-current).T@(self.d[:, None]*current)/len(current)
        np.testing.assert_allclose(x.T@x/len(x), gram, rtol=3e-5, atol=3e-7)
        np.testing.assert_allclose((fp-x).T@x/len(x), cross, rtol=3e-5, atol=5e-8)
        upper, stats = inverse_factor(x)
        damped = gram+.01*np.trace(gram)/32*np.eye(32)
        np.testing.assert_allclose(upper.numpy().T@upper.numpy(), np.linalg.inv(damped),
                                   rtol=5e-5, atol=5e-7)
        correction = asymmetric_factor(fp, x, upper).numpy()
        expected = np.zeros((32, 32))
        for col in range(31):
            expected[col, col+1:] = .25*np.linalg.solve(
                damped[col+1:, col+1:], cross[col, col+1:])
        np.testing.assert_allclose(correction, expected, rtol=8e-5, atol=1e-7)
        self.assertAlmostEqual(stats['damping'], .01*np.trace(gram)/32, places=8)
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.any(x != bf16(x)))

    def test_projection_factors_are_independent_and_inputs_are_unchanged(self):
        original = [v.copy() for v in (self.w, self.fp, self.x, self.d)]
        first = quantize_sensitive(self.w, self.fp, self.x, self.d)
        second = quantize_sensitive(self.w, self.fp, self.x, self.d[::-1])
        self.assertNotEqual(first[3]['hessian']['damping'], second[3]['hessian']['damping'])
        self.assertTrue(np.any(first[0] != second[0]))
        repeat = quantize_sensitive(self.w, self.fp, self.x, self.d)
        for got, want in zip(first[:3], repeat[:3], strict=True):
            np.testing.assert_array_equal(got, want)
        for got, want in zip((self.w, self.fp, self.x, self.d), original, strict=True):
            np.testing.assert_array_equal(got, want)
        same = quantize_sensitive(self.w, self.x, self.x, self.d)
        self.assertTrue(same[3]['zero_asymmetric_correction'])

    def test_invalid_gradient_and_weighting_inputs_are_rejected(self):
        for g in ([], np.zeros((1, 3)), np.zeros((3, 0)), np.ones((2, 2, 2)),
                  [[np.nan], [1]], [[np.inf], [1]], [[1j], [2j]], [[True], [False]],
                  [[1e308], [1e308]]):
            with self.subTest(gradient=np.shape(g)), self.assertRaises(ValueError):
                normalized_token_weights(g)
        for d in (np.zeros(80), -self.d, self.d[:-1], self.d[:, None],
                  np.full(80, np.nan), np.full(80, np.inf), np.ones(80, complex)):
            with self.subTest(weights=np.shape(d)), self.assertRaises(ValueError):
                weighted_inputs(self.fp, self.x, d)
        for fp, x in ((self.fp[:-1], self.x), (self.fp, self.x[:, :-1]),
                      (self.fp[0], self.x[0]), (self.fp*float('inf'), self.x)):
            with self.assertRaises(ValueError):
                weighted_inputs(fp, x, self.d)
        with self.assertRaises(ValueError):
            weighted_inputs(np.full((2, 16), 1e30), np.ones((2, 16)), np.full(2, 1e30))

    def test_invalid_projection_and_unobserved_columns_are_rejected(self):
        for w in (self.w[:, :-1], np.zeros((0, 32)), np.zeros(32),
                  np.full_like(self.w, np.nan)):
            with self.assertRaises(ValueError):
                quantize_sensitive(w, self.fp, self.x, self.d)
        dead = self.x.copy(); dead[:, 3] = 0
        with self.assertRaisesRegex(ValueError, 'observed input columns'):
            quantize_sensitive(self.w, self.fp, dead, self.d)
        with self.assertRaises(ValueError):
            quantize_sensitive(self.w, self.fp, self.x, np.zeros(80))


if __name__ == '__main__':
    unittest.main()
