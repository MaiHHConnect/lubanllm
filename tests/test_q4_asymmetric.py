"""CPU-only independent controls for GPTAQ Eq.15/17 and lazy block updates."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_q4_packed import asymmetric_factor, bf16, gptq_quantize, inverse_factor, metadata


def moments(fp, x):
    """Only reproduce the specified FP32 accumulation, not either factorization."""
    current = torch.from_numpy(x)
    gram = (current.T @ current / len(x)).double().numpy()
    gram += .01 * np.trace(gram) / gram.shape[0] * np.eye(gram.shape[0])
    cross = ((torch.from_numpy(fp) - current).T @ current / len(x)).double().numpy()
    return gram, cross


def constrained_reference(weights, scales, biases, gram, cross, refit):
    """Solve each constrained quadratic directly, without U, P or block feedback."""
    work = weights.astype(np.float64)
    scales, biases = scales.copy(), biases.copy()
    codes = np.empty(weights.shape, dtype=np.uint8)
    for col in range(weights.shape[1]):
        if refit and col % 16 == 0:
            _, _, s, b = metadata(work[:, col:col+16].astype(np.float32))
            scales[:, col//16], biases[:, col//16] = s[:, 0], b[:, 0]
        value = work[:, col].copy()
        raw = (value - biases[:, col//16]) / scales[:, col//16]
        code = np.clip(np.copysign(np.floor(np.abs(raw) + .5), raw), 0, 15)
        decoded = (scales[:, col//16] * code.astype(np.float32) + biases[:, col//16]).astype(np.float64)
        change = decoded - value
        codes[:, col] = code
        work[:, col] = decoded
        if col + 1 < weights.shape[1]:
            rhs = .25 * value[:, None] * cross[col, col+1:] - change[:, None] * gram[col, col+1:]
            work[:, col+1:] += np.linalg.solve(gram[col+1:, col+1:], rhs.T).T
    return codes, scales, biases


class AsymmetricQ4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng = np.random.default_rng(42)
        cls.x = rng.normal(size=(224, 160)).astype(np.float32)
        cls.fp = (cls.x + .35 * rng.normal(size=cls.x.shape)).astype(np.float32)
        cls.weights = bf16(rng.normal(0, .2, size=(6, 160)).astype(np.float32))
        _, _, cls.scales, cls.biases = metadata(cls.weights)
        cls.upper, _ = inverse_factor(cls.x)
        cls.gram, cls.cross = moments(cls.fp, cls.x)

    def test_factor_matches_independent_tail_solves_and_clears_unused_triangle(self):
        expected = np.zeros((160, 160))
        for col in range(159):
            expected[col, col+1:] = .25 * np.linalg.solve(
                self.gram[col+1:, col+1:], self.cross[col, col+1:])
        original = self.upper.clone()
        dirty = self.upper + torch.ones_like(self.upper).tril(-1)
        actual = asymmetric_factor(self.fp, self.x, dirty)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-4, atol=3e-8)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(actual.tril()), 0)
        self.assertTrue(torch.equal(self.upper, original))
        self.assertTrue(torch.equal(actual, asymmetric_factor(self.fp, self.x, self.upper)))

    def test_blocks_and_dynamic_grids_match_independent_constrained_solves(self):
        correction = asymmetric_factor(self.fp, self.x, self.upper)
        originals = [value.copy() for value in (self.weights, self.scales, self.biases)]
        original_correction = correction.clone()
        for refit in (False, True):
            expected = constrained_reference(self.weights, self.scales, self.biases,
                                             self.gram, self.cross, refit)
            baseline = gptq_quantize(self.weights, self.scales, self.biases, self.upper, refit_grid=refit)
            # Require a fixture where correction genuinely affects later blocks.
            self.assertTrue(np.any(expected[0][:, 128:] != baseline[0][:, 128:]))
            for block in (32, 128):
                with self.subTest(refit=refit, block=block):
                    actual = gptq_quantize(self.weights, self.scales, self.biases, self.upper,
                                           block, refit_grid=refit, correction=correction)
                    for got, want in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(got, want)
        for actual, expected in zip((self.weights, self.scales, self.biases), originals, strict=True):
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(torch.equal(correction, original_correction))

    def test_zero_deviation_or_alpha_reproduces_old_path_exactly(self):
        for correction in (asymmetric_factor(self.x, self.x, self.upper),
                           asymmetric_factor(self.fp, self.x, self.upper, alpha=0)):
            self.assertEqual(torch.count_nonzero(correction), 0)
            for refit in (False, True):
                for block in (32, 128):
                    kwargs = dict(block=block, refit_grid=refit)
                    expected = gptq_quantize(self.weights, self.scales, self.biases, self.upper, **kwargs)
                    actual = gptq_quantize(self.weights, self.scales, self.biases, self.upper,
                                           correction=correction, **kwargs)
                    for got, want in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(got, want)

    def test_invalid_alpha_and_pair_dimensions_are_rejected(self):
        for alpha in (True, False, -1, np.nan, np.inf, -np.inf, '.25', None, .25j):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                asymmetric_factor(self.fp, self.x, self.upper, alpha=alpha)
        for fp, x in ((self.fp[:-1], self.x), (self.fp[:, :-1], self.x),
                      (self.fp[:1], self.x[:1]), (np.zeros((2, 0)), np.zeros((2, 0))),
                      (np.zeros(4), np.zeros(4))):
            with self.subTest(shape=(fp.shape, x.shape)), self.assertRaises(ValueError):
                asymmetric_factor(fp, x, self.upper)

    def test_nonfinite_inputs_factors_and_overflow_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            for reference in (False, True):
                altered = self.x.copy(); altered[0, 0] = bad
                with self.subTest(bad=bad, reference=reference), self.assertRaises(ValueError):
                    asymmetric_factor(altered if reference else self.fp,
                                      self.x if reference else altered, self.upper)
            factor = self.upper.clone(); factor[0, 0] = bad
            with self.assertRaises(ValueError):
                asymmetric_factor(self.fp, self.x, factor)
        for factor in (self.upper[:-1], self.upper.double(), np.eye(160), -self.upper):
            with self.assertRaises(ValueError):
                asymmetric_factor(self.fp, self.x, factor)
        with self.assertRaisesRegex(ValueError, 'nonfinite asymmetric'):
            asymmetric_factor(np.full((2, 16), 1e30, np.float32),
                              np.full((2, 16), -1e30, np.float32), torch.eye(16))

    def test_invalid_correction_is_rejected_before_quantization(self):
        for correction in (torch.zeros(32, 32), torch.eye(160), torch.ones(160, 160).tril(-1),
                           torch.zeros(160, 160, dtype=torch.float64), np.zeros((160, 160))):
            with self.subTest(type=type(correction)), self.assertRaises(ValueError):
                gptq_quantize(self.weights, self.scales, self.biases, self.upper, correction=correction)
        for value in (float('nan'), float('inf'), -float('inf')):
            correction = torch.zeros(160, 160); correction[0, 1] = value
            with self.assertRaises(ValueError):
                gptq_quantize(self.weights, self.scales, self.biases, self.upper, correction=correction)


if __name__ == '__main__':
    unittest.main()
