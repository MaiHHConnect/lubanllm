"""Finite-direction protection properties, not a model-quality test."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from study_q4_protection import basis_for, compensated_error, q4


class ProtectionTests(unittest.TestCase):
    def test_restores_calibration_span_but_not_unseen_directions(self):
        rng = np.random.default_rng(42)
        inputs = np.zeros((16, 32), dtype=np.float32)
        inputs[:, :4] = rng.normal(size=(16, 4))
        delta = rng.normal(size=(8, 32)).astype(np.float32)
        basis = basis_for(inputs, 8)
        self.assertEqual(basis.shape[1], 4)
        self.assertLess(np.abs(compensated_error(inputs, delta, basis)).max(), 1e-4)
        unseen = np.zeros((1, 32), dtype=np.float32)
        unseen[0, 20] = 1
        np.testing.assert_allclose(compensated_error(unseen, delta, basis), unseen @ delta.T, atol=1e-5)

    def test_group_boundaries_and_zero(self):
        w = np.zeros((2, 64), dtype=np.float32)
        w[0, :32] = 1
        w[1, 32:] = -1
        np.testing.assert_array_equal(q4(w), w)
        with self.assertRaises(ValueError):
            q4(np.zeros((2, 33), dtype=np.float32))
        w[0, 0] = np.nan
        with self.assertRaises(ValueError):
            q4(w)


if __name__ == '__main__':
    unittest.main()
