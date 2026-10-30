"""CPU mathematics for frozen-budget column protection; no real model or GPU."""
from copy import deepcopy
from pathlib import Path
import sys
import struct
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from study_q4_column_protection import (LAYERS, MASK, ROOT, ROWS, bf16, calibration_gate,
                                      dequantize_payload, frozen_scope, output_errors, select_columns, select_all_columns)


class ColumnProtectionTests(unittest.TestCase):
    def test_all_columns_agrees_with_independent_exhaustive_residual_sse_steps(self):
        rng = np.random.default_rng(71)
        x = bf16(rng.normal(size=(21, 11)).astype(np.float32))
        fp = x+bf16(rng.normal(0, .15, x.shape).astype(np.float32))
        w = bf16(rng.normal(size=(4, 11)).astype(np.float32))
        q = w+bf16(rng.normal(0, .3, w.shape).astype(np.float32))
        residual = fp@w.T-x@q.T
        indices, factors, selected = select_all_columns(w, q, x, residual)
        current, available = residual.astype(np.float64), list(range(x.shape[1]))
        choices = []
        for slot in range(8):
            old_sse = np.square(current).sum()
            trials = {}
            for col in available:
                delta = x[:, col, None].astype(np.float64)*bf16(w[:, col]-q[:, col])[None, :]
                trials[col] = (old_sse-np.square(current-delta).sum(), delta)
            best = max(available, key=lambda col: (trials[col][0], -col))
            if trials[best][0] <= 0:
                break
            choices.append(best); available.remove(best)
            self.assertEqual(indices[slot], best)
            self.assertAlmostEqual(selected['selected'][slot]['predicted_sse_decrease'], trials[best][0], places=10)
            current -= trials[best][1]
        self.assertEqual([s['column'] for s in selected['selected']], choices)
        self.assertEqual(selected['gram_rows_computed'], len(choices))
        np.testing.assert_array_equal(factors[:, len(choices):], 0)

    def test_all_columns_finds_positive_that_symmetric_shortlist64_misses(self):
        # Xfp removes the first64 input directions. Their large W-Q errors have
        # high symmetric saliency but moving toward W worsens the asymmetric target.
        x = np.eye(65, dtype=np.float32)
        fp = np.zeros_like(x); fp[-1, -1] = 1
        w = np.full((1, 65), 2, np.float32); w[0, -1] = 1
        q = np.zeros_like(w); residual = fp@w.T-x@q.T
        old = select_columns(w, q, x, residual)
        new = select_all_columns(w, q, x, residual)
        self.assertEqual(old[2]['shortlist'], list(range(64)))
        self.assertEqual(old[2]['selected_count'], 0)
        self.assertEqual(new[2]['selected_count'], 1)
        self.assertEqual(new[0][0], 64)
        self.assertAlmostEqual(new[2]['selected'][0]['predicted_sse_decrease'], 1)

    def test_all_columns_matches_old_full_small_shortlist_and_tie_zero_finite_guards(self):
        rng = np.random.default_rng(19)
        x = bf16(rng.normal(size=(17, 6)).astype(np.float32))
        w = bf16(rng.normal(size=(3, 6)).astype(np.float32))
        q = w+bf16(rng.normal(0, .3, w.shape).astype(np.float32))
        residual = x@w.T-x@q.T
        old = select_columns(w, q, x, residual, shortlist_size=6)
        new = select_all_columns(w, q, x, residual)
        np.testing.assert_array_equal(old[0], new[0]); np.testing.assert_array_equal(old[1], new[1])
        np.testing.assert_allclose([s['gain_per_row'] for s in old[2]['selected']],
                                   [s['gain_per_row'] for s in new[2]['selected']], rtol=1e-12, atol=1e-12)
        w, q, x = np.ones((1, 3), np.float32), np.zeros((1, 3), np.float32), np.eye(3, dtype=np.float32)
        indices, factors, info = select_all_columns(w, q, x, x@w.T)
        np.testing.assert_array_equal(indices, [0, 1, 2, 0, 0, 0, 0, 0])
        self.assertEqual(info['padding'], 5)
        self.assertEqual(select_all_columns(w, w, x, np.zeros((3, 1), np.float32))[2]['selected_count'], 0)
        for args in ((w, q, x*np.nan, x@w.T), (w, q, x, np.full((3, 1), np.inf)),
                     (w, q[:, :2], x, x@w.T)):
            with self.assertRaises(ValueError): select_all_columns(*args)
        with self.assertRaises(ValueError): select_all_columns(w, q, x, x@w.T, rank=0)

    def test_actual_payload_layout_signed_bf16_scales_and_nibble_order(self):
        n, k = 2048, 6144
        groups = n*k//16
        scales = np.resize(np.array([0x3e00, 0xbe00], dtype='<u2'), groups)  # +.125, -.125
        biases = np.full(groups, 0x3e80, dtype='<u2')  # +.25
        raw = (struct.pack('<II', n, k) + bytes.fromhex('1032547698badcfe')*groups
               + scales.tobytes() + biases.tobytes())
        q = dequantize_payload(raw, n, k)
        self.assertEqual(q.shape, (n, k))
        expected = np.concatenate((np.arange(16)*.125+.25, np.arange(16)*-.125+.25))
        np.testing.assert_array_equal(q[0, :32], expected)
        np.testing.assert_array_equal(q[-1, -32:], expected)

    def test_each_gain_matches_full_residual_sse_and_correlated_update(self):
        rng = np.random.default_rng(42)
        x = bf16(rng.normal(size=(35, 7)).astype(np.float32))
        fp = x + bf16(rng.normal(0, .05, x.shape).astype(np.float32))
        w = bf16(rng.normal(size=(5, 7)).astype(np.float32))
        q = w + bf16(rng.normal(0, .2, w.shape).astype(np.float32))
        r = fp@w.T-x@q.T
        indices, factors, selection = select_columns(w, q, x, r, rank=8, shortlist_size=7)
        current = r.astype(np.float64)
        self.assertGreater(selection['selected_count'], 1)
        for step in selection['selected']:
            old = np.square(current).sum()
            current -= x[:, step['column'], None].astype(np.float64)*factors[:, step['slot']][None, :]
            actual = old-np.square(current).sum()
            self.assertAlmostEqual(actual, step['predicted_sse_decrease'], places=10)
            self.assertGreater(actual, 0)
            np.testing.assert_array_equal(factors[:, step['slot']], bf16(w[:, step['column']]-q[:, step['column']]))
        np.testing.assert_array_equal(indices[selection['selected_count']:], 0)
        np.testing.assert_array_equal(factors[:, selection['selected_count']:], 0)

    def test_correlated_error_cancellation_forbids_point_two_to_one(self):
        x = np.ones((4, 2), np.float32)
        w = np.array([[-.8, 1]], np.float32); q = np.zeros_like(w)
        r = x@w.T
        # A magnitude-only choice of column0 destroys cancellation: about .2 becomes1.
        harmful = r-x[:, :1]@bf16(w[:, :1]).T
        self.assertLess(float(np.abs(r[0, 0])), .21)
        self.assertGreater(float(np.abs(harmful[0, 0])), .99)
        indices, factors, result = select_columns(w, q, x, r)
        self.assertEqual(result['selected_count'], 0)
        self.assertEqual(result['padding'], 8)
        np.testing.assert_array_equal(factors, 0)
        np.testing.assert_array_equal(indices, 0)

    def test_ties_choose_column_index_and_padding_is_zero(self):
        x = np.eye(4, dtype=np.float32)
        w = np.ones((1, 4), np.float32); q = np.zeros_like(w)
        indices, factors, result = select_columns(w, q, x, x@w.T, rank=8, shortlist_size=2)
        self.assertEqual(result['shortlist'], [0, 1])
        self.assertEqual([s['column'] for s in result['selected']], [0, 1])
        np.testing.assert_array_equal(indices, [0, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(factors, [[1, 1, 0, 0, 0, 0, 0, 0]])

    def test_zero_and_invalid_inputs(self):
        x = np.eye(3, dtype=np.float32); w = np.ones((2, 3), np.float32)
        result = select_columns(w, w, x, np.zeros((3, 2), np.float32))
        self.assertEqual(result[2]['selected_count'], 0)
        for args in ((w, w, x*float('nan'), np.zeros((3, 2))),
                     (w, w, x, np.full((3, 2), np.inf)), (w, w[:, :2], x, np.zeros((3, 2))),
                     (w, w, x[:0], np.zeros((0, 2)))):
            with self.subTest(args=[a.shape for a in args]), self.assertRaises(ValueError):
                select_columns(*args)
        for kwargs in ({'rank': 0}, {'rank': True}, {'shortlist_size': 0}):
            with self.assertRaises(ValueError):
                select_columns(w, w, x, np.zeros((3, 2)), **kwargs)

    def test_native_staged_rounding_and_calibration_only_gate(self):
        x = np.ones((2, 1), np.float32)
        baseline = np.full((2, 1), 1.003, np.float32)
        target = np.full((2, 1), 1.008, np.float32)
        indices = np.zeros(8, np.int64)
        factors = np.zeros((1, 8), np.float32); factors[0, 0] = bf16(np.array(.005, np.float32))
        report = output_errors(target, baseline, x, indices, factors)
        expected = bf16(bf16(baseline)+x[:, indices]@factors.T)
        self.assertEqual(report['after_output_round']['candidate_sse'],
                         float(np.square(expected.astype(np.float64)-bf16(target)).sum()))
        kept_i, kept, gate = calibration_gate(indices, factors, report, {'selected_count': 1})
        self.assertTrue(gate['accepted']); np.testing.assert_array_equal(kept, factors)
        # Improvement before rounding is insufficient when the final BF16 output ties.
        factors[0, 0] = bf16(np.array(.001, np.float32))
        tied = output_errors(target, baseline, x, indices, factors)
        self.assertEqual(tied['after_output_round']['candidate_sse'], tied['after_output_round']['baseline_sse'])
        _, cleared, gate = calibration_gate(indices, factors, tied, {'selected_count': 1})
        self.assertFalse(gate['accepted']); np.testing.assert_array_equal(cleared, 0)
        _, cleared, gate = calibration_gate(indices, factors, report, {'selected_count': 0})
        self.assertFalse(gate['accepted']); np.testing.assert_array_equal(cleared, 0)

    def test_frozen_whole32_scope_cannot_switch_or_drop_layers(self):
        source = {'completed': True, 'mode': 'convert', 'accepted_layers': LAYERS, 'fixed_layers': LAYERS,
                  'layer_mask': MASK, 'mask_zero': False, 'artifact': {'quantized_projections': 32},
                  'reload': {'passed': True}, 'asymmetric_alpha': .25,
                  'asymmetric_reference': str(ROOT/'task-asymmetric-reference-v1'), 'input_rows': ROWS}
        scope = frozen_scope(source)
        self.assertEqual(len(scope), 32)
        self.assertEqual({s[0] for s in scope}, set(LAYERS))
        self.assertNotIn((23, 'down'), [(s[0], s[1]) for s in scope])
        for key, value in (('accepted_layers', LAYERS[:-1]), ('asymmetric_alpha', None),
                           ('input_rows', {'calibration': 128, 'observation': 688}), ('layer_mask', 1)):
            wrong = deepcopy(source); wrong[key] = value
            with self.assertRaises(ValueError):
                frozen_scope(wrong)


if __name__ == '__main__':
    unittest.main()
