"""Packed layout and independent numerical controls for source Q4 feedback."""
import io
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_q4_packed import (bf16, metadata, pack_codes, write_projection, decode,
                               inverse_factor, gptq_codes, gptq_quantize, materials, PROJECTIONS)
from diagnose_native_q4 import quantize


def legacy_codes(weights, scales, biases, upper, block):
    """Frozen pre-refactor static-grid algorithm, for bit-exact compatibility."""
    n, k = weights.shape
    work = torch.from_numpy(weights.copy())
    scale, bias = torch.from_numpy(scales), torch.from_numpy(biases)
    codes = torch.empty((n, k), dtype=torch.uint8)
    for begin in range(0, k, block):
        end = min(begin + block, k)
        current = work[:, begin:end].clone()
        errors = torch.empty_like(current)
        local_upper = upper[begin:end, begin:end]
        for offset in range(end - begin):
            col = begin + offset
            value = current[:, offset]
            raw = (value - bias[:, col//16]) / scale[:, col//16]
            code = torch.copysign(torch.floor(torch.abs(raw) + .5), raw).clamp_(0, 15)
            decoded = scale[:, col//16] * code + bias[:, col//16]
            error = (value - decoded) / local_upper[offset, offset]
            codes[:, col] = code.to(torch.uint8)
            current[:, offset:] -= error[:, None] * local_upper[offset, offset:][None, :]
            errors[:, offset] = error
        work[:, end:] -= errors @ upper[begin:end, end:]
    return codes.numpy()


def scalar_dynamic_reference(weights, upper):
    """Unblocked NumPy FP64 feedback; fit groups from updated weights, not a stale block."""
    work = weights.astype(np.float64)
    n, k = work.shape
    codes = np.empty((n, k), dtype=np.uint8)
    scales, biases = (np.empty((n, k//16), dtype=np.float32) for _ in range(2))
    factor = upper.double().numpy()
    for col in range(k):
        if col % 16 == 0:
            # Reuse only the preexisting signed-grid definition, not GPTQ code.
            grid = quantize(work[:, col:col+16].astype(np.float32))
            scales[:, col//16], biases[:, col//16] = grid[7][:, 0], grid[9][:, 0]
        for row in range(n):
            scale, bias = float(scales[row, col//16]), float(biases[row, col//16])
            raw = (work[row, col]-bias)/scale
            code = min(15, max(0, np.copysign(np.floor(abs(raw)+.5), raw)))
            decoded = float(np.float32(np.float32(scale)*np.float32(code)+np.float32(bias)))
            error = (work[row, col]-decoded)/factor[col, col]
            codes[row, col] = int(code)
            for following in range(col, k):
                work[row, following] -= error*factor[col, following]
    return codes, scales, biases


class PackedQ4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rng = np.random.default_rng(24)
        self.weights = bf16(self.rng.normal(0, .04, (8, 32)).astype(np.float32))

    def test_rtn_reuses_original_signed_grid_and_low_nibble_layout(self):
        rtn, nearest, scales, biases = metadata(self.weights)
        expected = quantize(self.weights.reshape(-1, 16))
        np.testing.assert_array_equal(decode(rtn, scales, biases), expected[0].reshape(8, 32))
        np.testing.assert_array_equal(decode(nearest, scales, biases), expected[1].reshape(8, 32))
        self.assertTrue(np.any(scales < 0))
        stream = io.BytesIO()
        write_projection(stream, rtn, scales, biases)
        raw = stream.getvalue()
        self.assertEqual(struct.unpack('<II', raw[:8]), (8, 32))
        words = np.frombuffer(raw[8:8+128], dtype='<u4')
        unpacked = ((words[:, None] >> (np.arange(8)*4)) & 15).reshape(8, 32)
        np.testing.assert_array_equal(unpacked, rtn)
        stored = (np.frombuffer(raw[136:], dtype='<u2').astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(stored[:16].reshape(8,2), scales)
        np.testing.assert_array_equal(stored[16:].reshape(8,2), biases)

    def test_diagonal_hessian_is_exact_stored_grid_control(self):
        _, nearest, scales, biases = metadata(self.weights)
        got = gptq_codes(self.weights, scales, biases, torch.eye(32), 16)
        np.testing.assert_array_equal(got, nearest)

    def test_quantize_default_is_bit_exact_legacy_and_owns_grids(self):
        weights = bf16(self.rng.normal(0, .04, (4, 160)).astype(np.float32))
        upper, _ = inverse_factor(self.rng.normal(size=(192, 160)).astype(np.float32))
        _, _, scales, biases = metadata(weights)
        for block in (16, 32, 128):
            codes, s, b = gptq_quantize(weights, scales, biases, upper, block)
            np.testing.assert_array_equal(codes, legacy_codes(weights, scales, biases, upper, block))
            np.testing.assert_array_equal(codes, gptq_codes(weights, scales, biases, upper, block))
            np.testing.assert_array_equal(s, scales)
            np.testing.assert_array_equal(b, biases)
            self.assertFalse(np.shares_memory(s, scales))
            self.assertFalse(np.shares_memory(b, biases))

    def test_refit_diagonal_factor_keeps_original_grid_and_codes(self):
        _, nearest, scales, biases = metadata(self.weights)
        codes, s, b = gptq_quantize(self.weights, scales, biases, torch.eye(32), 32, refit_grid=True)
        np.testing.assert_array_equal(codes, nearest)
        np.testing.assert_array_equal(s, scales)
        np.testing.assert_array_equal(b, biases)

    def test_refit_current_groups_and_cross_block_match_scalar_reference(self):
        weights = bf16(self.rng.normal(0, .04, (4, 160)).astype(np.float32))
        upper, _ = inverse_factor(self.rng.normal(size=(192, 160)).astype(np.float32))
        _, _, scales, biases = metadata(weights)
        original = [value.copy() for value in (weights, scales, biases)]
        reference = scalar_dynamic_reference(weights, upper)
        self.assertTrue(np.any(reference[1][:, 1:8] != scales[:, 1:8]))
        self.assertTrue(np.any(reference[1][:, 8:] != scales[:, 8:]))
        for block in (32, 128):
            actual = gptq_quantize(weights, scales, biases, upper, block, refit_grid=True)
            for got, expected in zip(actual, reference):
                np.testing.assert_array_equal(got, expected)
            for got, expected in zip((weights, scales, biases), original):
                np.testing.assert_array_equal(got, expected)
        codes, s, b = actual
        np.testing.assert_array_equal(s, bf16(s))
        np.testing.assert_array_equal(b, bf16(b))
        stream = io.BytesIO()
        write_projection(stream, codes, s, b)
        raw = stream.getvalue()
        n, k = struct.unpack('<II', raw[:8])
        self.assertEqual((n, k), weights.shape)
        end = 8+n*k//2
        words = np.frombuffer(raw[8:end], dtype='<u4')
        unpacked = ((words[:, None] >> (np.arange(8)*4)) & 15).reshape(n, k)
        grid = (np.frombuffer(raw[end:], dtype='<u2').astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(unpacked, codes)
        np.testing.assert_array_equal(grid[:s.size].reshape(s.shape), s)
        np.testing.assert_array_equal(grid[s.size:].reshape(b.shape), b)

    def test_refit_rejects_nonfinite_source_grid_factor_and_feedback(self):
        _, _, s, b = metadata(self.weights)
        for index in range(3):
            inputs = [self.weights.copy(), s.copy(), b.copy()]
            inputs[index].flat[0] = np.nan
            with self.subTest(index=index), self.assertRaises(ValueError):
                gptq_quantize(*inputs, torch.eye(32), refit_grid=True)
        for bad in (float('nan'), float('inf'), 0):
            upper = torch.eye(32); upper[0, 0] = bad
            with self.subTest(diagonal=bad), self.assertRaises(ValueError):
                gptq_quantize(self.weights, s, b, upper, refit_grid=True)
        weights = np.full((2, 16), 1e20, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'nonfinite GPTQ block'):
            gptq_quantize(weights, np.ones((2, 1), np.float32), np.zeros((2, 1), np.float32),
                          torch.eye(16)*1e-30, 16, refit_grid=True)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            weights[:, 0] = np.finfo(np.float32).max
            weights[:, 1] = -np.finfo(np.float32).max
            with self.assertRaisesRegex(ValueError, 'refitted GPTQ grid'):
                gptq_quantize(weights, np.ones((2, 1), np.float32), np.zeros((2, 1), np.float32),
                              torch.eye(16), 16, refit_grid=True)

    def test_blocked_feedback_matches_independent_scalar_reference(self):
        x = self.rng.normal(size=(80,32)).astype(np.float32)
        upper, _ = inverse_factor(x)
        _, _, scales, biases = metadata(self.weights)
        work = self.weights.astype(np.float64)
        reference = np.empty(work.shape, dtype=np.uint8)
        u = upper.double().numpy()
        for column in range(32):
            raw = (work[:,column]-biases[:,column//16])/scales[:,column//16]
            code = np.clip(np.copysign(np.floor(np.abs(raw)+.5),raw),0,15)
            q = scales[:,column//16]*code+biases[:,column//16]
            error = (work[:,column]-q)/u[column,column]
            reference[:,column] = code.astype(np.uint8)
            work[:,column:] -= error[:,None]*u[column,column:]
        np.testing.assert_array_equal(gptq_codes(self.weights,scales,biases,upper,16),reference)

    def test_correlated_calibration_feedback_reduces_projection_error(self):
        x = (self.rng.normal(size=(128,4)) @ self.rng.normal(size=(4,32))
             + .03*self.rng.normal(size=(128,32))).astype(np.float32)
        upper,_ = inverse_factor(x)
        _,nearest,s,b = metadata(self.weights)
        codes = gptq_codes(self.weights,s,b,upper,16)
        original = np.mean((x @ (self.weights-decode(nearest,s,b)).T)**2)
        candidate = np.mean((x @ (self.weights-decode(codes,s,b)).T)**2)
        self.assertLess(candidate, original*.5)
        np.testing.assert_array_equal(codes,gptq_codes(self.weights,s,b,upper,32))

    def test_unobserved_columns_keep_original_stored_grid(self):
        x=self.rng.normal(size=(64,32)).astype(np.float32);x[:,[0,5,31]]=0
        upper,report=inverse_factor(x)
        _,nearest,s,b=metadata(self.weights)
        got=gptq_codes(self.weights,s,b,upper,16)
        self.assertEqual(report['dead_columns'],[0,5,31])
        np.testing.assert_array_equal(got[:,[0,5,31]],nearest[:,[0,5,31]])

    def test_default_damping_preserves_legacy_factor_exactly(self):
        x = self.rng.normal(size=(80, 32)).astype(np.float32)
        tensor = torch.from_numpy(x)
        hessian = (tensor.T @ tensor / len(x)).double()
        damping = .01 * float(hessian.diagonal().mean())
        hessian.diagonal().add_(damping)
        legacy = torch.linalg.cholesky(torch.cholesky_inverse(
            torch.linalg.cholesky(hessian)), upper=True).float().triu_()
        default, report = inverse_factor(x)
        explicit, explicit_report = inverse_factor(x, damping_fraction=.01)
        self.assertTrue(torch.equal(default, legacy))
        self.assertTrue(torch.equal(default, explicit))
        self.assertEqual(report, explicit_report)
        self.assertEqual(report['damping_fraction'], .01)
        self.assertEqual(report['damping'], damping)

    def test_stronger_damping_matches_direct_rank_deficient_inverse(self):
        x = self.rng.normal(size=(12, 32)).astype(np.float32)
        x[:, 5] = 0
        upper, report = inverse_factor(x, damping_fraction=.1)
        # Reproduce only the specified FP32 Gram accumulation; solve the
        # regularized inverse independently with NumPy, without Cholesky.
        gram = (torch.from_numpy(x).T @ torch.from_numpy(x) / len(x)).double().numpy()
        damping = .1 * float(np.trace(gram) / gram.shape[0])
        expected = np.linalg.inv(gram + damping * np.eye(32))
        u = upper.double().numpy()
        np.testing.assert_allclose(u.T @ u, expected, rtol=3e-6, atol=3e-6)
        self.assertEqual(report['damping_fraction'], .1)
        self.assertAlmostEqual(report['damping'], damping)
        self.assertEqual(report['dead_columns'], [5])
        boundary, boundary_report = inverse_factor(x, damping_fraction=1)
        self.assertTrue(torch.isfinite(boundary).all())
        self.assertEqual(boundary_report['damping_fraction'], 1.)

    def test_invalid_damping_rejected_before_factorization(self):
        x = self.rng.normal(size=(12, 32)).astype(np.float32)
        for value in (0, -1, 1.0001, float('nan'), float('inf'), -float('inf'),
                      True, False, '0.1', None, .1j, [.1]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'damping fraction'):
                inverse_factor(x, damping_fraction=value)

    def test_invalid_metadata_codes_and_empty_energy_rejected(self):
        rtn,_,s,b=metadata(self.weights)
        with self.assertRaises(ValueError):pack_codes(np.full((1,16),16))
        with self.assertRaises(ValueError):pack_codes(np.full((1,16),.5))
        with self.assertRaises(ValueError):inverse_factor(np.zeros((16,32),np.float32))
        with self.assertRaises(ValueError):write_projection(io.BytesIO(),rtn,s+np.float32(1e-8),b)
        s[0,0]=0
        with self.assertRaises(ValueError):write_projection(io.BytesIO(),rtn,s,b)

    def test_protocol_covers_71_projections_and_disjoint_frozen_texts(self):
        self.assertEqual(len(PROJECTIONS),71)
        self.assertNotIn((23,'down',2048,6144),PROJECTIONS)
        rows=materials()
        self.assertEqual(len(rows),20)
        self.assertEqual(len({row['id'] for row in rows}),20)
        self.assertEqual(len({row['text'] for row in rows}),20)
        self.assertEqual(sum(row['split']=='calibration' for row in rows),16)

    def test_last_block_overflow_cannot_be_hidden_by_uint8_codes(self):
        weights = np.full((2,16), 1e20, dtype=np.float32)
        scales = np.ones((2,1), dtype=np.float32)
        biases = np.zeros_like(scales)
        with self.assertRaisesRegex(ValueError, 'nonfinite GPTQ block'):
            gptq_codes(weights,scales,biases,torch.eye(16)*1e-30,16)


if __name__ == '__main__':
    unittest.main()
