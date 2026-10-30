"""CPU checks for legacy bytes, sparse layer ordering and failed-carrier cleanup."""
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from q4_packed_format import (ALL_LAYERS, PAYLOAD_BYTES, encode_header, projections,
                             validate_projection, write_sparse, write_sparse_protected, protected_factors)
from prepare_q4_packed import PROJECTIONS, write_projection

MANIFEST = 'a' * 64
CORE = '0' * 64


class SparsePackedFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real dimensions, both signed scales; no model weights or GPU.
        raw = bytearray(8 + PAYLOAD_BYTES)
        struct.pack_into('<II', raw, 0, 6144, 2048)
        groups = 6144 * 2048 // 16
        scale = np.frombuffer(raw, dtype='<u2', count=groups, offset=8 + 6144 * 2048 // 2)
        scale[::2] = 0x3f80
        scale[1::2] = 0xbf80
        cls.payload = bytes(raw)

    def test_v1_exact_bytes_and_original_projection_order_unchanged(self):
        self.assertEqual(encode_header(MANIFEST, CORE),
                         b'LQ4PK01\n' + struct.pack('<II', 16, 71) + MANIFEST.encode() + CORE.encode())
        self.assertEqual(projections(ALL_LAYERS), PROJECTIONS)
        # Existing writer remains bit-compatible with the new pure validator.
        codes = np.zeros((6144, 2048), dtype=np.uint8)
        scales = np.ones((6144, 128), dtype=np.float32)
        scales.reshape(-1)[1::2] = -1
        stream = io.BytesIO()
        write_projection(stream, codes, scales, np.zeros_like(scales))
        self.assertEqual(stream.getvalue(), self.payload)
        validate_projection(stream.getvalue(), 6144, 2048)

    def test_v2_counts_every_single_layer_and_sparse_order(self):
        for layer in range(24):
            mask = 1 << layer
            raw = encode_header(MANIFEST, CORE, mask)
            self.assertEqual(len(raw), 148)
            self.assertEqual(raw[:8], b'LQ4PK02\n')
            self.assertEqual(struct.unpack_from('<II', raw, 8), (16, 2 if layer == 23 else 3))
            self.assertEqual(struct.unpack_from('<I', raw, 144), (mask,))
        got = projections(1 | (1 << 7) | (1 << 23))
        self.assertEqual([(l, k) for l, k, _, _ in got],
                         [(l, k) for l in (0, 7, 23) for k in ('gate', 'up', 'down') if (l, k) != (23, 'down')])

    def test_invalid_mask_and_source_rejected(self):
        for mask in (0, -1, 1 << 24, 0xffffffff, True, 1.5):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                encode_header(MANIFEST, CORE, mask)
        for source in ('A' * 64, 'g' * 64, 'a' * 63, None):
            with self.assertRaises(ValueError): encode_header(source, CORE, 1)

    def test_projection_rejects_shape_length_zero_nonfinite_and_endpoint_overflow(self):
        for raw in (self.payload[:-1], self.payload + b'x'):
            with self.assertRaises(ValueError): validate_projection(raw, 6144, 2048)
        with self.assertRaises(ValueError): validate_projection(self.payload, 2048, 6144)
        offset = 8 + 6144 * 2048 // 2
        for value in (0, 0x8000, 0x7f80, 0x7fc0, 0x7f7f):
            raw = bytearray(self.payload)
            struct.pack_into('<H', raw, offset, value)
            with self.subTest(value=value), self.assertRaises(ValueError): validate_projection(raw, 6144, 2048)
        raw = bytearray(self.payload)
        struct.pack_into('<H', raw, offset + (6144 * 2048 // 16) * 2, 0x7f80)
        with self.assertRaises(ValueError): validate_projection(raw, 6144, 2048)

    def test_sparse_writer_exact_eof_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'candidate.bin'
            rows = [(23, k, self.payload) for k in ('gate', 'up')]
            result = write_sparse(path, MANIFEST, CORE, 1 << 23, iter(rows))
            self.assertEqual(result['quantized_projections'], 2)
            self.assertEqual(path.stat().st_size, 148 + 2 * len(self.payload))
            with path.open('rb') as stream:
                self.assertEqual(stream.read(148), encode_header(MANIFEST, CORE, 1 << 23))
                for row in rows: self.assertEqual(stream.read(len(self.payload)), row[2])
                self.assertEqual(stream.read(1), b'')
            with self.assertRaises(FileExistsError): write_sparse(path, MANIFEST, CORE, 1 << 23, rows)
            self.assertEqual(path.stat().st_size, result['bytes'])

    def test_rejected_extra_missing_or_wrong_order_removes_only_new_output(self):
        rows = [(23, k, self.payload) for k in ('gate', 'up')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'candidate.bin'
            for invalid in (rows[:1], rows[::-1], rows + [rows[0]], rows + [None],
                            [(22, 'gate', self.payload), rows[1]],
                            [(23, 'gate', self.payload[:-2]), rows[1]]):
                with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                    write_sparse(path, MANIFEST, CORE, 1 << 23, invalid)
                self.assertFalse(path.exists())

    def test_v3_header_paired_bytes_both_shapes_and_zero_control(self):
        rows = []
        for layer, kind, n, k in projections(1):
            raw = struct.pack('<II', n, k)+self.payload[8:]
            b = np.zeros((8, k), np.float32); b[0, 17] = 1
            u = np.zeros((n, 8), np.float32); u[:, 0] = -.25
            rows.append((layer, kind, raw, b, u))
        with tempfile.TemporaryDirectory() as tmp:
            candidate, zero = Path(tmp)/'candidate.bin', Path(tmp)/'zero.bin'
            result = write_sparse_protected(candidate, MANIFEST, CORE, 1, iter(rows))
            write_sparse_protected(zero, MANIFEST, CORE, 1,
                                   ((l, k, raw, b, np.zeros_like(u)) for l, k, raw, b, u in rows))
            self.assertEqual(result['rank'], 8)
            self.assertEqual(result['quantized_projections'], 3)
            self.assertEqual(candidate.stat().st_size, 152+3*(len(self.payload)+8*(6144+2048)*2))
            with candidate.open('rb') as a, zero.open('rb') as z:
                header = b'LQ4PK03\n'+encode_header(MANIFEST, CORE, 1)[8:]+struct.pack('<I', 8)
                self.assertEqual(a.read(152), header); self.assertEqual(z.read(152), header)
                for _, _, raw, b, u in rows:
                    self.assertEqual(a.read(len(raw)), raw); self.assertEqual(z.read(len(raw)), raw)
                    b_bytes, u_bytes = protected_factors(b, u, u.shape[0], b.shape[1])
                    self.assertEqual(a.read(len(b_bytes)), b_bytes); self.assertEqual(z.read(len(b_bytes)), b_bytes)
                    self.assertEqual(a.read(len(u_bytes)), u_bytes); self.assertEqual(z.read(len(u_bytes)), bytes(len(u_bytes)))
                self.assertEqual(a.read(1), b''); self.assertEqual(z.read(1), b'')
            with self.assertRaises(FileExistsError):
                write_sparse_protected(candidate, MANIFEST, CORE, 1, rows)
            self.assertEqual(candidate.stat().st_size, result['bytes'])

    def test_v3_factor_shape_finite_onehot_unique_padding_and_exact_bf16(self):
        b = np.zeros((8, 2048), np.float32); b[0, 3] = 1
        u = np.zeros((6144, 8), np.float32); u[:, 0] = .5
        protected_factors(b, u, 6144, 2048)
        invalid = [(b[:7], u), (b, u[:, :7]), (b*.5, u)]
        for key in ('two_hot', 'duplicate', 'zero_basis_update', 'nan', 'non_bf16'):
            bad_b, bad_u = b.copy(), u.copy()
            if key == 'two_hot': bad_b[0, 4] = 1
            elif key == 'duplicate': bad_b[1, 3] = 1
            elif key == 'zero_basis_update': bad_u[0, 1] = .5
            elif key == 'nan': bad_u[0, 0] = np.nan
            else: bad_u[0, 0] = .1001
            invalid.append((bad_b, bad_u))
        for bad_b, bad_u in invalid:
            with self.assertRaises(ValueError): protected_factors(bad_b, bad_u, 6144, 2048)

    def test_v3_incomplete_wrong_scope_or_bad_factor_removes_only_new_output(self):
        b = np.zeros((8, 2048), np.float32); u = np.zeros((6144, 8), np.float32)
        rows = [(23, k, self.payload, b, u) for k in ('gate', 'up')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'candidate.bin'
            for invalid in (rows[:1], rows[::-1], rows+[rows[0]],
                            [rows[0], (23, 'down', self.payload, b, u)],
                            [rows[0], (23, 'up', self.payload, b, np.ones_like(u))]):
                with self.assertRaises(ValueError):
                    write_sparse_protected(path, MANIFEST, CORE, 1 << 23, invalid)
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
