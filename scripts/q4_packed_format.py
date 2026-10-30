"""Pure source-bound packed Q4 headers and sparse-layer carrier writing."""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

ALL_LAYERS = (1 << 24) - 1
PAYLOAD_BYTES = 2048 * 6144 * 3 // 4
PROTECTED_RANK = 8


def projections(layer_mask):
    if type(layer_mask) is not int or not 0 < layer_mask <= ALL_LAYERS:
        raise ValueError('nonzero 24-bit layer mask required')
    return [(layer, kind, 2048 if kind == 'down' else 6144,
             6144 if kind == 'down' else 2048)
            for layer in range(24) if layer_mask & (1 << layer)
            for kind in ('gate', 'up', 'down') if (layer, kind) != (23, 'down')]


def encode_header(manifest, core, layer_mask=None):
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
           for value in (manifest, core)):
        raise ValueError('exact lowercase source bindings required')
    mask = ALL_LAYERS if layer_mask is None else layer_mask
    count = len(projections(mask))
    return ((b'LQ4PK01\n' if layer_mask is None else b'LQ4PK02\n')
            + struct.pack('<II', 16, count) + manifest.encode('ascii') + core.encode('ascii')
            + (b'' if layer_mask is None else struct.pack('<I', mask)))


def validate_projection(raw, n, k):
    if ((n, k) not in {(6144, 2048), (2048, 6144)} or len(raw) != 8 + PAYLOAD_BYTES
            or struct.unpack_from('<II', raw) != (n, k)):
        raise ValueError('exact projection shape and payload length required')
    words = n * k // 2
    groups = n * k // 16
    scales = np.frombuffer(raw, dtype='<u2', count=groups, offset=8 + words)
    biases = np.frombuffer(raw, dtype='<u2', count=groups, offset=8 + words + groups * 2)
    if (np.any((scales & 0x7fff) == 0) or np.any((scales & 0x7f80) == 0x7f80)
            or np.any((biases & 0x7f80) == 0x7f80)):
        raise ValueError('finite nonzero signed BF16 scale and finite bias required')
    s = (scales.astype(np.uint32) << 16).view(np.float32)
    b = (biases.astype(np.uint32) << 16).view(np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        if not np.isfinite(s * np.float32(15) + b).all():
            raise ValueError('decoded FP32 endpoint overflow')


def write_sparse(path, manifest, core, layer_mask, payloads):
    """Write only accepted layers from ordered (layer, kind, dimensions+payload).

    Existing paths are never overwritten. Any failure removes only this call's
    new partial output, so a rejected extra/missing projection cannot be loaded.
    """
    header = encode_header(manifest, core, layer_mask)
    expected = projections(layer_mask)
    path = Path(path)
    stream = path.open('xb')
    try:
        with stream:
            stream.write(header)
            iterator = iter(payloads)
            for layer, kind, n, k in expected:
                try:
                    actual_layer, actual_kind, raw = next(iterator)
                except StopIteration as error:
                    raise ValueError('missing sparse projection') from error
                if type(actual_layer) is not int or (actual_layer, actual_kind) != (layer, kind):
                    raise ValueError('sparse projection order/layer mismatch')
                validate_projection(raw, n, k)
                stream.write(raw)
            end = object()
            if next(iterator, end) is not end:
                raise ValueError('trailing sparse projection')
            size = stream.tell()
            if size != len(header) + len(expected) * (8 + PAYLOAD_BYTES):
                raise ValueError('sparse carrier length mismatch')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {'layer_mask': layer_mask, 'quantized_projections': len(expected), 'bytes': size}


def protected_factors(basis, update, n, k):
    """Exact stored BF16 factors with unique one-hot (or padded zero) basis rows."""
    b, u = (np.asarray(value, dtype=np.float32) for value in (basis, update))
    if (b.shape != (PROTECTED_RANK, k) or u.shape != (n, PROTECTED_RANK)
            or not np.isfinite(b).all() or not np.isfinite(u).all()
            or not np.array_equal(basis, b) or not np.array_equal(update, u)
            or np.any((b != 0) & (b != 1))
            or np.any(u.view(np.uint32) & 0xffff)):
        raise ValueError('finite exact BF16 B[8,k] and U[n,8] with binary basis required')
    counts = np.count_nonzero(b, axis=1)
    active = counts == 1
    if (np.any(counts > 1) or len(set(np.argmax(b[active], axis=1).tolist())) != int(active.sum())
            or np.any(u[:, ~active] != 0)):
        raise ValueError('unique one-hot basis rows and zero updates for padded rows required')
    return tuple((value.view(np.uint32) >> 16).astype('<u2').tobytes(order='C') for value in (b, u))


def write_sparse_protected(path, manifest, core, layer_mask, records):
    """Write paired (layer, kind, raw Q4, B, U) records as rank8 packed-v3."""
    header = b'LQ4PK03\n' + encode_header(manifest, core, layer_mask)[8:] + struct.pack('<I', PROTECTED_RANK)
    expected = projections(layer_mask)
    path = Path(path)
    stream = path.open('xb')
    try:
        with stream:
            stream.write(header)
            iterator = iter(records)
            expected_size = len(header)
            for layer, kind, n, k in expected:
                try:
                    actual_layer, actual_kind, raw, basis, update = next(iterator)
                except StopIteration as error:
                    raise ValueError('missing protected projection') from error
                if type(actual_layer) is not int or (actual_layer, actual_kind) != (layer, kind):
                    raise ValueError('protected projection order/layer mismatch')
                validate_projection(raw, n, k)
                b, u = protected_factors(basis, update, n, k)
                stream.write(raw); stream.write(b); stream.write(u)
                expected_size += 8 + PAYLOAD_BYTES + (k+n)*PROTECTED_RANK*2
            sentinel = object()
            if next(iterator, sentinel) is not sentinel:
                raise ValueError('trailing protected projection')
            size = stream.tell()
            if size != expected_size:
                raise ValueError('protected carrier length mismatch')
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {'layer_mask': layer_mask, 'quantized_projections': len(expected),
            'rank': PROTECTED_RANK, 'bytes': size, 'format': 'LQ4PK03'}
