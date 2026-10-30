#!/usr/bin/env python3
"""Read-only row-sampled BF16 weight diagnostics; no model forward, GPU, hashes or writes to weights."""
import argparse
import json
import os
import platform
import struct
import time
from pathlib import Path

import numpy as np


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def away(values):
    return np.copysign(np.floor(np.abs(values) + np.float32(0.5)), values)


def quantize(values):
    """Mirror core/affine4.zig: zero-inclusive upper, signed endpoint, zero alignment, FP32 codes."""
    lower = values.min(axis=1, keepdims=True)
    upper = np.maximum(values.max(axis=1, keepdims=True), np.float32(0))
    bias = np.where(np.abs(lower) > np.abs(upper), lower, upper)
    scale = np.maximum((upper - lower) / np.float32(15), np.float32(1e-7))
    scale = np.where(bias >= 0, -scale, scale)
    zero = away(bias / scale)
    scale = np.divide(bias, zero, out=scale.copy(), where=zero != 0)
    bias = np.where(zero == 0, np.float32(0), bias)
    codes = np.clip(away((values - bias) / scale), 0, 15)
    stored_scale, stored_bias = bf16(scale), bf16(bias)
    current = stored_scale * codes + stored_bias
    nearest_codes = np.clip(away((values - stored_bias) / stored_scale), 0, 15)
    nearest = stored_scale * nearest_codes + stored_bias
    ideal = scale * codes + bias
    q8_scale = bf16(np.maximum(np.abs(values).max(axis=1, keepdims=True) / np.float32(127), np.float32(1e-30)))
    q8 = q8_scale * np.clip(away(values / q8_scale), -127, 127)
    return current, nearest, ideal, q8, codes, nearest_codes, scale, stored_scale, bias, stored_bias


def self_check():
    # Exact source tests: constant +1, -1 and tiny positive groups have all zero codes.
    values = bf16(np.repeat(np.array([1, -1, 1e-20], np.float32), 64).reshape(3, 64))
    result = quantize(values)
    assert np.all(result[4] == 0)
    assert (result[7].view(np.uint32).ravel() >> 16).tolist()[:2] == [0xb3d7, 0x3d89]
    assert result[9][2, 0] == 0
    values = bf16(np.random.default_rng(42).normal(0, 0.04, (256, 64)).astype(np.float32))
    current, nearest, *_ = quantize(values)
    assert np.all((nearest - values) ** 2 <= (current - values) ** 2 + 1e-12)


def selected(name, metadata):
    if not name.startswith('model.language_model.') or metadata.get('dtype') != 'BF16':
        return False
    if len(metadata.get('shape', [])) != 2 or metadata['shape'][1] % 64:
        return False
    if name.endswith(('in_proj_a.weight', 'in_proj_b.weight')):
        return False
    return not name.endswith('layers.23.mlp.down_proj.weight')


def metrics(values, outputs):
    energy = float(np.square(values.astype(np.float64)).sum())
    result = {'elements': int(values.size), 'squared_weight_sum': energy}
    for key, output in outputs.items():
        errors = output.astype(np.float64) - values
        sse = float(np.square(errors).sum())
        result[key] = {'squared_error_sum': sse, 'mse': sse / values.size,
                       'relative_mse': sse / energy if energy else 0,
                       'max_absolute_error': float(np.abs(errors).max()),
                       'exact_weight_fraction': float(np.mean(output == values))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.rows <= 128:
        raise ValueError('rows must be 1..128')
    self_check()
    start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    records, group_records = [], []
    bytes_read = 0
    for path in sorted(args.model.glob('*.safetensors')):
        if path.name.startswith('.'):
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            header_size = struct.unpack('<Q', os.pread(fd, 8, 0))[0]
            if header_size > 16 << 20:
                raise ValueError('unexpected safetensors header size')
            header = json.loads(os.pread(fd, header_size, 8))
            bytes_read += 8 + header_size
            for name, metadata in sorted(header.items()):
                if not selected(name, metadata):
                    continue
                n, k = metadata['shape']
                rows = np.sort(rng.choice(n, min(args.rows, n), replace=False))
                offset = 8 + header_size + metadata['data_offsets'][0]
                raw = [os.pread(fd, k * 2, offset + int(row) * k * 2) for row in rows]
                if any(len(row) != k * 2 for row in raw):
                    raise ValueError('short tensor row read')
                bytes_read += len(rows) * k * 2
                values = (np.frombuffer(b''.join(raw), dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(-1, 64)
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite original weight')
                current, nearest, ideal, q8, codes, new_codes, scale, stored_scale, bias, stored_bias = quantize(values)
                record = {'tensor': name, 'shape': [n, k], 'sampled_rows': rows.tolist(),
                          **metrics(values, {'q4_current': current, 'q4_nearest_stored_metadata': nearest,
                                            'q4_fp32_metadata': ideal, 'q8_symmetric_bf16_scale': q8}),
                          'changed_codes': int(np.count_nonzero(codes != new_codes)),
                          'bias_rounding_changed_groups': int(np.count_nonzero(bias != stored_bias)),
                          'scale_rounding_changed_groups': int(np.count_nonzero(scale != stored_scale)),
                          'groups': len(values)}
                current_sse = record['q4_current']['squared_error_sum']
                record['nearest_stored_mse_improvement_fraction'] = 1 - record['q4_nearest_stored_metadata']['squared_error_sum'] / current_sse if current_sse else 0
                rms = np.sqrt(np.square(values.astype(np.float64)).mean(axis=1))
                outlier = np.divide(np.abs(values).max(axis=1), rms, out=np.zeros_like(rms), where=rms != 0)
                sse = np.square(current.astype(np.float64) - values).sum(axis=1)
                energy = np.square(values.astype(np.float64)).sum(axis=1)
                for ratio, error, norm in zip(outlier, sse, energy):
                    group_records.append((float(ratio), float(error), float(norm)))
                record['max_abs_to_group_rms'] = {'median': float(np.median(outlier)), 'p95': float(np.quantile(outlier, 0.95)), 'max': float(outlier.max())}
                records.append(record)
        finally:
            os.close(fd)
    if not records:
        raise ValueError('no selected original BF16 tensors')
    elements = sum(r['elements'] for r in records)
    energy = sum(r['squared_weight_sum'] for r in records)
    aggregate = {'tensors': len(records), 'elements': elements, 'groups': sum(r['groups'] for r in records),
                 'changed_codes': sum(r['changed_codes'] for r in records),
                 'bias_rounding_changed_groups': sum(r['bias_rounding_changed_groups'] for r in records)}
    for method in ['q4_current', 'q4_nearest_stored_metadata', 'q4_fp32_metadata', 'q8_symmetric_bf16_scale']:
        sse = sum(r[method]['squared_error_sum'] for r in records)
        aggregate[method] = {'mse': sse / elements, 'relative_mse': sse / energy, 'squared_error_sum': sse}
    aggregate['nearest_stored_mse_improvement_fraction'] = 1 - aggregate['q4_nearest_stored_metadata']['squared_error_sum'] / aggregate['q4_current']['squared_error_sum']
    groups = np.array(group_records)
    for label, mask in [('max_abs_over_4rms', groups[:, 0] >= 4), ('max_abs_below_4rms', groups[:, 0] < 4)]:
        subset = groups[mask]
        aggregate[label] = {'groups': len(subset), 'group_fraction': float(mask.mean()),
                            'relative_mse': float(subset[:, 1].sum() / subset[:, 2].sum()) if len(subset) else None}
    report = {'protocol': 'q4-bf16-rows-readonly-v1', 'model': str(args.model), 'seed': args.seed,
              'rows_per_tensor': args.rows, 'numpy': np.__version__, 'hardware': platform.machine(),
              'elapsed_seconds': time.perf_counter() - start, 'file_bytes_read': bytes_read,
              'self_checks_passed': True, 'training': False, 'gpu_used': False,
              'scope': 'Original checkpoint row samples only; down23 and delta a/b excluded. Weight error is not evidence of which error caused answer regressions.',
              'rounding': 'BF16 nearest ties even; code selection nearest ties away; FP32 dequantized weight error before activation/input-dependent propagation.',
              'sampling_limitation': 'Equal sampled rows per tensor; aggregate is sample-weighted, not full-model parameter-weighted. No activation calibration or full-model answer comparison.',
              'memory_measurement': 'Not measured; one sampled tensor at a time. No full model or full tensor load.',
              'aggregate': aggregate, 'tensors': records}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'result.json'
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'aggregate': aggregate}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
