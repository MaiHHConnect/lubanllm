#!/usr/bin/env python3
"""Isolated synthetic group64 clipping study; no model, training or GPU access."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

from diagnose_native_q4 import bf16, quantize, self_check


def search(weight, inputs):
    weight = np.asarray(weight, dtype=np.float32)
    inputs = np.asarray(inputs, dtype=np.float32)
    if (weight.shape != (64,) or inputs.ndim != 2 or inputs.shape[1] != 64
            or not len(inputs) or not np.isfinite(weight).all() or not np.isfinite(inputs).all()):
        raise ValueError('expected finite weight[64] and nonempty inputs[T,64]')
    if not np.array_equal(weight, bf16(weight)):
        raise ValueError('study requires original BF16 weight values')
    records, candidates = [], []
    for ratio in np.linspace(1, .5, 20, dtype=np.float32):
        # Match the existing BF16-source prepare entry; no replacement quantizer.
        clipped = bf16(np.clip(weight, weight.min() * ratio, weight.max() * ratio))
        result = quantize(clipped[None, :])
        output, codes, scale, bias = result[0][0], result[4][0], result[7][0], result[9][0]
        if not np.isfinite(output).all() or not np.isfinite(scale).all() or np.any(scale == 0):
            raise ValueError('invalid stored affine metadata')
        # Verify nibble order and reconstruction using the stored BF16 metadata.
        shifts = np.arange(8, dtype=np.uint32) * 4
        words = np.bitwise_or.reduce(codes.astype(np.uint32).reshape(8, 8) << shifts, axis=1)
        unpacked = ((words[:, None] >> shifts) & 15).reshape(64)
        np.testing.assert_array_equal(unpacked, codes)
        np.testing.assert_array_equal(unpacked.astype(np.float32) * scale + bias, output)
        error = inputs.astype(np.float64) @ (weight.astype(np.float64) - output)
        mse = float(np.mean(error ** 2))
        if not np.isfinite(mse):
            raise ValueError('nonfinite objective')
        records.append({'range_ratio': float(ratio), 'activation_mse': mse})
        candidates.append(output)
    best = int(np.argmin([r['activation_mse'] for r in records]))
    assert records[best]['activation_mse'] <= records[0]['activation_mse']
    return candidates[0], candidates[best], best, records


def run():
    start = time.perf_counter()
    self_check()
    rng = np.random.default_rng(42)
    weight = bf16(rng.normal(0, .15, 64).astype(np.float32))
    weight[0] = 2
    train = bf16(rng.normal(size=(512, 64)).astype(np.float32))
    heldout = bf16(rng.normal(size=(512, 64)).astype(np.float32))
    inactive_train, inactive_heldout = train.copy(), heldout.copy()
    inactive_train[:, 0] = bf16(inactive_train[:, 0] * .02)
    inactive_heldout[:, 0] = bf16(inactive_heldout[:, 0] * .02)
    cases = [
        ('low_activation_outlier', weight, inactive_train, inactive_heldout),
        ('active_outlier', weight, train, heldout),
        ('distribution_shift', weight, inactive_train, heldout),
        ('zero_weights', np.zeros(64, np.float32), train, heldout),
        ('constant_weights', np.ones(64, np.float32), train, heldout),
    ]
    results = []
    for name, w, calibration, validation in cases:
        baseline, chosen, best, grid = search(w, calibration)
        def mse(q):
            return float(np.mean((validation.astype(np.float64) @ (w.astype(np.float64) - q)) ** 2))
        before, after = mse(baseline), mse(chosen)
        results.append({'case': name, 'selected_index': best, 'grid': grid,
                        'heldout_baseline_mse': before, 'heldout_clipped_mse': after,
                        'heldout_improvement_fraction': 1 - after / before if before else None})
    assert results[0]['heldout_clipped_mse'] < results[0]['heldout_baseline_mse']
    assert results[2]['heldout_clipped_mse'] > results[2]['heldout_baseline_mse']
    for bad_w, bad_x in [(np.full(64, np.nan), train), (weight, train[:0]),
                         (weight, train[:, :63]), (weight, np.full((1, 64), np.inf))]:
        try:
            search(bad_w, bad_x)
        except ValueError:
            continue
        raise AssertionError('invalid input accepted')
    # Unquantized SwiGLU permits folding a down-input inverse scale into up rows.
    # This proves the real-arithmetic identity, not BF16 bitwise equivalence.
    up = rng.normal(size=(8, 4))
    gate = rng.normal(size=(8, 4))
    down = rng.normal(size=(3, 4))
    scales = np.array([.25, .5, 2, 4])
    activated = gate / (1 + np.exp(-gate))
    np.testing.assert_allclose((activated * up) @ down.T,
                               (activated * (up / scales)) @ (down * scales).T,
                               rtol=1e-14, atol=1e-14)
    raw_norm = np.array([-.75, -.25, .5, 1])
    transformed = (1 + raw_norm) / scales - 1
    np.testing.assert_allclose((1 + transformed) * scales, 1 + raw_norm)
    return {'protocol': 'synthetic-affine4-weighted-clipping-v1', 'seed': 42,
            'calibration_tokens': 512, 'heldout_tokens': 512, 'groups_per_case': 1,
            'grid': '20 inclusive ratios linearly spaced from 1 to 0.5; strict first-min tie',
            'precision': 'BF16 source/clipped source/activation/metadata; FP32 dequant; FP64 objective',
            'hardware': platform.machine(), 'numpy': np.__version__, 'gpu_used': False,
            'model_loaded': False, 'training': False, 'weights_written': False,
            'checks_passed': True, 'invalid_input_cases': 4,
            'algebra_checks': ['unquantized SwiGLU up/down scale fold', 'Qwen RMSNorm gamma=1+weight transform'],
            'elapsed_seconds': time.perf_counter() - start,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
            'memory_method': 'process lifetime ru_maxrss; includes imports, not model memory',
            'scope': 'Synthetic single-group mathematical evidence only; no real layer calibration, Metal equality, model quality or speed claim.',
            'cases': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parts = args.output.absolute().parts
    if len(parts) > 2 and parts[1] == 'Volumes' and not Path(*parts[:3]).is_mount():
        raise ValueError('external volume is not mounted')
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'checks_passed': result['checks_passed'],
                      'cases': [{k: v for k, v in r.items() if k != 'grid'} for r in result['cases']]}, ensure_ascii=False))
