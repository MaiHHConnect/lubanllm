"""CPU-only G=1 raw-gradient weighting of the existing dynamic-grid GPTAQ path."""
from __future__ import annotations

import numpy as np

from prepare_q4_packed import (DAMPING, GROUP, asymmetric_factor, gptq_quantize,
                               inverse_factor, metadata)


def _finite_real(value, dtype, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must contain real numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        array = array.astype(dtype, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def normalized_token_weights(gradient):
    """Pool raw g² across outputs before token normalization; zero means fallback."""
    g = _finite_real(gradient, np.float64, 'gradient')
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] == 0:
        raise ValueError('gradient must have at least two rows and one output')
    with np.errstate(over='ignore', invalid='ignore'):
        energy = np.mean(np.square(g), axis=1)
        mean = float(np.mean(energy))
    if not np.isfinite(energy).all() or not np.isfinite(mean):
        raise ValueError('nonfinite squared gradient energy')
    if np.all(energy == energy[0]):
        mean = float(energy[0])  # Constant saliency must normalize to exactly one.
    d = energy/mean if mean > 0 else np.zeros_like(energy)
    if not np.isfinite(d).all():
        raise ValueError('nonfinite normalized token weights')
    return d, {'groups': 1, 'rows': g.shape[0], 'outputs': g.shape[1],
               'normalization': 'mean_output_raw_gradient_squared_then_token_mean',
               'raw_mean_square': mean, 'all_zero': mean == 0,
               'zero_weight_rows': int(np.count_nonzero(d == 0)),
               'minimum': float(d.min()), 'maximum': float(d.max()),
               'effective_rows': float(d.sum()**2/np.square(d).sum()) if mean > 0 else 0.}


def weighted_inputs(fp, x, d):
    """Scale both paired inputs in FP32, without introducing BF16 rounding."""
    fp = _finite_real(fp, np.float32, 'reference inputs')
    x = _finite_real(x, np.float32, 'current inputs')
    d = _finite_real(d, np.float64, 'token weights')
    if (x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0 or fp.shape != x.shape
            or d.ndim != 1 or d.shape[0] != x.shape[0]):
        raise ValueError('paired matrices and one token weight per row required')
    if np.any(d < 0) or not np.any(d > 0):
        raise ValueError('nonnegative token weights with positive energy required')
    with np.errstate(over='ignore', invalid='ignore'):
        root = np.sqrt(d).astype(np.float32)
        fp_scaled, x_scaled = fp*root[:, None], x*root[:, None]
    if not np.isfinite(fp_scaled).all() or not np.isfinite(x_scaled).all():
        raise ValueError('nonfinite weighted inputs')
    return fp_scaled, x_scaled


def quantize_sensitive(weights, fp, x, d):
    """One projection, one fresh factor; no cross-projection gate/up sharing."""
    weights = _finite_real(weights, np.float32, 'weights')
    fp_scaled, x_scaled = weighted_inputs(fp, x, d)
    if (weights.ndim != 2 or weights.shape[0] == 0
            or weights.shape[1] != x_scaled.shape[1] or weights.shape[1] % GROUP):
        raise ValueError('nonempty projection with paired group16 input columns required')
    upper, hessian = inverse_factor(x_scaled, damping_fraction=DAMPING)
    if hessian['dead_columns']:
        raise ValueError('dynamic-grid candidate requires observed input columns')
    correction = asymmetric_factor(fp_scaled, x_scaled, upper, alpha=.25)
    _, _, scales, biases = metadata(weights)
    codes, scales, biases = gptq_quantize(
        weights, scales, biases, upper, refit_grid=True, correction=correction)
    return codes, scales, biases, {'groups': 1, 'group_size': GROUP,
                                  'refit_grid': True, 'asymmetric_alpha': .25,
                                  'factor_scope': 'this_projection', 'hessian': hessian,
                                  'weighted_input_dtype': 'float32',
                                  'zero_asymmetric_correction': not bool(correction.count_nonzero())}
