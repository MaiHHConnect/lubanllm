"""Fixed rank8 greedy normalized-column ridge fit of frozen asymmetric residuals."""
import argparse
import os
from pathlib import Path

import numpy as np

from study_q4_column_protection import ROOT, RANK, bf16, finite_matrix, run

RIDGE = .01
POLICY = 'normalized_column_ridge_fit'


def fit_columns(inputs, residual, *, rank=RANK):
    """Schur-complement selection and joint refit; never allocate the full Gram."""
    x, r = finite_matrix(inputs, 'Xq'), finite_matrix(residual, 'R')
    if len(x) != len(r) or type(rank) is not int or not 1 <= rank <= RANK:
        raise ValueError('paired residual rows and rank in1..8 required')
    n, k = x.shape
    x64, r64 = x.astype(np.float64), r.astype(np.float64)
    scales = np.sqrt(np.einsum('ij,ij->j', x64, x64)/n)
    available = scales > 0
    z = x64/np.where(available, scales, 1)[None, :]
    del x64
    b = z.T@r64/n
    diagonal = np.einsum('ij,ij->j', z, z)/n + RIDGE
    if not all(np.isfinite(v).all() for v in (scales, z, b, diagonal)):
        raise ValueError('nonfinite normalized ridge statistics')
    initial = float(np.einsum('ij,ij->', r64, r64)/n)
    del r64
    selected, steps, gram_rows = [], [], []
    coefficients = np.empty((0, r.shape[1]), dtype=np.float64)
    lower = None
    for slot in range(rank):
        if not np.any(available):
            break
        if selected:
            a_rows = np.stack(gram_rows)
            # A_Sj contains lambda only when j is already selected; these columns
            # are excluded. All unselected cross moments are undamped.
            whitened = np.linalg.solve(lower, a_rows)
            denominator = diagonal-np.einsum('ij,ij->j', whitened, whitened)
            gradient = b-a_rows.T@coefficients
        else:
            denominator, gradient = diagonal.copy(), b
        if (not np.isfinite(denominator[available]).all() or np.any(denominator[available] <= 0)
                or not np.isfinite(gradient).all()):
            raise ValueError('nonpositive/nonfinite ridge Schur denominator or gradient')
        gains = np.full(k, -np.inf)
        gains[available] = np.einsum('ij,ij->i', gradient[available], gradient[available])/denominator[available]
        if not np.isfinite(gains[available]).all():
            raise ValueError('nonfinite regularized-objective gain')
        best = int(np.argmax(gains))  # Stable ascending original-column tie break.
        if gains[best] <= 0:
            break
        selected.append(best); available[best] = False
        row = z[:, best]@z/n
        row[best] += RIDGE
        if not np.isfinite(row).all():
            raise ValueError('nonfinite selected normalized Gram row')
        gram_rows.append(row)
        system = np.stack(gram_rows)[:, selected]
        # Each entry uses the same dot product; symmetrize only floating reduction noise.
        system = (system+system.T)*.5
        lower = np.linalg.cholesky(system)
        coefficients = np.linalg.solve(lower.T, np.linalg.solve(lower, b[selected]))
        if not np.isfinite(coefficients).all():
            raise ValueError('nonfinite joint ridge coefficients')
        steps.append({'slot': slot, 'column': best, 'rms': float(scales[best]),
            'regularized_objective_decrease_per_row': float(gains[best]),
            'joint_regularized_objective_per_row': float(initial-np.einsum('ij,ij->', b[selected], coefficients))})
    indices = np.zeros(rank, dtype=np.int64)
    factors = np.zeros((r.shape[1], rank), dtype=np.float32)
    if selected:
        # One export rounding, only after all selected columns have been jointly refit.
        unrounded = coefficients/scales[selected, None]
        exported = bf16(unrounded.T.astype(np.float32))
        if not np.isfinite(exported).all():
            raise ValueError('fitted update overflows stored BF16')
        indices[:len(selected)] = selected
        factors[:, :len(selected)] = exported
    return indices, factors, {'selection_policy': POLICY, 'ridge': RIDGE,
        'factor_semantics': 'BF16(final joint normalized ridge coefficient / selected input RMS), not W-Q',
        'candidate_columns': k, 'zero_rms_columns': int(np.count_nonzero(scales == 0)),
        'gram_rows_computed': len(selected), 'selected': steps, 'selected_count': len(selected),
        'padding': rank-len(selected), 'initial_regularized_objective_per_row': initial}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'column-fit-cpu-v1')
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('new output on mounted experiment volume required')
    run(args.output, column_fit=True)
