"""Small CPU counterexamples for unnecessary/unstable constraint corrections.

Uses the installed helper without changing it. No model, MPS work, hashes,
snapshots or optimizer execution. The existing final 64-epsilon bounds remain
identical in every comparison; this diagnostic does not evaluate model losses.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import resource
import tempfile
import time

import torch

from luban.core_update import constrain_residual, validate_basis


def final_checks(candidate, current, basis, gradients):
    """The helper's actual-step/final-subspace criteria, without any solver."""
    step = candidate - current
    step_norm = float(step.double().norm())
    epsilon = 64 * torch.finfo(torch.float32).eps
    interference = [float((g.double() * step.double()).sum()) for g in gradients]
    tolerances = [epsilon * float(g.double().norm()) * step_norm for g in gradients]
    residual = float((candidate @ basis).norm())
    subspace_tolerance = epsilon * max(1., float(candidate.double().norm()))
    finite = bool(torch.isfinite(candidate).all() and torch.isfinite(step).all())
    passed = (finite and math.isfinite(residual) and residual <= subspace_tolerance
              and all(value <= bound for value, bound in zip(interference, tolerances)))
    return {'passed': passed, 'interference': interference, 'linear_tolerances': tolerances,
            'step_norm': step_norm, 'subspace_residual': residual,
            'subspace_tolerance': subspace_tolerance}


def direction_checks(gradient, basis):
    scaled = gradient / gradient.abs().max()
    once = scaled - (scaled @ basis) @ basis.T
    twice = once - (once @ basis) @ basis.T
    return once, twice, {
        'once': {'norm': float(once.norm()), 'span_residual': float((once @ basis).norm()),
                 'unit_direction_span_residual': float(((once / once.norm()) @ basis).norm())},
        'twice': {'norm': float(twice.norm()), 'span_residual': float((twice @ basis).norm()),
                  'unit_direction_span_residual': float(((twice / twice.norm()) @ basis).norm())},
    }


def run():
    torch.set_num_threads(2)
    started = time.perf_counter()
    basis = torch.tensor([[1.], [1.], [0.]]) / 2 ** .5
    validation = validate_basis(basis)
    current = torch.zeros(1, 3)
    proposal = torch.tensor([[.1, -.1, .1]])
    projected = proposal - (proposal @ basis) @ basis.T
    cases = []
    for name, gradients in (
            ('one_near_span_signal_already_within_existing_bounds',
             torch.tensor([[[1., 1., 1e-8]]])),
            ('four_near_span_opposite_and_duplicate_signals_already_within_bounds',
             torch.tensor([[[1., 1., 1e-8]], [[2., 2., 2e-8]],
                           [[-1., -1., -1e-8]], [[.5, .5, -2e-8]]]))):
        accepted, report = constrain_residual(proposal, current, basis, gradients)
        checks = final_checks(projected, current, basis, gradients)
        assert checks['passed']
        cases.append({'name': name, 'gradients': gradients.tolist(),
                      'uncorrected_projected_proposal': checks,
                      'current_helper': report,
                      'helper_kept_current': bool(torch.equal(accepted, current)),
                      'unnecessary_correction_rejection_reproduced': report['rejected'] and checks['passed']})

    # Here the original proposal really violates the unchanged linear bound.
    # A second projection removes cancellation leakage before normalization;
    # correcting the real complement can then pass both original final checks.
    gradient = torch.tensor([[1., 1., 1e-3]])
    _, twice, directions = direction_checks(gradient, basis)
    pre = float((gradient.double() * (projected - current).double()).sum())
    projected_gradient = twice * float(gradient.abs().max())
    squared_norm = float(projected_gradient.double().square().sum())
    corrected = projected - (pre / squared_norm) * projected_gradient
    _, original_report = constrain_residual(proposal, current, basis, [gradient])
    before = final_checks(projected, current, basis, [gradient])
    after = final_checks(corrected, current, basis, [gradient])
    assert not before['passed'] and after['passed']
    _, _, near_span_directions = direction_checks(torch.tensor([[1., 1., 1e-8]]), basis)
    result = {
        'protocol': 'constraint-numerics-small-cpu-v1', 'passed': True,
        'device': 'cpu', 'threads': torch.get_num_threads(), 'torch': torch.__version__,
        'seed': None, 'randomness': 'none; all matrices are literal constants',
        'hardware': platform.platform(), 'dtype': 'FP32 matrices; FP64 reductions/small solve',
        'memory_measurement': 'process lifetime ru_maxrss; bytes on Darwin, KiB on Linux; no MPS',
        'lifetime_peak_rss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'basis': basis.tolist(), 'basis_validation': validation,
        'current': current.tolist(), 'proposal': proposal.tolist(),
        'unchanged_epsilon': 64 * torch.finfo(torch.float32).eps,
        'cases': cases, 'near_span_direction_diagnostics': near_span_directions,
        'genuinely_conflicting_direction': {
            'gradient': gradient.tolist(), 'uncorrected': before,
            'current_helper': original_report, 'directions': directions,
            'double_projected_correction': corrected.tolist(), 'corrected_final_checks': after,
        },
        'interpretation': [
            'Check the projected proposal with the existing final bounds before active-set correction.',
            'If those same bounds already pass, unnecessary correction can introduce violations.',
            'For real conflicts, reprojecting directions before normalization can remove amplified span leakage.',
            'Double projection alone is not a proof: every final FP32 candidate still needs all original checks.',
            'No protected CE or real-model guarantee is inferred; all 61 actual loss vetoes remain required.',
            'This reproduces a numerical mechanism on CPU, not the exact MPS trajectory or its training outcome.',
        ],
        'model_loads': 0, 'optimizer_updates': 0, 'existing_source_mutations': 0,
        'elapsed_seconds': time.perf_counter() - started,
    }
    volume = Path('/Volumes/移动数据库')
    if not volume.is_mount():
        raise RuntimeError('The research volume is not mounted')
    destination = Path(tempfile.mkdtemp(prefix='constraint-numerics-cpu-', dir=volume / 'lubanllm/experiments'))
    output = destination / 'result.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'result': str(output), **result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
