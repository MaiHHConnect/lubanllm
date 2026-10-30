"""One fixed-support Jacobi coefficient refit on frozen actual corrected-prefix inputs."""
from contextlib import ExitStack
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
from safetensors import safe_open

from benchmark_native_decode import MODEL, save
from capture_q4_task_reference import read_reference
from diagnose_native_q4 import bf16
from prepare_q4_packed import input_name, read_inputs
from q4_packed_format import protected_factors
from study_q4_column_protection import ROOT, ROWS, dequantize_payload, finite_matrix, output_errors
from study_q4_corrected_outputs import CAPTURE, CPU, paired_records, preflight, read

SCHEMA = 'q4-fixed-support-refit-cpu-v1'
POLICY = 'fixed_support_jacobi_ridge_refit'
RIDGE, RANK = .01, 8
OUTPUT = ROOT/'fixed-support-refit-cpu-v1'


def original_scales(row, indices, count):
    selected = row['selection']['selected']
    if (len(selected) != count or row['active_columns'] != count
            or row['selection']['selected_count'] != count
            or [s['column'] for s in selected] != np.asarray(indices[:count]).tolist()
            or [s['slot'] for s in selected] != list(range(count))):
        raise ValueError('original selected support and scale order changed')
    scales = np.asarray([s['rms'] for s in selected], dtype=np.float64)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('original positive finite RMS scales required')
    return scales


def refit_fixed_support(inputs, residual, indices, scales, active_count):
    """Solve only original support using original scales; return replacement, not delta U."""
    x, r = finite_matrix(inputs, 'Xfit'), finite_matrix(residual, 'Rnew')
    indices, scales = np.asarray(indices), np.asarray(scales, dtype=np.float64)
    if (len(x) != len(r) or type(active_count) is not int or not 0 <= active_count <= RANK
            or indices.shape != (RANK,) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= x.shape[1]) or np.any(indices[active_count:] != 0)
            or len(set(indices[:active_count].tolist())) != active_count
            or scales.shape != (active_count,) or not np.isfinite(scales).all() or np.any(scales <= 0)):
        raise ValueError('fixed unique support, zero padding and positive original scales required')
    factors = np.zeros((r.shape[1], RANK), np.float32)
    z = x[:, indices[:active_count]].astype(np.float64)/scales[None, :]
    system = z.T@z/len(x)+RIDGE*np.eye(active_count)
    rhs = z.T@r.astype(np.float64)/len(x)
    if not np.isfinite(system).all() or not np.isfinite(rhs).all():
        raise ValueError('nonfinite fixed-support ridge system')
    if active_count:
        lower = np.linalg.cholesky(system)
        coefficients = np.linalg.solve(lower.T, np.linalg.solve(lower, rhs))
        exported = bf16((coefficients/scales[:, None]).T.astype(np.float32))
        if not np.isfinite(coefficients).all() or not np.isfinite(exported).all():
            raise ValueError('nonfinite fixed-support solution or BF16 export')
        factors[:, :active_count] = exported
    else:
        coefficients = np.zeros((0, r.shape[1]), np.float64)
    return factors, {'scales': scales, 'system': system, 'coefficients': coefficients}


def replacement_gate(old, proposed, old_report, new_report):
    old, proposed = finite_matrix(old, 'old U'), finite_matrix(proposed, 'new U')
    before, after = old_report['after_output_round'], new_report['after_output_round']
    values = [before['baseline_sse'], before['candidate_sse'], after['baseline_sse'], after['candidate_sse']]
    if (old.shape != proposed.shape or not np.isfinite(values).all() or min(values) < 0
            or before['baseline_sse'] != after['baseline_sse']):
        raise ValueError('same frozen input/target baseline and finite nonnegative SSE required')
    accepted = bool(after['candidate_sse'] < before['candidate_sse']
                    and after['candidate_sse'] < before['baseline_sse'])
    gate = {'accepted': accepted, 'action': 'replace_coefficients' if accepted else 'retain_original_coefficients',
        'old_sse': before['candidate_sse'], 'zero_sse': before['baseline_sse'], 'proposed_sse': after['candidate_sse'],
        'reason': 'strictly better than old and zero on fixed calibration inputs' if accepted
        else 'proposal did not strictly beat both old and zero; preserve full original U'}
    return (proposed.copy() if accepted else old.copy()), gate


def run(output):
    started = time.perf_counter()
    source, old_cpu, capture, reference, scope = preflight()
    registration = read(Path(__file__).resolve().parents[1]/'research/q4_fixed_support_refit_20261016.json')
    if (registration.get('schema') != SCHEMA or registration.get('selection_policy') != POLICY
            or registration.get('registered_before_run') is not True or registration.get('rank') != RANK
            or registration.get('ridge') != RIDGE or registration.get('rows') != ROWS
            or registration.get('update_count') != 1 or registration.get('automatic_iterations') != 0):
        raise ValueError('one fixed-support preregistered update required')
    weight_map = read(Path(MODEL)/'model.safetensors.index.json')['weight_map']
    output.mkdir(exist_ok=False)
    metadata = {**registration, 'completed': False, 'source_receipt': source['receipt'],
        'capture_receipt': capture['receipt'], 'plan': source['plan'], 'shortlist': None,
        'hardware': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
        'device': 'cpu', 'gpu_used': False, 'native_execution_verified': False}
    save(output/'protocol.json', metadata)
    reports = []
    try:
        with ExitStack() as stack:
            readers, inputs, previous_name = {}, None, None
            records = iter(paired_records(old_cpu))
            for old_row, (layer, kind, n, k) in zip(old_cpu['projections'], scope, strict=True):
                got_layer, got_kind, raw, basis, old = next(records)
                if (got_layer, got_kind) != (layer, kind):
                    raise ValueError('original paired record order changed')
                protected_factors(basis, old, n, k)
                count = int(np.count_nonzero(np.any(basis != 0, axis=1)))
                indices = np.argmax(basis, axis=1).astype(np.int64)
                scales = original_scales(old_row, indices, count)
                name = input_name(layer, kind)
                if name != previous_name:
                    inputs = {split: {
                        'bf16': read_reference(reference, layer, 2 if kind == 'down' else 1, split, rows),
                        'fit': read_inputs(CAPTURE/'fitv3/inputs', name, split, k, rows)}
                        for split, rows in ROWS.items()}
                    previous_name = name
                key = f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
                filename = weight_map[key]
                if Path(filename).name != filename:
                    raise ValueError('local model shard basename required')
                if filename not in readers:
                    readers[filename] = stack.enter_context(safe_open(Path(MODEL)/filename, framework='pt', device='cpu'))
                tensor = readers[filename].get_tensor(key)
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (n, k):
                    raise ValueError('original BF16 W shape changed')
                weights, quantized = tensor.float().numpy(), dequantize_payload(raw, n, k)
                x = inputs['calibration']['fit']
                target = inputs['calibration']['bf16']@weights.T
                baseline = x@quantized.T
                proposed, solution = refit_fixed_support(x, target-baseline, indices, scales, count)
                old_report = output_errors(target, baseline, x, indices, old)
                new_report = output_errors(target, baseline, x, indices, proposed)
                applied, gate = replacement_gate(old, proposed, old_report, new_report)
                errors = {'calibration': {'old': old_report, 'new': new_report,
                                         'applied': new_report if gate['accepted'] else old_report}}
                # Observation is computed only after the fixed calibration decision.
                x = inputs['observation']['fit']
                target = inputs['observation']['bf16']@weights.T
                baseline = x@quantized.T
                old_obs = output_errors(target, baseline, x, indices, old)
                new_obs = output_errors(target, baseline, x, indices, proposed)
                errors['observation'] = {'old': old_obs, 'new': new_obs,
                                        'applied': new_obs if gate['accepted'] else old_obs}
                path = output/f'layer-{layer:02d}-{kind}.npz'
                # Keep standard physical fields; report schema distinguishes retain-old fallback.
                np.savez(path, layer=np.int32(layer), kind=kind, shape=np.array([n, k]),
                    q4_payload=np.frombuffer(raw, dtype=np.uint8), selected_indices=indices,
                    selected_factors=proposed, active_count=np.int32(count), indices=indices, factors=applied)
                solve_path = output/f'layer-{layer:02d}-{kind}-solve.npz'
                np.savez(solve_path, **solution)
                reports.append({'layer': layer, 'kind': kind, 'shape': [n, k], 'paired_record': str(path),
                    'old_paired_record': old_row['paired_record'], 'active_columns': count,
                    'fixed_support': {'indices': indices[:count].tolist(), 'old_rms': scales.tolist()},
                    'selection': {'selection_policy': POLICY, 'selected_count': count, 'padding': RANK-count},
                    'solve': {'path': str(solve_path), 'ridge': RIDGE, 'dtype': 'float64',
                        'shape': list(solution['coefficients'].shape),
                        'coefficient_norm': float(np.linalg.norm(solution['coefficients']))},
                    'gate': gate, 'errors': errors})
                save(output/'progress.json', {'completed': False, 'projections': reports})
                print(json.dumps({'layer': layer, 'kind': kind, 'gate': gate}), flush=True)
                del tensor, weights, quantized, raw, target, baseline, solution, proposed, applied, old
            if next(records, None) is not None or len(reports) != 32:
                raise ValueError('all32 fixed projections required')
        replacements = sum(r['gate']['accepted'] for r in reports)
        preserved = all(r['errors']['calibration']['applied']['after_output_round']['candidate_sse']
                        <= r['errors']['calibration']['old']['after_output_round']['candidate_sse'] for r in reports)
        save(output/'result.json', {**metadata, 'completed': True, 'projections': reports,
            'replaced_projections': replacements, 'retained_original_projections': 32-replacements,
            'cpu_prerequisite_for_native_study': bool(replacements and preserved),
            'calibration_old_preserved': preserved, 'weight_tensors_read': 32,
            'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'method': 'Darwin process peak RSS, CPU arrays and32 individually read weights; not deployment residency'}})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'error': repr(error), 'projections': reports,
                                    'elapsed_seconds': time.perf_counter()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('new output on mounted experiment volume required')
    run(args.output)


if __name__ == '__main__':
    main()
