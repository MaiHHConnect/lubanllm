"""One calibration-only token-sensitive refit of the original fixed eight columns."""
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
from capture_q4_task_reference import check_mapping, read_reference
from diagnose_native_q4 import bf16
from prepare_q4_packed import input_name, read_inputs
from q4_packed_format import protected_factors
from study_q4_column_protection import ROOT, ROWS, SOURCE, dequantize_payload, finite_matrix, output_errors
from study_q4_corrected_outputs import CAPTURE, CPU, paired_records, preflight, read
from study_q4_fixed_support_refit import original_scales

SCHEMA = 'q4-sensitivity-refit-cpu-v1'
POLICY = 'token_sensitive_fixed_support_refit'
RIDGE, RANK = .01, 8
SENSITIVITY = ROOT/'calibration-layerwise-sensitivity-v1'
OUTPUT = ROOT/'sensitivity-refit-cpu-v1'


def normalized_sensitivity(gradient):
    g = finite_matrix(gradient, 'calibration gradient').astype(np.float64)
    square = g*g
    means = np.mean(square, axis=0)
    active = means > 0
    weights = np.zeros_like(square)
    np.divide(square, means[None, :], out=weights, where=active[None, :])
    if not np.isfinite(weights).all() or not np.isfinite(means).all():
        raise ValueError('nonfinite normalized sensitivity')
    effective = len(g)**2/np.sum(weights[:, active]**2, axis=0) if active.any() else np.array([])
    stats = {'rows': len(g), 'outputs': g.shape[1], 'active_channels': int(active.sum()),
        'zero_channels': int((~active).sum()), 'max_normalized_weight': float(weights.max()),
        'effective_rows_min': float(effective.min()) if len(effective) else 0.,
        'effective_rows_max': float(effective.max()) if len(effective) else 0.,
        'normalization': 'g_squared / calibration_channel_mean; zero channels retain old U',
        'clipping': False, 'epsilon_floor': False}
    return weights, means, stats


def checked_sensitivity(weights, means, shape):
    d, mean = np.asarray(weights, np.float64), np.asarray(means, np.float64)
    if (d.shape != shape or mean.shape != (shape[1],) or not np.isfinite(d).all()
            or not np.isfinite(mean).all() or np.any(d < 0) or np.any(mean < 0)):
        raise ValueError('finite nonnegative paired sensitivity required')
    active = mean > 0
    if (np.any(d[:, ~active] != 0)
            or not np.allclose(d[:, active].mean(axis=0), 1., rtol=1e-12, atol=1e-12)):
        raise ValueError('active channels must have mean1, inactive channels zero')
    return d, mean, active


def refit_sensitive_support(inputs, residual, indices, scales, active_count, old, weights, mean_square):
    x, r, old = [finite_matrix(v, name) for v, name in
                 ((inputs, 'Xfit'), (residual, 'R'), (old, 'old U'))]
    indices, scales = np.asarray(indices), np.asarray(scales, np.float64)
    if (len(x) != len(r) or type(active_count) is not int or not 1 <= active_count <= RANK
            or indices.shape != (RANK,) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= x.shape[1]) or np.any(indices[active_count:] != 0)
            or len(set(indices[:active_count].tolist())) != active_count
            or scales.shape != (active_count,) or not np.isfinite(scales).all() or np.any(scales <= 0)
            or old.shape != (r.shape[1], RANK) or np.any(old[:, active_count:] != 0)
            or np.any(old.view(np.uint32) & 65535)):
        raise ValueError('original unique support, positive old RMS and exact BF16 old U required')
    d, means, active = checked_sensitivity(weights, mean_square, r.shape)
    z = x[:, indices[:active_count]].astype(np.float64)/scales
    # N x rank^2, never N x outputs x rank. Output tiles bound the temporary RHS.
    outer = np.einsum('ni,nj->nij', z, z).reshape(len(x), active_count**2)
    systems = np.zeros((r.shape[1], active_count, active_count), np.float64)
    coefficients = old[:, :active_count].T.astype(np.float64)*scales[:, None]
    proposed = old.copy()
    selected = np.flatnonzero(active)
    for start in range(0, len(selected), 128):
        columns = selected[start:start+128]
        tile = d[:, columns]
        system = (outer.T@tile/len(x)).T.reshape(-1, active_count, active_count)
        system = (system+system.transpose(0, 2, 1))*.5 + RIDGE*np.eye(active_count)
        rhs = (z.T@(tile*r[:, columns].astype(np.float64))/len(x)).T
        if not np.isfinite(system).all() or not np.isfinite(rhs).all():
            raise ValueError('nonfinite weighted ridge system')
        solved = np.linalg.solve(system, rhs[..., None])[..., 0]
        exported = bf16((solved/scales).astype(np.float32))
        if not np.isfinite(solved).all() or not np.isfinite(exported).all():
            raise ValueError('nonfinite weighted coefficients or BF16 export')
        systems[columns], coefficients[:, columns] = system, solved.T
        proposed[columns, :active_count] = exported
    return proposed, {'scales': scales, 'systems': systems, 'coefficients': coefficients,
        'active_channels': active, 'gradient_mean_square': means}


def weighted_output_errors(target, baseline, inputs, indices, factors, weights, mean_square):
    target, baseline, x, u = [finite_matrix(v, name) for v, name in
                            ((target, 'target'), (baseline, 'baseline'), (inputs, 'Xfit'), (factors, 'U'))]
    indices = np.asarray(indices)
    if (target.shape != baseline.shape or len(x) != len(target) or indices.shape != (RANK,)
            or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= x.shape[1])
            or u.shape != (target.shape[1], RANK) or np.any(u.view(np.uint32) & 65535)):
        raise ValueError('paired outputs/inputs and BF16 rank8 factors required')
    d, means, active = checked_sensitivity(weights, mean_square, target.shape)
    rounded_target, rounded_base = bf16(target), bf16(baseline)
    corrected = bf16(rounded_base+x[:, indices]@u.T)
    if not all(np.isfinite(v).all() for v in (rounded_target, rounded_base, corrected)):
        raise ValueError('nonfinite two-round weighted output')
    denominator = len(x)*int(active.sum())
    result = {'rows': len(x), 'outputs': target.shape[1], 'active_channels': int(active.sum()),
              'after_output_round': {}, 'raw_gradient_squared_sse_report_only': {}}
    for name, value in (('baseline', rounded_base), ('candidate', corrected)):
        error = value.astype(np.float64)-rounded_target.astype(np.float64)
        per_channel = np.einsum('no,no,no->o', d, error, error)
        total, raw = float(per_channel.sum()), float(per_channel@means)
        if not np.isfinite([total, raw]).all():
            raise ValueError('nonfinite weighted error accumulation')
        result['after_output_round'][name+'_sse'] = total
        result['after_output_round'][name+'_mse'] = total/denominator if denominator else 0.
        result['raw_gradient_squared_sse_report_only'][name+'_sse'] = raw
    return result


def weighted_replacement_gate(old, proposed, old_report, new_report):
    old, proposed = finite_matrix(old, 'old U'), finite_matrix(proposed, 'proposal U')
    before, after = old_report['after_output_round'], new_report['after_output_round']
    values = [before['baseline_sse'], before['candidate_sse'], after['baseline_sse'], after['candidate_sse']]
    if (old.shape != proposed.shape or not np.isfinite(values).all() or min(values) < 0
            or before['baseline_sse'] != after['baseline_sse']
            or any(old_report[k] != new_report[k] for k in ('rows', 'outputs', 'active_channels'))):
        raise ValueError('finite paired weighted gate reports required')
    accepted = bool(old_report['active_channels'] > 0 and after['candidate_sse'] < before['candidate_sse']
                    and after['candidate_sse'] < before['baseline_sse'])
    gate = {'accepted': accepted, 'action': 'replace_coefficients' if accepted else 'retain_original_coefficients',
        'metric': 'normalized_sensitivity_after_output_round_sse',
        'old_sse': before['candidate_sse'], 'zero_sse': before['baseline_sse'],
        'proposed_sse': after['candidate_sse'], 'raw_gradient_and_unweighted_errors_used': False,
        'reason': 'strict normalized weighted rounded improvement over old and zero' if accepted
        else 'no strict normalized weighted rounded improvement over both; retain full old U'}
    return (proposed.copy() if accepted else old.copy()), gate


def sensitivity_preflight(source):
    result = read(SENSITIVITY/'result.json')
    audit = read(SENSITIVITY/'independent-readback.json')
    if (result.get('schema') != 'q4-layerwise-calibration-sensitivity-v1' or result.get('completed') is not True
            or result.get('source') != str(SOURCE) or result.get('source_receipt') != source['receipt']
            or result.get('plan') != source['plan'] or result.get('core') is not None
            or result.get('loss_reduction') != 'sum' or result.get('attention_implementation') != 'eager'
            or result.get('all_forward_controls_exact') is not True or result.get('parameter_updates') != 0
            or result.get('use_cache') is not False or audit.get('audit_completed') is not True
            or audit.get('audit_passed') is not True
            or len(result.get('cases', [])) != 32 or len(result.get('records', [])) != 32):
        raise ValueError('audited same-source complete32 layerwise CE-sum gradients required')
    check_mapping(read(SENSITIVITY/'position-mapping.json'), source['plan'])
    equivalence = read(SENSITIVITY/'equivalence/result.json')
    if (equivalence.get('completed') is not True or equivalence.get('loss_reduction') != 'sum'
            or equivalence.get('logits_exact') is not True or equivalence.get('gradients_exact') is not True
            or len(equivalence.get('projections', [])) != 32):
        raise ValueError('complete32 full-graph/layerwise gradient control required')
    return result


def run(output):
    started = time.perf_counter()
    source, old_cpu, capture, reference, scope = preflight()
    sensitivity = sensitivity_preflight(source)
    registration = read(Path(__file__).resolve().parents[1]/'research/q4_sensitivity_refit_20261016.json')
    if (registration.get('schema') != SCHEMA or registration.get('selection_policy') != POLICY
            or registration.get('registered_before_run') is not True or registration.get('rank') != RANK
            or registration.get('ridge') != RIDGE or registration.get('rows') != ROWS
            or registration.get('projection_count') != 32 or registration.get('layers') != old_cpu['layers']
            or registration.get('loss_reduction') != 'sum' or registration.get('shortlist') is not None
            or registration.get('update_count') != 1 or registration.get('automatic_iterations') != 0
            or registration.get('source') != str(SOURCE) or registration.get('old_factors') != str(CPU)
            or registration.get('actual_inputs') != str(CAPTURE/'fitv3/inputs')
            or registration.get('reference') != source['asymmetric_reference']
            or registration.get('sensitivity') != str(SENSITIVITY) or registration.get('output') != str(output)):
        raise ValueError('single fixed-support token-sensitive preregistration required')
    weight_map = read(Path(MODEL)/'model.safetensors.index.json')['weight_map']
    output.mkdir(exist_ok=False)
    metadata = {**registration, 'completed': False, 'source_receipt': source['receipt'],
        'capture_receipt': capture['receipt'], 'plan': source['plan'], 'shortlist': None,
        'sensitivity_schema': sensitivity['schema'], 'hardware': platform.platform(),
        'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu', 'gpu_used': False,
        'native_execution_verified': False}
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
                if count != RANK:
                    raise ValueError('all original eight support slots required')
                indices = np.argmax(basis, axis=1).astype(np.int64)
                scales = original_scales(old_row, indices, count)
                name = input_name(layer, kind)
                if name != previous_name:
                    inputs = {split: {'bf16': read_reference(reference, layer, 2 if kind == 'down' else 1, split, rows),
                        'fit': read_inputs(CAPTURE/'fitv3/inputs', name, split, k, rows)} for split, rows in ROWS.items()}
                    previous_name = name
                gradient_path = SENSITIVITY/f'layer{layer:02d}-{kind}.bf16'
                if gradient_path.is_symlink() or gradient_path.stat().st_size != ROWS['calibration']*n*2:
                    raise ValueError('complete finite calibration gradient stream required')
                words = np.memmap(gradient_path, dtype='<u2', mode='r', shape=(ROWS['calibration'], n))
                gradient = (words.astype(np.uint32)<<16).view(np.float32)
                d, means, stats = normalized_sensitivity(gradient)
                del gradient, words
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
                target, baseline = inputs['calibration']['bf16']@weights.T, x@quantized.T
                proposed, solution = refit_sensitive_support(x, target-baseline, indices, scales, count, old, d, means)
                weighted = {mode: weighted_output_errors(target, baseline, x, indices, factor, d, means)
                            for mode, factor in (('old', old), ('new', proposed))}
                applied, gate = weighted_replacement_gate(old, proposed, weighted['old'], weighted['new'])
                weighted['applied'] = weighted['new' if gate['accepted'] else 'old']
                errors = {}
                for split in ROWS:
                    if split == 'observation':
                        x = inputs[split]['fit']
                        target, baseline = inputs[split]['bf16']@weights.T, x@quantized.T
                    split_errors = {mode: output_errors(target, baseline, x, indices, factor)
                                    for mode, factor in (('old', old), ('new', proposed))}
                    errors[split] = {**split_errors, 'applied': split_errors['new' if gate['accepted'] else 'old']}
                path, solve_path = output/f'layer-{layer:02d}-{kind}.npz', output/f'layer-{layer:02d}-{kind}-solve.npz'
                np.savez(path, layer=np.int32(layer), kind=kind, shape=np.array([n, k]),
                    q4_payload=np.frombuffer(raw, dtype=np.uint8), selected_indices=indices,
                    selected_factors=proposed, active_count=np.int32(count), indices=indices, factors=applied)
                np.savez(solve_path, **solution)
                reports.append({'layer': layer, 'kind': kind, 'shape': [n, k], 'paired_record': str(path),
                    'old_paired_record': old_row['paired_record'], 'gradient_stream': str(gradient_path),
                    'active_columns': count, 'fixed_support': {'indices': indices.tolist(), 'old_rms': scales.tolist()},
                    'selection': {'selection_policy': POLICY, 'selected_count': count, 'padding': 0},
                    'sensitivity_statistics': stats,
                    'solve': {'path': str(solve_path), 'ridge': RIDGE, 'dtype': 'float64',
                        'shape': list(solution['coefficients'].shape),
                        'systems_shape': list(solution['systems'].shape),
                        'inactive_channels': 'not solved; coefficients=old_U*s_old, systems=zero, exported U exactly old'},
                    'gate': gate, 'weighted_errors': weighted, 'errors': errors})
                save(output/'progress.json', {'completed': False, 'projections': reports})
                print(json.dumps({'layer': layer, 'kind': kind, 'gate': gate}), flush=True)
                del tensor, weights, quantized, raw, target, baseline, solution, proposed, applied, old, d, means
            if next(records, None) is not None or len(reports) != 32:
                raise ValueError('all32 fixed projections required')
        replacements = sum(row['gate']['accepted'] for row in reports)
        preserved = all(row['weighted_errors']['applied']['after_output_round']['candidate_sse']
                        <= row['weighted_errors']['old']['after_output_round']['candidate_sse'] for row in reports)
        save(output/'result.json', {**metadata, 'completed': True, 'projections': reports,
            'replaced_projections': replacements, 'retained_original_projections': 32-replacements,
            'calibration_weighted_old_preserved': preserved, 'weight_tensors_read': 32,
            'cpu_prerequisite_for_native_study': bool(replacements and preserved),
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
