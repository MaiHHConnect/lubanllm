"""CPU-only fixed-budget column residual study on frozen GPTAQ inputs and weights."""
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

from audit_q4_all_layers import unpack_words
from benchmark_native_decode import MANIFEST, MODEL, save
from capture_q4_task_reference import load_reference, read_reference
from diagnose_native_q4 import bf16
from prepare_q4_packed import decode, input_name, read_inputs
from prepare_q4_sequential import sparse_payloads
from q4_packed_format import projections, validate_projection

ROOT = Path('/Volumes/移动数据库/lubanllm/experiments/q4-native-protection-20261016')
SOURCE = ROOT/'asymmetric-conversion-v1'
LAYERS = [0, 2, 6, 9, 14, 15, 16, 18, 20, 21, 23]
MASK = sum(1 << layer for layer in LAYERS)
RANK, SHORTLIST = 8, 64
ROWS = {'calibration': 4165, 'observation': 688}


def finite_matrix(value, name):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not all(value.shape) or not np.isfinite(value).all():
        raise ValueError('finite nonempty matrix required: '+name)
    return value


def select_columns(weights, quantized, inputs, residual, *, rank=RANK, shortlist_size=SHORTLIST):
    """Greedy true residual gain; only a shortlist-by-shortlist Gram is constructed."""
    w, q, x, r = [finite_matrix(v, n) for v, n in
                   ((weights, 'W'), (quantized, 'Q'), (inputs, 'Xq'), (residual, 'R'))]
    if (q.shape != w.shape or x.shape[1] != w.shape[1] or r.shape != (len(x), len(w))
            or type(rank) is not int or rank <= 0 or type(shortlist_size) is not int or shortlist_size <= 0):
        raise ValueError('paired projection dimensions and positive integer budgets required')
    x64 = x.astype(np.float64)
    diagonal = np.einsum('ij,ij->j', x64, x64)/len(x)
    difference = w.astype(np.float64)-q.astype(np.float64)
    priority = diagonal*np.einsum('ij,ij->j', difference, difference)
    short = np.lexsort((np.arange(w.shape[1]), -priority))[:min(shortlist_size, w.shape[1])]
    # Quantize only the retained delta columns; Q remains decoded with its actual BF16 grids.
    candidates = bf16((w[:, short]-q[:, short]).astype(np.float32)).astype(np.float64)
    xs = x64[:, short]
    h = xs.T@xs/len(x)
    cross = r.astype(np.float64).T@xs/len(x)
    if not all(np.isfinite(v).all() for v in (priority, candidates, h, cross)):
        raise ValueError('nonfinite column objective')
    del x64, difference
    indices = np.zeros(rank, dtype=np.int64)
    factors = np.zeros((len(w), rank), dtype=np.float32)
    used = np.zeros(len(short), dtype=bool)
    steps = []
    for slot in range(rank):
        gains = 2*np.einsum('ij,ij->j', candidates, cross)-h.diagonal()*np.einsum('ij,ij->j', candidates, candidates)
        if not np.isfinite(gains).all():
            raise ValueError('nonfinite marginal gain')
        order = np.lexsort((short, -np.where(used, -np.inf, gains)))
        best = int(order[0])
        if used[best] or gains[best] <= 0:
            break
        indices[slot], factors[:, slot], used[best] = short[best], candidates[:, best], True
        steps.append({'slot': slot, 'column': int(short[best]), 'gain_per_row': float(gains[best]),
                      'predicted_sse_decrease': float(gains[best]*len(x))})
        cross -= candidates[:, best, None]*h[best, :][None, :]
    return indices, factors, {'shortlist': short.tolist(), 'shortlist_priority': priority[short].tolist(),
                             'selected': steps, 'selected_count': len(steps), 'padding': rank-len(steps)}


def select_all_columns(weights, quantized, inputs, residual, *, rank=RANK):
    """Evaluate every asymmetric column gain, computing only the chosen Gram rows."""
    w, q, x, r = [finite_matrix(v, n) for v, n in
                   ((weights, 'W'), (quantized, 'Q'), (inputs, 'Xq'), (residual, 'R'))]
    if (q.shape != w.shape or x.shape[1] != w.shape[1] or r.shape != (len(x), len(w))
            or type(rank) is not int or rank <= 0):
        raise ValueError('paired projection dimensions and positive integer budget required')
    x64 = x.astype(np.float64)
    diagonal = np.einsum('ij,ij->j', x64, x64)/len(x)
    candidates = bf16(w-q).astype(np.float64)
    cross = r.astype(np.float64).T@x64/len(x)
    norm = np.einsum('ij,ij->j', candidates, candidates)
    if not all(np.isfinite(v).all() for v in (diagonal, candidates, cross, norm)):
        raise ValueError('nonfinite all-column objective')
    indices = np.zeros(rank, dtype=np.int64)
    factors = np.zeros((len(w), rank), dtype=np.float32)
    used = np.zeros(w.shape[1], dtype=bool)
    steps = []
    for slot in range(rank):
        gains = 2*np.einsum('ij,ij->j', candidates, cross)-diagonal*norm
        if not np.isfinite(gains).all():
            raise ValueError('nonfinite marginal gain')
        # Array order is column order; argmax chooses the lowest column on an exact tie.
        best = int(np.argmax(np.where(used, -np.inf, gains)))
        if used[best] or gains[best] <= 0:
            break
        indices[slot], factors[:, slot], used[best] = best, candidates[:, best], True
        steps.append({'slot': slot, 'column': best, 'gain_per_row': float(gains[best]),
                      'predicted_sse_decrease': float(gains[best]*len(x))})
        h_row = x64[:, best]@x64/len(x)
        if not np.isfinite(h_row).all():
            raise ValueError('nonfinite selected Gram row')
        cross -= candidates[:, best, None]*h_row[None, :]
    return indices, factors, {'candidate_columns': w.shape[1], 'gram_rows_computed': len(steps),
                             'selected': steps, 'selected_count': len(steps), 'padding': rank-len(steps)}


def squared_error(actual, target):
    delta = actual.astype(np.float64)-target.astype(np.float64)
    value = float(np.einsum('ij,ij->', delta, delta))
    if not np.isfinite(value):
        raise ValueError('nonfinite full projection SSE')
    return value


def output_errors(target, baseline, inputs, indices, factors):
    # Native design would round Q4 output first, add its residual, then round again.
    # Dense CPU FP32 matmuls do not reproduce the Metal accumulation order.
    residual = inputs[:, indices]@factors.T
    rounded_base, rounded_target = bf16(baseline), bf16(target)
    corrected_pre_round = rounded_base+residual
    if not all(np.isfinite(v).all() for v in (residual, rounded_base, rounded_target, corrected_pre_round)):
        raise ValueError('nonfinite staged CPU projection output')
    corrected = bf16(corrected_pre_round)
    return {'rows': len(inputs), 'outputs': target.shape[1],
        'target_energy_before_round': squared_error(target, np.zeros_like(target)),
        'before_output_round': {'baseline_sse': squared_error(baseline, target),
                               'ideal_additive_sse': squared_error(baseline+residual, target)},
        'before_final_round': {'baseline_sse': squared_error(rounded_base, rounded_target),
                              'candidate_sse': squared_error(corrected_pre_round, rounded_target)},
        'after_output_round': {'baseline_sse': squared_error(rounded_base, rounded_target),
                              'candidate_sse': squared_error(corrected, rounded_target)}}


def calibration_gate(indices, factors, report, selection):
    metric = report['after_output_round']
    accepted = selection['selected_count'] > 0 and metric['candidate_sse'] < metric['baseline_sse']
    if accepted:
        return indices, factors, {'accepted': True, 'reason': 'strict rounded calibration SSE decrease'}
    return np.zeros_like(indices), np.zeros_like(factors), {
        'accepted': False, 'reason': 'no positive column' if not selection['selected_count']
        else 'rounded calibration SSE did not strictly decrease'}


def dequantize_payload(raw, n, k):
    validate_projection(raw, n, k)
    codes = unpack_words(np.frombuffer(raw, dtype='<u4', count=n*k//8, offset=8).reshape(n, k//8))
    groups = n*k//16
    scales, biases = [(np.frombuffer(raw, dtype='<u2', count=groups, offset=8+n*k//2+at*groups*2)
                       .astype(np.uint32)<<16).view(np.float32).reshape(n, k//16) for at in (0, 1)]
    return decode(codes, scales, biases)


def frozen_scope(source):
    if (source.get('completed') is not True or source.get('mode') != 'convert'
            or source.get('accepted_layers') != LAYERS or source.get('fixed_layers') != LAYERS
            or source.get('layer_mask') != MASK or source.get('mask_zero') is not False
            or source.get('artifact', {}).get('quantized_projections') != 32
            or source.get('reload', {}).get('passed') is not True
            or source.get('asymmetric_alpha') != .25
            or source.get('asymmetric_reference') != str(ROOT/'task-asymmetric-reference-v1')
            or source.get('input_rows') != ROWS):
        raise ValueError('exact frozen asymmetric32 projection scope and paired rows required')
    return projections(MASK)


def run(output, *, all_columns=False, column_fit=False):
    if all_columns and column_fit:
        raise ValueError('all-column W-Q protection and fitted updates are distinct methods')
    started = time.perf_counter()
    source = json.loads((SOURCE/'result.json').read_text())
    scope = frozen_scope(source)
    audit = json.loads((SOURCE/'independent-readback.json').read_text())
    if audit.get('audit_passed') is not True or audit.get('ordinary_reload16_exact') is not True:
        raise ValueError('completed independent source audit required')
    reference = load_reference(Path(source['asymmetric_reference']), source['receipt'], source['plan'],
        task_source=source['task_trajectories_from'], task_source_receipt=source['task_source_receipt'])
    registration_name = ('q4_column_fit_20261016.json' if column_fit else
                         'q4_column_protection_all_columns_20261016.json' if all_columns
                         else 'q4_column_protection_20261016.json')
    registration = json.loads((Path(__file__).resolve().parents[1]/'research'/registration_name).read_text())
    policy = ('normalized_column_ridge_fit' if column_fit else
              'all_columns_asymmetric_gain' if all_columns else 'symmetric_saliency64_then_asymmetric_gain')
    if (registration.get('rank') != RANK or registration.get('shortlist') != (None if all_columns or column_fit else SHORTLIST)
            or ((all_columns or column_fit) and registration.get('selection_policy') != policy)
            or (column_fit and registration.get('ridge') != .01)):
        raise ValueError('fixed preregistered column budget required')
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**registration, 'selection_policy': policy, 'source_receipt': source['receipt'], 'plan': source['plan'],
        'completed': False, 'hardware': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
        'device': 'cpu', 'gpu_used': False, 'seed': 42, 'selection_deterministic': True}
    save(output/'protocol.json', metadata)
    reports = []
    try:
        with ExitStack() as stack:
            readers = {p.name: stack.enter_context(safe_open(p, framework='pt', device='cpu'))
                       for p in sorted(Path(MODEL).glob('*.safetensors')) if not p.name.startswith('.')}
            mapping = {key: reader for reader in readers.values() for key in reader.keys()}
            payload = iter(sparse_payloads(SOURCE/'weights', LAYERS))
            last_name, inputs, references = None, None, None
            for layer, kind, n, k in scope:
                got_layer, got_kind, raw = next(payload)
                if (got_layer, got_kind) != (layer, kind):
                    raise ValueError('frozen stage projection order changed')
                name = input_name(layer, kind)
                if name != last_name:
                    inputs = {split: read_inputs(SOURCE/'inputs', name, split, k, count) for split, count in ROWS.items()}
                    references = {split: read_reference(reference, layer, 2 if kind == 'down' else 1, split, count)
                                  for split, count in ROWS.items()}
                    for split in ROWS:
                        finite_matrix(inputs[split], 'frozen '+split)
                    last_name = name
                key = f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
                tensor = mapping[key].get_tensor(key)
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (n, k):
                    raise ValueError('original BF16 projection shape changed')
                weights = tensor.float().numpy()  # Exactly one read of each of the frozen32 W tensors.
                quantized = dequantize_payload(raw, n, k)
                targets = {split: references[split]@weights.T for split in ROWS}
                baselines = {split: inputs[split]@quantized.T for split in ROWS}
                residual = targets['calibration']-baselines['calibration']
                if column_fit:
                    from study_q4_column_fit import fit_columns
                    indices, factors, selection = fit_columns(inputs['calibration'], residual)
                else:
                    selector = select_all_columns if all_columns else select_columns
                    indices, factors, selection = selector(weights, quantized, inputs['calibration'], residual)
                proposed = output_errors(targets['calibration'], baselines['calibration'], inputs['calibration'], indices, factors)
                applied_indices, applied, gate = calibration_gate(indices, factors, proposed, selection)
                # Observation enters only reporting, after the calibration-only selection and gate.
                proposed_observation = output_errors(targets['observation'], baselines['observation'],
                    inputs['observation'], indices, factors)
                errors = ({'calibration': proposed, 'observation': proposed_observation} if gate['accepted'] else
                          {split: output_errors(targets[split], baselines[split], inputs[split], applied_indices, applied)
                           for split in ROWS})
                # Keep the exact bytes that produced Q and their residual together. The
                # indices encode the one-hot rank8 basis; no independent factor path exists.
                record_path = output/f'layer-{layer:02d}-{kind}.npz'
                np.savez(record_path, layer=np.int32(layer), kind=kind, shape=np.array([n, k]),
                         q4_payload=np.frombuffer(raw, dtype=np.uint8),
                         selected_indices=indices, selected_factors=factors,
                         active_count=np.int32(selection['selected_count'] if gate['accepted'] else 0),
                         indices=applied_indices, factors=applied)
                report = {'layer': layer, 'kind': kind, 'shape': [n, k], 'selection': selection,
                          'paired_record': str(record_path),
                          'proposed_calibration': proposed, 'proposed_observation': proposed_observation,
                          'gate': gate, 'applied_errors': errors,
                          'active_columns': selection['selected_count'] if gate['accepted'] else 0}
                reports.append(report); save(output/'progress.json', {'completed': False, 'projections': reports})
                print(json.dumps({'layer': layer, 'kind': kind, 'active_columns': report['active_columns'],
                                  'calibration': errors['calibration']['after_output_round']}), flush=True)
                del tensor, weights, quantized, raw, targets, baselines, residual
            if next(payload, None) is not None or len(reports) != 32:
                raise ValueError('frozen32 projection coverage changed')
        promising = any(r['active_columns'] for r in reports) and all(
            r['applied_errors']['calibration']['after_output_round']['candidate_sse']
            <= r['applied_errors']['calibration']['after_output_round']['baseline_sse'] for r in reports)
        save(output/'result.json', {**metadata, 'completed': True, 'projections': reports,
             'cpu_prerequisite_for_native_study': bool(promising), 'native_execution_verified': False,
             'qualified': False, 'elapsed_seconds': time.perf_counter()-started,
             'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'method': 'Darwin process peak RSS, CPU tensors and arrays; no GPU/model forward'}})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'error': repr(error), 'projections': reports,
                                    'elapsed_seconds': time.perf_counter()-started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--all-columns', action='store_true', help='Use independently registered all-column asymmetric gain selection')
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT/('column-protection-all-columns-cpu-v1' if args.all_columns else 'column-protection-cpu-v1')
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('new output on mounted experiment volume required')
    run(args.output, all_columns=args.all_columns)
