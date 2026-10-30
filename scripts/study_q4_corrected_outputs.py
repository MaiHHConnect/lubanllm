"""CPU output-error proxies for fixed factors under captured corrected-prefix inputs."""
from contextlib import ExitStack
import argparse
import importlib.util
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
from capture_q4_corrected_inputs import checked_loaded, exact_stream_names
from capture_q4_task_reference import check_mapping, load_reference, read_reference
from diagnose_native_q4 import bf16
from prepare_q4_packed import input_name, read_inputs
from q4_packed_format import protected_factors
from study_q4_column_protection import (LAYERS, MASK, ROOT, ROWS, SOURCE, dequantize_payload,
    finite_matrix, frozen_scope, output_errors, squared_error)

CPU = ROOT/'column-fit-cpu-v1'
CAPTURE = ROOT/'corrected-inputs-diagnostic-v1'
OUTPUT = ROOT/'corrected-output-proxy-v1'
SCHEMA = 'q4-corrected-output-proxy-v1'
REPORT_RTOL, REPORT_ATOL = 1e-10, 1e-6


def read(path):
    return json.loads(Path(path).read_text())


def reproduce_report(actual, expected):
    """Compare every frozen output_errors field, not just its rounded headline SSE."""
    differences = []
    def compare(left, right, path):
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise ValueError('frozen output report structure changed: '+path)
            for key in left:
                compare(left[key], right[key], path+'.'+key)
        else:
            if (not isinstance(right, (int, float)) or not np.isfinite([left, right]).all()
                    or not np.isclose(left, right, rtol=REPORT_RTOL, atol=REPORT_ATOL)):
                raise ValueError('old fitted output-error report did not reproduce: '+path)
            differences.append(abs(float(left)-float(right)))
    compare(actual, expected, 'output_errors')
    return {'passed': True, 'fields': len(differences), 'max_absolute_difference': max(differences),
            'rtol': REPORT_RTOL, 'atol': REPORT_ATOL, 'exact': all(d == 0 for d in differences)}


def error_decomposition(target, baseline, old, actual):
    values = [finite_matrix(v, name) for v, name in zip(
        (target, baseline, old, actual), ('target', 'baseline', 'old', 'actual'), strict=True)]
    if len({v.shape for v in values}) != 1:
        raise ValueError('paired output shapes required')
    target, baseline, old, actual = values
    base_sse, old_sse, actual_sse = [squared_error(v, target) for v in (baseline, old, actual)]
    e = old.astype(np.float64)-target.astype(np.float64)
    d = actual.astype(np.float64)-old.astype(np.float64)
    cross = float(2*np.einsum('ij,ij->', e, d))
    energy = float(np.einsum('ij,ij->', d, d))
    delta, predicted = actual_sse-old_sse, cross+energy
    tolerance = 64*np.finfo(np.float64).eps*max(1., abs(cross), energy, old_sse, actual_sse)
    identity_error = delta-predicted
    if not np.isfinite([cross, energy, delta, predicted, identity_error]).all() or abs(identity_error) > tolerance:
        raise ValueError('FP64 propagation-error identity failed')
    return {'v2_sse': base_sse, 'fit_old_sse': old_sse, 'fit_actual_sse': actual_sse,
        'old_gain_vs_v2': base_sse-old_sse, 'actual_gain_vs_v2': base_sse-actual_sse,
        'old_gain_erased': old_sse < base_sse and actual_sse >= base_sse,
        'old_gain_strictly_reversed': old_sse < base_sse and actual_sse > base_sse,
        'propagation': {'twice_error_drift_inner_product': cross, 'drift_energy': energy,
            'sum': predicted, 'actual_minus_old_sse': delta, 'identity_error': identity_error,
            'identity_tolerance': tolerance, 'identity_passed': True}}


def projection_proxy(weights, quantized, xbf, xv2, xfit, indices, factors):
    w, q, fp, old_x, new_x, u = [finite_matrix(v, name) for v, name in
        ((weights, 'W'), (quantized, 'Q'), (xbf, 'Xbf'), (xv2, 'Xv2'), (xfit, 'Xfit'), (factors, 'U'))]
    indices = np.asarray(indices)
    if (w.shape != q.shape or fp.shape != old_x.shape or fp.shape != new_x.shape
            or fp.shape[1] != w.shape[1] or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or u.shape != (len(w), len(indices)) or np.any(indices < 0) or np.any(indices >= w.shape[1])
            or np.any(u.view(np.uint32) & 0xffff)):
        raise ValueError('paired projection/factor shapes and exact BF16 coefficients required')
    target = fp@w.T
    baseline = old_x@q.T
    # Reuse the frozen report implementation; U is fixed, jointly fitted BF16(C/s).
    old_report = output_errors(target, baseline, old_x, indices, u)
    rounded_target, rounded_baseline = bf16(target), bf16(baseline)
    old = bf16(rounded_baseline+old_x[:, indices]@u.T)
    actual = bf16(bf16(new_x@q.T)+new_x[:, indices]@u.T)
    result = error_decomposition(rounded_target, rounded_baseline, old, actual)
    if (result['v2_sse'] != old_report['after_output_round']['baseline_sse']
            or result['fit_old_sse'] != old_report['after_output_round']['candidate_sse']):
        raise ValueError('proxy and reused rounded report disagree')
    return {**result, 'rows': len(fp), 'outputs': len(w), 'old_report': old_report}


def preflight():
    source, cpu = read(SOURCE/'result.json'), read(CPU/'result.json')
    scope = frozen_scope(source)
    capture = read(CAPTURE/'result.json')  # Missing/incomplete capture fails before reading tensors.
    audit = read(CAPTURE/'independent-readback.json')
    if (capture.get('schema') != 'q4-corrected-inputs-diagnostic-v1' or capture.get('completed') is not True
            or audit.get('audit_passed') is not True or capture.get('plan') != source['plan']
            or capture.get('source_receipt') != source['receipt'] or capture.get('source') != str(SOURCE)
            or capture.get('bf16_reference') != source['asymmetric_reference']
            or capture.get('exact_sequential_streams') != exact_stream_names(LAYERS)
            or capture.get('trace_control', {}).get('passed') is not True
            or capture.get('seed') != 42 or capture.get('temperature') != 0 or capture.get('cache_reuse') is not False):
        raise ValueError('completed independently audited same-source actual-input diagnostic required')
    for mode, version, active in (('v2', 2, 0), ('fitv3', 3, 32)):
        record = capture['modes'][mode]
        artifact = SOURCE/'gptq-sequential.bin' if mode == 'v2' else ROOT/'column-fit-native-v1/candidate.bin'
        if record.get('source_artifact') != str(artifact) or record.get('rows_per_split') != ROWS:
            raise ValueError('actual input artifact/row binding changed')
        checked_loaded(record['loaded'], version, active, capture=True)
        check_mapping(read(CAPTURE/mode/'inputs/position-mapping.json'), source['plan'])
    exact = capture['modes']['v2'].get('sequential_exact', [])
    expected = {(name, split) for name in exact_stream_names(LAYERS) for split in ROWS}
    if (len(exact) != 68 or {(r['stream'], r['split']) for r in exact} != expected
            or any(r.get('exact') is not True for r in exact)):
        raise ValueError('34-stream final-prefix control required')
    cpu_audit = read(CPU/'independent-readback.json')
    if (cpu.get('schema') != 'q4-column-fit-cpu-v1' or cpu.get('completed') is not True
            or cpu.get('selection_policy') != 'normalized_column_ridge_fit' or cpu.get('ridge') != .01
            or cpu.get('rank') != 8 or cpu.get('shortlist') is not None or cpu.get('rows') != ROWS
            or cpu.get('source') != str(SOURCE) or cpu.get('source_receipt') != source['receipt']
            or cpu.get('plan') != source['plan'] or cpu.get('projection_count') != 32
            or cpu.get('layers') != LAYERS or cpu_audit.get('audit_passed') is not True
            or cpu_audit.get('directory') != str(CPU)):
        raise ValueError('original audited fitted32 CPU records required')
    reference = load_reference(Path(source['asymmetric_reference']), source['receipt'], source['plan'],
        task_source=source['task_trajectories_from'], task_source_receipt=source['task_source_receipt'])
    return source, cpu, capture, reference, scope


def paired_records(cpu):
    # Reuse the existing frozen validator, including original NPZ paths and applied-only gate.
    # This fixed external helper is already the dependency used to construct the audited carrier.
    spec = importlib.util.spec_from_file_location('frozen_proxy_assembly', ROOT/'prepare-column-protected.py')
    assembly = importlib.util.module_from_spec(spec); spec.loader.exec_module(assembly)
    assembly.CPU = CPU
    yield from assembly.paired_records(cpu)


def run(output):
    started = time.perf_counter()
    source, cpu, capture, reference, scope = preflight()
    registration = read(Path(__file__).resolve().parents[1]/'research/q4_corrected_outputs_20261016.json')
    if (registration.get('schema') != SCHEMA or registration.get('registered_before_run') is not True
            or registration.get('rows') != ROWS or registration.get('projection_count') != 32
            or registration['controls']['old_report_tolerance'] != {'rtol': REPORT_RTOL, 'atol': REPORT_ATOL}):
        raise ValueError('fixed pre-registered proxy calculation required')
    # Index lookup reads no model data and avoids scanning unrelated tensor names or weights.
    weight_map = read(Path(MODEL)/'model.safetensors.index.json')['weight_map']
    output.mkdir(exist_ok=False)
    metadata = {**registration, 'completed': False, 'source_receipt': source['receipt'],
        'capture_receipt': capture['receipt'], 'plan': source['plan'],
        'hardware': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
        'device': 'cpu', 'gpu_used': False, 'native_output_equivalence': False}
    save(output/'protocol.json', metadata)
    reports = []
    try:
        with ExitStack() as stack:
            readers, inputs, previous_name = {}, None, None
            records = iter(paired_records(cpu))
            for row, (layer, kind, n, k) in zip(cpu['projections'], scope, strict=True):
                got_layer, got_kind, raw, basis, update = next(records)
                if (got_layer, got_kind) != (layer, kind):
                    raise ValueError('original paired record order changed')
                protected_factors(basis, update, n, k)
                indices = np.argmax(basis, axis=1).astype(np.int64)  # Padded B slots have exactly zero U.
                name = input_name(layer, kind)
                if name != previous_name:
                    inputs = {split: {
                        'bf16': read_reference(reference, layer, 2 if kind == 'down' else 1, split, count),
                        'v2': read_inputs(CAPTURE/'v2/inputs', name, split, k, count),
                        'fit': read_inputs(CAPTURE/'fitv3/inputs', name, split, k, count)}
                        for split, count in ROWS.items()}
                    previous_name = name
                key = f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
                filename = weight_map[key]
                if Path(filename).name != filename:
                    raise ValueError('weight shard must be a local model basename')
                if filename not in readers:
                    readers[filename] = stack.enter_context(safe_open(Path(MODEL)/filename, framework='pt', device='cpu'))
                tensor = readers[filename].get_tensor(key)
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (n, k):
                    raise ValueError('original BF16 W shape changed')
                weights = tensor.float().numpy()  # One read per original named projection.
                quantized = dequantize_payload(raw, n, k)
                split_reports = {}
                for split in ROWS:
                    x = inputs[split]
                    report = projection_proxy(weights, quantized, x['bf16'], x['v2'], x['fit'], indices, update)
                    report['old_report_reproduction'] = reproduce_report(report['old_report'], row['applied_errors'][split])
                    split_reports[split] = report
                reports.append({'layer': layer, 'kind': kind, 'shape': [n, k],
                    'paired_record': row['paired_record'], 'active_columns': row['active_columns'], 'splits': split_reports})
                save(output/'progress.json', {'completed': False, 'projections': reports})
                print(json.dumps({'layer': layer, 'kind': kind,
                    'splits': {s: {k: v[k] for k in ('v2_sse', 'fit_old_sse', 'fit_actual_sse', 'old_gain_erased')}
                               for s, v in split_reports.items()}}), flush=True)
                del tensor, weights, quantized, raw, basis, update
            if next(records, None) is not None or len(reports) != 32:
                raise ValueError('exact32 output proxy scope required')
        summary = {split: {
            'old_improved_projections': sum(r['splits'][split]['old_gain_vs_v2'] > 0 for r in reports),
            'actual_improved_projections': sum(r['splits'][split]['actual_gain_vs_v2'] > 0 for r in reports),
            'old_gain_erased_projections': sum(r['splits'][split]['old_gain_erased'] for r in reports),
            'old_gain_strictly_reversed_projections': sum(r['splits'][split]['old_gain_strictly_reversed'] for r in reports)}
            for split in ROWS}
        save(output/'result.json', {**metadata, 'completed': True, 'projections': reports, 'summary': summary,
            'old_reports_reproduced': True, 'weight_tensors_read': 32, 'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'method': 'Darwin process peak RSS; CPU arrays and individual tensors, not deployment residency'}})
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
