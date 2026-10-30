"""One fixed-mask, sensitivity-weighted source conversion on actual native prefixes."""
from contextlib import ExitStack
import argparse
import gc
import io
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
from safetensors import safe_open

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from capture_q4_task_reference import layout, load_reference, read_reference
from diagnose_native_q4 import bf16
from luban.process_lock import ModelProcessLock
from prepare_q4_packed import decode, write_projection
from prepare_q4_sequential import (HEADER_BYTES, SequentialWorker, selected_inputs,
                                    sparse_payloads, stage_header)
from q4_packed_format import validate_projection, write_sparse
from q4_sensitivity_quantization import normalized_token_weights, quantize_sensitive
from study_q4_column_protection import (LAYERS, MASK, ROOT, ROWS, SOURCE,
                                         dequantize_payload, frozen_scope)
from study_q4_sensitivity_refit import SENSITIVITY, sensitivity_preflight

OUTPUT = ROOT/'sensitivity-source-sequential-v1'
SCHEMA = 'q4-sensitivity-source-sequential-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_sensitivity_source_20261016.json'


def read(path):
    return json.loads(Path(path).read_text())


def stage_path(directory, layer, mask):
    return directory/f'layer-{layer:02d}-{"gateup" if mask == 3 else "down"}.bin'


def old_stage(layer, mask):
    path = stage_path(SOURCE/'weights', layer, mask)
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular immutable original stage required')
    with path.open('rb') as stream:
        if stream.read(HEADER_BYTES) != stage_header(layer, mask, layer):
            raise ValueError('original temporary stage binding changed')
    return path


def old_payloads(layer, mask):
    with old_stage(layer, mask).open('rb') as stream:
        stream.seek(HEADER_BYTES)
        for kind in (('gate', 'up') if mask == 3 else ('down',)):
            n, k = (2048, 6144) if kind == 'down' else (6144, 2048)
            raw = stream.read(8+n*k*3//4)
            validate_projection(raw, n, k)
            yield kind, raw
        if stream.read(1):
            raise ValueError('trailing original stage data')


def output_metrics(weights, old_q, new_q, fp, x, d=None):
    """Full-output CPU proxy, diagnostic only; never selects projections or grids."""
    target = bf16(fp@weights.T)
    reports = {}
    for name, q in (('old', old_q), ('new', new_q)):
        error = bf16(x@q.T).astype(np.float64)-target.astype(np.float64)
        per_row = np.einsum('ij,ij->i', error, error)
        reports[name] = {'unweighted_sse': float(per_row.sum()),
                        'weighted_sse': float(d@per_row) if d is not None else None}
    if not all(np.isfinite(v) for report in reports.values() for v in report.values() if v is not None):
        raise ValueError('nonfinite diagnostic output error')
    return reports


def convert_stage(layer, mask, values, reference, readers, output):
    converted = {s: (raw.astype(np.uint32)<<16).view(np.float32) for s, raw in values.items()}
    fp = {s: read_reference(reference, layer, 2 if mask == 4 else 1, s, ROWS[s]) for s in ROWS}
    if layer == 0 and mask == 3 and any(not np.array_equal(fp[s], converted[s]) for s in ROWS):
        raise ValueError('first gate/up must exactly reproduce original BF16 inputs')
    path = stage_path(output/'weights', layer, mask)
    reports = []
    with path.open('xb') as stream:
        stream.write(stage_header(layer, mask, layer))
        for kind, old_raw in old_payloads(layer, mask):
            n, k = (2048, 6144) if kind == 'down' else (6144, 2048)
            gradient_path = SENSITIVITY/f'layer{layer:02d}-{kind}.bf16'
            if gradient_path.is_symlink() or gradient_path.stat().st_size != ROWS['calibration']*n*2:
                raise ValueError('exact original calibration gradient stream required')
            words = np.memmap(gradient_path, dtype='<u2', mode='r', shape=(ROWS['calibration'], n))
            gradient = (words.astype(np.uint32)<<16).view(np.float32)
            d, statistics = normalized_token_weights(gradient)
            del gradient, words
            np.save(output/'weights'/f'layer-{layer:02d}-{kind}-token-weights.npy', d)
            key = f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
            tensor = readers[key].get_tensor(key)
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (n, k):
                raise ValueError('original BF16 projection changed')
            weights = tensor.float().numpy()
            old_q = dequantize_payload(old_raw, n, k)
            if np.any(d):
                codes, scales, biases, quantizer = quantize_sensitive(
                    weights, fp['calibration'], converted['calibration'], d)
                buffer = io.BytesIO()
                write_projection(buffer, codes, scales, biases)
                raw = buffer.getvalue()
                new_q = decode(codes, scales, biases)
                action = 'new_source_quantization'
            else:
                raw, new_q, quantizer, action = old_raw, old_q, None, 'retain_zero_sensitivity_projection'
            validate_projection(raw, n, k)
            stream.write(raw)
            report = {'layer': layer, 'kind': kind, 'shape': [n, k], 'action': action,
                'gradient_stream': str(gradient_path), 'sensitivity': statistics,
                'quantizer': quantizer, 'payload_changed': raw != old_raw,
                'diagnostic_errors': {s: output_metrics(weights, old_q, new_q, fp[s], converted[s],
                    d if s == 'calibration' else None) for s in ROWS}}
            reports.append(report)
            print(json.dumps({'layer': layer, 'kind': kind, 'action': action,
                              'payload_changed': report['payload_changed']}), flush=True)
            del tensor, weights, old_q, new_q, raw, old_raw
            if action == 'new_source_quantization':
                del codes, scales, biases, buffer
    gc.collect()
    return path, reports


def preflight(output):
    source = read(SOURCE/'result.json')
    scope = frozen_scope(source)
    if read(SOURCE/'independent-readback.json').get('audit_passed') is not True:
        raise ValueError('audited original conversion required')
    sensitivity_preflight(source)
    binary = Path(source['binary'])
    receipt = read(Path(str(binary)+'.json'))
    control = read(Path(source['control'])/'result.json')
    if (receipt != source['receipt'] or control.get('receipt') != receipt
            or control.get('plan') != source['plan'] or control.get('completed') is not True
            or control.get('all_frozen_tokens_equal') is not True or control.get('streams_compared') != 47):
        raise ValueError('same frozen binary complete47-stream BF16 control required')
    reference = load_reference(Path(source['asymmetric_reference']), receipt, source['plan'],
        task_source=source['task_trajectories_from'], task_source_receipt=source['task_source_receipt'])
    registration = read(REGISTRATION)
    if (registration.get('schema') != SCHEMA or registration.get('registered_before_run') is not True
            or registration.get('output') != str(output) or registration.get('layers') != LAYERS
            or registration.get('rows') != ROWS or registration.get('groups_per_projection') != 1
            or registration.get('sensitivity') != str(SENSITIVITY)
            or registration.get('source') != str(SOURCE)):
        raise ValueError('fixed preregistered source conversion required')
    return source, scope, binary, reference, registration


def run(output):
    started = time.perf_counter()
    source, scope, binary, reference, registration = preflight(output)
    output.mkdir(exist_ok=False)
    for name in ('native', 'inputs', 'weights'):
        (output/name).mkdir()
    save(output/'plan.json', source['plan'])
    save(output/'inputs/position-mapping.json', layout(source['plan']))
    protocol = {**registration, 'completed': False, 'binary': str(binary), 'receipt': source['receipt'],
        'plan': source['plan'], 'manifest': MANIFEST, 'core': None, 'hardware': platform.platform(),
        'torch': torch.__version__, 'numpy': np.__version__, 'precision': 'Q4 MLP weights; BF16 activations/metadata; FP32 quantizer feedback; FP64 inverse factor',
        'seed': 42, 'cache_reuse': False, 'parameter_updates': 0}
    save(output/'protocol.json', protocol)
    index = read(Path(MODEL)/'model.safetensors.index.json')['weight_map']
    worker, lock, decisions, reports = None, None, [], []
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        worker = SequentialWorker(binary, output, output/'native', lock)
        save(output/'loaded.json', worker.loaded)
        worker.control('begin', 0, seq_plan=str(output/'plan.json'), seq_dry_run=False)
        with ExitStack() as stack:
            shards, readers = {}, {}
            for layer, kind, _, _ in scope:
                key = f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
                filename = index[key]
                if Path(filename).name != filename:
                    raise ValueError('local weight shard basename required')
                if filename not in shards:
                    shards[filename] = stack.enter_context(safe_open(Path(MODEL)/filename, framework='pt', device='cpu'))
                readers[key] = shards[filename]
            accepted = []
            for layer in range(24):
                selected = layer in LAYERS
                gate = worker.control('inputs', layer)
                if selected:
                    values = selected_inputs(gate, source['plan'], layer, 1, output/'native', output/'inputs')
                    path, rows = convert_stage(layer, 3, values, reference, readers, output)
                    reports.extend(rows); del values
                else:
                    path = old_stage(layer, 3)
                act = worker.control('gateup', layer, seq_payload=str(path))
                down = {}
                if layer != 23:
                    if selected:
                        values = selected_inputs(act, source['plan'], layer, 2, output/'native', output/'inputs')
                        path, rows = convert_stage(layer, 4, values, reference, readers, output)
                        reports.extend(rows); del values
                    else:
                        path = old_stage(layer, 4)
                    down = {'seq_payload': str(path)}
                worker.control('down', layer, **down)
                verification = worker.control('verify', layer) if layer in (0, 12, 23) else None
                if verification is not None and (type(verification.get('verified_values')) is not int
                                                 or verification['verified_values'] <= 0):
                    raise ValueError('real-prefix normal-forward control failed')
                decision = worker.control('decide', layer, seq_accept=selected)
                if selected:
                    accepted.append(layer)
                if decision.get('accepted_layer_mask') != sum(1<<number for number in accepted):
                    raise ValueError('fixed prefix mask changed')
                decisions.append({'layer': layer, 'accepted': selected, 'native': decision,
                    'normal_forward_control': verification,
                    'temporary_original_stage_restored': not selected})
                save(output/'progress.json', {'completed': False, 'completed_layers': layer+1,
                    'projections': reports, 'decisions': decisions, 'elapsed_seconds': time.perf_counter()-started})
                print(json.dumps({'completed_layers': layer+1, 'elapsed_seconds': time.perf_counter()-started}), flush=True)
        if accepted != LAYERS or len(reports) != 32 or decision.get('phase') != 'complete':
            raise ValueError('all24 stages and exactly32 fixed projections required')
        artifact = write_sparse(output/'candidate.bin', MANIFEST, '0'*64, MASK,
                                sparse_payloads(output/'weights', LAYERS))
        worker.close(); exit_code = worker.process.returncode; worker = None
        if exit_code:
            raise RuntimeError('sequential worker failed on close')
        save(output/'result.json', {**protocol, 'completed': True, 'projections': reports,
            'decisions': decisions, 'accepted_layers': accepted, 'layer_mask': MASK, 'artifact': artifact,
            'worker_exit_code': exit_code, 'ordinary_native_screen_run': False,
            'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'method': 'Darwin Python RSS; native worker memory separately in events, not total peak or deployment residency'}})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'error': repr(error), 'projections': reports,
            'completed_layers': len(decisions), 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            if lock is not None:
                lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('new output on mounted experiment volume required')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    run(args.output)


if __name__ == '__main__':
    main()
