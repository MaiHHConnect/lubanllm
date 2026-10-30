"""Diagnose actual packed-v2/v3 prefix inputs on frozen trajectories; never fit or qualify."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import resource
import struct
import subprocess
import threading
import time

import numpy as np

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from capture_q4_native import (HEADER_BYTES, ORDER, ROW_VALUES, append_streams,
                               integers, load_frozen, make_request)
from capture_q4_task_reference import layout, load_reference, stream_file
from diagnose_q4_task_candidate import IDS, diagnostic_result, exact_generations
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import make_plan
from q4_task_trajectories import load_task_trajectories
from study_q4_column_protection import LAYERS, MASK, ROOT, ROWS, SOURCE, frozen_scope
from trace_q4_divergence import WIDTHS, error_stats, read_bf16, trace_request

MAGIC = b'LNACQ01\n'
SCHEMA = 'q4-corrected-inputs-diagnostic-v1'
FIT = ROOT/'column-fit-native-v1'


def read(path):
    return json.loads(Path(path).read_text())


def checked_loaded(event, version, active, *, capture):
    expected = {'event': 'loaded', 'base_identity': IDENTITY, 'manifest_sha256': MANIFEST,
        'core_loaded': False, 'core_file_sha256': None, 'retained_original_weight_bytes': 0,
        'inference_precision': 'mixed-q4-g16-packed', 'quantized_projections': 32,
        'packed_version': version, 'residual_rank': 8 if version == 3 else 0,
        'active_residual_projections': active, 'research_capture_inputs': capture,
        'research_intervention': False, 'research_sequential': False}
    if version not in (2, 3) or not 0 <= active <= 32 or (version == 2 and active):
        raise ValueError('invalid packed capture scope')
    if any(event.get(k) != v or (isinstance(v, bool) and event.get(k) is not v) for k, v in expected.items()):
        raise ValueError('packed precision, source, capture scope or retained originals changed')


def checked_prediction(result, request, vocab):
    # Forced histories are frozen; emitted predictions are from this packed worker, not BF16 labels.
    expected = {'event': 'result', 'id': request['id'], 'prompt_tokens': len(request['tokens']),
        'generated_tokens': len(request['forced_tokens'])+1, 'finish_reason': 'capture',
        'complete': True, 'capture_only': True, 'cached_prompt_tokens': 0}
    tokens = result.get('tokens')
    if (not integers(tokens, 0, vocab) or len(tokens) != expected['generated_tokens']
            or request['max_tokens'] != expected['generated_tokens']
            or any(result.get(k) != v or (isinstance(v, bool) and result.get(k) is not v)
                   for k, v in expected.items())):
        raise ValueError('packed capture actual prediction scope changed')


def read_capture(event, request, directory):
    """Q4-only carrier reader. The original BF16 reader remains strict and unchanged."""
    positions = request['capture_positions']
    p, f, rows = len(request['tokens']), len(request['forced_tokens']), len(positions)
    if (not positions or positions != sorted(set(positions))
            or not integers(positions, 0, p+f) or not 1 <= rows <= 288):
        raise ValueError('invalid packed capture positions')
    path = Path(directory)/(request['id']+'.inputs.bin')
    expected = {'event': 'capture', 'id': request['id'], 'path': str(path), 'rows': rows,
        'row_values': ROW_VALUES, 'positions': positions, 'prompt_tokens': p, 'forced_tokens': f}
    if any(event.get(k) != v for k, v in expected.items()):
        raise ValueError('packed capture event differs from request')
    if path.is_symlink() or not path.is_file() or path.stat().st_size != HEADER_BYTES+rows*4+rows*ROW_VALUES*2:
        raise ValueError('complete regular packed carrier with exact size required')
    with path.open('rb') as stream:
        header = MAGIC+struct.pack('<6I', rows, ROW_VALUES, p, f, 24, 47)+MANIFEST.encode()+b'0'*64
        if stream.read(HEADER_BYTES) != header:
            raise ValueError('Q4 capture magic or source binding mismatch')
        if np.frombuffer(stream.read(rows*4), dtype='<u4').tolist() != positions:
            raise ValueError('packed capture positions mismatch')
    data = np.memmap(path, dtype='<u2', mode='r', offset=HEADER_BYTES+rows*4, shape=(rows, ROW_VALUES))
    if np.any((data & 0x7f80) == 0x7f80):
        raise ValueError('nonfinite packed BF16 capture')
    return data


def assemble(plan, requests, events, directory, output):
    if ([r['id'] for r in requests] != [c['id'] for c in plan['cases']]
            or len(events) != len(requests)):
        raise ValueError('all frozen cases in plan order required')
    mapping, paths = layout(plan), {}
    for case, request, event in zip(plan['cases'], requests, events, strict=True):
        if (request['tokens'], request['forced_tokens'], request['capture_positions']) != (
                case['tokens'], case['forced_tokens'], case['positions']):
            raise ValueError('capture request differs from frozen plan')
        data = read_capture(event, request, directory)
        del data
        paths[case['id']] = (event['path'], len(case['positions']))
    counts = {name: dict.fromkeys(ROWS, 0) for name, _ in ORDER}
    for segment in mapping['segments']:
        path, rows = paths[segment['id']]
        data = np.memmap(path, dtype='<u2', mode='r', offset=HEADER_BYTES+rows*4, shape=(rows, ROW_VALUES))
        append_streams(data[segment['carrier_row_start']:segment['carrier_row_end']], segment['split'], output, counts)
        segment['path'] = path
        del data
    if any(c != mapping['rows_per_split'] for c in counts.values()):
        raise ValueError('assembled stream row counts differ from plan')
    save(output/'position-mapping.json', mapping)
    return mapping


def exact_stream_names(layers):
    # Rejected-layer down inputs were saved after temporary Q4 gate/up, before rollback.
    return [name for name, _ in ORDER if name.endswith('gateup') or int(name[5:7]) in layers]


def compare_original_streams(directory, original, rows, layers):
    compared = []
    for name, width in ORDER:
        if name not in exact_stream_names(layers):
            continue
        for split, count in rows.items():
            left = np.fromfile(stream_file(directory, name, split, count, width), dtype='<u2')
            right = np.fromfile(stream_file(original, name, split, count, width), dtype='<u2')
            if not np.array_equal(left, right):
                raise ValueError('final packed-v2 differs from sequential accepted-prefix inputs: '+name+'/'+split)
            compared.append({'stream': name, 'split': split, 'values': len(left), 'exact': True})
    return compared


def trace_inputs(raw):
    raw = np.asarray(raw)
    if raw.dtype != np.dtype('uint16') or raw.shape != (24, sum(WIDTHS.values())):
        raise ValueError('24-layer BF16 trace layout required')
    offsets, at = {}, 0
    for name, width in WIDTHS.items():
        offsets[name] = at; at += width
    return np.concatenate([raw[int(name[5:7]), offsets['mlp_input' if name.endswith('gateup') else 'act']:
        offsets['mlp_input' if name.endswith('gateup') else 'act']+width] for name, width in ORDER])


class PackedCaptureWorker(MixedWorker):
    """Existing worker lifecycle and ordinary requests, plus the exclusive packed capture CLI."""
    def __init__(self, binary, output, directory, lock, profile, version, active):
        self.name, self.out = 'packed_capture', output
        self.events = (output/'native-events.jsonl').open('x')
        self.stderr = (output/'native-stderr.log').open('x')
        self.cancel_read, self.cancel_write = os.pipe()
        self.messages = queue.Queue()
        command = [str(binary), MODEL, MANIFEST, str(self.cancel_read),
                   '--mixed-q4-g16-packed='+str(profile), '--research-capture-inputs='+str(directory)]
        save(output/'invocation.json', {'command': command})
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, text=True, pass_fds=(self.cancel_read, lock.fd), bufsize=1)
        except BaseException:
            os.close(self.cancel_read); os.close(self.cancel_write)
            self.events.close(); self.stderr.close()
            raise
        def reader():
            try:
                for line in self.process.stdout:
                    self.events.write(line); self.events.flush()
                    self.messages.put(json.loads(line))
            except BaseException as error:
                self.messages.put(error)
            finally:
                self.messages.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            self.loaded = self.receive(300)
            checked_loaded(self.loaded, version, active, capture=True)
        except BaseException:
            self.close()
            raise

    def capture(self, request, vocab):
        with (self.out/'requests.jsonl').open('a') as log:
            log.write(json.dumps(request)+'\n')
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        captured = None
        while True:
            event = self.receive(300)
            if event.get('id') != request['id']:
                raise ValueError('capture event request ID changed')
            if event.get('event') == 'capture' and captured is None:
                captured = event
            elif event.get('event') == 'result' and captured is not None:
                checked_prediction(event, request, vocab)
                return captured, event
            else:
                raise ValueError('unexpected packed capture event: '+str(event))


def ordinary(worker, specs, expected, tokenizer, vocab, prefix):
    results = []
    for spec in specs:
        case, identifier = spec['case'], prefix+'-'+spec['id']
        raw = worker.request(case['tokens'], identifier, max_tokens=case['max_tokens'],
            temperature=0, seed=42, reuse=False, eos_ids=case['eos_ids'])
        if raw.get('event') != 'result' or raw.get('id') != identifier:
            raise ValueError('ordinary request identity changed')
        results.append(diagnostic_result(spec, raw, tokenizer, vocab))
    exact_generations(results, expected)
    return results


def preflight():
    source = read(SOURCE/'result.json'); frozen_scope(source)
    audit = read(SOURCE/'independent-readback.json')
    if audit.get('audit_passed') is not True or audit.get('ordinary_reload16_exact') is not True:
        raise ValueError('audited frozen GPTAQ source required')
    protocol, original = load_frozen(Path(source['source']))
    tasks = load_task_trajectories(Path(source['task_trajectories_from']), source['task_source_receipt'],
                                   make_plan(protocol, original), protocol)
    records, plan = original+tasks, source['plan']
    if make_plan(protocol, records) != plan or len(records) != 36 or layout(plan)['rows_per_split'] != ROWS:
        raise ValueError('unchanged36 actual frozen trajectories required')
    reference = load_reference(Path(source['asymmetric_reference']), source['receipt'], plan,
        task_source=source['task_trajectories_from'], task_source_receipt=source['task_source_receipt'])
    prepared, screen = read(FIT/'prepare-result.json'), read(FIT/'screen/result.json')
    old = read(ROOT/'asymmetric-four-screen-v1/result.json')
    if (prepared.get('completed') is not True or prepared.get('schema') != 'q4-column-fit-assembly-v1'
            or prepared.get('selection_policy') != 'normalized_column_ridge_fit' or prepared.get('ridge') != .01
            or prepared.get('source') != str(SOURCE) or prepared.get('source_receipt') != source['receipt']
            or prepared.get('plan') != plan or prepared.get('layer_mask') != MASK
            or prepared.get('rank') != 8 or prepared.get('quantized_projections') != 32
            or screen.get('completed') is not True or screen.get('prepared') != prepared
            or read(FIT/'screen/independent-readback.json').get('audit_passed') is not True
            or old.get('completed') is not True or old.get('conversion') != str(SOURCE)
            or screen.get('cases') != old.get('cases') or len(screen['cases']) != 4
            or {c['id'] for c in screen['cases']} != IDS):
        raise ValueError('audited fixed fitted carrier and original four compatibility evidence required')
    carrier = prepared['carriers']['candidate']
    if (carrier.get('path') != str(FIT/'candidate.bin') or carrier.get('packed_version') != 3
            or carrier.get('residual_rank') != 8 or carrier.get('active_residual_projections') != prepared['active_residual_projections']):
        raise ValueError('fitted carrier scope changed')
    modes = {'v2': {'path': SOURCE/'gptq-sequential.bin', 'version': 2, 'active': 0,
                    'expected': [r for r in old['observations'] if r['phase'] == 'candidate']},
             'fitv3': {'path': FIT/'candidate.bin', 'version': 3,
                       'active': prepared['active_residual_projections'], 'expected': screen['four']}}
    for name, mode in modes.items():
        expected_size = source['artifact']['bytes'] if name == 'v2' else carrier['bytes']
        if mode['path'].is_symlink() or not mode['path'].is_file() or mode['path'].stat().st_size != expected_size:
            raise ValueError('exact regular existing packed carrier required')
        checked_loaded(screen['loaded'], 3, prepared['active_residual_projections'], capture=False)
    return source, protocol, records, reference, screen['cases'], modes


def trace_control(binary, output, lock, mode, record, evidence, eos, specs, tokenizer, vocab):
    directory = output/'trace-control'; directory.mkdir()
    traces_dir = directory/'traces'; traces_dir.mkdir()
    worker = MixedWorker(binary, 'candidate_mixed', directory, lock, precision='mixed-q4-g16-packed',
                         calibration_profile=mode['path'], trace_directory=traces_dir, core=None)
    try:
        checked_loaded(worker.loaded, 3, mode['active'], capture=False)
        save(directory/'loaded.json', worker.loaded)
        request = make_request(record, eos)
        if record['id'] != 'cal-00' or len(request['forced_tokens']) < 1:
            raise ValueError('fixed first calibration trace requires one original continuation')
        traces = trace_request(worker, {'tokens': record['prompt_tokens'], 'eos_ids': eos},
                               request['forced_tokens'][:1], 'cal0-two-frames')
        trace_result = json.loads((directory/'candidate_mixed-events.jsonl').read_text().splitlines()[-1])
        expected_trace = {'event': 'result', 'id': 'cal0-two-frames',
            'tokens': [t['top1'] for t in traces], 'forced_tokens': request['forced_tokens'][:1],
            'prompt_tokens': len(record['prompt_tokens']), 'generated_tokens': 2, 'complete': True,
            'finish_reason': 'trace', 'trace_only': True, 'cached_prompt_tokens': 0}
        if any(trace_result.get(k) != v for k, v in expected_trace.items()):
            raise ValueError('trace completion or actual predictions differ from recorded frames')
        captured = read_capture(evidence['capture'], request, output/'fitv3/carriers')
        comparisons = []
        for step, trace in enumerate(traces):
            prefix = traces_dir/f'cal0-two-frames-step{step:04d}'
            if (trace.get('id') != 'cal0-two-frames' or trace.get('layers') != 24
                    or trace.get('layer_values') != sum(WIDTHS.values()) or trace.get('vocab') != vocab
                    or trace.get('committed_tokens') != len(record['prompt_tokens'])+step
                    or trace.get('layers_path') != str(prefix)+'.layers.bf16'
                    or trace.get('logits_path') != str(prefix)+'.logits.bf16'):
                raise ValueError('trace identity/layout/prefix changed')
            values = read_bf16(trace['layers_path'], 24*sum(WIDTHS.values()))
            raw = (values.view(np.uint32)>>16).astype(np.uint16).reshape(24, -1)
            position = len(record['prompt_tokens'])-1+step
            selected = captured[request['capture_positions'].index(position)]
            logits = read_bf16(trace['logits_path'], vocab)
            if (not np.array_equal(trace_inputs(raw), selected) or int(np.argmax(logits)) != trace['top1']
                    or trace['top1'] != evidence['result']['tokens'][step]):
                raise ValueError('separate ordinary trace differs from actual capture streams/prediction')
            comparisons.append({'step': step, 'position': position, 'streams': 47,
                                'values': ROW_VALUES, 'top1': trace['top1'], 'exact': True})
        mmlu = [s for s in specs if s['id'] == 'mmlu-2804']
        after = ordinary(worker, mmlu, [r for r in mode['expected'] if r['id'] == 'mmlu-2804'],
                         tokenizer, vocab, 'trace-after')
        worker.close()
        if worker.process.returncode:
            raise RuntimeError('trace worker failed')
        result = {'passed': True, 'comparisons': comparisons, 'ordinary_after': after, 'loaded': worker.loaded, 'trace_result': trace_result}
        save(directory/'result.json', result)
        return result
    finally:
        if not worker.events.closed:
            worker.close()


def input_metrics(output, reference):
    reports = []
    for name, width in ORDER:
        for split, count in ROWS.items():
            values = {}
            for mode, directory in [('bf16', reference['directory']), ('v2', output/'v2/inputs'),
                                    ('fitv3', output/'fitv3/inputs')]:
                raw = np.fromfile(stream_file(directory, name, split, count, width), dtype='<u2')
                values[mode] = (raw.astype(np.uint32)<<16).view(np.float32)
                if not np.isfinite(values[mode]).all():
                    raise ValueError('nonfinite diagnostic input stream')
            reports.append({'stream': name, 'split': split, 'rows': count, 'width': width,
                'v2_vs_bf16': error_stats(values['bf16'], values['v2']),
                'fitv3_vs_bf16': error_stats(values['bf16'], values['fitv3']),
                'fitv3_vs_v2': error_stats(values['v2'], values['fitv3'])})
    return reports


def run(args):
    started = time.perf_counter()
    source, protocol, records, reference, specs, modes = preflight()
    binary, output = args.binary.resolve(), args.output.resolve()
    receipt = read(Path(str(binary)+'.json'))
    registration = read(Path(__file__).resolve().parents[1]/'research/q4_corrected_inputs_20261016.json')
    if registration.get('schema') != SCHEMA or registration.get('exact_sequential_streams') != exact_stream_names(LAYERS):
        raise ValueError('fixed actual-input preregistration required')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    output.mkdir(exist_ok=False)
    metadata = {**registration, 'binary': str(binary), 'receipt': receipt, 'plan': source['plan'],
        'source_receipt': source['receipt'], 'core': None, 'seed': 42, 'temperature': 0, 'cache_reuse': False,
        'qualified': False, 'product_adopted': False, 'training': False, 'teacher_calls': 0}
    save(output/'protocol.json', metadata)
    worker, lock, reports = None, None, {}
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        for name, mode in modes.items():
            directory = output/name; directory.mkdir()
            carriers = directory/'carriers'; carriers.mkdir()
            inputs = directory/'inputs'; inputs.mkdir()
            worker = PackedCaptureWorker(binary, directory, carriers, lock, mode['path'], mode['version'], mode['active'])
            save(directory/'loaded.json', worker.loaded)
            before = ordinary(worker, specs, mode['expected'], tokenizer, protocol['vocab_size'], 'before')
            save(directory/'ordinary-before.json', before)
            requests, evidence = [], []
            for record in records:
                request = make_request(record, source['plan']['eos_ids'])
                captured, result = worker.capture(request, protocol['vocab_size'])
                requests.append(request); evidence.append({'id': record['id'], 'capture': captured, 'result': result})
                save(directory/'captures.json', evidence)
            mmlu = [s for s in specs if s['id'] == 'mmlu-2804']
            after = ordinary(worker, mmlu, [r for r in mode['expected'] if r['id'] == 'mmlu-2804'],
                             tokenizer, protocol['vocab_size'], 'after')
            loaded = worker.loaded
            worker.close(); code = worker.process.returncode; worker = None
            if code:
                raise RuntimeError('packed capture worker failed')
            mapping = assemble(source['plan'], requests, [e['capture'] for e in evidence], carriers, inputs)
            control = compare_original_streams(inputs, SOURCE/'inputs', ROWS, LAYERS) if name == 'v2' else None
            reports[name] = {'source_artifact': str(mode['path']), 'loaded': loaded, 'ordinary_before': before,
                'ordinary_after': after, 'records': evidence, 'rows_per_split': mapping['rows_per_split'],
                'sequential_exact': control, 'inference_precision': loaded['inference_precision']}
            save(directory/'result.json', {**reports[name], 'completed': True})
        trace = trace_control(binary, output, lock, modes['fitv3'], records[0], reports['fitv3']['records'][0],
                              source['plan']['eos_ids'], specs, tokenizer, protocol['vocab_size'])
        metrics = input_metrics(output, reference)
        save(output/'input-metrics.json', metrics)
        save(output/'result.json', {**metadata, 'completed': True, 'modes': reports, 'trace_control': trace,
            'input_metrics': metrics, 'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'method': 'Darwin process peak RSS; worker proc_pid_rusage retained in raw loaded/results; not deployment residency'}})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'qualified': False, 'error': repr(error),
                                    'completed_modes': list(reports), 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        if worker is not None:
            worker.close()
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'corrected-inputs-diagnostic-v1')
    args = parser.parse_args()
    if (not os.path.ismount('/Volumes/移动数据库') or not args.binary.is_absolute()
            or not args.output.resolve().is_relative_to(ROOT)):
        raise ValueError('absolute binary and mounted external experiment output required')
    run(args)


if __name__ == '__main__':
    main()
