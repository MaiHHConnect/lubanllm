"""Capture complete frozen native teacher/student tail trajectories; no fitting."""
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from benchmark_native_decode import IDENTITY, MANIFEST, save
from benchmark_native_mixed import MixedWorker
from capture_q4_native import load_frozen
from diagnose_q4_task_candidate import ordinary_loaded, read
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import make_plan
from probe_q4_paired_tail import tail_parts
from q4_packed_format import encode_header
from q4_task_trajectories import load_task_trajectories
from study_q4_column_protection import ROOT, SOURCE, MASK, frozen_scope
from trace_q4_divergence import VOCAB, WIDTHS

BINARY = ROOT/'paired-tail-trace-build/zig-out/bin/luban-worker'
OUTPUT = ROOT/'paired-q4-tail-capture-v1'
PROBE = ROOT/'paired-q4-tail-probe-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_paired_tail_capture_20261016.json'
TAIL_WIDTHS = {'h': 2048, 'act': 6144, 'down': 2048}
COUNTS = {'calibration': 3897, 'observation': 827}


def trajectory_layout(records, plan, vocab=VOCAB):
    """One row per actual next-token prediction, including the unforwarded final target."""
    counts = dict.fromkeys(COUNTS, 0)
    cases = []
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate trajectory IDs')
    for record, planned in zip(records, plan['cases'], strict=True):
        prompt, targets = record['prompt_tokens'], record['generated_tokens']
        split = record['split']; identifier = record['id']
        if (split not in counts or planned['split'] != split or identifier != planned['id']
                or not identifier or any(not (c.isascii() and (c.isalnum() or c in '-_')) for c in identifier)
                or not prompt or not 1 <= len(targets) <= 256 or len(prompt)+len(targets)-1 > 2048
                or any(type(t) is not int or not 0 <= t < vocab for t in prompt+targets)
                or prompt != planned['tokens'] or targets[:-1] != planned['forced_tokens']
                or any(t in plan['eos_ids'] for t in targets[:-1])
                or record['complete'] is not (targets[-1] in plan['eos_ids'])
                or record['finish_reason'] != ('stop' if record['complete'] else 'length')):
            raise ValueError('complete original bounded teacher-forced history required')
        start = counts[split]; counts[split] += len(targets)
        cases.append({'id': identifier, 'split': split, 'tokens': prompt, 'targets': targets,
                      'forced_tokens': targets[:-1], 'row_start': start, 'row_end': counts[split],
                      'prediction_positions': list(range(len(prompt)-1, len(prompt)+len(targets)-1)),
                      'source_complete': record['complete'], 'source_finish_reason': record['finish_reason']})
    return {'order': 'plan case order within split, then all predicted targets in order',
            'rows_per_split': counts, 'cases': cases}


def read_frame_file(path, count):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != count*2:
        raise ValueError('exact regular BF16 frame dimensions required')
    raw = np.fromfile(path, dtype='<u2')
    if len(raw) != count or np.any((raw & 0x7f80) == 0x7f80):
        raise ValueError('finite complete BF16 frame required')
    return (raw.astype(np.uint32) << 16).view(np.float32)


def check_frame(event, directory, request, step):
    tail = request.get('tail_only', False)
    if (event.get('event') != 'trace' or event.get('id') != request['id'] or event.get('step') != step
            or event.get('vocab') != VOCAB or event.get('committed_tokens') != len(request['tokens'])+step):
        raise ValueError('trace identity, sequence or committed prefix changed')
    if tail:
        if (event.get('tail_only') is not True or event.get('layer') != 23
                or event.get('tail_values') != sum(TAIL_WIDTHS.values())
                or any(k in event for k in ('layers', 'layer_values', 'layers_path'))):
            raise ValueError('tail-only layout must not impersonate full layer trace')
        kind, count = 'tail', sum(TAIL_WIDTHS.values())
    else:
        if event.get('layers') != 24 or event.get('layer_values') != sum(WIDTHS.values()) or event.get('tail_only', False):
            raise ValueError('full trace layout changed')
        kind, count = 'layers', 24*sum(WIDTHS.values())
    result = {}
    for name, width in [('logits', VOCAB), (kind, count)]:
        path = directory/f'{request["id"]}-step{step:04d}.{name}.bf16'
        if event.get(name+'_path') != str(path):
            raise ValueError('trace path escaped its exact request/frame')
        result[name] = read_frame_file(path, width)
    if int(np.argmax(result['logits'])) != event.get('top1'):
        raise ValueError('stored full logits disagree with native greedy')
    return result


def collect(worker, case, eos, identifier, *, tail=True, inspect=None):
    forced = case['forced_tokens']
    if len(forced) > (255 if tail else 64) or any(t in eos for t in forced):
        raise ValueError('invalid forced-token trace boundary')
    request = dict(id=identifier, tokens=case['tokens'], forced_tokens=forced,
                   max_tokens=len(forced)+1, eos_ids=eos, temperature=0, seed=42, cache_reuse=False)
    if tail:
        request['tail_only'] = True
    with (worker.out/'requests.jsonl').open('a') as log:
        log.write(json.dumps(request)+'\n')
    worker.process.stdin.write(json.dumps(request)+'\n'); worker.process.stdin.flush()
    traces = []
    while True:
        event = worker.receive()
        if event.get('event') == 'trace':
            step = len(traces)
            if step > len(forced):
                raise ValueError('extra trace frame')
            frame = check_frame(event, worker.out/'traces', request, step)
            if inspect is not None:
                inspect(step, frame)
            traces.append(event)
            del frame
        elif event.get('event') == 'result':
            expected = dict(event='result', id=identifier, tokens=[e['top1'] for e in traces],
                forced_tokens=forced, prompt_tokens=len(case['tokens']), generated_tokens=len(forced)+1,
                finish_reason='trace', complete=True, trace_only=True, cached_prompt_tokens=0)
            if len(traces) != len(forced)+1 or any(event.get(k) != v for k, v in expected.items()):
                raise ValueError('trace terminal differs from complete streamed predictions')
            return {'request': request, 'events': traces, 'result': event}
        else:
            raise ValueError('unexpected trace event: '+str(event))


def checked_worker(directory, mode, lock):
    (directory/'traces').mkdir(parents=True)
    precision = 'bf16' if mode == 'bf16' else 'mixed-q4-g16-packed'
    worker = MixedWorker(BINARY, 'bf16' if mode == 'bf16' else 'candidate_mixed', directory, lock,
        precision=precision, calibration_profile=None if mode == 'bf16' else SOURCE/'gptq-sequential.bin',
        trace_directory=directory/'traces', core=None)
    try:
        ordinary_loaded(worker.loaded, precision)
        expected = (0, 0, 0) if mode == 'bf16' else (2, 0, 0)
        if (tuple(worker.loaded.get(k) for k in ('packed_version', 'residual_rank', 'active_residual_projections')) != expected
                or worker.loaded.get('core_file_sha256') is not None or worker.loaded.get('core_tensor_sha256') is not None):
            raise ValueError('original BF16/packed-v2 without correction or core required')
    except BaseException:
        worker.close()
        raise
    return worker


def ordinary(worker, case, expected, identifier, eos):
    result = worker.request(case['tokens'], identifier, max_tokens=2, eos_ids=eos, temperature=0, seed=42, reuse=False)
    if (any(result.get(k) != expected.get(k) for k in ('tokens', 'finish_reason', 'complete'))
            or result.get('prompt_tokens') != len(case['tokens']) or result.get('generated_tokens') != 2
            or result.get('cached_prompt_tokens') != 0):
        raise ValueError('ordinary two-token frozen probe changed')
    return result


def preflight():
    source = read(SOURCE/'result.json'); frozen_scope(source)
    if read(SOURCE/'independent-readback.json').get('audit_passed') is not True:
        raise ValueError('audited original asymmetric Q4 required')
    protocol, originals = load_frozen(Path(source['source']))
    tasks = load_task_trajectories(Path(source['task_trajectories_from']), source['task_source_receipt'],
                                  make_plan(protocol, originals), protocol)
    records = originals+tasks
    if make_plan(protocol, records) != source['plan']:
        raise ValueError('original complete 36-case plan changed')
    mapping = trajectory_layout(records, source['plan'], protocol['vocab_size'])
    if (len(records) != 36 or sum(r['split']=='calibration' for r in records) != 32
            or mapping['rows_per_split'] != COUNTS):
        raise ValueError('exact registered full prediction counts required')
    carrier = SOURCE/'gptq-sequential.bin'
    if carrier.is_symlink() or not carrier.is_file() or carrier.stat().st_size != source['artifact']['bytes']:
        raise ValueError('frozen original packed carrier size changed')
    with carrier.open('rb') as stream:
        if stream.read(148) != encode_header(MANIFEST, '0'*64, MASK):
            raise ValueError('original coreNone packed-v2 fixed32 mask changed')
    probe = read(PROBE/'result.json')
    if (probe.get('completed') is not True or read(PROBE/'independent-readback.json').get('audit_passed') is not True
            or probe.get('source_receipt') != source['receipt'] or probe.get('carrier') != str(carrier)
            or probe.get('manifest') != MANIFEST or probe.get('core') is not None
            or probe['case']['tokens'] != mapping['cases'][0]['tokens']
            or probe['forced_tokens'] != mapping['cases'][0]['forced_tokens'][:1]):
        raise ValueError('audited original paired probe binding required')
    return source, mapping, probe


def main():
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('mounted experiment volume required')
    source, mapping, probe = preflight()
    registration = read(REGISTRATION); receipt = read(Path(str(BINARY)+'.json'))
    if (registration.get('registered_before_run') is not True or registration.get('case_count') != 36
            or registration.get('prediction_positions_per_model') != COUNTS
            or registration.get('schema') != 'q4-paired-tail-capture-v1'
            or registration.get('build') != str(BINARY.parents[2]) or not BINARY.is_file()):
        raise ValueError('fixed registered paired-tail build and scope required')
    OUTPUT.mkdir(exist_ok=False)
    metadata = {**registration, 'completed': False, 'receipt': receipt, 'binary': str(BINARY),
        'source_receipt': source['receipt'], 'source_plan': source['plan'], 'probe_receipt': probe['receipt'],
        'manifest': MANIFEST, 'core': None, 'seed': 42, 'temperature': 0, 'cache_reuse': False,
        'tail_widths': TAIL_WIDTHS, 'vocab': VOCAB, 'position_mapping': 'position-mapping.json',
        'hardware': platform.platform(), 'precision': 'original native BF16 teacher / packed-v2 Q4 student; stored BF16'}
    save(OUTPUT/'protocol.json', metadata); save(OUTPUT/'position-mapping.json', mapping)
    started = time.perf_counter(); lock = worker = None
    controls, reports = {}, {}
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        short = {'tokens': mapping['cases'][0]['tokens'], 'forced_tokens': mapping['cases'][0]['forced_tokens'][:1]}
        eos = source['plan']['eos_ids']
        # Both precision controls pass before any complete trajectory is captured.
        for mode in ('bf16', 'q4'):
            worker = checked_worker(OUTPUT/'controls'/mode, mode, lock)
            before = ordinary(worker, short, probe['native'][mode]['before'], 'before', eos)
            full_frames = []
            def full_control(step, frame):
                for kind in ('layers', 'logits'):
                    old = read_frame_file(PROBE/mode/'traces'/f'paired-cal00-step{step:04d}.{kind}.bf16', len(frame[kind]))
                    if not np.array_equal(old.view(np.uint32), frame[kind].view(np.uint32)):
                        raise ValueError('new full trace differs bitwise from original probe')
                parts = tail_parts(frame['layers'])
                full_frames.append({'tail': np.concatenate([parts[k] for k in TAIL_WIDTHS]), 'logits': frame['logits']})
            full = collect(worker, short, eos, 'control-full', tail=False, inspect=full_control)
            def tail_control(step, frame):
                if any(not np.array_equal(frame[k].view(np.uint32), full_frames[step][k].view(np.uint32)) for k in ('tail','logits')):
                    raise ValueError('new tail trace differs bitwise from full trace')
            tail = collect(worker, short, eos, 'control-tail', inspect=tail_control)
            after = ordinary(worker, short, probe['native'][mode]['after'], 'after', eos)
            controls[mode] = {'loaded': worker.loaded, 'before': before, 'full': full, 'tail': tail, 'after': after, 'exact': True}
            worker.close()
            if worker.process.returncode != 0:
                raise RuntimeError('control worker failed on close')
            worker = None
            save(OUTPUT/'controls.json', controls)
        for mode in ('bf16', 'q4'):
            worker = checked_worker(OUTPUT/mode, mode, lock)
            report = {'loaded': worker.loaded, 'cases': [], 'before': ordinary(worker, short, probe['native'][mode]['before'], 'before', eos)}
            reports[mode] = report
            for case in mapping['cases']:
                captured = collect(worker, case, eos, case['id'])
                # Teacher targets are frozen real outputs; student predictions may differ.
                if mode == 'bf16' and captured['result']['tokens'] != case['targets']:
                    raise ValueError('BF16 teacher differs from original complete trajectory')
                report['cases'].append({'id': case['id'], 'split': case['split'], **captured})
                save(OUTPUT/'progress.json', {'completed': False, 'mode': mode, 'cases_completed': len(report['cases']),
                                            'last_case': case['id'], 'elapsed_seconds': time.perf_counter()-started})
                print(json.dumps({'mode': mode, 'case': case['id'], 'positions': len(case['targets'])}), flush=True)
            report['after'] = ordinary(worker, short, probe['native'][mode]['after'], 'after', eos)
            worker.close()
            if worker.process.returncode != 0:
                raise RuntimeError('capture worker failed on close')
            worker = None
            save(OUTPUT/(mode+'-report.json'), report)
        save(OUTPUT/'result.json', {**metadata, 'completed': True, 'controls': controls, 'reports': reports,
            'all_teacher_targets_exact': True, 'rows_per_model': COUNTS, 'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'scope': 'Python high-water RSS; native loaded/results contain physical footprint samples and lifetime peak; not combined GPU peak'}})
    except BaseException as error:
        save(OUTPUT/'failure.json', {**metadata, 'completed': False, 'error': repr(error), 'controls': controls, 'reports': reports,
                                   'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            if lock is not None:
                lock.close()


if __name__ == '__main__':
    main()
