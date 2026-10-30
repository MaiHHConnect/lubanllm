"""Diagnose already observed Q4 failures, preserving native prefill/decode order.

This is post-hoc diagnosis, not calibration, a held-out test or product adoption.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer

from benchmark_native_decode import MODEL, IDENTITY, save
from benchmark_native_mixed import MixedWorker
from luban.process_lock import ModelProcessLock

IDS = ('coverage-new-train-1', 'preservation-replay-train-olive-path',
       'long-upper-16', 'mmlu-10358', '163648ee70fe43ca92746a1d27654c07')
WIDTHS = {'h': 2048, 'mixer_out': 2048, 'mlp_input': 2048,
          'gate': 6144, 'up': 6144, 'act': 6144, 'down': 2048}
VOCAB = 248320


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def read_bf16(path, count):
    raw = np.fromfile(path, dtype='<u2')
    if len(raw) != count:
        raise ValueError(f'wrong trace size: {path}: {len(raw)} != {count}')
    values = (raw.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f'nonfinite trace: {path}')
    return values


def error_stats(reference, actual):
    # Float64 analysis avoids confusing diagnostic rounding with model arithmetic.
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    diff = b - a
    return {'rmse': float(np.sqrt(np.mean(diff * diff))),
            'relative_l2': float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-30)),
            'max_abs': float(np.max(np.abs(diff))), 'different_values': int(np.count_nonzero(diff))}


def trace_request(worker, case, forced, name):
    request = {'id': name, 'tokens': case['tokens'], 'forced_tokens': forced,
               'max_tokens': len(forced)+1, 'eos_ids': case['eos_ids'],
               'temperature': 0, 'seed': 42, 'cache_reuse': False}
    with (worker.out/'requests.jsonl').open('a') as log:
        log.write(json.dumps(request)+'\n')
    worker.process.stdin.write(json.dumps(request)+'\n'); worker.process.stdin.flush()
    traces = []
    while True:
        event = worker.receive()
        if event['event'] == 'trace':
            traces.append(event)
        elif event['event'] == 'result':
            if len(traces) != len(forced)+1 or [x['step'] for x in traces] != list(range(len(forced)+1)):
                raise ValueError('missing or duplicate diagnostic steps')
            return traces
        else:
            raise RuntimeError(event)


def analyze(output, specs, tokenizer):
    reports, zero_exact = [], True
    for index, spec in enumerate(specs):
        for step in range(len(spec['forced_tokens'])+1):
            prefix = f'c{index}-step{step:04d}'
            logits = {mode: read_bf16(output/mode/'traces'/(prefix+'.logits.bf16'), VOCAB)
                      for mode in ('bf16', 'q4', 'zero', 'rank8', 'rank32')}
            states = {mode: read_bf16(output/mode/'traces'/(prefix+'.layers.bf16'), 24*sum(WIDTHS.values()))
                      .reshape(24, -1) for mode in logits}
            if not np.array_equal(logits['q4'], logits['zero']) or not np.array_equal(states['q4'], states['zero']):
                zero_exact = False
            nominated = sorted(set([int(np.argmax(x)) for x in logits.values()] +
                                   [x['tokens'][step] for x in spec['historical'].values() if len(x['tokens']) > step]))
            item = {'id': spec['id'], 'step': step, 'forced_prefix': spec['forced_tokens'][:step],
                    'prefix_text': tokenizer.decode(spec['forced_tokens'][:step]),
                    'token_labels': {str(t): tokenizer.decode([t]) for t in nominated}, 'modes': {}}
            ref_top = int(np.argmax(logits['bf16']))
            for mode in logits:
                values = logits[mode]
                # Stable sort and numpy argmax both select lowest ID on exact ties.
                top = np.argsort(-values, kind='stable')[:8]
                layers = []
                for layer in range(24):
                    at, metrics = 0, {}
                    for name, width in WIDTHS.items():
                        metrics[name] = error_stats(states['bf16'][layer, at:at+width], states[mode][layer, at:at+width])
                        at += width
                    # h is after post-attention residual/norm; down is pending MLP residual.
                    metrics['block_sum_fp32'] = error_stats(states['bf16'][layer, :2048]+states['bf16'][layer, -2048:],
                                                            states[mode][layer, :2048]+states[mode][layer, -2048:])
                    layers.append({'layer': layer, 'metrics': metrics})
                item['modes'][mode] = {'top1': int(top[0]), 'top8': [{'id': int(t), 'logit': float(values[t])} for t in top],
                    'nominated_logits': {str(t): float(values[t]) for t in nominated},
                    'bf16_top1_margin_to_best_other': float(values[ref_top] - np.max(np.delete(values, ref_top))),
                    'logits_error': error_stats(logits['bf16'], values), 'layers': layers}
            reports.append(item)
    if not zero_exact:
        raise ValueError('zero correction changed full traced logits or layer states')
    return {'steps': reports, 'zero_trace_exact': zero_exact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('binary', 'prior', 'profiles', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.resolve().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    rows = [json.loads(line) for line in (args.prior/'answers.jsonl').read_text().splitlines()]
    cases = {x['case']['id']: x['case'] for pairs in json.loads((args.prior/'cases.json').read_text()).values() for x in pairs}
    specs = []
    for id in IDS:
        history = {r['mode']: r for r in rows if r['case_id'] == id and r['mode'] != 'zero'}
        if set(history) != {'bf16', 'q4', 'rank8', 'rank32'}:
            raise ValueError('missing prior outputs')
        baseline = history['bf16']['tokens']
        differences = {m: first_difference(baseline, r['tokens']) for m, r in history.items()}
        last = max(d for d in differences.values() if d is not None)
        if last > 64 or last >= len(baseline) or any(t in cases[id]['eos_ids'] for t in baseline[:last]):
            raise ValueError('unsupported forced prefix')
        specs.append({'id': id, 'case': cases[id], 'forced_tokens': baseline[:last],
                      'first_differences': differences, 'historical': history})
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', {'method': 'original32-token-prefill-then-single-forced-decode',
         'prior': str(args.prior.resolve()), 'binary': str(args.binary.resolve()), 'profiles': str(args.profiles.resolve()),
         'seed': 42, 'temperature': 0, 'cache_reuse': False, 'layer_widths': WIDTHS,
         'limits': 'post-hoc diagnosis of known failures, no held-out claim; partial UTF8 token labels may be replacement characters',
         'scope': 'B45 core, no expert/router, no training or adoption; diagnostics timing is not inference performance',
         'cases': specs})
    tokenizer = Tokenizer.from_file(MODEL+'/tokenizer.json')
    start, events, controls = time.perf_counter(), {}, {}
    lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        for mode, precision, profile in [('bf16', 'bf16', None), ('q4', 'mixed-q4-g32', None),
                 ('zero', 'mixed-q4-g32-residual', 'rank8-zero.bin'), ('rank8', 'mixed-q4-g32-residual', 'rank8.bin'),
                 ('rank32', 'mixed-q4-g32-residual', 'rank32.bin')]:
            out = args.output/mode; (out/'traces').mkdir(parents=True)
            worker = MixedWorker(args.binary.resolve(), 'baseline_bf16' if mode=='bf16' else 'candidate_mixed', out, lock,
                                 precision, (args.profiles/profile).resolve() if profile else None, (out/'traces').resolve())
            events[mode], controls[mode] = [], []
            try:
                save(out/'loaded.json', worker.loaded)
                for index, spec in enumerate(specs):
                    c = spec['case']; old = spec['historical']['q4' if mode=='zero' else mode]
                    result = worker.request(c['tokens'], f'control-{index}', max_tokens=c['max_tokens'],
                                            temperature=0, seed=42, reuse=False, eos_ids=c['eos_ids'])
                    if any(result[k] != old[k] for k in ('tokens', 'finish_reason', 'complete')):
                        raise ValueError(f'trace build normal generation differs: {mode} {spec["id"]}')
                    controls[mode].append(result)
                    traces = trace_request(worker, c, spec['forced_tokens'], f'c{index}')
                    for step, event in enumerate(traces):
                        values = read_bf16(out/'traces'/f'c{index}-step{step:04d}.logits.bf16', VOCAB)
                        if int(np.argmax(values)) != event['top1']:
                            raise ValueError('CPU logits argmax differs from GPU greedy')
                        if old['tokens'][:step] == spec['forced_tokens'][:step] and event['top1'] != old['tokens'][step]:
                            raise ValueError('trace altered next token on matching natural prefix')
                    events[mode].append({'id': spec['id'], 'traces': traces})
                # A normal request after diagnostics catches leaked cache/trace state.
                c = specs[0]['case']; old = specs[0]['historical']['q4' if mode=='zero' else mode]
                after = worker.request(c['tokens'], 'control-after', max_tokens=c['max_tokens'], temperature=0,
                                       seed=42, reuse=False, eos_ids=c['eos_ids'])
                if any(after[k] != old[k] for k in ('tokens', 'finish_reason', 'complete')):
                    raise ValueError('diagnostics leaked into normal generation')
                controls[mode].append(after)
            finally:
                worker.close()
            if worker.process.returncode:
                raise RuntimeError('trace worker exited unsuccessfully')
            save(args.output/'progress.json', {'finished_mode': mode, 'elapsed_seconds': time.perf_counter()-start})
    finally:
        lock.close()
    save(args.output/'events.json', events); save(args.output/'controls.json', controls)
    result = analyze(args.output, specs, tokenizer)
    result.update(elapsed_seconds=time.perf_counter()-start, normal_generation_controls=30,
                  normal_controls_exact=True, cpu_gpu_argmax_exact=True, natural_prefix_top1_exact=True, product_adopted=False)
    save(args.output/'result.json', result)
    print(json.dumps({'completed': True, 'steps_per_mode': len(result['steps']), 'seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
