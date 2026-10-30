"""Two exposed first-token decisions; diagnostics never qualify or fit a model."""
import json
import os
from pathlib import Path
import time

import numpy as np
from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MODEL, save
from benchmark_native_mixed import MixedWorker
from diagnose_q4_task_candidate import diagnostic_result, ordinary_loaded, read
from luban.process_lock import ModelProcessLock
from trace_q4_divergence import WIDTHS, VOCAB, error_stats, read_bf16, trace_request
from verify_q4_sensitive_source import BINARY, CONTROLS, CONVERSION, preflight

ROOT = CONVERSION.parent
OUTPUT = ROOT/'sensitivity-source-graph-diagnostic-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_sensitive_graph_diagnostic_20261016.json'
IDS = ('seq-final-young-dog', 'mmlu-2804')


def logit_metrics(reference, actual, nominated):
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite([a, b]).all():
        raise ValueError('paired finite vocabulary vectors required')
    ref = int(np.argmax(a))
    top = np.argsort(-b, kind='stable')
    logp = a-a.max(); logp -= np.log(np.exp(logp).sum())
    logq = b-b.max(); logq -= np.log(np.exp(logq).sum())
    return {
        'top1': int(top[0]),
        'top10': [{'id': int(t), 'logit': float(b[t])} for t in top[:10]],
        'top_ties': np.flatnonzero(b == b[top[0]]).tolist(),
        'reference_top1_ties': np.flatnonzero(b == b[ref]).tolist(),
        'reference_top1_margin': float(b[ref]-np.max(np.delete(b, ref))),
        'nominated': {str(t): {'logit': float(b[t]),
            'rank': 1+int(np.count_nonzero(b > b[t]))+int(np.count_nonzero(b[:t] == b[t]))}
            for t in nominated},
        'logits_error': error_stats(a, b),
        'centered_logits_error': error_stats(a-a.mean(), b-b.mean()),
        'teacher_student_kl': float(np.sum(np.exp(logp)*(logp-logq))),
    }


def read_frame(event, case, directory, name, natural):
    if (event.get('id') != name or event.get('step') != 0 or event.get('vocab') != VOCAB
            or event.get('layers') != 24 or event.get('layer_values') != sum(WIDTHS.values())
            or event.get('committed_tokens') != len(case['tokens'])):
        raise ValueError('same complete first-prediction frame required')
    result = {}
    for kind, count in [('logits', VOCAB), ('layers', 24*sum(WIDTHS.values()))]:
        path = directory/'traces'/f'{name}-step0000.{kind}.bf16'
        if event.get(kind+'_path') != str(path):
            raise ValueError('trace location changed')
        result[kind] = read_bf16(path, count)
    if int(np.argmax(result['logits'])) != event['top1'] or event['top1'] != natural['tokens'][0]:
        raise ValueError('full vocabulary/native/ordinary first token mismatch')
    if natural['prompt_tokens'] != case['tokens']:
        raise ValueError('ordinary prompt differs from traced prefix')
    return result


def main():
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('external volume absent')
    registration = read(REGISTRATION)
    if registration.get('registered_before_run') is not True or registration['case_ids'] != list(IDS):
        raise ValueError('fixed diagnostic registration required')
    _, _, controls, receipt, carrier, specs, _, _ = preflight()
    screen = read(CONVERSION/'screen/result.json')
    audit = read(CONVERSION/'screen/independent-readback.json')
    if (screen.get('completed') is not True or screen.get('screen_passed') is not False
            or screen.get('receipt') != receipt or screen.get('cases') != specs
            or audit.get('audit_passed') is not True):
        raise ValueError('audited rejected candidate required')
    specs = {s['id']: s for s in specs if s['id'] in IDS}
    natural = {'bf16': controls['reports']['bf16']['four'],
               'old_q4': controls['reports']['v2']['four'], 'sensitive_q4': screen['four']}
    natural = {m: {r['id']: r for r in rows if r['id'] in IDS} for m, rows in natural.items()}
    frames, reports = {m: {} for m in natural}, {m: {} for m in natural}
    for item in controls['reports']['v2']['traces']:
        if item['id'] not in IDS:
            continue
        id = item['id']; event = item['events'][0]
        frames['old_q4'][id] = read_frame(event, specs[id]['case'], CONTROLS/'v2',
            'zero-control-'+id, natural['old_q4'][id])
        reports['old_q4'][id] = {'reused': True, 'event': event, 'source': str(CONTROLS/'result.json')}
    prior = ROOT/'column-fit-graph-diagnostic-v1'
    prior_protocol = read(prior/'protocol.json')
    prior_result = read(ROOT/'column-fit-graph-diagnostic-v2/result.json')
    prior_audit = read(ROOT/'column-fit-graph-diagnostic-v2/independent-readback.json')
    completed_bf16 = read(prior/'failure.json')['reports']['bf16']
    if (prior_protocol.get('receipt') != receipt or prior_protocol.get('spec') != specs['mmlu-2804']
            or prior_result.get('completed') is not True or prior_audit.get('audit_passed') is not True):
        raise ValueError('prior BF16 same-worker/prompt completion and audit required')
    ordinary_loaded(completed_bf16['loaded'], 'bf16')
    if any(completed_bf16['after_trace'][k] != natural['bf16']['mmlu-2804'][k]
           for k in ('prompt_tokens', 'tokens', 'text', 'complete')):
        raise ValueError('prior BF16 ordinary trajectory differs')
    frames['bf16']['mmlu-2804'] = read_frame(completed_bf16['events'][0], specs['mmlu-2804']['case'],
        prior/'bf16', 'prefill', natural['bf16']['mmlu-2804'])
    reports['bf16']['mmlu-2804'] = {'reused': True, 'event': completed_bf16['events'][0],
                                  'source': str(prior/'failure.json')}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    OUTPUT.mkdir(exist_ok=False)
    protocol = {**registration, 'binary': str(BINARY), 'receipt': receipt, 'candidate': str(carrier),
                'cases': specs, 'natural': natural, 'layer_widths': WIDTHS}
    save(OUTPUT/'protocol.json', protocol)
    started, worker, lock = time.perf_counter(), None, None
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        for mode, ids in [('bf16', IDS[:1]), ('sensitive_q4', IDS)]:
            directory = OUTPUT/mode; (directory/'traces').mkdir(parents=True)
            precision = 'bf16' if mode == 'bf16' else 'mixed-q4-g16-packed'
            name = 'bf16' if mode == 'bf16' else 'candidate_mixed'
            worker = MixedWorker(BINARY, name, directory, lock, precision=precision,
                calibration_profile=None if mode == 'bf16' else carrier,
                trace_directory=directory/'traces', core=None)
            ordinary_loaded(worker.loaded, precision)
            if mode != 'bf16' and tuple(worker.loaded.get(k) for k in
                    ('packed_version', 'residual_rank', 'active_residual_projections')) != (2, 0, 0):
                raise ValueError('packed-v2 source-only candidate required')
            save(directory/'loaded.json', worker.loaded)
            for id in ids:
                case = specs[id]['case']; name_trace = 'prefill-'+id
                events = trace_request(worker, case, [], name_trace)
                if len(events) != 1:
                    raise ValueError('single prefill frame required')
                frames[mode][id] = read_frame(events[0], case, directory, name_trace, natural[mode][id])
                raw = worker.request(case['tokens'], 'ordinary-'+id, max_tokens=case['max_tokens'],
                    eos_ids=case['eos_ids'], temperature=0, seed=42, reuse=False)
                actual = diagnostic_result(specs[id], raw, tokenizer, VOCAB)
                if raw.get('id') != 'ordinary-'+id or any(actual[k] != natural[mode][id][k]
                        for k in ('prompt_tokens', 'tokens', 'text', 'complete')):
                    raise ValueError('ordinary post-trace generation changed')
                reports[mode][id] = {'reused': False, 'event': events[0], 'after_trace': actual}
            worker.close(); code = worker.process.returncode; worker = None
            if code:
                raise RuntimeError('worker exit failed')
            events = [json.loads(line) for line in (directory/(name+'-events.jsonl')).read_text().splitlines()]
            results = [e for e in events if e.get('event') == 'result']
            if len(results) != 2*len(ids):
                raise ValueError('unexpected number of diagnostic results')
            for id, result in zip(ids, results[::2], strict=True):
                if (result.get('id') != 'prefill-'+id or result.get('forced_tokens') != []
                        or result.get('tokens') != [reports[mode][id]['event']['top1']]
                        or result.get('trace_only') is not True or result.get('complete') is not True
                        or result.get('cached_prompt_tokens') != 0):
                    raise ValueError('trace result consumed unexpected prefix')
        comparisons = {}
        for id in IDS:
            baseline = frames['bf16'][id]
            nominated = sorted({natural[m][id]['tokens'][0] for m in natural})
            pair = [47, 47868] if id == IDS[0] else [33, 39]
            item = {'token_labels': {str(t): tokenizer.decode([t]) for t in nominated},
                    'pair': pair, 'modes': {}}
            for mode in frames:
                actual = frames[mode][id]
                stats = logit_metrics(baseline['logits'], actual['logits'], nominated)
                stats['pair_margin'] = float(actual['logits'][pair[0]]-actual['logits'][pair[1]])
                a = baseline['layers'].reshape(24, -1); b = actual['layers'].reshape(24, -1)
                layers = []
                for layer in range(24):
                    offset, metrics = 0, {}
                    for name, width in WIDTHS.items():
                        metrics[name] = error_stats(a[layer, offset:offset+width], b[layer, offset:offset+width])
                        offset += width
                    metrics['block_sum_fp32'] = error_stats(a[layer,:2048]+a[layer,-2048:],
                                                          b[layer,:2048]+b[layer,-2048:])
                    layers.append({'layer': layer, 'metrics': metrics})
                stats['layers'] = layers; item['modes'][mode] = stats
            comparisons[id] = item
        save(OUTPUT/'result.json', {**protocol, 'completed': True, 'reports': reports,
            'comparisons': comparisons, 'ordinary_generations_exact': True,
            'new_trace_frames': 3, 'reused_trace_frames': 3, 'new_ordinary_generations': 3,
            'elapsed_seconds': time.perf_counter()-started})
        print('Completed two-case three-model fixed-prefix diagnosis', flush=True)
    except BaseException as error:
        save(OUTPUT/'failure.json', {'completed': False, 'error': repr(error), 'reports': reports})
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
