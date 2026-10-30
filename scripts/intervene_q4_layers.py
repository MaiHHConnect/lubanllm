"""Cold-cache BF16 single-layer restoration on known Q4 diagnostic cases.

All 24 MLP layers are scanned, then selected interventions get free generation.
Selection is explicitly post-hoc diagnosis, never product qualification.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from tokenizers import Tokenizer

from benchmark_native_decode import MODEL, IDENTITY, save
from benchmark_native_mixed import MixedWorker, score
from luban.process_lock import ModelProcessLock


def request(worker, spec, name, *, layer=None, layers=None, projection='all', probe=None):
    case = spec['case']
    data = {'id': name, 'tokens': case['tokens'], 'max_tokens': case['max_tokens'],
            'eos_ids': case['eos_ids'], 'temperature': 0, 'seed': 42, 'cache_reuse': False}
    if layer is not None:
        data.update(restore_layer=layer, restore_projection=projection)
    if layers is not None:
        if layer is not None or not layers or len(set(layers)) != len(layers) or any(not 0 <= x < 24 for x in layers):
            raise ValueError('invalid joint layer restoration')
        data.update(restore_layers=layers, restore_projection=projection)
    if probe is not None:
        data.update(forced_tokens=spec['forced_tokens'], probe_tokens=probe,
                    max_tokens=len(spec['forced_tokens'])+1)
    with (worker.out/'requests.jsonl').open('a') as f:
        f.write(json.dumps(data)+'\n')
    worker.process.stdin.write(json.dumps(data)+'\n'); worker.process.stdin.flush()
    traces, tokens = [], []
    while True:
        event = worker.receive()
        if event['event'] == 'trace':
            traces.append(event)
        elif event['event'] == 'token':
            tokens.append(event['token'])
        elif event['event'] == 'result':
            if probe is None:
                if event['tokens'] != tokens:
                    raise ValueError('stream/result mismatch')
                return event
            if [r['step'] for r in traces] != list(range(len(spec['forced_tokens'])+1)):
                raise ValueError('missing or duplicate forced-prefix steps')
            for i, trace in enumerate(traces):
                if trace['committed_tokens'] != len(case['tokens'])+i or trace['probe_tokens'] != probe:
                    raise ValueError('incorrect prefix state or nominated tokens')
            return {'traces': traces, 'result': event}
        else:
            raise RuntimeError(event)


def pair_result(probe):
    last = probe['traces'][-1]
    if len(last['probe_logits']) != 2:
        raise ValueError('expected two nominated logits')
    a, b = last['probe_logits']
    return {'top1': last['top1'], 'preferred_logit': a, 'alternative_logit': b,
            'margin': a-b, 'preferred_is_top1': last['top1'] == last['probe_tokens'][0]}


def select_layers(rows):
    """Predeclared diagnostic selection: best pair margin per case plus endpoints."""
    selected = {0, 23}
    for case in sorted({r['case_id'] for r in rows}):
        group = [r for r in rows if r['case_id'] == case]
        # Full-vocab top1 takes precedence; then nominated pair margin; lower layer wins ties.
        best = max(group, key=lambda r: (r['probe']['preferred_is_top1'], r['probe']['margin'], -r['layer']))
        selected.add(best['layer'])
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'prior', 'profiles', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.resolve().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    previous = json.loads((args.prior/'protocol.json').read_text())
    specs = previous['cases']
    evidence = json.loads((Path(__file__).resolve().parents[1]/'research/q4_fixed_prefix_divergence_20261016.json').read_text())
    decisions = {x['id']: x for x in evidence['cases']}
    fixtures = {x['fixture']['id']: x['fixture'] for pairs in json.loads(
        (Path(previous['prior'])/'cases.json').read_text()).values() for x in pairs}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'protocol.json', {'schema': 'q4-single-layer-intervention-v1',
        'binary': str(args.binary.resolve()), 'prior': str(args.prior.resolve()), 'profiles': str(args.profiles.resolve()),
        'modes': ['q4','rank8','rank32'], 'layers': list(range(24)), 'projection': 'all', 'cases': [s['id'] for s in specs],
        'forced_prefixes': {s['id']: s['forced_tokens'] for s in specs}, 'seed': 42, 'temperature': 0,
        'cache': 'cold per request; restore bound original BF16 Linear before entire prompt/forced decode; restore pointers on return/error',
        'numerical_control': 'no intervention equals prior logits/top1; layer23/down is no-op; original normal generation before/after',
        'generation_selection': 'per mode union of layers0/23 and best full-vocab target top1/pair margin per case, low-layer tie',
        'decomposition_selection': 'per mode best layer per case; measure gate/up/down separately for that case only',
        'limits': 'post-hoc known-case diagnosis; restored layer removes Q4 and residual together; no quality qualification',
        'memory': 'one original checkpoint retained in research process; resident payload and physical footprint separately logged',
        'training': False, 'teacher_calls': 0, 'product_adopted': False})
    tokenizer = Tokenizer.from_file(MODEL+'/tokenizer.json')
    start, sweep, normal, decomposed, controls, loaded = time.perf_counter(), [], [], [], [], {}
    lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        with (args.output/'observations.jsonl').open('x') as log:
            def record(row):
                log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush()
            for mode, precision, profile in [('q4','mixed-q4-g32',None),
                    ('rank8','mixed-q4-g32-residual','rank8.bin'), ('rank32','mixed-q4-g32-residual','rank32.bin')]:
                out=args.output/mode; (out/'traces').mkdir(parents=True)
                worker=MixedWorker(args.binary.resolve(),'candidate_mixed',out,lock,precision,
                    (args.profiles/profile).resolve() if profile else None,(out/'traces').resolve(),intervention=True)
                loaded[mode]=worker.loaded; save(args.output/'loaded.json',loaded)
                try:
                    if not worker.loaded.get('research_intervention') or worker.loaded.get('retained_original_weight_bytes',0)<=0:
                        raise ValueError('original bound weights not retained')
                    for index,spec in enumerate(specs):
                        old=spec['historical'][mode]
                        result=request(worker,spec,f'before-c{index}')
                        if any(result[k]!=old[k] for k in ('tokens','finish_reason','complete')):
                            raise ValueError('normal baseline changed')
                        decision=decisions[spec['id']]; ids=[decision['preferred_token']['id'],decision['alternative_token']['id']]
                        baseline=pair_result(request(worker,spec,f'base-c{index}',probe=ids))
                        expected=decision['modes'][mode]
                        if any(baseline[k]!=expected[k] for k in ('top1','preferred_logit','alternative_logit')):
                            raise ValueError('probe baseline changed')
                        noop=pair_result(request(worker,spec,f'noop-c{index}',layer=23,projection='down',probe=ids))
                        if noop!=baseline:
                            raise ValueError('layer23/down is not a no-op')
                        row={'phase':'control','mode':mode,'case_id':spec['id'],'baseline':baseline,'noop':noop,'normal_exact':True}
                        controls.append(row);record(row)
                    group=[]
                    for layer in range(24):
                        for index,spec in enumerate(specs):
                            decision=decisions[spec['id']];ids=[decision['preferred_token']['id'],decision['alternative_token']['id']]
                            result=request(worker,spec,f'all-l{layer}-c{index}',layer=layer,probe=ids)
                            row={'phase':'sweep','mode':mode,'layer':layer,'projection':'all','case_id':spec['id'],'probe':pair_result(result)}
                            group.append(row);sweep.append(row);record(row)
                        save(args.output/'progress.json',{'mode':mode,'layer':layer,'sweep_rows':len(sweep),'elapsed_seconds':time.perf_counter()-start})
                    selected=select_layers(group)
                    for layer in selected:
                        for index,spec in enumerate(specs):
                            result=request(worker,spec,f'gen-l{layer}-c{index}',layer=layer)
                            text=tokenizer.decode(result['tokens'],skip_special_tokens=True)
                            row={'phase':'generation','mode':mode,'layer':layer,'case_id':spec['id'],
                                 'text':text,'score':score(text,fixtures[spec['id']],result['complete']),
                                 'tokens':result['tokens'],'finish_reason':result['finish_reason'],'complete':result['complete']}
                            normal.append(row);record(row)
                    for index,spec in enumerate(specs):
                        best=max((r for r in group if r['case_id']==spec['id']),
                                 key=lambda r:(r['probe']['preferred_is_top1'],r['probe']['margin'],-r['layer']))
                        decision=decisions[spec['id']];ids=[decision['preferred_token']['id'],decision['alternative_token']['id']]
                        for projection in ('gate','up','down'):
                            probe=pair_result(request(worker,spec,f'{projection}-l{best["layer"]}-c{index}',
                                                      layer=best['layer'],projection=projection,probe=ids))
                            row={'phase':'decomposition','mode':mode,'layer':best['layer'],'case_id':spec['id'],
                                 'projection':projection,'probe':probe}
                            decomposed.append(row);record(row)
                    for index,spec in enumerate(specs):
                        result=request(worker,spec,f'after-c{index}');old=spec['historical'][mode]
                        if any(result[k]!=old[k] for k in ('tokens','finish_reason','complete')):
                            raise ValueError('intervention leaked into later generation')
                    print(json.dumps({'mode':mode,'selected_layers':selected,'seconds':time.perf_counter()-start}),flush=True)
                finally:
                    worker.close()
                if worker.process.returncode:
                    raise RuntimeError('intervention worker failed')
    finally:
        lock.close()
    save(args.output/'result.json',{'completed':True,'sweep':sweep,'generation':normal,'decomposition':decomposed,
         'controls':controls,'normal_controls_exact':30,'noop_controls_exact':15,'baseline_probes_exact':15,
         'elapsed_seconds':time.perf_counter()-start,'product_adopted':False})


if __name__=='__main__':
    main()
