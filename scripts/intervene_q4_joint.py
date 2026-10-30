"""Bounded joint-layer ablations; all-layer restoration is a positive control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from tokenizers import Tokenizer

from benchmark_native_decode import MODEL, IDENTITY, save
from benchmark_native_mixed import MixedWorker, score
from intervene_q4_layers import request, pair_result
from luban.process_lock import ModelProcessLock

# Fixed before observing this experiment; these are diagnostic contrasts, not a grid search.
CONFIGS = {'none': None, 'observed_pair': [8,22], 'pair_and_ends': [0,8,22,23],
           'first_half': list(range(12)), 'last_half': list(range(12,24)),
           'even_layers': list(range(0,24,2)), 'odd_layers': list(range(1,24,2)),
           'all_layers': list(range(24))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','prior','profiles','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.resolve().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    prior=json.loads((args.prior/'protocol.json').read_text());specs=prior['cases']
    fixtures={x['fixture']['id']:x['fixture'] for pairs in json.loads((Path(prior['prior'])/'cases.json').read_text()).values() for x in pairs}
    decisions={x['id']:x for x in json.loads((Path(__file__).resolve().parents[1]/'research/q4_fixed_prefix_divergence_20261016.json').read_text())['cases']}
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'protocol.json',{'schema':'q4-joint-restoration-v1','configurations':CONFIGS,
        'binary':str(args.binary.resolve()),'prior':str(args.prior.resolve()),'profiles':str(args.profiles.resolve()),
        'selection':'eight fixed diagnostic contrasts; pair uses prior known repairs; not independent validation',
        'seed':42,'temperature':0,'cache_reuse':False,'cases':[s['id'] for s in specs],
        'precision':'Q4/group32 + optional BF16 rank8/32 factors; restored MLP uses original BF16 and removes its residual',
        'positive_control':'all24 must match prior BF16 full tokens/stop and nominated logits/top1',
        'memory':'retained original BF16 checkpoint for research; no deployable memory/speed claim',
        'training':False,'teacher_calls':0,'product_adopted':False})
    start,rows,loaded,positive,time_controls=time.perf_counter(),[],{},0,0
    tokenizer=Tokenizer.from_file(MODEL+'/tokenizer.json')
    lock=ModelProcessLock(IDENTITY,'mps')
    try:
        with (args.output/'observations.jsonl').open('x') as log:
            for mode,precision,profile in [('q4','mixed-q4-g32',None),('rank8','mixed-q4-g32-residual','rank8.bin'),('rank32','mixed-q4-g32-residual','rank32.bin')]:
                out=args.output/mode;(out/'traces').mkdir(parents=True)
                worker=MixedWorker(args.binary.resolve(),'candidate_mixed',out,lock,precision,
                    (args.profiles/profile).resolve() if profile else None,(out/'traces').resolve(),intervention=True)
                loaded[mode]=worker.loaded;save(args.output/'loaded.json',loaded)
                try:
                    for label,layers in CONFIGS.items():
                        for index,spec in enumerate(specs):
                            decision=decisions[spec['id']];ids=[decision['preferred_token']['id'],decision['alternative_token']['id']]
                            probe=pair_result(request(worker,spec,f'p-{label}-c{index}',layers=layers,probe=ids))
                            result=request(worker,spec,f'g-{label}-c{index}',layers=layers)
                            if label in ('none','all_layers'):
                                reference_mode='bf16' if label=='all_layers' else mode
                                old=spec['historical'][reference_mode];expected=decision['modes'][reference_mode]
                                if any(result[k]!=old[k] for k in ('tokens','finish_reason','complete')):
                                    raise ValueError('positive or baseline generation control failed')
                                if any(probe[k]!=expected[k] for k in ('top1','preferred_logit','alternative_logit')):
                                    raise ValueError('positive or baseline logit control failed')
                                positive+=label=='all_layers'
                            text=tokenizer.decode(result['tokens'],skip_special_tokens=True)
                            row={'mode':mode,'configuration':label,'layers':layers,'case_id':spec['id'],'probe':probe,
                                 'text':text,'score':score(text,fixtures[spec['id']],result['complete']),
                                 'tokens':result['tokens'],'finish_reason':result['finish_reason'],'complete':result['complete']}
                            rows.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                        save(args.output/'progress.json',{'mode':mode,'configuration':label,'rows':len(rows),'elapsed_seconds':time.perf_counter()-start})
                    for index,spec in enumerate(specs):
                        old=spec['historical'][mode];result=request(worker,spec,f'after-c{index}')
                        if any(result[k]!=old[k] for k in ('tokens','finish_reason','complete')):
                            raise ValueError('joint intervention leaked into normal generation')
                        time_controls+=1
                finally:
                    worker.close()
                if worker.process.returncode:
                    raise RuntimeError('worker failed')
    finally:
        lock.close()
    summary={mode:{label:{'passed':sum(r['score']['passed'] for r in rows if r['mode']==mode and r['configuration']==label),
        'failed':[r['case_id'] for r in rows if r['mode']==mode and r['configuration']==label and not r['score']['passed']]}
        for label in CONFIGS} for mode in loaded}
    save(args.output/'result.json',{'completed':True,'rows':rows,'summary':summary,'all24_positive_controls_exact':positive,
        'after_controls_exact':time_controls,'elapsed_seconds':time.perf_counter()-start,'product_adopted':False})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
