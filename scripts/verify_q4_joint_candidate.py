"""Expand the smallest successful joint diagnostic configuration to all166 cases."""
import argparse
import json
import os
from pathlib import Path
import time

from tokenizers import Tokenizer

from benchmark_native_decode import MODEL, IDENTITY, save
from benchmark_native_mixed import MixedWorker, score
from intervene_q4_layers import request
from intervene_q4_joint import CONFIGS
from luban.process_lock import ModelProcessLock


def select_candidate(result):
    choices=[]
    for mode,groups in result['summary'].items():
        for name,data in groups.items():
            layers=CONFIGS[name]
            if layers and len(layers)<24 and data['passed']==5 and not data['failed']:
                choices.append((len(layers),('q4','rank8','rank32').index(mode),list(CONFIGS).index(name),mode,name))
    if not choices:
        raise ValueError('no partial-Q4 configuration passed the diagnostic cases')
    _,_,_,mode,name=min(choices)
    return mode,name,CONFIGS[name]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','joint','prior','profiles','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.resolve().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    joint=json.loads((args.joint/'result.json').read_text())
    if not joint['completed'] or joint['all24_positive_controls_exact']!=15:
        raise ValueError('joint experiment incomplete or control failed')
    mode,name,layers=select_candidate(joint)
    cases=json.loads((args.prior/'cases.json').read_text())
    old={(r['phase'],r['case_id']):r for r in map(json.loads,(args.prior/'answers.jsonl').read_text().splitlines())
         if r['mode']=='bf16' and 'score' in r}
    if len(old)!=166 or sum(map(len,cases.values()))!=166:
        raise ValueError('expected complete166 baseline')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'traces').mkdir()
    save(args.output/'protocol.json',{'schema':'q4-joint-full166-v1','binary':str(args.binary.resolve()),
         'joint':str(args.joint.resolve()),'prior':str(args.prior.resolve()),'mode':mode,'configuration':name,'layers':layers,
         'selection':'Among partial-Q4 five-case winners: fewest restored layers, then q4/rank8/rank32, then fixed configuration order',
         'seed':42,'temperature':0,'cache_reuse':False,'baseline':'historical same-boundBF16; original scores and tokens retained',
         'quality_rule':'No baseline-correct loss; gains do not cancel losses',
         'limits':'Existing160 regression +6 long probes, known5 diagnostic selection; not independent transfer or deployment qualification',
         'training':False,'teacher_calls':0,'product_adopted':False})
    tokenizer=Tokenizer.from_file(MODEL+'/tokenizer.json');start=time.perf_counter();rows=[]
    lock=ModelProcessLock(IDENTITY,'mps')
    try:
        worker=MixedWorker(args.binary.resolve(),'candidate_mixed',args.output,lock,
            'mixed-q4-g32' if mode=='q4' else 'mixed-q4-g32-residual',
            None if mode=='q4' else (args.profiles/f'{mode}.bin').resolve(),
            (args.output/'traces').resolve(),intervention=True)
        try:
            save(args.output/'loaded.json',worker.loaded)
            with (args.output/'answers.jsonl').open('x') as log:
                for phase,items in cases.items():
                    for index,item in enumerate(items):
                        result=request(worker,{'case':item['case']},f'{phase}-{index}',layers=layers)
                        text=tokenizer.decode(result['tokens'],skip_special_tokens=True)
                        result.update(phase=phase,case_id=item['case']['id'],text=text,score=score(text,item['fixture'],result['complete']))
                        rows.append(result);log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
                        save(args.output/'progress.json',{'cases':len(rows),'elapsed_seconds':time.perf_counter()-start})
        finally:
            worker.close()
        if worker.process.returncode:
            raise RuntimeError('joint candidate worker failed')
    finally:
        lock.close()
    quality={}
    for phase in cases:
        group=[r for r in rows if r['phase']==phase]
        quality[phase]={'cases':len(group),'baseline_passed':sum(old[(phase,r['case_id'])]['score']['passed'] for r in group),
            'passed':sum(r['score']['passed'] for r in group),
            'lost':[r['case_id'] for r in group if old[(phase,r['case_id'])]['score']['passed'] and not r['score']['passed']],
            'gained':[r['case_id'] for r in group if not old[(phase,r['case_id'])]['score']['passed'] and r['score']['passed']]}
    save(args.output/'result.json',{'completed':len(rows)==166,'mode':mode,'configuration':name,'layers':layers,
        'quality':quality,'quality_retained':all(not v['lost'] for v in quality.values()),
        'elapsed_seconds':time.perf_counter()-start,'product_adopted':False})
    print(json.dumps(quality,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
