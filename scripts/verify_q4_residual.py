"""Native BF16 / Q4 / rank8 / rank32 paired closed-book generation protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import time

from tokenizers import Tokenizer

from benchmark_native_decode import MODEL, CORE, CORE_SHA, MANIFEST, IDENTITY, save
from benchmark_native_mixed import MixedWorker, FIXTURES, load_cases, score
from luban.process_lock import ModelProcessLock


def summarize(rows):
    quality = {}
    baseline = {(r['phase'], r['case_id']): r for r in rows if r['mode'] == 'bf16' and 'score' in r}
    for mode in ('q4', 'rank8', 'rank32'):
        quality[mode] = {}
        for phase in ('ranking', 'preservation', 'long'):
            paired = [r for r in rows if r['mode'] == mode and r['phase'] == phase]
            expected = {key for key in baseline if key[0] == phase}
            if { (r['phase'], r['case_id']) for r in paired } != expected or len(paired) != len(expected):
                raise ValueError(f'missing or duplicate quality outputs: {mode} {phase}')
            lost, gained, changed = [], [], []
            for row in paired:
                old = baseline[(phase, row['case_id'])]
                if old['score']['passed'] and not row['score']['passed']:
                    lost.append(row['case_id'])
                if not old['score']['passed'] and row['score']['passed']:
                    gained.append(row['case_id'])
                if old['tokens'] != row['tokens']:
                    changed.append(row['case_id'])
            quality[mode][phase] = {'cases': len(paired),
                'baseline_passed': sum(baseline[key]['score']['passed'] for key in expected),
                'passed': sum(r['score']['passed'] for r in paired),
                'lost_baseline_correct': lost, 'gained_correct': gained, 'changed_tokens': changed}
    speed = {}
    for mode in ('bf16', 'q4', 'rank8', 'rank32'):
        group = [r for r in rows if r['mode'] == mode and r['phase'] == 'speed']
        if len(group) != 9 or any(r['generated_tokens'] != 96 or r['decode_ns'] <= 0 for r in group):
            raise ValueError(f'incomplete speed protocol: {mode}')
        speed[mode] = {'decode_tps': 95 * len(group) * 1e9 / sum(r['decode_ns'] for r in group),
                       'median_ttft_seconds': statistics.median(r['ttft_ns']/1e9 for r in group),
                       'median_wall_seconds': statistics.median(r['wall_seconds'] for r in group)}
    return {'quality': quality, 'speed': speed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.absolute().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted research volume required')
    for name in ('rank8.bin', 'rank32.bin', 'rank8-zero.bin'):
        if not (args.profiles / name).is_file():
            raise ValueError('missing residual profile ' + name)
    suites = load_cases(FIXTURES)
    tokenizer = Tokenizer.from_file(MODEL + '/tokenizer.json')

    def prompt(text):
        return tokenizer.encode('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + text +
                                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n', add_special_tokens=False).ids

    extra = []
    for repeats in (16, 48, 80):
        for suffix, text, expected in [
            ('copy', '开头登记的档案编号是 VX-4827。\n', 'VX-4827'),
            ('upper', '开头登记的字符串是 Cedar_Path。\n', 'CEDAR_PATH')]:
            question = '请只输出开头登记的档案编号。' if suffix == 'copy' else '请将开头登记的字符串的英文字母转为大写，只输出转换结果。'
            body = text + '以下是无关记录：仓库检查完成，货物摆放整齐，今天没有新的登记。\n' * repeats + question
            tokens = prompt(body)
            if len(tokens) + 128 > 4096:
                raise ValueError('long case exceeds reserved context')
            extra.append(({'id': f'long-{suffix}-{repeats}', 'tokens': tokens, 'max_tokens':128, 'eos_ids':[248044]},
                          {'id': f'long-{suffix}-{repeats}', 'check': {'type':'exact','expected':expected}, 'prompt':body}))
    suites['long'] = extra
    speed_prompts = [('zh', prompt('请用中文解释电脑运行内存与硬盘的区别，给日常使用的例子。')),
                     ('en', prompt('Explain how to organize a small software project and check that changes are correct.')),
                     ('long', prompt('请概括以下说明，并给出三个具体建议：' + '日常对话要简洁清楚，遇到不确定的问题先核实，学习需要持续练习。' * 12))]
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'protocol.json', {'schema':'native-q4-residual-generation-v1', 'created':datetime.now(timezone.utc).isoformat(),
        'binary':str(args.binary.resolve()), 'profiles':str(args.profiles.resolve()), 'model':MODEL, 'core':CORE, 'core_sha':CORE_SHA,
        'manifest':MANIFEST, 'seed':42, 'temperature':0, 'modes':['bf16','q4','zero','rank8','rank32'],
        'selection':'fixed rank8 and rank32 from five prior calibration prompts; no fitting to generation cases',
        'quality':'48 existing ranking + 112 prior training/replay + 6 fresh long probes; natural EOS and original budgets',
        'zero_control':'16 smoke plus two known uppercase regressions and six long probes, exact tokens/stop vs plain Q4',
        'speed':'three prompts x three repeats, 96 tokens EOS=0, greedy, no cache; 8-token warmup excluded',
        'hardware':platform.platform(), 'memory':'worker Darwin physical footprint, lifetime load peak separately',
        'limits':'B45 core only, no identity expert/router; old preservation probes are not held-out transfer; no adoption',
        'teacher_calls':0,'training':False,'product_changed':False})
    save(args.output / 'cases.json', {suite:[{'case':case,'fixture':fix} for case,fix in pairs] for suite,pairs in suites.items()})
    rows, loaded = [], {}
    modes = [('bf16','bf16',None), ('q4','mixed-q4-g32',None), ('zero','mixed-q4-g32-residual','rank8-zero.bin'),
             ('rank8','mixed-q4-g32-residual','rank8.bin'), ('rank32','mixed-q4-g32-residual','rank32.bin')]
    smoke_ids = {c['id'] for pairs in load_cases(FIXTURES,8).values() for c,_ in pairs}
    smoke_ids.update(['coverage-new-train-1','preservation-replay-train-olive-path'])
    lock = ModelProcessLock(IDENTITY,'mps')
    start = time.perf_counter()
    try:
        with (args.output/'answers.jsonl').open('x') as answers:
            for mode, precision, profile in modes:
                out = args.output / mode
                out.mkdir()
                worker = MixedWorker(args.binary.resolve(), 'baseline_bf16' if mode=='bf16' else 'candidate_mixed', out, lock,
                                     precision, (args.profiles/profile).resolve() if profile else None)
                loaded[mode] = worker.loaded
                save(args.output/'loaded.json', loaded)
                try:
                    expected_rank = 8 if mode in ('zero','rank8') else 32 if mode=='rank32' else 0
                    if worker.loaded.get('residual_rank') != expected_rank:
                        raise ValueError('worker loaded the wrong correction rank')
                    if worker.loaded.get('quantized_projections') != (0 if mode=='bf16' else 71):
                        raise ValueError('unexpected projection quantization count')
                    worker.request(speed_prompts[0][1], 'warmup', max_tokens=8,temperature=0,reuse=False)

                    def record(case, phase, fixture=None):
                        result = worker.request(case['tokens'], phase+'-'+case['id'], max_tokens=case['max_tokens'],temperature=0,seed=42,
                                                reuse=False,eos_ids=case['eos_ids'])
                        if result.get('cached_prompt_tokens') != 0:
                            raise ValueError('unexpected cache reuse')
                        result.update(mode=mode,phase=phase,case_id=case['id'],text=tokenizer.decode(result['tokens'],skip_special_tokens=True))
                        if fixture is not None:
                            result['score'] = score(result['text'],fixture,result['complete'])
                        answers.write(json.dumps(result,ensure_ascii=False)+'\n');answers.flush()
                        rows.append(result)
                        if mode=='zero':
                            old=next(r for r in rows if r['mode']=='q4' and r['phase']==phase and r['case_id']==case['id'])
                            if any(result[key]!=old[key] for key in ('tokens','finish_reason','complete')):
                                raise ValueError('zero correction changed Q4 output: '+case['id'])

                    for phase,pairs in suites.items():
                        for case,fixture in pairs:
                            if mode!='zero' or phase=='long' or case['id'] in smoke_ids:
                                record(case,phase,fixture)
                    if mode!='zero':
                        for name,tokens in speed_prompts:
                            for repeat in range(3):
                                record({'id':f'{name}-{repeat}','tokens':tokens,'max_tokens':96,'eos_ids':[0]},'speed')
                finally:
                    worker.close()
                if worker.process.returncode:
                    raise RuntimeError(f'{mode} failed: {worker.process.returncode}')
                save(args.output/'progress.json',{'finished_mode':mode,'answers':len(rows),'elapsed_seconds':time.perf_counter()-start})
    finally:
        lock.close()
    result=summarize(rows)
    result.update(completed=True,zero_control_exact=True,zero_control_cases=sum(r['mode']=='zero' for r in rows),
                  loaded=loaded,elapsed_seconds=time.perf_counter()-start,product_adopted=False)
    result['quality_retained']={mode:all(not suite['lost_baseline_correct'] for suite in data.values()) for mode,data in result['quality'].items()}
    save(args.output/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
