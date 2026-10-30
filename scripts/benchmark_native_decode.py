#!/usr/bin/env python3
"""Serial resident old/new BF16 worker comparison; no teacher calls or training."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from luban.process_lock import ModelProcessLock
from tokenizers import Tokenizer

IDENTITY = '673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df'
MODEL = '/Volumes/移动数据库/lubanllm/models/luban0'
MANIFEST = 'd8b0ead81edafada4883e6fea37d1750625a682c469b5f70eb760cd2ab0b1237'
CORE = '/Volumes/移动数据库/lubanllm/modules/luban0-v1/core-b45d965728bb.safetensors'
CORE_SHA = 'a2eb54773a4fdc18196acf966c4c21400bbbcd8b811fdc2518b7cbcc40442628'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class Worker:
    def __init__(self, binary, name, out):
        self.name, self.out = name, out
        self.events = (out / (name + '-events.jsonl')).open('w')
        self.stderr = (out / (name + '-stderr.log')).open('w')
        self.cancel_read, self.cancel_write = os.pipe()
        self.messages = queue.Queue()
        self.process = subprocess.Popen([str(binary), MODEL, MANIFEST, str(self.cancel_read), CORE,
            'model.language_model.layers.23.mlp.base.down_proj.weight', CORE_SHA],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True,
            pass_fds=(self.cancel_read,), bufsize=1)
        def read():
            try:
                for line in self.process.stdout:
                    self.events.write(line); self.events.flush()
                    self.messages.put(json.loads(line))
            except BaseException as error:
                self.messages.put(error)
            finally:
                self.messages.put(None)
        self.reader = threading.Thread(target=read, daemon=True); self.reader.start()
        try:
            self.loaded = self.receive(180)
            assert self.loaded['event'] == 'loaded', self.loaded
            assert self.loaded['base_identity'] == IDENTITY
        except BaseException:
            self.close(); raise

    def receive(self, timeout=120):
        event = self.messages.get(timeout=timeout)
        if not isinstance(event, dict):
            raise RuntimeError(f'{self.name}: worker ended or malformed: {event!r}')
        return event

    def request(self, tokens, name, max_tokens=96, temperature=.6, seed=42,
                reuse=False, eos_ids=None, cancel_after=None):
        request = dict(id=name, tokens=tokens, max_tokens=max_tokens, eos_ids=eos_ids or [0],
                       temperature=temperature, seed=seed)
        if self.name != 'baseline': request['cache_reuse'] = reuse
        with (self.out / 'requests.jsonl').open('a') as log:
            log.write(json.dumps(dict(worker=self.name, request=request)) + '\n')
        start = time.monotonic()
        self.process.stdin.write(json.dumps(request) + '\n'); self.process.stdin.flush()
        seen = []
        while True:
            event = self.receive()
            if event['event'] == 'token':
                seen.append(event['token'])
                if cancel_after and len(seen) == cancel_after: os.write(self.cancel_write, b'x')
            elif event['event'] == 'result':
                assert event['tokens'] == seen
                event.update(worker=self.name, wall_seconds=time.monotonic()-start,
                             decode_tps=(len(seen)-1)*1e9/event['decode_ns'] if len(seen)>1 else None)
                print(json.dumps({k:event.get(k) for k in ('worker','id','generated_tokens',
                    'cached_prompt_tokens','decode_tps','prefill_ns','ttft_ns')}),flush=True)
                return event
            else:
                raise RuntimeError(event)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try: self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate(); self.process.wait(timeout=10)
        self.reader.join(timeout=5)
        os.close(self.cancel_read); os.close(self.cancel_write)
        self.events.close(); self.stderr.close()
        assert self.process.returncode == 0, self.process.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    args=parser.parse_args(); out=Path(args.output)
    assert out.is_absolute() and out.parts[1] == 'Volumes' and os.path.ismount(Path(*out.parts[:3]))
    assert out.is_dir() and not (out/'results.json').exists()
    tokenizer=Tokenizer.from_file(MODEL+'/tokenizer.json')
    def prompt(text):
        return tokenizer.encode('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n', add_special_tokens=False).ids
    cases=[('zh', prompt('请用中文解释电脑运行内存与硬盘的区别，给日常使用的例子。')),
           ('en', prompt('Explain how to organize a small software project and check that changes are correct.')),
           ('long', prompt('请概括以下说明，并给出三个具体建议：'+('日常对话要简洁清楚，遇到不确定的问题先核实，学习需要持续练习。'*12)))]
    save(out/'cases.json',[dict(id=name,tokens=tokens,prompt_tokens=len(tokens)) for name,tokens in cases])
    rows=[]; controls={}; outputs={}; loaded={}
    lock=ModelProcessLock(IDENTITY,'mps')
    try:
        for name,binary in [('baseline',args.baseline),('candidate',args.candidate)]:
            worker=Worker(binary,name,out); loaded[name]=worker.loaded
            try:
                worker.request(cases[0][1], 'warmup', max_tokens=8)
                for temperature in (0,.6):
                    for case,tokens in cases:
                        for repeat in range(3):
                            result=worker.request(tokens,f'{case}-t{temperature}-r{repeat}',temperature=temperature)
                            rows.append(result)
                            key=result['id']
                            if name=='baseline': outputs[key]=result['tokens']
                            else: assert outputs[key]==result['tokens'], f'output mismatch {key}'
                if name=='candidate':
                    p=cases[2][1]
                    # Retain exactly the prompt (last emitted token was not forwarded).
                    worker.request(p,'cache-store',max_tokens=1,temperature=0,reuse=True)
                    hit=worker.request(p,'cache-new-settings',max_tokens=1,temperature=.6,seed=43,eos_ids=[1],reuse=True)
                    cold=worker.request(p,'cache-new-settings-fresh',max_tokens=1,temperature=.6,seed=43,eos_ids=[1])
                    assert hit['cached_prompt_tokens']==len(p) and hit['tokens']==cold['tokens']
                    worker.request(p,'cache-store-again',max_tokens=1,temperature=0,reuse=True)
                    extra=tokenizer.encode('请继续回答。',add_special_tokens=False).ids
                    extension=worker.request(p+extra,'cache-extended',temperature=0,reuse=True)
                    extension_cold=worker.request(p+extra,'cache-extended-fresh',temperature=0)
                    controls['cache_extension_tokens_equal']=extension['tokens']==extension_cold['tokens']
                    controls['cache_hit']={k:hit[k] for k in ('cached_prompt_tokens','prefill_ns','ttft_ns','memory')}
                    controls['cache_fresh']={k:cold[k] for k in ('cached_prompt_tokens','prefill_ns','ttft_ns','memory')}
                    controls['extension_hit']={k:extension[k] for k in ('cached_prompt_tokens','prefill_ns','ttft_ns')}
                    controls['extension_fresh']={k:extension_cold[k] for k in ('cached_prompt_tokens','prefill_ns','ttft_ns')}
                    assert controls['cache_extension_tokens_equal'], 'extension numerical comparison failed'
                    worker.request(p,'cancel-store',max_tokens=1,reuse=True)
                    aborted=worker.request(p,'cancel-hit',max_tokens=64,reuse=True,cancel_after=2)
                    after=worker.request(p,'after-cancel',max_tokens=1,temperature=.6,seed=43,eos_ids=[1],reuse=True)
                    assert aborted['finish_reason']=='cancelled' and after['cached_prompt_tokens']==0 and after['tokens']==cold['tokens']
                    controls['cancel_clears_cache']=True
                    unrelated=worker.request(cases[0][1],'unrelated-session',max_tokens=1,reuse=True)
                    assert unrelated['cached_prompt_tokens']==0
                    controls['unrelated_prefix_misses']=True
            finally: worker.close()
            save(out/'results-in-progress.json',dict(rows=rows,controls=controls,loaded=loaded))
    finally: lock.close()
    save(out/'results.json',dict(rows=rows,controls=controls,loaded=loaded,
        token_outputs_equal=True,teacher_calls=0,training=False,metric='continuous decode, first token excluded'))

if __name__=='__main__': main()
