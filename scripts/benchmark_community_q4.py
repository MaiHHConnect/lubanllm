#!/usr/bin/env python3
"""Serial, lease-protected community comparison; never changes product pointers."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import requests
from tokenizers import Tokenizer
from luban.native_engine import NativeEngine
from luban.process_lock import ModelProcessLock
from verify_direct_q4 import MODEL, IDENTITY, MANIFEST, CORE

ROOT = Path('/Volumes/移动数据库/lubanllm/models/community-q4-study-20261017')
APP = Path('/Users/linhao/Applications/Luban.app/Contents/Resources/backend/native/luban-worker')
EOS = [248046, 248044]


def observe(engine, cases, output):
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    rows = []
    result = {'engine': engine.name, 'seed': 42, 'temperature': 0, 'stop_ids': EOS,
              'prompt_cache': False, 'file_cache': 'not purged', 'training_steps': 0,
              'teacher_calls': 0, 'measurement': 'streamed tokens: (N-1)/(last-first), excludes load/prefill; engine timings also retained',
              'rows': rows, 'complete': False}
    try:
        at = time.monotonic()
        result['loaded'] = engine.start()
        result['load_seconds'] = time.monotonic() - at
        engine.generate({**cases[0], 'max_tokens': 8}, lambda token: None)
        for case in cases:
            times = []
            began = time.monotonic()
            actual = engine.generate(case, lambda token: times.append(time.monotonic()))
            tokens = actual['tokens']
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            span = times[-1] - times[0] if len(times) > 1 else 0
            row = {'id': case['id'], 'category': case['category'], 'tokens': tokens, 'text': text,
                   'seconds': time.monotonic() - began, 'ttft_seconds': times[0] - began if times else None,
                   'stream_decode_tps': (len(times)-1)/span if span else None,
                   'finish_reason': actual.get('finish_reason'), 'engine_timing': actual.get('timings')}
            if 'expected' in case:
                row['reference_text_exact'] = text == case['expected'].strip()
            rows.append(row)
            print(json.dumps({'engine': engine.name, 'id': row['id'], 'text': text[:100], 'tps': row['stream_decode_tps']}, ensure_ascii=False), flush=True)
        result['complete'] = True
    finally:
        engine.close()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


class Native:
    name = 'current-native-q4-with-core'

    def start(self):
        proof = json.loads(Path('/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/direct-q4-v1/result.json').read_text())['prepare']['event']
        path = Path('/Volumes/移动数据库/lubanllm/runtime/direct-q4') / (IDENTITY + '-' + proof['quantizer_sha256'] + '.safetensors')
        binding = {'file': str(path), 'sha256': proof['direct_q4_sha256'],
                   'quantizer_sha256': proof['quantizer_sha256'], 'weight_bytes': proof['direct_q4_weight_bytes']}
        self.engine = NativeEngine(APP, MODEL, MANIFEST, core=CORE, precision='mixed-q4-g16', direct_q4=binding)
        return self.engine.start()

    def generate(self, case, emit):
        result = self.engine.generate(case['input_tokens'], case['max_tokens'], EOS, 0, emit, threading.Event(), seed=42, cache_reuse=False)
        result['timings'] = {k: result.get(k) for k in ('prefill_ns', 'ttft_ns', 'decode_ns')}
        return result

    def close(self):
        if hasattr(self, 'engine'):
            self.engine.close()


class Llama:
    name = 'llama-b11713-unsloth-ud-q4-k-xl-original'

    def __init__(self, output, core=False):
        self.output, self.process, self.log = output, None, None
        self.core = core
        if core:
            self.name = 'llama-b11713-unsloth-with-luban-core'

    def start(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        self.http = requests.Session()
        self.http.trust_env = False
        self.log = (self.output / 'llama-server.log').open('w')
        binary = ROOT / 'llama-b11713/llama-b11713/llama-server'
        model = ROOT / ('unsloth-with-luban-core.gguf' if self.core else 'unsloth-ud-q4-k-xl/Qwen3.5-2B-UD-Q4_K_XL.gguf')
        self.process = subprocess.Popen([str(binary), '-m', str(model), '-c', '4096', '-np', '1', '-ngl', '99', '-fa', 'on', '--host', '127.0.0.1', '--port', str(port), '--no-webui', '--reasoning', 'off'], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('llama-server exited; see its log')
            try:
                response = self.http.get(self.url + '/health', timeout=1)
                if response.ok:
                    return {'pid': self.process.pid, 'model': str(model), 'core': CORE if self.core else None}
            except requests.RequestException:
                pass
            time.sleep(.2)
        raise TimeoutError('llama-server startup timed out')

    def generate(self, case, emit):
        tokens, last = [], {}
        body = {'prompt': case['input_tokens'], 'n_predict': case['max_tokens'], 'temperature': 0,
                'seed': 42, 'cache_prompt': False, 'return_tokens': True, 'stream': True,
                'repeat_penalty': 1.0, 'top_k': 0, 'top_p': 1.0}
        with self.http.post(self.url + '/completion', json=body, stream=True, timeout=(5, 120)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b'data: '):
                    continue
                raw = line[6:]
                if raw == b'[DONE]':
                    break
                last = json.loads(raw)
                if 'error' in last:
                    raise RuntimeError(str(last['error']))
                for token in last.get('tokens', []):
                    tokens.append(token)
                    emit(token)
        return {'tokens': tokens, 'finish_reason': 'length' if last.get('stopped_limit') else 'stop', 'timings': last.get('timings')}

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backends', default='native,llama,mlx,mlx-core')
    args = parser.parse_args()
    assert Path('/Volumes/移动数据库').is_mount()
    args.output.mkdir(exist_ok=False)
    cases = json.loads(args.cases.read_text())
    with_lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        for name in args.backends.split(','):
            if name in ('native', 'llama', 'llama-core'):
                observe(Native() if name == 'native' else Llama(args.output, name == 'llama-core'), cases, args.output / (name + '.json'))
            elif name in ('mlx', 'mlx-core'):
                command = [sys.executable, str(Path(__file__).with_name('community_mlx_study.py')),
                           '--cases', str(args.cases), '--output', str(args.output / (name + '.json'))]
                if name == 'mlx-core':
                    command += ['--core', CORE['file']]
                env = {**os.environ, 'PYTHONPATH': str(Path('build/community-mlx-runtime').resolve()) + os.pathsep + str(Path('src').resolve())}
                subprocess.run(command, env=env, check=True)
            else:
                raise ValueError('Unknown backend')
    finally:
        with_lock.close()


if __name__ == '__main__':
    main()
