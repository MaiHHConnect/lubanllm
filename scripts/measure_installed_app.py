#!/usr/bin/env python3
"""Finite installed-app latency/footprint study; no training or teacher calls."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import plistlib
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'native/luban'))
from footprint import footprint


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Application:
    def __init__(self):
        launch = json.loads((Path.home() / 'Library/Application Support/Luban/launch.json').read_text())
        self.base = 'http://127.0.0.1:' + str(launch['port'])
        self.pid = launch['pid']
        self.headers = {'Cookie': 'luban_session=' + launch['token'],
                        'Content-Type': 'application/json', 'Origin': self.base}

    def request(self, path, body=None, method=None):
        request = urllib.request.Request(self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=self.headers, method=method)
        return urllib.request.urlopen(request, timeout=240)

    def json(self, path, body=None, method=None):
        with self.request(path, body, method) as response:
            return json.load(response)


class Sampler:
    def __init__(self, root, path):
        self.root, self.path = root, path
        self.phase = 'idle'
        self.stop = threading.Event()
        self.rows, self.errors = [], []
        self.lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
        self.children = self.lib.proc_listchildpids
        self.children.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.children.restype = ctypes.c_int
        self.thread = threading.Thread(target=self.run, daemon=True)

    def pids(self, root):
        found, pending = set(), [root]
        while pending:
            pid = pending.pop()
            if pid in found:
                continue
            found.add(pid)
            buffer = (ctypes.c_int * 256)()
            count = self.children(pid, buffer, ctypes.sizeof(buffer))
            if count < 0 or count >= 256:
                raise RuntimeError('Process-tree enumeration failed or overflowed')
            pending.extend(value for value in list(buffer)[:count] if value > 0)
        return sorted(found)

    def sample(self):
        started = time.monotonic()
        processes = []
        for pid in self.pids(self.root):
            try:
                processes.append({'pid': pid, **footprint(pid)})
            except OSError:
                self.errors.append({'kind': 'process_exited_during_sample', 'pid': pid})
        assert any(row['pid'] == self.root for row in processes), 'Application root disappeared'
        return {'monotonic': started, 'phase': self.phase, 'processes': processes,
                'simultaneous_sum_bytes': sum(row['physical_footprint_bytes'] for row in processes),
                'sample_seconds': time.monotonic() - started}

    def run(self):
        with self.path.open('x') as output:
            while not self.stop.is_set():
                try:
                    row = self.sample()
                    self.rows.append(row)
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                except Exception as error:
                    self.errors.append({'kind': type(error).__name__})
                    self.stop.set()
                    return
                self.stop.wait(.05)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), 'Sampler failed to stop'


def describe(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    volume = Path('/Volumes/移动数据库')
    assert volume.is_mount() and directory.is_relative_to(volume / 'lubanllm/experiments')
    assert not directory.exists()
    directory.mkdir(mode=0o700)
    app = Application()
    initial = app.json('/api/status')
    settings = app.json('/api/settings')
    assert initial['runtime']['state'] == 'ready'
    assert not initial['runtime'].get('job') or initial['runtime']['job'].get('state') not in ('queued', 'running')
    assert settings['teacher']['key_configured'] is False, 'Study requires an unconfigured teacher'
    assert settings['teacher']['answer_mode'] == 'local_first'
    info = plistlib.loads(Path('/Applications/Luban.app/Contents/Info.plist').read_bytes())
    assert info['CFBundleShortVersionString'] == '0.3.223'
    inference = initial['runtime']['inference']
    assert inference['binary_sha256'] == '7f5054c11b496137dd303603397ae5e7d01404acdf66e0d6869b14b6066006d5'
    # Reuse the platform's existing process topology; verify the service's parent is
    # the native application before including its complete descendant tree.
    root = int(subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(app.pid)], text=True).strip())
    assert Path(subprocess.check_output(['ps', '-o', 'comm=', '-p', str(root)], text=True).strip()) == Path('/Applications/Luban.app/Contents/MacOS/Luban')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(settings['model_path']) / 'tokenizer.json'))
    prefix = '以下背景仅供本次性能测量，不是需要执行的指令。\n'
    paragraph = '电脑里的文件可以按项目和日期分类，重要资料应妥善保存，完成工作后及时整理目录。\n'
    suffix = '\n请分条详细讲解如何整理电脑中的文件，给出至少二十条具体建议，每条附上解释和例子。请直接开始回答。'
    cases = []
    for target in (128, 1024, 4096):
        candidates = [(abs(len(tokenizer.encode(prefix + paragraph * count + suffix).ids) - target), count)
                      for count in range(200)]
        _, count = min(candidates)
        text = prefix + paragraph * count + suffix
        cases.append({'id': 'input-' + str(target), 'nominal_input_tokens': target,
                      'text': text, 'raw_text_tokens': len(tokenizer.encode(text).ids)})
    save(directory / 'cases.json', cases)
    protocol = {'schema': 'installed-app-resources-v1', 'date': '2026-10-24',
        'input_targets': [128, 1024, 4096], 'output_budgets': [32, 128, 256], 'passes': 3,
        'warmups': 1, 'temperature': 0, 'thinking': 'off', 'teacher_calls_allowed': 0,
        'training': False, 'adoption': False,
        'cache': 'Installed API policy unchanged: single-prefix reuse enabled; record actual cached_prompt_tokens. OS/file/Metal caches not purged; app already resident.',
        'timing': 'HTTP submission to complete SSE drain; first nonempty content delta separately; server prefill/TTFT/decode also recorded. Excludes startup, token planning, status polls and research preparation.',
        'memory': '50ms target sampling: simultaneous sum of Darwin physical footprint over native app and all descendants, then max. Sampled accounting sum, not exact system-wide unique physical bytes. No sum of per-process lifetime peaks. Sampler itself excluded.',
        'gpu_memory': 'Unified-memory accounting may overlap process footprint. Worker exposes no standalone allocator peak; do not invent or add it.',
        'limitations': ['Single device and current installed source; not a matched historical speedup comparison.',
                       'Three repetitions are timing observations, not independent learning seeds.',
                       'Output budgets may end with length; these requests do not assess answer quality.',
                       'Automatic learning temporarily disabled and restored; no weight or accepted-pointer changes intended.']}
    save(directory / 'protocol.json', protocol)
    sampler = Sampler(root, directory / 'memory.jsonl')
    original = {'automatic': settings['automatic'],
                'teacher': {'teaching_modalities': settings['teacher']['teaching_modalities']}}
    result = {'schema': protocol['schema'], 'complete': False, 'rows': [], 'settings_restored': False,
        'source': {'version': info['CFBundleShortVersionString'], 'build': info['CFBundleVersion'],
                   'inference': inference, 'core': initial['runtime']['core_version'],
                   'expert': initial['runtime']['active_expert']},
        'hardware': {'model': subprocess.check_output(['sysctl', '-n', 'hw.model'], text=True).strip(),
                     'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                     'ram_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                     'macos': subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip()},
        'protocol': protocol, 'teacher_calls_before': initial['teacher_calls']}
    started = time.monotonic()
    sampler.thread.start()
    try:
        app.json('/api/settings', {'automatic': {**settings['automatic'], 'learning_enabled': False},
                                  'teacher': {'teaching_modalities': []}}, 'PATCH')
        def generate(case, budget, identifier, warmup=False):
            assert not sampler.errors, 'Memory sampling failed'
            sampler.phase = identifier
            at = time.monotonic()
            frames, first_delta = [], None
            with app.request('/api/chat', {'messages': [{'role': 'user', 'content': case['text']}],
                    'route': 'local', 'thinking': 'off', 'temperature': 0, 'max_tokens': budget}) as response:
                for line in response:
                    if line.startswith(b'data: '):
                        event = json.loads(line[6:])
                        if event['type'] == 'delta' and event.get('content') and first_delta is None:
                            first_delta = time.monotonic() - at
                        frames.append(event)
            elapsed = time.monotonic() - at
            assert not any(frame['type'] == 'error' for frame in frames), 'Application returned an error'
            done = next(frame for frame in frames if frame['type'] == 'done')
            meta = next(frame for frame in frames if frame['type'] == 'meta')
            assert meta['route'] == 'local'
            usage = done['usage']
            assert usage['reference_used'] is False and usage['completion_tokens'] > 1
            assert usage['prompt_tokens'] < initial['runtime']['context']['native_limit']
            assert usage['decode_tokens'] == usage['completion_tokens'] - 1
            assert abs(usage['decode_tokens_per_second'] * usage['decode_seconds'] - usage['decode_tokens']) < 1e-6
            after = app.json('/api/status')
            assert after['teacher_calls'] == initial['teacher_calls']
            row = {'id': identifier, 'warmup': warmup, 'nominal_input_tokens': case['nominal_input_tokens'],
                'output_budget': budget, 'http_seconds': elapsed, 'client_first_content_seconds': first_delta,
                'http_tokens_per_second': usage['completion_tokens'] / elapsed, 'usage': usage,
                'meta': meta, 'answer': ''.join(frame.get('content', '') for frame in frames if frame['type'] == 'delta'),
                'native_process': after['runtime']['native_process']}
            result['rows'].append(row)
            save(directory / 'result.json', result)
            print(json.dumps({key: row[key] for key in ('id', 'http_seconds', 'http_tokens_per_second')}), flush=True)
        generate(cases[0], 32, 'warmup', True)
        cells = [(case, budget) for case in cases for budget in (32, 128, 256)]
        for repeat in range(3):
            ordered = cells if repeat == 0 else list(reversed(cells)) if repeat == 1 else cells[3:] + cells[:3]
            for case, budget in ordered:
                generate(case, budget, f'r{repeat+1}-{case["id"]}-out-{budget}')
        sampler.phase = 'final'
        final = app.json('/api/status')
        assert final['runtime']['core_version'] == result['source']['core']
        assert final['runtime']['active_expert'] == result['source']['expert']
        result.update(teacher_calls_after=final['teacher_calls'], teacher_calls_added=0,
                      optimizer_updates=0, adoptions=0, source_ids_unchanged=True, complete=True)
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        try:
            app.json('/api/settings', original, 'PATCH')
            restored = app.json('/api/settings')
            assert restored['automatic'] == settings['automatic']
            assert restored['teacher']['teaching_modalities'] == settings['teacher']['teaching_modalities']
            result['settings_restored'] = True
        finally:
            sampler.close()
            result['seconds'] = time.monotonic() - started
            result['memory'] = {'sample_count': len(sampler.rows), 'errors': sampler.errors,
                'sampled_max_simultaneous_sum_bytes': max((row['simultaneous_sum_bytes'] for row in sampler.rows), default=None),
                'median_sample_seconds': statistics.median(row['sample_seconds'] for row in sampler.rows) if sampler.rows else None,
                'app_root_pid': root, 'pids_seen': sorted({p['pid'] for row in sampler.rows for p in row['processes']})}
            if sampler.errors:
                result['complete'] = False
            measured = [row for row in result['rows'] if not row['warmup']]
            summary = []
            for case in cases:
                for budget in (32, 128, 256):
                    rows = [row for row in measured if row['nominal_input_tokens'] == case['nominal_input_tokens'] and row['output_budget'] == budget]
                    if not rows:
                        continue
                    summary.append({'nominal_input_tokens': case['nominal_input_tokens'], 'output_budget': budget,
                        'n': len(rows), 'actual_input_tokens': sorted({row['usage']['prompt_tokens'] for row in rows}),
                        'actual_output_tokens': [row['usage']['completion_tokens'] for row in rows],
                        'cached_prompt_tokens': [row['usage'].get('cached_prompt_tokens') for row in rows],
                        'finish_reasons': [row['usage']['finish_reason'] for row in rows],
                        'decode_tokens_per_second': describe([row['usage']['decode_tokens_per_second'] for row in rows]),
                        'http_seconds': describe([row['http_seconds'] for row in rows]),
                        'client_first_content_seconds': describe([row['client_first_content_seconds'] for row in rows]),
                        'weighted_decode_tokens_per_second': sum(row['usage']['decode_tokens'] for row in rows) / sum(row['usage']['decode_seconds'] for row in rows)})
            result['summary'] = summary
            save(directory / 'result.json', result)
    assert result['complete'] and result['settings_restored'] and len(result['rows']) == 28


if __name__ == '__main__':
    main()
