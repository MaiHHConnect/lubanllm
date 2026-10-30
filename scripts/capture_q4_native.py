"""Capture real native BF16 inputs on frozen continuations; no quality-based selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import re
import resource
import struct
import subprocess
import threading
import time

import numpy as np

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from capture_q4_decode import LIMIT, SEED, STREAMS, replay_record, validate_source
from luban.process_lock import ModelProcessLock
from prepare_q4_packed import materials

MAGIC = b'LNAC001\n'
ROW_VALUES = 190464
MAX_ROWS = 288
HEADER_BYTES = 160
ORDER = [(f'layer{layer:02d}-'+kind, width) for layer in range(24)
         for kind, width in [('gateup', 2048), ('down', 6144)]
         if layer != 23 or kind != 'down']


def integers(values, minimum, maximum):
    return (isinstance(values, list) and all(type(v) is int and minimum <= v < maximum for v in values))


def capture_positions(record):
    p = len(record['prompt_tokens'])
    prompt = sorted(set(np.linspace(0, p-1, min(128, p), dtype=int).tolist()
                        + list(range(max(0, p-16), p))))
    positions = prompt + record['assistant_positions']
    if (not 1 <= len(positions) <= MAX_ROWS or positions != sorted(set(positions))
            or not integers(positions, 0, len(record['replay_tokens']))):
        raise ValueError('invalid frozen capture positions')
    return positions


def load_frozen(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    generated = json.loads((directory/'generation-result.json').read_text())
    source = protocol['source_protocol']
    validate_source(source)
    frozen_materials = [{k: r[k] for k in ('id', 'split', 'text')} for r in source['materials']]
    if (protocol.get('schema') != 'q4-decode-capture-v1' or protocol.get('seed') != SEED
            or protocol.get('temperature') != 0 or protocol.get('cache_reuse') is not False
            or protocol.get('max_new_tokens') != LIMIT or frozen_materials != materials()
            or generated.get('completed') is not True or len(generated['records']) != 20
            or len(protocol['cases']) != 20):
        raise ValueError('completed frozen original twenty-record generation required')
    eos, vocab = protocol['eos_ids'], protocol['vocab_size']
    if type(vocab) is not int or vocab <= 0 or not eos or not integers(eos, 0, vocab):
        raise ValueError('invalid frozen vocabulary or EOS IDs')
    config = json.loads((Path(MODEL)/'config.json').read_text())
    config = config.get('text_config', config)
    actual_eos = config['eos_token_id']
    if vocab != config['vocab_size'] or eos != (actual_eos if isinstance(actual_eos, list) else [actual_eos]):
        raise ValueError('frozen vocabulary disagrees with model configuration')
    records = []
    prompt_counts = {'calibration': 0, 'observation': 0}
    for material, case, record in zip(source['materials'], protocol['cases'], generated['records'], strict=True):
        if (case['id'], case['split']) != (material['id'], material['split']):
            raise ValueError('frozen material split/order changed')
        if (not re.fullmatch(r'[a-z0-9-]+', case['id']) or not case['prompt_tokens']
                or not integers(case['prompt_tokens'], 0, vocab)
                or len(case['prompt_tokens']) != material['tokens']):
            raise ValueError('invalid frozen prompt tokens')
        expected = replay_record(case, record['native_result'], eos, vocab)
        if expected != record or len(record['replay_tokens']) > 2048:
            raise ValueError('frozen replay record changed or context overflow')
        positions = capture_positions(record)
        prompt_counts[case['split']] += sum(p < len(case['prompt_tokens']) for p in positions)
        records.append(record)
    if any(counts != prompt_counts for counts in source['counts'].values()):
        raise ValueError('original prompt position totals changed')
    return protocol, records


def make_request(record, eos):
    return dict(id=record['id'], tokens=record['prompt_tokens'],
                max_tokens=len(record['generated_tokens']), eos_ids=eos,
                temperature=0, seed=SEED, cache_reuse=False,
                forced_tokens=record['generated_tokens'][:-1], capture_positions=capture_positions(record))


def check_loaded(event):
    expected = {'event': 'loaded', 'base_identity': IDENTITY, 'manifest_sha256': MANIFEST,
                'core_loaded': False, 'core_file_sha256': None, 'quantized_projections': 0,
                'inference_precision': 'bf16', 'research_capture_inputs': True}
    if any(event.get(k) != v or (isinstance(v, bool) and event.get(k) is not v) for k, v in expected.items()):
        raise ValueError('worker must load original BF16 core with native capture enabled')


def check_result(result, request, expected_tokens):
    expected = {'event': 'result', 'id': request['id'], 'tokens': expected_tokens,
                'prompt_tokens': len(request['tokens']), 'generated_tokens': len(expected_tokens),
                'finish_reason': 'capture', 'complete': True, 'capture_only': True, 'cached_prompt_tokens': 0}
    if (not integers(result.get('tokens'), 0, 2**32)
            or any(result.get(k) != v or (isinstance(v, bool) and result.get(k) is not v)
                   for k, v in expected.items())):
        raise ValueError('native forced-prefix greedy output differs from frozen BF16 generation')


def read_capture(event, request, directory):
    positions = request['capture_positions']
    p, f, rows = len(request['tokens']), len(request['forced_tokens']), len(positions)
    path = directory/(request['id']+'.inputs.bin')
    expected = {'event': 'capture', 'id': request['id'], 'path': str(path), 'rows': rows,
                'row_values': ROW_VALUES, 'positions': positions, 'prompt_tokens': p, 'forced_tokens': f}
    if any(event.get(k) != v for k, v in expected.items()):
        raise ValueError('native capture event differs from request')
    if path.is_symlink() or not path.is_file() or path.stat().st_size != HEADER_BYTES+rows*4+rows*ROW_VALUES*2:
        raise ValueError('capture must be a complete regular carrier with exact size')
    with path.open('rb') as stream:
        header = stream.read(HEADER_BYTES)
        expected_header = (MAGIC + struct.pack('<6I', rows, ROW_VALUES, p, f, 24, 47)
                           + MANIFEST.encode('ascii') + b'0'*64)
        if header != expected_header:
            raise ValueError('capture header/source binding mismatch')
        stored_positions = np.frombuffer(stream.read(rows*4), dtype='<u4').tolist()
        if stored_positions != positions:
            raise ValueError('capture positions mismatch')
    data = np.memmap(path, dtype='<u2', mode='r', offset=HEADER_BYTES+rows*4, shape=(rows, ROW_VALUES))
    if np.any((data & 0x7f80) == 0x7f80):
        raise ValueError('nonfinite BF16 capture input')
    return data


def append_streams(data, split, output, counts):
    if split not in {'calibration', 'observation'} or data.shape[1] != ROW_VALUES:
        raise ValueError('invalid stream split or width')
    at = 0
    for name, width in ORDER:
        with (output/f'{name}-{split}.bf16').open('ab') as stream:
            stream.write(data[:, at:at+width].tobytes(order='C'))
        counts[name][split] += len(data)
        at += width
    if at != ROW_VALUES:
        raise ValueError('native stream layout mismatch')


def assemble_streams(records, events, capture_dir, output, counts, eos):
    """Match frozen HF accumulation order: all prompts, then all assistant rows per split."""
    if len(records) != len(events):
        raise ValueError('missing completed native capture')
    mapping = []
    offsets = {'calibration': 0, 'observation': 0}
    for phase in ('prompt', 'assistant'):
        for record, event in zip(records, events, strict=True):
            request = make_request(record, eos)
            positions = request['capture_positions']
            prompt_rows = sum(p < len(request['tokens']) for p in positions)
            if phase == 'prompt':
                # Validate each immutable carrier once, after the worker has exited.
                data = read_capture(event, request, capture_dir)
                begin, end = 0, prompt_rows
            else:
                # Same exclusively-created, previously validated carrier; no repeated scan.
                data = np.memmap(event['path'], dtype='<u2', mode='r',
                                 offset=HEADER_BYTES+len(positions)*4, shape=(len(positions), ROW_VALUES))
                begin, end = prompt_rows, len(positions)
            selected = data[begin:end]
            split = record['split']
            first = offsets[split]
            append_streams(selected, split, output, counts)
            offsets[split] += len(selected)
            mapping.append({'id': record['id'], 'split': split, 'phase': phase,
                            'stream_row_start': first, 'stream_row_end': offsets[split],
                            'positions': positions[begin:end], 'carrier_row_start': begin,
                            'carrier_row_end': end, 'path': event['path']})
            del selected, data
    save(output/'position-mapping.json', {'order': 'all prompt rows then all assistant rows, original material order within each split',
                                         'streams': [name for name, _ in ORDER], 'rows_per_split': offsets, 'segments': mapping})
    return mapping


class CaptureWorker(MixedWorker):
    """Reuse worker receive/close, adding only the native research capture protocol."""
    def __init__(self, binary, output, capture_dir, lock):
        self.name, self.out = 'native_capture_bf16', output
        self.events = (output/'native-events.jsonl').open('x')
        self.stderr = (output/'native-stderr.log').open('x')
        self.cancel_read, self.cancel_write = os.pipe()
        self.messages = queue.Queue()
        command = [str(binary), MODEL, MANIFEST, str(self.cancel_read), '--research-capture-inputs='+str(capture_dir)]
        save(output/'invocation.json', {'command': command})
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, text=True, pass_fds=(self.cancel_read, lock.fd), bufsize=1)
        except BaseException:
            os.close(self.cancel_read); os.close(self.cancel_write)
            self.events.close(); self.stderr.close()
            raise
        def read():
            try:
                for line in self.process.stdout:
                    self.events.write(line); self.events.flush()
                    self.messages.put(json.loads(line))
            except BaseException as error:
                self.messages.put(error)
            finally:
                self.messages.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            self.loaded = self.receive(300)
            check_loaded(self.loaded)
        except BaseException:
            self.close()
            raise

    def capture(self, request, expected_tokens):
        with (self.out/'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(request)+'\n')
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        capture = None
        while True:
            event = self.receive(300)
            if event.get('id') != request['id']:
                raise ValueError('worker event request ID mismatch')
            if event.get('event') == 'capture' and capture is None:
                capture = event
            elif event.get('event') == 'result':
                check_result(event, request, expected_tokens)
                if capture is None:
                    raise ValueError('missing capture completion event')
                return capture, event
            else:
                raise ValueError(f'unexpected capture event: {event}')


def run(source, output, binary):
    protocol, records = load_frozen(source)
    if not binary.is_absolute() or not binary.is_file():
        raise ValueError('explicit existing absolute native binary required')
    receipt = json.loads(Path(str(binary)+'.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    capture_dir = output/'native'
    capture_dir.mkdir()
    counts = {name: {'calibration': 0, 'observation': 0} for name in STREAMS}
    for name in STREAMS:
        for split in ('calibration', 'observation'):
            (output/f'{name}-{split}.bf16').touch(exist_ok=False)
    native_protocol = {**protocol['source_protocol'], 'completed': False, 'counts': counts,
        'native_capture_source': str(source), 'binary': str(binary), 'receipt': receipt,
        'positions': 'same frozen prompt uniform128+tail16 and assistant_positions; all inputs captured natively',
        'activation_execution': 'original native BF16 prefill/decode forced frozen prefixes; no HF streams',
        'cache_reuse': False, 'temperature': 0, 'teacher_calls': 0, 'training': False,
        'stream_row_order': 'all prompts then all assistants within each split, frozen material order',
        'position_mapping': 'position-mapping.json',
        'product_adopted': False, 'quality_claim': False}
    save(output/'protocol.json', native_protocol)
    started, completed = time.perf_counter(), []
    lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        worker = CaptureWorker(binary, output, capture_dir, lock)
        try:
            save(output/'loaded.json', worker.loaded)
            for record in records:
                request = make_request(record, protocol['eos_ids'])
                event, result = worker.capture(request, record['generated_tokens'])
                completed.append({'id': record['id'], 'split': record['split'], 'capture': event,
                                  'result': result, 'tokens_equal_frozen': True})
                save(output/'progress.json', {'completed': False, 'records': completed, 'counts': counts})
                print(json.dumps({'captured': record['id'], 'rows': event['rows']}), flush=True)
        finally:
            worker.close()
        if worker.process.returncode != 0:
            raise RuntimeError('native capture worker failed')
    finally:
        lock.close()
    assemble_streams(records, [r['capture'] for r in completed], capture_dir, output, counts, protocol['eos_ids'])
    expected = {split: sum(len(capture_positions(r)) for r in records if r['split'] == split)
                for split in ('calibration', 'observation')}
    for name, width in STREAMS.items():
        for split in expected:
            if counts[name][split] != expected[split] or (output/f'{name}-{split}.bf16').stat().st_size != expected[split]*width*2:
                raise ValueError('native capture aggregate stream count/size mismatch')
    save(output/'result.json', {**native_protocol, 'completed': True, 'counts': counts,
        'records': completed, 'all_frozen_tokens_equal': True, 'elapsed_seconds': time.perf_counter()-started,
        'memory': {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'method': 'Darwin Python process peak RSS; worker Metal memory recorded in result events'}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    if (not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute()
            or not args.output.resolve().is_relative_to('/Volumes/移动数据库/lubanllm/experiments')):
        parser.error('new absolute output under mounted experiment volume required')
    run(args.source, args.output, args.binary)


if __name__ == '__main__':
    main()
