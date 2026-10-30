"""Append HF replay activations from fixed native BF16 continuations to Q4 calibration."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch
from tokenizers import Tokenizer

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from luban.backbone import load_backbone, text_backbone
from luban.process_lock import ModelProcessLock
from prepare_q4_packed import PROJECTIONS, input_name

LIMIT = 256
SEED = 42
BOUNDARY = ('Native BF16 generated token distribution, HF BF16 full-sequence replay activations; '
            'not bitwise native decode activations: prefill/decode grouping, kernel rounding and state updates differ.')
STREAMS = {input_name(layer, kind): k for layer, kind, _, k in PROJECTIONS}


def validate_source(source):
    if (source.get('schema') != 'q4-packed-calibration-v1' or source.get('completed') is not True or source.get('core') is not None
            or source.get('core_digest') != '0'*64 or source.get('manifest') != MANIFEST
            or source.get('identity') != IDENTITY or source.get('model') != MODEL):
        raise ValueError('completed original-core source binding required')
    rows = source['materials']
    if (len(rows) != 20 or len({r['id'] for r in rows}) != 20
            or [r['split'] for r in rows].count('calibration') != 16
            or [r['split'] for r in rows].count('observation') != 4
            or any(not isinstance(r['text'], str) or not r['text'] for r in rows)):
        raise ValueError('fixed sixteen calibration and four observation materials required')
    if set(source['counts']) != set(STREAMS) or any(
            type(count) is not int or count <= 0
            for counts in source['counts'].values() for count in counts.values()):
        raise ValueError('invalid source activation streams')
    if any(set(counts) != {'calibration', 'observation'} for counts in source['counts'].values()):
        raise ValueError('invalid source activation split')


def prompt_cases(source):
    """Render the original template, then verify native tokenizer IDs against HF on CPU."""
    validate_source(source)
    hf = __import__('transformers').AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False)
    native = Tokenizer.from_file(str(Path(MODEL)/'tokenizer.json'))
    cases = []
    for row in source['materials']:
        messages = [{'role': 'user', 'content': row['text']}]
        text = hf.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        tokens = native.encode(text, add_special_tokens=False).ids
        expected = hf.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                         return_tensors='pt', enable_thinking=False)
        if not isinstance(expected, torch.Tensor):
            expected = expected['input_ids']
        expected = expected[0].tolist()
        if tokens != expected or len(tokens) != row['tokens'] or len(tokens)+LIMIT > 2048:
            raise ValueError('original capture prompt tokenization mismatch or context overflow')
        cases.append({'id': row['id'], 'split': row['split'], 'prompt_tokens': tokens})
    config = json.loads((Path(MODEL)/'config.json').read_text())
    text_config = config.get('text_config', config)
    eos = text_config['eos_token_id']
    return cases, eos if isinstance(eos, list) else [eos], text_config['vocab_size']


def replay_record(case, result, eos_ids, vocab_size):
    emitted = result['tokens']
    if (result.get('id') != case['id'] or not isinstance(emitted, list) or not 1 <= len(emitted) <= LIMIT
            or any(type(t) is not int or not 0 <= t < vocab_size for t in emitted)
            or any(t in eos_ids for t in emitted[:-1]) or result.get('cached_prompt_tokens') != 0
            or result.get('prompt_tokens') != len(case['prompt_tokens'])
            or result.get('generated_tokens') != len(emitted)):
        raise ValueError('invalid native generation record')
    stopped = emitted[-1] in eos_ids
    if (result.get('complete') is not stopped or result.get('finish_reason') != ('stop' if stopped else 'length')
            or (not stopped and len(emitted) != LIMIT)):
        raise ValueError('generation must stop at EOS or the fixed budget; no output filtering')
    # Native emits its final token without forwarding it; the first output uses the final prompt state.
    forwarded = emitted[:-1]
    positions = sorted(set(np.linspace(0, len(forwarded)-1, min(128, len(forwarded)), dtype=int).tolist()
                           + list(range(max(0, len(forwarded)-16), len(forwarded))))) if forwarded else []
    begin = len(case['prompt_tokens'])
    return {**case, 'generated_tokens': emitted, 'complete': stopped, 'finish_reason': result['finish_reason'],
            'replay_tokens': case['prompt_tokens']+forwarded,
            'assistant_positions': [begin+i for i in positions], 'native_result': result}


def generate(source_path, output, binary):
    source = json.loads((source_path/'result.json').read_text())
    cases, eos, vocab = prompt_cases(source)
    receipt = json.loads(Path(str(binary)+'.json').read_text())
    save(output/'protocol.json', {'schema': 'q4-decode-capture-v1', 'source': str(source_path),
         'source_protocol': source, 'binary': str(binary), 'receipt': receipt, 'cases': cases,
         'eos_ids': eos, 'vocab_size': vocab, 'seed': SEED, 'temperature': 0, 'max_new_tokens': LIMIT,
         'cache_reuse': False, 'numerical_boundary': BOUNDARY, 'selection': 'all frozen materials; no answer filtering',
         'teacher_calls': 0, 'training': False, 'product_adopted': False})
    start, records = time.perf_counter(), []
    lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        worker = MixedWorker(binary, 'baseline_bf16', output, lock, precision='bf16', core=None)
        try:
            if worker.loaded.get('manifest_sha256') != MANIFEST or worker.loaded.get('quantized_projections') != 0:
                raise ValueError('native source is not the bound original BF16 graph')
            save(output/'loaded.json', worker.loaded)
            for case in cases:
                result = worker.request(case['prompt_tokens'], case['id'], max_tokens=LIMIT,
                                        temperature=0, seed=SEED, reuse=False, eos_ids=eos)
                records.append(replay_record(case, result, eos, vocab))
                save(output/'generation-progress.json', {'completed': False, 'records': records})
        finally:
            worker.close()
        if worker.process.returncode != 0:
            raise RuntimeError('native BF16 generation worker failed')
    finally:
        lock.close()
    save(output/'generation-result.json', {'completed': True, 'records': records,
         'elapsed_seconds': time.perf_counter()-start, 'numerical_boundary': BOUNDARY})


def capture(output):
    protocol = json.loads((output/'protocol.json').read_text())
    source, source_path = protocol['source_protocol'], Path(protocol['source'])
    validate_source(source)
    generated = json.loads((output/'generation-result.json').read_text())
    if generated.get('completed') is not True or len(generated['records']) != 20 or len(protocol['cases']) != 20:
        raise ValueError('complete generation stage required')
    records = []
    for material, case, record in zip(source['materials'], protocol['cases'], generated['records'], strict=True):
        if (case['id'], case['split']) != (material['id'], material['split']):
            raise ValueError('material split or order changed')
        expected = replay_record(case, record['native_result'], protocol['eos_ids'], protocol['vocab_size'])
        if expected != record:
            raise ValueError('replay tokens or assistant positions changed')
        records.append(record)
    counts = {name: dict(value) for name, value in source['counts'].items()}
    # New files assemble original prompt observations plus additional decode observations.
    # Existing capture files are never modified or replaced.
    for name, width in STREAMS.items():
        for split in ('calibration', 'observation'):
            old, new = source_path/f'{name}-{split}.bf16', output/f'{name}-{split}.bf16'
            if old.stat().st_size != counts[name][split]*width*2 or new.exists():
                raise ValueError('source size mismatch or output already exists')
    for name in STREAMS:
        for split in ('calibration', 'observation'):
            shutil.copyfile(source_path/f'{name}-{split}.bf16', output/f'{name}-{split}.bf16')
    start = time.perf_counter()
    lock = ModelProcessLock(IDENTITY, 'mps')
    handles, seen, model, decoder = [], set(), None, None
    current = None
    try:
        model = load_backbone(MODEL, torch.bfloat16, 'mps')
        decoder, _ = text_backbone(model)

        def hook_for(name):
            def hook(_module, arguments):
                x = arguments[0].detach().reshape(-1, STREAMS[name])
                if name in seen or len(x) != len(current['replay_tokens']):
                    raise ValueError('unexpected HF hook count or replay sequence size')
                seen.add(name)
                values = x[current['assistant_positions']].to(device='cpu', dtype=torch.bfloat16)
                if not torch.isfinite(values).all():
                    raise ValueError('nonfinite assistant activation')
                with (output/f'{name}-{current["split"]}.bf16').open('ab') as stream:
                    values.view(torch.uint16).numpy().astype('<u2').tofile(stream)
                counts[name][current['split']] += len(values)
            return hook

        for layer in range(24):
            mlp = decoder.layers[layer].mlp
            handles.append(mlp.gate_proj.register_forward_pre_hook(hook_for(input_name(layer, 'gate'))))
            if layer != 23:
                handles.append(mlp.down_proj.register_forward_pre_hook(hook_for(input_name(layer, 'down'))))
        with torch.inference_mode():
            for current in records:
                seen.clear()
                if current['assistant_positions']:
                    decoder(torch.tensor([current['replay_tokens']], dtype=torch.long, device='mps'), use_cache=False)
                    if seen != set(STREAMS):
                        raise ValueError('missing replay activation stream')
                save(output/'capture-progress.json', {'id': current['id'], 'counts': counts})
    finally:
        for handle in handles:
            handle.remove()
        del decoder, model
        gc.collect()
        torch.mps.empty_cache()
        lock.close()
    added = {split: sum(len(r['assistant_positions']) for r in records if r['split'] == split)
             for split in ('calibration', 'observation')}
    for name, width in STREAMS.items():
        for split in added:
            if (counts[name][split] != source['counts'][name][split]+added[split]
                    or (output/f'{name}-{split}.bf16').stat().st_size != counts[name][split]*width*2):
                raise ValueError('final activation count mismatch')
    save(output/'result.json', {**source, 'schema': 'q4-packed-decode-calibration-v1', 'completed': True,
         'counts': counts, 'source_capture': str(source_path), 'generation': 'generation-result.json',
         'additional_positions_per_stream': added, 'positions': 'original prompt rows plus assistant input128 uniform+last16',
         'numerical_boundary': BOUNDARY, 'elapsed_seconds': time.perf_counter()-start,
         'memory': {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'method': 'Darwin peak RSS for HF replay process; not combined Metal physical footprint'}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('generate', 'capture'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not os.path.ismount('/Volumes/移动数据库') or not output.is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    if args.stage == 'generate':
        if args.source is None or args.binary is None:
            parser.error('generate requires --source and --binary')
        output.mkdir(parents=True, exist_ok=False)
    elif (output/'result.json').exists():
        raise ValueError('completed capture cannot be overwritten')
    try:
        if args.stage == 'generate':
            generate(args.source.resolve(), output, args.binary.resolve())
        else:
            capture(output)
    except BaseException as error:
        save(output/f'{args.stage}-failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
