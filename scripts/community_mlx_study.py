#!/usr/bin/env python3
"""Research-only Qwen3.5 MLX comparison; caller holds the shared GPU model lease.

Run with PYTHONPATH=build/community-mlx-runtime:src .venv/bin/python ...
No chat templating: cases contain the exact input token IDs used by all engines.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

DEFAULT_MODEL = '/Volumes/移动数据库/lubanllm/models/community-q4-study-20261017/mlx-community-4bit'
CORE_KEY = 'model.language_model.layers.23.mlp.base.down_proj.weight'
STOP_TOKENS = (248046, 248044)


def read_cases(path, config):
    cases = json.loads(path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError('--cases must contain a nonempty JSON array')
    text_config = config.get('text_config', config)
    vocab = text_config['vocab_size']
    context = text_config['max_position_embeddings']
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f'case {index}: expected an object')
        tokens, maximum = case.get('input_tokens'), case.get('max_tokens')
        if not isinstance(tokens, list) or not tokens or any(
            type(token) is not int or not 0 <= token < vocab for token in tokens
        ):
            raise ValueError(f'case {index}: invalid input_tokens')
        if type(maximum) is not int or maximum < 1 or len(tokens) + maximum > context:
            raise ValueError(f'case {index}: invalid max_tokens/context length')
    return cases


def apply_core(model, path, key, mx, nn):
    if key not in (CORE_KEY, CORE_KEY.replace('.base.', '.')):
        raise ValueError('only the accepted layer 23 down_proj core is supported')
    weights = mx.load(str(path), stream=mx.cpu)
    if set(weights) != {key}:
        raise ValueError('core must contain exactly the requested tensor')
    weight = weights[key]
    if tuple(weight.shape) != (2048, 6144) or weight.dtype != mx.bfloat16:
        raise ValueError('core must be BF16 [2048,6144]')
    if not mx.all(mx.isfinite(weight)).item():
        raise ValueError('core contains nonfinite values')
    # Preserve the learned BF16 parameter; never requantize or alter the source.
    layer = nn.Linear(6144, 2048, bias=False)
    layer.weight = weight
    model.language_model.model.layers[23].mlp.down_proj = layer
    mx.eval(model.parameters())
    return {'file': str(path), 'tensor_key': key, 'dtype': 'bfloat16',
            'shape': list(weight.shape), 'weight_bytes': weight.nbytes,
            'scope': 'single down_proj replacement; product experts are absent'}


def run_case(case, model, tokenizer, mx, generate_step, make_prompt_cache, sampler, stream):
    mx.synchronize(stream)
    mx.clear_cache()
    mx.random.seed(42)
    cache = make_prompt_cache(model)
    prompt = mx.array(case['input_tokens'], dtype=mx.int32)
    mx.eval(prompt)
    mx.reset_peak_memory()
    tokens, first, last = [], None, None
    finish = 'length'
    started = time.perf_counter()
    iterator = generate_step(prompt, model, stream=stream, max_tokens=case['max_tokens'],
                             sampler=sampler, prompt_cache=cache)
    try:
        for token, _ in iterator:
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            tokens.append(token)
            if token in STOP_TOKENS:
                finish = 'stop'
                break
    finally:
        iterator.close()
        # generate_step can have an asynchronous lookahead when EOS stops early.
        # Drain before releasing this case's cache or starting the next case.
        mx.synchronize(stream)
    completed = time.perf_counter()
    peak = mx.get_peak_memory()
    decode = last - first if len(tokens) > 1 else 0.0
    row = {key: case[key] for key in ('id', 'prompt', 'category', 'expected') if key in case}
    row.update(input_tokens=case['input_tokens'], max_tokens=case['max_tokens'],
               tokens=tokens, text=tokenizer.decode(tokens, skip_special_tokens=True),
               finish_reason=finish, stop_token=tokens[-1] if finish == 'stop' else None,
               output_token_count=len(tokens), prompt_token_count=len(case['input_tokens']),
               ttft_seconds=first - started if first is not None else None,
               decode_seconds=decode,
               decode_tokens_per_second=(len(tokens) - 1) / decode if decode > 0 else None,
               generation_seconds=last - started if last is not None else None,
               completed_seconds=completed - started, peak_mlx_bytes=peak)
    expected = case.get('expected')
    if isinstance(expected, str):
        row['expected_match'] = row['text'] == expected
    elif isinstance(expected, list):
        row['expected_match'] = tokens == expected
    elif isinstance(expected, dict):
        fields = set(expected) & {'tokens', 'text', 'finish_reason'}
        row['expected_match'] = all(row[key] == expected[key] for key in fields) if fields else None
    return row


def execute(args, cases, report):
    # Import only after validation/check-only: importing MLX creates a GPU stream.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.generate import generate_step, generation_stream, wired_limit
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    report['versions'] = {name: importlib.metadata.version(name)
                          for name in ('mlx', 'mlx-metal', 'mlx-lm')}
    mx.random.seed(42)
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, tokenizer, config = load(str(args.model), return_config=True)
    mx.synchronize()
    report['model_load_seconds'] = time.perf_counter() - started
    report['model_load_peak_mlx_bytes'] = mx.get_peak_memory()
    if args.core:
        report['core'] = apply_core(model, args.core, args.core_key, mx, nn)
    parameters = tree_flatten(model.parameters())
    report['language_parameter_bytes'] = sum(value.nbytes for _, value in parameters)
    report['language_parameter_tensors'] = len(parameters)
    report['quantization'] = config.get('quantization')
    report['vision_loaded'] = False
    report['vision_exclusion'] = 'mlx_lm.models.qwen3_5.Model constructs TextModel; sanitize drops vision_tower/model.visual'
    del parameters
    sampler = make_sampler(temp=0.0)
    with wired_limit(model, [generation_stream]):
        warmup = dict(cases[0], max_tokens=min(args.warmup_tokens, cases[0]['max_tokens']))
        started = time.perf_counter()
        warmup_row = run_case(warmup, model, tokenizer, mx, generate_step,
                             make_prompt_cache, sampler, generation_stream)
        report['warmup'] = {'seconds': time.perf_counter() - started,
                            'input_tokens': warmup['input_tokens'],
                            'max_tokens': warmup['max_tokens'],
                            'output_token_count': warmup_row['output_token_count'],
                            'included_in_measurements': False}
        for index, case in enumerate(cases):
            row = run_case(case, model, tokenizer, mx, generate_step,
                           make_prompt_cache, sampler, generation_stream)
            row.setdefault('id', str(index))
            report['cases'].append(row)
            print(json.dumps({'case': row}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--core', type=Path)
    parser.add_argument('--core-key', default=CORE_KEY)
    parser.add_argument('--warmup-tokens', type=int, default=4)
    parser.add_argument('--check-only', action='store_true', help='validate inputs without importing MLX')
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    args.cases = args.cases.resolve(strict=True)
    if args.core:
        args.core = args.core.resolve(strict=True)
    if args.output.exists() or not args.output.is_absolute() or not args.output.parent.is_dir():
        parser.error('--output must be a new absolute filename in an existing directory')
    if args.warmup_tokens < 1:
        parser.error('--warmup-tokens must be positive')
    config = json.loads((args.model / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5':
        parser.error('this runner only supports qwen3_5')
    cases = read_cases(args.cases, config)
    if args.check_only:
        print(json.dumps({'validated_cases': len(cases), 'gpu_loaded': False,
                          'model': str(args.model), 'core': str(args.core) if args.core else None}))
        return
    report = {'schema': 'community-mlx-study-v1', 'model': str(args.model),
              'cases_file': str(args.cases), 'hardware': platform.platform(),
              'machine': platform.machine(), 'seed': 42, 'temperature': 0,
              'stop_tokens': list(STOP_TOKENS), 'core': None, 'cases': [],
              'cache': 'fresh prompt/recurrent cache per case; independent warmup; OS file cache not purged',
              'lease': 'caller must hold the shared product model lease for this process lifetime',
              'measurement': 'host wall clock; TTFT includes prefill; decode is first-to-last token excluding first token; text decode excluded',
              'memory_measurement': 'MLX allocator peak per-case reset, includes resident model; not process physical footprint or system peak',
              'limitations': ['research backend only; no product expert branches or learned router',
                              'existing exposed observations are not blind quality evaluation',
                              'raw tokens include stopping EOS; text omits all special tokens',
                              'generate_step async lookahead is drained after stop; completed_seconds includes that cleanup',
                              'exact expected comparisons do not measure semantic quality'],
              'optimizer_steps': 0, 'teacher_calls': 0}
    failed = None
    try:
        execute(args, cases, report)
    except BaseException as error:
        failed = error
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        with args.output.open('x') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    if failed is not None:
        raise failed


if __name__ == '__main__':
    main()
