#!/usr/bin/env python3
"""Lease-owned read-only HF route replay, then serial Swift MLX expert checks.

The caller obtains the application lease. --prepare-only is CPU-only. This does
not publish pointers, qualify training, refit routing or manufacture trial proof.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
from types import SimpleNamespace

from prepare_mlx_expert_trial import ROOT, STORE, STUDY, SOURCE_ID, TARGET_ID, save
from verify_direct_q4 import CORE, MODEL, MANIFEST, IDENTITY

SOURCE_ROUTER = 'router-a970b1fb305c'
TARGET_ROUTER = 'router-18ca8e6d702f'
SOURCE_DB = ROOT / 'experiments/preservation-first-20261016/identity-expert-candidate-09-preservation-48/storage/app/memory.sqlite3'


def prepare(directory):
    from transformers import AutoTokenizer
    router = json.loads((STORE / (SOURCE_ROUTER + '.json')).read_text())
    core = json.loads((STORE / 'core-fdfa0ce5a75c.json').read_text())
    assert router['core_version'] is None and router['expert_id'] == SOURCE_ID
    assert core['parameter'] == 'model.language_model.layers.23.mlp.base.down_proj.weight'
    assert router['depth'] == 4 and router['calibrated'] and router['expert_scope']['passed']
    ids = sum(router['split_ids'].values(), [])
    db = sqlite3.connect(SOURCE_DB.as_uri() + '?mode=ro', uri=True, timeout=3)
    db.row_factory = sqlite3.Row
    try:
        samples = {row['id']: dict(row) for row in db.execute(
            'SELECT * FROM samples WHERE id IN (' + ','.join('?' * len(ids)) + ')', ids)}
    finally:
        db.close()
    assert len(samples) == len(ids) == 46
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    rows, generation = [], []
    for split, names in router['split_ids'].items():
        for name in names:
            row = samples[name]
            provenance = json.loads(row['provenance'])
            messages = provenance.get('messages') or [{'role': 'user', 'content': row['prompt']}]
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
            if not isinstance(tokens, list):
                tokens = tokens['input_ids']
            entry = {'id': name, 'split': split, 'messages': messages, 'input_tokens': tokens,
                     'target_action': provenance['route_action'], 'prompt': row['prompt']}
            rows.append(entry)
            if name in router['expert_route_proof']['ids']:
                check = provenance.get('generation_check') or {'type': 'exact', 'expected': row['answer']}
                generation.append({**entry, 'category': 'old_router_expert_closed_book',
                                   'max_tokens': check.get('max_new_tokens', 192),
                                   'expected': row['answer'], 'generation_check': check})
    assert len(generation) == 24
    candidate = copy.deepcopy(router)
    # Old scope/temperature/threshold describe the unchanged HF four-layer head.
    # The old expert's closed-book pass is not a current MLX quality assertion.
    inherited = candidate.pop('expert_route_proof')
    candidate.update(id=TARGET_ROUTER, expert_id=TARGET_ID, core_version=core['version'],
                     calibrated=False, execution_scope='local-trial', quality_admitted=False,
                     trial_engine='mlx-q4', migration={
                         'source_router_id': SOURCE_ROUTER, 'source_core_version': None,
                         'source_expert_id': SOURCE_ID, 'source_database': str(SOURCE_DB),
                         'source_router_sha256': router['sha256'],
                         'inherited_expert_route_proof': inherited,
                         'route_checked': False, 'optimizer_updates': 0,
                         'scope': 'HF layers 0..3 and original decision head unchanged; MLX downstream expert is separately trialed'})
    save(directory / 'route-cases.json', rows)
    save(directory / 'route-expert-cases.json', generation)
    save(directory / 'candidate-router.json', candidate)
    return rows


def route_replay(directory, result):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from luban.backbone import load_backbone, text_backbone
    from luban.coldpaging import ColdBackbone
    from luban.decision import FastDecisionHead
    from luban.runtime import Runtime
    rows = json.loads((directory / 'route-cases.json').read_text())
    extra = json.loads((directory / 'cases.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    runtime = Runtime.__new__(Runtime)
    runtime.device, runtime.dtype = 'mps', torch.bfloat16
    runtime.model = load_backbone(MODEL, runtime.dtype, runtime.device, cold=True)
    runtime.decoder, runtime.text_config = text_backbone(runtime.model)
    runtime.cold_backbone = ColdBackbone(runtime, MODEL)
    runtime.router = FastDecisionHead(2048).to('mps', dtype=torch.float32).eval()
    runtime.router.load_state_dict(load_file(str(STORE / (SOURCE_ROUTER + '.safetensors')), device='cpu'))
    runtime.router_meta = json.loads((STORE / (SOURCE_ROUTER + '.json')).read_text())
    runtime.branch = SimpleNamespace(expert_id=SOURCE_ID, _paths=lambda _: (
        STORE / 'layer-23' / (SOURCE_ID + '.safetensors'), STORE / 'layer-23' / (SOURCE_ID + '.json')))
    runtime._ready = lambda: None
    runtime.external_hint = lambda _: None
    entered, handles = [], []
    for index, layer in enumerate(runtime.decoder.layers):
        handles.append(layer.register_forward_pre_hook(lambda _m, _a, index=index: entered.append(index)))
    observed = []
    result['route_replay'] = {'rows': observed, 'precision': 'original HF BF16 four layers; FP32 FastDecisionHead',
                              'old_calibration_refit': False, 'current_core_parameter': CORE['tensor_key'],
                              'late_core_excluded_by_real_early_hook': False, 'released': False}
    try:
        for row in rows + extra:
            ids = row['input_tokens']
            messages = row.get('messages') or [{'role': 'user', 'content': row['prompt']}]
            if 'messages' in row:
                check = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
                if not isinstance(check, list):
                    check = check['input_ids']
                assert check == ids, 'Original template tokens differ: ' + row['id']
            runtime._tokens = lambda _messages, shallow=False, ids=ids: torch.tensor([ids], dtype=torch.long, device='mps')
            entered.clear()
            at = time.perf_counter()
            decision = Runtime.choose_route(runtime, messages)
            torch.mps.synchronize()
            assert entered == [0, 1, 2, 3], entered
            assert all(p.is_meta for p in runtime.decoder.layers[23].parameters())
            record = {'id': row['id'], 'decision': decision, 'executed_layers': list(entered),
                      'target_action': row.get('target_action'), 'seconds': time.perf_counter() - at}
            if 'target_action' in row:
                record['target_matched'] = decision['action'] == row['target_action']
            observed.append(record)
            print(json.dumps({'route': record}, ensure_ascii=False), flush=True)
        result['route_replay']['late_core_excluded_by_real_early_hook'] = True
        result['route_replay']['old_46_correct'] = sum(row.get('target_matched') is True for row in observed)
        result['route_replay']['route_checked'] = all(row['target_matched'] for row in observed if 'target_matched' in row)
    finally:
        for handle in handles:
            handle.remove()
        runtime.cold_backbone.evict_resident()
        runtime.router = runtime.cold_backbone = runtime.decoder = runtime.model = None
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        result['route_replay']['released'] = True
        result['route_replay']['mps_current_allocated_after_release'] = torch.mps.current_allocated_memory()


def native_replay(directory, binary, result):
    from tokenizers import Tokenizer
    from luban.native_engine import NativeEngine
    from luban.generation_checks import evaluate_generation
    from luban.native_mlx import model_binding
    model = model_binding(SimpleNamespace(native_inference_precision='mlx-q4',
        native_mlx_model_path=str(STUDY / 'mlx-community-4bit'), _native_mlx_binding=None))
    source = json.loads((STORE / 'layer-23' / (SOURCE_ID + '.json')).read_text())
    expert = {'file': str(STORE / 'layer-23' / (SOURCE_ID + '.safetensors')),
              'sha256': source['sha256'], 'width': source['width'], 'core_sha256': CORE['sha256']}
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    short = json.loads((directory / 'cases.json').read_text())
    old = json.loads((directory / 'route-expert-cases.json').read_text())
    result['native_replay'] = []
    for enabled in (False, True):
        run = {'expert_enabled': enabled, 'rows': [], 'closed': False}
        result['native_replay'].append(run)
        engine = NativeEngine(binary, MODEL, MANIFEST, core=CORE, expert=expert if enabled else None,
                              precision='mlx-q4', mlx_model=model)
        try:
            run['loaded'] = engine.start()
            assert run['loaded']['mlx_training'] is False, 'MLX must use inference kernels'
            run['worker_binary_sha256'] = engine.binary_sha256
            run['model_binding'] = model
            run['core_binding'] = CORE
            run['expert_binding'] = expert if enabled else None
            for row in short + (old if enabled else []):
                times = []
                response = engine.generate(row['input_tokens'], row['max_tokens'], [248046, 248044], 0,
                    lambda _token: times.append(time.perf_counter()), threading.Event(), seed=42, cache_reuse=False)
                text = tokenizer.decode(response['tokens'], skip_special_tokens=True).strip()
                observed = {'id': row['id'], 'text': text, 'response': response,
                            'tps': (len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None}
                if 'generation_check' in row:
                    observed['generation_check'] = evaluate_generation(text, row['generation_check'], response['finish_reason'] == 'stop')
                elif row.get('expected'):
                    observed['expected_exact'] = text == row['expected'].strip()
                run['rows'].append(observed)
                print(json.dumps({'native_expert': enabled, 'id': row['id'], 'text': text}, ensure_ascii=False), flush=True)
            if enabled:
                speed = next(row for row in json.loads((STUDY / 'comparison-cases.json').read_text())
                             if row['category'] == 'speed')
                speed_times = []
                speed_result = engine.generate(speed['input_tokens'], 128, [248046, 248044], 0,
                    lambda _token: speed_times.append(time.perf_counter()), threading.Event(), seed=42, cache_reuse=False)
                run['speed_probe'] = {'id': speed['id'], 'response': speed_result,
                    'stream_decode_tps': (len(speed_times)-1)/(speed_times[-1]-speed_times[0]) if len(speed_times)>1 else None,
                    'scope': 'single warm request, expert on, no prefix cache; not median or end-to-end app throughput'}
                stop, emitted = threading.Event(), []
                def cancel_after_four(token):
                    emitted.append(token)
                    if len(emitted) == 4:
                        stop.set()
                cancelled = engine.generate(speed['input_tokens'], 128, [248046, 248044], 0,
                    cancel_after_four, stop, seed=42, cache_reuse=False)
                run['controls'] = {'cancel': cancelled, 'emitted': emitted}
                assert cancelled['finish_reason'] == 'cancelled' and cancelled['tokens'] == emitted
                case = short[0]
                def repeat():
                    value = engine.generate(case['input_tokens'], case['max_tokens'], [248046, 248044], 0,
                        lambda _token: None, threading.Event(), seed=42, cache_reuse=False)
                    assert value['tokens'] == run['rows'][0]['response']['tokens']
                    assert value['finish_reason'] == run['rows'][0]['response']['finish_reason']
                    return value
                run['controls']['after_cancel'] = repeat()
                engine.close()
                run['controls']['restart_loaded'] = engine.start()
                assert run['controls']['restart_loaded']['mlx_training'] is False
                run['controls']['after_restart'] = repeat()
        finally:
            engine.close()
            run['closed'] = engine.process is None


def expert_math(directory, binary, result, expert=None):
    from luban.native_engine import _sha
    source = expert or json.loads((STORE / 'layer-23' / (SOURCE_ID + '.json')).read_text())
    receipt = json.loads(Path(str(binary) + '.json').read_text())
    actual_sha = _sha(binary)
    assert actual_sha == receipt['binary_sha256']
    process = subprocess.run([str(binary), '--check-expert',
        source.get('file', str(STORE / 'layer-23' / (SOURCE_ID + '.safetensors'))), source['sha256'],
        str(directory / 'expert-math-oracle.safetensors')], capture_output=True, text=True, timeout=180)
    result['expert_math'] = {'worker_binary_sha256': actual_sha, 'exit_code': process.returncode,
                            'stdout': process.stdout, 'stderr': process.stderr}
    assert process.returncode == 0, process.stderr
    event = json.loads(process.stdout)
    result['expert_math']['event'] = event
    assert event['event'] == 'expert_math' and event['schema'] == 'luban-mlx-expert-math-v1'
    assert event['expert_file_sha256'] == source['sha256'] and event['width'] == source['width']
    assert event['fp32_pass'] is True and event['bf16_pass_assumed'] is False
    stages = event['stages']
    assert len(stages) == 8 and {(x['dtype'], x['stage']) for x in stages} == {
        (dtype, stage) for dtype in ('fp32', 'bf16') for stage in ('gate', 'up', 'activation', 'output')}
    assert all(row['finite'] is True and row['nonzero'] is True for row in stages)
    assert all(0 <= row['relative_l2'] <= .03 for row in stages if row['dtype'] == 'bf16'), 'BF16 expert relative L2 exceeds preregistered 0.03'
    result['expert_math']['bf16_relative_l2_limit'] = .03
    result['expert_math']['bf16_numerical_pass'] = True
    result['expert_math']['passed'] = True
    print(json.dumps({'expert_math': result['expert_math']}, ensure_ascii=False), flush=True)


def main():
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-routes', action='store_true')
    parser.add_argument("--model-metadata", type=Path, help="Use a metadata-only original-model directory")
    args = parser.parse_args()
    if args.model_metadata is not None:
        MODEL = args.model_metadata.resolve()
        assert (MODEL / "config.json").is_file() and (MODEL / "tokenizer.json").is_file()
        assert not list(MODEL.glob("model*.safetensors")), "Acceptance requires no original backbone weights"
    if args.prepare_only:
        rows = prepare(args.directory)
        print(json.dumps({'prepared_routes': len(rows), 'gpu_loaded': False}))
        return
    assert args.output and not args.output.exists()
    from luban.process_lock import ModelProcessLock
    lock = ModelProcessLock(IDENTITY, 'mps')
    result = {'schema': 'mlx-expert-router-local-trial-v1', 'scope': 'local-trial',
              'quality_admitted': False, 'optimizer_updates': 0, 'teacher_calls': 0,
              'source_core_version': None, 'target_core_version': 'core-fdfa0ce5a75c',
              'source_expert_id': SOURCE_ID, 'target_expert_id': TARGET_ID,
              'source_router_id': SOURCE_ROUTER, 'target_router_id': TARGET_ROUTER,
              'app_lease': 'held by caller', 'complete': False}
    at = time.perf_counter()
    try:
        if not args.skip_routes:
            route_replay(args.directory, result)
        if args.binary:
            expert_math(args.directory, args.binary, result)
            native_replay(args.directory, args.binary, result)
        result['complete'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        lock.close()
        result['model_process_lock_released'] = True
        result['seconds'] = time.perf_counter() - at
        save(args.output, result)


if __name__ == '__main__':
    main()
