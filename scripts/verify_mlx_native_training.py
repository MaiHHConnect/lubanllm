#!/usr/bin/env python3
"""Finite native MLX training qualification. Caller owns the app GPU lease.

Only approved TRAIN rows are used. This tests mathematics and lifecycle, never
adopts an expert or grants closed-book semantic quality or a production proof.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import select
import sqlite3
import subprocess
import threading
import time
from types import SimpleNamespace

from prepare_mlx_expert_trial import ROOT, STORE, STUDY, SOURCE_ID, save
from verify_direct_q4 import CORE, MODEL, MANIFEST, IDENTITY
from verify_mlx_trial_router import SOURCE_DB


def frozen_stat(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def authorized_inputs():
    db = sqlite3.connect(SOURCE_DB.as_uri() + '?mode=ro', uri=True, timeout=3)
    db.row_factory = sqlite3.Row
    try:
        candidates = [dict(row) for row in db.execute(
            "SELECT * FROM samples WHERE purpose='training' AND approved=1 AND training_allowed=1 ORDER BY created,id")]
    finally:
        db.close()
    chosen, families = [], set()
    for row in candidates:
        provenance = json.loads(row['provenance'])
        family = provenance.get('split_family')
        if (row['source'] != 'human' or not provenance.get('verification_evidence')
                or provenance.get('base_identity') != IDENTITY or not family or family in families):
            continue
        families.add(family)
        chosen.append({**row, 'messages': provenance.get('messages') or [{'role': 'user', 'content': row['prompt']}],
                       'provenance': provenance})
        if len(chosen) == 3:
            break
    assert len(chosen) == 3, 'Need three approved distinct-family TRAIN rows'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    assert tokenizer.eos_token_id == 248046
    for row in chosen:
        prompt = tokenizer.apply_chat_template(row['messages'], add_generation_prompt=True,
                                               tokenize=True, enable_thinking=False)
        if not isinstance(prompt, list):
            prompt = prompt['input_ids']
        answer = tokenizer.encode(row['answer'], add_special_tokens=False) + [248046]
        row['reference'] = {'tokens': prompt + answer, 'labels': [-100] * len(prompt) + answer,
                            'prompt_tokens': len(prompt), 'eos': 248046}
        assert 2 <= len(row['reference']['tokens']) <= 2048
    return chosen


def rpc(engine, op, events, *, cancel=False, **arguments):
    """One private serialized request; NativeEngine retains lifecycle/FD ownership."""
    with engine._serial:
        engine.start()
        engine._request_id += 1
        identifier = 'training-' + str(engine._request_id)
        deadline = time.monotonic() + engine.request_timeout
        # Do not allow a cancellation byte from a completed request into the next one.
        while True:
            try:
                if not os.read(engine._cancel_read_fd, 4096):
                    break
            except BlockingIOError:
                break
        payload = memoryview((json.dumps({'id': identifier, 'op': op, **arguments}, allow_nan=False) + '\n').encode())
        started = time.monotonic()
        while payload:
            remaining = deadline - time.monotonic()
            assert remaining > 0, 'Request submission timed out'
            fd = engine.process.stdin.fileno()
            if select.select([], [fd], [], min(remaining, .05))[1]:
                try:
                    payload = payload[os.write(fd, payload):]
                except BlockingIOError:
                    pass
        if cancel:
            # Signal immediately after the complete JSON frame is submitted. The worker
            # must either cancel before its atomic commit, or explicitly report a full step.
            os.write(engine._cancel_fd, b'\x01')
        while (event := engine._next_event(deadline)) is None:
            pass
        assert event.get('id') == identifier, 'Training response identity differs'
        record = {'op': op, 'seconds': time.monotonic() - started, 'event': event,
                  'memory': engine.status()}
        events.append(record)
        print(json.dumps({'op': op, 'seconds': record['seconds'], 'event': event}, ensure_ascii=False), flush=True)
        assert event['event'] != 'train_error', event
        return event


def compare_checkpoint(first, second):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    a, b = load_file(str(first), device='cpu'), load_file(str(second), device='cpu')
    with safe_open(str(first), framework='pt', device='cpu') as handle:
        am = handle.metadata()
    with safe_open(str(second), framework='pt', device='cpu') as handle:
        bm = handle.metadata()
    assert am == bm, 'Checkpoint metadata changed'
    assert set(a) == set(b), 'Checkpoint tensor keys changed'
    rows = []
    for key in sorted(a):
        x, y = a[key], b[key]
        exact = x.dtype == y.dtype and tuple(x.shape) == tuple(y.shape) and torch.equal(x, y)
        rows.append({'name': key, 'dtype': str(x.dtype), 'shape': list(x.shape), 'exact': exact})
        assert exact, 'Checkpoint replay differs: ' + key
    assert all(a[key].dtype == torch.bfloat16 for key in ('gate.weight', 'up.weight', 'down.weight'))
    assert all(a[key].dtype == torch.float32 for key in a if key.startswith('adam.'))
    assert a['rng'].dtype == torch.uint32
    return {'metadata_exact': True, 'tensors': rows, 'step': int(am['step'])}


def generation(engine, prompt):
    started = time.monotonic()
    response = engine.generate(prompt, 128, [248044, 248046], 0, lambda _: None,
                               threading.Event(), seed=42, cache_reuse=False)
    assert response['finish_reason'] in ('stop', 'length')
    return {'response': response, 'seconds': time.monotonic() - started}


def same_generation(a, b):
    return (a['response']['tokens'] == b['response']['tokens']
            and a['response']['finish_reason'] == b['response']['finish_reason'])


def main():
    global MODEL, STORE, SOURCE_ID, CORE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument("--model-metadata", type=Path, help="Use a metadata-only original-model directory")
    parser.add_argument("--source-root", type=Path, help="Explicit isolated factory source; never changes accepted pointers")
    args = parser.parse_args()
    if args.source_root is not None:
        source_root = args.source_root.resolve(strict=True)
        assert source_root.is_relative_to(ROOT / 'experiments')
        STORE = source_root / 'modules/luban0-v1'
        proof = json.loads((source_root / 'runtime/mlx/expert-proof.json').read_text())
        assert proof['expert_id'] == 'learned-bafb4050374f' and proof['core_version'] == 'core-fdfa0ce5a75c'
        SOURCE_ID = proof['expert_id']
        CORE = {**CORE, 'file': str(STORE / (proof['core_version'] + '.safetensors')), 'sha256': proof['core_file_sha256']}
    if args.model_metadata is not None:
        MODEL = args.model_metadata.resolve()
        assert (MODEL / "config.json").is_file() and (MODEL / "tokenizer.json").is_file()
        assert not list(MODEL.glob("model*.safetensors")), "Acceptance requires no original backbone weights"
    assert Path('/Volumes/移动数据库').is_mount(), 'External volume not mounted'
    directory, output, binary = args.directory.resolve(), args.output.resolve(), args.binary.resolve()
    assert directory.is_relative_to(ROOT / 'experiments') and output.is_relative_to(directory)
    assert not output.exists() and binary.is_file()
    directory.mkdir(parents=True, exist_ok=True)
    names = ['step1', 'step2', 'step2-replayed', 'after-cancel', 'after-continue']
    paths = {name: {'path': str(directory / (name + '-expert.safetensors')),
                    'checkpoint_path': str(directory / (name + '-checkpoint.safetensors'))} for name in names}
    assert all(not Path(path).exists() for value in paths.values() for path in value.values())
    result = {'schema': 'mlx-native-training-qualification-v1', 'complete': False,
              'quality_admitted': False, 'teacher_calls': 0,
              'scope': 'real model TRAIN mathematics/lifecycle regression; no semantic admission',
              'app_lease': 'caller must hold application lease',
              'hardware': {'platform': platform.platform(), 'machine': platform.machine(),
                  'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  'system_ram_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))},
              'precision': {'base': 'MLX affine Q4 group64', 'core': 'BF16', 'expert': 'BF16',
                            'adam': 'FP32', 'loss': 'FP32 CE+5*KL'},
              'seed': 42, 'temperature': 0, 'cache_reuse': False,
              'file_cache': 'not purged; one resident training session plus explicit saved-expert restart',
              'memory_measurement': 'Darwin proc_pid_rusage v4 child current/lifetime peak after commands; excludes Python/app',
              'gates': {name: False for name in ('prepare_equivalent', 'real_adam_updates',
                  'checkpoint_replay_exact', 'cancellation_atomic', 'continued_after_cancel',
                  'saved_expert_reload_equal', 'base_core_frozen')}, 'events': [], 'generations': {}}
    started = time.monotonic()
    lock = None
    engine = None
    try:
        rows = authorized_inputs()
        result['samples'] = rows
        result['sample_split'] = {'training': [rows[0]['id'], rows[1]['id']],
                                  'frozen_teacher_protection': [rows[2]['id']],
                                  'all_original_purpose': 'training', 'held_out_validation_used': False,
                                  'source_database': str(SOURCE_DB)}
        if args.prepare_only:
            result['prepared_only'] = True
            return
        from luban.native_engine import NativeEngine, _sha
        from luban.native_mlx import model_binding
        from luban.process_lock import ModelProcessLock
        model = model_binding(SimpleNamespace(native_inference_precision='mlx-q4',
            native_mlx_model_path=str(STUDY / 'mlx-community-4bit'), _native_mlx_binding=None))
        metadata = json.loads((STORE / 'layer-23' / (SOURCE_ID + '.json')).read_text())
        width = metadata['width']
        assert width == (32 if args.source_root is not None else 64)
        expert = {'file': str(STORE / 'layer-23' / (SOURCE_ID + '.safetensors')),
                  'sha256': metadata['sha256'], 'width': width, 'core_sha256': CORE['sha256']}
        frozen_paths = [Path(model['path']) / 'model.safetensors', Path(CORE['file']), Path(expert['file'])]
        before = {str(path): frozen_stat(path) for path in frozen_paths}
        result.update(core_file_sha256=CORE['sha256'], mlx_model_sha256=model['sha256'],
                      parent_expert_file_sha256=expert['sha256'], frozen_file_stat_before=before)
        lock = ModelProcessLock(IDENTITY, 'mps')
        engine = NativeEngine(binary, MODEL, MANIFEST, core=CORE, expert=expert,
            precision='mlx-q4', mlx_model=model, load_timeout=300, request_timeout=600, cancel_timeout=120)
        result['loaded'] = engine.start()
        result['worker_binary_sha256'] = engine.binary_sha256
        assert result['loaded']['core_loaded'] and result['loaded']['expert_loaded']
        events = result['events']
        for row in rows:
            prepared = rpc(engine, 'prepare', events, messages=row['messages'], answer=row['answer'])
            assert prepared['event'] == 'prepared'
            assert {key: prepared[key] for key in row['reference']} == row['reference'], 'Swift/HF preparation differs'
        result['gates']['prepare_equivalent'] = True
        prompt = rows[0]['reference']['tokens'][:rows[0]['reference']['prompt_tokens']]
        selected = rpc(engine, 'train_select', events, source='parent')
        assert selected['source'] == 'parent' and selected['compilation_enabled'] is False
        result['learning_observation_mode'] = 'compilation_disabled'
        result['generations']['parent'] = generation(engine, prompt)
        binding = 'real-mlx-training-qualification-v1:' + ':'.join(row['id'] for row in rows)
        begin = rpc(engine, 'train_begin', events, seed=42, width=width, learning_rate=.0002,
                    preservation_weight=5, initialize='parent', binding=binding)
        assert begin['event'] == 'train_started' and begin['step'] == 0 and begin['base_frozen'] is True
        result['generations']['initial_candidate'] = generation(engine, prompt)
        assert same_generation(result['generations']['parent'], result['generations']['initial_candidate'])
        sample = lambda row: {key: row['reference'][key] for key in ('tokens', 'labels')}
        protection = sample(rows[2])
        def update(row, cancel=False):
            return rpc(engine, 'train_step', events, sample=sample(row), preservation=protection,
                       teacher='parent', cancel=cancel)
        first = update(rows[0]); assert first['event'] == 'train_step' and first['step'] == 1
        assert first['gradient_norm'] > 0
        rpc(engine, 'train_save', events, **paths['step1'])
        second = update(rows[1]); assert second['event'] == 'train_step' and second['step'] == 2
        rpc(engine, 'train_save', events, **paths['step2'])
        from safetensors.torch import load_file
        import torch
        original_arrays = load_file(expert['file'], device='cpu')
        updated_arrays = load_file(paths['step2']['path'], device='cpu')
        changed = {key: int(torch.count_nonzero(original_arrays[key] != updated_arrays[key]))
                   for key in original_arrays}
        assert sum(changed.values()) > 0, 'Two updates did not alter BF16 expert weights'
        result['changed_bf16_parameters'] = changed
        result['gates']['real_adam_updates'] = True
        result['generations']['step2'] = generation(engine, prompt)
        restored = rpc(engine, 'train_restore', events, checkpoint_path=paths['step1']['checkpoint_path'], binding=binding)
        assert restored['step'] == 1
        repeated = update(rows[1]); assert repeated['step'] == 2
        rpc(engine, 'train_save', events, **paths['step2-replayed'])
        result['checkpoint_replay'] = compare_checkpoint(paths['step2']['checkpoint_path'], paths['step2-replayed']['checkpoint_path'])
        result['gates']['checkpoint_replay_exact'] = True
        result['generations']['step2_replayed'] = generation(engine, prompt)
        assert same_generation(result['generations']['step2'], result['generations']['step2_replayed'])
        cancelled = update(rows[0], cancel=True)
        result['cancellation'] = {'event': cancelled, 'method': 'pipe signalled immediately after complete train_step frame; no artificial sleep'}
        assert cancelled['event'] == 'train_cancelled' and cancelled['step'] == 2, 'Cancellation was not observed before commit; do not claim atomic cancellation'
        rpc(engine, 'train_save', events, **paths['after-cancel'])
        result['cancellation']['checkpoint_exact'] = compare_checkpoint(paths['step2']['checkpoint_path'], paths['after-cancel']['checkpoint_path'])
        result['gates']['cancellation_atomic'] = True
        continued = update(rows[0]); assert continued['event'] == 'train_step' and continued['step'] == 3
        rpc(engine, 'train_save', events, **paths['after-continue'])
        result['gates']['continued_after_cancel'] = True
        result['generations']['after_continue'] = generation(engine, prompt)
        result['training_process_memory'] = engine.status()
        rpc(engine, 'train_end', events, restore_parent=True)
        result['generations']['restored_parent'] = generation(engine, prompt)
        assert same_generation(result['generations']['parent'], result['generations']['restored_parent'])
        engine.close()
        result['training_process_closed'] = engine.process is None
        # One required new-artifact digest binds the standard export for the real reload.
        saved_expert = {**expert, 'file': paths['after-continue']['path'], 'sha256': _sha(paths['after-continue']['path'])}
        result['saved_expert'] = saved_expert
        engine = NativeEngine(binary, MODEL, MANIFEST, core=CORE, expert=saved_expert,
            precision='mlx-q4', mlx_model=model, load_timeout=300, request_timeout=600)
        result['saved_expert_loaded'] = engine.start()
        assert result['saved_expert_loaded']['expert_file_sha256'] == saved_expert['sha256']
        assert engine.binary_sha256 == result['worker_binary_sha256']
        # Keep the default product-mode gate before selecting any learning mode.
        result['generations']['saved_expert_restarted'] = generation(engine, prompt)
        assert same_generation(result['generations']['after_continue'], result['generations']['saved_expert_restarted'])
        selected = rpc(engine, 'train_select', events, source='parent')
        assert selected['source'] == 'parent' and selected['compilation_enabled'] is False
        result['generations']['saved_expert_restarted_learning'] = generation(engine, prompt)
        assert same_generation(result['generations']['after_continue'], result['generations']['saved_expert_restarted_learning'])
        result['gates']['saved_expert_reload_equal'] = True
        after = {str(path): frozen_stat(path) for path in frozen_paths}
        result['frozen_file_stat_after'] = after
        assert before == after, 'Frozen model/core/parent source file changed'
        result['gates']['base_core_frozen'] = True
        result['base_core_freeze_scope'] = 'immutable original model/core/parent disk stats plus worker trainable parameter contract; no whole-base GPU readback'
        assert all(result['gates'].values())
        result['complete'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if engine is not None:
            result['final_process_memory'] = engine.status()
            engine.close()
            if 'error' in result:
                result['worker_stderr'] = b''.join(tuple(engine._stderr))[-8192:].decode(errors='replace')
            result['final_process_closed'] = engine.process is None
        if lock is not None:
            lock.close()
            result['model_process_lock_released'] = True
        result['seconds'] = time.monotonic() - started
        save(output, result)


if __name__ == '__main__':
    main()
