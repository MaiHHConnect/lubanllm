#!/usr/bin/env python3
"""Bounded teacher-guided repair of the existing Q4 expert, under an app lease.

Historical TRAIN only; DEVELOPMENT monitors progress, remaining splits are
reported once at the end. No live pointers or qualification files are changed.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import sqlite3
import threading
import time
from types import SimpleNamespace

from prepare_mlx_expert_trial import ROOT, STORE, STUDY, save
from verify_direct_q4 import CORE, IDENTITY, MANIFEST
from verify_mlx_native_training import frozen_stat, rpc

PRIOR = ROOT / 'experiments/mlx-without-original-20261021'
MODEL = ROOT / 'models/luban0-mlx-metadata-20261020'
TEACHER_IDS = ('39489a31c218417b83e460957be3b341', '98111cd5898c4d41b5fbddd7efba6add')


def select_material(prepared, teacher_records):
    """Teacher archive supplies targets, never its answer-bearing system prompt."""
    train = [copy.deepcopy(r) for r in prepared['splits']['train'] if r['route_action'] == 'neural_expert']
    validation = [copy.deepcopy(r) for s, rows in prepared['splits'].items() if s != 'train'
                  for r in rows if r['route_action'] == 'neural_expert']
    assert train and validation
    assert all(r['purpose'] == 'training' and r['approved'] and r['training_allowed'] for r in train)
    assert all(r['purpose'] == 'validation' for r in validation)
    assert not {r['prompt'] for r in train} & {r['prompt'] for r in validation}
    by_prompt = {r['prompt']: r for r in train}
    for teacher in teacher_records:
        row = by_prompt.get(teacher['prompt'])
        assert row is not None, 'Held-out question cannot become TRAIN through teacher capture'
        assert teacher['answer'] == row['answer'], 'Teacher target conflicts with the authorized historical target'
        row['teacher_material_id'] = teacher['id']
    assert len(teacher_records) == 2 and len({r['prompt'] for r in teacher_records}) == 2
    for row in train + validation:
        provenance = json.loads(row['provenance']) if isinstance(row['provenance'], str) else row['provenance']
        messages = provenance.get('messages') or [{'role': 'user', 'content': row['prompt']}]
        assert messages == [{'role': 'user', 'content': row['prompt']}], 'Repair requires bare original questions'
        row['messages'] = messages
    return train, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert Path('/Volumes/移动数据库').is_mount() and output.is_relative_to(ROOT / 'experiments')
    assert not output.exists()
    prepared = json.loads((PRIOR / 'router-calibration/prepared.json').read_text())
    with sqlite3.connect((ROOT / 'app/memory.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        teachers = []
        for sample_id in TEACHER_IDS:
            # Deleted hinted samples remain as auditable raw teacher archives.
            rows = db.execute('SELECT * FROM teacher_materials WHERE answer=? AND prompt=? ORDER BY created',
                (prepared['splits']['train'][0]['answer'],
                 '你属于哪个模型系列？' if sample_id == TEACHER_IDS[0] else '请说明麦好火与 luban0 的关系。')).fetchall()
            assert rows
            teachers.append(dict(rows[-1]))
    train, valid = select_material(prepared, teachers)
    extra = json.loads((PRIOR / 'router/cases.json').read_text())
    active = json.loads((STORE / 'active.json').read_text())
    meta = json.loads((STORE / 'layer-23' / (active['id'] + '.json')).read_text())
    assert active['id'] == 'trial-mlx-8ec555e81b20' and meta['width'] == 64
    expert = {'file': str(STORE / 'layer-23' / (active['id'] + '.safetensors')),
              'sha256': meta['sha256'], 'width': meta['width'], 'core_sha256': CORE['sha256']}
    qualification = json.loads((PRIOR / 'training/result.json').read_text())
    build = json.loads(Path(str(args.binary) + '.json').read_text())
    assert qualification['complete'] and all(qualification['gates'].values())
    assert qualification['worker_binary_sha256'] == build['binary_sha256']
    assert qualification['core_file_sha256'] == CORE['sha256']
    from luban.native_engine import NativeEngine
    from luban.native_mlx import model_binding
    from luban.process_lock import ModelProcessLock
    from luban.generation_checks import evaluate_generation
    from tokenizers import Tokenizer
    model = model_binding(SimpleNamespace(native_inference_precision='mlx-q4',
        native_mlx_model_path=str(STUDY / 'mlx-community-4bit'), _native_mlx_binding=None))
    assert model['sha256'] == qualification['mlx_model_sha256']
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    output.mkdir()
    report = {'schema': 'mlx-teacher-expert-repair-v1', 'complete': False, 'passed': False,
        'published': False, 'teacher_calls': 0, 'teacher_archive_ids': [r['id'] for r in teachers],
        'scope': 'exposed historical migration cases; not a blind/generalization benchmark',
        'seed': 42, 'precision': 'MLX Q4 backbone, BF16 core/expert, FP32 Adam',
        'hardware': platform.platform(), 'cache_reuse': False, 'file_cache': 'warm/not purged',
        'memory_method': 'Darwin proc_pid_rusage v4 child footprint/peak',
        'train': train, 'validation': valid, 'events': [], 'observations': [], 'steps': 0,
        'worker_binary_sha256': build['binary_sha256'], 'core': CORE, 'expert': expert, 'model': model}
    save(output / 'protocol.json', {k: v for k, v in report.items() if k != 'events'})
    pointers = {name: (STORE / name).read_bytes() for name in ('active.json', 'active-core.json', 'active-router.json')}
    frozen = {p: frozen_stat(p) for p in [expert['file'], CORE['file'], str(Path(model['path']) / 'model.safetensors')]}
    engine = None
    lock = None
    started = time.monotonic()
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        engine = NativeEngine(args.binary, MODEL, MANIFEST, core=CORE, expert=expert,
                              precision='mlx-q4', mlx_model=model, request_timeout=600)
        report['loaded'] = engine.start()
        assert engine.binary_sha256 == qualification['worker_binary_sha256']
        events = report['events']
        def command(op, **kw):
            return rpc(engine, op, events, **kw)
        def observe(rows, source):
            command('train_select', source=source)
            observed = []
            for row in rows:
                result = engine.generate(row['input_tokens'], row.get('max_tokens', 128), [248044, 248046],
                    0, lambda _: None, threading.Event(), seed=42, cache_reuse=False)
                answer = tokenizer.decode(result['tokens'], skip_special_tokens=True).strip()
                check = row.get('generation_check') or {'type': 'exact', 'expected': row.get('expected', row.get('answer'))}
                scored = check.get('type') != 'exact' or check.get('expected') is not None
                record = {'id': row['id'], 'answer': answer, 'tokens': result['tokens'],
                    'finish_reason': result['finish_reason'],
                    'passed': evaluate_generation(answer, check, result['finish_reason'] == 'stop')['passed'] if scored else None}
                observed.append(record)
            print(json.dumps({'source': source, 'correct': sum(r['passed'] is True for r in observed), 'total': len(observed)}), flush=True)
            return observed
        all_rows = train + valid + extra
        report['baseline'] = observe(all_rows, 'parent')
        samples = {}
        for row in train:
            value = command('prepare', messages=row['messages'], answer=row['answer'])
            assert value['tokens'][:value['prompt_tokens']] == row['input_tokens'] and value['eos'] == 248046
            assert len(value['tokens']) <= 512
            samples[row['id']] = {k: value[k] for k in ('tokens', 'labels')}
        correct = {r['id'] for r in report['baseline'] if r['passed']}
        preservation = [r for r in train if r['id'] in correct]
        assert preservation
        dev = [r for r in valid if r['split'] == 'development']
        command('train_begin', seed=42, width=64, learning_rate=.0002, preservation_weight=5,
                initialize='parent', binding='mlx-teacher-expert-repair-v1')
        for step in range(1, 65):
            row, old = train[(step - 1) % len(train)], preservation[(step - 1) % len(preservation)]
            event = command('train_step', sample=samples[row['id']], preservation=samples[old['id']], teacher='parent')
            assert event['event'] == 'train_step' and event['step'] == step
            report['steps'] = step
            if step % 8 == 0:
                weights = output / f'step-{step}.safetensors'
                command('train_save', path=str(weights), checkpoint_path=str(output / f'step-{step}.checkpoint.safetensors'))
                seen = observe(train + dev, 'candidate')
                report['observations'].append({'step': step, 'rows': seen})
        first, second = observe(all_rows, 'candidate'), observe(all_rows, 'candidate')
        report.update(candidate_first=first, candidate_second=second, candidate_file=str(weights))
        retained = all(r['passed'] for r in first if r['id'] in correct)
        report['gates'] = {'deterministic': first == second,
            'all_route_expert_correct': all(r['passed'] for r in first[:len(train) + len(valid)]),
            'previous_correct_retained': retained,
            'closed_book_gain': sum(r['passed'] is True for r in first) > sum(r['passed'] is True for r in report['baseline'])}
        report.update(complete=True, passed=all(report['gates'].values()))
        command('train_end', restore_parent=True)
        report['memory'] = engine.status()
    finally:
        if engine:
            engine.close()
        if lock:
            lock.close()
        report['sources_unchanged'] = (all((STORE / n).read_bytes() == v for n, v in pointers.items())
            and all(frozen_stat(p) == v for p, v in frozen.items()))
        report['seconds'] = time.monotonic() - started
        save(output / 'result.json', report)
    assert report['sources_unchanged']
    print(json.dumps({'passed': report['passed'], 'steps': report['steps'], 'seconds': report['seconds'],
                      'published': False, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
