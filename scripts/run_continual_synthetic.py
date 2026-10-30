#!/usr/bin/env python3
"""Prespecified three-seed fixed-width study. Never admits a product candidate."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from continual_synthetic_data import build_dataset, schedule
from measure_installed_app import Application
from verify_mlx_native_training import rpc, frozen_stat
from luban.generation_checks import evaluate_generation
from luban.native_engine import NativeEngine, _sha
from luban.native_mlx import model_binding
from luban.process_lock import ModelProcessLock
from verify_direct_q4 import IDENTITY, MANIFEST

ROOT = Path('/Volumes/移动数据库/lubanllm')
METADATA = ROOT / 'models/luban0-mlx-metadata-20261020'
BINARY = Path('/Applications/Luban.app/Contents/Resources/backend/native/luban-mlx-worker')
WORKER_SHA = '7f5054c11b496137dd303603397ae5e7d01404acdf66e0d6869b14b6066006d5'
FACTORY = ROOT / 'experiments/mlx-factory-engine-20261023-v3/factory'
SEEDS = (7, 42, 123)
ARMS = ('current_only', 'replay')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def matrix_metrics(matrix):
    """Keep unlearned tasks and signed changes; rows 0..3, columns A..C."""
    assert len(matrix) == 4 and all(len(row) == 3 for row in matrix)
    return {'final_average_accuracy': statistics.mean(matrix[3]),
            'immediate_acquisition': [matrix[j][j-1] for j in (1, 2, 3)],
            'backward_transfer': statistics.mean(matrix[3][j-1] - matrix[j][j-1] for j in (1, 2)),
            'forgetting': statistics.mean(max(matrix[k][j-1] for k in range(j, 3)) - matrix[3][j-1]
                                        for j in (1, 2)),
            'forward_transfer': statistics.mean(matrix[j-1][j-1] - matrix[0][j-1] for j in (2, 3))}


def wait_state(app, target, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = app.json('/api/status')['runtime']['state']
        if state == target:
            return
        if state == 'error':
            raise RuntimeError('Installed application entered error while restoring model')
        time.sleep(.5)
    raise TimeoutError('Installed application did not reach ' + target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    assert Path('/Volumes/移动数据库').is_mount() and directory.is_relative_to(ROOT / 'experiments')
    assert not directory.exists()
    directory.mkdir(mode=0o700)
    dataset = build_dataset()
    save(directory / 'dataset.json', dataset)
    # Separate files document which material the optimizer can consume.
    train, dev = dataset['groups']['TRAIN'], dataset['groups']['DEV']
    save(directory / 'training.json', train)
    save(directory / 'development.json', dev)
    save(directory / 'final-test.json', dataset['groups']['TEST'])
    save(directory / 'protection.json', dataset['protection'])
    plan = [{'seed': seed, 'arm': arm, 'stage': stage,
             'row_ids': [row['id'] for row in schedule(train, stage, arm, seed)]}
            for seed in SEEDS for arm in ARMS for stage in (1, 2, 3)]
    protocol = {'schema': 'continual-synthetic-three-seed-v1', 'registered_at': timestamp(),
        'seeds': list(SEEDS), 'arms': list(ARMS), 'width': 32, 'task_order': ['A', 'B', 'C'],
        'actual_adam_per_stage': 64, 'planned_adam_total': 1152,
        'learning_rate': .0002, 'loss': 'CE only; no preservation/teacher forward',
        'precision': 'MLX affine Q4/group64 base; BF16 accepted core/expert; FP32 Adam',
        'stage_lifecycle': 'fresh stage 1; saved parent for later stages; new candidate/Adam each stage, native 64-step cap unchanged',
        'evaluation': 'all 18 candidates frozen before any TEST; exact/finite checks and EOS required; temperature0 seed42 fresh request cache; task-time matrices include unlearned/future tasks',
        'dev': 'fixed observation at step32 and64, current task only; never changes budget or selects weights',
        'baseline': 'frozen accepted core is a zero-update anchor, not equal training compute',
        'compute_boundary': 'arms match actual Adam count and one differentiated forward per update; train token counts differ; replay halves current exposures; not equal FLOPs',
        'memory': 'native child footprint/lifetime peak snapshots from existing RPC; not full app peak or separate GPU allocator',
        'scope': 'forced single fixed-width expert mechanism; no growth/routing training/adoption; novel wording for seen synthetic facts',
        'teacher_api_calls': 0, 'product_adoptions': 0, 'training_plan': plan}
    save(directory / 'protocol.json', protocol)
    proof = json.loads((FACTORY / 'runtime/mlx/expert-proof.json').read_text())
    store = FACTORY / 'modules/luban0-v1'
    core = {'file': str(store / (proof['core_version'] + '.safetensors')),
            'tensor_key': 'model.language_model.layers.23.mlp.base.down_proj.weight',
            'sha256': proof['core_file_sha256']}
    model = model_binding(SimpleNamespace(native_inference_precision='mlx-q4',
        native_mlx_model_path=str(ROOT / 'models/community-q4-study-20261017/mlx-community-4bit')))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(METADATA), local_files_only=True, trust_remote_code=False)
    def tokens(row):
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
        return ids if isinstance(ids, list) else ids['input_ids']
    dev_inputs = {row['id']: tokens(row) for row in dev}
    sample_cache, prepared_inputs = {}, {}
    app = Application()
    before = app.json('/api/status')
    settings = app.json('/api/settings')
    assert before['runtime']['state'] == 'ready'
    assert settings['teacher']['key_configured'] is False
    assert settings['teacher']['answer_mode'] == 'local_first'
    assert before['runtime']['inference']['binary_sha256'] == WORKER_SHA
    original_settings = {'automatic': settings['automatic'],
                         'teacher': {'teaching_modalities': settings['teacher']['teaching_modalities']}}
    frozen_paths = [Path(core['file']), Path(model['path']) / 'model.safetensors']
    frozen_before = {str(path): frozen_stat(path) for path in frozen_paths}
    pointers = {name: (store / name).read_bytes() for name in ('active.json', 'active-core.json', 'active-router.json')}
    result = {'schema': protocol['schema'], 'complete': False, 'teacher_api_calls': 0,
        'adoptions': 0, 'started_at': timestamp(), 'protocol': protocol,
        'hardware': {'platform': platform.platform(),
          'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
          'model': subprocess.check_output(['sysctl', '-n', 'hw.model'], text=True).strip(),
          'ram_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True))},
        'worker_sha256': WORKER_SHA, 'core': core, 'model': model,
        'stages': [], 'evaluations': [], 'actual_adam_total': 0,
        'app_restored': False, 'settings_restored': False, 'test_opened': False}
    started = time.monotonic()
    engine, lease = None, None
    stopped_app = False
    def persist():
        result['seconds'] = time.monotonic() - started
        save(directory / 'result.json', result)
    def make_engine(expert=None):
        return NativeEngine(BINARY, METADATA, MANIFEST, core=core, expert=expert,
            precision='mlx-q4', mlx_model=model, request_timeout=600, cancel_timeout=60)
    def generate(owner, row, ids):
        at = time.monotonic()
        response = owner.generate(ids, row['check']['max_new_tokens'], [248044, 248046],
            0, lambda _: None, threading.Event(), seed=42, cache_reuse=False)
        answer = tokenizer.decode(response['tokens'], skip_special_tokens=True).strip()
        complete = response['finish_reason'] == 'stop' and bool(response['tokens']) and response['tokens'][-1] in (248044, 248046)
        scored = evaluate_generation(answer, row['check'], complete)
        return {'id': row['id'], 'task': row['task'], 'fact_id': row['fact_id'],
                'input_tokens': ids, 'tokens': response['tokens'], 'answer': answer,
                'complete': complete, 'finish_reason': response['finish_reason'],
                'passed': scored['passed'], 'seconds': time.monotonic() - at}
    try:
        app.json('/api/settings', {'automatic': {**settings['automatic'], 'learning_enabled': False},
                                  'teacher': {'teaching_modalities': []}}, 'PATCH')
        stopped_app = True
        app.json('/api/model/stop', {})
        wait_state(app, 'stopped', 30)
        lease = ModelProcessLock(IDENTITY, 'mps')
        persist()
        for seed in SEEDS:
            for arm in ARMS:
                parent = None
                for stage in (1, 2, 3):
                    path = directory / f'seed-{seed}' / arm / f'stage-{stage}'
                    path.mkdir(parents=True)
                    at = time.monotonic()
                    stage_result = {'seed': seed, 'arm': arm, 'stage': stage,
                        'events': [], 'complete': False, 'dev': [], 'parent': parent,
                        'training_token_count': 0, 'answer_target_count': 0}
                    result['stages'].append(stage_result)
                    engine = make_engine(parent)
                    stage_result['loaded'] = engine.start()
                    assert engine.binary_sha256 == WORKER_SHA
                    events = stage_result['events']
                    def command(op, **kw):
                        return rpc(engine, op, events, **kw)
                    if not sample_cache:
                        for row in train:
                            assert row['training_allowed'] and row['split'] == 'training'
                            prepared = command('prepare', messages=row['messages'], answer=row['answer'])
                            ids = tokens(row)
                            assert prepared['tokens'][:prepared['prompt_tokens']] == ids
                            assert prepared['eos'] == 248046 and len(prepared['tokens']) <= 256
                            sample_cache[row['id']] = {k: prepared[k] for k in ('tokens', 'labels')}
                            prepared_inputs[row['id']] = ids
                        save(directory / 'prepared-training.json', sample_cache)
                    sequence = schedule(train, stage, arm, seed)
                    probes = [sequence[0], sequence[-1]]
                    command('train_select', source='parent')
                    base_losses = [command('train_observe', sample=sample_cache[row['id']], source='parent')['loss']
                                   for row in probes]
                    original_generations = [generate(engine, row, prepared_inputs[row['id']]) for row in probes]
                    bind = f'continual-synthetic-20261024:{seed}:{arm}:{stage}'
                    begin = command('train_begin', seed=seed, width=32, learning_rate=.0002,
                        preservation_weight=5, initialize='fresh' if stage == 1 else 'parent', binding=bind)
                    assert begin['step'] == 0 and begin['base_frozen'] is True
                    initial_losses = [command('train_observe', sample=sample_cache[row['id']], source='candidate')['loss']
                                      for row in probes]
                    initial_generations = [generate(engine, row, prepared_inputs[row['id']]) for row in probes]
                    assert base_losses == initial_losses, 'Initial real-graph loss identity failed'
                    assert all(a['tokens'] == b['tokens'] and a['finish_reason'] == b['finish_reason']
                               for a, b in zip(original_generations, initial_generations)), 'Initial real-graph generation identity failed'
                    stage_result['initial_real_graph_identity'] = {'loss_exact': True, 'tokens_exact': True,
                        'probe_ids': [row['id'] for row in probes], 'losses': base_losses}
                    stage_result['schedule_counts'] = dict(Counter(row['task'] for row in sequence))
                    stage_result['schedule_ids'] = [row['id'] for row in sequence]
                    for step, row in enumerate(sequence, 1):
                        sample = sample_cache[row['id']]
                        value = command('train_step', sample=sample)
                        assert value['event'] == 'train_step' and value['step'] == step
                        assert value['teacher'] == 'none' and value['preservation_kl'] == 0
                        assert all(math.isfinite(value[key]) for key in ('loss', 'ce', 'gradient_norm'))
                        result['actual_adam_total'] += 1
                        stage_result['training_token_count'] += len(sample['tokens'])
                        stage_result['answer_target_count'] += sum(x != -100 for x in sample['labels'])
                        if step in (32, 64):
                            seen = [generate(engine, item, dev_inputs[item['id']]) for item in dev if item['task'] == 'ABC'[stage-1]]
                            stage_result['dev'].append({'step': step, 'rows': seen})
                            persist()
                    weights = path / 'expert.safetensors'
                    checkpoint = path / 'checkpoint.safetensors'
                    command('train_save', path=str(weights), checkpoint_path=str(checkpoint))
                    # A new saved-source binding is required for the next lifecycle.
                    parent = {'id': f'synthetic-{seed}-{arm}-{stage}', 'file': str(weights),
                        'sha256': _sha(weights), 'width': 32, 'core_sha256': core['sha256']}
                    stage_result.update(expert=parent, checkpoint=str(checkpoint),
                        actual_adam=64, differentiated_forward_calls=64, complete=True,
                        seconds=time.monotonic() - at, memory=engine.status())
                    command('train_end', restore_parent=True)
                    engine.close()
                    engine = None
                    persist()
                    print(json.dumps({'stage_complete': [seed, arm, stage], 'adam_total': result['actual_adam_total'],
                                      'seconds': stage_result['seconds']}, ensure_ascii=False), flush=True)
        assert result['actual_adam_total'] == 1152 and len(result['stages']) == 18
        from safetensors.torch import load_file
        import torch
        paired = []
        for seed in SEEDS:
            a, b = [next(row for row in result['stages'] if row['seed'] == seed and row['arm'] == arm and row['stage'] == 1)
                    for arm in ARMS]
            aw, bw = load_file(a['expert']['file']), load_file(b['expert']['file'])
            assert set(aw) == set(bw) and all(torch.equal(aw[key], bw[key]) for key in aw)
            paired.append({'seed': seed, 'stage1_tensor_exact': True})
        result['paired_stage1'] = paired
        manifest = {'sealed_at': timestamp(), 'complete': True,
                    'candidates': [{'seed': row['seed'], 'arm': row['arm'], 'stage': row['stage'],
                                    'expert': row['expert']} for row in result['stages']]}
        save(directory / 'frozen-candidates.json', manifest)
        result['all_candidates_frozen_at'] = manifest['sealed_at']
        # TEST labels and prompts are consumed by the evaluator only after the above
        # complete manifest exists. No subsequent training/selection is permitted.
        test = json.loads((directory / 'final-test.json').read_text())
        protection = json.loads((directory / 'protection.json').read_text())
        evaluation_rows = test + protection
        evaluation_inputs = {row['id']: tokens(row) for row in evaluation_rows}
        result.update(test_opened=True, test_opened_at=timestamp())
        persist()
        for descriptor in [{'seed': None, 'arm': 'frozen_core', 'stage': 0, 'expert': None}] + manifest['candidates']:
            engine = make_engine(descriptor['expert'])
            engine.start()
            events = []
            rpc(engine, 'train_select', events, source='parent')
            observations = [generate(engine, row, evaluation_inputs[row['id']]) for row in evaluation_rows]
            item = {**descriptor, 'rows': observations, 'task_accuracy': {
                task: sum(row['passed'] for row in observations if row['task'] == task) / 8 for task in 'ABC'},
                'protection_accuracy': sum(row['passed'] for row in observations if row['task'] == 'P') / 8,
                'native_memory': engine.status()}
            result['evaluations'].append(item)
            engine.close()
            engine = None
            persist()
            print(json.dumps({'evaluation': [item['seed'], item['arm'], item['stage']],
                              'task_accuracy': item['task_accuracy'], 'protection': item['protection_accuracy']}), flush=True)
        anchor = result['evaluations'][0]
        summaries = []
        for seed in SEEDS:
            for arm in ARMS:
                items = [next(x for x in result['evaluations'] if x['seed'] == seed and x['arm'] == arm and x['stage'] == stage)
                         for stage in (1, 2, 3)]
                matrix = [list(anchor['task_accuracy'].values())] + [[item['task_accuracy'][task] for task in 'ABC'] for item in items]
                summaries.append({'seed': seed, 'arm': arm, 'matrix': matrix,
                    'metrics': matrix_metrics(matrix),
                    'protection_by_stage': [anchor['protection_accuracy']] + [item['protection_accuracy'] for item in items],
                    'training_tokens': sum(row['training_token_count'] for row in result['stages'] if row['seed'] == seed and row['arm'] == arm)})
        result['summaries'] = summaries
        lease.close()
        lease = None
        app.json('/api/model/reload', {})
        wait_state(app, 'ready')
        result['app_restored'] = True
        stopped_app = False
        # This is the unchanged installed model, not one of the experimental experts.
        # Keep product behavior separate from forced-candidate mechanism matrices.
        product = []
        for row in test:
            with app.request('/api/chat', {'messages': row['messages'], 'route': 'local',
                    'thinking': 'off', 'temperature': 0, 'max_tokens': row['check']['max_new_tokens']}) as response:
                frames = [json.loads(line[6:]) for line in response if line.startswith(b'data: ')]
            assert not any(frame['type'] == 'error' for frame in frames)
            done = next(frame for frame in frames if frame['type'] == 'done')
            meta = next(frame for frame in frames if frame['type'] == 'meta')
            answer = ''.join(frame.get('content', '') for frame in frames if frame['type'] == 'delta').strip()
            events = app.json('/api/events')['events']
            observed = next(event['body'] for event in events if event['kind'] == 'decision' and event['body']['request_id'] == meta['request_id'])
            assert observed['usage']['reference_used'] is False and observed['decision']['route'] == 'local'
            product.append({'id': row['id'], 'task': row['task'], 'answer': answer,
                'passed': evaluate_generation(answer, row['check'], done['usage']['finish_reason'] == 'stop')['passed'],
                'decision': observed['decision'], 'model_context': observed['model_context'], 'usage': done['usage']})
        result['unchanged_installed_product'] = {'rows': product,
            'correct': sum(row['passed'] for row in product), 'total': len(product),
            'action_counts': dict(Counter(row['decision'].get('action') for row in product)),
            'experimental_candidates_installed': False}
        final = app.json('/api/status')
        assert final['teacher_calls'] == before['teacher_calls']
        assert final['runtime']['core_version'] == before['runtime']['core_version']
        assert final['runtime']['active_expert'] == before['runtime']['active_expert']
        result['complete'] = True
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        try:
            try:
                if engine is not None:
                    engine.close()
            finally:
                if lease is not None:
                    lease.close()
            result['frozen_sources_unchanged'] = all(frozen_stat(path) == state for path, state in frozen_before.items())
            result['factory_pointers_unchanged'] = all((store / name).read_bytes() == data for name, data in pointers.items())
            assert result['frozen_sources_unchanged'] and result['factory_pointers_unchanged']
        finally:
            try:
                try:
                    if stopped_app:
                        if app.json('/api/status')['runtime']['state'] != 'ready':
                            app.json('/api/model/reload', {})
                            wait_state(app, 'ready')
                        result['app_restored'] = True
                finally:
                    app.json('/api/settings', original_settings, 'PATCH')
                    restored = app.json('/api/settings')
                    assert restored['automatic'] == settings['automatic']
                    assert restored['teacher']['teaching_modalities'] == settings['teacher']['teaching_modalities']
                    result['settings_restored'] = True
            finally:
                result['finished_at'] = timestamp()
                persist()
    assert result['complete'] and result['app_restored'] and result['settings_restored']


if __name__ == '__main__':
    main()
