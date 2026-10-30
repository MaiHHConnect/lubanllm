#!/usr/bin/env python3
"""Refit the fixed historical route split on actual Q4 prefix features.

No original BF16 weights. Preparation is CPU-only; execution requires the
caller to hold the app lease. Publication is explicit and rolls back pointers.
This grants only the existing bounded local trial, never training admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from types import SimpleNamespace
import uuid

from prepare_mlx_expert_trial import ROOT, STORE, STUDY, save
from verify_mlx_trial_router import SOURCE_ROUTER, SOURCE_DB


def read(path):
    return json.loads(path.read_text())


def prepare(args):
    from transformers import AutoTokenizer
    from luban.calibration import SPLITS, split_route_samples, validate_route_splits
    from luban.decision import FastDecisionHead
    source = read(args.store / (args.source_router + '.json'))
    if (source.get('depth') != 4 or source.get('calibrated') is not True
            or source.get('feature_source', 'hf-bf16-prefix') != 'hf-bf16-prefix'
            or not isinstance(source.get('expert_route_proof'), dict)):
        raise ValueError('Source router is not the pinned calibrated BF16 four-layer head')
    ids = [name for split in SPLITS for name in source['split_ids'][split]]
    if len(ids) != len(set(ids)):
        raise ValueError('Historical route split contains duplicate IDs')
    with sqlite3.connect(args.database.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        records = [dict(row) for row in db.execute(
            'SELECT * FROM samples WHERE id IN (' + ','.join('?' * len(ids)) + ')', ids)]
    splits = split_route_samples(records, FastDecisionHead.actions)
    supported = validate_route_splits(splits, FastDecisionHead.actions)
    for split in SPLITS:
        if set(source['split_ids'][split]) != {row['id'] for row in splits[split]}:
            raise ValueError('Frozen historical split differs: ' + split)
    tokenizer = AutoTokenizer.from_pretrained(str(args.metadata), local_files_only=True, trust_remote_code=False)
    for rows in splits.values():
        for row in rows:
            provenance = row['provenance']
            provenance = json.loads(provenance) if isinstance(provenance, str) else provenance
            messages = provenance.get('messages') or [{'role': 'user', 'content': row['prompt']}]
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=True, enable_thinking=False)
            row['input_tokens'] = tokens if isinstance(tokens, list) else tokens['input_ids']
            row['generation_check'] = provenance.get('generation_check') or {'type': 'exact', 'expected': row['answer']}
    old_expert_ids = set(source['expert_route_proof']['ids'])
    actual_expert_ids = {row['id'] for rows in splits.values() for row in rows if row['route_action'] == 'neural_expert'}
    if old_expert_ids != actual_expert_ids:
        raise ValueError('Closed-book expert case set differs from historical proof')
    args.output.mkdir()
    prepared = {'schema': 'mlx-prefix-route-preparation-v1', 'splits': splits, 'supported_actions': supported,
                'source_router_id': args.source_router, 'source_router_sha256': source['sha256'],
                'database': str(args.database), 'metadata': str(args.metadata), 'store': str(args.store),
                'old_pointers': {name: read(args.store / name) for name in
                                 ('active.json', 'active-core.json', 'active-router.json')}}
    save(args.output / 'prepared.json', prepared)
    return prepared


def fit(features, targets, supported):
    import torch
    from luban.decision import FastDecisionHead
    from luban.calibration import SPLITS, evaluate_router, calibrate_expert_scope
    torch.manual_seed(42)
    head = FastDecisionHead(2048).float()
    head.projection.reset_parameters()
    optimizer = torch.optim.Adam(head.parameters(), lr=.005)
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(head.projection(features['train']), targets['train'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite router loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
        optimizer.step()
    head.eval()
    with torch.no_grad():
        logits = {split: head.projection(features[split]) for split in SPLITS}
    result = evaluate_router(logits, targets, head.actions, supported)
    scope = calibrate_expert_scope(features, targets, head.actions, supported)
    result.update(expert_scope=scope, calibrated=result['calibrated'] and scope['passed'])
    return head, result


def validate_math(report, worker_sha, expert):
    if report.get('complete') is not True or report.get('model_process_lock_released') is not True:
        raise ValueError('Math/runtime report incomplete')
    validate_expert_math(report['expert_math'], worker_sha, expert)


def validate_expert_math(math, worker_sha, expert):
    event = math['event']
    stages = event['stages']
    if (math.get('passed') is not True or math.get('exit_code') != 0
            or math.get('worker_binary_sha256') != worker_sha
            or math.get('bf16_relative_l2_limit') != .03 or math.get('bf16_numerical_pass') is not True
            or event.get('expert_file_sha256') != expert['sha256'] or event.get('width') != expert['width']
            or event.get('fp32_pass') is not True or event.get('bf16_pass_assumed') is not False
            or len(stages) != 8 or {(x['dtype'], x['stage']) for x in stages} != {
                (dtype, stage) for dtype in ('fp32', 'bf16') for stage in ('gate', 'up', 'activation', 'output')}
            or any(x.get('finite') is not True or x.get('nonzero') is not True for x in stages)
            or any(not 0 <= x['relative_l2'] <= .03 for x in stages if x['dtype'] == 'bf16')):
        raise ValueError('Current worker/expert math proof failed')


def execute(args, prepared):
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from luban.calibration import SPLITS
    from luban.decision import FastDecisionHead
    from luban.generation_checks import evaluate_generation
    from luban.native_engine import NativeEngine
    from luban.native_mlx import model_binding
    from luban.native_mlx_routing import feature_source, pooled_features
    from luban.process_lock import ModelProcessLock
    old = prepared['old_pointers']
    core_pointer, expert_pointer = old['active-core.json'], old['active.json']
    core_meta = read(args.store / (core_pointer['version'] + '.json'))
    expert_id = expert_pointer['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', expert_id):
        raise ValueError('Invalid expert ID')
    expert_meta = read(args.store / 'layer-23' / (expert_id + '.json'))
    identity = core_pointer['identity']
    if (expert_meta.get('identity') != identity + ':layer-23'
            or expert_meta.get('core_version') != core_pointer['version']
            or expert_meta.get('execution_scope') != 'local-trial'):
        raise ValueError('Expected an existing local-trial expert bound to the current core')
    core = {'file': str(args.store / (core_pointer['version'] + '.safetensors')),
            'sha256': core_pointer['sha256'], 'tensor_key': core_meta['parameter']}
    expert = {'file': str(args.store / 'layer-23' / (expert_id + '.safetensors')),
              'sha256': expert_meta['sha256'], 'width': expert_meta['width'], 'core_sha256': core['sha256']}
    runtime = SimpleNamespace(native_inference_precision='mlx-q4', native_mlx_model_path=str(args.mlx_model), device='cpu')
    model = model_binding(runtime)
    manifest = hashlib.sha256((args.metadata / 'upstream-file-manifest.json').read_bytes()).hexdigest()
    engine = NativeEngine(args.binary, args.metadata, manifest, core=core, expert=expert,
                          precision='mlx-q4', mlx_model=model)
    runtime.native_engine = engine
    # Bind the worker and all small model/core artifacts before taking the
    # process lease.  This reads only metadata plus the worker receipt; the
    # Q4 worker is started below while the lease is held.
    engine._verify()
    validate_math(read(args.runtime_report), engine.binary_sha256, expert)
    tokenizer = Tokenizer.from_file(str(args.metadata / 'tokenizer.json'))
    result = {'schema': 'mlx-prefix-route-calibration-v1', 'complete': False, 'passed': False,
              'scope': 'local-trial', 'quality_admitted': False, 'training_admitted': False,
              'feature_source': feature_source(runtime), 'optimizer_updates': 160, 'teacher_calls': 0,
              'seed': 42, 'head_precision': 'torch.float32 CPU', 'backbone_precision': 'MLX Q4',
              'hardware': __import__('platform').platform(), 'cache_reuse': False,
              'source_router_id': prepared['source_router_id'], 'core_binding': core,
              'expert_binding': expert, 'model_binding': model, 'worker_binary_sha256': engine.binary_sha256,
              'runtime_report': str(args.runtime_report), 'closed_book': []}
    started = time.monotonic()
    lock = ModelProcessLock(identity, 'mps')
    try:
        result['loaded'] = engine.start()
        features, targets = {}, {}
        for split in SPLITS:
            rows = prepared['splits'][split]
            features[split] = torch.cat([pooled_features(runtime, torch.tensor([row['input_tokens']])).reshape(1, 2048)
                                         for row in rows])
            targets[split] = torch.tensor([FastDecisionHead.actions.index(row['route_action']) for row in rows])
            print(json.dumps({'feature_split': split, 'rows': len(rows)}), flush=True)
        head, calibrated = fit(features, targets, prepared['supported_actions'])
        result['calibration'] = calibrated
        for split in SPLITS:
            for row in prepared['splits'][split]:
                if row['route_action'] != 'neural_expert':
                    continue
                response = engine.generate(row['input_tokens'], row['generation_check'].get('max_new_tokens', 192),
                    [248046, 248044], 0, lambda _: None, threading.Event(), seed=42, cache_reuse=False)
                text = tokenizer.decode(response['tokens'], skip_special_tokens=True).strip()
                check = evaluate_generation(text, row['generation_check'], response['finish_reason'] == 'stop')
                result['closed_book'].append({'id': row['id'], 'text': text, 'response': response, 'check': check})
                print(json.dumps({'closed_book': row['id'], 'check': check}, ensure_ascii=False), flush=True)
        passed = bool(calibrated['calibrated'] and result['closed_book']
                      and all(row['check']['passed'] is True for row in result['closed_book']))
        result.update(complete=True, passed=passed)
        if passed:
            version = 'router-' + uuid.uuid4().hex[:12]
            weights = args.output / (version + '.safetensors')
            save_file({key: tensor.detach().contiguous() for key, tensor in head.state_dict().items()}, str(weights))
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            meta = {**calibrated, 'id': version, 'sha256': digest, 'identity': identity,
                    'depth': 4, 'seed': 42, 'feature_source': feature_source(runtime),
                    'core_version': core_pointer['version'], 'expert_id': expert_id, 'expert_sha256': expert['sha256'],
                    'source_router_id': prepared['source_router_id'],
                    'source_router_sha256': prepared['source_router_sha256'],
                    'gates_expert': True, 'execution_scope': 'local-trial', 'quality_admitted': False,
                    'trial_engine': 'mlx-q4', 'training_admitted': False,
                    'split_ids': {split: [row['id'] for row in prepared['splits'][split]] for split in SPLITS},
                    'expert_route_proof': {'required': True, 'passed': True, 'count': len(result['closed_book']),
                                           'ids': [row['id'] for row in result['closed_book']]}}
            proof = {'schema': 'mlx-expert-local-trial-v1', 'scope': 'local-trial', 'quality_admitted': False,
                     'math_passed': True, 'runtime_checked': True, 'route_checked': True,
                     'identity': identity, 'expert_id': expert_id, 'core_version': core_pointer['version'],
                     'core_file_sha256': core['sha256'], 'expert_file_sha256': expert['sha256'],
                     'worker_binary_sha256': engine.binary_sha256, 'mlx_model_sha256': model['sha256'],
                     'width': expert['width'], 'router_id': version, 'router_sha256': digest,
                     'feature_source': feature_source(runtime), 'route_report': str(args.output / 'result.json'),
                     'runtime_report': str(args.runtime_report), 'training_admitted': False,
                     'source_router_id': prepared['source_router_id'],
                     'source_router_sha256': prepared['source_router_sha256'],
                     'optimizer_updates': 160, 'teacher_calls': 0,
                     'closed_book_case_ids': [row['id'] for row in result['closed_book']],
                     'closed_book_passed': all(row['check']['passed'] is True for row in result['closed_book'])}
            save(args.output / (version + '.json'), meta)
            save(args.output / 'candidate-proof.json', proof)
            result['router_id'] = version
    finally:
        engine.close()
        lock.close()
        result.update(model_process_lock_released=True, seconds=time.monotonic() - started)
        save(args.output / 'result.json', result)
    return result


def publish(args, prepared, result):
    from luban.settings import private_json
    if result.get('complete') is not True or result.get('passed') is not True:
        raise ValueError('Cannot publish a failed calibration')
    version = result['router_id']
    if not re.fullmatch(r'router-[a-f0-9]{12}', version):
        raise ValueError('Invalid candidate router ID')
    old = prepared['old_pointers']
    if any(read(args.store / name) != value for name, value in old.items()):
        raise ValueError('Active source changed since preparation')
    meta = read(args.output / (version + '.json'))
    proof = read(args.output / 'candidate-proof.json')
    weight_bytes = (args.output / (version + '.safetensors')).read_bytes()
    digest = hashlib.sha256(weight_bytes).hexdigest()
    if (meta.get('sha256') != digest or proof.get('router_sha256') != digest
            or meta.get('calibrated') is not True or meta.get('feature_source') != 'mlx-q4-prefix'
            or proof.get('router_id') != version or proof.get('quality_admitted') is not False):
        raise ValueError('Candidate evidence differs')
    proof_path = args.store.parent.parent / 'runtime/mlx/expert-proof.json'
    previous_proof = read(proof_path)
    router_path = args.store / 'active-router.json'
    pointer = {'id': version, 'identity': meta['identity'], 'core_version': meta['core_version'], 'sha256': digest}
    receipt = {'published': False, 'old_router': old['active-router.json'], 'old_proof': previous_proof,
               'new_router_id': version, 'expert_pointer_unchanged': True}
    changed = False
    copied = []
    try:
        for suffix in ('.safetensors', '.json'):
            target = args.store / (version + suffix)
            with target.open('xb') as handle:
                handle.write((args.output / (version + suffix)).read_bytes())
            copied.append(target)
        pointer['metadata_sha256'] = hashlib.sha256((args.store / (version + '.json')).read_bytes()).hexdigest()
        if any(read(args.store / name) != value for name, value in old.items()) or read(proof_path) != previous_proof:
            raise ValueError('Active source changed before publication')
        changed = True
        private_json(proof_path, proof)
        private_json(router_path, pointer)
        receipt['published'] = True
    except BaseException:
        if changed:
            private_json(proof_path, previous_proof)
            private_json(router_path, old['active-router.json'])
        for target in copied:
            target.unlink(missing_ok=True)
        raise
    finally:
        save(args.output / 'publication.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--runtime-report', type=Path)
    parser.add_argument('--store', type=Path, default=STORE)
    parser.add_argument('--database', type=Path, default=SOURCE_DB)
    parser.add_argument('--source-router', default=SOURCE_ROUTER)
    parser.add_argument('--mlx-model', type=Path, default=STUDY / 'mlx-community-4bit')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    for name in ('metadata', 'output', 'store', 'database', 'mlx_model'):
        if not getattr(args, name).is_absolute():
            parser.error('--' + name + ' must be absolute')
    if args.prepare_only and args.publish:
        parser.error('Preparation cannot publish')
    if not args.prepare_only and (not args.binary or not args.runtime_report):
        parser.error('Execution requires --binary and --runtime-report')
    prepared = read(args.output / 'prepared.json') if args.output.exists() else prepare(args)
    if any(prepared[name] != str(getattr(args, name)) for name in ('metadata', 'store', 'database')):
        parser.error('Prepared sources differ')
    if args.prepare_only:
        print(json.dumps({'prepared': sum(map(len, prepared['splits'].values())), 'gpu_loaded': False}))
        return
    if (args.output / 'result.json').exists():
        parser.error('Result exists; use a new output directory')
    result = execute(args, prepared)
    if args.publish:
        publish(args, prepared, result)
    if not result['passed']:
        raise SystemExit('Calibration/closed-book gate failed; no proof or pointers issued')


if __name__ == '__main__':
    main()
