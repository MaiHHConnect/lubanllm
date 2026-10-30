#!/usr/bin/env python3
"""Adopt a bounded teacher repair by held-out net gain, under the app lease.

Keeps local-trial scope: exposed migration cases are not a new blind benchmark.
A remaining wrong answer is retained, never relabelled as mastered.
"""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

from prepare_mlx_expert_trial import ROOT, STORE, STUDY, save, expert_oracle
from recalibrate_mlx_trial_router import fit, validate_expert_math
from repair_mlx_teacher_expert import MODEL, PRIOR
from verify_direct_q4 import CORE, MANIFEST, IDENTITY
from verify_mlx_native_training import frozen_stat, rpc
from verify_mlx_trial_router import expert_math


def read(path):
    return json.loads(Path(path).read_text())


def net_progress(rows, baseline, candidate):
    """Re-score every observation; count distinct held-out contexts only."""
    from luban.generation_checks import evaluate_generation
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'Duplicate observation IDs'
    def checked(pair):
        assert len(pair) == 2 and pair[0] == pair[1], 'Non-deterministic closed-book outputs'
        assert [r['id'] for r in pair[0]] == ids, 'Incomplete observation coverage'
        correct = set()
        for material, observed in zip(rows, pair[0]):
            assert observed['input_tokens'] == material['input_tokens'], 'Changed input'
            assert observed['reference_used'] is False and isinstance(observed['answer'], str)
            check = material.get('generation_check') or {'type': 'exact', 'expected': material.get('expected', material.get('answer'))}
            scored = check.get('type') != 'exact' or check.get('expected') is not None
            passed = evaluate_generation(observed['answer'], check, observed['finish_reason'] == 'stop')['passed'] if scored else None
            assert observed['passed'] is passed, 'Forged score'
            if passed:
                correct.add(tuple(material['input_tokens']))
        return correct
    before, after = checked(baseline), checked(candidate)
    train = {tuple(r['input_tokens']) for r in rows if r.get('purpose') == 'training'}
    valid = {tuple(r['input_tokens']) for r in rows if r.get('purpose') == 'validation'}
    assert not train & valid, 'TRAIN/VALID context overlap'
    gained, forgotten = (after - before) & valid, before - after
    remaining = [r['id'] for r in rows if r.get('purpose') == 'validation' and tuple(r['input_tokens']) not in after]
    return {'policy': 'historical-repair-heldout-net-gain-v1', 'gained': len(gained),
            'forgotten': len(forgotten), 'net_gain': len(gained) - len(forgotten),
            'pending_validation_ids': remaining,
            'gained_ids': [r['id'] for r in rows if r.get('purpose') == 'validation' and tuple(r['input_tokens']) in gained],
            'passed': len(gained) > len(forgotten)}


def publish(output, old, artifacts, pointers):
    """Existing app lease excludes readers; compare sources, rollback on errors."""
    from luban.settings import private_json
    assert all(read(path) == value for path, value in old.items()), 'Active source changed'
    receipt = {'published': False, 'before': old, 'after': pointers, 'created': time.time()}
    save(output / 'publication.json', receipt)
    written = []
    try:
        for path, raw in artifacts.items():
            with Path(path).open('xb') as f:
                f.write(raw)
            written.append(path)
        assert all(read(path) == value for path, value in old.items()), 'Rival publication'
        for path, value in pointers.items():
            private_json(Path(path), value)
        assert all(read(path) == value for path, value in pointers.items())
        receipt['published'] = True
    except BaseException:
        for path, value in reversed(list(pointers.items())):
            if read(path) == value:
                private_json(Path(path), old[path])
        for path in written:
            Path(path).unlink(missing_ok=True)
        receipt['rolled_back'] = True
        raise
    finally:
        private_json(output / 'publication.json', receipt)


def main():
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    from luban.native_engine import NativeEngine
    from luban.native_mlx import model_binding
    from luban.native_mlx_routing import pooled_features
    from luban.process_lock import ModelProcessLock
    from luban.runtime import Runtime
    from luban.decision import FastDecisionHead
    from luban.generation_checks import evaluate_generation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--training-proof', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    assert Path('/Volumes/移动数据库').is_mount()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / 'experiments') and not output.exists()
    repair = read(args.repair)
    assert repair['complete'] and repair['sources_unchanged'] and not repair['published']
    assert repair['gates']['deterministic'] and repair['gates']['closed_book_gain']
    prepared = read(PRIOR / 'router-calibration/prepared.json')
    original = [r for group in prepared['splits'].values() for r in group if r['route_action'] == 'neural_expert']
    rows = repair['train'] + repair['validation']
    assert {r['id'] for r in rows} == {r['id'] for r in original}
    for r in rows:
        before = next(x for x in original if x['id'] == r['id'])
        assert all(r[k] == before[k] for k in ('purpose', 'prompt', 'answer', 'split', 'input_tokens', 'generation_check'))
    rows += read(PRIOR / 'router/cases.json')
    # Don't retroactively overwrite the strict failure report.
    historical = {'strict_gate_passed': repair['passed'], 'steps': repair['steps'],
                  'correct_before': sum(r['passed'] is True for r in repair['baseline'][:24]),
                  'correct_after': sum(r['passed'] is True for r in repair['candidate_first'][:24])}
    paths = [STORE / n for n in ('active.json', 'active-core.json', 'active-router.json')]
    paths += [ROOT / 'runtime/mlx/expert-proof.json', ROOT / 'runtime/mlx/training-proof.json']
    old = {str(p): read(p) for p in paths}
    active, core = old[str(paths[0])], old[str(paths[1])]
    parent_meta = read(STORE / 'layer-23' / (active['id'] + '.json'))
    parent = repair['expert']
    assert Path(parent['file']) == STORE / 'layer-23' / (active['id'] + '.safetensors')
    assert parent['sha256'] == parent_meta['sha256'] and core['sha256'] == CORE['sha256']
    candidate_raw = Path(repair['candidate_file']).read_bytes()
    candidate = {**parent, 'file': repair['candidate_file'], 'sha256': hashlib.sha256(candidate_raw).hexdigest()}
    runtime = SimpleNamespace(native_inference_precision='mlx-q4', native_mlx_model_path=str(STUDY / 'mlx-community-4bit'), device='cpu')
    model = model_binding(runtime)
    build = read(str(args.binary) + '.json')
    training = read(args.training_proof)
    keys = ('prepare_equivalent','real_adam_updates','checkpoint_replay_exact','cancellation_atomic','continued_after_cancel','saved_expert_reload_equal','base_core_frozen')
    assert training['complete'] and all(training['gates'].get(k) is True for k in keys)
    assert training['worker_binary_sha256'] == build['binary_sha256']
    assert training['core_file_sha256'] == CORE['sha256'] and training['mlx_model_sha256'] == model['sha256']
    assert model == repair['model'] and repair['core'] == CORE
    output.mkdir(mode=0o700)
    report = {'schema': 'mlx-teacher-net-repair-v1', 'complete': False, 'passed': False,
              'scope': 'user-authorized local migration; exposed historical cases, not blind generalization',
              'historical': historical, 'repair_report': str(args.repair), 'precision': repair['precision'],
              'hardware': training['hardware'], 'seed':42, 'teacher_calls':0, 'new_expert_adam_steps':0,
              'worker_binary_sha256': build['binary_sha256'], 'model': model, 'core': CORE,
              'parent':parent, 'candidate':candidate, 'cache_reuse':False, 'events':[]}
    save(output / 'protocol.json', report)
    stamps = {p: frozen_stat(p) for p in (parent['file'], candidate['file'], CORE['file'], str(Path(model['path']) / 'model.safetensors'))}
    engine, lock = None, ModelProcessLock(IDENTITY, 'mps')
    started = time.monotonic()
    try:
        expert_oracle(load_file(candidate['file']), output / 'expert-math-oracle.safetensors')
        expert_math(output, args.binary, report, expert=candidate)
        validate_expert_math(report['expert_math'], build['binary_sha256'], candidate)
        tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
        def start(binding):
            result = NativeEngine(args.binary, MODEL, MANIFEST, core=CORE, expert=binding, precision='mlx-q4', mlx_model=model, request_timeout=600)
            loaded = result.start()
            assert result.binary_sha256 == build['binary_sha256']
            return result, loaded
        engine, report['parent_loaded'] = start(parent)
        runtime.native_engine = engine
        features, targets = {}, {}
        cached = {}
        for split, group in prepared['splits'].items():
            values = []
            for row in group:
                vector = pooled_features(runtime, torch.tensor([row['input_tokens']])).reshape(1,2048)
                cached[tuple(row['input_tokens'])] = vector
                values.append(vector)
            features[split] = torch.cat(values)
            targets[split] = torch.tensor([FastDecisionHead.actions.index(row['route_action']) for row in group])
        head, calibrated = fit(features, targets, prepared['supported_actions'])
        assert calibrated['calibrated'], 'Q4 router calibration failed'
        runtime.router, runtime.router_meta = head, {**calibrated, 'feature_source':'mlx-q4-prefix'}
        runtime._ready = lambda: None
        runtime.external_hint = lambda _: None
        routes = {}
        for row in rows:
            key = tuple(row['input_tokens'])
            if key not in cached:
                cached[key] = pooled_features(runtime, torch.tensor([row['input_tokens']])).reshape(1,2048)
            runtime._tokens = lambda _messages, shallow=False, key=key: torch.tensor([key])
            runtime.shallow = lambda _tokens, key=key: cached[key].reshape(1,1,2048)
            routes[row['id']] = Runtime.choose_route(runtime, [])
        report['routes'] = routes
        def observe():
            pair = []
            for _ in range(2):
                observations = []
                current = None
                for row in rows:
                    source = 'parent' if routes[row['id']].get('action') == 'neural_expert' else 'core'
                    if current != source:
                        rpc(engine, 'train_select', report['events'], source=source)
                        current = source
                    value = engine.generate(row['input_tokens'], row.get('max_tokens',128), [248046,248044], 0, lambda _:None, threading.Event(), seed=42, cache_reuse=False)
                    answer = tokenizer.decode(value['tokens'], skip_special_tokens=True).strip()
                    check = row.get('generation_check') or {'type':'exact','expected':row.get('expected',row.get('answer'))}
                    scored = check.get('type') != 'exact' or check.get('expected') is not None
                    observations.append({'id':row['id'],'answer':answer,'input_tokens':row['input_tokens'],
                        'tokens':value['tokens'],'finish_reason':value['finish_reason'],'reference_used':False,
                        'route_action':routes[row['id']].get('action'),
                        'passed':evaluate_generation(answer,check,value['finish_reason']=='stop')['passed'] if scored else None})
                pair.append(observations)
                print(json.dumps({'pass':len(pair),'correct':sum(x['passed'] is True for x in observations)},ensure_ascii=False),flush=True)
            return pair
        report['baseline'] = observe()
        engine.close()
        engine, report['candidate_loaded'] = start(candidate)
        report['candidate_observations'] = observe()
        report['progress'] = net_progress(rows, report['baseline'], report['candidate_observations'])
        report['pending_records'] = [r for r in rows if r['id'] in report['progress']['pending_validation_ids']]
        report['memory'] = engine.status()
        report.update(complete=True, passed=report['progress']['passed'])
    finally:
        if engine: engine.close()
        lock.close()
        report['sources_unchanged'] = all(frozen_stat(p)==s for p,s in stamps.items()) and all(read(p)==v for p,v in old.items())
        report['seconds'] = time.monotonic()-started
        save(output / 'result.json', report)
    assert report['sources_unchanged'] and report['passed'], 'Net gain or source check failed; no publication'
    uid = uuid.uuid4().hex[:12]
    expert_id, router_id = 'trial-mlx-'+uid, 'router-'+uid
    flags = {'execution_scope':'local-trial','quality_admitted':False,'trial_engine':'mlx-q4','training_admitted':False}
    weights_path = output / (router_id+'.safetensors')
    save_file({k:v.detach().contiguous() for k,v in head.state_dict().items()}, str(weights_path))
    router_raw = weights_path.read_bytes()
    router_sha = hashlib.sha256(router_raw).hexdigest()
    meta = {**parent_meta, **flags, 'id':expert_id,'sha256':candidate['sha256'], 'migration':{
        'scope':'local-trial','source_id':active['id'],'source_sha256':parent['sha256'],
        'repair_report':str(args.repair),'admission_report':str(output/'result.json'),
        'policy':report['progress']['policy'],'pending_records':report['pending_records']}}
    router = {**calibrated, **flags, 'id':router_id,'sha256':router_sha,'identity':IDENTITY,'depth':4,'seed':42,
              'core_version':core['version'],'expert_id':expert_id,'expert_sha256':candidate['sha256'],
              'feature_source':'mlx-q4-prefix','gates_expert':True,
              'split_ids':{k:[r['id'] for r in v] for k,v in prepared['splits'].items()},
              'expert_route_proof':{'required':True,'passed':True,'policy':report['progress']['policy'],
                                    'all_answers_correct':False,'progress':report['progress'],'report':str(output/'result.json')}}
    proof = {'schema':'mlx-expert-local-trial-v1','scope':'local-trial','quality_admitted':False,
             'training_admitted':False,'math_passed':True,'runtime_checked':True,'route_checked':True,
             'identity':IDENTITY,'expert_id':expert_id,'width':candidate['width'],'core_version':core['version'],
             'core_file_sha256':CORE['sha256'],'expert_file_sha256':candidate['sha256'],
             'worker_binary_sha256':build['binary_sha256'],'mlx_model_sha256':model['sha256'],
             'router_id':router_id,'router_sha256':router_sha,'feature_source':'mlx-q4-prefix',
             'route_report':str(output/'result.json'),'runtime_report':str(output/'result.json'),
             'admission_policy':report['progress']['policy'],'net_progress':report['progress'],
             'closed_book_passed':False,'net_gain_admitted':True,'teacher_calls':0,'optimizer_updates':repair['steps']}
    def encoded(v): return (json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
    artifacts = {str(STORE/'layer-23'/(expert_id+'.safetensors')):candidate_raw,
                 str(STORE/'layer-23'/(expert_id+'.json')):encoded(meta),
                 str(STORE/(router_id+'.safetensors')):router_raw,str(STORE/(router_id+'.json')):encoded(router)}
    pointers = {str(paths[0]):{'id':expert_id,'width':candidate['width'],'identity':IDENTITY,'core_version':core['version'],**flags},
                str(paths[2]):{'id':router_id,'identity':IDENTITY,'core_version':core['version'],'sha256':router_sha,
                              'metadata_sha256':hashlib.sha256(encoded(router)).hexdigest()},
                str(paths[3]):proof,str(paths[4]):training}
    save(output/'candidate-publication.json',{'pointers':pointers,'expert_metadata':meta,'router_metadata':router})
    if args.publish: publish(output,old,artifacts,pointers)
    print(json.dumps({'progress':report['progress'],'published':args.publish},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
