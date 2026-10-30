#!/usr/bin/env python3
"""Qualify one unchanged factory source against an exact new packaged MLX worker.

Read-only accepted history. Finite exposed regressions, not new knowledge or a
blind benchmark. Raw evidence stays on the research volume; product receipt has
only immutable source binding and verified summaries.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from prepare_mlx_expert_trial import ROOT, expert_oracle, save
from verify_direct_q4 import IDENTITY, MANIFEST
from verify_mlx_native_training import frozen_stat
from verify_mlx_trial_router import expert_math

METADATA = ROOT / 'models/luban0-mlx-metadata-20261020'
MODEL = ROOT / 'models/community-q4-study-20261017/mlx-community-4bit'
SEED = Path('/Volumes/LubanRelease/complete-218/Luban.app/Contents/Resources/foundation/Luban-Factory-Expert.zip')


def read(path):
    return json.loads(Path(path).read_text())


def messages(row):
    provenance = row['provenance']
    provenance = json.loads(provenance) if isinstance(provenance, str) else provenance
    return provenance.get('messages') or [{'role': 'user', 'content': row['prompt']}]


def prepare(directory):
    directory.mkdir(mode=0o700)
    seed = directory / 'factory'
    seed.mkdir()
    with zipfile.ZipFile(SEED) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            assert parts[0] == 'Luban-Factory-Expert' and '..' not in parts and not item.is_dir()
            target = seed.joinpath(*parts[1:]); target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(archive.read(item))
    return seed


def quality(binary, seed, rows, expert, core, model, tokenizer, result):
    import torch
    from safetensors.torch import load_file
    from luban.native_engine import NativeEngine
    from luban.runtime import Runtime
    from luban.decision import FastDecisionHead
    from luban.generation_checks import evaluate_generation, validate_generation_check
    store = seed / 'modules/luban0-v1'
    pointer = read(store / 'active-router.json')
    router = read(store / (pointer['id'] + '.json'))
    observations, routes = [], []
    for repeat in range(2):
        current = {}; route_pass = {}
        for enabled in (True, False):
            engine = NativeEngine(binary, METADATA, MANIFEST, core=core, expert=expert if enabled else None,
                precision='mlx-q4', mlx_model=model, request_timeout=600)
            try:
                loaded = engine.start()
                assert loaded['mlx_training'] is False
                assert loaded['execution_exit_after'] == 24 and loaded['execution_skip_mask'] == 0
                assert loaded['layer_paging_enabled'] is False
                if enabled:
                    runtime = Runtime.__new__(Runtime)
                    runtime.native_inference_precision = 'mlx-q4'; runtime.device = 'cpu'
                    runtime.router_meta = router; runtime._ready = lambda: None
                    runtime.external_hint = lambda _: None
                    runtime.branch = SimpleNamespace(expert_id=expert['id'], _paths=lambda _: (
                        Path(expert['file']), Path(expert['file']).with_suffix('.json')))
                    runtime.router = FastDecisionHead(2048).float().eval()
                    runtime.router.load_state_dict(load_file(str(store / (pointer['id'] + '.safetensors'))))
                    runtime.shallow = lambda tokens: torch.tensor(engine.route_features(tokens[0].tolist(), threading.Event()), dtype=torch.float32).reshape(1, 1, 2048)
                    for row in rows:
                        ids = row['input_tokens']
                        runtime._tokens = lambda _messages, shallow=False, ids=ids: torch.tensor([ids], dtype=torch.long)
                        # This pre-admission probe supplies the exact checked source;
                        # all live product admissions stay closed until this run passes.
                        with patch('luban.product_inference.trial_expert_binding', return_value=expert), patch('luban.product_inference.native_binary', return_value=binary):
                            decision = Runtime.choose_route(runtime, messages(row))
                        assert decision['action'] in ('local_core', 'neural_expert')
                        route_pass[row['id']] = decision['action']
                chosen = [row for row in rows if (route_pass[row['id']] == 'neural_expert') == enabled]
                for row in chosen:
                    provenance = row['provenance']
                    provenance = json.loads(provenance) if isinstance(provenance, str) else provenance
                    check = validate_generation_check(provenance.get('generation_check'), default_expected=row['answer'])
                    answer = engine.generate(row['input_tokens'], check['max_new_tokens'], [248044, 248046], 0,
                        lambda _: None, threading.Event(), seed=42, cache_reuse=False)
                    text = tokenizer.decode(answer['tokens'], skip_special_tokens=True).strip()
                    complete = answer['finish_reason'] == 'stop' and bool(answer['tokens']) and answer['tokens'][-1] in (248044, 248046)
                    passed = evaluate_generation(text, check, complete)['passed']
                    current[row['id']] = {'id': row['id'], 'tokens': answer['tokens'], 'input_tokens': row['input_tokens'],
                        'answer': text, 'complete': complete, 'finish_reason': answer['finish_reason'], 'passed': passed,
                        'route_action': route_pass[row['id']], 'reference_used': False}
                result.setdefault('loads', []).append({'repeat': repeat, 'expert_enabled': enabled, 'loaded': loaded, 'memory': engine.status()})
            finally:
                engine.close()
                assert engine.process is None
        observations.append([current[row['id']] for row in rows]); routes.append(route_pass)
        print(json.dumps({'binary': str(binary), 'pass': repeat, 'rows': len(rows), 'correct': sum(v['passed'] for v in current.values())}), flush=True)
    result.update(observations=observations, routes=routes)
    assert observations[0] == observations[1], 'Repeated default product generation changed'
    return observations[0], routes[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert Path('/Volumes/移动数据库').is_mount()
    directory = args.directory.resolve()
    assert directory.is_relative_to(ROOT / 'experiments') and not directory.exists() and not args.receipt.exists()
    seed = prepare(directory); store = seed / 'modules/luban0-v1'
    proof = read(seed / 'runtime/mlx/expert-proof.json')
    assert proof['expert_id'] == 'learned-bafb4050374f' and proof['width'] == 32
    assert read(str(args.baseline) + '.json')['binary_sha256'] == proof['worker_binary_sha256']
    from luban.factory_expert import validate
    with patch('luban.mlx_model_packages.installed_binary', return_value=args.baseline):
        # Factory validation also needs the preverified model records. Its ZIP
        # inventory was pinned by the release; no large weight copy is needed.
        from luban import model_packages as packages
        payload = packages._read(seed / 'accepted.json'); packages._validate_journal(payload)
        from luban.factory_expert import MANIFEST as FACTORY
        packages._validate_modules(seed, payload, {k:v for k,v in FACTORY['files'].items() if k.startswith('modules/') or k == 'accepted.json'})
        from luban.mlx_model_packages import _proof_runtime
        _proof_runtime(store, proof)
    from luban.native_mlx import model_binding
    model = model_binding(SimpleNamespace(native_inference_precision='mlx-q4', native_mlx_model_path=str(MODEL)))
    core = {'file': str(store / (proof['core_version'] + '.safetensors')), 'tensor_key': 'model.language_model.layers.23.mlp.base.down_proj.weight', 'sha256': proof['core_file_sha256']}
    expert = {'id': proof['expert_id'], 'file': str(store / 'layer-23' / (proof['expert_id'] + '.safetensors')),
              'sha256': proof['expert_file_sha256'], 'width': proof['width'], 'core_sha256': core['sha256']}
    report = read(store / proof['report']); retention = read(store / (proof['core_version'] + '.retention.json'))
    rows = {}; groups = list(report['records'].values()) + [retention['groups'][k] for k in ('new_validation', 'replay_validation')] + [retention['regression_validation']]
    for group in groups:
        for row in group:
            if row['id'] in rows:
                assert messages(rows[row['id']]) == messages(row) and rows[row['id']]['answer'] == row['answer']
            else: rows[row['id']] = row
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    from safetensors.torch import load_file
    tokenizer = AutoTokenizer.from_pretrained(str(METADATA), local_files_only=True, trust_remote_code=False)
    for row in rows.values():
        tokens = tokenizer.apply_chat_template(messages(row), tokenize=True, add_generation_prompt=True, enable_thinking=False)
        row['input_tokens'] = tokens if isinstance(tokens, list) else tokens['input_ids']
    rows = list(rows.values())
    for observation in report['observations']['candidate'][0]:
        assert observation['input_tokens'] == next(row['input_tokens'] for row in rows if row['id'] == observation['id'])
    result = {'schema':'factory-engine-requalification-raw-v1', 'complete':False, 'teacher_calls':0, 'adopted':False,
        'source_proof':proof, 'scope':'unchanged factory accepted source; all exposed adoption and core retention regressions',
        'hardware':platform.platform(), 'seed':42, 'precision':'MLX affine Q4/group64, BF16 core/expert; FP32 Adam',
        'file_cache':'not purged; fresh default compiled product processes per pass/branch; no KV reuse',
        'memory_measurement':'Darwin proc_pid_rusage v4 child current/lifetime peak, excludes Python', 'rows':rows}
    frozen = [store / name for name in ('active.json','active-core.json','active-router.json')] + [seed / 'runtime/mlx/expert-proof.json', seed / 'runtime/mlx/training-proof.json']
    before = {str(path):path.read_bytes() for path in frozen}
    stats = {str(path): frozen_stat(path) for path in [core['file'],expert['file'],MODEL/'model.safetensors']}
    started = time.monotonic()
    from luban.process_lock import ModelProcessLock
    try:
        with_lock = ModelProcessLock(IDENTITY, 'mps')
        try:
            expert_oracle(load_file(expert['file']), directory/'expert-math-oracle.safetensors')
            expert_math(directory, args.binary, result, expert)
            result['baseline'] = {}; old, routes = quality(args.baseline, seed, rows, expert, core, model, Tokenizer.from_file(str(METADATA/'tokenizer.json')), result['baseline'])
            result['candidate'] = {}; new, new_routes = quality(args.binary, seed, rows, expert, core, model, Tokenizer.from_file(str(METADATA/'tokenizer.json')), result['candidate'])
        finally: with_lock.close()
        assert routes == new_routes, 'Current source actual routes changed'
        assert all(not a['passed'] or b['passed'] for a,b in zip(old,new)), 'Previously correct answer lost'
        assert all(a['tokens']==b['tokens'] and a['input_tokens']==b['input_tokens'] and a['complete']==b['complete'] for a,b in zip(old,new)), 'Protected default generation changed'
        # Existing seven-gate real Adam/replay/cancel/save/reload verifier;
        # only the explicit isolated source changes from its historical default.
        training_dir=directory/'training'
        command=[str(Path(__file__).resolve().parents[1]/'.venv/bin/python'),str(Path(__file__).with_name('verify_mlx_native_training.py')),
            '--directory',str(training_dir),'--output',str(training_dir/'result.json'),'--binary',str(args.binary),
            '--model-metadata',str(METADATA),'--source-root',str(seed)]
        with (directory/'training.log').open('x') as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        assert completed.returncode==0, 'Factory lifecycle failed; inspect training.log'
        training=read(training_dir/'result.json'); assert training['complete'] and all(training['gates'].values())
        assert all(path.read_bytes()==before[str(path)] for path in frozen)
        assert all(frozen_stat(path)==state for path,state in stats.items())
        result.update(complete=True,sources_unchanged=True,training=training)
        event=result['expert_math']['event']
        receipt={'schema':'luban-factory-engine-qualification-v1','complete':True,'teacher_calls':0,'sources_unchanged':True,
            'source_proof':proof,'worker_binary_sha256':training['worker_binary_sha256'],
            'training':{k:training[k] for k in ('schema','complete','worker_binary_sha256','core_file_sha256','mlx_model_sha256','parent_expert_file_sha256','gates')},
            'math':{k:event[k] for k in ('expert_file_sha256','fp32_pass','bf16_pass_assumed','stages')},
            'quality':{'complete':True,'rows':len(rows),'two_passes_equal':True,'routes_equal':True,
                'protected_answers_equal':True,'all_previously_correct_still_pass':True,
                'previously_correct':sum(row['passed'] for row in old),'factory_adoption_rows':len(report['observations']['candidate'][0])}}
        receipt['math'].update(passed=True,bf16_relative_l2_limit=.03,bf16_numerical_pass=True)
        save(args.receipt,receipt)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['seconds']=time.monotonic()-started;save(directory/'result.json',result)


if __name__=='__main__':main()
