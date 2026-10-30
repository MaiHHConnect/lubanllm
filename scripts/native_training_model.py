"""Actual frozen Python-prefix fixture producer and full-model quality consumer.

Default and explicit preparation are stdlib only. Actual model work requires a
fresh separately sealed protocol and Root's exact MODEL GO. No training here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import compare_native_training as abi
import run_training_continuation450 as closed
import run_training_end_only450 as end
import run_training_step_curve as curve
import run_training_method_pilot as prior
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, safe, write_new, production_snapshot

PROTOCOL = 'actual-original57-frozen-prefix-native-training-material-quality-v1'
WALL_SECONDS = 2400
PARENT = tuple(ROOT / ('research/' + name) for name in (
    'training_end_only450_result.json', 'training_end_only450_result_backup.json',
    'training_end_only450_root_result_readback.json'))


def require(value, message):
    if value is not True:
        raise ValueError(message)


def read(path):
    return json.loads(safe(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def schedule(groups, steps):
    require(type(steps) is int and steps in (50, 450), 'Only actual50/450 original schedule prefixes')
    batches = [b for segment in end.prefix_schedule(groups)['segments'] for b in segment['batches']][:steps]
    result = [{'ordinal': i + 1, 'role': b['ce_role'], 'ce': b['ce_ids'][0], 'replay': b['replay_kl_ids'][0]}
              for i, b in enumerate(batches)]
    require(len(result) == steps and all(b['ce_event_ids'] == [i + 1] for i, b in enumerate(batches)),
            'Original CE event ordering differs')
    return result


def sources(parent):
    result = dict(parent['plan']['source_sha256'])
    for name in ('scripts/native_training_model.py', 'scripts/compare_native_training.py',
                 'scripts/run_training_continuation450.py', 'tests/test_training_continuation450.py',
                 'scripts/run_training_subset450.py', 'tests/test_training_subset450.py',
                 'tests/test_native_training_model.py', 'native/Training.cpp', 'scripts/native-training.ts'):
        result[str(ROOT / name)] = sha(ROOT / name)
    # Bind native-prefix module implementations without importing Torch/HF.
    packages = ROOT / '.venv/lib/python3.12/site-packages'
    paths = list((packages / 'torch/nn/modules').glob('*.py'))
    paths += list((packages / 'transformers/models/qwen3_5').glob('*.py'))
    paths += [packages / name for name in ('transformers/activations.py', 'transformers/pytorch_utils.py',
        'transformers/modeling_utils.py', 'transformers/modeling_layers.py', 'transformers/masking_utils.py',
        'transformers/cache_utils.py', 'transformers/generation/utils.py', 'transformers/__init__.py',
        'torch/optim/adam.py', 'torch/nn/functional.py', 'torch/version.py')]
    for path in paths:
        result[str(path)] = sha(path)
    return dict(sorted(result.items()))


def external_inputs(fixture_dir, python_output, native_output, comparison):
    require(all(v is not None for v in (fixture_dir, python_output, native_output, comparison)),
            'Quality requires actual same-fixture paired outputs and numeric receipt')
    root = safe(fixture_dir); manifest_sha = sha(root / 'manifest.tsv')
    manifest = abi.fixture(root, manifest_sha)
    metadata = read(root / 'fixture.json')
    require(metadata['metadata']['scope'] == 'actual_python_frozen_prefix'
            and metadata['metadata']['identity'] == IDENTITY and metadata['metadata']['core_version'] is None
            and metadata['metadata']['original_native_tensor_sha256'] == ORIGINAL
            and len(manifest['steps']) in (50, 450) and manifest['config']['identity'] == IDENTITY
            and manifest['config']['device'] == 'mps' and manifest['tensors']['candidate']['sha256'] == ORIGINAL,
            'Only actual bound original fixture quality')
    producer_path = root.parent / 'result.json'; producer = read(producer_path)
    closure_path = root.parent / 'producer-root-receipt.json'; closure = read(closure_path)
    require(closure.get('scope') == 'actual_closed_native_training_fixture_receipt'
            and type(closure.get('process_exit_code')) is int and closure['process_exit_code'] == 0
            and closure.get('pid_dead') is True and closure.get('pid') == producer.get('pid')
            and closure.get('producer_result') == {'path': str(producer_path), 'sha256': sha(producer_path)}
            and bool(closure.get('checks')) and all(v is True for v in closure['checks'].values()),
            'Actual Root closed producer exit/PID/post receipt required')
    formal_ref = closure['producer_protocol']; formal = read(formal_ref['path'])
    require(sha(formal_ref['path']) == formal_ref['sha256'] == producer.get('protocol_sha256')
            and formal.get('protocol') == PROTOCOL and formal.get('action') == 'produce'
            and formal.get('sealed') is True and formal.get('output') == str(root.parent),
            'Closed producer protocol/result/fixture root differ')
    require(producer.get('protocol') == PROTOCOL and producer.get('passed') is True and producer.get('model_started') is True
            and producer.get('preserved') is True and producer.get('model_released') is True
            and producer.get('actual_os_lock_reacquired_closed') is True and producer.get('teacher_calls') == 0
            and producer.get('publication') is False and set(producer.get('restoration', {})) == closed.saved_point.RESTORATION
            and all(v is True for v in producer['restoration'].values())
            and producer.get('model', {}).get('fixture' + str(len(manifest['steps']))) == {
                **producer['model']['fixture' + str(len(manifest['steps']))], 'directory': str(root), 'manifest_sha256': manifest_sha},
            'Actual producer exit/restoration/fixture receipt missing')
    outputs = {}
    for name, path in (('python', python_output), ('native', native_output)):
        path = safe(path); receipt = abi.output_receipt(path, manifest)
        require(receipt['report']['worker'] == ('Python-minimal-cached-tail-torch.optim.Adam' if name == 'python'
                else 'C++LibTorch-autograd-aligned-single-tensor-Adam') and receipt['report']['entry_adam_step'] == 0,
                'Fresh paired actual engine identity/Adam entry differs')
        outputs[name] = {'path': str(path), 'receipt_sha256': receipt['receipt_sha256'],
                         'candidate': receipt['tensors']['final_candidate']}
    comparison = safe(comparison); report = read(comparison)
    require(report.get('passed_numeric') is True, 'Actual paired numeric gate must pass before quality')
    require(report.get('python_receipt_sha256') == outputs['python']['receipt_sha256']
            and report.get('native_receipt_sha256') == outputs['native']['receipt_sha256']
            and report.get('python_final_candidate_sha256') == outputs['python']['candidate']['sha256']
            and report.get('native_final_candidate_sha256') == outputs['native']['candidate']['sha256'],
            'Numeric comparison outputs differ')
    require(len(report.get('scalar_metrics', [])) == len(manifest['steps'])
            and bool(report.get('tensor_metrics')) and all(v.get('passed') is True for v in report['tensor_metrics'].values()),
            'Complete actual numeric comparison checks required')
    return {'fixture': str(root), 'manifest_sha256': manifest_sha, 'fixture_json_sha256': sha(root / 'fixture.json'),
            'producer_result': {'path': str(producer_path), 'sha256': sha(producer_path)},
            'producer_root_receipt': {'path': str(closure_path), 'sha256': sha(closure_path)},
            'outputs': outputs, 'comparison': {'path': str(comparison), 'sha256': sha(comparison)}}


def prepare(output, action, *, fixture_dir=None, python_output=None, native_output=None, comparison=None):
    require(action in ('produce', 'quality'), 'Explicit produce or quality action required')
    parent = closed.closed_parent(*PARENT)
    require(parent['plan']['protocol'] == end.PROTOCOL, 'Only closed original END57 authority')
    output = safe(output); production = safe(parent['plan']['production_root'])
    experiments = production / 'experiments'
    require(output.parent.is_dir() and not output.exists() and experiments in output.parents
            and output.parent != experiments, 'Fresh private nested research execution required')
    inputs = external_inputs(fixture_dir, python_output, native_output, comparison) if action == 'quality' else None
    if inputs:
        require(not any(output == safe(p) or output.is_relative_to(safe(p)) or safe(p).is_relative_to(output)
                        for p in [inputs['fixture'], *[v['path'] for v in inputs['outputs'].values()]]), 'Input/output overlap')
        m = abi.fixture(inputs['fixture'], inputs['manifest_sha256'])
        expected = schedule(parent['groups'], len(m['steps']))
        exported = read(Path(inputs['fixture']) / 'fixture.json')
        original_rows = parent['groups']['new_training'] + parent['groups']['replay_training']
        by_id = {row['id']: 'r' + str(i).zfill(3) for i, row in enumerate(original_rows)}
        require(exported['samples'] == {by_id[row['id']]: row for row in original_rows}
                and m['steps'] == [{**s, 'ce': by_id[s['ce']], 'replay': by_id[s['replay']]} for s in expected]
                and all(m['rows'][by_id[row['id']]]['sample_sha256'] == abi.digest(row) for row in original_rows),
                'Actual raw fixture complete ordered114 material/original prefix differs')
        require(exported['metadata']['groups_sha256'] == parent['raw']['groups_sha256']
                and exported['metadata']['fixed_schedule_sha256'] == abi.digest(expected)
                and exported['metadata']['parent_raw_sha256'] == parent['descriptor']['raw']['sha256']
                and exported['metadata']['sources'] == sources(parent)
                and read(inputs['producer_result']['path'])['source_before'] == sources(parent),
                'Actual fixture ordered material/schedule/closed source differs')
    return {'protocol': PROTOCOL, 'state': 'unbound_waiting_root_review', 'sealed': False, 'model_go': False,
        'action': action, 'output': str(output), 'model_path': parent['plan']['model_path'],
        'production_root': str(production), 'device': 'mps', 'dtype': 'torch.bfloat16', 'seed': 42,
        'identity': IDENTITY, 'original_tensor_sha256': ORIGINAL, 'core_version': None,
        'torch_version': '2.8.0', 'transformers_version': '5.13.1',
        'parent_references': {str(p): sha(p) for p in PARENT}, 'source_sha256': sources(parent),
        'groups_sha256': parent['raw']['groups_sha256'], 'baseline_sha256': abi.digest(parent['baseline']),
        'schedule_sha256': {str(n): abi.digest(schedule(parent['groups'], n)) for n in (50, 450)},
        'inputs': inputs, 'official_model_binding': None, 'production_snapshot': None,
        'deadline_seconds': WALL_SECONDS, 'maximum_attempts': 1, 'teacher_calls': 0, 'publication': False,
        'generation_upper_bound': 0 if action == 'produce' else 338,
        'baseline_scope': 'Historical closed END57 originalNone1706 full114 baseline13NEW/55REPLAY; no fresh current baseline answer. Same source/official/dtype/W binding required.',
        'scope': 'Actual Python full frozen prefix capture; C++/Python cached-tail arms share preparation. No full native backbone, warm start, production adoption or transfer claim.',
        'timing': '2400 whole phase includes preflight/hash/load/capture/SVD/export or169proof+loss per distinct candidate, restore/release/post preservation. Shared fixture preparation is separately charged, never zero native cost.'}


def bind(plan):
    from verify_router_coverage_study import model_binding
    inp = plan['inputs'] or {}; outputs = inp.get('outputs', {})
    require(prepare(plan['output'], plan['action'], fixture_dir=inp.get('fixture'),
        python_output=outputs.get('python', {}).get('path'), native_output=outputs.get('native', {}).get('path'),
        comparison=inp.get('comparison', {}).get('path')) == plan, 'Prepared source/material/input bindings changed')
    official = model_binding(plan['model_path'], IDENTITY); parent = closed.closed_parent(*PARENT)
    require(official == parent['plan']['official_model_binding'] == parent['raw']['official_after'],
            'Historical baseline requires exact old13 official members and manifest, including tokenizer')
    return {**copy.deepcopy(plan), 'official_model_binding': official,
            'production_snapshot': production_snapshot(plan['production_root']), 'state': 'bound_waiting_root_seal_MODEL_GO'}


def validate_bound(plan):
    template = {**plan, 'state': 'unbound_waiting_root_review', 'sealed': False, 'model_go': False,
                'official_model_binding': None, 'production_snapshot': None}
    require(bind(template) == {**plan, 'sealed': False, 'model_go': False, 'state': 'bound_waiting_root_seal_MODEL_GO'},
            'Bound actual native model protocol changed')


def produce(runtime, parent, plan, output, deadline):
    import torch
    from luban.evolution import _target, _capture_replay_subspace
    from verify_cached_training import capture_tail, _binding, _sync, tensor_sha
    parameter = _target(runtime)[1]; groups = parent['groups']; start = time.perf_counter()
    native_basis, report = _capture_replay_subspace(runtime, parameter, groups['replay_training'])
    basis = native_basis.to(parameter.device); _sync(parameter.device)
    basis_seconds = time.perf_counter() - start; start = time.perf_counter(); caches = {}
    for row in groups['new_training'] + groups['replay_training']:
        require(time.perf_counter() < deadline, 'Actual cache preparation exceeded whole deadline')
        caches[row['id']] = capture_tail(runtime, row)
    _sync(parameter.device); cache_seconds = time.perf_counter() - start
    require(all(plan['source_sha256'].get(p) == h for p, h in _binding(runtime, parameter)['sources'].items()),
            'Actual Python frozen graph source is not preregistered')
    official = plan['official_model_binding']; model_root = safe(plan['model_path'])
    members = {k: {'sha256': h, 'bytes': (model_root / k).stat().st_size} for k, h in official['files'].items()}
    members['upstream-file-manifest.json'] = {'sha256': official['manifest_sha256'],
        'bytes': (model_root / 'upstream-file-manifest.json').stat().st_size}
    old_schedule = schedule(groups, 450)
    metadata = {'scope': 'actual_python_frozen_prefix', 'identity': IDENTITY, 'core_version': None,
        'original_native_tensor_sha256': ORIGINAL, 'official_files': members, 'sources': plan['source_sha256'],
        'groups_sha256': parent['raw']['groups_sha256'], 'parent_raw_sha256': parent['descriptor']['raw']['sha256'],
        'historical_baseline_sha256': plan['baseline_sha256'], 'fixed_schedule_sha256': abi.digest(old_schedule)}
    start = time.perf_counter()
    full = abi.export_runtime_fixture(runtime, groups['new_training'], groups['replay_training'],
        parameter.detach().float(), basis, caches, output / 'fixture450', steps=450, metadata=metadata, schedule=old_schedule)
    # ExFAT has no hardlinks: one capture, two verified raw copies; charge I/O.
    manifest = abi.fixture(full['directory'], full['manifest_sha256']); pilot = output / 'fixture50'; pilot.mkdir(mode=0o700)
    for value in manifest['tensors'].values(): shutil.copyfile(Path(full['directory']) / value['file'], pilot / value['file'])
    config = {**manifest['config'], 'steps': '50', 'checkpoint_steps': '1,2,50'}
    abi.write_text(pilot / 'manifest.tsv', abi.manifest_text(config, manifest['tensors'], manifest['rows'], manifest['steps'][:50]))
    receipt = read(Path(full['directory']) / 'fixture.json'); receipt['metadata']['fixed_schedule_sha256'] = abi.digest(schedule(groups, 50))
    receipt['manifest_sha256'] = sha(pilot / 'manifest.tsv'); receipt['shared_raw_source'] = full
    abi.write_json(pilot / 'fixture.json', receipt); abi.fixture(pilot, receipt['manifest_sha256'])
    require(tensor_sha(parameter) == ORIGINAL and time.perf_counter() < deadline, 'Producer original W/deadline failure')
    return {'fixture450': full, 'fixture50': {'directory': str(pilot), 'manifest_sha256': receipt['manifest_sha256']},
        'basis': report, 'basis_seconds': basis_seconds, 'cache_seconds': cache_seconds,
        'export_hash_save_seconds': time.perf_counter() - start, 'cache_bytes': sum(sum(c['bytes'].values()) for c in caches.values()),
        'capture_count': 114, 'raw_fixture_copies': 2, 'generation_calls': 0, 'baseline_reused': True, 'training_updates': 0}


def quality(runtime, parent, plan, deadline, progress):
    import torch
    from luban.evolution import _target
    from verify_cached_training import tensor_sha
    parameter = _target(runtime)[1]; inputs = plan['inputs']; groups = parent['groups']; baseline = parent['baseline']
    old_ids = prior.correct(groups['replay_training'], baseline['replay'])
    old = [r for r in groups['replay_training'] if r['id'] in old_ids]
    require(len(old) == 55 and len(prior.correct(groups['new_training'], baseline['new'])) == 13, 'Closed original protection changed')
    proofs = progress['candidates'] = {}; prior_sha = None
    for name in ('python', 'native'):
        item = inputs['outputs'][name]; spec = item['candidate']; path = safe(Path(item['path']) / spec['file'])
        require(spec['dtype'] == 'bf16' and spec['shape'] == [2048, 6144]
                and path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Actual full BF16 candidate bytes differ')
        if spec['sha256'] == prior_sha:
            proofs[name] = {'same_exact_raw_W_proof_reused_from': 'python', 'proof': proofs['python']['proof']}
            continue
        value = torch.from_file(str(path), shared=False, size=2048 * 6144, dtype=torch.bfloat16).reshape(2048, 6144)
        require(bool(torch.isfinite(value).all()), 'Nonfinite actual raw candidate')
        with torch.no_grad(): parameter.copy_(value.to(parameter.device))
        value = None; candidate_sha = tensor_sha(parameter)
        require(candidate_sha == spec['sha256'], 'Actual loaded candidate full tensor SHA differs')
        proof = curve.frozen_check(runtime, groups, baseline, old, candidate_sha, deadline, second=True)
        proofs[name] = {'same_exact_raw_W_proof_reused_from': None, 'proof': proof}
        require(proof['safety_passed'] is True and proof['mastery_confirmed'] is True
                and proof['forgotten_ids'] == [] and tensor_sha(parameter) == candidate_sha,
                'Actual paired candidate old protection/frozen complete proof failed')
        prior_sha = spec['sha256']
    python_ids = proofs['python']['proof']['newly_confirmed_ids']; native_ids = proofs['native']['proof']['newly_confirmed_ids']
    require(len(native_ids) >= len(python_ids), 'Native actual finite growth is below current paired Python control')
    progress.update(python_confirmed_ids=python_ids, native_confirmed_ids=native_ids)
    return {**progress,
        'growth_noninferior_by_count': True, 'same_ids': native_ids == python_ids,
        'generation_calls': 169 if proofs['native']['same_exact_raw_W_proof_reused_from'] else 338,
        'historical_27_not_used_as_control': True, 'publication': False}


def execute(path, approved_sha):
    path = safe(path); require(isinstance(approved_sha, str) and abi.SHA.fullmatch(approved_sha) is not None
                              and sha(path) == approved_sha, 'Exact sealed Root SHA required')
    plan = read(path); require(plan['protocol'] == PROTOCOL and plan['sealed'] is True
        and plan['state'] == 'sealed_waiting_explicit_MODEL_GO', 'Separately sealed single-attempt model protocol required')
    began = time.perf_counter(); deadline = began + WALL_SECONDS; validate_bound(plan)
    output = safe(plan['output']); require(time.perf_counter() < deadline and not output.exists(), 'Fresh bounded output required')
    write_new(path.parent / 'execution-started.json', {'protocol_sha256': approved_sha, 'action': plan['action'], 'time': time.time()})
    output.mkdir(mode=0o700); report = {'protocol': PROTOCOL, 'protocol_sha256': approved_sha, 'pid': os.getpid(),
        'model_started': False, 'passed': False, 'teacher_calls': 0, 'publication': False, 'source_before': plan['source_sha256']}
    write_new(output / 'run-started.json', report)
    runtime = parameter = original = flags = modes = hooks = backhooks = rng = mps_rng = pointer = None
    try:
        import torch
        import transformers
        require(torch.__version__ == plan['torch_version'] and transformers.__version__ == plan['transformers_version'],
                'Actual installed Torch/HF versions differ from closed original study')
        from luban.runtime import Runtime
        from luban.settings import Settings
        from luban.memory import Memory
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target
        from verify_cached_training import tensor_sha
        from verify_automatic_batches import measured
        parent = closed.closed_parent(*PARENT)
        with tempfile.TemporaryDirectory(prefix='luban-native-model-') as home:
            settings = Settings(home); settings.update({'storage_root': str(output / 'private'), 'model_path': plan['model_path'],
                                                       'temperature': 0, 'local_thinking': 'off'})
            runtime = Runtime(settings, Memory(output / 'private'), research=True)
            require(str(runtime.device) == plan['device'] and str(runtime.dtype) == plan['dtype'], 'Actual MPS/BF16 hardware changed')
            runtime._model_process_lock = ModelProcessLock(IDENTITY, runtime.device)
            with measured(runtime) as resources:
                start = time.perf_counter(); runtime.load(paging=False); report['model_started'] = True
                report['load_seconds'] = time.perf_counter() - start; parameter = _target(runtime)[1]; original = parameter.detach().clone()
                require(runtime.identity == IDENTITY and runtime.active_core is None and runtime.branch.expert is None
                        and tensor_sha(parameter) == ORIGINAL, 'Actual originalNone1706 OFF required')
                flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
                modes = [(m, m.training) for m in runtime.model.modules()]
                hooks = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backhooks = [(p, len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone()
                pointer = Path(runtime.store) / 'active-core.json'; pointer_before = pointer.read_bytes() if pointer.exists() else None
                torch.manual_seed(42); runtime.model.eval().requires_grad_(False); parameter.requires_grad_(True)
                from run_identity_coverage import verify_training_prefix_alignment
                report['current_TRAIN_prefix_alignment'] = verify_training_prefix_alignment(runtime,
                    {'prefix_alignment_required': True, 'groups_before_original_baseline': parent['groups'],
                     'replay_training_candidates': []})
                report['model'] = {}
                report['model'] = produce(runtime, parent, plan, output, deadline) if plan['action'] == 'produce' \
                    else quality(runtime, parent, plan, deadline, report['model'])
                report['resources'] = resources; report['passed'] = True
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if runtime is not None:
            try:
                if original is not None:
                    import torch
                    from verify_cached_training import tensor_sha
                    with torch.no_grad(): parameter.copy_(original)
                    if flags is not None:
                        for p, flag, grad in flags: p.requires_grad_(flag); p.grad = grad
                        for m, mode in modes: m.training = mode
                        torch.set_rng_state(rng); torch.mps.set_rng_state(mps_rng)
                        report['restoration'] = {'original_tensor': tensor_sha(parameter) == ORIGINAL,
                            'flags_grads': all(p.requires_grad == f and p.grad is g for p, f, g in flags),
                            'modes': all(m.training == v for m, v in modes), 'rng': torch.equal(rng, torch.get_rng_state()),
                            'mps_rng': torch.equal(mps_rng, torch.mps.get_rng_state()), 'core_branch': runtime.active_core is None and runtime.branch.expert is None,
                            'forward_hooks': all(len(m._forward_hooks) == a and len(m._forward_pre_hooks) == b for m, a, b in hooks),
                            'backward_hooks': all(len(p._backward_hooks or {}) == n for p, n in backhooks),
                            'pointer': (pointer.read_bytes() if pointer.exists() else None) == pointer_before}
            except BaseException as error:
                report['restoration_error'] = {'type': type(error).__name__, 'message': str(error)}
            parameter = original = flags = modes = hooks = backhooks = rng = mps_rng = p = m = None
            try:
                runtime._release_model()
                from luban.process_lock import ModelProcessLock
                lock = ModelProcessLock(IDENTITY, plan['device']); lock.close()
                report['model_released'] = runtime.model is None and runtime._model_process_lock is None
                report['actual_os_lock_reacquired_closed'] = True
            except BaseException as error:
                report['release_error'] = {'type': type(error).__name__, 'message': str(error)}
        try:
            from verify_router_coverage_study import model_binding
            report['source_after'] = {p: sha(p) for p in plan['source_sha256']}
            report['preserved'] = report['source_after'] == plan['source_sha256'] and production_snapshot(plan['production_root']) == plan['production_snapshot']
            report['official_after'] = model_binding(plan['model_path'], IDENTITY)
            report['preserved'] &= report['official_after'] == plan['official_model_binding']
            require({str(p): sha(p) for p in PARENT} == plan['parent_references'], 'Closed parent evidence changed')
            if plan['inputs']:
                inp = plan['inputs']; outputs = inp['outputs']
                require(external_inputs(inp['fixture'], outputs['python']['path'], outputs['native']['path'], inp['comparison']['path']) == inp,
                        'Actual paired fixture/output evidence changed during quality')
        except BaseException as error:
            report['preservation_error'] = {'type': type(error).__name__, 'message': str(error)}; report['preserved'] = False
        report['wall_seconds_before_final_write'] = time.perf_counter() - began
        report['passed'] = report['passed'] and report.get('preserved') is True and len(report.get('restoration', {})) == 9 \
            and all(report['restoration'].values()) and report.get('model_released') is True and report.get('actual_os_lock_reacquired_closed') is True \
            and not any(k in report for k in ('error', 'restoration_error', 'release_error', 'preservation_error')) and time.perf_counter() < deadline
        write_new(output / 'result.json', report)
        report['whole_wall_after_first_final_write_seconds'] = time.perf_counter() - began
        if time.perf_counter() >= deadline:
            report['passed'] = False; report['deadline_error'] = 'Whole phase including restoration/post/final raw exceeded2400'
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if time.perf_counter() >= deadline:
            report['passed'] = False; report['deadline_error'] = 'Final result save exceeded whole2400 deadline'
            (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'passed': report['passed'], 'whole_wall_after_final_write_seconds': time.perf_counter() - began, 'publication': False}), flush=True)
    if not report['passed']: raise RuntimeError('Native model phase failed; preserve raw, no retry or publication')


def main():
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', type=Path); mode.add_argument('--bind', type=Path); mode.add_argument('--seal', type=Path); mode.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--action', choices=('produce', 'quality')); parser.add_argument('--approved-sha')
    parser.add_argument('--fixture', type=Path); parser.add_argument('--python-output', type=Path); parser.add_argument('--native-output', type=Path); parser.add_argument('--comparison', type=Path)
    args = parser.parse_args()
    if args.run: execute(args.run, args.approved_sha); return
    if args.prepare: write_new(args.prepare, prepare(args.output, args.action, fixture_dir=args.fixture, python_output=args.python_output, native_output=args.native_output, comparison=args.comparison))
    elif args.bind: write_new(args.output, bind(read(args.bind)))
    elif args.seal:
        value = read(args.seal); validate_bound(value)
        write_new(args.output, {**value, 'sealed': True, 'state': 'sealed_waiting_explicit_MODEL_GO'})
    else: print(json.dumps({'protocol': PROTOCOL, 'SOURCE_only': True, 'torch_imported': False, 'model_started': False, 'training_updates': 0, 'quality_generation_max': 338}))


if __name__ == '__main__': main()
