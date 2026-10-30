"""One separately approved native2B cached-tail audit; default/prepare are stdlib only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PRIOR = Path('/Volumes/移动数据库/lubanllm/experiments/luban0-identity-fields-preregister-20261009T194226Z-876dd471/protocol.json')
PRIOR_SHA = 'fa6a3a720558dab906e5bc6137e54d081f96edda03da22b37173469ca5acb2f6'
PROTOCOL = 'native2b-three-train-two-clone-cache-audit-v1'
IDS = ('field-paraphrase-train-relation-fresh-dev-api-ordinary', 'field-context-train-name-multi', 'preservation-replay-train-olive-path')
IDENTITY = '673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df'
ORIGINAL = '1706e50833fffb25da45ad958af043a688e5ef495d45f33fa4b1ab623c696719'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Audit path may not contain symlinks')
    return path


def write_new(path, value):
    with safe(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare(output):
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError('Historical fixed TRAIN protocol changed')
    old = json.loads(PRIOR.read_text())
    rows = {r['id']: r for key in ('new_training', 'replay_training') for r in old['groups_before_original_baseline'][key]}
    samples = [copy.deepcopy(rows[key]) for key in IDS]
    for index, row in enumerate(samples):
        source = json.loads(row['provenance'])
        check = source['generation_check']
        exact_gold = (check['type'] == 'semantic' and check.get('validator') == 'integer'
                      and check['expected'] == 11 and row['answer'] == '11') if index == 0 else (
                      check['type'] == 'exact' and check['expected'] == row['answer'])
        if (row['purpose'] != 'training' or not row['approved'] or not row['training_allowed']
                or source['evolution_role'] != ('replay' if index == 2 else 'new')
                or not exact_gold
                or source.get('human_verified') is not False or source.get('program_verified') is not True):
            raise ValueError('Only these explicitly program-verified exact TRAIN rows are allowed')
    model = safe(Path(old['model_path']).resolve(strict=True))
    product, output = safe(old['production_root']), safe(output)
    for protected in (ROOT, product / 'app', product / 'modules', model, PRIOR.parent):
        if output == protected or protected in output.parents:
            raise ValueError('Fresh audit output must be outside source/model/product/historical experiment')
    if output.exists():
        raise ValueError('Audit output is single-attempt and must not exist')
    names = set(old['execution_manifest'])
    names.update(str(ROOT / p) for p in ('scripts/run_cached_training_audit.py', 'scripts/verify_cached_training.py',
        'scripts/verify_training_operators.py', 'scripts/diagnose_field_training.py', 'scripts/verify_automatic_batches.py',
        'scripts/verify_router_coverage_study.py', 'scripts/verify_router_coverage_counts.py', 'src/luban/process_lock.py',
        'src/luban/evolution_progress.py', 'tests/test_cached_training.py', 'tests/test_cached_training_runner.py'))
    # These actual installed graph sources were used by the existing offline official tiny audit.
    cpu_path = ROOT / 'research/cached_training_current_source_cpu.json'
    cpu = json.loads(cpu_path.read_text())
    if any(sha(name) != value for name, value in cpu['sources'].items()):
        raise ValueError('Current CPU cache proof does not bind the actual audit execution sources')
    names.update(cpu['sources'])
    proof_paths = [cpu_path, ROOT / 'research/partial_cached_training_independent_review.json',
                   ROOT / 'research/partial_evolution_closed_baseline_source_final.json']
    sources = {str(safe(name)): sha(safe(name)) for name in sorted(names)}
    manifest = model / 'upstream-file-manifest.json'
    official_count = len(json.loads(manifest.read_text())['Data']['Files']) + 1
    if official_count != 13:
        raise ValueError('Exactly twelve official members plus their upstream manifest are required')
    return {'protocol': PROTOCOL, 'state': 'unbound_waiting_root_review', 'sealed': False, 'model_go': False,
        'output': str(output), 'identity': IDENTITY, 'original_tensor_sha256': ORIGINAL, 'original_core': None,
        'model_path': str(model), 'production_root': str(product), 'prior_protocol_path': str(PRIOR),
        'prior_protocol_sha256': PRIOR_SHA, 'source_sha256': sources, 'samples': samples,
        'current_source_cpu_and_review_proofs': {str(p): sha(p) for p in proof_paths},
        'sample_sha256': [content_sha(r) for r in samples], 'device': 'mps', 'dtype': 'torch.bfloat16', 'seed': 42,
        'official_manifest_sha256': sha(manifest), 'official_model_binding': None, 'production_snapshot': None,
        'official_member_count_including_manifest': official_count,
        'optimizer': {'steps_per_native_and_cache_clone': 2, 'lr': 1e-5, 'eps': 1e-6, 'kl_weight': 5,
            'clip': 1, 'master': 'torch.float32', 'backward_order': ['CE.backward', '(5*KL).backward'], 'cumulative_replay_projection': True},
        'qualification': {'row': IDS[-1], 'once': True, 'core': None, 'branch': 'OFF', 'reference_used': False,
            'complete_EOS': True, 'exact_gold': 'OLIVE_PATH', 'max_new_tokens': 192, 'failure': 'reject; no retry or substitution'},
        'teacher_calls': 0, 'audit_generation_calls': 0, 'qualification_generation_calls': 1,
        'publication': False, 'research_runtime': True, 'paging': False, 'maximum_attempts': 1,
        'scope': 'three fixed TRAIN rows only; no DEV, archived candidate, warm start, generation quality or speed claim',
        'timing': 'wall including source/official validation, load, prefix/qualification, cache/native two-step comparison, per-use source/hash verification and synchronized transfers, restore/release and preservation; not optimized training throughput; RSS/MPS sampled every0.1s',
        'numerical_tolerances': {'torch.bfloat16': {'logits_atol': 2**-10, 'rtol': 2**-7, 'relative_l2': 2**-7,
            'loss_atol': 2**-10, 'gradient_atol': 2**-16, 'adam_atol': 1e-7, 'update_relative_l2': 2**-6}},
        'restoration': ['original last-down full tensor', 'requires_grad', 'grad object identities', 'module modes',
            'expert OFF', 'core pointer/version', 'Torch and MPS RNG', 'owned hooks/cache removed', 'model released', 'OS lock reacquired/closed']}


def production_snapshot(root):
    from diagnose_field_training import production_snapshot as snapshot
    result = snapshot(root)
    home = Path.home() / 'Library/Application Support/Luban'
    result['settings_and_credentials_sha256'] = {name: sha(safe(home / name)) if (home / name).exists() else None
        for name in ('settings.json', 'credentials.json')}
    return result


def bind(plan):
    from verify_router_coverage_study import model_binding
    if prepare(plan['output']) != plan:
        raise ValueError('Static protocol or source changed before actual binding')
    value = copy.deepcopy(plan)
    value.update(official_model_binding=model_binding(plan['model_path'], IDENTITY),
        production_snapshot=production_snapshot(plan['production_root']), state='bound_waiting_root_seal_MODEL_GO')
    return value


def validate_bound(plan):
    template = copy.deepcopy(plan)
    template.update(state='unbound_waiting_root_review', sealed=False, official_model_binding=None, production_snapshot=None)
    if prepare(plan['output']) != template or bind(template) != {**plan, 'sealed': False, 'state': 'bound_waiting_root_seal_MODEL_GO'}:
        raise ValueError('Bound source/data/official/production contract changed')


def execute(path, approved_sha):
    path = safe(path)
    if sha(path) != approved_sha:
        raise ValueError('Exact independently approved sealed protocol SHA required')
    plan = json.loads(path.read_text())
    if plan.get('protocol') != PROTOCOL or plan.get('sealed') is not True or plan.get('state') != 'sealed_waiting_explicit_MODEL_GO':
        raise ValueError('Only the separately sealed fixed native audit can execute')
    started = time.perf_counter()
    validate_bound(plan)  # No Torch, model allocation, or output before all binding checks pass.
    output = safe(plan['output'])
    output.mkdir(mode=0o700)
    report = {'protocol_sha256': approved_sha, 'protocol': PROTOCOL, 'pid': os.getpid(), 'model_started': False,
        'passed': False, 'teacher_calls': 0, 'publication': False, 'generation_calls': 0, 'phases': {},
        'source_sha256_before': plan['source_sha256'], 'production_before': plan['production_snapshot'], 'started': time.time()}
    write_new(output / 'run-started.json', report)
    runtime = None
    def persist():
        (output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    try:
        from luban.runtime import Runtime
        from luban.settings import Settings
        from luban.memory import Memory
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target, _generation
        from verify_cached_training import audit_cached_training, tensor_sha
        from run_identity_coverage import verify_training_prefix_alignment
        from verify_automatic_batches import measured
        import torch
        phase = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='luban-native-cache-settings-') as home:
            settings = Settings(home)
            settings.update({'storage_root': str(output / 'private'), 'model_path': plan['model_path'],
                             'temperature': 0, 'local_thinking': 'off'})
            runtime = Runtime(settings, Memory(output / 'private'), research=True)
            if str(runtime.device) != plan['device'] or str(runtime.dtype) != plan['dtype']:
                raise ValueError('Actual device/native dtype changed')
            runtime._model_process_lock = ModelProcessLock(IDENTITY, runtime.device)
            with measured(runtime) as resources:
                parameter = flags = modes = hooks = backward_hooks = rng = mps_rng = None
                try:
                    report['model_started'] = True
                    runtime.load(paging=False)
                    report['phases']['load_seconds'] = time.perf_counter() - phase
                    parameter = _target(runtime)[1]
                    report['original_tensor_sha256'] = tensor_sha(parameter)
                    if runtime.identity != IDENTITY or runtime.active_core is not None or runtime.branch.expert is not None or report['original_tensor_sha256'] != ORIGINAL:
                        raise ValueError('Loaded graph is not exact official originalNone1706 branchOFF')
                    torch.manual_seed(plan['seed'])
                    phase = time.perf_counter()
                    report['prefix'] = verify_training_prefix_alignment(runtime, {'prefix_alignment_required': True,
                        'groups_before_original_baseline': {'new_training': plan['samples'][:2], 'replay_training': plan['samples'][2:]},
                        'replay_training_candidates': []})
                    report['fixed_training_tokens'] = []
                    for sample in plan['samples']:
                        ids, labels = runtime._sample_tokens(sample)
                        report['fixed_training_tokens'].append({'id': sample['id'], 'sample_sha256': content_sha(sample),
                            'input_ids': ids.detach().cpu().tolist(), 'labels': labels.detach().cpu().tolist(),
                            'causal_prediction_positions': (labels[:, 1:] != -100).nonzero().detach().cpu().tolist()})
                    ids = labels = None
                    report['phases']['prefix_seconds'] = time.perf_counter() - phase
                    write_new(output / 'qualification-started.json', {'row': IDS[-1], 'protocol_sha256': approved_sha})
                    phase = time.perf_counter()
                    actual = _generation(runtime, plan['samples'][2:])
                    report['generation_calls'] = 1
                    report['qualification'] = actual
                    report['phases']['qualification_seconds'] = time.perf_counter() - phase
                    proof = actual[0]
                    if not (proof['passed'] is True and proof['complete'] is True and proof['branch_enabled'] is False
                            and proof['reference_used'] is False and proof['answer'] == 'OLIVE_PATH'):
                        raise ValueError('Current original replay qualification failed; gold/budget unchanged')
                    flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
                    modes = [(m, m.training) for m in runtime.model.modules()]
                    hooks = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                    backward_hooks = [(p, len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                    rng = torch.get_rng_state().clone()
                    mps_rng = torch.mps.get_rng_state().clone() if runtime.device == 'mps' else None
                    core_pointer = Path(runtime.store) / 'active-core.json'
                    core_pointer_before = core_pointer.read_bytes() if core_pointer.exists() else None
                    phase = time.perf_counter()
                    report['audit'] = audit_cached_training(runtime, plan['samples'])
                    report['phases']['cache_native_audit_seconds'] = time.perf_counter() - phase
                    report['fieldwise_restoration'] = {
                        'requires_grad': all(p.requires_grad == flag for p, flag, grad in flags),
                        'grad_object_identity': all(p.grad is grad for p, flag, grad in flags),
                        'module_modes': all(m.training == mode for m, mode in modes),
                        'owned_forward_hooks_removed': all(len(m._forward_hooks) == one and len(m._forward_pre_hooks) == two for m, one, two in hooks),
                        'owned_backward_hooks_removed': all(len(p._backward_hooks or {}) == count for p, count in backward_hooks),
                        'torch_rng': torch.equal(rng, torch.get_rng_state()),
                        'mps_rng': mps_rng is None or torch.equal(mps_rng, torch.mps.get_rng_state()),
                        'core_version_and_branch': runtime.active_core is None and runtime.branch.expert is None}
                    report['fieldwise_restoration']['core_pointer'] = (core_pointer.read_bytes() if core_pointer.exists() else None) == core_pointer_before
                    report['fieldwise_restoration_counts'] = {'parameters': len(flags), 'modules': len(modes)}
                    flags = modes = hooks = backward_hooks = rng = mps_rng = None
                    if not all(report['fieldwise_restoration'].values()):
                        raise ValueError('Actual fieldwise state restoration was incomplete')
                    if not report['audit']['passed'] or not report['audit']['restored']:
                        raise ValueError('Native/cache two-step algebra or restoration failed')
                    report['original_tensor_sha256_after'] = tensor_sha(parameter)
                    if report['original_tensor_sha256_after'] != ORIGINAL:
                        raise ValueError('Actual original full last-down tensor was not restored')
                except BaseException as error:
                    report['error'] = {'type': type(error).__name__, 'message': str(error)}
                # Exception traceback is gone before dropping the only model and OS lock.
                if flags is not None:
                    report['failure_restoration'] = {
                        'requires_grad': all(p.requires_grad == flag for p, flag, grad in flags),
                        'grad_object_identity': all(p.grad is grad for p, flag, grad in flags),
                        'module_modes': all(m.training == mode for m, mode in modes),
                        'owned_forward_hooks_removed': all(len(m._forward_hooks) == one and len(m._forward_pre_hooks) == two for m, one, two in hooks),
                        'owned_backward_hooks_removed': all(len(p._backward_hooks or {}) == count for p, count in backward_hooks),
                        'torch_rng': torch.equal(rng, torch.get_rng_state()),
                        'mps_rng': mps_rng is None or torch.equal(mps_rng, torch.mps.get_rng_state()),
                        'original_tensor': tensor_sha(parameter) == ORIGINAL}
                phase = time.perf_counter()
                parameter = flags = modes = hooks = backward_hooks = rng = mps_rng = None
                runtime._release_model()
                report['model_released'] = runtime.model is None and runtime._model_process_lock is None
                lock = ModelProcessLock(IDENTITY, plan['device']); lock.close()
                report['actual_process_lock_reacquired_and_closed'] = True
                report['phases']['release_seconds'] = time.perf_counter() - phase
            report['resources'] = resources
            report['passed'] = 'error' not in report
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        if runtime is not None:
            try:
                runtime._release_model()
                report['model_released'] = runtime.model is None and runtime._model_process_lock is None
                from luban.process_lock import ModelProcessLock
                lock = ModelProcessLock(IDENTITY, plan['device']); lock.close()
                report['actual_process_lock_reacquired_and_closed'] = True
            except BaseException as error:
                report['release_error'] = {'type': type(error).__name__, 'message': str(error)}
                report['passed'] = False
        phase = time.perf_counter()
        try:
            from verify_router_coverage_study import model_binding
            report['source_sha256_after'] = {name: sha(name) for name in plan['source_sha256']}
            report['production_after'] = production_snapshot(plan['production_root'])
            report['official_model_after'] = model_binding(plan['model_path'], IDENTITY)
            report['official_member_count_including_manifest'] = len(report['official_model_after'].get('files', {})) + 1
            report['preserved'] = report['source_sha256_after'] == plan['source_sha256'] and report['production_after'] == plan['production_snapshot'] and report['official_model_after'] == plan['official_model_binding']
        except BaseException as error:
            report['preservation_error'] = {'type': type(error).__name__, 'message': str(error)}
            report['preserved'] = False
        report['phases']['post_preservation_seconds'] = time.perf_counter() - phase
        report['wall_seconds_including_validation_load_audit_transfer_restore'] = time.perf_counter() - started
        report['passed'] = report['passed'] and report['preserved'] and report.get('model_released') is True and report.get('actual_process_lock_reacquired_and_closed') is True
        persist()
    if not report['passed']:
        raise RuntimeError('Fixed native cache audit rejected; preserve result, do not retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare', type=Path)
    actions.add_argument('--bind', type=Path)
    actions.add_argument('--seal', type=Path)
    actions.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--approved-sha')
    args = parser.parse_args()
    if args.run:
        if not args.approved_sha: parser.error('--run requires exact root-approved sealed SHA and MODEL GO')
        execute(args.run, args.approved_sha)
    elif args.prepare:
        if not args.output: parser.error('--prepare requires unique future result directory')
        write_new(args.prepare, prepare(args.output))
    elif args.bind:
        if not args.output: parser.error('--bind requires NEW bound protocol file')
        write_new(args.output, bind(json.loads(safe(args.bind).read_text())))
    elif args.seal:
        if not args.output: parser.error('--seal requires NEW formal protocol file')
        value = json.loads(safe(args.seal).read_text()); validate_bound(value)
        value.update(sealed=True, state='sealed_waiting_explicit_MODEL_GO')
        write_new(args.output, value)
    else:
        print(json.dumps({'protocol': PROTOCOL, 'fixed_TRAIN_ids': IDS, 'model_started': False,
            'torch_imported': False, 'state': 'waiting_root_SOURCE_review_BIND_SEAL_MODEL_GO'}))


if __name__ == '__main__':
    main()
