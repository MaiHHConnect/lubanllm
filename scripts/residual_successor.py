"""Research-only frozen-parent contracts and TRAIN protection for fixed450 E."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from luban.core_update import apply_core_residual, validate_basis
from luban.evolution import _target
from luban.evolution_progress import extend_protected_basis
from study_core_constraint import capture

PROTOCOL = 'preservation-first-residual-fixed450-v4-successor'
FORMAT = 'residual-successor-v1'
ARITHMETIC = 'one_fp32_parent_plus_increment'


def inherited_loss_tolerance(identifier, bound, successor):
    """Match the verifier's fixed FP32 measurement allowance for inherited rows."""
    if successor is None or identifier in set(successor.get('original_fixed', {}).get('ids', ())):
        return 0.0
    return 64 * torch.finfo(torch.float32).eps * max(1., abs(float(bound)))


def read(path):
    return json.loads(Path(path).read_text())


def load_parent(directory, verification, prepared_root, identity, *, ancestry=()):
    """One bound parent load; no models, hashes, snapshots, or parent writes."""
    from verify_residual_round import load_round
    directory, verification = Path(directory).resolve(), Path(verification).resolve()
    if directory.parent != Path(prepared_root).resolve() or str(directory) in ancestry:
        raise ValueError('Parent must share the original prepared materials without a cycle')
    bundle = load_round(directory, identity, ancestry=ancestry)
    groups, rows, result, checkpoint, data, fixed, baseline, final = bundle
    prior = data.get('successor')
    by_id = {row['id']: row for row in final}
    replay_ids = {row['id'] for row in groups['replay_training']}
    required_correct = (prior['parent_correct_replay_ids'] if prior else
                        [row['id'] for row in baseline if row['id'] in replay_ids and row['passed']])
    raw_required = (prior['protected_raw'] if prior else
                    {row['id']: row for row in baseline if row['id'] in fixed['ids'][:4]})
    if (result.get('baseline_correct_replay') != len(required_correct)
            or any(not by_id[key]['passed'] for key in required_correct)
            or any(not by_id[key]['complete'] or by_id[key]['raw_completion_token_ids'] != value['raw_completion_token_ids']
                   for key, value in raw_required.items())):
        raise ValueError('Parent final records lost their fixed raw or correct replay obligations')
    checked = read(verification / 'result.json')
    flags = ('carrier_consistent', 'generation_tokens_exact', 'generation_validators_exact',
             'full_model_loss_guard_passed', 'original_parameter_versions_unchanged', 'model_lock_released')
    if (checked.get('status') != 'diagnostic_verification_passed'
            or checked.get('identity') != identity
            or Path(checked.get('round_directory', '')).resolve() != directory
            or checked.get('source_protocol') != result['protocol']
            or checked.get('source_candidate_qualified') != result['candidate_qualified']
            or checked.get('production_adoption') is not False
            or checked.get('training_performed') is not False
            or checked.get('teacher_calls') != 0 or any(checked.get(key) is not True for key in flags)
            or result.get('probe_tokens_unchanged') is not True
            or result.get('retained_correct_replay') != result.get('baseline_correct_replay')):
        raise ValueError('Parent needs a completed, matching independent verification and retention')
    actual = read(verification / 'generation.json')
    if ([row['id'] for row in actual] != [row['id'] for row in final]
            or any(any(a.get(key) != b.get(key) for key in
                       ('raw_completion_token_ids', 'passed', 'correct', 'complete', 'forbidden_found'))
                   for a, b in zip(actual, final))):
        raise ValueError('Independent raw generations or validators differ from the parent')
    losses = read(verification / 'protection.json')
    if len(losses) != len(fixed['ids']):
        raise ValueError('Independent fixed protection does not bind the parent')
    for row, identifier, bound in zip(losses, fixed['ids'], fixed['losses']):
        tolerance = inherited_loss_tolerance(identifier, bound, prior)
        measured = row.get('full_model_loss')
        if (row.get('id') != identifier or row.get('fixed_baseline') != bound
                or row.get('nonincreasing') is not True
                or not isinstance(measured, (int, float))
                or not torch.isfinite(torch.tensor(measured))
                or measured < 0
                or measured - bound > tolerance
                or ('tolerance' in row and row['tolerance'] != tolerance)):
            raise ValueError('Independent fixed protection does not bind the parent')
    artifact = Path(checked.get('diagnostic_delta_artifact', '')).resolve()
    if artifact.parent != verification:
        raise ValueError('Independent residual must belong to its verification directory')
    with safe_open(artifact, framework='pt', device='cpu') as saved:
        if (list(saved.keys()) != ['delta'] or (saved.metadata() or {}).get('base_identity') != identity
                or not torch.equal(saved.get_tensor('delta'), checkpoint['delta'])):
            raise ValueError('Independently loaded residual differs from parent D')
    successor = data.get('successor')
    basis = data.get('basis')
    if basis is None:
        if result.get('expanded_span'):
            basis = torch.load(directory / 'expanded-basis.pt', map_location='cpu', weights_only=True)['basis']
        else:
            raise ValueError('A bound expanded parent basis is required')
    validate_basis(basis)
    if successor is not None and (result.get('inherited_raw_unchanged') is not True
                                  or result.get('parent_delta_unchanged') is not True):
        raise ValueError('Parent lost inherited mastered TRAIN trajectories')
    # The caller needs only this parent, not every ancestor's tensors/optimizer.
    data = {key: value for key, value in data.items() if key not in ('parent_bundle', 'parent_delta')}
    checkpoint = {key: value for key, value in checkpoint.items() if key != 'optimizer'}
    return dict(groups=groups, rows=rows, result=result, checkpoint=checkpoint, data=data,
                fixed=fixed, baseline=baseline, final=final, basis=basis, successor=successor,
                verification=checked, binding={'round_directory': str(directory),
                'verification_directory': str(verification), 'protocol': result['protocol'], 'identity': identity})


def inherited_train(parent):
    """Cumulative successful TRAIN, never DEV; earlier raw obligations survive."""
    prior = parent['successor']
    mastered = set(prior['mastered_train_ids'] if prior else [])
    final = {row['id']: row for row in parent['final']}
    training = parent['groups']['new_training']
    mastered.update(row['id'] for row in training if final[row['id']]['passed'])
    ordered = [row['id'] for row in training if row['id'] in mastered]
    if set(ordered) != mastered or any(not final[key]['passed'] or not final[key]['complete'] for key in ordered):
        raise ValueError('Inherited mastery must still pass complete TRAIN generation')
    raw = dict(prior['protected_raw']) if prior else {}
    for key in ordered:
        if key in raw and raw[key]['raw_completion_token_ids'] != final[key]['raw_completion_token_ids']:
            raise ValueError('Parent changed an inherited mastered TRAIN trajectory')
        raw.setdefault(key, final[key])
    return ordered, raw


def replay_correct(parent):
    prior = parent['successor']
    replay_ids = [row['id'] for row in parent['groups']['replay_training']]
    final = {row['id']: row for row in parent['final']}
    original = (prior['original_correct_replay_ids'] if prior else
                [row['id'] for row in parent['baseline'] if row['id'] in replay_ids and row['passed']])
    current = [key for key in replay_ids if final[key]['passed']]
    required = prior['parent_correct_replay_ids'] if prior else original
    if not set(original).issubset(required) or not set(required).issubset(current):
        raise ValueError('Parent lost an original or subsequently correct replay')
    return original, current


def parent_equivalence(runtime, cache, parent_delta):
    """One lifecycle check, using the existing nonzero full/cached math audit."""
    from diagnose_residual_generation import full_cached_math
    active = [parent_delta]
    handle = runtime.branch.base.down_proj.register_forward_hook(
        lambda module, inputs, output: apply_core_residual(inputs[0], output, active[0]))
    try:
        with torch.no_grad():
            ids = cache['ids'].to(parent_delta.device)
            before = runtime.model(ids, use_cache=False).logits.detach()
            active[0] = parent_delta + torch.zeros_like(parent_delta)
            after = runtime.model(ids, use_cache=False).logits.detach()
            zero_exact = torch.equal(before, after)
        active[0] = parent_delta.detach().clone().requires_grad_()
        math, _ = full_cached_math(runtime, cache, active[0])
        if not zero_exact or not math['exact_equivalence']:
            raise ValueError('Parent plus zero E or nonzero parent full/cached math differs')
        return {'zero_E_parent_logits_exact': zero_exact, 'nonzero_parent_math': math}
    finally:
        handle.remove()


def prepare_successor(runtime, parent, caches, parent_delta):
    """Capture only new TRAIN protection and append to the actual parent span."""
    from probe_core_constraint import probes
    from verify_residual_round import capture_greedy_trajectory
    mastered, raw = inherited_train(parent)
    original_ids = [row['id'] for row in probes() + parent['groups']['replay_training']]
    prior = parent['successor']
    previous_mastered = set(prior['mastered_train_ids'] if prior else [])
    original_raw = ({row['id']: row for row in parent['baseline'] if row['id'] in original_ids[:4]}
                    if prior is None else {key: prior['protected_raw'][key] for key in original_ids[:4]})
    raw.update(original_raw)
    entries = [{'id': key, 'row_id': key, 'kind': 'original_raw' if index < 4 else 'original_gold',
                **{name: caches[key][name] for name in ('ids', 'labels')}}
               for index, key in enumerate(original_ids)]
    directions, trajectories = [], []
    handle = runtime.branch.base.down_proj.register_forward_hook(
        lambda module, inputs, output: apply_core_residual(inputs[0], output, parent_delta))
    try:
        for row in parent['groups']['new_training']:
            key = row['id']
            if key not in mastered:
                continue
            observed = raw[key]
            raw_cache = capture(runtime, row, generated=observed)
            for kind, cache in (('mastered_gold', caches[key]), ('mastered_raw', raw_cache)):
                identifier = 'successor-' + ('gold:' if kind == 'mastered_gold' else 'raw:') + key
                caches[identifier] = cache
                entries.append({'id': identifier, 'row_id': key, 'kind': kind,
                                **{name: cache[name] for name in ('ids', 'labels')}})
                if key not in previous_mastered:
                    directions.append(cache['x'][:, :-1][cache['labels'][:, 1:] != -100].float())
            if key not in previous_mastered:
                trajectory = capture_greedy_trajectory(runtime.model, runtime.tokenizer, row, observed)
                trajectories.append(trajectory)
    finally:
        handle.remove()
    basis = parent['basis']
    if directions:
        generated = torch.cat(directions + [value['inputs'].float() for value in trajectories])
        basis, extension = extend_protected_basis(basis, generated)
    else:
        extension = {'added_rank': 0, 'extra_tolerance': 0., 'old_span_residual': 0.}
    validate_basis(basis)  # A full-rank extension has no learning complement.
    extension.update(gold_raw_teacher_forced_positions=sum(len(value) for value in directions),
                     actual_kv_positions=sum(len(value['inputs']) for value in trajectories))
    original_correct, parent_correct = replay_correct(parent)
    original_fixed = (prior['original_fixed'] if prior else
                      {'ids': parent['fixed']['ids'], 'losses': parent['fixed']['losses']})
    contract = {'format': FORMAT, 'arithmetic': ARITHMETIC, 'parent_binding': parent['binding'],
        'round_index': (prior['round_index'] if prior else 0) + 1,
        'material_groups': parent['groups'], 'mastered_train_ids': mastered, 'protected_raw': raw,
        'original_correct_replay_ids': original_correct, 'parent_correct_replay_ids': parent_correct,
        'original_fixed': original_fixed,
        'protection_entries': entries, 'inherited_bounds': dict(zip(parent['fixed']['ids'], parent['fixed']['losses']))}
    return basis, extension, contract


def validate_successor_contract(directory, result, checkpoint, fixed, artifact, identity, *, ancestry=()):
    """Validate v4 E geometry separately from deployed total, including ancestry."""
    directory = Path(directory).resolve()
    if str(directory) in ancestry:
        raise ValueError('Cyclic residual successor ancestry')
    contract = torch.load(directory / 'successor.pt', map_location='cpu', weights_only=True)
    binding = contract.get('parent_binding') or {}
    parent = load_parent(binding.get('round_directory', ''), binding.get('verification_directory', ''),
                         directory.parent, identity, ancestry=(*ancestry, str(directory)))
    if (result.get('protocol') != PROTOCOL or contract.get('format') != FORMAT
            or contract.get('arithmetic') != ARITHMETIC or checkpoint.get('parent_binding') != binding
            or binding != parent['binding'] or contract.get('material_groups') != parent['groups']
            or result.get('parent_binding') != binding):
        raise ValueError('Successor protocol, parent or material binding differs')
    mastered, raw = inherited_train(parent)
    prior = parent['successor']
    if (contract.get('round_index') != (prior['round_index'] if prior else 0) + 1
            or contract.get('mastered_train_ids') != mastered
            or result.get('mastered_train_ids') != mastered):
        raise ValueError('Cumulative mastered TRAIN or round index differs')
    from probe_core_constraint import probes
    original_ids = [row['id'] for row in probes() + parent['groups']['replay_training']]
    original_raw = ({row['id']: row for row in parent['baseline'] if row['id'] in original_ids[:4]}
                    if prior is None else {key: prior['protected_raw'][key] for key in original_ids[:4]})
    raw.update(original_raw)
    original_correct, parent_correct = replay_correct(parent)
    original_fixed = prior['original_fixed'] if prior else {'ids': parent['fixed']['ids'], 'losses': parent['fixed']['losses']}
    if (contract.get('protected_raw') != raw or contract.get('original_correct_replay_ids') != original_correct
            or contract.get('parent_correct_replay_ids') != parent_correct
            or contract.get('original_fixed') != original_fixed):
        raise ValueError('Inherited raw, original replay or fixed baseline differs')
    expected = [(key, key, 'original_raw' if index < 4 else 'original_gold') for index, key in enumerate(original_ids)]
    expected += [(f'successor-{kind}:{key}', key, f'mastered_{kind}') for key in mastered for kind in ('gold', 'raw')]
    entries = contract.get('protection_entries') or []
    if ([(row.get('id'), row.get('row_id'), row.get('kind')) for row in entries] != expected
            or fixed.get('ids') != [row[0] for row in expected]):
        raise ValueError('Successor protection order differs')
    for entry in entries:
        ids, labels = entry['ids'], entry['labels']
        if (not isinstance(ids, torch.Tensor) or not isinstance(labels, torch.Tensor)
                or ids.dtype != torch.long or labels.dtype != torch.long or ids.ndim != 2
                or ids.shape[0] != 1 or labels.shape != ids.shape):
            raise ValueError('Invalid successor protection tokens')
        selected = labels[0].ne(-100).nonzero().flatten()
        if (not selected.numel() or int(selected[0]) < 1 or int(selected[-1]) != ids.shape[1] - 1
                or not torch.equal(labels[0, selected[0]:], ids[0, selected[0]:])):
            raise ValueError('Successor protection needs a masked prefix and complete targets')
        if entry['kind'].endswith('raw'):
            if ids[0, selected[0]:].tolist() != raw[entry['row_id']]['raw_completion_token_ids']:
                raise ValueError('Successor raw target differs from inherited tokens')
        else:
            gold = parent['data']['caches'][entry['row_id']]
            if any(not torch.equal(entry[name], gold[name]) for name in ('ids', 'labels')):
                raise ValueError('Successor gold differs from bound TRAIN tokenization')
    inherited = dict(zip(parent['fixed']['ids'], parent['fixed']['losses']))
    values = torch.tensor(contract.get('parent_losses', []), dtype=torch.float32)
    declared = torch.tensor(fixed.get('losses', []), dtype=torch.float32)
    if (contract.get('inherited_bounds') != inherited or values.shape != (len(entries),)
            or declared.shape != values.shape or not bool(torch.isfinite(values).all()) or bool((values < 0).any())):
        raise ValueError('Missing finite parent protection loss binding')
    expected_bounds = torch.tensor([min(inherited.get(row['id'], float(value)), float(value))
                                    for row, value in zip(entries, values)], dtype=torch.float32)
    if not torch.equal(declared, expected_bounds) or not torch.equal(checkpoint['fixed_protection_losses'], declared):
        raise ValueError('Successor protection thresholds were raised or changed')
    increment, total = checkpoint.get('increment'), checkpoint['delta']
    parent_delta = parent['checkpoint']['delta']
    if (not isinstance(increment, torch.Tensor) or increment.dtype != torch.float32
            or increment.shape != parent_delta.shape or not bool(torch.isfinite(increment).all())
            or not torch.equal(total, parent_delta + increment)):
        raise ValueError('Saved FP32 total is not frozen parent D plus E')
    basis = artifact['basis']
    geometry = validate_basis(basis)
    old = parent['basis']
    span_error = float((old - basis @ (basis.T @ old)).abs().max())
    residual = float((increment @ basis).norm())
    tolerance = 64 * torch.finfo(torch.float32).eps * max(1., float(increment.double().norm()))
    extension = artifact.get('extension', {})
    new_mastered = set(mastered) - set(prior['mastered_train_ids'] if prior else [])
    tf_count = sum(int(entry['labels'].ne(-100).sum()) for entry in entries
                   if entry['row_id'] in new_mastered)
    kv_count = sum(len(raw[key]['raw_completion_token_ids']) for key in new_mastered)
    if (span_error > 2e-5 or residual > tolerance or extension != result.get('extension')
            or extension.get('added_rank') != basis.shape[1] - old.shape[1]
            or extension.get('gold_raw_teacher_forced_positions') != tf_count
            or extension.get('actual_kv_positions') != kv_count
            or fixed.get('basis_rank') != basis.shape[1] or result.get('basis', {}).get('rank') != basis.shape[1]):
        raise ValueError('Successor E geometry or inherited protected span differs')
    return {'parent_delta': parent_delta, 'successor': contract, 'basis': basis, 'parent_basis': old,
            'parent_bundle': parent, 'basis_validation': {**geometry, 'rank': basis.shape[1],
            'old_span_residual': span_error, 'increment_subspace_residual': residual,
            'increment_subspace_tolerance': tolerance}}


def audit_parent(directory, verification):
    """Independent one-model migration audit; never enters the training loop."""
    import platform
    import resource
    import tempfile
    import time
    from train_luban_identity import MODEL, runtime_for
    from verify_local_review import model_identity
    from luban.settings import private_json
    directory = Path(directory).resolve()
    if not Path('/Volumes/移动数据库').is_mount() or not directory.is_relative_to(
            Path('/Volumes/移动数据库/lubanllm/experiments')):
        raise ValueError('Use the mounted external experiment directory')
    destination = Path(tempfile.mkdtemp(prefix='successor-parent-math-', dir=directory.parent))
    began = time.perf_counter()
    report = {'status': 'preparing', 'seed': 42, 'device': 'mps', 'hardware': platform.platform(),
        'precision': 'BF16 original graph; FP32 frozen parent D plus zero E',
        'scope': 'Independent finite parent migration math, not learning or adoption',
        'optimizer_updates': 0, 'teacher_calls': 0, 'production_adoption': False,
        'memory_measurement': 'Darwin lifetime peak RSS bytes; MPS driver separately, overlapping'}
    runtime = None
    try:
        torch.manual_seed(42)
        identity = model_identity(MODEL)
        parent = load_parent(directory, verification, directory.parent, identity)
        report['parent_binding'] = parent['binding']
        if not torch.backends.mps.is_available():
            raise RuntimeError('Independent real parent audit requires MPS')
        runtime = runtime_for(destination)
        parameter = _target(runtime)[1]
        if runtime.active_core is not None or runtime.identity != identity or not torch.equal(
                parameter.detach().cpu(), parent['data']['reference']):
            raise ValueError('Independent parent audit requires the bound original graph')
        runtime.model.requires_grad_(False)
        versions = [(value, value._version) for value in runtime.model.parameters()]
        row = parent['groups']['new_training'][0]
        cache = capture(runtime, row)
        if any(not torch.equal(cache[key], parent['data']['caches'][row['id']][key]) for key in ('ids', 'labels')):
            raise ValueError('Independent parent audit TRAIN tokens differ')
        delta = parent['checkpoint']['delta'].to(runtime.device).detach()
        report['audit'] = parent_equivalence(runtime, cache, delta)
        report['original_parameters_unchanged'] = ([id(value) for value in runtime.model.parameters()] ==
            [id(value) for value, _ in versions] and all(value._version == version for value, version in versions))
        report['parent_delta_unchanged'] = torch.equal(delta.cpu(), parent['checkpoint']['delta'])
        if not report['original_parameters_unchanged'] or not report['parent_delta_unchanged']:
            raise ValueError('Independent parent audit changed a frozen tensor')
        report['status'] = 'parent_math_verified'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if runtime is not None:
                try:
                    report['mps_driver_bytes_before_release'] = torch.mps.driver_allocated_memory()
                finally:
                    runtime._release_model()
                    report['model_lock_released'] = runtime._model_process_lock is None
        except BaseException as error:
            report.update(status='failed_release', release_error=str(error))
            raise
        finally:
            report.update(elapsed_seconds=time.perf_counter() - began,
                          lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            private_json(destination / 'result.json', report)
            print(json.dumps({'output': str(destination), **report}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Independent real parent math audit; no training')
    parser.add_argument('parent_round', type=Path)
    parser.add_argument('parent_verification', type=Path)
    args = parser.parse_args()
    audit_parent(args.parent_round, args.parent_verification)
