"""Independent fixed450 HF residual diagnostic; never adoption or training.

Execution is explicit (--execute), after the training owner has confirmed exit.
Model identity reads the existing manifest; it is not full weight provenance.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import tempfile
import time

import psutil
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, __version__ as transformers_version

from verify_local_review import model_identity
from probe_core_constraint import probes
from luban.backbone import load_backbone, text_backbone
from luban.core_update import apply_core_residual, validate_basis
from luban.context import check_input, window
from luban.evolution import _answer_logits
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.process_lock import ModelProcessLock
from luban.residual_core import (ResidualProjection, install_residual_core, load_residual_core,
                                 save_residual_core, uninstall_residual_core, _slot)
from luban.retention import messages, provenance
from luban.settings import private_json
from verify_cached_batch_training import _labels

GENERATION_SETTINGS = {'context_mode': 'auto', 'context_tokens': 262144}
SUCCESSOR_PROTOCOL = 'preservation-first-residual-fixed450-v4-successor'


def read(path):
    return json.loads(Path(path).read_text())


def inherited_loss_tolerance(identifier, bound, successor):
    """Allow only fixed FP32 measurement noise on newly inherited entries."""
    if successor is None or identifier in set(successor.get('original_fixed', {}).get('ids', ())):
        return 0.0
    return 64 * torch.finfo(torch.float32).eps * max(1., abs(float(bound)))


def finite(value, dtype, shape, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != tuple(shape)
            or value.is_meta or not bool(torch.isfinite(value).all())):
        raise ValueError(f'Invalid finite {dtype} {name}')


def completed(result):
    protocol = result.get('protocol')
    if (result.get('status') != 'fixed450_complete_research_only'
            or type(result.get('optimizer_steps')) is not int or result['optimizer_steps'] != 450
            or result.get('steps_required') != 450 or result.get('seed') != 42
            or result.get('original_runtime_weight_unchanged') is not True
            or result.get('production_adoption') is not False
            or result.get('production_weights_changed') is not False
            or (result.get('final_loss_guard') or {}).get('rejected') is not False
            or protocol not in ('preservation-first-residual-fixed450-v1',
                'preservation-first-residual-fixed450-v2-expanded',
                'preservation-first-residual-fixed450-v3-stable',
                'preservation-first-residual-fixed450-v3-expanded-stable', SUCCESSOR_PROTOCOL)
            or result.get('precision') != 'BF16 frozen model and deployed tail; FP32 unmerged delta/Adam'
            or type(result.get('candidate_qualified')) is not bool):
        raise ValueError('A completed, unchanged, protected fixed450 round is required')
    if protocol != 'preservation-first-residual-fixed450-v1':
        expanded = 'expanded' in protocol or protocol == SUCCESSOR_PROTOCOL
        scales = ([1.] if expanded else []) + [.25, .0625, .015625, .00390625]
        if result.get('expanded_span') is not expanded or result.get('scales') != scales:
            raise ValueError('Bound fixed450 span recipe or scales differ')
    elif result.get('expanded_span', False) is not False:
        raise ValueError('Legacy v1 did not expand its protected span')
    solver = 'fp32-feasible-first-double-projection-v1' if 'v3-' in protocol or protocol == SUCCESSOR_PROTOCOL else None
    if result.get('constraint_solver') != solver:
        raise ValueError('Bound fixed450 constraint solver differs')


def validate_expanded_span(result, checkpoint, original, fixed, artifact):
    """An expanded round must retain its original span and its actual delta bound."""
    validate_basis(original)
    basis = artifact.get('basis')
    rank = result.get('basis', {}).get('rank')
    if type(rank) is not int:
        raise ValueError('Missing expanded basis rank')
    finite(basis, torch.float32, (original.shape[0], rank), 'expanded basis')
    geometry = validate_basis(basis)
    extension = artifact.get('extension')
    if (not isinstance(extension, dict) or extension != result.get('extension')
            or type(extension.get('added_rank')) is not int
            or extension['added_rank'] < 0 or rank != original.shape[1] + extension['added_rank']
            or fixed.get('basis_rank') != rank):
        raise ValueError('Expanded basis metadata differs')
    old_residual = float((original - basis @ (basis.T @ original)).abs().max())
    if old_residual > 2e-5:
        raise ValueError('Expanded basis dropped original protection')
    delta = checkpoint['delta']
    residual = float((delta @ basis).norm())
    tolerance = 64 * torch.finfo(torch.float32).eps * max(1., float(delta.double().norm()))
    if not torch.isfinite(torch.tensor(residual)) or residual > tolerance:
        raise ValueError('Saved delta is outside its expanded protected span')
    return {**geometry, 'rank': rank, 'old_span_residual': old_residual,
            'delta_subspace_residual': residual, 'delta_subspace_tolerance': tolerance}


def validate_checkpoint(result, checkpoint, reference, fixed, identity):
    """Validate one real Adam parameter; shape follows its bound BF16 reference."""
    completed(result)
    if (result.get('identity') != identity or checkpoint.get('identity') != identity
            or type(checkpoint.get('steps')) is not int or checkpoint['steps'] != 450
            or checkpoint.get('production_adoption') is not False
            or type(checkpoint.get('accepted_updates')) is not int
            or not 0 <= checkpoint['accepted_updates'] <= 450
            or checkpoint['accepted_updates'] != result.get('accepted_updates')):
        raise ValueError('Round/checkpoint identity or progress differs')
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError('Missing original projection reference')
    finite(reference, torch.bfloat16, reference.shape, 'reference')
    finite(checkpoint.get('delta'), torch.float32, reference.shape, 'delta')
    if result['protocol'] == SUCCESSOR_PROTOCOL:
        finite(checkpoint.get('increment'), torch.float32, reference.shape, 'successor increment')
        if not isinstance(checkpoint.get('parent_binding'), dict):
            raise ValueError('Successor checkpoint needs an explicit parent binding')
    elif 'increment' in checkpoint or 'parent_binding' in checkpoint:
        raise ValueError('Legacy checkpoint cannot claim successor state')
    baseline = checkpoint.get('fixed_protection_losses')
    finite(baseline, torch.float32, (len(fixed['ids']),), 'fixed protection baseline')
    declared = torch.tensor(fixed['losses'], dtype=torch.float32)
    if not torch.equal(baseline, declared) or bool((baseline < 0).any()):
        raise ValueError('Fixed protection baselines differ')
    optimizer = checkpoint.get('optimizer') or {}
    groups, states = optimizer.get('param_groups', []), optimizer.get('state', {})
    expected_adam = {'lr': 1e-5, 'eps': 1e-6, 'betas': (0.9, 0.999), 'weight_decay': 0,
        'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False,
        'differentiable': False, 'fused': None, 'decoupled_weight_decay': False}
    if (len(groups) != 1 or len(groups[0].get('params', [])) != 1
            or set(states) != set(groups[0]['params'])
            or set(groups[0]) != set(expected_adam) | {'params'}
            or any(groups[0][key] != value for key, value in expected_adam.items())):
        raise ValueError('Expected one bound residual Adam state')
    state = states[groups[0]['params'][0]]
    if set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
        raise ValueError('Unexpected Adam state')
    finite(state['step'], torch.float32, (), 'Adam step')
    if float(state['step']) != 450:
        raise ValueError('Adam step must be 450')
    for key in ('exp_avg', 'exp_avg_sq'):
        finite(state[key], torch.float32, reference.shape, key)
    if bool((state['exp_avg_sq'] < 0).any()):
        raise ValueError('Negative Adam second moment')


def load_round(directory, identity, *, ancestry=()):
    directory = Path(directory).resolve()
    if str(directory) in {str(Path(path).resolve()) for path in ancestry}:
        raise ValueError('Cyclic successor round ancestry')
    result = read(directory / 'result.json')
    completed(result)  # Reject unfinished training before even reading recovery tensors.
    settings = read(directory / 'home/settings.json')
    if {key: settings.get(key) for key in GENERATION_SETTINGS} != GENERATION_SETTINGS:
        raise ValueError('Original generation context settings differ')
    checkpoint = torch.load(directory / 'recovery.pt', map_location='cpu', weights_only=True)
    preparation = read(directory.parent / 'preparation.json')
    if preparation.get('status') != 'prepared_no_optimizer_updates' or preparation.get('identity') != identity:
        raise ValueError('Successful preparation must bind this original identity')
    groups = read(directory.parent / 'material.json')['groups']
    if any(not groups.get(key) for key in ('new_training', 'new_validation', 'replay_training')) or groups.get('replay_validation'):
        raise ValueError('Expected complete original residual material groups')
    rows = groups['new_training'] + groups['new_validation'] + groups['replay_training'] + probes()
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids) or len(groups['new_training']) <= 31:
        raise ValueError('Material IDs or the three preregistered TRAIN audits differ')
    fixed = read(directory / 'fixed_protection.json')
    successor = result['protocol'] == SUCCESSOR_PROTOCOL
    if not successor and fixed['ids'] != [row['id'] for row in probes() + groups['replay_training']]:
        raise ValueError('Fixed protection order differs')
    data = torch.load(directory.parent / 'prepared.pt', map_location='cpu', weights_only=True)
    validate_checkpoint(result, checkpoint, data['reference'], fixed, identity)
    if data['reference'].shape != (2048, 6144):
        raise ValueError('Expected the real 2B final down projection')
    # Full-model verification needs IDs/labels, not cached prefix activations.
    # Drop activations before parent recursion so ancestors do not keep them resident.
    data = {'reference': data['reference'], 'basis': data['basis'], 'caches': {
        key: {name: value[name] for name in ('ids', 'labels')} for key, value in data['caches'].items()}}
    span_check, successor_data, basis = None, {}, data['basis']
    if result.get('expanded_span') is True:
        artifact = torch.load(directory / 'expanded-basis.pt', map_location='cpu', weights_only=True)
        basis = artifact.get('basis')
        if successor:
            from residual_successor import validate_successor_contract
            successor_data = validate_successor_contract(directory, result, checkpoint, fixed, artifact,
                                                          identity, ancestry=ancestry)
            span_check = successor_data['basis_validation']
        else:
            span_check = validate_expanded_span(result, checkpoint, data['basis'], fixed, artifact)
    data.update(successor_data, basis=basis, span_verification=span_check)
    baseline, final = read(directory / 'baseline_generation.json'), read(directory / 'final_generation.json')
    if successor and baseline != successor_data['parent_bundle']['final']:
        raise ValueError('Successor baseline must preserve every complete parent generation record')
    if [row['id'] for row in baseline] != ids or [row['id'] for row in final] != ids:
        raise ValueError('Generation records must contain every bound material row in order')
    for record in baseline + final:
        raw = record.get('raw_completion_token_ids')
        if (not isinstance(raw, list) or not raw or any(type(token) is not int or token < 0 for token in raw)
                or record.get('branch_enabled') is not False or record.get('reference_used') is not False
                or record.get('thinking') != 'off' or record.get('do_sample') is not False):
            raise ValueError('Invalid closed-book raw generation record')
    if span_check is not None and not successor:
        original = {row['id']: row for row in baseline}
        count = sum(len(original[row['id']]['raw_completion_token_ids']) for row in probes())
        if any(result['extension'].get(key) != count
               for key in ('teacher_forced_positions', 'actual_kv_positions')):
            raise ValueError('Expanded span predictor counts differ from original TRAIN trajectories')
    return groups, rows, result, checkpoint, data, fixed, baseline, final


def generation_context(model):
    _, config = text_backbone(model)
    device = str(next(model.parameters()).device)
    status = window(GENERATION_SETTINGS, config, psutil.virtual_memory().available,
                    device=device, dtype_bytes=2)
    status['current_limit'] = min(status['current_limit'], 32768)
    return status


def prompt_tokens(tokenizer, row, maximum, status):
    ids = tokenizer.apply_chat_template(messages(row), add_generation_prompt=True,
        tokenize=True, return_tensors='pt', enable_thinking=False)
    ids = ids if isinstance(ids, torch.Tensor) else ids['input_ids']
    if (ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0
            or type(maximum) is not int or not 0 < maximum <= 32768):
        raise ValueError('Invalid complete generation prompt or output budget')
    check_input(ids.shape[1], maximum, status)
    return ids


def gold_sequence(tokenizer, row, cache):
    prefix = tokenizer.apply_chat_template(messages(row), tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    first = tokenizer.encode(prefix, add_special_tokens=False)
    full = first + tokenizer.encode(row['answer'], add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = torch.tensor([full], dtype=torch.long)
    labels = ids.clone()
    labels[:, :len(first)] = -100
    if ids.shape[1] > 512:
        raise ValueError('Gold sequence exceeds 512 tokens')
    _labels(ids, labels, {tokenizer.eos_token_id})
    if not torch.equal(ids, cache['ids']) or not torch.equal(labels, cache['labels']):
        raise ValueError('Prepared gold IDs/labels differ from the loaded tokenizer and context')
    return cache['ids'], cache['labels']


def probe_sequence(tokenizer, model, row, observed):
    check = validate_generation_check(provenance(row).get('generation_check'), default_expected=row['answer'])
    if observed.get('complete') is not True or observed.get('max_new_tokens') != check['max_new_tokens']:
        raise ValueError('Probe needs its complete bound original generation')
    prefix = prompt_tokens(tokenizer, row, check['max_new_tokens'], generation_context(model))
    raw = observed['raw_completion_token_ids']
    eos = model.generation_config.eos_token_id
    if raw[-1] not in (eos if isinstance(eos, (list, tuple)) else [eos]) or len(raw) > check['max_new_tokens']:
        raise ValueError('Probe completion lacks its actual generation EOS or exceeds its budget')
    ids = torch.cat((prefix, torch.tensor([raw], dtype=torch.long)), dim=1)
    labels = ids.clone()
    labels[:, :prefix.shape[1]] = -100
    return ids, labels


@torch.no_grad()
def logits(model, ids):
    values = model(ids.to(next(model.parameters()).device), use_cache=False).logits.detach().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Nonfinite full-model logits')
    return values


def audit_carrier(model, delta, identity, sequences, artifact):
    """Audit all rows per phase; leave the restored carrier installed on success."""
    decoder, _ = text_backbone(model)
    mlp = decoder.layers[-1].mlp
    original = mlp.down_proj
    delta_device = delta.to(original.weight.device)
    checks = [{'id': identifier} for identifier, _ in sequences]
    handle, success = None, False
    try:
        baseline = [logits(model, ids) for _, ids in sequences]
        install_residual_core(model, torch.zeros_like(delta_device), base_identity=identity, base_core=None)
        for item, (_, ids), before in zip(checks, sequences, baseline):
            item['zero_exact'] = bool(torch.equal(logits(model, ids), before))
        del baseline, before
        uninstall_residual_core(model)
        handle = original.register_forward_hook(
            lambda module, inputs, output: apply_core_residual(inputs[0], output, delta_device))
        try:
            expected = [logits(model, ids) for _, ids in sequences]
        finally:
            handle.remove()
            handle = None
        install_residual_core(model, delta_device, base_identity=identity, base_core=None)
        save_residual_core(model, artifact)
        uninstall_residual_core(model)
        restored = load_residual_core(model, artifact, base_identity=identity, base_core=None)
        delta_exact = bool(torch.equal(restored.delta.cpu(), delta.cpu()))
        for item, (_, ids), before in zip(checks, sequences, expected):
            item.update(research_hook_logits_exact=bool(torch.equal(logits(model, ids), before)),
                        serialized_delta_exact=delta_exact)
        if not all(item[key] for item in checks for key in
                   ('zero_exact', 'research_hook_logits_exact', 'serialized_delta_exact')):
            raise ValueError(f'Carrier full-graph equivalence failed: {checks}')
        success = True
    finally:
        if handle is not None:
            handle.remove()
        if not success and isinstance(mlp.down_proj, ResidualProjection):
            uninstall_residual_core(model)
    return checks


@torch.no_grad()
def generate(model, tokenizer, row, expected):
    check = validate_generation_check(provenance(row).get('generation_check'), default_expected=row['answer'])
    maximum = check['max_new_tokens']
    ids = prompt_tokens(tokenizer, row, maximum, generation_context(model)).to(next(model.parameters()).device)
    if expected.get('max_new_tokens') != maximum:
        raise ValueError('Bound generation budget differs')
    output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=maximum,
        do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    raw = output[0, ids.shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    complete = bool(raw and raw[-1] in (eos if isinstance(eos, (list, tuple)) else [eos]))
    answer = tokenizer.decode(raw, skip_special_tokens=True).strip()
    verdict = evaluate_generation(answer, check, complete)
    return {'id': row['id'], 'raw_completion_token_ids': raw, 'answer': answer,
            'tokens_match_round': raw == expected['raw_completion_token_ids'],
            'validator_matches_round': all(verdict[key] == expected.get(key)
                for key in ('passed', 'correct', 'complete', 'forbidden_found')), **verdict}


def capture_greedy_trajectory(model, tokenizer, row, expected):
    """Record actual prefill/decode predictor states, not teacher-forced ones.

    Research-only TRAIN capture. The unchanged last projection is downstream
    of the final attention cache; each recorded position predicts one raw token.
    This is a finite trajectory record, not proof about another prompt or graph.
    """
    if row.get('purpose') != 'training' or model.training:
        raise ValueError('Trajectory capture requires TRAIN and an eval model')
    _, projection = _slot(model)
    if isinstance(projection, ResidualProjection):
        raise ValueError('Original trajectory capture requires the unmodified projection')
    inputs, outputs, lengths = [], [], []

    def capture(module, arguments, output):
        value = arguments[0]
        if (value.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                or value.ndim != 3 or value.shape[0] != 1 or not value.shape[1]
                or value.shape[-1] != projection.in_features
                or output.shape != (*value.shape[:-1], projection.out_features)
                or (lengths and value.shape[1] != 1)):
            raise ValueError('Expected BF16 single-sequence prefill then cached decode')
        x, y = value[0, -1].detach().cpu().clone(), output[0, -1].detach().cpu().clone()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError('Nonfinite generation trajectory')
        inputs.append(x)
        outputs.append(y)
        lengths.append(value.shape[1])

    handle = projection.register_forward_hook(capture)
    try:
        generated = generate(model, tokenizer, row, expected)
    finally:
        handle.remove()
    raw = generated['raw_completion_token_ids']
    if (not generated['tokens_match_round'] or not generated['complete']
            or not generated['validator_matches_round'] or len(inputs) != len(raw) or not inputs):
        raise ValueError('Actual generation must reproduce the complete bound raw trajectory')
    return {'id': row['id'], 'messages': messages(row), 'raw_completion_token_ids': raw,
            'inputs': torch.stack(inputs), 'outputs': torch.stack(outputs), 'forward_lengths': lengths}


@torch.no_grad()
def protection_loss(model, ids, labels):
    device = next(model.parameters()).device
    labels = labels.to(device)
    output = model(ids.to(device), use_cache=False).logits
    value = F.cross_entropy(_answer_logits(output, labels), labels[:, 1:][labels[:, 1:] != -100])
    if not bool(torch.isfinite(value)):
        raise ValueError('Nonfinite full-model protection loss')
    return float(value)


def successor_sequences(tokenizer, model, rows, caches, successor):
    """Rebuild every saved protection tensor from the actual template and gold/raw."""
    by_id = {row['id']: row for row in rows if row.get('purpose') == 'training'}
    result = []
    for entry in successor['protection_entries']:
        row = by_id.get(entry['row_id'])
        if row is None:
            raise ValueError('Successor protection must refer to a complete TRAIN row')
        if entry['kind'] in ('original_gold', 'mastered_gold'):
            ids, labels = gold_sequence(tokenizer, row, caches[row['id']])
        elif entry['kind'] in ('original_raw', 'mastered_raw'):
            ids, labels = probe_sequence(tokenizer, model, row, successor['protected_raw'][row['id']])
        else:
            raise ValueError('Unknown successor protection kind')
        if not torch.equal(ids, entry['ids']) or not torch.equal(labels, entry['labels']):
            raise ValueError('Successor protection tensors differ from tokenizer-bound gold/raw')
        result.append((entry['id'], ids, labels))
    return result


def successor_retention(successor, by_id):
    checks = {'successor_protected_raw_exact': all(
        by_id[identifier]['raw_completion_token_ids'] == prior['raw_completion_token_ids']
        for identifier, prior in successor['protected_raw'].items()),
        'original_correct_replay_retained': all(by_id[identifier]['passed']
            for identifier in successor['original_correct_replay_ids']),
        'parent_correct_replay_retained': all(by_id[identifier]['passed']
            for identifier in successor['parent_correct_replay_ids']),
        'previously_mastered_train_retained': all(by_id[identifier]['passed']
            for identifier in successor['mastered_train_ids'])}
    checks['successor_retention_passed'] = all(checks.values())
    return checks


@torch.no_grad()
def audit_parent_zero(model, parent_delta, sequences, protection=()):
    """One parent-D phase and one parent-D+zero-E phase; no parent re-projection."""
    mlp, projection = _slot(model)
    if isinstance(projection, ResidualProjection) or model.training:
        raise ValueError('Parent audit requires the frozen original projection')
    finite(parent_delta, torch.float32, projection.weight.shape, 'parent delta')
    parent = parent_delta.to(projection.weight.device)
    active = [parent]
    handle = projection.register_forward_hook(
        lambda module, arguments, output: apply_core_residual(arguments[0], output, active[0]))
    try:
        before = [logits(model, ids) for _, ids in sequences]
        losses = [protection_loss(model, ids, labels) for _, ids, labels in protection]
        active[0] = parent + torch.zeros_like(parent)
        checks = [{'id': identifier, 'parent_plus_zero_logits_exact': bool(torch.equal(logits(model, ids), expected))}
                  for (identifier, ids), expected in zip(sequences, before)]
        if not all(check['parent_plus_zero_logits_exact'] for check in checks):
            raise ValueError('Parent plus zero increment changed full-graph logits')
        return {'logits': checks, 'protection_losses': losses}
    finally:
        handle.remove()


def new_output(directory):
    volume = Path('/Volumes/移动数据库')
    experiments = volume / 'lubanllm/experiments'
    if not volume.is_mount() or not directory.is_relative_to(experiments):
        raise ValueError('Use the mounted external research experiment directory')
    return Path(tempfile.mkdtemp(prefix='independent-residual-verification-', dir=directory.parent))


def run(directory, model_path):
    directory, model_path = Path(directory).resolve(), Path(model_path).resolve()
    destination = new_output(directory)
    print(json.dumps({'output': str(destination)}, ensure_ascii=False), flush=True)
    began = time.perf_counter()
    report = {'status': 'preparing', 'production_adoption': False, 'teacher_calls': 0,
        'training_performed': False, 'model_path': str(model_path), 'round_directory': str(directory),
        'seed': 42, 'device': 'mps', 'base_core': None, 'base_precision': 'BF16', 'delta_precision': 'FP32',
        'hardware': platform.platform(), 'model_lock_acquired': False, 'carrier_consistent': False,
        'torch_version': torch.__version__, 'transformers_version': transformers_version,
        'memory_measurement': 'Darwin lifetime peak RSS bytes; MPS driver bytes separately, overlapping',
        'scope': 'Diagnostic delta export and finite full-model checks; no adoption or generalization proof'}
    private_json(destination / 'result.json', report)
    lock = model = decoder = projection = versions = None
    used_mps = False
    try:
        identity = model_identity(model_path)
        groups, rows, source, checkpoint, data, fixed, baseline, expected = load_round(directory, identity)
        successor = data.get('successor')
        report.update(identity=identity, source_candidate_qualified=source['candidate_qualified'],
            counts={key: len(value) for key, value in groups.items()}, generation_count=len(rows),
            protection_count=len(fixed['ids']), source_protocol=source['protocol'],
            training_span_verification=data.get('span_verification'))
        private_json(destination / 'result.json', report)
        lock = ModelProcessLock(identity, 'mps')
        report['model_lock_acquired'] = True
        if not torch.backends.mps.is_available():
            raise RuntimeError('This frozen real-model protocol requires MPS')
        used_mps = True
        torch.manual_seed(42)
        phase = time.perf_counter()
        model = load_backbone(model_path, torch.bfloat16, 'mps', cold=False)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        decoder, config = text_backbone(model)
        projection = decoder.layers[-1].mlp.down_proj
        if (config.hidden_size, config.intermediate_size, config.num_hidden_layers) != (2048, 6144, 24):
            raise ValueError('Expected the original real 2B Qwen3.5 graph')
        if not torch.equal(projection.weight.detach().cpu(), data['reference']):
            raise ValueError('Original last down projection differs from the prepared reference')
        versions = [(value, value._version) for value in model.parameters()]
        report['original_projection_exact'] = True
        report['model_load_seconds'] = time.perf_counter() - phase
        status = generation_context(model)
        totals = []
        for row in rows:
            maximum = validate_generation_check(provenance(row).get('generation_check'),
                                                default_expected=row['answer'])['max_new_tokens']
            totals.append(prompt_tokens(tokenizer, row, maximum, status).shape[1] + maximum)
        report['generation_prompt_preflight'] = {'rows': len(rows), 'maximum_prompt_plus_reserve': max(totals),
            'context': status, 'truncation': False, 'training_gold_limit_separate': 512}
        original = {row['id']: row for row in baseline}
        trajectories = [capture_greedy_trajectory(model, tokenizer, row, original[row['id']]) for row in probes()]
        trajectory_path = destination / 'original-probe-kv-trajectories.pt'
        torch.save({'identity': identity, 'base_core': None, 'precision': 'BF16', 'rows': trajectories,
                    'scope': 'Four original complete greedy TRAIN trajectories; no candidate updates'}, trajectory_path)
        report['original_probe_kv_capture'] = {'path': str(trajectory_path),
            'predictor_counts': {row['id']: len(row['raw_completion_token_ids']) for row in trajectories},
            'raw_tokens_match_training_baseline': True, 'used_for_training_here': False}
        del trajectories
        audit_rows = [groups['new_training'][0], groups['new_training'][31], groups['replay_training'][0]]
        sequences = [(row['id'], gold_sequence(tokenizer, row, data['caches'][row['id']])[0]) for row in audit_rows]
        successor_protection = None
        if successor is not None:
            successor_protection = successor_sequences(tokenizer, model, rows, data['caches'], successor)
            report['successor_parent_checks'] = audit_parent_zero(model, data['parent_delta'], sequences,
                                                                 successor_protection)
            parent_losses = torch.tensor(report['successor_parent_checks']['protection_losses'], dtype=torch.float32)
            if not torch.equal(parent_losses, torch.tensor(successor['parent_losses'], dtype=torch.float32)):
                raise ValueError('Saved parent protection losses differ from the actual parent graph')
            expected_bounds = torch.tensor([min(successor['inherited_bounds'].get(identifier, value), value)
                for (identifier, _, _), value in zip(successor_protection, parent_losses.tolist())], dtype=torch.float32)
            if not torch.equal(expected_bounds, checkpoint['fixed_protection_losses']):
                raise ValueError('Successor fixed bounds differ from inherited and actual parent bounds')
            report.update(parent_binding=checkpoint['parent_binding'],
                successor_round_index=successor['round_index'],
                cumulative_mastered_train_ids=successor['mastered_train_ids'],
                successor_protection_tokenizer_bound=True, successor_parent_losses_exact=True)
        artifact = destination / 'diagnostic-residual.safetensors'
        report['carrier_checks'] = audit_carrier(model, checkpoint['delta'], identity, sequences, artifact)
        report['diagnostic_delta_artifact'] = str(artifact)
        report['carrier_consistent'] = True
        report['preparation_seconds'] = time.perf_counter() - began
        private_json(destination / 'result.json', report)
        phase = time.perf_counter()
        generated = []
        for index, (row, prior) in enumerate(zip(rows, expected), 1):
            generated.append(generate(model, tokenizer, row, prior))
            if index % 10 == 0 or index == len(rows):
                print(json.dumps({'stage': 'generation', 'completed': index, 'total': len(rows)}), flush=True)
        report['generation_seconds'] = time.perf_counter() - phase
        private_json(destination / 'generation.json', generated)
        losses = []
        phase = time.perf_counter()
        probe_ids = {row['id'] for row in probes()}
        protection = (successor_protection if successor is not None else [
            (row['id'], *(probe_sequence(tokenizer, model, row, original[row['id']]) if row['id'] in probe_ids
                else gold_sequence(tokenizer, row, data['caches'][row['id']])))
            for row in probes() + groups['replay_training']])
        for (identifier, ids, labels), bound in zip(protection, fixed['losses']):
            value = protection_loss(model, ids, labels)
            tolerance = inherited_loss_tolerance(identifier, bound, successor)
            excess = value - bound - tolerance
            losses.append({'id': identifier, 'fixed_baseline': bound, 'full_model_loss': value,
                           'tolerance': tolerance, 'excess_over_tolerance': excess,
                           'nonincreasing': excess <= 0})
        private_json(destination / 'protection.json', losses)
        report['protection_seconds'] = time.perf_counter() - phase
        uninstall_residual_core(model)
        by_id = {row['id']: row for row in generated}
        correct_replay = (successor['parent_correct_replay_ids'] if successor is not None else
            [row['id'] for row in baseline if row['id'] in {item['id'] for item in groups['replay_training']}
             and row['passed']])
        report.update(generation_tokens_exact=all(row['tokens_match_round'] for row in generated),
            generation_validators_exact=all(row['validator_matches_round'] for row in generated),
            full_model_loss_guard_passed=all(row['nonincreasing'] for row in losses),
            new_counts={key: {'total': len(groups[key]), 'passed': sum(by_id[row['id']]['passed'] for row in groups[key])}
                        for key in ('new_training', 'new_validation')},
            original_parameter_versions_unchanged=(all(value._version == version for value, version in versions)
                and [id(value) for value in model.parameters()] == [id(value) for value, _ in versions]),
            probe_tokens_unchanged=all(by_id[row['id']]['raw_completion_token_ids'] ==
                original[row['id']]['raw_completion_token_ids'] for row in probes()),
            baseline_correct_replay=len(correct_replay),
            retained_correct_replay=sum(by_id[identifier]['passed'] for identifier in correct_replay))
        report['new_knowledge_mastered'] = all(row['passed'] == row['total'] for row in report['new_counts'].values())
        if successor is not None:
            report['training_targets_complete'] = all(by_id[row['id']]['passed'] for row in groups['new_training'])
            report['parent_selection_scope'] = 'TRAIN only; existing TRAIN+DEV task qualification stays unchanged'
            report['mastered_train_ids_after_round'] = [row['id'] for row in groups['new_training'] if by_id[row['id']]['passed']]
            report.update(successor_retention(successor, by_id))
            report['inherited_raw_unchanged'] = report['successor_protected_raw_exact']
        report['status'] = ('diagnostic_verification_passed' if report['generation_tokens_exact']
            and report['generation_validators_exact'] and report['full_model_loss_guard_passed']
            and report['original_parameter_versions_unchanged']
            and (successor is None or report['successor_retention_passed'])
            else 'diagnostic_verification_failed')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if model is not None:
                decoder, _ = text_backbone(model)
                if isinstance(decoder.layers[-1].mlp.down_proj, ResidualProjection):
                    uninstall_residual_core(model)
            model = decoder = projection = versions = None
            gc.collect()
            report['lifetime_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if used_mps:
                report['mps_driver_bytes_before_empty_cache'] = torch.mps.driver_allocated_memory()
                torch.mps.empty_cache()
        except BaseException as error:
            report.update(status='failed_cleanup', cleanup_error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if lock is not None:
                lock.close()
            report.update(model_lock_released=lock is not None and lock.fd is None, elapsed_seconds=time.perf_counter() - began)
            private_json(destination / 'result.json', report)
    return destination, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round_directory', type=Path)
    parser.add_argument('model_path', type=Path)
    parser.add_argument('--execute', action='store_true', help='Run only after the training owner has confirmed it stopped')
    args = parser.parse_args()
    if not args.execute:
        print('Prepared code only. Actual execution requires --execute after training exit; no model loaded.')
    else:
        _, outcome = run(args.round_directory, args.model_path)
        raise SystemExit(0 if outcome['status'] == 'diagnostic_verification_passed' else 1)
