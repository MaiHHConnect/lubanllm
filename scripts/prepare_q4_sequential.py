"""Native sequential Q4 calibration with per-layer full-model TRAIN protection."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import resource
import struct
import time

import numpy as np
import torch
from safetensors import safe_open

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from capture_q4_native import capture_positions, integers, load_frozen
from luban.process_lock import ModelProcessLock
from prepare_q4_packed import (BLOCK, DAMPING, GROUP, gptq_quantize, inverse_factor,
                               metadata, projection_error, write_projection, decode)
from q4_sequential_protection import (GENERATION, check_proposal, establish_baseline,
                                      protection_cases)

ROOT = Path('/Volumes/移动数据库/lubanllm/experiments/q4-native-protection-20261016')
FRAME_MAGIC = b'LNQLA01\n'
STAGE_MAGIC = b'LQ4ST01\n'
HEADER_BYTES = 160


def make_plan(protocol, records):
    return {'schema': 'q4-sequential-plan-v1', 'manifest': MANIFEST, 'core': None,
            'eos_ids': protocol['eos_ids'], 'cases': [
                {'id': row['id'], 'split': row['split'], 'tokens': row['prompt_tokens'],
                 'forced_tokens': row['generated_tokens'][:-1], 'positions': capture_positions(row)}
                for row in records]}


def read_frame(record, case, layer, kind, native_root):
    width = 6144 if kind == 2 else 4096
    rows = len(case['tokens']) + len(case['forced_tokens'])
    path = Path(record.get('path', ''))
    category = {1: 'inputs', 2: 'act', 3: 'output'}[kind]
    expected_path = native_root / f'layer-{layer:02d}' / category / (case['id']+'.bin')
    expected = {'id': case['id'], 'split': case['split'], 'path': str(expected_path),
                'rows': rows, 'prompt_tokens': len(case['tokens']),
                'forced_tokens': len(case['forced_tokens']), 'row_values': width, 'kind': kind}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('sequential frame descriptor differs from frozen plan')
    if path != expected_path or path.is_symlink() or not path.is_file() or path.stat().st_size != HEADER_BYTES+rows*width*2:
        raise ValueError('sequential frame path or length mismatch')
    header = (FRAME_MAGIC+struct.pack('<6I', layer, kind, rows, width, len(case['tokens']),
                                     len(case['forced_tokens']))+MANIFEST.encode()+b'0'*64)
    with path.open('rb') as stream:
        if stream.read(HEADER_BYTES) != header:
            raise ValueError('sequential frame binding/header mismatch')
    values = np.memmap(path, dtype='<u2', mode='r', offset=HEADER_BYTES, shape=(rows, width))
    if np.any((values & 0x7f80) == 0x7f80):
        raise ValueError('nonfinite sequential activation')
    return values


def selected_inputs(reply, plan, layer, kind, native_root, save_to=None):
    records = reply.get('records')
    if not isinstance(records, list) or [r.get('id') for r in records] != [c['id'] for c in plan['cases']]:
        raise ValueError('complete ordered calibration and observation records required')
    collected = {split: {'prompt': [], 'assistant': []} for split in ('calibration', 'observation')}
    for record, case in zip(records, plan['cases'], strict=True):
        frame = read_frame(record, case, layer, kind, native_root)
        selected = frame[case['positions']]
        if kind == 1:
            selected = selected[:, 2048:]
        prompt_count = sum(position < len(case['tokens']) for position in case['positions'])
        collected[case['split']]['prompt'].append(selected[:prompt_count].copy())
        collected[case['split']]['assistant'].append(selected[prompt_count:].copy())
        del selected, frame
    result = {}
    name = f'layer{layer:02d}-'+('gateup' if kind == 1 else 'down')
    for split, phases in collected.items():
        raw = np.concatenate(phases['prompt']+phases['assistant'])
        if save_to is not None:
            with (save_to/f'{name}-{split}.bf16').open('xb') as stream:
                stream.write(raw.tobytes())
        result[split] = raw
    return result


def stage_header(layer, mask, revision):
    if (type(layer) is not int or not 0 <= layer < 24 or type(revision) is not int
            or revision != layer or type(mask) is not int or mask not in (3, 4) or (layer == 23 and mask == 4)):
        raise ValueError('invalid sequential stage binding')
    return (STAGE_MAGIC+struct.pack('<4I', layer, mask, mask.bit_count(), GROUP)
            +MANIFEST.encode()+b'0'*64+struct.pack('<Q', revision))


def asymmetric_errors(weights, codes, scales, biases, inputs, reference):
    rows = np.linspace(0, len(weights)-1, min(64, len(weights)), dtype=int)
    target = reference @ weights[rows].T
    actual = inputs @ decode(codes[rows], scales[rows], biases[rows]).T
    if not np.isfinite(target).all() or not np.isfinite(actual).all():
        raise ValueError('nonfinite asymmetric projection observation')
    delta = actual.astype(np.float64)-target.astype(np.float64)
    energy = float(np.square(target.astype(np.float64)).sum())
    return {'sampled_output_rows': rows.tolist(), 'input_rows': len(inputs),
            'relative_mse': float(np.square(delta).sum())/energy if energy else None,
            'mse': float(np.square(delta).mean()), 'max_abs': float(np.abs(delta).max()),
            'target': 'original BF16 weights on original BF16 reference input; pre-output-rounding'}


def quantize_stage(layer, kinds, values, readers, output, *, damping_fraction=DAMPING,
                   refit_grid=False, reference_values=None):
    mask = 3 if kinds == ('gate', 'up') else 4
    if kinds not in (('gate', 'up'), ('down',)):
        raise ValueError('gate/up must share one proposal; down follows quantized gate/up')
    path = output/f'layer-{layer:02d}-{ "gateup" if mask == 3 else "down"}.bin'
    converted = {split: (raw.astype(np.uint32) << 16).view(np.float32) for split, raw in values.items()}
    correction, drift = None, None
    if reference_values is not None:
        if not refit_grid or damping_fraction != DAMPING or set(reference_values) != set(converted):
            raise ValueError('asymmetric reference requires dynamic grids, original damping and both splits')
        drift = {}
        for split, value in reference_values.items():
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != converted[split].shape or not np.isfinite(value).all()):
                raise ValueError('reference must match current input rows/columns and finite FP32 values')
            if layer == 0 and mask == 3 and not np.array_equal(value, converted[split]):
                raise ValueError('first-layer gate/up must exactly match original BF16 reference input')
            difference = converted[split].astype(np.float64)-value.astype(np.float64)
            energy = float(np.square(value.astype(np.float64)).sum())
            delta_energy = float(np.square(difference).sum())
            drift[split] = {'rows': len(value), 'squared_l2_difference': delta_energy,
                            'reference_squared_l2': energy,
                            'relative_l2': float(np.sqrt(delta_energy/energy)) if energy else None}
    upper, hessian = inverse_factor(converted['calibration'], damping_fraction=damping_fraction)
    if refit_grid and hessian['dead_columns']:
        raise ValueError('dynamic-grid candidate requires observed input columns')
    if reference_values is not None:
        from prepare_q4_packed import asymmetric_factor
        correction = asymmetric_factor(reference_values['calibration'], converted['calibration'], upper, alpha=.25)
        if layer == 0 and mask == 3 and torch.count_nonzero(correction).item():
            raise ValueError('first-layer gate/up asymmetric correction must be exactly zero')
    records = []
    with path.open('xb') as stream:
        stream.write(stage_header(layer, mask, layer))
        for kind in kinds:
            n, k = (2048, 6144) if kind == 'down' else (6144, 2048)
            key = f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
            tensor = readers[key].get_tensor(key)
            if tensor.shape != (n, k) or tensor.dtype != torch.bfloat16:
                raise ValueError('unexpected original checkpoint projection')
            weights = tensor.float().numpy()
            rtn, nearest, scales, biases = metadata(weights)
            kwargs = {'correction': correction} if correction is not None else {}
            codes, fitted_scales, fitted_biases = gptq_quantize(weights, scales, biases, upper, refit_grid=refit_grid, **kwargs)
            dead = hessian['dead_columns']
            if dead and not np.array_equal(codes[:, dead], nearest[:, dead]):
                raise ValueError('feedback changed unobserved columns')
            write_projection(stream, codes, fitted_scales, fitted_biases)
            records.append({'layer': layer, 'kind': kind, 'shape': [n, k], 'hessian': hessian,
                'refit_grid': refit_grid,
                'changed_grid_groups': int(np.count_nonzero((fitted_scales != scales) | (fitted_biases != biases))),
                'changed_vs_rtn': int(np.count_nonzero(codes != rtn)),
                'errors': {method: {split: projection_error(weights, candidate, grid_scale, grid_bias, inputs)
                                    for split, inputs in converted.items()}
                           for method, candidate, grid_scale, grid_bias in
                           (('rtn', rtn, scales, biases), ('gptq', codes, fitted_scales, fitted_biases))}})
            if reference_values is not None:
                records[-1].update(asymmetric_alpha=.25, input_drift=drift,
                    original_input_zero_correction_control=(layer == 0 and mask == 3),
                    asymmetric_errors={method: {split: asymmetric_errors(weights, candidate, grid_scale, grid_bias,
                        inputs, reference_values[split]) for split, inputs in converted.items()}
                        for method, candidate, grid_scale, grid_bias in
                        (('rtn', rtn, scales, biases), ('gptq', codes, fitted_scales, fitted_biases))})
            del weights, tensor, rtn, nearest, scales, biases, codes, fitted_scales, fitted_biases
    del upper, converted
    gc.collect()
    return path, records


class SequentialWorker(MixedWorker):
    def __init__(self, binary, output, native_root, lock):
        super().__init__(binary, 'sequential', output, lock, precision='bf16', core=None,
                         sequential_directory=native_root)
        if (self.loaded.get('research_sequential') is not True or self.loaded.get('manifest_sha256') != MANIFEST
                or self.loaded.get('quantized_projections') != 0):
            self.close()
            raise ValueError('worker must begin original BF16 with sequential research enabled')
        self.serial = 0

    def control(self, op, layer, **extra):
        self.serial += 1
        request = {'id': f'seq-{self.serial:04d}-{op}-{layer}', 'seq_op': op,
                   'seq_layer': layer, 'seq_revision': layer, **extra}
        with (self.out/'control-requests.jsonl').open('a') as stream:
            stream.write(json.dumps(request)+'\n')
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        reply = self.receive(600)
        if reply.get('event') != 'sequential' or reply.get('id') != request['id'] or reply.get('op') != op:
            raise ValueError(f'invalid sequential response: {reply}')
        expected_revision = layer+1 if op == 'decide' else layer
        phases = {'begin': 'ready', 'inputs': 'inputs', 'gateup': 'act', 'down': 'candidate',
                  'verify': 'candidate', 'decide': 'complete' if layer == 23 else 'ready'}
        if reply.get('revision') != expected_revision or reply.get('layer') != layer or reply.get('phase') != phases[op]:
            raise ValueError('sequential worker phase/revision did not match transaction')
        return reply


def protection_generations(worker, cases, tokenizer, eos, label, vocab_size):
    outputs = []
    for case in cases:
        tokens = tokenizer.apply_chat_template(case['messages'], add_generation_prompt=True,
                                               tokenize=True, enable_thinking=False)
        if isinstance(tokens, Mapping):
            tokens = tokens['input_ids']
        if not integers(tokens, 0, vocab_size) or not tokens:
            raise ValueError('protected prompt must contain valid model token IDs')
        request_id = label+'-'+case['id']
        result = worker.request(tokens, request_id, max_tokens=GENERATION['max_tokens'],
                                temperature=0, seed=42, reuse=False, eos_ids=eos)
        generated = result.get('tokens')
        if (result.get('id') != request_id or result.get('prompt_tokens') != len(tokens)
                or result.get('cached_prompt_tokens') != 0 or not integers(generated, 0, vocab_size)
                or not generated or len(generated) > GENERATION['max_tokens']):
            raise ValueError('invalid protected generation result')
        complete = result.get('finish_reason') == 'stop' and generated[-1] in eos
        if result.get('complete') is not complete:
            raise ValueError('protected completion must reflect actual EOS')
        outputs.append({'id': case['id'], 'text': tokenizer.decode(generated, skip_special_tokens=True),
                        'complete': complete, 'tokens': generated, 'request_id': request_id,
                        'prompt_tokens': tokens, 'native_result': result})
    save(worker.out/(label+'-generations.json'), outputs)
    return outputs


def sparse_payloads(directory, accepted_layers):
    for layer in accepted_layers:
        for mask, kinds in ((3, ('gate', 'up')), (4, ('down',))):
            if layer == 23 and mask == 4:
                continue
            path = directory/f'layer-{layer:02d}-{ "gateup" if mask == 3 else "down"}.bin'
            with path.open('rb') as stream:
                if stream.read(HEADER_BYTES) != stage_header(layer, mask, layer):
                    raise ValueError('committed stage header changed')
                for kind in kinds:
                    n, k = (2048, 6144) if kind == 'down' else (6144, 2048)
                    raw = stream.read(8+n*k*3//4)
                    yield layer, kind, raw
                if stream.read(1):
                    raise ValueError('trailing committed stage payload')


def verify_reload(binary, output, lock, cases, tokenizer, eos, expected, artifact, vocab_size):
    """The deployment loader must reproduce the accepted resident model, once."""
    directory = output/'reload'; directory.mkdir()
    worker = MixedWorker(binary, 'candidate_mixed', directory, lock,
                         precision='mixed-q4-g16-packed', calibration_profile=output/'gptq-sequential.bin', core=None)
    try:
        if (worker.loaded.get('quantized_projections') != artifact['quantized_projections']
                or worker.loaded.get('manifest_sha256') != MANIFEST
                or worker.loaded.get('retained_original_weight_bytes') != 0):
            raise ValueError('independent packed reload binding/count/storage mismatch')
        actual = protection_generations(worker, cases, tokenizer, eos, 'reload', vocab_size)
        changes = [{'id': before['id'], 'fields': [key for key in ('id', 'tokens', 'complete', 'text', 'prompt_tokens')
                    if before[key] != after[key]]} for before, after in zip(expected, actual, strict=True)]
        changes = [row for row in changes if row['fields']]
        result = {'passed': not changes, 'cases': len(cases), 'changes': changes, 'loaded': worker.loaded,
                  'scope': 'Independent ordinary packed-v2 loader versus final protected resident16; not final product quality'}
        save(directory/'result.json', result)
        if changes:
            raise ValueError('independent packed reload changed protected model outputs')
    finally:
        worker.close()
    if worker.process.returncode != 0:
        raise RuntimeError('independent packed reload worker did not exit cleanly')
    return result


def compare_control(values, reference, layer, kind):
    name = f'layer{layer:02d}-'+('gateup' if kind == 1 else 'down')
    checks = []
    for split, actual in values.items():
        expected = np.fromfile(reference/f'{name}-{split}.bf16', dtype='<u2')
        if expected.size != actual.size or not np.array_equal(expected.reshape(actual.shape), actual):
            raise ValueError(f'BF16 sequential input differs from normal native capture: {name}/{split}')
        checks.append({'stream': name, 'split': split, 'rows': len(actual),
                       'values_compared': actual.size, 'exact': True})
    return checks


def regularization_source(directory, receipt, plan):
    prior = json.loads((directory/'result.json').read_text())
    layers = prior.get('accepted_layers', [])
    if (prior.get('completed') is not True or prior.get('receipt') != receipt or prior.get('plan') != plan
            or prior.get('mode') != 'convert' or prior.get('reload', {}).get('passed') is not True
            or not layers or any(type(layer) is not int or not 0 <= layer < 24 for layer in layers)
            or layers != sorted(set(layers)) or prior.get('layer_mask') != sum(1 << layer for layer in layers)
            or prior.get('artifact', {}).get('quantized_projections') != 3*len(layers)-(23 in layers)):
        raise ValueError('completed same-source conversion and exact nonempty fixed mask required')
    previous = json.loads((directory/'accepted-generations.json').read_text())
    # Reuse the original scored TRAIN contract; do not silently weaken its union.
    baseline = json.loads((directory/'protection-baseline.json').read_text())
    check_proposal(protection_cases(), baseline, previous, previous)
    return layers, previous


def fixed_protection(cases, baseline, previous, candidate, *, original_bf16_preservation=False):
    union = check_proposal(cases, baseline, previous, candidate)
    if not original_bf16_preservation:
        return union
    report = check_proposal(cases, baseline, baseline['generations'], candidate)
    report.update(protection_policy='original_bf16_preservation',
        prior_q4_correct_ids=union['previous_accepted_correct_ids'],
        prior_q4_only_correct_ids=sorted(set(union['previous_accepted_correct_ids'])-set(union['original_correct_ids'])),
        prior_q4_only_lost_ids=sorted(set(union['lost_accepted_correct_ids'])-set(union['original_correct_ids'])),
        prior_union_would_accept=union['accepted'],
        scope='New preregistered source-fidelity method: original BF16 correctness gates; prior Q4-only gains are diagnostic. No old rejection is reclassified.')
    return report


def exact_original_baseline(actual, source_directory):
    expected = json.loads((source_directory/'baseline-generations.json').read_text())
    keys = ('id', 'prompt_tokens', 'tokens', 'complete', 'text')
    if len(actual) != 16 or len(expected) != 16 or any(
            any(a[key] != b[key] for key in keys) for a,b in zip(actual,expected,strict=True)):
        raise ValueError('fresh original BF16 sixteen trajectories changed')


def validate_asymmetric_options(args):
    if args.asymmetric_reference and (args.mode != 'convert' or not args.refit_grid_from
            or not args.task_trajectories_from or args.regularize_from
            or args.refit_grid_from.resolve() != args.task_trajectories_from.resolve()
            or not args.asymmetric_reference.is_absolute()):
        raise ValueError('asymmetric reference requires convert, absolute reference and identical fixed/task source')


def reference_stage(reference, layer, kind, values):
    if reference is None:
        return None
    from capture_q4_task_reference import read_reference
    return {split: read_reference(reference, layer, kind, split, len(raw)) for split,raw in values.items()}


def run(args):
    validate_asymmetric_options(args)
    source_protocol, records = load_frozen(args.source)
    original_plan = make_plan(source_protocol, records)
    binary = args.binary.resolve()
    receipt = json.loads(Path(str(binary)+'.json').read_text())
    fixed_from = args.regularize_from or args.refit_grid_from
    refit_grid = args.refit_grid_from is not None
    # The augmented plan needs a new validator build; its mathematical bridge
    # is checked below by a new same-binary BF16 control against original traces.
    source_receipt = (json.loads((args.task_trajectories_from/'result.json').read_text())['receipt']
                      if args.task_trajectories_from else receipt)
    fixed_layers, previous = (regularization_source(fixed_from, source_receipt, original_plan)
                              if fixed_from else (None, None))
    if args.task_trajectories_from:
        from q4_task_trajectories import load_task_trajectories
        records = records + load_task_trajectories(args.task_trajectories_from, source_receipt, original_plan, source_protocol)
    plan = make_plan(source_protocol, records)
    reference = None
    if args.asymmetric_reference:
        from capture_q4_task_reference import load_reference
        reference = load_reference(args.asymmetric_reference, receipt, plan,
                                   task_source=args.task_trajectories_from, task_source_receipt=source_receipt)
    if args.mode == 'convert':
        control = json.loads((args.control/'result.json').read_text())
        if (control.get('completed') is not True or control.get('mode') != 'control'
                or control.get('receipt') != receipt or control.get('all_frozen_tokens_equal') is not True
                or control.get('streams_compared') != 47 or control.get('plan') != plan):
            raise ValueError('same-binary complete BF16 sequential control required before conversion')
        if args.task_trajectories_from and (control.get('task_trajectories_from') != str(args.task_trajectories_from)
                or control.get('task_source_receipt') != source_receipt
                or any(not control['decisions'][layer].get('normal_forward_control', {}).get('verified_values')
                       for layer in (0, 12, 23))):
            raise ValueError('augmented-plan BF16 control must bind task source and full-forward comparisons')
    args.output.mkdir(parents=True, exist_ok=False)
    native_root = args.output/'native'; native_root.mkdir()
    weights_dir = args.output/'weights'; weights_dir.mkdir()
    inputs_dir = args.output/'inputs'; inputs_dir.mkdir()
    plan_path = args.output/'plan.json'; save(plan_path, plan)
    cases = protection_cases()
    protocol = {'schema': 'q4-sequential-run-v1', 'mode': args.mode, 'binary': str(binary), 'receipt': receipt,
        'plan': plan, 'source': str(args.source), 'control': str(args.control) if args.control else None,
        'quantizer': {'group': GROUP, 'block': BLOCK, 'damping_fraction': DAMPING, 'static_bf16_grid': not refit_grid},
        'regularize_from': str(args.regularize_from) if args.regularize_from else None,
        'refit_grid_from': str(args.refit_grid_from) if refit_grid else None,
        'task_trajectories_from': str(args.task_trajectories_from) if args.task_trajectories_from else None,
        'task_source_receipt': source_receipt if args.task_trajectories_from else None,
        'asymmetric_reference': str(args.asymmetric_reference) if args.asymmetric_reference else None,
        'asymmetric_alpha': .25 if args.asymmetric_reference else None,
        'protection_policy': 'original_bf16_preservation' if args.asymmetric_reference else 'original_plus_prior_q4_union',
        'input_rows': {split: sum(len(case['positions']) for case in plan['cases'] if case['split'] == split)
                       for split in ('calibration', 'observation')},
        'added_task_rows': [{'id': row['id'], 'prompt': sum(p < len(row['prompt_tokens']) for p in capture_positions(row)),
                             'assistant': len(row['assistant_positions'])} for row in records[len(original_plan['cases']):]],
        'fixed_layers': fixed_layers, 'down_damping_fraction': .1 if args.regularize_from else DAMPING,
        'protection_cases': cases, 'generation': GENERATION, 'teacher_calls': 0,
        'product_adopted': False, 'original_weights_changed': False}
    save(args.output/'protocol.json', protocol)
    started = time.perf_counter()
    decisions, checks, accepted_layers = [], [], []
    worker = None
    lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        worker = SequentialWorker(binary, args.output, native_root, lock)
        save(args.output/'loaded.json', worker.loaded)
        worker.control('begin', 0, seq_plan=str(plan_path), seq_dry_run=args.mode == 'control')
        if args.mode == 'convert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
            accepted = protection_generations(worker, cases, tokenizer, plan['eos_ids'], 'baseline', source_protocol['vocab_size'])
            if args.asymmetric_reference:
                exact_original_baseline(accepted, args.task_trajectories_from)
            baseline = establish_baseline(cases, accepted)
            save(args.output/'protection-baseline.json', baseline)
        with ExitStack() as stack:
            readers = {}
            if args.mode == 'convert':
                for file in sorted(Path(MODEL).glob('*.safetensors')):
                    if file.name.startswith('.'):
                        continue
                    reader = stack.enter_context(safe_open(file, framework='pt', device='cpu'))
                    readers.update({key: reader for key in reader.keys()})
            for layer in range(24):
                gate = worker.control('inputs', layer)
                values = selected_inputs(gate, plan, layer, 1, native_root, inputs_dir if args.mode == 'convert' else None)
                quantized = []
                if args.mode == 'control':
                    reference_values = (selected_inputs({**gate, 'records': gate['records'][:20]}, original_plan,
                                                        layer, 1, native_root) if args.task_trajectories_from else values)
                    checks.extend(compare_control(reference_values, args.reference, layer, 1))
                    del reference_values
                    gate_kwargs = {}
                else:
                    path, stats = quantize_stage(layer, ('gate', 'up'), values, readers, weights_dir,
                                                 refit_grid=refit_grid,
                                                 reference_values=reference_stage(reference, layer, 1, values))
                    quantized.extend(stats); gate_kwargs = {'seq_payload': str(path)}
                del values
                act = worker.control('gateup', layer, **gate_kwargs)
                down_kwargs = {}
                if layer != 23:
                    values = selected_inputs(act, plan, layer, 2, native_root, inputs_dir if args.mode == 'convert' else None)
                    if args.mode == 'control':
                        reference_values = (selected_inputs({**act, 'records': act['records'][:20]}, original_plan,
                                                            layer, 2, native_root) if args.task_trajectories_from else values)
                        checks.extend(compare_control(reference_values, args.reference, layer, 2))
                        del reference_values
                    else:
                        path, stats = quantize_stage(layer, ('down',), values, readers, weights_dir,
                                                     damping_fraction=.1 if args.regularize_from else DAMPING,
                                                     refit_grid=refit_grid,
                                                     reference_values=reference_stage(reference, layer, 2, values))
                        quantized.extend(stats); down_kwargs = {'seq_payload': str(path)}
                    del values
                proposal = worker.control('down', layer, **down_kwargs)
                if proposal.get('phase') != 'candidate':
                    raise ValueError('layer must be a complete candidate before generation protection')
                verification = None
                if (args.mode == 'convert' or args.task_trajectories_from) and layer in (0, 12, 23):
                    verification = worker.control('verify', layer)
                    if type(verification.get('verified_values')) is not int or verification['verified_values'] <= 0:
                        raise ValueError('quantized-prefix normal-forward control did not compare values')
                if args.mode == 'control':
                    report, accept = None, True
                elif fixed_layers is not None:
                    # Fixed conversion scope: never select or drop a layer from answer scores.
                    report, accept = None, layer in fixed_layers
                else:
                    candidate = protection_generations(worker, cases, tokenizer, plan['eos_ids'], f'layer-{layer:02d}', source_protocol['vocab_size'])
                    report = check_proposal(cases, baseline, accepted, candidate)
                    accept = report['accepted']
                decision = worker.control('decide', layer, seq_accept=accept)
                if args.mode == 'convert' and accept:
                    accepted_layers.append(layer)
                    if fixed_layers is None:
                        accepted = candidate
                expected_mask = sum(1 << number for number in accepted_layers)
                if decision.get('accepted_layer_mask') != expected_mask:
                    raise ValueError('worker accepted layer mask disagrees with protected decisions')
                decisions.append({'layer': layer, 'accepted': accept if args.mode == 'convert' else False,
                                  'quantizer': quantized, 'protection': report, 'native': decision,
                                  'normal_forward_control': verification})
                save(args.output/'decisions.json', decisions)
                save(args.output/'progress.json', {'completed_layers': layer+1, 'accepted_layers': accepted_layers,
                    'elapsed_seconds': time.perf_counter()-started})
                print(json.dumps({'layer': layer, 'accepted': accept, 'accepted_layers': accepted_layers,
                                  'elapsed_seconds': time.perf_counter()-started}), flush=True)
        if decision.get('phase') != 'complete':
            raise ValueError('all24 sequential layers did not complete')
        if args.mode == 'control':
            expected = [{'id': row['id'], 'tokens': row['generated_tokens']} for row in records]
            if decision.get('predictions') != expected:
                raise ValueError('BF16 sequential final predictions differ from frozen native tokens')
            save(args.output/'control-inputs.json', checks)
            final = {'all_frozen_tokens_equal': True, 'tokens_compared': sum(len(row['tokens']) for row in expected),
                     'streams_compared': len(checks)//2, 'values_compared': sum(row['values_compared'] for row in checks)}
        else:
            from q4_packed_format import write_sparse
            if fixed_layers is not None:
                if accepted_layers != fixed_layers:
                    raise ValueError('fixed quantization scope changed')
                accepted = protection_generations(worker, cases, tokenizer, plan['eos_ids'],
                                                   'fixed-final', source_protocol['vocab_size'])
                report = fixed_protection(cases, baseline, previous, accepted,
                                          original_bf16_preservation=bool(args.asymmetric_reference))
                report.update(decision='keep_fixed_candidate' if report['accepted'] else 'reject_entire_fixed_candidate',
                              rollback='Close research worker; original files and product pointer remain unchanged')
                save(args.output/'fixed-protection.json', report)
                if not report['accepted']:
                    if args.asymmetric_reference:
                        raise ValueError('asymmetric fixed-mask candidate lost original BF16 TRAIN answers; reject whole candidate')
                    raise ValueError('fixed-mask candidate lost protected TRAIN answers; reject whole candidate')
            save(args.output/'accepted-generations.json', accepted)
            mask = sum(1 << layer for layer in accepted_layers)
            final = {'accepted_layers': accepted_layers, 'layer_mask': mask, 'protection_baseline_correct': baseline['correct_ids'],
                     'mask_zero': not mask, 'artifact': None}
            if mask:
                final['artifact'] = write_sparse(args.output/'gptq-sequential.bin', MANIFEST, '0'*64, mask,
                                                  sparse_payloads(weights_dir, accepted_layers))
        worker.close()
        code = worker.process.returncode
        worker = None
        if code != 0:
            raise RuntimeError('sequential worker did not exit cleanly')
        if args.mode == 'convert' and final['artifact'] is not None:
            final['reload'] = verify_reload(binary, args.output, lock, cases, tokenizer, plan['eos_ids'],
                                             accepted, final['artifact'], source_protocol['vocab_size'])
        save(args.output/'result.json', {**protocol, **final, 'completed': True, 'decisions': decisions,
            'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'method': 'Darwin Python RSS separately from worker proc_pid_rusage events; not deployment weight memory'}})
    except BaseException as error:
        save(args.output/'failure.json', {'type': type(error).__name__, 'error': str(error),
                                        'completed_layers': len(decisions), 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        if worker is not None:
            worker.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('control', 'convert'))
    parser.add_argument('--source', type=Path, default=ROOT/'packed-decode-calibration-v1')
    parser.add_argument('--reference', type=Path, default=ROOT/'packed-native-calibration-v1')
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path)
    method = parser.add_mutually_exclusive_group()
    method.add_argument('--regularize-from', type=Path,
                        help='Keep this completed conversion mask; down damping .1, final whole-candidate TRAIN gate')
    method.add_argument('--refit-grid-from', type=Path,
                        help='Keep this completed conversion mask; refit BF16 grids after feedback, damping .01')
    parser.add_argument('--task-trajectories-from', type=Path,
                        help='Append all sixteen actual original BF16 TRAIN trajectories as calibration')
    parser.add_argument('--asymmetric-reference', type=Path,
                        help='Bound original BF16 reference streams; GPTAQ alpha .25 with original-BF16 correctness gate')
    args = parser.parse_args()
    if (not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute()
            or not args.output.resolve().is_relative_to(ROOT) or not args.binary.is_absolute()
            or not args.binary.is_file() or (args.mode == 'convert' and args.control is None)):
        parser.error('existing absolute binary, mounted new experiment directory and conversion control required')
    if (args.regularize_from or args.refit_grid_from) and args.mode != 'convert':
        parser.error('fixed-scope candidates require conversion mode')
    if args.task_trajectories_from and args.mode == 'convert' and (
            not args.refit_grid_from or args.task_trajectories_from.resolve() != args.refit_grid_from.resolve()):
        parser.error('task calibration requires the same prior fixed-mask dynamic-grid source')
    try:
        validate_asymmetric_options(args)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    run(args)


if __name__ == '__main__':
    main()
