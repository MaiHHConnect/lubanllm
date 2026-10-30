"""Frozen HF end-loss gradients on the existing 32 calibration trajectories only."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from capture_q4_native import load_frozen
from capture_q4_task_reference import layout
from luban.backbone import load_backbone, text_backbone
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import make_plan
from probe_q4_output_sensitivity import PROJECTIONS
from q4_task_trajectories import load_task_trajectories
from study_q4_column_protection import ROOT, ROWS, SOURCE

SCHEMA = 'q4-calibration-output-sensitivity-v1'
OUTPUT = ROOT/'calibration-output-sensitivity-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_calibration_sensitivity_20261016.json'


def calibration_case(record, planned, vocab):
    prompt, generated = record['prompt_tokens'], record['generated_tokens']
    tokens = prompt + generated[:-1]
    positions = planned['positions']
    if (record['split'] != 'calibration' or planned['split'] != 'calibration'
            or record['id'] != planned['id'] or prompt != planned['tokens']
            or generated[:-1] != planned['forced_tokens'] or not prompt or not generated
            or not 1 <= len(tokens) <= 2048 or len(generated) > 256
            or any(type(t) is not int or not 0 <= t < vocab for t in prompt + generated)
            or not positions or positions != sorted(set(positions))
            or any(type(p) is not int or not 0 <= p < len(tokens) for p in positions)):
        raise ValueError('complete bounded calibration trajectory matching original plan required')
    return {'id': record['id'], 'tokens': tokens, 'targets': generated,
            'target_positions': list(range(len(prompt)-1, len(tokens))),
            'selected_positions': positions, 'source_finish_reason': record['finish_reason']}


def case_rows(mapping, case):
    """Map per-case selected positions to prompt-first global calibration rows."""
    rows, positions = [], []
    for segment in mapping['segments']:
        if segment['split'] == 'calibration' and segment['id'] == case['id']:
            rows.extend(range(segment['stream_row_start'], segment['stream_row_end']))
            positions.extend(segment['positions'])
    if positions != case['selected_positions'] or len(rows) != len(set(rows)):
        raise ValueError('selected gradient positions differ from original input row mapping')
    return np.array(rows, dtype=np.int64)


def preflight():
    source = json.loads((SOURCE/'result.json').read_text())
    protocol, original = load_frozen(Path(source['source']))
    tasks = load_task_trajectories(Path(source['task_trajectories_from']),
                                  source['task_source_receipt'], make_plan(protocol, original), protocol)
    plan = make_plan(protocol, original + tasks)
    if (source.get('completed') is not True or source.get('mode') != 'convert'
            or plan != source['plan'] or plan['manifest'] != MANIFEST or plan['core'] is not None):
        raise ValueError('completed frozen original conversion and identical trajectory plan required')
    mapping = layout(plan)
    if mapping['rows_per_split'] != ROWS:
        raise ValueError('frozen calibration/observation row counts changed')
    cases = [calibration_case(r, p, protocol['vocab_size'])
             for r, p in zip(original+tasks, plan['cases'], strict=True) if r['split'] == 'calibration']
    if len(cases) != 32 or sorted(np.concatenate([case_rows(mapping, c) for c in cases]).tolist()) != list(range(4165)):
        raise ValueError('exactly 32 calibration trajectories covering every selected row once required')
    return source, protocol, mapping, cases


def raw_summary(raw):
    """All selected BF16 values are inspected; zero channels and token rows are valid."""
    if raw.dtype != np.uint16 or raw.ndim != 2 or not all(raw.shape):
        raise ValueError('nonempty 2D BF16 gradient carrier required')
    if np.any((raw & 0x7f80) == 0x7f80):
        raise ValueError('nonfinite selected gradient')
    values = (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    energies = np.einsum('ij,ij->i', values, values)
    return {'shape': list(raw.shape), 'nonzero': int(np.count_nonzero(values)),
            'squared_norm': float(energies.sum()), 'token_squared_norms': energies.tolist()}


def run(output):
    source, protocol, mapping, cases = preflight()
    registration = json.loads(REGISTRATION.read_text())
    if (registration.get('schema') != SCHEMA or registration.get('registered_before_run') is not True
            or registration.get('loss_reduction') != 'sum' or registration.get('rows') != ROWS
            or registration.get('calibration_cases') != 32):
        raise ValueError('matching calibration-only preregistration required')
    # Capacity choice uses length only, not content, losses, answers or sensitivity.
    longest = max(range(len(cases)), key=lambda i: len(cases[i]['tokens']))
    run_cases = [cases[longest]] + cases[:longest] + cases[longest+1:]
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**registration, 'model': MODEL, 'identity': IDENTITY, 'manifest': MANIFEST,
        'core': None, 'source': str(SOURCE), 'source_receipt': source['receipt'],
        'plan': source['plan'], 'cases': cases, 'run_order': [c['id'] for c in run_cases],
        'hardware': platform.platform(), 'torch': torch.__version__, 'seed': 42,
        'attention_implementation': 'eager', 'precision': 'BF16 frozen weights/activations/gradients; FP32 sum CE',
        'memory_method': 'Darwin peak process RSS and maximum sampled MPS allocated/driver, not GPU peak',
        'completed': False, 'parameter_updates': 0, 'teacher_calls': 0, 'qualified': False,
        'product_adopted': False, 'cache_reuse': False, 'use_cache': False}
    save(output/'protocol.json', metadata)
    save(output/'position-mapping.json', mapping)
    streams, handles, captured, reports = {}, [], {}, []
    model = decoder = lock = None
    input_grad_enabled = False
    logits = loss = gradients = gradient = selected = baseline = ids = targets = positions = value = module = None
    started = time.perf_counter()
    sampled = {'allocated_bytes': 0, 'driver_bytes': 0}
    stage = 'load'
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = load_backbone(MODEL, torch.bfloat16, 'mps')
        decoder, config = text_backbone(model)
        if (config.num_hidden_layers, config.hidden_size, config.intermediate_size, config.vocab_size) != (24, 2048, 6144, protocol['vocab_size']):
            raise ValueError('unexpected original decoder graph')
        model.set_attn_implementation('eager')
        if config._attn_implementation != 'eager':
            raise ValueError('official eager attention did not reach decoder')
        model.enable_input_require_grads()
        input_grad_enabled = True

        def hook_for(name):
            def hook(_module, _arguments, value):
                if not torch.is_grad_enabled():
                    return
                if name in captured or not value.requires_grad:
                    raise ValueError('duplicate projection or broken gradient path: '+name)
                captured[name] = value
            return hook

        for layer, kind in PROJECTIONS:
            name, width = f'layer{layer:02d}-{kind}', 2048 if kind == 'down' else 6144
            module = getattr(decoder.layers[layer].mlp, kind+'_proj')
            handles.append(module.register_forward_hook(hook_for(name)))
            streams[name] = np.memmap(output/(name+'.bf16'), dtype='<u2', mode='w+', shape=(4165, width))

        def forward():
            hidden = decoder(ids, use_cache=False, past_key_values=None).last_hidden_state
            logits = model.lm_head(hidden[:, positions, :])[0]
            return logits, F.cross_entropy(logits.float(), targets, reduction='sum')

        for case in run_cases:
            case_started = time.perf_counter()
            stage = case['id'] + ':baseline'
            save(output/'progress.json', {'stage': stage, 'completed_cases': len(reports)})
            ids = torch.tensor([case['tokens']], dtype=torch.long, device='mps')
            targets = torch.tensor(case['targets'], dtype=torch.long, device='mps')
            positions = torch.tensor(case['target_positions'], dtype=torch.long, device='mps')
            with torch.no_grad():
                logits, loss = forward()
                baseline, baseline_loss = logits.cpu(), float(loss.item())
            logits = loss = None
            stage = case['id'] + ':gradient_forward'
            save(output/'progress.json', {'stage': stage, 'completed_cases': len(reports)})
            logits, loss = forward()
            if (len(captured) != 32 or not torch.equal(baseline, logits.detach().cpu())
                    or baseline_loss != float(loss.item()) or not np.isfinite(baseline_loss)):
                raise ValueError('full-vocabulary grad/no-grad forward control failed')
            argmax = baseline.argmax(-1).tolist()
            baseline = None
            stage = case['id'] + ':backward'
            save(output/'progress.json', {'stage': stage, 'completed_cases': len(reports)})
            gradients = torch.autograd.grad(loss, tuple(captured.values()))
            torch.mps.synchronize()
            for key, value in [('allocated_bytes', torch.mps.current_allocated_memory()),
                               ('driver_bytes', torch.mps.driver_allocated_memory())]:
                sampled[key] = max(sampled[key], value)
            selected_positions = torch.tensor(case['selected_positions'], device='mps', dtype=torch.long)
            rows = case_rows(mapping, case)
            summaries = []
            for (name, value), gradient in zip(captured.items(), gradients, strict=True):
                width = streams[name].shape[1]
                if (gradient.dtype != torch.bfloat16 or tuple(gradient.shape) != (1, len(case['tokens']), width)
                        or not torch.isfinite(gradient).all().item() or not torch.count_nonzero(gradient).item()):
                    raise ValueError('invalid complete projection gradient: '+name)
                selected = gradient[0, selected_positions].detach().cpu().contiguous().view(torch.uint16).numpy()
                summary = raw_summary(selected)
                streams[name][rows] = selected
                summaries.append({'name': name, **summary})
            report = {'id': case['id'], 'input_tokens': len(case['tokens']), 'target_tokens': len(case['targets']),
                'selected_rows': rows.tolist(), 'loss_sum': baseline_loss, 'forward_exact': True,
                'argmax': argmax, 'gradient_projections': summaries, 'elapsed_seconds': time.perf_counter()-case_started}
            save(output/(case['id']+'.json'), report)
            reports.append({'id': case['id'], 'loss_sum': baseline_loss, 'elapsed_seconds': report['elapsed_seconds']})
            print(json.dumps(reports[-1]), flush=True)
            captured.clear()
            gradients = gradient = selected = logits = loss = value = None
            gc.collect()
            torch.mps.empty_cache()
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError('parameter freezing violated')
        for stream in streams.values():
            stream.flush()
        save(output/'result.json', {**metadata, 'completed': True, 'records': reports,
            'all_forward_controls_exact': True, 'gradient_values': 4165*sum(s.shape[1] for s in streams.values()),
            'elapsed_seconds': time.perf_counter()-started,
            'memory': {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'maximum_sampled_mps': sampled}})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'stage': stage, 'error': repr(error),
            'records': reports, 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        try:
            for handle in handles:
                handle.remove()
            if input_grad_enabled:
                model.disable_input_require_grads()
            captured.clear()
            streams.clear()
            gradients = gradient = selected = logits = loss = baseline = value = module = None
            ids = targets = positions = model = decoder = None
            gc.collect()
            torch.mps.empty_cache()
        finally:
            if lock is not None:
                lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('new output on mounted experiment volume required')
    run(args.output)


if __name__ == '__main__':
    main()
