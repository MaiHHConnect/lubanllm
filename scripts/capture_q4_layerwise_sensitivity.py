"""Validate exact layerwise VJPs, then collect the frozen 32-case calibration gradients."""
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
from capture_q4_output_sensitivity import case_rows, preflight, raw_summary
from luban.backbone import load_backbone, text_backbone
from luban.process_lock import ModelProcessLock
from probe_q4_output_sensitivity import PROJECTIONS
from q4_layerwise_sensitivity import layerwise_gradients
from study_q4_column_protection import ROOT

OUTPUT = ROOT/'calibration-layerwise-sensitivity-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_layerwise_sensitivity_20261016.json'


def tensors(case):
    return (torch.tensor([case['tokens']], device='mps', dtype=torch.long),
            torch.tensor(case['target_positions'], device='mps', dtype=torch.long),
            torch.tensor(case['targets'], device='mps', dtype=torch.long))


def full_graph_reference(model, decoder, case):
    """Small complete-trajectory sum-loss reference; no multiplication of old mean gradients."""
    ids, positions, targets = tensors(case)
    captured, handles = {}, []
    logits = loss = gradients = value = None
    try:
        model.enable_input_require_grads()

        def hook_for(name):
            def hook(_module, _arguments, value):
                if name in captured or not value.requires_grad:
                    raise ValueError('reference gradient hook invalid')
                captured[name] = value
            return hook

        for layer, kind in PROJECTIONS:
            handles.append(getattr(decoder.layers[layer].mlp, kind+'_proj').register_forward_hook(
                hook_for(f'layer{layer:02d}-{kind}')))
        hidden = decoder(ids, use_cache=False, past_key_values=None).last_hidden_state
        logits = model.lm_head(hidden[:, positions, :])[0]
        loss = F.cross_entropy(logits.float(), targets, reduction='sum')
        gradients = torch.autograd.grad(loss, tuple(captured.values()))
        result = (logits.detach().cpu(), float(loss.item()),
                  {name: g.detach().cpu() for name, g in zip(captured, gradients, strict=True)})
        return result
    finally:
        for handle in handles:
            handle.remove()
        model.disable_input_require_grads()
        captured.clear()
        gradients = logits = loss = value = None


def validate_full_gradient(model, decoder, case, output, progress):
    if case['id'] != 'cal-00' or len(case['tokens']) != 281 or len(case['targets']) != 256:
        raise ValueError('original 281-input/256-target calibration control required')
    reference_logits, reference_loss, reference = full_graph_reference(model, decoder, case)
    gc.collect()
    torch.mps.empty_cache()
    ids, positions, targets = tensors(case)
    logits, loss, gradients, controls = layerwise_gradients(
        model, decoder, ids, positions, targets, PROJECTIONS, reduction='sum', progress=progress)
    if not torch.equal(logits, reference_logits) or loss != reference_loss or gradients.keys() != reference.keys():
        raise ValueError('layerwise/full-graph forward/loss/projection control failed')
    reports = []
    for name, expected in reference.items():
        actual = gradients[name]
        if not torch.equal(actual, expected):
            raise ValueError('layerwise/full-graph gradient mismatch: '+name)
        np.savez(output/(name+'.npz'), full_graph=expected.view(torch.uint16).numpy(),
                 layerwise=actual.view(torch.uint16).numpy())
        reports.append({'name': name, 'shape': list(actual.shape), 'values': actual.numel(), 'exact': True})
    result = {'completed': True, 'case': case, 'loss_reduction': 'sum', 'loss': loss,
              'logits_exact': True, 'gradients_exact': True, 'projections': reports, 'controls': controls}
    save(output/'result.json', result)
    return result


def run(output):
    source, protocol, mapping, cases = preflight()
    registration = json.loads(REGISTRATION.read_text())
    if (registration.get('schema') != 'q4-layerwise-calibration-sensitivity-v1'
            or registration.get('registered_before_run') is not True
            or registration.get('loss_reduction') != 'sum'):
        raise ValueError('layerwise complete-sequence preregistration required')
    longest = max(range(len(cases)), key=lambda i: len(cases[i]['tokens']))
    run_cases = [cases[longest]] + cases[:longest] + cases[longest+1:]
    output.mkdir(parents=True, exist_ok=False)
    (output/'equivalence').mkdir()
    metadata = {**registration, 'model': MODEL, 'identity': IDENTITY, 'manifest': MANIFEST,
        'core': None, 'source': str(ROOT/'asymmetric-conversion-v1'),
        'source_receipt': source['receipt'], 'plan': source['plan'], 'cases': cases,
        'run_order': [c['id'] for c in run_cases], 'hardware': platform.platform(),
        'torch': torch.__version__, 'seed': 42, 'attention_implementation': 'eager',
        'precision': 'BF16 weights/activations/boundary gradients; FP32 sum CE',
        'completed': False, 'parameter_updates': 0, 'teacher_calls': 0, 'qualified': False,
        'product_adopted': False, 'cache_reuse': False, 'use_cache': False}
    save(output/'protocol.json', metadata)
    save(output/'position-mapping.json', mapping)
    streams, reports = {}, []
    model = decoder = lock = None
    logits = gradients = value = None
    stage = 'load'
    started = time.perf_counter()
    sampled = {'allocated_bytes': 0, 'driver_bytes': 0}

    def progress(event):
        memory = {'allocated_bytes': torch.mps.current_allocated_memory(),
                  'driver_bytes': torch.mps.driver_allocated_memory()}
        for key, value in memory.items():
            sampled[key] = max(sampled[key], value)
        save(output/'progress.json', {'case_or_control': stage, 'stage': stage,
            'completed_cases': len(reports), 'sampled_mps': memory, **event})

    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = load_backbone(MODEL, torch.bfloat16, 'mps')
        decoder, config = text_backbone(model)
        if (config.num_hidden_layers, config.hidden_size, config.intermediate_size, config.vocab_size) != (24, 2048, 6144, protocol['vocab_size']):
            raise ValueError('unexpected original model graph')
        model.set_attn_implementation('eager')
        stage = '281-token-full-graph-equivalence'
        progress({})
        equivalence = validate_full_gradient(model, decoder, cases[0], output/'equivalence', progress)
        print(json.dumps({'equivalence_passed': True, 'gradient_values': sum(r['values'] for r in equivalence['projections'])}), flush=True)
        gc.collect()
        torch.mps.empty_cache()
        for layer, kind in PROJECTIONS:
            name, width = f'layer{layer:02d}-{kind}', 2048 if kind == 'down' else 6144
            streams[name] = np.memmap(output/(name+'.bf16'), dtype='<u2', mode='w+', shape=(4165, width))
        for case in run_cases:
            case_started = time.perf_counter()
            stage = case['id']
            progress({})
            ids, positions, targets = tensors(case)
            logits, loss, gradients, controls = layerwise_gradients(
                model, decoder, ids, positions, targets, PROJECTIONS, reduction='sum', progress=progress)
            rows, summaries = case_rows(mapping, case), []
            for name, value in gradients.items():
                if (value.dtype != torch.bfloat16 or value.device.type != 'cpu'
                        or tuple(value.shape) != (1, len(case['tokens']), streams[name].shape[1])
                        or not torch.count_nonzero(value).item()):
                    raise ValueError('complete finite nonzero BF16 projection gradient required')
                raw = value[0, case['selected_positions']].contiguous().view(torch.uint16).numpy()
                summaries.append({'name': name, **raw_summary(raw)})
                streams[name][rows] = raw
            report = {'id': case['id'], 'input_tokens': len(case['tokens']), 'target_tokens': len(case['targets']),
                'selected_rows': rows.tolist(), 'loss_sum': loss, 'forward_exact': True,
                'argmax': logits.argmax(-1).tolist(), 'gradient_projections': summaries, 'controls': controls,
                'elapsed_seconds': time.perf_counter()-case_started}
            save(output/(case['id']+'.json'), report)
            reports.append({'id': case['id'], 'loss_sum': loss, 'elapsed_seconds': report['elapsed_seconds']})
            print(json.dumps(reports[-1]), flush=True)
            gradients = logits = value = None
            gc.collect()
            torch.mps.empty_cache()
        for stream in streams.values():
            stream.flush()
        save(output/'result.json', {**metadata, 'completed': True, 'records': reports,
            'all_forward_controls_exact': True, 'full_graph_equivalence': 'equivalence/result.json',
            'gradient_values': 4165*sum(s.shape[1] for s in streams.values()),
            'elapsed_seconds': time.perf_counter()-started,
            'memory': {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'maximum_sampled_mps': sampled,
                       'method': 'Darwin peak process RSS; MPS samples at phase/layer callbacks, not GPU peak'}})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'stage': stage, 'error': repr(error),
            'records': reports, 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        try:
            streams.clear()
            gradients = logits = value = model = decoder = None
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
