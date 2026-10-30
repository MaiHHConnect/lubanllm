"""Frozen HF output-gradient feasibility probe; never trains or qualifies native Q4."""
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
from luban.backbone import load_backbone, text_backbone
from luban.process_lock import ModelProcessLock
from study_q4_column_protection import LAYERS, ROOT

SOURCE = ROOT/'packed-decode-calibration-v1'
OUTPUT = ROOT/'hf-output-sensitivity-probe-v1'
PROJECTIONS = [(layer, kind) for layer in LAYERS for kind in ('gate', 'up', 'down')
               if (layer, kind) != (23, 'down')]
BOUNDARY = ('Frozen original HF BF16, full-sequence fallback graph, FP32 mean CE, '
            'ordinary autograd through casts. Not the derivative of discrete Metal inference; '
            'HF/native forward differences are already established. No Q4 qualification.')


def probe_case(records, vocab, *, full_continuation=False):
    # Fixed source order, not selected by model loss, answer correctness or gradients.
    record = records[0]
    if record.get('id') != 'cal-00' or record.get('split') != 'calibration':
        raise ValueError('first original calibration record required')
    prompt, generated = record['prompt_tokens'], record['generated_tokens']
    if (not prompt or not 3 <= len(generated) <= 256 or len(prompt) > 128
            or any(type(t) is not int or not 0 <= t < vocab for t in prompt+generated)):
        raise ValueError('bounded valid frozen token sequence required')
    count = len(generated) if full_continuation else 3
    return {'id': record['id'], 'tokens': prompt+generated[:count-1], 'targets': generated[:count],
            'target_positions': list(range(len(prompt)-1, len(prompt)+count-1)),
            'source_finish_reason': record['finish_reason'], 'source_tokens': len(generated),
            'scope': ('all frozen teacher-forced predictions; preserve original finish reason' if full_continuation
                      else 'first three teacher-forced predictions only; not complete generation')}


def gradient_summary(value):
    x = value.detach().float().cpu().numpy()
    if x.ndim != 3 or x.shape[0] != 1 or not np.isfinite(x).all() or not np.any(x):
        raise ValueError('nonempty finite nonzero projection gradient required')
    energy = np.sum(x[0].astype(np.float64)**2, axis=1)
    return {'shape': list(x.shape), 'dtype': str(value.dtype),
            'nonzero': int(np.count_nonzero(x)), 'squared_norm': float(energy.sum()),
            'token_squared_norms': energy.tolist(),
            'token_energy_varies': bool(np.ptp(energy) > 0)}


def run(output, attention='sdpa', *, full_continuation=False):
    protocol, records = load_frozen(SOURCE)
    case = probe_case(records, protocol['vocab_size'], full_continuation=full_continuation)
    output.mkdir(parents=True, exist_ok=False)
    registration = {'schema': 'q4-hf-output-sensitivity-probe-v1', 'source': str(SOURCE),
        'model': MODEL, 'identity': IDENTITY, 'manifest': MANIFEST, 'core': None,
        'source_receipt': protocol['receipt'], 'case': case, 'projections': PROJECTIONS,
        'precision': 'BF16 weights/activations/returned gradients, FP32 CE',
        'attention_implementation': attention,
        'full_continuation': full_continuation,
        'seed': 42, 'cache_reuse': False, 'use_cache': False, 'training': False,
        'parameter_updates': 0, 'teacher_calls': 0, 'product_adopted': False,
        'new_holdout_read': False, 'qualified': False, 'numerical_boundary': BOUNDARY,
        'hardware': platform.platform(), 'torch': torch.__version__,
        'selection': 'first frozen calibration record, '+case['scope']+', no answer filtering',
        'controls': ['same resident no-hook/no-grad forward logits exactly match hooked grad forward',
                     'all 32 projection gradients finite, nonzero and complete',
                     'separate earliest-output-leaf backward matches its multi-hook gradient exactly',
                     'all parameters remain frozen with grad=None; no optimizer or save_pretrained'],
        'memory_method': 'Darwin process peak RSS plus sampled MPS allocated/driver bytes; not GPU peak'}
    save(output/'protocol.json', registration)
    started, reports, handles = time.perf_counter(), [], []
    model = decoder = None
    input_grad_enabled = False
    lock = None
    captured, leaf = {}, []
    logits = loss = gradients = value = gradient = module = None
    ids = targets = positions = None
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = load_backbone(MODEL, torch.bfloat16, 'mps')
        decoder, config = text_backbone(model)
        if (config.num_hidden_layers, config.hidden_size, config.intermediate_size,
                config.vocab_size) != (24, 2048, 6144, protocol['vocab_size']):
            raise ValueError('unexpected original decoder graph')
        ids = torch.tensor([case['tokens']], device='mps', dtype=torch.long)
        targets = torch.tensor(case['targets'], device='mps', dtype=torch.long)
        positions = torch.tensor(case['target_positions'], device='mps', dtype=torch.long)

        def forward():
            hidden = decoder(ids, use_cache=False, past_key_values=None).last_hidden_state
            logits = model.lm_head(hidden[:, positions, :])[0]
            return logits, F.cross_entropy(logits.float(), targets, reduction='mean')

        with torch.no_grad():
            baseline, loss = forward()
            baseline = baseline.cpu()
            baseline_loss = float(loss.item())
        default_baseline, default_loss = baseline, baseline_loss
        default_attention = config._attn_implementation
        if attention != default_attention:
            model.set_attn_implementation(attention)
            if config._attn_implementation != attention:
                raise ValueError('official attention selection did not reach text decoder')
            with torch.no_grad():
                baseline, loss = forward()
                baseline = baseline.cpu()
                baseline_loss = float(loss.item())
        save(output/'attention-control.json', {'default_attention': default_attention,
            'probe_attention': config._attn_implementation, 'default_loss': default_loss,
            'probe_loss': baseline_loss,
            'default_vs_probe_exact': torch.equal(default_baseline, baseline),
            'rmse': float((default_baseline.float()-baseline.float()).square().mean().sqrt())})
        save(output/'progress.json', {'stage': 'baseline_complete', 'loss': baseline_loss,
                                     'elapsed_seconds': time.perf_counter()-started})

        def hook_for(name):
            def hook(_module, _arguments, value):
                if name in captured or not value.requires_grad:
                    raise ValueError('duplicate projection or broken gradient path: '+name)
                captured[name] = value
            return hook

        model.enable_input_require_grads()
        input_grad_enabled = True
        for layer, kind in PROJECTIONS:
            name = f'layer{layer:02d}-{kind}'
            module = getattr(decoder.layers[layer].mlp, kind+'_proj')
            handles.append(module.register_forward_hook(hook_for(name)))
        logits, loss = forward()
        actual = logits.detach().cpu()
        difference = actual.float()-baseline.float()
        save(output/'forward-control.json', {
            'baseline_loss': baseline_loss, 'instrumented_loss': float(loss.item()),
            'logits_equal': torch.equal(actual, baseline),
            'different_logits': int(torch.count_nonzero(difference)),
            'logit_values': baseline.numel(), 'max_abs': float(difference.abs().max()),
            'rmse': float(difference.square().mean().sqrt()),
            'baseline_argmax': baseline.argmax(-1).tolist(),
            'instrumented_argmax': actual.argmax(-1).tolist()})
        np.savez(output/'logits.npz', baseline=baseline.view(torch.uint16).numpy(),
                 instrumented=actual.view(torch.uint16).numpy(), targets=np.array(case['targets']),
                 default_no_grad=default_baseline.view(torch.uint16).numpy())
        if not torch.equal(actual, baseline) or float(loss.item()) != baseline_loss:
            raise ValueError('gradient instrumentation changed forward logits/loss')
        if len(captured) != 32:
            raise ValueError('missing projection outputs')
        save(output/'progress.json', {'stage': 'grad_forward_exact', 'loss': baseline_loss,
            'elapsed_seconds': time.perf_counter()-started,
            'mps_allocated_bytes': torch.mps.current_allocated_memory(),
            'mps_driver_bytes': torch.mps.driver_allocated_memory()})
        gradients = torch.autograd.grad(loss, tuple(captured.values()))
        torch.mps.synchronize()
        for (name, value), gradient in zip(captured.items(), gradients, strict=True):
            summary = gradient_summary(gradient)
            if tuple(value.shape) != tuple(gradient.shape) or value.shape[1] != len(case['tokens']):
                raise ValueError('projection output/gradient shape mismatch')
            np.savez(output/(name+'.npz'),
                     output=value.detach().cpu().view(torch.uint16).numpy(),
                     gradient=gradient.detach().cpu().view(torch.uint16).numpy())
            reports.append({'name': name, **summary})
        earliest = gradients[0].detach().cpu().clone()
        for handle in handles:
            handle.remove()
        handles.clear()
        model.disable_input_require_grads()
        input_grad_enabled = False
        captured.clear()
        gradients = value = gradient = logits = loss = None
        gc.collect()
        torch.mps.empty_cache()

        def leaf_hook(_module, _arguments, value):
            if leaf or value.requires_grad:
                raise ValueError('independent first-output leaf scope changed')
            leaf.append(value.requires_grad_(True))

        handles.append(decoder.layers[0].mlp.gate_proj.register_forward_hook(leaf_hook))
        logits, loss = forward()
        independent = torch.autograd.grad(loss, leaf)[0].detach().cpu()
        if (not torch.equal(logits.detach().cpu(), baseline) or float(loss.item()) != baseline_loss
                or not torch.equal(independent, earliest)):
            raise ValueError('independent earliest-output backward mismatch')
        np.savez(output/'earliest-leaf-control.npz', multi_hook=earliest.view(torch.uint16).numpy(),
                 single_leaf=independent.view(torch.uint16).numpy())
        if any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError('model parameter freezing violated')
        save(output/'result.json', {**registration, 'completed': True, 'forward_exact': True,
            'earliest_leaf_gradient_exact': True,
            'baseline_loss': baseline_loss, 'gradient_projections': reports,
            'elapsed_seconds': time.perf_counter()-started,
            'memory': {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                'mps_driver_bytes': torch.mps.driver_allocated_memory()},
            'limitation': 'Feasibility on recorded sequence only; no independent full-graph derivative proof, calibration-wide fitting or Q4 quality qualification.'})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'error': repr(error),
            'gradient_projections': reports, 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        try:
            for handle in handles:
                handle.remove()
            if input_grad_enabled:
                model.disable_input_require_grads()
            captured.clear()
            leaf.clear()
            gradients = value = gradient = logits = loss = module = None
            ids = targets = positions = model = decoder = None
            gc.collect()
            torch.mps.empty_cache()
        finally:
            if lock is not None:
                lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--attention', choices=('sdpa', 'eager'), default='sdpa')
    parser.add_argument('--full-continuation', action='store_true')
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('new output on mounted experiment volume required')
    run(args.output, args.attention, full_continuation=args.full_continuation)


if __name__ == '__main__':
    main()
