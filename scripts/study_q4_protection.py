"""Read-only real-activation study of group protection and subspace compensation.

No candidate is installed. Full BF16 forwards supply frozen inputs; sampled
projection outputs are diagnostics, not end-to-end model quality evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from diagnose_native_q4 import bf16, quantize
from diagnose_native_q4_activation import PROMPTS
from luban.backbone import load_backbone, text_backbone


def q4(weights):
    if weights.ndim != 2 or weights.shape[1] % 32 or not np.isfinite(weights).all():
        raise ValueError('finite row-major weights with group32 required')
    return quantize(weights.reshape(-1, 32))[0].reshape(weights.shape)


def basis_for(inputs, rank):
    if inputs.ndim != 2 or not np.isfinite(inputs).all() or not 0 < rank < min(inputs.shape):
        raise ValueError('finite calibration matrix and bounded rank required')
    _, singular, vh = torch.linalg.svd(torch.from_numpy(inputs), full_matrices=False)
    tolerance = max(inputs.shape) * torch.finfo(torch.float32).eps * float(singular[0])
    rank = min(rank, int((singular > tolerance).sum()))
    if not rank:
        raise ValueError('no nonzero calibration directions')
    return vh[:rank].T.contiguous().numpy()


def compensated_error(inputs, delta, basis, *, rounded=False):
    # delta = W - Q(W). Restore only the error along protected input directions.
    factors = delta @ basis
    if rounded:
        basis, factors = bf16(basis), bf16(factors)
    return inputs @ delta.T - (inputs @ basis) @ factors.T


def scores(error, reference):
    if not np.isfinite(error).all():
        raise ValueError('nonfinite projection error')
    sse = float(np.square(error.astype(np.float64)).sum())
    energy = float(np.square(reference.astype(np.float64)).sum())
    return {'mse': sse / error.size, 'relative_mse': sse / energy if energy else None,
            'max_abs': float(np.abs(error).max()), 'elements': error.size}


def fixture(path, weight, inputs):
    path.mkdir(exist_ok=False)
    n, k = weight.shape
    values = quantize(weight.reshape(-1, 32))
    codes = values[4].astype(np.uint32).reshape(-1, 8)
    words = np.bitwise_or.reduce(codes << (np.arange(8, dtype=np.uint32) * 4), axis=1)
    words.astype('<u4').tofile(path / 'words.u32')
    for name, value in [('x', inputs), ('scales', values[7]), ('biases', values[9])]:
        (value.astype(np.float32).view(np.uint32) >> 16).astype('<u2').tofile(path / f'{name}.bf16')
    (path / 'shape.json').write_text(json.dumps({'m': len(inputs), 'n': n, 'k': k, 'group_size': 32}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.absolute().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('use the mounted research volume')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    started = time.perf_counter()
    materials = [{'id': f'cal-{i}', 'split': 'calibration', 'text': PROMPTS[i]} for i in (0, 1, 2, 3, 6)]
    materials += [{'id': f'observe-{i}', 'split': 'observation', 'text': PROMPTS[i]} for i in (4, 5, 7)]
    materials += [{'id': 'observe-long-retrieval', 'split': 'observation', 'text':
                   '档案约定：项目代号是青松，截止日期是九月二十三日。\n' +
                   '以下是无关的每日记录：天气晴，仓库库存正常，例行检查完成。\n' * 48 +
                   '请只回答开头约定的项目代号和截止日期。'}]
    names = [f'layers.{layer}.mlp.{kind}_proj' for layer in (0, 12, 22) for kind in ('gate', 'up', 'down')]
    protocol = {'schema': 'q4-protection-local-v1', 'seed': 42, 'layers': names, 'materials': materials,
                'sampled_output_rows': 64, 'positions': 'up to 32 evenly spaced plus final 8 per prompt',
                'ranks': [8, 32], 'protected_group_fractions': [.01, .05, .1],
                'selection': 'calibration only; observation never selects groups or basis',
                'precision': 'BF16 graph, FP32 Q4 decode/reference, FP32 or BF16 residual factors',
                'core_override': False, 'training': False, 'product_changed': False,
                'scope': 'local linear outputs on frozen BF16 inputs; no autoregressive Q4 graph or quality claim'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = load_backbone(args.model, torch.bfloat16, device)
    decoder, _ = text_backbone(model)
    tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    modules = dict(decoder.named_modules())
    captures = {name: {} for name in names}
    current = None

    def hook_for(name):
        def hook(_module, arguments):
            x = arguments[0].detach().reshape(-1, arguments[0].shape[-1])
            positions = sorted(set(np.linspace(0, len(x) - 1, min(32, len(x)), dtype=int).tolist() + list(range(max(0, len(x) - 8), len(x)))))
            captures[name][current] = x[positions].float().cpu().numpy()
        return hook

    handles = [modules[name].register_forward_pre_hook(hook_for(name)) for name in names]
    try:
        with torch.inference_mode():
            for row in materials:
                current = row['id']
                ids = tokenizer(row['text'], return_tensors='pt').input_ids.to(device)
                if ids.shape[1] > 2048:
                    raise ValueError('unexpected prompt length; do not silently truncate')
                row['tokens'] = int(ids.shape[1])
                decoder(ids, use_cache=False)
                print(json.dumps({'captured': current, 'tokens': row['tokens']}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    sampled = {}
    for name in names:
        weight = modules[name].weight
        rows = np.sort(rng.choice(weight.shape[0], 64, replace=False))
        sampled[name] = (weight[rows.tolist()].detach().float().cpu().numpy(), rows, list(weight.shape))
    fixture(args.output / 'real-gate-fixture', modules[names[0]].weight.detach().float().cpu().numpy(),
            captures[names[0]][materials[0]['id']][:9])
    del modules, decoder, model, weight
    if device == 'mps':
        torch.mps.empty_cache()
    records = []
    for name in names:
        weight, rows, shape = sampled[name]
        decoded = q4(weight)
        delta = weight - decoded
        cal = np.concatenate([captures[name][row['id']] for row in materials if row['split'] == 'calibration'])
        groups = delta.reshape(64, -1, 32)
        cal_groups = cal.reshape(len(cal), -1, 32)
        errors = np.einsum('tgi,ogi->tgo', cal_groups, groups, optimize=True)
        group_energy = np.square(errors.astype(np.float64)).sum(axis=(0, 2))
        group_order = np.argsort(-group_energy, kind='stable')
        bases = {rank: basis_for(cal, rank) for rank in protocol['ranks']}
        alternatives = []
        for fraction in protocol['protected_group_fractions']:
            count = math.ceil(len(group_order) * fraction)
            protected = delta.copy().reshape(64, -1, 32)
            protected[:, group_order[:count], :] = 0
            alternatives.append((f'groups-{fraction}', protected.reshape(weight.shape), None,
                                 {'group_ids': group_order[:count].tolist(), 'actual_fraction': count / len(group_order),
                                  'extra_bytes_full_projection': count * 32 * shape[0] * 2}))
        for rank, basis in bases.items():
            for rounded in (False, True):
                alternatives.append((f'subspace-{rank}-' + ('bf16' if rounded else 'fp32'), delta, (basis, rounded),
                                     {'rank': basis.shape[1], 'extra_bytes_full_projection':
                                      (shape[0] + shape[1]) * basis.shape[1] * (2 if rounded else 4)}))
        record = {'module': name, 'shape': shape, 'output_rows': rows.tolist(),
                  'nominal_q4_bytes_full_projection': shape[0] * shape[1] * .625, 'cases': []}
        for row in materials:
            x = captures[name][row['id']]
            reference = x @ weight.T
            baseline_error = x @ delta.T
            case = {'id': row['id'], 'split': row['split'], 'tokens': row['tokens'],
                    'positions': len(x), 'baseline': scores(baseline_error, reference), 'alternatives': {}}
            for label, error_weight, correction, metadata in alternatives:
                error = x @ error_weight.T if correction is None else compensated_error(x, error_weight, correction[0], rounded=correction[1])
                result = scores(error, reference)
                result['mse_ratio_to_q4'] = result['mse'] / case['baseline']['mse'] if case['baseline']['mse'] else None
                # Include output BF16 rounding; mathematical subspace cancellation alone is insufficient.
                result['bf16_output_mse'] = scores(bf16(reference - error) - bf16(reference), bf16(reference))['mse']
                case['alternatives'][label] = {**metadata, **result}
            case['baseline']['bf16_output_mse'] = scores(bf16(reference - baseline_error) - bf16(reference), bf16(reference))['mse']
            record['cases'].append(case)
        records.append(record)
        print(json.dumps({'analyzed': name}), flush=True)
    report = {**protocol, 'model': str(args.model.resolve()), 'materials': materials, 'device': device,
              'hardware': platform.platform(), 'torch': torch.__version__, 'numpy': np.__version__,
              'elapsed_seconds': time.perf_counter() - started,
              'memory': {'process_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         'method': 'Darwin getrusage ru_maxrss; process high-water RSS, not GPU footprint'},
              'records': records}
    (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'result': str(args.output / 'result.json'), 'seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
