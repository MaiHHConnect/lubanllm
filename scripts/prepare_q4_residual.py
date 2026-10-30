"""Prepare frozen-calibration Q4 error factors for the explicit native experiment.

Uses the same five calibration prompts and rank budgets as the local study.
No generation labels, test answers, optimizer, product writes or weight hashes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import struct
import time

import numpy as np
import torch
from safetensors.torch import load_file

from benchmark_native_decode import MODEL, CORE, CORE_SHA, MANIFEST, IDENTITY, save
from benchmark_native_mixed import CORE_KEY
from diagnose_native_q4 import bf16
from diagnose_native_q4_activation import PROMPTS
from study_q4_protection import basis_for, q4
from luban.backbone import load_backbone, text_backbone
from luban.process_lock import ModelProcessLock


def encode_bf16(values):
    rounded = bf16(values)
    if not np.isfinite(rounded).all():
        raise ValueError('nonfinite residual factors after BF16 rounding')
    return (rounded.view(np.uint32) >> 16).astype('<u2').tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.absolute().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted research volume required')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    prompts = [{'id': f'cal-{i}', 'text': PROMPTS[i]} for i in (0, 1, 2, 3, 6)]
    names = [f'layers.{i}.mlp.{p}_proj' for i in range(24) for p in ('gate', 'up', 'down')
             if not (i == 23 and p == 'down')]
    report = {'protocol': 'native-q4-subspace-factors-v1', 'ranks': [8, 32], 'seed': 42,
              'calibration': prompts, 'model': MODEL, 'core': CORE, 'manifest': MANIFEST,
              'core_sha_declared': CORE_SHA, 'binding': 'native worker verifies base/core at load; no extra Python weight hash',
              'factor_rule': 'B = right singular vectors of frozen calibration inputs; U = (W-Q4group32(W)) B',
              'layout': 'LQ4RES1 newline (8 bytes), LE u32 rank, 64 ASCII manifest, 64 ASCII core, then 71 B[r,k],U[n,r] BF16',
              'native_arithmetic': 'Q4 BF16 output + FP32 (X BF16_basis^T) BF16_U^T, final BF16',
              'scope': 'same B45 core as existing 160-case native benchmark; no trained identity expert',
              'training': False, 'teacher_calls': 0, 'product_changed': False}
    save(args.output / 'protocol.json', report)
    lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        model = load_backbone(MODEL, torch.bfloat16, 'mps')
        decoder, _ = text_backbone(model)
        core = load_file(CORE, device='cpu')[CORE_KEY]
        target = decoder.layers[23].mlp.down_proj.weight
        if target.shape != core.shape or core.dtype != torch.bfloat16 or not torch.isfinite(core).all():
            raise ValueError('unexpected B45 core')
        with torch.no_grad():
            target.copy_(core.to('mps'))
        del core, target
        tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        modules = dict(decoder.named_modules())
        captures = {name: [] for name in names}

        def capture(name):
            def hook(_module, arguments):
                x = arguments[0].detach().reshape(-1, arguments[0].shape[-1])
                positions = sorted(set(np.linspace(0, len(x)-1, min(32, len(x)), dtype=int).tolist()
                                       + list(range(max(0, len(x)-8), len(x)))))
                captures[name].append(x[positions].float().cpu().numpy())
            return hook

        handles = [modules[name].register_forward_pre_hook(capture(name)) for name in names]
        try:
            with torch.inference_mode():
                for prompt in prompts:
                    ids = tokenizer(prompt['text'], return_tensors='pt').input_ids.to('mps')
                    if ids.shape[1] > 2048:
                        raise ValueError('no implicit truncation')
                    prompt['tokens'] = int(ids.shape[1])
                    decoder(ids, use_cache=False)
                    print(json.dumps({'captured': prompt['id'], 'tokens': prompt['tokens']}), flush=True)
        finally:
            for handle in handles:
                handle.remove()
        # Release GPU model after one pass; CPU reference weights remain read-only.
        model.to('cpu')
        torch.mps.empty_cache()
        records = []
        with ExitStack() as stack:
            files = {rank: stack.enter_context((args.output / f'rank{rank}.bin').open('xb')) for rank in (8, 32)}
            zero = stack.enter_context((args.output / 'rank8-zero.bin').open('xb'))
            for rank, output in [*files.items(), (8, zero)]:
                output.write(b'LQ4RES1\n' + struct.pack('<I', rank) + MANIFEST.encode('ascii') + CORE_SHA.encode('ascii'))
            for name in names:
                cal = np.concatenate(captures.pop(name))
                basis = basis_for(cal, 32)
                if basis.shape[1] != 32:
                    raise ValueError(f'calibration rank below fixed 32: {name}')
                weight = modules[name].weight.detach().float().numpy()
                factors = np.empty((weight.shape[0], 32), dtype=np.float32)
                for begin in range(0, len(weight), 256):
                    piece = weight[begin:begin+256]
                    factors[begin:begin+256] = (piece - q4(piece)) @ basis
                for rank, output in files.items():
                    output.write(encode_bf16(basis[:, :rank].T.copy()))
                    output.write(encode_bf16(factors[:, :rank].copy()))
                zero.write(encode_bf16(basis[:, :8].T.copy()))
                zero.write(bytes(weight.shape[0] * 8 * 2))
                records.append({'module': name, 'shape': list(weight.shape), 'calibration_positions': len(cal),
                                'factor_max_abs': float(np.abs(factors).max())})
                print(json.dumps({'prepared': name, 'count': len(records)}), flush=True)
        report.update(calibration=prompts, projections=records, elapsed_seconds=time.perf_counter()-start,
                      memory={'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              'method': 'Darwin getrusage peak RSS, not GPU footprint'})
        report['files'] = {p.name: {'bytes': p.stat().st_size} for p in args.output.glob('*.bin') if not p.name.startswith('.')}
        save(args.output / 'result.json', report)
        print(json.dumps({'output': str(args.output), 'seconds': report['elapsed_seconds']}), flush=True)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
