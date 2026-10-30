#!/usr/bin/env python3
"""Prepare an independent CPU NeuralExpert reference; never starts Metal or a model."""
import argparse
import json
import math
import struct
from pathlib import Path
import torch
from luban.dynamic import NeuralExpert


def scalar_bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return struct.unpack('<f', struct.pack('<I', rounded << 16))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.output.is_absolute() or not Path('/Volumes/移动数据库').is_mount():
        raise ValueError('Mounted external output required')
    args.output.relative_to('/Volumes/移动数据库/lubanllm')
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    plans = [dict(rows=rows, width=width, hidden=2048) for rows, width in [(1, 32), (9, 32), (9, 65), (32, 128), (1, 512), (9, 512)]]
    records = []
    for index, plan in enumerate(plans):
        rows, width, hidden = plan['rows'], plan['width'], plan['hidden']
        module = NeuralExpert(hidden, width).to(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for tensor in module.parameters():
                tensor.zero_()
            for i in range(width):
                module.gate.weight[i, (i * 7) % hidden] = (1 if i % 2 else -1) * 0.5
                module.up.weight[i, (i * 11 + 3) % hidden] = 0.25
            for i in range(hidden):
                module.down.weight[i, i % width] = (1 if i % 3 else -1) * 0.5
            x = ((torch.arange(rows * hidden).reshape(rows, hidden) % 31 - 15) / 16).to(torch.bfloat16)
            base = ((torch.arange(rows * hidden).reshape(rows, hidden) % 13 - 6) / 8).to(torch.bfloat16)
            gate, up = module.gate(x), module.up(x)
            act = torch.nn.functional.silu(gate) * up
            expert = module(x)
            combined = base + expert
            # Separate scalar arithmetic checks the actual NeuralExpert BF16
            # SiLU -> multiply -> down -> base-add boundaries, not a float fuse.
            scalar_act = []
            for g, u in zip(gate.float().flatten().tolist(), up.float().flatten().tolist()):
                scalar_act.append(scalar_bf16(scalar_bf16(g / (1 + math.exp(-g))) * u))
            expected_act = torch.tensor(scalar_act).reshape(rows, width).to(torch.bfloat16)
            assert torch.equal(act, expected_act), 'CPU activation reference mismatch'
            scalar_output = [[scalar_bf16(scalar_act[r * width + i % width] * ((1 if i % 3 else -1) * 0.5)) for i in range(hidden)] for r in range(rows)]
            expected_expert = torch.tensor(scalar_output).to(torch.bfloat16)
            assert torch.equal(expert, expected_expert), 'CPU projection order mismatch'
            expected_combined = torch.tensor([[scalar_bf16(float(base[r, i]) + scalar_output[r][i]) for i in range(hidden)] for r in range(rows)]).to(torch.bfloat16)
            assert torch.equal(combined, expected_combined), 'CPU residual order mismatch'
        directory = args.output / str(index)
        directory.mkdir()
        tensors = {'x': x, 'base': base, 'gate': gate, 'up': up, 'act': act, 'expert': expert, 'combined': combined, **module.state_dict()}
        for name, tensor in tensors.items():
            (directory / (name + '.raw')).write_bytes(tensor.detach().contiguous().view(torch.int16).numpy().tobytes())
        records.append(dict(**plan, cpu_independent_equal=True, elements=rows * (width + 2 * hidden)))
    (args.output / 'cases.json').write_text(json.dumps(plans, indent=2) + '\n')
    result = dict(schema='native-q4-expert-cpu-reference-v1', passed=True, seed=42, precision='BF16 boundaries, independent scalar FP32/BF16 reference', cases=records, metal_run=False, model_loaded=False, teacher_calls=0)
    (args.output / 'cpu-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
