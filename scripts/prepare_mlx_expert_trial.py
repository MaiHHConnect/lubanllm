#!/usr/bin/env python3
"""Prepare read-only old-expert migration inputs; never publish or grant a proof.

CPU only. The parent run owns the GPU lease and validates the final Swift worker.
"""
import argparse
import json
from pathlib import Path
import platform

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer


ROOT = Path('/Volumes/移动数据库/lubanllm')
STORE = ROOT / 'modules/luban0-v1'
STUDY = ROOT / 'models/community-q4-study-20261017'
SOURCE_ID = 'learned-8ec555e81b20'
TARGET_ID = 'trial-mlx-8ec555e81b20'


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def expert_oracle(weights, path):
    """Independent CPU reference for the exact candidate's three matrices."""
    generator = torch.Generator(device='cpu').manual_seed(42)
    x = torch.randn(1, 17, 2048, generator=generator, dtype=torch.float32)
    x = (x / x.square().mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
    tensors = {'input': x}
    with torch.no_grad():
        for label, dtype in [('fp32', torch.float32), ('bf16', torch.bfloat16)]:
            gate = torch.nn.functional.linear(x.to(dtype), weights['gate.weight'].to(dtype))
            up = torch.nn.functional.linear(x.to(dtype), weights['up.weight'].to(dtype))
            activated = torch.nn.functional.silu(gate) * up
            result = torch.nn.functional.linear(activated, weights['down.weight'].to(dtype))
            assert torch.isfinite(result).all() and result.abs().max() > 0
            for key, value in [('gate', gate), ('up', up), ('activation', activated), ('output', result)]:
                tensors[label + '_' + key] = value
    save_file(tensors, str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/Volumes/移动数据库').is_mount():
        parser.error('external research volume is not mounted')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'experiments') or output.exists():
        parser.error('--output must be a new external experiments directory')
    active = json.loads((STORE / 'active.json').read_text())
    core = json.loads((STORE / 'active-core.json').read_text())
    source_meta_path = STORE / 'layer-23' / (SOURCE_ID + '.json')
    source = json.loads(source_meta_path.read_text())
    weights_path = source_meta_path.with_suffix('.safetensors')
    assert active['id'] == SOURCE_ID and active['core_version'] is None
    assert source['core_version'] is None and source['hidden'] == 2048 and source['width'] == 64
    assert source['identity'] == core['identity'] + ':layer-23' and source['requires_gate'] is True
    weights = load_file(str(weights_path), device='cpu')
    shapes = {'gate.weight': (64, 2048), 'up.weight': (64, 2048), 'down.weight': (2048, 64)}
    assert set(weights) == set(shapes)
    for name, tensor in weights.items():
        assert tuple(tensor.shape) == shapes[name] and tensor.dtype == torch.bfloat16
        assert torch.isfinite(tensor).all()
    output.mkdir(mode=0o700)
    # A candidate descriptor is not an active pointer or an accepted new expert.
    candidate = {**source, 'id': TARGET_ID, 'core_version': core['version'],
                 'quality_admitted': False,
                 'migration': {'scope': 'local-trial', 'status': 'prepared-unvalidated',
                               'source_id': SOURCE_ID, 'source_core_version': None,
                               'source_metadata': str(source_meta_path),
                               'source_weights': str(weights_path),
                               'source_sha256': source['sha256'],
                               'target_backend': 'mlx', 'optimizer_updates': 0}}
    save(output / 'candidate-metadata.json', candidate)
    existing = json.loads((STUDY / 'comparison-cases.json').read_text())
    chosen_ids = {'basic-arithmetic', 'basic-unit', 'basic-user_given',
                  'q4-id-user-boundary-0', 'q4-id-user-boundary-1', 'q4-id-user-boundary-2'}
    cases = [row for row in existing if row['category'] == 'natural_identity' or row['id'] in chosen_ids]
    tokenizer = Tokenizer.from_file(str(STUDY / 'mlx-community-4bit/tokenizer.json'))
    for row in source['preservation_rows']:
        provenance = json.loads(row['provenance'])
        messages = provenance['messages']
        prompt = ''.join('<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'
                         for message in messages) + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        cases.append({'id': row['id'], 'category': 'old_expert_preservation',
                      'prompt': prompt, 'input_tokens': tokenizer.encode(prompt).ids,
                      'max_tokens': 128, 'expected': row['answer'], 'messages': messages})
    assert len(cases) == 16 and len({row['id'] for row in cases}) == 16
    save(output / 'cases.json', cases)
    # Fixed externally normalized layer-23 inputs; the expert has no internal RMSNorm.
    expert_oracle(weights, output / 'expert-math-oracle.safetensors')
    save(output / 'preparation.json', {
        'schema': 'mlx-expert-trial-preparation-v1', 'prepared': True,
        'gpu_loaded': False, 'math_passed': False, 'runtime_checked': False,
        'quality_admitted': False, 'production_pointers_changed': False,
        'source_expert_id': SOURCE_ID, 'target_expert_id': TARGET_ID,
        'source_core_version': None, 'target_core_version': core['version'],
        'source_expert_file_sha256': source['sha256'], 'core_file_sha256': core['sha256'],
        'checksum_note': 'Inherited source receipt; final consumer must verify checkpoint binding once at load.',
        'identity': core['identity'], 'width': 64, 'layer': 23, 'cases': len(cases),
        'oracle': {'device': 'cpu', 'torch_version': torch.__version__,
                   'platform': platform.platform(), 'seed': 42, 'shape': [1, 17, 2048],
                   'formula': 'down(silu(gate(x)) * up(x)); layer RMSNorm is outside branch',
                   'comparison_pending': True,
                   'fp32_tolerance': {'atol': 1e-5, 'rtol': 1e-4},
                   'bf16_reporting': 'Report exact equality/max error/relative L2; no presumed cross-kernel bitwise equality.'},
        'optimizer_updates': 0, 'teacher_calls': 0,
        'limitations': ['Prepared candidate is not published and grants no inference or training qualification.',
                       'Old router also has core_version=null; its old calibration is not transferred.',
                       'Six natural questions and all fixed checks were already exposed; not a blind evaluation.',
                       'The source preservation prompt about 我是谁 is ambiguous; unknown-user cases stay separate.']})
    print(json.dumps({'output': str(output), 'cases': len(cases), 'gpu_loaded': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
