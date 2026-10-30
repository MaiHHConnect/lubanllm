# SPDX-License-Identifier: Apache-2.0
"""Export a closed research BF16 replacement; never substitute the FP32 master."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from run_native import acquire, model_identity, OFFICIAL_MANIFEST_SHA

RAW_SHA = '23e4080f2b1e5bd32d75eb98de5b639dacfc300c10061c607e33e8b85fe0cab4'
PT_SHA = '97b43ca20d1344e55c7059de44c1f4103f65af1be44589015e87bbddc4bf7a89'
PARAMETER = 'model.language_model.layers.23.mlp.down_proj.weight'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--closed-index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    closed = json.loads(args.closed_index.read_bytes())
    if not all(closed.get(k) is True for k in ('execution_passed', 'quality_passed', 'pid_dead', 'full_state_payload_verified')):
        raise ValueError('Research candidate did not close validation')
    if closed['product_accepted'] or closed['process_exit_code'] != 0:
        raise ValueError('This export only accepts the closed isolated research candidate')
    raw_path = Path(closed['raw']['path'])
    if closed['raw']['sha256'] != RAW_SHA or digest(raw_path) != RAW_SHA:
        raise ValueError('Closed validation changed')
    raw = json.loads(raw_path.read_bytes())
    training = raw['training']
    saved = training['saved_weights']
    pt = Path(saved['path'])
    if saved['file_sha256'] != PT_SHA or digest(pt) != PT_SHA:
        raise ValueError('Native BF16 checkpoint changed')
    if not raw['passed'] or not training['completed'] or not training['quality_passed'] or not training['checks'][-1]['safety_passed']:
        raise ValueError('Candidate failed final safety or quality checks')
    lock = acquire(model_identity(args.model))
    try:
        import torch
        from safetensors.torch import save_file
        weight = torch.load(pt, weights_only=True, map_location='cpu')
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16 or list(weight.shape) != [2048, 6144]:
            raise ValueError('Expected complete native BF16 down23')
        if not torch.isfinite(weight).all():
            raise ValueError('Nonfinite candidate')
        tensor_sha = hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if tensor_sha != saved['tensor_sha256']:
            raise ValueError('Candidate tensor changed')
        # Read only the original target tensor, not a dequantized/reconstructed copy.
        manifest = json.loads((args.model / 'upstream-file-manifest.json').read_bytes())
        shard = next(args.model / x['Path'] for x in manifest['Data']['Files'] if x['Path'].endswith('.safetensors'))
        with shard.open('rb') as source:
            size = struct.unpack('<Q', source.read(8))[0]
            if size > 4 << 20:
                raise ValueError('Oversized original tensor header')
            header = json.loads(source.read(size))
            entry = header[PARAMETER]
            if entry['dtype'] != 'BF16' or entry['shape'] != [2048, 6144]:
                raise ValueError('Original target dtype or shape changed')
            begin, end = entry['data_offsets']
            source.seek(8 + size + begin)
            original = source.read(end - begin)
        if len(original) != 2048 * 6144 * 2:
            raise ValueError('Original target truncated')
        args.output.mkdir(parents=True)
        target = args.output / 'down23.bf16.safetensors'
        save_file({PARAMETER: weight.contiguous()}, str(target))
        record = {'format': 'luban-native-bf16-core-v1', 'base_identity': model_identity(args.model),
                  'official_manifest_sha256': OFFICIAL_MANIFEST_SHA,
                  'original_tensor_sha256': hashlib.sha256(original).hexdigest(),
                  'parameter': PARAMETER, 'source_parameter': 'model.language_model.layers.23.mlp.base.down_proj.weight',
                  'dtype': str(weight.dtype), 'shape': list(weight.shape), 'file_sha256': digest(target),
                  'native_tensor_sha256': tensor_sha, 'closed_validation_sha256': RAW_SHA,
                  'source_checkpoint': str(pt), 'source_checkpoint_sha256': PT_SHA,
                  'safety_passed': True, 'quality_passed': True, 'branch_enabled': False, 'reference_used': False,
                  'product_adopted': False, 'training_master_exported': False}
        descriptor = args.output / 'provenance.json'
        descriptor.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'descriptor': str(descriptor), 'approved_provenance_sha256': digest(descriptor),
                          'tensor_sha256': tensor_sha, 'bf16_bytes': weight.numel() * weight.element_size()}), flush=True)
        return 0
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
