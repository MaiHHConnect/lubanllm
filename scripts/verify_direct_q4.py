#!/usr/bin/env python3
"""One saved carrier, two cold worker starts; frozen prior Q4 observations."""
import argparse
import json
from pathlib import Path
import platform
import threading
import time

from luban.native_engine import NativeEngine
from luban.process_lock import ModelProcessLock
from luban.context import available_memory_bytes, GIB

MODEL = Path('/Volumes/移动数据库/lubanllm/models/luban0')
ROOT = Path('/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017')
IDENTITY = '673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df'
MANIFEST = 'd8b0ead81edafada4883e6fea37d1750625a682c469b5f70eb760cd2ab0b1237'
CORE = {'file': '/Volumes/移动数据库/lubanllm/modules/luban0-v1/core-fdfa0ce5a75c.safetensors',
        'tensor_key': 'model.language_model.layers.23.mlp.base.down_proj.weight',
        'sha256': 'bed06ece53a5ee6a4eff0c00161868d0011cc39d16ece246b8bffb3f1a360e64'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.is_absolute() and not args.output.exists()
    assert Path('/Volumes/移动数据库').is_mount()
    args.output.mkdir()
    prior_path = ROOT / 'identity-import-v4/result.json'
    prior = json.loads(prior_path.read_text())['report']
    expected = prior['partial_candidate_first']
    assert len(expected) == 61 and expected == prior['partial_candidate_second']
    assert prior['generation_precision'] == 'mixed-q4-g16'
    receipt = json.loads(Path(str(args.binary) + '.json').read_text())
    quantizer = receipt['binary_sha256']
    carrier = args.output / 'model-q4.safetensors'
    stop = threading.Event()
    result = {'schema': 'direct-q4-real-regression-v1', 'hardware': platform.platform(),
              'system_ram_bytes': 16 * GIB, 'precision': 'mixed-q4-g16', 'layers': 24,
              'seed': 42, 'temperature': 0, 'cache_reuse': False, 'core': CORE,
              'reference': str(prior_path), 'reference_scope': 'existing nonblind same-Q4 observations',
              'optimizer_steps': 0, 'teacher_calls': 0, 'passes': [],
              'file_cache': 'not purged; second start reuses the same saved carrier',
              'measurement': 'native proc_pid_rusage physical footprint and lifetime peak, not system peak'}
    lock = ModelProcessLock(IDENTITY, 'mps')
    started = time.monotonic()
    try:
        assert available_memory_bytes() >= GIB + (128 << 20), 'Insufficient prepare memory'
        prepare = NativeEngine(args.binary, MODEL, MANIFEST, load_timeout=1800)
        at = time.monotonic()
        event = prepare.prepare_direct_q4(carrier, quantizer, stop)
        result['prepare'] = {'seconds': time.monotonic() - at, 'event': event}
        print(json.dumps({'prepare': result['prepare']}), flush=True)
        binding = {'file': str(carrier), 'sha256': event['direct_q4_sha256'],
                   'quantizer_sha256': quantizer, 'weight_bytes': event['direct_q4_weight_bytes']}
        for repeat in range(2):
            available = available_memory_bytes()
            assert available >= binding['weight_bytes'] + 50_331_648 + GIB, 'Insufficient direct load memory'
            engine = NativeEngine(args.binary, MODEL, MANIFEST, core=CORE,
                                  precision='mixed-q4-g16', direct_q4=binding)
            rows = []
            actual_pass = {'repeat': repeat, 'available_before': available, 'rows': rows}
            result['passes'].append(actual_pass)
            try:
                at = time.monotonic()
                actual_pass['loaded'] = engine.start()
                actual_pass['load_seconds'] = time.monotonic() - at
                assert engine.loaded['base_identity'] == IDENTITY
                for sample in expected:
                    at = time.monotonic()
                    # A length-ended reference uses its original exact budget.
                    maximum = len(sample['tokens']) if sample['finish_reason'] == 'length' else 192
                    actual = engine.generate(sample['input_tokens'], maximum, [248044], 0,
                                             lambda _: None, stop, seed=42, cache_reuse=False)
                    row = {'id': sample['id'], 'seconds': time.monotonic() - at, 'result': actual,
                           'exact': actual['tokens'] == sample['tokens'] and actual['finish_reason'] == sample['finish_reason']}
                    rows.append(row)
                    print(json.dumps({'repeat': repeat, 'id': row['id'], 'exact': row['exact']}), flush=True)
                    assert row['exact'], 'Direct Q4 differs: ' + sample['id']
            finally:
                engine.close()
        result['passed'] = True
    except BaseException as error:
        result['passed'] = False
        result['error'] = repr(error)
        raise
    finally:
        lock.close()
        result['seconds'] = time.monotonic() - started
        (args.output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
