#!/usr/bin/env python3
"""Repeat existing real-model observation/lifecycle diagnostics; never adopt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[1]
VOLUME = Path('/Volumes/移动数据库')
STORE = VOLUME / 'lubanllm/modules/luban0-v1'


def read(path):
    return json.loads(path.read_text())


def signature(row):
    return [row.get(key) for key in ('id', 'input_tokens', 'tokens', 'complete',
                                     'finish_reason', 'route_action', 'source')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--bun', required=True, type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    assert VOLUME.is_mount() and directory.is_relative_to(VOLUME / 'lubanllm/experiments')
    assert directory != VOLUME / 'lubanllm/experiments' and not directory.exists()
    assert args.bun.is_absolute() and args.bun.is_file()
    assert not os.environ.get('MLX_DISABLE_COMPILE'), 'Test must exercise the worker ACK mode switch'
    directory.mkdir(mode=0o700)
    binary = PROJECT / 'build/native/luban-mlx-worker'
    receipt = read(Path(str(binary) + '.json'))
    # Metadata bytes bind this finite run without rescanning model weights.
    pointers = [STORE / name for name in ('active-core.json', 'active.json', 'active-router.json')]
    before = {str(path): path.read_bytes() for path in pointers}
    result = {'schema': 'mlx-observation-stress-20-v1', 'requested_rounds': 20,
              'scope': 'Isolated fixed-source repetitions; not 20 product learning rounds',
              'worker_sha256': receipt['binary_sha256'], 'complete': False,
              'adopted': False, 'teacher_calls': 0, 'rounds': [], 'started': time.time(),
              'file_cache': 'Not purged; new processes each round; no cross-request KV reuse',
              'seed': 42, 'temperature': 0}
    started = time.monotonic()
    environment = {**os.environ, 'PYTHONPATH': str(PROJECT / 'src')}
    python = PROJECT / '.venv/bin/python'

    def persist():
        result['seconds'] = time.monotonic() - started
        temporary = directory / '.result.json.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(directory / 'result.json')

    def run(command, log):
        with log.open('x') as output:
            return subprocess.run([str(arg) for arg in command], cwd=PROJECT,
                                  env=environment, stdout=output, stderr=subprocess.STDOUT).returncode

    reference = None
    try:
        for index in range(1, 21):
            assert all(path.read_bytes() == before[str(path)] for path in pointers), 'Active source changed'
            assert read(Path(str(binary) + '.json')) == receipt, 'Worker receipt changed'
            folder = directory / f'round-{index:02d}'
            folder.mkdir()
            item = {'round': index, 'complete': False}
            result['rounds'].append(item)
            persist()
            at = time.monotonic()
            item['observation_exit_code'] = run([args.bun, PROJECT / 'scripts/benchmark-mlx-observation.ts',
                '--output', folder / 'observation.json'], folder / 'observation.log')
            observed = read(folder / 'observation.json')
            assert item['observation_exit_code'] == 0 and observed['complete'] and observed['outputs_exact']
            assert observed['worker_sha256'] == receipt['binary_sha256']
            assert observed['learning_observation_compilation_enabled'] is False
            actual = [signature(row) for row in observed['full_pair'][0]]
            if reference is None:
                reference = actual
                result.update(hardware=observed['hardware'], os=observed['os'], precision=observed['precision'],
                              core=observed['core'], observation_expert=observed['expert'])
            item.update(cross_round_exact=actual == reference, checked_generations=148,
                        warmup_generations=2, observation_full_seconds=observed['full_seconds'])
            assert item['cross_round_exact'], 'Full-record output changed across fresh workers'
            qualification = folder / 'qualification'
            item['qualification_exit_code'] = run([python, PROJECT / 'scripts/verify_mlx_native_training.py',
                '--directory', qualification, '--output', qualification / 'result.json', '--binary', binary,
                '--model-metadata', VOLUME / 'lubanllm/models/luban0-mlx-metadata-20261020'],
                folder / 'qualification.log')
            trained = read(qualification / 'result.json')
            item.update(gates=trained['gates'], qualification_seconds=trained['seconds'],
                        actual_adam_updates=sum(row['op'] == 'train_step' and row['event']['event'] == 'train_step'
                                                for row in trained['events']),
                        qualification_generations=len(trained['generations']))
            assert item['qualification_exit_code'] == 0 and trained['complete'] and all(trained['gates'].values())
            assert trained['worker_binary_sha256'] == receipt['binary_sha256']
            assert trained['model_process_lock_released'] and trained['final_process_closed']
            assert trained['teacher_calls'] == 0 and trained['quality_admitted'] is False
            assert all(path.read_bytes() == before[str(path)] for path in pointers), 'Active source changed'
            item.update(complete=True, seconds=time.monotonic() - at)
            persist()
            print(json.dumps(item, ensure_ascii=False), flush=True)
        result.update(complete=True, pointers_unchanged=True,
                      checked_generations=sum(row['checked_generations'] for row in result['rounds']),
                      actual_adam_updates=sum(row['actual_adam_updates'] for row in result['rounds']))
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        persist()


if __name__ == '__main__':
    main()
