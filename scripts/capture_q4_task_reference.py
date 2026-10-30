"""Capture only sixteen frozen BF16 task traces and merge the existing twenty references."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from benchmark_native_decode import IDENTITY, MANIFEST, save
from capture_q4_native import (CaptureWorker, STREAMS, assemble_streams, check_loaded,
                               check_result, load_frozen, make_request)
from luban.process_lock import ModelProcessLock
from prepare_q4_packed import read_inputs
from q4_task_trajectories import load_task_trajectories

SPLITS = ('calibration', 'observation')
SCHEMA = 'q4-task-reference-v1'


def layout(plan):
    """The exact phase/case ordering used by selected_inputs, including empty phases."""
    cases = plan['cases']
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate reference case IDs')
    offsets = dict.fromkeys(SPLITS, 0)
    segments = []
    for phase in ('prompt', 'assistant'):
        for case in cases:
            split, positions = case['split'], case['positions']
            prompt = len(case['tokens'])
            if (split not in SPLITS or not prompt or not positions
                    or any(type(p) is not int or not 0 <= p < prompt + len(case['forced_tokens']) for p in positions)
                    or positions != sorted(set(positions))):
                raise ValueError('invalid reference positions or split')
            n_prompt = sum(p < prompt for p in positions)
            begin, end = (0, n_prompt) if phase == 'prompt' else (n_prompt, len(positions))
            first = offsets[split]
            offsets[split] += end - begin
            segments.append({'id': case['id'], 'split': split, 'phase': phase,
                'stream_row_start': first, 'stream_row_end': offsets[split],
                'positions': positions[begin:end], 'carrier_row_start': begin, 'carrier_row_end': end})
    return {'streams': list(STREAMS), 'rows_per_split': offsets, 'segments': segments}


def check_mapping(mapping, plan):
    expected = layout(plan)
    actual = [{k: row.get(k) for k in expected['segments'][0]} for row in mapping.get('segments', [])]
    if (mapping.get('streams') != expected['streams'] or mapping.get('rows_per_split') != expected['rows_per_split']
            or actual != expected['segments']):
        raise ValueError('reference mapping differs from sequential input row order')


def merged_mapping(plan, original_count, original, task):
    result = layout(plan)
    lookup = {}
    for directory, cases in ((original, plan['cases'][:original_count]), (task, plan['cases'][original_count:])):
        for segment in layout({'cases': cases})['segments']:
            lookup[(segment['id'], segment['phase'])] = (str(directory), segment)
    rows = []
    for segment in result['segments']:
        directory, source = lookup[(segment['id'], segment['phase'])]
        segment.update(source_directory=directory, source_row_start=source['stream_row_start'],
                       source_row_end=source['stream_row_end'])
        for i, position in enumerate(segment['positions']):
            rows.append({'split': segment['split'], 'row': segment['stream_row_start'] + i,
                         'id': segment['id'], 'phase': segment['phase'], 'position': position,
                         'source_directory': directory, 'source_row': source['stream_row_start'] + i})
    result.update(order='all prompts then all assistants, plan case order within each split', rows=rows)
    return result


def stream_file(directory, name, split, rows, width):
    path = Path(directory)/f'{name}-{split}.bf16'
    if path.is_symlink() or not path.is_file() or path.stat().st_size != rows * width * 2:
        raise ValueError('reference stream must be regular with exact BF16 dimensions')
    return path


def merge_streams(plan, original_count, original, task, output):
    """Copy immutable sources by phase, never concatenate the two whole calibration files."""
    mapping = merged_mapping(plan, original_count, original, task)
    sources = [(original, {'cases': plan['cases'][:original_count]}),
               (task, {'cases': plan['cases'][original_count:]})]
    for directory, subplan in sources:
        check_mapping(json.loads((directory/'position-mapping.json').read_text()), subplan)
    for name, width in STREAMS.items():
        for split in SPLITS:
            data = {}
            for directory, subplan in sources:
                count = layout(subplan)['rows_per_split'][split]
                path = stream_file(directory, name, split, count, width)
                # Each source stream is read once, without hashing or modifying it.
                raw = np.fromfile(path, dtype='<u2').reshape(count, width)
                if np.any((raw & 0x7f80) == 0x7f80):
                    raise ValueError('nonfinite BF16 reference source')
                data[str(directory)] = raw
            with (output/f'{name}-{split}.bf16').open('xb') as target:
                for segment in mapping['segments']:
                    if segment['split'] == split:
                        target.write(data[segment['source_directory']][segment['source_row_start']:
                                     segment['source_row_end']].tobytes(order='C'))
    save(output/'position-mapping.json', mapping)
    return mapping


def load_reference(directory, receipt, plan, task_source=None, task_source_receipt=None):
    """Bind small provenance/mapping metadata and sizes once; stage reads validate values."""
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text())
    if (result.get('schema') != SCHEMA or result.get('completed') is not True
            or result.get('receipt') != receipt or result.get('plan') != plan
            or result.get('inference_precision') != 'bf16' or result.get('core') is not None
            or result.get('manifest') != MANIFEST or result.get('all_frozen_tokens_equal') is not True
            or result.get('cache_reuse') is not False or result.get('seed') != 42
            or result.get('temperature') != 0 or result.get('captured_task_records') != 16
            or len(plan['cases']) != 36 or result.get('original_count') != 20):
        raise ValueError('completed same-binary thirty-six-trace BF16 reference required')
    if (task_source is not None and result.get('task_trajectories_from') != str(task_source)
            or task_source_receipt is not None and result.get('task_source_receipt') != task_source_receipt):
        raise ValueError('reference task trajectory source changed')
    tasks, captured = result.get('task_records', []), result.get('records', [])
    if len(tasks) != 16 or len(captured) != 16:
        raise ValueError('all sixteen task capture results required')
    for case, record, evidence in zip(plan['cases'][20:], tasks, captured, strict=True):
        request = make_request(record, plan['eos_ids'])
        if (record['split'] != 'calibration' or evidence.get('id') != case['id']
                or record['complete'] is not True or record['generated_tokens'][-1] not in plan['eos_ids']
                or (request['id'], request['tokens'], request['forced_tokens'], request['capture_positions'])
                != (case['id'], case['tokens'], case['forced_tokens'], case['positions'])):
            raise ValueError('captured task trajectory differs from bound plan')
        check_result(evidence['result'], request, record['generated_tokens'])
    mapping = json.loads((directory/'position-mapping.json').read_text())
    expected = merged_mapping(plan, 20, result['original_reference'], directory/'task-capture')
    if mapping != expected or result.get('rows_per_split') != expected['rows_per_split']:
        raise ValueError('reference row provenance differs from bound sequential plan')
    check_loaded(json.loads((directory/'task-capture/loaded.json').read_text()))
    for name, width in STREAMS.items():
        for split in SPLITS:
            stream_file(directory, name, split, expected['rows_per_split'][split], width)
    return {'directory': directory, 'manifest': result}


def read_reference(reference, layer, kind, split, expected_rows):
    if kind not in (1, 2) or split not in SPLITS:
        raise ValueError('invalid reference stream kind/split')
    name = f'layer{layer:02d}-' + ('gateup' if kind == 1 else 'down')
    if name not in STREAMS or reference['manifest']['rows_per_split'][split] != expected_rows:
        raise ValueError('reference stream or row count differs from candidate inputs')
    values = read_inputs(reference['directory'], name, split, STREAMS[name], expected_rows)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite BF16 reference input')
    return values


def run(args):
    # Local import avoids a cycle when the sequential driver loads this reference.
    from prepare_q4_sequential import make_plan
    started = time.perf_counter()
    protocol, original_records = load_frozen(args.source)
    original_plan = make_plan(protocol, original_records)
    prior = json.loads((args.task_trajectories_from/'result.json').read_text())
    source_receipt = prior['receipt']
    tasks = load_task_trajectories(args.task_trajectories_from, source_receipt, original_plan, protocol)
    plan = make_plan(protocol, original_records + tasks)
    binary = args.binary.resolve()
    receipt = json.loads(Path(str(binary)+'.json').read_text())
    control = json.loads((args.control/'result.json').read_text())
    original = json.loads((args.original_reference/'result.json').read_text())
    if (control.get('completed') is not True or control.get('mode') != 'control'
            or control.get('receipt') != receipt or control.get('plan') != plan
            or control.get('all_frozen_tokens_equal') is not True or control.get('streams_compared') != 47
            or control.get('task_trajectories_from') != str(args.task_trajectories_from)
            or control.get('task_source_receipt') != source_receipt
            or any(not control['decisions'][i].get('normal_forward_control', {}).get('verified_values')
                   for i in (0, 12, 23))):
        raise ValueError('existing same-binary full augmented BF16 control required')
    if (original.get('completed') is not True or original.get('all_frozen_tokens_equal') is not True
            or original.get('native_capture_source') != str(args.source) or original.get('manifest') != MANIFEST
            or original.get('core') is not None or len(original.get('records', [])) != 20):
        raise ValueError('completed original twenty-trace BF16 native reference required')
    check_mapping(json.loads((args.original_reference/'position-mapping.json').read_text()), original_plan)
    expected_counts = layout(original_plan)['rows_per_split']
    if original.get('counts') != {name: expected_counts for name in STREAMS}:
        raise ValueError('original reference counts changed')
    check_loaded(json.loads((args.original_reference/'loaded.json').read_text()))
    for record, evidence in zip(original_records, original['records'], strict=True):
        check_result(evidence['result'], make_request(record, protocol['eos_ids']), record['generated_tokens'])
    args.output.mkdir(parents=True, exist_ok=False)
    task_dir = args.output/'task-capture'; task_dir.mkdir()
    native = task_dir/'native'; native.mkdir()
    counts = {name: dict.fromkeys(SPLITS, 0) for name in STREAMS}
    for name in STREAMS:
        for split in SPLITS:
            (task_dir/f'{name}-{split}.bf16').touch(exist_ok=False)
    metadata = {'schema': SCHEMA, 'completed': False, 'receipt': receipt, 'binary': str(binary),
        'plan': plan, 'source': str(args.source), 'original_reference': str(args.original_reference),
        'original_reference_receipt': original['receipt'], 'original_count': 20,
        'control': str(args.control), 'task_trajectories_from': str(args.task_trajectories_from),
        'task_source_receipt': source_receipt, 'inference_precision': 'bf16', 'core': None,
        'manifest': MANIFEST, 'seed': 42, 'temperature': 0, 'cache_reuse': False,
        'teacher_calls': 0, 'training': False, 'product_adopted': False, 'quality_claim': False,
        'hardware': original.get('hardware'), 'activation_execution': 'original native BF16 forced frozen prefixes',
        'task_records': tasks, 'position_mapping': 'position-mapping.json'}
    save(args.output/'protocol.json', metadata)
    completed, worker, lock = [], None, None
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        worker = CaptureWorker(binary, task_dir, native, lock)
        save(task_dir/'loaded.json', worker.loaded)
        for record in tasks:
            event, result = worker.capture(make_request(record, protocol['eos_ids']), record['generated_tokens'])
            completed.append({'id': record['id'], 'capture': event, 'result': result})
            save(args.output/'progress.json', {'completed': False, 'records': completed})
            print(json.dumps({'captured': record['id'], 'rows': event['rows']}), flush=True)
        worker.close()
        if worker.process.returncode != 0:
            raise RuntimeError('native task capture worker failed')
        worker = None
        assemble_streams(tasks, [r['capture'] for r in completed], native, task_dir, counts, protocol['eos_ids'])
        mapping = merge_streams(plan, len(original_records), args.original_reference, task_dir, args.output)
        save(args.output/'result.json', {**metadata, 'completed': True, 'all_frozen_tokens_equal': True,
            'captured_task_records': len(completed), 'records': completed, 'rows_per_split': mapping['rows_per_split'],
            'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'method': 'Darwin Python peak RSS; worker Metal memory in individual result events'}})
    except BaseException as error:
        save(args.output/'failure.json', {'completed': False, 'error': repr(error),
             'records': completed, 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        if worker is not None:
            worker.close()
        if lock is not None:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'original-reference', 'task-trajectories-from', 'control', 'binary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if (not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute()
            or not args.output.resolve().is_relative_to('/Volumes/移动数据库/lubanllm/experiments')
            or not args.binary.is_absolute() or not args.binary.is_file()):
        parser.error('existing absolute worker and new output under mounted experiment volume required')
    run(args)


if __name__ == '__main__':
    main()
