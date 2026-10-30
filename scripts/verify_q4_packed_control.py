"""Same-binary original-core RTN loader equivalence and all71 projection audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from audit_q4_all_layers import request as audit_request, read_projection
from benchmark_native_decode import IDENTITY, MODEL, MANIFEST, save
from benchmark_native_mixed import MixedWorker
from luban.process_lock import ModelProcessLock
from prepare_q4_packed import PROJECTIONS, input_name


def controls(prior):
    rows = [json.loads(line) for line in (prior / 'answers.jsonl').read_text().splitlines()]
    chosen = []
    for temperature, count in ((0, 4), (.6, 2)):
        for phase in ('ranking', 'preservation', 'fresh'):
            candidates = [r for r in rows if r['precision'] == 'mixed-q4-g16' and r['phase'] == phase
                          and r['temperature'] == temperature and r['execution_action'] == 'local_core'
                          and r['engine']['engine'] == 'luban-native-mixed-q4-g16-metal']
            if len(candidates) < count:
                raise ValueError('insufficient existing native cases')
            for i in np.linspace(0, len(candidates)-1, count, dtype=int):
                row = candidates[i]
                chosen.append({'id': f'{phase}-t{temperature}-{row["case_id"]}',
                               'tokens': row['prompt_tokens'], 'max_tokens': row['usage']['output_limit'],
                               'temperature': temperature, 'seed': 42})
    if len(chosen) != 18 or len({r['id'] for r in chosen}) != 18:
        raise ValueError('invalid control selection')
    return chosen


def prepare_reference(args):
    source = json.loads((args.captures / 'result.json').read_text())
    if not source['completed'] or source['core'] is not None or source['manifest'] != MANIFEST:
        raise ValueError('expected completed original-core activation capture')
    cases = controls(args.prior)
    config = json.loads((Path(MODEL) / 'config.json').read_text())
    eos = config.get('text_config', config)['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    for case in cases:
        case['eos_ids'] = eos
    (args.output / 'inputs').mkdir()
    for name, k in {(input_name(layer, kind), k) for layer, kind, _, k in PROJECTIONS}:
        # Fixed first nine observation rows, used only as operator inputs, never code selection.
        raw = np.fromfile(args.captures / f'{name}-observation.bf16', dtype='<u2', count=9*k)
        if raw.size != 9*k or np.any((raw & 0x7f80) == 0x7f80):
            raise ValueError('invalid audit input rows')
        raw.tofile(args.output / 'inputs' / f'{name}.bf16')
    protocol = {'schema': 'q4-packed-rtn-loader-control-v1', 'binary': str(args.binary),
                'receipt': json.loads(Path(str(args.binary) + '.json').read_text()),
                'manifest': MANIFEST, 'core': None, 'captures': str(args.captures),
                'cases': cases, 'case_selection': '4 greedy and2 seed42 sampled native cases per known phase, uniformly spaced in original order',
                'audit_inputs': 'first9 frozen observation positions per layer, same across both modes',
                'operator_sampling': 'all71 projections;16 equally spaced output rows, all input columns; GPU rows1 and9',
                'scope': 'loader/operator equivalence only; no semantic quality or final-holdout claim',
                'cache_reuse': False, 'teacher_calls': 0, 'training': False, 'product_adopted': False}
    save(args.output / 'protocol.json', protocol)
    return protocol


def run(args, protocol):
    out = args.output / args.stage
    out.mkdir()
    (out / 'traces').mkdir(); (out / 'native').mkdir()
    precision = 'mixed-q4-g16' if args.stage == 'reference' else 'mixed-q4-g16-packed'
    sidecar = None
    if args.stage == 'packed':
        prepared = json.loads((args.weights / 'result.json').read_text())
        if not prepared['completed'] or len(prepared['projections']) != 71:
            raise ValueError('packed preparation is not complete')
        sidecar = args.weights / 'rtn.bin'
        if sidecar.stat().st_size != 670040776:
            raise ValueError('incorrect packed size')
        with sidecar.open('rb') as stream:
            header = stream.read(144)
        if header[:8] != b'LQ4PK01\n' or header[16:80] != MANIFEST.encode() or header[80:] != b'0'*64:
            raise ValueError('incorrect packed source header')
    started = time.perf_counter()
    lock = ModelProcessLock(IDENTITY, 'mps')
    generated, audited = [], []
    try:
        worker = MixedWorker(args.binary, 'candidate_mixed', out, lock, precision, sidecar,
                             trace_directory=out/'traces', intervention=True, audit_directory=out/'native', core=None)
        try:
            save(out / 'loaded.json', worker.loaded)
            if worker.loaded['quantized_projections'] != 71 or not worker.loaded['research_intervention']:
                raise ValueError('wrong quantized graph')
            for case in protocol['cases']:
                result = worker.request(case['tokens'], case['id'], max_tokens=case['max_tokens'],
                                        temperature=case['temperature'], seed=42, reuse=False, eos_ids=case['eos_ids'])
                generated.append(result)
                save(out / 'generation.json', generated)
            for layer, kind, _, _ in PROJECTIONS:
                path = args.output / 'inputs' / f'{input_name(layer, kind)}.bf16'
                metadata = audit_request(worker, protocol['cases'][0], layer, kind, path)
                if metadata['group_size'] != 16 or metadata['layer'] != layer or metadata['projection'] != kind:
                    raise ValueError('audit projection mismatch')
                audited.append(read_projection(out/'native', metadata, path))
                save(out / 'progress.json', {'generated': len(generated), 'audited': len(audited),
                                             'elapsed_seconds': time.perf_counter()-started})
        finally:
            worker.close()
        if worker.process.returncode:
            raise RuntimeError('worker failed')
    finally:
        lock.close()
    audit_ok = all(r['rows']['1']['conversion']['passed'] and
                   all(r['rows'][str(n)]['kernel']['passed'] for n in (1,9)) for r in audited)
    result = {'completed': len(generated)==18 and len(audited)==71, 'conversion_and_kernel_passed': audit_ok,
              'generations': generated, 'projections': audited, 'elapsed_seconds': time.perf_counter()-started}
    save(out / 'result.json', result)
    if not result['completed'] or not audit_ok:
        raise ValueError('RTN operator control failed')
    return result


def compare(output, candidate):
    reference = json.loads((output / 'reference/result.json').read_text())
    if not reference['completed'] or not reference['conversion_and_kernel_passed']:
        raise ValueError('reference control did not pass')
    before = {row['id']: row for row in reference['generations']}
    after = {row['id']: row for row in candidate['generations']}
    if len(before) != 18 or before.keys() != after.keys():
        raise ValueError('generation control coverage mismatch')
    changed = [key for key in before if any(before[key][field] != after[key][field]
               for field in ('tokens','complete','finish_reason'))]
    source = {p.name:p for p in (output/'reference/native').iterdir() if not p.name.startswith('._')}
    loaded = {p.name:p for p in (output/'packed/native').iterdir() if not p.name.startswith('._')}
    if source.keys() != loaded.keys() or len(source) != 71*6:
        raise ValueError('audit file coverage mismatch')
    # Byte comparison is the numerical equivalence test, not an extra provenance hash.
    different = [name for name in source if source[name].read_bytes() != loaded[name].read_bytes()]
    report = {'completed': True, 'passed': not changed and not different,
              'exact_generation_controls': 18-len(changed), 'changed_generation_ids': changed,
              'exact_audit_files': len(source)-len(different), 'different_audit_files': different,
              'full_projection_coverage': 71, 'source_and_loaded_kernel_checks_passed': True,
              'codes_compared_per_mode': sum(r['rows']['1']['conversion']['codes_compared'] for r in candidate['projections']),
              'metadata_values_compared_per_mode': sum(r['rows']['1']['conversion']['metadata_values_compared'] for r in candidate['projections']),
              'gpu_outputs_compared_per_mode': sum(r['rows'][str(n)]['kernel']['outputs_compared'] for r in candidate['projections'] for n in (1,9)),
              'elapsed_seconds': reference['elapsed_seconds'] + candidate['elapsed_seconds'],
              'memory_scope': 'loaded native footprint includes retained BF16 research source; not app production peak',
              'scope': 'same-binary original-core RTN load and operator equivalence; no GPTQ quality claim',
              'teacher_calls': 0, 'product_adopted': False}
    save(output/'result.json', report)
    if not report['passed']:
        raise ValueError('packed RTN differs from on-load RTN')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('reference','packed'))
    for name in ('binary','output','captures','weights','prior'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('binary','output','captures','weights','prior'):
        setattr(args,name,getattr(args,name).absolute())
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    if args.stage == 'reference':
        args.output.mkdir(exist_ok=False)
        protocol = prepare_reference(args)
    else:
        protocol = json.loads((args.output/'protocol.json').read_text())
        if protocol['binary'] != str(args.binary) or protocol['captures'] != str(args.captures):
            raise ValueError('frozen control source changed')
    try:
        result = run(args,protocol)
        if args.stage == 'packed':
            result = compare(args.output,result)
        print(json.dumps({key:value for key,value in result.items() if key not in ('projections','generations')},ensure_ascii=False),flush=True)
    except BaseException as error:
        save(args.output/f'{args.stage}-failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__ == '__main__':
    main()
