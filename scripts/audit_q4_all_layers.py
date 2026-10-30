"""Audit every quantized MLP projection at fixed real inputs, without training.

Weight rows are sampled; layer/projection coverage is complete. Conversion-byte
equality, GPU arithmetic tolerance, and lossy quantization are separate results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmark_native_decode import IDENTITY, save
from benchmark_native_mixed import MixedWorker
from diagnose_native_q4 import bf16, quantize
from luban.process_lock import ModelProcessLock
from trace_q4_divergence import WIDTHS, error_stats, read_bf16


def unpack_words(words):
    return ((np.asarray(words, dtype=np.uint32)[..., None] >>
             (4*np.arange(8, dtype=np.uint32))) & 15).reshape(words.shape[0], -1)


def bf16_ulp(values):
    """Spacing at magnitude, including zero/subnormal BF16 values."""
    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    exponent = np.floor(np.log2(np.maximum(magnitude, 2.**-126)))
    return np.maximum(np.exp2(exponent-7), 2.**-133)


def compare_projection(original, words, scales, biases, inputs, observed, group_size=32):
    n, k = original.shape
    if (group_size not in (16,32) or k % group_size or words.shape != (n, k//8) or scales.shape != (n, k//group_size)
            or biases.shape != scales.shape or inputs.ndim != 2 or inputs.shape[1] != k
            or observed.shape != (len(inputs), n)):
        raise ValueError('projection shape mismatch')
    if not all(np.isfinite(a).all() for a in (original, scales, biases, inputs, observed)):
        raise ValueError('nonfinite audit data')
    if np.any(scales == 0):
        raise ValueError('zero Q4 scale')
    expected = quantize(original.reshape(-1, group_size))
    codes = unpack_words(words)
    conversion = {
        'codes_compared': int(codes.size),
        'code_mismatches': int(np.count_nonzero(codes != expected[4].reshape(n, k))),
        'metadata_values_compared': int(scales.size+biases.size),
        'scale_mismatches': int(np.count_nonzero(scales.view(np.uint32) != expected[7].reshape(scales.shape).view(np.uint32))),
        'bias_mismatches': int(np.count_nonzero(biases.view(np.uint32) != expected[9].reshape(biases.shape).view(np.uint32))),
    }
    conversion['passed'] = not any(conversion[key] for key in ('code_mismatches','scale_mismatches','bias_mismatches'))
    decoded = (codes.reshape(n, k//group_size, group_size).astype(np.float64)*scales[...,None] + biases[...,None]).reshape(n,k)
    reference64 = inputs.astype(np.float64) @ decoded.T
    reference32 = inputs.astype(np.float32) @ decoded.astype(np.float32).T
    rounded64 = bf16(reference64.astype(np.float32))
    rounded32 = bf16(reference32)
    sumabs = np.abs(inputs.astype(np.float64)) @ np.abs(decoded).T
    tolerance = bf16_ulp(rounded64) + 1e-5*sumabs
    errors = np.abs(observed.astype(np.float64)-rounded64)
    source64 = inputs.astype(np.float64) @ original.astype(np.float64).T
    kernel = {
        'outputs_compared': int(observed.size), 'passed': bool(np.all(errors <= tolerance)),
        'outside_tolerance': int(np.count_nonzero(errors > tolerance)),
        'exact_bf16_fp64_fraction': float(np.mean(observed == rounded64)),
        'exact_bf16_fp32_fraction': float(np.mean(observed == rounded32)),
        'max_abs_vs_bf16_fp64': float(errors.max()),
        'max_abs_vs_unrounded_fp64': float(np.max(np.abs(observed-reference64))),
        'max_error_to_tolerance_ratio': float(np.max(errors/tolerance)),
        'fp32_vs_fp64_max_abs': float(np.max(np.abs(reference32-reference64))),
    }
    return {'conversion': conversion, 'kernel': kernel,
            'weight_quantization_loss': error_stats(original, decoded),
            'same_input_output_quantization_loss': error_stats(source64, reference64),
            'gpu_output_vs_source_cpu': error_stats(bf16(source64.astype(np.float32)), observed)}


def prepare_inputs(prior, output):
    protocol = json.loads((prior/'protocol.json').read_text())
    if protocol['layer_widths'] != WIDTHS:
        raise ValueError('unexpected prior layer layout')
    prefixes = [(i, step, spec['id']) for i,spec in enumerate(protocol['cases'])
                for step in range(len(spec['forced_tokens'])+1)]
    if len(prefixes) != 9:
        raise ValueError('expected nine prior real input states')
    states = np.stack([read_bf16(prior/'q4'/'traces'/f'c{i}-step{step:04d}.layers.bf16',
                                24*sum(WIDTHS.values())).reshape(24,-1) for i,step,_ in prefixes])
    offsets, offset = {}, 0
    for name,width in WIDTHS.items():
        offsets[name] = (offset, offset+width)
        offset += width
    output.mkdir()
    for layer in range(24):
        for name in ('mlp_input','act'):
            begin,end = offsets[name]
            # Values came from exact BF16 trace reads; preserve their original bits.
            values = np.ascontiguousarray(states[:,layer,begin:end],dtype=np.float32)
            ((values.view(np.uint32)>>16).astype('<u2')).tofile(output/f'l{layer}-{name}.bf16')
    return protocol, [{'case_index':i,'step':step,'case_id':name} for i,step,name in prefixes]


def request(worker, case, layer, projection, input_path):
    name = f'l{layer}-{projection}'
    payload = {'id':name, 'tokens':case['tokens'], 'max_tokens':1, 'eos_ids':case['eos_ids'],
               'temperature':0, 'seed':42, 'cache_reuse':False, 'audit_quantization':True,
               'audit_layer':layer, 'audit_projection':projection, 'audit_input_path':str(input_path)}
    with (worker.out/'requests.jsonl').open('a') as stream:
        stream.write(json.dumps(payload)+'\n')
    worker.process.stdin.write(json.dumps(payload)+'\n'); worker.process.stdin.flush()
    audit = None
    while True:
        event = worker.receive()
        if event.get('id') != name:
            raise ValueError('audit response id mismatch')
        if event['event'] == 'audit' and audit is None:
            audit = event['audit']
        elif event['event'] == 'result' and audit is not None:
            if not event.get('audit_only') or not event.get('complete') or event.get('finish_reason')!='audit':
                raise ValueError('audit did not complete')
            return audit
        else:
            raise RuntimeError(event)


def read_projection(root, metadata, input_path):
    n,k = metadata['outputs'],metadata['inputs']
    group_size = metadata['group_size']
    indices = metadata['channel_indices']
    expected_shape=(2048,6144) if metadata['projection']=='down' else (6144,2048)
    if (indices != [i*(n-1)//15 for i in range(16)] or (n,k)!=expected_shape
            or group_size not in (16,32) or metadata['rows']!=[1,9]
            or Path(metadata['directory']).resolve()!=root.resolve()):
        raise ValueError('unexpected projection sampling')
    stem = metadata['stem']
    if Path(stem).name != stem:
        raise ValueError('invalid audit stem')
    original = read_bf16(root/(stem+'.original.bf16'),16*k).reshape(16,k)
    words = np.fromfile(root/(stem+'.words.u32'),dtype='<u4')
    if words.size != 16*k//8:
        raise ValueError('wrong packed word count')
    scales = read_bf16(root/(stem+'.scales.bf16'),16*k//group_size).reshape(16,k//group_size)
    biases = read_bf16(root/(stem+'.biases.bf16'),16*k//group_size).reshape(16,k//group_size)
    inputs = read_bf16(input_path,9*k).reshape(9,k)
    results = {}
    for rows in (1,9):
        observed = read_bf16(root/(stem+f'.y{rows}.bf16'),rows*n).reshape(rows,n)[:,indices]
        results[str(rows)] = compare_projection(original,words.reshape(16,k//8),scales,biases,inputs[:rows],observed,group_size)
    return {'layer':metadata['layer'], 'projection':metadata['projection'], 'shape':[n,k],
            'sampled_output_channels':indices, 'rows':results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary','prior','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--group-size', type=int, choices=(16,32), default=32)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.resolve().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    previous,prefixes = prepare_inputs(args.prior,args.output/'inputs')
    exported = (args.output/'native').resolve(); exported.mkdir()
    traces = (args.output/'traces').resolve(); traces.mkdir()
    save(args.output/'protocol.json',{
        'schema':'q4-all-layer-conversion-operator-audit-v1', 'binary':str(args.binary.resolve()),
        'prior':str(args.prior.resolve()), 'input_states':prefixes, 'layers':24,'quantized_projections':71,
        'weight_sampling':'16 equally spaced complete output rows per projection, all input channels',
        'precision':f'original BF16, group{args.group_size} Q4 codes and BF16 metadata, native GPU output BF16',
        'reference':'existing independent NumPy FP32 ties-away quantizer; literal low-nibble-first unpack',
        'kernel_tolerance':'abs(GPU-BF16(CPUFloat64dot)) <= one BF16 ULP(reference) + 1e-5 sum(abs(x*wq))',
        'operator_rows':[1,9], 'quantization_loss_reference':'CPU float64 dot using original bound BF16 vs decoded Q4 on identical inputs',
        'limits':'All projection coverage, sampled output channels and nine known-case inputs; no BF16 GPU equivalence or all-weight proof',
        'memory':'worker payload/footprint from loaded event; no full-process peak sampling',
        'seed':42,'training':False,'teacher_calls':0,'product_adopted':False})
    lock = ModelProcessLock(IDENTITY,'mps')
    records=[]
    try:
        worker = MixedWorker(args.binary.resolve(),'candidate_mixed',args.output,lock,f'mixed-q4-g{args.group_size}',
                             trace_directory=traces,intervention=True,audit_directory=exported)
        try:
            save(args.output/'loaded.json',worker.loaded)
            if not worker.loaded.get('research_intervention') or worker.loaded.get('quantized_projections') != 71:
                raise ValueError('wrong worker graph')
            with (args.output/'observations.jsonl').open('x') as stream:
                for layer in range(24):
                    for projection in ('gate','up','down'):
                        if layer==23 and projection=='down':
                            continue
                        path=(args.output/'inputs'/f'l{layer}-{"act" if projection=="down" else "mlp_input"}.bf16').resolve()
                        metadata=request(worker,previous['cases'][0]['case'],layer,projection,path)
                        if metadata['group_size']!=args.group_size:
                            raise ValueError('returned wrong quantization group')
                        if metadata['layer']!=layer or metadata['projection']!=projection:
                            raise ValueError('returned wrong projection')
                        row=read_projection(exported,metadata,path);records.append(row)
                        stream.write(json.dumps(row)+'\n');stream.flush()
                        save(args.output/'progress.json',{'completed_projections':len(records),'elapsed_seconds':time.perf_counter()-start})
        finally:
            worker.close()
        if worker.process.returncode:
            raise RuntimeError('audit worker failed')
    finally:
        lock.close()
    conversion_passed=all(r['rows']['1']['conversion']['passed'] for r in records)
    kernel_passed=all(r['rows'][str(n)]['kernel']['passed'] for r in records for n in (1,9))
    result={'completed':len(records)==71,'conversion_passed':conversion_passed,'kernel_passed':kernel_passed,
            'projections':records,'elapsed_seconds':time.perf_counter()-start,'product_adopted':False}
    save(args.output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='projections'}),flush=True)
    if not conversion_passed or not kernel_passed:
        raise SystemExit('audit failed; preserve evidence and investigate')


if __name__=='__main__':
    main()
