"""Frozen real-activation GPTQ preparation for the original core; no model writes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
from numbers import Real
import os
from pathlib import Path
import platform
import resource
import struct
import time

import numpy as np
import torch
from safetensors import safe_open

from benchmark_native_decode import MODEL, MANIFEST, IDENTITY, save
from diagnose_native_q4 import bf16, quantize
from luban.backbone import load_backbone, text_backbone
from luban.process_lock import ModelProcessLock

GROUP = 16
BLOCK = 128
DAMPING = .01
METHODS = ('rtn', 'stored-grid', 'gptq')
PROJECTIONS = [(layer, kind, 2048 if kind == 'down' else 6144,
                6144 if kind == 'down' else 2048)
               for layer in range(24) for kind in ('gate', 'up', 'down')
               if (layer, kind) != (23, 'down')]


def materials():
    """Authored calibration inputs; no benchmark answers or teacher outputs."""
    topics = [
        '请介绍潮汐为什么具有周期性，并区分太阳和月球的影响。',
        'Explain how a library catalog helps readers find books by author, title and subject.',
        '请解释整数、小数、负数、分数之间的区别，并说明什么时候需要保留小数位。',
        'Describe stable sorting, duplicate removal, and why input order sometimes matters.',
        '请介绍UTF-8编码，以及汉字、空格、下划线、连字符在文本处理中有什么区别。',
        'Explain the difference between a JSON string, number, list, object, boolean and null.',
        '请说明Python函数参数、返回值、异常处理和字典查询的关系。',
        'Write a short explanation of HTTP methods, status codes, headers and request bodies.',
        '请比较温度、长度、面积、质量的常见单位，说明换算时怎样保持单位一致。',
        'Describe a small test plan for a text editor, including selection, insertion and saving.',
        '请解释递归、循环、边界条件和终止条件，不需要引用任何具体产品。',
        'Explain how plants use sunlight and how water moves between roots and leaves.',
    ]
    rows = [{'id': f'cal-{i:02d}', 'split': 'calibration', 'text': text}
            for i, text in enumerate(topics)]
    for i, (prefix, line, question) in enumerate([
        ('档案记录的是港口仓储流程。\n', '每日完成到货登记、货架盘点、温度测量和交接确认。\n', '总结这些记录的共同主题。'),
        ('This document describes a city bus network.\n', 'Stops are inspected, timetables reviewed, and vehicle maintenance logged.\n', 'Summarize the activities described.'),
        ('说明：以下为程序设计笔记。\n', '输入先经过验证，数据保留原顺序，异常单独处理，结束时释放资源。\n', '整理出笔记中的执行顺序。'),
        ('Notes about a community garden follow.\n', 'Volunteers water seedlings, check soil, label plants, and record growth.\n', 'Explain the purpose of keeping these records.'),
    ]):
        rows.append({'id': f'cal-long-{i}', 'split': 'calibration',
                     'text': prefix + line * (24 if i % 2 else 48) + question})
    for i, text in enumerate([
        '请介绍博物馆如何保存和登记历史器物，并解释温湿度控制的作用。',
        'Explain why a database transaction should either complete or roll back.',
        '写一段关于山区步道维护的说明，包含雨水排放和路标检查。',
        'The notes concern a railway workshop.\n' + 'Technicians inspect wheels, record measurements, and schedule repairs.\n' * 36 + 'Summarize the maintenance process.',
    ]):
        rows.append({'id': f'observe-{i}', 'split': 'observation', 'text': text})
    return rows


def input_name(layer, kind):
    return f'layer{layer:02d}-' + ('down' if kind == 'down' else 'gateup')


def pack_codes(codes):
    values = np.asarray(codes)
    if (values.ndim != 2 or values.shape[1] % 8 or not np.isfinite(values).all()
            or np.any(values != np.floor(values)) or np.any((values < 0) | (values > 15))):
        raise ValueError('finite integral 4-bit row-major codes required')
    return np.bitwise_or.reduce(values.astype(np.uint32).reshape(*values.shape[:1], -1, 8)
                               << (4 * np.arange(8, dtype=np.uint32)), axis=-1).astype('<u4')


def metadata(weights):
    if weights.ndim != 2 or weights.shape[1] % GROUP or not np.isfinite(weights).all():
        raise ValueError('finite matrix with group16 columns required')
    result = quantize(weights.reshape(-1, GROUP))
    shape = (weights.shape[0], weights.shape[1] // GROUP)
    return (result[4].reshape(weights.shape).astype(np.uint8),
            result[5].reshape(weights.shape).astype(np.uint8),
            result[7].reshape(shape), result[9].reshape(shape))


def inverse_factor(inputs, *, damping_fraction=DAMPING):
    """Full cross-group Hessian; FP64 factorization, FP32 weight feedback."""
    if (isinstance(damping_fraction, bool) or not isinstance(damping_fraction, Real)
            or not np.isfinite(damping_fraction) or not 0 < damping_fraction <= 1):
        raise ValueError('damping fraction must be finite and in (0, 1]')
    damping_fraction = float(damping_fraction)
    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('finite calibration inputs required')
    tensor = torch.from_numpy(x)
    hessian = (tensor.T @ tensor / len(x)).double()
    diagonal = hessian.diagonal()
    mean = float(diagonal.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('calibration has no activation energy')
    dead = diagonal == 0
    damping = damping_fraction * mean
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError('damping must produce a positive finite diagonal increment')
    diagonal.add_(damping)
    lower = torch.linalg.cholesky(hessian)
    del hessian
    inverse = torch.cholesky_inverse(lower)
    del lower
    upper = torch.linalg.cholesky(inverse, upper=True).float().triu_()
    if not torch.isfinite(upper).all() or torch.any(upper.diagonal() <= 0):
        raise ValueError('invalid inverse Hessian factor')
    return upper, {'rows': len(x), 'columns': x.shape[1], 'damping': damping,
                   'damping_fraction': damping_fraction,
                   'dead_columns': torch.where(dead)[0].tolist()}


def asymmetric_factor(fp_inputs, inputs, upper, *, alpha=.25):
    """GPTAQ Eq.17: FP32 paired-input cross moment and strict-upper correction."""
    if (isinstance(alpha, bool) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or alpha < 0):
        raise ValueError('asymmetric alpha must be finite and nonnegative')
    fp, x = (np.asarray(value, dtype=np.float32) for value in (fp_inputs, inputs))
    if (x.ndim != 2 or len(x) < 2 or x.shape[1] == 0 or fp.shape != x.shape
            or not np.isfinite(x).all() or not np.isfinite(fp).all()
            or not isinstance(upper, torch.Tensor) or upper.shape != (x.shape[1], x.shape[1])
            or upper.device.type != 'cpu' or upper.dtype != torch.float32
            or not torch.isfinite(upper).all() or torch.any(upper.diagonal() <= 0)):
        raise ValueError('finite paired calibration matrices and FP32 CPU factor required')
    # Explicitly clear the unused triangle; some CPU Cholesky backends leave it populated.
    factor = upper.triu()
    current, reference = torch.from_numpy(x), torch.from_numpy(fp)
    cross = (reference - current).T @ current / len(x)
    correction = (float(alpha) * (cross @ factor.T).triu_(diagonal=1)) @ factor
    correction.triu_(diagonal=1)
    if not torch.isfinite(cross).all() or not torch.isfinite(correction).all():
        raise ValueError('nonfinite asymmetric input correction')
    return correction


def gptq_quantize(weights, scales, biases, upper, block=BLOCK, *, refit_grid=False, correction=None):
    """Stored BF16 GPTQ grids, optionally with GPTAQ pre-quantization-column feedback."""
    if weights.ndim != 2 or type(refit_grid) is not bool:
        raise ValueError('invalid GPTQ matrix or grid mode')
    n, k = weights.shape
    if (n == 0 or k == 0 or k % GROUP or block <= 0 or block % GROUP or upper.shape != (k, k)
            or scales.shape != (n, k // GROUP) or biases.shape != scales.shape
            or not all(np.isfinite(v).all() for v in (weights, scales, biases))
            or np.any(scales == 0) or not torch.isfinite(upper).all()
            or torch.any(upper.diagonal() <= 0)):
        raise ValueError('invalid GPTQ dimensions or finite grid')
    if correction is not None:
        if (not isinstance(correction, torch.Tensor) or correction.shape != (k, k)
                or correction.device.type != 'cpu' or correction.dtype != torch.float32
                or not torch.isfinite(correction).all() or torch.count_nonzero(correction.tril())):
            raise ValueError('finite strict-upper FP32 CPU correction required')
        if not torch.count_nonzero(correction):
            correction = None  # Zero input deviation must reproduce the old path exactly.
    work = torch.from_numpy(weights.copy())
    scale, bias = torch.from_numpy(scales.copy()), torch.from_numpy(biases.copy())
    codes = torch.empty((n, k), dtype=torch.uint8)
    for begin in range(0, k, block):
        end = min(begin + block, k)
        current = work[:, begin:end].clone()
        errors = torch.empty_like(current)
        preupdates = torch.empty_like(current) if correction is not None else None
        local_upper = upper[begin:end, begin:end]
        for offset in range(end - begin):
            col = begin + offset
            if refit_grid and offset % GROUP == 0:
                _, _, group_scale, group_bias = metadata(current[:, offset:offset+GROUP].numpy())
                if (not np.isfinite(group_scale).all() or not np.isfinite(group_bias).all()
                        or np.any(group_scale == 0)):
                    raise ValueError('nonfinite or zero refitted GPTQ grid')
                scale[:, col//GROUP:col//GROUP+1] = torch.from_numpy(group_scale)
                bias[:, col//GROUP:col//GROUP+1] = torch.from_numpy(group_bias)
            value = current[:, offset]
            if preupdates is not None:
                value = value.clone()
                preupdates[:, offset] = value
            raw = (value - bias[:, col // GROUP]) / scale[:, col // GROUP]
            code = torch.copysign(torch.floor(torch.abs(raw) + .5), raw).clamp_(0, 15)
            decoded = scale[:, col // GROUP] * code + bias[:, col // GROUP]
            error = (value - decoded) / local_upper[offset, offset]
            codes[:, col] = code.to(torch.uint8)
            feedback = error[:, None] * local_upper[offset, offset:][None, :]
            if correction is not None:
                feedback -= value[:, None] * correction[col, col:end][None, :]
            current[:, offset:] -= feedback
            errors[:, offset] = error
        if not torch.isfinite(current).all() or not torch.isfinite(errors).all():
            raise ValueError('nonfinite GPTQ block feedback')
        feedback = errors @ upper[begin:end, end:]
        if correction is not None:
            # Each column was captured after earlier feedback, before its own quantization.
            # The mutated block contains decoded weights and is not equivalent here.
            feedback -= preupdates @ correction[begin:end, end:]
        work[:, end:] -= feedback
    if not torch.isfinite(work).all():
        raise ValueError('nonfinite GPTQ feedback')
    return codes.numpy(), scale.numpy(), bias.numpy()


def gptq_codes(weights, scales, biases, upper, block=BLOCK):
    """Compatibility wrapper retaining the original static-grid computation."""
    return gptq_quantize(weights, scales, biases, upper, block)[0]


def write_projection(stream, codes, scales, biases):
    n, k = codes.shape
    if (k % GROUP or scales.shape != (n, k // GROUP) or biases.shape != scales.shape
            or not np.isfinite(scales).all() or not np.isfinite(biases).all()
            or np.any(scales == 0) or not np.array_equal(scales, bf16(scales))
            or not np.array_equal(biases, bf16(biases))):
        raise ValueError('exact finite stored BF16 metadata required')
    stream.write(struct.pack('<II', n, k))
    stream.write(pack_codes(codes).tobytes())
    for value in (scales, biases):
        stream.write((value.astype(np.float32).view(np.uint32) >> 16).astype('<u2').tobytes())


def decode(codes, scales, biases):
    return (codes.reshape(*scales.shape, GROUP).astype(np.float32) * scales[..., None]
            + biases[..., None]).reshape(codes.shape)


def projection_error(weights, codes, scales, biases, inputs):
    rows = np.linspace(0, len(weights) - 1, min(64, len(weights)), dtype=int)
    reference = inputs @ weights[rows].T
    actual = inputs @ decode(codes[rows], scales[rows], biases[rows]).T
    delta = actual.astype(np.float64) - reference
    energy = float(np.square(reference.astype(np.float64)).sum())
    return {'sampled_output_rows': rows.tolist(), 'input_rows': len(inputs),
            'relative_mse': float(np.square(delta).sum()) / energy if energy else None,
            'mse': float(np.square(delta).mean()), 'max_abs': float(np.abs(delta).max())}


def capture(output):
    rows = materials()
    protocol = {'schema': 'q4-packed-calibration-v1', 'model': MODEL, 'manifest': MANIFEST,
                'identity': IDENTITY, 'core': None, 'core_digest': '0' * 64, 'seed': 42,
                'materials': rows, 'training': False, 'teacher_calls': 0,
                'positions': 'up to128 evenly spaced plus final16 per prompt; same rows for all projections',
                'selection': 'calibration only; observation never selects grids, damping or codes',
                'methods': list(METHODS), 'group_size': GROUP, 'block_size': BLOCK, 'damping_fraction': DAMPING,
                'grid': 'static original affine signed-endpoint scale/bias rounded to actual BF16; no actorder',
                'feedback': 'full X.T X / N; FP64 inverse Cholesky then FP32; dead columns remain stored-grid RTN',
                'scope': 'all71 MLP projections; attention/down23 BF16; original core, no B45 or experts',
                'hardware': platform.platform(), 'torch': torch.__version__, 'numpy': np.__version__}
    save(output / 'protocol.json', protocol)
    names = sorted({input_name(layer, kind) for layer, kind, _, _ in PROJECTIONS})
    for name in names:
        for split in ('calibration', 'observation'):
            (output / f'{name}-{split}.bf16').touch(exist_ok=False)
    started = time.perf_counter()
    lock = ModelProcessLock(IDENTITY, 'mps')
    model = None
    handles = []
    counts = {name: {'calibration': 0, 'observation': 0} for name in names}
    current = None
    try:
        model = load_backbone(MODEL, torch.bfloat16, 'mps')
        decoder, _ = text_backbone(model)
        tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)

        def make_hook(name):
            def hook(_module, arguments):
                x = arguments[0].detach().reshape(-1, arguments[0].shape[-1])
                positions = sorted(set(np.linspace(0, len(x) - 1, min(128, len(x)), dtype=int).tolist()
                                       + list(range(max(0, len(x) - 16), len(x)))))
                value = x[positions].to(device='cpu', dtype=torch.bfloat16)
                if not torch.isfinite(value).all():
                    raise ValueError('nonfinite real calibration activation')
                with (output / f'{name}-{current["split"]}.bf16').open('ab') as stream:
                    value.view(torch.uint16).numpy().astype('<u2').tofile(stream)
                counts[name][current['split']] += len(value)
            return hook

        for layer in range(24):
            mlp = decoder.layers[layer].mlp
            handles.append(mlp.gate_proj.register_forward_pre_hook(make_hook(input_name(layer, 'gate'))))
            if layer != 23:
                handles.append(mlp.down_proj.register_forward_pre_hook(make_hook(input_name(layer, 'down'))))
        with torch.inference_mode():
            for row in rows:
                current = row
                ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['text']}],
                    add_generation_prompt=True, tokenize=True, return_tensors='pt', enable_thinking=False)
                if not isinstance(ids, torch.Tensor):
                    ids = ids['input_ids']
                if ids.shape[1] > 2048:
                    raise ValueError('no implicit calibration truncation')
                row['tokens'] = int(ids.shape[1])
                decoder(ids.to('mps'), use_cache=False)
                print(json.dumps({'captured': row['id'], 'tokens': row['tokens']}), flush=True)
        protocol.update(completed=True, counts=counts, elapsed_seconds=time.perf_counter()-started,
                        memory={'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                'method': 'Darwin peak process RSS; not combined Metal physical footprint'})
        save(output / 'result.json', protocol)
    finally:
        for handle in handles:
            handle.remove()
        del model
        gc.collect()
        torch.mps.empty_cache()
        lock.close()


def read_inputs(root, name, split, width, count):
    raw = np.fromfile(root / f'{name}-{split}.bf16', dtype='<u2')
    if raw.size != count * width:
        raise ValueError('capture dimensions mismatch')
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(count, width)


def prepare(captures, output, *, gptq_only=False):
    protocol = json.loads((captures / 'result.json').read_text())
    if (protocol.get('completed') is not True or protocol['manifest'] != MANIFEST
            or protocol['identity'] != IDENTITY or protocol['core'] is not None
            or protocol['grid'] != 'static original affine signed-endpoint scale/bias rounded to actual BF16; no actorder'):
        raise ValueError('unexpected frozen original-core calibration binding')
    save(output / 'protocol.json', {**protocol, 'captures': str(captures), 'completed': False})
    started = time.perf_counter()
    records = []
    model_root = Path(MODEL)
    with ExitStack() as stack:
        readers = {p.name: stack.enter_context(safe_open(p, framework='pt', device='cpu'))
                   for p in sorted(model_root.glob('*.safetensors')) if not p.name.startswith('.')}
        mapping = {key: reader for reader in readers.values() for key in reader.keys()}
        methods = ('gptq',) if gptq_only else METHODS
        streams = {method: stack.enter_context((output / (method + '.bin')).open('xb')) for method in methods}
        for stream in streams.values():
            stream.write(b'LQ4PK01\n' + struct.pack('<II', GROUP, len(PROJECTIONS)) + MANIFEST.encode() + b'0' * 64)
        last_name = None
        for layer, kind, n, k in PROJECTIONS:
            name = input_name(layer, kind)
            if name != last_name:
                counts = protocol['counts'][name]
                inputs = read_inputs(captures, name, 'calibration', k, counts['calibration'])
                observed = read_inputs(captures, name, 'observation', k, counts['observation'])
                upper, hessian = inverse_factor(inputs)
                last_name = name
            key = f'model.language_model.layers.{layer}.mlp.{kind}_proj.weight'
            tensor = mapping[key].get_tensor(key)
            if tensor.shape != (n, k) or tensor.dtype != torch.bfloat16:
                raise ValueError('unexpected checkpoint projection')
            weights = tensor.float().numpy()
            rtn, nearest, scales, biases = metadata(weights)
            gptq = gptq_codes(weights, scales, biases, upper)
            dead = hessian['dead_columns']
            if dead and not np.array_equal(gptq[:, dead], nearest[:, dead]):
                raise ValueError('feedback changed an unobserved dead column')
            record = {'layer': layer, 'projection': kind, 'shape': [n, k], 'hessian': hessian,
                      'changed_vs_rtn': int(np.count_nonzero(gptq != rtn)), 'errors': {}}
            for method, codes in zip(METHODS, (rtn, nearest, gptq)):
                if method in streams:
                    write_projection(streams[method], codes, scales, biases)
                record['errors'][method] = {split: projection_error(weights, codes, scales, biases, values)
                    for split, values in (('calibration', inputs), ('observation', observed))}
            records.append(record)
            save(output / 'progress.json', {'projections': len(records), 'elapsed_seconds': time.perf_counter()-started})
            save(output / 'projection-results.json', records)
            print(json.dumps({'prepared': key, 'projections': len(records), 'seconds': time.perf_counter()-started}), flush=True)
    result = {'completed': True, 'protocol': 'protocol.json', 'projections': records,
              'elapsed_seconds': time.perf_counter()-started,
              'files': {method: {'path': str(output / (method+'.bin')), 'bytes': (output / (method+'.bin')).stat().st_size} for method in methods},
              'memory': {'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'method': 'Darwin process peak RSS; CPU preparation only'},
              'product_adopted': False, 'quality_claim': False, 'teacher_calls': 0}
    save(output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture', 'prepare'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--captures', type=Path)
    parser.add_argument('--gptq-only', action='store_true', help='Reuse existing RTN controls; write only the new GPTQ carrier')
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.absolute().is_relative_to('/Volumes/移动数据库/lubanllm/experiments'):
        raise ValueError('mounted experiment volume required')
    if args.stage == 'prepare' and args.captures is None:
        parser.error('prepare requires --captures')
    if args.stage == 'capture' and args.gptq_only:
        parser.error('--gptq-only requires prepare')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.stage == 'capture':
        capture(args.output)
    else:
        prepare(args.captures, args.output, gptq_only=args.gptq_only)


if __name__ == '__main__':
    main()
