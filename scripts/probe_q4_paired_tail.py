"""Paired native BF16/Q4 tails and differentiable-tail feasibility, without updates."""
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from diagnose_q4_task_candidate import ordinary_loaded, read
from diagnose_q4_sensitive_graph import logit_metrics
from luban.process_lock import ModelProcessLock
from study_q4_column_protection import ROOT, SOURCE, MASK
from trace_q4_divergence import WIDTHS, VOCAB, error_stats, read_bf16, trace_request
from verify_q4_sensitive_source import BINARY, CONTROLS

OUTPUT = ROOT/'paired-q4-tail-probe-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_paired_tail_probe_20261016.json'


def tail_parts(values):
    if values.shape != (24*sum(WIDTHS.values()),) or not np.isfinite(values).all():
        raise ValueError('complete finite24-layer trace required')
    last = values.reshape(24, -1)[23]
    parts, at = {}, 0
    for name, width in WIDTHS.items():
        if name in ('h', 'act', 'down'):
            parts[name] = last[at:at+width].copy()
        at += width
    return parts


def tail_head(down, residual, norm, head):
    z = down+residual
    f = z.float()
    normalized = (f*torch.rsqrt(f.square().mean(-1, keepdim=True)+1e-6)*(1+norm.float())).to(z.dtype)
    return F.linear(normalized, head)


def tail_forward(x, residual, weight, norm, head):
    down = F.linear(x, weight)
    return down, tail_head(down, residual, norm, head)


def tensor_array(value):
    return value.detach().float().cpu().numpy().reshape(-1)


def write_bf16(path, value):
    value.detach().to('cpu').contiguous().view(torch.uint16).numpy().tofile(path)


def main():
    if not os.path.ismount('/Volumes/移动数据库') or not torch.backends.mps.is_available():
        raise ValueError('mounted external storage and MPS required')
    registration = read(REGISTRATION)
    source, controls = read(SOURCE/'result.json'), read(CONTROLS/'result.json')
    receipt = read(Path(str(BINARY)+'.json'))
    prior_audit = read(CONTROLS/'independent-readback.json')
    if (registration.get('registered_before_run') is not True
            or source.get('completed') is not True or source.get('layer_mask') != MASK
            or source['artifact'].get('quantized_projections') != 32
            or controls.get('receipt') != receipt or controls.get('controls_passed') is not True
            or controls.get('source_receipt') != source['receipt'] or prior_audit.get('audit_passed') is not True):
        raise ValueError('original32 source and same frozen worker audited controls required')
    case = source['plan']['cases'][0]
    if case['id'] != 'cal-00' or case['split'] != 'calibration' or len(case['tokens']) != 26:
        raise ValueError('fixed first calibration trajectory required')
    forced = case['forced_tokens'][:1]
    ordinary_case = {'tokens': case['tokens'], 'max_tokens': 2, 'eos_ids': source['plan']['eos_ids']}
    carrier = SOURCE/'gptq-sequential.bin'
    if not carrier.is_file() or carrier.is_symlink() or carrier.stat().st_size != source['artifact']['bytes']:
        raise ValueError('original frozen Q4 carrier required')
    config = read(Path(MODEL)/'config.json')
    if config.get('tie_word_embeddings') is not True:
        raise ValueError('original tied head required')
    OUTPUT.mkdir(exist_ok=False)
    protocol = {**registration, 'binary': str(BINARY), 'receipt': receipt, 'source_receipt': source['receipt'],
        'carrier': str(carrier), 'manifest': MANIFEST, 'case': ordinary_case, 'forced_tokens': forced,
        'hardware': platform.platform(), 'torch': torch.__version__, 'core': None,
        'timing_scope': 'two native loads/paired trace and controls, then one differentiable tail load/forward/backward; no optimization'}
    save(OUTPUT/'protocol.json', protocol)
    started, lock, worker = time.perf_counter(), None, None
    reports, frames = {}, {}
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        for mode in ('bf16', 'q4'):
            directory = OUTPUT/mode; (directory/'traces').mkdir(parents=True)
            name, precision = ('bf16', 'bf16') if mode == 'bf16' else ('candidate_mixed', 'mixed-q4-g16-packed')
            worker = MixedWorker(BINARY, name, directory, lock, precision=precision,
                calibration_profile=None if mode == 'bf16' else carrier,
                trace_directory=directory/'traces', core=None)
            ordinary_loaded(worker.loaded, precision)
            if mode == 'q4' and tuple(worker.loaded.get(k) for k in
                    ('packed_version', 'residual_rank', 'active_residual_projections')) != (2, 0, 0):
                raise ValueError('fixed source-only Q4 graph required')
            before = worker.request(case['tokens'], 'before', max_tokens=2, eos_ids=ordinary_case['eos_ids'],
                temperature=0, seed=42, reuse=False)
            events = trace_request(worker, ordinary_case, forced, 'paired-cal00')
            after = worker.request(case['tokens'], 'after', max_tokens=2, eos_ids=ordinary_case['eos_ids'],
                temperature=0, seed=42, reuse=False)
            if (any(before[k] != after[k] for k in ('tokens', 'finish_reason', 'complete'))
                    or any(r.get('cached_prompt_tokens') != 0 or r.get('prompt_tokens') != len(case['tokens'])
                           or r.get('generated_tokens') != 2 for r in (before, after))):
                raise ValueError('ordinary two-token controls changed')
            if mode == 'bf16' and before['tokens'] != case['forced_tokens'][:2]:
                raise ValueError('original BF16 calibration prefix changed')
            frames[mode] = []
            for step, event in enumerate(events):
                if (event.get('id') != 'paired-cal00' or event.get('step') != step or event.get('vocab') != VOCAB
                        or event.get('layers') != 24 or event.get('layer_values') != sum(WIDTHS.values())
                        or event.get('committed_tokens') != len(case['tokens'])+step):
                    raise ValueError('paired native prefix/shape changed')
                paths = {kind: directory/'traces'/f'paired-cal00-step{step:04d}.{kind}.bf16' for kind in ('layers', 'logits')}
                if any(event.get(kind+'_path') != str(path) for kind, path in paths.items()):
                    raise ValueError('paired frame path changed')
                logits = read_bf16(paths['logits'], VOCAB)
                if int(np.argmax(logits)) != event['top1']:
                    raise ValueError('native greedy differs from full vocabulary')
                if before['tokens'][:step] == forced[:step] and event['top1'] != before['tokens'][step]:
                    raise ValueError('trace differs on same natural prefix')
                parts = tail_parts(read_bf16(paths['layers'], 24*sum(WIDTHS.values())))
                frames[mode].append({**parts, 'logits': logits})
            reports[mode] = {'loaded': worker.loaded, 'before': before, 'events': events, 'after': after}
            worker.close(); code = worker.process.returncode; worker = None
            if code:
                raise RuntimeError('native worker failed on close')
            raw = [json.loads(line) for line in (directory/(name+'-events.jsonl')).read_text().splitlines()]
            results = [e for e in raw if e.get('event') == 'result']
            if (len(results) != 3 or results[1].get('id') != 'paired-cal00'
                    or results[1].get('tokens') != [e['top1'] for e in events]
                    or results[1].get('forced_tokens') != forced or results[1].get('trace_only') is not True
                    or results[1].get('complete') is not True or results[1].get('cached_prompt_tokens') != 0):
                raise ValueError('forced paired trace did not complete as registered')
        save(OUTPUT/'native.json', {'completed': True, 'reports': reports})
        # Native workers are closed before allocating the differentiable tail.
        torch.set_num_threads(4); torch.manual_seed(42)
        index = read(Path(MODEL)/'model.safetensors.index.json')['weight_map']
        keys = {'weight': 'model.language_model.layers.23.mlp.down_proj.weight',
                'norm': 'model.language_model.norm.weight', 'head': 'model.language_model.embed_tokens.weight'}
        tensors = {}
        for name, key in keys.items():
            with safe_open(Path(MODEL)/index[key], framework='pt', device='cpu') as stream:
                value = stream.get_tensor(key)
                if value.dtype != torch.bfloat16:
                    raise ValueError('original BF16 tail weights required')
                tensors[name] = value.to('mps')
        if [tuple(tensors[n].shape) for n in ('weight', 'norm', 'head')] != [(2048,6144),(2048,),(VOCAB,2048)]:
            raise ValueError('original complete tail dimensions changed')
        metrics = {}
        for mode in frames:
            metrics[mode] = []
            for step, frame in enumerate(frames[mode]):
                x = torch.tensor(frame['act'], device='mps', dtype=torch.bfloat16).reshape(1,-1)
                h = torch.tensor(frame['h'], device='mps', dtype=torch.bfloat16).reshape(1,-1)
                with torch.no_grad():
                    down, logits = tail_forward(x, h, **tensors)
                    captured_down = torch.tensor(frame['down'], device='mps', dtype=torch.bfloat16).reshape(1,-1)
                    isolated = tail_head(captured_down, h, tensors['norm'], tensors['head'])
                write_bf16(OUTPUT/f'{mode}-{step}-down.bf16', down)
                write_bf16(OUTPUT/f'{mode}-{step}-logits.bf16', logits)
                write_bf16(OUTPUT/f'{mode}-{step}-captured-down-logits.bf16', isolated)
                row = {'step': step, 'down': error_stats(frame['down'], tensor_array(down)),
                    'logits': logit_metrics(frame['logits'], tensor_array(logits), [int(np.argmax(frame['logits']))]),
                    'captured_down_head_logits': logit_metrics(frame['logits'], tensor_array(isolated),
                                                              [int(np.argmax(frame['logits']))])}
                row['exact_native_down'] = np.array_equal(frame['down'], tensor_array(down))
                row['exact_native_logits'] = np.array_equal(frame['logits'], tensor_array(logits))
                metrics[mode].append(row)
        # Measure gradients without Adam, parameter updates, or new candidate artifacts.
        weight = tensors['weight'].detach().clone().requires_grad_(True)
        losses, forward_exact = [], []
        for step, frame in enumerate(frames['q4']):
            x = torch.tensor(frame['act'], device='mps', dtype=torch.bfloat16).reshape(1,-1)
            h = torch.tensor(frame['h'], device='mps', dtype=torch.bfloat16).reshape(1,-1)
            teacher = torch.tensor(frames['bf16'][step]['logits'], device='mps').reshape(1,-1).log_softmax(-1)
            _, logits = tail_forward(x, h, weight, tensors['norm'], tensors['head'])
            expected = read_bf16(OUTPUT/f'q4-{step}-logits.bf16', VOCAB)
            forward_exact.append(np.array_equal(tensor_array(logits), expected))
            if not forward_exact[-1]:
                raise ValueError('differentiable tail forward changed when gradient enabled')
            loss = F.kl_div(logits.float().log_softmax(-1), teacher, reduction='batchmean', log_target=True)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite paired KL')
            losses.append(float(loss.detach().cpu())); (loss/2).backward()
        gradient = weight.grad
        if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
            raise ValueError('finite nonzero full down gradient required')
        write_bf16(OUTPUT/'paired-kl-gradient.bf16', gradient)
        if not torch.equal(weight.detach(), tensors['weight']):
            raise ValueError('probe changed original weight')
        signal = [logit_metrics(frames['bf16'][i]['logits'], frames['q4'][i]['logits'],
                    [int(np.argmax(frames['bf16'][i]['logits']))]) for i in range(2)]
        torch.mps.synchronize()
        save(OUTPUT/'result.json', {**protocol, 'completed': True, 'native': reports,
            'tail_metrics': metrics, 'native_teacher_student': signal, 'paired_proxy_kl': losses,
            'no_grad_grad_forward_exact': forward_exact,
            'native_tail_bitexact': all(r['exact_native_down'] and r['exact_native_logits'] for rows in metrics.values() for r in rows),
            'gradient': {'dtype': str(gradient.dtype), 'shape': list(gradient.shape),
                'finite': True, 'nonzero': int(torch.count_nonzero(gradient).cpu()),
                'norm': float(gradient.float().norm().detach().cpu())},
            'original_weight_unchanged': True, 'elapsed_seconds': time.perf_counter()-started,
            'memory': {'python_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'sampled_mps_allocated_bytes': torch.mps.current_allocated_memory(),
                'sampled_mps_driver_bytes': torch.mps.driver_allocated_memory(),
                'scope': 'Python high-water RSS and final sampled MPS memory; not combined/native/GPU peak'}})
        print('Completed paired native-tail feasibility probe; no updates', flush=True)
    except BaseException as error:
        save(OUTPUT/'failure.json', {'completed': False, 'error': repr(error), 'native': reports})
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            if lock is not None:
                lock.close()


if __name__ == '__main__':
    main()
