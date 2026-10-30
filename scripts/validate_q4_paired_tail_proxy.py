"""Measure every frozen paired tail against native arithmetic, without updates."""
import contextlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
from safetensors import safe_open
import torch

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from capture_q4_paired_tails import OUTPUT as CAPTURE, BINARY, COUNTS, TAIL_WIDTHS, check_frame
from diagnose_q4_task_candidate import read
from luban.process_lock import ModelProcessLock
from probe_q4_paired_tail import tail_forward, tail_head, tensor_array, write_bf16
from study_q4_column_protection import ROOT, SOURCE, frozen_scope
from trace_q4_divergence import VOCAB

OUTPUT = ROOT/'paired-q4-tail-proxy-validation-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_paired_tail_proxy_validation_20261017.json'
MODES = ('bf16', 'q4')
ERROR_COLUMNS = ('rmse', 'max_abs', 'different')
LOGIT_COLUMNS = ('forward_kl', *ERROR_COLUMNS, 'reference_argmax', 'actual_argmax',
                 'reference_margin', 'actual_margin', 'reference_top_ties', 'actual_top_ties')


def finite_pair(reference, actual):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if a.ndim != 1 or a.size < 2 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('same finite one-dimensional vectors required')
    return a, b


def errors(reference, actual):
    a, b = finite_pair(reference, actual)
    delta = b-a
    return np.array([np.sqrt(np.mean(delta*delta)), np.max(np.abs(delta)), np.count_nonzero(delta)], np.float64)


def logit_summary(reference, actual):
    """Full-vocabulary FP64 forward KL; linear selection rather than full sorting."""
    a, b = finite_pair(reference, actual)
    top_a, top_b = int(np.argmax(a)), int(np.argmax(b))
    la, lb = a-a[top_a], b-b[top_b]
    la -= np.log(np.exp(la).sum()); lb -= np.log(np.exp(lb).sum())
    kl = np.sum(np.exp(la)*(la-lb))
    return np.array([kl, *errors(a, b), top_a, top_b,
        a[top_a]-np.partition(a, -2)[-2], b[top_b]-np.partition(b, -2)[-2],
        np.count_nonzero(a == a[top_a]), np.count_nonzero(b == b[top_b])], np.float64)


def validate_mapping(mapping, plan, counts):
    seen, offsets, rows = set(), dict.fromkeys(counts, 0), []
    for case, planned in zip(mapping['cases'], plan['cases'], strict=True):
        identifier, split, targets = case['id'], case['split'], case['targets']
        prompt, forced = case['tokens'], case['forced_tokens']
        if (identifier in seen or split not in offsets or not prompt or not 1 <= len(targets) <= 256
                or any(type(t) is not int or not 0 <= t < VOCAB for t in prompt+targets)
                or any(not (c.isascii() and (c.isalnum() or c in '-_')) for c in identifier)
                or not identifier or forced != targets[:-1]
                or any(t in plan['eos_ids'] for t in forced)
                or any(case[k] != planned[p] for k, p in
                       [('id','id'), ('split','split'), ('tokens','tokens'), ('forced_tokens','forced_tokens')])
                or case['row_start'] != offsets[split] or case['row_end'] != offsets[split]+len(targets)
                or case['prediction_positions'] != list(range(len(prompt)-1, len(prompt)+len(targets)-1))
                or case['source_complete'] is not (targets[-1] in plan['eos_ids'])
                or case['source_finish_reason'] != ('stop' if case['source_complete'] else 'length')):
            raise ValueError('original complete trajectory mapping changed')
        rows.append({**case, 'file_row_start': sum(offsets.values()),
                     'file_row_end': sum(offsets.values())+len(targets)})
        offsets[split] += len(targets); seen.add(identifier)
    if offsets != counts or mapping['rows_per_split'] != counts:
        raise ValueError('original prediction counts changed')
    return rows


def preflight():
    registration = read(REGISTRATION)
    capture, audit = read(CAPTURE/'result.json'), read(CAPTURE/'independent-readback.json')
    source = read(SOURCE/'result.json'); frozen_scope(source)
    receipt = read(Path(str(BINARY)+'.json'))
    if (registration.get('registered_before_run') is not True
            or registration.get('schema') != 'q4-paired-tail-proxy-validation-v1'
            or registration.get('capture_directory') != str(CAPTURE)
            or registration.get('prediction_positions_per_model') != COUNTS
            or capture.get('completed') is not True or capture.get('all_teacher_targets_exact') is not True
            or capture.get('schema') != 'q4-paired-tail-capture-v1' or capture.get('rows_per_model') != COUNTS
            or capture.get('source_receipt') != source['receipt'] or capture.get('source_plan') != source['plan']
            or capture.get('receipt') != receipt or capture.get('binary') != str(BINARY)
            or capture.get('manifest') != MANIFEST or capture.get('core') is not None
            or capture.get('tail_widths') != TAIL_WIDTHS or capture.get('vocab') != VOCAB
            or audit.get('audit_completed') is not True or audit.get('audit_passed') is not True
            or audit.get('directory') != str(CAPTURE) or audit.get('rows_per_model') != COUNTS
            or audit.get('complete_teacher_targets_exact') is not True):
        raise ValueError('completed independently audited fixed-source paired capture required')
    mapping = read(CAPTURE/'position-mapping.json')
    cases = validate_mapping(mapping, source['plan'], COUNTS)
    if len(cases) != 36 or sum(c['split']=='calibration' for c in cases) != 32:
        raise ValueError('fixed 32 calibration and 4 observation trajectories required')
    for mode in MODES:
        reports = capture['reports'][mode]['cases']
        for case, report in zip(cases, reports, strict=True):
            expected = dict(id=case['id'], tokens=case['tokens'], forced_tokens=case['forced_tokens'],
                max_tokens=len(case['targets']), eos_ids=source['plan']['eos_ids'], temperature=0,
                seed=42, cache_reuse=False, tail_only=True)
            if (report['request'] != expected or report['id'] != case['id'] or report['split'] != case['split']
                    or len(report['events']) != len(case['targets'])
                    or (mode == 'bf16' and report['result']['tokens'] != case['targets'])):
                raise ValueError('audited frame requests differ from full original trajectory')
    if read(Path(MODEL)/'config.json').get('tie_word_embeddings') is not True:
        raise ValueError('original tied BF16 head required')
    return registration, capture, audit, cases


def summarize(matrix, columns):
    if not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError('complete finite metric rows required')
    result = {name: {'mean': float(matrix[:,i].mean()), 'min': float(matrix[:,i].min()),
                     'max': float(matrix[:,i].max())}
              for i, name in enumerate(columns) if 'argmax' not in name}
    if 'reference_argmax' in columns:
        result['argmax_mismatches'] = int(np.count_nonzero(matrix[:,4] != matrix[:,5]))
        result['exact_rows'] = int(np.count_nonzero(matrix[:,3] == 0))
    else:
        result['exact_rows'] = int(np.count_nonzero(matrix[:,2] == 0))
    result['different_values_total'] = int(matrix[:,columns.index('different')].sum())
    return result


def main():
    if not os.path.ismount('/Volumes/移动数据库') or not torch.backends.mps.is_available():
        raise ValueError('mounted experiment storage and MPS required')
    registration, capture, audit, cases = preflight()
    OUTPUT.mkdir(exist_ok=False)
    metadata = {**registration, 'capture_directory': str(CAPTURE), 'capture_receipt': capture['receipt'],
        'source_receipt': capture['source_receipt'], 'capture_audit': audit,
        'manifest': MANIFEST, 'core': None, 'torch': torch.__version__, 'hardware': platform.platform(),
        'case_mapping': cases, 'modes': MODES, 'error_columns': ERROR_COLUMNS, 'logit_columns': LOGIT_COLUMNS,
        'output_layout': 'per-mode files in case_mapping order; BF16 row-major logits[248320], down[2048]',
        'isolated_arrays_saved': False, 'isolated_audit_scope': 'statistics only; no independent array readback',
        'timing_scope': 'after metadata preflight and output preparation; lock, tensor load, all forwards/statistics/writes',
        'updates': 0, 'optimizer': False, 'autograd': False, 'batch_rows': 1,
        'quality_adoption': False, 'numerical_tolerance_gate': None}
    save(OUTPUT/'protocol.json', metadata)
    total = sum(COUNTS.values())
    down_metrics = np.full((2,total,3), np.nan)
    full_metrics = np.full((2,total,len(LOGIT_COLUMNS)), np.nan)
    isolated_metrics = np.full_like(full_metrics, np.nan)
    signal = np.full((total,len(LOGIT_COLUMNS)), np.nan)
    completed = 0; started = time.perf_counter(); lock = None
    summaries = []
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        torch.set_num_threads(4); torch.manual_seed(42)
        index = read(Path(MODEL)/'model.safetensors.index.json')['weight_map']
        keys = {'weight':'model.language_model.layers.23.mlp.down_proj.weight',
                'norm':'model.language_model.norm.weight', 'head':'model.language_model.embed_tokens.weight'}
        tensors = {}
        for name, key in keys.items():
            with safe_open(Path(MODEL)/index[key], framework='pt', device='cpu') as stream:
                value = stream.get_tensor(key)
                if value.dtype != torch.bfloat16 or not all(bool(torch.isfinite(part).all()) for part in value.split(4096)):
                    raise ValueError('finite original BF16 tail weights required')
                tensors[name] = value.to('mps').requires_grad_(False)
        del value
        if [tuple(tensors[n].shape) for n in keys] != [(2048,6144),(2048,),(VOCAB,2048)]:
            raise ValueError('original full tail shape changed')
        versions = {name: tensor._version for name,tensor in tensors.items()}
        with contextlib.ExitStack() as stack, torch.no_grad():
            outputs = {mode: {kind: stack.enter_context((OUTPUT/f'{mode}-{kind}.bf16').open('wb', buffering=0))
                      for kind in ('logits','down')} for mode in MODES}
            for ci, case in enumerate(cases):
                for step in range(len(case['targets'])):
                    row = case['file_row_start']+step
                    frames = {}
                    for mode in MODES:
                        report = capture['reports'][mode]['cases'][ci]
                        frames[mode] = check_frame(report['events'][step], CAPTURE/mode/'traces', report['request'], step)
                    signal[row] = logit_summary(frames['bf16']['logits'], frames['q4']['logits'])
                    for mi,mode in enumerate(MODES):
                        frame = frames[mode]; tail = frame['tail']
                        h, x, native_down = [torch.tensor(v, device='mps', dtype=torch.bfloat16).reshape(1,-1)
                            for v in (tail[:2048],tail[2048:8192],tail[8192:])]
                        down, logits = tail_forward(x, h, **tensors)
                        isolated = tail_head(native_down, h, tensors['norm'], tensors['head'])
                        down_array, logits_array, isolated_array = map(tensor_array, (down,logits,isolated))
                        down_metrics[mi,row] = errors(tail[8192:],down_array)
                        full_metrics[mi,row] = logit_summary(frame['logits'],logits_array)
                        isolated_metrics[mi,row] = logit_summary(frame['logits'],isolated_array)
                        write_bf16(outputs[mode]['down'],down); write_bf16(outputs[mode]['logits'],logits)
                    completed = row+1
                sl = slice(case['file_row_start'],case['file_row_end'])
                report = {'id':case['id'],'split':case['split'],'rows':len(case['targets']),
                    'native_teacher_student':summarize(signal[sl],LOGIT_COLUMNS),
                    'modes':{mode:{'down':summarize(down_metrics[mi,sl],ERROR_COLUMNS),
                        'full':summarize(full_metrics[mi,sl],LOGIT_COLUMNS),
                        'isolated':summarize(isolated_metrics[mi,sl],LOGIT_COLUMNS)} for mi,mode in enumerate(MODES)}}
                summaries.append(report)
                np.savez(OUTPUT/'metrics.npz', down=down_metrics[:,:completed], full=full_metrics[:,:completed],
                         isolated=isolated_metrics[:,:completed], native_teacher_student=signal[:completed])
                save(OUTPUT/'progress.json', {'completed':False,'rows_per_model':completed,'cases':summaries,
                                             'elapsed_seconds':time.perf_counter()-started})
                print(json.dumps({'case':case['id'],'rows_per_model':completed,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        if completed != total or any(t.requires_grad or t.grad is not None or t._version != versions[n] for n,t in tensors.items()):
            raise ValueError('incomplete stream or frozen parameter mutation')
        for mode in MODES:
            for kind,width in [('logits',VOCAB),('down',2048)]:
                if (OUTPUT/f'{mode}-{kind}.bf16').stat().st_size != total*width*2:
                    raise ValueError('proxy stream length mismatch')
        split_summaries = {}
        for split in COUNTS:
            rows = np.concatenate([np.arange(c['file_row_start'],c['file_row_end']) for c in cases if c['split']==split])
            split_summaries[split] = {'rows':len(rows),'native_teacher_student':summarize(signal[rows],LOGIT_COLUMNS),'modes':{}}
            for mi,mode in enumerate(MODES):
                positive = signal[rows,0] > 0
                ratios = full_metrics[mi,rows,0][positive]/signal[rows,0][positive]
                split_summaries[split]['modes'][mode] = {
                    'down':summarize(down_metrics[mi,rows],ERROR_COLUMNS),'full':summarize(full_metrics[mi,rows],LOGIT_COLUMNS),
                    'isolated':summarize(isolated_metrics[mi,rows],LOGIT_COLUMNS),
                    'proxy_to_native_signal_kl_ratio':{'positive_signal_rows':int(positive.sum()),
                        'mean':float(ratios.mean()) if len(ratios) else None,'max':float(ratios.max()) if len(ratios) else None}}
        torch.mps.synchronize()
        save(OUTPUT/'result.json', {**metadata,'completed':True,'cases':summaries,'splits':split_summaries,
            'native_tail_bitexact':bool(np.all(down_metrics[:,:,2]==0) and np.all(full_metrics[:,:,3]==0)),
            'original_weight_unchanged':True,'elapsed_seconds':time.perf_counter()-started,
            'memory':{'python_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'sampled_mps_allocated_bytes':torch.mps.current_allocated_memory(),
                'sampled_mps_driver_bytes':torch.mps.driver_allocated_memory(),
                'scope':'Python high-water RSS and final MPS samples; not GPU peak'}})
    except BaseException as error:
        save(OUTPUT/'failure.json', {'completed':False,'rows_per_model':completed,'error':repr(error),
                                  'elapsed_seconds':time.perf_counter()-started})
        raise
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
