"""Ordinary native four-case and original union screen for the new source conversion."""
import os
from pathlib import Path
import time

from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MODEL, save
from benchmark_native_mixed import MixedWorker
from diagnose_q4_task_candidate import diagnostic_result, ordinary_loaded, read
from luban.process_lock import ModelProcessLock
from prepare_q4_sensitive_sequential import OUTPUT as CONVERSION, REGISTRATION, SCHEMA
from prepare_q4_sequential import protection_generations
from q4_sequential_protection import check_proposal, protection_cases
from study_q4_column_protection import LAYERS, MASK, ROOT, SOURCE

BINARY = ROOT/'column-protection-build/zig-out/bin/luban-worker'
CONTROLS = ROOT/'column-protection-native-v1/screen'
OUTPUT = CONVERSION/'screen'


def preflight():
    candidate = read(CONVERSION/'result.json')
    audit = read(CONVERSION/'independent-readback.json')
    source = read(SOURCE/'result.json')
    controls = read(CONTROLS/'result.json')
    control_audit = read(CONTROLS/'independent-readback.json')
    receipt = read(Path(str(BINARY)+'.json'))
    registration = read(REGISTRATION)
    if (candidate.get('schema') != SCHEMA or candidate.get('completed') is not True
            or candidate.get('source') != str(SOURCE) or candidate.get('receipt') != source['receipt']
            or candidate.get('plan') != source['plan'] or candidate.get('accepted_layers') != LAYERS
            or candidate.get('layer_mask') != MASK or candidate.get('groups_per_projection') != 1
            or candidate.get('worker_exit_code') != 0 or len(candidate.get('projections', [])) != 32
            or audit.get('audit_completed') is not True or audit.get('audit_passed') is not True
            or audit.get('directory') != str(CONVERSION)
            or registration.get('native_followup', {}).get('registered_before_run') is not True):
        raise ValueError('completed independently audited fixed source conversion required')
    path = CONVERSION/'candidate.bin'
    if (candidate['artifact'].get('quantized_projections') != 32 or path.is_symlink()
            or not path.is_file() or path.stat().st_size != candidate['artifact']['bytes']):
        raise ValueError('complete ordinary packed-v2 carrier required')
    if (controls.get('completed') is not True or controls.get('controls_passed') is not True
            or controls.get('binary') != str(BINARY) or controls.get('receipt') != receipt
            or controls.get('source_receipt') != source['receipt']
            or control_audit.get('audit_completed') is not True or control_audit.get('audit_passed') is not True
            or control_audit.get('old_bf16_four_exact') is not True
            or control_audit.get('new_v2_old20_exact') is not True):
        raise ValueError('same ordinary worker BF16/v2 source-bridge controls required')
    for name, precision in (('bf16', 'bf16'), ('v2', 'mixed-q4-g16-packed')):
        ordinary_loaded(controls['reports'][name]['loaded'], precision)
    if tuple(controls['reports']['v2']['loaded'].get(k) for k in
             ('packed_version', 'residual_rank', 'active_residual_projections')) != (2, 0, 0):
        raise ValueError('original bridge must use packed-v2 without corrections')
    specs = controls['cases']
    if len(specs) != 4 or {s['id'] for s in specs} != {
            'mmlu-2804', 'preservation-replay-train-olive-path', 'decode-holdout-unit-volume', 'seq-final-young-dog'}:
        raise ValueError('only original four exposed cases allowed')
    baseline, previous = controls['baseline'], controls['previous']
    union = check_proposal(protection_cases(), baseline, previous, previous)
    if (len(union['original_correct_ids']) != 13 or len(union['required_correct_ids']) != 14
            or 'seq-train-copy-en' not in union['required_correct_ids']):
        raise ValueError('original13 and prior14 correctness union changed')
    return candidate, source, controls, receipt, path, specs, baseline, previous


def main():
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('experiment volume must be mounted')
    candidate, source, controls, receipt, path, specs, baseline, previous = preflight()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    vocab = read(Path(source['source'])/'protocol.json')['vocab_size']
    for spec, recorded in zip(specs, controls['reports']['bf16']['four'], strict=True):
        checked = diagnostic_result(spec, recorded['native_result'], tokenizer, vocab)
        if any(checked[key] != recorded[key] for key in checked) or checked['score']['passed'] is not True:
            raise ValueError('original BF16 correctness evidence changed')
    OUTPUT.mkdir(exist_ok=False)
    protocol = {'schema': 'q4-sensitive-source-native-screen-v1', 'conversion': str(CONVERSION),
        'source': str(SOURCE), 'source_receipt': source['receipt'], 'binary': str(BINARY), 'receipt': receipt,
        'candidate': str(path), 'artifact': candidate['artifact'], 'conversion_audit': str(CONVERSION/'independent-readback.json'),
        'controls_source': str(CONTROLS), 'cases': specs, 'baseline': baseline, 'previous': previous,
        'seed': 42, 'temperature': 0, 'cache_reuse': False, 'core': None,
        'qualified': False, 'product_adopted': False, 'training': False, 'teacher_calls': 0,
        'new_holdout_read': False, 'full_regression_run': False, 'final_holdout_run': False}
    save(OUTPUT/'protocol.json', protocol)
    started, worker, lock = time.perf_counter(), None, None
    four, train, report = [], None, None
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        worker = MixedWorker(BINARY, 'candidate_mixed', OUTPUT, lock,
            precision='mixed-q4-g16-packed', calibration_profile=path, core=None)
        loaded = worker.loaded
        ordinary_loaded(loaded, 'mixed-q4-g16-packed')
        if (loaded.get('core_file_sha256') is not None or tuple(loaded.get(k) for k in
                ('packed_version', 'residual_rank', 'active_residual_projections')) != (2, 0, 0)):
            raise ValueError('ordinary source-only v2 candidate required')
        save(OUTPUT/'loaded.json', loaded)
        for spec in specs:
            case, request_id = spec['case'], 'four-'+spec['id']
            raw = worker.request(case['tokens'], request_id, max_tokens=case['max_tokens'],
                temperature=0, seed=42, reuse=False, eos_ids=case['eos_ids'])
            if raw.get('event') != 'result' or raw.get('id') != request_id:
                raise ValueError('request/result identity changed')
            four.append(diagnostic_result(spec, raw, tokenizer, vocab))
            save(OUTPUT/'four.json', four)
        passed = all(row['score']['passed'] for row in four)
        if passed:
            train = protection_generations(worker, protection_cases(), tokenizer, source['plan']['eos_ids'], 'train', vocab)
            report = check_proposal(protection_cases(), baseline, previous, train)
        worker.close(); code = worker.process.returncode; worker = None
        if code:
            raise RuntimeError('ordinary worker failed on close')
        accepted = passed and report is not None and report['accepted'] is True
        save(OUTPUT/'result.json', {**protocol, 'completed': True, 'loaded': loaded, 'four': four,
            'four_passed': passed, 'train_skipped': not passed, 'train': train, 'protection': report,
            'screen_passed': accepted, 'full_regression_permitted': accepted,
            'full_regression_skipped': not accepted, 'final_holdout_skipped': not accepted,
            'selected': str(path) if accepted else None, 'worker_exit_code': code,
            'elapsed_seconds': time.perf_counter()-started})
    except BaseException as error:
        save(OUTPUT/'failure.json', {'completed': False, 'error': repr(error), 'four': four,
            'train': train, 'protection': report, 'elapsed_seconds': time.perf_counter()-started})
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
