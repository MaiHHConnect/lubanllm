"""Screen fixed Q8/BF16 allocations on exposed cases; never qualify or adopt."""
import os
from pathlib import Path
import platform
import statistics
import time

from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from check_q8_original_baseline import OUTPUT as REFERENCE, performance
from diagnose_q4_task_candidate import diagnostic_result, ordinary_loaded, read
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import protection_generations
from q4_sequential_protection import check_proposal, protection_cases
from study_q4_column_protection import ROOT
from verify_q4_paired_recovery import CONTROLS, four_generations

BINARY = ROOT/'higher-precision-combinations-build/zig-out/bin/luban-worker'
OUTPUT = ROOT/'higher-precision-combinations-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/higher_precision_combinations_20261017.json'
COUNTS = {'bf16': 0, 'mixed-q8': 150, 'mixed-q8-mlp': 71, 'mixed-q8-gateup': 48}
COMBINATIONS = ('mixed-q8-mlp', 'mixed-q8-gateup')


def check_loaded(loaded, mode, original):
    if mode == 'bf16':
        ordinary_loaded(loaded, mode)
    if (mode not in COUNTS or loaded.get('base_identity') != IDENTITY
            or loaded.get('manifest_sha256') != MANIFEST
            or loaded.get('inference_precision') != mode
            or loaded.get('quantized_projections') != COUNTS[mode]
            or loaded.get('core_loaded') is not False
            or loaded.get('core_file_sha256') is not None or loaded.get('core_tensor_sha256') is not None
            or loaded.get('retained_original_weight_bytes') != 0
            or any(loaded.get(k) != original.get(k) for k in ('checkpoint_sha256', 'original_down_sha256'))
            or any(loaded.get(k) is not False for k in
                   ('research_sequential', 'research_capture_inputs', 'research_intervention'))
            or any(loaded.get(k) != 0 for k in ('packed_version', 'residual_rank', 'active_residual_projections'))):
        raise ValueError('exact original coreNone ordinary allocation required')


def exact_rows(actual, previous, scored=False):
    keys = ['id', 'prompt_tokens', 'tokens', 'text', 'complete'] + (['score'] if scored else [])
    if len(actual) != len(previous) or any(any(row[k] != old[k] for k in keys)
            for row, old in zip(actual, previous, strict=True)):
        raise ValueError('new worker must exactly reproduce old original BF16/Q8 outputs')


def screen_passed(report):
    # The prior accepted copy-en gain remains required even though original BF16 missed it.
    return (report['four_passed'] is True and report['protection']['accepted'] is True
            and not report['protection']['lost_required_correct_ids'])


def speed_cases(tokenizer):
    # Exact prompts/template from benchmark_native_mixed; only greedy is measured here.
    prompts = [
        ('zh', '请用中文解释电脑运行内存与硬盘的区别，给日常使用的例子。'),
        ('en', 'Explain how to organize a small software project and check that changes are correct.'),
        ('long', '请概括以下说明，并给出三个具体建议：' + '日常对话要简洁清楚，遇到不确定的问题先核实，学习需要持续练习。' * 12),
    ]
    return [{'id': name, 'tokens': tokenizer.encode('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
        '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n',
        add_special_tokens=False)} for name, prompt in prompts]


def speed_report(loaded, rows):
    if (len(rows) != 9 or any(row.get('generated_tokens') != 96 or len(row['tokens']) != 96
            or row.get('cached_prompt_tokens') != 0 or row['decode_ns'] <= 0
            or row['temperature'] != 0 for row in rows)):
        raise ValueError('nine uncached greedy 96-token speed results required')
    # Same aggregation as speed_summary, restricted explicitly to its temperature=0 group.
    return {'cases': len(rows), 'temperature': 0,
        'decode_tps': 95*len(rows)*1e9/sum(row['decode_ns'] for row in rows),
        'median_ttft_seconds': statistics.median(row['ttft_ns']/1e9 for row in rows),
        'median_wall_seconds': statistics.median(row['wall_seconds'] for row in rows),
        'lifetime_peak_physical_footprint_bytes': max(loaded['memory']['lifetime_peak_physical_footprint_bytes'],
            *(row['memory']['lifetime_peak_physical_footprint_bytes'] for row in rows)),
        'resident_weight_bytes': loaded['resident_weight_bytes'], 'load_seconds': loaded['load_ns']/1e9}


def main():
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('mounted experiment volume required')
    controls, audit = read(CONTROLS/'result.json'), read(CONTROLS/'independent-readback.json')
    reference = read(REFERENCE/'result.json')
    receipt, registration = read(Path(str(BINARY)+'.json')), read(REGISTRATION)
    if (registration.get('registered_before_run') is not True
            or reference.get('completed') is not True or reference.get('core') is not None
            or reference.get('modes') != ['bf16', 'mixed-q8']
            or controls.get('completed') is not True or controls.get('controls_passed') is not True
            or audit.get('audit_completed') is not True or audit.get('audit_passed') is not True
            or audit.get('old_bf16_four_exact') is not True or audit.get('new_v2_old20_exact') is not True):
        raise ValueError('registered fixed allocation and prior complete original controls required')
    cases, specs = protection_cases(), controls['cases']
    if len(specs) != 4 or {s['id'] for s in specs} != {
            'mmlu-2804', 'preservation-replay-train-olive-path', 'decode-holdout-unit-volume', 'seq-final-young-dog'}:
        raise ValueError('only existing four exposed checks allowed')
    if any(reference[k] != controls[k] for k in ('cases', 'baseline', 'previous')):
        raise ValueError('old reference cases and protection policy must match')
    union = check_proposal(cases, controls['baseline'], controls['previous'], controls['previous'])
    if (len(union['original_correct_ids']) != 13 or len(union['required_correct_ids']) != 14
            or 'seq-train-copy-en' not in union['required_correct_ids']):
        raise ValueError('original13/prior14 union changed')
    eos = specs[0]['case']['eos_ids']
    if any(spec['case']['eos_ids'] != eos for spec in specs):
        raise ValueError('shared original EOS required')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    for mode in ('bf16', 'mixed-q8'):
        old = reference['reports'][mode]
        check_loaded(old['loaded'], mode, reference['reports']['bf16']['loaded'])
        for spec, row in zip(specs, old['four'], strict=True):
            if diagnostic_result(spec, row['native_result'], tokenizer, 248320) != row:
                raise ValueError('prior four-case scores do not reproduce')
        if check_proposal(cases, controls['baseline'], controls['previous'], old['train']) != old['protection']:
            raise ValueError('prior protection scoring changed')
    OUTPUT.mkdir(exist_ok=False)
    speed = speed_cases(tokenizer)
    protocol = {'schema': 'higher-precision-combinations-v1', 'registered_before_run': True,
        'registration': registration, 'binary': str(BINARY), 'receipt': receipt,
        'reference': str(REFERENCE), 'controls_source': str(CONTROLS), 'cases': specs,
        'baseline': controls['baseline'], 'previous': controls['previous'], 'core': None,
        'modes': list(COUNTS), 'expected_quantized_projections': COUNTS,
        'seed': 42, 'temperature': 0, 'cache_reuse': False, 'training': False, 'teacher_calls': 0,
        'hardware': {'platform': platform.platform(), 'machine': platform.machine()},
        'qualified': False, 'product_adopted': False, 'new_holdout_read': False,
        'full_regression_run': False, 'final_holdout_run': False,
        'speed_cases': speed, 'speed_repeats': 3, 'speed_output_tokens': 96,
        'speed_policy': 'Only after a combination passes four and unchanged prior14 union. Reload BF16 and passing combinations serially; 8-token warmup excluded; EOS=[0], 96 actual tokens, greedy. No semantic score on speed prompts.',
        'measurement_scope': 'Native process Darwin physical footprint and lifetime peak, including load; not total machine memory. Decode TTFT and pipe wall, not application HTTP latency.',
        'scope': 'Exposed diagnosis and candidate screen only; Q8/BF16 combinations contain no Q4. No automatic qualification, later suites or adoption.'}
    save(OUTPUT/'protocol.json', protocol)
    started, worker, lock, reports, benchmarks = time.perf_counter(), None, None, {}, {}
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        for mode in COUNTS:
            directory = OUTPUT/mode; directory.mkdir()
            worker = MixedWorker(BINARY, 'bf16' if mode == 'bf16' else 'candidate_mixed',
                directory, lock, precision=mode, core=None)
            loaded = worker.loaded
            check_loaded(loaded, mode, reference['reports']['bf16']['loaded'])
            save(directory/'loaded.json', loaded)
            four = four_generations(worker, specs, tokenizer, mode+'-four')
            train = protection_generations(worker, cases, tokenizer, eos, 'train', 248320)
            if any(row['native_result'].get('cached_prompt_tokens') != 0 for row in four+train):
                raise ValueError('cached prompts invalidate fixed comparison')
            if mode in ('bf16', 'mixed-q8'):
                exact_rows(four, reference['reports'][mode]['four'], scored=True)
                exact_rows(train, reference['reports'][mode]['train'])
            protection = check_proposal(cases, controls['baseline'], controls['previous'], train)
            worker.close(); code = worker.process.returncode; worker = None
            if code != 0:
                raise RuntimeError('screen worker failed on close')
            report = {'loaded': loaded, 'four': four, 'four_passed': all(r['score']['passed'] for r in four),
                'train': train, 'protection': protection, 'performance': performance(loaded, four+train),
                'worker_exit_code': code, 'old_reference_exact': mode in ('bf16', 'mixed-q8')}
            report['screen_passed'] = screen_passed(report)
            reports[mode] = report
            save(directory/'report.json', report)
        eligible = [mode for mode in COMBINATIONS if reports[mode]['screen_passed']]
        for mode in (['bf16'] + eligible if eligible else []):
            directory = OUTPUT/('speed-'+mode); directory.mkdir()
            worker = MixedWorker(BINARY, 'bf16' if mode == 'bf16' else 'candidate_mixed',
                directory, lock, precision=mode, core=None)
            loaded = worker.loaded
            check_loaded(loaded, mode, reports['bf16']['loaded'])
            save(directory/'loaded.json', loaded)
            worker.request(speed[0]['tokens'], 'warmup', max_tokens=8, temperature=0, seed=42, reuse=False)
            rows = []
            for case in speed:
                for repeat in range(3):
                    identifier = f'speed-{case["id"]}-r{repeat}'
                    raw = worker.request(case['tokens'], identifier, max_tokens=96, temperature=0,
                        seed=42, reuse=False, eos_ids=[0])
                    if raw.get('event') != 'result' or raw.get('id') != identifier:
                        raise ValueError('speed result identity changed')
                    raw.update(temperature=0, phase='speed', case_id=case['id'], repeat=repeat)
                    rows.append(raw); save(directory/'rows.json', rows)
            summary = speed_report(loaded, rows)
            worker.close(); code = worker.process.returncode; worker = None
            if code != 0:
                raise RuntimeError('benchmark worker failed on close')
            benchmarks[mode] = {'loaded': loaded, 'rows': rows, 'summary': summary, 'worker_exit_code': code}
            save(directory/'report.json', benchmarks[mode])
        save(OUTPUT/'result.json', {**protocol, 'completed': True, 'reports': reports,
            'eligible_for_performance': eligible, 'benchmarks': benchmarks,
            'elapsed_seconds': time.perf_counter()-started})
    except BaseException as error:
        save(OUTPUT/'failure.json', {'completed': False, 'error': repr(error), 'reports': reports,
            'benchmarks': benchmarks, 'elapsed_seconds': time.perf_counter()-started})
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
