"""Original BF16/Q8 reference on existing exposed checks; no quality qualification."""
import os
from pathlib import Path
import time

from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MODEL, save
from benchmark_native_mixed import MixedWorker
from diagnose_q4_task_candidate import diagnostic_result, ordinary_loaded, read
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import protection_generations
from q4_sequential_protection import check_proposal, protection_cases
from study_q4_column_protection import ROOT
from verify_q4_paired_recovery import BINARY, CONTROLS, four_generations

OUTPUT = ROOT/'q8-original-mixed-reference-20261017'


def performance(loaded, rows):
    native = [row['native_result'] for row in rows]
    return {'load_seconds': loaded['load_ns']/1e9, 'resident_weight_bytes': loaded['resident_weight_bytes'],
        'loaded_memory': loaded['memory'],
        'maximum_reported_lifetime_peak_physical_footprint_bytes': max(
            value['lifetime_peak_physical_footprint_bytes'] for value in [loaded['memory'], *[r['memory'] for r in native]]),
        'cases': [{'id': row['id'], 'prompt_tokens': raw['prompt_tokens'], 'generated_tokens': raw['generated_tokens'],
            'ttft_seconds': raw['ttft_ns']/1e9, 'decode_seconds': raw['decode_ns']/1e9,
            'decode_tps': raw['decode_tps'], 'wall_seconds': raw['wall_seconds'], 'memory': raw['memory']}
            for row, raw in zip(rows, native, strict=True)],
        'scope': 'Actual exposed correctness cases, variable prompt/output lengths; not a standard throughput benchmark. Native process footprint, not total machine memory.'}


def main():
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('mounted experiment volume required')
    controls = read(CONTROLS/'result.json')
    audit = read(CONTROLS/'independent-readback.json')
    receipt = read(Path(str(BINARY)+'.json'))
    if (controls.get('completed') is not True or controls.get('controls_passed') is not True
            or audit.get('audit_completed') is not True or audit.get('audit_passed') is not True
            or audit.get('old_bf16_four_exact') is not True or audit.get('new_v2_old20_exact') is not True):
        raise ValueError('existing independently checked BF16 controls required')
    cases, specs = protection_cases(), controls['cases']
    if len(specs) != 4 or {s['id'] for s in specs} != {
            'mmlu-2804', 'preservation-replay-train-olive-path', 'decode-holdout-unit-volume', 'seq-final-young-dog'}:
        raise ValueError('only existing four exposed checks allowed')
    union = check_proposal(cases, controls['baseline'], controls['previous'], controls['previous'])
    if len(union['original_correct_ids']) != 13 or len(union['required_correct_ids']) != 14 or 'seq-train-copy-en' not in union['required_correct_ids']:
        raise ValueError('original13/prior14 protection union changed')
    eos = specs[0]['case']['eos_ids']
    if any(spec['case']['eos_ids'] != eos for spec in specs):
        raise ValueError('shared original EOS required')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    for spec, old in zip(specs, controls['reports']['bf16']['four'], strict=True):
        if diagnostic_result(spec, old['native_result'], tokenizer, 248320) != old or old['score']['passed'] is not True:
            raise ValueError('original BF16 four-case correctness changed')
    OUTPUT.mkdir(exist_ok=False)
    protocol = {'schema': 'q8-original-mixed-reference-v1', 'registered_before_run': True,
        'binary': str(BINARY), 'receipt': receipt, 'controls_source': str(CONTROLS), 'cases': specs,
        'baseline': controls['baseline'], 'previous': controls['previous'], 'core': None,
        'modes': ['bf16', 'mixed-q8'], 'expected_quantized_projections': {'bf16': 0, 'mixed-q8': 150},
        'seed': 42, 'temperature': 0, 'cache_reuse': False, 'training': False, 'teacher_calls': 0,
        'qualified': False, 'product_adopted': False, 'new_holdout_read': False,
        'full_regression_run': False, 'final_holdout_run': False,
        'scope': 'Reference only for later precision allocation. Both modes run all sixteen exposed protection cases even when Q8 fails four. No automatic acceptance or later suite execution.'}
    save(OUTPUT/'protocol.json', protocol)
    started, worker, lock, reports = time.perf_counter(), None, None, {}
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        for mode in protocol['modes']:
            directory = OUTPUT/mode; directory.mkdir()
            worker = MixedWorker(BINARY, 'bf16' if mode == 'bf16' else 'candidate_mixed',
                directory, lock, precision=mode, core=None)
            loaded = worker.loaded
            if mode == 'bf16':
                ordinary_loaded(loaded, mode)
            elif (any(loaded.get(k) != reports['bf16']['loaded'].get(k) for k in
                      ['base_identity', 'manifest_sha256', 'checkpoint_sha256', 'original_down_sha256'])
                    or loaded.get('quantized_projections') != 150 or loaded.get('retained_original_weight_bytes') != 0
                    or any(loaded.get(k) is not False for k in ['research_sequential', 'research_capture_inputs', 'research_intervention'])
                    or any(loaded.get(k) != 0 for k in ['packed_version', 'residual_rank', 'active_residual_projections'])):
                raise ValueError('ordinary original Q8 with exactly150 projections required')
            if (loaded.get('inference_precision') != mode or loaded.get('core_loaded') is not False
                    or loaded.get('core_file_sha256') is not None or loaded.get('core_tensor_sha256') is not None):
                raise ValueError('original coreNone required')
            save(directory/'loaded.json', loaded)
            four = four_generations(worker, specs, tokenizer, mode+'-four')
            if mode == 'bf16':
                for row, old in zip(four, controls['reports']['bf16']['four'], strict=True):
                    if any(row[k] != old[k] for k in ['id', 'prompt_tokens', 'tokens', 'text', 'complete', 'score']):
                        raise ValueError('original BF16 must reproduce old four tokens and scores')
            train = protection_generations(worker, cases, tokenizer, eos, 'train', 248320)
            protection = check_proposal(cases, controls['baseline'], controls['previous'], train)
            if mode == 'bf16':
                for row, old in zip(train, controls['baseline']['generations'], strict=True):
                    if any(row[k] != old[k] for k in ['id', 'prompt_tokens', 'tokens', 'text', 'complete']):
                        raise ValueError('original BF16 sixteen-case baseline changed')
            worker.close(); code = worker.process.returncode; worker = None
            if code != 0:
                raise RuntimeError('reference worker failed on close')
            reports[mode] = {'loaded': loaded, 'four': four, 'four_passed': all(r['score']['passed'] for r in four),
                'train': train, 'protection': protection, 'performance': performance(loaded, four+train), 'worker_exit_code': code}
            save(directory/'report.json', reports[mode])
        save(OUTPUT/'result.json', {**protocol, 'completed': True, 'reports': reports,
            'elapsed_seconds': time.perf_counter()-started})
    except BaseException as error:
        save(OUTPUT/'failure.json', {'completed': False, 'error': repr(error), 'reports': reports,
            'elapsed_seconds': time.perf_counter()-started})
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
