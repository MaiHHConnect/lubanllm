"""Fixed local Q4 layer-policy and paging screen; no training or adoption."""
import argparse
import os
from pathlib import Path
import platform
import statistics
import time

from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker
from capture_q4_paired_tails import collect
from check_higher_precision_combinations import speed_cases
from diagnose_q4_task_candidate import read
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import protection_generations
from q4_sequential_protection import protection_cases, _score
from study_q4_column_protection import ROOT
from trace_q4_divergence import VOCAB, WIDTHS
from verify_q4_paired_recovery import CONTROLS, four_generations

BINARY = ROOT/'layered-inference-build/zig-out/bin/luban-worker'
OLD = ROOT/'higher-precision-combinations-build/zig-out/bin/luban-worker'
OUTPUT = ROOT/'layered-inference-v1'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/layered_inference_20261017.json'
MODES = {'old_full': (0, 24), 'new_full': (0, 24), 'skip22': (1 << 22, 24),
         'exit23': (0, 23), 'paged_full': (0, 24)}
FULL = {'old_full', 'new_full', 'paged_full'}


class BoundedWorker(MixedWorker):
    def receive(self, timeout=120):
        return super().receive(min(timeout, 120))


def compare_scores(baseline, candidate):
    def indexed(rows):
        result = {r['id']: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError('nonempty unique score coverage required')
        return result
    before, after = indexed(baseline), indexed(candidate)
    if before.keys() != after.keys():
        raise ValueError('same complete case coverage required')
    correct = {key for key, row in before.items() if row['score']['passed']}
    actual = {key for key, row in after.items() if row['score']['passed']}
    if not correct:
        raise ValueError('empty baseline-correct set is not a protection gate')
    lost = sorted(correct-actual)
    return {'baseline_correct': len(correct), 'candidate_correct': len(actual),
            'lost_current_q4_correct_ids': lost, 'gained_ids': sorted(actual-correct), 'passed': not lost}


def exact_rows(actual, previous):
    keys = ('id', 'prompt_tokens', 'tokens', 'text', 'complete')
    if len(actual) != len(previous) or any(any(a[k] != b[k] for k in keys)
            for a, b in zip(actual, previous, strict=True)):
        raise ValueError('full graph outputs differ from identical current Q4 baseline')


def same_bytes(actual, expected, size):
    for path in (actual, expected):
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError('complete regular trace payload required')
    if actual.read_bytes() != expected.read_bytes():
        raise ValueError('native layer/logit trace bytes differ')


def exact_trace(actual, previous):
    if len(actual['events']) != 2 or len(previous['events']) != 2:
        raise ValueError('two complete trace frames required')
    for a, b in zip(actual['events'], previous['events'], strict=True):
        if any(a.get(k) != b.get(k) for k in ('step', 'committed_tokens', 'top1', 'layers', 'layer_values', 'vocab')):
            raise ValueError('trace frame identity or geometry differs')
        for kind, count in [('layers', 24*sum(WIDTHS.values())), ('logits', VOCAB)]:
            same_bytes(Path(a[kind+'_path']), Path(b[kind+'_path']), count*2)
    return {'frames': 2, 'layers': 24, 'vocab': VOCAB, 'all_layer_and_logit_bytes_exact': True}


def check_loaded(loaded, mode, reference=None):
    if (loaded.get('base_identity') != IDENTITY or loaded.get('manifest_sha256') != MANIFEST
            or loaded.get('inference_precision') != 'mixed-q4-g16'
            or loaded.get('quantized_projections') != 71 or loaded.get('core_loaded') is not False
            or loaded.get('core_file_sha256') is not None or loaded.get('core_tensor_sha256') is not None
            or loaded.get('retained_original_weight_bytes') != 0
            or any(loaded.get(k) != 0 for k in ('packed_version', 'residual_rank', 'active_residual_projections'))
            or any(loaded.get(k) is not False for k in
                   ('research_sequential', 'research_capture_inputs', 'research_intervention'))):
        raise ValueError('original coreNone ordinary Q4/group16 graph required')
    if reference and any(loaded.get(k) != reference.get(k) for k in ('checkpoint_sha256', 'original_down_sha256')):
        raise ValueError('base binding changed')
    if mode != 'old_full' and (loaded.get('execution_skip_mask') != MODES[mode][0]
            or loaded.get('execution_exit_after') != MODES[mode][1]
            or loaded.get('layer_paging_enabled') is not (mode == 'paged_full')
            or loaded.get('layer_paging', {}).get('enabled') is not (mode == 'paged_full')):
        raise ValueError('actual native execution policy differs from requested mode')


def cache_probe(worker, case):
    rows = []
    for label, reuse in [('cold', False), ('prime', True), ('cached', True)]:
        raw = worker.request(case['tokens'], 'cache-'+label, max_tokens=1, eos_ids=[0],
                             temperature=0, seed=42, reuse=reuse)
        if (raw.get('id') != 'cache-'+label or raw['generated_tokens'] != 1 or len(raw['tokens']) != 1
                or raw['cached_prompt_tokens'] != (len(case['tokens']) if label == 'cached' else 0)):
            raise ValueError('full-prompt cache probe did not exercise expected cold/cached path')
        if rows and any(raw[k] != rows[0][k] for k in ('tokens', 'complete', 'finish_reason')):
            raise ValueError('cached greedy output differs from cold replay')
        rows.append(raw); save(worker.out/'cache-probe.json', rows)
    return {'passed': True, 'rows': rows}


def benchmark(worker, cases):
    warm = worker.request(cases[0]['tokens'], 'warmup', max_tokens=8, eos_ids=[0],
                          temperature=0, seed=42, reuse=False)
    save(worker.out/'warmup.json', warm)
    if warm.get('id') != 'warmup' or len(warm['tokens']) != 8 or warm['cached_prompt_tokens'] != 0 or warm['finish_reason'] != 'length':
        raise ValueError('eight uncached warmup tokens required')
    rows = []
    for case in cases:
        raw = worker.request(case['tokens'], 'speed-'+case['id'], max_tokens=32, eos_ids=[0],
                             temperature=0, seed=42, reuse=False)
        if (raw.get('id') != 'speed-'+case['id'] or raw['generated_tokens'] != 32 or len(raw['tokens']) != 32
                or raw['cached_prompt_tokens'] != 0 or raw['decode_ns'] <= 0
                or raw['finish_reason'] != 'length'):
            raise ValueError('three uncached fixed 32-token generations required')
        rows.append(raw); save(worker.out/'speed-rows.json', rows)
    return {'rows': rows, 'warmup_excluded': True, 'decode_tps': 31*len(rows)*1e9/sum(r['decode_ns'] for r in rows),
            'median_ttft_seconds': statistics.median(r['ttft_ns']/1e9 for r in rows),
            'median_wall_seconds': statistics.median(r['wall_seconds'] for r in rows)}


def main():
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('mounted experiment volume required')
    registration = read(REGISTRATION)
    if registration.get('registered_before_run') is not True:
        raise ValueError('fixed protocol must be registered before execution')
    controls = read(CONTROLS/'result.json')
    # Existing controls call these four exposed diagnostic specifications cases.
    specs = controls['cases']; cases = protection_cases()
    if (len(specs) != 4 or [s['id'] for s in specs] != ['seq-final-young-dog',
            'decode-holdout-unit-volume', 'preservation-replay-train-olive-path', 'mmlu-2804']
            or controls.get('completed') is not True or controls.get('controls_passed') is not True):
        raise ValueError('existing four exposed specifications required')
    eos = specs[0]['case']['eos_ids']
    if any(s['case']['eos_ids'] != eos for s in specs):
        raise ValueError('fixed shared EOS required')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    speed = speed_cases(tokenizer)
    OUTPUT.mkdir(exist_ok=False)
    protocol = {'schema': 'layered-inference-v1', 'registered_before_run': True, 'registration': registration,
        'binary': str(BINARY), 'old_binary': str(OLD), 'receipt': read(Path(str(BINARY)+'.json')),
        'old_receipt': read(Path(str(OLD)+'.json')), 'controls_source': str(CONTROLS), 'cases': specs,
        'precision': 'mixed-q4-g16', 'core': None, 'modes': MODES,
        'seed': 42, 'temperature': 0, 'cache_reuse': False, 'training': False, 'teacher_calls': 0,
        'qualified': False, 'product_adopted': False, 'new_holdout_read': False,
        'full_regression_run': False, 'final_holdout_run': False, 'speed_cases': speed,
        'speed_output_tokens': 32, 'speed_repeats': 1, 'warmup_tokens': 8,
        'hardware': {'platform': platform.platform(), 'machine': platform.machine()},
        'memory_scope': 'Native Darwin physical footprint and lifetime peak including load, not total machine memory.',
        'paging_scope': 'Cumulative pread calls/bytes; OS file cache allowed, physical SSD reads not measured. No cache purge.',
        'scope': 'Local exposed-case regression and engineering screen only. Current plain Q4 baseline may already miss cases.'}
    save(OUTPUT/'protocol.json', protocol)
    started = time.perf_counter(); reports = {}; worker = lock = None
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        for mode, (skip, depth) in MODES.items():
            start = time.perf_counter(); directory = OUTPUT/mode; directory.mkdir()
            trace_dir = directory/'traces' if mode in FULL else None
            if trace_dir:
                trace_dir.mkdir()
            worker = BoundedWorker(OLD if mode == 'old_full' else BINARY, 'candidate_mixed', directory,
                lock, precision='mixed-q4-g16', core=None, trace_directory=trace_dir,
                skip_layer_mask=skip, exit_after=depth,
                layer_paging_path=directory/'store.bin' if mode == 'paged_full' else None)
            loaded = worker.loaded
            check_loaded(loaded, mode, reports.get('old_full', {}).get('loaded'))
            save(directory/'loaded.json', loaded)
            four = four_generations(worker, specs, tokenizer, mode+'-four')
            report = {'loaded': loaded, 'four': four, 'train': [], 'screen_passed': False}
            reports[mode] = report
            reference = reports['old_full']
            report['four_comparison'] = compare_scores(reference['four'], four)
            if mode in FULL or report['four_comparison']['passed']:
                report['train'] = protection_generations(worker, cases, tokenizer, eos, 'train', VOCAB)
                report['protection_comparison'] = compare_scores(_score(cases, reference['train']), _score(cases, report['train']))
                report['screen_passed'] = report['four_comparison']['passed'] and report['protection_comparison']['passed']
            else:
                report['skipped'] = ['sixteen_protection', 'performance', 'cache_probe']
            if mode in FULL:
                exact_rows(four, reference['four']); exact_rows(report['train'], reference['train'])
                first = reference['four'][0]['tokens'][0]
                if first in eos:
                    raise ValueError('first baseline prediction must allow a second forced trace frame')
                trace_case = {'tokens': specs[0]['case']['tokens'], 'forced_tokens': [first]}
                report['trace'] = collect(worker, trace_case, eos, 'full-graph', tail=False)
                save(directory/'trace.json', report['trace'])
                report['trace_comparison'] = exact_trace(report['trace'], reference['trace'])
                report['full_outputs_exact'] = True
            if report['screen_passed']:
                report['cache_probe'] = cache_probe(worker, specs[0]['case'])
                if mode != 'old_full':
                    report['performance'] = benchmark(worker, speed)
            native = [r['native_result'] for r in four+report['train']]
            native += report.get('cache_probe', {}).get('rows', []) + report.get('performance', {}).get('rows', [])
            report['peak_physical_footprint_bytes'] = max(r['memory']['lifetime_peak_physical_footprint_bytes'] for r in [loaded]+native)
            report['paging_counters'] = {'loaded': loaded.get('layer_paging'),
                'final': native[-1].get('layer_paging') if native else None}
            if mode == 'paged_full' and (report['paging_counters']['final']['read_calls'] <= 0
                    or report['paging_counters']['final']['read_bytes'] <= 0):
                raise ValueError('paged mode did not actually read layer payloads')
            report['elapsed_seconds'] = time.perf_counter()-start
            save(directory/'report.json', report)
            worker.close(); report['worker_exit_code'] = worker.process.returncode; worker = None
            if report['worker_exit_code'] != 0:
                raise RuntimeError('native worker failed on close')
            save(directory/'report.json', report)
        save(OUTPUT/'result.json', {**protocol, 'completed': True, 'reports': reports,
            'local_screen_passed_modes': [m for m, r in reports.items() if r['screen_passed']],
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


def performance_only():
    """One idle-host repeat; preserve the original quality and timing evidence."""
    if not os.path.ismount('/Volumes/移动数据库'):
        raise ValueError('mounted experiment volume required')
    original = read(OUTPUT/'result.json')
    if (original.get('completed') is not True or original.get('binary') != str(BINARY)
            or original.get('core') is not None or original.get('precision') != 'mixed-q4-g16'
            or any(original['reports'][mode].get('screen_passed') is not True
                   for mode in ('new_full', 'paged_full'))):
        raise ValueError('completed local quality screen required before timing-only repeat')
    output = OUTPUT/'performance-idle'
    output.mkdir(exist_ok=False)
    protocol = {'schema': 'layered-inference-idle-performance-v1', 'source': str(OUTPUT/'result.json'),
        'binary': str(BINARY), 'receipt': original['receipt'], 'core': None, 'precision': 'mixed-q4-g16',
        'modes': ['new_full', 'paged_full'], 'speed_cases': original['speed_cases'],
        'speed_repeats': 1, 'speed_output_tokens': 32, 'warmup_tokens': 8,
        'temperature': 0, 'seed': 42, 'cache_reuse': False, 'trace': False,
        'teacher_calls': 0, 'training': False, 'qualified': False, 'product_adopted': False,
        'quality_repeated': False, 'original_results_modified': False,
        'reason': 'Application packaging overlapped the end of the original paged performance run with CPU work. Repeat only fixed performance after packaging finishes; retain original timing evidence.',
        'idle_precondition': 'Operator must finish building/packaging and stop application model work before invocation; driver serializes model access but cannot prove whole-host idleness.',
        'memory_scope': original['memory_scope'], 'paging_scope': original['paging_scope'],
        'hardware': {'platform': platform.platform(), 'machine': platform.machine()}}
    save(output/'protocol.json', protocol)
    started = time.perf_counter(); reports = {}; worker = lock = None
    try:
        lock = ModelProcessLock(IDENTITY, 'mps')
        for mode in protocol['modes']:
            directory = output/mode; directory.mkdir()
            start = time.perf_counter()
            worker = BoundedWorker(BINARY, 'candidate_mixed', directory, lock,
                precision='mixed-q4-g16', core=None,
                layer_paging_path=directory/'store.bin' if mode == 'paged_full' else None)
            loaded = worker.loaded
            check_loaded(loaded, mode, original['reports']['old_full']['loaded'])
            save(directory/'loaded.json', loaded)
            performance = benchmark(worker, protocol['speed_cases'])
            rows = performance['rows']
            report = {'loaded': loaded, 'performance': performance,
                'elapsed_seconds': time.perf_counter()-start,
                'peak_physical_footprint_bytes': max(r['memory']['lifetime_peak_physical_footprint_bytes'] for r in [loaded]+rows),
                'paging_counters': {'loaded': loaded['layer_paging'], 'final': rows[-1]['layer_paging']}}
            if mode == 'paged_full' and (rows[-1]['layer_paging']['read_calls'] <= 0
                    or rows[-1]['layer_paging']['read_bytes'] <= 0):
                raise ValueError('paged repeat did not actually read layer payloads')
            reports[mode] = report
            worker.close(); report['worker_exit_code'] = worker.process.returncode; worker = None
            save(directory/'report.json', report)
            if report['worker_exit_code'] != 0:
                raise RuntimeError('performance worker failed on close')
        save(output/'result.json', {**protocol, 'completed': True, 'reports': reports,
                                   'elapsed_seconds': time.perf_counter()-started})
    except BaseException as error:
        save(output/'failure.json', {'completed': False, 'error': repr(error), 'reports': reports,
                                    'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            if lock is not None:
                lock.close()


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--performance-only', action='store_true', help='Repeat only fixed timing after building/packaging has finished')
    args = parser.parse_args(argv)
    performance_only() if args.performance_only else main()


if __name__ == '__main__':
    run()
