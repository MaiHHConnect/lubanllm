#!/usr/bin/env python3
"""Resident BF16 versus mixed-precision experiment; no training, adoption or teacher calls."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import platform
import queue
import re
import statistics
import subprocess
import threading

from benchmark_native_decode import CORE, CORE_SHA, IDENTITY, MANIFEST, MODEL, Worker, save
from luban.generation_checks import evaluate_generation, is_finite_generation_check
from luban.process_lock import ModelProcessLock
from tokenizers import Tokenizer

FIXTURES = Path('/Volumes/移动数据库/lubanllm/experiments/native-inference-20261010T060000Z-b68d710e')
CORE_KEY = 'model.language_model.layers.23.mlp.base.down_proj.weight'


class MixedWorker(Worker):
    """Reuse the existing request protocol, with an explicit experimental CLI mode."""
    def __init__(self, binary, name, out, lock, precision="mixed-q4", calibration_profile=None, trace_directory=None,
                 intervention=False, audit_directory=None, core=CORE, core_sha=CORE_SHA, sequential_directory=None,
                 skip_layer_mask=0, exit_after=24, layer_paging_path=None):
        from luban.settings import native_execution_policy
        native_execution_policy(skip_layer_mask, exit_after, layer_paging_path is not None)
        if layer_paging_path is not None and (not Path(layer_paging_path).is_absolute() or os.path.lexists(layer_paging_path)):
            raise ValueError('Layer paging requires a new absolute file path')
        if sequential_directory is not None and (not Path(sequential_directory).is_absolute() or core is not None
                or name == 'candidate_mixed' or trace_directory is not None or intervention or audit_directory is not None):
            raise ValueError('Sequential research requires original BF16 and an absolute directory without other research modes')
        if name == 'candidate_mixed' and precision in {'mixed-q4-g32-residual', 'mixed-q4-g16-packed'} and (
                calibration_profile is None or not Path(calibration_profile).is_absolute()):
            raise ValueError('Residual or packed mode requires an absolute sidecar path')
        if trace_directory is not None and not Path(trace_directory).is_absolute():
            raise ValueError('Research trace requires an absolute directory')
        if audit_directory is not None and (not Path(audit_directory).is_absolute() or not intervention):
            raise ValueError('Quantization audit requires an absolute directory and intervention mode')
        self.name, self.out = name, out
        self.events = (out / (name + '-events.jsonl')).open('x')
        self.stderr = (out / (name + '-stderr.log')).open('x')
        self.cancel_read, self.cancel_write = os.pipe()
        self.messages = queue.Queue()
        command = [str(binary), MODEL, MANIFEST, str(self.cancel_read)]
        if skip_layer_mask:
            command.append('--skip-layer-mask=' + str(skip_layer_mask))
        if exit_after != 24:
            command.append('--exit-after=' + str(exit_after))
        if layer_paging_path is not None:
            command.append('--layer-paging=' + str(layer_paging_path))
        if core is not None:
            command.extend([str(core), CORE_KEY, core_sha])
        if name == 'candidate_mixed' and precision in {'mixed-q4-g32-residual', 'mixed-q4-g16-packed'}:
            command.append('--' + precision + '=' + str(calibration_profile))
        elif name == 'candidate_mixed':
            command.append(('--mixed-q4-g32-group-early8-calibrated=' if precision == 'mixed-q4-g32-group-early8-calibrated' else '--mixed-q4-g32-group-early4-calibrated=' if precision == 'mixed-q4-g32-group-early4-calibrated' else '--mixed-q4-g32-group-calibrated=' if precision == 'mixed-q4-g32-group-calibrated' else '--mixed-q4-g32-calibrated=' if precision == 'mixed-q4-g32-calibrated' else '--mixed-q4-calibrated=') + str(calibration_profile) if precision in {'mixed-q4-calibrated', 'mixed-q4-g32-calibrated', 'mixed-q4-g32-group-calibrated', 'mixed-q4-g32-group-early4-calibrated', 'mixed-q4-g32-group-early8-calibrated'} else '--' + precision)
        if trace_directory is not None:
            command.append('--research-trace=' + str(trace_directory))
        if intervention:
            command.append('--research-intervention')
        if audit_directory is not None:
            command.append('--research-audit-quantization=' + str(audit_directory))
        if sequential_directory is not None:
            command.append('--research-sequential=' + str(sequential_directory))
        save(out / (name + '-invocation.json'), dict(command=command))
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, text=True, pass_fds=(self.cancel_read, lock.fd), bufsize=1)
        except BaseException:
            os.close(self.cancel_read); os.close(self.cancel_write)
            self.events.close(); self.stderr.close()
            raise

        def read():
            try:
                for line in self.process.stdout:
                    self.events.write(line); self.events.flush()
                    self.messages.put(json.loads(line))
            except BaseException as error:
                self.messages.put(error)
            finally:
                self.messages.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        try:
            self.loaded = self.receive(300)
            if (self.loaded.get('event') != 'loaded' or self.loaded.get('base_identity') != IDENTITY
                    or self.loaded.get('core_loaded') is not (core is not None)
                    or self.loaded.get('core_file_sha256') != (core_sha if core is not None else None)):
                raise RuntimeError(f'Worker binding mismatch: {self.loaded}')
            expected_precision = precision if name == 'candidate_mixed' else 'bf16'
            if self.loaded.get('inference_precision') != expected_precision:
                raise RuntimeError(f'Worker precision mismatch: {self.loaded}')
        except BaseException:
            self.close()
            raise

    def close(self):
        # Do not mask an earlier worker error with an assertion during cleanup.
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=10)
        self.reader.join(timeout=5)
        self.process.stdout.close()
        os.close(self.cancel_read); os.close(self.cancel_write)
        self.events.close(); self.stderr.close()


def score(text, fixture, complete):
    """Use original labels; changed prose/token IDs are not themselves a failure."""
    if 'check' in fixture or 'expected' in fixture:
        check = fixture.get('check', {'type': 'exact', 'expected': fixture.get('expected')})
        return dict(passed=evaluate_generation(text, check, complete)['passed'],
                    finite=is_finite_generation_check(check), method=check)
    method, answer = fixture['scoring'], fixture['answer']
    if method == 'choice':
        correct = bool(re.fullmatch('[A-Ja-j]', text.strip())) and text.strip().upper() == str(answer).upper()
    elif method == 'exact_number':
        correct = bool(re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', text.strip()))
        correct = correct and Decimal(text.strip()) == Decimal(str(answer))
    elif method == 'json':
        correct = evaluate_generation(text, {'type': 'semantic', 'validator': 'json', 'expected': answer}, complete)['passed']
    else:
        raise ValueError(f'Unsupported fixture scoring: {method}')
    return dict(passed=bool(complete and correct), finite=True, method=method)


def load_cases(directory, limit=None):
    suites = {}
    for suite, count in [('ranking', 48), ('preservation', 112)]:
        cases = json.loads((directory / f'cases-{suite}.json').read_text())
        fixtures = json.loads((directory / f'fixtures-{suite}.json').read_text())
        if len(cases) != count or len(fixtures) != count or len({c['id'] for c in cases}) != count:
            raise ValueError(f'Incomplete or duplicate {suite} fixtures')
        for case, fixture in zip(cases, fixtures, strict=True):
            if case['id'] != fixture['id']:
                raise ValueError('Fixture order mismatch')
            score('', fixture, False)  # Validate labels before starting a model.
        paired = list(zip(cases, fixtures))
        if limit is not None:
            # Smoke probes cover different validators before filling remaining slots.
            selected, kinds = [], set()
            for pair in paired:
                fixture = pair[1]
                check = fixture.get('check', {})
                kind = fixture.get('scoring', check.get('validator', check.get('type', 'exact')))
                if kind not in kinds:
                    selected.append(pair); kinds.add(kind)
            paired = (selected + [pair for pair in paired if pair not in selected])[:limit]
        suites[suite] = paired
    return suites


def quality_summary(rows):
    result = {}
    for suite in ('ranking', 'preservation'):
        for temperature in sorted({r['temperature'] for r in rows if r['phase'] == suite}):
            group = [r for r in rows if r['phase'] == suite and r['temperature'] == temperature]
            by_worker = {name: {r['case_id']: r for r in group if r['worker'] == name}
                         for name in ('baseline_bf16', 'candidate_mixed')}
            baseline, candidate = by_worker.values()
            if not baseline or baseline.keys() != candidate.keys():
                raise ValueError('Missing paired quality outputs')
            lost = [key for key in baseline if baseline[key]['score']['passed'] and not candidate[key]['score']['passed']]
            result[f'{suite}-t{temperature}'] = dict(cases=len(baseline),
                baseline_passed=sum(r['score']['passed'] for r in baseline.values()),
                candidate_passed=sum(r['score']['passed'] for r in candidate.values()),
                lost_baseline_correct_ids=lost,
                lost_finite_correct_ids=[key for key in lost if baseline[key]['score']['finite']],
                gained_correct_ids=[key for key in baseline if not baseline[key]['score']['passed'] and candidate[key]['score']['passed']],
                finite_cases=sum(r['score']['finite'] for r in baseline.values()),
                permissive_check_cases=sum(not r['score']['finite'] for r in baseline.values()))
    return result


def speed_summary(rows):
    result = {}
    for worker in ('baseline_bf16', 'candidate_mixed'):
        result[worker] = {}
        for temperature in (0, .6):
            group = [r for r in rows if r['worker'] == worker and r['phase'] == 'speed' and r['temperature'] == temperature]
            result[worker][str(temperature)] = dict(cases=len(group),
                decode_tps=95 * len(group) * 1e9 / sum(r['decode_ns'] for r in group),
                median_ttft_seconds=statistics.median(r['ttft_ns'] / 1e9 for r in group),
                median_wall_seconds=statistics.median(r['wall_seconds'] for r in group),
                lifetime_peak_physical_footprint_bytes=max(r['memory']['lifetime_peak_physical_footprint_bytes'] for r in group))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=['mixed-q4', 'mixed-q4-g16', 'mixed-q4-g32-middle', 'mixed-q4-g32-clip97', 'mixed-q4-g32-calibrated', 'mixed-q4-g32-group-calibrated', 'mixed-q4-g32-group-early4-calibrated', 'mixed-q4-g32-group-early8-calibrated', 'mixed-q4-g32-gateup', 'mixed-q4-g32-down', 'mixed-q4-g32-early4', 'mixed-q4-g32-mse', 'mixed-q4-g32', 'mixed-q4-g32-skip0-early8', 'mixed-q4-g32-skip0', 'mixed-q4-g32-early12', 'mixed-q4-mlp', 'mixed-q4-gateup', 'mixed-q4-down', 'mixed-q4-first1', 'mixed-q4-first2', 'mixed-q4-first4', 'mixed-q4-first6', 'mixed-q4-early', 'mixed-q4-early8', 'mixed-q4-early12', 'mixed-q4-calibrated', 'mixed-q8'], default='mixed-q4')
    parser.add_argument('--binary', type=Path, required=True, help='Same worker binary for both modes')
    parser.add_argument('--output', type=Path, required=True, help='New directory on mounted external volume')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--quality-limit', type=int, help='Smoke only: cap each suite, never full qualification')
    parser.add_argument('--sampled-quality', action='store_true', help='Also run quality at temperature 0.6')
    parser.add_argument('--calibration-profile', type=Path, help='Absolute Q4 clip sidecar for mixed-q4-calibrated')
    args = parser.parse_args()
    if args.precision in {'mixed-q4-calibrated', 'mixed-q4-g32-calibrated', 'mixed-q4-g32-group-calibrated', 'mixed-q4-g32-group-early4-calibrated', 'mixed-q4-g32-group-early8-calibrated'}:
        if args.calibration_profile is None or not args.calibration_profile.is_absolute() or not args.calibration_profile.is_file():
            parser.error('--calibration-profile must name an existing absolute profile for mixed-q4-calibrated')
    out = args.output
    if (not out.is_absolute() or len(out.parts) < 4 or out.parts[1] != 'Volumes'
            or not os.path.ismount(Path(*out.parts[:3]))):
        raise ValueError('Output must be on a mounted external volume')
    if args.quality_limit is not None and args.quality_limit < 1:
        parser.error('--quality-limit must be positive')
    suites = load_cases(args.fixtures, args.quality_limit)
    quality_temperatures = [0, .6] if args.sampled_quality else [0]
    tokenizer = Tokenizer.from_file(MODEL + '/tokenizer.json')
    prompts = [
        ('zh', '请用中文解释电脑运行内存与硬盘的区别，给日常使用的例子。'),
        ('en', 'Explain how to organize a small software project and check that changes are correct.'),
        ('long', '请概括以下说明，并给出三个具体建议：' + '日常对话要简洁清楚，遇到不确定的问题先核实，学习需要持续练习。' * 12),
    ]
    speed_cases = [(name, tokenizer.encode('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        + prompt + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n', add_special_tokens=False).ids) for name, prompt in prompts]
    out.mkdir()
    save(out / 'protocol.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        hardware=dict(platform=platform.platform(), machine=platform.machine(),
            processor=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()),
        binary=str(args.binary.resolve()), model=MODEL, core=CORE, core_sha256=CORE_SHA,
        manifest_sha256=MANIFEST, precision=dict(baseline='BF16', candidate=args.precision + ', learned core BF16'),
        seed=42, speed_temperatures=[0, .6], quality_temperatures=quality_temperatures,
        quality_limit=args.quality_limit, full_quality_suite=args.quality_limit is None,
        cache_reuse=False, resident_workers='serial, one per mode',
        memory_method='worker Darwin proc_pid_rusage RUSAGE_INFO_V4 physical footprint and lifetime peak, includes load',
        timing_scope='worker TTFT/decode and pipe request wall; load separate; not HTTP/app latency',
        split='48 existing ranking probes; 112 prior training/replay preservation cases, not held-out transfer',
        quality_policy='natural EOS, original budgets and labels; contains/nonempty checks reported as permissive',
        speed_policy='96 actual output tokens, EOS=0; not semantic quality; 8-token warmup excluded',
        teacher_calls=0, training=False, product_adopted=False))
    save(out / 'cases.json', dict(quality={key: [dict(case=c, fixture=f) for c, f in values] for key, values in suites.items()},
                                  speed=[dict(id=name, tokens=tokens) for name, tokens in speed_cases]))
    rows, loaded, closed = [], {}, []
    lock = ModelProcessLock(IDENTITY, 'mps')
    try:
        with (out / 'answers.jsonl').open('x') as answers:
            for name in ('baseline_bf16', 'candidate_mixed'):
                worker = MixedWorker(args.binary.resolve(), name, out, lock, args.precision, args.calibration_profile.resolve() if args.calibration_profile else None)
                loaded[name] = worker.loaded
                save(out / 'loaded.json', loaded)
                try:
                    worker.request(speed_cases[0][1], 'warmup', max_tokens=8, reuse=False)

                    def record(case, phase, temperature, fixture=None):
                        result = worker.request(case['tokens'], f'{phase}-{case["id"]}-t{temperature}',
                            max_tokens=case['max_tokens'], temperature=temperature, seed=42,
                            reuse=False, eos_ids=case['eos_ids'])
                        if result.get('cached_prompt_tokens') != 0:
                            raise RuntimeError('Cache reuse invalidates this comparison')
                        result.update(phase=phase, case_id=case['id'], temperature=temperature,
                            text=tokenizer.decode(result['tokens'], skip_special_tokens=True), cache_reuse=False)
                        if fixture is not None:
                            result['score'] = score(result['text'], fixture, result['complete'])
                        answers.write(json.dumps(result, ensure_ascii=False) + '\n'); answers.flush()
                        rows.append(result)
                        if phase == 'speed' and (len(result['tokens']) != 96 or not result['decode_ns']):
                            raise RuntimeError('Speed request did not generate exactly 96 tokens')

                    for temperature in (0, .6):
                        for case_id, tokens in speed_cases:
                            for repeat in range(3):
                                record(dict(id=f'{case_id}-r{repeat}', tokens=tokens, max_tokens=96, eos_ids=[0]), 'speed', temperature)
                    for temperature in quality_temperatures:
                        for suite, cases in suites.items():
                            for case, fixture in cases:
                                record(case, suite, temperature, fixture)
                finally:
                    worker.close()
                    closed.append(dict(worker=name, pid=worker.process.pid, returncode=worker.process.returncode))
                    save(out / 'closed.json', closed)
                if worker.process.returncode:
                    raise RuntimeError(f'Worker exited with {worker.process.returncode}')
    finally:
        lock.close()
    quality = quality_summary(rows)
    save(out / 'results.json', dict(completed=True, quality=quality, speed=speed_summary(rows),
        loaded=loaded, closed=closed, full_quality_suite=args.quality_limit is None,
        quality_retained=all(not r['lost_baseline_correct_ids'] for r in quality.values()),
        product_adopted=False, teacher_calls=0, training=False, token_equivalence_required=False,
        limitations='Finite existing probes only; trained-set preservation is not transfer. No adoption or universal quality claim.'))


if __name__ == '__main__':
    main()
