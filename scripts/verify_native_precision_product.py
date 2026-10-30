"""Paired precision checks through the current product Runtime; no model writes."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import threading
import time
import unicodedata
from unittest.mock import patch

import torch

from benchmark_native_decode import save
from benchmark_native_mixed import FIXTURES, load_cases, score
from luban import product_inference
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings

PRECISIONS = ('bf16', 'mixed-q4-g16', 'mixed-q8')
PACKED_PRECISION = 'mixed-q4-g16-packed'
IDENTITY_ANSWER = '我是麦好火旗下的luban0，luban0模型，基于qwen3.5开始生长的，能无限进化的模型。'


def fresh_cases():
    """Fixed before any candidate output; exposure makes later reuse regression."""
    simple = [
        ('upper', '把 Maple_Ridge42 转为大写，只输出转换结果。', 'MAPLE_RIDGE42'),
        ('copy', '请逐字复制 K7-nova_503，不要添加其他内容。', 'K7-nova_503'),
        ('count', '列表 [4, 9, 4, 7, 9, 2, 7] 去重后有几个不同的整数？只输出数字。', '4'),
        ('characters', '字符串 abcaac 中小写字母 a 出现几次？只输出数字。', '3'),
        ('arithmetic', '计算 28 + 17 - 9，只输出整数。', '36'),
        ('json', '输出一个 JSON 对象，键 count 的值为整数 3，键 ok 的值为布尔值 true，不要解释。', {'count': 3, 'ok': True}),
        ('nested', '读取 JSON {"box":{"count":23},"count":8}，box 中的 count 是多少？只输出整数。', '23'),
        ('unicode', '原样输出字符串「青山-Blue_29」，不要输出引号或其他内容。', '青山-Blue_29'),
    ]
    rows = []
    for name, prompt, expected in simple:
        check = {'type': 'semantic', 'validator': 'json', 'expected': expected} if isinstance(expected, dict) else {'type': 'exact', 'expected': expected}
        rows.append({'id': 'fresh-' + name, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 192, 'check': check})
    for repeats in (24, 72):
        for kind, prefix, question, expected in (
            ('copy', '本次登记编号为 NX-7306。\n', '只输出本次登记编号。', 'NX-7306'),
            ('upper', '本次登记字符串为 Birch_Trail9。\n', '将本次登记字符串转为大写，只输出结果。', 'BIRCH_TRAIL9')):
            prompt = prefix + '附加日志：设备检查结束，文件整理完成，没有变更登记信息。\n' * repeats + question
            rows.append({'id': f'fresh-long-{kind}-{repeats}', 'messages': [{'role': 'user', 'content': prompt}],
                         'max_tokens': 192, 'check': {'type': 'exact', 'expected': expected}})
    return rows


def route_cases():
    entries = [
        ('identity-entry', '你是谁？', True),
        ('identity-brand', '请说明你的品牌归属。', True),
        ('identity-source', '请说出你的身份和基座来源。', True),
        ('ordinary-integer', '计算17加8，只输出整数。', False),
        ('ordinary-uppercase', '把字符串“PyTorch”转为大写，只输出结果。', False),
        ('ordinary-json', 'JSON {"n":7} 中 n 的值是多少？', False),
        ('ordinary-translation', '请把 hello 翻译成中文，只输出译文。', False),
        ('ordinary-fact', 'H2O 的中文名称是什么？只输出名称。', False),
        ('unknown-weather', '今天天气如何？', False),
        ('unknown-quantum', '请解释量子纠缠。', False),
        ('unknown-web', '帮我写一个网页。', False),
        ('unknown-http', '写一个最短的 HTTP GET 请求示例。', False),
        ('unknown-clothing', '今天适合穿什么衣服？', False),
    ]
    return [{'id': name, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 192,
             'expected_action': 'neural_expert' if identity else 'local_core',
             **({'check': {'type': 'exact', 'expected': IDENTITY_ANSWER}} if identity else {})}
            for name, prompt, identity in entries]


def packed_candidates(values, reserved=PRECISIONS):
    result = []
    used = set(reserved) | {PACKED_PRECISION}
    for value in values:
        label, separator, filename = value.partition('=')
        if not separator or not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', label) or label in used:
            raise ValueError('packed candidate requires a unique NAME=PATH, distinct from precision names')
        path = Path(filename)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('packed candidate requires an existing absolute file')
        used.add(label)
        result.append({'label': label, 'precision': PACKED_PRECISION, 'profile': str(path)})
    return result


def read_holdout(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get('schema') != 'q4-packed-final-holdout-v1' or payload.get('frozen') is not True:
        raise ValueError('fresh holdout must use the frozen final-holdout schema')
    rows = payload.get('cases')
    if not isinstance(rows, list) or not rows or len(rows) > 256:
        raise ValueError('fresh holdout needs 1..256 cases')
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'] or row['id'] in identifiers:
            raise ValueError('fresh holdout contains missing or duplicate IDs')
        identifiers.add(row['id'])
        if type(row.get('max_tokens')) is not int or not 1 <= row['max_tokens'] <= 8192:
            raise ValueError('fresh holdout needs a bounded token budget')
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or m.get('role') not in {'system','user','assistant'} or not isinstance(m.get('content'), str) or not m['content'] for m in messages):
            raise ValueError('fresh holdout messages are invalid')
        if 'check' not in row or not score('', row, False)['finite']:
            raise ValueError('fresh holdout requires a finite exact or structural check')
    return rows


def cases_for_run(fixtures, fresh_holdout=None, regression_holdout=None):
    existing = load_cases(fixtures)
    suites = {phase: [dict(fixture, max_tokens=case['max_tokens']) for case, fixture in pairs]
              for phase, pairs in existing.items()}
    suites.update(fresh=fresh_cases(), routes=route_cases())
    sampled = {phase: {case['id'] for case, _ in pairs} for phase, pairs in load_cases(fixtures, 4).items()}
    sampled.update(fresh={row['id'] for row in suites['fresh'][:4]}, routes={row['id'] for row in suites['routes'][:5]})
    plan = [(phase, fixture, temperature) for temperature in (0, .6) for phase, rows in suites.items()
            for fixture in rows if temperature == 0 or fixture['id'] in sampled[phase]]
    normalized_messages = lambda messages: json.dumps([
        {'role': message['role'], 'content': ''.join(unicodedata.normalize('NFKC', message['content']).split())}
        for message in messages], sort_keys=True, ensure_ascii=False)
    for phase, path in (('holdout-regression', regression_holdout), ('holdout', fresh_holdout)):
        if path is None:
            continue
        holdout = read_holdout(path)
        existing_ids = {row['id'] for _, row, _ in plan}
        existing_messages = {normalized_messages(row['messages']) for _, row, _ in plan}
        new_messages = [normalized_messages(row['messages']) for row in holdout]
        if (len(set(new_messages)) != len(new_messages) or any(row['id'] in existing_ids for row in holdout)
                or set(new_messages) & existing_messages):
            raise ValueError('holdout overlaps existing regression material or itself')
        plan.extend((phase, row, 0) for row in holdout)
    keys = [(phase, row['id'], temperature) for phase, row, temperature in plan]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate planned cases')
    for phase, fixture, _ in plan:
        if phase != 'routes' or 'check' in fixture:
            score('', fixture, False)
    return plan


def result_score(text, complete, usage, fixture, phase):
    if phase != 'routes':
        return score(text, fixture, complete)
    route_ok = usage['local_action'] == fixture['expected_action']
    if 'check' in fixture:
        result = score(text, fixture, complete)
        result['passed'] = bool(route_ok and result['passed'])
        result['route_passed'] = route_ok
        return result
    return {'passed': route_ok, 'finite': False, 'method': 'route-only; no semantic quality claim', 'route_passed': route_ok}


def summarize(rows, plan, modes):
    expected = {(phase, fixture['id'], temperature) for phase, fixture, temperature in plan}
    indexed = {}
    for mode in modes:
        subset = [row for row in rows if row.get('label', row['precision']) == mode]
        mapped = {(row['phase'], row['case_id'], row['temperature']): row for row in subset}
        if len(mapped) != len(subset) or set(mapped) != expected:
            raise ValueError('missing or duplicate paired outputs: ' + mode)
        indexed[mode] = mapped
    baseline = indexed['bf16']
    result = {}
    for mode in modes:
        phases = {}
        for phase, _, temperature in sorted(expected):
            name = f'{phase}-t{temperature}'
            if name in phases:
                continue
            keys = sorted(key for key in expected if key[0] == phase and key[2] == temperature)
            candidate = indexed[mode]
            lost = [key[1] for key in keys if baseline[key]['score']['passed'] and not candidate[key]['score']['passed']]
            phases[name] = {'cases': len(keys), 'baseline_passed': sum(baseline[key]['score']['passed'] for key in keys),
                'passed': sum(candidate[key]['score']['passed'] for key in keys), 'lost_baseline_correct': lost,
                'gained_correct': [key[1] for key in keys if not baseline[key]['score']['passed'] and candidate[key]['score']['passed']],
                'changed_tokens': [key[1] for key in keys if baseline[key]['tokens'] != candidate[key]['tokens']],
                'incomplete': [key[1] for key in keys if not candidate[key]['complete']]}
        route_rows = [row for key, row in indexed[mode].items() if key[0] == 'routes']
        routes_passed = all(row['score']['passed'] for row in route_rows)
        execution = {}
        for row in indexed[mode].values():
            engine_name = row.get('engine', {}).get('engine', 'unrecorded')
            execution[engine_name] = execution.get(engine_name, 0) + 1
        result[mode] = {'quality': phases, 'route_checks_passed': routes_passed, 'execution_paths': execution,
                        'quality_retained': routes_passed and all(not group['lost_baseline_correct'] for group in phases.values())}
    return result


def evaluation_stages(plan, modes, rows, output):
    """Finish paired exposed regression before allowing any final-holdout request."""
    holdout = [entry for entry in plan if entry[0] == 'holdout']
    regression = [entry for entry in plan if entry[0] != 'holdout']
    if not holdout:
        for candidate in modes:
            yield 'all', candidate, plan
        return
    if not regression:
        raise ValueError('final holdout requires a nonempty paired regression stage')
    for candidate in modes:
        yield 'regression', candidate, regression
    results = summarize(rows, regression, [mode['label'] for mode in modes])
    rejected = [mode['label'] for mode in modes if mode['label'] != 'bf16'
                and not results[mode['label']]['quality_retained']]
    save(output / 'regression-result.json', {'completed': True, 'results': results,
        'cases_per_mode': len(regression), 'answers': len(rows), 'rejected_candidates': rejected,
        'final_holdout_allowed': not rejected, 'final_holdout_skipped': bool(rejected),
        'scope': 'All paired exposed regressions completed; no final holdout has been generated yet.'})
    if rejected:
        return
    for candidate in modes:
        yield 'holdout', candidate, holdout


def capture_generation(runtime, original_generate, captured, tokens, maximum, **kwargs):
    # The caller is still inside answer_backbone and the selected expert scope.
    action = 'neural_expert' if runtime.branch.expert is not None else 'local_core'
    engine = runtime.inference_identity(actual_action=action)
    depth = getattr(runtime.cold_backbone, 'depth_count', 0)
    result = original_generate(tokens, maximum, **kwargs)
    captured.update(prompt_tokens=tokens[0].tolist(),
                    tokens=result[0, tokens.shape[1]:].detach().cpu().tolist(),
                    engine=engine, execution_action=action, execution_hf_depth=depth)
    return result


def pointer_bytes(store):
    return {name: (store / name).read_bytes() if (store / name).exists() else None
            for name in ('active-core.json', 'active.json', 'active-router.json')}


def readonly_product_state(store, identity):
    """Refuse production recovery; retain its unaccepted-research pointer guard."""
    from luban.training_checkpoint import ensure_product_state
    database = store.parents[1] / 'app/memory.sqlite3'
    if database.is_symlink() or not database.is_file():
        raise ValueError('production journal must exist as a regular file')
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_runs'").fetchone()
        if not exists:
            raise ValueError('production training journal table is missing')
        records = db.execute('SELECT body,state FROM training_runs WHERE identity=?', (identity,)).fetchall()
        rows = [{**json.loads(body), 'state': state} for body, state in records]
    if any(row['state'] in {'queued', 'running', 'paused', 'committing'} for row in rows):
        raise ValueError('production journal has pending training or recovery; validation is read-only')
    # Reuse product's active unaccepted-research checks, without its schema-writing reader.
    with patch('luban.training_checkpoint.training_runs', return_value=rows):
        ensure_product_state(None, store, identity)


@contextmanager
def isolated_runtime_io(memory, store):
    def check_only(actual_memory, actual_store, identity, *, research=False):
        if actual_memory is not memory or Path(actual_store) != store or research:
            raise ValueError('validation runtime isolation mismatch')
        readonly_product_state(store, identity)
    with patch('luban.runtime.Memory', return_value=memory), patch('luban.runtime.recover_commits', side_effect=check_only):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=FIXTURES)
    parser.add_argument('--model', type=Path, default=Path('/Volumes/移动数据库/lubanllm/models/luban0'))
    parser.add_argument('--storage', type=Path, default=Path('/Volumes/移动数据库/lubanllm'))
    parser.add_argument('--precisions', nargs='+', choices=PRECISIONS, default=list(PRECISIONS))
    parser.add_argument('--packed-candidate', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--fresh-holdout', type=Path, help='Frozen final-observation JSON, appended at t0 only')
    parser.add_argument('--regression-holdout', type=Path, help='Previously exposed holdout JSON; separate t0 regression before new holdout')
    args = parser.parse_args()
    if args.precisions[0] != 'bf16' or len(set(args.precisions)) != len(args.precisions):
        parser.error('precisions must start with bf16 and contain no duplicates')
    out = args.output.absolute()
    if len(out.parts) < 4 or out.parts[1] != 'Volumes' or not os.path.ismount(Path(*out.parts[:3])):
        raise ValueError('output requires mounted external volume')
    if not args.binary.is_absolute() or not args.binary.is_file() or not Path(str(args.binary) + '.json').is_file():
        raise ValueError('explicit native worker and build receipt required')
    if os.environ.get('LUBAN_BUN_RPC_URL') or os.environ.get('LUBAN_BUN_RPC_TOKEN'):
        raise ValueError('run outside the application RPC environment')
    modes = [{'label': mode, 'precision': mode, 'profile': None} for mode in args.precisions]
    modes.extend(packed_candidates(args.packed_candidate))
    labels = [mode['label'] for mode in modes]
    plan = cases_for_run(args.fixtures, args.fresh_holdout, args.regression_holdout)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'cases.json', [{'phase': phase, 'fixture': fixture, 'temperature': temp} for phase, fixture, temp in plan])
    save(out / 'protocol.json', {'schema': 'native-precision-current-product-v1', 'created': datetime.now(timezone.utc).isoformat(),
        'binary': str(args.binary), 'model': str(args.model), 'production_storage': str(args.storage), 'precisions': args.precisions, 'candidates': modes,
        'fresh_holdout': str(args.fresh_holdout) if args.fresh_holdout else None,
        'regression_holdout': str(args.regression_holdout) if args.regression_holdout else None,
        'runtime_research': False, 'paging': True, 'seed': 42, 'cache_reuse': False, 'samples_per_mode': len(plan),
        'final_holdout_policy': 'All modes finish non-holdout regression first; any candidate quality-retention failure skips final holdout for every mode.',
        'baseline': 'fresh current Runtime admitted core/expert/router; never historical B45 answers',
        'scope': '160 prior regression fixtures +12 probes +13 route probes, t0; fixed17-case t0.6 subset; optional exposed holdout-regression then frozen holdout last at t0',
        'memory': 'native loaded Darwin footprint; not combined app peak', 'hardware': platform.platform(),
        'observation': 'generate_tokens wrapper records actual engine/action/HF depth within execution scope and returned IDs; emit=None gives native seed42',
        'isolation': 'temporary private settings and empty Memory; current production modules read-only; no references or observation writes',
        'limits': 'fresh probes are authored held-out from prior experiments, not a broad transfer benchmark; route-only checks do not score semantics; experts use HF',
        'teacher_calls': 0, 'training': False, 'product_adopted': False})
    settings = Settings(out / 'settings')
    settings.update({'model_path': str(args.model), 'storage_root': str(args.storage), 'local_thinking': 'off', 'temperature': 0})
    private_memory = Memory(out / 'memory')
    runtime = Runtime(settings, private_memory, research=False)
    store = args.storage / 'modules/luban0-v1'
    before = pointer_bytes(store)
    if any(value is None for value in before.values()):
        raise ValueError('current product pointers required; do not auto-install foundation in this validation')
    original_native_binary = product_inference.native_binary

    def selected_binary(instance):
        return args.binary if original_native_binary(instance) is not None else None

    started = time.perf_counter()
    rows, loaded, captured, load_history = [], {}, {}, []
    original_generate = runtime.generate_tokens

    def capture(tokens, maximum, **kwargs):
        return capture_generation(runtime, original_generate, captured, tokens, maximum, **kwargs)

    runtime.generate_tokens = capture
    try:
        with isolated_runtime_io(private_memory, store), patch('luban.product_inference.native_binary', side_effect=selected_binary):
            runtime.load()
            if runtime.memory is not private_memory or private_memory.model_identity != runtime.identity:
                raise ValueError('Runtime.load replaced isolated memory')
            binding = {'identity': runtime.identity, 'active_core': runtime.active_core,
                'core_requires_review': runtime.core_requires_review,
                'expert_pointer': json.loads(before['active.json']), 'router_pointer': json.loads(before['active-router.json']),
                'router_metadata': runtime.router_meta}
            save(out / 'binding.json', binding)
            with (out / 'answers.jsonl').open('x') as log:
                for stage, candidate, stage_plan in evaluation_stages(plan, modes, rows, out):
                    mode, label = candidate['precision'], candidate['label']
                    if pointer_bytes(store) != before:
                        raise ValueError('production bindings changed between modes')
                    readonly_product_state(store, runtime.identity)
                    product_inference.close(runtime)
                    runtime.native_inference_precision = mode
                    runtime.native_q4_profile = Path(candidate['profile']) if candidate['profile'] else None
                    engine = product_inference.start(runtime)
                    if engine is None or engine.loaded['inference_precision'] != mode:
                        raise ValueError('product native precision was not loaded')
                    if (engine.loaded.get('base_identity') != binding['identity'] or
                            bool(engine.loaded.get('core_loaded')) != (binding['active_core'] is not None)):
                        raise ValueError('native model binding differs from current Runtime')
                    loaded[label] = dict(engine.loaded, binary_sha256=engine.binary_sha256, candidate_profile=candidate['profile'])
                    save(out / 'loaded.json', loaded)
                    load_history.append({'stage': stage, 'label': label, 'loaded': loaded[label]})
                    save(out / 'load-history.json', load_history)
                    for phase, fixture, temperature in stage_plan:
                        captured.clear()
                        torch.manual_seed(42)
                        tick = time.perf_counter()
                        usage = runtime.generate(fixture['messages'], None, threading.Event(), thinking='off',
                            max_tokens=fixture['max_tokens'], temperature=temperature,
                            allow_reference=True, observe_external=False, cache_reuse=False)
                        if usage['reference_used'] or usage.get('cached_prompt_tokens', 0):
                            raise ValueError('unexpected reference or prefix cache use')
                        text = runtime.tokenizer.decode(captured['tokens'], skip_special_tokens=True)
                        if captured['execution_action'] != usage['local_action']:
                            raise ValueError('recorded execution action differs from generation usage')
                        row = {'label': label, 'precision': mode, 'phase': phase, 'case_id': fixture['id'], 'temperature': temperature,
                            **captured, 'text': text, 'complete': usage['local_complete'], 'usage': usage,
                            'wall_seconds': time.perf_counter() - tick,
                            'score': result_score(text, usage['local_complete'], usage, fixture, phase)}
                        rows.append(row)
                        log.write(json.dumps(row, ensure_ascii=False) + '\n'); log.flush()
                        save(out / 'progress.json', {'label': label, 'precision': mode, 'answers': len(rows), 'elapsed_seconds': time.perf_counter() - started})
                    if pointer_bytes(store) != before:
                        raise ValueError('production bindings changed during mode')
            holdout_skipped = bool(args.fresh_holdout) and not any(row['phase'] == 'holdout' for row in rows)
            completed_plan = [entry for entry in plan if not holdout_skipped or entry[0] != 'holdout']
            save(out / 'completed-cases.json', [{'phase': phase, 'fixture': fixture, 'temperature': temp}
                                               for phase, fixture, temp in completed_plan])
            result = {'completed': True, 'full_plan_completed': not holdout_skipped,
                      'final_holdout_skipped': holdout_skipped,
                      'final_holdout_skip_reason': 'exposed_regression_quality_retention_failed' if holdout_skipped else None,
                      'planned_samples_per_mode': len(plan), 'completed_samples_per_mode': len(completed_plan),
                      'results': summarize(rows, completed_plan, labels), 'loaded': loaded,
                      'elapsed_seconds': time.perf_counter() - started, 'product_pointers_unchanged': True,
                      'product_adopted': False, 'teacher_calls': 0}
            save(out / 'result.json', result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as error:
        save(out / 'failure.json', {'type': type(error).__name__, 'error': str(error), 'answers': len(rows),
                                  'elapsed_seconds': time.perf_counter() - started})
        raise
    finally:
        runtime.stop()


if __name__ == '__main__':
    main()
