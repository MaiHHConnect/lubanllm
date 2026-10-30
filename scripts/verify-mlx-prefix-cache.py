#!/usr/bin/env python3
"""Read-only real-model prefix-cache equivalence and latency qualification.

Run only after releasing the application's model lease. No teacher calls,
optimizer updates, production pointers, or cache files are written. The factory
is an existing isolated extraction from qualify_factory_mlx_engine.py.
"""
import argparse
import json
from pathlib import Path
import platform
import threading
import time
import traceback
from types import SimpleNamespace

from luban.native_engine import NativeEngine
from luban.native_mlx import model_binding
from luban.process_lock import ModelProcessLock
from verify_direct_q4 import IDENTITY, MANIFEST


ROOT = Path('/Volumes/移动数据库/lubanllm')
METADATA = ROOT / 'models/luban0-mlx-metadata-20261020'
MODEL = ROOT / 'models/community-q4-study-20261017/mlx-community-4bit'
EOS = [248044, 248046]


def read(path):
    return json.loads(path.read_text())


def stat(path):
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def token_list(value):
    if not isinstance(value, list):
        value = value['input_ids']
    assert value and all(type(token) is int and 0 <= token < 248320 for token in value)
    return value


def make_prompt(tokenizer, length):
    """Exact-length synthetic text; contains no private conversation history."""
    marker = '__LUBAN_CACHE_FILLER__'
    rendered = tokenizer.apply_chat_template([
        {'role': 'system', 'content': '阅读用户材料。只输出最后一道算术题的数字答案。'},
        {'role': 'user', 'content': marker + '\n材料结束。2+3等于多少？只回答数字。'},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert rendered.count(marker) == 1
    left, right = rendered.split(marker)
    prefix = tokenizer.encode(left, add_special_tokens=False)
    suffix = tokenizer.encode(right, add_special_tokens=False)
    unit = tokenizer.encode('这是一段固定的合成背景材料，没有额外问题。\n', add_special_tokens=False)
    remaining = length - len(prefix) - len(suffix)
    assert remaining > 0 and unit
    filler = (unit * ((remaining + len(unit) - 1) // len(unit)))[:remaining]
    return token_list(prefix + filler + suffix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--factory', type=Path, required=True)
    parser.add_argument('--long-tokens', type=int, default=8192)
    parser.add_argument('--prefill-step', type=int, choices=(32, 64, 128, 256, 512), default=128)
    parser.add_argument('--reservation-long-checks', action='store_true', help='long-prefix reuse and real 2GiB quota checks')
    parser.add_argument('--only-cancel', action='store_true', help='run only cancellation/recovery checks')
    parser.add_argument('--reservation-checks', action='store_true',
                        help='run only 18 KV reservation/growth/cancellation checks')
    args = parser.parse_args()
    if sum((args.only_cancel, args.reservation_checks, args.reservation_long_checks)) > 1:
        parser.error('select only one cancellation/reservation check group')
    if not Path('/Volumes/移动数据库').is_mount():
        parser.error('research volume is not mounted')
    directory = args.directory.resolve()
    if directory.exists() or not directory.is_relative_to(ROOT / 'experiments'):
        parser.error('--directory must be a new external experiments directory')
    if not 512 <= args.long_tokens <= 250000:
        parser.error('--long-tokens must be between 512 and 250000')
    factory = args.factory.resolve()
    proof_path = factory / 'runtime/mlx/expert-proof.json'
    if not proof_path.is_file() or not args.binary.is_file():
        parser.error('factory proof or worker binary is missing')
    directory.mkdir(mode=0o700)
    result = {'schema': 'mlx-prefix-cache-real-model-v1', 'complete': False,
              'hardware': platform.platform(), 'precision': 'MLX affine Q4/group64; factory BF16 core/expert',
              'scope': 'synthetic exact-token-prefix equivalence; not new semantic quality or full Agent qualification',
              'teacher_calls': 0, 'optimizer_steps': 0, 'adopted': False,
              'seed': 42, 'prefill_step': args.prefill_step, 'long_prompt_tokens': args.long_tokens,
              'only_cancellation': args.only_cancel,
              'reservation_checks': args.reservation_checks,
              'reservation_long_checks': args.reservation_long_checks,
              'file_cache': 'not purged; one resident worker; no cross-process cache reuse',
              'memory_measurement': 'Darwin proc_pid_rusage v4 worker samples; excludes Python and application',
              'cases': [], 'checks': {}, 'memory_samples': []}
    engine = lease = None
    sample_stop = threading.Event()
    sampler = None
    started = time.monotonic()
    frozen_files, frozen_stats = {}, {}
    try:
        proof = read(proof_path)
        store = factory / 'modules/luban0-v1'
        core = {'file': str(store / (proof['core_version'] + '.safetensors')),
                'tensor_key': 'model.language_model.layers.23.mlp.base.down_proj.weight',
                'sha256': proof['core_file_sha256']}
        expert = {'id': proof['expert_id'],
                  'file': str(store / 'layer-23' / (proof['expert_id'] + '.safetensors')),
                  'sha256': proof['expert_file_sha256'], 'width': proof['width'],
                  'core_sha256': core['sha256']}
        model = model_binding(SimpleNamespace(native_inference_precision='mlx-q4', native_mlx_model_path=str(MODEL)))
        frozen_files = {str(path): path.read_bytes() for path in [proof_path] + [
            store / name for name in ('active.json', 'active-core.json', 'active-router.json')]}
        frozen_stats = {str(path): stat(path) for path in [MODEL / 'model.safetensors',
                        Path(core['file']), Path(expert['file'])]}
        result.update(factory=str(factory), source_proof=proof, core=core, expert=expert,
                      model=model, binary=str(args.binary.resolve()))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(METADATA), local_files_only=True, trust_remote_code=False)
        base = make_prompt(tokenizer, args.long_tokens)
        short = make_prompt(tokenizer, 640)
        write(directory / 'inputs.json', {'base': base, 'short': short})
        lease = ModelProcessLock(IDENTITY, 'mps')
        engine = NativeEngine(args.binary, METADATA, MANIFEST, core=core, expert=expert,
                              precision='mlx-q4', mlx_model=model, request_timeout=600)
        result['loaded'] = engine.start()
        assert result['loaded']['mlx_training'] is False
        assert result['loaded']['execution_skip_mask'] == 0 and result['loaded']['execution_exit_after'] == 24
        assert result['loaded']['layer_paging_enabled'] is False

        def sample():
            while not sample_stop.is_set():
                result['memory_samples'].append({'seconds': time.monotonic() - started, **engine.status()})
                sample_stop.wait(.25)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()

        def run(name, tokens, *, reuse=True, maximum=12, temperature=0, seed=42, eos=None, hit=None,
                cancel=None, prefill_step=None):
            prefill_step = args.prefill_step if prefill_step is None else prefill_step
            row = {'name': name, 'input_tokens': tokens, 'cache_reuse': reuse,
                   'prefill_step': prefill_step,
                   'maximum': maximum, 'temperature': temperature, 'seed': seed,
                   'eos_ids': EOS if eos is None else eos, 'cancellation': cancel, 'complete': False}
            result['cases'].append(row)
            events = []
            begin = time.monotonic()
            stop = threading.Event()
            timer = threading.Timer(.05, stop.set) if cancel == 'prefill' else None

            def token_event(token):
                events.append({'token': token, 'seconds': time.monotonic() - begin})
                if cancel == 'decode':
                    stop.set()

            try:
                if timer is not None:
                    timer.start()
                answer = engine.generate(tokens, maximum, row['eos_ids'], temperature,
                    token_event, stop, seed=seed, cache_reuse=reuse,
                    context_limit=262144, prefill_step=prefill_step)
                row.update(result=answer, events=events, seconds=time.monotonic() - begin,
                           memory=engine.status(), text=tokenizer.decode(answer['tokens'], skip_special_tokens=True))
                assert [event['token'] for event in events] == answer['tokens'], name + ': stream differs'
                if cancel is None:
                    assert answer['tokens'] and answer['finish_reason'] in ('length', 'stop'), name + ': incomplete run'
                else:
                    assert stop.is_set() and answer['finish_reason'] == 'cancelled', name + ': cancellation not acknowledged'
                    assert answer.get('cache_capacity', 0) == 0, name + ': cancelled cache survived'
                    assert not engine._prefix_tokens, name + ': cancelled local ledger survived'
                    if cancel == 'prefill':
                        assert not events, name + ': cancellation happened after prefill'
                    else:
                        assert events, name + ': decode callback did not run'
                cached = answer.get('cached_prompt_tokens', 0)
                assert type(cached) is int and 0 <= cached < len(tokens), name + ': invalid prefix count'
                if hit is not None:
                    assert (cached > 0) == hit, name + ': unexpected cache reuse'
                if not reuse:
                    assert cached == 0, name + ': no_reuse hit cache'
                row['complete'] = True
                print(json.dumps({'case': name, 'tokens': len(tokens), 'cached': cached,
                                  'seconds': row['seconds'], 'finish_reason': answer['finish_reason']}, ensure_ascii=False), flush=True)
                return row
            except BaseException as error:
                row.update(error=repr(error), seconds=time.monotonic() - begin)
                raise
            finally:
                if timer is not None:
                    timer.cancel()
                    timer.join(timeout=1)
                write(directory / (f'{len(result["cases"]):02d}-' + name + '.json'), row)

        def equal(name, warm, cold):
            checks = {'tokens_equal': warm['result']['tokens'] == cold['result']['tokens'],
                      'finish_reason_equal': warm['result']['finish_reason'] == cold['result']['finish_reason']}
            result['checks'][name] = checks
            assert all(checks.values()), name + ': warm/cold output differs'

        def cancellation_checks():
            cold = run('cancel-reference-cold', short, reuse=False, hit=False)
            run('decode-cancel-fill', short)
            # Exclude natural EOS so the small answer cannot race cancellation.
            run('decode-cancel', short, maximum=128, eos=[0], hit=True, cancel='decode')
            equal('decode_cancel_recovery', run('decode-cancel-recovery', short, hit=False), cold)
            prefill_prompt = make_prompt(tokenizer, 8192)
            run('prefill-cancel', prefill_prompt, maximum=128, eos=[0], cancel='prefill')
            equal('prefill_cancel_recovery', run('prefill-cancel-recovery', short, hit=False), cold)

        def finish():
            assert all(path_value == Path(path).read_bytes() for path, path_value in frozen_files.items())
            assert all(value == stat(Path(path)) for path, value in frozen_stats.items())
            result.update(complete=True, sources_unchanged=True,
                          case_count=len(result['cases']), generation_count=len(result['cases']))

        if args.only_cancel:
            cancellation_checks()
            finish()
            return

        if args.reservation_long_checks:
            assert args.long_tokens >= 158701 and result['loaded'].get('mlx_kv_reservation') is True
            reference = run('long-reserved-fill', base, maximum=16, hit=False)
            equal('long_reserved_repeat', run('long-reserved-repeat', base, maximum=16, hit=True), reference)
            prompt = base + reference['result']['tokens'] + tokenizer.encode('\n只回答2+3的数字结果。', add_special_tokens=False)
            appended = run('long-reserved-append', prompt, maximum=16, hit=True)
            equal('long_reserved_append', appended, run('long-reserved-append-cold', prompt, reuse=False, maximum=16, hit=False))
            # Actual full capacity exceeds the unchanged 2GiB prefix-slot quota.
            # Generation must finish while the retained slot and ledger are empty.
            quota = run('reservation-real-quota', make_prompt(tokenizer, 176000), maximum=16, hit=False)
            assert quota['text'].strip() == '5' and quota['result']['finish_reason'] == 'stop'
            assert quota['result']['cache_capacity'] == quota['result']['prefix_cache_bytes'] == 0
            short_reference = run('quota-short-reference', short, reuse=False, hit=False)
            equal('quota_recovery', run('quota-short-recovery', short, hit=False), short_reference)
            finish()
            return

        if args.reservation_checks:
            assert result['loaded'].get('mlx_kv_reservation') is True, 'worker has no KV reservation support'
            # This exact 640-token prompt leaves a short suffix at every supported
            # step. Repeated requests therefore reserve more capacity without a
            # new capture: the existing checkpoint and its byte ledger must survive.
            filled = run('reserve-short-fill', short, hit=False)
            checkpoint = filled['result']['cache_capacity']
            initial_bytes = filled['result']['prefix_cache_bytes']
            assert checkpoint > 0 and type(initial_bytes) is int and initial_bytes > 0
            grown = []
            for maximum in (256, 512):
                row = run(f'reserve-no-capture-{maximum}', short, maximum=maximum, eos=[0], hit=True)
                answer = row['result']
                assert answer['cached_prompt_tokens'] == checkpoint, 'saved boundary changed before reserve'
                assert answer['cache_capacity'] == checkpoint, 'short suffix unexpectedly captured'
                assert len(engine._prefix_tokens) == checkpoint, 'Python checkpoint ledger changed'
                assert type(answer['prefix_cache_bytes']) is int, 'missing capacity byte ledger'
                assert engine._prefix_bytes == answer['prefix_cache_bytes'], 'Python capacity byte ledger differs'
                assert len(answer['tokens']) == maximum and answer['finish_reason'] == 'length', 'long decode ended early'
                grown.append(row)
            first_delta = grown[0]['result']['prefix_cache_bytes'] - initial_bytes
            second_delta = grown[1]['result']['prefix_cache_bytes'] - grown[0]['result']['prefix_cache_bytes']
            byte_checks = {'checkpoint_unchanged_without_capture': all(
                               row['result']['cache_capacity'] == checkpoint for row in grown),
                           'capacity_bytes_increased': first_delta > 0 and second_delta > 0,
                           'capacity_bytes_follow_reserved_tokens': first_delta * (512 - 256) == second_delta * (256 - 12)}
            result['checks']['reservation_no_capture_bytes'] = byte_checks
            assert all(byte_checks.values()), 'reserved capacity bytes were stale or incorrectly counted'
            equal('reservation_same_prompt_512', grown[1],
                  run('reserve-same-prompt-cold-512', short, reuse=False, maximum=512, eos=[0], hit=False))

            first = run('reserve-segment-fill-256', short, maximum=256, eos=[0], hit=False)
            equal('reservation_same_prompt_256', grown[0], first)
            segments = [first]
            for number, maximum in ((1, 256), (2, 512)):
                previous = segments[-1]
                prompt = previous['input_tokens'] + previous['result']['tokens']
                row = run(f'reserve-append-segment-{number}', prompt, maximum=maximum, eos=[0], hit=True)
                assert len(row['result']['tokens']) == maximum and row['result']['finish_reason'] == 'length'
                assert row['result']['prefix_cache_bytes'] > previous['result']['prefix_cache_bytes'], 'append did not grow capacity ledger'
                segments.append(row)
            for number in (2, 1):
                row = segments[number]
                equal(f'reservation_append_segment_{number}', row,
                      run(f'reserve-segment-{number}-cold', row['input_tokens'], reuse=False,
                          maximum=row['maximum'], eos=[0], hit=False))

            run('reserve-before-shortening', short, hit=False)
            shortened = run('reserve-shortened-miss', short[:64], hit=False)
            equal('reservation_shortening', shortened,
                  run('reserve-shortened-cold', short[:64], reuse=False, hit=False))
            cancellation_checks()
            finish()
            return

        # Finish the warm chain before cold controls, because no_reuse is allowed
        # to evict the cache slot. The input includes the actual preceding output.
        rounds = [run('round1-cold', base, hit=False)]
        for number in (2, 3):
            suffix = tokenizer.encode('\n<|im_start|>user\n仍然只回答2+3的数字结果。<|im_end|>\n'
                                      '<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
            prompt = rounds[-1]['input_tokens'] + rounds[-1]['result']['tokens'] + suffix
            rounds.append(run(f'round{number}-warm', prompt, hit=True))
        repeated = run('same-prompt-warm', rounds[-1]['input_tokens'], hit=True)
        equal('same_prompt', repeated, rounds[-1])
        for index, warm in enumerate(rounds, 1):
            equal(f'round{index}', warm, run(f'round{index}-cold-control', warm['input_tokens'], reuse=False, hit=False))

        changed = list(base)
        changed[0] = 1 if changed[0] != 1 else 2
        run('seed-before-prefix-change', base)
        changed_warm = run('changed-prefix', changed, hit=False)
        equal('changed_prefix', changed_warm, run('changed-prefix-cold', changed, reuse=False, hit=False))
        run('seed-before-shortening', base)
        shortened = base[:64]
        shortened_warm = run('shortened-prefix', shortened, hit=False)
        equal('shortened_prefix', shortened_warm, run('shortened-prefix-cold', shortened, reuse=False, hit=False))
        run('seed-before-no-reuse', short)
        run('no-reuse', short, reuse=False, hit=False)

        # A fresh iterator must seed its own sampler. Caches must not preserve a
        # preceding request's random stream, temperature, or sampled lookahead.
        for seed in (42, 947):
            for temperature in (0, .7):
                label = f'seed{seed}-temp{temperature}'
                cold = run(label + '-cold', short, reuse=False, seed=seed, temperature=temperature, hit=False)
                run(label + '-fill', short, seed=seed, temperature=temperature)
                warm = run(label + '-warm', short, seed=seed, temperature=temperature, hit=True)
                equal(label, warm, cold)

        one = run('max1-cold', short, reuse=False, maximum=1, hit=False)
        assert len(one['result']['tokens']) == 1
        run('max1-fill', short, maximum=1)
        equal('max1', run('max1-warm', short, maximum=1, hit=True), one)
        # Use the proven first sample as an EOS id so this does not depend on
        # the model choosing a natural end token within a finite short budget.
        forced_eos = [one['result']['tokens'][0]]
        eos_warm = run('eos-max1-warm', short, maximum=1, eos=forced_eos, hit=True)
        assert eos_warm['result']['finish_reason'] == 'stop' and eos_warm['result'].get('complete') is True
        eos_cold = run('eos-max1-cold', short, reuse=False, maximum=1, eos=forced_eos, hit=False)
        equal('eos_max1', eos_warm, eos_cold)
        # A change of chunk policy must evict the recurrent/attention snapshot;
        # subsequent requests under that same policy may establish a new hit.
        previous_step = None
        for index, step in enumerate((args.prefill_step, 128, 32, 64, 32, 128, args.prefill_step)):
            if step == previous_step:
                continue
            label = f'step-binding-{index}-{step}'
            cold = run(label + '-miss', short, prefill_step=step, hit=False)
            warm = run(label + '-hit', short, prefill_step=step, hit=True)
            equal(label, warm, cold)
            previous_step = step
        cancellation_checks()
        finish()
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        sample_stop.set()
        if sampler is not None:
            sampler.join(timeout=2)
        if engine is not None:
            result['final_worker_status'] = engine.status()
            engine.close()
        if lease is not None:
            lease.close()
        result['seconds'] = time.monotonic() - started
        result['source_stats_unchanged'] = all(Path(path).is_file() and value == stat(Path(path))
                                               for path, value in frozen_stats.items())
        write(directory / 'result.json', result)


if __name__ == '__main__':
    main()
