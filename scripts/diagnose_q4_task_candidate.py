"""Diagnose four exposed failures with a rejected fixed Q4 candidate; never admit it."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from transformers import AutoTokenizer

from benchmark_native_decode import IDENTITY, MANIFEST, MODEL, save
from benchmark_native_mixed import MixedWorker, score
from capture_q4_native import integers
from diagnose_q4_sequential import ROOT, diagnostic_cases
from luban.process_lock import ModelProcessLock
from prepare_q4_sequential import protection_generations, sparse_payloads
from q4_packed_format import write_sparse
from q4_sequential_protection import GENERATION, check_proposal, protection_cases

LAYERS = [0, 2, 6, 9, 14, 15, 16, 18, 20, 21, 23]
IDS = {'mmlu-2804', 'preservation-replay-train-olive-path',
       'decode-holdout-unit-volume', 'seq-final-young-dog'}


def read(path):
    return json.loads(path.read_text())


def rejected_source(directory):
    failure, protocol = read(directory/'failure.json'), read(directory/'protocol.json')
    if ((directory/'result.json').exists() or (directory/'gptq-sequential.bin').exists()
            or failure.get('completed_layers') != 24
            or failure.get('error') != 'fixed-mask candidate lost protected TRAIN answers; reject whole candidate'
            or protocol.get('mode') != 'convert' or protocol.get('fixed_layers') != LAYERS
            or protocol.get('protection_cases') != protection_cases() or protocol.get('generation') != GENERATION):
        raise ValueError('complete fixed-mask rejected source required; no successful conversion claim')
    audit = read(directory/'independent-readback.json')
    if audit.get('audit_passed') is not True or audit.get('candidate_accepted') is not False:
        raise ValueError('independent rejected-source audit required')
    final = read(directory/'fixed-final-generations.json')
    baseline = read(directory/'protection-baseline.json')
    previous = read(Path(protocol['refit_grid_from'])/'accepted-generations.json')
    computed = check_proposal(protection_cases(), baseline, previous, final)
    recorded = read(directory/'fixed-protection.json')
    if (computed['accepted'] or computed['lost_required_correct_ids'] != ['seq-train-copy-en']
            or any(recorded.get(key) != value for key, value in computed.items()
                   if key not in {'decision', 'rollback'})):
        raise ValueError('original TRAIN union rejection or scoring changed')
    if any(any(a[key] != b[key] for key in ('id', 'prompt_tokens', 'tokens', 'text', 'complete'))
           for a, b in zip(baseline['generations'], final, strict=True)):
        raise ValueError('expected all sixteen final trajectories to reproduce original BF16')
    return protocol, final, recorded


def exact_generations(actual, expected):
    if len(actual) != len(expected) or any(any(a[k] != b[k] for k in ('id', 'prompt_tokens', 'tokens', 'text', 'complete'))
                                          for a, b in zip(actual, expected, strict=True)):
        raise ValueError('ordinary reload differs from recorded rejected final trajectories')


def ordinary_loaded(loaded, precision):
    if (loaded.get('inference_precision') != precision or loaded.get('base_identity') != IDENTITY
            or loaded.get('manifest_sha256') != MANIFEST or loaded.get('core_loaded') is not False
            or loaded.get('retained_original_weight_bytes') != 0
            or any(loaded.get(flag) is not False for flag in
                   ('research_sequential', 'research_capture_inputs', 'research_intervention'))
            or loaded.get('quantized_projections') != (0 if precision == 'bf16' else 32)):
        raise ValueError('ordinary bound worker without research modes or retained original weights required')


def diagnostic_result(spec, raw, tokenizer, vocab):
    case, tokens = spec['case'], raw.get('tokens')
    if (not tokens or not integers(tokens, 0, vocab) or len(tokens) > case['max_tokens']
            or raw.get('cached_prompt_tokens') != 0 or raw.get('prompt_tokens') != len(case['tokens'])
            or raw.get('generated_tokens') != len(tokens) or any(t in case['eos_ids'] for t in tokens[:-1])):
        raise ValueError('diagnostic generation scope, tokens or lengths changed')
    stopped = tokens[-1] in case['eos_ids']
    if (raw.get('complete') is not stopped or raw.get('finish_reason') != ('stop' if stopped else 'length')
            or (not stopped and len(tokens) != case['max_tokens'])):
        raise ValueError('diagnostic completion must match natural EOS or full original budget')
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {'id': spec['id'], 'prompt_tokens': case['tokens'], 'tokens': tokens, 'text': text,
            'complete': stopped, 'score': score(text, spec['fixture'], stopped), 'native_result': raw}


def prepare(directory, conversion, prior):
    protocol, final, rejection = rejected_source(conversion)
    binary = Path(protocol['binary'])
    if read(Path(str(binary)+'.json')) != protocol['receipt']:
        raise ValueError('same conversion worker receipt required')
    source_protocol = read(Path(protocol['source'])/'protocol.json')
    cases = diagnostic_cases(read(prior/'result.json'),
        [json.loads(line) for line in (prior/'answers.jsonl').read_text().splitlines()],
        read(prior/'cases.json'), protocol['plan']['eos_ids'])
    if len(cases) != 4 or {case['id'] for case in cases} != IDS:
        raise ValueError('exact four already exposed failures required')
    directory.mkdir()
    save(directory/'protocol.json', {'schema': 'q4-task-rejected-diagnostic-v1',
        'registered_at': datetime.now(timezone.utc).isoformat(), 'conversion': str(conversion), 'prior': str(prior),
        'binary': str(binary), 'receipt': protocol['receipt'], 'eos_ids': protocol['plan']['eos_ids'],
        'vocab_size': source_protocol['vocab_size'], 'cases': cases, 'layers': LAYERS,
        'seed': 42, 'temperature': 0, 'cache_reuse': False, 'teacher_calls': 0, 'product_adopted': False,
        'candidate_rejected': True, 'original_union_rejection': rejection,
        'scope': 'Diagnostic only. Ordinary packed reload16, fresh BF16 four versus historical exact tokens, then candidate four. No complete regression or unseen holdout. No training or tuning.'})
    save(directory/'expected-final.json', final)
    artifact = write_sparse(directory/'rejected-candidate.bin', MANIFEST, '0'*64,
                            sum(1 << layer for layer in LAYERS), sparse_payloads(conversion/'weights', LAYERS))
    save(directory/'prepared.json', {'prepared': True, 'artifact': artifact, 'candidate_rejected': True,
                                    'gpu_used': False, 'product_adopted': False})


def run(directory):
    protocol, prepared = read(directory/'protocol.json'), read(directory/'prepared.json')
    if (prepared.get('prepared') is not True or protocol.get('candidate_rejected') is not True
            or prepared['artifact']['quantized_projections'] != 32 or (directory/'result.json').exists()):
        raise ValueError('new prepared rejected-candidate diagnosis required')
    binary = Path(protocol['binary'])
    if read(Path(str(binary)+'.json')) != protocol['receipt']:
        raise ValueError('worker receipt changed')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    started = time.perf_counter()
    rows, loaded = [], {}
    lock, worker = ModelProcessLock(IDENTITY, 'mps'), None
    try:
        for phase, precision in [('reload16', 'mixed-q4-g16-packed'), ('bf16', 'bf16'), ('candidate', 'mixed-q4-g16-packed')]:
            stage = directory/phase; stage.mkdir()
            worker = MixedWorker(binary, 'baseline_bf16' if precision == 'bf16' else 'candidate_mixed', stage,
                lock, precision=precision, calibration_profile=directory/'rejected-candidate.bin', core=None)
            ordinary_loaded(worker.loaded, precision)
            loaded[phase] = worker.loaded; save(directory/'loaded.json', loaded)
            if phase == 'reload16':
                actual = protection_generations(worker, protection_cases(), tokenizer, protocol['eos_ids'],
                                                 'diagnostic-reload', protocol['vocab_size'])
                exact_generations(actual, read(directory/'expected-final.json'))
                save(stage/'control.json', {'passed': True, 'cases': 16, 'candidate_rejected': True})
            else:
                for spec in protocol['cases']:
                    case = spec['case']
                    raw = worker.request(case['tokens'], phase+'-'+spec['id'], max_tokens=case['max_tokens'],
                                         temperature=0, seed=42, reuse=False, eos_ids=case['eos_ids'])
                    row = {'phase': phase, **diagnostic_result(spec, raw, tokenizer, protocol['vocab_size'])}
                    rows.append(row)
                    save(directory/'answers.json', rows)
                    if phase == 'bf16' and any(row[k] != spec['baseline'][k] for k in ('tokens', 'text', 'complete')):
                        raise ValueError('fresh BF16 does not reproduce the old four observations')
            worker.close(); code = worker.process.returncode; worker = None
            if code:
                raise RuntimeError('ordinary diagnostic worker failed')
        candidate = [row for row in rows if row['phase'] == 'candidate']
        save(directory/'result.json', {'completed': True, 'diagnostic_only': True,
            'candidate_rejected': True, 'product_adopted': False, 'teacher_calls': 0,
            'reload16_exact': True, 'bf16_four_exact': True, 'observations': rows,
            'correct_known_failures': [r['id'] for r in candidate if r['score']['passed']],
            'remaining_known_failures': [r['id'] for r in candidate if not r['score']['passed']],
            'lost_previous_q4_only_train_correct': protocol['original_union_rejection']['lost_required_correct_ids'],
            'original_union_rejection_unchanged': True, 'elapsed_seconds': time.perf_counter()-started,
            'scope': 'Four exposed failures only; no unseen holdout or complete product-quality admission.'})
    except BaseException as error:
        save(directory/'diagnostic-failure.json', {'type': type(error).__name__, 'error': str(error), 'rows': len(rows)})
        raise
    finally:
        if worker is not None:
            worker.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=ROOT/'task-calibration-diagnostic-v1')
    parser.add_argument('--conversion', type=Path, default=ROOT/'task-calibration-conversion-v1')
    parser.add_argument('--prior', type=Path, default=ROOT/'sequential-product-v1')
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.is_relative_to(ROOT):
        parser.error('mounted new experiment directory required')
    if args.stage == 'prepare':
        prepare(args.output, args.conversion, args.prior)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
