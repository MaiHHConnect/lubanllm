"""Locate observed sparse-Q4 regressions; interventions do not qualify a product."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from tokenizers import Tokenizer

from benchmark_native_decode import IDENTITY, MODEL, save
from benchmark_native_mixed import MixedWorker, score
from intervene_q4_layers import request, pair_result
from luban.process_lock import ModelProcessLock
from trace_q4_divergence import first_difference

ROOT = Path('/Volumes/移动数据库/lubanllm/experiments/q4-native-protection-20261016')


def diagnostic_cases(result, answers, fixtures, eos):
    if result.get('completed') is not True:
        raise ValueError('complete product result required')
    losses = [(phase.rsplit('-t', 1)[0], identifier, float(phase.rsplit('-t', 1)[1]))
              for phase, group in result['results']['gptq-sequential']['quality'].items()
              for identifier in group['lost_baseline_correct']]
    if not losses or any(temp != 0 for _, _, temp in losses):
        raise ValueError('this fixed-prefix diagnosis requires nonempty greedy regressions')
    indexed = {}
    for row in answers:
        key = (row['label'], row['phase'], row['case_id'], row['temperature'])
        if key in indexed:
            raise ValueError('duplicate product observation')
        indexed[key] = row
    source = {(row['phase'], row['fixture']['id'], row['temperature']): row['fixture'] for row in fixtures}
    cases = []
    for phase, identifier, temperature in losses:
        key = (phase, identifier, temperature)
        base, quant = (indexed[(label, *key)] for label in ('bf16', 'gptq-sequential'))
        if (base['prompt_tokens'] != quant['prompt_tokens'] or not base['complete'] or not quant['complete']
                or base['execution_action'] != 'local_core' or quant['execution_action'] != 'local_core'):
            raise ValueError('diagnosis requires matching native prompts and complete outputs')
        fixture = source[key]
        if not score(base['text'], fixture, True)['passed'] or score(quant['text'], fixture, True)['passed']:
            raise ValueError('observed regression does not reproduce scoring')
        at = first_difference(base['tokens'], quant['tokens'])
        if at is None or at >= min(len(base['tokens']), len(quant['tokens'])):
            raise ValueError('first differing token required')
        forced = base['tokens'][:at]
        if any(t in eos for t in forced):
            raise ValueError('cannot force a completed prefix')
        cases.append({'id': identifier, 'phase': phase, 'fixture': fixture,
                      'case': {'tokens': base['prompt_tokens'], 'max_tokens': fixture['max_tokens'], 'eos_ids': eos},
                      'forced_tokens': forced, 'probe_tokens': [base['tokens'][at], quant['tokens'][at]],
                      'baseline': base, 'candidate': quant})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=ROOT/'sequential-product-v1')
    parser.add_argument('--conversion', type=Path, default=ROOT/'sequential-conversion-v2')
    parser.add_argument('--binary', type=Path, default=ROOT/'sequential-build/zig-out/bin/luban-worker')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not os.path.ismount('/Volumes/移动数据库') or not args.output.is_absolute() or not args.output.resolve().is_relative_to(ROOT):
        parser.error('new directory on mounted experiment volume required')
    converted = json.loads((args.conversion/'result.json').read_text())
    receipt = json.loads(Path(str(args.binary)+'.json').read_text())
    if (converted.get('completed') is not True or converted['receipt'] != receipt
            or converted['reload']['passed'] is not True or converted['mask_zero']):
        raise ValueError('same frozen worker and completed nonempty reloaded conversion required')
    result = json.loads((args.prior/'result.json').read_text())
    answers = [json.loads(line) for line in (args.prior/'answers.jsonl').read_text().splitlines()]
    fixtures = json.loads((args.prior/'cases.json').read_text())
    config = json.loads((Path(MODEL)/'config.json').read_text())
    eos = config.get('text_config', config)['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    cases = diagnostic_cases(result, answers, fixtures, eos)
    layers = converted['accepted_layers']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'traces').mkdir()
    protocol = {'schema': 'q4-sequential-regression-diagnosis-v1', 'registered_at': datetime.now(timezone.utc).isoformat(),
        'binary': str(args.binary), 'prior': str(args.prior), 'conversion': str(args.conversion), 'cases': cases,
        'layers': layers, 'projections': ['all', 'gate', 'up', 'down'], 'seed': 42, 'temperature': 0,
        'cache_reuse': False, 'core': None, 'teacher_calls': 0, 'training': False, 'product_adopted': False,
        'method': 'One resident packed model retaining original BF16 views; each request restores only nominated views and always returns to candidate. Scan every accepted layer/projection at first differing fixed prefix; free-generate each whole-layer restore. Never select a product from4 exposed failures.',
        'controls': 'Candidate full tokens before/after; restore all accepted layers reproduces BF16 full tokens; unquantized layer1 restore and final down no-op; final fixed-prefix probes repeat exactly.',
        'memory': 'Research retains original BF16 plus candidate; not deployment footprint'}
    save(args.output/'protocol.json', protocol)
    tokenizer = Tokenizer.from_file(str(Path(MODEL)/'tokenizer.json'))
    started = time.perf_counter()
    lock, worker = ModelProcessLock(IDENTITY, 'mps'), None
    rows, controls = [], []
    try:
        worker = MixedWorker(args.binary, 'candidate_mixed', args.output, lock, precision='mixed-q4-g16-packed',
            calibration_profile=args.conversion/'gptq-sequential.bin', trace_directory=args.output/'traces',
            intervention=True, core=None)
        if not worker.loaded.get('research_intervention') or worker.loaded.get('retained_original_weight_bytes', 0) <= 0:
            raise ValueError('bound original weights must be retained for interventions')
        save(args.output/'loaded.json', worker.loaded)

        def generation(spec, name, **kwargs):
            reply = request(worker, spec, name, **kwargs)
            text = tokenizer.decode(reply['tokens'], skip_special_tokens=True)
            return {'tokens': reply['tokens'], 'complete': reply['complete'], 'text': text,
                    'score': score(text, spec['fixture'], reply['complete'])}

        def compare(actual, expected):
            if any(actual[k] != expected[k] for k in ('tokens', 'complete', 'text')):
                raise ValueError('normal/full-restore output differs from frozen paired observation')

        for i, spec in enumerate(cases):
            normal = generation(spec, f'before-{i}'); compare(normal, spec['candidate'])
            restored = generation(spec, f'bf16-{i}', layers=layers); compare(restored, spec['baseline'])
            base = pair_result(request(worker, spec, f'probe-{i}', probe=spec['probe_tokens']))
            for layer, projection in ((1, 'all'), (23, 'down')):
                noop = pair_result(request(worker, spec, f'noop-{layer}-{i}', layer=layer,
                                            projection=projection, probe=spec['probe_tokens']))
                if noop != base:
                    raise ValueError('unquantized projection restoration changed logits')
            controls.append({'id': spec['id'], 'candidate_exact': True, 'bf16_restore_exact': True, 'probe': base})
        save(args.output/'controls.json', controls)
        with (args.output/'observations.jsonl').open('x') as log:
            for layer in layers:
                for projection in ('all', 'gate', 'up', 'down'):
                    if layer == 23 and projection == 'down':
                        continue
                    for i, spec in enumerate(cases):
                        probe = pair_result(request(worker, spec, f'l{layer}-{projection}-{i}', layer=layer,
                                                    projection=projection, probe=spec['probe_tokens']))
                        row = {'layer': layer, 'projection': projection, 'id': spec['id'], 'probe': probe}
                        if projection == 'all':
                            row['generation'] = generation(spec, f'free-l{layer}-{i}', layer=layer)
                        rows.append(row); log.write(json.dumps(row, ensure_ascii=False)+'\n'); log.flush()
                save(args.output/'progress.json', {'layer': layer, 'observations': len(rows), 'elapsed_seconds': time.perf_counter()-started})
                print(json.dumps({'layer': layer, 'observations': len(rows)}), flush=True)
        for i, spec in enumerate(cases):
            compare(generation(spec, f'after-{i}'), spec['candidate'])
            if pair_result(request(worker, spec, f'after-probe-{i}', probe=spec['probe_tokens'])) != controls[i]['probe']:
                raise ValueError('interventions changed subsequent candidate fixed-prefix logits')
        worker.close(); code = worker.process.returncode; worker = None
        if code:
            raise RuntimeError('diagnostic worker did not exit cleanly')
        summary = {spec['id']: {'whole_layer_answer_restored': [r['layer'] for r in rows if r['id'] == spec['id']
                    and r['projection'] == 'all' and r['generation']['score']['passed']],
                    'target_top1_restored': [{'layer': r['layer'], 'projection': r['projection'], 'margin': r['probe']['margin']}
                        for r in rows if r['id'] == spec['id'] and r['probe']['preferred_is_top1']]}
                   for spec in cases}
        save(args.output/'result.json', {'completed': True, 'summary': summary, 'controls': controls,
             'observations': len(rows), 'elapsed_seconds': time.perf_counter()-started, 'product_adopted': False,
             'scope': 'Exposed4-case causal interventions, not independent quality or universal root-cause proof'})
    except BaseException as error:
        save(args.output/'failure.json', {'error': str(error), 'type': type(error).__name__, 'observations': len(rows)})
        raise
    finally:
        if worker is not None:
            worker.close()
        lock.close()


if __name__ == '__main__':
    main()
