# SPDX-License-Identifier: Apache-2.0
"""Report precision, answer losses and measured native/HF performance honestly."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('existing_generation_checks', ROOT / 'src/luban/generation_checks.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def rows(directory):
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    return [r for r in events if r['event'] == 'result']


def correct(text, fixture, complete):
    if 'check' in fixture:
        return checks.evaluate_generation(text, fixture['check'], complete)['passed']
    if 'expected' in fixture:
        return checks.evaluate_generation(text, {'type': 'exact', 'expected': fixture['expected']}, complete)['passed']
    if 'scoring' not in fixture:
        return None
    value = fixture['answer']
    if not complete:
        return False
    if fixture['scoring'] == 'choice':
        return bool(re.fullmatch('[A-Ja-j]', text.strip())) and text.strip().upper() == str(value).upper()
    if fixture['scoring'] == 'exact_number':
        return bool(re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', text.strip())) and float(text.strip()) == float(value)
    return checks.evaluate_generation(text, {'type': 'semantic', 'validator': 'json', 'expected': value}, complete)['passed']


def performance(items):
    return {'median_ttft_seconds': statistics.median(r['ttft_ns'] / 1e9 for r in items),
            'median_first_visible_text_seconds': statistics.median(r['client_first_visible_text_ns'] / 1e9 for r in items if r.get('client_first_visible_text_ns') is not None),
            'median_decode_tokens_per_second': statistics.median((len(r['tokens']) - 1) * 1e9 / r['decode_ns']
                for r in items if r['decode_ns'] and len(r['tokens']) > 1),
            'peak_physical_footprint_bytes': max(r['memory']['lifetime_peak_physical_footprint_bytes'] for r in items),
            'cases': len(items)}


def logits(path):
    words = np.fromfile(path, np.uint16)
    return (words.astype(np.uint32) << 16).view(np.float32).reshape(-1, 248320)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    fixtures = json.loads(args.fixtures.read_bytes())
    native, reference = rows(args.native), rows(args.reference)
    if len(native) != len(fixtures) or len(reference) != len(fixtures):
        raise ValueError('Missing results; cannot report a complete comparison')
    report = {'scope': 'Isolated BF16 text comparison; finite fixtures are not universal capability proof',
              'native': performance(native), 'reference': performance(reference), 'rows': [], 'logits': []}
    for index, (f, a, b) in enumerate(zip(fixtures, native, reference, strict=True)):
        if a['id'] != f['id'] or b['id'] != f['id']:
            raise ValueError('Case order or binding mismatch')
        texts = [tokenizer.decode(r['tokens'], skip_special_tokens=True) for r in (a, b)]
        scores = [correct(text, f, r['complete']) for text, r in zip(texts, (a, b))]
        report['rows'].append({'id': f['id'], 'native_answer': texts[0], 'reference_answer': texts[1],
            'native_correct': scores[0], 'reference_correct': scores[1],
            'native_complete': a['complete'], 'reference_complete': b['complete'],
            'tokens_exact': a['tokens'] == b['tokens'], 'kind': f['kind']})
        if a['logits_dumped'] and b['logits_dumped']:
            x, y = logits(args.native / f'{index}.logits.bf16'), logits(args.reference / f'{index}.logits.bf16')
            if x.shape != y.shape:
                raise ValueError('Logit shape mismatch')
            delta = x - y
            report['logits'].append({'id': f['id'], 'shape': list(x.shape), 'finite': bool(np.isfinite(x).all() and np.isfinite(y).all()),
                'top1_agreement': float(np.mean(x.argmax(-1) == y.argmax(-1))),
                'max_abs': float(np.abs(delta).max()), 'rms': float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
                'exact_bf16_bytes': bool(np.array_equal(x, y))})
    scored = [r for r in report['rows'] if r['native_correct'] is not None]
    report['quality'] = {'scored': len(scored), 'native_correct': sum(r['native_correct'] for r in scored),
        'reference_correct': sum(r['reference_correct'] for r in scored),
        'lost_reference_correct_ids': [r['id'] for r in scored if r['reference_correct'] and not r['native_correct']],
        'gained_correct_ids': [r['id'] for r in scored if r['native_correct'] and not r['reference_correct']],
        'exact_token_cases': sum(r['tokens_exact'] for r in report['rows'])}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('native', 'reference', 'quality', 'logits')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
