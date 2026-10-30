"""Read a fixed local-study prefix and emit public aggregates, never raw material."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def mastered(pair):
    first, second = pair
    other = {row['id']: row for row in second}
    return {row['id'] for row in first if row['id'] in other
            and all(v['passed'] is True and v['complete'] is True
                    and v['reference_used'] is False for v in [row, other[row['id']]])
            and all(row[key] == other[row['id']][key] for key in ['tokens', 'input_tokens'])}


def instability(pair):
    first, second = pair
    assert [row['id'] for row in first] == [row['id'] for row in second]
    changed = [a for a, b in zip(first, second)
               if any(a[key] != b[key] for key in ['tokens', 'input_tokens', 'complete'])]
    return {'observed_records': len(first), 'inconsistent_records': len(changed),
            'by_source': dict(sorted(Counter(row['source'] for row in changed).items()))}


def summarize(evidence, database, count):
    state = json.loads((evidence / 'controller-state.json').read_text())
    attempts = state['attempts'][:count]
    assert len(attempts) == count and all(a['actual_steps'] > 0 for a in attempts)
    rounds = []
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        for number, attempt in enumerate(attempts, 1):
            terminal = json.loads(Path(attempt['report_path']).read_text())
            report = terminal['last_result']['report']
            journal_id = report.get('id') or terminal['last_result'].get('run_id')
            if journal_id:
                rows = connection.execute('SELECT state, body FROM training_runs WHERE id=?', (journal_id,)).fetchall()
            else:
                rows = [(status, body) for status, body in connection.execute('SELECT state, body FROM training_runs')
                        if json.loads(body).get('started', 0) >= attempt['started'] - 1
                        and json.loads(body).get('started', 0) <= attempt['finished']]
            assert len(rows) == 1, 'Study attempt must resolve to exactly one training journal'
            status, raw = rows[0]
            body = json.loads(raw)
            baseline = body['baseline']['core']
            candidate = body['last_validation']['observations']
            before, after = mastered(baseline), mastered(candidate)
            records = body['records']
            progress = body['last_validation']['decision']['learning_progress']
            candidates = body['candidates']
            actual_steps = sum(c['actual_steps'] for c in candidates)
            assert actual_steps == attempt['actual_steps']
            assert (status == 'committed') == attempt['adopted']
            rounds.append({'round': number, 'application': '0.3.208', 'build': 216,
                'adam_updates': actual_steps, 'adopted': attempt['adopted'],
                'outcome': attempt['phase'], 'submission_to_lease_release_seconds': attempt['elapsed_seconds'],
                'engine_reported_seconds': attempt['engine_elapsed_seconds'],
                'training_count': len(records['train']), 'validation_count': len(records['valid']),
                'validation_before_correct': sum(r['id'] in before for r in records['valid']),
                'last_candidate_validation_correct': sum(r['id'] in after for r in records['valid']),
                'historical_protection_count': len(records['protection']),
                'historical_protection_before_correct': sum(r['id'] in before for r in records['protection']),
                'last_candidate_protection_correct': sum(r['id'] in after for r in records['protection']),
                'gained': len(progress['gained_ids']), 'forgotten': len(progress['forgotten_ids']),
                'net_gain': progress['net_gain'],
                'baseline_validation_loss': body['baseline']['validation']['mean'],
                'last_candidate_validation_loss': candidates[-1]['validation_loss'],
                'baseline_instability': instability(baseline), 'last_candidate_instability': instability(candidate),
                'candidates': [{k: c[k] for k in ['width', 'initialization', 'actual_steps', 'accepted', 'gates']}
                               for c in candidates]})
    return {'id': 'E4', 'date': '2026-10-22', 'status': 'ongoing-study-fixed-three-round-prefix',
        'scheduled_actual_training_rounds': 20, 'reported_actual_training_rounds': count,
        'accepted_rounds': sum(r['adopted'] for r in rounds),
        'adam_updates': sum(r['adam_updates'] for r in rounds),
        'hardware_model': 'Mac16,10', 'ram_bytes': 17179869184, 'os': 'macOS 26.4',
        'model': 'Qwen3.5-2B-derived MLX 4bit with frozen accepted core',
        'expert_precision': 'BF16 weights; FP32 Adam moments and update arithmetic',
        'seed': 42, 'learning_rate': 0.0002, 'preservation_weight': 5,
        'candidate_update_limit': 64, 'validation_interval_updates': 8,
        'decoding': {'temperature': 0, 'seed': 42, 'thinking': 'off', 'cache_reuse': False,
                     'new_material_max_tokens': 192, 'some_old_protection_max_tokens': 64},
        'peak_memory_bytes': None, 'memory_note': 'No all-process physical or Metal peak sampler',
        'facts': ['minimum macOS 26', 'Apple Silicon', 'Luban.app filename', 'brand 麦好火'],
        'validation_role': 'Repeated development validation on paraphrases of the same fact families; no transfer test',
        'rounds': rounds,
        'teacher': {'unexpected_preparation_calls': 2, 'provider_reported_tokens': 1958,
                    'controlled_training_calls': 0, 'charge': None, 'charge_note': 'Unknown; both preparation outputs incomplete'},
        'interpretation': 'Rejected last-candidate gains are not committed capabilities; instability root cause unresolved',
        'continuation': 'Existing-data repetition under application 0.3.218 is a separate protocol phase, not this fixed prefix'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--rounds', type=int, default=3, choices=[3])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.evidence, args.database, args.rounds)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'rounds': result['reported_actual_training_rounds'],
                      'adam_updates': result['adam_updates'], 'accepted': result['accepted_rounds']}))
