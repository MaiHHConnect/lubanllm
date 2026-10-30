"""Frozen TRAIN generation protection for sequential Q4 proposals; pure CPU checks."""
from __future__ import annotations

from copy import deepcopy

from luban.generation_checks import evaluate_generation, is_finite_generation_check

SCHEMA = 'q4-sequential-train-protection-v1'
GENERATION = {'temperature': 0, 'seed': 42, 'max_tokens': 192, 'cache_reuse': False}
GROUPS = ('characters', 'numbers_units', 'json', 'long_context')


def protection_cases():
    """Sixteen authored TRAIN checks; no external regression or final holdout inputs."""
    specs = [
        ('characters', 'copy-zh', '逐字复制方括号内的内容，只输出内容，不含方括号：[石径·暮钟_6]', '石径·暮钟_6'),
        ('characters', 'copy-en', 'Copy exactly the text between the brackets. Output only that text: [pR7-zT]', 'pR7-zT'),
        ('characters', 'remove-mark', 'Remove every # from ab##Q#4#de. Keep every other character unchanged. Output only the result.', 'abQ4de'),
        ('characters', 'replace-letter', '把字符串 x_2xx-M 中每个小写 x 替换成大写 K，其余字符不变。只输出结果。', 'K_2KK-M'),
        ('numbers_units', 'remaining', '有7盒卡片，每盒8张，取走5张。剩下多少张？只输出整数。', '51'),
        ('numbers_units', 'signed-sum', 'Calculate -14 + 9 + 12. Reply with the integer only.', '7'),
        ('numbers_units', 'mass', '一袋材料重3千克450克，合计多少克？只输出整数，不写单位。', '3450'),
        ('numbers_units', 'area', 'Use 1 square metre = 10000 square centimetres. Add 2 square metres and 3500 square centimetres. Reply only with the total number of square centimetres.', '23500'),
        ('json', 'object', 'Return only a JSON object with tag equal to "saffron", enabled equal to false, and count equal to 4.', {'tag': 'saffron', 'enabled': False, 'count': 4}),
        ('json', 'sort', '把JSON数组[8,3,6]按从大到小排序。只输出JSON数组。', [8, 6, 3]),
        ('json', 'nested', 'Read {"rack":{"cells":[{"k":"R4","n":2},{"k":"T9","n":5}]}}. Return a JSON object mapping each k to its n; no explanation.', {'R4': 2, 'T9': 5}),
        ('json', 'null-boolean', '只输出JSON对象：外层键为state，值为对象；该对象的ready为true，error为null。', {'state': {'ready': True, 'error': None}}),
    ]
    # Deterministic long records, authored independently of any model outputs.
    ledger = '\n'.join(f'箱号 C{i:03d}：登记值 {700+i}' for i in range(72))
    specs.append(('long_context', 'ledger-middle', '阅读完整登记表，找到箱号C037的登记值。只输出该整数。\n'+ledger, '737'))
    notices = '\n'.join(['ARCHIVE OWNER: NERIS'] + [f'Entry {i:03d}: routine inspection recorded.' for i in range(72)])
    specs.append(('long_context', 'header-owner', 'Read the archive below. Reply only with its ARCHIVE OWNER, preserving case.\n'+notices, 'NERIS'))
    records = '\n'.join(f'记录{i:03d}|标记=苔石|值={i+210}' for i in range(64))
    specs.append(('long_context', 'two-records', '从以下记录读取记录015与记录049的值，按此顺序仅输出两个整数构成的JSON数组。\n'+records, [225, 259]))
    notes = '\n'.join([f'Inspection {i:03d}: no change.' for i in range(72)] + ['FINAL SEAL: vC6_hN'])
    specs.append(('long_context', 'footer-seal', 'Read every line below and return only the value of FINAL SEAL.\n'+notes, 'vC6_hN'))
    return [{'id': 'seq-train-'+name, 'split': 'train', 'group': group,
             'messages': [{'role': 'user', 'content': prompt}],
             'check': ({'type': 'exact', 'expected': expected, 'max_new_tokens': 192}
                       if isinstance(expected, str) else
                       {'type': 'semantic', 'validator': 'json', 'expected': expected, 'max_new_tokens': 192})}
            for group, name, prompt, expected in specs]


def _score(cases, generations):
    if cases != protection_cases():
        raise ValueError('exact frozen sixteen TRAIN protection cases required')
    if not isinstance(generations, list) or len(generations) != len(cases):
        raise ValueError('complete sixteen-result coverage required; do not filter baseline errors')
    by_id = {}
    for row in generations:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in by_id:
            raise ValueError('invalid or duplicate generation ID')
        tokens = row.get('tokens')
        if (not isinstance(row.get('text'), str) or type(row.get('complete')) is not bool
                or not isinstance(tokens, list) or not 1 <= len(tokens) <= GENERATION['max_tokens']
                or any(type(token) is not int or not 0 <= token < 2**32 for token in tokens)):
            raise ValueError('actual text, boolean completion and bounded token IDs required')
        by_id[row['id']] = row
    if set(by_id) != {case['id'] for case in cases}:
        raise ValueError('missing or foreign generation IDs')
    scored = []
    for case in cases:
        if not is_finite_generation_check(case['check']):
            raise ValueError('finite frozen checks required')
        row = by_id[case['id']]
        scored.append({'id': case['id'], 'group': case['group'], 'text': row['text'],
                       'complete': row['complete'], 'tokens': list(row['tokens']),
                       'score': deepcopy(evaluate_generation(row['text'], case['check'], row['complete']))})
    return scored


def _correct(rows):
    return {row['id'] for row in rows if row['score']['passed']}


def _coverage(rows):
    return {group: {'cases': sum(row['group'] == group for row in rows),
                    'correct': sum(row['group'] == group and row['score']['passed'] for row in rows),
                    'incomplete': sum(row['group'] == group and not row['complete'] for row in rows)}
            for group in GROUPS}


def establish_baseline(cases, generations):
    """Freeze actual original-BF16 results, retaining incorrect and incomplete cases."""
    scored = _score(cases, generations)
    correct = sorted(_correct(scored))
    if not correct:
        raise ValueError('empty original-correct set cannot establish meaningful protection')
    return {'schema': SCHEMA, 'cases': deepcopy(cases), 'generation': dict(GENERATION),
            'generations': deepcopy(generations), 'correct_ids': correct,
            'coverage': _coverage(scored), 'scored': scored,
            'scope': 'TRAIN selection only; sixteen cases are not a broad or universal guarantee'}


def check_proposal(cases, baseline, accepted_generations, candidate_generations):
    """Protect the union of original-correct and previous accepted-correct, without mutation."""
    if (not isinstance(baseline, dict) or baseline.get('schema') != SCHEMA
            or baseline.get('cases') != cases or baseline.get('generation') != GENERATION):
        raise ValueError('baseline protocol or frozen cases changed')
    original = _score(cases, baseline['generations'])
    original_correct = _correct(original)
    if not original_correct or baseline.get('correct_ids') != sorted(original_correct):
        raise ValueError('baseline original-correct record inconsistent')
    accepted = _score(cases, accepted_generations)
    accepted_correct = _correct(accepted)
    missing_original = sorted(original_correct-accepted_correct)
    if missing_original:
        raise ValueError('accepted state already lost original-correct cases: '+', '.join(missing_original))
    candidate = _score(cases, candidate_generations)
    candidate_correct = _correct(candidate)
    required = original_correct | accepted_correct
    lost_original = sorted(original_correct-candidate_correct)
    lost_accepted = sorted(accepted_correct-candidate_correct)
    lost = sorted(required-candidate_correct)
    report = {'schema': SCHEMA, 'accepted': not lost, 'decision': 'accept_proposal' if not lost else 'reject_restore_proposed_layer_bf16',
              'original_correct_ids': sorted(original_correct), 'previous_accepted_correct_ids': sorted(accepted_correct),
              'required_correct_ids': sorted(required), 'candidate_correct_ids': sorted(candidate_correct),
              'lost_original_correct_ids': lost_original, 'lost_accepted_correct_ids': lost_accepted,
              'lost_required_correct_ids': lost, 'gained_vs_original_ids': sorted(candidate_correct-original_correct),
              'gained_vs_accepted_ids': sorted(candidate_correct-accepted_correct),
              'coverage': {'original': _coverage(original), 'accepted': _coverage(accepted), 'candidate': _coverage(candidate)},
              'rows': [{'id': a['id'], 'group': a['group'], 'original': a, 'accepted': b, 'candidate': c}
                       for a, b, c in zip(original, accepted, candidate, strict=True)],
              'scope': 'Full-model generation on frozen TRAIN checks only; final regression/holdout must not select layers',
              'rollback': 'Caller must preserve this report and restore the proposed layer to BF16 on rejection; helper changes no model state'}
    return report
