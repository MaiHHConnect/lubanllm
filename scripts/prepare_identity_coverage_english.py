"""One new unsealed TRAIN-only uppercase qualification protocol; no model calls."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from run_identity_coverage import ARGUMENTS, check_original_production
from verify_luban0_final import digest, write_new
from verify_luban0_product import content_digest
from luban.retention import _validate_record, normalized_context
from luban.settings import require_storage

REPO = Path(__file__).resolve().parents[1]
WORDS = ('PyTorch', 'gamma', 'orange', 'hello', 'test', 'alpha2', 'OpenAI', 'ai_lab',
         'banana', 'monkey', 'TableTop', 'audio_7')
DATASET = 'identity-core-coverage-user-only-english-v1'


def prepare():
    descriptor = REPO / 'research/core_identity_coverage_english_preparation.json'
    if descriptor.exists():
        raise FileExistsError('This new fixed preparation already exists; never replace its pool')
    previous = json.loads((REPO / 'research/core_identity_coverage_protocol.json').read_text())
    path = Path(previous['protocol_path'])
    if digest(path) != previous['protocol_sha256']:
        raise ValueError('Previous sealed protocol changed')
    old = json.loads(path.read_text())
    result_path = path.parent / 'result.json'
    result = json.loads(result_path.read_text())
    if (result.get('passed') or result['original_pool']['qualified_count'] != 2
            or result['original_pool']['minimum_qualified'] != 4 or result.get('training')
            or result['engineering'] or result['retirement']['version'] is not None):
        raise ValueError('New preparation needs the preserved original2/12 rejection with no optimization/heldout calls')
    check_original_production(old)
    plan = copy.deepcopy(old)
    now = time.time()
    pool = []
    for index, word in enumerate(WORDS):
        answer = word.upper()
        prompt = ('Convert the following string to ALL UPPERCASE: map every ASCII letter a-z to A-Z '
                  'and keep digits and underscores unchanged. Output only the complete converted string, '
                  'with no quotes or explanation.\nString: ' + word)
        identifier = f'coverage-en-replay-candidate-{index}'
        context = [{'role': 'user', 'content': prompt}]
        value = {'base_identity': old['base_identity'], 'dataset': DATASET,
            'protocol_id': identifier, 'split_family': identifier, 'evolution_role': 'replay',
            'messages': context, 'context_sha256': content_digest(context),
            'generation_check': {'type': 'exact', 'expected': answer, 'max_new_tokens': 192},
            'program_verified': True, 'human_verified': False,
            'verification_evidence': f'Python str.upper: {word!r} -> {answer!r}',
            'authoring_scope': 'locally self-authored under explicit user-authorized teacher/review instruction; no paid teacher or third-party output',
            'baseline_preservation_eligible': None}
        row = {'id': identifier, 'prompt': prompt, 'answer': answer, 'source': 'synthetic',
               'purpose': 'training', 'created': now, 'approved': 0, 'training_allowed': True,
               'provenance': json.dumps(value, ensure_ascii=False)}
        checked = copy.deepcopy(row)
        checked['approved'] = 1
        _validate_record(checked)
        pool.append(row)
    old_ids = {r['id'] for r in old['replay_training_candidates']}
    contexts = {normalized_context(r) for rows in old['groups_before_original_baseline'].values() for r in rows}
    if any(r['id'] in old_ids or normalized_context(r) in contexts for r in pool):
        raise ValueError('New complete TRAIN-only contexts overlap old IDs or declared groups')
    if [r['answer'] for r in pool] != [r['answer'] for r in old['replay_training_candidates']]:
        raise ValueError('Fixed twelve words/Python uppercase gold changed')
    root = require_storage(old['production_root'])
    directory = root / 'experiments' / ('luban0-identity-coverage-english-preregister-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    plan.update(sealed=False, status='prepared new complete English user-only TRAIN qualification; not sealed/run',
        reason='Original Chinese context produced2/12 exact uppercase with completeEOS; no budget cause claimed. New explicit ALL UPPERCASE context has not been generated and is not assumed correct.',
        previous_qualification_protocol_path=str(path), previous_qualification_protocol_sha256=digest(path),
        previous_qualification_result_path=str(result_path), previous_qualification_result_sha256=digest(result_path),
        session_id=uuid.uuid4().hex[:12], prepared=now, research_root=str(directory / 'research-storage'),
        replay_training_candidates=pool, replay_candidate_sha256=content_digest(pool), minimum_replay_candidates=4,
        fixed_candidate_words=list(WORDS), candidate_dataset=DATASET, prefix_alignment_required=True,
        prefix_alignment_policy='Before qualification/Adam, check inference tokens versus TRAIN masked-prefix and exact gold+EOS labels for NEW34+REPLAY44+candidate12 only. Any mismatch rejects; no DEV tokenization/generation.',
        old_known_development_scope='Known25 engineering regressions only: previousNEW8+REPLAY17. PyTorch/gamma and other word-level TRAIN overlap is declared; no independent transfer claim.',
        fresh_dev_scope='Unchanged ungenerated fresh8: four identity contexts plus cedar/LumenGrid/vector8/olive_path; never TRAIN or feedback-driven retry.',
        engineering_scope='Original ungenerated fixed16, used once only after frozen TRAIN/freshDEV/known25 all pass; failure retiresNone.',
        arguments=ARGUMENTS, maximum_attempts=1, teacher_calls=0,
        source_binding='Pending parent/engine final source review; no seal/model allocation now',
        run_descriptor_path=str(REPO / 'research/core_identity_coverage_english_run.json'),
        script_sha256=None, execution_manifest=None)
    plan['historical_files'] += [{'path': str(p), 'sha256': digest(p)} for p in (path, result_path)]
    for field in ('data_draft_path', 'data_draft_sha256', 'planned_counts_if_baseline_all8_pass'):
        plan.pop(field, None)
    if (plan['groups_before_original_baseline'] != old['groups_before_original_baseline']
            or plan['source_rows'] != old['source_rows'] or plan['retention_regressions'] != old['retention_regressions']
            or content_digest(plan['engineering_cases']) != old['engineering_cases_sha256']):
        raise ValueError('Unchanged corpus or complete regression obligations changed')
    directory.mkdir(mode=0o700, exist_ok=False)
    draft = directory / 'draft-protocol.json'
    write_new(draft, plan)
    record = {'draft_path': str(draft), 'draft_sha256': digest(draft), 'prepared_without_model': True,
        'sealed': False, 'baseline_verified': False, 'pool_count': 12, 'minimum_qualified': 4,
        'protocol_descriptor_path': str(REPO / 'research/core_identity_coverage_english_protocol.json')}
    write_new(descriptor, record)
    print(json.dumps({**record, 'groups': {k: len(v) for k, v in plan['groups_before_original_baseline'].items()},
                     'pool_sha256': plan['replay_candidate_sha256'], 'engineering_sha256': plan['engineering_cases_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        print(json.dumps({'words': WORDS, 'dataset': DATASET, 'minimum_qualified': 4, 'budget': 192,
                          'prepared_without_model': True}, ensure_ascii=False))
