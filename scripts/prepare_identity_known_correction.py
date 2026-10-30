"""Prepare two explicitly exposed NEW corrections; never seal or allocate a model."""
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
from luban.retention import (_validate_groups, _validate_record, messages,
                            provenance, validate_retention_conflicts)
from luban.settings import require_storage

REPO = Path(__file__).resolve().parents[1]
DATASET = 'identity-core-exposed-uppercase-correction-v1'
CORRECTIONS = (
    ('30745d9410f14a748c7029096adaeed0', 'tokyo', 'known-correction-new-train-tokyo'),
    ('7140260be6f44bd49c3aa5a59f8c95e4', 'signal7', 'known-correction-new-train-signal7'),
)
COUNTS = {'new_training': 34, 'new_validation': 8, 'replay_training': 44, 'replay_validation': 17}


def corrected_plan(old, result, *, prepared, session_id, directory):
    """Keep all old gold/obligations; only add the two already exposed TRAIN rows."""
    if ({key: len(rows) for key, rows in old['groups_before_original_baseline'].items()} != COUNTS
            or len(old['source_rows']) != 103 or old['arguments'] != ARGUMENTS
            or old['minimum_replay_candidates'] != 4 or len(old['replay_training_candidates']) != 12
            or result.get('passed') is not False or result.get('engineering')
            or result['retirement'].get('version') is not None
            or result['retirement'].get('restored') is not True
            or result['retirement'].get('tensor_sha256') != old['original_tensor_sha256']):
        raise ValueError('Needs the unchanged finite34 corpus and genuinely retired original-core rejection')
    original = {row['id']: row for row in old['source_rows']}
    known = {row['id']: row for row in old['old_known_development_regressions']}
    observed = {row['id']: row for row in result['known_regression']}
    plan = copy.deepcopy(old)
    incoming = []
    for old_id, word, new_id in CORRECTIONS:
        source, readonly, actual = original[old_id], known[old_id], observed[old_id]
        value = provenance(source)
        if (source['purpose'] != 'validation' or value.get('evolution_role') != 'new'
                or source['answer'] != word.upper() or readonly['answer'] != source['answer']
                or messages(readonly) != messages(source)
                or provenance(readonly)['generation_check'] != value['generation_check']
                or value['generation_check']['max_new_tokens'] != 192
                or actual.get('passed') is not False or actual.get('complete') is not True):
            raise ValueError('Correction must bind the exact exposed original NEW target/check/context')
        correction = {'base_identity': old['base_identity'], 'dataset': DATASET,
            'protocol_id': new_id, 'split_family': new_id, 'evolution_role': 'new',
            'messages': copy.deepcopy(messages(source)),
            'generation_check': copy.deepcopy(value['generation_check']),
            'rights_basis': 'User explicitly authorized local teaching and review; self-authored, no paid teacher',
            'human_verified': False, 'program_verified': True, 'source_string': word,
            'verification_evidence': f'Python str.upper: {word!r} -> {word.upper()!r}',
            'exposed_old_validation_id': old_id, 'original_record_sha256': content_digest(source),
            'observed_failure': copy.deepcopy(actual),
            'scope': 'Known failed NEW goal now explicitly TRAIN; same-context regression is no longer independent validation'}
        row = {'id': new_id, 'prompt': source['prompt'], 'answer': word.upper(), 'source': 'synthetic',
            'purpose': 'training', 'created': prepared, 'approved': 1, 'training_allowed': True,
            'provenance': json.dumps(correction, ensure_ascii=False)}
        _validate_record(row)
        incoming.append(row)
    all_rows = old['source_rows'] + [row for rows in old['groups_before_original_baseline'].values() for row in rows]
    all_rows += old['replay_training_candidates'] + old['retention_regressions']
    if {row['id'] for row in incoming} & {row['id'] for row in all_rows}:
        raise ValueError('New correction stable ID conflicts with an anchored or protected record')
    plan['groups_before_original_baseline']['new_training'] += incoming
    plan['groups_before_original_baseline']['new_training'].sort(key=lambda row: row['id'])
    groups = plan['groups_before_original_baseline']
    _validate_groups(groups)
    for row in plan['retention_regressions']:
        _validate_record(row, generation_only=True)
    # Same-context TRAIN/readonly repetition is permitted only with identical gold.
    snapshot = {'groups': old['groups_before_original_baseline'], 'regression_validation': old['retention_regressions']}
    validate_retention_conflicts(snapshot, incoming)
    for field in ('source_rows', 'source_groups', 'replay_training_candidates', 'engineering_cases',
                  'retention_regressions', 'old_known_development_regressions'):
        if plan[field] != old[field]:
            raise ValueError('Anchors, fixed pool or original readonly obligations changed')
    for key in ('new_validation', 'replay_training', 'replay_validation'):
        if groups[key] != old['groups_before_original_baseline'][key]:
            raise ValueError('Existing split or development gold changed')
    plan.update(sealed=False, status='prepared exposed-goal teaching correction; not sealed/run',
        reason='Original base never mastered these two SAME contexts. Last candidate failed fixed known goals; teach exact known contexts explicitly, without calling this independent generalization.',
        session_id=session_id, prepared=prepared, research_root=str(directory / 'research-storage'),
        groups_sha256=content_digest(groups), correction_dataset=DATASET,
        correction_rows_sha256=content_digest(incoming), correction_training_rows=copy.deepcopy(incoming),
        known_regression_training_reuse=[old_id for old_id, _, _ in CORRECTIONS],
        fresh_dev_scope='Unchanged fixed8 development regressions, observed on previous candidate; not untouched or independent scientific final',
        old_known_development_scope='Unchanged known25: original-correct17 plus oldNEW8; two explicitly reused as TRAIN, no independent-test claim',
        engineering_scope='Unchanged16 never generated; once only after frozen TRAIN/development/known regression gates',
        prefix_alignment_policy='Actual inference masked-prefix and exact gold+EOS on NEW36+REPLAY44+candidate12 =92; never tokenize DEV/engineering',
        selection='One bounded512 attempt from originalNone; min4 fixed pool; first two consecutive TRAIN NEW36all+qualified replay generationall+same protected loss+minNEW2; then unchanged development and known regression; no DEV-driven retry',
        training_selection_change_proposed='Two consecutive fixed TRAIN checks: allNEW36 generation/EOS, all qualified replay TRAIN generation/EOS, unchanged replay loss budget and minNEW2; no development feedback',
        planned_counts_if_baseline_qualifies={'new_training': 36, 'new_validation': 8,
            'replay_training': '44 plus4..12 baseline-qualified uppercase rows', 'replay_validation': 17},
        snapshot_if_all_pass='All49 fixed obligations =previously observed NEW development8+original-correct replay17+knownoldNEW8+untouched engineering16; same-context TRAIN-reused2 are regressions, not independent tests; real backup/original restore/full reload required',
        initial_core_version=None, warm_start_candidate=False, maximum_attempts=1, teacher_calls=0,
        implementation_blockers=[],
        source_binding='Unsealed: exact-two NEW36 publication guard and29 execution sources require seal/readback; model run requires parent explicit GO',
        script_sha256=None, execution_manifest=None,
        run_descriptor_path=str(REPO / 'research/core_identity_known_correction_run.json'))
    return plan


def prepare():
    descriptor = REPO / 'research/core_identity_known_correction_preparation.json'
    if descriptor.exists():
        raise FileExistsError('Known correction already prepared; never replace its fixed data')
    previous = json.loads((REPO / 'research/core_identity_coverage_english_guard_protocol.json').read_text())
    path = Path(previous['protocol_path'])
    if digest(path) != previous['protocol_sha256']:
        raise ValueError('Previous sealed protocol changed')
    old = json.loads(path.read_text())
    result_path = path.parent / 'result.json'
    result = json.loads(result_path.read_text())
    check_original_production(old)
    root = require_storage(old['production_root'])
    directory = root / 'experiments' / ('luban0-identity-known-correction-preregister-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    plan = corrected_plan(old, result, prepared=time.time(), session_id=uuid.uuid4().hex[:12], directory=directory)
    plan.update(previous_qualification_protocol_path=str(path), previous_qualification_protocol_sha256=digest(path),
        previous_qualification_result_path=str(result_path), previous_qualification_result_sha256=digest(result_path))
    plan['historical_files'] += [{'path': str(p), 'sha256': digest(p)} for p in (path, result_path)]
    directory.mkdir(mode=0o700, exist_ok=False)
    draft = directory / 'draft-protocol.json'
    write_new(draft, plan)
    record = {'draft_path': str(draft), 'draft_sha256': digest(draft), 'prepared_without_model': True,
        'sealed': False, 'baseline_verified': False, 'pool_count': 12, 'minimum_qualified': 4,
        'protocol_descriptor_path': str(REPO / 'research/core_identity_known_correction_protocol.json')}
    write_new(descriptor, record)
    print(json.dumps({**record, 'groups': {key: len(rows) for key, rows in plan['groups_before_original_baseline'].items()},
        'correction_sha256': plan['correction_rows_sha256'], 'pool_sha256': plan['replay_candidate_sha256'],
        'protection_count': 49, 'blockers': plan['implementation_blockers']}, ensure_ascii=False))


def prepare_corrigendum():
    """New original-core attempt; disclose stopped provenance error, not a resume."""
    descriptor = REPO / 'research/core_identity_correction_metadata_preparation.json'
    if descriptor.exists():
        raise FileExistsError('Metadata corrigendum already prepared; do not replace it')
    stopped_path = REPO / 'research/core_identity_known_correction_stopped.json'
    stopped = json.loads(stopped_path.read_text())
    bad_path = Path(stopped['protocol_path'])
    if digest(bad_path) != stopped['protocol_sha256']:
        raise ValueError('Stopped protocol changed')
    bad = json.loads(bad_path.read_text())
    if (stopped.get('pid_dead') is not True or stopped.get('exit_code') != 130
            or stopped.get('engineering_count') != 0 or stopped.get('known_regression_count') != 0
            or stopped.get('original_restored', {}).get('version') is not None
            or stopped['original_restored'].get('tensor_sha256') != bad['original_tensor_sha256']
            or digest(bad_path.parent / 'result.json') != stopped['raw_result_sha256']):
        raise ValueError('Corrigendum requires cooperative stop and proven original restoration before heldout')
    check_original_production(bad)
    previous_path = Path(bad['previous_qualification_protocol_path'])
    previous_result = Path(bad['previous_qualification_result_path'])
    if (digest(previous_path) != bad['previous_qualification_protocol_sha256']
            or digest(previous_result) != bad['previous_qualification_result_sha256']):
        raise ValueError('Prior same-context failure evidence changed')
    directory = require_storage(bad['production_root']) / 'experiments' / (
        'luban0-identity-correction-metadata-preregister-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    plan = corrected_plan(json.loads(previous_path.read_text()), json.loads(previous_result.read_text()),
                          prepared=bad['prepared'], session_id=uuid.uuid4().hex[:12], directory=directory)
    expected = copy.deepcopy(bad['groups_before_original_baseline'])
    for row in expected['new_training']:
        if row['id'] in {item[2] for item in CORRECTIONS}:
            value = provenance(row)
            if value.get('human_verified') is not True or value.get('program_verified') is not True:
                raise ValueError('Stopped defect differs from the declared two metadata flags')
            value['human_verified'] = False
            row['provenance'] = json.dumps(value, ensure_ascii=False)
    if plan['groups_before_original_baseline'] != expected:
        raise ValueError('Corrigendum changed more than two human-verification flags')
    for key in ('source_rows', 'source_groups', 'replay_training_candidates', 'retention_regressions',
                'old_known_development_regressions', 'engineering_cases', 'arguments', 'training_checkpoints'):
        if plan[key] != bad[key]:
            raise ValueError('Corrigendum changed anchored materials, gold or budget')
    plan.update(previous_qualification_protocol_path=str(previous_path),
        previous_qualification_protocol_sha256=digest(previous_path),
        previous_qualification_result_path=str(previous_result),
        previous_qualification_result_sha256=digest(previous_result),
        status='new metadata-corrected independent original-core attempt; not sealed or run',
        provenance_corrigendum={'stopped_protocol_path': str(bad_path), 'stopped_protocol_sha256': digest(bad_path),
            'stop_descriptor_path': str(stopped_path), 'stop_descriptor_sha256': digest(stopped_path),
            'changed_field': 'human_verified', 'old_value': True, 'new_value': False,
            'program_verified_remains': True, 'material_ids': [item[2] for item in CORRECTIONS],
            'reason': 'Agent program verification was incorrectly labeled human verification; no per-item human review occurred',
            'tokens_gold_checks_unchanged': True},
        repeat_budget_scope='Prior malformed-provenance run stopped at durable144. This is a new independent maximum512 originalNone run if authorized; no checkpoint/master/Adam/RNG reuse, not a resumed single budget.',
        run_descriptor_path=str(REPO / 'research/core_identity_correction_metadata_run.json'))
    plan['historical_files'] += [{'path': str(p), 'sha256': digest(p)} for p in
        (bad_path, bad_path.parent / 'result.json', stopped_path, previous_path, previous_result)]
    directory.mkdir(mode=0o700, exist_ok=False)
    draft = directory / 'draft-protocol.json'
    write_new(draft, plan)
    record = {'draft_path': str(draft), 'draft_sha256': digest(draft), 'prepared_without_model': True,
        'sealed': False, 'baseline_verified': False, 'pool_count': 12, 'minimum_qualified': 4,
        'protocol_descriptor_path': str(REPO / 'research/core_identity_correction_metadata_protocol.json')}
    write_new(descriptor, record)
    print(json.dumps({**record, 'session_id': plan['session_id'], 'metadata_only_change': True,
                     'groups': {key: len(rows) for key, rows in expected.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--correct-provenance', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.correct_provenance:
        prepare_corrigendum()
    else:
        print(json.dumps({'dataset': DATASET, 'corrections': CORRECTIONS, 'prepared_without_model': True,
                          'sealed': False, 'model_go': False}, ensure_ascii=False))
