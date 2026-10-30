"""Fixed three-task correction/replay draft; no model, seal or paid teacher."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import uuid

from run_identity_coverage import ARGUMENTS, REPLAY_GENERATION_POLICY, check_original_production
from verify_luban0_final import digest, write_new
from verify_luban0_product import content_digest
from luban.core_publication import PRESERVATION_CORRECTIONS, PRESERVATION_DATASET, SUPERSEDED_DUPLICATES
from luban.retention import _validate_groups, _validate_record, messages, provenance
from luban.settings import require_storage

REPO = Path(__file__).resolve().parents[1]
COUNTS = {'new_training': 36, 'new_validation': 8, 'replay_training': 44, 'replay_validation': 17}


def obligations(groups, regressions):
    return Counter(content_digest({'id': row['id'], 'messages': messages(row), 'gold': row['answer'],
        'check': provenance(row)['generation_check']}) for row in
        groups['new_validation'] + groups['replay_validation'] + regressions)


def preservation_plan(old, result, *, prepared, session_id, directory):
    """Reclassify exposed goals without changing any of the 49 obligations."""
    groups = old['groups_before_original_baseline']
    if ({key: len(rows) for key, rows in groups.items()} != COUNTS or len(old['source_rows']) != 103
            or old['arguments'] != ARGUMENTS or old['minimum_replay_candidates'] != 4
            or len(old['replay_training_candidates']) != 12 or len(old['retention_regressions']) != 24
            or result.get('passed') is not False or result.get('engineering') or result.get('known_regression')
            or result.get('publication') or result.get('recovery') or result['training'].get('promoted') is not False
            or result['retirement'].get('version') is not None or result['retirement'].get('restored') is not True
            or result['retirement'].get('tensor_sha256') != old['original_tensor_sha256']):
        raise ValueError('Needs the exact finite36 rejected corpus, untouched engineering, and restored original core')
    original = {row['id']: row for rows in groups.values() for row in rows}
    observed = {row['id']: row for key in ('baseline_new_generation', 'baseline_replay_generation')
                for row in result['training'][key]}
    existing = {row['id'] for row in old['source_rows'] + old['replay_training_candidates'] +
                old['retention_regressions'] + list(original.values())}
    plan, incoming, readonly = copy.deepcopy(old), [], []
    for new_id, (old_id, word, role) in PRESERVATION_CORRECTIONS.items():
        source, baseline = original[old_id], observed[old_id]
        value = provenance(source)
        # Literal source extraction, not a guessed answer or historical model output.
        quoted = re.findall(r'[「“]([^」”]+)[」”]', messages(source)[-1]['content'])
        if (quoted != [word] or source['answer'] != word.upper()
                or value['generation_check'].get('expected') != word.upper()
                or value['generation_check'].get('max_new_tokens') != 192
                or source['purpose'] != 'validation' or new_id in existing
                or baseline.get('complete') is not True or baseline.get('branch_enabled') is not False
                or baseline.get('passed') is not (role == 'replay')
                or (role == 'replay' and baseline.get('answer') != word.upper())):
            raise ValueError('Three tasks must bind their complete quoted source, unchanged gold/check and actual historical classification')
        correction = {'base_identity': old['base_identity'], 'dataset': PRESERVATION_DATASET,
            'protocol_id': new_id, 'split_family': new_id, 'evolution_role': role,
            'messages': copy.deepcopy(messages(source)), 'generation_check': copy.deepcopy(value['generation_check']),
            'rights_basis': 'User authorized local self-authored teaching correction; no paid teacher',
            'human_verified': False, 'program_verified': True, 'source_string': quoted[0],
            'verification_evidence': f'Python str.upper: {quoted[0]!r} -> {quoted[0].upper()!r}',
            'exposed_old_validation_id': old_id, 'original_record_sha256': content_digest(source),
            'scope': 'Explicitly reused exposed development goal; not independent validation or migration evidence'}
        if role == 'replay':
            # Historical success classifies the task, but cannot qualify this new run.
            correction.update(baseline_preservation_eligible=None, training_generation_gate=False)
        row = {'id': new_id, 'prompt': source['prompt'], 'answer': word.upper(), 'source': 'synthetic',
            'purpose': 'training', 'created': prepared, 'approved': 1, 'training_allowed': True,
            'provenance': json.dumps(correction, ensure_ascii=False)}
        _validate_record(row)
        incoming.append(row)
        copied = copy.deepcopy(source)
        copied['training_allowed'] = False
        flags = provenance(copied)
        flags.update(evolution_role='replay', retention_generation_only=True,
                     scope='Exposed development context reused in TRAIN; readonly regression, not independent test',
                     training_reused_as=new_id, original_development_id=old_id)
        copied['provenance'] = json.dumps(flags, ensure_ascii=False)
        _validate_record(copied, generation_only=True)
        readonly.append(copied)
        previous_role = value['evolution_role']
        plan['groups_before_original_baseline'][previous_role + '_validation'] = [
            item for item in plan['groups_before_original_baseline'][previous_role + '_validation'] if item['id'] != old_id]
        plan['groups_before_original_baseline'][role + '_training'].append(row)
    groups = plan['groups_before_original_baseline']
    for rows in groups.values():
        rows.sort(key=lambda row: row['id'])
    _validate_groups(groups)
    plan['retention_regressions'] += readonly
    plan['old_known_development_regressions'] += readonly
    if obligations(groups, plan['retention_regressions']) != obligations(old['groups_before_original_baseline'], old['retention_regressions']):
        raise ValueError('Original complete 49 context/gold/check obligations changed')
    archived = {row['id'] for row in old['source_rows'] if row['purpose'] == 'validation'
                and provenance(row).get('evolution_role') == 'new'} | set(SUPERSEDED_DUPLICATES)
    archived.add(PRESERVATION_CORRECTIONS['preservation-replay-train-pytorch'][0])
    if len(archived) != 13 or not archived <= {row['id'] for row in old['source_rows']}:
        raise ValueError('Exact13 archive must remain anchored in unchanged103 original records')
    plan.update(sealed=False, status='prepared explicit three-task preservation; NOT sealed/run', prepared=prepared,
        session_id=session_id, research_root=str(directory / 'research-storage'),
        reason='Known CEDAR goal remains wrong; original-correct OLIVE_PATH/PYTORCH regressed. Teach the first and protect the exact two decoder paths in TRAIN; no independent-transfer claim.',
        policy_version=REPLAY_GENERATION_POLICY, groups_sha256=content_digest(groups), retention_regressions_sha256=content_digest(plan['retention_regressions']),
        preservation_training_rows=copy.deepcopy(incoming), preservation_training_sha256=content_digest(incoming),
        required_replay_training_ids=sorted(row['id'] for row in incoming if provenance(row)['evolution_role'] == 'replay'),
        required_original_baseline_policy='Qualification stage: both new TRAIN IDs each generated once on current original-core SHA/OFF/no reference/full EOS/unchanged gold; marker forbids retry, both mandatory or reject. Each shared training entry, including same-session checkpoint resume, separately regenerates every flagged pool+2 original baseline once; any failure rejects. Qualification is not retried. No fixed total baseline-call count for a resumed run; historical DEV cannot substitute.',
        superseded_source_ids=sorted(archived), archive_scope='Exact13: original4 duplicates + original8 NEW development + original1 PyTorch replay development; retain all raw fields/gold/source.',
        prefix_alignment_policy='NEW37 + REPLAY46 + fixed pool12 =95 actual token prefixes/gold+EOS; no DEV/engineering tokenization',
        fresh_dev_scope='Fixed6 previously observed development regressions; three exposed goals explicitly TRAIN/readonly, not new independent tests',
        old_known_development_scope='Known11 =oldNEW8+three exposed TRAIN-reused contexts; total readonly27 adds untouched engineering16, original49 obligations unchanged',
        known_regression_training_reuse=old.get('known_regression_training_reuse', []) + [item[0] for item in PRESERVATION_CORRECTIONS.values()],
        engineering_scope=old['engineering_scope'],
        selection='One originalNone attempt max512. Pool12 min4 plus mandatory2 original qualification; each once at qualification, then all flagged originals once per shared training entry including resume, any failure rejects; no fixed total baseline-call claim. First two consecutive fixed TRAIN NEW37all + all flagged protected replay generation + unchanged loss/minNEW2, then fixed6/16 and readonly11; engineering16 once only.',
        training_selection_change_proposed='Two consecutive fixed TRAIN checks: allNEW37 + every qualified original-correct replay whole-string generation/EOS; unchanged loss budget/minNEW2, no DEV feedback',
        planned_counts_if_baseline_qualifies={'new_training': 37, 'new_validation': 6,
            'replay_training': '46 plus4..12 fixed-pool qualifiers', 'replay_validation': 16},
        snapshot_if_all_pass='49 complete obligations =NEWDEV6+REPLAYDEV16+readonly27; derived count, no fixed41 assumption; actual backup/original restoration/full reload/all49 proof before production transaction',
        publication='Exact37 layout only; same durable productionjournal+matching records+immutable evidence+exact13 archive+pointer+event transaction after all outer gates',
        exposure_limit='256 NEW optimizer exposures /37 yields min6/max7 at512; min2 and two consecutive whole TRAIN passes still required, no success guarantee',
        initial_core_version=None, warm_start_candidate=False, maximum_attempts=1, teacher_calls=0,
        implementation_blockers=[], script_sha256=None, execution_manifest=None,
        source_binding='Unsealed: shared strict normalized_exact guard, exact37 publisher and30 execution sources; root review/explicit seal/GO required',
        run_descriptor_path=str(REPO / 'research/core_identity_preservation_run.json'))
    for field in ('source_rows', 'source_groups', 'replay_training_candidates', 'engineering_cases', 'arguments', 'training_checkpoints'):
        if plan[field] != old[field]:
            raise ValueError('Original103, pool12, engineering16 or finite budget changed')
    return plan


def prepare():
    descriptor = REPO / 'research/core_identity_preservation_preparation.json'
    if descriptor.exists():
        raise FileExistsError('Preservation already prepared; never rewrite a fixed draft')
    previous = json.loads((REPO / 'research/core_identity_correction_metadata_protocol.json').read_text())
    path = Path(previous['protocol_path'])
    result_path = path.parent / 'result.json'
    evidence = json.loads((REPO / 'research/core_identity_correction_metadata_result.json').read_text())
    if digest(path) != previous['protocol_sha256'] or digest(result_path) != evidence['raw_result_sha256']:
        raise ValueError('Historical protocol/result evidence changed')
    old = json.loads(path.read_text())
    check_original_production(old)
    directory = require_storage(old['production_root']) / 'experiments' / ('luban0-identity-preservation-preregister-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    plan = preservation_plan(old, json.loads(result_path.read_text()), prepared=time.time(),
                             session_id=uuid.uuid4().hex[:12], directory=directory)
    plan.update(previous_preservation_protocol_path=str(path), previous_preservation_protocol_sha256=digest(path),
                previous_preservation_result_path=str(result_path), previous_preservation_result_sha256=digest(result_path))
    plan['historical_files'] += [{'path': str(p), 'sha256': digest(p)} for p in (path, result_path)]
    directory.mkdir(mode=0o700, exist_ok=False)
    draft = directory / 'draft-protocol.json'
    write_new(draft, plan)
    record = {'draft_path': str(draft), 'draft_sha256': digest(draft), 'prepared_without_model': True,
        'sealed': False, 'baseline_verified': False, 'pool_count': 12, 'minimum_qualified': 4,
        'mandatory_original_replay_baselines': 2,
        'protocol_descriptor_path': str(REPO / 'research/core_identity_preservation_protocol.json')}
    write_new(descriptor, record)
    print(json.dumps({**record, 'groups': {key: len(rows) for key, rows in plan['groups_before_original_baseline'].items()},
                      'protection_count': 49, 'archive_count': 13, 'prefix_count': 95}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        print(json.dumps({'dataset': PRESERVATION_DATASET, 'corrections': PRESERVATION_CORRECTIONS,
                          'prepared_without_model': True, 'sealed': False, 'model_go': False}))
