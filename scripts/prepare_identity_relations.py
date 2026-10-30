"""Prepare the fixed literal-relation correction; no model, seal or teacher."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from run_identity_coverage import (RELATION_ARGUMENTS, RELATION_TRAIN_CHECKPOINTS,
    check_original_production, validate_protocol_policy)
from verify_luban0_final import digest, write_new
from verify_luban0_product import content_digest
from luban.core_publication import (RELATION_DATASET, RELATION_ENGINEERING_IDS, RELATION_REUSE_IDS,
                                   validate_relation_teaching)
from luban.settings import require_storage

REPO = Path(__file__).resolve().parents[1]
PROPOSAL = REPO / 'research/core_identity_relation_teaching_draft.json'
PROPOSAL_SHA = '7947e20eccb6ccc9b4a6c6092662e629f8ac2b4d085ac8765d519fdd2ce7435f'


def relation_plan(old, result, proposal, *, prepared, session_id, directory):
    """Preserve original records; exposed engineering copies are TRAIN, not fresh."""
    groups, readonly = copy.deepcopy(proposal['groups']), copy.deepcopy(proposal['regression_validation'])
    validate_relation_teaching(groups, readonly)
    failed = {row['id'] for row in result['engineering'] if not row['passed']}
    if (result.get('passed') is not False or result.get('engineering_complete') is not True
            or failed != set(RELATION_REUSE_IDS.values()) or len(result['engineering']) != 16
            or result.get('production_publication') or result.get('recovery')
            or result['retirement'].get('version') is not None or result['retirement'].get('restored') is not True
            or result['retirement'].get('tensor_sha256') != old['original_tensor_sha256']
            or len(old['source_rows']) != 103 or len(old['superseded_source_ids']) != 13
            or proposal['fixed_replay_candidates'] != old['replay_training_candidates']
            or len(old['engineering_cases']) != 16 or len(old['retention_regressions']) != 27):
        raise ValueError('Requires the exact rejected37 original-restored result, unchanged103/pool12/archive13')
    incoming = {row['id']: row for row in groups['new_training']}
    if any(incoming.get(row['id']) != row for row in old['groups_before_original_baseline']['new_training']):
        raise ValueError('Original37 complete TRAIN records must remain unchanged')
    indexed = {row['id']: row for row in readonly}
    if any(indexed.get(row['id']) != row for row in old['retention_regressions']):
        raise ValueError('Original27 complete readonly records must remain unchanged')
    original_engineering = {row['id']: row for row in old['engineering_cases']}
    if any(indexed.get(key) != row for key, row in original_engineering.items()):
        raise ValueError('Exposed engineering16 permissions, full contexts and gold cannot be rewritten')
    plan = copy.deepcopy(old)
    plan.update(sealed=False, status='prepared fixed44 literal-relation teaching; NOT sealed/run', prepared=prepared,
        session_id=session_id, research_root=str(directory / 'research-storage'), relation_dataset=RELATION_DATASET,
        groups_before_original_baseline=groups, groups_sha256=content_digest(groups),
        retention_regressions=readonly, retention_regressions_sha256=content_digest(readonly), protection_count=63,
        old_known_development_regressions=[row for row in readonly if row['id'] not in RELATION_ENGINEERING_IDS],
        engineering_cases=[row for row in readonly if row['id'] in RELATION_ENGINEERING_IDS],
        arguments=copy.deepcopy(RELATION_ARGUMENTS), training_checkpoints=list(RELATION_TRAIN_CHECKPOINTS),
        relation_proposal_path=str(PROPOSAL), relation_proposal_sha256=PROPOSAL_SHA,
        reason='Prior TRAIN37/development/protected replay passed, but exposed engineering identity0/3/4/5 failed exact identity and trainer relations. Teach those complete contexts and three literal attributes; old gold is unchanged.',
        engineering_scope='New fixed8 untouched engineering contexts, once only after all developmental and exposed-regression gates; old16 are exposed known regressions, not fresh tests.',
        fresh_dev_scope='Original6 observed developmental regressions plus new6 fixed contexts; no development feedback into TRAIN checkpoint selection.',
        old_known_development_scope='Readonly27 exposed regressions =prior11 + old engineering16; preserve full contexts/checks/permissions. Four failures explicitly copied as seven new literal teaching rows, no independent-transfer claim.',
        old_replay_dev_scope='Unchanged16 known old replay obligations; original PyTorch validation is in readonly27 and explicitly reused TRAIN, gold unchanged.',
        prefix_alignment_policy='NEW44 + REPLAY46 + fixed candidate12 =102 TRAIN-only token prefixes/gold+EOS; no DEV or engineering tokenization.',
        selection='New originalNone session, maximum640. Fixed pool12 min4 and mandatory2 each qualification once; all flagged originals separately rechecked per shared training entry including resume. Two consecutive fixed TRAIN allNEW44 + all flagged replay/EOS + unchanged replay loss/minNEW2. Then NEWDEV12/REPLAYDEV16/known27 and freshengineering8 once, all63 mandatory.',
        snapshot_if_all_pass='63 unique complete obligations =old49 + freshDEV6 + freshengineering8; old engineering16 already in49. Real backup/original restoration/native full reload verifies exact target tensor SHA, complete63 snapshot/hash, two observed identity smoke answers, then actually regenerates all63 closed-book/EOS/gold before publication.',
        planned_counts_if_baseline_qualifies={'new_training': 44, 'new_validation': 12,
            'replay_training': '46 plus4..12 current fixed-pool qualifiers', 'replay_validation': 16},
        exposure_limit='320 NEW optimizer exposures /44 yields min7/max8 at640; optimizer batch1, min2 and two consecutive whole TRAIN passes still mandatory. Coverage/budget correction, no optimal-batch or guaranteed-success claim.',
        publication='Exact44 dataset only; unchanged103 anchors and exact13 archives, complete63 retention and raw matching journal/report/files. No retrospective gold changes or permission edits to old engineering rows.',
        repeat_budget_scope='New independent originalNone maximum640; no prior480/master/Adam/RNG/checkpoint reuse and no extension of any historical single-run budget.',
        scope_limit='Finite literal identity-name/tuner/position and exact exposed contexts; not unlimited intelligence, universal identity generalization or optimal batch count.',
        training_selection_change_proposed='All NEW44 plus every qualified original-correct replay generation/EOS and unchanged loss/min2/two consecutive checkpoints. No DEV or engineering feedback.',
        initial_core_version=None, warm_start_candidate=False, maximum_attempts=1, teacher_calls=0,
        implementation_blockers=[], script_sha256=None, execution_manifest=None,
        source_binding='Unsealed fixed44/640 path; tokenizer CPU proof, independent source review, source seal and explicit model GO still required.',
        run_descriptor_path=str(REPO / 'research/core_identity_relations_run.json'))
    plan['engineering_cases_sha256'] = content_digest(plan['engineering_cases'])
    plan['old_known_development_regressions_sha256'] = content_digest(plan['old_known_development_regressions'])
    plan['relation_teaching_rows'] = [row for row in groups['new_training'] if row['id'].startswith('relation-')]
    plan['relation_teaching_sha256'] = content_digest(plan['relation_teaching_rows'])
    plan['historical_rebind_scope'] = plan.pop('rebind_scope', None)
    plan['historical_data_draft'] = {key: plan[key] for key in ('data_draft_path', 'data_draft_sha256') if key in plan}
    plan.update(data_draft_path=str(PROPOSAL), data_draft_sha256=PROPOSAL_SHA,
        previous_relation_protocol_path=proposal['prior_protocol']['path'],
        previous_relation_protocol_sha256=proposal['prior_protocol']['sha256'],
        previous_relation_result_path=proposal['prior_raw_result']['path'],
        previous_relation_result_sha256=proposal['prior_raw_result']['sha256'])
    validate_protocol_policy(plan)
    for key in ('source_rows', 'source_groups', 'replay_training_candidates', 'superseded_source_ids',
                'required_replay_training_ids', 'original_tensor_sha256', 'operator_audit'):
        if plan[key] != old[key]:
            raise ValueError('Original103, qualification pool, archive, teacher operator or base SHA changed')
    return plan


def prepare():
    descriptor = REPO / 'research/core_identity_relations_preparation.json'
    if descriptor.exists():
        raise FileExistsError('Never overwrite an existing relation preparation')
    if digest(PROPOSAL) != PROPOSAL_SHA:
        raise ValueError('Immutable relation teaching proposal SHA changed')
    proposal = json.loads(PROPOSAL.read_text())
    previous, raw = proposal['prior_protocol'], proposal['prior_raw_result']
    paths = [Path(previous['path']), Path(raw['path'])]
    if any(digest(path) != record['sha256'] for path, record in zip(paths, (previous, raw))):
        raise ValueError('Original37 protocol or rejected raw evidence changed')
    old = json.loads(paths[0].read_text())
    check_original_production(old)
    directory = require_storage(old['production_root']) / 'experiments' / ('luban0-identity-relations-preregister-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8])
    plan = relation_plan(old, json.loads(paths[1].read_text()), proposal, prepared=time.time(),
                         session_id=uuid.uuid4().hex[:12], directory=directory)
    plan['historical_files'] += [{'path': str(path), 'sha256': digest(path)} for path in paths + [PROPOSAL]]
    directory.mkdir(mode=0o700, exist_ok=False)
    draft = directory / 'draft-protocol.json'
    write_new(draft, plan)
    record = {'draft_path': str(draft), 'draft_sha256': digest(draft), 'prepared_without_model': True,
        'sealed': False, 'baseline_verified': False, 'pool_count': 12, 'minimum_qualified': 4,
        'mandatory_original_replay_baselines': 2, 'protection_count': 63, 'prefix_count': 102,
        'protocol_descriptor_path': str(REPO / 'research/core_identity_relations_protocol.json')}
    write_new(descriptor, record)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    if parser.parse_args().prepare:
        prepare()
    else:
        print(json.dumps({'dataset': RELATION_DATASET, 'arguments': RELATION_ARGUMENTS,
            'model_go': False, 'sealed': False, 'teacher_calls': 0}))
