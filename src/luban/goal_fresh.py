"""Prospective goal teaching: fresh B45, complete450 rounds, one final holdout.

Research only. This helper is excluded from the client product payload.
"""
from __future__ import annotations
import copy
import math
import re
import time
from pathlib import Path
import torch
from . import goal_sequence as common
from .retention import eligible_learning_sample, messages, provenance, verified_source
from .generation_checks import validate_generation_check, evaluate_generation, is_finite_generation_check

PROTOCOL = 'evolution-goal-prospective-fixed450-v1'
RESUME_PROTOCOL = 'evolution-goal-prospective-fixed450-v2'
RESUME_PROTOCOL_V3 = 'evolution-goal-prospective-fixed450-v3'
RESUME_PROTOCOL_V4 = 'evolution-goal-prospective-fixed450-v4'
DATA_PHASE = 'evolution-goal-new-data-fixed450-v1'
RESUME_PROTOCOLS = {RESUME_PROTOCOL, RESUME_PROTOCOL_V3, RESUME_PROTOCOL_V4}
POLICY = 'goal-prospective-same-weight450-v1'
CHECKPOINTS = list(range(450, 2701, 450))
OLD_FILES = {'old_retention_complete': 'b0874f236a9cb25d850775ec149ea7e444b691095973bdc6a4925c18f7aa251e',
    'old_training_original': 'b8c61cfc2db9627d1b2f8848ff311d28f7ac3d9bf3e81be62e6aeaee87bc6924',
    'old_protection': '4c93cb78215d698d7dc8733cbfbe247cde8bd0b7b48228529197c2f19487945e'}
BASE_ARGUMENTS = {'steps': 2700, 'learning_rate': 1e-5, 'distillation_weight': 5, 'preserve_replay_subspace': True}
need, content_sha, tensor_sha, read_json = common.need, common.content_sha, common.tensor_sha, common.read_json


def parent_step(protocol):
    return 6750 if protocol.get('protocol') == DATA_PHASE else 5400 if protocol.get('protocol') == RESUME_PROTOCOL_V4 else 4050 if protocol.get('protocol') == RESUME_PROTOCOL_V3 else 2700


def checkpoints(protocol):
    if protocol.get('protocol') == DATA_PHASE:
        return list(range(7200, 9451, 450))
    maximum = parent_step(protocol) + 1350 if protocol.get('protocol') in RESUME_PROTOCOLS else 2700
    return list(range(450, maximum + 1, 450))


def load_verified(path):
    path = common.plain(path)
    protocol = read_json(path)
    if protocol.get('protocol') == DATA_PHASE:
        return load_data_phase(path, protocol)
    if protocol.get('protocol') in RESUME_PROTOCOLS:
        return load_continuation(path, protocol)
    need(protocol.get('protocol') == PROTOCOL and protocol.get('status') == 'PRE_REGISTERED_NOT_TRAINED', 'fresh prospective protocol required')
    need(protocol.get('segment_steps') == 450 and type(protocol['segment_steps']) is int
         and protocol.get('total_steps') == 2700 and type(protocol['total_steps']) is int
         and protocol.get('policy_version') == POLICY, 'fresh goal budget must be six complete450 rounds')
    need(re.fullmatch(r'[a-f0-9]{12}', protocol.get('new_run_id', '')), 'new independent run ID required')
    core = protocol['source_core']
    need(core['version'] == 'core-b45d965728bb' and core['identity'] == protocol['identity']
         and protocol['original_tensor_sha256'] == 'c887e507841eb4f0cc81b1dd2fe82f58b4baa80db8a1d30eedb286bc8f36a4d6'
         and core['sha256'] == 'a2eb54773a4fdc18196acf966c4c21400bbbcd8b811fdc2518b7cbcc40442628', 'fresh experiment starts from accepted B45')
    need(protocol['new_counts'] == {'TRAIN': 24, 'VALID': 8} and protocol['acceptance_count'] == 8
         and protocol['old_counts']['original_training'] == 83 and protocol['old_counts']['protected_generation'] == 89,
         'prospective24/8/8 and complete old83/89 required')
    need(Path(protocol['manifest']).name == protocol['manifest'], 'manifest must be a plain directory member')
    manifest = read_json(path.parent / protocol['manifest'])
    files = {}
    for field in ('new_material', 'old_retention_complete', 'old_training_original', 'old_protection', 'acceptance'):
        name = protocol[field]
        need(isinstance(name, str) and Path(name).name == name, 'material must be a plain member of this registered directory')
        member = common.plain(path.parent / name)
        expected = manifest['files'][name]['sha256']
        need(common.file_sha(member) == expected, 'material SHA mismatch: ' + field)
        files[field] = {'path': str(member), 'sha256': expected}
        if field in OLD_FILES:
            need(expected == OLD_FILES[field], 'the full original B45 old material cannot be edited')
    material = read_json(files['new_material']['path'])
    # Materials are kept as the independent author's frozen schema; accept list or
    # the existing records member without rewriting a row or weakening its check.
    rows = material if isinstance(material, list) else material['TRAIN'] + material['VALID']
    need(len(rows) == 32 and sum(r['purpose'] == 'training' for r in rows) == 24
         and sum(r['purpose'] == 'validation' for r in rows) == 8, 'new split count changed')
    need(len({r['id'] for r in rows}) == 32, 'duplicate material IDs')
    for record in rows:
        need(provenance(record).get('draft') is False, 'draft material cannot enter a formal training session')
        check = validate_generation_check(provenance(record).get('generation_check'), default_expected=record['answer'])
        need(is_finite_generation_check(check), 'finite prospectively declared checks required')
        if record['purpose'] == 'training':
            need(eligible_learning_sample(record), 'TRAIN provenance or local learning rights missing')
        else:
            need(record['purpose'] == 'validation', 'fresh VALID must remain evaluation-only by purpose; training_allowed is a rights flag')
    acceptance = read_json(files['acceptance']['path'])
    need(isinstance(acceptance, list) and len(acceptance) == 8 and len({r['id'] for r in acceptance}) == 8
         and not set(r['id'] for r in acceptance) & set(r['id'] for r in rows), 'distinct8 ACCEPTANCE required')
    for record in acceptance:
        p = provenance(record)
        check = validate_generation_check(p.get('generation_check'), default_expected=record['answer'])
        need(p.get('draft') is False and record['purpose'] == 'validation' and not record.get('training_allowed')
             and p.get('retention_generation_only') is True and record.get('approved') and verified_source(record)
             and is_finite_generation_check(check), 'ACCEPTANCE must be verified finite formal generation-only material')
    capture = protocol['protection_capture']
    need(common.file_sha(capture['path']) == capture['sha256'], 'old baseline capture descriptor SHA mismatch')
    descriptor = read_json(capture['path'])
    receipt = descriptor['baseline_receipt']
    need(common.file_sha(receipt['path']) == receipt['sha256'], 'actual B45 baseline receipt SHA changed')
    baseline = read_json(receipt['path'])
    need(baseline['starting_tensor_sha256'] == protocol['original_tensor_sha256']
         and descriptor['results'] == baseline['training']['baseline_core_retention_generation'], 'capture cannot fabricate or substitute baseline outputs')
    old = read_json(files['old_protection']['path'])
    old = old if isinstance(old, list) else old['records']
    need(len(old) == 89 and descriptor['records'] == old and len(descriptor['results']) == 89,
         'capture must use the entire immutable known old89, never new validation')
    need(descriptor['original_tensor_sha256'] == descriptor['baseline_tensor_sha256'] == protocol['original_tensor_sha256']
         and descriptor['scope'] == 'known_old_protection_only' and descriptor['policy'] == 'known-old-generated-states-v1'
         and descriptor['authorized_record_sha256'] == {r['id']: content_sha(r) for r in old},
         'B45 baseline or explicitly authorized old inventory changed')
    need(set(protocol['new_implementation_sha256']) == {'coldpaging.py', 'native_core_training.py', 'calibration.py', 'goal_topics.py', 'evolution_progress.py', 'goal_sequence.py', 'foundation.py', 'process_lock.py', 'backbone.py', 'core_publication.py', 'evolution.py', 'decision.py', 'generation_checks.py', 'training_checkpoint.py', 'goal_fresh.py', 'context.py', 'external_brain.py', 'automatic.py', 'retention.py', 'luban_training.py', 'runtime.py', 'dynamic.py'},
         'fresh implementation must cover original20 plus both new research helpers')
    need(all(re.fullmatch(r'[a-f0-9]{64}', value) for value in protocol['new_implementation_sha256'].values()), 'implementation SHA missing')
    need(all(protocol['new_source_sha256'].get('src/luban/' + name) == value for name, value in protocol['new_implementation_sha256'].items()), 'full source ledger omits training implementation')
    protocol.update(_protocol_sha256=common.file_sha(path), _protocol_path=str(path), _files=files)
    return protocol


def metadata(protocol):
    if protocol.get('protocol') == DATA_PHASE:
        return {'protocol_sha256': protocol['_protocol_sha256'], 'parent_run_id': protocol['parent_run_id'],
                'parent_steps': 6750, 'total_budget': 9450, 'new_steps_budget': 2700}
    parent = parent_step(protocol) if protocol.get('protocol') in RESUME_PROTOCOLS else 0
    return {'protocol_sha256': protocol['_protocol_sha256'], 'parent_run_id': protocol.get('parent_run_id'),
            'parent_steps': parent, 'total_budget': parent + 1350 if parent else 2700, 'new_steps_budget': 1350 if parent else 2700}


def arguments(protocol):
    return {**BASE_ARGUMENTS, 'steps': metadata(protocol)['total_budget'], 'goal_sequence': metadata(protocol)}


def report_metadata(protocol, completed=None):
    completed = (6750 if protocol.get('protocol') == DATA_PHASE else 0) if completed is None else completed
    need(type(completed) is int and 0 <= completed <= metadata(protocol)['total_budget'], 'invalid prospective completed step')
    if protocol.get('protocol') == DATA_PHASE:
        need(6750 <= completed <= 9450, 'invalid new-data phase step')
        return {**metadata(protocol), 'protocol': DATA_PHASE, 'new_steps_executed': completed-6750,
                'phase_steps': completed-6750, 'cumulative_steps': completed, 'selected_segment': (completed-6750+449)//450,
                'training_budget_scope': 'new frozen56TRAIN/12VALID/12ACCEPT episode; six complete450 rounds; parent6750 archived separately'}
    return {**metadata(protocol), 'protocol': protocol['protocol'], 'new_steps_executed': max(0, completed - metadata(protocol)['parent_steps']),
            'cumulative_steps': completed, 'selected_segment': (completed + 449) // 450,
            'training_budget_scope': (f"bounded fresh{parent_step(protocol)} full-state continuation; at most three additional complete450 rounds" if protocol['protocol'] in RESUME_PROTOCOLS else 'six finite complete450 research rounds; ordinary product512 budget unchanged')}


def source_metadata(protocol, selection=None):
    if protocol.get('protocol') == DATA_PHASE:
        return phase_sources(protocol)
    if protocol.get('protocol') in RESUME_PROTOCOLS:
        value = copy.deepcopy(protocol['_parent_report']['goal_sequence_sources'])
        value.update(new_implementation=protocol['new_implementation_sha256'], new_source_sha256=protocol['new_source_sha256'])
        value['lineage_parent'] = {key: copy.deepcopy(protocol[key]) for key in ('parent_run_id', 'files', 'parent_binding_sha256', 'parent_report_sha256', 'parent_history_sha256', 'parent_source_sha256', 'parent_implementation_sha256', 'basis_sha256', 'reference_sha256', 'parent_final_tensor_sha256')}
        value['lineage_parent']['parent_stage_sha256'] = content_sha(protocol['_parent_report']['goal_sequence']['stage_observations'])
        if protocol['protocol'] in {RESUME_PROTOCOL_V3, RESUME_PROTOCOL_V4}:
            value['lineage_parent']['ancestor_lineage'] = copy.deepcopy(protocol['_parent_report']['goal_sequence_sources']['lineage_parent'])
            value['lineage_parent']['parent_goal_sequence_sources_sha256'] = content_sha(protocol['_parent_report']['goal_sequence_sources'])
        value['scope'] = f"closed fresh{parent_step(protocol)} full-state continuation; originalB45 reference/basis; no failed old v3 ancestry"
        value['lineage_parent']['protocol_sha256'] = protocol['_protocol_sha256']
        return value
    return {'protocol_sha256': protocol['_protocol_sha256'], 'original_tensor_sha256': protocol['original_tensor_sha256'],
            'starting_core': protocol['source_core'], 'new_implementation': protocol['new_implementation_sha256'],
            'new_source_sha256': protocol['new_source_sha256'], 'files': protocol['_files'],
            'protection_capture': protocol['protection_capture'], 'scope': 'fresh B45; no rejected-candidate optimizer ancestry'}


def check_call(runtime, path, steps, learning_rate, distillation_weight, preserve, field_weight, partial_policy):
    protocol = load_verified(path)
    session = runtime.training_session
    need(session is not None and session.id == protocol['new_run_id'] and session.binding['scope'] == 'product'
         and getattr(runtime, 'research', False) is not True and session.binding['arguments'] == arguments(protocol)
         and session.body['arguments'] == arguments(protocol), 'distinct frozen fresh product session required')
    need(steps == metadata(protocol)['total_budget'] and type(steps) is int and learning_rate == 1e-5 and distillation_weight == 5
         and preserve is True and field_weight == 1 and partial_policy is None, 'native LR/KL/mean1 objective changed')
    need(runtime.identity == protocol['identity'] and runtime.active_core == protocol['source_core']['version']
         and session.binding['implementation_sha256'] == protocol['new_implementation_sha256'], 'accepted B45 or implementation changed')
    records = session.body['records']
    need({k: len(v) for k, v in records['groups'].items()} == {'new_training': 56 if protocol['protocol'] == DATA_PHASE else 24, 'new_validation': 12 if protocol['protocol'] == DATA_PHASE else 8, 'replay_training': 83, 'replay_validation': 8}
         and len(records['protection']) == 89 and len(records['retention_regressions']) == (12 if protocol['protocol'] == DATA_PHASE else 8), 'frozen cohort inventory changed')
    need(content_sha(records) == session.binding['records_sha256']
         and (protocol['protocol'] != DATA_PHASE or records == protocol['_records']), 'frozen cohort SHA changed')
    return protocol


def initial_saved(runtime=None, protocol=None, session=None, parameter=None, previous=None, groups=None, generation=None, loss=None):
    if protocol is not None and protocol.get('protocol') == DATA_PHASE:
        return initial_phase_saved(runtime, protocol, session, parameter, previous, groups, generation, loss)
    if protocol is None or protocol.get('protocol') not in RESUME_PROTOCOLS:
        return None
    from . import luban_training as engine
    files = protocol['files']
    closed = torch.load(files['parent_checkpoint']['path'], map_location='cpu', weights_only=True, mmap=True)
    fixture = engine.validate_fixture(Path(files['native_manifest']['path']).parent, files['native_manifest']['sha256'])
    output = Path(files['native_output_manifest']['path']).parent
    engine.validate_output(output, fixture, protocol['binary_sha256'])
    native = engine.load_checkpoint(output, fixture)
    payload = derive_payload(protocol, closed, native, session.binding, session.body['records'])
    torch.set_rng_state(closed['rng'])
    if str(runtime.device) == 'mps':
        torch.mps.set_rng_state(closed['mps_rng'])
    session.save(payload, {'step': parent_step(protocol), 'phase': 'derived-parent', 'steps': parent_step(protocol) + 1350})
    return session.load()


def capture_descriptor(protocol):
    need(common.file_sha(protocol['protection_capture']['path']) == protocol['protection_capture']['sha256'], 'capture file changed after preflight')
    descriptor = read_json(protocol['protection_capture']['path'])
    need(content_sha(descriptor) == protocol['protection_capture'].get('content_sha256', content_sha(descriptor)), 'capture content changed')
    return descriptor


def observe(runtime, protocol, groups, parameter, step, generation, loss, reference=(), prior=()):
    started = time.time()
    old = runtime.training_session.body['records']['protection']
    # Shared observation primitive: no new VALID is passed to it.
    proof = common.observe_training_state(runtime, groups, protocol['replay_generation_ids'], parameter, step,
        generation, loss, reference_confirmed=reference, prior_confirmed=prior,
        label=('data-phase-entry6750' if step == 6750 else 'data-phase-complete450') if protocol['protocol'] == DATA_PHASE else 'prospective-reference' if step == 0 else 'prospective-complete450', expected_new=56 if protocol['protocol'] == DATA_PHASE else 24)
    digest, expert = tensor_sha(parameter), runtime.branch.expert
    runtime.branch.expert = None
    try:
        for observed in proof['observations']:
            observed['old89'] = generation(runtime, old)
            need(tensor_sha(parameter) == digest == proof['tensor_sha256'], 'old protection observation changed W')
            need([r['id'] for r in observed['old89']] == [r['id'] for r in old], 'complete old89 order/inventory changed')
            need(all(type(r['passed']) is bool and r.get('branch_enabled') is False and r.get('reference_used') is False
                     and (not r['passed'] or r.get('complete') is True) for r in observed['old89']), 'old protection observation source/EOS invalid')
        proof['old89_both_passed'] = all(r['passed'] for o in proof['observations'] for r in o['old89'])
        proof['elapsed_seconds'] = time.time() - started
        return proof
    finally:
        runtime.branch.expert = expert


def capture_basis(runtime, protocol, report, parameter, previous, groups, generation, loss):
    need(tensor_sha(previous) == tensor_sha(parameter) == protocol['original_tensor_sha256'], 'basis must execute on accepted B45')
    need(report['training_selection']['replay_generation_ids'] == protocol['replay_generation_ids'], 'old13 qualified guard inventory changed')
    report['goal_sequence'].setdefault('stage_observations', [])
    report['goal_sequence']['reference_baseline_before_training'] = observe(runtime, protocol, groups, parameter, 0, generation, loss)
    need(report['goal_sequence']['reference_baseline_before_training']['old89_both_passed'], 'B45 failed full old protection')
    descriptor = capture_descriptor(protocol)
    runtime.original_tensor_sha256 = protocol['original_tensor_sha256']
    runtime.tokenizer_sha256 = common.file_sha(Path(protocol['model_realpath']) / 'tokenizer.json')
    return descriptor


def observe_boundary(runtime, protocol, report, groups, parameter, step, generation, loss):
    need(report['training_selection']['replay_generation_ids'] == protocol['replay_generation_ids'], 'old13 qualified guard inventory changed')
    stages = report['goal_sequence'].setdefault('stage_observations', [])
    need(step == (6750 if protocol['protocol'] == DATA_PHASE else 0) + 450 * (len(stages) + 1) and step in checkpoints(protocol)
         and not any(o['safety_failed'] or o['round_result'] == 'satisfied' for o in stages), 'complete450 contiguous rounds required')
    base = report['goal_sequence']['reference_baseline_before_training']['confirmed_ids']
    prior = set(base)
    for observation in stages:
        prior.update(observation['confirmed_ids'])
    proof = observe(runtime, protocol, groups, parameter, step, generation, loss, base, prior)
    satisfied = common.classify_round(proof, report['baseline_training_replay_loss'] * 1.05 + .01)
    if not proof['old89_both_passed']:
        proof.update(safety_failed=True, round_result='protection_failed'); satisfied = False
    stages.append(proof)
    report['goal_sequence'].update(report_metadata(protocol, step))
    first = proof['observations'][0]
    report['training_selection']['results'].append({'step': step, 'new_generation': first['new'], 'replay_training_generation': first['old13'],
        'replay_loss': first['replay_training_loss'], 'accepted': satisfied, 'consecutive_passes': 0,
        'same_weight_passes': 2 if satisfied else 0, 'minimum_new_exposures': (step-6750)//2//56 if protocol['protocol'] == DATA_PHASE else (step // 2) // 24, 'tensor_sha256': proof['tensor_sha256']})
    if satisfied:
        report['training_selection']['selected_step'] = step
    runtime.job['learning_progress'] = {'last_check_step': step, 'candidate_mastered': proof['confirmed_count'], 'pending': proof['pending_count'],
        'forgotten_ids': proof['forgotten_since_prior'], 'scope': 'completed research candidate round, not production adoption'}
    return proof


def stopped(report):
    return common.stopped(report)


def finish_metadata(protocol, report):
    report['goal_sequence'].update(report_metadata(protocol, report['completed_steps']))
    report['new_episode_until_validation_elapsed_seconds'] = time.time() - report['started']
    report['new_episode_elapsed_scope'] = 'fresh preparation and optimization until entering once-only VALID; not complete run elapsed'


def validate_resume_history(saved, declared, groups):
    report, step = saved['report'], saved['step']
    selection, expected = report['training_selection'], declared['training_selection']
    meta = report['goal_sequence']
    phase = meta['protocol'] == DATA_PHASE
    continuation = meta['protocol'] in RESUME_PROTOCOLS
    need(meta['protocol'] in {PROTOCOL, DATA_PHASE, *RESUME_PROTOCOLS} and meta['parent_steps'] == (parent_step(meta) if continuation or phase else 0)
         and (not phase or 6750 <= step <= 9450)
         and selection['checkpoints'] == checkpoints(meta) and all(selection[k] == expected[k] for k in
         ('policy_version', 'required_consecutive_passes', 'required_same_weight_passes', 'required_new_exposures', 'replay_generation_ids', 'minimum_replay_generation')),
         'fresh resume policy changed')
    need(report['goal_sequence_sources'] == declared['goal_sequence_sources']
         and saved['original_sha256'] == report['goal_sequence_sources']['original_tensor_sha256']
         and tensor_sha(saved['basis']) == report['replay_subspace']['basis_sha256'], 'resume source/baseline changed')
    capture = read_json(report['goal_sequence_sources']['protection_capture']['path'])
    need(report['replay_subspace'].get('protection_capture_sha256') == content_sha(capture), 'frozen OLD-state basis policy changed')
    old89_ids = [r['id'] for r in capture['records']]
    meta = report['goal_sequence']; stages = meta['stage_observations']
    need([p['step'] for p in stages] == [s for s in checkpoints(meta) if s <= step] and len(selection['results']) == len(stages), 'complete boundary history missing')
    need(not phase or 'reference_baseline_before_training' in meta, 'actual6750 new56 entry baseline has not been observed')
    baseline = meta['reference_baseline_before_training']
    need(baseline['step'] == (6750 if phase else 0) and baseline['tensor_sha256'] == (report['goal_sequence_sources']['lineage_parent']['parent_final_tensor_sha256'] if phase else saved['original_sha256']) and baseline['old89_both_passed']
         and (not phase or set(report['goal_sequence_sources']['prior_training_ids']) <= set(baseline['confirmed_ids'])), 'original B45 evidence changed')
    if phase:
        phase_proofs(report, groups, read_json(report['goal_sequence_sources']['protection_capture']['path'])['records'])
        need(all(meta[k] == declared['goal_sequence'][k] for k in ('protocol_sha256','parent_run_id','parent_steps','total_budget','new_steps_budget')), 'phase protocol/budget binding changed')
    ids = [r['id'] for r in groups['new_training']]; old_ids = selection['replay_generation_ids']
    prior = set(baseline['confirmed_ids']); selected = None
    for index, proof in enumerate(stages):
        need(len(proof['observations']) == 2 and not any(p['safety_failed'] or p['round_result'] == 'satisfied' for p in stages[:index]), 'continued after failure/satisfaction')
        for obs in proof['observations']:
            need([r['id'] for r in obs['new']] == ids and [r['id'] for r in obs['old13']] == old_ids
                 and [r['id'] for r in obs['old89']] == old89_ids and math.isfinite(obs['replay_training_loss']), 'resumed proof inventory changed')
            need(all(type(r['passed']) is bool and r.get('branch_enabled') is False and r.get('reference_used') is False
                 and (not r['passed'] or r.get('complete') is True) for field in ('new', 'old13', 'old89') for r in obs[field]), 'proof raw source/EOS changed')
        confirmed = set.intersection(*[{r['id'] for r in o['new'] if r['passed']} for o in proof['observations']])
        need(proof['confirmed_ids'] == sorted(confirmed) and proof['confirmed_count'] == len(confirmed)
             and proof['pending_ids'] == sorted(set(ids)-confirmed) and proof['pending_count'] == len(ids)-len(confirmed)
             and proof['forgotten_from_reference'] == sorted(set(baseline['confirmed_ids'])-confirmed)
             and proof['forgotten_since_prior'] == sorted(prior-confirmed), 'mastery/pending/forget accounting changed')
        checked = copy.deepcopy(proof); satisfied = common.classify_round(checked, report['baseline_training_replay_loss']*1.05+.01)
        old_ok = all(r['passed'] for o in proof['observations'] for r in o['old89'])
        need(proof['old89_both_passed'] == old_ok, 'old protection boolean changed')
        if not old_ok:checked.update(safety_failed=True,round_result='protection_failed');satisfied=False
        need(all(proof[k] == checked[k] for k in ('safety_failed','round_result','actual_updates','replay_loss_budget','replay_loss_both_passed')), 'round safety result changed')
        row = selection['results'][index];first=proof['observations'][0]
        need(row == {'step':proof['step'],'new_generation':first['new'],'replay_training_generation':first['old13'],'replay_loss':first['replay_training_loss'],
             'accepted':satisfied,'consecutive_passes':0,'same_weight_passes':2 if satisfied else 0,'minimum_new_exposures':(proof['step']-6750)//2//56 if phase else (proof['step']//2)//24,'tensor_sha256':proof['tensor_sha256']}, 'selection proof changed')
        if satisfied:selected=proof['step']
        prior.update(confirmed)
    need(selection['selected_step'] == selected and saved['consecutive_passes'] == 0, 'invalid selected state')
    if continuation:
        lineage = report['goal_sequence_sources']['lineage_parent']
        count = parent_step(meta) // 450
        need(content_sha(selection['results'][:count]) == lineage['parent_history_sha256']
             and content_sha(stages[:count]) == lineage['parent_stage_sha256'], 'immutable complete parent history changed')
        if meta['protocol'] in {RESUME_PROTOCOL_V3, RESUME_PROTOCOL_V4}:
            ancestor = lineage['ancestor_lineage']
            inherited = 9 if meta['protocol'] == RESUME_PROTOCOL_V4 else 6
            need(content_sha(selection['results'][:inherited]) == ancestor['parent_history_sha256']
                 and content_sha(stages[:inherited]) == ancestor['parent_stage_sha256'], 'immutable earlier ancestry changed')
            if meta['protocol'] == RESUME_PROTOCOL_V4:
                original = ancestor['ancestor_lineage']
                need(content_sha(selection['results'][:6]) == original['parent_history_sha256']
                     and content_sha(stages[:6]) == original['parent_stage_sha256'], 'immutable original six-round ancestry changed')
    exposures = (step + 1)//2
    need(saved['new_exposures'] == (phase_exposures(report['goal_sequence_sources'], groups, step) if phase else {s:exposures//24+int(i<exposures%24) for i,s in enumerate(ids)}), 'absolute NEW exposures changed')
    state, param_groups = saved['optimizer']['state'], saved['optimizer']['param_groups']
    need(len(state) == len(param_groups) == 1 and len(param_groups[0]['params']) == 1, 'one complete Adam parameter state required')
    adam = next(iter(state.values()))
    need(set(adam) == {'step', 'exp_avg', 'exp_avg_sq'} and float(adam['step']) == step
         and all(adam[name].dtype == torch.float32 and adam[name].shape == saved['master'].shape
                 and torch.isfinite(adam[name]).all().item() for name in ('exp_avg', 'exp_avg_sq'))
         and torch.isfinite(saved['master']).all().item() and torch.isfinite(saved['basis']).all().item(), 'complete finite master/Adam/basis required')
    pg = param_groups[0]
    need(pg['lr'] == 1e-5 and pg['betas'] == (.9, .999) and pg['eps'] == 1e-6
         and pg['weight_decay'] == 0 and pg['amsgrad'] is False and pg['maximize'] is False, 'Adam mathematical policy changed')
    need(set(state) == set(pg['params']), 'Adam parameter binding changed')
    if step in checkpoints(meta):
        need(stages[-1]['tensor_sha256'] == tensor_sha(saved['master'].to(torch.bfloat16)), 'saved full master differs from sameW boundary proof')


def validated_protection_tokens(runtime, descriptor):
    # The preregistrar must bind this explicit allowlist to immutable old89 files,
    # original B45 tensor, actual fresh baseline receipt and tokenizer source SHA.
    if (descriptor.get('policy') != 'known-old-generated-states-v1'
            or descriptor.get('baseline_tensor_sha256') != descriptor.get('original_tensor_sha256')
            or descriptor.get('original_tensor_sha256') != runtime.original_tensor_sha256
            or descriptor.get('scope') != 'known_old_protection_only'
            or descriptor.get('tokenizer_sha256') != runtime.tokenizer_sha256):
        raise ValueError('old protection source/reference scope mismatch')
    samples, results, allowed = descriptor['records'], descriptor['results'], descriptor['authorized_record_sha256']
    ids = [sample['id'] for sample in samples]
    if (not samples or len(set(ids)) != len(ids) or ids != [row['id'] for row in results]
            or set(ids) != set(allowed)):
        raise ValueError('old protection inventory mismatch')
    eos_ids = runtime.model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    output = []
    for sample, row in zip(samples, results):
        if (allowed[sample['id']] != content_sha(sample) or not sample.get('approved')
                or sample.get('purpose') not in {'training', 'validation'}
                or not verified_source(sample)
                or row.get('branch_enabled') is not False or row.get('reference_used') is not False
                or row.get('thinking') != 'off' or row.get('do_sample') is not False):
            raise ValueError('old protection rights/raw observation mismatch')
        raw = row.get('raw_completion_token_ids')
        if (not isinstance(raw, list) or not raw or len(raw) > row.get('max_new_tokens', 0)
                or any(type(token) is not int or not 0 <= token < len(runtime.tokenizer) for token in raw)
                or raw[-1] not in eos_ids or row.get('completion_tokens') != len(raw)):
            raise ValueError('old protection must end naturally in genuine EOS')
        decoded = runtime.tokenizer.decode(raw, skip_special_tokens=True)
        check = validate_generation_check(provenance(sample).get('generation_check'), default_expected=sample['answer'])
        if (not row.get('passed') or not row.get('complete') or decoded != row.get('decoded_completion_unstripped')
                or decoded.strip() != row.get('answer') or not evaluate_generation(decoded.strip(), check, True)['passed']):
            raise ValueError('old protection output is not independently correct')
        prefix = runtime._tokens(messages(sample), 'off', output_reserve=len(raw))
        if prefix.shape[0] != 1 or prefix.shape[1] + len(raw) > 512:
            raise ValueError('old protection path exceeds frozen 512-token capture bound')
        tokens = torch.cat((prefix, torch.tensor([raw], dtype=prefix.dtype, device=prefix.device)), dim=1)
        labels = tokens.clone(); labels[:, :prefix.shape[1]] = -100
        output.append((sample['id'], tokens, labels))
    return output


def load_continuation(path, protocol):
    """A new bounded episode; original fresh data/source files remain immutable."""
    parent = parent_step(protocol)
    v3 = protocol.get('protocol') == RESUME_PROTOCOL_V3
    v4 = protocol.get('protocol') == RESUME_PROTOCOL_V4
    need(protocol.get('status') == 'FROZEN_CLOSED_PARENT' and protocol.get('parent_steps') == parent
         and type(protocol['parent_steps']) is int and protocol.get('total_steps') == parent + 1350
         and type(protocol['total_steps']) is int and protocol.get('segment_steps') == 450
         and type(protocol['segment_steps']) is int and protocol.get('policy_version') == POLICY,
         'fresh continuation requires its exact closed parent plus three complete450 rounds')
    need(protocol.get('parent_run_id') == ('1d80fd7152eb' if v4 else '25bb1aa918d8' if v3 else 'e3d9ef987253') and re.fullmatch(r'[a-f0-9]{12}', protocol.get('new_run_id', ''))
         and protocol['new_run_id'] != protocol['parent_run_id'], 'only the actual prospective parent may be continued')
    files = protocol.get('files', {})
    required = {'parent_plan', 'parent_result', 'parent_journal', 'parent_checkpoint', 'parent_binding', 'parent_report',
                'records', 'parent_protocol', 'native_manifest', 'native_output_manifest', 'native_trace', 'baseline_old_training'}
    need(set(files) == required, 'complete actual parent closure and native evidence required')
    for entry in files.values():
        need(set(entry) == {'path', 'sha256'} and re.fullmatch(r'[a-f0-9]{64}', entry['sha256'])
             and common.file_sha(common.plain(entry['path'])) == entry['sha256'], 'parent evidence SHA changed')
    need(files['parent_plan']['sha256'] == ('d60e2c189ba98807135dc31b9fb10b2dea3323daedf27cc90c631c52eafd67df' if v4 else '459499c4a14c15f54cd842d41fd64ffbb65432e454c4e9713d36b4af078cadd9' if v3 else '850d3db950657ae1962be975a18994ceec93cb7f0743c110207df2e874c21de9'), 'wrong actual prospective plan')
    ancestor = load_verified(files['parent_protocol']['path'])
    plan, result, binding, report, journal, records = [read_json(files[key]['path']) for key in
        ('parent_plan', 'parent_result', 'parent_binding', 'parent_report', 'parent_journal', 'records')]
    need(plan['protocol'] == (RESUME_PROTOCOL_V3 if v4 else RESUME_PROTOCOL if v3 else PROTOCOL) and ancestor['protocol'] == plan['protocol']
         and plan['goal_sequence_protocol'] == files['parent_protocol']
         and plan['arguments'] == arguments(ancestor) and plan['goal_sequence'] == metadata(ancestor)
         and ancestor['new_run_id'] == protocol['parent_run_id'], 'original fresh ancestry changed')
    need(result.get('protocol') == plan['protocol'] and result.get('plan_sha256') == files['parent_plan']['sha256']
         and result.get('released') is True and result.get('passed') is False and result.get('adopted') is False
         and result.get('production_written') is False and result.get('teacher_api_calls') == 0
         and type(result['teacher_api_calls']) is int and result.get('source_core') == plan['starting_core'], 'parent not closed/released/unadopted')
    final = result['training']
    need(final.get('completed_steps') == parent and type(final['completed_steps']) is int and final.get('promoted') is False
         and final.get('gates') is None and not (common.VALIDATION_FIELDS - {'gates'}) & set(final)
         and final['id'] == protocol['parent_run_id'] and final['previous_version'] == plan['starting_core']
         and final['training_selection'] == report['training_selection'] and final['goal_sequence'] == report['goal_sequence']
         and final['goal_sequence_sources'] == report['goal_sequence_sources'], 'parent read heldout or lost immutable terminal observations')
    need(set(result.get('rows', {})) == set() and result.get('error_type') == 'ValueError'
         and result.get('error') == '隔离候选没有通过正式核心训练门禁', 'only completed TRAIN pending rejection can continue')
    need(content_sha(binding) == protocol['parent_binding_sha256'] and content_sha(report) == protocol['parent_report_sha256']
         and content_sha(records) == protocol['records_sha256'] == binding['records_sha256']
         and binding['arguments'] == arguments(ancestor) and binding['scope'] == 'product'
         and binding['core_version'] == plan['starting_core'] and binding['identity'] == plan['identity'], 'parent binding/report/records mismatch')
    need(records == {k: plan[k] for k in ('groups', 'protection', 'retention_regressions')}
         and journal['id'] == protocol['parent_run_id'] and journal['state'] == 'rejected'
         and journal['binding'] == binding and journal['records'] == records
         and journal['checkpoint']['sha256'] == files['parent_checkpoint']['sha256']
         and journal['checkpoint']['progress']['step'] == parent, 'closed Bun journal and fullstate binding mismatch')
    need(report.get('promoted') is False and not common.VALIDATION_FIELDS & set(report)
         and report['training_selection']['selected_step'] is None and len(report['goal_sequence']['stage_observations']) == parent // 450
         and [s['step'] for s in report['goal_sequence']['stage_observations']] == checkpoints(ancestor)
         and all(s['round_result'] in {'pending', 'partial'} and s['safety_failed'] is False
                 and s['old89_both_passed'] is True and s['old13_both_passed'] is True
                 and not s['forgotten_from_reference'] and not s['forgotten_since_prior']
                 for s in report['goal_sequence']['stage_observations'])
         and report['goal_sequence']['stage_observations'][-1]['pending_count'] > 0,
         'parent safety failed, satisfied, or incomplete; cannot continue')
    if v3 or v4:
        need(protocol.get('parent_goal_sequence_sources_sha256') == content_sha(report['goal_sequence_sources'])
             and protocol.get('ancestor_lineage') == report['goal_sequence_sources']['lineage_parent'], 'explicit parent source/ancestor anchors changed')
        need(report['goal_sequence_sources'] == source_metadata(ancestor)
             and report['goal_sequence']['protocol'] == ancestor['protocol']
             and all(report['goal_sequence'][k] == v for k, v in metadata(ancestor).items())
             and content_sha(report['goal_sequence']['stage_observations'][:parent_step(ancestor)//450]) == report['goal_sequence_sources']['lineage_parent']['parent_stage_sha256']
             and content_sha(report['training_selection']['results'][:parent_step(ancestor)//450]) == report['goal_sequence_sources']['lineage_parent']['parent_history_sha256'], 'immutable earlier fullstate ancestry changed')
    need(content_sha(report['training_selection']['results']) == protocol['parent_history_sha256'], 'complete parent history SHA changed')
    closure = protocol.get('closure', {})
    need(closure == {'exit_code': 1, 'released': True, 'adopted': False, 'production_written': False, 'validation_reads': 0,
                    'final_step': parent, 'parent_journal_state': 'rejected', 'active_core': plan['starting_core'],
                    'error_type': result['error_type'], 'error': result['error']}, 'exact safe pending closure required')
    old_sources, new_sources = protocol['parent_source_sha256'], protocol['new_source_sha256']
    snapshots = protocol['parent_source_snapshot']
    allowed = {'src/luban/goal_fresh.py', 'src/luban/goal_sequence.py', 'src/luban/evolution.py',
               'src/luban/training_checkpoint.py', 'scripts/train_evolution_goal.py'}
    need(old_sources == plan['source_sha256'] == ancestor['new_source_sha256'] and set(snapshots) == set(old_sources)
         and set(new_sources) == set(old_sources), 'all original38 and all new38 source bindings required')
    for name, digest in old_sources.items():
        entry = snapshots[name]
        need(set(entry) == {'path', 'sha256'} and entry['sha256'] == digest
             and common.file_sha(common.plain(entry['path'])) == digest
             and (name in allowed or new_sources[name] == digest), 'unapproved original source/math/native change')
    old_impl, new_impl = protocol['parent_implementation_sha256'], protocol['new_implementation_sha256']
    need(old_impl == binding['implementation_sha256'] == ancestor['new_implementation_sha256']
         and set(new_impl) == set(old_impl) and all(new_sources['src/luban/' + k] == v for k, v in new_impl.items()),
         'complete22 implementation ancestry changed')
    baseline = read_json(files['baseline_old_training']['path'])
    need(baseline == result['baseline_old_training'] and len(baseline) == 83
         and [r['id'] for r in baseline] == [r['id'] for r in plan['old_training_original']], 'original83 baseline changed')
    need(all(re.fullmatch(r'[a-f0-9]{64}', v) for v in new_sources.values()), 'new source digests missing')
    need(all(protocol.get(k, ancestor[k]) == ancestor[k] for k in ('identity', 'source_core', 'original_tensor_sha256', 'protection_capture', 'replay_generation_ids', 'new_counts', 'old_counts', 'acceptance_count', 'model_realpath')), 'original B45 reference/capture/cohort changed')
    need(protocol['parameter'] == report['parameter'] and protocol['dataset_sha256'] == report['dataset_sha256']
         and protocol['binary_sha256'] == report['training_engine']['binary_sha256']
         and protocol['engine'] == report['training_engine']['engine']
         and type(closure['validation_reads']) is int and type(closure['exit_code']) is int, 'parent native mathematical identity or closure type changed')
    from .training_checkpoint import _owner_alive
    need(not _owner_alive(journal), 'parent process still owns the rejected journal; wait for actual exit')
    inherited = copy.deepcopy(ancestor)
    inherited.update(protocol)
    inherited.update(_protocol_path=str(path), _protocol_sha256=common.file_sha(path), _parent_report=report, _parent_binding=binding)
    return inherited


def derive_payload(protocol, closed, native, new_binding, records):
    if protocol.get('protocol') == DATA_PHASE:
        return derive_phase_payload(protocol, closed, native, new_binding, records)
    """Preserve all actual optimizer/RNG states; only new session/source/budget change."""
    need(protocol.get('protocol') in RESUME_PROTOCOLS and set(closed) == {'binding', 'payload', 'rng', 'mps_rng'}, 'explicit complete continuation envelope required')
    parent = parent_step(protocol)
    old = closed['payload']; report = old['report']
    need(closed['binding'] == protocol['_parent_binding'] and report == protocol['_parent_report']
         and content_sha(report) == protocol['parent_report_sha256'], 'actual checkpoint envelope mismatch')
    need(set(old) == {'format', 'id', 'parameter', 'dataset_sha256', 'original_sha256', 'step', 'master', 'optimizer',
                     'basis', 'new_exposures', 'kl_total', 'kl_maximum', 'consecutive_passes', 'report'}
         and old['format'] == 1 and type(old['format']) is int and old['step'] == parent and type(old['step']) is int
         and old['id'] == protocol['parent_run_id'] and report['id'] == old['id'] and report['steps'] == parent
         and old['parameter'] == protocol['parameter'] and old['dataset_sha256'] == report['dataset_sha256'] == protocol['dataset_sha256']
         and old['original_sha256'] == protocol['original_tensor_sha256'] and math.isfinite(old['kl_total'])
         and math.isfinite(old['kl_maximum']), 'actual complete terminal payload required')
    previous_binding = closed['binding']
    need({k:v for k,v in previous_binding.items() if k not in {'arguments','implementation_sha256'}}
         == {k:v for k,v in new_binding.items() if k not in {'arguments','implementation_sha256'}}
         and new_binding['arguments'] == arguments(protocol)
         and new_binding['implementation_sha256'] == protocol['new_implementation_sha256']
         and content_sha(records) == protocol['records_sha256'] == new_binding['records_sha256'], 'unapproved binding/material change')
    validate_resume_history(old, report, records['groups'])
    common.validate_state(old, native, protocol, expected_step=parent)
    for key in ('rng', 'mps_rng'):
        value = closed[key]
        need(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.uint8
             and value.ndim == 1 and value.numel() > 0, 'actual CPU/MPS RNG state required')
    derived = copy.deepcopy(old)
    derived.update(id=protocol['new_run_id'])
    inherited = copy.deepcopy(report['goal_sequence'])
    inherited.update(report_metadata(protocol, parent))
    derived['report'].update(id=protocol['new_run_id'], version='core-' + protocol['new_run_id'], steps=parent + 1350,
        started=time.time(), parent_started=report['started'], goal_sequence=inherited,
        goal_sequence_sources=source_metadata(protocol), elapsed_scope='new episode wall time; parent duration remains immutable')
    derived['report']['training_selection'].update(checkpoints=checkpoints(protocol))
    return derived


def phase_groups(protocol):
    parent = read_json(protocol['files']['parent_plan']['path'])
    material = read_json(protocol['new_files']['new_material']['path'])
    return {**copy.deepcopy(parent['groups']), 'new_training': copy.deepcopy(material['TRAIN']),
            'new_validation': copy.deepcopy(material['VALID'])}


def phase_sources(protocol):
    value = copy.deepcopy(protocol['_parent_report']['goal_sequence_sources'])
    value.update(protocol_sha256=protocol['_protocol_sha256'], files=protocol['_files'],
        new_implementation=protocol['new_implementation_sha256'], new_source_sha256=protocol['new_source_sha256'],
        original_formal_protocol=protocol['original_formal_protocol'], original_formal_files=protocol['original_formal_files'],
        formal_manifest=protocol['material_manifest'],
        prior_training_ids=[r['id'] for r in protocol['_records']['groups']['new_training'][:24]],
        inherited_new_exposures=protocol['parent_new_exposures'],
        scope='new frozen data phase after observed V4 VALID failure; parent6750 full state; originalB45 reference/basis')
    value['lineage_parent'] = {k: copy.deepcopy(protocol[k]) for k in ('parent_run_id', 'files',
        'parent_binding_sha256', 'parent_report_sha256', 'parent_history_sha256', 'parent_stage_sha256',
        'parent_source_sha256', 'parent_implementation_sha256', 'basis_sha256', 'reference_sha256',
        'parent_final_tensor_sha256', 'ancestor_lineage', 'parent_goal_sequence_sources_sha256')}
    value['lineage_parent']['protocol_sha256'] = protocol['_protocol_sha256']
    return value


def phase_plan_fields(protocol):
    parent = read_json(protocol['files']['parent_plan']['path'])
    files = copy.deepcopy(parent['formal_files']); files.update(protocol['new_files'])
    return {'formal_protocol': protocol['_protocol_path'], 'formal_protocol_sha256': protocol['_protocol_sha256'],
        'formal_files': files, 'materials': files['new_material']['path'], 'materials_sha256': files['new_material']['sha256'],
        'groups': protocol['_records']['groups'], 'acceptance': protocol['_records']['retention_regressions'],
        'retention_regressions': protocol['_records']['retention_regressions'],
        'original_formal_protocol': protocol['original_formal_protocol'], 'original_formal_files': protocol['original_formal_files'],
        'budget_policy': 'new56TRAIN/12VALID/12ACCEPT; parent6750 archived; at most six complete450 new-data rounds to9450; TRAIN-only selection then one fresh heldout'}


def load_data_phase(path, protocol):
    need(protocol.get('status') == 'FROZEN_NEW_DATA_PHASE' and protocol.get('parent_steps') == 6750
         and type(protocol['parent_steps']) is int and protocol.get('total_steps') == 9450
         and type(protocol['total_steps']) is int and protocol.get('segment_steps') == 450
         and type(protocol['segment_steps']) is int and protocol.get('policy_version') == POLICY
         and protocol.get('parent_run_id') == '03fd69e9f865'
         and re.fullmatch(r'[a-f0-9]{12}', protocol.get('new_run_id', ''))
         and protocol['new_run_id'] != protocol['parent_run_id'], 'exact actual6750 new-data phase required')
    files = protocol['files']
    need(set(files) == {'parent_plan','parent_result','parent_journal','parent_checkpoint','parent_binding','parent_report',
        'records','parent_protocol','native_manifest','native_output_manifest','native_trace','baseline_old_training'}
        and set(protocol['new_files']) == {'new_material','acceptance'}, 'complete parent/new material inventory required')
    for entry in [*files.values(), *protocol['new_files'].values()]:
        need(set(entry) == {'path','sha256'} and re.fullmatch(r'[a-f0-9]{64}', entry['sha256'])
             and common.file_sha(common.plain(entry['path'])) == entry['sha256'], 'parent/new material evidence changed')
    need(files['parent_plan']['sha256'] == 'ffc07d6604310da5eab9c188a3bedba647a11bc069e959f0a61c272744275f53'
         and files['parent_result']['sha256'] == 'c32c0b00f14e827790d88e94fa999a451b6330b1474eb07b65d7dc2d9170ec91'
         and files['parent_checkpoint']['sha256'] == '9c75504a277067a911b987e12aa74f169ac143f98646af0d704d57daa70bd632', 'not the actual optimized6750 closure')
    ancestor = load_verified(files['parent_protocol']['path'])
    plan, result, binding, report, journal, records = [read_json(files[k]['path']) for k in
        ('parent_plan','parent_result','parent_binding','parent_report','parent_journal','records')]
    need(ancestor['protocol'] == RESUME_PROTOCOL_V4 and ancestor['new_run_id'] == protocol['parent_run_id']
         and plan['protocol'] == RESUME_PROTOCOL_V4 and plan['goal_sequence_protocol'] == files['parent_protocol']
         and plan['arguments'] == arguments(ancestor) and plan['goal_sequence'] == metadata(ancestor), 'actual V4 ancestry changed')
    need(result.get('released') is True and result.get('passed') is False and result.get('adopted') is False
         and result.get('production_written') is False and result.get('teacher_api_calls') == 0
         and type(result['teacher_api_calls']) is int and not result.get('rows')
         and result['plan_sha256'] == files['parent_plan']['sha256']
         and result.get('error_type') == 'ValueError' and result.get('error') == '隔离候选没有通过正式核心训练门禁', 'parent not an actual released formal rejection')
    final = result['training']; stages = report['goal_sequence']['stage_observations']
    need(final['completed_steps'] == 6750 and final['promoted'] is False and len(final['new_generation']) == 8
         and sum(r['passed'] for r in final['new_generation']) == 1
         and final['gates']['heldout_new_generation'] is False
         and all(v is True for k,v in final['gates'].items() if k != 'heldout_new_generation')
         and not common.VALIDATION_FIELDS & set(report) and report['promoted'] is False
         and len(stages) == 15 and [s['step'] for s in stages] == list(range(450,6751,450))
         and report['training_selection']['selected_step'] == 6750 and stages[-1]['round_result'] == 'satisfied'
         and stages[-1]['confirmed_count'] == 24 and not stages[-1]['pending_count']
         and all(not s['safety_failed'] and s['old89_both_passed'] is True and s['old13_both_passed'] is True
                 and not s['forgotten_from_reference'] and not s['forgotten_since_prior'] for s in stages)
         and final['goal_sequence'] == report['goal_sequence'] and final['training_selection'] == report['training_selection']
         and final['goal_sequence_sources'] == report['goal_sequence_sources'] == source_metadata(ancestor), 'parent optimized state or separate failed VALID evidence changed')
    need(content_sha(binding) == protocol['parent_binding_sha256'] and content_sha(report) == protocol['parent_report_sha256']
         and content_sha(records) == protocol['records_sha256'] == binding['records_sha256']
         and content_sha(stages) == protocol['parent_stage_sha256']
         and content_sha(report['training_selection']['results']) == protocol['parent_history_sha256']
         and content_sha(report['goal_sequence_sources']) == protocol['parent_goal_sequence_sources_sha256']
         and protocol['ancestor_lineage'] == report['goal_sequence_sources']['lineage_parent']
         and records == {k: plan[k] for k in ('groups','protection','retention_regressions')}
         and journal['binding'] == binding and journal['records'] == records and journal['id'] == protocol['parent_run_id']
         and journal['state'] == 'rejected' and journal['checkpoint']['sha256'] == files['parent_checkpoint']['sha256']
         and journal['checkpoint']['progress']['step'] == 6750 and binding['arguments'] == arguments(ancestor)
         and binding['scope'] == 'product', 'exact parent report/journal/source archive required')
    from .training_checkpoint import _owner_alive
    need(not _owner_alive(journal), 'actual parent journal still owned')
    need(protocol['original_formal_protocol'] == {'path': plan['formal_protocol'], 'sha256': plan['formal_protocol_sha256']}
         and protocol['original_formal_protocol']['sha256'] == 'a94531b7802ee06b6e286d434be052c38b3fb1ff8594e019c768265d58866d3f'
         and protocol['original_formal_files'] == plan['formal_files'], 'original formal/old14 material anchors changed')
    old_sources, new_sources = protocol['parent_source_sha256'], protocol['new_source_sha256']
    allowed = {'src/luban/goal_fresh.py','src/luban/goal_sequence.py','src/luban/evolution.py',
               'src/luban/training_checkpoint.py','scripts/train_evolution_goal.py'}
    need(old_sources == plan['source_sha256'] == ancestor['new_source_sha256'] and len(old_sources) == 38
         and set(new_sources) == set(old_sources) and set(protocol['parent_source_snapshot']) == set(old_sources)
         and protocol['parent_implementation_sha256'] == binding['implementation_sha256'] == ancestor['new_implementation_sha256']
         and len(binding['implementation_sha256']) == 22
         and set(protocol['new_implementation_sha256']) == set(binding['implementation_sha256']), 'complete38/22 source ancestry required')
    for name, digest in old_sources.items():
        entry = protocol['parent_source_snapshot'][name]
        need(entry == {'path':entry['path'],'sha256':digest} and common.file_sha(common.plain(entry['path'])) == digest
             and (name in allowed or new_sources[name] == digest)
             and re.fullmatch(r'[a-f0-9]{64}',new_sources[name]), 'unapproved source/native mathematical change')
    need(all(new_sources['src/luban/'+n] == v for n,v in protocol['new_implementation_sha256'].items()), 'implementation22 mapping changed')
    material = read_json(protocol['new_files']['new_material']['path']); acceptance = read_json(protocol['new_files']['acceptance']['path'])
    manifest = read_json(protocol['material_manifest']['path'])
    need(protocol['material_manifest'] == {'path': str(path.parent / protocol['manifest']), 'sha256': protocol['material_manifest']['sha256']}
         and common.file_sha(common.plain(protocol['material_manifest']['path'])) == protocol['material_manifest']['sha256']
         and manifest['files'] == {k: {**v, 'bytes': Path(v['path']).stat().st_size} for k,v in {**plan['formal_files'],**protocol['new_files']}.items()}
         and manifest['original_formal_protocol'] == protocol['original_formal_protocol'], 'current five-file material manifest changed')
    need(set(material) == {'TRAIN','VALID'} and len(material['TRAIN']) == 56 and len(material['VALID']) == 12
         and len(acceptance) == 12 and material['TRAIN'][:24] == plan['groups']['new_training'], '56/12/12 or unchanged original24 changed')
    from .retention import _validate_groups, _validate_record, normalized_context
    contexts, families = set(), {'TRAIN':set(), 'VALID':set(), 'ACCEPTANCE':set()}
    for key, rows in [('TRAIN',material['TRAIN']),('VALID',material['VALID']),('ACCEPTANCE',acceptance)]:
        for record in rows:
            _validate_record(record, generation_only=key == 'ACCEPTANCE')
            p = provenance(record); check = validate_generation_check(p.get('generation_check'),default_expected=record['answer'])
            need(p.get('draft') is False and p.get('teacher_calls') == 0 and p.get('teacher_generated') is False
                 and check.get('type') == 'semantic' and check.get('validator') == 'goal_topics'
                 and check.get('expected') in {'framework','fast_slow','own_mastery','policy'} and is_finite_generation_check(check)
                 and evaluate_generation(record['answer'],check,True)['passed'], 'new data rights/finite frozen sentence contract changed')
            need((record['purpose'] == 'training' and eligible_learning_sample(record)) if key == 'TRAIN' else record['purpose'] == 'validation', 'new data split/rights changed')
            context = normalized_context(record)
            need(context not in contexts and isinstance(p.get('split_family'),str) and p['split_family'], 'duplicate phase context/family')
            contexts.add(context); families[key].add(p['split_family'])
    need(not families['TRAIN'] & families['VALID'] and not families['TRAIN'] & families['ACCEPTANCE']
         and not families['VALID'] & families['ACCEPTANCE'] and len({r['id'] for r in material['TRAIN']+material['VALID']+acceptance}) == 80, 'new data split family/ID leak')
    for row, original in zip(material['TRAIN'][24:32],plan['groups']['new_validation']):
        p = provenance(row)
        need(p.get('source_original_record') == {'id':original['id'],'sha256':content_sha(original)}
             and p.get('parent_acceptance_sha256') == files['parent_result']['sha256']
             and p.get('now_known_TRAIN_not_unseen') is True and p.get('exposed_as_V4_VALID') is True
             and row['prompt'] == original['prompt'] and row['answer'] == original['answer']
             and messages(row) == messages(original) and p['generation_check'] == provenance(original)['generation_check'], 'known oldVALID8 derivatives changed provenance/context/judge')
    fresh_contexts = {normalized_context(r) for r in material['TRAIN'][32:]+material['VALID']+acceptance}
    prior_contexts = {normalized_context(r) for r in plan['groups']['new_training']+plan['groups']['new_validation']+plan['acceptance']}
    need(not fresh_contexts & prior_contexts, 'observed/old acceptance context cannot be called new')
    review = protocol['material_contract_review']
    need(review.get('completed') is True and review.get('material_sha256') == protocol['new_files']['new_material']['sha256']
         and review.get('acceptance_sha256') == protocol['new_files']['acceptance']['sha256'], 'exact new material contract review missing')
    groups = phase_groups(protocol); _validate_groups(groups)
    new_records = {'groups':groups,'protection':records['protection'],'retention_regressions':acceptance}
    need(content_sha(new_records) == protocol['new_records_sha256'] and content_sha(groups) == protocol['new_dataset_sha256']
         and protocol['parameter'] == report['parameter'] and protocol['dataset_sha256'] == report['dataset_sha256']
         and protocol['binary_sha256'] == report['training_engine']['binary_sha256'] and protocol['engine'] == report['training_engine']['engine']
         and protocol['basis_sha256'] == report['replay_subspace']['basis_sha256']
         and protocol['reference_sha256'] == ancestor['reference_sha256']
         and protocol['parent_final_tensor_sha256'] == stages[-1]['tensor_sha256'], 'new dataset or original math/reference/basis changed')
    baseline = read_json(files['baseline_old_training']['path'])
    need(baseline == result['baseline_old_training'] and len(baseline) == 83, 'old83 baseline archive changed')
    base_files = copy.deepcopy(ancestor['_files']); base_files.update(protocol['new_files'])
    inherited = copy.deepcopy(ancestor); inherited.update(protocol)
    inherited.update(_protocol_path=str(path), _protocol_sha256=common.file_sha(path), _parent_report=report,
        _parent_binding=binding, _parent_protocol=ancestor, _records=new_records, _files=base_files)
    return inherited


def phase_exposures(sources, groups, step):
    need(type(step) is int and 6750 <= step <= 9450, 'invalid data-phase exposure step')
    ids = [r['id'] for r in groups['new_training']]
    need(len(ids) == 56 and set(sources['inherited_new_exposures']) == set(ids[:24]), 'exact inherited24 exposure ledger required')
    values = {i:sources['inherited_new_exposures'].get(i,0) for i in ids}
    need(all(type(v) is int and v >= 0 for v in values.values()), 'invalid inherited exposure count')
    for ordinal in range(6750,step):
        if ordinal % 2 == 0:values[ids[(ordinal//2)%56]] += 1
    return values


def derive_phase_payload(protocol, closed, native, new_binding, records):
    need(protocol['protocol'] == DATA_PHASE and set(closed) == {'binding','payload','rng','mps_rng'}, 'complete new-data continuation envelope required')
    old = closed['payload']; report = old['report']; binding = closed['binding']
    need(binding == protocol['_parent_binding'] and report == protocol['_parent_report']
         and content_sha(report) == protocol['parent_report_sha256'] and old['step'] == 6750
         and old['id'] == protocol['parent_run_id'] and old['dataset_sha256'] == protocol['dataset_sha256']
         and old['new_exposures'] == protocol['parent_new_exposures']
         and old['original_sha256'] == protocol['original_tensor_sha256'], 'actual6750 complete optimized payload changed')
    validate_resume_history(old, report, read_json(protocol['files']['records']['path'])['groups'])
    common.validate_state(old,native,protocol,expected_step=6750)
    need({k:v for k,v in binding.items() if k not in {'arguments','implementation_sha256','records_sha256'}}
         == {k:v for k,v in new_binding.items() if k not in {'arguments','implementation_sha256','records_sha256'}}
         and new_binding['arguments'] == arguments(protocol) and new_binding['implementation_sha256'] == protocol['new_implementation_sha256']
         and records == protocol['_records'] and content_sha(records) == protocol['new_records_sha256'] == new_binding['records_sha256'], 'unapproved new data session binding change')
    for key in ('rng','mps_rng'):
        value = closed[key]
        need(isinstance(value,torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.uint8
             and value.ndim == 1 and value.numel() > 0, 'complete original RNG state required')
    q = copy.deepcopy(old); q.update(id=protocol['new_run_id'],dataset_sha256=protocol['new_dataset_sha256'],consecutive_passes=0)
    q['new_exposures'] = phase_exposures(phase_sources(protocol),records['groups'],6750)
    clean = q['report']
    for key in common.VALIDATION_FIELDS | {'rejection_reason','publication_state','elapsed_seconds','sampled_memory_maxima','new_episode_until_validation_elapsed_seconds','new_episode_elapsed_scope','validation_scope','resumed_from_steps'}:clean.pop(key,None)
    clean.update(id=q['id'],version='core-'+q['id'],steps=9450,completed_steps=6750,started=time.time(),parent_started=report['started'],
        dataset_sha256=protocol['new_dataset_sha256'],promoted=False,goal_sequence={**report_metadata(protocol,6750),'stage_observations':[]},
        goal_sequence_sources=phase_sources(protocol),split_ids={k:[r['id'] for r in v] for k,v in records['groups'].items()},
        sample_sha256={r['id']:content_sha(r) for v in records['groups'].values() for r in v},
        elapsed_scope='new data episode wall time; immutable parent15 history and failed VALID are separate archived evidence')
    clean['goal_sequence']['parent_archive'] = phase_archive(protocol)
    clean['training_selection'].update(checkpoints=checkpoints(protocol),results=[],selected_step=None,
        scope='56TRAIN-only complete450 double checks; original24 mastery mandatory; fresh VALID once after first all56 satisfaction')
    clean['training_engine'].pop('artifacts',None); clean['training_engine']['segments']=[]
    clean['memory_samples']=[]
    return q


def initial_phase_saved(runtime, protocol, session, parameter, previous, groups, generation, loss):
    from . import luban_training as engine
    files=protocol['files'];closed=torch.load(files['parent_checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
    fixture=engine.validate_fixture(Path(files['native_manifest']['path']).parent,files['native_manifest']['sha256'])
    output=Path(files['native_output_manifest']['path']).parent
    engine.validate_output(output,fixture,protocol['binary_sha256'])
    payload=derive_phase_payload(protocol,closed,engine.load_checkpoint(output,fixture),session.binding,session.body['records'])
    current=parameter.detach().clone()
    try:
        with torch.no_grad():parameter.copy_(payload['master'].to(parameter.device))
        proof=observe(runtime,protocol,groups,parameter,6750,generation,loss,
            reference=payload['report']['goal_sequence_sources']['prior_training_ids'],prior=payload['report']['goal_sequence_sources']['prior_training_ids'])
        need(proof['tensor_sha256']==protocol['parent_final_tensor_sha256'] and not proof['forgotten_from_reference']
             and proof['old13_both_passed'] and proof['old89_both_passed']
             and all(o['replay_training_loss'] <= payload['report']['baseline_training_replay_loss']*1.05+.01 for o in proof['observations']), 'actual6750 new-data baseline lost prior24/old89/13 or replay loss')
        payload['report']['goal_sequence']['reference_baseline_before_training']=proof
        validate_resume_history(payload,payload['report'],groups)
    finally:
        with torch.no_grad():parameter.copy_(current)
    torch.set_rng_state(closed['rng'])
    if str(runtime.device)=='mps':torch.mps.set_rng_state(closed['mps_rng'])
    session.save(payload,{'step':6750,'phase':'new-data-entry-baseline','steps':9450})
    return session.load()


def validation_reference(protocol, previous):
    if protocol.get('protocol') != DATA_PHASE:return previous
    closed=torch.load(protocol['files']['parent_checkpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
    need(closed['binding']==protocol['_parent_binding'] and closed['payload']['report']==protocol['_parent_report'], 'phase validation reference archive changed')
    value=closed['payload']['master'].to(torch.bfloat16)
    need(tensor_sha(value)==protocol['parent_final_tensor_sha256'], 'phase validation comparator is not actual6750')
    return value


def phase_observation_binding(plan, report, protocol):
    """TRAIN-only report proof; final heldout fields do not select checkpoints."""
    need(plan['protocol']==DATA_PHASE and report['goal_sequence_sources']==phase_sources(protocol)
         and report['dataset_sha256']==protocol['new_dataset_sha256'] and report['steps']==9450
         and report['goal_sequence']['phase_steps']==report['completed_steps']-6750
         and report['goal_sequence']['cumulative_steps']==report['completed_steps'], 'new phase report identity/budget changed')
    phase_proofs(report, plan['groups'], plan['protection'])
    stages=report['goal_sequence']['stage_observations']
    return {'protocol':DATA_PHASE,'phase_steps':report['completed_steps']-6750,'cumulative_steps':report['completed_steps'],
            'prior_training_count':24,'new_training_count':56,'stage_count':len(stages),'parent_history_sha256':protocol['parent_history_sha256']}


def phase_archive(protocol):
    return {'parent_run_id': protocol['parent_run_id'], 'parent_steps': 6750,
            **{k: protocol[k] for k in ('parent_report_sha256','parent_history_sha256','parent_stage_sha256','parent_goal_sequence_sources_sha256')},
            'parent_checkpoint_sha256': protocol['files']['parent_checkpoint']['sha256'],
            'parent_result_sha256': protocol['files']['parent_result']['sha256']}


def phase_proofs(report, groups, protection):
    meta=report['goal_sequence'];source=report['goal_sequence_sources'];lineage=source['lineage_parent']
    archive={'parent_run_id':lineage['parent_run_id'],'parent_steps':6750,
        **{k:lineage[k] for k in ('parent_report_sha256','parent_history_sha256','parent_stage_sha256','parent_goal_sequence_sources_sha256')},
        'parent_checkpoint_sha256':lineage['files']['parent_checkpoint']['sha256'], 'parent_result_sha256':lineage['files']['parent_result']['sha256']}
    need(meta['parent_archive']==archive and meta['parent_steps']==6750 and meta['total_budget']==9450
         and meta['new_steps_budget']==2700 and len(groups['new_training'])==56
         and source['prior_training_ids']==[r['id'] for r in groups['new_training'][:24]], 'phase archive/budget/prior24 inventory changed')
    baseline=meta['reference_baseline_before_training'];stages=meta['stage_observations'];selection=report['training_selection']
    # During interrupted optimization completed_steps is the entry value; stage history
    # nevertheless binds the last complete boundary and no heldout can select it.
    expected=[s for s in checkpoints(meta) if s<=meta['cumulative_steps']]
    need(baseline['step']==6750 and baseline['tensor_sha256']==lineage['parent_final_tensor_sha256']
         and baseline['label']=='data-phase-entry6750' and [s['step'] for s in stages]==expected
         and meta['phase_steps']==meta['cumulative_steps']-6750
         and meta['new_steps_executed']==meta['phase_steps'] and len(selection['results'])==len(stages), 'phase entry/boundary history changed')
    replay={r['id']:r for r in groups['replay_training']};old13=[replay[i] for i in selection['replay_generation_ids']]
    inventories={'new':groups['new_training'],'old13':old13,'old89':protection}
    reference=set(baseline['confirmed_ids']);prior=set(reference);selected=None
    for index,proof in enumerate([baseline,*stages]):
        need(len(proof['observations'])==2 and proof['validation_rows_accessed']==0
             and proof['reference_used'] is False and proof['branch_enabled'] is False
             and proof['label']==('data-phase-entry6750' if index==0 else 'data-phase-complete450')
             and re.fullmatch(r'[a-f0-9]{64}',proof['tensor_sha256']), 'phase proof source/double observation changed')
        confirmed_sets=[]
        for observed in proof['observations']:
            need(set(observed)=={'new','old13','old89','replay_training_loss'} and math.isfinite(observed['replay_training_loss']), 'phase double observation schema/loss changed')
            for field,samples in inventories.items():
                need([r['id'] for r in observed[field]]==[r['id'] for r in samples], 'phase double row inventory changed')
                for row,sample in zip(observed[field],samples):
                    check=validate_generation_check(provenance(sample).get('generation_check'),default_expected=sample['answer'])
                    need(type(row['passed']) is bool and row['passed']==evaluate_generation(row['answer'],check,row.get('complete') is True)['passed']
                         and row.get('reference_used') is False and row.get('branch_enabled') is False
                         and (not row['passed'] or row.get('complete') is True), 'phase generation finite judge/source/EOS changed')
            confirmed_sets.append({r['id'] for r in observed['new'] if r['passed']})
        confirmed=set.intersection(*confirmed_sets);ids={r['id'] for r in groups['new_training']}
        need(proof['confirmed_ids']==sorted(confirmed) and proof['confirmed_count']==len(confirmed)
             and proof['pending_ids']==sorted(ids-confirmed) and proof['pending_count']==56-len(confirmed)
             and proof['old89_both_passed']==all(r['passed'] for o in proof['observations'] for r in o['old89'])
             and proof['old13_both_passed']==all(r['passed'] for o in proof['observations'] for r in o['old13']), 'phase correctness accounting changed')
        if index==0:
            need(proof['forgotten_from_reference']==sorted(set(source['prior_training_ids'])-confirmed)
                 and proof['forgotten_since_prior']==sorted(set(source['prior_training_ids'])-confirmed)
                 and set(source['prior_training_ids'])<=confirmed and proof['old89_both_passed'] and proof['old13_both_passed']
                 and all(o['replay_training_loss']<=report['baseline_training_replay_loss']*1.05+.01 for o in proof['observations']), 'phase actual6750 baseline lost prior knowledge')
            continue
        need(selected is None and not any(s['safety_failed'] for s in stages[:index-1])
             and proof['forgotten_from_reference']==sorted(reference-confirmed) and proof['forgotten_since_prior']==sorted(prior-confirmed), 'phase continued after satisfaction/safety failure or forgetting accounting changed')
        checked=copy.deepcopy(proof);accepted=common.classify_round(checked,report['baseline_training_replay_loss']*1.05+.01)
        if not proof['old89_both_passed']:checked.update(safety_failed=True,round_result='protection_failed');accepted=False
        need(all(proof[k]==checked[k] for k in ('safety_failed','round_result','actual_updates','replay_loss_budget','replay_loss_both_passed')), 'phase complete450 safety result changed')
        first=proof['observations'][0]
        need(selection['results'][index-1]=={'step':proof['step'],'new_generation':first['new'],'replay_training_generation':first['old13'],
            'replay_loss':first['replay_training_loss'],'accepted':accepted,'consecutive_passes':0,'same_weight_passes':2 if accepted else 0,
            'minimum_new_exposures':(proof['step']-6750)//2//56,'tensor_sha256':proof['tensor_sha256']}, 'phase selection mixed old/new dataset or changed TRAIN evidence')
        if accepted:selected=proof['step']
        prior.update(confirmed)
    need(selection['selected_step']==selected, 'phase selected checkpoint not established solely by current TRAIN56')
