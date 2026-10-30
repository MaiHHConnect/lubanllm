"""Real fresh native foundation validation; optional authorized local adoption."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import time

from train_evolution_goal import data_service, native_rows, all_passed, need, read, sha, write_new, plain, goal_observation_binding, terminal_selected_confirmation, CONTINUATION_PROTOCOLS, CONTINUATION_PROTOCOL_V3, FRESH_PROTOCOL, FRESH_RESUME_PROTOCOL, FRESH_PROTOCOLS

FRESH_RESUME_PROTOCOL_V3 = 'evolution-goal-prospective-fixed450-v3'
FRESH_RESUME_PROTOCOL_V4 = 'evolution-goal-prospective-fixed450-v4'
FRESH_RESUME_PROTOCOLS = {FRESH_RESUME_PROTOCOL, FRESH_RESUME_PROTOCOL_V3, FRESH_RESUME_PROTOCOL_V4}
DATA_PHASE = 'evolution-goal-new-data-fixed450-v1'
FRESH_PROTOCOLS = FRESH_PROTOCOLS | FRESH_RESUME_PROTOCOLS | {DATA_PHASE}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def sequence_sources(sequence, parent_report):
    files = sequence['files']
    sources = {'parent_history_sha256': digest(parent_report['training_selection']['results']),
        'parent_final_tensor_sha256': sequence['parent_final_tensor_sha256'], 'original_tensor_sha256': sequence['original_tensor_sha256'],
        'parent_checkpoint_sha256': files['parent_checkpoint']['sha256'], 'parent_binding_sha256': sequence['parent_binding_sha256'],
        'parent_report_sha256': sequence['parent_report_sha256'], 'parent_journal_sha256': files['parent_journal']['sha256'],
        'native_output_manifest_sha256': files['native_output_manifest']['sha256'],
        'parent_implementation_sha256': sequence['parent_implementation_sha256'], 'new_implementation_sha256': sequence['new_implementation_sha256']}
    if sequence['protocol'] == CONTINUATION_PROTOCOL_V3:
        immediate = {**sources, 'parent_goal_sequence_sources_sha256': sequence['parent_goal_sequence_sources_sha256'],
                     'protocol_sha256': sequence['files']['parent_sequence_protocol']['sha256'],
                     'inherited_observations_sha256': sequence['inherited_observations_sha256']}
        sources = {**parent_report['goal_sequence_sources'], 'new_implementation_sha256': sequence['new_implementation_sha256'],
                   'lineage_parent': immediate}
    return sources


def closed_sequence_binding(plan, formal_path):
    """Read frozen closed-parent ancestry with standard-library checks only."""
    need(plan.get('protocol') in CONTINUATION_PROTOCOLS, 'Unknown closed-parent goal protocol')
    descriptor = plan['goal_sequence_protocol']
    sequence_path = Path(descriptor['path'])
    sequence = read(sequence_path)
    sequence_sha = sha(sequence_path)
    need(descriptor['sha256'] == sequence_sha
         and sequence.get('protocol') == plan['protocol'] and sequence.get('status') == 'FROZEN_CLOSED_PARENT'
         and plan['formal_protocol_sha256'] == sha(formal_path)
         and Path(plan['formal_protocol']) == formal_path
         and sequence['files']['formal_protocol']['sha256'] == sha(formal_path)
         and sequence['identity'] == plan['identity']
         and sequence['starting_core'] == plan['starting_core'],
         'Continued protocol/formal material/source identity binding differs')
    is_v3 = plan['protocol'] == CONTINUATION_PROTOCOL_V3
    parent_steps, total = (1800, 3150) if is_v3 else (450, 1800)
    meta = {'protocol_sha256': sequence_sha, 'parent_run_id': sequence['parent_run_id'],
            'parent_steps': parent_steps, 'total_budget': total, 'new_steps_budget': 1350}
    arguments = {'steps': total, 'learning_rate': 1e-5, 'distillation_weight': 5,
                 'preserve_replay_subspace': True, 'goal_sequence': meta}
    need(type(sequence.get('parent_steps')) is int and sequence['parent_steps'] == parent_steps
         and type(sequence.get('max_steps')) is int and sequence['max_steps'] == total
         and sequence.get('segment_steps') == 450 and sequence.get('max_segments') == total // 450
         and plan.get('arguments') == arguments and plan.get('goal_sequence') == meta
         and plan.get('source_sha256') == sequence.get('new_source_sha256')
         and isinstance(plan.get('source_sha256'), dict) and 'src/luban/goal_sequence.py' in plan['source_sha256'],
         'Continued budget or complete frozen implementation manifest differs')
    old_source = sequence.get('parent_source_sha256', {})
    old_impl = sequence.get('parent_implementation_sha256', {})
    new_impl = sequence.get('new_implementation_sha256', {})
    need(len(old_impl) == (20 if is_v3 else 19) and set(new_impl) == set(old_impl) | {'goal_sequence.py'}
         and set(plan['source_sha256']) == set(old_source) | {'src/luban/' + name for name in old_impl} | {'src/luban/goal_sequence.py'}
         and all(plan['source_sha256'].get('src/luban/' + name) == checksum for name, checksum in new_impl.items())
         and all(name in ({'evolution.py', 'training_checkpoint.py', 'goal_sequence.py'} if is_v3 else {'evolution.py', 'training_checkpoint.py'}) or new_impl[name] == checksum for name, checksum in old_impl.items())
         and all(name in ({'scripts/train_evolution_goal.py', 'src/luban/evolution.py', 'src/luban/training_checkpoint.py', 'src/luban/goal_sequence.py'} if is_v3 else {'scripts/train_evolution_goal.py', 'src/luban/evolution.py', 'src/luban/training_checkpoint.py'})
                 or plan['source_sha256'][name] == checksum for name, checksum in old_source.items()),
         'Old19/new20 complete implementation closure differs')
    for inventory, expected in ((sequence.get('parent_source_snapshot', {}), old_source),
                                (sequence.get('parent_implementation_snapshot', {}), old_impl)):
        need(set(inventory) == set(expected), 'Original source snapshot incomplete')
        for name, entry in inventory.items():
            need(not Path(name).is_absolute() and '..' not in Path(name).parts
                 and set(entry) == {'path', 'sha256'} and entry['sha256'] == expected[name]
                 and sha(plain(entry['path'])) == entry['sha256'], 'Original source snapshot changed')
    required = {'parent_plan', 'parent_result', 'parent_journal', 'parent_checkpoint', 'parent_binding', 'parent_report',
                'records', 'formal_protocol', 'formal_manifest', 'materials', 'old_retention', 'old_training',
                'old_protection', 'acceptance', 'native_manifest', 'native_output_manifest', 'native_trace', 'baseline_old_training'}
    if is_v3:
        required.add('parent_sequence_protocol')
    need(set(sequence['files']) == required, 'Closed-parent evidence inventory incomplete')
    for entry in sequence['files'].values():
        need(set(entry) == {'path', 'sha256'} and re.fullmatch('[a-f0-9]{64}', entry['sha256'])
             and sha(plain(entry['path'])) == entry['sha256'], 'Closed-parent protocol evidence file changed')
    native_files = sequence.get('native_files', {})
    need(isinstance(native_files, dict) and native_files and all(sha(plain(path)) == checksum
         for path, checksum in native_files.items()), 'Original actual native tensor/cache inventory changed')
    files = sequence['files']
    parent_plan = read(Path(files['parent_plan']['path']))
    parent_result = read(Path(files['parent_result']['path']))
    parent_report = read(Path(files['parent_report']['path']))
    parent_journal = read(Path(files['parent_journal']['path']))
    parent_binding = read(Path(files['parent_binding']['path']))
    records = read(Path(files['records']['path']))
    baseline = read(Path(files['baseline_old_training']['path']))
    closure = sequence['closure']
    validation_fields = {'baseline_new_loss', 'baseline_replay_loss', 'baseline_new_generation', 'baseline_replay_generation',
                         'validation_new_loss', 'validation_replay_loss', 'new_generation', 'replay_generation', 'core_retention_generation', 'gates'}
    original_arguments = {'steps': 450, 'learning_rate': 1e-5, 'distillation_weight': 5, 'preserve_replay_subspace': True}
    expected_parent_protocol = 'evolution-goal-current-core-native-v1'
    if is_v3:
        expected_parent_protocol = 'evolution-goal-fixed450-rounds-v2'
        need(parent_plan.get('protocol') == expected_parent_protocol, 'V3 must descend from one closed v2, without recursive/cyclic plans')
        ancestor, ancestor_report, _, _, original_arguments, _ = closed_sequence_binding(parent_plan, formal_path)
        need(files['parent_sequence_protocol'] == parent_plan['goal_sequence_protocol']
             and sequence['inherited_observations_sha256'] == digest({key: parent_report['goal_sequence'][key] for key in ('reference_baseline_after_parent_closed', 'entry450', 'stage_observations')})
             and sequence['parent_goal_sequence_sources_sha256'] == digest(parent_report['goal_sequence_sources'])
             and parent_report['goal_sequence_sources'] == sequence_sources(ancestor, ancestor_report), 'Closed1800 ancestry and observation source proof differs')
        ancestor['_protocol_path'] = parent_plan['goal_sequence_protocol']['path']
        goal_observation_binding(parent_plan, parent_result['training'], ancestor)
        need(parent_result['training']['goal_sequence'] == parent_report['goal_sequence']
             and parent_report['training_selection']['selected_step'] is None
             and parent_report['goal_sequence']['stage_observations'][-1]['step'] == 1800
             and parent_report['goal_sequence']['stage_observations'][-1]['pending_count'] > 0
             and all(not proof['safety_failed'] for proof in parent_report['goal_sequence']['stage_observations']), 'Parent1800 must be pending and safe with no heldout access')
    need(parent_plan.get('protocol') == expected_parent_protocol and parent_plan.get('arguments') == original_arguments
         and parent_plan.get('formal_protocol_sha256') == sha(formal_path)
         and parent_plan.get('source_sha256') == old_source and parent_plan.get('identity') == sequence['identity']
         and parent_plan.get('starting_core') == sequence['starting_core']
         and parent_result.get('plan_sha256') == files['parent_plan']['sha256']
         and parent_result.get('passed') is False and parent_result.get('released') is True
         and parent_result.get('production_written') is False and parent_result.get('adopted') is False
         and parent_result.get('training', {}).get('completed_steps') == parent_steps
         and parent_result['training'].get('promoted') is False and not validation_fields.intersection(parent_result['training'])
         and parent_report['id'] == parent_journal.get('id') == sequence['parent_run_id'] and parent_journal.get('state') == 'rejected'
         and parent_journal.get('binding') == parent_binding and parent_journal.get('records') == records
         and parent_journal.get('checkpoint', {}).get('sha256') == files['parent_checkpoint']['sha256']
         and parent_binding.get('implementation_sha256') == old_impl and parent_binding.get('records_sha256') == digest(records)
         and sequence.get('parent_binding_sha256') == digest(parent_binding)
         and sequence.get('parent_report_sha256') == digest(parent_report) and sequence.get('records_sha256') == digest(records)
         and len(baseline) == 83 and baseline == parent_result.get('baseline_old_training')
         and [row['id'] for row in baseline] == [row['id'] for row in records['groups']['replay_training']]
         and all(records[key] == plan[key] for key in ('groups', 'protection', 'retention_regressions'))
         and closure.get('validation_reads') == 0 and closure.get('final_step') == parent_steps
         and closure.get('released') is True and closure.get('adopted') is False and closure.get('production_written') is False
         and closure.get('parent_journal_state') == 'rejected' and closure.get('active_core') == plan['starting_core']
         and (closure.get('exit_code') == 0 or closure.get('exit_code') == 1 and closure.get('error_type') == 'ValueError'
              and closure.get('error') == '隔离候选没有通过正式核心训练门禁')
         and closure.get('error_type') == parent_result.get('error_type') and closure.get('error') == parent_result.get('error'),
         'Original450 lineage is incomplete, not closed or heldout was visited')
    formal = read(formal_path)
    aliases = {'new_material': 'materials', 'old_retention_complete': 'old_retention',
               'old_training_original': 'old_training', 'old_protection': 'old_protection', 'acceptance': 'acceptance'}
    need(all(plan['formal_files'][key]['sha256'] == sequence['files'][alias]['sha256']
             == sha(formal_path.parent / formal[key]) for key, alias in aliases.items()),
         'Continued formal16/4/83/89/8 material SHA differs')
    return sequence, parent_report, old_impl, new_impl, arguments, meta


def continued_training_binding(candidate_path, formal_path, foundation_report):
    """Root-local STD-only proof chain, independent of later repository/default changes."""
    candidate = read(candidate_path)
    plan_path = candidate_path.parent / 'plan.json'
    plan = read(plan_path)
    need(candidate.get('passed') is True and candidate.get('released') is True
         and candidate.get('production_written') is False and candidate.get('adopted') is False
         and candidate.get('plan_sha256') == sha(plan_path)
         and candidate.get('protocol') == plan.get('protocol') and plan.get('protocol') in CONTINUATION_PROTOCOLS,
         'Need completed actual continued candidate and matching new plan')
    sequence, parent_report, old_impl, new_impl, arguments, meta = closed_sequence_binding(plan, formal_path)
    descriptor = plan['goal_sequence_protocol']
    sequence_path, sequence_sha = Path(descriptor['path']), descriptor['sha256']
    files = sequence['files']
    records = read(Path(files['records']['path']))
    need(sequence_sha == foundation_report.get('source_protocol_sha256')
         and sequence['identity'] == foundation_report['identity']
         and sequence['starting_core'] == foundation_report['previous_version'], 'Foundation training protocol identity differs')
    version = candidate['candidate_version']
    source_name = 'evolution-' + version[5:] + '.json'
    source_store = Path(plan['storage']) / 'modules/luban0-v1'
    source_report = read(source_store / source_name)
    completed = source_report['completed_steps']
    expected_meta = {**meta, 'protocol': plan['protocol'], 'new_steps_executed': completed - sequence['parent_steps'],
                     'cumulative_steps': completed, 'selected_segment': (completed + 449) // 450,
                     'training_budget_scope': 'bounded goal-specific episode, ordinary product512 limit unchanged'}
    need(type(completed) is int and sequence['parent_steps'] < completed <= sequence['max_steps']
         and source_report.get('steps') == arguments['steps']
         and source_report == candidate.get('training') and version == 'core-' + sequence['new_run_id']
         and version == foundation_report.get('source_training_version')
         and sha(source_store / source_name) == candidate['artifacts'][source_name] == foundation_report['source_quality_sha256']
         and sha(candidate_path) == foundation_report.get('native_identity_sha256')
         and source_report.get('goal_sequence') == candidate.get('goal_sequence') == foundation_report.get('goal_sequence')
         and all(source_report['goal_sequence'].get(key) == value for key, value in expected_meta.items())
         and candidate.get('native_loaded', {}).get('core_loaded') is True
         and candidate['native_loaded']['core_tensor_sha256'] == candidate['candidate_tensor_sha256']
         and candidate['native_loaded']['core_file_sha256'] == candidate['artifacts'][version + '.safetensors']
         == foundation_report['checkpoint']['sha256'], 'Final actual C/F/native or cumulative-update proof differs')
    expected_sources = sequence_sources(sequence, parent_report)
    need(source_report.get('goal_sequence_sources') == expected_sources, 'Actual source450 history/tensor/implementation proof differs')
    sequence['_protocol_path'] = str(sequence_path)
    goal_observation_binding(plan, source_report, sequence)
    need(candidate.get('terminal_selected_training_confirmation') == terminal_selected_confirmation(
        plan, source_report, candidate['rows'], candidate['candidate_tensor_sha256']), 'Actual selected-W HF/native double TRAIN confirmation differs')
    expected_members = {version + suffix for suffix in ('.json', '.safetensors', '.retention.json')} | {source_name}
    need(set(candidate['artifacts']) == expected_members and all(sha(plain(source_store / name)) == checksum
         for name, checksum in candidate['artifacts'].items()), 'Actual C four-file source SHA differs')
    from luban.upgrades import CORE_GATES
    need(set(source_report['gates']) == CORE_GATES and all(value is True for value in source_report['gates'].values()), 'Original eight gates did not pass')
    checkpoints = list(range(64, 449, 32)) + list(range(900, sequence['max_steps'] + 1, 450))
    parent_history = parent_report['training_selection']['results']
    selection = source_report.get('training_selection', {})
    history = selection.get('results', [])
    journal = candidate.get('training_journal', {})
    need(completed in checkpoints[13:] and selection.get('selected_step') == completed
         and selection.get('checkpoints') == checkpoints and history[:len(parent_history)] == parent_history
         and [row['step'] for row in history] == [step for step in checkpoints if step <= completed], 'Actual cumulative selection history differs')
    need(journal.get('state') == 'committed' and journal.get('id') == source_report['id'] == sequence['new_run_id']
         and journal.get('scope') == journal.get('binding', {}).get('scope') == 'product'
         and journal.get('arguments') == journal['binding'].get('arguments') == arguments
         and journal.get('records') == records and journal.get('commit', {}).get('artifacts') == candidate['artifacts']
         and journal['commit'].get('event_kind') == 'evolution'
         and candidate['training_journal']['commit']['report'] == source_report
         and candidate.get('training_event') == {'kind': 'evolution', 'body': source_report}, 'Actual C adoption journal/event missing')
    return {'candidate_version': version, 'candidate_acceptance_sha256': sha(candidate_path),
            'candidate_plan_sha256': sha(plan_path), 'training_protocol_sha256': sequence_sha,
            'formal_protocol_sha256': sha(formal_path), 'goal_sequence': expected_meta,
            'scope': 'Root-local immutable proof-chain verification; no dependence on current repository or production defaults'}


def fresh_protocol_binding(plan, formal_path):
    """Immutable local prospective evidence, independent of future repository defaults."""
    need(plan.get('protocol') in FRESH_PROTOCOLS, 'Unknown prospective goal protocol')
    descriptor = plan['goal_sequence_protocol']
    formal = read(formal_path)
    phase = plan['protocol'] == DATA_PHASE
    resumed = plan['protocol'] in FRESH_RESUME_PROTOCOLS
    need(plan['formal_protocol'] == str(formal_path) and plan['formal_protocol_sha256'] == sha(formal_path), 'Fresh formal materials path/SHA differs')
    if resumed or phase:
        from luban import goal_fresh
        protocol = goal_fresh.load_verified(descriptor['path'])
        need(descriptor['sha256'] == sha(descriptor['path']) and protocol['_protocol_sha256'] == descriptor['sha256']
             and protocol['_files'] == plan['formal_files']
             and protocol['_parent_report']['goal_sequence']['protocol_sha256'] == protocol['files']['parent_protocol']['sha256'], 'Fresh child/parent protocol SHA differs')
        args, meta = goal_fresh.arguments(protocol), goal_fresh.metadata(protocol)
        if phase:need(all(plan.get(k) == v for k,v in goal_fresh.phase_plan_fields(protocol).items()), 'Phase exact formal/material/known24 fields differ')
    else:
        need(descriptor == {'path': str(formal_path), 'sha256': sha(formal_path)}, 'Fresh protocol path/SHA differs')
        protocol = formal
    need((formal.get('protocol') == DATA_PHASE and formal.get('status') == 'FROZEN_NEW_DATA_PHASE'
          and formal.get('total_steps') == 9450 and descriptor == {'path': str(formal_path), 'sha256': sha(formal_path)} if phase else
          formal.get('protocol') == FRESH_PROTOCOL and formal.get('status') == 'PRE_REGISTERED_NOT_TRAINED'
          and formal.get('total_steps') == 2700)
         and formal.get('segment_steps') == 450 and protocol.get('new_run_id')
         and formal.get('policy_version') == 'goal-prospective-same-weight450-v1',
         'Fresh must start B45 with complete450 same-W double rounds, not failed2700 Adam')
    if not (resumed or phase):
        meta = {'protocol_sha256': sha(formal_path), 'parent_run_id': None, 'parent_steps': 0, 'total_budget': 2700, 'new_steps_budget': 2700}
        args = {'steps': 2700, 'learning_rate': 1e-5, 'distillation_weight': 5, 'preserve_replay_subspace': True, 'goal_sequence': meta}
    need(plan['arguments'] == args and plan['goal_sequence'] == meta
         and protocol['source_core']['version'] == plan['starting_core'] == 'core-b45d965728bb'
         and protocol['source_core']['identity'] == plan['identity']
         and protocol['new_source_sha256'] == plan['source_sha256'], 'Fresh budget/B45/source binding differs')
    sources = plan['source_sha256']
    needed = {'src/luban/goal_fresh.py', 'src/luban/goal_topics.py', 'src/luban/generation_checks.py'}
    need(needed <= sources.keys(), 'Fresh source closure omits math or finite judge')
    impl = protocol['new_implementation_sha256']
    need(len(impl) == 22 and {'goal_fresh.py', 'goal_topics.py', 'generation_checks.py'} <= impl.keys()
         and all(sources.get('src/luban/' + name) == checksum for name, checksum in impl.items()), 'Fresh complete implementation closure differs')
    for name, checksum in sources.items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts
             and sha(plain(Path(plan['output']) / 'source-snapshot' / name)) == checksum, 'Fresh immutable source snapshot changed')
    aliases = ('new_material', 'old_retention_complete', 'old_training_original', 'old_protection', 'acceptance')
    for key in aliases:
        entry = plan['formal_files'][key]
        need((entry == protocol['_files'][key] if phase else Path(protocol[key]).name == protocol[key]
              and entry['path'] == str(formal_path.parent / protocol[key]))
             and sha(plain(entry['path'])) == entry['sha256'], 'Fresh frozen member changed')
    material = read(plan['materials'])
    need((len(material['TRAIN']), len(material['VALID'])) == ((56, 12) if phase else (24, 8))
         and plan['acceptance'] == read(plan['formal_files']['acceptance']['path']) and len(plan['acceptance']) == (12 if phase else 8)
         and plan['old_training_original'] == read(plan['formal_files']['old_training_original']['path']) and len(plan['old_training_original']) == 83
         and plan['protection'] == read(plan['formal_files']['old_protection']['path']) and len(plan['protection']) == 89
         and plan['retention_regressions'] == plan['acceptance'], 'Fresh24/8/8/83/89 full records differ')
    need(len({row['id'] for key in ('TRAIN','VALID') for row in material[key]} | {row['id'] for row in plan['acceptance']}) == (80 if phase else 40), 'Fresh IDs collide')
    need(plan['groups']['new_training'] == material['TRAIN'] and plan['groups']['new_validation'] == material['VALID'], 'Fresh grouping differs')
    capture = protocol['protection_capture']
    need(set(capture) == {'path', 'sha256'} and sha(plain(capture['path'])) == capture['sha256'], 'B45 actual generation capture changed')
    return protocol, args, meta


def fresh_observation_binding(plan, report, protocol):
    """Verify complete per-round raw TRAIN/old89 proof; no model, no new evaluation."""
    meta = report['goal_sequence']; completed = report['completed_steps']
    phase = plan['protocol'] == DATA_PHASE
    resumed = plan['protocol'] in FRESH_RESUME_PROTOCOLS
    parent = {FRESH_RESUME_PROTOCOL:2700, FRESH_RESUME_PROTOCOL_V3:4050, FRESH_RESUME_PROTOCOL_V4:5400, DATA_PHASE:6750}.get(plan['protocol'],0)
    budget = 9450 if phase else parent+1350 if resumed else 2700
    need(type(completed) is int and completed in range(parent+450, budget+1, 450), 'Fresh selected step must be a complete450 boundary')
    base = {**plan['goal_sequence'], 'protocol': plan['protocol'],
            'new_steps_executed': completed-parent, 'cumulative_steps': completed, 'selected_segment': (completed-parent)//450 if phase else completed // 450}
    if phase:
        from luban import goal_fresh
        need(report['goal_sequence_sources'] == goal_fresh.phase_sources(protocol), 'Phase exact new sources/archive binding differs')
        need(meta.get('parent_archive') == goal_fresh.phase_archive(protocol), 'Phase immutable parent archive differs')
        base.update(phase_steps=completed-parent, parent_archive=goal_fresh.phase_archive(protocol))
        goal_fresh.phase_proofs(report, plan['groups'], plan['protection'])
    base['training_budget_scope'] = ('new frozen56TRAIN/12VALID/12ACCEPT episode; six complete450 rounds; parent6750 archived separately' if phase else f'bounded fresh{parent} full-state continuation; at most three additional complete450 rounds' if resumed else 'six finite complete450 research rounds; ordinary product512 budget unchanged')
    if resumed:
        inherited=protocol['_parent_report']
        need(meta['reference_baseline_before_training']==inherited['goal_sequence']['reference_baseline_before_training']
             and meta['stage_observations'][:parent//450]==inherited['goal_sequence']['stage_observations']
             and report['training_selection']['results'][:parent//450]==inherited['training_selection']['results'], 'Fresh immutable complete parent history changed')
    need(set(meta) == set(base) | {'reference_baseline_before_training', 'stage_observations'} and all(meta.get(k) == v for k, v in base.items()), 'Fresh executed/budget metadata differs')
    training = plan['groups']['new_training']; replay = {r['id']: r for r in plan['groups']['replay_training']}
    old13 = [replay[i] for i in protocol['replay_generation_ids']]; old89 = plan['protection']
    need(len(training) == (56 if phase else 24) and len(old13) == 13 and len(old89) == 89, 'Fresh proof row inventories differ')
    wanted = {r['id'] for r in training}; prior = set(); reference = set()
    def observe(proof, step, label):
        nonlocal prior, reference
        need(proof.get('step') == step and type(proof['step']) is int and proof.get('label') == label
             and re.fullmatch('[a-f0-9]{64}', proof.get('tensor_sha256', ''))
             and proof.get('branch_enabled') is False and proof.get('reference_used') is False and proof.get('validation_rows_accessed') == 0
             and len(proof.get('observations', [])) == 2, 'Fresh requires two actual same-W branch/reference-off observations')
        sets=[]
        for obs in proof['observations']:
            need(set(obs) == {'new', 'old13', 'old89', 'replay_training_loss'}
                 and type(obs['replay_training_loss']) in {int, float} and math.isfinite(obs['replay_training_loss']) and obs['replay_training_loss'] >= 0, 'Invalid fresh TRAIN loss/old89 evidence')
            for key, samples in [('new', training), ('old13', old13), ('old89', old89)]:
                need([r['id'] for r in obs[key]] == [r['id'] for r in samples], 'Fresh same-W raw rows reordered/omitted')
                for row in obs[key]:
                    need(type(row.get('passed')) is bool and row.get('reference_used') is False and row.get('branch_enabled') is False
                         and isinstance(row.get('raw_completion_token_ids'), list) and row['raw_completion_token_ids']
                         and isinstance(row.get('answer'), str) and (not row['passed'] or row.get('complete') is True), 'Fresh evidence lacks actual independent answer/EOS')
            sets.append({r['id']for r in obs['new']if r['passed']})
        confirmed=set.intersection(*sets)
        old_ok=all(r['passed']for obs in proof['observations'] for k in ('old13','old89')for r in obs[k])
        expected={'single_observation_correct':[len(s)for s in sets], 'confirmed_ids':sorted(confirmed), 'confirmed_count':len(confirmed),
            'pending_ids':sorted(wanted-confirmed), 'pending_count':len(wanted-confirmed),
            'newly_confirmed_from_reference':sorted(confirmed-reference), 'newly_confirmed_since_prior':sorted(confirmed-prior),
            'forgotten_from_reference':sorted(reference-confirmed), 'forgotten_since_prior':sorted(prior-confirmed),
            'old13_both_passed':all(r['passed']for obs in proof['observations']for r in obs['old13']),
            'old89_both_passed':all(r['passed']for obs in proof['observations']for r in obs['old89'])}
        need(all(proof.get(k)==v for k,v in expected.items()), 'Fresh mastery/forgetting/old89 accounting differs')
        if step != (parent if phase else 0):
            budget=report['baseline_training_replay_loss']*1.05+.01
            loss_ok=all(obs['replay_training_loss']<=budget for obs in proof['observations'])
            safety=not(old_ok and loss_ok and not reference-confirmed and not prior-confirmed)
            satisfied=not safety and confirmed==wanted
            need(proof.get('actual_updates')==450 and proof.get('replay_loss_budget')==budget
                 and proof.get('replay_loss_both_passed')==loss_ok and proof.get('safety_failed')==safety
                 and proof.get('round_result')==('protection_failed' if safety else 'satisfied' if satisfied else 'partial' if confirmed else 'pending'), 'Fresh round protection/selection differs')
        else:
            need(old_ok, 'B45 old baseline protection failed')
        prior |= confirmed
        return confirmed
    baseline=meta['reference_baseline_before_training']; reference=observe(baseline,parent if phase else 0,'data-phase-entry6750' if phase else 'prospective-reference')
    need(baseline['tensor_sha256']==(protocol['parent_final_tensor_sha256'] if phase else protocol['original_tensor_sha256']), 'Fresh entry/reference tensor differs')
    if phase:need(set(report['goal_sequence_sources']['prior_training_ids']) <= reference, 'Phase entry lost original TRAIN24')
    stages=meta['stage_observations']; expected_steps=list(range(parent+450 if phase else 450,completed+1,450))
    need([p['step']for p in stages]==expected_steps, 'Fresh full450 ledger has missing/early boundaries')
    for proof in stages:
        observe(proof,proof['step'],'data-phase-complete450' if phase else 'prospective-complete450')
        need(not(proof['safety_failed'] or proof['round_result']=='satisfied') or proof['step']==completed, 'Fresh optimized after safety failure or satisfaction')
    satisfied=stages[-1]['round_result']=='satisfied'
    need(satisfied or stages[-1]['safety_failed'] or completed==budget, 'Fresh stopped an incomplete safe round before budget')
    selection=report['training_selection']
    need(selection.get('checkpoints')==list(range(parent+450 if phase else 450,budget+1,450)) and selection.get('selected_step')==(completed if satisfied else None)
         and selection.get('required_consecutive_passes')==0 and selection.get('required_same_weight_passes')==2, 'Fresh same-W policy differs')
    history=selection['results'];need(len(history)==len(stages), 'Fresh selection history inventory differs')
    for row, proof in zip(history,stages):
        first=proof['observations'][0];satisfied=proof['round_result']=='satisfied'
        expected={'step':proof['step'],'new_generation':first['new'],'replay_training_generation':first['old13'],
            'replay_loss':first['replay_training_loss'],'accepted':satisfied,'consecutive_passes':0,
            'same_weight_passes':2 if satisfied else 0,'minimum_new_exposures':((proof['step']-parent)//2)//56 if phase else (proof['step']//2)//24,'tensor_sha256':proof['tensor_sha256']}
        need(row==expected, 'Fresh history is not actual same-W evidence')
    return reference,prior


def fresh_training_binding(candidate_path, formal_path, foundation_report=None):
    candidate=read(candidate_path);plan_path=candidate_path.parent/'plan.json';plan=read(plan_path)
    protocol,args,meta=fresh_protocol_binding(plan,formal_path)
    need(candidate.get('passed') is True and candidate.get('released') is True and candidate.get('adopted') is False
         and candidate.get('production_written') is False and candidate.get('teacher_api_calls')==0
         and candidate.get('protocol')==plan['protocol'] and candidate.get('plan_sha256')==sha(plan_path), 'Fresh actual candidate closure differs')
    report=candidate['training']; version=candidate['candidate_version'];store=Path(plan['storage'])/'modules/luban0-v1'
    report_name='evolution-'+version[5:]+'.json';need(report==read(store/report_name) and report['version']==version=='core-'+protocol['new_run_id'], 'Fresh actual report/version differs')
    from luban.upgrades import CORE_GATES
    need(report['gates']==dict.fromkeys(CORE_GATES,True) and report.get('partial_policy') is None, 'Fresh full ordinary eight gates required')
    fresh_observation_binding(plan,report,protocol)
    sources = {'protocol_sha256': sha(formal_path), 'original_tensor_sha256': protocol['original_tensor_sha256'],
        'starting_core': protocol['source_core'], 'new_implementation': protocol['new_implementation_sha256'],
        'new_source_sha256': protocol['new_source_sha256'], 'files': plan['formal_files'],
        'protection_capture': protocol['protection_capture'], 'scope': 'fresh B45; no rejected-candidate optimizer ancestry'}
    if plan['protocol'] in FRESH_RESUME_PROTOCOLS | {DATA_PHASE}:
        from luban import goal_fresh
        sources=goal_fresh.phase_sources(protocol) if plan['protocol'] == DATA_PHASE else goal_fresh.source_metadata(protocol)
    need(report.get('goal_sequence_sources') == sources, 'Fresh exact originalB45/material/implementation/capture lineage differs')
    need(candidate['goal_sequence']==report['goal_sequence']
         and candidate.get('terminal_selected_training_confirmation')==terminal_selected_confirmation(plan,report,candidate['rows'],candidate['candidate_tensor_sha256']), 'Fresh final selected-W double HF/nativeTRAIN24 differs')
    wanted=plan['groups']['new_training']+plan['groups']['new_validation']+plan['acceptance']
    need(set(candidate['rows'])=={'hf_new_first','hf_new_second','hf_old','native_new_first','native_new_second','native_old'}, 'Fresh six actual generation phases missing')
    for phase,rows in candidate['rows'].items():all_passed(rows,plan['protection']if phase.endswith('old')else wanted,native=phase.startswith('native'))
    artifacts={version+s for s in ('.json','.safetensors','.retention.json')}|{report_name}
    need(set(candidate['artifacts'])==artifacts and all(sha(plain(store/n))==h for n,h in candidate['artifacts'].items()), 'Fresh C four actual artifact SHA differs')
    loaded=candidate['native_loaded'];need(loaded['core_loaded'] is True and loaded['core_tensor_sha256']==candidate['candidate_tensor_sha256']
         and loaded['core_file_sha256']==candidate['artifacts'][version+'.safetensors'], 'Fresh native did not load actual C tensor')
    need(report['completed_steps'] == candidate.get('completed_steps') and report['steps'] == args['steps']
         and report['goal_sequence']['stage_observations'][-1]['tensor_sha256'] == candidate['candidate_tensor_sha256'], 'Fresh selected proof is a different tensor or execution count')
    records={k:plan[k]for k in ('groups','protection','retention_regressions')};journal=candidate['training_journal'];commit=journal['commit']
    need(journal['state']=='committed' and journal['id']==report['id'] and journal['scope']==journal['binding']['scope']=='product'
         and journal['arguments']==journal['binding']['arguments']==args and journal['records']==records
         and commit['report']==report and commit['artifacts']==candidate['artifacts'] and commit['event_kind']=='evolution'
         and candidate['training_event']=={'kind':'evolution','body':report}, 'Fresh actual C product journal/event differs')
    if foundation_report is not None:
        need(foundation_report['source_protocol_sha256']==meta['protocol_sha256'] and foundation_report['source_training_version']==version
             and foundation_report['source_quality_sha256']==candidate['artifacts'][report_name]
             and foundation_report['native_identity_sha256']==sha(candidate_path)
             and foundation_report['checkpoint']['sha256']==candidate['artifacts'][version+'.safetensors']
             and foundation_report['goal_sequence']==report['goal_sequence'], 'Fresh F is not actualC/protocol/native/SF')
    return {'candidate_version':version,'candidate_acceptance_sha256':sha(candidate_path),'candidate_plan_sha256':sha(plan_path),
            'training_protocol_sha256':meta['protocol_sha256'],'formal_protocol_sha256':sha(formal_path),'goal_sequence':report['goal_sequence'],
            'scope':('New frozen data phase; actual6750 parent archived separately; originalB45 reference/basis' if plan['protocol'] == DATA_PHASE else f"Closed safe fresh{meta['parent_steps']} continuation; originalB45 reference/basis; no failed old v3 ancestry"
                if plan['protocol'] in FRESH_RESUME_PROTOCOLS else 'Prospective fresh B45 immutable evidence; no failed2700 optimizer inheritance or current-source dependency')}


def bound_materials(model, path, manifest, candidate_acceptance=None):
    protocol = read(path)
    phase = protocol.get('protocol') == DATA_PHASE
    if phase:
        from luban import goal_fresh
        checked = goal_fresh.load_verified(path)
        formal = read(checked['material_manifest']['path'])['files']
        hashes = {'protocol.json': sha(path)}
        need(formal == {k: {**v, 'bytes': Path(v['path']).stat().st_size} for k,v in checked['_files'].items()}, 'Phase exact five material manifest differs')
        for key, entry in checked['_files'].items():
            need(sha(plain(entry['path'])) == entry['sha256'], 'Phase material bytes differ')
            hashes[key] = entry['sha256']
    else:
        formal = read(path.parent / 'manifest.json')['files']
        hashes = {}
    for name in () if phase else (path.name, *(protocol[key] for key in ('new_material', 'acceptance', 'old_protection'))):
        need(Path(name).name == name, 'Protocol member must be a filename')
        hashes[name] = sha(path.parent / name)
        need(hashes[name] == formal[name]['sha256'], 'Frozen protocol/material SHA differs')
    report_path = model / 'luban-foundation' / manifest['report']
    need(sha(report_path) == manifest['files'][manifest['report']], 'Foundation report SHA differs')
    foundation_report = read(report_path)
    if 'goal_sequence' in foundation_report:
        need(candidate_acceptance is not None, 'Continued F requires explicit candidate acceptance')
        binding = (fresh_training_binding(candidate_acceptance, path, foundation_report)
                   if protocol.get('protocol') in {FRESH_PROTOCOL, DATA_PHASE} else continued_training_binding(candidate_acceptance, path, foundation_report))
        hashes['fresh_training_binding' if protocol.get('protocol') in {FRESH_PROTOCOL, DATA_PHASE} else 'continued_training_binding'] = binding
    else:
        need(foundation_report['source_protocol_sha256'] == hashes[path.name], 'Foundation protocol binding differs')
    material = read(checked['_files']['new_material']['path'] if phase else path.parent / protocol['new_material'])
    acceptance = read(checked['_files']['acceptance']['path'] if phase else path.parent / protocol['acceptance'])
    old = read(checked['_files']['old_protection']['path'] if phase else path.parent / protocol['old_protection'])
    fresh = protocol.get('protocol') in {FRESH_PROTOCOL, DATA_PHASE}
    need((len(material['TRAIN']), len(material['VALID'])) == ((56,12) if phase else (24, 8) if fresh else (16, 4))
         and len(acceptance) == (12 if phase else 8) and len(old) == 89, 'Frozen TRAIN/VALID/acceptance/old row counts differ')
    new = material['TRAIN'] + material['VALID'] + acceptance
    need(len({row['id'] for row in new}) == (80 if phase else 40 if fresh else 28) and len({row['id'] for row in old}) == 89,
         'Generation material IDs must be unique')
    return protocol, new, old, hashes


def user_rows(memory):
    result = {}
    with memory.lock, memory.connect() as db:
        for table in ('samples', 'conversations', 'teacher_materials', 'usage', 'events', 'training_runs'):
            key = 'route' if table == 'usage' else 'id'
            result[table] = {row[key]: digest(dict(row)) for row in db.execute('SELECT * FROM ' + table)}
    return result


def row_summary(rows):
    return {table: {'count': len(values), 'sha256': digest(values)} for table, values in rows.items()}


def verify_user_rows(before, after, version):
    added = {}
    allowed = {'events': 'training:' + version[5:], 'training_runs': version[5:]}
    for table, original in before.items():
        current = after[table]
        need(all(current.get(key) == value for key, value in original.items()), 'Existing user rows changed: ' + table)
        extra = set(current) - set(original)
        need(extra <= ({allowed[table]} if table in allowed else set()), 'Unexpected new user rows: ' + table)
        added[table] = len(extra)
    return added


def rollback_adoption(runtime, settings, before, storage, previous, target, report):
    outcome = report['rollback'] = {'attempted': True, 'passed': False}
    try:
        settings.update({'model_path': before['model_path']})
        outcome['model_path_restored'] = settings.snapshot() == before
        if runtime is not None:
            runtime.close_native()
        pointer_path = storage / 'modules/luban0-v1/active-core.json'
        pointer = read(pointer_path)
        if pointer['version'] == target:
            need(runtime is not None, 'Committed foundation rollback requires runtime')
            if runtime.model is None or runtime.loaded_path != before['model_path'] or runtime.state == 'error':
                runtime._release_model()
                runtime.load()
            from luban.evolution import restore_core
            outcome['core_restore'] = restore_core(runtime, previous['version'])
        need(read(pointer_path) == previous and outcome['model_path_restored'], 'Adoption rollback incomplete')
        outcome['passed'] = True
    except BaseException as error:
        outcome.update(error_type=type(error).__name__, error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--adopt-home', type=Path)
    parser.add_argument('--expected-previous')
    parser.add_argument('--candidate-acceptance', type=Path)
    args = parser.parse_args()
    need(args.output.parent.is_dir() and not args.output.exists(), 'Evidence output must be new')
    need(not args.model.is_symlink(), 'Foundation root must be a real directory')
    manifest = read(args.model / 'luban-foundation.json')
    protocol, new_rows, old_rows, material_hashes = bound_materials(args.model, args.protocol, manifest, args.candidate_acceptance)
    if args.adopt_home:
        need(args.expected_previous == protocol['source_core']['version'], 'Explicit previous binding required')
    args.output.mkdir(mode=0o700)
    home = args.adopt_home or args.output / 'home'
    report = {'passed': False, 'fresh_store': not bool(args.adopt_home), 'teacher_calls': 0,
              'started': time.time(), 'target_version': manifest['version'], 'model': str(args.model),
              'model_manifest_sha256': sha(args.model / 'luban-foundation.json'), 'training_updates_executed': 0,
              'material_sha256': material_hashes, 'protocol_sha256': sha(args.protocol)}
    runtime = None
    try:
        with data_service(home, args.output / 'data-service.json'):
            import torch
            from luban.settings import Settings, private_json
            from luban.memory import Memory
            from luban.runtime import Runtime
            from luban.foundation import install_foundation
            from luban.training_checkpoint import training_runs
            from luban import model_packages, product_inference
            torch.set_num_threads(1)
            settings = Settings(home)
            before = settings.snapshot()
            report['settings_before_sha256'] = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
            storage = Path(before['storage_root']) if args.adopt_home else args.output / 'storage'
            if args.adopt_home:
                pointer = read(storage / 'modules/luban0-v1/active-core.json')
                need(pointer == protocol['source_core'], 'Production core changed; do not adopt')
            memory = Memory(storage)
            previous_rows = user_rows(memory) if args.adopt_home else None
            if args.adopt_home:
                report['user_data_before'] = row_summary(previous_rows)
            try:
                updates = {'model_path': str(args.model)}
                if not args.adopt_home:
                    updates.update(storage_root=str(storage), temperature=0, local_thinking='off',
                                   automatic={'mode': 'manual', 'scheduled_time': '02:00'})
                settings.update(updates)
                runtime = Runtime(settings, memory, research=False)
                runtime.load()
                if args.adopt_home:
                    need(runtime.active_core == args.expected_previous, 'Loaded previous core changed')
                    runtime.close_native()
                    installed = install_foundation(runtime)
                    need(installed and installed['version'] == manifest['version'], 'Validated adoption failed')
                    runtime._release_model()
                    runtime.load()
                need(runtime.active_core == manifest['version'] and runtime.branch.expert is None,
                     'Actual loaded core or branch differs')
                native = product_inference.start(runtime)
                need(native is not None and native.loaded.get('core_loaded') is True
                     and native.loaded.get('base_identity') == manifest['identity']
                     and native.loaded.get('core_file_sha256') == manifest['files'][manifest['version'] + '.safetensors'],
                     'Fresh native binding differs')
                report['native_loaded'] = native.loaded
                for name, samples in (('new80' if protocol.get('protocol') == DATA_PHASE else 'new40' if protocol.get('protocol') == FRESH_PROTOCOL else 'new28', new_rows), ('old89', old_rows)):
                    rows = native_rows(runtime, samples, native)
                    report[name] = rows
                    private_json(args.output / 'progress.json', report)
                    all_passed(rows, samples, native=True)
                runs = training_runs(runtime.memory, runtime.identity)
                selected = [run for run in runs if run['id'] == manifest['version'][5:]]
                need(len(selected) == 1 and selected[0]['state'] == 'committed'
                     and selected[0]['commit']['report']['foundation_install'] is True
                     and selected[0]['commit']['report']['training_updates_executed'] == 0
                     and selected[0]['commit']['artifacts'] == manifest['files'], 'Real installation journal differs')
                report['installation_journal'] = selected[0]
                report['journal_count'] = len(runs)
                if not args.adopt_home:
                    need(len(runs) == 1, 'Fresh store unexpectedly contains personal training')
                    runtime.close_native()
                    report['package'] = model_packages.export_package(runtime, args.output / 'Luban-Goal-Base.zip')
                    report['package_sha256'] = sha(report['package']['path'])
                    managed = args.output / 'import-stage'
                    managed.mkdir()
                    report['import'] = model_packages.stage_import(Path(report['package']['path']), managed)
                    need(report['import']['kind'] == 'learned', 'Wrong package interpretation')
                else:
                    after = settings.snapshot()
                    need(after == {**before, 'model_path': str(args.model)}, 'Unrelated user settings changed')
                    current_rows = user_rows(memory)
                    report['user_data_after'] = row_summary(current_rows)
                    report['new_user_rows'] = verify_user_rows(previous_rows, current_rows, manifest['version'])
                    report['production_adopted'] = True
                report.update(passed=True, scope=f'Real BF16 native fresh reload, new{len(new_rows)}/old89 complete frozen-contract EOS; no teacher/reference/branch')
            except BaseException as error:
                report.update(error_type=type(error).__name__, error=str(error))
                if args.adopt_home:
                    rollback_adoption(runtime, settings, before, storage, protocol['source_core'], manifest['version'], report)
                    try:
                        current_rows = user_rows(memory)
                        report['user_data_after_failure'] = row_summary(current_rows)
                        report['new_user_rows_after_failure'] = verify_user_rows(previous_rows, current_rows, manifest['version'])
                    except BaseException as check_error:
                        report['user_data_check_error'] = {'type': type(check_error).__name__, 'message': str(check_error)}
                raise
    finally:
        try:
            if runtime is not None:
                runtime._release_model()
                report['released'] = runtime.model is None and runtime.native_engine is None and runtime._model_process_lock is None
        except BaseException as error:
            report.update(passed=False, released=False,
                          release_error={'type': type(error).__name__, 'message': str(error)})
            raise
        finally:
            report['elapsed_seconds'] = time.time() - report['started']
            write_new(args.output / 'result.json', report)
    print(json.dumps({'passed': report['passed'], 'version': manifest['version'], 'fresh_store': report['fresh_store'],
                      'result': str(args.output / 'result.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
