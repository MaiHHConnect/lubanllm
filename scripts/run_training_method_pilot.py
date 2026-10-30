"""Fixed original57 native/cache B1 pilot150; default/prepare never allocate a model."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time

from run_cached_training_audit import (ROOT, PRIOR, PRIOR_SHA, IDENTITY, ORIGINAL,
    sha, content_sha, safe, write_new, prepare as audit_prepare, production_snapshot)

PROTOCOL = 'original57-native-cache-b1-pilot150-tokenizer-eos-v2'
CHECKS = (50, 100, 150)
METHODS = ('native', 'cache')


def prepare(output):
    old = json.loads(PRIOR.read_text())
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError('Historical full TRAIN source changed')
    base = audit_prepare(output)
    prior_audit = ROOT / 'research/native_cached_training_audit_result.json'
    receipt = json.loads(prior_audit.read_text())
    raw = json.loads(Path(receipt['result_path']).read_text())
    if (sha(receipt['result_path']) != receipt['result_sha256'] or raw.get('passed') is not True
            or raw.get('preserved') is not True or raw['audit'].get('restored') is not True
            or len(raw['audit']['steps']) != 2 or any(c.get('exact') is not True
                for step in raw['audit']['steps'] for c in step['checks'].values())):
        raise ValueError('Actual native2B B1 cache algebra proof required before method pilot')
    # Only complete TRAIN rows are part of this study. No held-out tokens or answers.
    groups = {key: copy.deepcopy(old['groups_before_original_baseline'][key])
              if key.endswith('_training') else [] for key in old['groups_before_original_baseline']}
    if {k: len(v) for k, v in groups.items()} != {
            'new_training': 57, 'new_validation': 0, 'replay_training': 46, 'replay_validation': 0}:
        raise ValueError('Pilot requires unchanged full57 NEW and46 old REPLAY TRAIN')
    sources = dict(base['source_sha256'])
    for name in ('scripts/run_training_method_pilot.py', 'scripts/cached_method_steps.py',
                 'tests/test_training_method_pilot.py', 'tests/test_cached_method_steps.py',
                 'scripts/verify_cached_batch_training.py', 'tests/test_training_eos_contract.py'):
        sources[str(ROOT / name)] = sha(ROOT / name)
    labels_path=ROOT/'research/training_method_pilot_eos_cpu_runtime.json'
    labels=json.loads(labels_path.read_text())
    contract_sources={str(ROOT/name):sha(ROOT/name) for name in (
        'src/luban/runtime.py','scripts/verify_cached_batch_training.py','scripts/cached_method_steps.py')}
    if (labels.get('passed') is not True or labels.get('model_allocated') is not False
            or labels.get('runtime_construction')!='Runtime.__new__: tokenizer/CPU only, modelNone'
            or labels.get('cpu_script_sha256')!=sha(labels['cpu_script_path'])
            or labels.get('contract')!='runtime-sample-tokens-tokenizer-eos-v1'
            or len(labels.get('rows',[]))!=114 or not all(row.get('passed') is True for row in labels['rows'])
            or labels.get('source_sha256')!=contract_sources):
        raise ValueError('Current official-tokenizer complete114 TRAIN label contract CPU proof required')
    failed_path=ROOT/'research/training_method_pilot_result.json'
    failed=json.loads(failed_path.read_text())
    if sha(failed['result_path'])!=failed['result_sha256'] or failed.get('native_optimizer_steps')!=0 or failed.get('cache_optimizer_steps')!=0:
        raise ValueError('Immutable previous zero-update failed pilot reference changed')
    failed_raw=json.loads(Path(failed['result_path']).read_text())
    training=[row for key in ('new_training','replay_training') for row in failed_raw['groups'][key]]
    if (labels.get('qualified_training_groups_sha256')!=failed_raw['groups_sha256']
            or [(row['id'],row['sample_sha256']) for row in labels['rows']]!=[(row['id'],content_sha(row)) for row in training]):
        raise ValueError('Official tokenizer CPU labels must cover exact previously qualified full114 TRAIN, not held-out or replacement rows')
    return {k: v for k, v in base.items() if k not in {
        'samples', 'sample_sha256', 'optimizer', 'qualification', 'scope', 'timing'}} | {
        'protocol': PROTOCOL, 'source_sha256': dict(sorted(sources.items())),
        'method_cpu_source_path': str(ROOT/'research/cached_method_steps_source.json'),
        'method_cpu_source_sha256': sha(ROOT/'research/cached_method_steps_source.json'),
        'method_cpu_protocol_sha256': sha(ROOT/'research/cached_method_steps_protocol.json'),
        'audit_receipt_path': str(prior_audit), 'audit_receipt_sha256': sha(prior_audit),
        'audit_result_sha256': receipt['result_sha256'], 'base_identity': IDENTITY,
        'previous_failed_pilot_descriptor_sha256':sha(failed_path),
        'previous_failed_pilot_descriptor_path':str(failed_path),
        'previous_failed_pilot_raw_sha256':failed['result_sha256'],
        'training_labels_cpu_path':str(labels_path),'training_labels_cpu_sha256':sha(labels_path),
        'training_labels_cpu_script_path':labels['cpu_script_path'],
        'training_labels_cpu_script_sha256':labels['cpu_script_sha256'],
        'training_eos_contract':'runtime-sample-tokens-tokenizer-eos-v1',
        'training_label_contract':'exact Runtime tokenizerEOS gold suffix; generation completeness still model generation EOS, unchanged; CPU labels do not replace new original qualification',
        'single_attempt':'fsynced O_EXCL protocol-directory execution-started marker before output/model allocation; changing output cannot repeat the same preregister',
        'groups_before_original_baseline': groups, 'groups_sha256': content_sha(groups),
        'replay_training_candidates': copy.deepcopy(old['replay_training_candidates']),
        'replay_candidate_sha256': old['replay_candidate_sha256'],
        'minimum_replay_candidates': 4, 'required_qualified_count': 11,
        'required_replay_training_ids': old['required_replay_training_ids'],
        'prepared': old['prepared'], 'methods': list(METHODS), 'optimizer_steps': 150,
        'checkpoints': list(CHECKS), 'tensor_batch': 1, 'gradient_accumulation': 1,
        'learning_rate': 1e-5, 'eps': 1e-6, 'distillation_weight': 5, 'clip': 1,
        'field_first_token_weight': 1, 'preserve_replay_subspace': True,
        'schedule': '0-based even NEW CE, odd REPLAY CE; sample and KL replay index=(step//2)%57; CE.backward then5KL.backward',
        'exposure_ledger_per_arm': {'new_ce': 75, 'replay_ce': 75, 'replay_kl': 150, 'optimizer_steps': 150},
        'qualified_shape': {'new_training': 57, 'replay_training': 57, 'protected_original_replay': 13},
        'baseline': 'current original closedbook all114 qualified TRAIN; reuse only exact same ID/context/check qualification observations for13 of114, with raw IDs and check retained',
        'generation_upper_bound': 913, 'qualification_generation_calls': 14, 'original_baseline_remaining_generations': 101,
        'generation_cost': '14 qualifications+101 remaining original TRAIN=115; each arm worst-case50/100:(57NEW+57original-correctREPLAY),150:(2*57NEW+57REPLAY)=399; total<=913; no DEV/engineering',
        'mastery': '50/100 single probes are provisional correctness only;150 frozen same tensor2NEW probes confirm finite-check baseline-wrong to twice-correct; no product acceptance',
        'preservation': 'every checkpoint preserves ALL original-correct NEW/REPLAY plus all13 qualified mandatory/pool tasks; native mean REPLAY TRAIN loss<=1.05 baseline',
        'stop': 'hard150 updates; no gain-based early stopping; any original-correct regression/loss budget/nonfinite/deadline stops that arm without retry',
        'deadline_seconds_per_arm_including_half_common_setup': 3600,
        'timing': 'real synchronized phase wall; full checks/cache build/hash validation/transfers/save/restore included; common load+qualification+baseline+basis cost split half into each arm; actual end-to-end sum retained',
        'scope': 'single fixed-order native-then-cache historical57 TRAIN pilot; warmup/order effect unseparated; passed means complete protected/equivalent execution and recovery, NOT mastery of all57; no sweetspot/generalization/publication/teacher claim'}


def bind(plan):
    from verify_router_coverage_study import model_binding
    if prepare(plan['output']) != plan:
        raise ValueError('Pilot source/data/algorithm changed before binding')
    return {**copy.deepcopy(plan), 'official_model_binding': model_binding(plan['model_path'], IDENTITY),
        'production_snapshot': production_snapshot(plan['production_root']), 'state': 'bound_waiting_root_seal_MODEL_GO'}


def validate_bound(plan):
    template = {**copy.deepcopy(plan), 'state': 'unbound_waiting_root_review', 'sealed': False,
                'official_model_binding': None, 'production_snapshot': None}
    if prepare(plan['output']) != template or bind(template) != {
            **plan, 'sealed': False, 'state': 'bound_waiting_root_seal_MODEL_GO'}:
        raise ValueError('Pilot fixed source/material/official/production binding changed')


def schedule(groups):
    if len(groups['new_training']) != 57 or len(groups['replay_training']) != 57:
        raise ValueError('Exactly114 current-qualified TRAIN required')
    return [(('new' if i % 2 == 0 else 'replay'),
        groups['new_training' if i % 2 == 0 else 'replay_training'][(i // 2) % 57],
        groups['replay_training'][(i // 2) % 57]) for i in range(150)]


def correct(rows, observations, *, finite=False):
    from luban.generation_checks import is_finite_generation_check, evaluate_generation
    if [r['id'] for r in rows] != [o.get('id') for o in observations]:
        raise ValueError('Actual closedbook observation coverage/order changed')
    result = set()
    for sample, actual in zip(rows, observations):
        if (type(actual.get('passed')) is not bool or type(actual.get('complete')) is not bool
                or actual.get('branch_enabled') is not False or actual.get('reference_used') is not False):
            raise ValueError('Every probe must be actual unreferenced branchOFF and strict boolean EOS/result')
        check = json.loads(sample['provenance'])['generation_check']
        expected = evaluate_generation(actual['answer'], check, actual['complete'])['passed']
        if actual['passed'] != expected:
            raise ValueError('Raw answer/EOS does not match fixed check')
        if actual['passed'] and (not finite or is_finite_generation_check(check)):
            result.add(sample['id'])
    return result


def probes(runtime, rows, deadline=None, expected_tensor_sha=None):
    from luban.evolution import _generation, _target
    from verify_cached_training import tensor_sha
    actual = []
    for row in rows:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Predeclared arm end-to-end deadline; no extra updates/probes')
        before = tensor_sha(_target(runtime)[1])
        if expected_tensor_sha is not None and before != expected_tensor_sha:
            raise ValueError('Frozen probe candidate tensor changed before generation')
        one = _generation(runtime, [row])
        after = tensor_sha(_target(runtime)[1])
        if before != after:
            raise ValueError('Frozen probe candidate tensor changed during generation')
        for record in one:
            record.update(sample_sha256=content_sha(row), actual_tensor_sha256=before,
                          core_version=runtime.active_core, actual_base_identity=getattr(runtime,'identity',None), reference_used=record.get('reference_used'))
        actual.extend(one)
    correct(rows, actual)
    return actual


def checkpoint(runtime, groups, baseline, old_rows, before_loss, step, candidate_sha, deadline):
    from luban.evolution import _loss, _target
    from verify_cached_training import tensor_sha
    def frozen():
        if tensor_sha(_target(runtime)[1]) != candidate_sha:
            raise ValueError('Frozen checkpoint candidate changed before/between/after actual proofs')
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Complete checkpoint including quality loss exceeded fixed deadline')
    frozen()
    news = groups['new_training']
    before = correct(news, baseline['new'], finite=True)
    first = probes(runtime, news, deadline, candidate_sha)
    frozen()
    first_correct = correct(news, first, finite=True)
    second = probes(runtime, news, deadline, candidate_sha) if step == 150 else []
    frozen()
    confirmations = correct(news, second, finite=True) if second else set()
    protected = probes(runtime, old_rows, deadline, candidate_sha)
    frozen()
    original_new = correct(news, baseline['new'])
    current_new = correct(news, first)
    current_old = correct(old_rows, protected)
    forgotten = sorted((original_new - current_new) | ({r['id'] for r in old_rows} - current_old))
    if second: forgotten = sorted(set(forgotten) | (original_new - correct(news, second)))
    replay_loss = _loss(runtime, groups['replay_training'])
    frozen()
    safety = not forgotten and replay_loss <= before_loss * 1.05
    return {'step': step, 'candidate_tensor_sha256': candidate_sha, 'new_first': first,
        'new_frozen_second': second, 'old_preservation': protected, 'replay_loss': replay_loss,
        'baseline_replay_loss': before_loss, 'replay_loss_ratio_limit': 1.05,
        'provisional_new_correct_ids': sorted(first_correct),
        'provisional_growth_ids': sorted(first_correct - before),
        'newly_confirmed_ids': sorted((first_correct & confirmations) - before) if second else [],
        'mastery_confirmed': bool(second), 'pending_ids': sorted({r['id'] for r in news} - (first_correct & confirmations)) if second else None,
        'forgotten_ids': forgotten, 'safety_passed': safety,
        'publication': False, 'scope': 'finite baseline->two frozen actualEOS proof at150 only;50/100 single scores provisional'}


def arm(runtime, groups, baseline, old_rows, original, basis, caches, method, common_seconds, output, cache_setup_seconds=0, sources=None):
    import torch
    from cached_method_steps import train_method_step
    from luban.evolution import _target, _loss
    from verify_cached_training import _sync, tensor_sha
    parameter = _target(runtime)[1]
    _sync(parameter.device); began = time.perf_counter()
    deadline = began + max(0, 3600 - common_seconds / 2 - cache_setup_seconds)
    master = torch.nn.Parameter(original.float().clone())
    optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
    reference = original.float()
    report = {'method': method, 'steps': [], 'checks': [], 'completed_steps': 0,
        'exposures': {'new_ce': {}, 'replay_ce': {}, 'replay_kl': {}}, 'optimizer_steps': 0,
        'timing': {'optimization_seconds': 0., 'check_seconds': 0., 'save_seconds': 0.,
                   'common_setup_charged_seconds': common_seconds / 2, 'cache_build_seconds': cache_setup_seconds}, 'completed': False,
        'research_only': True, 'product_accepted': False}
    before_loss = baseline['replay_loss']
    flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
    modes = [(m, m.training) for m in runtime.model.modules()]
    hooks = [(m,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
    rng = torch.get_rng_state().clone()
    mps_rng = torch.mps.get_rng_state().clone() if parameter.device.type == 'mps' else None
    def persist():
        start = time.perf_counter()
        (output / (method + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        report['timing']['save_seconds'] += time.perf_counter() - start
    try:
        for i, (role, sample, replay) in enumerate(schedule(groups)):
            if i in (0,50,100) and sources is not None and any(sha(p)!=h for p,h in sources.items()):
                raise ValueError('Full fixed execution sources changed between checkpoints')
            if time.perf_counter() >= deadline:
                raise TimeoutError('Arm fixed3600sec including half common setup expired')
            _sync(parameter.device); start = time.perf_counter()
            actual = train_method_step(runtime, [sample], [replay], master, optimizer, reference, basis,
                method=method, ce_role=role, caches=caches if method == 'cache' else None)
            _sync(parameter.device); actual['step'] = i + 1
            actual['synchronized_step_wall_seconds'] = time.perf_counter() - start
            report['timing']['optimization_seconds'] += actual['synchronized_step_wall_seconds']
            if (actual.get('runtime_restored') is not True or actual.get('optimizer_steps') != 1
                    or actual.get('ce_role') != role or actual.get('ce_ids') != [sample['id']]
                    or actual.get('replay_kl_ids') != [replay['id']]):
                raise ValueError('Actual training helper changed optimizer/role/IDs/restoration')
            report['steps'].append(actual); report['completed_steps'] = i + 1; report['optimizer_steps'] += 1
            for key, row in ((role + '_ce', sample), ('replay_kl', replay)):
                counts = report['exposures'][key]; counts[row['id']] = counts.get(row['id'], 0) + 1
            if i + 1 in CHECKS:
                start = time.perf_counter()
                with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
                actual = checkpoint(runtime, groups, baseline, old_rows, before_loss, i + 1,
                    tensor_sha(parameter), deadline)
                if sources is not None and any(sha(p)!=h for p,h in sources.items()):
                    raise ValueError('Full fixed execution sources changed during checkpoint')
                state = optimizer.state[master]
                actual['optimizer_state_sha256'] = {key: tensor_sha(value.reshape(-1))
                    for key, value in state.items()}
                actual['adam_step'] = int(float(state['step']))
                actual['fp32_master_sha256'] = tensor_sha(master)
                actual['cumulative_delta_sha256'] = tensor_sha(master - reference)
                report['checks'].append(actual)
                with torch.no_grad(): parameter.copy_(original)
                _sync(parameter.device)
                report['timing']['check_seconds'] += time.perf_counter() - start
                persist()
                print(json.dumps({'method': method, 'step': i+1, 'candidate_correct': len(actual['provisional_new_correct_ids']),
                    'newly_confirmed': len(actual['newly_confirmed_ids']), 'forgotten': len(actual['forgotten_ids']),
                    'safety': actual['safety_passed'], 'elapsed_seconds': time.perf_counter()-began+common_seconds/2}), flush=True)
                if not actual['safety_passed']:
                    raise ValueError('Actual original-correct regression or original replay loss budget failed; arm stops')
        if time.perf_counter() >= deadline:
            raise TimeoutError('Final checks and save exceeded fixed arm deadline')
        report['completed'] = True
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        with torch.no_grad(): parameter.copy_(original)
        for p, flag, grad in flags: p.requires_grad_(flag); p.grad = grad
        for m, mode in modes: m.training = mode
        torch.set_rng_state(rng)
        if mps_rng is not None: torch.mps.set_rng_state(mps_rng)
        _sync(parameter.device)
        report['arm_state_restoration'] = {
            'flags_grads': all(p.requires_grad == f and p.grad is g for p, f, g in flags),
            'modes': all(m.training == v for m, v in modes),
            'hooks': all(len(m._forward_hooks) == a and len(m._forward_pre_hooks) == b for m, a, b in hooks),
            'rng': torch.equal(rng,torch.get_rng_state()),
            'mps_rng': mps_rng is None or torch.equal(mps_rng,torch.mps.get_rng_state())}
        if not all(report['arm_state_restoration'].values()): report['completed'] = False
        report['timing']['arm_wall_seconds'] = time.perf_counter() - began
        report['timing']['end_to_end_charged_seconds'] = report['timing']['arm_wall_seconds'] + common_seconds / 2 + cache_setup_seconds
        report['original_tensor_restored'] = tensor_sha(parameter) == tensor_sha(original)
        master = optimizer = reference = flags = modes = hooks = rng = mps_rng = p = m = None
        persist()
    return report


def execute(path, approved_sha):
    if sha(safe(path)) != approved_sha:
        raise ValueError('Exact root-approved sealed pilot SHA required')
    plan = json.loads(safe(path).read_text())
    if plan.get('protocol') != PROTOCOL or plan.get('sealed') is not True or plan.get('state') != 'sealed_waiting_explicit_MODEL_GO':
        raise ValueError('Only fixed separately sealed pilot150 may execute')
    began = time.perf_counter(); validate_bound(plan)
    output = safe(plan['output'])
    if output.exists():
        raise FileExistsError('Fresh single-attempt output required')
    write_new(safe(path).parent/'execution-started.json',
        {'protocol_sha256':approved_sha,'output':str(output),'started':time.time()})
    output.mkdir(mode=0o700)
    report = {'protocol': PROTOCOL, 'protocol_sha256': approved_sha, 'pid': os.getpid(),
        'model_started': False, 'passed': False, 'publication': False, 'teacher_calls': 0,
        'methods': [], 'source_before': plan['source_sha256']}
    write_new(output/'run-started.json', report)
    runtime = parameter = original = flags = modes = rng = mps_rng = basis = caches = hooks = backward_hooks = pointer_before = None
    try:
        import torch
        from luban.runtime import Runtime
        from luban.memory import Memory
        from luban.settings import Settings
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target, _capture_replay_subspace, _training_replay_generation, _loss
        from run_identity_coverage import qualify_required_replay, qualified_groups, verify_training_prefix_alignment
        from prepare_identity_coverage import qualify_original_replay_pool
        from verify_cached_training import capture_tail, _sync, tensor_sha
        from verify_automatic_batches import measured
        with tempfile.TemporaryDirectory(prefix='luban-method-pilot150-') as home:
            settings=Settings(home); settings.update({'storage_root':str(output/'private'),
                'model_path':plan['model_path'],'temperature':0,'local_thinking':'off'})
            runtime=Runtime(settings,Memory(output/'private'),research=True)
            if str(runtime.device)!=plan['device'] or str(runtime.dtype)!=plan['dtype']:
                raise ValueError('Actual native hardware/dtype changed')
            runtime._model_process_lock=ModelProcessLock(IDENTITY,runtime.device)
            common_started=time.perf_counter()
            with measured(runtime) as resources:
                runtime.load(paging=False); report['model_started']=True
                parameter=_target(runtime)[1]; original=parameter.detach().clone()
                if runtime.identity!=IDENTITY or runtime.active_core is not None or runtime.branch.expert is not None or tensor_sha(original)!=ORIGINAL:
                    raise ValueError('Exact originalNone1706 graph required')
                flags=[(p,p.requires_grad,p.grad) for p in runtime.model.parameters()]
                modes=[(m,m.training) for m in runtime.model.modules()]
                hooks=[(m,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backward_hooks=[(p,len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                pointer=Path(runtime.store)/'active-core.json'
                pointer_before=pointer.read_bytes() if pointer.exists() else None
                rng=torch.get_rng_state().clone(); mps_rng=torch.mps.get_rng_state().clone() if runtime.device=='mps' else None
                torch.manual_seed(42); runtime.model.eval().requires_grad_(False); parameter.requires_grad_(True)
                report['prefix']=verify_training_prefix_alignment(runtime,{**plan,'prefix_alignment_required':True})
                write_new(output/'qualification-started.json',{'protocol_sha256':approved_sha})
                pool=qualify_original_replay_pool(runtime,plan); required=qualify_required_replay(runtime,plan)
                report['qualification']={'pool':pool,'required':required}
                if pool['qualified_count']!=11 or required['passed'] is not True:
                    raise ValueError('Exact full57REPLAY/13old qualification failed; no substitution')
                groups=qualified_groups(plan,pool,required); old=_training_replay_generation(runtime,groups)
                if len(old)!=13 or len(groups['replay_training'])!=57:
                    raise ValueError('Actual full114/13 old groups changed')
                report['groups']=groups; report['groups_sha256']=content_sha(groups)
                # Qualification evidence can stand for the very same original TRAIN context/check only.
                qualified_actual={a['id']:a for a in pool['generation']+required['generation'] if a['passed']}
                baseline={}
                for key, rows in (('new',groups['new_training']),('replay',groups['replay_training'])):
                    baseline[key]=[]
                    for row in rows:
                        actual=qualified_actual.get(row['id'])
                        if actual is not None:
                            actual=copy.deepcopy(actual)
                            actual.update(sample_sha256=content_sha(row),actual_tensor_sha256=ORIGINAL,
                                core_version=None, actual_base_identity=IDENTITY, qualification_reused_same_context_check=True)
                        baseline[key].extend([actual] if actual is not None else probes(runtime,[row],expected_tensor_sha=ORIGINAL))
                    correct(rows,baseline[key])
                baseline['replay_loss']=_loss(runtime,groups['replay_training'])
                report['baseline']=baseline
                before_correct=correct(groups['replay_training'],baseline['replay'])
                old_rows=[r for r in groups['replay_training'] if r['id'] in before_correct]
                if not {r['id'] for r in old}.issubset(before_correct):
                    raise ValueError('Mandatory old proofs missing actual original correctness')
                report['baseline_preserved_replay_ids']=[r['id'] for r in old_rows]
                native_basis,basis_report=_capture_replay_subspace(runtime,parameter,groups['replay_training'])
                basis=native_basis.to(parameter.device); report['replay_svd']=basis_report
                common_seconds=time.perf_counter()-common_started + common_started-began
                report['common_setup_seconds']=common_seconds
                for method in METHODS:
                    torch.manual_seed(42)
                    caches={}
                    cache_started=time.perf_counter()
                    if method=='cache':
                        for row in groups['new_training']+groups['replay_training']:
                            if time.perf_counter()-cache_started+common_seconds/2>=3600:
                                raise TimeoutError('Cache setup alone exceeded arm end-to-end cap')
                            caches[row['id']]=capture_tail(runtime,row)
                    cache_seconds=time.perf_counter()-cache_started
                    arm_started=time.perf_counter()
                    actual=arm(runtime,groups,baseline,old_rows,original,basis,caches,method,
                        common_seconds,output,cache_setup_seconds=cache_seconds,sources=plan['source_sha256'])
                    actual['timing']['end_to_end_charged_seconds']=time.perf_counter()-arm_started+cache_seconds+common_seconds/2
                    if actual['timing']['end_to_end_charged_seconds']>3600:
                        actual['completed']=False
                        actual['error']={'type':'TimeoutError','message':'Complete arm including final save exceeded fixed3600 deadline'}
                    actual['cache_bytes']=sum(sum(c['bytes'].values()) for c in caches.values())
                    report['methods'].append(actual); caches=None
                    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                if len(report['methods'])==2:
                    one,two=report['methods']
                    report['same_update_trajectory']={
                        'paired_checkpoints': [{'step': a['step'], 'master_exact':a.get('fp32_master_sha256')==b.get('fp32_master_sha256'),
                            'adam_exact':a.get('optimizer_state_sha256')==b.get('optimizer_state_sha256'),
                            'delta_exact':a.get('cumulative_delta_sha256')==b.get('cumulative_delta_sha256')}
                            for a,b in zip(one['checks'],two['checks']) if a['step']==b['step']],
                        'scope':'actual150 paired checkpoint full-tensor hashes, not inferred from prior three-row audit'}
                pairs=report.get('same_update_trajectory',{}).get('paired_checkpoints',[])
                report['method_equivalence_passed']=(len(pairs)==3 and [p['step'] for p in pairs]==list(CHECKS)
                    and all(p['master_exact'] and p['adam_exact'] and p['delta_exact'] for p in pairs))
                report['passed']=report['method_equivalence_passed'] and all(r['completed'] and r['original_tensor_restored'] for r in report['methods'])
                report['resources']=resources
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        if runtime is not None:
            try:
                import torch
                from verify_cached_training import tensor_sha
                if runtime.model is not None and original is not None:
                    with torch.no_grad(): parameter.copy_(original)
                if flags is not None and modes is not None and rng is not None:
                    for p,flag,grad in flags:p.requires_grad_(flag);p.grad=grad
                    for m,mode in modes:m.training=mode
                    torch.set_rng_state(rng)
                    if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
                    report['restoration']={'original_tensor':tensor_sha(parameter)==ORIGINAL,
                        'flags_grads':all(p.requires_grad==f and p.grad is g for p,f,g in flags),
                        'modes':all(m.training==v for m,v in modes),'rng':torch.equal(rng,torch.get_rng_state()),
                        'mps_rng':mps_rng is None or torch.equal(mps_rng,torch.mps.get_rng_state()),
                        'core_branch':runtime.active_core is None and runtime.branch.expert is None,
                        'forward_hooks':all(len(m._forward_hooks)==a and len(m._forward_pre_hooks)==b for m,a,b in hooks),
                        'backward_hooks':all(len(p._backward_hooks or {})==v for p,v in backward_hooks),
                        'pointer':(pointer.read_bytes() if pointer.exists() else None)==pointer_before}
            except BaseException as error:
                report['restoration_error']={'type':type(error).__name__,'message':str(error)}
                report['passed']=False
            # Never let traceback/local tensor aliases keep the graph live when releasing the OS lock.
            parameter=original=flags=modes=rng=mps_rng=basis=caches=native_basis=hooks=backward_hooks=pointer_before=None
            p=m=None
            try:
                runtime._release_model()
                from luban.process_lock import ModelProcessLock
                lock=ModelProcessLock(IDENTITY,plan['device']);lock.close()
                report['model_released']=runtime.model is None and runtime._model_process_lock is None
                report['actual_os_lock_reacquired_closed']=True
            except BaseException as error:
                report['release_error']={'type':type(error).__name__,'message':str(error)}
                report['passed']=False
        try:
            from verify_router_coverage_study import model_binding
            report['source_after']={name:sha(name) for name in plan['source_sha256']}
            report['production_after']=production_snapshot(plan['production_root'])
            report['official_after']=model_binding(plan['model_path'],IDENTITY)
            report['preserved']=report['source_after']==plan['source_sha256'] and report['production_after']==plan['production_snapshot'] and report['official_after']==plan['official_model_binding']
        except BaseException as error:
            report['preservation_error']={'type':type(error).__name__,'message':str(error)};report['preserved']=False
        report['wall_seconds']=time.perf_counter()-began
        report['passed']=report['passed'] and report.get('preserved') is True and len(report.get('restoration',{}))==9 and all(report['restoration'].values()) and report.get('model_released') is True
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if not report['passed']:raise RuntimeError('Fixed pilot incomplete/rejected; preserve raw, no retries or publication')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for action in ('prepare','bind','seal','run'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--approved-sha');args=parser.parse_args()
    if args.run:
        if not args.approved_sha:parser.error('Exact approvedSHA and rootMODELGO required')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if not args.output:parser.error('Fresh future result directory required')
        write_new(args.prepare,prepare(args.output))
    elif args.bind:
        if not args.output:parser.error('NEW bound file required')
        write_new(args.output,bind(json.loads(safe(args.bind).read_text())))
    elif args.seal:
        if not args.output:parser.error('NEW formal file required')
        value=json.loads(safe(args.seal).read_text());validate_bound(value)
        value.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO');write_new(args.output,value)
    else:print(json.dumps({'protocol':PROTOCOL,'model_started':False,'torch_imported':False,'steps':150,'checks':list(CHECKS),'publication':False}))


if __name__=='__main__':main()
