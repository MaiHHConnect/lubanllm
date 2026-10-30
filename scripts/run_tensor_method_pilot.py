"""Independent original57 cacheB2/B4 G1 pilot150 CE events; SOURCE until explicit GO."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import run_training_method_pilot as prior
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, content_sha, safe, write_new, production_snapshot
from prepare_tensor_method_schedule import build_schedule

PROTOCOL = 'original57-cache-b2-b4-g1-pilot150-ce-tokenizer-eos-v2'
CHECKS = (50, 100, 150)
BATCHES = (2, 4)
MATERIAL_CONTRACT = 'qualified114-fullrow-except-original-generation-v1'


def training_materials(groups):
    """Keep complete rows/ordered IDs; omit only each run's generation proof."""
    if not isinstance(groups, dict) or set(groups)!={'new_training','replay_training','new_validation','replay_validation'} or {k:len(v) for k,v in groups.items() if isinstance(v,list)} != {
            'new_training':57,'replay_training':57,'new_validation':0,'replay_validation':0}:
        raise ValueError('B1 comparison requires complete fixed114 TRAIN and no held-out rows')
    result={}
    ids=[]
    for key in ('new_training','replay_training'):
        result[key]=[]
        for row in groups[key]:
            if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id']:
                raise ValueError('Complete ordered training material IDs required')
            material=copy.deepcopy(row)
            provenance=json.loads(material['provenance'])
            if not isinstance(provenance,dict):raise ValueError('Training provenance must be an object')
            provenance.pop('original_baseline_generation',None)
            material['provenance']=provenance
            result[key].append({'id':row['id'],'material_sha256':content_sha(material)})
            ids.append(row['id'])
    if len(set(ids))!=114:raise ValueError('Duplicate full114 training IDs')
    return result


def verify_training_materials(groups, reference):
    if reference.get('training_material_contract')!=MATERIAL_CONTRACT:
        raise ValueError('Explicit full114 material comparison contract required')
    path=safe(reference['path'])
    if sha(path)!=reference['sha256']:raise ValueError('Actual B1 raw SHA changed')
    value=json.loads(path.read_text())
    if (sha(path)!=reference['sha256'] or content_sha(value['groups'])!=value.get('groups_sha256')
            or value['groups_sha256']!=reference['groups_sha256']
            or training_materials(value['groups'])!=reference['training_materials']
            or content_sha(reference['training_materials'])!=reference['training_material_sha256']):
        raise ValueError('Actual B1 raw/group/material binding changed')
    if training_materials(groups)!=reference['training_materials']:
        raise ValueError('B2/B4 qualified ordered IDs or complete training material differ from actual B1')


def reference_receipt(path):
    path = safe(path)
    before=sha(path)
    value = json.loads(path.read_text())
    if (value.get('protocol') not in ('original57-native-cache-b1-pilot150-v1',
            'original57-native-cache-b1-pilot150-tokenizer-eos-v2') or value.get('model_started') is not True
            or type(value.get('passed')) is not bool or value.get('preserved') is not True
            or value.get('model_released') is not True or value.get('actual_os_lock_reacquired_closed') is not True
            or value.get('teacher_calls') != 0 or value.get('publication') is not False
            or len(value.get('restoration', {})) != 9 or not all(v is True for v in value['restoration'].values())
            or [arm.get('method') for arm in value.get('methods', [])] not in
                (['native'], ['native', 'cache'])):
        raise ValueError('B1 reference requires actual released/restored raw, not quality success or invented150')
    if sha(path)!=before or content_sha(value.get('groups'))!=value.get('groups_sha256'):
        raise ValueError('Actual B1 complete groups/raw binding missing or changed')
    materials=training_materials(value['groups'])
    return {'path': str(path), 'sha256': before, 'protocol':value['protocol'], 'passed': value['passed'],
            'groups_sha256':value['groups_sha256'],'training_material_contract':MATERIAL_CONTRACT,
            'training_materials':materials,'training_material_sha256':content_sha(materials),
            'available_optimizer_steps': {arm['method']: arm.get('completed_steps', 0) for arm in value['methods']},
            'scope': 'Separate actual B1 raw only. Partial/failedB1 is not a complete150 comparator; no shared baseline observations borrowed.'}


def prepare(output, b1_result=None):
    base = prior.prepare(output)
    sources = dict(base['source_sha256'])
    for name in ('scripts/run_tensor_method_pilot.py', 'scripts/prepare_tensor_method_schedule.py',
                 'tests/test_tensor_method_pilot.py', 'tests/test_tensor_method_schedule.py'):
        sources[str(ROOT/name)] = sha(ROOT/name)
    reference = reference_receipt(b1_result) if b1_result is not None else None
    return {k:v for k,v in base.items() if k not in {
        'methods','optimizer_steps','tensor_batch','schedule','exposure_ledger_per_arm','baseline',
        'generation_upper_bound','original_baseline_remaining_generations','generation_cost','scope','timing','stop'}} | {
        'protocol': PROTOCOL, 'source_sha256': dict(sorted(sources.items())), 'b1_actual_reference': reference,
        'training_eos_contract': 'runtime-sample-tokens-tokenizer-eos-v1',
        'training_material_contract': MATERIAL_CONTRACT,
        'methods': ['cacheB2','cacheB4'], 'tensor_batches': list(BATCHES), 'ce_events': 150,
        'actual_optimizer_updates': {'cacheB2':78,'cacheB4':42},
        'schedule_protocol': 'fixed57-g1-ce-events-same-role-batch-exact-checks-v1',
        'schedule': 'Exact50CE-event flush; same-role consecutiveG1 batches, NEWbatch thenREPLAYbatch; original pair/cycle IDs preserved; cross-role physical order andAdam trajectory differ fromB1.',
        'exposure_ledger_per_arm': {'new_ce':75,'replay_ce':75,'replay_kl':150},
        'baseline': 'This execution independently generates all114 originalNone TRAIN once after fresh qualification; no oldB1 or qualification answer is reused as baseline.',
        'original_baseline_remaining_generations':114, 'generation_upper_bound':926,
        'generation_cost': '14 qualification +114 independent original baseline + at most399 candidate checks/arm=926; noDEV.',
        'quality_criteria': {'positive_finite_confirmed_growth':True,'all_original_correct_protected':True,
                             'native_replay_loss_ratio_max':1.05,'complete_ce_events':150,'arm_seconds_max':3600},
        'stop': 'Each arm stops on safety/nonfinite/deadline; no retries, substitution, gain earlystop or automatic publication. No gain is qualityFalse, not an operator-equivalence failure.',
        'cache': 'All114 original fullseq captured once, shared immutableCPU cache. Its setup cost is common and split half into each arm.',
        'timing': 'Synchronized phasewall; common validation/load/qualification/all114baseline/basis/cache half charged per arm; own compute/transfers/hash/check/save/restore fully charged. FixedB2thenB4 order confounded.',
        'scope': 'Independent optimization methods, not sequentialAdam numerical equivalence; oldstrict CPU failures remainFalse. Actual B1 raw separately hash-referenced; no product/acceptance/sweetspot/generalization claim.'}


def bind(plan):
    from verify_router_coverage_study import model_binding
    if plan.get('b1_actual_reference') is None:
        raise ValueError('Actual released B1 raw hash required before binding; SOURCE preparation may omit it')
    if prepare(plan['output'], plan['b1_actual_reference']['path']) != plan:
        raise ValueError('Tensor method source/data/reference changed before binding')
    return {**copy.deepcopy(plan), 'official_model_binding':model_binding(plan['model_path'],IDENTITY),
            'production_snapshot':production_snapshot(plan['production_root']), 'state':'bound_waiting_root_seal_MODEL_GO'}


def validate_bound(plan):
    reference = plan.get('b1_actual_reference')
    if not isinstance(reference,dict): raise ValueError('ActualB1 reference required')
    template={**copy.deepcopy(plan),'state':'unbound_waiting_root_review','sealed':False,
              'official_model_binding':None,'production_snapshot':None}
    if prepare(plan['output'],reference['path']) != template or bind(template) != {
            **plan,'sealed':False,'state':'bound_waiting_root_seal_MODEL_GO'}:
        raise ValueError('Fixed tensor source/material/reference/official/production binding changed')


def arm(runtime, groups, baseline, old_rows, original, basis, caches, batch_size, common_seconds, output, sources=None):
    import torch
    from cached_method_steps import train_method_step
    from luban.evolution import _target
    from verify_cached_training import _sync, tensor_sha
    if type(batch_size) is not int or batch_size not in BATCHES or not math.isfinite(common_seconds) or common_seconds < 0:
        raise ValueError('Only fixedB2/B4 and finite common cost')
    plan=build_schedule([r['id'] for r in groups['new_training']], [r['id'] for r in groups['replay_training']],150,batch_size)
    rows={r['id']:r for r in groups['new_training']+groups['replay_training']}
    parameter=_target(runtime)[1]
    if tensor_sha(parameter)!=tensor_sha(original): raise ValueError('Each arm must start from originalW')
    _sync(parameter.device);began=time.perf_counter();deadline=began+3600-common_seconds/2
    master=torch.nn.Parameter(original.float().clone());reference=original.float()
    optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
    flags=[(p,p.requires_grad,p.grad) for p in runtime.model.parameters()]
    modes=[(m,m.training) for m in runtime.model.modules()]
    hooks=[(m,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
    rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone() if parameter.device.type=='mps' else None
    report={'method':'cacheB'+str(batch_size),'tensor_batch':batch_size,'schedule':plan,'updates':[],'checks':[],
        'completed_ce_events':0,'optimizer_steps':0,'exposures':{k:{} for k in ('new_ce','replay_ce','replay_kl')},
        'completed':False,'quality_passed':False,'operator_equivalence_claim':False,'product_accepted':False,
        'timing':{'optimization_seconds':0.,'check_seconds':0.,'save_seconds':0.,'common_setup_charged_seconds':common_seconds/2}}
    def persist():
        start=time.perf_counter()
        (output/(report['method']+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report['timing']['save_seconds']+=time.perf_counter()-start
    def sources_unchanged():
        if sources is not None and any(sha(p)!=h for p,h in sources.items()):
            raise ValueError('Full bound sources changed during tensor method')
    try:
        for segment in plan['segments']:
            sources_unchanged()
            for batch in segment['batches']:
                if time.perf_counter()>=deadline: raise TimeoutError('Tensor arm3600sec end-to-end cap')
                samples=[rows[i] for i in batch['ce_ids']];replays=[rows[i] for i in batch['replay_kl_ids']]
                _sync(parameter.device);start=time.perf_counter()
                actual=train_method_step(runtime,samples,replays,master,optimizer,reference,basis,
                    method='cache',ce_role=batch['ce_role'],caches=caches)
                _sync(parameter.device);wall=time.perf_counter()-start
                if (actual.get('runtime_restored') is not True or actual.get('optimizer_steps')!=1
                        or actual.get('ce_role')!=batch['ce_role'] or actual.get('ce_ids')!=batch['ce_ids']
                        or actual.get('replay_kl_ids')!=batch['replay_kl_ids']):
                    raise ValueError('Actual batch helper changed fixed role/IDs/optimizer/restoration')
                actual.update(schedule_batch=batch,synchronized_update_wall_seconds=wall)
                report['updates'].append(actual);report['optimizer_steps']+=1
                report['completed_ce_events']+=len(samples);report['timing']['optimization_seconds']+=wall
                for key,ids in ((batch['ce_role']+'_ce',batch['ce_ids']),('replay_kl',batch['replay_kl_ids'])):
                    for id in ids: report['exposures'][key][id]=report['exposures'][key].get(id,0)+1
            start=time.perf_counter()
            with torch.no_grad():parameter.copy_(master.to(parameter.dtype))
            actual=prior.checkpoint(runtime,groups,baseline,old_rows,baseline['replay_loss'],
                segment['check_ce_event'],tensor_sha(parameter),deadline)
            actual['ce_events']=actual.pop('step');actual['optimizer_updates']=report['optimizer_steps']
            state=optimizer.state[master]
            actual['adam_step']=int(float(state['step']))
            if actual['adam_step']!=report['optimizer_steps']:raise ValueError('Adam updates do not match batch ledger')
            actual['optimizer_state_sha256']={k:tensor_sha(v.reshape(-1)) for k,v in state.items()}
            actual['fp32_master_sha256']=tensor_sha(master);actual['cumulative_delta_sha256']=tensor_sha(master-reference)
            report['checks'].append(actual)
            with torch.no_grad():parameter.copy_(original)
            _sync(parameter.device);report['timing']['check_seconds']+=time.perf_counter()-start
            sources_unchanged();persist()
            print(json.dumps({'method':report['method'],'ce_events':actual['ce_events'],
                'optimizer_updates':report['optimizer_steps'],'newly_confirmed':len(actual['newly_confirmed_ids']),
                'forgotten':len(actual['forgotten_ids']),'safety':actual['safety_passed']}),flush=True)
            if actual['safety_passed'] is not True:raise ValueError('Original-correct/loss safety failed; arm stopped')
        if time.perf_counter()>=deadline:raise TimeoutError('Checks/save exceeded fixed arm cap')
        if report['optimizer_steps']!=plan['optimizer_updates'] or any(
                {id:report['exposures'][key].get(id,0) for id in counts}!=counts for key,counts in plan['per_id_exposures'].items()):
            raise ValueError('Final tensor exposure/optimizer ledger changed')
        report['completed']=True
        report['quality_passed']=bool(report['checks'][-1]['newly_confirmed_ids'])
        if not report['quality_passed']:report['quality_rejection']='No positive finite baseline-to-double-proof learning gain'
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:
        with torch.no_grad():parameter.copy_(original)
        for p,flag,grad in flags:p.requires_grad_(flag);p.grad=grad
        for m,mode in modes:m.training=mode
        torch.set_rng_state(rng)
        if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
        _sync(parameter.device)
        report['arm_state_restoration']={'flags_grads':all(p.requires_grad==f and p.grad is g for p,f,g in flags),
            'modes':all(m.training==v for m,v in modes),'rng':torch.equal(rng,torch.get_rng_state()),
            'mps_rng':mps_rng is None or torch.equal(mps_rng,torch.mps.get_rng_state()),
            'hooks':all(len(m._forward_hooks)==a and len(m._forward_pre_hooks)==b for m,a,b in hooks)}
        report['original_tensor_restored']=tensor_sha(parameter)==tensor_sha(original)
        report['timing']['arm_wall_seconds']=time.perf_counter()-began
        report['timing']['end_to_end_charged_seconds']=report['timing']['arm_wall_seconds']+common_seconds/2
        if (not all(report['arm_state_restoration'].values()) or not report['original_tensor_restored']
                or report['timing']['end_to_end_charged_seconds']>=3600):
            report['completed']=report['quality_passed']=False
        master=optimizer=reference=flags=modes=hooks=rng=mps_rng=p=m=None
        persist()
        report['timing']['arm_wall_seconds']=time.perf_counter()-began
        report['timing']['end_to_end_charged_seconds']=report['timing']['arm_wall_seconds']+common_seconds/2
        if time.perf_counter()>=deadline:
            report['completed']=report['quality_passed']=False
            report['error']={'type':'TimeoutError','message':'Final raw save exceeded fixed3600 arm cap'}
            persist()
    return report


def execute(path, approved_sha):
    if not isinstance(approved_sha,str) or re.fullmatch(r'[0-9a-f]{64}',approved_sha) is None or sha(safe(path)) != approved_sha:
        raise ValueError('Exact root-approved sealed pilot SHA required')
    plan = json.loads(safe(path).read_text())
    if plan.get('protocol') != PROTOCOL or plan.get('sealed') is not True or plan.get('state') != 'sealed_waiting_explicit_MODEL_GO':
        raise ValueError('Only fixed separately sealed pilot150 may execute')
    began = time.perf_counter(); validate_bound(plan)
    output = safe(plan['output'])
    if output.exists(): raise FileExistsError('Fresh single-attempt output required')
    write_new(safe(path).parent/'execution-started.json',{'protocol_sha256':approved_sha,'output':str(output),'started':time.time()})
    output.mkdir(mode=0o700)
    report = {'protocol': PROTOCOL, 'protocol_sha256': approved_sha, 'pid': os.getpid(),
        'model_started': False, 'passed': False, 'publication': False, 'teacher_calls': 0,
        'methods': [], 'source_before': plan['source_sha256'], 'production_before':plan['production_snapshot'],
        'b1_actual_reference':plan['b1_actual_reference'], 'operator_equivalence_claim':False, 'started':time.time()}
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
                groups=qualified_groups(plan,pool,required)
                verify_training_materials(groups,plan['b1_actual_reference'])
                old=_training_replay_generation(runtime,groups)
                if len(old)!=13 or len(groups['replay_training'])!=57:
                    raise ValueError('Actual full114/13 old groups changed')
                report['groups']=groups; report['groups_sha256']=content_sha(groups)
                baseline={key:prior.probes(runtime,rows,expected_tensor_sha=ORIGINAL)
                    for key,rows in (('new',groups['new_training']),('replay',groups['replay_training']))}
                baseline['replay_loss']=_loss(runtime,groups['replay_training'])
                if not math.isfinite(baseline['replay_loss']) or baseline['replay_loss']<=0:
                    raise ValueError('Original native REPLAY baseline loss must be finite and positive')
                report['baseline']=baseline
                before_correct=prior.correct(groups['replay_training'],baseline['replay'])
                old_rows=[r for r in groups['replay_training'] if r['id'] in before_correct]
                if not {r['id'] for r in old}.issubset(before_correct):
                    raise ValueError('Mandatory old proofs missing actual original correctness')
                report['baseline_preserved_replay_ids']=[r['id'] for r in old_rows]
                native_basis,basis_report=_capture_replay_subspace(runtime,parameter,groups['replay_training'])
                basis=native_basis.to(parameter.device); report['replay_svd']=basis_report
                caches={};cache_started=time.perf_counter()
                for row in groups['new_training']+groups['replay_training']:
                    if (time.perf_counter()-began)/2>=3600:
                        raise TimeoutError('Common setup alone exhausted both arm budgets')
                    caches[row['id']]=capture_tail(runtime,row)
                _sync(parameter.device)
                report['common_cache_build_seconds']=time.perf_counter()-cache_started
                report['cache_bytes']=sum(sum(c['bytes'].values()) for c in caches.values())
                common_seconds=time.perf_counter()-began;report['common_setup_seconds']=common_seconds
                for batch_size in BATCHES:
                    if tensor_sha(parameter)!=ORIGINAL:raise ValueError('Fresh originalW required between arms')
                    torch.manual_seed(plan['seed'])
                    arm_started=time.perf_counter()
                    actual=arm(runtime,groups,baseline,old_rows,original,basis,caches,batch_size,
                        common_seconds,output,sources=plan['source_sha256'])
                    actual['timing']['end_to_end_charged_seconds']=time.perf_counter()-arm_started+common_seconds/2
                    if actual['timing']['end_to_end_charged_seconds']>=3600:
                        actual['completed']=actual['quality_passed']=False
                        actual['error']={'type':'TimeoutError','message':'Arm restore/final save exceeded fixed3600 cap'}
                    report['methods'].append(actual)
                    (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                    if not actual['original_tensor_restored'] or not all(actual['arm_state_restoration'].values()):
                        raise ValueError('Unsafe arm restoration; do not start the next arm')
                report['execution_completed']=len(report['methods'])==2 and all(r['completed'] for r in report['methods'])
                report['passed']=report['execution_completed'] and all(r['quality_passed'] for r in report['methods'])
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
            report['b1_reference_after_sha256']=sha(plan['b1_actual_reference']['path'])
            report['preserved']=report['source_after']==plan['source_sha256'] and report['production_after']==plan['production_snapshot'] and report['official_after']==plan['official_model_binding'] and report['b1_reference_after_sha256']==plan['b1_actual_reference']['sha256']
        except BaseException as error:
            report['preservation_error']={'type':type(error).__name__,'message':str(error)};report['preserved']=False
        report['wall_seconds']=time.perf_counter()-began
        report['passed']=report['passed'] and report.get('preserved') is True and len(report.get('restoration',{}))==9 and all(report['restoration'].values()) and report.get('model_released') is True and report.get('actual_os_lock_reacquired_closed') is True
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if not report['passed']:raise RuntimeError('Fixed pilot incomplete/rejected; preserve raw, no retries or publication')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for action in ('prepare','bind','seal','run'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--approved-sha');parser.add_argument('--b1-result',type=Path)
    args=parser.parse_args()
    if args.b1_result and not args.prepare:parser.error('b1-result is only an explicit static prepare input')
    if args.approved_sha and not args.run:parser.error('approved-sha is only valid for the separate run entry')
    if args.output and not (args.prepare or args.bind or args.seal):parser.error('output is only a static preparation destination')
    if args.run:
        if not args.approved_sha:parser.error('Exact root-approvedSHA and separate MODELGO required')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if not args.output:parser.error('Fresh future result directory required')
        write_new(args.prepare,prepare(args.output,args.b1_result))
    elif args.bind:
        if not args.output:parser.error('NEW bound file required')
        write_new(args.output,bind(json.loads(safe(args.bind).read_text())))
    elif args.seal:
        if not args.output:parser.error('NEW sealed file required')
        value=json.loads(safe(args.seal).read_text());validate_bound(value)
        value.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO');write_new(args.output,value)
    else:print(json.dumps({'protocol':PROTOCOL,'model_started':False,'torch_imported':False,'ce_events':150,
        'actual_optimizer_updates':{'cacheB2':78,'cacheB4':42},'checks_ce_events':list(CHECKS),'publication':False}))


if __name__=='__main__':main()
