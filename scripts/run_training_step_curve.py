"""Independent original57 TRAIN curve at50..700+704 CE events; default SOURCE."""
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

import run_tensor_method_pilot as tensor
import run_training_method_pilot as prior
from prepare_tensor_method_schedule import build_schedule
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, safe, write_new, production_snapshot

PROTOCOL='original57-single-method-curve704-ce-tokenizer-eos-v1'
CHECKS=tuple(range(50,701,50))+(704,)
DEADLINE_SECONDS=3600


def prepare(output, b1_result, method, batch_size, confirm_ce_event=None):
    if method not in ('native','cache') or type(batch_size) is not int or batch_size not in (1,2,4) or method=='native' and batch_size!=1:
        raise ValueError('Explicit nativeB1 or cacheB1/B2/B4 required; root selects method after study')
    if confirm_ce_event is not None and (type(confirm_ce_event) is not int or confirm_ce_event not in CHECKS[:-1]):
        raise ValueError('Optional second proof must preregister one saved50..700 point')
    base=tensor.prepare(output,b1_result)
    if base['b1_actual_reference'] is None:raise ValueError('Complete actual B1 reference required for curve SOURCE')
    raw=json.loads(safe(b1_result).read_text())
    if base['b1_actual_reference']['available_optimizer_steps']!={'native':150,'cache':150}:
        raise ValueError('Curve comparison requires complete actual B1 native/cache150')
    new_correct=prior.correct(raw['groups']['new_training'],raw['baseline']['new'])
    replay_correct=prior.correct(raw['groups']['replay_training'],raw['baseline']['replay'])
    if len(new_correct)!=13 or len(replay_correct)!=55:raise ValueError('Fixed actual original13NEW/55REPLAY protection required')
    schedule=build_schedule([r['id'] for r in raw['groups']['new_training']],
                            [r['id'] for r in raw['groups']['replay_training']],704,batch_size)
    sources=dict(base['source_sha256'])
    for name in ('scripts/run_training_step_curve.py','tests/test_training_step_curve.py'):
        sources[str(ROOT/name)]=sha(ROOT/name)
    plan={k:v for k,v in base.items() if k not in ('methods','tensor_batches','ce_events','actual_optimizer_updates',
        'schedule','exposure_ledger_per_arm','generation_upper_bound','generation_cost','quality_criteria','stop','cache','timing','scope')}
    plan.update(protocol=PROTOCOL,source_sha256=dict(sorted(sources.items())),method=method,tensor_batch=batch_size,
        ce_events=704,actual_optimizer_updates=schedule['optimizer_updates'],checkpoints=list(CHECKS),
        confirm_ce_event=confirm_ce_event,protection_ids={'new':sorted(new_correct),'replay':sorted(replay_correct)},
        schedule_sha256=tensor.content_sha(schedule),exposure_ledger={'new_ce':352,'replay_ce':352,'replay_kl':704},
        deadline_seconds=DEADLINE_SECONDS,generation_upper_bound=1865+(112 if confirm_ce_event else 0),
        generation_cost='14 qualifications+114 independent original baseline+15*(57NEW+55oldREPLAY)+57final secondNEW; optional savedpoint adds57NEW+55REPLAY; noDEV',
        checkpoint_weights='Every point actually saves complete native BF16lastdown CPU tensor and verifies file/tensorSHA before first proof; actualbytes recorded, not assumed25MiB.',
        mastery='50..700 only provisional single-proof correct/growth;704 frozenW twoNEWproofs. Optional preregistered middlepoint reloads actual savedW for secondNEW+allold proof andloss, charged.',
        timing='Single run3600s includesvalidation/load/qualification/baseline/SVD/cache/optimization/check/save/restore/finalraw; no common cost halving. Perinterval, pureopt, fullrunwall andsampledresources separate.',
        stop='Safety/finite/deadline failure aborts without retry/substitute; no gain earlystop. No production/teacher/publication/automatic selectedcheckpoint activation.',
        scope='One explicitly selected method/B, fixed historicalTRAIN curve;704 isCEevents, notAdam forB>1. Not sequentialAdam equivalence, generalization or proven sweetspot.')
    return plan


def bind(plan):
    from verify_router_coverage_study import model_binding
    if prepare(plan['output'],plan['b1_actual_reference']['path'],plan['method'],plan['tensor_batch'],plan['confirm_ce_event'])!=plan:
        raise ValueError('Curve source/material/method/reference changed before binding')
    return {**copy.deepcopy(plan),'official_model_binding':model_binding(plan['model_path'],IDENTITY),
            'production_snapshot':production_snapshot(plan['production_root']),'state':'bound_waiting_root_seal_MODEL_GO'}


def validate_bound(plan):
    template={**copy.deepcopy(plan),'state':'unbound_waiting_root_review','sealed':False,
              'official_model_binding':None,'production_snapshot':None}
    if bind(template)!={**plan,'sealed':False,'state':'bound_waiting_root_seal_MODEL_GO'}:
        raise ValueError('Full curve source/model/production/reference binding changed')


def frozen_check(runtime,groups,baseline,old_rows,candidate_sha,deadline,*,second=False):
    """All original knowledge protection; no special150 checkpoint behavior."""
    from luban.evolution import _target,_loss
    from verify_cached_training import tensor_sha
    def frozen():
        if time.perf_counter()>=deadline:raise TimeoutError('Curve proof/loss deadline exceeded')
        if tensor_sha(_target(runtime)[1])!=candidate_sha:raise ValueError('Actual frozen check tensor changed')
    news=groups['new_training'];frozen()
    first=prior.probes(runtime,news,deadline,candidate_sha);frozen()
    again=prior.probes(runtime,news,deadline,candidate_sha) if second else [];frozen()
    old=prior.probes(runtime,old_rows,deadline,candidate_sha);frozen()
    old_new=prior.correct(news,baseline['new']);now=prior.correct(news,first)
    old_replay={r['id'] for r in old_rows};replayed=prior.correct(old_rows,old)
    forgotten=(old_new-now)|(old_replay-replayed)
    if second:forgotten|=old_new-prior.correct(news,again)
    replay_loss=_loss(runtime,groups['replay_training']);frozen()
    if not math.isfinite(replay_loss):raise ValueError('Curve native replayloss is nonfinite')
    finite_before=prior.correct(news,baseline['new'],finite=True)
    finite_now=prior.correct(news,first,finite=True)
    confirmed=(finite_now&prior.correct(news,again,finite=True))-finite_before if second else None
    return {'candidate_tensor_sha256':candidate_sha,'new_first':first,'new_second':again,'old_preservation':old,
        'provisional_correct_ids':sorted(finite_now),'provisional_growth_ids':sorted(finite_now-finite_before),
        'newly_confirmed_ids':sorted(confirmed) if confirmed is not None else None,'mastery_confirmed':second,
        'forgotten_ids':sorted(forgotten),'replay_loss':replay_loss,'baseline_replay_loss':baseline['replay_loss'],
        'safety_passed':not forgotten and replay_loss<=baseline['replay_loss']*1.05,'publication':False}


def save_weights(parameter,path):
    import torch
    from verify_cached_training import tensor_sha
    if parameter.dtype!=torch.bfloat16 or not torch.isfinite(parameter).all():
        raise ValueError('Checkpoint requires actual finite native BF16 full lastdown')
    candidate_sha=tensor_sha(parameter)
    cpu=parameter.detach().cpu().clone()
    with safe(path).open('xb') as stream:
        torch.save(cpu,stream);stream.flush();os.fsync(stream.fileno())
    if tensor_sha(cpu)!=candidate_sha or tensor_sha(parameter)!=candidate_sha:
        raise ValueError('Checkpoint transfer changed actual frozen weights')
    return {'path':str(path),'file_sha256':sha(path),'tensor_sha256':candidate_sha,
            'tensor_bytes':cpu.numel()*cpu.element_size(),'file_bytes':path.stat().st_size,
            'shape':list(cpu.shape),'dtype':str(cpu.dtype)}


def confirm_saved_point(runtime,groups,baseline,old_rows,point,deadline):
    import torch
    from luban.evolution import _target,_loss
    from verify_cached_training import tensor_sha
    saved=point['saved_weights'];path=safe(saved['path']);parameter=_target(runtime)[1]
    if sha(path)!=saved['file_sha256']:raise ValueError('Saved intermediate checkpoint bytes changed')
    weight=torch.load(path,map_location='cpu',weights_only=True)
    if (not isinstance(weight,torch.Tensor) or weight.dtype!=torch.bfloat16 or list(weight.shape)!=saved['shape']
            or weight.shape!=parameter.shape or not torch.isfinite(weight).all()
            or tensor_sha(weight)!=saved['tensor_sha256'] or sha(path)!=saved['file_sha256']):
        raise ValueError('Saved intermediate native checkpoint binding changed')
    before=parameter.detach().clone()
    try:
        with torch.no_grad():parameter.copy_(weight)
        news=groups['new_training'];candidate=saved['tensor_sha256']
        again=prior.probes(runtime,news,deadline,candidate)
        old=prior.probes(runtime,old_rows,deadline,candidate)
        loss=_loss(runtime,groups['replay_training'])
        if time.perf_counter()>=deadline or tensor_sha(parameter)!=candidate or not math.isfinite(loss):
            raise ValueError('Savedpoint second proof changedW/nonfinite/exceededdeadline')
        before_correct=prior.correct(news,baseline['new']);now=prior.correct(news,again)
        forgotten=(before_correct-now)|({r['id'] for r in old_rows}-prior.correct(old_rows,old))
        confirmed=(prior.correct(news,point['new_first'],finite=True)&prior.correct(news,again,finite=True))-prior.correct(news,baseline['new'],finite=True)
        return {'ce_event':point['ce_events'],'saved_weights':saved,'new_second':again,'old_preservation':old,
            'newly_confirmed_ids':sorted(confirmed),'forgotten_ids':sorted(forgotten),'replay_loss':loss,
            'mastery_confirmed':True,'safety_passed':point['safety_passed'] and not forgotten and loss<=baseline['replay_loss']*1.05,
            'publication':False}
    finally:
        with torch.no_grad():parameter.copy_(before)


def curve(runtime,groups,baseline,old_rows,original,basis,caches,plan,output,deadline):
    import torch
    from cached_method_steps import train_method_step
    from luban.evolution import _target
    from verify_cached_training import _sync,tensor_sha
    schedule=build_schedule([r['id'] for r in groups['new_training']],[r['id'] for r in groups['replay_training']],704,plan['tensor_batch'])
    if tensor.content_sha(schedule)!=plan['schedule_sha256']:raise ValueError('Complete bound curve schedule changed')
    rows={r['id']:r for r in groups['new_training']+groups['replay_training']};parameter=_target(runtime)[1]
    if tensor_sha(parameter)!=tensor_sha(original):raise ValueError('Fresh originalW curve required')
    master=torch.nn.Parameter(original.float().clone());reference=original.float()
    optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
    began=time.perf_counter();interval_start=began;interval_opt=0.
    report={'method':plan['method'],'tensor_batch':plan['tensor_batch'],'checks':[],'updates':[],
        'completed_ce_events':0,'optimizer_updates':0,'completed':False,'quality_passed':False,
        'exposures':{k:{} for k in ('new_ce','replay_ce','replay_kl')},
        'timing':{'optimization_seconds':0.,'check_seconds':0.,'save_seconds':0.},'publication':False}
    def persist():
        start=time.perf_counter()
        (output/'curve.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report['timing']['save_seconds']+=time.perf_counter()-start
    def check_guard(*,sources=True):
        if time.perf_counter()>=deadline:raise TimeoutError('Full stepcurve3600 deadline exceeded')
        if sources and any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Curve bound execution sources changed')
    try:
        for segment in schedule['segments']:
            check_guard()
            for batch in segment['batches']:
                check_guard(sources=False);_sync(parameter.device);start=time.perf_counter()
                actual=train_method_step(runtime,[rows[i] for i in batch['ce_ids']],[rows[i] for i in batch['replay_kl_ids']],
                    master,optimizer,reference,basis,method=plan['method'],ce_role=batch['ce_role'],caches=caches)
                _sync(parameter.device);wall=time.perf_counter()-start
                if (actual.get('runtime_restored') is not True or actual.get('optimizer_steps')!=1
                        or actual.get('ce_role')!=batch['ce_role'] or actual.get('ce_ids')!=batch['ce_ids']
                        or actual.get('replay_kl_ids')!=batch['replay_kl_ids']):
                    raise ValueError('Actual curve optimizer/roles/exposure/restoration changed')
                report['updates'].append({**actual,'schedule_batch':batch,'synchronized_update_wall_seconds':wall})
                report['optimizer_updates']+=1;report['completed_ce_events']+=len(batch['ce_ids'])
                report['timing']['optimization_seconds']+=wall;interval_opt+=wall
                for key,ids in ((batch['ce_role']+'_ce',batch['ce_ids']),('replay_kl',batch['replay_kl_ids'])):
                    for id in ids:report['exposures'][key][id]=report['exposures'][key].get(id,0)+1
            with torch.no_grad():parameter.copy_(master.to(parameter.dtype))
            check_guard();event=segment['check_ce_event'];start=time.perf_counter()
            saved=save_weights(parameter,output/('lastdown-ce'+str(event)+'.pt'))
            report['timing']['save_seconds']+=time.perf_counter()-start
            start=time.perf_counter();point=frozen_check(runtime,groups,baseline,old_rows,saved['tensor_sha256'],deadline,second=event==704)
            _sync(parameter.device);check_wall=time.perf_counter()-start
            state=optimizer.state[master]
            if int(float(state['step']))!=report['optimizer_updates']:raise ValueError('Actual Adam step differs from ledger')
            point.update(ce_events=event,optimizer_updates=report['optimizer_updates'],saved_weights=saved,
                fp32_master_sha256=tensor_sha(master),optimizer_state_sha256={k:tensor_sha(v.reshape(-1)) for k,v in state.items()},
                cumulative_delta_sha256=tensor_sha(master-reference),check_seconds=check_wall,
                interval_optimization_seconds=interval_opt,interval_wall_seconds=time.perf_counter()-interval_start)
            previous=report['checks'][-1] if report['checks'] else None
            point['interval_ce_events']=event-(previous['ce_events'] if previous else 0)
            point['interval_optimizer_updates']=report['optimizer_updates']-(previous['optimizer_updates'] if previous else 0)
            point['cumulative_exposure_totals']={k:sum(v.values()) for k,v in report['exposures'].items()}
            point['interval_exposure_totals']={k:v-(previous['cumulative_exposure_totals'][k] if previous else 0)
                                              for k,v in point['cumulative_exposure_totals'].items()}
            point['cumulative_per_id_exposures']=copy.deepcopy(report['exposures'])
            report['checks'].append(point);report['timing']['check_seconds']+=check_wall
            with torch.no_grad():parameter.copy_(original)
            check_guard();raw_save_started=time.perf_counter();persist()
            point['interval_raw_save_seconds']=time.perf_counter()-raw_save_started
            point['interval_wall_seconds']=time.perf_counter()-interval_start
            print(json.dumps({'ce_events':event,'optimizer_updates':report['optimizer_updates'],
                'provisional_correct':len(point['provisional_correct_ids']),'newly_confirmed':point['newly_confirmed_ids'],
                'forgotten':len(point['forgotten_ids']),'safety':point['safety_passed']}),flush=True)
            if point['safety_passed'] is not True:raise ValueError('Curve original knowledge/loss safety failed')
            interval_start=time.perf_counter();interval_opt=0.
        if report['optimizer_updates']!=schedule['optimizer_updates'] or any(
            {i:report['exposures'][k].get(i,0) for i in counts}!=counts for k,counts in schedule['per_id_exposures'].items()):
            raise ValueError('Complete704 CE/Adam/exposure ledger changed')
        selected=plan['confirm_ce_event']
        if selected is not None:
            check_guard();start=time.perf_counter()
            report['selected_saved_point']=confirm_saved_point(runtime,groups,baseline,old_rows,
                next(p for p in report['checks'] if p['ce_events']==selected),deadline)
            _sync(parameter.device);report['timing']['check_seconds']+=time.perf_counter()-start
            if report['selected_saved_point']['safety_passed'] is not True:raise ValueError('Savedpoint second-proof safety failed')
        check_guard();report['completed']=True
        report['quality_passed']=bool(report['checks'][-1]['newly_confirmed_ids'])
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:
        with torch.no_grad():parameter.copy_(original)
        _sync(parameter.device);report['original_tensor_restored']=tensor_sha(parameter)==tensor_sha(original)
        report['timing']['curve_wall_seconds']=time.perf_counter()-began
        master=optimizer=reference=state=None
        persist()
        if time.perf_counter()>=deadline or report['original_tensor_restored'] is not True:
            report['completed']=report['quality_passed']=False
            report['error']={'type':'TimeoutError' if time.perf_counter()>=deadline else 'ValueError','message':'Curve finalsave/restore deadline ororiginalW failed'}
            persist()
    return report


def execute(path,approved_sha):
    if not isinstance(approved_sha,str) or re.fullmatch('[0-9a-f]{64}',approved_sha) is None or sha(safe(path))!=approved_sha:
        raise ValueError('Exact root approved sealed curveSHA required')
    plan=json.loads(safe(path).read_text())
    if plan.get('protocol')!=PROTOCOL or plan.get('sealed') is not True or plan.get('state')!='sealed_waiting_explicit_MODEL_GO':
        raise ValueError('Separately sealed curve+explicit MODEL GO required')
    began=time.perf_counter();deadline=began+DEADLINE_SECONDS;validate_bound(plan)
    output=safe(plan['output'])
    if output.exists():raise FileExistsError('Fresh curve output required')
    write_new(safe(path).parent/'execution-started.json',{'protocol_sha256':approved_sha,'output':str(output),'started':time.time()})
    output.mkdir(mode=0o700)
    report={'protocol':PROTOCOL,'protocol_sha256':approved_sha,'pid':os.getpid(),'started':time.time(),
        'model_started':False,'passed':False,'teacher_calls':0,'publication':False,'source_before':plan['source_sha256']}
    write_new(output/'run-started.json',report)
    runtime=parameter=original=flags=modes=rng=mps_rng=basis=caches=native_basis=hooks=backward_hooks=pointer_before=None
    resources=None
    try:
        import torch
        from luban.runtime import Runtime
        from luban.memory import Memory
        from luban.settings import Settings
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target,_capture_replay_subspace,_training_replay_generation,_loss
        from run_identity_coverage import qualify_required_replay,qualified_groups,verify_training_prefix_alignment
        from prepare_identity_coverage import qualify_original_replay_pool
        from verify_cached_training import capture_tail,_sync,tensor_sha
        from verify_automatic_batches import measured
        with tempfile.TemporaryDirectory(prefix='luban-stepcurve-') as home:
            settings=Settings(home);settings.update({'storage_root':str(output/'private'),'model_path':plan['model_path'],'temperature':0,'local_thinking':'off'})
            runtime=Runtime(settings,Memory(output/'private'),research=True)
            if str(runtime.device)!=plan['device'] or str(runtime.dtype)!=plan['dtype']:raise ValueError('Curve native hardware/dtype changed')
            runtime._model_process_lock=ModelProcessLock(IDENTITY,runtime.device)
            with measured(runtime) as resources:
                phase=time.perf_counter();runtime.load(paging=False);report['model_started']=True
                report['load_seconds']=time.perf_counter()-phase
                parameter=_target(runtime)[1];original=parameter.detach().clone()
                if runtime.identity!=IDENTITY or runtime.active_core is not None or runtime.branch.expert is not None or tensor_sha(original)!=ORIGINAL:
                    raise ValueError('Exact originalNone1706 frozen graph required')
                flags=[(p,p.requires_grad,p.grad) for p in runtime.model.parameters()];modes=[(m,m.training) for m in runtime.model.modules()]
                hooks=[(m,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backward_hooks=[(p,len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                pointer=Path(runtime.store)/'active-core.json';pointer_before=pointer.read_bytes() if pointer.exists() else None
                rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone() if runtime.device=='mps' else None
                torch.manual_seed(plan['seed']);runtime.model.eval().requires_grad_(False);parameter.requires_grad_(True)
                phase=time.perf_counter();report['prefix']=verify_training_prefix_alignment(runtime,{**plan,'prefix_alignment_required':True})
                report['prefix_seconds']=time.perf_counter()-phase
                write_new(output/'qualification-started.json',{'protocol_sha256':approved_sha})
                phase=time.perf_counter();pool=qualify_original_replay_pool(runtime,plan);required=qualify_required_replay(runtime,plan)
                report['qualification']={'pool':pool,'required':required};report['qualification_seconds']=time.perf_counter()-phase
                if pool['qualified_count']!=11 or required['passed'] is not True:raise ValueError('Fixed11+2 qualification failed')
                groups=qualified_groups(plan,pool,required);tensor.verify_training_materials(groups,plan['b1_actual_reference'])
                old=_training_replay_generation(runtime,groups)
                if len(old)!=13:raise ValueError('Fixed13 mandatory old qualification changed')
                report['groups']=groups;report['groups_sha256']=tensor.content_sha(groups)
                phase=time.perf_counter();baseline={k:prior.probes(runtime,rows,deadline,ORIGINAL) for k,rows in
                    (('new',groups['new_training']),('replay',groups['replay_training']))}
                baseline['replay_loss']=_loss(runtime,groups['replay_training'])
                if not math.isfinite(baseline['replay_loss']) or baseline['replay_loss']<=0:raise ValueError('Original baseline native loss nonfinite')
                protected={'new':sorted(prior.correct(groups['new_training'],baseline['new'])),
                           'replay':sorted(prior.correct(groups['replay_training'],baseline['replay']))}
                if protected!=plan['protection_ids'] or not {r['id'] for r in old}.issubset(protected['replay']):
                    raise ValueError('Actual original13NEW/55REPLAY protection changed')
                old_rows=[r for r in groups['replay_training'] if r['id'] in protected['replay']]
                report['baseline']=baseline;report['baseline_seconds']=time.perf_counter()-phase
                phase=time.perf_counter();native_basis,basis_report=_capture_replay_subspace(runtime,parameter,groups['replay_training'])
                basis=native_basis.to(parameter.device);report['replay_svd']=basis_report;_sync(parameter.device)
                report['basis_seconds']=time.perf_counter()-phase;caches={};phase=time.perf_counter()
                if plan['method']=='cache':
                    for row in groups['new_training']+groups['replay_training']:
                        if time.perf_counter()>=deadline:raise TimeoutError('Curve cache setup exceeded entiredeadline')
                        caches[row['id']]=capture_tail(runtime,row)
                _sync(parameter.device);report['cache_build_seconds']=time.perf_counter()-phase
                report['cache_bytes']=sum(sum(c['bytes'].values()) for c in caches.values())
                report['common_seconds']=time.perf_counter()-began
                report['curve']=curve(runtime,groups,baseline,old_rows,original,basis,caches,plan,output,deadline)
                report['passed']=report['curve']['completed'] and report['curve']['quality_passed']
                report['resources']=resources
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        if isinstance(resources,dict):report['resources']=resources
        restore_started=time.perf_counter()
        if runtime is not None:
            try:
                import torch
                from verify_cached_training import tensor_sha
                if runtime.model is not None and original is not None:
                    with torch.no_grad():parameter.copy_(original)
                if flags is not None:
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
            except BaseException as error:report['restoration_error']={'type':type(error).__name__,'message':str(error)};report['passed']=False
            parameter=original=flags=modes=rng=mps_rng=basis=caches=native_basis=hooks=backward_hooks=pointer_before=None;p=m=None
            try:
                runtime._release_model()
                from luban.process_lock import ModelProcessLock
                lock=ModelProcessLock(IDENTITY,plan['device']);lock.close()
                report['model_released']=runtime.model is None and runtime._model_process_lock is None
                report['actual_os_lock_reacquired_closed']=True
            except BaseException as error:report['release_error']={'type':type(error).__name__,'message':str(error)};report['passed']=False
        report['restore_release_seconds']=time.perf_counter()-restore_started
        try:
            from verify_router_coverage_study import model_binding
            report['source_after']={p:sha(p) for p in plan['source_sha256']}
            report['production_after']=production_snapshot(plan['production_root'])
            report['official_after']=model_binding(plan['model_path'],IDENTITY)
            report['b1_reference_after_sha256']=sha(plan['b1_actual_reference']['path'])
            report['preserved']=report['source_after']==plan['source_sha256'] and report['production_after']==plan['production_snapshot'] and report['official_after']==plan['official_model_binding'] and report['b1_reference_after_sha256']==plan['b1_actual_reference']['sha256']
        except BaseException as error:report['preservation_error']={'type':type(error).__name__,'message':str(error)};report['preserved']=False
        report['passed']=report['passed'] and report.get('preserved') is True and len(report.get('restoration',{}))==9 and all(report['restoration'].values()) and report.get('model_released') is True and report.get('actual_os_lock_reacquired_closed') is True
        report['wall_seconds_before_final_result_write']=time.perf_counter()-began
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report['wall_seconds']=time.perf_counter()-began
        if time.perf_counter()>=deadline:
            report['passed']=False;report['error']={'type':'TimeoutError','message':'Complete curve restore/preservation/finalraw exceeded3600'}
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'curve_full_wall_after_final_result_write_seconds':report['wall_seconds'],
                         'passed':report['passed'],'publication':False}),flush=True)
    if not report['passed']:raise RuntimeError('Curve incomplete/rejected; keepraw/weights, no retry/publication')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for action in ('prepare','bind','seal','run'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--b1-result',type=Path)
    parser.add_argument('--method',choices=('native','cache'));parser.add_argument('--batch',type=int,choices=(1,2,4))
    parser.add_argument('--confirm-ce-event',type=int);parser.add_argument('--approved-sha')
    args=parser.parse_args()
    if not any(v is not None for v in vars(args).values()):
        print(json.dumps({'protocol':PROTOCOL,'model_started':False,'torch_imported':False,'requires_explicit_method_batch':True,'checks_ce_events':list(CHECKS)}));return
    if args.run:
        if not args.approved_sha or any(v is not None for v in (args.output,args.b1_result,args.method,args.batch,args.confirm_ce_event)):parser.error('run requires only sealedpath/exact approvedsha and separateGO')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if any(v is None for v in (args.output,args.b1_result,args.method,args.batch)) or args.approved_sha:parser.error('prepare requires explicit freshoutput/actualB1/method/batch')
        write_new(args.prepare,prepare(args.output,args.b1_result,args.method,args.batch,args.confirm_ce_event))
    elif args.bind or args.seal:
        if args.output is None or any(v is not None for v in (args.b1_result,args.method,args.batch,args.confirm_ce_event,args.approved_sha)):parser.error('bind/seal requires only input andNEW output')
        value=json.loads(safe(args.bind or args.seal).read_text())
        if args.bind:value=bind(value)
        else:validate_bound(value);value.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO')
        write_new(args.output,value)
    else:parser.error('No implicit model or operation')


if __name__=='__main__':main()
