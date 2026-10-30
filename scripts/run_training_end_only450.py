"""Independent original57 cacheB1 first upgrade450; only final frozen double proof.

Default SOURCE is stdlib only. No model until fresh bound/sealed exact Root GO.
Historical450 equality is diagnostic, never a substitute for actual quality.
"""
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

import run_training_step_curve as prior_curve
import verify_curve_saved_point as saved_point
import run_tensor_method_pilot as tensor
import run_training_method_pilot as prior
from prepare_tensor_method_schedule import build_schedule
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, safe, write_new, production_snapshot

PROTOCOL = 'original57-cache-b1-end-only450-tokenizer-eos-v1'
DEADLINE_SECONDS = 2400


def prefix_schedule(groups):
    whole = build_schedule([r['id'] for r in groups['new_training']],
                          [r['id'] for r in groups['replay_training']], 704, 1)
    segments = copy.deepcopy(whole['segments'][:9])
    batches = [batch for segment in segments for batch in segment['batches']]
    if (segments[-1]['check_ce_event'] != 450 or len(batches) != 450
            or [b['ce_event_ids'] for b in batches] != [[i] for i in range(1,451)]):
        raise ValueError('Only exact original704 B1 first450-event prefix')
    ledger = {key: {} for key in ('new_ce','replay_ce','replay_kl')}
    for batch in batches:
        for key, ids in ((batch['ce_role']+'_ce',batch['ce_ids']),('replay_kl',batch['replay_kl_ids'])):
            for id in ids: ledger[key][id] = ledger[key].get(id,0)+1
    return {'original704_schedule_sha256':tensor.content_sha(whole), 'segments':segments,
            'ce_events':450,'optimizer_updates':450,'per_id_exposures':ledger,
            'exposure_totals':{k:sum(v.values()) for k,v in ledger.items()}}


def prepare(output, receipt_path):
    data = saved_point.closed_receipt(receipt_path, 450)
    old = data['curve_plan']; groups = data['groups']; point = data['point']
    if old['method'] != 'cache' or old['tensor_batch'] != 1 or old['seed'] != 42:
        raise ValueError('Closed original704 cacheB1/seed42 reference required')
    plan = prior_curve.prepare(output, old['b1_actual_reference']['path'], 'cache', 1)
    schedule = prefix_schedule(groups)
    if schedule['original704_schedule_sha256'] != old['schedule_sha256']:
        raise ValueError('Original bound704 schedule prefix differs')
    sources = dict(plan['source_sha256'])
    for name in ('scripts/verify_curve_saved_point.py','tests/test_curve_saved_point.py',
                 'scripts/run_training_end_only450.py','tests/test_training_end_only450.py'):
        sources[str(ROOT/name)] = sha(ROOT/name)
    if safe(output).exists(): raise FileExistsError('Fresh end-only output required')
    plan.update(protocol=PROTOCOL, source_sha256=dict(sorted(sources.items())), ce_events=450,
        actual_optimizer_updates=450, checkpoints=[450], confirm_ce_event=None,
        schedule_sha256=tensor.content_sha(schedule), original704_schedule_sha256=old['schedule_sha256'],
        exposure_ledger={'new_ce':225,'replay_ce':225,'replay_kl':450}, deadline_seconds=DEADLINE_SECONDS,
        curve_receipt_path=str(safe(receipt_path)), curve_receipt_sha256=sha(safe(receipt_path)),
        historical450={k:copy.deepcopy(point[k]) for k in ('candidate_tensor_sha256','fp32_master_sha256',
            'optimizer_state_sha256','cumulative_delta_sha256','cumulative_per_id_exposures')},
        generation_upper_bound=297, generation_cost='14qualification+114freshbaseline+169final:57NEWtwice+55oldREPLAY; full57native loss, no intermediate generation',
        checkpoint_weights='One complete native BF16 W and one CPU full-state master/Adam/RNG/ledger/source fsync checkpoint. Basis/reference hashes only; future reconstruction not implemented.',
        mastery='Actual end-only450 sameW twoNEW finite intersection minus originalfinite12, all original13NEW/55REPLAY plus nativeREPLAY loss protection. No inherited27.',
        stop='Only per-step finite/input/cancel/source/deadline guards; final safety failure rejects/restores, no intermediate generation/quality earlystop.',
        timing='One cold original research process2400 fullwall incl preflight/load/freshbaseline/SVD/cache/450updates/finalproof/checkpoint/restore/lastsave; report actual stdout after lastwrite separately.',
        scope='First-upgrade end-only validation policy, not product adoption or second upgrade. Historical native/master/Adam/delta equality diagnostic only.')
    return plan


def bind(plan):
    from verify_router_coverage_study import model_binding
    if prepare(plan['output'],plan['curve_receipt_path']) != plan:
        raise ValueError('End-only source/material/closedreceipt changed')
    return {**copy.deepcopy(plan),'official_model_binding':model_binding(plan['model_path'],IDENTITY),
            'production_snapshot':production_snapshot(plan['production_root']), 'state':'bound_waiting_root_seal_MODEL_GO'}


def validate_bound(plan):
    template={**copy.deepcopy(plan),'state':'unbound_waiting_root_review','sealed':False,
              'official_model_binding':None,'production_snapshot':None}
    if bind(template) != {**plan,'sealed':False,'state':'bound_waiting_root_seal_MODEL_GO'}:
        raise ValueError('End-only fixed source/model/production/receipt binding changed')


def save_training_state(parameter,master,optimizer,reference,basis,ledger,saved,plan,protocol_sha,path):
    import torch
    from verify_cached_training import tensor_sha
    state = optimizer.state_dict()
    cpu_state = {'state':{key:{k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v)
                             for k,v in value.items()} for key,value in state['state'].items()},
                 'param_groups':copy.deepcopy(state['param_groups'])}
    payload={'protocol':PROTOCOL,'protocol_sha256':protocol_sha,'ce_events':450,'optimizer_updates':450,
        'candidate':parameter.detach().cpu().clone(),'master':master.detach().cpu().clone(),
        'optimizer_state_dict':cpu_state,'cpu_rng':torch.get_rng_state().clone(),
        'mps_rng':torch.mps.get_rng_state().clone() if parameter.device.type=='mps' else None,
        'exposures':copy.deepcopy(ledger),'exposure_totals':{k:sum(v.values()) for k,v in ledger.items()},
        'native_checkpoint':copy.deepcopy(saved),'source_sha256':plan['source_sha256'],
        'identity':IDENTITY,'original_tensor_sha256':ORIGINAL,'reference_sha256':tensor_sha(reference),
        'basis_sha256':tensor_sha(basis),'basis_reference_reconstruction':'Future continuation protocol must define reconstruction; tensors not saved here. No resume/publication.',
        'seed':42,'method':'cache','tensor_batch':1,'publication':False,'teacher_calls':0}
    before=tensor_sha(master)
    with safe(path).open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    if tensor_sha(master)!=before or tensor_sha(payload['master'])!=before or tensor_sha(payload['candidate'])!=saved['tensor_sha256']:
        raise ValueError('Full checkpoint transfer changed master/nativeW')
    return {'path':str(path),'file_sha256':sha(path),'file_bytes':path.stat().st_size,
            'master_sha256':before,'candidate_tensor_sha256':saved['tensor_sha256'],
            'contains':['nativeBF16','FP32master','fullAdam','CPU/MPSrng','perIDledger','sourceBinding'],
            'basis_reference_tensors_saved':False,'continuation_implemented':False}


def end_only(runtime,groups,baseline,old_rows,original,basis,caches,plan,output,deadline,protocol_sha):
    import torch
    from cached_method_steps import train_method_step
    from luban.evolution import _target
    from verify_cached_training import _sync,tensor_sha
    schedule=prefix_schedule(groups)
    if tensor.content_sha(schedule)!=plan['schedule_sha256']:
        raise ValueError('Exact450 prefix schedule binding changed')
    rows={r['id']:r for r in groups['new_training']+groups['replay_training']}
    parameter=_target(runtime)[1]
    if tensor_sha(parameter)!=tensor_sha(original):raise ValueError('Fresh originalW required')
    master=torch.nn.Parameter(original.float().clone());reference=original.float()
    optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
    began=time.perf_counter()
    report={'method':'cache','tensor_batch':1,'ce_events':0,'optimizer_updates':0,'updates':[],
        'exposures':{k:{} for k in ('new_ce','replay_ce','replay_kl')},'checks':[],
        'completed':False,'quality_passed':False,'publication':False,
        'timing':{'optimization_seconds':0.,'check_seconds':0.,'checkpoint_seconds':0.}}
    def guard(sources=True):
        if time.perf_counter()>=deadline:raise TimeoutError('End-only2400 fullwall exceeded')
        if sources and any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('Bound execution source changed')
    def persist():
        (output/'training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        for segment in schedule['segments']:
            guard()
            for batch in segment['batches']:
                guard(False);_sync(parameter.device);start=time.perf_counter()
                actual=train_method_step(runtime,[rows[i] for i in batch['ce_ids']],[rows[i] for i in batch['replay_kl_ids']],
                    master,optimizer,reference,basis,method='cache',ce_role=batch['ce_role'],caches=caches)
                _sync(parameter.device);wall=time.perf_counter()-start
                if (actual.get('runtime_restored') is not True or actual.get('optimizer_steps')!=1
                        or actual.get('ce_role')!=batch['ce_role'] or actual.get('ce_ids')!=batch['ce_ids']
                        or actual.get('replay_kl_ids')!=batch['replay_kl_ids']):raise ValueError('Actual update/exposure/restoration differs')
                report['updates'].append({**actual,'schedule_batch':batch,'synchronized_update_wall_seconds':wall})
                report['ce_events']+=1;report['optimizer_updates']+=1;report['timing']['optimization_seconds']+=wall
                for key,ids in ((batch['ce_role']+'_ce',batch['ce_ids']),('replay_kl',batch['replay_kl_ids'])):
                    for id in ids:report['exposures'][key][id]=report['exposures'][key].get(id,0)+1
            persist()  # Durable exposure prefix; no generation, checkpoint or candidate selection.
        guard()
        if report['ce_events']!=450 or report['optimizer_updates']!=450 or report['exposures']!=schedule['per_id_exposures']:
            raise ValueError('Complete450 three-account exposure ledger differs')
        state=optimizer.state[master]
        if int(float(state['step']))!=450:raise ValueError('Actual Adam step differs')
        with torch.no_grad():parameter.copy_(master.to(parameter.dtype))
        start=time.perf_counter();saved=prior_curve.save_weights(parameter,output/'lastdown-ce450.pt')
        summary={'candidate_tensor_sha256':saved['tensor_sha256'],'fp32_master_sha256':tensor_sha(master),
            'optimizer_state_sha256':{k:tensor_sha(v.reshape(-1)) for k,v in state.items()},
            'cumulative_delta_sha256':tensor_sha(master-reference),'cumulative_per_id_exposures':copy.deepcopy(report['exposures'])}
        report['actual450_state_summary']=summary
        report['historical450_comparison']={k:summary[k]==plan['historical450'][k] for k in summary}
        report['historical450_comparison_is_quality_gate']=False
        report['saved_weights']=saved
        report['full_training_state']=save_training_state(parameter,master,optimizer,reference,basis,report['exposures'],saved,plan,protocol_sha,output/'training-state-ce450.pt')
        _sync(parameter.device);report['timing']['checkpoint_seconds']+=time.perf_counter()-start;guard()
        start=time.perf_counter();point=prior_curve.frozen_check(runtime,groups,baseline,old_rows,saved['tensor_sha256'],deadline,second=True)
        _sync(parameter.device);report['timing']['check_seconds']+=time.perf_counter()-start
        point.update(ce_events=450,optimizer_updates=450,saved_weights=saved)
        report['checks'].append(point);guard()
        report['completed']=point['safety_passed'] is True
        report['quality_passed']=report['completed'] and bool(point['newly_confirmed_ids'])
        if not report['completed']:report['error']={'type':'ValueError','message':'Actual final old-protection/loss failed; no adoption'}
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:
        with torch.no_grad():parameter.copy_(original)
        _sync(parameter.device);report['original_tensor_restored']=tensor_sha(parameter)==tensor_sha(original)
        report['timing']['training_wall_seconds']=time.perf_counter()-began
        master=optimizer=reference=state=None
        persist()
        if time.perf_counter()>=deadline or report['original_tensor_restored'] is not True:
            report['completed']=report['quality_passed']=False
            report['error']={'type':'TimeoutError' if time.perf_counter()>=deadline else 'ValueError','message':'Final raw save/restore failed or exceeded2400'}
            persist()
    return report


def execute(path,approved_sha):
    if not isinstance(approved_sha,str) or re.fullmatch('[0-9a-f]{64}',approved_sha) is None or sha(safe(path))!=approved_sha:
        raise ValueError('Exact root approved sealed end-onlySHA required')
    plan=json.loads(safe(path).read_text())
    if plan.get('protocol')!=PROTOCOL or plan.get('sealed') is not True or plan.get('state')!='sealed_waiting_explicit_MODEL_GO':
        raise ValueError('Separately sealed end-only+explicit MODEL GO required')
    began=time.perf_counter();deadline=began+DEADLINE_SECONDS;validate_bound(plan)
    if time.perf_counter()>=deadline:raise TimeoutError('Static preflight exceeded2400 before execution consumption')
    output=safe(plan['output'])
    if output.exists():raise FileExistsError('Fresh end-only output required')
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
        with tempfile.TemporaryDirectory(prefix='luban-endonly450-') as home:
            settings=Settings(home);settings.update({'storage_root':str(output/'private'),'model_path':plan['model_path'],'temperature':0,'local_thinking':'off'})
            runtime=Runtime(settings,Memory(output/'private'),research=True)
            if str(runtime.device)!=plan['device'] or str(runtime.dtype)!=plan['dtype']:raise ValueError('End-only native hardware/dtype changed')
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
                report['training']=end_only(runtime,groups,baseline,old_rows,original,basis,caches,plan,output,deadline,approved_sha)
                report['passed']=report['training']['completed'] and report['training']['quality_passed']
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
            report['curve_receipt_after_sha256']=sha(plan['curve_receipt_path'])
            report['preserved']=report['source_after']==plan['source_sha256'] and report['production_after']==plan['production_snapshot'] and report['official_after']==plan['official_model_binding'] and report['b1_reference_after_sha256']==plan['b1_actual_reference']['sha256'] and report['curve_receipt_after_sha256']==plan['curve_receipt_sha256']
        except BaseException as error:report['preservation_error']={'type':type(error).__name__,'message':str(error)};report['preserved']=False
        report['passed']=report['passed'] and report.get('preserved') is True and len(report.get('restoration',{}))==9 and all(report['restoration'].values()) and report.get('model_released') is True and report.get('actual_os_lock_reacquired_closed') is True
        report['wall_seconds_before_final_result_write']=time.perf_counter()-began
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report['wall_seconds_after_first_final_result_write']=time.perf_counter()-began
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report['wall_seconds']=time.perf_counter()-began
        if time.perf_counter()>=deadline:
            report['passed']=False;report['error']={'type':'TimeoutError','message':'Complete end-only restore/preservation/finalraw exceeded2400'}
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'end_only_full_wall_after_final_result_write_seconds':report['wall_seconds'],
                         'passed':report['passed'],'publication':False}),flush=True)
    if not report['passed']:raise RuntimeError('End-only incomplete/rejected; keepraw/weights, no retry/publication')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for action in ('prepare','bind','seal','run'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--curve-receipt',type=Path);parser.add_argument('--approved-sha')
    args=parser.parse_args()
    if not any(v is not None for v in vars(args).values()):
        print(json.dumps({'protocol':PROTOCOL,'model_started':False,'torch_imported':False,'ce_events':450,
            'optimizer_updates':450,'method':'cache','tensor_batch':1,'checks':[450],'fullwall_cap_seconds':2400,
            'implementation_scope':'SOURCE only; first upgrade, no second upgrade/publication'}));return
    if args.run:
        if not args.approved_sha or args.output or args.curve_receipt:parser.error('run only exact sealedpath/RootapprovedSHA after independent GO')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if args.output is None or args.curve_receipt is None or args.approved_sha:parser.error('prepare requires freshoutput/closed actualcurve receipt')
        write_new(args.prepare,prepare(args.output,args.curve_receipt))
    elif args.bind or args.seal:
        if args.output is None or args.curve_receipt or args.approved_sha:parser.error('bind/seal only input and freshoutput')
        value=saved_point.read(args.bind or args.seal)
        if args.bind:value=bind(value)
        else:validate_bound(value);value.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO')
        write_new(args.output,value)
    else:parser.error('No implicit operation/model')


if __name__=='__main__':main()
