"""Independent closed first-upgrade450 -> cumulative900 research continuation.

SOURCE/default is stdlib-only. Parent selection, CPU proof and MODEL GO remain
separate Root actions. This is a new replay constraint phase, not old-basis resume.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import os
import re
import tempfile
import time

import run_training_subset450 as first
import run_training_end_only450 as end
import run_training_step_curve as curve
import run_tensor_method_pilot as tensor
import run_training_method_pilot as prior
import verify_curve_saved_point as saved_point
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, safe, write_new, production_snapshot

PROTOCOL = 'closed-original57-cumulative-second-upgrade450-tokenizer-eos-v1'
WALL_SECONDS = 2400
ACCOUNTS = ('new_ce', 'replay_ce', 'replay_kl')


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(safe(path).read_text())


def closed_parent(descriptor_path, backup_path, readback_path):
    """Immutable closed evidence only; never a live run, weak receipt or new parent."""
    descriptor, backup, readback = (read(p) for p in (descriptor_path, backup_path, readback_path))
    require(descriptor.get('process_exit_code') == 0 and type(descriptor['process_exit_code']) is int
        and all(descriptor.get(k) is True for k in ('pid_dead','evidence_preserved','execution_passed',
            'quality_passed','positive_confirmed_growth','full_state_payload_verified')),
        'Closed eligible parent actual exit/quality/full-state proof required')
    values = {}
    for key in ('formal','raw','root_post'):
        ref = descriptor[key]
        require(sha(safe(ref['path'])) == ref['sha256'], 'Parent '+key+' immutable SHA differs')
        values[key] = read(ref['path'])
    plan, raw, post = (values[k] for k in ('formal','raw','root_post'))
    require(plan.get('protocol') in (end.PROTOCOL, first.PROTOCOL) and plan.get('sealed') is True
        and raw.get('protocol') == plan['protocol'] and raw.get('protocol_sha256') == descriptor['formal']['sha256']
        and post.get('scope') == 'actual_root_post' and post.get('result_sha256') == descriptor['raw']['sha256']
        and post.get('protocol_sha256') == descriptor['formal']['sha256']
        and post.get('process_exit_code') == 0 and post.get('pid_dead') is True
        and post.get('evidence_preserved') is True and post.get('full_state_payload_verified') is True
        and bool(post.get('checks')) and all(v is True for v in post['checks'].values()),
        'Closed parent formal/raw/actual Root full-state post disagree')
    require(raw.get('passed') is True and raw.get('preserved') is True
        and raw.get('model_released') is True and raw.get('actual_os_lock_reacquired_closed') is True
        and raw.get('publication') is False and type(raw.get('teacher_calls')) is int and raw['teacher_calls'] == 0
        and set(raw.get('restoration',{})) == saved_point.RESTORATION
        and all(v is True for v in raw['restoration'].values()), 'Parent complete recovery/closed state missing')
    require(backup.get('result_sha256') == descriptor['raw']['sha256']
        and backup.get('protocol_sha256') == descriptor['formal']['sha256']
        and backup.get('root_post_sha256') == descriptor['root_post']['sha256']
        and backup.get('both_complete_second_readback_passed') is True
        and readback.get('passed') is True and readback.get('model_pid_dead') is True
        and readback.get('index') == {'path':str(safe(backup_path)), 'sha256':sha(safe(backup_path))}
        and readback.get('both_manifest_sha256') == backup.get('manifest_sha256')
        and readback.get('actual_full_file_reads') == 2*backup.get('payload_files',0)
        and readback.get('new_full_PT_count_each_volume') == 2,
        'Actual complete dual archive and Root byte-readback receipt required')
    groups, baseline, training = raw['groups'], raw['baseline'], raw['training']
    require(len(groups['new_training']) == len(groups['replay_training']) == 57
        and tensor.content_sha(groups) == raw['groups_sha256'] and training.get('completed') is True
        and training.get('quality_passed') is True and training.get('ce_events') == training.get('optimizer_updates') == 450
        and len(training.get('checks',[])) == 1, 'Exact first-upgrade114/full450 required')
    tensor.verify_training_materials(groups, plan['b1_actual_reference'])
    sources = plan['source_sha256']
    require(len(sources) == (65 if plan['protocol'] == end.PROTOCOL else 67)
        and sources == raw['source_before'] == raw['source_after']
        and all(sha(p) == h for p,h in sources.items()), 'Closed parent execution sources changed')
    point = training['checks'][0]; candidate = training['saved_weights']['tensor_sha256']
    require(point.get('safety_passed') is True and point.get('mastery_confirmed') is True
        and point.get('candidate_tensor_sha256') == candidate and point.get('forgotten_ids') == [],
        'Actual parent final frozen two-proof safety required')
    # Recorded completion IDs/EOS are verified by the immutable Root post. These
    # independent row/binding/finite-rule checks do not invent a new generation.
    for name in ('new_first','new_second'):
        prior.correct(groups['new_training'], point[name])
        for row, obs in zip(groups['new_training'], point[name]):
            require(obs.get('sample_sha256') == tensor.content_sha(row)
                and obs.get('actual_tensor_sha256') == candidate and obs.get('actual_base_identity') == IDENTITY,
                'Parent actual row/core binding differs')
    newly = (prior.correct(groups['new_training'],point['new_first'],finite=True)
        & prior.correct(groups['new_training'],point['new_second'],finite=True)) - prior.correct(groups['new_training'],baseline['new'],finite=True)
    require(sorted(newly) == point['newly_confirmed_ids'] == descriptor['confirmed_new_ids']
        and bool(newly), 'Parent finite two-proof growth differs')
    if plan['protocol'] == first.PROTOCOL:
        require(first.final_gain(point,groups,baseline,plan['selected_new_pool_n']) == training['gain']
            and training['gain']['selected_taught_finite_gain'] > 0, 'Parent positive taught gain required')
    else: require(descriptor.get('confirmed_new_count') == len(newly), 'Parent confirmed count differs')
    require(training['exposures'] == post.get('actual_per_ID_ledger')
        and all(sum(training['exposures'][k].values()) == v for k,v in zip(ACCOUNTS,(225,225,450))),
        'Parent actual per-ID three-account post differs')
    checkpoint_refs = (training['saved_weights'], training['full_training_state'])
    for record in checkpoint_refs:
        path = safe(record['path']); expected = record['file_sha256']
        require(sha(path) == expected and path.stat().st_size == record['file_bytes'], 'Parent actual full PT changed')
        copies = backup['new_checkpoints'][str(path)]
        require(copies['sha256'] == expected and copies['bytes'] == record['file_bytes'], 'Parent checkpoint receipt changed')
        for root, key in ((Path('/Volumes/移动数据库/lubanllm'),'primary'),(Path('/Volumes/Untitled/lubanllm-backup'),'backup')):
            target = safe(copies[key])
            require(root in target.parents and sha(target) == expected and target.stat().st_size == record['file_bytes'],
                'Parent complete native/master/Adam/RNG dual PT bytes differ')
    finite_f1 = prior.correct(groups['new_training'],point['new_first'],finite=True) & prior.correct(groups['new_training'],point['new_second'],finite=True)
    require(prior.correct(groups['new_training'],baseline['new']).issubset(
        prior.correct(groups['new_training'],point['new_first']) & prior.correct(groups['new_training'],point['new_second'])),
        'Actual parent originalknown protection disappeared')
    return {'descriptor':descriptor,'plan':plan,'raw':raw,'post':post,'groups':groups,'baseline':baseline,
            'point':point,'training':training,'confirmed_ids':sorted(newly),'finite_f1_ids':sorted(finite_f1),'candidate_sha256':candidate}


def curriculum(parent):
    groups=parent['groups']; news=groups['new_training']; finite_f1=set(parent['finite_f1_ids'])
    known=prior.correct(news,parent['baseline']['new'])
    protected=known|finite_f1
    pending=[r for r in news if r['id'] not in protected]
    parent_n=parent['plan'].get('selected_new_pool_n',57)
    require(type(parent_n) is int and parent_n in (16,32,57) and pending, 'Bound parentN and pending required')
    selected=pending[:parent_n];expanded=copy.deepcopy(groups['replay_training'])
    for row in news:
        if row['id'] not in protected:continue
        replay=copy.deepcopy(row); provenance=json.loads(replay['provenance'])
        provenance.update(evolution_role='replay', continuation_source_sha256=tensor.content_sha(row),
            continuation_source_id=row['id'], continuation_parent_tensor_sha256=parent['candidate_sha256'],
            continuation_proof=('Parent currentW finite two-complete-closedbook EOS' if row['id'] in finite_f1 else 'Originalknown weak-rule obligation; not finite mastery'))
        replay['provenance']=json.dumps(provenance,ensure_ascii=False,sort_keys=True)
        expanded.append(replay)
    require(len({r['id'] for r in expanded}) == len(expanded), 'Expanded replay IDs collide')
    from luban.generation_checks import is_finite_generation_check
    return {'pending_ids':[r['id'] for r in pending], 'selected_ids':[r['id'] for r in selected],
        'finite_pending_ids':[r['id'] for r in selected if is_finite_generation_check(json.loads(r['provenance'])['generation_check'])],
        'parent_n_cap':parent_n,'selected_count':len(selected),'replay_training':expanded,
        'expanded_replay_sha256':tensor.content_sha(expanded),'first_finite_f1_ids':sorted(finite_f1),
        'parent_newly_confirmed_ids':parent['confirmed_ids'],
        'original_known_ids':sorted(known),'original_known_nonfinite_ids':sorted(known-finite_f1),'protection_ids':sorted(protected),
        'contains_pending_scope':'Weak contains-rule passes remain pending unless originalknown; never counted finite mastery.'}


def continuation_schedule(parent, selected):
    news=selected['selected_ids'];replays=[r['id'] for r in selected['replay_training']]
    require(news and replays and len(set(news)|set(replays)) == len(news)+len(replays), 'Disjoint pending NEW/expanded replay required')
    batches=[];ledger={k:{} for k in ACCOUNTS}
    for i in range(450):
        role='new' if i%2==0 else 'replay';pair=i//2;replay=replays[pair%len(replays)]
        ce=news[pair%len(news)] if role=='new' else replay
        batches.append({'ce_role':role,'ce_ids':[ce],'replay_kl_ids':[replay],'round_event':i+1,'cumulative_optimizer_step':451+i})
        for key,id in ((role+'_ce',ce),('replay_kl',replay)):ledger[key][id]=ledger[key].get(id,0)+1
    cumulative=copy.deepcopy(parent['training']['exposures'])
    for key in ACCOUNTS:
        for id,count in ledger[key].items():cumulative[key][id]=cumulative[key].get(id,0)+count
    return {'batches':batches,'round_exposures':ledger,'cumulative_exposures':cumulative,
        'round_totals':{k:sum(v.values()) for k,v in ledger.items()},
        'cumulative_totals':{k:sum(v.values()) for k,v in cumulative.items()}}


def prepare(output,descriptor_path,backup_path,readback_path):
    cpu_path=ROOT/'research/training_continuation450_cpu.json'
    cpu=read(cpu_path)
    expected={str(ROOT/name):sha(ROOT/name) for name in ('scripts/run_training_continuation450.py','tests/test_training_continuation450.py')}
    require(cpu.get('run') is True and cpu.get('passed') is True and type(cpu.get('exit_code')) is int
        and cpu['exit_code']==0 and cpu.get('source_sha256')==expected
        and sha(safe(cpu['log']))==cpu.get('log_sha256'), 'Actual matching continuation SOURCE CPU proof required before parent access/preparation')
    parent=closed_parent(descriptor_path,backup_path,readback_path);selected=curriculum(parent)
    schedule=continuation_schedule(parent,selected);old=parent['plan'];output=safe(output)
    product=safe(old['production_root']);experiments=product/'experiments'
    require(experiments in output.parents and len(output.relative_to(experiments).parts)>=2,
        'Fresh continuation experiment child under production/experiments required')
    require(output!=product,'Production root cannot be a continuation output')
    for forbidden in (ROOT,product/'app',product/'modules',safe(old['model_path']),safe(old['output'])):
        require(output!=forbidden and forbidden not in output.parents,'Fresh private continuation output required')
    require(not output.exists(),'Continuation output already consumed')
    sources=dict(old['source_sha256'])
    for name in ('scripts/run_training_subset450.py','tests/test_training_subset450.py',
                 'scripts/run_training_continuation450.py','tests/test_training_continuation450.py'):
        sources[str(ROOT/name)]=sha(ROOT/name)
    require(len(sources)==69,'Exact inherited67 plus continuation2 source closure required')
    return {'protocol':PROTOCOL,'state':'unbound_waiting_Root_review','sealed':False,'model_go':False,
        'output':str(output),'parent_descriptor_path':str(safe(descriptor_path)),'parent_descriptor_sha256':sha(safe(descriptor_path)),
        'parent_backup_path':str(safe(backup_path)),'parent_backup_sha256':sha(safe(backup_path)),
        'parent_readback_path':str(safe(readback_path)),'parent_readback_sha256':sha(safe(readback_path)),
        'parent_protocol_sha256':parent['descriptor']['formal']['sha256'],'parent_candidate_tensor_sha256':parent['candidate_sha256'],
        'parent_full_state':parent['training']['full_training_state'],'curriculum':selected,
        'schedule_sha256':tensor.content_sha(schedule),'groups_sha256':tensor.content_sha(parent['groups']),
        'method':'cache','tensor_batch':1,'gradient_accumulation':1,'round_optimizer_updates':450,
        'entry_optimizer_step':450,'final_optimizer_step':900,'round_exposure_totals':schedule['round_totals'],
        'cumulative_exposure_totals':schedule['cumulative_totals'],'deadline_seconds':WALL_SECONDS,
        'identity':IDENTITY,'original_tensor_sha256':ORIGINAL,'device':'mps','dtype':'torch.bfloat16',
        'model_path':old['model_path'],'production_root':old['production_root'],'paging':False,'research_runtime':True,
        'source_sha256':dict(sorted(sources.items())),'official_model_binding':None,'production_snapshot':None,
        'source_cpu_receipt_path':str(cpu_path),'source_cpu_receipt_sha256':sha(cpu_path),
        'learning_rate':1e-5,'eps':1e-6,'distillation_weight':5,'clip':1,'field_first_token_weight':1,
        'reference':'FP32parentmaster.detach.clone; BF16cast must equal parentnative; never official originalW/candidate.float',
        'rng':'Restore full parent savedCPU/MPS RNG after entryloss/SVD/cache preparation immediately before update451',
        'constraints':'New expanded replay subspace built at parent entryW; not same-basis old trajectory equivalence',
        'entry_generation_calls':len(selected['protection_ids'])+55,'final_generation_calls':169,
        'generation_calls':len(selected['protection_ids'])+55+169,
        'checks':'Entry originalknown13/allfiniteF1 NEW once+original55REPLAY once; final full57NEW sameW twice+original55REPLAY once',
        'quality':'Actual selected finite doubleproof growth vs parent finitecurrent; all old protections; expanded replay native meanloss<=entry1.05',
        'single_attempt':True,'maximum_attempts':1,'teacher_calls':0,'publication':False,'product_accepted':False}


def bind(plan):
    rebuilt=prepare(plan['output'],plan['parent_descriptor_path'],plan['parent_backup_path'],plan['parent_readback_path'])
    require(rebuilt==plan,'Continuation fixed parent/material/source arguments changed')
    parent=closed_parent(plan['parent_descriptor_path'],plan['parent_backup_path'],plan['parent_readback_path'])
    from verify_router_coverage_study import model_binding
    official=model_binding(plan['model_path'],IDENTITY);production=production_snapshot(plan['production_root'])
    require(official==parent['plan']['official_model_binding']==parent['raw']['official_after']
        and production==parent['plan']['production_snapshot']==parent['raw']['production_after'],
        'Continuation parent originalofficial/current product binding changed')
    return {**copy.deepcopy(plan),'official_model_binding':official,'production_snapshot':production,
            'state':'bound_waiting_Root_seal_MODEL_GO'}


def validate_bound(plan):
    template={**copy.deepcopy(plan),'state':'unbound_waiting_Root_review','sealed':False,
              'official_model_binding':None,'production_snapshot':None}
    require(bind(template)=={**plan,'sealed':False,'state':'bound_waiting_Root_seal_MODEL_GO'},'Bound continuation parent changed')


def load_parent_state(runtime,parent):
    """Only MODEL_GO execution/CPU fixture loads tensors, never SOURCE/prepare."""
    import torch
    from luban.evolution import _target
    from verify_cached_training import tensor_sha
    record=parent['training']['full_training_state'];path=safe(record['path'])
    require(sha(path)==record['file_sha256'],'Parent full-state bytes changed before load')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(sha(path)==record['file_sha256'] and payload.get('protocol')==parent['plan']['protocol']
        and payload.get('protocol_sha256')==parent['descriptor']['formal']['sha256']
        and payload.get('source_sha256')==parent['plan']['source_sha256']
        and payload.get('ce_events')==payload.get('optimizer_updates')==450
        and payload.get('identity')==IDENTITY and payload.get('original_tensor_sha256')==ORIGINAL
        and payload.get('exposures')==parent['training']['exposures']
        and payload.get('native_checkpoint')==parent['training']['saved_weights'], 'Parent tensor-state metadata changed')
    parameter=_target(runtime)[1];candidate=payload.get('candidate');master_cpu=payload.get('master')
    require(isinstance(candidate,torch.Tensor) and isinstance(master_cpu,torch.Tensor)
        and candidate.dtype==torch.bfloat16 and master_cpu.dtype==torch.float32
        and candidate.shape==master_cpu.shape==parameter.shape
        and bool(torch.isfinite(candidate).all()) and bool(torch.isfinite(master_cpu).all())
        and torch.equal(master_cpu.to(candidate.dtype),candidate)
        and tensor_sha(candidate)==parent['candidate_sha256']
        and tensor_sha(master_cpu)==record['master_sha256'], 'Exact BF16candidate/FP32master residual binding required')
    require(isinstance(payload.get('cpu_rng'),torch.Tensor) and payload['cpu_rng'].dtype==torch.uint8
        and payload['cpu_rng'].ndim==1 and payload['cpu_rng'].numel()>0,'Parent CPU RNG bytes missing')
    try:torch.Generator(device='cpu').set_state(payload['cpu_rng'])
    except Exception as error:raise ValueError('Parent CPU RNG state is invalid') from error
    if parameter.device.type=='mps':
        require(isinstance(payload.get('mps_rng'),torch.Tensor) and payload['mps_rng'].dtype==torch.uint8
            and payload['mps_rng'].ndim==1 and payload['mps_rng'].numel()>0,'Parent MPS RNG bytes missing')
    master=torch.nn.Parameter(master_cpu.to(parameter.device).clone());reference=master.detach().clone()
    optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
    state_dict=payload.get('optimizer_state_dict')
    require(isinstance(state_dict,dict) and set(state_dict)=={'state','param_groups'}
        and len(state_dict['state'])==len(state_dict['param_groups'])==1,'Complete single-parent Adam required')
    optimizer.load_state_dict(state_dict)
    from cached_method_steps import _adam_state
    require(_adam_state(optimizer,master)==450,'Parent actual Adam step must450, not fresh optimizer')
    group=optimizer.param_groups[0]
    require(group['params']==[master] and all(group[k]==v for k,v in
        (('lr',1e-5),('eps',1e-6),('weight_decay',0),('betas',(.9,.999)),('amsgrad',False))), 'Parent Adam settings/ownership differ')
    require({k:tensor_sha(v.reshape(-1)) for k,v in optimizer.state[master].items()}
        ==parent['training']['actual450_state_summary']['optimizer_state_sha256'], 'Parent Adam moments/step content differs')
    with torch.no_grad():parameter.copy_(candidate)
    require(tensor_sha(parameter)==parent['candidate_sha256'],'Actual parent nativeW import differs')
    return master,optimizer,reference,payload['cpu_rng'].clone(),payload.get('mps_rng')


def restore_parent_rng(cpu_rng,mps_rng):
    import torch
    torch.set_rng_state(cpu_rng)
    if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
    restored=torch.equal(torch.get_rng_state(),cpu_rng) and (mps_rng is None or torch.equal(torch.mps.get_rng_state(),mps_rng))
    require(restored,'Actual saved parent CPU/MPS RNG restoration failed before update451')
    return restored


def entry_protection(runtime,parent,selected,deadline,report=None):
    """One real entry probe per protected context; fail before SVD/cache/updates."""
    from luban.evolution import _target
    from verify_cached_training import tensor_sha
    record={} if report is None else report
    candidate=parent['candidate_sha256'];protect=set(selected['protection_ids'])
    news=[r for r in parent['groups']['new_training'] if r['id'] in protect]
    old_ids=prior.correct(parent['groups']['replay_training'],parent['baseline']['replay'])
    replay=[r for r in parent['groups']['replay_training'] if r['id'] in old_ids]
    require(len(news)==len(protect) and len(replay)==55,'Exact fixed entry protection coverage required')
    record.update(candidate_tensor_sha256=candidate,new=[],replay=[],passed=False,generation_calls=0,
        protected_new_ids=[r['id'] for r in news],protected_replay_ids=[r['id'] for r in replay],
        source_rows_sha256={'new':tensor.content_sha(news),'replay':tensor.content_sha(replay)})
    for name,rows in (('new',news),('replay',replay)):
        for row in rows:
            require(time.perf_counter()<deadline,'Entry closedbook protection exceeded2400')
            require(tensor_sha(_target(runtime)[1])==candidate,'Entry actual parent weight changed before probe')
            record[name].extend(prior.probes(runtime,[row],deadline,candidate));record['generation_calls']+=1
            require(tensor_sha(_target(runtime)[1])==candidate,'Entry actual parent weight changed after probe')
    forgotten=(protect-prior.correct(news,record['new']))|(old_ids-prior.correct(replay,record['replay']))
    record['forgotten_ids']=sorted(forgotten);record['passed']=not forgotten
    return record


def final_check(runtime,parent,selected,candidate_sha,entry_loss,deadline):
    from luban.evolution import _loss,_target
    from verify_cached_training import tensor_sha
    def frozen():
        require(time.perf_counter()<deadline,'Continuation fullwall exceeded during final proof')
        require(tensor_sha(_target(runtime)[1])==candidate_sha,'Continuation frozen proof actualW changed')
    news=parent['groups']['new_training'];old_ids=set(prior.correct(parent['groups']['replay_training'],parent['baseline']['replay']))
    olds=[r for r in parent['groups']['replay_training'] if r['id'] in old_ids]
    require(len(olds)==55,'Actual original55 replay proof required')
    frozen();one=prior.probes(runtime,news,deadline,candidate_sha);frozen()
    two=prior.probes(runtime,news,deadline,candidate_sha);frozen()
    old=prior.probes(runtime,olds,deadline,candidate_sha);frozen()
    now=prior.correct(news,one)&prior.correct(news,two);finite=prior.correct(news,one,finite=True)&prior.correct(news,two,finite=True)
    previous=prior.correct(news,parent['point']['new_first'],finite=True)&prior.correct(news,parent['point']['new_second'],finite=True)
    forgotten=(set(selected['protection_ids'])-now)|(old_ids-prior.correct(olds,old))
    loss=_loss(runtime,selected['replay_training']);frozen()
    require(math.isfinite(loss) and math.isfinite(entry_loss) and entry_loss>0,'Finite expanded replay quality loss required')
    gains=finite-previous;taught=gains&set(selected['selected_ids'])
    return {'candidate_tensor_sha256':candidate_sha,'new_first':one,'new_second':two,'old_preservation':old,
        'newly_confirmed_ids':sorted(gains),'selected_taught_finite_gain_ids':sorted(taught),
        'selected_taught_finite_gain':len(taught),'first_finite_f1_ids':selected['first_finite_f1_ids'],
        'first_finite_f1_preserved':set(selected['first_finite_f1_ids']).issubset(now),
        'forgotten_ids':sorted(forgotten),'expanded_replay_loss':loss,'entry_expanded_replay_loss':entry_loss,
        'safety_passed':not forgotten and loss<=entry_loss*1.05,'mastery_confirmed':True,'generation_calls':169,
        'quality_passed':not forgotten and loss<=entry_loss*1.05 and bool(taught),'publication':False}


def train_round(runtime,parent,selected,master,optimizer,reference,basis,caches,entry_loss,plan,output,deadline,protocol_sha):
    import torch
    from luban.evolution import _target
    from cached_method_steps import train_method_step,_adam_state
    from verify_cached_training import tensor_sha,_sync
    schedule=continuation_schedule(parent,selected)
    require(tensor.content_sha(schedule)==plan['schedule_sha256'],'Continuation frozen schedule differs')
    parameter=_target(runtime)[1];entry=parameter.detach().clone()
    rows={r['id']:r for r in parent['groups']['new_training']+selected['replay_training']}
    report={'completed':False,'quality_passed':False,'entry_optimizer_step':450,'optimizer_updates':0,'final_optimizer_step':450,
        'updates':[],'exposures':{k:{} for k in ACCOUNTS},'checks':[],'publication':False,
        'timing':{'optimization_seconds':0.,'checkpoint_seconds':0.,'check_seconds':0.}}
    def guard():
        require(time.perf_counter()<deadline,'Continuation2400 fullwall exceeded')
        require(all(sha(p)==h for p,h in plan['source_sha256'].items()),'Continuation execution source changed')
    def persist():
        (output/'training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    try:
        for batch in schedule['batches']:
            guard();_sync(parameter.device);began=time.perf_counter()
            actual=train_method_step(runtime,[rows[id] for id in batch['ce_ids']],[rows[id] for id in batch['replay_kl_ids']],
                master,optimizer,reference,basis,method='cache',ce_role=batch['ce_role'],caches=caches)
            _sync(parameter.device);wall=time.perf_counter()-began
            require(actual.get('runtime_restored') is True and actual.get('optimizer_steps')==1
                and actual.get('ce_ids')==batch['ce_ids'] and actual.get('replay_kl_ids')==batch['replay_kl_ids']
                and actual.get('ce_role')==batch['ce_role'] and _adam_state(optimizer,master)==batch['cumulative_optimizer_step'],
                'Actual continuation Adam/exposure/restoration differs')
            report['updates'].append({**actual,'schedule_batch':batch,'synchronized_update_wall_seconds':wall})
            report['optimizer_updates']+=1;report['final_optimizer_step']=_adam_state(optimizer,master)
            report['timing']['optimization_seconds']+=wall
            for key,ids in ((batch['ce_role']+'_ce',batch['ce_ids']),('replay_kl',batch['replay_kl_ids'])):
                for id in ids:report['exposures'][key][id]=report['exposures'][key].get(id,0)+1
            if report['optimizer_updates']%50==0:persist()
        guard();require(report['exposures']==schedule['round_exposures'] and _adam_state(optimizer,master)==900,'Complete continuation900 ledger differs')
        report['cumulative_exposures']=schedule['cumulative_exposures'];report['cumulative_totals']=schedule['cumulative_totals']
        with torch.no_grad():parameter.copy_(master.to(parameter.dtype))
        began=time.perf_counter();saved=curve.save_weights(parameter,output/'lastdown-cumulative900.pt')
        report['saved_weights']=saved
        state=optimizer.state[master]
        report['state_summary']={'native_tensor_sha256':saved['tensor_sha256'],'master_sha256':tensor_sha(master),
            'optimizer_state_sha256':{k:tensor_sha(v.reshape(-1)) for k,v in state.items()},
            'entry_reference_sha256':tensor_sha(reference),'new_phase_delta_sha256':tensor_sha(master-reference),
            'new_basis_sha256':tensor_sha(basis),'cumulative_optimizer_step':900}
        full={'protocol':PROTOCOL,'protocol_sha256':protocol_sha,'identity':IDENTITY,'source_sha256':plan['source_sha256'],
            'parent_full_state':plan['parent_full_state'],'candidate':parameter.detach().cpu().clone(),
            'master':master.detach().cpu().clone(),'optimizer_state_dict':copy.deepcopy(optimizer.state_dict()),
            'cpu_rng':torch.get_rng_state().clone(),'mps_rng':torch.mps.get_rng_state().clone() if parameter.device.type=='mps' else None,
            'reference':reference.detach().cpu().clone(),'basis':basis.detach().cpu().clone(),
            'optimizer_updates':900,'round_updates':450,'exposures':schedule['cumulative_exposures'],
            'round_exposures':report['exposures'],'native_checkpoint':saved,'curriculum':selected,'state_summary':report['state_summary']}
        for values in full['optimizer_state_dict']['state'].values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor):values[key]=value.detach().cpu().clone()
        target=output/'training-state-cumulative900.pt'
        with safe(target).open('xb') as stream:torch.save(full,stream);stream.flush();os.fsync(stream.fileno())
        report['full_training_state']={'path':str(target),'file_sha256':sha(target),'file_bytes':target.stat().st_size}
        del full;_sync(parameter.device);report['timing']['checkpoint_seconds']+=time.perf_counter()-began;guard()
        began=time.perf_counter();point=final_check(runtime,parent,selected,saved['tensor_sha256'],entry_loss,deadline)
        _sync(parameter.device);report['timing']['check_seconds']+=time.perf_counter()-began;report['checks'].append(point);guard()
        report['completed']=point['safety_passed'];report['quality_passed']=point['quality_passed']
        if not report['quality_passed']:report['error']={'type':'ValueError','message':'Actual new finite taught growth/old protection failed; no adoption'}
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:
        with torch.no_grad():parameter.copy_(entry)
        report['entry_tensor_restored']=tensor_sha(parameter)==tensor_sha(entry);persist()
        if time.perf_counter()>=deadline or report['entry_tensor_restored'] is not True:
            report['completed']=report['quality_passed']=False
            report['error']={'type':'TimeoutError' if time.perf_counter()>=deadline else 'ValueError','message':'Final continuation raw save/entry restore failed or exceeded2400'}
            persist()
    return report


def execute(path,approved_sha):
    require(isinstance(approved_sha,str) and re.fullmatch('[0-9a-f]{64}',approved_sha) and sha(safe(path))==approved_sha,'Exact Root approved formal continuationSHA required')
    plan=read(path);require(plan.get('protocol')==PROTOCOL and plan.get('sealed') is True
        and plan.get('state')=='sealed_waiting_explicit_MODEL_GO','Separately sealed continuation and Root MODEL GO required')
    began=time.perf_counter();deadline=began+WALL_SECONDS;validate_bound(plan)
    require(time.perf_counter()<deadline,'Continuation static preflight exceeded fullwall')
    output=safe(plan['output']);require(not output.exists(),'Fresh single-attempt output required')
    write_new(safe(path).parent/'execution-started.json',{'protocol_sha256':approved_sha,'output':str(output),'started':time.time()});output.mkdir(mode=0o700)
    report={'protocol':PROTOCOL,'protocol_sha256':approved_sha,'pid':os.getpid(),'model_started':False,'passed':False,
        'teacher_calls':0,'publication':False,'source_before':plan['source_sha256']}
    write_new(output/'run-started.json',report)
    runtime=parameter=original=flags=modes=rng=mps_rng=basis=caches=hooks=backward_hooks=pointer=None
    master=optimizer=reference=parent_rng=parent_mps_rng=None;pointer_before=None;resources=None
    try:
        import torch
        from luban.runtime import Runtime
        from luban.settings import Settings
        from luban.memory import Memory
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target,_capture_replay_subspace,_loss
        from verify_cached_training import tensor_sha,capture_tail,_sync
        from verify_automatic_batches import measured
        parent=closed_parent(plan['parent_descriptor_path'],plan['parent_backup_path'],plan['parent_readback_path'])
        selected=curriculum(parent);require(selected==plan['curriculum'],'Bound pending/expanded replay changed')
        with tempfile.TemporaryDirectory(prefix='luban-continuation450-') as home:
            settings=Settings(home);settings.update({'storage_root':str(output/'private'),'model_path':plan['model_path'],'temperature':0,'local_thinking':'off'})
            runtime=Runtime(settings,Memory(output/'private'),research=True)
            require(str(runtime.device)==plan['device'] and str(runtime.dtype)==plan['dtype'],'Bound continuation hardware/dtype changed')
            runtime._model_process_lock=ModelProcessLock(IDENTITY,runtime.device)
            with measured(runtime) as resources:
                phase=time.perf_counter();runtime.load(paging=False);report['model_started']=True;report['load_seconds']=time.perf_counter()-phase
                parameter=_target(runtime)[1];original=parameter.detach().clone()
                require(runtime.identity==IDENTITY and runtime.active_core is None and runtime.branch.expert is None
                    and tensor_sha(original)==ORIGINAL,'Actual originalNone1706 loading required before parent import')
                flags=[(p,p.requires_grad,p.grad) for p in runtime.model.parameters()];modes=[(m,m.training) for m in runtime.model.modules()]
                hooks=[(m,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backward_hooks=[(p,len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                pointer=Path(runtime.store)/'active-core.json';pointer_before=pointer.read_bytes() if pointer.exists() else None
                rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone() if parameter.device.type=='mps' else None
                runtime.model.eval().requires_grad_(False);parameter.requires_grad_(True)
                phase=time.perf_counter();master,optimizer,reference,parent_rng,parent_mps_rng=load_parent_state(runtime,parent)
                report['parent_import_seconds']=time.perf_counter()-phase
                report['parent_import']={'native_sha256':tensor_sha(parameter),'master_sha256':tensor_sha(master),
                    'entry_reference_sha256':tensor_sha(reference),'optimizer_step':450,'reference_is_original':False,
                    'native_equals_reference_BF16':torch.equal(parameter,reference.to(parameter.dtype))}
                phase=time.perf_counter();report['entry_protection']={}
                try:entry_protection(runtime,parent,selected,deadline,report['entry_protection'])
                finally:report['entry_protection_seconds']=time.perf_counter()-phase
                require(report['entry_protection']['passed'] is True,'Actual parent entry protected knowledge failed; no update or SVD/cache')
                phase=time.perf_counter();entry_loss=_loss(runtime,selected['replay_training'])
                require(math.isfinite(entry_loss) and entry_loss>0,'Actual entry expandedreplay loss nonfinite')
                report['entry_expanded_replay_loss']=entry_loss;report['entry_loss_seconds']=time.perf_counter()-phase
                phase=time.perf_counter();native_basis,basis_report=_capture_replay_subspace(runtime,parameter,selected['replay_training'])
                basis=native_basis.to(parameter.device);del native_basis;_sync(parameter.device)
                report['expanded_replay_svd']=basis_report;report['basis_seconds']=time.perf_counter()-phase
                caches={};rows={r['id']:r for r in parent['groups']['new_training']+selected['replay_training']};phase=time.perf_counter()
                for id in selected['selected_ids']+[r['id'] for r in selected['replay_training']]:
                    require(time.perf_counter()<deadline,'Continuation setup fullwall exceeded')
                    caches[id]=capture_tail(runtime,rows[id])
                _sync(parameter.device);report['cache_build_seconds']=time.perf_counter()-phase
                report['cache_bytes']=sum(sum(v['bytes'].values()) for v in caches.values())
                report['parent_rng_restored_before_update451']=restore_parent_rng(parent_rng,parent_mps_rng)
                report['common_seconds']=time.perf_counter()-began;report['parent_confirmed_ids']=parent['confirmed_ids'];report['curriculum']=selected
                report['training']=train_round(runtime,parent,selected,master,optimizer,reference,basis,caches,entry_loss,plan,output,deadline,approved_sha)
                report['passed']=report['training']['completed'] and report['training']['quality_passed'];report['resources']=resources
    except BaseException as error:report['error']={'type':type(error).__name__,'message':str(error)}
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
                        'flags_grads':all(p.requires_grad==f and p.grad is g for p,f,g in flags),'modes':all(m.training==v for m,v in modes),
                        'rng':torch.equal(rng,torch.get_rng_state()),'mps_rng':mps_rng is None or torch.equal(mps_rng,torch.mps.get_rng_state()),
                        'core_branch':runtime.active_core is None and runtime.branch.expert is None,
                        'forward_hooks':all(len(m._forward_hooks)==a and len(m._forward_pre_hooks)==b for m,a,b in hooks),
                        'backward_hooks':all(len(p._backward_hooks or {})==v for p,v in backward_hooks),
                        'pointer':(pointer.read_bytes() if pointer.exists() else None)==pointer_before}
            except BaseException as error:report['restoration_error']={'type':type(error).__name__,'message':str(error)};report['passed']=False
            master=optimizer=reference=parent_rng=parent_mps_rng=parameter=original=flags=modes=rng=mps_rng=basis=caches=hooks=backward_hooks=pointer=None;p=m=None
            try:
                runtime._release_model()
                from luban.process_lock import ModelProcessLock
                lock=ModelProcessLock(IDENTITY,plan['device']);lock.close()
                report['model_released']=runtime.model is None and runtime._model_process_lock is None;report['actual_os_lock_reacquired_closed']=True
            except BaseException as error:report['release_error']={'type':type(error).__name__,'message':str(error)};report['passed']=False
        report['restore_release_seconds']=time.perf_counter()-restore_started
        try:
            from verify_router_coverage_study import model_binding
            report['source_after']={p:sha(p) for p in plan['source_sha256']};report['production_after']=production_snapshot(plan['production_root'])
            report['official_after']=model_binding(plan['model_path'],IDENTITY)
            report['parent_after']={k:sha(plan[k+'_path']) for k in ('parent_descriptor','parent_backup','parent_readback')}
            report['preserved']=report['source_after']==plan['source_sha256'] and report['production_after']==plan['production_snapshot']
            report['preserved']=report['preserved'] and report['official_after']==plan['official_model_binding'] and all(
                report['parent_after'][k]==plan[k+'_sha256'] for k in report['parent_after'])
        except BaseException as error:report['preservation_error']={'type':type(error).__name__,'message':str(error)};report['preserved']=False
        report['passed']=report['passed'] and report.get('preserved') is True and len(report.get('restoration',{}))==9 and all(report['restoration'].values()) and report.get('model_released') is True and report.get('actual_os_lock_reacquired_closed') is True
        report['wall_seconds_before_final_result_write']=time.perf_counter()-began
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        report['wall_seconds']=time.perf_counter()-began
        if time.perf_counter()>=deadline:report['passed']=False;report['error']={'type':'TimeoutError','message':'Continuation final restore/preservation/write exceeded2400'}
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        if time.perf_counter()>=deadline and report['passed']:
            report['passed']=False;report['error']={'type':'TimeoutError','message':'Last continuation raw write exceeded2400'}
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'continuation_fullwall_after_write_seconds':time.perf_counter()-began,'passed':report['passed'],'publication':False}),flush=True)
    if not report['passed']:raise RuntimeError('Continuation rejected/incomplete; preserve candidate/raw, no retry or adoption')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for action in ('prepare','bind','seal','run'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--parent-descriptor',type=Path)
    parser.add_argument('--parent-backup',type=Path);parser.add_argument('--parent-readback',type=Path);parser.add_argument('--approved-sha')
    args=parser.parse_args()
    if not any(v is not None for v in vars(args).values()):
        print(json.dumps({'protocol':PROTOCOL,'SOURCE_only':True,'torch_imported':False,'model_started':False,'CPU_tests_run':False,
            'entry_Adam':450,'exit_Adam':900,'new_updates':450,'source_count':69,'parent':'Waiting Root selected eligible closed parent','publication':False}));return
    if args.run:
        if not args.approved_sha or any((args.output,args.parent_descriptor,args.parent_backup,args.parent_readback)):parser.error('run only formal+exact approvedSHA')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if not all((args.output,args.parent_descriptor,args.parent_backup,args.parent_readback)) or args.approved_sha:parser.error('prepare fresh output plus three closed parent receipts')
        write_new(args.prepare,prepare(args.output,args.parent_descriptor,args.parent_backup,args.parent_readback))
    elif args.bind or args.seal:
        if not args.output or any((args.parent_descriptor,args.parent_backup,args.parent_readback,args.approved_sha)):parser.error('bind/seal only source/fresh output')
        plan=read(args.bind or args.seal)
        if args.bind:plan=bind(plan)
        else:validate_bound(plan);plan.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO')
        write_new(args.output,plan)
    else:parser.error('No implicit operation')

if __name__=='__main__':main()
