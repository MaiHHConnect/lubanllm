"""Independent nested NEW16/32/57 fixed450 Adam; all57 final frozen double proof.

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

import run_training_end_only450 as end_only_reference
import run_training_step_curve as prior_curve
import verify_curve_saved_point as saved_point
import run_tensor_method_pilot as tensor
import run_training_method_pilot as prior
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, safe, write_new, production_snapshot

PROTOCOL = 'original57-nested-new-subset450-tokenizer-eos-v1'
DEADLINE_SECONDS = 2400


# Frozen task-family/context selection, not picked from candidate answers.
# Historical design SHA3679579bc5eb432a83e88078b1d42be14f8c19cb82f90e3e070cf9c61ab49cbb.
# No runtime planning-document dependency or new data source.
SELECTED_IDS = {16: ('1e78f266890a42a2a6ac11af53475b3d', '34cb8858bf3547e7947e0e688b8c7f12', '53290358b5444801b707dd87990ec814', '6642c4f89ef34fd2b369dfc102066a07', '70f728bbd2d64deea60b83c27eee4f42', '8dea8474b4c5490b9907c42d3e107511', '8e02c20e1b41458fb5c7b23982f459a7', 'ca35a1163ee1423d9533724254b6c8a8', 'f5a57b458fb74508a8c45a7d73a76d90', 'field-context-train-name-multi', 'field-context-train-name-single', 'field-context-train-name-system', 'field-context-train-position-system', 'field-context-train-tuner-single', 'field-paraphrase-train-relation-fresh-dev-api-ordinary', 'known-correction-new-train-tokyo'), 32: ('1e78f266890a42a2a6ac11af53475b3d', '34cb8858bf3547e7947e0e688b8c7f12', '4eab333276ad4158a2c40cee7513d7bd', '53290358b5444801b707dd87990ec814', '6583229d5bbb4ec59fa47f971ac35357', '6642c4f89ef34fd2b369dfc102066a07', '6bcce926e5024c82b2053e48d822aaa2', '70f728bbd2d64deea60b83c27eee4f42', '7b2a1e7d54a443088d2294fc8a1f3da4', '89ebbf69523442bca346fadef0e34d55', '8dea8474b4c5490b9907c42d3e107511', '8e02c20e1b41458fb5c7b23982f459a7', 'bb75c7ae3b8b4b6d89753834f04e5cb7', 'ca35a1163ee1423d9533724254b6c8a8', 'coverage-new-train-1', 'coverage-new-train-3', 'f5a57b458fb74508a8c45a7d73a76d90', 'field-context-train-name-multi', 'field-context-train-name-single', 'field-context-train-name-system', 'field-context-train-position-multi', 'field-context-train-position-single', 'field-context-train-position-system', 'field-context-train-tuner-multi', 'field-context-train-tuner-single', 'field-context-train-tuner-system', 'field-paraphrase-train-relation-fresh-dev-api-ordinary', 'known-correction-new-train-signal7', 'known-correction-new-train-tokyo', 'preservation-new-train-cedar', 'relation-reuse-train-v6-engineering-identity-3', 'relation-reuse-train-v6-engineering-identity-5'), 57: ('0a629f198e2946378efe7afdae3668d2', '150d39379ac04a59a4d3d5ca87f579e6', '163648ee70fe43ca92746a1d27654c07', '189b53c8eec241818719806aacc9db0f', '1e78f266890a42a2a6ac11af53475b3d', '24c9a4f5580b47d791b40a7e2a060f18', '34cb8858bf3547e7947e0e688b8c7f12', '3a7606813c4047acaf4db465eae1bdcf', '4eab333276ad4158a2c40cee7513d7bd', '4edd50e1d73c49b2806fee7e5b89caec', '519d475f17c44bbba3070ae8926e4103', '53290358b5444801b707dd87990ec814', '6583229d5bbb4ec59fa47f971ac35357', '6642c4f89ef34fd2b369dfc102066a07', '6bcce926e5024c82b2053e48d822aaa2', '6eba34ec96ce4c58866621ad374b3f6f', '70f728bbd2d64deea60b83c27eee4f42', '790bf39e957c4b869494c1c8382995f9', '7b2a1e7d54a443088d2294fc8a1f3da4', '853f65d5263f4167840dbedbfd24628c', '89ebbf69523442bca346fadef0e34d55', '8dea8474b4c5490b9907c42d3e107511', '8e02c20e1b41458fb5c7b23982f459a7', '95fba656796c40578a8956720947a311', 'a3ba1b1e3a81437388c2768f697143d1', 'a7d73b554bc343c980ce0752e8c7dc2c', 'bb75c7ae3b8b4b6d89753834f04e5cb7', 'ca35a1163ee1423d9533724254b6c8a8', 'coverage-new-train-0', 'coverage-new-train-1', 'coverage-new-train-2', 'coverage-new-train-3', 'e53800a3fdf34b8089541173d29f07ab', 'f5a57b458fb74508a8c45a7d73a76d90', 'field-context-train-name-multi', 'field-context-train-name-single', 'field-context-train-name-system', 'field-context-train-position-multi', 'field-context-train-position-single', 'field-context-train-position-system', 'field-context-train-tuner-multi', 'field-context-train-tuner-single', 'field-context-train-tuner-system', 'field-paraphrase-train-relation-fresh-dev-api-ordinary', 'field-paraphrase-train-relation-fresh-dev-name', 'field-paraphrase-train-relation-fresh-dev-position', 'field-paraphrase-train-relation-fresh-dev-tuner', 'known-correction-new-train-signal7', 'known-correction-new-train-tokyo', 'preservation-new-train-cedar', 'relation-attribute-train-name', 'relation-attribute-train-position', 'relation-attribute-train-tuner', 'relation-reuse-train-v6-engineering-identity-0', 'relation-reuse-train-v6-engineering-identity-3', 'relation-reuse-train-v6-engineering-identity-4', 'relation-reuse-train-v6-engineering-identity-5')}
KNOWN_IDS = ('189b53c8eec241818719806aacc9db0f', '1e78f266890a42a2a6ac11af53475b3d', '3a7606813c4047acaf4db465eae1bdcf', '4edd50e1d73c49b2806fee7e5b89caec', '6eba34ec96ce4c58866621ad374b3f6f', '89ebbf69523442bca346fadef0e34d55', 'ca35a1163ee1423d9533724254b6c8a8', 'coverage-new-train-0', 'coverage-new-train-1', 'coverage-new-train-3', 'e53800a3fdf34b8089541173d29f07ab', 'f5a57b458fb74508a8c45a7d73a76d90', 'field-paraphrase-train-relation-fresh-dev-api-ordinary')
NONFINITE_IDS = ('163648ee70fe43ca92746a1d27654c07', '24c9a4f5580b47d791b40a7e2a060f18', '34cb8858bf3547e7947e0e688b8c7f12', '4eab333276ad4158a2c40cee7513d7bd', '6bcce926e5024c82b2053e48d822aaa2', '790bf39e957c4b869494c1c8382995f9', '853f65d5263f4167840dbedbfd24628c', '8dea8474b4c5490b9907c42d3e107511', 'a3ba1b1e3a81437388c2768f697143d1', 'ca35a1163ee1423d9533724254b6c8a8')
CLASS_COUNTS = {16: {'known': 4, 'finite_pending': 10, 'contains_pending': 2}, 32: {'known': 7, 'finite_pending': 21, 'contains_pending': 4}, 57: {'known': 13, 'finite_pending': 35, 'contains_pending': 9}}


def subset_schedule(groups, n):
    if type(n) is not int or n not in SELECTED_IDS:raise ValueError('Explicit nested NEW pool16/32/57 only')
    if [r['id'] for r in groups['new_training']]!=list(SELECTED_IDS[57]):
        raise ValueError('Full ordered original57 NEW material pool changed')
    schedule=end_only_reference.prefix_schedule(groups)
    ledger={k:{} for k in ('new_ce','replay_ce','replay_kl')}
    for segment in schedule['segments']:
        for batch in segment['batches']:
            pair=batch['pair_indices'][0]
            batch['replay_kl_sample_indices']=[pair%57]
            if batch['ce_role']=='new':
                batch['ce_ids']=[SELECTED_IDS[n][pair%n]]
                batch['sample_indices']=[pair%n]
            for key,values in ((batch['ce_role']+'_ce',batch['ce_ids']),('replay_kl',batch['replay_kl_ids'])):
                for id in values:ledger[key][id]=ledger[key].get(id,0)+1
    schedule.update(selected_new_pool_n=n,selected_new_ids=list(SELECTED_IDS[n]),
        per_id_exposures=ledger,exposure_totals={k:sum(x.values()) for k,x in ledger.items()},
        meaning='450 actualAdam, only NEW cycle changes; original REPLAYCE/KL pairindex%57 untouched. Equalwork, unequal per-item dose.')
    return schedule


def selected_classes(groups, baseline, n):
    from luban.generation_checks import is_finite_generation_check,validate_generation_check
    news=groups['new_training'];known=prior.correct(news,baseline['new'])
    actual_nonfinite={r['id'] for r in news if not is_finite_generation_check(
        validate_generation_check(json.loads(r['provenance'])['generation_check'],default_expected=r['answer']))}
    if known!=set(KNOWN_IDS) or actual_nonfinite!=set(NONFINITE_IDS):
        raise ValueError('Fresh baseline or finite/contains rule population changed')
    selected=set(SELECTED_IDS[n])
    counts={'known':len(selected&known),'finite_pending':len(selected-known-actual_nonfinite),
            'contains_pending':len((selected-known)&actual_nonfinite)}
    if counts!=CLASS_COUNTS[n]:raise ValueError('Fixed subset nominal/known/pending counts changed')
    materials={r['id']:r for r in tensor.training_materials(groups)['new_training']}
    selected_materials=[materials[id] for id in SELECTED_IDS[n]]
    return {'nominal_new_pool_n':n,'selected_new_ids':list(SELECTED_IDS[n]),'classes':counts,
        'selected_training_materials':selected_materials,'selected_training_materials_sha256':tensor.content_sha(selected_materials),
        'finite_baseline_correct_ids':sorted(prior.correct(news,baseline['new'],finite=True)),
        'unselected_original_TRAIN_ids':sorted(set(SELECTED_IDS[57])-selected)}


def final_gain(point, groups, baseline, n):
    selected_classes(groups,baseline,n)
    selected=set(SELECTED_IDS[n]);confirmed=set(point['newly_confirmed_ids'])
    finite_twice=prior.correct(groups['new_training'],point['new_first'],finite=True)&prior.correct(groups['new_training'],point['new_second'],finite=True)
    if confirmed!=finite_twice-prior.correct(groups['new_training'],baseline['new'],finite=True):
        raise ValueError('Actual final finite two-proof growth differs')
    twice=prior.correct(groups['new_training'],point['new_first'])&prior.correct(groups['new_training'],point['new_second'])
    return {'selected_taught_finite_gain_ids':sorted(confirmed&selected),
        'selected_taught_finite_gain':len(confirmed&selected),'full57_finite_gain':len(confirmed),
        'unselected_original_TRAIN_observed_finite_gain_ids':sorted(confirmed-selected),
        'unselected_original_TRAIN_observed_finite_correct_ids':sorted(finite_twice-selected),
        'contains_rule_passed_twice_ids':sorted(twice&set(NONFINITE_IDS)),
        'scope':'Unselected rows already belong to original TRAIN; observed correctness is not transfer. Known/contains rows are never finite new learning.'}


def prepare(output, receipt_path, n):
    if type(n) is not int or n not in SELECTED_IDS:raise ValueError('Explicit nested16/32/57 only')
    plan=end_only_reference.prepare(output,receipt_path)
    data=saved_point.closed_receipt(receipt_path,450)
    schedule=subset_schedule(data['groups'],n);classes=selected_classes(data['groups'],data['baseline'],n)
    sources=dict(plan['source_sha256'])
    for name in ('scripts/run_training_subset450.py','tests/test_training_subset450.py'):
        sources[str(ROOT/name)]=sha(ROOT/name)
    plan.update(protocol=PROTOCOL,source_sha256=dict(sorted(sources.items())),selected_new_pool_n=n,
        selection=classes,schedule_sha256=tensor.content_sha(schedule),
        sampling='Only original ordered selected NEW pool cycles at pairindex%N; REPLAYCE and KL keep pairindex%57, whole57NEW+57REPLAY records unchanged.',
        mastery='Final full57NEW sameW doubleproof and55oldREPLAY/loss; primary positive taught finite gain. Unselected originalTRAIN observed correct is reported separately, not transfer.',
        scope='One independently bound nested subset16/32/57 first-upgrade450, equal Adam work but different NEW exposure distribution. No product/second upgrade/CPU or model measurement claimed by preparation.')
    return plan


def bind(plan):
    from verify_router_coverage_study import model_binding
    if prepare(plan['output'],plan['curve_receipt_path'],plan['selected_new_pool_n']) != plan:
        raise ValueError('Subset450 source/material/closedreceipt changed')
    return {**copy.deepcopy(plan),'official_model_binding':model_binding(plan['model_path'],IDENTITY),
            'production_snapshot':production_snapshot(plan['production_root']), 'state':'bound_waiting_root_seal_MODEL_GO'}


def validate_bound(plan):
    template={**copy.deepcopy(plan),'state':'unbound_waiting_root_review','sealed':False,
              'official_model_binding':None,'production_snapshot':None}
    if bind(template) != {**plan,'sealed':False,'state':'bound_waiting_root_seal_MODEL_GO'}:
        raise ValueError('Subset450 fixed source/model/production/receipt binding changed')


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
        'seed':42,'method':'cache','tensor_batch':1,'selected_new_pool_n':plan['selected_new_pool_n'],
        'selection':plan['selection'],'publication':False,'teacher_calls':0}
    before=tensor_sha(master)
    with safe(path).open('xb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    if tensor_sha(master)!=before or tensor_sha(payload['master'])!=before or tensor_sha(payload['candidate'])!=saved['tensor_sha256']:
        raise ValueError('Full checkpoint transfer changed master/nativeW')
    return {'path':str(path),'file_sha256':sha(path),'file_bytes':path.stat().st_size,
            'master_sha256':before,'candidate_tensor_sha256':saved['tensor_sha256'],
            'contains':['nativeBF16','FP32master','fullAdam','CPU/MPSrng','perIDledger','sourceBinding'],
            'basis_reference_tensors_saved':False,'continuation_implemented':False}


def subset(runtime,groups,baseline,old_rows,original,basis,caches,plan,output,deadline,protocol_sha):
    import torch
    from cached_method_steps import train_method_step
    from luban.evolution import _target
    from verify_cached_training import _sync,tensor_sha
    schedule=subset_schedule(groups,plan['selected_new_pool_n'])
    if tensor.content_sha(schedule)!=plan['schedule_sha256']:
        raise ValueError('Exact450 prefix schedule binding changed')
    rows={r['id']:r for r in groups['new_training']+groups['replay_training']}
    parameter=_target(runtime)[1]
    if tensor_sha(parameter)!=tensor_sha(original):raise ValueError('Fresh originalW required')
    master=torch.nn.Parameter(original.float().clone());reference=original.float()
    optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
    began=time.perf_counter()
    report={'method':'cache','tensor_batch':1,'selected_new_pool_n':plan['selected_new_pool_n'],'selection':selected_classes(groups,baseline,plan['selected_new_pool_n']),'ce_events':0,'optimizer_updates':0,'updates':[],
        'exposures':{k:{} for k in ('new_ce','replay_ce','replay_kl')},'checks':[],
        'completed':False,'quality_passed':False,'publication':False,
        'timing':{'optimization_seconds':0.,'check_seconds':0.,'checkpoint_seconds':0.}}
    def guard(sources=True):
        if time.perf_counter()>=deadline:raise TimeoutError('Subset4502400 fullwall exceeded')
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
        report['gain']=final_gain(point,groups,baseline,plan['selected_new_pool_n'])
        report['quality_passed']=report['completed'] and report['gain']['selected_taught_finite_gain']>0
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
        raise ValueError('Exact root approved sealed subset450SHA required')
    plan=json.loads(safe(path).read_text())
    if plan.get('protocol')!=PROTOCOL or plan.get('sealed') is not True or plan.get('state')!='sealed_waiting_explicit_MODEL_GO':
        raise ValueError('Separately sealed subset450+explicit MODEL GO required')
    began=time.perf_counter();deadline=began+DEADLINE_SECONDS;validate_bound(plan)
    if time.perf_counter()>=deadline:raise TimeoutError('Static preflight exceeded2400 before execution consumption')
    output=safe(plan['output'])
    if output.exists():raise FileExistsError('Fresh subset450 output required')
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
        with tempfile.TemporaryDirectory(prefix='luban-subset450-') as home:
            settings=Settings(home);settings.update({'storage_root':str(output/'private'),'model_path':plan['model_path'],'temperature':0,'local_thinking':'off'})
            runtime=Runtime(settings,Memory(output/'private'),research=True)
            if str(runtime.device)!=plan['device'] or str(runtime.dtype)!=plan['dtype']:raise ValueError('Subset450 native hardware/dtype changed')
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
                report['selection']=selected_classes(groups,baseline,plan['selected_new_pool_n'])
                if report['selection']!=plan['selection']:raise ValueError('Fresh actual subset baseline/selection binding changed')
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
                report['training']=subset(runtime,groups,baseline,old_rows,original,basis,caches,plan,output,deadline,approved_sha)
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
            report['passed']=False;report['error']={'type':'TimeoutError','message':'Complete subset450 restore/preservation/finalraw exceeded2400'}
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'subset450_full_wall_after_final_result_write_seconds':report['wall_seconds'],
                         'passed':report['passed'],'publication':False}),flush=True)
    if not report['passed']:raise RuntimeError('Subset450 incomplete/rejected; keepraw/weights, no retry/publication')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for action in ('prepare','bind','seal','run'):actions.add_argument('--'+action,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--curve-receipt',type=Path);parser.add_argument('--approved-sha')
    parser.add_argument('--n',type=int,choices=(16,32,57))
    args=parser.parse_args()
    if not any(v is not None for v in vars(args).values()):
        print(json.dumps({'protocol':PROTOCOL,'model_started':False,'torch_imported':False,'ce_events':450,
            'optimizer_updates':450,'method':'cache','tensor_batch':1,'checks':[450],'fullwall_cap_seconds':2400,
            'subset_choices':[16,32,57],'CPU_tests_run':False,'implementation_scope':'SOURCE only; no CPU/model, first upgrade only'}));return
    if args.run:
        if not args.approved_sha or args.output or args.curve_receipt or args.n is not None:parser.error('run only exact sealedpath/RootapprovedSHA after independent GO')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if args.output is None or args.curve_receipt is None or args.n is None or args.approved_sha:parser.error('prepare requires n16/32/57, freshoutput and closed actualcurve receipt')
        write_new(args.prepare,prepare(args.output,args.curve_receipt,args.n))
    elif args.bind or args.seal:
        if args.output is None or args.curve_receipt or args.approved_sha or args.n is not None:parser.error('bind/seal only input and freshoutput')
        value=saved_point.read(args.bind or args.seal)
        if args.bind:value=bind(value)
        else:validate_bound(value);value.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO')
        write_new(args.output,value)
    else:parser.error('No implicit operation/model')


if __name__=='__main__':main()
