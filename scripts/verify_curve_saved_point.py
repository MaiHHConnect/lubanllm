"""One research-only saved-point second proof; zero optimizer/product adoption.

Default/prepare are stdlib only. No selection or closed actual curve receipt
means waiting. Root selects one point, independently binds/seals a fresh output,
then separately grants MODEL GO. Historical/live curve files are never polled.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time

import run_training_step_curve as curve
import run_tensor_method_pilot as tensor
import run_training_method_pilot as prior
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, content_sha, safe, write_new, production_snapshot

PROTOCOL = 'original57-saved450-500-550-second-proof-zero-update-v1'
POINTS = (450, 500, 550)
WALL_SECONDS = 900
RESTORATION = {'original_tensor', 'flags_grads', 'modes', 'rng', 'mps_rng', 'core_branch',
               'forward_hooks', 'backward_hooks', 'pointer'}


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(safe(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def require(value, message):
    if value is not True: raise ValueError(message)


def selected_summary(groups, baseline, point, ce_event):
    """Inspect SHA-bound first proof, no tensor deserialization or new answer."""
    require(type(ce_event) is int and ce_event in POINTS and point.get('ce_events') == ce_event,
            'Root must select exactly one existing450/500/550 point')
    require(tensor.training_materials(groups) is not None, 'Full ordered57NEW+57REPLAY TRAIN required')
    news, replay = groups['new_training'], groups['replay_training']
    before = prior.correct(news, baseline['new']); finite_before = prior.correct(news, baseline['new'], finite=True)
    replay_before = prior.correct(replay, baseline['replay'])
    first = prior.correct(news, point['new_first'], finite=True)
    require(len(before) == 13 and len(finite_before) == 12 and len(replay_before) == 55,
            'Historical original13NEW/finite12/55REPLAY binding changed')
    old_rows = [r for r in replay if r['id'] in replay_before]
    require(point.get('new_second') == [] and point.get('mastery_confirmed') is False
        and point.get('newly_confirmed_ids') is None and point.get('safety_passed') is True
        and point.get('publication') is False and not point.get('forgotten_ids')
        and prior.correct(news, point['new_first']) >= before
        and prior.correct(old_rows, point['old_preservation']) == replay_before,
        'Only complete safe first-proof point, never invented mastery')
    saved = point['saved_weights']; candidate = point['candidate_tensor_sha256']
    require(saved['tensor_sha256'] == candidate and saved['dtype'] == 'torch.bfloat16'
        and saved['shape'] == [2048, 6144] and saved['tensor_bytes'] == 2048*6144*2
        and point['provisional_correct_ids'] == sorted(first)
        and point['provisional_growth_ids'] == sorted(first-finite_before), 'Full nativeW/finite first-proof metadata differs')
    for rows, observations in ((news, point['new_first']), (old_rows, point['old_preservation'])):
        for row, observed in zip(rows, observations):
            require(observed.get('sample_sha256') == content_sha(row)
                and observed.get('actual_tensor_sha256') == candidate and observed.get('actual_base_identity') == IDENTITY
                and observed.get('core_version') is None, 'Original first proof full context/candidate binding differs')
    original_loss = baseline['replay_loss']; loss = point['replay_loss']
    require(type(original_loss) in (int,float) and math.isfinite(original_loss) and original_loss > 0
        and type(loss) in (int,float) and math.isfinite(loss) and 0 <= loss <= original_loss*1.05
        and point['baseline_replay_loss'] == original_loss, 'Existing native replay loss gate differs')
    return {'ce_event': ce_event, 'candidate_tensor_sha256': candidate, 'saved_weights': saved,
        'point_sha256': content_sha(point), 'new_first_sha256': content_sha(point['new_first']),
        'groups_sha256': content_sha(groups), 'baseline_sha256': content_sha(baseline),
        'original_correct_ids': {'new': sorted(before), 'finite_new': sorted(finite_before), 'replay': sorted(replay_before)},
        'first_finite_correct_ids': sorted(first), 'first_provisional_growth_ids': sorted(first-finite_before)}


def closed_receipt(path, ce_event):
    """Only actual exited evidence; file copies do not authenticate GPU execution."""
    receipt = read(path)
    require(receipt.get('scope') == 'actual_closed_stepcurve_saved_point_receipt'
        and type(receipt.get('process_exit_code')) is int
        and receipt.get('pid_dead') is True, 'Actual closed curve/Root process receipt required')
    values = {}
    for name in ('protocol', 'result', 'root_post'):
        target = safe(receipt[name+'_path'])
        require(sha(target) == receipt[name+'_sha256'], 'Closed '+name+' byte SHA differs')
        values[name] = read(target)
    plan, raw, post = (values[name] for name in ('protocol','result','root_post'))
    require(plan.get('protocol') == curve.PROTOCOL and plan.get('sealed') is True
        and raw.get('protocol') == curve.PROTOCOL and raw.get('protocol_sha256') == receipt['protocol_sha256']
        and post.get('scope') == 'actual_root_post' and post.get('evidence_preserved') is True
        and post.get('pid_dead') is True and post.get('process_exit_code') == receipt['process_exit_code']
        and post.get('protocol_sha256') == receipt['protocol_sha256'] and post.get('result_sha256') == receipt['result_sha256']
        and bool(post.get('checks')) and all(v is True for v in post['checks'].values())
        and post.get('checkpoint_payload_requested') is True
        and post.get('curve_recomputed', {}).get('point_payloads_verified') is True,
        'Actual immutable curve/rootpost preservation and CPU savedW proof required')
    require(set(raw.get('restoration',{})) == RESTORATION and all(v is True for v in raw['restoration'].values())
        and raw.get('preserved') is True and raw.get('model_released') is True
        and raw.get('actual_os_lock_reacquired_closed') is True and raw.get('publication') is False
        and type(raw.get('teacher_calls')) is int and raw['teacher_calls'] == 0,
        'Closed curve original state/release evidence missing')
    actual = raw.get('curve')
    if actual is None:
        durable = safe(Path(plan['output'])/'curve.json')
        require(sha(durable) == post.get('durable_curve_file_sha256'), 'Closed interrupted durable curve must bind actual Root post')
        actual = read(durable)
    groups, baseline = raw['groups'], raw['baseline']
    require(content_sha(groups) == raw['groups_sha256'], 'Closed full114 group SHA differs')
    tensor.verify_training_materials(groups, plan['b1_actual_reference'])
    points = [p for p in actual['checks'] if p.get('ce_events') == ce_event]
    require(len(points) == 1, 'Selected point absent/incomplete: remain unqualified, never refill')
    point = points[0]; summary = selected_summary(groups, baseline, point, ce_event)
    audited = [p for p in post['curve_recomputed']['checks'] if p.get('ce_events') == ce_event]
    require(len(audited) == 1 and audited[0]['candidate_tensor_sha256'] == summary['candidate_tensor_sha256']
        and audited[0]['safety_passed'] is True, 'Selected point not actually preserved by Root post')
    original = safe(point['saved_weights']['path'])
    require(sha(original) == point['saved_weights']['file_sha256'], 'Original savedW file changed')
    copies = receipt.get('saved_weight_copies')
    require(isinstance(copies,list) and len(copies) == 2 and len({str(safe(c['path'])) for c in copies}) == 2,
            'Two separately preserved full PT copies required')
    roots = (Path('/Volumes/移动数据库/lubanllm'), Path('/Volumes/Untitled/lubanllm-backup'))
    for target, copy_record in zip(roots, copies):
        saved = safe(copy_record['path'])
        require(target in saved.parents and copy_record['file_sha256'] == point['saved_weights']['file_sha256']
            and sha(saved) == copy_record['file_sha256'] and saved.stat().st_size == point['saved_weights']['file_bytes'],
            'Full selected nativeW primary/backup byte mismatch')
    sources = plan['source_sha256']
    require(len(sources) == 61 and sources == raw['source_before'] == raw['source_after']
        and all(sha(p) == value for p,value in sources.items()), 'Closed actual61 execution sources changed')
    return {'receipt': receipt, 'curve_plan': plan, 'raw': raw, 'post': post,
            'groups': groups, 'baseline': baseline, 'point': point, 'summary': summary}


def prepare(output, receipt_path=None, ce_event=None):
    require(ce_event is None or type(ce_event) is int and ce_event in POINTS, 'Explicit450/500/550, no automatic best selection')
    output = safe(output)
    value = {'protocol': PROTOCOL, 'state': 'waiting_closed_curve_root_post_dual_PT_and_one_Root_selected_point',
        'sealed': False, 'model_go': False, 'output': str(output), 'ce_event': ce_event,
        'curve_receipt_path': str(safe(receipt_path)) if receipt_path is not None else None,
        'identity': IDENTITY, 'original_core': None, 'original_tensor_sha256': ORIGINAL,
        'device': 'mps', 'dtype': 'torch.bfloat16', 'seed':42, 'wall_seconds':WALL_SECONDS,
        'optimizer_steps':0, 'teacher_calls':0, 'publication':False, 'product_accepted':False,
        'generation_calls':112, 'scope':'Second proof of one saved nativeW point; not another training/upgrade, DEV evaluation or product adoption'}
    if receipt_path is None or ce_event is None: return value
    data = closed_receipt(receipt_path, ce_event); old = data['curve_plan']
    for protected in (ROOT, safe(old['production_root'])/'app', safe(old['production_root'])/'modules',
                      safe(old['model_path']), safe(old['output'])):
        require(output != protected and protected not in output.parents, 'Fresh verifier outside source/model/product/curve output')
    require(not output.exists(), 'Single saved-point verification output must not exist')
    sources = dict(old['source_sha256'])
    for name in ('scripts/verify_curve_saved_point.py','tests/test_curve_saved_point.py'):
        sources[str(ROOT/name)] = sha(ROOT/name)
    value.update(state='unbound_waiting_Root_review', curve_receipt_sha256=sha(safe(receipt_path)),
        selected=data['summary'], model_path=old['model_path'], production_root=old['production_root'],
        source_sha256=dict(sorted(sources.items())), official_model_binding=None, production_snapshot=None,
        original_baseline_scope='Actual original114 from closed curve:13NEW/finite12/55REPLAY, no regeneration or oldlabel substitution',
        objective='Exactly one extra57NEW+55REPLAY naturalEOS/branchOFF/noRef generation and native57REPLAY meanloss; full first-proof sameW intersection',
        checkpoint_scope='Only native BF16 W is loaded. No master, Adam, optimizer or continuation from the old run')
    return value


def bind(plan):
    require(plan.get('state') == 'unbound_waiting_Root_review', 'Unselected/missing closed evidence remains waiting')
    require(prepare(plan['output'],plan['curve_receipt_path'],plan['ce_event']) == plan, 'Full saved-point source/material/selection changed')
    data = closed_receipt(plan['curve_receipt_path'],plan['ce_event'])
    from verify_router_coverage_study import model_binding
    official = model_binding(plan['model_path'],IDENTITY)
    production = production_snapshot(plan['production_root'])
    require(official == data['curve_plan']['official_model_binding'] == data['raw']['official_after']
        and production == data['curve_plan']['production_snapshot'] == data['raw']['production_after'],
        'Actual official13/production differ from immutable closed curve')
    return {**copy.deepcopy(plan), 'official_model_binding':official, 'production_snapshot':production,
            'state':'bound_waiting_Root_seal_explicit_MODEL_GO'}


def validate_bound(plan):
    template={**copy.deepcopy(plan), 'sealed':False, 'state':'unbound_waiting_Root_review',
        'official_model_binding':None, 'production_snapshot':None}
    require(bind(template) == {**plan,'sealed':False,'state':'bound_waiting_Root_seal_explicit_MODEL_GO'},
            'Exact bound saved-point contract changed')


def raw_proofs(rows, observations, candidate, eos):
    """Natural generation EOS is distinct from the gold training tokenizer EOS."""
    from luban.generation_checks import validate_generation_check, evaluate_generation
    require(prior.correct(rows, observations) is not None, 'Full fixed observation order/grade required')
    require(isinstance(eos, list) and eos and all(type(v) is int and v>=0 for v in eos)
        and len(set(eos))==len(eos), 'Actual generation EOS singleton/list required')
    for row, value in zip(rows, observations):
        rule=validate_generation_check(json.loads(row['provenance'])['generation_check'])
        maximum=rule.get('max_new_tokens',192); tokens=value.get('raw_completion_token_ids')
        require(value.get('sample_sha256')==content_sha(row) and value.get('actual_tensor_sha256')==candidate
            and value.get('actual_base_identity')==IDENTITY and value.get('core_version') is None
            and value.get('thinking')=='off' and value.get('do_sample') is False
            and isinstance(tokens,list) and 1<=len(tokens)<=maximum
            and all(type(v) is int and v>=0 for v in tokens) and not any(v in eos for v in tokens[:-1])
            and value['complete']==(tokens[-1] in eos) and (value['complete'] or len(tokens)==maximum)
            and type(value.get('completion_tokens')) is int and value['completion_tokens']==len(tokens)
            and type(value.get('max_new_tokens')) is int and value['max_new_tokens']==maximum
            and isinstance(value.get('decoded_completion_unstripped'),str)
            and value['answer']==value['decoded_completion_unstripped'].strip(), 'Full context/raw/naturalEOS metadata drift')
        grade=evaluate_generation(value['answer'],rule,value['complete'])
        require(all(type(value.get(k)) is type(v) and value[k]==v for k,v in grade.items()), 'Fixed gold/rule grade differs')
    return True


def verify_second(groups, baseline, point, second):
    """Regrade both proofs; the actual generation/raw EOS details remain mandatory."""
    summary=selected_summary(groups,baseline,point,point['ce_events'])
    news,replay=groups['new_training'],groups['replay_training']
    before=set(summary['original_correct_ids']['new']);finite_before=set(summary['original_correct_ids']['finite_new'])
    replay_before=set(summary['original_correct_ids']['replay']);old_rows=[r for r in replay if r['id'] in replay_before]
    again=prior.correct(news,second['new_second']);finite_again=prior.correct(news,second['new_second'],finite=True)
    old=prior.correct(old_rows,second['old_preservation'])
    confirmed=sorted((set(summary['first_finite_correct_ids']) & finite_again)-finite_before)
    forgotten=sorted((before-again)|(replay_before-old))
    loss=second['replay_loss'];safe_loss=type(loss) in (int,float) and math.isfinite(loss) and 0<=loss<=baseline['replay_loss']*1.05
    require(second.get('ce_event') == point['ce_events'] and second.get('saved_weights') == point['saved_weights']
        and second.get('newly_confirmed_ids') == confirmed and second.get('forgotten_ids') == forgotten
        and second.get('mastery_confirmed') is True and second.get('safety_passed') is (not forgotten and safe_loss)
        and second.get('publication') is False, 'Actual second proof/growth/loss differs from immutable full first proof')
    candidate=summary['candidate_tensor_sha256']
    for rows,observed in ((news,second['new_second']),(old_rows,second['old_preservation'])):
        for row,value in zip(rows,observed):
            require(value.get('sample_sha256')==content_sha(row) and value.get('actual_tensor_sha256')==candidate
                and value.get('actual_base_identity')==IDENTITY and value.get('core_version') is None,
                'Full candidate/context raw evidence differs')
    return {'candidate_confirmed_growth_ids':confirmed,'candidate_confirmed_growth':len(confirmed),
        'forgotten_ids':forgotten,'safety_passed':not forgotten and safe_loss,
        'original_correct_new':13,'original_finite_correct_new':12,'original_correct_replay':55,
        'production_learned_ids':[],'product_accepted':False,'optimizer_steps':0}


def execute(path, approved_sha):
    require(sha(safe(path)) == approved_sha, 'Exact Root-approved protocol SHA required')
    plan=read(path)
    require(plan.get('protocol')==PROTOCOL and plan.get('sealed') is True
        and plan.get('state')=='sealed_waiting_explicit_MODEL_GO', 'Fresh separately sealed one-point proof required')
    began=time.perf_counter();deadline=began+WALL_SECONDS;validate_bound(plan)
    output=safe(plan['output']);require(not output.exists(),'Fresh once output required')
    write_new(safe(path).parent/'execution-started.json',{'protocol_sha256':approved_sha,'output':str(output),'started':time.time()})
    output.mkdir(mode=0o700)
    report={'protocol':PROTOCOL,'protocol_sha256':approved_sha,'pid':os.getpid(),'passed':False,
        'model_started':False,'optimizer_steps':0,'teacher_calls':0,'publication':False,'product_accepted':False,
        'source_before':plan['source_sha256'],'production_before':plan['production_snapshot'],'ce_event':plan['ce_event']}
    write_new(output/'run-started.json',report)
    runtime=parameter=original=flags=modes=rng=mps_rng=hooks=backward=pointer_before=resources=None
    try:
        data=closed_receipt(plan['curve_receipt_path'],plan['ce_event'])
        import torch
        from luban.runtime import Runtime
        from luban.memory import Memory
        from luban.settings import Settings
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target
        from verify_cached_training import tensor_sha
        from verify_automatic_batches import measured
        from run_capacity_qualification import generation_eos
        with tempfile.TemporaryDirectory(prefix='luban-saved-point-proof-') as home:
            settings=Settings(home);settings.update({'storage_root':str(output/'private'),'model_path':plan['model_path'],'temperature':0,'local_thinking':'off'})
            runtime=Runtime(settings,Memory(output/'private'),research=True)
            require(str(runtime.device)==plan['device'] and str(runtime.dtype)==plan['dtype'],'Fixed native MPS/BF16 changed')
            runtime._model_process_lock=ModelProcessLock(IDENTITY,runtime.device)
            with measured(runtime) as resources:
                runtime.load(paging=False);report['model_started']=True;parameter=_target(runtime)[1];original=parameter.detach().clone()
                require(runtime.identity==IDENTITY and runtime.active_core is None and runtime.branch.expert is None
                    and tensor_sha(original)==ORIGINAL,'Official originalNone1706 before loading saved nativeW')
                flags=[(p,p.requires_grad,p.grad) for p in runtime.model.parameters()];modes=[(m,m.training) for m in runtime.model.modules()]
                rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone() if runtime.device=='mps' else None
                hooks=[(m,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backward=[(p,len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                pointer=Path(runtime.store)/'active-core.json';pointer_before=pointer.read_bytes() if pointer.exists() else None
                torch.manual_seed(42);runtime.model.eval().requires_grad_(False)
                if time.perf_counter()>=deadline:raise TimeoutError('Saved-point full900 preflight/load cap exceeded')
                report['generation_eos_token_ids']=generation_eos(runtime)
                groups,baseline,point=data['groups'],data['baseline'],data['point']
                replay_before=set(plan['selected']['original_correct_ids']['replay'])
                old_rows=[r for r in groups['replay_training'] if r['id'] in replay_before]
                report['second_proof']=curve.confirm_saved_point(runtime,groups,baseline,old_rows,point,deadline)
                report['proof_recomputed']=verify_second(groups,baseline,point,report['second_proof'])
                raw_proofs(groups['new_training'],report['second_proof']['new_second'],plan['selected']['candidate_tensor_sha256'],report['generation_eos_token_ids'])
                raw_proofs(old_rows,report['second_proof']['old_preservation'],plan['selected']['candidate_tensor_sha256'],report['generation_eos_token_ids'])
                require(tensor_sha(parameter)==ORIGINAL,'Inner savedW proof did not restore official original')
                report['passed']=report['proof_recomputed']['safety_passed']
    except BaseException as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        if isinstance(resources,dict):report['resources']=resources
        if runtime is not None:
            try:
                import torch
                from verify_cached_training import tensor_sha
                if runtime.model is not None and original is not None:
                    with torch.no_grad():parameter.copy_(original)
                if flags is not None:
                    for p,f,g in flags:p.requires_grad_(f);p.grad=g
                    for m,mode in modes:m.training=mode
                    torch.set_rng_state(rng)
                    if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
                    report['restoration']={'original_tensor':tensor_sha(parameter)==ORIGINAL,
                        'flags_grads':all(p.requires_grad==f and p.grad is g for p,f,g in flags),
                        'modes':all(m.training==mode for m,mode in modes),'rng':torch.equal(rng,torch.get_rng_state()),
                        'mps_rng':mps_rng is None or torch.equal(mps_rng,torch.mps.get_rng_state()),
                        'core_branch':runtime.active_core is None and runtime.branch.expert is None,
                        'forward_hooks':all(len(m._forward_hooks)==a and len(m._forward_pre_hooks)==b for m,a,b in hooks),
                        'backward_hooks':all(len(p._backward_hooks or {})==n for p,n in backward),
                        'pointer':(pointer.read_bytes() if pointer.exists() else None)==pointer_before}
            except BaseException as error:report['restoration_error']={'type':type(error).__name__,'message':str(error)}
            parameter=original=flags=modes=rng=mps_rng=hooks=backward=pointer_before=None;p=m=None
            try:
                runtime._release_model()
                from luban.process_lock import ModelProcessLock
                lock=ModelProcessLock(IDENTITY,plan['device']);lock.close()
                report['model_released']=runtime.model is None and runtime._model_process_lock is None
                report['actual_os_lock_reacquired_closed']=True
            except BaseException as error:report['release_error']={'type':type(error).__name__,'message':str(error)}
        try:
            from verify_router_coverage_study import model_binding
            report['source_after']={p:sha(p) for p in plan['source_sha256']}
            report['official_after']=model_binding(plan['model_path'],IDENTITY)
            report['production_after']=production_snapshot(plan['production_root'])
            report['preserved']=report['source_after']==plan['source_sha256'] and report['official_after']==plan['official_model_binding'] and report['production_after']==plan['production_snapshot']
            report['closed_receipt_preserved']=sha(plan['curve_receipt_path'])==plan['curve_receipt_sha256'] and closed_receipt(plan['curve_receipt_path'],plan['ce_event'])['summary']==plan['selected']
        except BaseException as error:report['preservation_error']={'type':type(error).__name__,'message':str(error)}
        report['wall_seconds_before_final_result_write']=time.perf_counter()-began
        report['passed']=bool(report['passed'] and not any(k.endswith('error') for k in report)
            and report.get('preserved') is True and report.get('closed_receipt_preserved') is True
            and set(report.get('restoration',{}))==RESTORATION and all(v is True for v in report['restoration'].values())
            and report.get('model_released') is True and report.get('actual_os_lock_reacquired_closed') is True
            and report['wall_seconds_before_final_result_write']<WALL_SECONDS)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        wall=time.perf_counter()-began
        if wall>=WALL_SECONDS:
            report['passed']=False;report['error']={'type':'TimeoutError','message':'Entire saved-point restore/preservation/finalsave exceeded900'}
            report['wall_seconds_after_first_final_result_write']=wall
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'saved_point_full_wall_after_first_final_result_write_seconds':wall,'passed':report['passed'],
                         'optimizer_steps':0,'publication':False}),flush=True)
    if report['passed'] is not True:raise RuntimeError('Saved-point proof rejected/incomplete; no retries/upgrade/publication')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for name in ('prepare','bind','seal','run'):actions.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--curve-receipt',type=Path)
    parser.add_argument('--ce-event',type=int,choices=POINTS);parser.add_argument('--approved-sha')
    args=parser.parse_args()
    if not any(value is not None for value in vars(args).values()):
        print(json.dumps({'protocol':PROTOCOL,'state':'waiting_Root_selected_saved_point_and_closed_evidence',
            'model_started':False,'torch_imported':False,'optimizer_steps':0,'publication':False}));return
    if args.run:
        if args.approved_sha is None or any(v is not None for v in (args.output,args.curve_receipt,args.ce_event)):parser.error('Only exact separately sealed path/SHA for run')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if args.output is None or args.approved_sha is not None:parser.error('prepare needs fresh output, no approvedSHA')
        write_new(args.prepare,prepare(args.output,args.curve_receipt,args.ce_event))
    elif args.bind or args.seal:
        if args.output is None or any(v is not None for v in (args.curve_receipt,args.ce_event,args.approved_sha)):parser.error('bind/seal only input+fresh output')
        value=read(args.bind or args.seal)
        if args.bind:value=bind(value)
        else:validate_bound(value);value.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO')
        write_new(args.output,value)
    else:parser.error('No implicit model/action')


if __name__=='__main__':main()
