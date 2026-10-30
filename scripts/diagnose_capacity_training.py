"""Closed N50 taught-TRAIN diagnostic: original/native candidate, zero updates.

Default/prepare are stdlib-only. Root must bind actual exited evidence and the
last complete saved checkpoint, seal a fresh output, then grant separate GO.
No optimizer, backward, generation, teacher or DEV; no training continuation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import time

import run_capacity_training as capacity
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, content_sha, safe, write_new, production_snapshot

PROTOCOL = 'closed-n50-original-candidate-taught-TRAIN-zero-update-unused-mps-cache-v2'
WALL_SECONDS = 900
RESTORATION = {'original_tensor','flags_grads','modes','rng','mps_rng','core_branch','forward_hooks','backward_hooks','pointer'}
read = capacity.select.read_json


def require(value, message):
    if value is not True: raise ValueError(message)


def last_complete_point(post, training, ce_events):
    points = post.get('completed_points',[])
    require(points and type(ce_events) is int and ce_events == max(p['ce_events'] for p in points),
        'Only Root-selected last complete point, never live current W')
    matches = [p for p in points if p['ce_events'] == ce_events]
    require(len(matches) == 1 and matches[0].get('payload_verified') is True, 'Actual Root full checkpoint payload proof')
    point = matches[0]; artifact = point['checkpoint_artifact']
    checks = [p for p in training['checks'] if p.get('ce_events') == ce_events]
    require(len(checks) == 1 and checks[0].get('checkpoint_artifact') == artifact
        and checks[0]['candidate_tensor_sha256'] == artifact['candidate_tensor_sha256'], 'Actual complete check/nativeW binding')
    return point, artifact


def closed(path):
    receipt = read(safe(path))
    require(receipt.get('scope') == 'actual_closed_N50_last_complete_checkpoint'
        and type(receipt.get('process_exit_code')) is int and receipt.get('pid_dead') is True,
        'Actual exited N50 receipt; partial exit must remain actual')
    objects = {}
    for name in ('protocol','result','root_post'):
        p = safe(receipt[name+'_path'])
        require(sha(p) == receipt[name+'_sha256'], 'Actual closed '+name+' byte SHA')
        objects[name] = read(p)
    plan, raw, post = (objects[k] for k in ('protocol','result','root_post'))
    require(plan.get('protocol') == capacity.PROTOCOL and plan.get('sealed') is True
        and plan.get('capacity_n') == 50 and type(plan['capacity_n']) is int
        and plan.get('original_core') is None and plan.get('original_tensor_sha256') == ORIGINAL
        and plan.get('identity') == IDENTITY and raw.get('protocol') == capacity.PROTOCOL
        and raw.get('protocol_sha256') == receipt['protocol_sha256'] and raw.get('capacity_n') == 50
        and post.get('scope') == 'actual_root_post' and post.get('pid_dead') is True
        and type(post.get('process_exit_code')) is int and post['process_exit_code'] == receipt['process_exit_code']
        and post.get('protocol_sha256') == receipt['protocol_sha256']
        and post.get('result_sha256') == receipt['result_sha256']
        and post.get('evidence_preserved') is True and bool(post.get('checks'))
        and all(v is True for v in post['checks'].values()), 'Closed actual N50/Root preservation, not quality substitution')
    require(type(post.get('execution_passed')) is bool and type(post.get('quality_passed')) is bool
        and set(raw.get('restoration',{})) == RESTORATION and all(v is True for v in raw['restoration'].values())
        and raw.get('preserved') is True and raw.get('model_released') is True
        and raw.get('actual_os_lock_reacquired_closed') is True, 'Actual nine restorations/release required')
    capacity.steps._qualification(read(plan['qualification_receipt_path']))
    training = raw.get('training')
    if training is None:
        durable = Path(plan['output'])/'training.json'
        require(sha(durable) == post.get('training_json_sha256'), 'Closed partial training bytes bound by actual post')
        training = read(durable)
    point, artifact = last_complete_point(post,training,receipt.get('ce_events'))
    checkpoint = safe(artifact['path'])
    require(sha(checkpoint) == artifact['file_sha256'], 'Original complete checkpoint unchanged')
    copies = receipt.get('checkpoint_copies')
    require(isinstance(copies,list) and len(copies) == 2 and len({str(safe(c['path'])) for c in copies}) == 2,
        'Two complete primary/backup payload copies required')
    for prefix, item in zip(('/Volumes/移动数据库/lubanllm','/Volumes/Untitled/lubanllm-backup'), copies):
        p = safe(item['path'])
        require(Path(prefix) in p.parents and item['file_sha256'] == artifact['file_sha256']
            and sha(p) == artifact['file_sha256'] and p.stat().st_size == checkpoint.stat().st_size, 'Full checkpoint dual byte identity')
    rows = plan['new_training']
    require(len(rows) == 50 and len({r['id'] for r in rows}) == 50
        and [r['id'] for r in rows] == plan['schedule']['new_ids']
        and all(r.get('purpose') == 'training' for r in rows), 'Fixed full taught50 TRAIN only')
    require(plan['source_sha256'] == raw['source_before'] == raw['source_after']
        and len(plan['source_sha256']) == 59 and all(sha(p) == h for p,h in plan['source_sha256'].items()), 'Closed59 current source identity')
    return {'plan':plan,'raw':raw,'post':post,'rows':rows,'artifact':artifact,'point':point,
        'summary':{'ce_events':receipt['ce_events'],'artifact':artifact,'rows_sha256':content_sha(rows),
            'source_process_exit_code':receipt['process_exit_code'],'source_execution_passed':post['execution_passed'],
            'source_quality_passed':post['quality_passed'],'last_complete_point_sha256':content_sha(point)}}


def prepare(output, receipt=None):
    output = safe(output)
    value = {'protocol':PROTOCOL,'state':'waiting_closed_N50_Root_post_dual_last_complete_checkpoint',
        'sealed':False,'model_go':False,'output':str(output),'receipt_path':str(safe(receipt)) if receipt else None,
        'identity':IDENTITY,'original_core':None,'original_tensor_sha256':ORIGINAL,'device':'mps','dtype':'torch.bfloat16',
        'seed':42,'wall_seconds':WALL_SECONDS,'optimizer_steps':0,'gradient_calls':0,'generation_calls':0,
        'teacher_calls':0,'publication':False,'product_accepted':False,'taught_TRAIN_count':50}
    if receipt is None: return value
    data = closed(receipt); previous = data['plan']
    product=safe(previous['production_root']);experiments=product/'experiments'
    require(experiments in output.parents, 'Fresh diagnostic child output under product/experiments, never its root')
    for target in (ROOT,product/'app',product/'modules',safe(previous['model_path']),safe(previous['output'])):
        require(output != target and target not in output.parents, 'Fresh independent diagnostic output')
    require(not output.exists(), 'Once output must not exist')
    sources = dict(previous['source_sha256'])
    for name in ('scripts/diagnose_capacity_training.py','tests/test_capacity_training_diagnostic.py'):
        sources[str(ROOT/name)] = sha(ROOT/name)
    value.update(state='unbound_waiting_Root_review',receipt_sha256=sha(safe(receipt)),selected=data['summary'],
        model_path=previous['model_path'],production_root=previous['production_root'],source_sha256=dict(sorted(sources.items())),
        official_model_binding=None,production_snapshot=None)
    return value


def bind(plan):
    require(plan.get('state') == 'unbound_waiting_Root_review', 'Actual closed evidence required')
    require(prepare(plan['output'],plan['receipt_path']) == plan, 'Exact source/material/checkpoint receipt')
    from verify_router_coverage_study import model_binding
    data = closed(plan['receipt_path']); official = model_binding(plan['model_path'],IDENTITY)
    production = production_snapshot(plan['production_root'])
    require(official == data['plan']['official_model_binding'] == data['raw']['official_after']
        and production == data['plan']['production_snapshot'] == data['raw']['production_after'], 'Official13/production actual closed equality')
    return {**copy.deepcopy(plan),'official_model_binding':official,'production_snapshot':production,'state':'bound_waiting_Root_seal_GO'}


def validate_bound(plan):
    original = {**plan,'sealed':False,'state':'unbound_waiting_Root_review','official_model_binding':None,'production_snapshot':None}
    require(bind(original) == {**plan,'sealed':False,'state':'bound_waiting_Root_seal_GO'}, 'Exact bound diagnostic')


def token_statistics(logits, labels):
    import torch
    from torch.nn import functional as F
    from luban.evolution import _answer_logits
    selected = _answer_logits(logits,labels); targets = labels[:,1:][labels[:,1:] != -100]
    ce = F.cross_entropy(selected,targets,reduction='none')
    first = selected[0]; gold = int(targets[0]); competitors = first.clone(); competitors[gold] = -torch.inf
    other = int(competitors.argmax())
    require(bool(torch.isfinite(ce).all()), 'Finite complete gold+EOS CE')
    return {'gold_eos_token_ids':targets.cpu().tolist(),'per_token_ce_fp32':ce.cpu().tolist(),'mean_ce':float(ce.mean().cpu()),
        'first_gold_token':gold,'gold_rank':1+int((first > first[gold]).sum()),
        'rank_policy':'One plus strictly greater logits; ties share rank','top1_token':int(first.argmax()),
        'maximum_non_gold_token':other,'gold_minus_maximum_non_gold_logit':float((first[gold]-first[other]).cpu())}


def release_unused_mps(runtime, phase):
    """Release unoccupied allocator blocks only; keep the admission guard intact."""
    if str(runtime.device) != 'mps':
        return {'phase':phase,'enabled':False,'reason':'not_mps','seconds':0.0}
    import torch
    import psutil
    def snapshot():
        return {'available_bytes':int(psutil.virtual_memory().available),
            'mps_current_allocated_bytes':int(torch.mps.current_allocated_memory()),
            'mps_driver_allocated_bytes':int(torch.mps.driver_allocated_memory()),
            'context':runtime.context_status()}
    began=time.perf_counter();torch.mps.synchronize()
    before=snapshot();torch.mps.empty_cache();torch.mps.synchronize();after=snapshot()
    return {'phase':phase,'enabled':True,'before':before,'after':after,
        'seconds':time.perf_counter()-began,'scope':'unused cached blocks; not live model/tensors or CPU captures'}


def diagnose_row(runtime, row, cache, *, memory=None):
    import torch
    from luban.evolution import _target
    from verify_cached_training import cached_logits, _ce, _compare, tensor_sha
    from verify_cached_batch_training import _labels, _eos
    from verify_training_operators import TOLERANCES
    memory={} if memory is None else memory
    began=time.perf_counter()
    ids, labels = runtime._sample_tokens(row); _labels(ids,labels,_eos(runtime))
    memory['input_tokens']=int(ids.shape[1])
    memory['before_admission']=release_unused_mps(runtime,'after_CPU_capture_before_tokens')
    prefix = runtime._tokens(json.loads(row['provenance'])['messages'],'off')
    gold = runtime.tokenizer.encode(row['answer'],add_special_tokens=False)+[runtime.tokenizer.eos_token_id]
    n = prefix.shape[1]
    require(torch.equal(prefix,ids[:,:n]) and bool(labels[:,:n].eq(-100).all())
        and labels[0,n:].tolist() == gold, 'Exact inference prefix/all gold+TRAIN EOS supervision')
    with torch.no_grad():
        native = runtime.model(ids,labels=labels,use_cache=False)
        tail, _ = cached_logits(runtime,row,cache,_target(runtime)[1])
        tolerance = TOLERANCES[str(_target(runtime)[1].dtype)]
        logits = _compare(tail,native.logits,tolerance,'logits')
        loss = _compare(_ce(tail,labels),native.loss,tolerance,'loss')
        stats = token_statistics(native.logits,labels)
    require(logits['passed'] and loss['passed']
        and abs(stats['mean_ce']-float(native.loss.float().cpu())) <= tolerance['loss_atol'], 'Native/cache fixed tolerance, no loosening')
    result={'id':row['id'],'sample_sha256':content_sha(row),'input_ids_sha256':tensor_sha(ids),
        'labels_sha256':tensor_sha(labels),'candidate_tensor_sha256':tensor_sha(_target(runtime)[1]),
        'prefix_tokens':n,**stats,'native_cached_logits':logits,'native_cached_mean_ce':loss}
    # All returned statistics are CPU scalars/lists. Drop full-vocabulary live
    # outputs before measuring/releasing allocator blocks for the next row.
    del native,tail,ids,labels,prefix
    memory['after_statistics']=release_unused_mps(runtime,'after_statistics_live_outputs_dropped')
    memory['diagnose_seconds']=time.perf_counter()-began
    return result


def execute(path, approved_sha):
    require(sha(safe(path)) == approved_sha, 'Root approved exact protocol SHA')
    plan = read(safe(path)); require(plan.get('protocol') == PROTOCOL and plan.get('sealed') is True
        and plan.get('state') == 'sealed_waiting_explicit_MODEL_GO', 'Separately sealed explicit GO')
    began = time.perf_counter(); deadline = began+WALL_SECONDS; validate_bound(plan)
    output = safe(plan['output']); require(not output.exists(), 'Once output')
    write_new(safe(path).parent/'execution-started.json',{'protocol_sha256':approved_sha,'output':str(output)})
    output.mkdir(mode=0o700)
    report = {'protocol':PROTOCOL,'protocol_sha256':approved_sha,'pid':os.getpid(),'passed':False,'model_started':False,
        'optimizer_steps':0,'gradient_calls':0,'generation_calls':0,'teacher_calls':0,'publication':False,'product_accepted':False,
        'source_before':plan['source_sha256'],'production_before':plan['production_snapshot'],'states':[]}
    write_new(output/'run-started.json',report)
    runtime = parameter = original = flags = modes = rng = mps_rng = hooks = backward = pointer_before = caches = payload = resources = None
    try:
        data = closed(plan['receipt_path'])
        import torch
        from luban.runtime import Runtime
        from luban.settings import Settings
        from luban.memory import Memory
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target
        from verify_cached_training import capture_tail, tensor_sha
        from verify_automatic_batches import measured
        with tempfile.TemporaryDirectory(prefix='luban-capacity-diagnostic-') as home:
            settings=Settings(home);settings.update({'storage_root':str(output/'private'),'model_path':plan['model_path'],'temperature':0,'local_thinking':'off'})
            runtime=Runtime(settings,Memory(output/'private'),research=True)
            require(str(runtime.device)==plan['device'] and str(runtime.dtype)==plan['dtype'],'Fixed native dtype/device')
            runtime._model_process_lock=ModelProcessLock(IDENTITY,runtime.device)
            with measured(runtime) as resources:
                runtime.load(paging=False);report['model_started']=True;parameter=_target(runtime)[1];original=parameter.detach().clone()
                require(runtime.identity==IDENTITY and runtime.active_core is None and runtime.branch.expert is None
                    and tensor_sha(original)==ORIGINAL,'Official originalNone1706 before candidate')
                flags=[(p,p.requires_grad,p.grad) for p in runtime.model.parameters()];modes=[(m,m.training) for m in runtime.model.modules()]
                rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone() if runtime.device=='mps' else None
                hooks=[(m,len(m._forward_hooks),len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backward=[(p,len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                pointer=Path(runtime.store)/'active-core.json';pointer_before=pointer.read_bytes() if pointer.exists() else None
                torch.manual_seed(42);runtime.model.eval().requires_grad_(False);parameter.requires_grad_(True)
                # Existing cache graph guard requires this flag; all forwards remain no_grad.
                artifact=data['artifact'];payload=torch.load(safe(artifact['path']),weights_only=True,mmap=True,map_location='cpu')
                candidate=payload['candidate']
                require(type(candidate) is torch.Tensor and candidate.dtype==parameter.dtype and candidate.shape==parameter.shape
                    and torch.isfinite(candidate).all().item() and tensor_sha(candidate)==artifact['candidate_tensor_sha256']
                    and payload['identity']==IDENTITY and payload['original_core'] is None
                    and payload['source_sha256']==data['plan']['source_sha256']
                    and payload['progress']['ce_events']==plan['selected']['ce_events']
                    and payload['progress']['capacity_n']==50 and payload['progress']['check_policy']==data['plan']['check_policy']
                    and payload['progress']['optimizer_updates']==artifact['optimizer_updates'],'Only exact last saved native candidate, never master continuation')
                caches={}
                for state in ('originalNone','last_complete_candidate'):
                    with torch.no_grad():parameter.copy_(original if state=='originalNone' else candidate)
                    actual=tensor_sha(parameter);record={'state':state,'native_weight_sha256':actual,'rows':[],'row_memory':[]};report['states'].append(record)
                    for row in data['rows']:
                        require(time.perf_counter()<deadline,'Complete900 diagnostic cap')
                        row_started=time.perf_counter();memory={'id':row['id']};record['row_memory'].append(memory)
                        capture_started=time.perf_counter()
                        if state=='originalNone':caches[row['id']]=capture_tail(runtime,row)
                        memory['capture_seconds']=time.perf_counter()-capture_started
                        diagnose_started=time.perf_counter()
                        try:
                            record['rows'].append(diagnose_row(runtime,row,caches[row['id']],memory=memory))
                        finally:
                            memory['diagnose_seconds']=time.perf_counter()-diagnose_started
                            memory['row_seconds']=time.perf_counter()-row_started
                        write_path=output/'diagnostic.json';write_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                report['passed']=True
    except BaseException as error:report['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        if isinstance(resources,dict):report['resources']=resources
        if runtime is not None:
            try:
                import torch
                from verify_cached_training import tensor_sha
                if original is not None:
                    with torch.no_grad():parameter.copy_(original)
                if flags is not None:
                    for p,f,g in flags:p.requires_grad_(f);p.grad=g
                    for m,v in modes:m.training=v
                    torch.set_rng_state(rng)
                    if mps_rng is not None:torch.mps.set_rng_state(mps_rng)
                    report['restoration']={'original_tensor':tensor_sha(parameter)==ORIGINAL,
                        'flags_grads':all(p.requires_grad==f and p.grad is g for p,f,g in flags),'modes':all(m.training==v for m,v in modes),
                        'rng':torch.equal(rng,torch.get_rng_state()),'mps_rng':mps_rng is None or torch.equal(mps_rng,torch.mps.get_rng_state()),
                        'core_branch':runtime.active_core is None and runtime.branch.expert is None,
                        'forward_hooks':all(len(m._forward_hooks)==a and len(m._forward_pre_hooks)==b for m,a,b in hooks),
                        'backward_hooks':all(len(p._backward_hooks or {})==n for p,n in backward),
                        'pointer':(pointer.read_bytes() if pointer.exists() else None)==pointer_before}
            except BaseException as error:report['restoration_error']={'type':type(error).__name__,'message':str(error)}
            parameter=original=flags=modes=rng=mps_rng=hooks=backward=pointer_before=caches=payload=candidate=p=m=None
            try:
                runtime._release_model()
                from luban.process_lock import ModelProcessLock
                lock=ModelProcessLock(IDENTITY,plan['device']);lock.close()
                report['model_released']=runtime.model is None and runtime._model_process_lock is None;report['actual_os_lock_reacquired_closed']=True
            except BaseException as error:report['release_error']={'type':type(error).__name__,'message':str(error)}
        try:
            from verify_router_coverage_study import model_binding
            report['source_after']={p:sha(p) for p in plan['source_sha256']};report['official_after']=model_binding(plan['model_path'],IDENTITY)
            report['production_after']=production_snapshot(plan['production_root'])
            report['preserved']=report['source_after']==plan['source_sha256'] and report['official_after']==plan['official_model_binding'] and report['production_after']==plan['production_snapshot']
            report['closed_receipt_preserved']=sha(plan['receipt_path'])==plan['receipt_sha256'] and closed(plan['receipt_path'])['summary']==plan['selected']
        except BaseException as error:report['preservation_error']={'type':type(error).__name__,'message':str(error)}
        report['wall_seconds_before_finalwrite']=time.perf_counter()-began
        report['passed']=bool(report['passed'] and not any(k.endswith('error') for k in report) and report.get('preserved') is True
            and report.get('closed_receipt_preserved') is True and set(report.get('restoration',{}))==RESTORATION
            and all(v is True for v in report['restoration'].values()) and report.get('model_released') is True
            and report.get('actual_os_lock_reacquired_closed') is True and report['wall_seconds_before_finalwrite']<WALL_SECONDS)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');wall=time.perf_counter()-began
        if wall>=WALL_SECONDS:
            report['passed']=False;report['error']={'type':'TimeoutError','message':'Full diagnostic finalwrite exceeded900'}
            (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'diagnostic_full_wall_after_first_finalwrite_seconds':wall,'passed':report['passed'],'optimizer_steps':0}),flush=True)
    if report['passed'] is not True:raise RuntimeError('Diagnostic incomplete/rejected; no automatic retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group()
    for name in ('prepare','bind','seal','run'):actions.add_argument('--'+name,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--receipt',type=Path);parser.add_argument('--approved-sha')
    args=parser.parse_args()
    if not any(v is not None for v in vars(args).values()):
        print(json.dumps({'protocol':PROTOCOL,'state':'waiting_closed_N50_last_complete_Root_receipt','torch_imported':False,'model_started':False,'optimizer_steps':0}));return
    if args.run:
        if args.approved_sha is None or args.output is not None or args.receipt is not None:parser.error('Only exact sealed path/SHA for run')
        execute(args.run,args.approved_sha)
    elif args.prepare:
        if args.output is None or args.approved_sha is not None:parser.error('prepare needs fresh output, no implicit run')
        write_new(args.prepare,prepare(args.output,args.receipt))
    elif args.bind or args.seal:
        if args.output is None or args.receipt is not None or args.approved_sha is not None:parser.error('Input plus fresh plan output only')
        plan=read(args.bind or args.seal)
        if args.bind:plan=bind(plan)
        else:validate_bound(plan);plan.update(sealed=True,state='sealed_waiting_explicit_MODEL_GO')
        write_new(args.output,plan)
    else:parser.error('No implicit action')


if __name__ == '__main__':main()
