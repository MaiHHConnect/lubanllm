"""One fixed Q4-tail recovery run; final native generation alone decides quality."""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from benchmark_native_decode import IDENTITY, MODEL, save
from benchmark_native_mixed import CORE_KEY
from capture_q4_paired_tails import OUTPUT as CAPTURE, COUNTS, check_frame
from diagnose_q4_task_candidate import read
from luban.process_lock import ModelProcessLock
from probe_q4_paired_tail import tail_forward, tensor_array, write_bf16
from study_q4_column_protection import ROOT, SOURCE
from validate_q4_paired_tail_proxy import (OUTPUT as PROXY, VOCAB, LOGIT_COLUMNS,
    preflight as capture_preflight, logit_summary, summarize)

OUTPUT = ROOT/'paired-q4-tail-recovery-v2'
REGISTRATION = Path(__file__).resolve().parents[1]/'research/q4_paired_tail_recovery_20261017.json'
SCHEMA = 'q4-paired-tail-recovery-v1'
OPTIMIZER = dict(lr=1e-6, betas=(0.9,0.999), eps=1e-6, weight_decay=0)
STEPS, BATCH, SEED = 450, 9, 42


def split_rows(cases, split):
    rows = [i for c in cases if c['split']==split for i in range(c['file_row_start'], c['file_row_end'])]
    if not rows or len(set(rows)) != len(rows):
        raise ValueError('nonempty disjoint split rows required')
    return np.asarray(rows, dtype=np.int64)


def schedule(rows, steps=STEPS, batch=BATCH, seed=SEED):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows) or steps <= 0 or batch <= 0:
        raise ValueError('unique nonempty rows and positive fixed update budget required')
    rng = np.random.default_rng(seed)
    cycles = [rng.permutation(rows) for _ in range((steps*batch+len(rows)-1)//len(rows))]
    return np.concatenate(cycles)[:steps*batch].reshape(steps,batch)


def teacher_kl(logits, teacher):
    logp = F.log_softmax(teacher.float(), dim=-1)
    return (logp.exp()*(logp-F.log_softmax(logits.float(),dim=-1))).sum()


def apply_update(parameter, master, optimizer, accumulated):
    if accumulated.dtype != torch.float32 or accumulated.shape != master.shape or not bool(torch.isfinite(accumulated).all()):
        raise ValueError('finite FP32 accumulated gradient required')
    master.grad = accumulated
    norm = torch.nn.utils.clip_grad_norm_([master], 1.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        if not bool(torch.isfinite(master).all()):
            raise ValueError('nonfinite optimizer state')
        parameter.copy_(master)
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def main():
    if not os.path.ismount('/Volumes/移动数据库') or not torch.backends.mps.is_available():
        raise ValueError('mounted storage and MPS required')
    registration=read(REGISTRATION)
    fixed=registration['fixed_protocol']
    if (fixed['steps'],fixed['batch_rows'],fixed['seed']) != (STEPS,BATCH,SEED) or fixed['optimizer'] != {
            'name':'Adam',**OPTIMIZER,'betas':list(OPTIMIZER['betas']),'clip_gradient_norm':1}:
        raise ValueError('registered fixed optimizer changed')
    _,capture,_,cases=capture_preflight()
    proxy, audit = read(PROXY/'result.json'), read(PROXY/'independent-readback.json')
    if (registration.get('registered_before_run') is not True or registration.get('schema') != SCHEMA
            or proxy.get('completed') is not True or proxy.get('case_mapping') != cases
            or audit.get('audit_completed') is not True or audit.get('audit_passed') is not True
            or audit.get('directory') != str(PROXY)):
        raise ValueError('registered run and audited paired forward required')
    cal, obs = split_rows(cases,'calibration'), split_rows(cases,'observation')
    if (len(cal),len(obs)) != (3897,827) or set(cal)&set(obs):
        raise ValueError('fixed calibration/observation split changed')
    order=schedule(cal)
    OUTPUT.mkdir(exist_ok=False)
    protocol={**registration,'source_receipt':capture['source_receipt'],'case_mapping':cases,
        'capture_directory':str(CAPTURE),'proxy_directory':str(PROXY),'hardware':platform.platform(),
        'torch':torch.__version__,'optimizer':OPTIMIZER,'steps':STEPS,'batch_rows':BATCH,'seed':SEED,
        'calibration_rows':cal.tolist(),'observation_rows':obs.tolist(),
        'timing_scope':'after metadata preflight; includes lock, tensor load, training, final evaluation and save'}
    save(OUTPUT/'protocol.json',protocol); np.save(OUTPUT/'schedule.npy',order)
    started=time.perf_counter(); completed=0; lock=None
    try:
        lock=ModelProcessLock(IDENTITY,'mps'); torch.set_num_threads(4); torch.manual_seed(SEED)
        torch.mps.set_per_process_memory_fraction(0.5)
        index=read(Path(MODEL)/'model.safetensors.index.json')['weight_map']
        keys={'weight':'model.language_model.layers.23.mlp.down_proj.weight',
              'norm':'model.language_model.norm.weight','head':'model.language_model.embed_tokens.weight'}
        tensors={}
        for name,key in keys.items():
            with safe_open(Path(MODEL)/index[key],framework='pt',device='cpu') as stream:
                value=stream.get_tensor(key)
                if value.dtype != torch.bfloat16 or not all(bool(torch.isfinite(p).all()) for p in value.split(4096)):
                    raise ValueError('finite original BF16 tensors required')
                tensors[name]=value.to('mps').requires_grad_(False)
        del value
        if [tuple(tensors[n].shape) for n in keys] != [(2048,6144),(2048,),(VOCAB,2048)]:
            raise ValueError('original tail shapes changed')
        weight=tensors['weight']; original=weight.detach().cpu().clone()
        master=torch.nn.Parameter(weight.detach().float().clone()); weight.requires_grad_(True)
        optimizer=torch.optim.Adam([master],**OPTIMIZER)
        frozen_versions={n:tensors[n]._version for n in ('norm','head')}
        frames=[]
        for ci,c in enumerate(cases):
            for step in range(len(c['targets'])):
                q=capture['reports']['q4']['cases'][ci]; t=capture['reports']['bf16']['cases'][ci]
                frames.append((q,t,step))
        def inputs(row):
            q,t,step=frames[row]
            qf=check_frame(q['events'][step],CAPTURE/'q4/traces',q['request'],step)
            tf=check_frame(t['events'][step],CAPTURE/'bf16/traces',t['request'],step)
            h=torch.tensor(qf['tail'][:2048],device='mps',dtype=torch.bfloat16).reshape(1,-1)
            x=torch.tensor(qf['tail'][2048:8192],device='mps',dtype=torch.bfloat16).reshape(1,-1)
            teacher=torch.tensor(tf['logits'],device='mps',dtype=torch.float32).reshape(1,-1)
            return h,x,teacher,tf['logits']
        # Control the differentiable forward before the first update; no extra full-model run.
        h,x,teacher,_=inputs(int(order[0,0]))
        with torch.no_grad(): _,plain=tail_forward(x,h,**tensors)
        _,grad_forward=tail_forward(x,h,**tensors)
        if not torch.equal(plain,grad_forward.detach()):
            raise ValueError('autograd changes forward')
        del plain,grad_forward,h,x,teacher
        with (OUTPUT/'updates.jsonl').open('x') as log:
            for batch in order:
                accumulated=torch.zeros_like(master); losses=[]
                for row in batch:
                    h,x,teacher,_=inputs(int(row)); weight.grad=None
                    _,logits=tail_forward(x,h,**tensors)
                    loss=teacher_kl(logits,teacher)/BATCH
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('nonfinite KL')
                    loss.backward()
                    if weight.grad is None or not bool(torch.isfinite(weight.grad).all()):
                        raise ValueError('missing/nonfinite tail gradient')
                    accumulated.add_(weight.grad.float()); losses.append(float(loss.detach()))
                    del logits,loss,h,x,teacher
                weight.grad=None
                norm=apply_update(weight,master,optimizer,accumulated)
                completed+=1
                log.write(json.dumps({'step':completed,'rows':batch.tolist(),'mean_kl':sum(losses),'gradient_norm':norm})+'\n')
                log.flush()
                if completed%25==0 or completed==1:
                    log.flush(); progress={'completed':False,'steps_completed':completed,'mean_kl':sum(losses),
                        'elapsed_seconds':time.perf_counter()-started,
                        'mps_allocated_bytes':torch.mps.current_allocated_memory(),
                        'mps_driver_bytes':torch.mps.driver_allocated_memory(),
                        'python_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    save(OUTPUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        torch.mps.synchronize(); training_seconds=time.perf_counter()-started
        core=weight.detach().cpu().contiguous()
        changed=int(torch.count_nonzero(core != original)); max_change=float((core.float()-original.float()).abs().max())
        if not changed or not torch.equal(core,master.detach().cpu().to(torch.bfloat16)):
            raise ValueError('missing update or final BF16 rounding mismatch')
        core_path=OUTPUT/'core.safetensors';save_file({CORE_KEY:core},str(core_path))
        core_sha=hashlib.sha256(core_path.read_bytes()).hexdigest() # Required native core lifecycle binding, once.
        torch.save({'master':master.detach().cpu(),'optimizer':optimizer.state_dict(),'steps':completed},OUTPUT/'optimizer-final.pt')
        weight.requires_grad_(False)
        total=sum(COUNTS.values()); metrics=np.empty((total,2,len(LOGIT_COLUMNS)),np.float64)
        old=np.memmap(PROXY/'q4-logits.bf16',dtype='<u2',mode='r',shape=(total,VOCAB))
        final_proxy=[]
        with torch.no_grad(), (OUTPUT/'final-logits.bf16').open('wb') as lf:
            for row in range(total):
                h,x,teacher,teacher_np=inputs(row)
                down,logits=tail_forward(x,h,**tensors)
                before=(old[row].astype(np.uint32)<<16).view(np.float32)
                metrics[row,0]=logit_summary(teacher_np,before)
                metrics[row,1]=logit_summary(teacher_np,tensor_array(logits))
                write_bf16(lf,logits)
                if row<2:
                    dp,lp=OUTPUT/f'final-cal00-step{row}.down.bf16',OUTPUT/f'final-cal00-step{row}.logits.bf16'
                    write_bf16(dp,down);write_bf16(lp,logits)
                    final_proxy.append({'step':row,'down_path':str(dp),'logits_path':str(lp)})
                if (row+1)%512==0:
                    print(json.dumps({'evaluation_rows':row+1,'total':total,'elapsed_seconds':time.perf_counter()-started}),flush=True)
        np.save(OUTPUT/'metrics.npy',metrics)
        if any(tensors[n]._version != v or tensors[n].grad is not None for n,v in frozen_versions.items()):
            raise ValueError('frozen norm/head changed')
        splits={name:{'rows':len(rows),'before':summarize(metrics[rows,0],LOGIT_COLUMNS),
                      'after':summarize(metrics[rows,1],LOGIT_COLUMNS)} for name,rows in [('calibration',cal),('observation',obs)]}
        save(OUTPUT/'result.json',{**protocol,'schema':SCHEMA,'completed':True,'trainingcompleted':True,'originalbase':True,
            'steps_completed':completed,'core_path':str(core_path),'core_file_sha256':core_sha,'source':str(SOURCE),
            'final_proxy':final_proxy,'splits':splits,'changed_bf16_weights':changed,'max_weight_change':max_change,
            'forward_autograd_exact':True,'native_tail_bitexact':False,'frozen_norm_head_unchanged':True,
            'training_seconds':training_seconds,'elapsed_seconds':time.perf_counter()-started,
            'memory':{'python_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'sampled_mps_allocated_bytes':torch.mps.current_allocated_memory(),
                'sampled_mps_driver_bytes':torch.mps.driver_allocated_memory(),'gpu_peak_measured':False},
            'qualified':False,'product_adopted':False,'new_holdout_read':False})
    except BaseException as error:
        save(OUTPUT/'failure.json',{'completed':False,'steps_completed':completed,'error':repr(error),
            'elapsed_seconds':time.perf_counter()-started,'original_source_modified':False})
        raise
    finally:
        if lock is not None: lock.close()


if __name__=='__main__':
    main()
