"""Verify native closed-book identity, then commit the user-authorized base core."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid
import threading
import math
import re

import torch
from safetensors.torch import save_file
from luban.runtime import Runtime
from luban.settings import Settings, private_json, require_storage
from luban.evolution import _target, _read_version
from luban.retention import save_core_retention, provenance, messages, normalized_context
from luban.generation_checks import evaluate_generation
from luban.training_checkpoint import TrainingSession, training_runs
from luban.backups import save_training_backup
from luban.product_inference import start
from train_luban_identity import TARGET, MODEL


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verified_quality(root, output):
    if not re.fullmatch(r'native[1-9][0-9]*', output):
        raise ValueError('Candidate output name must bind actual cumulative steps')
    path=root/(output+'-quality.json')
    if not path.exists() and output=='native450':path=root/'quality.json'
    value=json.loads(path.read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    baseline=json.loads((root/'baseline.json').read_text())
    new={r['id'] for key in ('new_training','new_validation') for r in protocol['groups'][key]}
    old={r['id'] for r in baseline['replay'] if r['passed']}
    if value.get('passed') is not True or not new or not old or value.get('target')!=TARGET:
        raise ValueError('Candidate lacks the complete bound identity/protection proof')
    for key,expected in (('new_first',new),('new_second',new),('old_protection',old)):
        rows=value.get(key)
        if (not isinstance(rows,list) or len(rows)!=len(expected) or {r.get('id') for r in rows}!=expected
            or not all(r.get('passed') is True and r.get('complete') is True
                and r.get('reference_used') is False and r.get('branch_enabled') is False for r in rows)):
            raise ValueError('Candidate proof omitted or failed bound rows: '+key)
    loss,previous=value.get('replay_loss'),baseline.get('replay_loss')
    if (type(loss) not in (int,float) or type(previous) not in (int,float)
            or not math.isfinite(loss) or not math.isfinite(previous) or loss>previous*1.05+.01
            or value.get('baseline_replay_loss')!=previous or value.get('teacher_api_calls')!=0):
        raise ValueError('Candidate replay budget or teacher-call binding differs')
    candidate=root/(output+'-candidate.safetensors')
    if not candidate.exists() and output=='native450':candidate=root/'candidate.safetensors'
    value.update(candidate_file=candidate.name,quality_sha256=sha(path),total_steps=int(output.removeprefix('native')))
    return value


def files(runtime, root, quality, run_id):
    version='core-'+run_id
    groups=copy.deepcopy(json.loads((root/'protocol.json').read_text())['groups'])
    # Publication-only metadata; actual prompts, golds and split assignments stay unchanged.
    for rows in groups.values():
        for row in rows:
            row.setdefault('created',time.time())
            value=provenance(row);value['base_identity']=runtime.identity
            row['provenance']=json.dumps(value,ensure_ascii=False)
    old_correct={r['id'] for r in quality['old_protection'] if r['passed']}
    protected=[]
    for rows in groups.values():
        for row in rows:
            if row['purpose']=='training' and (provenance(row)['evolution_role']=='new' or row['id'] in old_correct):
                item=copy.deepcopy(row);item.update(id='foundation-protected-'+row['id'],purpose='validation',training_allowed=False)
                value=provenance(item);value.update(retention_generation_only=True,evolution_role='replay')
                item['provenance']=json.dumps(value,ensure_ascii=False);protected.append(item)
    if quality.get('identity')!=runtime.identity:
        raise ValueError('Candidate identity differs from the actually loaded official base')
    source=root/quality['candidate_file']
    weight=runtime.store/(version+'.safetensors')
    with source.open('rb') as incoming,weight.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)
    retention=save_core_retention(runtime,version,groups,regression_samples=protected)
    report={'id':run_id,'version':version,'identity':runtime.identity,'previous_version':None,'promoted':True,
        'seed':42,'steps':quality['total_steps'],'completed_steps':quality['total_steps'],
        'training_engine':{'name':'Luban 训练引擎','engine':'luban-training-ce-release-v1'},
        'target':TARGET,'native_identity_verified':False,'teacher_api_calls':0,
        'publication_state':'validated_candidate','gates':{'two_complete_new_checks':True,'old_protection':True,'replay_loss':True},
        'source_quality_sha256':quality['quality_sha256'],'source_protocol_sha256':sha(root/'protocol.json'),
        'scope':'User-authorized foundational persona correction; finite curated checks, not broad or unlimited capability guarantee',
        'parameter':_target(runtime)[0],'trained_parameters':12582912,'added_parameters':0,
        'dtype':'torch.bfloat16','optimizer_dtype':'torch.float32','learning_rate':1e-5,'distillation_weight':5}
    metadata={'version':version,'identity':runtime.identity,'previous_version':None,
        'retention_sha256':retention['file_sha256'],'parameter':report['parameter'],'shape':[2048,6144],
        'dtype':'torch.bfloat16','sha256':sha(weight),'report':'evolution-'+run_id+'.json',
        'format':'cumulative core tensor replacement','trained_parameters':12582912}
    private_json(runtime.store/(version+'.json'),metadata)
    private_json(runtime.store/metadata['report'],report)
    _,checked_weight=_read_version(runtime,version)
    raw_sha=hashlib.sha256(checked_weight.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
    if raw_sha!=quality.get('candidate_sha256'):
        raise ValueError('Candidate safetensors bytes differ from the actual quality-tested tensor')
    metadata['candidate_tensor_sha256']=raw_sha
    private_json(runtime.store/(version+'.json'),metadata)
    return groups,protected,report,metadata


def native_check(root, output):
    quality=verified_quality(root,output)
    preview=root/(output+'-preview');preview.mkdir(exist_ok=False)
    settings=Settings(preview/'home');settings.data.update(model_path=str(MODEL),storage_root=str(preview/'storage'),temperature=0)
    runtime=Runtime(settings,None)
    runtime.load();runtime.close_native();run_id=uuid.uuid4().hex[:12]
    try:
        runtime.close_native()
        groups,protected,report,metadata=files(runtime,root,quality,run_id)
        version=metadata['version'];private_json(runtime.store/'active-core.json',{'version':version,'identity':runtime.identity,'sha256':metadata['sha256']})
        runtime.active_core=version
        engine=start(runtime)
        if (engine is None or engine.loaded.get('core_loaded') is not True
                or engine.loaded.get('core_file_sha256')!=metadata['sha256']
                or engine.loaded.get('core_tensor_sha256')!=metadata['candidate_tensor_sha256']
                or engine.loaded.get('base_identity')!=runtime.identity):
            raise RuntimeError('Product native worker did not load the exact quality-tested candidate')
        eos=runtime.model.generation_config.eos_token_id
        eos=eos if isinstance(eos,(list,tuple)) else [eos]
        raw=[]
        wanted=groups['new_training']+groups['new_validation']
        protected_old=[r for r in groups['replay_training'] if r['id'] in {x['id'] for x in quality['old_protection']}]
        for phase,rows in [('first',wanted),('second',wanted),('old',protected_old)]:
            for row in rows:
                ids=runtime._tokens(messages(row),'off',output_reserve=128)
                result=engine.generate(ids[0].tolist(),max_tokens=128,eos_ids=eos,temperature=0,seed=42,
                    on_token=lambda _:None,stop=threading.Event())
                decoded=runtime.tokenizer.decode(result['tokens'],skip_special_tokens=True)
                answer=decoded.strip()
                complete=bool(result.get('finish_reason')=='stop' and result['tokens'] and result['tokens'][-1] in eos)
                check=provenance(row)['generation_check']
                raw.append({'phase':phase,'id':row['id'],'answer':answer,'tokens':result['tokens'],
                    'raw_completion_token_ids':result['tokens'],'decoded_completion_unstripped':decoded,
                    'completion_tokens':len(result['tokens']),'max_new_tokens':128,'finish_reason':result['finish_reason'],
                    **evaluate_generation(answer,check,complete),'reference_used':False,'branch_enabled':False})
                private_json(root/(output+'-native-identity.json'),{'engine':runtime.inference_identity(),'rows':raw,'passed':False})
            print(json.dumps({'phase':phase,'passed':sum(x['passed'] for x in raw if x['phase']==phase),'total':len(rows)}),flush=True)
        if not all(x['passed'] for x in raw):raise RuntimeError('Native product identity/protection failed; no adoption')
        private_json(root/(output+'-native-identity.json'),{'engine':runtime.inference_identity(),
            'loaded':engine.loaded,'rows':raw,'passed':True,'candidate_file_sha256':metadata['sha256'],
            'candidate_tensor_sha256':metadata['candidate_tensor_sha256'],'quality_sha256':quality['quality_sha256']})
        report.update(native_identity_verified=True,native_identity_sha256=sha(root/(output+'-native-identity.json')))
        private_json(runtime.store/metadata['report'],report)
        receipt={'passed':True,'run_id':run_id,'preview_store':str(runtime.store),'groups':groups,'protected':protected,
            'report':report,'metadata':metadata,'native_engine':runtime.inference_identity(),'output':output,'rows':raw}
        private_json(root/'native-acceptance.json',receipt)
    finally:runtime._release_model()


def adopt(root):
    acceptance=json.loads((root/'native-acceptance.json').read_text())
    native_path=root/(acceptance['output']+'-native-identity.json')
    observed=json.loads(native_path.read_text())
    if (acceptance.get('passed') is not True or observed.get('passed') is not True
            or acceptance['report'].get('native_identity_verified') is not True
            or acceptance['report'].get('native_identity_sha256')!=sha(native_path)
            or observed.get('rows')!=acceptance['rows'] or observed.get('engine')!=acceptance['native_engine']
            or observed.get('candidate_file_sha256')!=acceptance['metadata']['sha256']
            or observed.get('candidate_tensor_sha256')!=acceptance['metadata']['candidate_tensor_sha256']):
        raise ValueError('Adoption is not bound to the completed native candidate proof')
    runtime=Runtime(Settings(),None);runtime.load()
    try:
        report=copy.deepcopy(acceptance['report']);metadata=acceptance['metadata'];version=metadata['version']
        if runtime.active_core not in (None,version):
            raise RuntimeError('Production core changed; preserve it instead of overwriting')
        before_path=root/'production-before.json'
        if before_path.exists():
            before=json.loads(before_path.read_text());backup=before['backup_id'];old_rows=before['samples']
        else:
            backup=save_training_backup(runtime.store,runtime.identity,'learn')
            old_rows=runtime.memory.samples()
            private_json(before_path,{'backup_id':backup,'samples':old_rows,'runs':training_runs(runtime.memory,runtime.identity)})
        names=[version+s for s in ('.safetensors','.json','.retention.json')]+[metadata['report']]
        report.update(publication_state='published',backup_id=backup)
        if runtime.active_core==version:
            existing=json.loads((runtime.store/metadata['report']).read_text())
            if existing!=report:raise ValueError('Existing adopted report differs from this verified release')
            runs=training_runs(runtime.memory,runtime.identity)
            if not any(run['id']==acceptance['run_id'] and run['state']=='committed' for run in runs):
                raise ValueError('Existing pointer has no completed adoption receipt')
        else:
        # Retire stale unadopted candidates; their full artifacts/materials remain in the backup.
            pending_runs=training_runs(runtime.memory,runtime.identity)
            with runtime.memory.lock,runtime.memory.connect() as db:
                for run in pending_runs:
                    if run['id']!=acceptance['run_id'] and run['state'] in {'queued','running','paused'}:
                        run.update(reason='Superseded unadopted candidate: new user-authorized identity foundation/engine; original checkpoint preserved',owner=None)
                        db.execute("UPDATE training_runs SET state='rejected',body=?,updated=? WHERE id=?",(json.dumps(run,ensure_ascii=False),time.time(),run['id']))
            session=TrainingSession(runtime,'evolution',{'steps':report['steps'],'learning_rate':1e-5,
                'distillation_weight':5,'preserve_replay_subspace':True},
                {'groups':acceptance['groups'],'protection':acceptance['protected'],'retention_regressions':[]},run_id=acceptance['run_id'])
            for name in names:
                target=runtime.store/name
                data=(Path(acceptance['preview_store'])/name).read_bytes()
                if name==metadata['report']:continue
                if target.exists():
                    if target.is_symlink() or target.read_bytes()!=data:
                        raise ValueError('Existing adoption artifact differs; no overwrite')
                else:
                    with target.open('xb') as outgoing:outgoing.write(data)
            _read_version(runtime,version)
            private_json(runtime.store/metadata['report'],report)
            session.prepare_commit(report,{'active-core.json':{'version':version,'identity':runtime.identity,'sha256':metadata['sha256']}},names,event_kind='evolution')
            session.finish_commit();runtime.active_core=version
        # The previous persona golds are retained as history and excluded from future training.
        taught_contexts={normalized_context(row):row['answer'] for key in ('new_training','new_validation')
                        for row in acceptance['groups'][key]}
        retired=[row['id'] for row in old_rows if row['approved'] and row['answer']!=TARGET and (
            (normalized_context(row) in taught_contexts and row['answer']!=taught_contexts[normalized_context(row)]) or (
                any(word in row['answer'] for word in ('我是','I am','I\'m'))
                and any(word in row['answer'] for word in ('鲁班','通义千问','Qwen','Luban','luban0'))))]
        with runtime.memory.lock,runtime.memory.connect() as db:
            for identifier in retired:db.execute('UPDATE samples SET approved=0,training_allowed=0 WHERE id=?',(identifier,))
        # New installations/reset use the same validated release core, preserving official files.
        foundation=MODEL/'luban-foundation'
        if foundation.is_symlink():raise ValueError('Foundation directory cannot be a symlink')
        foundation.mkdir(exist_ok=True)
        for name in names:
            source=runtime.store/name;destination=foundation/name
            if destination.exists():
                if destination.is_symlink() or sha(destination)!=sha(source):
                    raise ValueError('Existing foundation artifact differs; no overwrite')
            else:
                with source.open('rb') as incoming,destination.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)
        manifest={'identity':runtime.identity,'version':version,'quality_passed':True,
            'report':metadata['report'],'files':{name:sha(foundation/name) for name in names},'download_url':'https://ai.maihh.net/models/luban0/base.zip'}
        release_path=MODEL/'luban-foundation.json'
        if release_path.exists():
            if release_path.is_symlink() or json.loads(release_path.read_text())!=manifest:
                raise ValueError('Existing foundation manifest differs; no overwrite')
        else:private_json(release_path,manifest)
        private_json(root/'adoption.json',{'passed':True,'version':version,'metadata':metadata,'backup_id':backup,
            'retired_persona_ids':retired,'source_native_acceptance_sha256':sha(root/'native-acceptance.json'),'uploaded':False})
        print(json.dumps({'adopted':version,'foundation':str(MODEL/'luban-foundation.json')},ensure_ascii=False),flush=True)
    finally:runtime._release_model()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','adopt']);p.add_argument('root',type=Path);p.add_argument('--output',default='native900')
    a=p.parse_args();torch.set_num_threads(1)
    require_storage('/Volumes/移动数据库/lubanllm');require_storage('/Volumes/Untitled/lubanllm-backup')
    if a.action=='verify':native_check(a.root,a.output)
    else:adopt(a.root)
