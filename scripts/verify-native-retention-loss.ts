/** Fixed v19 candidate, zero Adam, diagnostic CE only. Never resumes or adopts the rejected journal. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {readFile,writeFile,mkdir,mkdtemp,realpath} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {homedir} from 'node:os';
import {parseArgs} from 'node:util';
import {openNativeEvolutionEngine,nativeEvolutionRuntimeResources,NativeCore450Owner} from '../backend/native-evolution-worker.ts';
import type {NativeEvolutionLease} from '../backend/native-evolution-service.ts';
import {applicationConnection} from './run-native-identity-round.ts';
import {retentionLossRecords} from '../backend/native-evolution-loss-targets.ts';
import {projectKnownOldSpan,replaceDiagnosticCore,diagnosticIdentityGeneration,oldSpanCheckpointInputs,assertPreparedProjection} from './old-span-diagnostic.ts';
import {nativeRetentionLossGuard} from '../backend/native-evolution-quality.ts';
import {coreCheckpointField as field,coreCheckpointMetadata as metadata} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {withEvolutionRound,enterEvolutionBranch,assertEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {withObservation} from '../.native-migration/research/native-observation-python-parity-v2/scope.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {openCheckpointRaw} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
import {durable,sha} from '../.native-migration/research/native-full-training/io.ts';

const {values}=parseArgs({args:process.argv.slice(2),strict:true,options:{output:{type:'string'},home:{type:'string',default:join(homedir(),'Library/Application Support/Luban')},'check-only':{type:'boolean'},'old-span-projection':{type:'boolean'},'prepared-projection':{type:'string'}}});
assert(values.output,'--output must name a new diagnostic directory');
const output=resolve(values.output),home=resolve(values.home!);
const evidence=JSON.parse(await readFile(resolve('research/native_evolution_final_policy_20261017.json'),'utf8'));
assert.equal(evidence.summary.run_id,'84dbd6c8f41a');
const directory=evidence.reader_directory,manifest=JSON.parse(await readFile(join(directory,'manifest.json'),'utf8'));
assert.equal(manifest.checkpoint_sha256,evidence.summary.checkpoint.sha256);
const tree=JSON.parse(await readFile(join(directory,'checkpoint-tree.json'),'utf8')),payload=field(tree,'payload')!,report=metadata(field(payload,'report'));
const candidate=field(field(payload,'native_state')!,'candidate') as any,reference=field(payload,'learning_reference') as any;
assert.deepEqual(candidate,evidence.candidate);assert.deepEqual(candidate.shape,[2048,6144]);assert.equal(candidate.dtype,'bf16');assert.equal(candidate.bytes,25165824);
assert(manifest.tensors.some((v:any)=>v.file===candidate.file&&v.sha256===candidate.sha256&&v.bytes===candidate.bytes));
const rows=retentionLossRecords([...evidence.records.protection,...evidence.records.retention_regressions],report.partial_baseline);
assert.equal(rows.length,11);assert(rows.every(row=>typeof row.answer==='string'&&row.answer.trim()));
assert(!(values['old-span-projection']&&values['prepared-projection']),'Use one projection source, never recapture a prepared candidate');
const preparedProjection=values['prepared-projection']?JSON.parse(await readFile(values['prepared-projection'],'utf8')):null;
if(values['old-span-projection']||preparedProjection)oldSpanCheckpointInputs(payload);
if(preparedProjection)assertPreparedProjection(preparedProjection,{run:evidence.summary.run_id,candidate:candidate.sha256,master:oldSpanCheckpointInputs(payload)[1].sha256,previous:reference.sha256,identity:report.identity});
const resources=nativeEvolutionRuntimeResources();assert.equal(resources.base.identity,report.identity);assert.equal(resources.base.original_down_sha256,reference.sha256);
if(values['check-only']){console.log(JSON.stringify({ready:true,model_loaded:false,GPU:false,training_steps:0,old_span_projection:!!values['old-span-projection'],prepared_projection:preparedProjection?.core??null,projection_helper:values['old-span-projection']?resolve('build/old-span-diagnostic/project-old-span'):null,rows:rows.length,candidate:join(directory,candidate.file),output,application_lease_required:true,product_admitted:false}));process.exit(0);}
await mkdir(output,{mode:0o700});
const connection=await applicationConnection(home),leaseID=randomUUID(),controller=new AbortController();
const stop=()=>controller.abort(new Error('Diagnostic cancelled'));process.on('SIGINT',stop);process.on('SIGTERM',stop);
const command=async(path:string,body:Record<string,unknown>={})=>{const response=await fetch(connection.url+path,{method:'POST',headers:{authorization:'Bearer '+connection.token,'content-type':'application/json'},body:JSON.stringify(body),redirect:'manual',proxy:null} as RequestInit & {proxy:null});const value:any=await response.json();assert(response.ok,typeof value?.detail==='string'?value.detail:'Application lease command failed: '+path);return value;};
const began=performance.now(),stages:any[]=[];
const event=(phase:string)=>{const value={phase,seconds:(performance.now()-began)/1000};stages.push(value);console.log(JSON.stringify(value));};
async function closeUntilConfirmed(close:()=>Promise<void>){let last='';for(;;){try{await close();return;}catch(error){const message=String(error).replaceAll(connection.token,'[private-token]');if(message!==last){console.error('Retaining real diagnostic owner: '+message);last=message;}await Bun.sleep(1000);}}}
const pointers=async(store:string)=>Promise.all(['active-core.json','active.json','active-router.json'].map(async name=>{try{return await readFile(join(store,name));}catch(error:any){if(error.code==='ENOENT')return null;throw error;}}));
let pointersBefore:(Buffer|null)[]|undefined,pointersUnchanged=false;
let opened:Awaited<ReturnType<typeof openNativeEvolutionEngine>>|undefined,lease:NativeEvolutionLease|undefined,work:string|undefined,result:any,error:unknown,engineClosed=false,leaseReleased=false;
try{
 event('acquiring_application_lease');lease=await command('/internal/native-evolution/acquire',{lease_id:leaseID});
 assert(lease&&lease.lease_id===leaseID&&lease.identity===report.identity);pointersBefore=await pointers(lease.store);controller.signal.throwIfAborted();
 const parent=join(home,'native-evolution-work');await mkdir(parent,{recursive:true,mode:0o700});work=await mkdtemp(join(parent,'loss-diagnostic-'));
 opened=await openNativeEvolutionEngine({home,work,model:await realpath(lease.model),root:lease.root,store:lease.store,identity:lease.identity});
 const {engine}=opened,owners=new NativeCore450Owner();
 result=await withEvolutionRound(engine,()=>engine.withSourceQuiescent(async(guard,control)=>{
  const peer=borrowNativeEnginePeer(engine,guard),worker=peer.worker;assert.equal(peer.source.runtime.core_label??null,null);
  const raw=await openCheckpointRaw(join(directory,candidate.file),candidate.bytes),held={assertUnchanged:()=>unchangedNativeFile(join(directory,candidate.file),raw.handle,raw.stat),close:()=>raw.handle.close()};owners.own(held,()=>held.close());
  const originalOwners={source:peer.source,core:peer.core,expert:peer.previous,router:peer.routerOwner};
  const unchanged=()=>{for(const[key,value]of Object.entries(originalOwners))assert.equal(key==='expert'?peer.previous:key==='router'?peer.routerOwner:(peer as any)[key],value);};
  let candidateSelected=false,originalDescriptor:Awaited<ReturnType<typeof writePeerDescriptor>>|undefined;
  const swap=async(descriptor:{path:string;sha256:string},name:string,expected:string,selected:boolean)=>{await worker.stageAcceptedCoreSource({...descriptor,name},owners);const ack=await worker.applyAcceptedCoreSource(owners);assert.equal(ack.actual_core_tensor_sha256,expected);assert.equal(ack.pid,worker.pid);assert.equal(ack.core_version,selected?'core-'+expected.slice(0,12):null);await worker.confirmAcceptedCoreSource(true,()=>{candidateSelected=selected;});unchanged();return{pid:ack.pid,core_sha256:ack.actual_core_tensor_sha256,core_version:ack.core_version};};
  try{
   await enterEvolutionBranch(engine,worker);const preparedRows:any[]=[];
   for(const row of rows){controller.signal.throwIfAborted();const prepared=await control.prepareTrainingSample(row as any);assert(prepared.native_labels_compatible&&prepared.ids.length<=512&&prepared.labels.length===prepared.ids.length);
    const ids=new Uint8Array(new BigInt64Array(prepared.ids.map(BigInt)).buffer),labels=new Uint8Array(new BigInt64Array(prepared.labels.map(BigInt)).buffer),id='retention-'+preparedRows.length;
    const labelsPath=join(work!,id+'-labels.raw');await durable(labelsPath,labels);
    const descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:id,work:worker.options.directory,prompt_ids:prepared.ids,messages_sha256:sha(ids),max_tokens:1,raw_steps_limit:0,core_version:null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
    const replacement:Record<string,string>={core_version:'core-'+candidate.sha256.slice(0,12),core_path:join(directory,candidate.file),core_file_sha256:candidate.sha256,core_tensor_sha256:candidate.sha256,core_tensor_key:resources.base.core_parameter,core_offset:'0',core_bytes:String(candidate.bytes)};
    const seen=new Set<string>(),text=(await readFile(descriptor.path,'utf8')).split('\n').map(line=>{const fields=line.split('\t');if(fields[0]==='config'&&Object.hasOwn(replacement,fields[1]!)){assert.equal(fields.length,3);assert(!seen.has(fields[1]!));seen.add(fields[1]!);return 'config\t'+fields[1]+'\t'+replacement[fields[1]!];}return line;}).join('\n');assert.equal(seen.size,7);
    const current={path:join(work!,id+'-candidate.tsv'),sha256:sha(text)};await durable(current.path,text);preparedRows.push({row,ids:prepared.ids,labels:prepared.labels,original:descriptor,current,labelFile:{path:labelsPath,sha256:sha(labels)}});
   }
   originalDescriptor=preparedRows[0].original;await worker.retainCoreBackbone({...originalDescriptor!,core_tensor_sha256:reference.sha256});
   const cap={research_only:true};const observation=async<T>(action:()=>Promise<T>)=>withObservation({engine,guard,worker,cap,value:cap,owners:[peer.owner,peer.source,peer.build,peer.core,peer.previous,held].filter(Boolean),fast:async()=>{guard.assertQuiescent();worker.assertLive();assertEvolutionBranch(engine,worker);}},action);
   const observe=async(mode:'original'|'current'|'projected')=>withObservation({engine,guard,worker,cap,value:cap,owners:[peer.owner,peer.source,peer.build,peer.core,peer.previous,held].filter(Boolean),fast:async()=>{guard.assertQuiescent();worker.assertLive();assertEvolutionBranch(engine,worker);}},async()=>{const losses=[];for(const p of preparedRows){controller.signal.throwIfAborted();const value=await worker.lossDescriptor({...p[mode],labels:p.labelFile,name:mode+'-'+losses.length,signal:controller.signal});assert(Number.isFinite(value.loss)&&value.loss>=0);losses.push({id:p.row.id,loss:value.loss});}return losses;});
   event('original_ce');const baseline=await observe('original');
   let projection:any=preparedProjection,projected:any,identity:any,identityTrainingPreserved:boolean|undefined;
   if(values['old-span-projection']){event('capture_known_old_and_project');projection=await observation(()=>projectKnownOldSpan({worker,owners,preparedRows,originalRows:evidence.records.protection,payload,directory,work:work!,signal:controller.signal,coreSHA:reference.sha256}));
   }
   if(projection){const raw=await openCheckpointRaw(projection.core.path,projection.core.bytes),projectedOwner={assertUnchanged:()=>unchangedNativeFile(projection.core.path,raw.handle,raw.stat),close:()=>raw.handle.close()};owners.own(projectedOwner,()=>projectedOwner.close());
    for(const[index,p]of preparedRows.entries())p.projected=await replaceDiagnosticCore(p.original,join(work!,'retention-'+index+'-projected.tsv'),projection.core,resources.base.core_parameter);
   }
   event('candidate_source');const candidateACK=await swap(preparedRows[0].current,'diagnostic-candidate',candidate.sha256,true);
   event('candidate_ce');const current=await observe('current');
   if(projection){event('projected_source');await swap(preparedRows[0].projected,'diagnostic-projected',projection.core.sha256,true);event('projected_ce');projected=await observe('projected');
    event('projected_identity_train');identity=await observation(()=>diagnosticIdentityGeneration({control,peer,guard,core:projection.core,parameter:resources.base.core_parameter,records:evidence.records.groups.new_training,scope:'training',work:work!,signal:controller.signal}));
    identityTrainingPreserved=identity.every((r:any)=>r.passed&&r.complete);
    if(identityTrainingPreserved&&nativeRetentionLossGuard(baseline,projected).passed){event('projected_identity_valid_once');identity.push(...await observation(()=>diagnosticIdentityGeneration({control,peer,guard,core:projection.core,parameter:resources.base.core_parameter,records:evidence.records.groups.new_validation,scope:'validation',work:work!,signal:controller.signal})));}
   }
   event('restore_original');const restored=await swap(originalDescriptor!,'diagnostic-original-restored',reference.sha256,false);await held.assertUnchanged();
   return{baseline,candidate:current,...(projection?{projection,projected,projected_loss_guard:nativeRetentionLossGuard(baseline,projected),projected_identity:identity,projected_identity_training_preserved:identityTrainingPreserved,projected_new_validation_evaluated:identityTrainingPreserved&&nativeRetentionLossGuard(baseline,projected).passed,projected_identity_scope:'20 TRAIN first; evaluate 10 NEW VALID once only if all TRAIN preserved and strict retention CE passed; NEW VALID never enters candidate fitting'}:{}),loss_guard:nativeRetentionLossGuard(baseline,current),candidateACK,restored,targets:rows.map(row=>({id:row.id,answer:row.answer})),same_prepared_inputs:true,token_lengths:preparedRows.map(p=>({id:p.row.id,input:p.ids.length,labeled:p.labels.filter((v:number)=>v!==-100).length}))};
  }finally{
   await closeUntilConfirmed(async()=>{await worker.settleAcceptedCoreSource();if(candidateSelected){assert(originalDescriptor);await swap(originalDescriptor,'diagnostic-failure-restore',reference.sha256,false);}});
   await closeUntilConfirmed(()=>owners.close());
  }
 }),async(worker,enabled)=>engine.withSourceQuiescent(async guard=>{assert.equal(borrowNativeEnginePeer(engine,guard).worker,worker);if(worker.isBranchEnabled!==enabled)await worker.setBranchEnabled(enabled);}));
}catch(value){error=value;}
finally{
 if(opened){await closeUntilConfirmed(()=>opened!.close());engineClosed=true;}
 try{if(lease&&pointersBefore){const after=await pointers(lease.store);assert(pointersBefore.every((v,i)=>v===null?after[i]===null:after[i]!==null&&v.equals(after[i]!)),'Diagnostic changed product pointer');pointersUnchanged=true;}}catch(value){error=error?new AggregateError([error,value],'Diagnostic and pointer check failed'):value;}
 try{const released=await command('/internal/native-evolution/release',{lease_id:leaseID});assert(released.released||!lease&&!released.was_ready);leaseReleased=true;if(released.was_ready)await command('/api/model/reload');}catch(value){error=error?new AggregateError([error,value],'Diagnostic and lease cleanup failed'):value;}
 process.off('SIGINT',stop);process.off('SIGTERM',stop);
}
const final={schema:'native-retention-loss-diagnostic-v1',passed:!error,source_run:evidence.summary.run_id,identity:report.identity,candidate_sha256:candidate.sha256,previous_sha256:reference.sha256,precision:'torch.bfloat16',seed:42,steps:0,optimizer_steps:0,product_admitted:false,journal_mutations_requested:false,product_pointers_unchanged:pointersUnchanged,stages,work,engine_closed:engineClosed,lease_released:leaseReleased,total_seconds:(performance.now()-began)/1000,...result,...(error?{error:String(error).replaceAll(connection.token,'[private-token]'),stack:error instanceof Error?error.stack?.replaceAll(connection.token,'[private-token]'):undefined}:{})};
await writeFile(join(output,'result.json'),JSON.stringify(final,null,2)+'\n',{flag:'wx',mode:0o600});console.log(JSON.stringify({phase:'finished',passed:final.passed,loss_passed:result?.loss_guard?.passed,steps:0,engine_closed:engineClosed,lease_released:leaseReleased}));if(error)throw error;
