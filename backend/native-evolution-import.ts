/** Explicit zero-update import of the fixed, independently checked v19 refined candidate. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {mkdir,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {openNativeEvolutionEngine,nativeEvolutionRuntimeResources,nativeEvolutionQ4Resources,NativeCore450Owner,Memory,type NativeEvolutionRuntimeInput} from './native-evolution-worker.ts';
import {withNativeEvolutionMaterial,selectIndependentNativeEvolutionCohort,type NativeEvolutionMaterial} from './native-evolution-cohort.ts';
import {readNativeEvolutionHeldSource} from './native-evolution-held-source.ts';
import {nativeEvolutionTrainingSource} from './native-evolution-training-source.ts';
import {startNativeTrainingSessionV4} from '../.native-migration/backend/native-training-start-v4.ts';
import {nativeTrainingPending} from '../.native-migration/backend/native-training-journal-start.ts';
import {openCheckpointRaw} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
import {readNativeModelTensors} from '../.native-migration/backend/native-model-tensor.ts';
import {withEvolutionRound,enterEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {coreCheckpointField as field} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {core450ControlMetadata} from '../.native-migration/research/native-core450-owner-controller/metadata.ts';
import {validateCore450TrainingHistory} from '../.native-migration/research/native-core450-runner/training-safety.ts';
import {prepareNativeCoreCommit,applyNativeCoreCommit,finishNativeCoreCommit,restoreNativeCoreCommitPointer,finishNativeCoreCommitRejection} from '../.native-migration/research/native-core-product-publication-v1/commit-storage.ts';
import {settleRetainedCoreReload} from './native-evolution-reload.ts';
import {core450CandidateRetention,immutableCore450Artifact} from './native-evolution-artifacts.ts';
import {finishNativeEvolutionQuality,evolutionContentSHA,correctNativeEvolutionIds} from './native-evolution-quality.ts';
import {finishNativeQ4Quality} from './native-evolution-q4-quality.ts';
import {NativeQ4Observer,assertNativeQ4Observation,type NativeQ4Observation} from './native-evolution-q4.ts';
import {uniqueNativeEvolutionRecords,provenance} from './native-evolution-material.ts';
import {encoded} from './canonical-json.ts';
import {ledgerParse,ledgerStringify} from './memory.ts';
import {readNativeEvolutionGraphQualification,assertNativeEvolutionGraphQualification} from './native-evolution-graph-qualification.ts';
import {rejectFailedNativeEvolutionJournal} from './native-evolution-journal.ts';
import {observeNativeImport,closeImportOwner} from './native-evolution-import-observe.ts';

export const REFINED_IDENTITY_SOURCE=Object.freeze({id:'84dbd6c8f41a',steps:450,
 tensor:'ae14867f1aaba5d0b6d8fd49abb551759c063184af53596bd29f574f99f0d6fb',
 file:'bed06ece53a5ee6a4eff0c00161868d0011cc39d16ece246b8bffb3f1a360e64'});
type Input=NativeEvolutionRuntimeInput&{material:NativeEvolutionMaterial;candidateFile:string;candidateRawFile:string;sourceTree:string;sourceManifest:string;
 signal:AbortSignal;onPhase?:(phase:string)=>void;onCleanupError?:(error:unknown)=>void;onEvidence?:(name:string,value:unknown)=>Promise<void>};

/** Source history stays rejected; only original records and actual completed updates are inherited. */
export function validateImportSource(row:any,manifest:any,tree:any,records:any){
 assert(row?.id===REFINED_IDENTITY_SOURCE.id&&row.kind==='evolution'&&row.state==='rejected');
 const body=ledgerParse(row.body),payload=field(tree,'payload')!,saved=core450ControlMetadata(payload);
 assert(body.id===row.id&&body.binding.identity===row.identity&&body.binding.core_version===null&&body.owner==null&&body.commit==null);
 assert(body.checkpoint?.progress?.step===450&&body.checkpoint.sha256===manifest.checkpoint_sha256&&saved.step===450&&saved.id===row.id);
 assert.equal(encoded(body.records.groups),encoded(records.groups),'Import cannot change original TRAIN or development VALID');
 assert.equal(encoded(body.records.protection),encoded(records.protection));
 assert.equal(records.retention_regressions.length,body.records.retention_regressions.length);
 for(let i=0;i<records.retention_regressions.length;i++){
  const revised=structuredClone(records.retention_regressions[i]),old=body.records.retention_regressions[i],p=provenance(revised),prior=provenance(old),revision=p.generation_check_revision;
  assert(revised.training_allowed===false&&revised.purpose==='validation'&&revision?.policy==='identity-boundary-punctuation-v1'&&revision.training===false);
  assert.equal(encoded(revision.previous_check),encoded(prior.generation_check));
  const{expected,type,...rest}=prior.generation_check;assert(type==='exact');
  assert.equal(encoded(p.generation_check),encoded({...rest,type:'semantic',validator:'one_of',accepted:[expected,expected+'。']}));
  delete p.generation_check_revision;p.generation_check=prior.generation_check;
  revised.provenance=typeof old.provenance==='string'?JSON.stringify(p):p;
  assert.equal(encoded({...revised,provenance:p}),encoded({...old,provenance:prior}),'Import can revise only readonly punctuation scoring');
 }
 // v19 FINAL transport overwrote only the training policy label; never rewrite its original file.
 const history={...saved,report:{...saved.report,partial_policy:'closed-book-growth-450-v1'}};
 assert(['closed-book-growth-v1','closed-book-growth-450-v1'].includes(saved.report.partial_policy));
 const safety=validateCore450TrainingHistory(history,body.records.groups);
 assert(safety.complete_training_safety&&safety.completed_steps===450&&!safety.partial_safety_failed);
 const candidate=field(field(payload,'native_state')!,'candidate') as any,reference=field(payload,'learning_reference') as any;
 assert(candidate?.kind==='tensor'&&reference?.kind==='tensor'&&candidate.dtype==='bf16'&&reference.dtype==='bf16');
 assert(manifest.tensors.some((item:any)=>item.file===candidate.file&&item.sha256===candidate.sha256&&item.bytes===candidate.bytes));
 return{body,safety,checkpoint_sha256:body.checkpoint.sha256,previous_sha256:reference.sha256,source_candidate_sha256:candidate.sha256};
}

/** Same durable intent / real source ACK / rollback ordering as ordinary publication. */
export async function commitImportedCore(r:any,control:any,owners:NativeCore450Owner,report:any,artifacts:string[],pointer:any,messages:any,assertReady:()=>Promise<void>,onCleanupError:(error:unknown)=>void){
 let before:any,prepared:any,intent=false,committed=false,rollbackSelected=false,recovery:any,receipt:any,failure:unknown;
 try{
  await assertReady();prepared=await control.prepareRouteTokens(messages,{signal:owners.signal});before=await control.prepareRetainedCoreReload(owners,prepared);
  await assertReady();prepareNativeCoreCommit(r,{report,artifacts,pointer});intent=true;applyNativeCoreCommit(r);
  const actual=await control.reloadRetainedCoreSource(before,prepared,owners,r.expected.id);finishNativeCoreCommit(r,actual);committed=true;
  return{state:'committed',actual_reload_confirmed:true};
 }catch(error){failure=error;throw error;}
 finally{if(!committed&&before&&prepared)await closeImportOwner(async()=>{
   const changed=await settleRetainedCoreReload(before);recovery??=changed?await control.prepareRetainedCoreReload(owners,prepared):before;
   if(intent&&!rollbackSelected){restoreNativeCoreCommitPointer(r,String(failure));rollbackSelected=true;}
   receipt??=await control.reloadRetainedCoreSource(recovery,prepared,owners,r.expected.id);
   if(intent)finishNativeCoreCommitRejection(r,receipt);
  },onCleanupError);}
}

export async function importNativeIdentityCandidate(input:Input){
 const resources=nativeEvolutionRuntimeResources(),base=resources.base,id=randomUUID().replaceAll('-','').slice(0,12),version='core-'+id;
 const phase=input.onPhase??(()=>{}),cleanupError=input.onCleanupError??(error=>console.error(String(error))),evidence=input.onEvidence??(async()=>{});
 assert.equal(input.identity,base.identity);assert(input.material);
 const material=structuredClone(input.material),manifest=JSON.parse(await readFile(input.sourceManifest,'utf8')),tree=JSON.parse(await readFile(input.sourceTree,'utf8'));
 let opened:Awaited<ReturnType<typeof openNativeEvolutionEngine>>|undefined,observer:NativeQ4Observer|undefined;
 let raw:Awaited<ReturnType<typeof openCheckpointRaw>>|undefined;
 let journal:any,records:any,source:ReturnType<typeof validateImportSource>|undefined,bf16:any,report:any,adopted=false;
 const held=await readNativeModelTensors({path:input.candidateFile,file_sha256:REFINED_IDENTITY_SOURCE.file,tensors:[{tensor_key:base.core_parameter,dtype:'BF16',shape:base.core_shape}]});
 assert.equal(held.tensors[base.core_parameter]!.sha256,REFINED_IDENTITY_SOURCE.tensor);
 const closeEngine=async()=>{if(opened){await closeImportOwner(()=>opened!.close(),cleanupError);opened=undefined;}};
 const scope=async<T>(body:(guard:any,control:any,owners:NativeCore450Owner)=>Promise<T>)=>{
  assert(opened);const{engine}=opened,owners=new NativeCore450Owner();
  return withEvolutionRound(engine,()=>engine.withSourceQuiescent(async(guard,control)=>{
   try{return await body(guard,control,owners);}finally{await closeImportOwner(()=>owners.close(),cleanupError);}
  }),async(worker,enabled)=>engine.withSourceQuiescent(async guard=>{assert.equal(borrowNativeEnginePeer(engine,guard).worker,worker);if(worker.isBranchEnabled!==enabled)await worker.setBranchEnabled(enabled);}));
 };
 try{
  const tensor=held.tensors[base.core_parameter]!;raw=await openCheckpointRaw(input.candidateRawFile,tensor.bytes);assert.equal(raw.stat.size,BigInt(tensor.bytes));
  const candidateRaw={path:input.candidateRawFile,bytes:tensor.bytes,sha256:tensor.sha256,assertUnchanged:()=>unchangedNativeFile(input.candidateRawFile,raw!.handle,raw!.stat)};
  const preview=new Memory(input.root);preview.model_identity=input.identity;
  try{preview.connect(db=>nativeTrainingPending(db,input.identity,id));}finally{preview.close();}
  await mkdir(join(input.work,'bf16'),{mode:0o700});phase('open_bf16_source');opened=await openNativeEvolutionEngine({...input,work:join(input.work,'bf16')});
  await withNativeEvolutionMaterial(opened.memory,material,()=>scope(async(guard,control,owners)=>{
   const{engine,memory}=opened!,cohort=selectIndependentNativeEvolutionCohort(memory,input.store,base);records=cohort.records;
   source=validateImportSource(memory.connect(db=>db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(REFINED_IDENTITY_SOURCE.id,input.identity)),manifest,tree,records);
   assert.equal(source.previous_sha256,base.original_down_sha256);assert.equal(cohort.core,null);
   const heldSource=await readNativeEvolutionHeldSource(engine,guard);await enterEvolutionBranch(engine,heldSource.peer.worker);
   const trainingSource=nativeEvolutionTrainingSource(engine,guard,cohort);
   const session=await startNativeTrainingSessionV4({memory,store:input.store,kind:'evolution',scope:'product',id,identity:input.identity,
    core_version:null,expert_id:trainingSource.expert_id,evolutionSource:trainingSource,device:'mps',dtype:'torch.bfloat16',
    arguments:{steps:450,partial_policy:'closed-book-growth-v1',operation:'validated-core-import',source_training_steps:450,current_updates:0,source_run:REFINED_IDENTITY_SOURCE.id,
     source_checkpoint_sha256:source.checkpoint_sha256,candidate_tensor_sha256:REFINED_IDENTITY_SOURCE.tensor,evaluation_revision:'identity-boundary-punctuation-v1'},
    records,guard,release:resources.release,signal:input.signal,assertSource:heldSource.assertCurrent});
   owners.own(session,()=>session.close());journal={id,binding:session.binding};await session.running();
   const graph=readNativeEvolutionGraphQualification({identity:input.identity,graph_binary_sha256:heldSource.peer.build.binary_sha256,training_binary_sha256:resources.role('trainer').sha256});
   assertNativeEvolutionGraphQualification(graph,{...graph,checkpoint_sha256:source.checkpoint_sha256,candidate_tensor_sha256:REFINED_IDENTITY_SOURCE.tensor,previous_tensor_sha256:source.previous_sha256});
   bf16=await observeNativeImport({engine,guard,control,owners,candidate:held,raw:candidateRaw,version,records,work:input.work,signal:input.signal,onPhase:phase,onCleanupError:cleanupError});
   await evidence('bf16',bf16);
   const baseline=bf16.baseline[0].map((row:any,i:number)=>row.passed?row:bf16.baseline[1][i]);
   report=finishNativeEvolutionQuality({records,baseline,first:bf16.candidate[0],second:bf16.candidate[1],
    baselineCoreGeneration:records.protection.map((row:any)=>baseline.find((proof:any)=>proof.id===row.id)),
    baselineCoreLoss:bf16.baselineCoreLoss,candidateCoreLoss:bf16.candidateCoreLoss,baselineReplayLoss:bf16.baselineReplayLoss,candidateReplayLoss:bf16.candidateReplayLoss,
    completedSteps:source.safety.completed_steps,trainingSafetyFailed:source.safety.partial_safety_failed,identity:input.identity,
    binding:{core_version:version,core_sha256:held.file_sha256,expert_id:null,expert_sha256:null},previousCoreSHA:source.previous_sha256,
    frozenRecordsSHA:evolutionContentSHA(uniqueNativeEvolutionRecords(records.groups,records.protection,records.retention_regressions))});
   await evidence('bf16-quality',report);assert(report.promoted,'Import BF16 twelve gates failed');
  }));
  await closeEngine();assert(source&&journal&&report);input.signal.throwIfAborted();
  const q4Resources=await nativeEvolutionQ4Resources(),all=uniqueNativeEvolutionRecords(records.groups,records.protection,records.retention_regressions);
  const q4Request={...q4Resources,model:input.model,identity:input.identity,manifestSHA:base.model_manifest_sha256,records:all,work:input.work,signal:input.signal};
  const pairs: (readonly[NativeQ4Observation,NativeQ4Observation])[]=[];
  for(const current of [false,true]){phase(current?'q4_candidate_two_passes':'q4_baseline_two_passes');await held.assertUnchanged();observer=new NativeQ4Observer();
   const pair=await observer.observePair({...q4Request,...(current?{core:{file:held.path,tensor_key:base.core_parameter,sha256:held.file_sha256}}:{})});
   for(const proof of pair){assertNativeQ4Observation(proof,all);assert.equal(proof.core?.sha256??null,current?held.file_sha256:null);}
   pairs.push(pair);await closeImportOwner(()=>observer!.close(),cleanupError);observer=undefined;await evidence(current?'q4-candidate':'q4-baseline',pair);
  }
  const q4={baseline:pairs[0]!,candidate:pairs[1]!};for(const proof of [...q4.baseline,...q4.candidate])assert.equal(encoded(proof.rows.map(row=>row.input_tokens)),encoded(q4.baseline[0].rows.map(row=>row.input_tokens)));
  const published=finishNativeQ4Quality(records,report,q4,{core_version:version,core_sha256:held.file_sha256,expert_id:null,expert_sha256:null},source.previous_sha256);
  const required=new Set([...report.learning_progress.candidate_learned_ids,...records.protection.map((r:any)=>r.id),...records.retention_regressions.map((r:any)=>r.id)]);
  for(const proof of q4.baseline)for(const known of correctNativeEvolutionIds([...proof.rows],all,false))required.add(known);
  for(const proof of q4.candidate){const correct=correctNativeEvolutionIds([...proof.rows],all,false);assert([...required].every(key=>correct.has(key as string)),'Import Q4 lost a required actual capability');}
  report={...published,id,operation:'validated-core-import',source_training_steps:450,current_updates:0,training_updates_executed:0,
   source_training_run:REFINED_IDENTITY_SOURCE.id,source_training_state:'rejected',source_checkpoint_sha256:source.checkpoint_sha256,
   source_candidate_tensor_sha256:source.source_candidate_sha256,converted_candidate_tensor_sha256:REFINED_IDENTITY_SOURCE.tensor,
   conversion:'gptq-bf16-refine-v1',core_retention_loss_targets:bf16.targets,source_training_safety:source.safety,
   measurement_scope:'Fresh BF16 and ordinary Q4 observations; completed_steps describes source training, current_updates is zero'};
  await evidence('import-quality',report);await held.assertUnchanged();input.signal.throwIfAborted();
  await mkdir(join(input.work,'publication'),{mode:0o700});opened=await openNativeEvolutionEngine({...input,work:join(input.work,'publication')});
  await scope(async(guard,control,owners)=>{
   const{engine,memory}=opened!,peer=borrowNativeEnginePeer(engine,guard);await enterEvolutionBranch(engine,peer.worker);
   assert.equal(peer.source.runtime.core_label??null,null);const sourceNow=await readNativeEvolutionHeldSource(engine,guard);
   assert.equal(encoded(sourceNow.pins),encoded(journal.binding.active_files_sha256),'Accepted source changed during import evaluation');
   const mastered=[...new Set<string>([...report.learning_progress.mastered_ids,...report.learning_progress.retained_ids])];
   const retained=core450CandidateRetention(records,mastered,base,version),retentionName=version+'.retention.json',reportName='evolution-'+id+'.json';
   const metadata={version,identity:input.identity,previous_version:null,retention_sha256:evolutionContentBytes(retained.bytes),
    dataset_sha256:report.partial_records_sha256,parameter:base.core_parameter,shape:base.core_shape,dtype:'torch.bfloat16',sha256:held.file_sha256,
    candidate_tensor_sha256:REFINED_IDENTITY_SOURCE.tensor,report:reportName,format:'cumulative core tensor replacement',trained_parameters:12582912};
   report={...report,checkpoint:metadata,retention:{path:join(input.store,retentionName),file_sha256:metadata.retention_sha256,content_sha256:retained.retention.content_sha256},
    accepted_retention_groups:retained.retention.groups,accepted_retention_regressions:retained.retention.regression_validation,
    promoted:true,quality_checked:true,product_admission:true,adopted:true,publication_state:'published'};
   const bytes=await readFile(held.path);await held.assertUnchanged();assert.equal(evolutionContentBytes(bytes),held.file_sha256);
   const artifacts=[[version+'.safetensors',bytes],[version+'.json',Buffer.from(encoded(metadata))],[retentionName,retained.bytes],[reportName,Buffer.from(encoded(report))]] as const;
   for(const[name,data]of artifacts)await immutableCore450Artifact(input.store,name,data);
   await commitImportedCore({memory,store:input.store,base,guard,expected:journal},control,owners,report,artifacts.map(([name])=>name),
    {version,identity:input.identity,sha256:held.file_sha256},[{role:'user',content:'你好'}],async()=>{input.signal.throwIfAborted();await held.assertUnchanged();await sourceNow.assertUnchanged();},cleanupError);
   adopted=true;
  });
  await closeEngine();await evidence('published',report);return{state:'committed',id,version,current_updates:0,source_training_steps:450,adopted:true,pending:report.learning_progress.pending_ids.length>0,report};
 }catch(error){
  if(observer){await closeImportOwner(()=>observer!.close(),cleanupError);observer=undefined;}await closeEngine();
  if(journal&&!adopted){const memory=new Memory(input.root);try{memory.connect(db=>{const row:any=db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(id,input.identity);
    if(row&&['queued','running','paused'].includes(row.state))rejectFailedNativeEvolutionJournal(db,row,String(error),{owned:true,restored:true,cancelled:false});});}finally{memory.close();}}
  if(adopted)throw Object.assign(new Error('Candidate committed, but final recording/cleanup failed',{cause:error}),{adopted:true,id,version});
  throw error;
 }finally{if(observer)await closeImportOwner(()=>observer!.close(),cleanupError);await closeEngine();if(raw)await closeImportOwner(()=>raw!.handle.close(),cleanupError);await closeImportOwner(()=>held.close(),cleanupError);}
}
const evolutionContentBytes=(bytes:Buffer)=>Bun.CryptoHasher.hash('sha256',bytes,'hex');
