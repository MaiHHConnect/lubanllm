/** One accepted-source calibration per new round; never a TRAIN/quality capability. */
import assert from 'node:assert/strict';
import {AsyncLocalStorage} from 'node:async_hooks';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {NativeQ4Observer,assertNativeQ4Observation,nativeQ4CalibrationInputs} from './native-evolution-q4.ts';
import {retentionLossRecords} from './native-evolution-loss-targets.ts';
import {predictorInputs,FINAL_CONVERSION} from './native-evolution-constraint-inputs.ts';
import {exportNativePeerProtectionTail} from './native-evolution-tail.ts';
import {borrowResidentNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {assertNativeExpertScopeGuard} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {withObservation} from '../.native-migration/research/native-observation-python-parity-v2/scope.ts';
import {assertEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {prepareBaselineSamples,baselineGenerationRow} from '../.native-migration/research/native-ordinary-baseline/rows.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {durable,sha} from '../.native-migration/research/native-full-training/io.ts';
import {openCheckpointRaw} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
import {encoded} from './canonical-json.ts';

type Capture=Awaited<ReturnType<NativeQ4Observer['captureTraining']>>;
const scope=new AsyncLocalStorage<{capture:Capture;records:string;active:boolean}>();
export async function withNativeEvolutionCalibration<T>(capture:Capture,records:unknown,consume:()=>Promise<T>){
 assert(!scope.getStore(),'Overlapping calibration scope');const value={capture,records:encoded(records),active:true};
 return scope.run(value,async()=>{try{return await consume();}finally{value.active=false;}});
}
export async function prepareNativeEvolutionCalibration(engine:Parameters<typeof exportNativePeerProtectionTail>[0],guard:Parameters<typeof exportNativePeerProtectionTail>[1],cohort:Parameters<typeof exportNativePeerProtectionTail>[2],control:Parameters<Parameters<Parameters<typeof exportNativePeerProtectionTail>[0]["withQuiescent"]>[0]>[1],owners:Parameters<typeof exportNativePeerProtectionTail>[6],work:string,signal?:AbortSignal){
 const frozen=scope.getStore();assert(frozen?.active,'New evolution round requires actual current Q4 calibration');
 assert.equal(frozen.records,encoded(cohort.records),'Q4 calibration frozen materials changed');
 const q4=frozen.capture,q4Raw=nativeQ4CalibrationInputs(q4,cohort.records.groups.replay_training),peer=borrowResidentNativeEnginePeer(engine,guard),worker=peer.worker;
 assertNativeQ4Observation(q4.baseline,cohort.records.groups.replay_training);
 assert(q4.baseline.identity===peer.base.identity&&q4.baseline.manifest_sha256===peer.base.model_manifest_sha256);
 const effective=cohort.core;
 assert(effective?q4.baseline.core?.sha256===effective.sha256:q4.baseline.core===null,'Q4 calibration accepted Core changed');
 const check=async()=>{signal?.throwIfAborted();assertNativeExpertScopeGuard(engine,guard);cohort.assertUnchanged();worker.assertLive();assertEvolutionBranch(engine,worker);};
 const newValidation=new Set(cohort.records.groups.new_validation.map((row:any)=>row.id));assert(cohort.records.protection.every((row:any)=>!newValidation.has(row.id)),"NEW VALID cannot enter conversion calibration");
 const baseline=await withObservation({engine,guard,worker,cap:cohort,value:cohort,owners:[peer.source,peer.owner,peer.build,...peer.core?[peer.core]:[]],fast:check,signal},async()=>{
  const rows=[];
  for(const sample of prepareBaselineSamples(cohort.records.protection)){
   await check();const id='constraint-old-'+ ++peer.serial;let finished=false;
   const prompt=await control.tokenizer({operation:'generation_prepare',generation_id:id,purpose:'quality',messages:sample.messages,max_tokens:Number(sample.check.max_new_tokens)},signal);
   try{
    assert(prompt.event==='generation_prepared'&&prompt.generation_id===id&&prompt.thinking==='off'&&Array.isArray(prompt.ids));
    const descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:id,work:worker.options.directory,prompt_ids:prompt.ids,messages_sha256:sha(encoded(sample.messages)),max_tokens:Number(sample.check.max_new_tokens),raw_steps_limit:0,core_version:peer.source.runtime.core_label??null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
    const result=await worker.runDescriptor({...descriptor,name:id,signal}),tokens=result.events.filter((e:any)=>e.event==='token').map((e:any)=>e.token),completion=result.events.find((e:any)=>e.event==='complete');assert(completion);
    for(const[step,token]of tokens.entries()){const decoded=await control.tokenizer({operation:'generation_append',generation_id:id,step,token},signal);assert(decoded.event==='generation_decoded'&&decoded.step===step&&decoded.token===token);}
    const decoded=await control.tokenizer({operation:'generation_finish',generation_id:id});finished=true;assert(decoded.event==='generation_decode_finished');assert.deepEqual(decoded.tokens,tokens);
    rows.push(baselineGenerationRow(sample,{tokens,text:decoded.text,finish_reason:completion.finish_reason,thinking:'off',sampling_seed:'42'}));
   }finally{if(!finished)await control.tokenizer({operation:'generation_finish',generation_id:id});}
  }
  assert(rows.every(row=>row.passed),'Original protection failed before conversion calibration');return rows;
 });
 const oldParts:Buffer[]=[],oldRows:{id:string;positions:number}[]=[];
 for(const row of retentionLossRecords(cohort.records.protection,baseline)){
  await check();const prepared=await control.prepareTrainingSample(row,{signal});
  const output=await exportNativePeerProtectionTail(engine,guard,cohort,row,prepared,baseline,owners,signal),x=output.tensors.tail_x!;
  const raw=await readFile(x.path);await unchangedNativeFile(x.path,x.handle,x.stat);
  const selected=predictorInputs(raw,prepared.labels,6144);assert(oldParts.reduce((n,p)=>n+p.length,0)+selected.length+q4Raw.length<=40000000,"Complete OLD+Q4 constraint exceeds budget; no truncation");oldParts.push(selected);oldRows.push({id:row.id,positions:selected.length/(6144*4)});
 }
 const oldPositions=oldRows.reduce((n,row)=>n+row.positions,0),q4Positions=q4.rows.reduce((n,row)=>n+row.positions,0);
 assert(oldPositions>0&&q4Positions>0&&q4Raw.length===q4Positions*6144*4&&(oldPositions+q4Positions)*6144<=10000000,'Complete conversion calibration exceeds bound; no truncation');
 const path=join(work,'constraint-inputs.raw'),raw=Buffer.concat([...oldParts,q4Raw]);await durable(path,raw);
 const held=await openCheckpointRaw(path,raw.length);owners.own(held,()=>held.handle.close());
 const tensor={...held,path,bytes:raw.length,dtype:'f32' as const,shape:[oldPositions+q4Positions,6144],sha256:sha(raw),assertUnchanged:async()=>{await check();await unchangedNativeFile(path,held.handle,held.stat);}};
 return{tensor,metadata:{method:FINAL_CONVERSION,old_positions:oldPositions,q4_positions:q4Positions,old_rows:oldRows,q4_rows:q4.rows,old_baseline:baseline,identity:peer.base.identity,core_version:effective?.version??null,core_tensor_sha256:peer.source.runtime.core_raw?.sha256??peer.base.original_down_sha256,records_sha256:cohort.records_sha256,training:false,new_validation_used:false,teacher_calls:0}};
}
