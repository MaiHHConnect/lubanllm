/** Live, branch-OFF observations of a held candidate; no training or publication authority. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {prepareBaselineSamples,baselineGenerationRow} from '../.native-migration/research/native-ordinary-baseline/rows.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {withObservation} from '../.native-migration/research/native-observation-python-parity-v2/scope.ts';
import {assertEvolutionBranch,enterEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import type {NativeModelTensors} from '../.native-migration/backend/native-model-tensor.ts';
import {durable,sha} from '../.native-migration/research/native-full-training/io.ts';
import {uniqueNativeEvolutionRecords} from './native-evolution-material.ts';
import {retentionLossRecords} from './native-evolution-loss-targets.ts';

/** Retry only lifecycle cleanup, never observations/transactions or a failed computation. */
export async function closeImportOwner(close:()=>Promise<void>,onError:(error:unknown)=>void){
 for(;;){try{await close();return;}catch(error){onError(error);await Bun.sleep(1000);}}
}

export async function observeNativeImport(o:{engine:any;guard:any;control:any;owners:any;candidate:NativeModelTensors;
 raw:{path:string;bytes:number;sha256:string;assertUnchanged():void|Promise<void>};version:string;records:any;work:string;signal:AbortSignal;onPhase:(phase:string)=>void;onCleanupError:(error:unknown)=>void}){
 const peer=borrowNativeEnginePeer(o.engine,o.guard),worker=peer.worker,base=peer.base;
 assert.equal(peer.source.runtime.core_label??null,null,'This validated import is bound to the original effective Core');
 const tensor=o.candidate.tensors[base.core_parameter]!;assert(tensor);
 assert.equal(o.raw.sha256,tensor.sha256);assert.equal(o.raw.bytes,tensor.bytes);await o.raw.assertUnchanged();
 const all=uniqueNativeEvolutionRecords(o.records.groups,o.records.protection,o.records.retention_regressions);
 const originalOwners={source:peer.source,core:peer.core,expert:peer.previous,router:peer.routerOwner};
 let serial=0,selected=false,original:any;
 const unchanged=async()=>{o.guard.assertQuiescent();worker.assertLive();assertEvolutionBranch(o.engine,worker);await o.candidate.assertUnchanged();await o.raw.assertUnchanged();
  assert(peer.source===originalOwners.source&&peer.core===originalOwners.core&&peer.previous===originalOwners.expert&&peer.routerOwner===originalOwners.router);};
 const descriptor=async(ids:number[],messagesSHA:string,maximum:number,current:boolean)=>{
  const request_id='import-'+ ++serial;
  const prior=await writePeerDescriptor({runtime:peer.source.runtime,guard:o.guard,owner:peer.owner,core_owner:peer.core,
   request_id,work:worker.options.directory,prompt_ids:ids,messages_sha256:messagesSHA,max_tokens:maximum,raw_steps_limit:0,
   core_version:null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
  original??=prior;if(!current)return prior;
  const values:Record<string,string>={core_version:o.version,core_path:o.raw.path,core_file_sha256:o.raw.sha256,
   core_tensor_sha256:tensor.sha256,core_tensor_key:base.core_parameter,core_offset:'0',core_bytes:String(o.raw.bytes)};
  const seen=new Set<string>(),text=(await readFile(prior.path,'utf8')).split('\n').map(line=>{const parts=line.split('\t');
   if(parts[0]!=='config'||!Object.hasOwn(values,parts[1]!))return line;
   assert(parts.length===3&&!seen.has(parts[1]!));seen.add(parts[1]!);return 'config\t'+parts[1]+'\t'+values[parts[1]!];}).join('\n');
  assert.equal(seen.size,7);const path=join(o.work,request_id+'-candidate.tsv');await durable(path,text);return{path,sha256:sha(text)};
 };
 const swap=async(d:any,current:boolean)=>{await worker.stageAcceptedCoreSource({...d,name:'import-source-'+ ++serial},o.owners);
  const ack=await worker.applyAcceptedCoreSource(o.owners);assert.equal(ack.pid,worker.pid);
  assert.equal(ack.actual_core_tensor_sha256,current?tensor.sha256:base.original_down_sha256);assert.equal(ack.core_version,current?o.version:null);
  await worker.confirmAcceptedCoreSource(true,()=>{selected=current;});return{pid:ack.pid,tensor_sha256:ack.actual_core_tensor_sha256};};
 const cap=Object.freeze({operation:'candidate-import-observation'});
 const observation=<T>(body:()=>Promise<T>)=>withObservation({engine:o.engine,guard:o.guard,worker,cap,value:cap,
  owners:[peer.owner,peer.source,peer.build,peer.core,peer.previous,o.candidate,o.raw].filter(Boolean),fast:unchanged},body);
 const generate=async(current:boolean,pass:number)=>observation(async()=>{
  const rows=[];for(const sample of prepareBaselineSamples(all)){
   o.signal.throwIfAborted();const id='import-generation-'+ ++serial;let prepared=false,finished=false;
   try{const prompt=await o.control.tokenizer({operation:'generation_prepare',generation_id:id,purpose:'quality',messages:sample.messages,max_tokens:Number(sample.check.max_new_tokens)},o.signal);prepared=true;
    assert(prompt.event==='generation_prepared'&&prompt.generation_id===id&&prompt.thinking==='off'&&Array.isArray(prompt.ids));
    const d=await descriptor(prompt.ids,sha(JSON.stringify(sample.messages)),Number(sample.check.max_new_tokens),current);
    const actual=await worker.runDescriptor({...d,name:'import-generate-'+current+'-'+pass+'-'+rows.length,signal:o.signal});
    const tokens=actual.events.filter((e:any)=>e.event==='token').map((e:any)=>e.token),done=actual.events.find((e:any)=>e.event==='complete');assert(done&&tokens.length<=Number(sample.check.max_new_tokens));
    for(const[step,token]of tokens.entries()){const frame=await o.control.tokenizer({operation:'generation_append',generation_id:id,step,token},o.signal);assert(frame.event==='generation_decoded'&&frame.generation_id===id&&frame.step===step&&frame.token===token);}
    const decoded=await o.control.tokenizer({operation:'generation_finish',generation_id:id});finished=true;assert(decoded.event==='generation_decode_finished'&&decoded.generation_id===id);assert.deepEqual(decoded.tokens,tokens);
    rows.push({...baselineGenerationRow(sample,{tokens,text:decoded.text,finish_reason:done.finish_reason,thinking:'off',sampling_seed:'42'}),input_tokens:prompt.ids});
   }finally{if(prepared&&!finished)await o.control.tokenizer({operation:'generation_finish',generation_id:id});}
  }return rows;
 });
 const preparedLoss:any[]=[];
 const prepareLoss=async(rows:readonly Record<string,any>[])=>{
  for(const row of rows){o.signal.throwIfAborted();const p=await o.control.prepareTrainingSample(row);assert(p.native_labels_compatible&&p.ids.length<=512&&p.labels.length===p.ids.length);
   const labels=new Uint8Array(new BigInt64Array(p.labels.map(BigInt)).buffer),path=join(o.work,'import-labels-'+preparedLoss.length+'.raw');await durable(path,labels);
   preparedLoss.push({id:row.id,original:await descriptor(p.ids,sha(JSON.stringify(p.ids)),1,false),current:await descriptor(p.ids,sha(JSON.stringify(p.ids)),1,true),labels:{path,sha256:sha(labels)}});
  }
 };
 const losses=async(current:boolean)=>observation(async()=>{const rows=[];for(const p of preparedLoss){o.signal.throwIfAborted();const value=await worker.lossDescriptor({...p[current?'current':'original'],labels:p.labels,name:'import-loss-'+ ++serial,signal:o.signal});assert(Number.isFinite(value.loss)&&value.loss>=0);rows.push({id:p.id,loss:value.loss});}return rows;});
 try{
  await enterEvolutionBranch(o.engine,worker);
  o.onPhase('bf16_baseline_two_passes');const baseline=[await generate(false,0),await generate(false,1)] as const;
  const targets=retentionLossRecords(uniqueNativeEvolutionRecords({},o.records.protection,o.records.retention_regressions),baseline[0]);
  await prepareLoss([...targets,...o.records.groups.replay_validation]);
  o.onPhase('bf16_baseline_losses');const before=await losses(false);
  assert(original);await worker.retainCoreBackbone({...original,core_tensor_sha256:base.original_down_sha256});
  const install=preparedLoss[0].current;await swap(install,true);
  o.onPhase('bf16_candidate_two_passes');const candidate=[await generate(true,0),await generate(true,1)] as const;
  o.onPhase('bf16_candidate_losses');const after=await losses(true);
  const restored=await swap(original,false);await unchanged();
  const mean=(rows:any[])=>{assert(rows.length);return rows.reduce((n,row)=>n+row.loss,0)/rows.length;};
  return{baseline,candidate,baselineCoreLoss:before.slice(0,targets.length),candidateCoreLoss:after.slice(0,targets.length),
   baselineReplayLoss:mean(before.slice(targets.length)),candidateReplayLoss:mean(after.slice(targets.length)),
   targets:targets.map(({id,answer})=>({id,answer})),restored,optimizer_steps:0 as const};
 }finally{await closeImportOwner(async()=>{await worker.settleAcceptedCoreSource();if(selected){assert(original);await swap(original,false);}},o.onCleanupError);}
}
