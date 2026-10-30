/** Daily frozen-prefix capture. Source admission belongs to the existing Engine lifecycle. */
import assert from 'node:assert/strict';
import {retentionLossRecords} from './native-evolution-loss-targets.ts';
import {encoded} from './canonical-json.ts';
import {readFile,readdir} from 'node:fs/promises';
import {join} from 'node:path';
import {NativeEngine,assertNativeExpertScopeGuard,validateNativeCorePreparedSample,type NativeSampleTokens} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {borrowResidentNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {assertEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {withObservation} from '../.native-migration/research/native-observation-python-parity-v2/scope.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {assertTailTrainingInput} from '../.native-migration/research/native-evolution-retained-producer-v1/train-input.ts';
import {openCheckpointRaw,nativeCheckpointJSON} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
import {durable,sha} from '../.native-migration/research/native-full-training/io.ts';
import type {NativeCore450Owner} from '../.native-migration/research/native-core450-owner-controller/owner.ts';
import type {selectIndependentNativeEvolutionCohort} from '../.native-migration/research/native-evolution-producer/cohort.ts';
type Guard=Parameters<typeof assertNativeExpertScopeGuard>[1];
type Cohort=ReturnType<typeof selectIndependentNativeEvolutionCohort>;

export async function exportNativePeerTrainingTail(engine:NativeEngine,guard:Guard,cohort:Cohort,sample:any,prepared:NativeSampleTokens,owners:NativeCore450Owner,signal?:AbortSignal){
 return exportTail(engine,guard,cohort,sample,prepared,owners,assertTailTrainingInput(cohort,sample,prepared),signal);
}

export async function exportNativePeerProtectionTail(engine:NativeEngine,guard:Guard,cohort:Cohort,sample:any,prepared:NativeSampleTokens,baseline:any[],owners:NativeCore450Owner,signal?:AbortSignal){
 const expected=retentionLossRecords(cohort.records.protection,baseline).find(row=>row.id===sample.id);
 assert(expected&&encoded(expected)===encoded(sample),"OLD constraint must be this original protection and verified baseline target");
 cohort.assertUnchanged();assert(prepared.ids.length===prepared.labels.length&&prepared.ids.length>=2&&prepared.ids.length<=512&&prepared.prefix_count>0&&prepared.prefix_count<prepared.ids.length&&prepared.native_labels_compatible&&prepared.ids.at(-1)===prepared.eos&&prepared.labels.every((v,i)=>v===(i<prepared.prefix_count?-100:prepared.ids[i])),"OLD complete predictor labels differ");
 const approved={sample:Buffer.from(encoded(sample)),sample_sha256:sha(encoded(sample,true)),ids:Buffer.from(new BigInt64Array(prepared.ids.map(BigInt)).buffer),labels:Buffer.from(new BigInt64Array(prepared.labels.map(BigInt)).buffer)};
 return exportTail(engine,guard,cohort,sample,prepared,owners,approved,signal);
}

async function exportTail(engine:NativeEngine,guard:Guard,cohort:Cohort,sample:any,prepared:NativeSampleTokens,owners:NativeCore450Owner,approved:ReturnType<typeof assertTailTrainingInput>,signal?:AbortSignal){
 assertNativeExpertScopeGuard(engine,guard);validateNativeCorePreparedSample(engine,guard,prepared);
 const peer=borrowResidentNativeEnginePeer(engine,guard),worker=peer.worker,pid=worker.pid;
 assert(worker.options.device==='mps'&&worker.options.hidden===2048&&!worker.hasInstalledCore);
 const fast=async()=>{signal?.throwIfAborted();assertNativeExpertScopeGuard(engine,guard);worker.assertLive();assert.equal(worker.pid,pid);assertEvolutionBranch(engine,worker);};
 return withObservation({engine,guard,worker,cap:cohort,value:cohort,owners:[peer.source,peer.owner,peer.build,...peer.core?[peer.core]:[]],fast,signal},async()=>{
  const id='training-tail-'+ ++peer.serial,labelsSHA=sha(approved.labels),idsSHA=sha(approved.ids),sampleRawSHA=sha(approved.sample),labelPath=join(worker.options.directory,id+'-labels.raw');
  await durable(labelPath,approved.labels);
  const descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:id,work:worker.options.directory,prompt_ids:[...prepared.ids],messages_sha256:idsSHA,max_tokens:1,raw_steps_limit:0,core_version:peer.source.runtime.core_label??null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
  const output=await worker.exportOrdinaryTrainingTail({...descriptor,labels:{path:labelPath,sha256:labelsSHA},sample_sha256:approved.sample_sha256,sample_raw_sha256:sampleRawSHA,name:id,signal});
  const directory=output.directory,raw=await readFile(join(directory,'manifest.json')),m=nativeCheckpointJSON(raw) as Record<string,any>;
  const coreSHA=peer.source.runtime.core_raw?.sha256??peer.base.original_down_sha256;
  assert(m.schema==='luban-ordinary-training-tail-v1'&&m.pid===pid&&m.device==='mps'&&m.layers===24&&m.hidden===2048&&m.intermediate===6144&&m.tokens===prepared.ids.length&&m.descriptor_sha256===descriptor.sha256&&m.sample_sha256===approved.sample_sha256&&m.sample_raw_sha256===sampleRawSHA&&m.ids_sha256===idsSHA&&m.labels_sha256===labelsSHA&&m.core_tensor_sha256===coreSHA&&m.core_version===(peer.source.runtime.core_label??'null')&&!m.expert_used&&!m.branch_enabled&&m.cache_fresh&&m.stop_before_last_down&&m.optimizer_steps===0&&m.teacher_calls===0&&!m.product_admission);
  assert.deepEqual((await readdir(directory)).sort(),['ids.raw','labels.raw','manifest.json','tail_residual.raw','tail_x.raw']);
  const tensors:Record<string,Awaited<ReturnType<typeof openCheckpointRaw>>&{path:string;bytes:number;sha256:string;dtype:'bf16'|'i64';shape:readonly number[]}>={};
  const specs=[['tail_x','bf16',6144],['tail_residual','bf16',2048],['ids','i64',1],['labels','i64',1]] as const;
  assert(Array.isArray(m.tensors)&&m.tensors.length===4);
  for(const[name,dtype,width]of specs){
   const tensor:any=m.tensors.find((value:any)=>value.name===name),shape=width===1?[1,prepared.ids.length]:[1,prepared.ids.length,width],bytes=prepared.ids.length*width*(dtype==='bf16'?2:8);
   assert(tensor&&tensor.file===name+'.raw'&&tensor.dtype===(dtype==='bf16'?'bfloat16':'int64')&&tensor.bytes===bytes&&/^[a-f0-9]{64}$/.test(tensor.sha256));assert.deepEqual(tensor.shape,shape);
   if(name==='ids'||name==='labels')assert.equal(tensor.sha256,name==='ids'?idsSHA:labelsSHA);
   const path=join(directory,tensor.file),held=await openCheckpointRaw(path,bytes);owners.own(held,()=>held.handle.close());assert.equal(held.stat.size,BigInt(bytes));
   tensors[name]={...held,path,bytes,sha256:tensor.sha256,dtype,shape};
  }
  const check=async()=>{await fast();for(const tensor of Object.values(tensors))await unchangedNativeFile(tensor.path,tensor.handle,tensor.stat);};
  await check();const source=peer.build.proof.build.sources.find((value:any)=>value.path===peer.build.proof.build.macros.LUBAN_PREFIX_SOURCE_SHA);assert(source);
  return{directory,pid,manifest:m,manifest_sha256:sha(raw),native_source_sha256:source.sha256,tensors,sample_sha256:approved.sample_sha256,sample_raw_sha256:sampleRawSHA,assertUnchanged:check,source_mode:'same_retained_accepted_backbone',product_admitted:false as const,qualified:false as const};
 });
}
