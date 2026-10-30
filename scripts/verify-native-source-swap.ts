/** Independent real-model source transaction check. No training, pointer writes or product qualification. */
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,readdir} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {openNativeEvolutionEngine,nativeEvolutionRuntimeResources,NativeCore450Owner} from '../backend/native-evolution-worker.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {withEvolutionRound,enterEvolutionBranch,assertEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {withObservation} from '../.native-migration/research/native-observation-python-parity-v2/scope.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {openCheckpointRaw} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
import {sha} from '../.native-migration/research/native-full-training/io.ts';

const args=process.argv.slice(2),flags=new Map<string,string>();let checkOnly=false;
for(let i=0;i<args.length;i++){const key=args[i]!;if(key==='--check-only'){checkOnly=true;continue;}assert(['--native','--output','--root','--store','--reference'].includes(key)&&!flags.has(key)&&args[i+1], 'Unknown, repeated or incomplete option: '+key);flags.set(key,args[++i]!);}
assert(flags.has('--native'),'Use --native COMPLETED_MATH_WORK [--output source-swap-v1] [--check-only]');
const work=resolve(flags.get('--native')!),name=flags.get('--output')??'source-swap-v1';assert(/^[a-z][a-z0-9-]*$/.test(name));
const root=resolve(flags.get('--root')??'/Volumes/移动数据库/lubanllm'),store=resolve(flags.get('--store')??join(root,'modules/luban0-v1'));
const native=JSON.parse(await readFile(join(work,'native-result.json'),'utf8'));
assert(native.passed&&native.engine_closed&&native.cleanup_confirmed&&native.optimizer_steps===1&&native.captures.length===2);
const candidate=native.state.candidate;assert(candidate.dtype==='bf16'&&candidate.bytes===25165824&&JSON.stringify(candidate.shape)==='[2048,6144]'&&/^[a-f0-9]{64}$/.test(candidate.sha256));
const referencePath=resolve(flags.get('--reference')??join(work,'selection-exfat-effective-none-v3/result.json'));
const reference=JSON.parse(await readFile(referencePath,'utf8'));
assert(reference.passed&&reference.engine_closed&&reference.identity===native.identity&&reference.core_version===null&&reference.candidate.sha256===candidate.sha256);
assert.deepEqual(reference.prepared,native.captures[0].prepared);
const resources=nativeEvolutionRuntimeResources();assert.equal(resources.base.identity,native.identity);
if(checkOnly){console.log(JSON.stringify({ready:true,model_loaded:false,GPU:false,training_steps:0,root,store,reference:referencePath,candidate:candidate.path,output:join(work,name)}));process.exit(0);}

const directory=join(work,name);await mkdir(directory,{mode:0o700});
const pointerNames=['active-core.json','active.json','active-router.json'];
async function pointers(){return Promise.all(pointerNames.map(async file=>{try{return await readFile(join(store,file));}catch(error:any){if(error.code==='ENOENT')return null;throw error;}}));}
const originalPointers=await pointers(),started=performance.now();let opened:Awaited<ReturnType<typeof openNativeEvolutionEngine>>|undefined,result:any,primary:unknown,peerDirectory:string|undefined;
const stages:{phase:string;elapsed_seconds:number}[]=[];
const progress=(phase:string)=>{const event={phase,elapsed_seconds:(performance.now()-started)/1000};stages.push(event);console.log(JSON.stringify(event));};
async function retryClose(close:()=>Promise<void>){let previous='';for(;;){try{await close();return;}catch(error){const message=String(error);if(message!==previous){console.error('Retaining actual owner for close retry: '+message);previous=message;}await Bun.sleep(1000);}}}
try{
 progress('open_source');opened=await openNativeEvolutionEngine({home:directory,work:directory,root,store,model:native.model,identity:native.identity});
 const {engine}=opened,owners=new NativeCore450Owner();
 result=await withEvolutionRound(engine,()=>engine.withSourceQuiescent(async(guard,control)=>{
  const peer=borrowNativeEnginePeer(engine,guard),worker=peer.worker;peerDirectory=worker.options.directory;
  try{
  assert.equal(peer.source.runtime.core_label??null,null,'This check requires the current effective original source; it never changes the product pointer');
  assert.equal(worker.status.backbone_resident,false,'A fresh cold Engine is required');
  const expert=peer.previous,router=peer.routerOwner,source=peer.source,core=peer.core;
  await enterEvolutionBranch(engine,worker);
  const prepared=await control.prepareTrainingSample(native.captures[0].sample);assert.deepEqual(prepared.ids,native.captures[0].prepared.ids);assert.deepEqual(prepared.labels,native.captures[0].prepared.labels);assert.equal(prepared.eos,native.captures[0].prepared.eos);
  const descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:'source-swap-original',work:worker.options.directory,prompt_ids:prepared.ids,messages_sha256:native.captures[0].manifest.ids_sha256,max_tokens:1,raw_steps_limit:8,core_version:null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
  const originalText=await readFile(descriptor.path,'utf8'),replacement:Record<string,string>={core_version:'core-'+candidate.sha256.slice(0,12),core_path:candidate.path,core_file_sha256:candidate.sha256,core_tensor_sha256:candidate.sha256,core_tensor_key:resources.base.core_parameter,core_offset:'0',core_bytes:String(candidate.bytes)};
  const seen=new Set<string>();const candidateText=originalText.split('\n').map(line=>{const fields=line.split('\t');if(fields[0]==='config'&&Object.hasOwn(replacement,fields[1]!)){assert.equal(fields.length,3);assert(!seen.has(fields[1]!));seen.add(fields[1]!);return 'config\t'+fields[1]+'\t'+replacement[fields[1]!];}return line;}).join('\n');
  assert.equal(seen.size,7);const candidateDescriptor={path:join(directory,'source-swap-candidate.tsv'),sha256:sha(candidateText)};await writeFile(candidateDescriptor.path,candidateText,{mode:0o600,flag:'wx'});
  // The native source loader performs the existing finite/full parameter check once. Host keeps its FD/stat binding.
  const raw=await openCheckpointRaw(candidate.path,candidate.bytes);
  const heldCandidate={assertUnchanged:()=>unchangedNativeFile(candidate.path,raw.handle,raw.stat),close:()=>raw.handle.close()};owners.own(heldCandidate,()=>heldCandidate.close());assert.equal(Number(raw.stat.size),candidate.bytes);
  const cap={research_only:true},held=()=>[peer.owner,peer.source,peer.build,peer.core,peer.previous,heldCandidate].filter(Boolean);
  const fast=async()=>{guard.assertQuiescent();worker.assertLive();assertEvolutionBranch(engine,worker);};
  const state=()=>{const value=worker.status;return{pid:worker.pid,expert_storage_ids:value.storage_ids,expert_constructions:value.constructions,expert_optimizer_step:value.optimizer_step,router_storage_ids:value.router_storage_ids??null,router_constructions:value.router_constructions??null,branch_enabled:worker.isBranchEnabled};};
  const initial=state();let changed=false,operationError:unknown;
  const unchanged=()=>{assert.equal(peer.previous,expert);assert.equal(peer.routerOwner,router);assert.equal(peer.source,source);assert.equal(peer.core,core);assert.deepEqual(state(),initial);};
  const observe=async(label:string,input:typeof descriptor,expected:any)=>{
   const observed=await withObservation({engine,guard,worker,cap,value:cap,owners:held(),fast},async()=>{
    const loss=await worker.lossDescriptor({...input,labels:{path:join(work,'input',native.captures[0].key+'_labels.raw'),sha256:native.captures[0].manifest.labels_sha256},name:label+'-ce'});
    const generation=await worker.runDescriptor({...input,name:label+'-head'}),head=await readFile(join(worker.options.directory,label+'-head','step_0_head.raw'));
    const expectedHead=await readFile(expected.head.path);assert.equal(head.length,248320*2);assert(head.equals(expectedHead),'Full BF16 head differs: '+label);assert.equal(loss.loss,expected.loss,'CE differs: '+label);
    const token=generation.events.find((event:any)=>event.event==='token')?.token;assert.equal(token,expected.token);
    const path=join(directory,label+'-head.raw');await writeFile(path,head,{flag:'wx',mode:0o600});
    assert.deepEqual((await readdir(join(worker.options.directory,label+'-head'))).sort(),['manifest.json','step_0_head.raw','step_0_scores.raw']);
    assert.deepEqual((await readdir(join(worker.options.directory,label+'-ce'))).sort(),['loss.json','loss.raw']);
    return{loss:loss.loss,token,head:{path,bytes:head.length,dtype:'bf16',shape:[1,1,248320]},reference_head:expected.head.path,full_head_exact:true,CE_exact:true};
   });unchanged();return observed;
  };
  const swap=async(label:string,input:typeof descriptor,expectedSHA:string,version:string|null)=>{
   await worker.stageAcceptedCoreSource({...input,name:label},owners);const ack=await worker.applyAcceptedCoreSource(owners);
   assert.equal(ack.pid,worker.pid);assert.equal(ack.core_version,version);assert.equal(ack.actual_core_tensor_sha256,expectedSHA);assert.equal(ack.native.base_identity,native.identity);assert.equal(ack.native.checkpoint_sha256,resources.base.checkpoint_sha256);assert.equal(ack.native.manifest_sha256,resources.base.model_manifest_sha256);
   assert.equal(ack.native.resident_core_readback,false);assert.equal(ack.native.parent_rng_snapshot,false);await worker.confirmAcceptedCoreSource(true,()=>{changed=version!==null;});unchanged();
   return{pid:ack.pid,protocol:ack.native.protocol,core_version:ack.core_version,actual_core_tensor_sha256:ack.actual_core_tensor_sha256,source_confirmed:true,product_admitted:false};
  };
  try{
   progress('retain_cold_backbone');const retained=await worker.retainCoreBackbone({...descriptor,core_tensor_sha256:resources.base.original_down_sha256});assert.equal(worker.status.backbone_resident,true);unchanged();
   progress('original_observation');const before=await observe('source-original',descriptor,reference.previous);
   progress('apply_existing_candidate');const candidateACK=await swap('source-candidate',candidateDescriptor,candidate.sha256,replacement.core_version!);changed=true;
   const current=await observe('source-candidate',candidateDescriptor,reference.current);
   progress('restore_original_source');const restoreACK=await swap('source-original-restored',descriptor,resources.base.original_down_sha256,null);changed=false;
   const restored=await observe('source-restored',descriptor,reference.previous);await heldCandidate.assertUnchanged();
   const entries=await readdir(worker.options.directory);for(const file of ['source-candidate','source-original-restored'])assert(!entries.includes(file),'Source transaction emitted audit files');
   return{passed:true,cold_start:true,actual_backbone_retained:retained.response.backbone_resident===true,retained_core_tensor_sha256:retained.response.retained_core_tensor_sha256,before,current,restored,candidateACK,restoreACK,resource_state:initial,expert_storage_exact:true,router_owner_reference_exact:true,router_storage_measured:initial.router_storage_ids!==null,router_evidence:'Same TS owner retained; C++ source transaction does not access router; GPU router addresses were not measured',source_owner_reference_exact:true,ordinary_audit_files_absent:true,graph_binary_sha256:worker.options.binary_sha256,training_steps:0,product_admitted:false,qualification:false};
  }catch(error){operationError=error;throw error;}finally{
   try{await worker.settleAcceptedCoreSource();if(changed)await swap('source-failure-restore',descriptor,resources.base.original_down_sha256,null);}
   catch(cleanup){if(operationError)throw new AggregateError([operationError,cleanup],'Source verification and restoration failed');throw cleanup;}
  }
  }finally{await retryClose(()=>owners.close());}
 }),async(worker,enabled)=>engine.withSourceQuiescent(async guard=>{assert.equal(borrowNativeEnginePeer(engine,guard).worker,worker);if(worker.isBranchEnabled!==enabled)await worker.setBranchEnabled(enabled);}));
}catch(error){primary=error;}
finally{if(opened)await retryClose(()=>opened!.close());}
let pointerUnchanged=false,closeAuditAbsent=false;
try{
 if(peerDirectory){const entries=await readdir(peerDirectory);for(const file of ['parent_rng.raw','restored_rng.raw','parent_mps_rng.raw','restored_mps_rng.raw'])assert(!entries.includes(file),'Process close emitted an extra RNG audit file');closeAuditAbsent=true;}
 const after=await pointers();assert(originalPointers.every((value,index)=>value===null?after[index]===null:after[index]!==null&&value.equals(after[index]!)),'Product pointers changed during independent source verification');pointerUnchanged=true;
}catch(error){primary=primary?new AggregateError([primary,error],'Source verification and final checks failed'):error;}

const report={...(result??{}),passed:!primary&&result?.passed===true,identity:native.identity,model:native.model,candidate,reference:referencePath,root,store,stages,total_seconds:(performance.now()-started)/1000,engine_closed:!!opened,product_pointers_unchanged:pointerUnchanged,close_rng_audit_files_absent:closeAuditAbsent,training_steps:0,product_admitted:false,qualification:false,GPU:true,...(primary?{error:String(primary),stack:primary instanceof Error?primary.stack:undefined}:{}),runtime_manifest_sha256:resources.release.sha256};
await writeFile(join(directory,'result.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});if(primary)throw primary;progress('complete');
