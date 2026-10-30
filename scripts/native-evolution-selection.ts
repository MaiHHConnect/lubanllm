/** Independent resident previous/current regression on an already completed one-step candidate. */
import assert from 'node:assert/strict';
import {join,resolve} from 'node:path';
import {readFile,readdir,mkdir,writeFile,lstat} from 'node:fs/promises';
import {openNativeEvolutionEngine,NativeCore450Owner} from '../backend/native-evolution-worker.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {withEvolutionRound,enterEvolutionBranch,assertEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {withObservation} from '../.native-migration/research/native-observation-python-parity-v2/scope.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {writeNativeModelTensors} from '../.native-migration/backend/native-model-write.ts';
import {exportNativePeerTrainingTail} from '../backend/native-evolution-tail.ts';
import {beginNativeEvolutionRound,restoreNativeEvolutionRound} from '../backend/native-evolution-ambient.ts';
import {sha} from '../.native-migration/research/native-full-training/io.ts';

assert(process.argv[2]==='--native'&&(process.argv.length===4||process.argv.length===5),'Use --native COMPLETED_MATH_WORK [comparison-name]');
const comparisonName=process.argv[4]??'selection';assert(/^[a-z][a-z0-9-]*$/.test(comparisonName));
const work=resolve(process.argv[3]!),native=JSON.parse(await readFile(join(work,'native-result.json'),'utf8'));
assert(native.passed&&native.engine_closed&&native.cleanup_confirmed&&native.optimizer_steps===1&&native.captures.length===2);
const root=resolve(import.meta.dir,'../..'),plan=JSON.parse(await readFile(join(root,'.native-migration/research/native-core-final-math-actual-v3/protocol-final.json'),'utf8'));
assert(native.identity===plan.config.identity);const directory=join(work,comparisonName);await mkdir(directory,{mode:0o700});
const opened=await openNativeEvolutionEngine({home:directory,work:directory,root:plan.root,store:plan.config.routing.store,model:native.model,identity:native.identity});
const {engine}=opened,owners=new NativeCore450Owner();let result:any,actualWorkerDirectory:string|undefined;
try{result=await withEvolutionRound(engine,()=>engine.withSourceQuiescent(async(guard,control)=>{
 const previousOwner=await beginNativeEvolutionRound(engine,guard,owners);let roundRestored=false;try{
 const peer=borrowNativeEnginePeer(engine,guard),worker=peer.worker;actualWorkerDirectory=worker.options.directory;await enterEvolutionBranch(engine,worker);
 const sample=native.captures[0].sample,prepared=await control.prepareTrainingSample(sample);assert.deepEqual(prepared.ids,native.captures[0].prepared.ids);assert.deepEqual(prepared.labels,native.captures[0].prepared.labels);
 const labels=join(work,'input',native.captures[0].key+'_labels.raw'),rawLabels=await readFile(labels),cap={research_only:true};let currentOwner:Awaited<ReturnType<typeof writeNativeModelTensors>>|undefined;
 const held=()=>[peer.owner,peer.source,peer.build,peer.core,peer.previous,currentOwner].filter(Boolean);
 const fast=async()=>{guard.assertQuiescent();worker.assertLive();assertEvolutionBranch(engine,worker);};
 const select=async(mode:'previous'|'current')=>withObservation({engine,guard,worker,cap,value:{selection_command:true},owners:held(),fast},()=>worker.selectCoreObservation(mode));
 const observe=async(name:string)=>{
  const descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:name,work:worker.options.directory,prompt_ids:prepared.ids,messages_sha256:sha(Buffer.from(new BigInt64Array(prepared.ids.map(BigInt)).buffer)),max_tokens:1,raw_steps_limit:8,core_version:peer.source.runtime.core_label??null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
  return withObservation({engine,guard,worker,cap,value:cap,owners:held(),fast},async()=>{
   const loss=await worker.lossDescriptor({...descriptor,labels:{path:labels,sha256:sha(rawLabels)},name:name+'-ce'});
   const generation=await worker.runDescriptor({...descriptor,name:name+'-head'}),head=await readFile(join(worker.options.directory,name+'-head','step_0_head.raw'));
   assert.equal(head.length,248320*2);const path=join(directory,name+'-head.raw');await writeFile(path,head,{mode:0o600,flag:'wx'});
   return{name,loss:loss.loss,head:{path,dtype:'bf16',shape:[1,1,248320],bytes:head.length,sha256:sha(head)},token:generation.events.find((v:any)=>v.event==='token')?.token,observation:worker.coreObservationMode,pid:worker.pid,optimizer_step:worker.status.optimizer_step};
  });
 };
 const previous=await observe('previous-before');
 const tailPairs:any[]=[];
 if(comparisonName!=='selection'){
  const cohort={records:{groups:{new_training:[native.captures[0].sample],replay_training:[native.captures[1].sample]}},assertUnchanged(){guard.assertQuiescent();}} as unknown as Parameters<typeof exportNativePeerTrainingTail>[2];
  for(const captured of native.captures){const row=await control.prepareTrainingSample(captured.sample),tail=await exportNativePeerTrainingTail(engine,guard,cohort,captured.sample,row,owners);
   for(const tensor of tail.manifest.tensors){const before=captured.manifest.tensors.find((t:any)=>t.name===tensor.name);assert(before&&before.sha256===tensor.sha256&&before.bytes===tensor.bytes);tailPairs.push({name:captured.key+'-'+tensor.name,bytes:tensor.bytes,sha256:tensor.sha256,exact:true});}
  }
 }

 currentOwner=await writeNativeModelTensors({path:join(directory,'candidate-state.safetensors.partial'),tensors:Object.entries(native.state).map(([tensor_key,v]:[string,any])=>({tensor_key,path:v.path,sha256:v.sha256,bytes:v.bytes,shape:v.shape,dtype:tensor_key==='candidate'?'BF16':'F32'}))});owners.own(currentOwner,()=>currentOwner!.close());
 let installed=false;
 try{
  await worker.loadCoreOwner(currentOwner,1,currentOwner);installed=true;
  const current=await observe('current-first');const previousACK=await select('previous'),previousAgain=await observe('previous-again');
  const currentACK=await select('current'),currentAgain=await observe('current-again');
  assert(previous.head.sha256===previousAgain.head.sha256&&previous.loss===previousAgain.loss);
  assert(current.head.sha256===currentAgain.head.sha256&&current.loss===currentAgain.loss);
  const acceptedVersion=peer.source.runtime.core_label??null,acceptedSHA=peer.source.runtime.core_raw?.sha256??engine.config.aten!.base.original_down_sha256;
  assert.equal(previousACK.core_tensor_sha256,acceptedSHA);assert.equal(currentACK.core_tensor_sha256,native.state.candidate.sha256);
  assert.equal(previousACK.step,0);assert.equal(currentACK.step,1);assert.equal(previousACK.optimizer_step,currentACK.optimizer_step);
  if(comparisonName!=='selection'){
   const prior=JSON.parse(await readFile(join(work,'selection/result.json'),'utf8'));
   assert(current.head.sha256===prior.current.head.sha256&&current.loss===prior.current.loss);
   const noneReference=JSON.parse(await readFile(join(work,'selection-exfat-effective-none-v3/result.json'),'utf8'));
   assert.equal(noneReference.accepted_tensor_sha256,acceptedSHA);
   for(const [actual,reference]of [[previous,noneReference.previous],[current,noneReference.current]]){assert.equal(actual.loss,reference.loss);assert((await readFile(actual.head.path)).equals(await readFile(reference.head.path)),'Previous qualified None/current head differs');}
   for(const row of [previous,current,previousAgain,currentAgain]){const files=await readdir(join(worker.options.directory,row.name+'-head'));assert.deepEqual(files.sort(),['manifest.json','step_0_head.raw','step_0_scores.raw']);const lossFiles=await readdir(join(worker.options.directory,row.name+'-ce'));assert.deepEqual(lossFiles.sort(),['loss.json','loss.raw']);}
  }
  const restoration=await restoreNativeEvolutionRound(engine,guard,previousOwner);roundRestored=true;const previousRestored=await observe('previous-restored');assert.equal(previousRestored.head.sha256,previous.head.sha256);assert.equal(previousRestored.loss,previous.loss);
  return{passed:true,restoration,previousRestored,native_round_restored:true,tailPairs,audit_files_absent:comparisonName!=='selection',current_previous_binary_exact:comparisonName!=='selection',none_current_reference:comparisonName!=='selection'?'selection-exfat-effective-none-v3':null,none_current_reference_exact:comparisonName!=='selection',previous_baseline_changed:acceptedSHA!==native.captures[0].manifest.core_tensor_sha256,identity:native.identity,model:native.model,core_file:engine.config.core?{path:engine.config.core.file,tensor_key:engine.config.core.tensor_key,sha256:engine.config.core.sha256}:null,core_version:acceptedVersion,accepted_tensor_sha256:acceptedSHA,candidate:native.state.candidate,prepared,sample,previous,current,previousAgain,currentAgain,previousACK,currentACK,training_steps:0,product_admitted:false};
 }finally{if(!roundRestored){await restoreNativeEvolutionRound(engine,guard,previousOwner);roundRestored=true;}}
 }finally{if(!roundRestored)await restoreNativeEvolutionRound(engine,guard,previousOwner);}
}),async(worker,enabled)=>engine.withSourceQuiescent(async guard=>{assert.equal(borrowNativeEnginePeer(engine,guard).worker,worker);if(worker.isBranchEnabled!==enabled)await worker.setBranchEnabled(enabled);}));}
finally{await owners.close();await opened.close();}
assert(actualWorkerDirectory);for(const name of ['parent_rng.raw','restored_rng.raw','parent_mps_rng.raw','restored_mps_rng.raw'])await assert.rejects(lstat(join(actualWorkerDirectory,name)),{code:'ENOENT'});
await writeFile(join(directory,'result.json'),JSON.stringify({...result,engine_closed:true,process_close_rng_audit_files_absent:true},null,2)+'\n',{flag:'wx',mode:0o600});console.log(JSON.stringify({directory,passed:true,engine_closed:true,training_steps:0}));
