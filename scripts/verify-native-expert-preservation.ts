/** Research-only two-update original-HF comparison input. Bundle with the normal native composition. */
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,mkdtemp,copyFile,realpath,lstat} from 'node:fs/promises';
import {constants} from 'node:fs';
import {join,resolve,relative} from 'node:path';
import {issueNativeExpertQualification} from './issue-native-expert-qualification.ts';
import {privateJSON} from '../backend/settings.ts';
import {closeImportOwner} from '../backend/native-evolution-import-observe.ts';
import {homedir,cpus,totalmem,arch,platform} from 'node:os';
import {randomUUID} from 'node:crypto';
import {parseArgs} from 'node:util';
import {applicationConnection} from './run-native-identity-round.ts';
import {openNativeEvolutionEngine,nativeEvolutionRuntimeResources} from '../backend/native-evolution-worker.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {validateExpertCurrentSnapshot} from '../.native-migration/research/native-unified-expert/expert-process.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {durable,sha} from '../.native-migration/research/native-full-training/io.ts';

const KEYS=['gate.weight','up.weight','down.weight'];
/** Native label input requires POSIX 0600; ExFAT is for durable research output only. */
async function privateTrainingLabels(directory:string,labels:number[]){
 const bytes=new Uint8Array(new BigInt64Array(labels.map(BigInt)).buffer),path=join(directory,'expert-qualification-labels.raw');await durable(path,bytes);
 const stat=await lstat(path);assert(stat.isFile()&&!stat.isSymbolicLink()&&stat.uid===process.getuid!()&&(stat.mode&0o7777)===0o600&&stat.size===bytes.length,'Native labels need the local private worker directory');
 return{path,sha256:sha(bytes)};
}
/** Research fixture only: restore actual warmup weights through the existing previous ABI, explicitly discarding its Adam. */
async function freezeFixtureParent(worker:any,cap:any){
 validateExpertCurrentSnapshot(worker,cap);await cap.assertUnchanged();assert.equal(cap.step,1);assert.equal(worker.status.optimizer_step,1);
 const fields=[cap.width,JSON.stringify(cap.owner.path),cap.owner.file_sha256,...KEYS.flatMap(key=>[cap.owner.tensors[key].byte_offset,cap.owner.tensors[key].sha256])];
 const result=await worker.command('previous '+fields.join(' '));assert.equal(result.response.operation,'previous');assert.equal(result.response.optimizer_step,0);assert.equal(worker.status.optimizer_step,0);await cap.assertUnchanged();
 return{operation:'previous',optimizer_step_before:1,optimizer_step_after:0,weights:cap.owner.tensors,actual_response:result.response};
}

async function saveState(cap:any,output:string,name:string){
 assert(cap.owner&&cap.mps_rng,'A real resident expert and MPS state are required');
 const path=join(output,name+'.safetensors');await copyFile(cap.owner.path,path,constants.COPYFILE_EXCL);
 const cpu=join(output,name+'-cpu.raw'),mps=join(output,name+'-mps.raw');
 await copyFile(cap.cpu_rng.path,cpu,constants.COPYFILE_EXCL);await copyFile(cap.mps_rng.path,mps,constants.COPYFILE_EXCL);
 return{path,tensors:cap.owner.tensors,width:cap.width??cap.previous_width,step:cap.step??0,cpu_rng:cpu,mps_rng:mps};
}
async function saveTrace(directory:string,output:string,name:string){
 const index=await readFile(join(directory,'trace.tsv'),'utf8'),raw=await readFile(join(directory,'trace.raw'));
 const rows=index.trim().split('\n').map(line=>{const [name,dtype,shape,offset,bytes]=line.split('\t');return{name,dtype,shape,offset:Number(offset),bytes:Number(bytes)};});
 let at=0;for(const row of rows){assert(Number.isSafeInteger(row.bytes)&&row.bytes>0&&row.offset===at);at+=row.bytes;}assert.equal(at,raw.length);
 const path=join(output,name+'.raw');await writeFile(path,raw,{flag:'wx',mode:0o600});return{path,rows};
}
function equalState(a:any,b:any){
 assert(a.width===b.width&&a.step===b.step&&a.optimizer_empty===b.optimizer_empty);
 assert.deepEqual(Object.keys(a.owner.tensors).sort(),Object.keys(b.owner.tensors).sort());
 for(const key of Object.keys(a.owner.tensors))assert.equal(a.owner.tensors[key].sha256,b.owner.tensors[key].sha256,key);
 assert.equal(a.cpu_rng.sha256,b.cpu_rng.sha256);assert.equal(a.mps_rng.sha256,b.mps_rng.sha256);
}
/** This intentionally performs restoration qualification once, not in daily per-sample training. */
export async function probeNativeExpertPreservation(engine:any,request:{output:string;sample:Record<string,any>;width:number;signal?:AbortSignal}){
 assert(Number.isSafeInteger(request.width)&&request.width>=32&&request.width<=512);
 const began=performance.now(),hardware={cpu_models:[...new Set(cpus().map(cpu=>cpu.model))],logical_cpu_count:cpus().length,host_total_memory_bytes:totalmem(),arch:arch(),platform:platform()};
 return engine.withSourceQuiescent(async(guard:any,control:any)=>{
  const peer=borrowNativeEnginePeer(engine,guard),worker:any=peer.worker,original=await worker.snapshotInitial('expert-qualification-original');
  const wasEnabled=worker.isBranchEnabled,caps:any[]=[];let restored=false,result:any;assert.equal(typeof wasEnabled,'boolean');
  const restore=async()=>{await worker.restoreExpertInitialSnapshot(original);if(peer.previous)await worker.adoptAcceptedPrevious(peer.previous);await worker.setBranchEnabled(wasEnabled);restored=true;};
  try{
   assert(peer.core,'Qualification requires the current accepted core');
   const prepared=await control.prepareTrainingSample(request.sample,{signal:request.signal});
   assert(prepared.native_labels_compatible&&prepared.ids.length>=2&&prepared.ids.length<=64);
   const descriptorFor=async(name:string,ids:number[],maximum=8)=>writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:name,work:worker.options.directory,prompt_ids:ids,messages_sha256:sha(JSON.stringify(request.sample)),max_tokens:maximum,raw_steps_limit:0,core_version:peer.core.research_label,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
   const descriptor=await descriptorFor('expert-qualification-training',prepared.ids);
   const fields={...descriptor,labels:await privateTrainingLabels(worker.options.directory,prepared.labels),signal:request.signal};
   await worker.retainCoreBackbone({...descriptor,core_tensor_sha256:peer.core.sha256});
   const firstAnswer=prepared.labels.findIndex((n:number)=>n!==-100);assert(firstAnswer>0);
   const prompt=prepared.ids.slice(0,firstAnswer),generationDescriptor=await descriptorFor('expert-qualification-prompt',prompt);
   const generate=async(name:string)=>{const response=await worker.runDescriptor({...generationDescriptor,name,signal:request.signal});const tokens=response.events.filter((e:any)=>e.event==='token').map((e:any)=>e.token),complete=response.events.find((e:any)=>e.event==='complete');assert(complete&&tokens.length<=8);return{tokens,finish_reason:complete.finish_reason,pid:worker.pid};};
   const sourceBefore=await generate('expert-qualification-source-before');
   // A private nonzero neural fixture exercises the parent-teacher graph even with no accepted expert.
   await worker.seedFresh();await worker.initializeCandidate(request.width,.0002,null);await worker.setBranchEnabled(true);
   const warmup=await worker.trainDescriptor({...fields,learning_rate:.0002,steps:1,name:'expert-qualification-parent-warmup'});assert.equal(warmup.step,1);
   const warmupState=await worker.snapshotCurrent('expert-qualification-parent-warmup-state');caps.push(warmupState);const fixtureParentReceipt=await freezeFixtureParent(worker,warmupState);
   const parent=await worker.snapshotInitial('expert-qualification-parent');caps.push(parent);for(const key of KEYS)assert.equal(parent.owner.tensors[key].sha256,warmupState.owner.tensors[key].sha256);await worker.restoreExpertInitialSnapshot(parent);await worker.setBranchEnabled(true);
   const originalState=await saveState(parent,request.output,'previous');const previous=await generate('expert-qualification-before');
   await worker.clearPreservation();await worker.setBranchEnabled(false);const coreTeacher=await worker.capturePreservation({...fields,name:'expert-qualification-teacher-core'});assert.equal(coreTeacher.response.preservation_rows,1);
   await worker.setBranchEnabled(true);const parentTeacher=await worker.capturePreservation({...fields,name:'expert-qualification-teacher-parent'});assert.equal(parentTeacher.response.preservation_rows,2);
   await worker.seedFresh();const fresh=await worker.initializeCandidate(request.width,.0002,null);assert.equal(fresh.response.preservation_rows,2);await worker.setBranchEnabled(true);
   const initial=await worker.snapshotCurrent('expert-qualification-cp0');caps.push(initial);
   const initialState=await saveState(initial,request.output,'initial');
   const steps:any[]=[],trainedStates:any[]=[];
   for(let step=1;step<=2;step++){
    request.signal?.throwIfAborted();const trained=await worker.trainDescriptor({...fields,preservation:true,audit:true,learning_rate:.0002,steps:1,name:'expert-qualification-step'+step});
    assert.equal(trained.step,step);const cap=await worker.snapshotCurrent('expert-qualification-cp'+step);caps.push(cap);trainedStates.push(cap);
    const trace=await saveTrace(trained.directory,request.output,'step'+step),names=new Set(trace.rows.map(row=>row.name));
    for(const suffix of ['ce','loss','preservation_kl','preservation_teacher','preservation_logits',...KEYS.flatMap((_,i)=>['gradient','clipped','weight','m','v','adamstep'].map(key=>key+'_'+i))])assert(names.has('step'+step+'_'+suffix),'Missing actual full trace '+suffix);
    steps.push({step,trace,state:await saveState(cap,request.output,'step'+step+'-state')});
   }
   const candidate=await generate('expert-qualification-candidate');
   const restoredStep=await worker.restoreExpertCurrentSnapshot(trainedStates[0]);assert.equal(restoredStep.response.preservation_rows,2,'Actual optimizer resume must retain both fixed teachers');
   const repeat=await worker.trainDescriptor({...fields,preservation:true,audit:true,learning_rate:.0002,steps:1,name:'expert-qualification-resume-step2'});assert.equal(repeat.step,2);
   const resumed=await worker.snapshotCurrent('expert-qualification-resumed');caps.push(resumed);equalState(trainedStates[1],resumed);
   const resumeState=await saveState(resumed,request.output,'resumed');
   await worker.restoreExpertInitialSnapshot(parent);await worker.setBranchEnabled(true);const after=await generate('expert-qualification-after');
   assert.deepEqual(after.tokens,previous.tokens);assert.equal(after.finish_reason,previous.finish_reason);
   await restore();const sourceAfter=await generate('expert-qualification-source-after');assert.deepEqual(sourceAfter.tokens,sourceBefore.tokens);assert.equal(sourceAfter.finish_reason,sourceBefore.finish_reason);
   const checked=await worker.snapshotInitial('expert-qualification-restored');caps.push(checked);assert.equal(!!checked.owner,!!original.owner);
   if(original.owner)for(const key of KEYS)assert.equal(checked.owner.tensors[key].sha256,original.owner.tensors[key].sha256);
   assert.equal(checked.cpu_rng.sha256,original.cpu_rng.sha256);assert.equal(checked.mps_rng.sha256,original.mps_rng.sha256);
   await worker.setBranchEnabled(wasEnabled);await worker.clearPreservation();
   const corePath=join(request.output,'accepted-core.bf16.raw');await copyFile(peer.core.path,corePath,constants.COPYFILE_EXCL);
   result={binding:{identity:peer.base.identity,core_tensor_sha256:peer.core.sha256,graph_binary_sha256:peer.build.binary_sha256,source_build_sha256:peer.build.proof.source_build_sha256,hidden:2048,width:request.width},
    core:{path:corePath,tensor_sha256:peer.core.sha256},input:{sample:request.sample,ids:prepared.ids,labels:prepared.labels,prompt_ids:prompt},previous_state:originalState,initial_state:initialState,steps,resumed_state:resumeState,
    fixture_parent:{kind:'private-ce-warmup',accepted:false,optimizer_updates:1},fixture_parent_receipt:fixtureParentReceipt,source_generation:{before:sourceBefore,after:sourceAfter},generation:{previous,candidate,after},native_restore:{previous_generation_equal:true,optimizer_resume_equal:true,source_rng_restored:true},optimizer_updates_executed:3,compared_steps:2};
  }finally{
   let cleanupError:unknown;try{if(!restored)await restore();await worker.setBranchEnabled(wasEnabled);}catch(error){cleanupError=error;}
   const closed=await Promise.allSettled([...caps.toReversed(),original].map(cap=>cap.close()));
   const failures=closed.filter((row):row is PromiseRejectedResult=>row.status==='rejected').map(row=>row.reason);if(cleanupError)failures.unshift(cleanupError);if(failures.length)throw new AggregateError(failures,'Expert qualification restoration/close failed');
  }
  await worker.close();
  const closeACK=worker.status;assert(closeACK.event==='closed'&&closeACK.parent_restored===true&&closeACK.resident_parameters===0&&Number.isSafeInteger(closeACK.ordinal)&&closeACK.ordinal>0&&worker.isClosed);
  // The same native process compares/restores its in-memory parent RNG before this ACK; close() also verifies exit/PGID.
  result.native_restore.parent_rng_restored=true;result.native_restore.owners_closed=worker.isClosed;
  result.native_close={pid:worker.pid,response:closeACK,is_closed:worker.isClosed};
  result.measurement={scope:'native qualification including private parent preparation, two compared updates, resume replay and close',elapsed_seconds:(performance.now()-began)/1000,hardware,precision:'BF16 graph and expert; FP32 full-vocabulary preservation KL',seed:42,sample:{id:request.sample.id??null,purpose:request.sample.purpose??null,split:request.sample.purpose==='training'?'TRAIN':'unspecified'},memory:{peak_measured:false,method:'No peak sampler; host_total_memory_bytes is physical system RAM, not process usage'},performance_qualified:false,memory_qualified:false};return result;
 },request.signal);
}

if(import.meta.main){
 const {values}=parseArgs({args:process.argv.slice(2),strict:true,options:{output:{type:'string'},sample:{type:'string'},width:{type:'string',default:'128'},home:{type:'string',default:join(homedir(),'Library/Application Support/Luban')},'check-only':{type:'boolean'}}});
 assert(values.output&&values.sample);const output=resolve(values.output),home=resolve(values.home!),sample=JSON.parse(await readFile(resolve(values.sample),'utf8')),width=Number(values.width);
 assert(Number.isSafeInteger(width)&&width>=32&&width<=512&&typeof sample.prompt==='string'&&typeof sample.answer==='string'&&sample.answer.trim());
 const resources=nativeEvolutionRuntimeResources();
 if(values['check-only']){console.log(JSON.stringify({ready:true,identity:resources.base.identity,width,output,model_loaded:false,GPU:false,requires_accepted_core:true,synthetic_parent_fixture:true,compared_steps:2,optimizer_updates_including_resume:3}));process.exit(0);}
 assert(!relative('/Volumes/移动数据库/lubanllm',output).startsWith('..')&&output!=='/Volumes/移动数据库/lubanllm');
 const mount=await Bun.$`mount`.text();assert(mount.includes(' on /Volumes/移动数据库 ('),'Research volume is not mounted');
 await mkdir(output,{mode:0o700});const connection=await applicationConnection(home),leaseID=randomUUID(),controller=new AbortController();
 let child:ReturnType<typeof Bun.spawn>|undefined;
 const cancel=()=>{controller.abort(new Error('Expert qualification cancelled'));child?.kill('SIGINT');};process.on('SIGINT',cancel);process.on('SIGTERM',cancel);
 const command=async(path:string,body:any)=>{const response=await fetch(connection.url+path,{method:'POST',headers:{authorization:'Bearer '+connection.token,'content-type':'application/json'},body:JSON.stringify(body),proxy:null} as RequestInit&{proxy:null});assert(response.ok,'Application lease failed');return response.json() as Promise<any>;};
 await writeFile(join(output,'lease.json'),JSON.stringify({lease_id:leaseID,acquire_attempted:false,acquired:false,was_ready:null,released:false})+'\n',{flag:'wx',mode:0o600});
 const leaseState=async(value:Record<string,unknown>)=>privateJSON(join(output,'lease.json'),{lease_id:leaseID,...value});
 const runBegan=performance.now();let acquireAttempted=false,acquired=false;
 let lease:any,opened:Awaited<ReturnType<typeof openNativeEvolutionEngine>>|undefined,result:any,error:unknown,engineClosed=false,leaseReleased=false,qualification:any;
 const closeEngine=async()=>{if(opened&&!engineClosed){await opened.close();engineClosed=true;}};
 try{
  await leaseState({acquire_attempted:true,acquired:false,was_ready:null,released:false});acquireAttempted=true;lease=await command('/internal/native-evolution/acquire',{lease_id:leaseID});acquired=true;await leaseState({acquire_attempted:true,acquired:true,was_ready:lease.was_ready,released:false});assert.equal(lease.lease_id,leaseID);assert.equal(lease.identity,resources.base.identity);
  await mkdir(join(home,'native-evolution-work'),{recursive:true,mode:0o700});const work=await mkdtemp(join(home,'native-evolution-work/expert-qualification-'));
  opened=await openNativeEvolutionEngine({home,work,model:await realpath(lease.model),root:lease.root,store:lease.store,identity:lease.identity});
  result=await probeNativeExpertPreservation(opened.engine,{output,sample,width,signal:controller.signal});await closeEngine();controller.signal.throwIfAborted();
  const nativePath=join(output,'native-result.json');
  await writeFile(nativePath,JSON.stringify({schema:'native-expert-preservation-probe-v1',passed:true,model:lease.model,engine_closed:true,lease_held:true,product_admitted:false,teacher_calls:0,...result},null,2)+'\n',{flag:'wx',mode:0o600});
  child=Bun.spawn([resolve('.venv/bin/python'),resolve('scripts/verify-native-expert-preservation.py'),'--native-result',nativePath,'--output',join(output,'original-hf')],{env:{...process.env,PYTHONPATH:resolve('src')},stdout:'inherit',stderr:'inherit'});
  assert.equal(await child.exited,0,'Original HF comparison failed');child=undefined;controller.signal.throwIfAborted();
  qualification=await issueNativeExpertQualification(nativePath,join(output,'original-hf'));await writeFile(join(output,'qualification.json'),JSON.stringify(qualification,null,2)+'\n',{flag:'wx',mode:0o600});
 }catch(value){error=value;}
 finally{
  const cleanupError=(value:unknown)=>console.error(String(value).replaceAll(connection.token,'[private-token]'));
  if(opened&&!engineClosed)await closeImportOwner(closeEngine,cleanupError);
  let wasReady=false;
  if(acquireAttempted)await closeImportOwner(async()=>{const released=await command('/internal/native-evolution/release',{lease_id:leaseID});assert(released.released||!acquired&&released.was_ready===false);wasReady=released.was_ready===true;leaseReleased=true;await leaseState({acquire_attempted:true,acquired,was_ready:wasReady,released:true});},cleanupError);
  if(leaseReleased&&wasReady)try{await command('/api/model/reload',{});}catch(value){error=error?new AggregateError([error,value]):value;}
  process.removeListener('SIGINT',cancel);process.removeListener('SIGTERM',cancel);
 }
 const report={schema:'native-expert-preservation-run-v1',passed:!error&&engineClosed&&leaseReleased&&!!qualification,engine_closed:engineClosed,lease_released:leaseReleased,product_admitted:false,teacher_calls:0,measurement:{...result?.measurement,scope:'complete native and original-HF qualification plus lease cleanup',elapsed_seconds:(performance.now()-runBegan)/1000,performance_qualified:false,memory_qualified:false},...(error?{error:String(error).replaceAll(connection.token,'[private-token]')}:{} )};
 await writeFile(join(output,'result.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx',mode:0o600});console.log(JSON.stringify({...report,output}));if(error)throw error;
}
