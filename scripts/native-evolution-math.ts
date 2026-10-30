/** Isolated one-update migration comparison. Build separately; never part of the application payload. */
import assert from 'node:assert/strict';
import {join,resolve} from 'node:path';
import {mkdir,readFile,open,writeFile,stat} from 'node:fs/promises';
import {constants} from 'node:fs';
import {createHash} from 'node:crypto';
import {openNativeEvolutionEngine,nativeEvolutionRuntimeResources,NativeCore450Owner} from '../backend/native-evolution-worker.ts';
import {exportNativePeerTrainingTail} from '../backend/native-evolution-tail.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {withEvolutionRound,enterEvolutionBranch,assertEvolutionBranch} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {withObservation} from '../.native-migration/research/native-observation-python-parity-v2/scope.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';
import {NativeTrainingMPSProcess,type NativeTrainingMPSRequest} from '../.native-migration/research/native-core450-runner/process.ts';
import {ledgerParse} from '../backend/memory.ts';

const root=resolve(import.meta.dir,'../..'),planPath=join(root,'.native-migration/research/native-core-final-math-actual-v3/protocol-final.json');
const p=JSON.parse(await readFile(planPath,'utf8')),prior=JSON.parse(await readFile(p.preparation.actual.path,'utf8'));
const round=ledgerParse(await readFile(p.preparation.round.path,'utf8'));
const fixture=prior.fixture.directory,original=(await readFile(join(fixture,'manifest.tsv'),'utf8')).trimEnd().split('\n').map(line=>line.split('\t'));
const step=original.find(line=>line[0]==='step'&&line[1]==='1')!;assert(step?.length===5&&step[2]==='new');
const keys=step.slice(3),allRows=[...round.records.groups.new_training,...round.records.groups.replay_training];
const samples=keys.map(key=>allRows[Number(key.slice(1))]);assert(samples.length===2&&samples.every(Boolean));
const selected=original.filter(line=>line[0]==='tensor'&&(!/^r\d+_/.test(line[1]!)||keys.some(key=>line[1]!.startsWith(key+'_'))));
const minimumInput=selected.reduce((n,line)=>n+Number(line[5]),0);
const mode=process.argv[2],work=process.argv[3]?resolve(process.argv[3]):null;
if(mode==='--describe'){
 console.log(JSON.stringify({model:p.model,root:p.root,store:p.config.routing.store,identity:p.config.identity,samples:samples.map((s:any)=>({id:s.id,purpose:s.purpose})),input_bytes:minimumInput,steps:1,product_admitted:false,GPU:false}));
}else{
 assert((mode==='--open-only'||mode==='--native')&&work,'Use --describe, --open-only WORK, or --native WORK');
 process.umask(0o077);await mkdir(work,{mode:0o700});
 const resources=nativeEvolutionRuntimeResources(),opened=await openNativeEvolutionEngine({home:work,work,root:p.root,store:p.config.routing.store,model:p.model,identity:p.config.identity});
 const {engine}=opened,owners=new NativeCore450Owner(),began=performance.now();let failure:unknown,result:any;
 const save=async(name:string,value:unknown)=>writeFile(join(work!,name),JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});
 const sha=(raw:Uint8Array|string)=>createHash('sha256').update(raw).digest('hex');
 // Copy only the exact two-row trainer inputs; no model directory or research snapshot copy.
 const copyBytes=async(source:string,target:string,bytes:number)=>{
  const input=await open(source,constants.O_RDONLY|constants.O_NOFOLLOW),out=await open(target,constants.O_WRONLY|constants.O_CREAT|constants.O_EXCL|constants.O_NOFOLLOW,0o600);
  try{const before=await input.stat({bigint:true});assert.equal(before.size,BigInt(bytes));const buffer=Buffer.alloc(1<<20);for(let at=0;at<bytes;){const n=(await input.read(buffer,0,Math.min(buffer.length,bytes-at),at)).bytesRead;assert(n>0);let written=0;while(written<n)written+=(await out.write(buffer,written,n-written,at+written)).bytesWritten;at+=n;}await out.sync();const after=await input.stat({bigint:true});assert(before.ino===after.ino&&before.size===after.size&&before.mtimeNs===after.mtimeNs&&before.ctimeNs===after.ctimeNs);}
  finally{await input.close();await out.close();}
 };
 try{
  if(mode==='--open-only')result={opened:true,status:engine.status(),GPU:true,optimizer_steps:0};
  else result=await withEvolutionRound(engine,()=>engine.withSourceQuiescent(async(guard,control)=>{
   const peer=borrowNativeEnginePeer(engine,guard);await enterEvolutionBranch(engine,peer.worker);
   const cohort={records:{groups:{new_training:[samples[0]],replay_training:[samples[1]]}},assertUnchanged(){guard.assertQuiescent();}} as unknown as Parameters<typeof exportNativePeerTrainingTail>[2];
   const input=join(work!,'input');await mkdir(input,{mode:0o700});const captures:any[]=[];
   for(const[index,sample]of samples.entries()){
    const prepared=await control.prepareTrainingSample(sample),tail=await exportNativePeerTrainingTail(engine,guard,cohort,sample,prepared,owners);
    captures.push({key:keys[index],sample,prepared,manifest:tail.manifest});
    for(const[name,part]of [['x','tail_x'],['residual','tail_residual'],['ids','ids'],['labels','labels']]){
     const tensor=tail.tensors[part!]!,line=selected.find(line=>line[1]===keys[index]+'_'+name)!;assert(line);line[3]=tensor.shape.join(',');line[5]=String(tensor.bytes);line[6]=tensor.sha256;await copyBytes(tensor.path,join(input,line[4]!),tensor.bytes);
    }
   }
   for(const line of selected.filter(line=>!/^r\d+_/.test(line[1]!)))await copyBytes(join(fixture,line[4]!),join(input,line[4]!),Number(line[5]));
   const lines=original.filter(line=>line[0]==='LUBAN_NATIVE_TRAINING_V3'||line[0]==='config'||line[0]==='row'&&keys.includes(line[1]!)).map(line=>line[0]==='config'&&['steps','checkpoint_steps'].includes(line[1]!)?[line[0],line[1],'1']:line);
   lines.push(...selected,step);const manifest=lines.map(line=>line.join('\t')).join('\n')+'\n';await writeFile(join(input,'manifest.tsv'),manifest,{flag:'wx',mode:0o600});
   const prepared=captures[0].prepared,descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:'math-ce',work:peer.worker.options.directory,prompt_ids:prepared.ids,messages_sha256:sha(Buffer.from(new BigInt64Array(prepared.ids.map(BigInt)).buffer)),max_tokens:1,raw_steps_limit:0,core_version:peer.source.runtime.core_label??null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
   const labels=selected.find(line=>line[1]===keys[0]+'_labels')!;
   const loss=await withObservation({engine,guard,worker:peer.worker,cap:cohort,value:cohort,owners:[peer.owner,peer.source,peer.build,...peer.core?[peer.core]:[]],fast:async()=>{guard.assertQuiescent();peer.worker.assertLive();assertEvolutionBranch(engine,peer.worker);}},()=>peer.worker.lossDescriptor({...descriptor,labels:{path:join(input,labels[4]!),sha256:labels[6]!},name:'math-ce'}));
   const trainer=resources.role('trainer'),spec=(name:string)=>selected.find(line=>line[1]===name)!;
   const request:NativeTrainingMPSRequest={binary:trainer.path,binary_sha256:trainer.sha256,package_receipt:resources.release.path,package_receipt_sha256:resources.release.sha256,...resources.training,input,output:join(work!,'output'),identity:resources.base.identity,device:'mps',core_shape:[2048,6144],manifest_sha256:sha(manifest),rng_mode:'fresh',cpu_rng_sha256:spec('cpu_rng')[6]!,mps_rng_sha256:spec('mps_rng')[6]!,reference_sha256:spec('reference')[6]!,basis_sha256:spec('basis')[6]!,stdoutLog:join(work!,'trainer.stdout'),stderrLog:join(work!,'trainer.stderr')};
   const training=new NativeTrainingMPSProcess(request,guard);owners.own(training,()=>training.cleanup());const actual=await training.run();owners.own(actual.parsed,()=>actual.parsed.close());assert(actual.summary.steps===1&&actual.summary.final_adam_step===1);
   const pin=(v:any)=>({path:v.path,dtype:v.dtype,shape:v.shape,bytes:v.bytes,sha256:v.sha256});
   return{passed:true,model:p.model,core_file:p.core_file,identity:resources.base.identity,baseline_loss:loss.loss,captures,input,output:request.output,request,summary:actual.summary,trace:actual.parsed.trace,state:Object.fromEntries(Object.entries(actual.parsed.state).map(([k,v])=>[k,pin(v)])),cpu_rng:pin(actual.parsed.cpu_rng),mps_rng:pin(actual.parsed.mps_rng),rng_status:actual.parsed.rng_status,pid:actual.pid,cleanup_confirmed:actual.cleanup_confirmed,optimizer_steps:1,product_admitted:false};
  }),async(worker,enabled)=>engine.withSourceQuiescent(async guard=>{assert.equal(borrowNativeEnginePeer(engine,guard).worker,worker);if(worker.isBranchEnabled!==enabled)await worker.setBranchEnabled(enabled);}));
 }catch(error){failure=error;await save('failure.json',{error:String(error),stack:(error as Error).stack,status:engine.status(),GPU:true});}
 finally{await owners.close();await opened.close();}
 if(failure)throw failure;await save('native-result.json',{...result,engine_closed:true,wall_seconds:(performance.now()-began)/1000,GPU:true});console.log(JSON.stringify({work,optimizer_steps:result.optimizer_steps,engine_closed:true}));
}
