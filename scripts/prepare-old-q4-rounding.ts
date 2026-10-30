/** CPU-only bounded preparation from existing captures. No model, lease, Adam or validation data. */
import assert from 'node:assert/strict';
import {open,mkdir,readFile,writeFile,lstat} from 'node:fs/promises';
import {constants} from 'node:fs';
import {resolve,join,dirname} from 'node:path';
import {parseArgs} from 'node:util';
import {spawn} from 'node:child_process';
import {oldSpanCheckpointInputs} from './old-span-diagnostic.ts';
import {coreCheckpointField as field,coreCheckpointMetadata as metadata} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {sha} from '../.native-migration/research/native-full-training/io.ts';

export function q4Activation(raw:Buffer){
 assert.equal(raw.length,(2048+6144+2048)*2,'Actual Q4 tail h/act/down byte layout differs');
 const result=Buffer.alloc(6144*4);
 for(let i=0;i<6144;i++){result.writeUInt32LE(raw.readUInt16LE((2048+i)*2)*65536,i*4);assert(Number.isFinite(result.readFloatLE(i*4)));}
 return result;
}
async function main(){
 const{values}=parseArgs({args:process.argv.slice(2),strict:true,options:{'old-result':{type:'string'},'q4-result':{type:'string'},output:{type:'string'},method:{type:'string',default:'feedback2'}}});
 assert(['feedback2','gptq-bf16','gptq-bf16-refine'].includes(values.method!),'Only fixed feedback2 or GPTQ diagnostics are supported');
 assert(values['old-result']&&values['q4-result']&&values.output,'--old-result --q4-result --output are required');
 const old=JSON.parse(await readFile(values['old-result'],'utf8')),q4=JSON.parse(await readFile(values['q4-result'],'utf8'));
 assert(old.passed&&old.engine_closed&&old.lease_released&&old.product_pointers_unchanged&&old.source_run==='84dbd6c8f41a'&&old.optimizer_steps===0);
 assert(q4.schema==='native-q4-replay-tail-v1'&&q4.passed&&q4.worker_closed&&q4.lease_released&&q4.product_pointers_unchanged&&q4.steps===0&&q4.precision==='mixed-q4-g16');
 const evidence=JSON.parse(await readFile('research/native_evolution_final_policy_20261017.json','utf8'));
 const tree=JSON.parse(await readFile(join(evidence.reader_directory,'checkpoint-tree.json'),'utf8')),payload=field(tree,'payload')!,specs=oldSpanCheckpointInputs(payload),report=metadata(field(payload,'report'));
 assert(old.candidate_sha256===specs[2].sha256&&old.previous_sha256===(field(payload,'learning_reference')as any).sha256&&old.identity===report.identity);
 assert(q4.loaded.base_identity===old.identity&&q4.loaded.original_down_sha256===old.previous_sha256&&q4.loaded.core_loaded===false&&q4.loaded.execution_skip_mask===0&&q4.loaded.execution_exit_after===24&&q4.loaded.layer_paging_enabled===false);
 const authorized=evidence.records.groups.replay_training;assert.equal(authorized.length,16);
 assert.deepEqual(q4.captured.map((r:any)=>r.id),authorized.map((r:any)=>r.id));assert(authorized.every((r:any)=>r.purpose==='training'&&r.training_allowed));
 const output=resolve(values.output);assert(output.startsWith('/Volumes/移动数据库/lubanllm/experiments/'),'Prepared candidate must be saved on the authorized external experiment volume');await mkdir(output,{mode:0o700});
 const held:{path:string;handle:Awaited<ReturnType<typeof open>>;stat:any}[]=[];
 const input=async(path:string,bytes:number)=>{const handle=await open(path,constants.O_RDONLY|constants.O_NOFOLLOW);const owner={path,handle,stat:await handle.stat({bigint:true})};held.push(owner);assert(owner.stat.isFile()&&owner.stat.size===BigInt(bytes));return owner;};
 const began=performance.now();let child:ReturnType<typeof spawn>|undefined,aborted=false;
 const cancel=()=>{aborted=true;child?.kill('SIGTERM');};process.on('SIGINT',cancel);process.on('SIGTERM',cancel);
 try{
  assert.equal(old.projection.positions_per_sample.reduce((a:number,b:number)=>a+b,0),47);
  const original=await input(join(old.work,'known-old-predictors.raw'),47*6144*4),buffers=[Buffer.from(await original.handle.readFile())];
  let count=0;
  for(const row of q4.captured){
   assert(!aborted);assert(row.result.complete&&row.result.tokens.at(-1)===248044&&row.frames.length===row.result.tokens.length);
   for(const[index,frame]of row.frames.entries()){
    assert.equal(frame.step,index);assert.equal(frame.target,row.result.tokens[index]);assert.equal(frame.prediction_position,row.request.tokens.length-1+index);
    const path=resolve(frame.tail_path);assert.equal(dirname(path),resolve(dirname(values['q4-result']),'traces'));
    const raw=await input(path,20480);buffers.push(q4Activation(Buffer.from(await raw.handle.readFile())));count++;
   }
  }
  assert.equal(count,78);const matrix=join(output,'old-and-q4-states.f32.raw');await writeFile(matrix,Buffer.concat(buffers),{flag:'wx',mode:0o600});
  const paths=[];for(const spec of specs){const path=join(evidence.reader_directory,spec.file);await input(path,spec.bytes);paths.push(path);}
  assert(!aborted);const binary=resolve('build/old-span-diagnostic/project-old-span'),args=[...paths,matrix,'2048','6144',String(specs[3].shape[1]),'125',join(output,'projection'),values.method,'47'];
  const run=child=spawn(binary,args,{stdio:['ignore','pipe','pipe']});let stdout='',stderr='';const timer=setTimeout(()=>run.kill('SIGTERM'),values.method!=='feedback2'?720000:120000);
  try{const code=await new Promise((done,reject)=>{run.once('error',reject);run.once('close',done);run.stdout!.on('data',b=>{stdout+=b;if(stdout.length>65536)run.kill('SIGTERM');});run.stderr!.on('data',b=>{stderr+=b;if(stderr.length>65536)run.kill('SIGTERM');});});assert.equal(code,0,stderr);assert(!aborted);}
  finally{clearTimeout(timer);}
  for(const owner of held){const current=await owner.handle.stat({bigint:true}),named=await lstat(owner.path,{bigint:true});for(const key of ['dev','ino','size','mtimeNs','ctimeNs']as const){assert.equal(current[key],owner.stat[key]);assert.equal(named[key],owner.stat[key]);}}
  const metrics=JSON.parse(stdout);assert(metrics.optimizer_steps===0&&metrics.new_validation_used===false);
  if(values.method==='feedback2'){assert(metrics.schema==='known-old-q4-rounding-feedback-v1'&&metrics.fixed_feedback_steps===2);assert.deepEqual(metrics.rounds.map((r:any)=>r.iteration),[0,1,2]);}
  else{const refined=values.method==='gptq-bf16-refine';assert(metrics.schema===(refined?'known-old-q4-gptq-bf16-refine-v1':'known-old-q4-gptq-bf16-v1')&&metrics.damping===0.01&&metrics.block_size===128&&metrics.column_order==='natural'&&metrics.selected_iteration===(refined?2:1));if(refined)assert.equal(metrics.fixed_compensation_passes,2);assert.deepEqual(metrics.rounds.map((r:any)=>r.iteration),refined?[0,1,2]:[0,1]);}
  const path=join(output,'projection/candidate.raw'),raw=await readFile(path);assert.equal(raw.length,25165824);
  const result={schema:'prepared-old-q4-rounding-candidate-v1',passed:true,source_run:old.source_run,source_master_sha256:specs[1].sha256,source_candidate_sha256:old.candidate_sha256,previous_sha256:old.previous_sha256,identity:old.identity,
   core:{path,sha256:sha(raw),bytes:raw.length},metrics,known_old_ids:old.projection.protection_ids,q4_training_ids:q4.captured.map((r:any)=>r.id),
   old_result:resolve(values['old-result']),q4_result:resolve(values['q4-result']),model_loaded:false,GPU:false,optimizer_steps:0,product_admitted:false,new_validation_used:false,
   eligible_for_model_diagnostic:values.method==='feedback2'?metrics.feedback_improved===true:metrics.compensation_improved===true,seconds:(performance.now()-began)/1000};
  await writeFile(join(output,'result.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});console.log(JSON.stringify(result));
 }finally{process.off('SIGINT',cancel);process.off('SIGTERM',cancel);const closed=await Promise.allSettled(held.map(v=>v.handle.close()));assert(closed.every(v=>v.status==='fulfilled'),'CPU capture owners failed to close');}
}
if(import.meta.main)await main();
