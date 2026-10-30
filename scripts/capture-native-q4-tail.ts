/** Research-only full-depth Q4 trace on already-correct authorized replay TRAIN histories. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {readFile,writeFile,mkdir,realpath,lstat} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {homedir} from 'node:os';
import {parseArgs} from 'node:util';
import type {Writable} from 'node:stream';
import {applicationConnection} from './run-native-identity-round.ts';
import {evaluateGeneration} from '../backend/generation-checks.ts';

const V19='/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/identity-native450-v19';
const EOS=248044,VOCAB=248320,TAIL=10240;
export function replayTraceCases(evidence:any,baseline:any,all:boolean){
 assert(Array.isArray(baseline)&&baseline.length===2);assert.deepEqual(baseline[0],baseline[1]);
 const b=baseline[0];assert(b.precision==='mixed-q4-g16'&&b.core===null&&b.seed===42&&b.temperature===0);
 const authorized=evidence.records.groups.replay_training;assert.equal(authorized.length,16);
 const rows=authorized.map((material:any)=>{
  assert(material.approved&&material.training_allowed&&material.purpose==='training'&&material.provenance.evolution_role==='replay');
  const row=b.rows.find((r:any)=>r.id===material.id);assert(row&&row.passed&&row.complete&&row.finish_reason==='stop'&&!row.branch_enabled&&!row.reference_used);
  assert(evaluateGeneration(row.answer,material.provenance.generation_check,true).passed);
  for(const ids of [row.input_tokens,row.tokens])assert(Array.isArray(ids)&&ids.length&&ids.every((n:any)=>Number.isSafeInteger(n)&&n>=0&&n<VOCAB));
  assert(row.tokens.length<=256&&row.tokens.at(-1)===EOS&&!row.tokens.slice(0,-1).includes(EOS));
  return {id:row.id,tokens:row.input_tokens,targets:row.tokens,answer:row.answer,forced_tokens:row.tokens.slice(0,-1),prediction_positions:row.tokens.map((_:number,i:number)=>row.input_tokens.length-1+i)};
 });
 assert.equal(new Set(rows.map((r:any)=>r.id)).size,16);
 return all?rows:rows.filter((r:any)=>r.id==='q4-id-replay-training-8');
}
export function validateTailFrame(event:any,row:any,step:number,directory:string,tail:Buffer,logits:Buffer){
 assert(event.event==='trace'&&event.id===row.id&&event.step===step&&event.committed_tokens===row.tokens.length+step);
 assert(event.tail_only===true&&event.layer===23&&event.tail_values===TAIL&&event.vocab===VOCAB&&!('layers_path'in event));
 assert.equal(event.tail_path,join(directory,`${row.id}-step${String(step).padStart(4,'0')}.tail.bf16`));
 assert.equal(event.logits_path,join(directory,`${row.id}-step${String(step).padStart(4,'0')}.logits.bf16`));
 assert.equal(tail.length,TAIL*2);assert.equal(logits.length,VOCAB*2);
 let top=0,max=-Infinity;const bits=new DataView(new ArrayBuffer(4));
 for(const [name,bytes]of [['tail',tail],['logits',logits]] as const)for(let i=0;i<bytes.length;i+=2){const u=bytes.readUInt16LE(i);assert((u&0x7f80)!==0x7f80,'Nonfinite '+name);if(name==='logits'){bits.setUint32(0,u<<16,true);const value=bits.getFloat32(0,true);if(value>max){max=value;top=i/2;}}}
 assert.equal(event.top1,top);assert.equal(top,row.targets[step],'Actual Q4 fixed-history prediction differs from saved baseline');
 return {step,prediction_position:row.prediction_positions[step],target:row.targets[step],tail_path:event.tail_path,logits_path:event.logits_path};
}

async function main(){
 const {values}=parseArgs({args:process.argv.slice(2),strict:true,options:{output:{type:'string'},worker:{type:'string',default:resolve('build/native-evolution/worker.js')},home:{type:'string',default:join(homedir(),'Library/Application Support/Luban')},'all-replay':{type:'boolean'},'check-only':{type:'boolean'}}});
 assert(values.output,'--output must name a new external directory');const output=resolve(values.output!),home=resolve(values.home!);
 assert(output.startsWith(V19+'/')&&await realpath('/Volumes/移动数据库')==='/Volumes/移动数据库');
 const evidence=JSON.parse(await readFile(resolve('research/native_evolution_final_policy_20261017.json'),'utf8'));
 const baseline=JSON.parse(await readFile(join(V19,'q4-candidate-v1/baseline.json'),'utf8')),cases=replayTraceCases(evidence,baseline,!!values['all-replay']);assert(cases.length);
 const worker=await import(pathToFileURL(resolve(values.worker!)).href),resources=worker.nativeEvolutionRuntimeResources(),q4=await worker.nativeEvolutionQ4Resources();
 assert.equal(resources.base.identity,baseline[0].identity);assert.equal(resources.base.model_manifest_sha256,baseline[0].manifest_sha256);
 if(values['check-only']){console.log(JSON.stringify({ready:true,GPU:false,model_loaded:false,cases:cases.map((r:any)=>({id:r.id,predictions:r.targets.length})),binary:q4.binary,precision:'mixed-q4-g16',full_layers:24,steps:0,output,application_lease_required:true}));return;}
 await mkdir(output,{mode:0o700});const traces=join(output,'traces');await mkdir(traces,{mode:0o700});
 const connection=await applicationConnection(home),leaseID=randomUUID(),abort=new AbortController();
 const command=async(path:string,body:any={})=>{const response=await fetch(connection.url+path,{method:'POST',headers:{authorization:'Bearer '+connection.token,'content-type':'application/json'},body:JSON.stringify(body),redirect:'manual',proxy:null} as RequestInit & {proxy:null});const value:any=await response.json();assert(response.ok,typeof value?.detail==='string'?value.detail:'Application lease operation failed');return value;};
 const pointers=async(store:string)=>Promise.all(['active-core.json','active.json','active-router.json'].map(async name=>{try{return await readFile(join(store,name));}catch(e:any){if(e.code==='ENOENT')return null;throw e;}}));
 let lease:any,child:ReturnType<typeof spawn>|undefined,exited:Promise<{code:number|null;signal:string|null}>|undefined,cancel:Writable|undefined,error:unknown,loaded:any,before:(Buffer|null)[]|undefined,closed=false,released=false,unchanged=false,stderr='';
 const stop=()=>{abort.abort(new Error('Q4 tail capture cancelled'));cancel?.write(Buffer.from([1]));};process.on('SIGINT',stop);process.on('SIGTERM',stop);
 const began=performance.now(),captured:any[]=[];
 try{
  lease=await command('/internal/native-evolution/acquire',{lease_id:leaseID});assert(lease.lease_id===leaseID&&lease.identity===resources.base.identity);before=await pointers(lease.store);abort.signal.throwIfAborted();
  const args=[await realpath(lease.model),resources.base.model_manifest_sha256,'3','--mixed-q4-g16','--research-trace='+traces];
  await writeFile(join(output,'protocol.json'),JSON.stringify({binary:q4.binary,args,cases,precision:'mixed-q4-g16',core:null,seed:42,temperature:0,skip:0,exit_after:24,cache_reuse:false,layout:{h:[0,2048],act:[2048,8192],down:[8192,10240],logits:[0,VOCAB]},dtype:'bf16-le',position:'P-1+s predicts completion[s]; only completion[:-1] is forwarded; final EOS is predicted, not forwarded',training:false,product_admitted:false},null,2)+'\n',{flag:'wx'});
  child=spawn(q4.binary,args,{stdio:['pipe','pipe','pipe','pipe']});cancel=child.stdio[3] as Writable;cancel.on('error',()=>{});
  const lines=createInterface({input:child.stdout!})[Symbol.asyncIterator]();let processError:Error|undefined;
  child.on('error',e=>{processError=e;});child.stdin!.on('error',e=>{processError=e;});child.stderr!.on('data',data=>{stderr=(stderr+String(data)).slice(-32768);});
  exited=new Promise(resolve=>child!.once('close',(code,signal)=>resolve({code,signal})));
  const next=async()=>{let timer:ReturnType<typeof setTimeout>|undefined;try{const item=await Promise.race([lines.next(),new Promise<never>((_,reject)=>{timer=setTimeout(()=>{stop();reject(new Error('Q4 trace event timeout'));},180000);})]);assert(!item.done,processError?.message??'Q4 worker exited before trace completed');const event=JSON.parse(item.value);assert(event.event!=='error','Q4 trace: '+event.error);return event;}finally{clearTimeout(timer);}};
  loaded=await next();assert(loaded.event==='loaded'&&loaded.base_identity===resources.base.identity&&loaded.manifest_sha256===resources.base.model_manifest_sha256&&loaded.inference_precision==='mixed-q4-g16'&&loaded.core_loaded===false&&loaded.core_file_sha256===null&&loaded.execution_skip_mask===0&&loaded.execution_exit_after===24&&loaded.layer_paging_enabled===false&&loaded.packed_version===0&&loaded.residual_rank===0);
  for(const row of cases){abort.signal.throwIfAborted();const request={id:row.id,tokens:row.tokens,forced_tokens:row.forced_tokens,max_tokens:row.targets.length,eos_ids:[EOS],temperature:0,seed:42,cache_reuse:false,tail_only:true};child.stdin!.write(JSON.stringify(request)+'\n');const frames=[];
   for(let step=0;step<row.targets.length;step++){const event=await next();abort.signal.throwIfAborted();for(const kind of ['tail','logits']){const path=join(traces,`${row.id}-step${String(step).padStart(4,'0')}.${kind}.bf16`);assert.equal(event[kind+'_path'],path);const stat=await lstat(path);assert(stat.isFile()&&!stat.isSymbolicLink());}
    frames.push(validateTailFrame(event,row,step,traces,await readFile(event.tail_path),await readFile(event.logits_path)));}
   const result=await next();assert(result.event==='result'&&result.id===row.id&&result.trace_only===true&&result.complete===true&&result.finish_reason==='trace'&&result.cached_prompt_tokens===0&&result.prompt_tokens===row.tokens.length&&result.generated_tokens===row.targets.length);assert.deepEqual(result.tokens,row.targets);assert.deepEqual(result.forced_tokens,row.forced_tokens);
   captured.push({id:row.id,request,frames,result});console.log(JSON.stringify({id:row.id,predictions:frames.length,seconds:(performance.now()-began)/1000}));
  }
 }catch(e){error=e;}finally{
  if(child&&exited){child.stdin!.end();const timeout=setTimeout(()=>child?.kill('SIGTERM'),30000);const result=await exited;clearTimeout(timeout);closed=true;if(result.code!==0||result.signal)error??=new Error('Q4 worker failed close: '+JSON.stringify(result));cancel?.destroy();}
  try{if(lease&&before){const after=await pointers(lease.store);assert(before.every((v,i)=>v===null?after[i]===null:after[i]!==null&&v.equals(after[i]!)));unchanged=true;}}catch(e){error??=e;}
  try{const r=await command('/internal/native-evolution/release',{lease_id:leaseID});assert(r.released||!lease&&!r.was_ready);released=true;if(r.was_ready)await command('/api/model/reload');}catch(e){error??=e;}
  process.off('SIGINT',stop);process.off('SIGTERM',stop);
 }
 const result={schema:'native-q4-replay-tail-v1',passed:!error,precision:'mixed-q4-g16',loaded,captured,worker_closed:closed,lease_released:released,product_pointers_unchanged:unchanged,steps:0,product_admitted:false,seconds:(performance.now()-began)/1000,training_math_equivalence:false,...(error?{error:String(error).replaceAll(connection.token,'[private-token]')}:{})};
 await writeFile(join(output,'result.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});await writeFile(join(output,'stderr.log'),stderr,{flag:'wx'});console.log(JSON.stringify({passed:result.passed,seconds:result.seconds,cases:captured.length,worker_closed:closed,lease_released:released}));if(error)throw error;
}
if(import.meta.main)await main();
