/** Research-only OLD constraint diagnostic, not TRAIN permission or independent validation. */
import assert from 'node:assert/strict';
import {readFile,lstat} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {coreCheckpointField as field} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {openCheckpointRaw} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
import {durable,sha} from '../.native-migration/research/native-full-training/io.ts';
import {prepareBaselineSamples,baselineGenerationRow} from '../.native-migration/research/native-ordinary-baseline/rows.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-peer-descriptor-v2/descriptor.ts';

export function selectOldPredictorInputs(raw:Buffer,labels:number[],width:number) {
 assert(labels.length>=2&&width>0&&raw.length===labels.length*width*2);
 const positions=labels.slice(1).map((value,i)=>value===-100?-1:i).filter(i=>i>=0);assert(positions.length>0);
 const output=Buffer.alloc(positions.length*width*4);
 for(const [row,position]of positions.entries())for(let col=0;col<width;col++){
  const bits=raw.readUInt16LE((position*width+col)*2);output.writeUInt32LE(bits*65536,(row*width+col)*4);
  assert(Number.isFinite(output.readFloatLE((row*width+col)*4)),'Nonfinite OLD predictor input');
 }
 return{raw:output,positions:positions.length};
}

export async function replaceDiagnosticCore(descriptor:{path:string},destination:string,core:{path:string;sha256:string;bytes:number},parameter:string){
 const replacement:Record<string,string>={core_version:'core-'+core.sha256.slice(0,12),core_path:core.path,core_file_sha256:core.sha256,core_tensor_sha256:core.sha256,core_tensor_key:parameter,core_offset:'0',core_bytes:String(core.bytes)};
 const seen=new Set<string>(),text=(await readFile(descriptor.path,'utf8')).split('\n').map(line=>{const parts=line.split('\t');if(parts[0]==='config'&&Object.hasOwn(replacement,parts[1]!)){assert.equal(parts.length,3);assert(!seen.has(parts[1]!));seen.add(parts[1]!);return 'config\t'+parts[1]+'\t'+replacement[parts[1]!];}return line;}).join('\n');
 assert.equal(seen.size,7);await durable(destination,text);return{path:destination,sha256:sha(text)};
}

export function oldSpanCheckpointInputs(payload:any){
 const native=field(payload,'native_state')!,specs=[field(native,'reference'),field(native,'master'),field(native,'candidate'),field(native,'basis')] as any[];
 for(const[entry,dtype]of [[specs[0],'f32'],[specs[1],'f32'],[specs[2],'bf16']]as const){assert(entry?.kind==='tensor'&&entry.dtype===dtype);assert.deepEqual(entry.shape,[2048,6144]);assert.equal(entry.bytes,2048*6144*(dtype==='bf16'?2:4));}
 const learning=field(payload,'learning_reference') as any;assert(learning?.kind==='tensor'&&learning.dtype==='bf16');assert.deepEqual(learning.shape,[2048,6144]);
 // The actual FINAL contains separate transport copies; compare its existing bindings, not file names.
 assert.equal(specs[1].sha256,(field(payload,'master') as any)?.sha256);
 assert.equal(specs[3].sha256,(field(payload,'basis') as any)?.sha256);
 const basis=specs[3];assert(basis?.kind==='tensor'&&basis.dtype==='f32'&&basis.shape[0]===6144&&basis.shape[1]>0&&basis.shape[1]<6144);
 return specs;
}

export async function projectKnownOldSpan(o:{worker:any;owners:any;preparedRows:any[];originalRows:any[];payload:any;directory:string;work:string;signal:AbortSignal;coreSHA:string}){
 assert.equal(o.originalRows.length,8);assert.deepEqual(o.originalRows.map(r=>r.id),Array.from({length:8},(_,i)=>'product-ordinary-'+i));
 assert(o.originalRows.every(r=>r.training_allowed===false&&r.purpose==='validation'),'OLD must remain read-only known protection, never TRAIN');
 const states:Buffer[]=[],positions:number[]=[];
 for(const row of o.originalRows){
  o.signal.throwIfAborted();const p=o.preparedRows.find(p=>p.row.id===row.id);assert(p&&p.row.answer.trim());
  const sample=JSON.stringify(p.row),sampleSHA=sha(sample);
  const output=await o.worker.exportOrdinaryTrainingTail({...p.original,labels:p.labelFile,sample_sha256:sampleSHA,sample_raw_sha256:sampleSHA,name:'old-span-'+positions.length,signal:o.signal});
  const manifest=JSON.parse(await readFile(join(output.directory,'manifest.json'),'utf8'));
  assert(manifest.schema==='luban-ordinary-training-tail-v1'&&manifest.pid===o.worker.pid&&manifest.core_tensor_sha256===o.coreSHA&&manifest.tokens===p.ids.length&&manifest.optimizer_steps===0&&!manifest.expert_used&&!manifest.branch_enabled&&manifest.stop_before_last_down);
  assert.equal(manifest.descriptor_sha256,p.original.sha256);assert.equal(manifest.labels_sha256,p.labelFile.sha256);
  const tensor=manifest.tensors.find((t:any)=>t.name==='tail_x');assert(tensor&&tensor.file==='tail_x.raw'&&tensor.dtype==='bfloat16');assert.deepEqual(tensor.shape,[1,p.ids.length,6144]);assert.equal(tensor.bytes,p.ids.length*6144*2);
  const held=await openCheckpointRaw(join(output.directory,tensor.file),tensor.bytes);o.owners.own(held,()=>held.handle.close());
  const selected=selectOldPredictorInputs(Buffer.from(await held.handle.readFile()),p.labels,6144);await unchangedNativeFile(join(output.directory,tensor.file),held.handle,held.stat);
  states.push(selected.raw);positions.push(selected.positions);
 }
 const matrix=join(o.work,'known-old-predictors.raw');await durable(matrix,Buffer.concat(states));
 const specs=oldSpanCheckpointInputs(o.payload),basis=specs[3];
 const inputs=[];
 for(const entry of specs){const path=join(o.directory,entry.file),held=await openCheckpointRaw(path,entry.bytes);o.owners.own(held,()=>held.handle.close());inputs.push({path,held});}
 const binary=resolve('build/old-span-diagnostic/project-old-span'),stat=await lstat(binary);assert(stat.isFile()&&!stat.isSymbolicLink());
 const destination=join(o.work,'old-span-projected'),args=[...inputs.map(v=>v.path),matrix,'2048','6144',String(basis.shape[1]),String(positions.reduce((a,b)=>a+b,0)),destination];
 o.signal.throwIfAborted();const child=spawn(binary,args,{stdio:['ignore','pipe','pipe']});let stdout='',stderr='';
 const stop=()=>child.kill('SIGTERM');o.signal.addEventListener('abort',stop,{once:true});const timeout=setTimeout(stop,120000);if(o.signal.aborted)stop();
 try{const code=await new Promise<number|null>((done,reject)=>{child.once('error',reject);child.once('close',done);child.stdout.on('data',b=>{stdout+=b;if(stdout.length>65536)child.kill('SIGTERM');});child.stderr.on('data',b=>{stderr+=b;if(stderr.length>65536)child.kill('SIGTERM');});});assert.equal(code,0,stderr);o.signal.throwIfAborted();}
 finally{clearTimeout(timeout);o.signal.removeEventListener('abort',stop);}
 for(const{path,held}of inputs)await unchangedNativeFile(path,held.handle,held.stat);
 const metrics=JSON.parse(stdout);assert(metrics.schema==='known-old-span-projection-v1'&&metrics.optimizer_steps===0&&!metrics.product_admitted&&metrics.source==='checkpoint_fp32_master_delta_then_one_bf16_cast');
 const path=join(destination,'candidate.raw'),held=await openCheckpointRaw(path,25165824);o.owners.own(held,()=>held.handle.close());
 const raw=Buffer.from(await held.handle.readFile());assert.equal(raw.length,25165824);await unchangedNativeFile(path,held.handle,held.stat);
 return{core:{path,sha256:sha(raw),bytes:raw.length},metrics,checkpoint_inputs:specs.map(v=>({file:v.file,dtype:v.dtype,sha256:v.sha256})),protection_ids:o.originalRows.map(r=>r.id),positions_per_sample:positions,scope:'known OLD constraint adaptation; not independent VALID; no CE/KL/Adam training',new_validation_used_for_projection:false};
}

export async function diagnosticIdentityGeneration(o:{control:any;peer:any;guard:any;core:{path:string;sha256:string;bytes:number};parameter:string;records:any[];work:string;signal:AbortSignal;scope?:'training'|'validation'}){
 assert.equal(o.records.length,o.scope==='training'?20:o.scope==='validation'?10:30);if(o.scope)assert(o.records.every(row=>row.purpose===o.scope));const observations:ReturnType<typeof baselineGenerationRow>[]=[];
 for(const sample of prepareBaselineSamples(o.records)){
  o.signal.throwIfAborted();const id='old-span-identity-'+(o.scope??'all')+'-'+observations.length;let prepared=false,finished=false;
  try{
   const prompt=await o.control.tokenizer({operation:'generation_prepare',generation_id:id,purpose:'quality',messages:sample.messages,max_tokens:Number(sample.check.max_new_tokens)},o.signal);prepared=true;
   assert(prompt.event==='generation_prepared'&&prompt.generation_id===id&&prompt.thinking==='off'&&Array.isArray(prompt.ids));
   const descriptor=await writePeerDescriptor({runtime:o.peer.source.runtime,guard:o.guard,owner:o.peer.owner,core_owner:o.peer.core,request_id:id,work:o.peer.worker.options.directory,prompt_ids:prompt.ids,messages_sha256:sha(JSON.stringify(sample.messages)),max_tokens:Number(sample.check.max_new_tokens),raw_steps_limit:0,core_version:null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
   const current=await replaceDiagnosticCore(descriptor,join(o.work,id+'.tsv'),o.core,o.parameter);
   const actual=await o.peer.worker.runDescriptor({...current,name:id,signal:o.signal});
   const tokens=actual.events.filter((e:any)=>e.event==='token').map((e:any)=>e.token),completion=actual.events.find((e:any)=>e.event==='complete');assert(completion&&tokens.length<=Number(sample.check.max_new_tokens));
   for(const[tokenIndex,token]of tokens.entries()){const frame=await o.control.tokenizer({operation:'generation_append',generation_id:id,step:tokenIndex,token},o.signal);assert(frame.event==='generation_decoded'&&frame.generation_id===id&&frame.step===tokenIndex&&frame.token===token);}
   const decoded=await o.control.tokenizer({operation:'generation_finish',generation_id:id});finished=true;
   assert(decoded.event==='generation_decode_finished'&&decoded.generation_id===id);assert.deepEqual(decoded.tokens,tokens);assert.equal(typeof decoded.text,'string');
   observations.push(baselineGenerationRow(sample,{tokens,text:decoded.text,finish_reason:completion.finish_reason,thinking:'off',sampling_seed:'42'}));
  }finally{if(prepared&&!finished)await o.control.tokenizer({operation:'generation_finish',generation_id:id});}
 }
 return observations;
}

/** Shared admission for the two research observers; never a publication capability. */
export function assertPreparedProjection(preparedProjection:any,binding:{run:string;candidate:string;master:string;previous:string;identity:string}){
 const p=preparedProjection,m=p.metrics;assert(p.schema==='prepared-old-q4-rounding-candidate-v1'&&p.passed&&p.eligible_for_model_diagnostic&&p.optimizer_steps===0&&p.new_validation_used===false&&!p.product_admitted);
 assert(p.source_run===binding.run&&p.source_candidate_sha256===binding.candidate&&p.source_master_sha256===binding.master&&p.previous_sha256===binding.previous&&p.identity===binding.identity);
 assert(m.new_validation_used===false);
 if(m.schema==='known-old-q4-rounding-feedback-v1'){assert(m.fixed_feedback_steps===2&&m.feedback_improved);assert.deepEqual(m.rounds.map((r:any)=>r.iteration),[0,1,2]);}
 else{const refined=m.schema==='known-old-q4-gptq-bf16-refine-v1';assert((refined||m.schema==='known-old-q4-gptq-bf16-v1')&&m.damping===0.01&&m.block_size===128&&m.column_order==='natural'&&m.selected_iteration===(refined?2:1)&&m.compensation_improved);if(refined)assert.equal(m.fixed_compensation_passes,2);assert.deepEqual(m.rounds.map((r:any)=>r.iteration),refined?[0,1,2]:[0,1]);}
 assert(m.old_positions===47&&m.q4_train_positions===78&&m.rounds.every((r:any)=>['reconstruction_frobenius','known_old_frobenius','q4_train_frobenius'].every(key=>Number.isFinite(r[key])&&r[key]>=0)));
 const selected=m.rounds[m.selected_iteration],initial=m.rounds[0];assert(selected&&selected.reconstruction_frobenius<initial.reconstruction_frobenius&&selected.known_old_frobenius<=initial.known_old_frobenius&&selected.q4_train_frobenius<=initial.q4_train_frobenius);
 if(m.schema==='known-old-q4-gptq-bf16-refine-v1'){const previous=m.rounds[1];assert(selected.reconstruction_frobenius<previous.reconstruction_frobenius&&selected.known_old_frobenius<=previous.known_old_frobenius&&selected.q4_train_frobenius<=previous.q4_train_frobenius);}
 assert.equal(selected.reconstruction_frobenius,Math.min(...m.rounds.map((r:any)=>r.reconstruction_frobenius)));assert(p.core.bytes===25165824&&/^[a-f0-9]{64}$/.test(p.core.sha256));
}
