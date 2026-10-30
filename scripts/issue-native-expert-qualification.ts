/** Research-only issuer: compare actual trace bytes again; never accept a passed flag alone. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {validateNativeExpertGraphEvidence,type NativeExpertGraphEvidence} from '../backend/native-expert-qualification.ts';

const DTYPES:Record<string,number>={bf16:2,f32:4,i64:8,u8:1};
export async function issueNativeExpertQualification(nativePath:string,oracleDirectory:string):Promise<NativeExpertGraphEvidence>{
 const n=JSON.parse(await readFile(nativePath,'utf8'));
 const [value,oracle,closed]=await Promise.all([ 'evidence.json','result.json','close.json'].map(async file=>JSON.parse(await readFile(join(oracleDirectory,file),'utf8'))));
 assert(n.schema==='native-expert-preservation-probe-v1'&&n.passed===true&&n.engine_closed===true&&n.lease_held===true);
 assert(n.teacher_calls===0&&n.product_admitted===false&&n.compared_steps===2&&n.optimizer_updates_executed===3);
 assert.deepEqual(n.native_restore,{previous_generation_equal:true,optimizer_resume_equal:true,source_rng_restored:true,parent_rng_restored:true,owners_closed:true});
 const nativeClose=n.native_close;assert(nativeClose?.is_closed===true&&Number.isSafeInteger(nativeClose.pid)&&nativeClose.pid>0);
 assert(nativeClose.response.event==='closed'&&nativeClose.response.parent_restored===true&&nativeClose.response.resident_parameters===0&&Number.isSafeInteger(nativeClose.response.ordinal)&&nativeClose.response.ordinal>0);
 assert(oracle.passed===true&&oracle.actual_original_hf===true&&oracle.compared_steps===2&&oracle.optimizer_updates_including_resume===3&&oracle.teacher_calls===0&&oracle.product_admitted===false);
 assert.deepEqual(closed,{previous_generation_equal:true,optimizer_resume_equal:true,parent_rng_restored:true,owners_closed:true});
 const evidence=validateNativeExpertGraphEvidence(value);assert.deepEqual(evidence.restored,closed);assert.equal(evidence.evidence_id,oracleDirectory);
 for(const [key,item]of Object.entries(n.binding))assert.equal(value[key],item,'Native/HF source binding '+key);
 assert.deepEqual(n.fixture_parent,{kind:'private-ce-warmup',accepted:false,optimizer_updates:1});
 assert.deepEqual(n.source_generation.before.tokens,n.source_generation.after.tokens);assert.equal(n.source_generation.before.finish_reason,n.source_generation.after.finish_reason);
 assert.equal(n.steps.length,2);assert.equal(oracle.pairs,evidence.pairs.length);
 const native=new Map<string,{bytes:Buffer;elements:number;shape:number[];dtype:string}>();
 for(const [index,step] of n.steps.entries()){
  assert.equal(step.step,index+1);const data=await readFile(step.trace.path),localNames=new Set<string>();let offset=0;
  for(const row of step.trace.rows){
   assert(Number.isSafeInteger(row.offset)&&row.offset===offset&&Number.isSafeInteger(row.bytes)&&row.bytes>0&&DTYPES[row.dtype]);
   const shape=row.shape==='-'?[]:row.shape.split(',').map(Number);assert(shape.every((x:number)=>Number.isSafeInteger(x)&&x>0));
   const elements=shape.reduce((a:number,b:number)=>a*b,1);assert.equal(row.bytes,elements*DTYPES[row.dtype]!);assert(!localNames.has(row.name),'Duplicate name within native trace');localNames.add(row.name);
   if(row.name.startsWith('step')){
    assert(row.name.startsWith('step'+step.step+'_'),'Step belongs to a different native trace');assert(!native.has(row.name),'Duplicate native step pair');
    native.set(row.name,{bytes:data.subarray(offset,offset+row.bytes),elements,shape,dtype:row.dtype});
   }else assert(['entry_rng','final_rng','final_mps_rng'].includes(row.name),'Unknown native trace header');
   offset+=row.bytes;
  }
  assert.equal(offset,data.length);
 }
 let initialBytes:Buffer|undefined;
 // Required qualification pairs come from actual command traces; initial tensors retain their native container offsets.
 for(const pair of evidence.pairs){
  assert(pair.max_abs_error===0&&pair.atol===0&&pair.rtol===0&&pair.allclose_passed===true,'Only exact comparisons may be issued');
  if(pair.name.startsWith('initial_')){
   const key=({'initial_gate_weight':'gate.weight','initial_up_weight':'up.weight','initial_down_weight':'down.weight'} as Record<string,string>)[pair.name];assert(key);
   const tensor=n.initial_state.tensors[key];assert(tensor&&tensor.dtype==='bf16');assert.deepEqual(tensor.shape,key==='down.weight'?[2048,n.binding.width]:[n.binding.width,2048]);
   initialBytes??=await readFile(n.initial_state.path);const bytes=initialBytes.subarray(tensor.byte_offset,tensor.byte_offset+tensor.bytes);
   assert.equal(pair.elements*2,bytes.length);assert(bytes.equals(await readFile(join(oracleDirectory,pair.name+'.raw'))),'Initial expert differs');continue;
  }
  assert(/^(teacher_core|teacher_expert|step[12]_[a-z0-9_]+)$/.test(pair.name),'Unexpected pair name');
  const key=pair.name==='teacher_core'?'step1_preservation_teacher':pair.name==='teacher_expert'?'step2_preservation_teacher':pair.name;
  const actual=native.get(key);assert(actual&&pair.elements===actual.elements,'Missing actual native trace '+key);
  const parameter=pair.name.match(/^step[12]_(gradient|clipped|weight|m|v|adamstep)_([012])$/);
  if(parameter){assert.equal(actual.dtype,parameter[1]==='adamstep'?'f32':'bf16');assert.deepEqual(actual.shape,parameter[1]==='adamstep'?[]:parameter[2]==='2'?[2048,n.binding.width]:[n.binding.width,2048]);}
  else if(pair.name.startsWith('teacher_')){assert.equal(actual.dtype,'f32');assert.deepEqual(actual.shape,[n.input.labels.slice(1).filter((v:number)=>v!==-100).length,248320]);}
  else if(/^step[12]_(ce|loss|preservation_kl)$/.test(pair.name)){assert.equal(actual.dtype,'f32');assert.deepEqual(actual.shape,[]);}
  if(actual.dtype==='f32')for(let at=0;at<actual.bytes.length;at+=4)assert(Number.isFinite(actual.bytes.readFloatLE(at)),'Nonfinite actual trace');
  if(actual.dtype==='bf16')for(let at=0;at<actual.bytes.length;at+=2)assert((actual.bytes.readUInt16LE(at)&0x7f80)!==0x7f80,'Nonfinite actual trace');
  assert(actual.bytes.equals(await readFile(join(oracleDirectory,pair.name+'.raw'))),'Original HF/native bytes differ: '+pair.name);
 }
 assert(!native.get('step1_preservation_teacher')!.bytes.equals(native.get('step2_preservation_teacher')!.bytes),'Core and actual parent must differ');
 assert.deepEqual(oracle.generations.map((row:any)=>row.label),['previous','candidate','after']);
 for(const row of oracle.generations){assert(row.original_Runtime_generate_tokens===true&&row.original_HF===true);assert.deepEqual(row.tokens,n.generation[row.label].tokens);assert.deepEqual(row.native_tokens,row.tokens);}
 assert.deepEqual(n.generation.previous.tokens,n.generation.after.tokens);assert.equal(n.generation.previous.finish_reason,n.generation.after.finish_reason);
 return evidence;
}
