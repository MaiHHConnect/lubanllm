import {test,expect} from 'bun:test';
import {mkdtemp,writeFile,readFile,rm,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {issueNativeExpertQualification} from '../scripts/issue-native-expert-qualification.ts';

async function fixture(){
 const root=await mkdtemp(join(tmpdir(),'luban-expert-issuer-')),oracle=join(root,'hf');await mkdir(oracle);
 const binding={identity:'1'.repeat(64),core_tensor_sha256:'2'.repeat(64),graph_binary_sha256:'3'.repeat(64),source_build_sha256:'4'.repeat(64),hidden:2048,width:32};
 const restored={previous_generation_equal:true,optimizer_resume_equal:true,parent_rng_restored:true,owners_closed:true};
 const pairs:any[]=[],steps:any[]=[];
 for(const step of [1,2]){
  const names=[`step${step}_preservation_teacher`,...['ce','preservation_kl','loss'].map(x=>`step${step}_${x}`),...[0,1,2].flatMap(i=>['gradient','weight','m','v','adamstep'].map(k=>`step${step}_${k}_${i}`))];
  const pieces:Buffer[]=[],rows:any[]=[];let offset=0;
  for(const [i,name]of names.entries()){
   const parameter=name.match(/_(gradient|weight|m|v)_([012])$/),shape=parameter?(parameter[2]==='2'?[2048,32]:[32,2048]):name.endsWith('preservation_teacher')?[1,248320]:[];
   const dtype=parameter?'bf16':'f32',elements=shape.reduce((a,b)=>a*b,1),bytes=Buffer.alloc(elements*(parameter?2:4));
   for(let at=0;at<bytes.length;at+=parameter?2:4){if(parameter)bytes.writeUInt16LE(0x3f80,at);else bytes.writeFloatLE(step+i,at);}
   rows.push({name,dtype,shape:shape.join(',')||'-',offset,bytes:bytes.length});offset+=bytes.length;pieces.push(bytes);
   const alias=name.endsWith('preservation_teacher')?(step===1?'teacher_core':'teacher_expert'):name;
   pairs.push({name:alias,elements,max_abs_error:0,atol:0,rtol:0,allclose_passed:true});await writeFile(join(oracle,alias+'.raw'),bytes);
  }
  // Real TracePack emits these per-command names again in both step traces.
  for(const [name,count] of [['entry_rng',5056],['final_rng',5056],['final_mps_rng',44]] as const){const bytes=Buffer.alloc(count,step);rows.push({name,dtype:'u8',shape:String(count),offset,bytes:count});offset+=count;pieces.push(bytes);}
  const raw=Buffer.concat(pieces);
  const path=join(root,'step'+step+'.raw');await writeFile(path,raw);steps.push({step,trace:{path,rows}});
 }
 const generation=Object.fromEntries(['previous','candidate','after'].map(label=>[label,{tokens:[8,9],finish_reason:'stop'}]));
 const native={schema:'native-expert-preservation-probe-v1',passed:true,engine_closed:true,lease_held:true,teacher_calls:0,product_admitted:false,compared_steps:2,optimizer_updates_executed:3,binding,input:{labels:[-100,42]},steps,generation,fixture_parent:{kind:'private-ce-warmup',accepted:false,optimizer_updates:1},source_generation:{before:generation.previous,after:generation.after},native_restore:{...restored,source_rng_restored:true},native_close:{pid:123,response:{event:'closed',ordinal:39,parent_restored:true,resident_parameters:0},is_closed:true}};
 const evidence={...binding,schema:'native-expert-preservation-graph-v1',evidence_id:oracle,dtype:'BF16',learning_rate:.0002,preservation_weight:5,actual_steps:2,teacher_modes:['core','expert'],pairs,restored,actual_original_hf:true};
 const result={passed:true,actual_original_hf:true,compared_steps:2,optimizer_updates_including_resume:3,teacher_calls:0,product_admitted:false,pairs:pairs.length,generations:['previous','candidate','after'].map(label=>({label,tokens:[8,9],native_tokens:[8,9],original_Runtime_generate_tokens:true,original_HF:true}))};
 const nativePath=join(root,'native.json');await writeFile(nativePath,JSON.stringify(native));await writeFile(join(oracle,'evidence.json'),JSON.stringify(evidence));await writeFile(join(oracle,'result.json'),JSON.stringify(result));await writeFile(join(oracle,'close.json'),JSON.stringify(restored));
 return{root,oracle,nativePath,native,evidence,result};
}
test('issuer rereads native/HF trace bytes and rejects forged metadata, source, restore, teacher, generation and bytes',async()=>{
 const f=await fixture();try{
  expect((await issueNativeExpertQualification(f.nativePath,f.oracle)).pairs.length).toBe(38);
  const evidencePath=join(f.oracle,'evidence.json');
  for(const mutate of [(e:any)=>e.core_tensor_sha256='a'.repeat(64),(e:any)=>e.pairs[0].atol=.1,(e:any)=>e.pairs[0].elements=2,(e:any)=>e.restored.parent_rng_restored=false,(e:any)=>e.pairs.splice(0,1)]){
   const value=structuredClone(f.evidence);mutate(value);await writeFile(evidencePath,JSON.stringify(value));await expect(issueNativeExpertQualification(f.nativePath,f.oracle)).rejects.toThrow();
  }
  await writeFile(evidencePath,JSON.stringify(f.evidence));
  const bytesPath=join(f.oracle,'step2_weight_2.raw'),bytes=await readFile(bytesPath);await writeFile(bytesPath,Buffer.alloc(4));await expect(issueNativeExpertQualification(f.nativePath,f.oracle)).rejects.toThrow('bytes differ');await writeFile(bytesPath,bytes);
  const result=structuredClone(f.result);result.generations[1]!.tokens=[7];await writeFile(join(f.oracle,'result.json'),JSON.stringify(result));await expect(issueNativeExpertQualification(f.nativePath,f.oracle)).rejects.toThrow();await writeFile(join(f.oracle,'result.json'),JSON.stringify(f.result));
  for(const field of ['parent_restored','resident_parameters','event']){
   const changed=structuredClone(f.native);(changed.native_close.response as any)[field]=field==='event'?'done':field==='resident_parameters'?1:false;
   await writeFile(f.nativePath,JSON.stringify(changed));await expect(issueNativeExpertQualification(f.nativePath,f.oracle)).rejects.toThrow();
  }
  for(const name of ['final_rng','step1_ce','step2_ce']){
   const changed=structuredClone(f.native);changed.steps[0].trace.rows.at(-1).name=name;
   await writeFile(f.nativePath,JSON.stringify(changed));await expect(issueNativeExpertQualification(f.nativePath,f.oracle)).rejects.toThrow();
  }
  f.native.lease_held=false;await writeFile(f.nativePath,JSON.stringify(f.native));await expect(issueNativeExpertQualification(f.nativePath,f.oracle)).rejects.toThrow();
 }finally{await rm(f.root,{recursive:true,force:true});}
});
