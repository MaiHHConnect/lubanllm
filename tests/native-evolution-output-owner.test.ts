import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,open,rm,realpath,lstat} from 'node:fs/promises';
import {join,isAbsolute,normalize} from 'node:path';
import {tmpdir,endianness} from 'node:os';
import {createHash} from 'node:crypto';
import {patchNativeEvolutionTrainingOutput} from '../scripts/native-evolution-output-patches.ts';
import {cloneNativeEvolutionInput,isNativeEvolutionClone,retainNativeEvolutionVerifiedRaw,isNativeEvolutionVerifiedRaw} from '../backend/native-evolution-segment-files.ts';
import {openCheckpointRaw,checkpointRawSHA,nativeCheckpointJSON} from '../.native-migration/backend/native-checkpoint.ts';
import {needNative as need,nativeSHA as sha,unchangedNativeFile as unchanged,readNativeSmallFile as small} from '../.native-migration/backend/native-features.ts';
import {parseJSON} from '../.native-migration/backend/wire.ts';
import {encoded} from '../.native-migration/backend/canonical-json.ts';
const source=await readFile(new URL('../.native-migration/backend/native-training-output.ts',import.meta.url),'utf8');
const replace=(s:string,a:string,b:string)=>{assert.equal(s.split(a).length,2,a);return s.replace(a,b);};
const hash=(v:string|Buffer)=>createHash('sha256').update(v).digest('hex');
const FINAL=['candidate','master','exp_avg','exp_avg_sq','adam_step','reference','basis'];
const PROBES=['ce_logits','teacher_log_probs','student_log_probs','ce_grad','kl_grad','total_grad','clipped_grad','delta','master','candidate','exp_avg','exp_avg_sq','adam_step'];
async function fixture(cloned=true,bad?:'nan'|'sha'|'shape'){
 const root=await realpath(await mkdtemp(join(tmpdir(),'luban-output-owner-'))),input=join(root,'input'),output=join(root,'output'),original=join(root,'original');
 for(const p of [input,output,original])await mkdir(p,{mode:0o700});
 const entries:any={},out:any={},shaReads=new Map<string,number>(),handles=new Map<any,string>();
 async function tensor(name:string,dtype:string,shape:number[],data?:Buffer){const bytes=shape.reduce((a,b)=>a*b,1)*({bf16:2,f32:4,i64:8,u8:1}as any)[dtype],raw=data??Buffer.alloc(bytes);return{name,dtype,shape,file:name+'.raw',bytes,sha256:hash(raw),raw};}
 async function add(name:string,dtype:string,shape:number[]){const spec=await tensor(name,dtype,shape);entries[name]=spec;const from=join(original,spec.file);await writeFile(from,spec.raw,{mode:0o600});if(cloned){const f=await open(from,'r');try{await cloneNativeEvolutionInput({path:from,handle:f,stat:await f.stat({bigint:true}),sha256:spec.sha256},join(input,spec.file));}finally{await f.close();}}else await writeFile(join(input,spec.file),spec.raw,{mode:0o600});}
 for(const n of FINAL)await add(n,n==='candidate'?'bf16':'f32',n==='adam_step'?[]:n==='basis'?[2,1]:[2,2]);
 await add('norm','bf16',[2]);await add('head','bf16',[4,2]);await add('cpu_rng','u8',[5056]);
 for(const row of ['r0','r1'])for(const[suffix,dtype,shape]of [['x','bf16',[1,2,2]],['residual','bf16',[1,2,2]],['ids','i64',[1,2]],['labels','i64',[1,2]]]as const)await add(row+'_'+suffix,dtype,[...shape]);
 const line=(s:any)=>['tensor',s.name,s.dtype,s.shape.length?s.shape.join(','):'-',s.file,s.bytes,s.sha256].join('\t');
 const config={identity:'a'.repeat(64),device:'cpu',steps:'2',norm_eps:'1e-6',train_eos:'1',checkpoint_steps:'2',rng_mode:'resume',training_offset:'0'};
 const manifest=['LUBAN_NATIVE_TRAINING_V3',...Object.entries(config).map(([k,v])=>`config\t${k}\t${v}`),...Object.values(entries).map(line),`row\tr0\tnew\t${'b'.repeat(64)}\t2\t1`,`row\tr1\treplay\t${'c'.repeat(64)}\t2\t1`,'step\t1\tnew\tr0\tr1','step\t2\treplay\tr1\tr1'].join('\n')+'\n';
 await writeFile(join(input,'manifest.tsv'),manifest,{mode:0o600});
 for(const n of PROBES){const logits=n==='ce_logits'||n.endsWith('log_probs'),initial=entries[['ce_grad','kl_grad','total_grad','candidate'].includes(n)?'candidate':FINAL.includes(n)?n:'master'];out['checkpoint000002_'+n]=await tensor('checkpoint000002_'+n,logits?'f32':initial.dtype,logits?[1,4]:initial.shape);}
 for(const n of FINAL)out['final_'+n]=await tensor('final_'+n,entries[n].dtype,entries[n].shape);
 for(const name of ['checkpoint000002_adam_step','final_adam_step']){out[name].raw.writeFloatLE(2);out[name].sha256=hash(out[name].raw);}
 if(bad==='nan'){out.final_master.raw.writeFloatLE(NaN);out.final_master.sha256=hash(out.final_master.raw);}
 if(bad==='sha')out.final_master.sha256='f'.repeat(64);
 if(bad==='shape')out.final_master.shape=[1,4];
 for(const spec of Object.values(out)as any[])await writeFile(join(output,spec.file),spec.raw,{mode:0o600});
 await writeFile(join(output,'tensors.tsv'),['LUBAN_NATIVE_TRAINING_OUTPUT_V3',...Object.values(out).map(line)].join('\n')+'\n',{mode:0o600});
 await writeFile(join(output,'trace.tsv'),'step\tce_role\tce_row\treplay_row\tce\tkl\tpreclip_norm\tloss\twall_seconds\n1\tnew\tr0\tr1\t1\t0\t0\t1\t0.1\n2\treplay\tr1\tr1\t1\t0\t0\t1\t0.1\n',{mode:0o600});
 const request={input,output,manifest_sha256:hash(manifest),binary_sha256:'d'.repeat(64),identity:config.identity,device:'cpu',core_shape:[2,2],reference_sha256:entries.reference.sha256,basis_sha256:entries.basis.sha256};
 await writeFile(join(output,'result.json'),JSON.stringify({abi:'LUBAN_NATIVE_TRAINING_V3',passed_execution:true,manifest_sha256:request.manifest_sha256,binary_sha256:request.binary_sha256,device:'cpu',release_ce_graph:true,reuse_replay_forward:true,publication:false,equivalence_tested:false,all_input_tensors_preserved:true,input_tensor_count:Object.keys(entries).length,steps:2,new_ce_exposures:1,replay_ce_exposures:1,replay_kl_exposures:2,entry_adam_step:0,final_adam_step:2,teacher_calls:0,generation_calls:0,ce_forwards:2,teacher_forwards:2,student_forwards:1,reused_student_forwards:1,forward_total:5}),{mode:0o600});
 const env={createHash,lstat,realpath,isAbsolute,join,normalize,endianness,need,sha,unchanged,small,parseJSON,encoded,nativeCheckpointJSON,isNativeEvolutionClone,retainNativeEvolutionVerifiedRaw,
  openCheckpointRaw:async(path:string,...args:any[])=>{const opened=await (openCheckpointRaw as any)(path,...args);handles.set(opened.handle,path);return opened;},
  checkpointRawSHA:async(handle:any,...args:any[])=>{const path=handles.get(handle)!;shaReads.set(path,(shaReads.get(path)??0)+1);return (checkpointRawSHA as any)(handle,...args);}};
 const patched=patchNativeEvolutionTrainingOutput(source,replace,'unused').replace(/^import .*?;$/gm,'').replaceAll('export ','');
 const parser=new Function(...Object.keys(env),new Bun.Transpiler({loader:'ts'}).transformSync(patched+'\nreturn readNativeTrainingMPSRNGTensors;'))(...Object.values(env));
 return{root,input,output,shaReads,entries,out,run:()=>parser(request,{cpu_sha256:entries.cpu_rng.sha256,mps_sha256:null}),async close(){for(const h of handles.keys())if(h.fd>=0)await h.close();await rm(root,{recursive:true,force:true});}};
}
test('actual parser reuses private cloned inputs, scans each necessary new output once; repeated guards perform zero full reads',async()=>{
 const f=await fixture();try{const parsed=await f.run();const before=[...f.shaReads];expect(before.filter(([p])=>p.startsWith(f.input))).toEqual([]);expect(before.filter(([p])=>p.startsWith(f.output))).toHaveLength(12);expect(before.every(([,n])=>n===1)).toBe(true);
 for(let i=0;i<4;i++)await parsed.assertUnchanged();await parsed.checkpoint(2);expect([...f.shaReads]).toEqual(before);
 const raw=parsed.state.master;expect(await isNativeEvolutionVerifiedRaw(raw)).toBe(true);await writeFile(raw.path,Buffer.alloc(raw.bytes));await expect(parsed.assertUnchanged()).rejects.toThrow();await expect(isNativeEvolutionVerifiedRaw(raw)).rejects.toThrow();await parsed.close();
 }finally{await f.close();}
});
test('unbound independent parser retains full input and numerical probe admission',async()=>{
 const f=await fixture(false);try{const parsed=await f.run();expect([...f.shaReads].filter(([p])=>p.startsWith(f.input))).toHaveLength(18);expect([...f.shaReads].filter(([p])=>p.startsWith(f.output))).toHaveLength(20);expect([...f.shaReads.values()].every(n=>n===1)).toBe(true);await parsed.close();}finally{await f.close();}
});
test('new output nonfinite values, wrong SHA and wrong shape remain rejected by actual parser',async()=>{
 for(const reason of ['nan','sha','shape']as const){const f=await fixture(true,reason);try{await expect(f.run()).rejects.toThrow();}finally{await f.close();}}
});
test('input inode changes and closed new-output FD cannot borrow an earlier binding',async()=>{
 const f=await fixture();try{await writeFile(join(f.input,'head.raw'),Buffer.alloc(16));await expect(f.run()).rejects.toThrow('Private segment clone was replaced or changed');}finally{await f.close();}
 const g=await fixture();try{const parsed=await g.run();await parsed.state.master.handle.close();await expect(isNativeEvolutionVerifiedRaw(parsed.state.master)).rejects.toThrow();await expect(parsed.assertUnchanged()).rejects.toThrow();await parsed.close();}finally{await g.close();}
});
