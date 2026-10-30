import {expect,test} from 'bun:test';
import {mkdtemp,readFile,rm,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {selectOldPredictorInputs,replaceDiagnosticCore,oldSpanCheckpointInputs,assertPreparedProjection} from '../scripts/old-span-diagnostic.ts';
import {coreCheckpointField as field} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {q4Activation} from '../scripts/prepare-old-q4-rounding.ts';
import assert from 'node:assert/strict';

test('actual pre-generated candidate consumer rejects wrong source and one-group regression before lease',async()=>{
 const p={schema:'prepared-old-q4-rounding-candidate-v1',passed:true,eligible_for_model_diagnostic:true,optimizer_steps:0,new_validation_used:false,product_admitted:false,source_run:'v19',source_candidate_sha256:'candidate',source_master_sha256:'master',previous_sha256:'original',identity:'base',core:{bytes:25165824,sha256:'a'.repeat(64)},metrics:{schema:'known-old-q4-rounding-feedback-v1',fixed_feedback_steps:2,feedback_improved:true,new_validation_used:false,old_positions:47,q4_train_positions:78,selected_iteration:1,rounds:[{iteration:0,reconstruction_frobenius:5,known_old_frobenius:3,q4_train_frobenius:4},{iteration:1,reconstruction_frobenius:2.5,known_old_frobenius:1.5,q4_train_frobenius:2},{iteration:2,reconstruction_frobenius:4,known_old_frobenius:2.4,q4_train_frobenius:3.2}]}};
 const run=(v:any)=>assertPreparedProjection(v,{run:'v19',candidate:'candidate',previous:'original',identity:'base',master:'master'});
 expect(()=>run(p)).not.toThrow();
 for(const key of ['source_run','source_candidate_sha256','source_master_sha256','previous_sha256','identity'])expect(()=>run({...p,[key]:'wrong'})).toThrow();
 const bad=structuredClone(p);bad.metrics.rounds[1]!.known_old_frobenius=4;expect(()=>run(bad)).toThrow();
 const nan=structuredClone(p);nan.metrics.rounds[1]!.known_old_frobenius=NaN;expect(()=>run(nan)).toThrow();
 const gptq:any=structuredClone(p);Object.assign(gptq.metrics,{schema:'known-old-q4-gptq-bf16-v1',damping:0.01,block_size:128,column_order:'natural',compensation_improved:true,rounds:gptq.metrics.rounds.slice(0,2)});expect(()=>run(gptq)).not.toThrow();
 for(const [key,value]of [['damping',0.02],['block_size',64],['column_order','optimized'],['selected_iteration',0],['compensation_improved',false]])expect(()=>run({...gptq,metrics:{...gptq.metrics,[key as string]:value}})).toThrow();
 const worse=structuredClone(gptq);worse.metrics.rounds[1].q4_train_frobenius=5;expect(()=>run(worse)).toThrow();
 const refined=structuredClone(gptq);Object.assign(refined.metrics,{schema:'known-old-q4-gptq-bf16-refine-v1',selected_iteration:2,fixed_compensation_passes:2});refined.metrics.rounds.push({iteration:2,reconstruction_frobenius:1.25,known_old_frobenius:.75,q4_train_frobenius:1});expect(()=>run(refined)).not.toThrow();
 const regressed=structuredClone(refined);regressed.metrics.rounds[2].known_old_frobenius=1.6;expect(()=>run(regressed)).toThrow();refined.metrics.fixed_compensation_passes=3;expect(()=>run(refined)).toThrow();
});

test('actual diagnostic checks TRAIN first and never consumes NEW VALID after TRAIN regression',async()=>{
 const source=await readFile('scripts/verify-native-retention-loss.ts','utf8'),start=source.indexOf("event('projected_identity_train');"),end=source.indexOf("\n   }\n   event('restore_original')",start);expect(start>0&&end>start).toBe(true);
 const code=new Bun.Transpiler({loader:'ts'}).transformSync('async function run(){let identity,identityTrainingPreserved;'+source.slice(start,end)+'\nreturn{identity,identityTrainingPreserved};}');
 const make=new Function('event','observation','diagnosticIdentityGeneration','control','peer','guard','projection','resources','evidence','work','controller','nativeRetentionLossGuard','baseline','projected',code+';return run;');
 for(const [success,retained]of [[false,true],[true,false],[true,true]]){
  const calls:string[]=[],run=make(()=>{},(fn:any)=>fn(),async(o:any)=>{calls.push(o.scope);return Array.from({length:o.scope==='training'?20:10},(_,i)=>({passed:success||i!==0,complete:true}));},{},{},{},{core:{}},{base:{core_parameter:'down'}},{records:{groups:{new_training:[],new_validation:[]}}},'/unused',new AbortController(),()=>({passed:retained}),[],[]);
  const result=await run();expect(calls).toEqual(success&&retained?['training','validation']:['training']);expect(result.identityTrainingPreserved).toBe(success);expect(result.identity.length).toBe(success&&retained?30:20);
 }
});

test('actual Q4 act slice excludes h and down and rejects nonfinite/incorrect layout',()=>{
 const raw=Buffer.alloc(20480);raw.writeUInt16LE(0x7f80,0);raw.writeUInt16LE(0x3fc0,2048*2);raw.writeUInt16LE(0xc000,8191*2);raw.writeUInt16LE(0x7f80,8192*2);
 const act=q4Activation(raw);expect(act.length).toBe(6144*4);expect(act.readFloatLE(0)).toBe(1.5);expect(act.readFloatLE(act.length-4)).toBe(-2);
 expect(()=>q4Activation(raw.subarray(0,20478))).toThrow();raw.writeUInt16LE(0x7f80,2048*2);expect(()=>q4Activation(raw)).toThrow();
});

test('actual FINAL metadata selects native FP32 reference, not the BF16 learning reference',async()=>{
 const evidence=JSON.parse(await readFile('research/native_evolution_final_policy_20261017.json','utf8'));
 const tree=JSON.parse(await readFile(join(evidence.reader_directory,'checkpoint-tree.json'),'utf8')),payload=field(tree,'payload')!;
 const inputs=oldSpanCheckpointInputs(payload),native=field(payload,'native_state')!;
 expect(inputs[0]).toEqual(field(native,'reference'));expect(inputs[0].dtype).toBe('f32');expect(inputs[0].bytes).toBe(50331648);
 expect((field(payload,'learning_reference')as any).dtype).toBe('bf16');expect(inputs[0].file).not.toBe((field(payload,'learning_reference')as any).file);
 const bad=structuredClone(payload),state=field(bad,'native_state')!;assertDict(state);const reference=state.entries.find(([key])=>key.kind==='string'&&key.value==='reference')!;reference[1]=structuredClone(field(payload,'learning_reference')!);
 expect(()=>oldSpanCheckpointInputs(bad)).toThrow();
});
function assertDict(value:any):asserts value is {entries:[any,any][]}{if(value.kind!=='dict')throw new Error('Expected actual checkpoint dict');}

test('known OLD capture selects causal answer/EOS predictors, excluding prefix and final unpredicted position',()=>{
 const x=Buffer.alloc(4*2*2);[1,2,3,4,5,6,7,8].forEach((v,i)=>{const f=Buffer.alloc(4);f.writeFloatLE(v);x.writeUInt16LE(f.readUInt16LE(2),i*2);});
 const result=selectOldPredictorInputs(x,[-100,-100,10,99],2);
 expect(result.positions).toBe(2);expect(Array.from({length:4},(_,i)=>result.raw.readFloatLE(i*4))).toEqual([3,4,5,6]);
 expect(()=>selectOldPredictorInputs(x,[-100,-100,-100,-100],2)).toThrow();
 expect(()=>selectOldPredictorInputs(x,[-100,99],2)).toThrow();
 x.writeUInt16LE(0x7f80,2*2);expect(()=>selectOldPredictorInputs(x,[-100,-100,10,99],2)).toThrow();
});

test('diagnostic descriptor substitutes all seven actual source fields without changing generation settings',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'old-span-descriptor-'));try{
  const path=join(dir,'original.tsv'),names=['core_version','core_path','core_file_sha256','core_tensor_sha256','core_tensor_key','core_offset','core_bytes'];
  await writeFile(path,names.map(n=>'config\t'+n+'\told').join('\n')+'\nconfig\ttemperature\t0\nconfig\tsampling_seed\t42\n');
  const result=await replaceDiagnosticCore({path},join(dir,'next.tsv'),{path:'/actual/candidate.raw',sha256:'a'.repeat(64),bytes:16},'layers.23.down');
  const text=await readFile(result.path,'utf8');expect(text).toContain('config\tcore_path\t/actual/candidate.raw');expect(text).toContain('config\tcore_bytes\t16');expect(text).toContain('config\tsampling_seed\t42');expect(text).not.toContain('\told');
  expect(await readFile(path,'utf8')).toContain('\told');
  await writeFile(path,'config\tcore_path\told\n');await expect(replaceDiagnosticCore({path},join(dir,'bad.tsv'),{path:'/actual/candidate.raw',sha256:'a'.repeat(64),bytes:16},'layers.23.down')).rejects.toThrow();
 }finally{await rm(dir,{recursive:true,force:true});}
});
