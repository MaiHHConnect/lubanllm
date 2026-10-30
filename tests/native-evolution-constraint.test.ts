import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,writeFile,rm} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {patchNativeEvolutionConstraint} from '../scripts/native-evolution-constraint-patches.ts';
import {coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {nativePartialCorrectIDs} from '../.native-migration/research/native-core-controller-candidate/progress.ts';
import {CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary} from '../.native-migration/research/native-core-controller-candidate/segments.ts';
const code=await readFile(new URL('../build/native-evolution/worker.js',import.meta.url),'utf8');
const normalize=(s:string)=>s.replace(/\b(assert|readFile|join|finite)\d+\b/g,'$1');
function fn(name:string){const start=code.search(new RegExp('(?:async )?function '+name+'\\(')),end=code.indexOf('\n}',start);assert(start>=0&&end>start);return normalize(code.slice(start,end+2));}
const finite=(v:any)=>{assert(typeof v==='number'&&Number.isFinite(v)&&v>=0);return v;};
const scope={assert,finite,nativePartialCorrectIDs,CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary,exposures:coreNewExposures,eq:pythonMetadataEqual,ordinary:validateOrdinaryCoreHistory,CORE450_SAFETY_POLICY:'closed-book-growth-450-train-safety-v1'};
const safety=new Function(...Object.keys(scope),['validateCore450ConstraintResidual','deriveCore450TrainingCheck','validateCore450TrainingHistory','nextCore450TrainingStep'].map(fn).join('\n')+'\nreturn {validateCore450ConstraintResidual,deriveCore450TrainingCheck,validateCore450TrainingHistory,nextCore450TrainingStep};')(...Object.values(scope));
const productStart=code.indexOf('async function runCore450ProductFromCheckpoint(');assert(productStart>=0);
const start=code.indexOf('          assert',code.indexOf('const residual = await control.coreConstraintResidual(candidate',productStart));
const end=code.indexOf('\n        }',start);assert(start>=0&&end>start);
const block=normalize(code.slice(start,end));
const fixed=block.includes('quantized_frobenius:')?block:patchNativeEvolutionConstraint(block,(b,a,z)=>{assert.equal(b.split(a).length,2);return b.replace(a,z);});
const consumer=(source:string)=>new Function('assert','validateCore450ConstraintResidual','residual','candidate','new_generation','replay_generation','replay_loss','let observed;'+source+';return observed;').bind(null,assert,safety.validateCore450ConstraintResidual);
const consume=consumer(fixed);
// Exact numeric report from v14 step50; native provenance is replayed separately below.
const metrics={quantized_frobenius:0.0038955763448029757,quantized_max_abs:0.00035581123665906489,quantized_relative_to_delta:0.0082160700112581253,master_frobenius:8.7840994922316895e-8,weight_dtype:'torch.bfloat16'};
const record=(id:string)=>({id,answer:'正确',provenance:{generation_check:{type:'exact',expected:'正确',max_new_tokens:16}}});
const proof=(id:string,passed:boolean)=>({id,answer:passed?'正确':'错误',passed,complete:true,branch_enabled:false,reference_used:false});
async function replay(directory:string){
 const meta=JSON.parse(await readFile(join(directory,'constraint.json'),'utf8'));
 const cap={approval:{step:50},state:{candidate:{sha256:'b'.repeat(64)}},async assertUnchanged(){}};
 const engine={},guard={},receipt={state_sha256:'c'.repeat(64)},worker={pid:meta.pid,async exportCoreConstraint(){return{directory};}},peer={worker,serial:0};
 const bindings={assert,readFile,join,installations:new WeakMap([[engine,{cap,receipt,owner:{file_sha256:meta.source_sha256}}]]),assertCoreInstalled(e:any,c:any,r:any){assert.equal(e,engine);assert.equal(c,cap);assert.equal(r,receipt);},borrowNativeEnginePeer:()=>peer,borrowCoreProcess:(w:any)=>w};
 const producer=new Function(...Object.keys(bindings),fn('coreConstraintResidual')+'\nreturn coreConstraintResidual;')(...Object.values(bindings));
 const residual=await producer(engine,guard,cap);
 expect(Object.keys(residual).length).toBe(11);
 if(!block.includes('quantized_frobenius:'))expect(()=>consumer(block)(residual,cap,[],[],1)).toThrow();
 const observed=consume(residual,cap,[proof('new',true)],[proof('known',true)],1);
 expect(observed.constraint_residual).toEqual(meta.constraint);
 expect(Object.keys(observed.constraint_residual).length).toBe(5);
 // Direct successors derive and revalidate the serialized TRAIN row using the original schema.
 const groups={new_training:[record('new')],replay_training:[record('known'),record('unknown')]};
 const derived=safety.deriveCore450TrainingCheck({step:50,new_records:groups.new_training,replay_records:[groups.replay_training[0]],baseline:[proof('new',false)],previous:null,...observed,baseline_replay_loss:1,previous_streak:0,prior_safety_failed:false});
 const saved={step:50,consecutive_passes:derived.row.consecutive_passes,new_exposures:coreNewExposures(50,groups.new_training),report:{steps:450,training_safety_policy:scope.CORE450_SAFETY_POLICY,partial_safety_failed:derived.partial_safety_failed,partial_training_baseline:[proof('new',false),proof('known',true),proof('unknown',false)],baseline_training_replay_generation:[proof('known',true)],baseline_training_replay_loss:1,baseline_correct_replay_training_ids:['known'],training_selection:{checkpoints:[...CORE450_CHECKS],policy_version:scope.CORE450_SAFETY_POLICY,required_consecutive_passes:2,required_new_exposures:2,minimum_replay_generation:0,replay_generation_ids:['known'],results:[derived.row],selected_step:derived.selected_step}}};
 expect(safety.validateCore450TrainingHistory(saved,groups).observed_check_steps).toEqual([50]);
 expect(safety.nextCore450TrainingStep(saved,groups)).toBe(64);
 expect(derived.row.accepted).toBe(false); // first check cannot prove two consecutive observations.
 for(const bad of [{...residual,step:49},{...residual,core_tensor_sha256:'wrong'},{...residual,actual_artifacts:''},{...residual,quantized_max_abs:NaN},{...residual,master_frobenius:-1},{...residual,weight_dtype:'torch.float32'}])expect(()=>consume(bad,cap,[],[],1)).toThrow();
 expect(()=>safety.validateCore450ConstraintResidual(residual)).toThrow(); // strict metric validator is unchanged.
 return {producer,engine,guard,cap};
}
test('actual compiled constraint producer, TRAIN projection and next history schema preserve v14 metrics and guards',async()=>{
 const directory=await mkdtemp(join(tmpdir(),'luban-constraint-'));
 try{
  const meta={schema:'luban-retained-core-constraint-v1',pid:5877,step:50,source_sha256:'d600892f3e90a79bd0cb42fca136b1ab9d3477bb4f14a80943fe1c8741a253a4',constraint:metrics,actual_tensors:7,optimizer_unchanged:true,RNG_unchanged:true,product_admitted:false};
  await writeFile(join(directory,'constraint.json'),JSON.stringify(meta));
  for(const [key,value]of Object.entries(metrics))if(typeof value==='number'){const bytes=Buffer.alloc(4);bytes.writeFloatLE(value);await writeFile(join(directory,key+'.raw'),bytes);}
  const h=await replay(directory);
  await writeFile(join(directory,'quantized_frobenius.raw'),Buffer.alloc(4));
  await expect(h.producer(h.engine,h.guard,h.cap)).rejects.toThrow();
 }finally{await rm(directory,{recursive:true,force:true});}
});
test.skipIf(!process.env.LUBAN_CONSTRAINT_ARTIFACTS)('read-only replay of actual v14 step50 native JSON and scalar artifacts',async()=>{await replay(process.env.LUBAN_CONSTRAINT_ARTIFACTS!);});
