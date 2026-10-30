import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {patchNativeEvolutionTrainingHistory} from '../scripts/native-evolution-training-history-patches.ts';
import {coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {nativePartialCorrectIDs} from '../.native-migration/research/native-core-controller-candidate/progress.ts';
import {CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary} from '../.native-migration/research/native-core-controller-candidate/segments.ts';
const original=await readFile(new URL('../.native-migration/research/native-core450-runner/training-safety.ts',import.meta.url),'utf8');
const replace=(body:string,before:string,after:string)=>{assert.equal(body.split(before).length,2);return body.replace(before,after);};
const scope={assert,coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory,nativePartialCorrectIDs,CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary};
function load(source:string){
 const code=source.replace(/^import .*?;$/gm,'').replaceAll('export ','');
 return new Function(...Object.keys(scope),new Bun.Transpiler({loader:'ts'}).transformSync(code+'\nreturn {validateCore450TrainingHistory,deriveCore450TrainingCheck};'))(...Object.values(scope));
}
const old=load(original),fixed=load(patchNativeEvolutionTrainingHistory(original,replace));
const record=(id:string)=>({id,answer:'正确',provenance:{generation_check:{type:'exact',expected:'正确',max_new_tokens:16}}});
const groups={new_training:[record('new')],replay_training:[record('known'),record('unknown')]};
const proof=(id:string,passed:boolean)=>({id,answer:passed?'正确':'错误',passed,complete:true,branch_enabled:false,reference_used:false});
function checkpoint(allWrong=false){
 const ids=allWrong?[]:['known'];
 return {step:0,consecutive_passes:0,new_exposures:coreNewExposures(0,groups.new_training),report:{steps:450,
  training_safety_policy:'closed-book-growth-450-train-safety-v1',partial_safety_failed:false,
  partial_training_baseline:[proof('new',false),proof('known',!allWrong),proof('unknown',false)],
  baseline_training_replay_generation:ids.map(id=>proof(id,true)),baseline_training_replay_loss:1,
  baseline_correct_replay_training_ids:ids,training_selection:{checkpoints:[...CORE450_CHECKS],policy_version:'closed-book-growth-450-train-safety-v1',required_consecutive_passes:2,required_new_exposures:2,minimum_replay_generation:0,replay_generation_ids:ids,results:[],selected_step:null}}};
}
test('real history validator reproduces old CP0 rejection and accepts Python replay-subset semantics',()=>{
 const saved=checkpoint();expect(()=>old.validateCore450TrainingHistory(saved,groups)).toThrow('partial observations are incomplete or unordered');
 expect(fixed.validateCore450TrainingHistory(saved,groups).completed_steps).toBe(0);
 expect(fixed.validateCore450TrainingHistory(checkpoint(true),groups).completed_steps).toBe(0);
 expect(saved.report.baseline_training_replay_generation.map(row=>row.id)).toEqual(['known']);
});
test('replay subset still requires all original-correct items and complete truthful baseline',()=>{
 for(const mutate of [
  (v:any)=>{v.report.baseline_training_replay_generation=[];v.report.baseline_correct_replay_training_ids=[];v.report.training_selection.replay_generation_ids=[];},
  (v:any)=>{v.report.baseline_training_replay_generation.push(proof('unknown',true));},
  (v:any)=>{v.report.partial_training_baseline.pop();},
  (v:any)=>{v.report.partial_training_baseline[2].passed=true;},
  (v:any)=>{v.report.baseline_training_replay_generation=[proof('known',false)];}
 ]){const saved=checkpoint();mutate(saved);expect(()=>fixed.validateCore450TrainingHistory(saved,groups)).toThrow();}
});
test('subsequent TRAIN observation still marks loss of protected replay as safety failure',()=>{
 const saved=checkpoint(),result=fixed.deriveCore450TrainingCheck({step:50,new_records:groups.new_training,replay_records:[groups.replay_training[0]],baseline:[proof('new',false)],previous:null,new_generation:[proof('new',true)],replay_generation:[proof('known',false)],baseline_replay_loss:1,replay_loss:1,previous_streak:0,prior_safety_failed:false,constraint_residual:{quantized_frobenius:0,quantized_max_abs:0,quantized_relative_to_delta:0,master_frobenius:0,weight_dtype:'torch.bfloat16'}});
 expect(result.partial_safety_failed).toBe(true);expect(result.row.accepted).toBe(false);
 Object.assign(saved,{step:50,new_exposures:coreNewExposures(50,groups.new_training)});
 saved.report.partial_safety_failed=true;(saved.report.training_selection.results as any[]).push(result.row);
 expect(fixed.validateCore450TrainingHistory(saved,groups).partial_safety_failed).toBe(true);
});
