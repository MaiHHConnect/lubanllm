/** Separate factory/current-engine evidence never changes an accepted proof or journal. */
import qualifications from '../src/luban/factory-engine-qualifications.json';
import {encoded} from './canonical-json.ts';

const gates=['prepare_equivalent','real_adam_updates','checkpoint_replay_exact','cancellation_atomic','continued_after_cancel','saved_expert_reload_equal','base_core_frozen'];
function mathValid(m:Record<string,any>,p:Record<string,any>){
 const stages=m.stages;
 return m.passed===true&&m.fp32_pass===true&&m.bf16_pass_assumed===false&&m.bf16_numerical_pass===true
  &&m.bf16_relative_l2_limit===.03&&m.expert_file_sha256===p.expert_file_sha256&&Array.isArray(stages)&&stages.length===8
  &&['fp32','bf16'].every(dtype=>['gate','up','activation','output'].every(stage=>stages.filter(s=>s.dtype===dtype&&s.stage===stage).length===1))
  &&stages.every(s=>s.finite===true&&s.nonzero===true&&s.fp32_tolerance_pass===true&&(s.dtype!=='bf16'||typeof s.relative_l2==='number'&&s.relative_l2>=0&&s.relative_l2<=.03));
}
function qualified(){
 return (qualifications as Record<string,any>[]).filter(row=>{
  const t=row.training??{},q=row.quality??{},m=row.math??{},p=row.source_proof??{};
  return row.schema==='luban-factory-engine-qualification-v1'&&row.complete===true&&row.teacher_calls===0&&row.sources_unchanged===true
   &&typeof row.worker_binary_sha256==='string'&&/^[a-f0-9]{64}$/.test(row.worker_binary_sha256)
   &&p.expert_id==='learned-bafb4050374f'&&p.schema==='mlx-expert-learning-v1'&&p.width===32&&p.quality_admitted===true&&p.training_admitted===true&&p.core_version==='core-fdfa0ce5a75c'
   &&t.schema==='mlx-native-training-qualification-v1'&&t.complete===true&&gates.every(k=>t.gates?.[k]===true)
   &&t.worker_binary_sha256===row.worker_binary_sha256&&['core_file_sha256','mlx_model_sha256'].every(k=>t[k]===p[k])
   &&t.parent_expert_file_sha256===p.expert_file_sha256&&mathValid(m,p)
   &&q.complete===true&&q.two_passes_equal===true&&q.routes_equal===true&&q.protected_answers_equal===true
   &&q.all_previously_correct_still_pass===true&&Number.isSafeInteger(q.rows)&&q.rows>0;
 });
}
export function factoryWorkerQualified(proof:Record<string,any>,workerSHA:string){
 return typeof workerSHA==='string'&&/^[a-f0-9]{64}$/.test(workerSHA)&&(proof.worker_binary_sha256===workerSHA||qualified().some(row=>row.worker_binary_sha256===workerSHA&&encoded(row.source_proof)===encoded(proof)));
}
export function factoryTrainingQualification(original:Record<string,any>,workerSHA:string,coreSHA:string,modelSHA:string){
 return [original,...qualified().map(row=>row.training)].find(t=>t.worker_binary_sha256===workerSHA&&t.core_file_sha256===coreSHA&&t.mlx_model_sha256===modelSHA&&t.complete===true&&gates.every(k=>t.gates?.[k]===true))??original;
}
