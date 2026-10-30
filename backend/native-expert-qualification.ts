/** Independent actual-HF operator evidence, bound at build time; never a caller's passed flag. */
import assert from 'node:assert/strict';
export type NativeExpertGraphBinding={identity:string;core_tensor_sha256:string;graph_binary_sha256:string;source_build_sha256:string;hidden:2048;width:number};
type Pair={name:string;elements:number;max_abs_error:number;atol:number;rtol:number;allclose_passed:boolean};
export type NativeExpertGraphEvidence=NativeExpertGraphBinding&{schema:'native-expert-preservation-graph-v1';evidence_id:string;dtype:'BF16';learning_rate:.0002;preservation_weight:5;actual_steps:2;teacher_modes:['core','expert'];pairs:Pair[];restored:{previous_generation_equal:true;optimizer_resume_equal:true;parent_rng_restored:true;owners_closed:true};actual_original_hf:true};
// The normal builder replaces this fixed placeholder only after the independent verifier.
const compiled='BUILD_BOUND_NATIVE_EXPERT_QUALIFICATION';
function records():NativeExpertGraphEvidence[]{return compiled.startsWith('BUILD_BOUND_')?[]:JSON.parse(compiled);}
export function validateNativeExpertGraphEvidence(value:NativeExpertGraphEvidence){
 assert(value.schema==='native-expert-preservation-graph-v1'&&value.actual_original_hf===true&&value.dtype==='BF16'&&value.hidden===2048&&Number.isSafeInteger(value.width)&&value.width>=32&&value.width<=512);
 for(const key of ['identity','core_tensor_sha256','graph_binary_sha256','source_build_sha256'] as const)assert(/^[a-f0-9]{64}$/.test(value[key]));
 assert(typeof value.evidence_id==='string'&&value.evidence_id.length>0&&value.actual_steps===2&&value.learning_rate===.0002&&value.preservation_weight===5);
 assert.deepEqual(value.teacher_modes,['core','expert']);
 const required=['teacher_core','teacher_expert',...[1,2].flatMap(step=>[`step${step}_ce`,`step${step}_preservation_kl`,`step${step}_loss`,...[0,1,2].flatMap(i=>['gradient','weight','m','v','adamstep'].map(key=>`step${step}_${key}_${i}`))])];
 assert(new Set(value.pairs.map(row=>row.name)).size===value.pairs.length);
 for(const name of required){const pair=value.pairs.find(row=>row.name===name);assert(pair&&Number.isSafeInteger(pair.elements)&&pair.elements>0&&pair.allclose_passed===true&&Number.isFinite(pair.max_abs_error)&&pair.max_abs_error>=0&&Number.isFinite(pair.atol)&&pair.atol>=0&&Number.isFinite(pair.rtol)&&pair.rtol>=0,`Missing actual expert comparison: ${name}`);}
 assert.deepEqual(value.restored,{previous_generation_equal:true,optimizer_resume_equal:true,parent_rng_restored:true,owners_closed:true});return value;
}
export function readNativeExpertGraphQualification(binding:NativeExpertGraphBinding){
 const value=records().find(value=>Object.entries(binding).every(([key,item])=>value[key as keyof NativeExpertGraphBinding]===item));
 assert(value,'当前核心、原生专家宽度及CE/KL/Adam/恢复尚未取得独立真实图资格');return Object.freeze(validateNativeExpertGraphEvidence(value));
}
