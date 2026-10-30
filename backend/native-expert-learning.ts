/** Current Qwen expert policy. These calculations grant no model or publication authority. */
import assert from 'node:assert/strict';
import {materialProvenance} from './native-evolution-material.ts';
import {evaluateGeneration,validateGenerationCheck} from './generation-checks.ts';

export const NATIVE_EXPERT_PROTOCOL='native-expert-preservation-v1';
export const NATIVE_EXPERT_CANDIDATE_STEPS=64;
export const NATIVE_EXPERT_PRESERVATION_WEIGHT=5;
export function nativeExpertHasParent(previous:{owner:unknown;approval:{core_version:string|null;branch_core_version:string|null}},core:string|null){
 return previous.owner!==null&&previous.approval.core_version===core&&previous.approval.branch_core_version===core;
}
export function nativeExpertSessionChoices(previous:{owner:unknown;approval:{width:number;core_version:string|null;branch_core_version:string|null}},core:string|null){
 const hasParent=nativeExpertHasParent(previous,core);return nativeExpertChoices(hasParent?previous.approval.width:128,hasParent);
}
export function nativeExpertChoices(previousWidth=128,hasPrevious=false){
 assert(Number.isSafeInteger(previousWidth)&&previousWidth>=32&&previousWidth<=512&&typeof hasPrevious==='boolean');
 const widths=[...new Set([previousWidth,Math.min(previousWidth*2,512),Math.max(32,Math.floor(previousWidth/2))])].sort((a,b)=>a-b);
 return [...widths.map(width=>({width,reuse:hasPrevious,learning_rate:.0002})),...(hasPrevious?[{width:previousWidth,reuse:false,learning_rate:.0002}]:[])];
}
export function nativeExpertStepBudget(previousWidth=128,hasPrevious=false){return nativeExpertChoices(previousWidth,hasPrevious).length*NATIVE_EXPERT_CANDIDATE_STEPS;}
export function nativeExpertQualityBoundary(step:number){return Number.isSafeInteger(step)&&step>=8&&step<=64&&step%8===0;}
export function nativeExpertAccumulatedStep(candidates:readonly Record<string,any>[],step:number){
 assert(Number.isSafeInteger(step)&&step>=0&&step<=64);
 return candidates.reduce((sum,row)=>{assert(nativeExpertQualityBoundary(row.actual_steps),'Candidate history lacks its actual saved update count');return sum+row.actual_steps;},step);
}
/** Freeze already-observed old protection only; never turn NEW VALID into loss material. */
export function freezeNativeExpertPreservation(rows:readonly Record<string,any>[],observations:readonly Record<string,any>[],saved?:readonly Record<string,any>[]){
 assert(rows.length>0&&observations.length===rows.length,'完整核心和父专家保护不能为空');
 const byId=new Map(observations.map(row=>[row.id,row]));assert.equal(byId.size,rows.length,'重复保护观察');
 const result:Record<string,any>[]=rows.map(row=>{
  const value=byId.get(row.id),p=materialProvenance(row);
  assert(value?.passed===true&&value.complete===true&&value.reference_used===false&&typeof value.answer==='string'&&value.answer.trim()&&typeof value.branch_enabled==='boolean','保护教师需要正确、完整、无参考的实际原答案');
  const teacher=p.retention_scope==='accepted_branch'?'expert':'core';assert.equal(value.branch_enabled,teacher==='expert','核心/父专家保护来源不一致');
  const check=validateGenerationCheck(p.generation_check??{type:'exact',expected:row.answer,max_new_tokens:512},row.answer);
  assert(evaluateGeneration(value.answer,check,true).passed,'保护基线未通过原规则');
  return {...structuredClone(row),answer:value.answer,purpose:'validation',approved:true,training_allowed:false,
   provenance:JSON.stringify({...p,preservation_only:true,preservation_teacher:teacher,preservation_baseline:{answer:value.answer,complete:true},
    preservation_evidence:NATIVE_EXPERT_PROTOCOL,generation_check:{...check,type:'exact',expected:value.answer}})};
 });
 if(saved){assert.equal(saved.length,result.length,'恢复丢失固定教师');for(let i=0;i<result.length;i++){
  const before:Record<string,any>=saved[i]!,current:Record<string,any>=result[i]!;assert.equal(before.id,current.id);assert.equal(before.answer,current.answer,'恢复时旧教师答案已改变');
  assert.deepEqual(materialProvenance(before),materialProvenance(current),'恢复时固定教师来源已改变');
 }}
 return Object.freeze(result.map(row=>Object.freeze(row)));
}
export function assertNativeExpertPreservation(row:Record<string,any>){
 const p=materialProvenance(row);assert(row.purpose==='validation'&&row.approved===true&&row.training_allowed===false&&p.preservation_only===true&&['core','expert'].includes(p.preservation_teacher)&&p.preservation_baseline?.complete===true&&typeof row.answer==='string'&&row.answer.trim()&&row.answer===p.preservation_baseline.answer&&p.generation_check?.type==='exact'&&p.generation_check.expected===row.answer,'固定专家保护记录无效');return p.preservation_teacher as 'core'|'expert';
}
