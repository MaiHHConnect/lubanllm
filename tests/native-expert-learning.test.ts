import {test,expect} from 'bun:test';
import {nativeExpertChoices,nativeExpertStepBudget,nativeExpertAccumulatedStep,freezeNativeExpertPreservation,assertNativeExpertPreservation,nativeExpertQualityBoundary,nativeExpertSessionChoices,nativeExpertHasParent} from '../backend/native-expert-learning.ts';
const rows=[{id:'old-core',answer:'',purpose:'validation',approved:true,training_allowed:false,provenance:JSON.stringify({generation_check:{type:'semantic',validator:'integer',expected:23,max_new_tokens:24}})},
 {id:'old-parent',answer:'learned answer',purpose:'validation',approved:true,training_allowed:false,provenance:JSON.stringify({retention_scope:'accepted_branch',generation_check:{type:'exact',expected:'learned answer',max_new_tokens:24}})}];
const observations=rows.map((row,i)=>({id:row.id,answer:i?'learned answer':'23',passed:true,complete:true,reference_used:false,branch_enabled:i===1}));
test('current conservative fresh/reuse rates, bounded width search and actual progress',()=>{
 expect(nativeExpertChoices()).toEqual([64,128,256].map(width=>({width,reuse:false,learning_rate:.0002})));
 expect(nativeExpertChoices(128,true).at(-1)).toEqual({width:128,reuse:false,learning_rate:.0002});
 expect(nativeExpertStepBudget()).toBe(192);expect(nativeExpertStepBudget(128,true)).toBe(256);
 expect(nativeExpertAccumulatedStep([{actual_steps:64},{actual_steps:16}],8)).toBe(88);
 expect(()=>nativeExpertAccumulatedStep([{actual_steps:0}],8)).toThrow();
 for(const step of [0,7,9,65])expect(nativeExpertQualityBoundary(step)).toBe(false);
});
test('fixed teacher targets use real branch-off/on answers, preserve read-only provenance and resume',()=>{
 const result=freezeNativeExpertPreservation(rows,observations);expect(result.map(assertNativeExpertPreservation)).toEqual(['core','expert']);
 expect(result.map(row=>row.answer)).toEqual(['23','learned answer']);expect(rows[0]!.answer).toBe('');
 expect(result.every(row=>row.training_allowed===false&&row.purpose==='validation')).toBe(true);
 expect(freezeNativeExpertPreservation(rows,observations,result)).toEqual(result);
 expect(()=>freezeNativeExpertPreservation(rows,[{...observations[0],answer:'24'},observations[1]!],result)).toThrow();
});
test('physical previous expert stays available for rollback but cannot teach a different core',()=>{
 const previous={owner:{},approval:{width:256,core_version:'old',branch_core_version:'old'}};
 expect(nativeExpertHasParent(previous,'new')).toBe(false);
 expect(nativeExpertSessionChoices(previous,'new')).toEqual(nativeExpertChoices());
 expect(previous.owner).toEqual({});
 expect(nativeExpertHasParent(previous,'old')).toBe(true);
 expect(nativeExpertSessionChoices(previous,'old')).toEqual(nativeExpertChoices(256,true));
 expect(nativeExpertHasParent({...previous,approval:{...previous.approval,branch_core_version:'mismatch'}},'old')).toBe(false);
});
test('wrong, empty, truncated, reference or swapped teachers never enter KL',()=>{
 for(const change of [{answer:''},{answer:'24'},{complete:false},{passed:false},{reference_used:true},{branch_enabled:true}]){
  expect(()=>freezeNativeExpertPreservation(rows,[{...observations[0],...change},observations[1]!])).toThrow();
 }
 expect(()=>freezeNativeExpertPreservation(rows,[observations[0]!,observations[0]!])).toThrow();
});
