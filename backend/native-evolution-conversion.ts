/** Validate terminal conversion evidence already carried by the original trainer summary. */
import assert from 'node:assert/strict';
export function validateNativeEvolutionTerminalConversion(report:Record<string,any>,input:{config:Record<string,string>;tensors:Record<string,any>;steps:readonly unknown[]}){
 const mode=input.config.final_conversion,offset=Number(input.config.training_offset??0),terminal=offset+input.steps.length===450;
 const value=report.terminal_conversion;
 if(!mode||!terminal){assert(value===null||value===undefined,'非末尾训练不得声称已转换');return null;}
 assert.equal(mode,'gptq-bf16-refine-v1');assert(value&&typeof value==='object'&&!Array.isArray(value),'末尾缺少实际转换证据');
 const tensor=input.tensors.constraint_inputs,old=Number(input.config.constraint_old_positions);
 assert(tensor&&Array.isArray(tensor.shape)&&tensor.shape.length===2&&Number.isSafeInteger(old)&&old>0&&old<tensor.shape[0]);
 assert.equal(value.method,mode);assert.equal(value.absolute_step,450);assert.equal(value.passes,2);
 assert.equal(value.old_positions,old);assert.equal(value.q4_train_positions,tensor.shape[0]-old);
 assert(Number.isSafeInteger(value.expanded_rank)&&value.expanded_rank>0&&value.expanded_rank<tensor.shape[1]);
 assert(Number.isFinite(value.original_span_max_abs)&&value.original_span_max_abs>=0&&value.original_span_max_abs<=2e-5);
 assert(Array.isArray(value.rounds)&&value.rounds.length===3);
 for(const [index,row]of value.rounds.entries()){
  assert.equal(row.iteration,index);
  for(const key of ['reconstruction_frobenius','known_old_frobenius','q4_train_frobenius','original_replay_span_frobenius'])assert(Number.isFinite(row[key])&&row[key]>=0,'转换残余必须真实有限');
  const square=row.known_old_frobenius**2+row.q4_train_frobenius**2;
  assert(Math.abs(row.reconstruction_frobenius**2-square)<=Math.max(1e-12,1e-5*square),'转换分组与总误差矛盾');
 }
 const final=value.rounds[2];for(const prior of value.rounds.slice(0,2))assert(final.reconstruction_frobenius<prior.reconstruction_frobenius&&final.known_old_frobenius<=prior.known_old_frobenius&&final.q4_train_frobenius<=prior.q4_train_frobenius,'末尾转换未通过固定重构门');
 assert.equal(value.eligible,true);return structuredClone(value);
}
