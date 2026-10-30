/** Pure MLX gates and actual resident-worker observations. No BF16/Q4 receipt substitution. */
import assert from 'node:assert/strict';
import {MLX_LEARNING_POLICY,MLX_NET_GATES,mlxNetProgress,mlxMastered} from './mlx-learning-policy.ts';
import {MLXLearningClient,type PreparedMLX} from './mlx-learning-client.ts';
import {materialProvenance} from './native-evolution-material.ts';
import {evaluateRetention,retentionAcceptsGrowth,retentionClass} from './retention-policy.ts';
import type {MLXRecords,MLXRow} from './mlx-learning-source.ts';

export const MLX_QUALITY_GATES=['mlx_training_generation','mlx_validation_generation','mlx_core_protection_baseline','mlx_parent_protection_baseline','mlx_protection_generation','mlx_baseline_correct_retained','mlx_deterministic','mlx_same_inputs','mlx_validation_loss_improved','mlx_replay_loss_retained','mlx_routes_available']as const;
/** Restored update boundaries must be validated before another Adam or candidate disposal. */
export function mlxCandidateAction(step:number,validatedStep:number){assert(Number.isSafeInteger(step)&&step>=0&&step<=64&&Number.isSafeInteger(validatedStep)&&validatedStep>=0&&validatedStep<=step);if(step>0&&step%8===0&&validatedStep<step)return'validate';return step<64?'train':'done';}
export function uniqueMLXRecords(records:MLXRecords){const ids=new Map<string,MLXRow>();for(const row of [...records.train,...records.valid,...records.protection]){const old=ids.get(row.id);assert(!old||JSON.stringify(old)===JSON.stringify(row),'MLX row ID collision');ids.set(row.id,row);}return [...ids.values()];}
export async function observeMLXPair(client:MLXLearningClient,rows:MLXRow[],source:'core'|'parent'|'candidate'|'current',routes:Map<string,string>,signal?:AbortSignal){const pair:MLXRow[][]=[];for(let i=0;i<2;i++){const pass:MLXRow[]=[];for(const row of rows){signal?.throwIfAborted();const actual=source==='current'?(routes.get(row.id)==='neural_expert'?'parent':'core'):source==='candidate'&&routes.get(row.id)==='local_core'?'core':source;pass.push({...await client.generate(row,actual,signal),route_action:routes.get(row.id)??null});}pair.push(pass);}return pair;}
export async function observeMLXLoss(client:MLXLearningClient,rows:MLXRow[],source:'core'|'parent'|'candidate'|'protection',prepared:Map<string,PreparedMLX>,signal?:AbortSignal,routes?:Map<string,string>){const observations=[];for(const row of rows){const actual=source==='protection'?(materialProvenance(row).preservation_teacher==='expert'?'parent':'core'):routes?.get(row.id)==='local_core'?'core':source;observations.push({...await client.loss(row,actual,prepared.get(row.id)!,signal),source:actual});}return{mean:observations.length?observations.reduce((n,r)=>n+r.loss,0)/observations.length:0,observations};}
export function decideMLXQuality(input:{records:MLXRecords;baseline:MLXRow[][];parent:MLXRow[][];candidate:MLXRow[][];baselineLoss:number|Number;candidateLoss:number|Number;replayBefore:number|Number;replayAfter:number|Number;routes:Map<string,string>;policy?:string}){
 const {records,baseline,parent,candidate}=input;for(const v of [input.baselineLoss,input.candidateLoss,input.replayBefore,input.replayAfter])assert((typeof v==='number'||v instanceof Number)&&Number.isFinite(Number(v))&&Number(v)>=0,'MLX损失记录必须是非负有限数');
 const baselineLoss=Number(input.baselineLoss),candidateLoss=Number(input.candidateLoss),replayBefore=Number(input.replayBefore),replayAfter=Number(input.replayAfter);
 const ids=(rows:MLXRow[])=>rows.map(row=>row.id),passes=(pair:MLXRow[][],wanted:string[])=>pair.length===2&&pair.every(pass=>wanted.every(id=>pass.some(row=>row.id===id&&row.passed===true&&row.complete===true&&row.reference_used===false)));
 const mastered=(pair:MLXRow[][])=>new Set(pair.length===2?pair[0]!.filter((row,i)=>{const other=pair[1]!.find(next=>next.id===row.id);return !!other&&row.passed===true&&row.complete===true&&row.reference_used===false&&other.passed===true&&other.complete===true&&other.reference_used===false}).map(row=>row.id):[]);
 const deterministic=(pair:MLXRow[][])=>pair.length===2&&pair[0]!.length===pair[1]!.length&&pair[0]!.every((row,i)=>{const next=pair[1]![i]!;return row.id===next.id&&JSON.stringify(row.tokens)===JSON.stringify(next.tokens)&&JSON.stringify(row.input_tokens)===JSON.stringify(next.input_tokens)&&row.complete===next.complete;});
 if(input.policy===MLX_LEARNING_POLICY){
  const progress=mlxNetProgress(records,baseline,candidate),before=mlxMastered(baseline);
  const all=uniqueMLXRecords(records),covers=(pair:MLXRow[][])=>pair.length===2&&pair.every(pass=>pass.length===all.length&&new Set(pass.map(r=>r.id)).size===all.length&&all.every(row=>pass.some(v=>v.id===row.id)));
  const gates={mlx_closed_book_gain:progress.gained_ids.length>0,mlx_net_gain:progress.net_gain>0,
   mlx_deterministic:covers(baseline)&&covers(candidate)&&deterministic(baseline)&&deterministic(candidate),
   mlx_same_inputs:candidate.length===2&&candidate.every(pass=>pass.every(row=>JSON.stringify(row.input_tokens)===JSON.stringify(baseline[0]?.find(old=>old.id===row.id)?.input_tokens))),
   mlx_validation_loss_improved:candidateLoss<baselineLoss*.995,
   mlx_routes_available:records.valid.every(row=>input.routes.get(row.id)==='neural_expert'||input.routes.get(row.id)==='local_core'&&before.has(row.id))&&candidate.every(pass=>pass.every(row=>input.routes.get(row.id)!=='local_core'||JSON.stringify(row.tokens)===JSON.stringify(baseline[0]?.find(old=>old.id===row.id)?.tokens)))};
  assert.deepEqual(Object.keys(gates),[...MLX_NET_GATES]);return{gates,accepted:Object.values(gates).every(Boolean),old_correct_ids:[...before],learning_progress:progress};
 }
 const core=records.protection.filter(row=>materialProvenance(row).retention_scope!=='accepted_branch'),branch=records.protection.filter(row=>materialProvenance(row).retention_scope==='accepted_branch');
 const baselineCorrect=new Set([...mastered(baseline),...mastered(parent)]),candidateCorrect=mastered(candidate);
 const oldCorrect=[...baselineCorrect];
 const retention=evaluateRetention(records.protection,baselineCorrect,candidateCorrect);
 const gained=new Set([...candidateCorrect].filter(id=>!baselineCorrect.has(id)&&records.train.concat(records.valid).some(row=>row.id===id)));
 const hardCore=core.filter(row=>retentionClass(row)==='hard'),hardBranch=branch.filter(row=>retentionClass(row)==='hard');
 const gates={mlx_training_generation:passes(candidate,ids(records.train)),mlx_validation_generation:passes(candidate,ids(records.valid)),mlx_core_protection_baseline:passes(baseline,ids(hardCore)),mlx_parent_protection_baseline:hardBranch.length===0||passes(parent,ids(hardBranch)),mlx_protection_generation:retentionAcceptsGrowth(retention,gained.size),mlx_baseline_correct_retained:retention.hard_forgotten_ids.length===0&&retention.soft_forgotten_ids.length<=retention.soft_forget_budget,mlx_deterministic:deterministic(baseline)&&deterministic(candidate)&&(branch.length===0||deterministic(parent)),mlx_same_inputs:candidate.length===2&&candidate.every(pass=>pass.every(row=>JSON.stringify(row.input_tokens)===JSON.stringify(baseline[0]!.find(old=>old.id===row.id)?.input_tokens))),mlx_validation_loss_improved:candidateLoss<baselineLoss*.995,mlx_replay_loss_retained:replayAfter<=replayBefore*1.05+.01,mlx_routes_available:[...records.train,...records.valid].every(row=>input.routes.get(row.id)==='neural_expert'||input.routes.get(row.id)==='local_core'&&passes(baseline,[row.id]))};
 assert.deepEqual(Object.keys(gates),[...MLX_QUALITY_GATES]);return{gates,accepted:Object.values(gates).every(Boolean),old_correct_ids:oldCorrect};
}
