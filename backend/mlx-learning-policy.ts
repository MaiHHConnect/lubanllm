/** Daily learning measures gains on held-out questions and revisits adopted knowledge. */
import assert from 'node:assert/strict';
import {materialProvenance,materialContext} from './native-evolution-material.ts';
type Row=Record<string,any>;
export const MLX_LEARNING_POLICY='verified-knowledge-net-gain-v1';
export const MLX_TRAINING_ADMISSION='direct-train-routed-validation-v1';
export const MLX_NET_GATES=['mlx_closed_book_gain','mlx_net_gain','mlx_deterministic','mlx_same_inputs','mlx_validation_loss_improved','mlx_routes_available'] as const;
export function mlxMastered(pair:Row[][]){
 if(pair.length!==2)return new Set<string>();
 return new Set<string>(pair[0]!.filter(row=>{const other=pair[1]!.find(v=>v.id===row.id);return other&&[row,other].every(v=>v.passed===true&&v.complete===true&&v.reference_used===false)&&JSON.stringify(row.tokens)===JSON.stringify(other.tokens)&&JSON.stringify(row.input_tokens)===JSON.stringify(other.input_tokens);}).map(row=>row.id));
}
export function mlxNetProgress(records:{valid:Row[];protection:Row[]},baseline:Row[][],candidate:Row[][]){
 const before=mlxMastered(baseline),after=mlxMastered(candidate);
 const mastered_ids=records.valid.filter(r=>after.has(r.id)).map(r=>r.id).sort();
 const gained_ids=mastered_ids.filter(id=>!before.has(id));
 const forgotten_ids=records.protection.filter(r=>!after.has(r.id)).map(r=>r.id).sort();
 const retained_ids=records.protection.filter(r=>after.has(r.id)).map(r=>r.id).sort();
 return{policy:MLX_LEARNING_POLICY,gained_ids,forgotten_ids,retained_ids,mastered_ids,net_gain:gained_ids.length-forgotten_ids.length};
}
/** All verified history is retained; a rotating window bounds each round's check cost. */
export function mlxKnowledgeSample(report:Row|null){
 const pool:Row[]=report?.learning_policy===MLX_LEARNING_POLICY?report.knowledge_pool??[]:[];
 if(!pool.length)return[];
 const ordered=[...pool].sort((a,b)=>String(a.id).localeCompare(String(b.id))),offset=Number(report?.review_cursor??0)%ordered.length;
 assert(Number.isSafeInteger(offset)&&offset>=0,'Invalid knowledge review cursor');
 return Array.from({length:Math.min(16,ordered.length)},(_,i)=>structuredClone(ordered[(offset+i)%ordered.length]!));
}
export function mlxNextKnowledge(previous:Row|null,records:{train:Row[];valid:Row[];protection:Row[]},candidate:Row[][]){
 const after=mlxMastered(candidate),prior:Row[]=previous?.learning_policy===MLX_LEARNING_POLICY?previous.knowledge_pool??[]:[];
 const forgotten=records.protection.filter(r=>!after.has(r.id));
 const forgetContexts=new Set(forgotten.map(materialContext)),pool=new Map(prior.filter(r=>!forgetContexts.has(materialContext(r))).map(r=>[materialContext(r),r]));
 for(const row of records.valid)if(after.has(row.id))pool.set(materialContext(row),{...row,provenance:JSON.stringify({...materialProvenance(row),learning_train_ids:records.train.map(r=>r.id)})});
 const debt=new Map<string,Row>((previous?.relearning_records??[]).map((r:Row)=>[r.id,r]));
 for(const row of forgotten)debt.set(row.id,row);
 for(const row of records.valid)if(after.has(row.id))debt.delete(row.id);
 return{knowledge_pool:[...pool.values()],review_cursor:Number(previous?.review_cursor??0)+records.protection.length,relearning_records:[...debt.values()]};
}
