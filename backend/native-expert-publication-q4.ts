/** Real ordinary-Q4 checks after the BF16 owner has closed; no caller-issued approval. */
import assert from 'node:assert/strict';
import {join} from 'node:path';
import {writeFile} from 'node:fs/promises';
import {NativeQ4Observer,assertNativeQ4Observation,type NativeQ4Expert,type NativeQ4Observation} from './native-evolution-q4.ts';
import {materialProvenance} from './native-evolution-material.ts';
import {closeImportOwner} from './native-evolution-import-observe.ts';

export type NativeExpertQ4Options={executable:string;binary:string;model:string;manifestSHA:string;identity:string;work:string;signal?:AbortSignal};
type Row=Record<string,any>;
const proofs=new WeakSet<object>();
export function assertNativeExpertQ4Publication(value:object){assert(proofs.has(value),'Expert publication needs this live Q4 observation');}
export async function observeNativeExpertPublication(o:NativeExpertQ4Options,input:{records:{train:Row[];valid:Row[];protection:Row[]};core:{file:string;sha256:string;tensor_key:string};coreVersion:string;coreTensorSHA:string;graphSHA:string;candidate:NativeQ4Expert;previous?:NativeQ4Expert;reportName:string}){
 const byID=new Map<string,Row>();
 for(const row of [...input.records.train,...input.records.valid,...input.records.protection]){const previous=byID.get(row.id);assert(!previous||JSON.stringify(previous)===JSON.stringify(row));byID.set(row.id,row);}
 const records=[...byID.values()];assert(records.length>0&&input.coreVersion===input.candidate.core_version);
 const observer=new NativeQ4Observer();
 const observe=async(rows:Row[],expert?:NativeQ4Expert)=>{
  const pair=await observer.observePair({...o,core:input.core,expert,records:rows});
  for(const value of pair)assertNativeQ4Observation(value,rows,expert);
  return pair;
 };
 try{
  const baseline=await observe(records);
  const branchRows=input.records.protection.filter(row=>materialProvenance(row).retention_scope==='accepted_branch');
  assert(!branchRows.length||input.previous,'Protected parent rows require the current-core parent');
  const parent=branchRows.length?await observe(branchRows,input.previous):null;
  const candidate=await observe(records,input.candidate);
  const matched=(pair:readonly NativeQ4Observation[])=>pair[0]!.rows.every((row,i)=>{const other=pair[1]!.rows[i]!;return row.id===other.id&&JSON.stringify(row.tokens)===JSON.stringify(other.tokens)&&JSON.stringify(row.input_tokens)===JSON.stringify(other.input_tokens)&&row.complete===other.complete;});
  const passed=(pair:readonly NativeQ4Observation[],ids:readonly string[])=>pair.every(value=>ids.every(id=>value.rows.some(row=>row.id===id&&row.passed&&row.complete)));
  const ids=(rows:Row[])=>rows.map(row=>row.id),coreRows=input.records.protection.filter(row=>materialProvenance(row).retention_scope!=='accepted_branch');
  const oldCorrect=[...new Set(baseline.flatMap(value=>value.rows.filter(row=>row.passed&&row.complete).map(row=>row.id)))];
  const gates={q4_training:passed(candidate,ids(input.records.train)),q4_validation:passed(candidate,ids(input.records.valid)),
   q4_core_protection_baseline:passed(baseline,ids(coreRows)),q4_parent_protection_baseline:parent?passed(parent,ids(branchRows)):branchRows.length===0,
   q4_protection:passed(candidate,ids(input.records.protection)),q4_baseline_correct_retained:passed(candidate,oldCorrect),
   q4_deterministic:matched(baseline)&&matched(candidate)&&(!parent||matched(parent)),
   q4_same_inputs:candidate.every(value=>value.rows.every((row,i)=>JSON.stringify(row.input_tokens)===JSON.stringify(baseline[0].rows[i]!.input_tokens)))};
  const binary=baseline[0].worker_binary_sha256;assert([...baseline,...candidate,...(parent??[])].every(value=>value.worker_binary_sha256===binary&&value.identity===o.identity&&value.core?.sha256===input.core.sha256));
  await writeFile(join(o.work,'expert-q4-'+input.reportName),JSON.stringify({identity:o.identity,core_version:input.coreVersion,gates,baseline,parent,candidate},null,2),{flag:'wx',mode:0o600});
  assert(Object.values(gates).every(Boolean),'Expert ordinary Q4 quality/protection failed');
  const native_q4=Object.freeze({schema:'native-expert-q4-v1',identity:o.identity,passed:true as const,core_version:input.coreVersion,core_file_sha256:input.core.sha256,
   core_tensor_sha256:input.coreTensorSHA,expert_file_sha256:input.candidate.sha256,worker_binary_sha256:binary,graph_binary_sha256:input.graphSHA,report:input.reportName});
  const value=Object.freeze({native_q4,gates:Object.freeze(gates),baseline,parent,candidate});proofs.add(value);return value;
 }finally{await closeImportOwner(()=>observer.close(),error=>console.error('Expert Q4 owner retained: '+String(error)));}
}
