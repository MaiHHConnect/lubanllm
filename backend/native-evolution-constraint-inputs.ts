/** Actual predictor positions, kept separate from TRAIN rows and the seven optimizer tensors. */
import assert from 'node:assert/strict';
import {evaluateGeneration,validateGenerationCheck} from './generation-checks.ts';
import {materialProvenance} from './native-evolution-material.ts';
export const FINAL_CONVERSION = 'gptq-bf16-refine-v1';
export function predictorInputs(raw:Buffer,labels:readonly number[],width:number){
 assert(Number.isSafeInteger(width)&&width>0&&labels.length>=2&&raw.length===labels.length*width*2);
 const positions=labels.slice(1).flatMap((value,index)=>value===-100?[]:[index]);assert(positions.length>0);
 const output=Buffer.alloc(positions.length*width*4);
 for(const[row,position]of positions.entries())for(let col=0;col<width;col++){
  const bits=raw.readUInt16LE((position*width+col)*2);assert((bits&0x7f80)!==0x7f80,'Nonfinite constraint activation');output.writeUInt32LE(bits*65536,(row*width+col)*4);
 }
 return output;
}
export function q4ConstraintFrame(event:any,request:{id:string;input_tokens:readonly number[];tokens:readonly number[]},step:number,tail:Buffer,logits:Buffer){
 assert(event.request_id===request.id&&event.step===step&&event.committed_tokens===request.input_tokens.length+step);
 assert(event.tail_only===true&&event.layer===23&&event.tail_values===10240&&event.vocab===248320&&tail.length===20480&&logits.length===496640);
 let top=0,max=-Infinity;const f=Buffer.alloc(4);
 for(let i=0;i<logits.length;i+=2){const bits=logits.readUInt16LE(i);assert((bits&0x7f80)!==0x7f80,'Nonfinite Q4 logits');f.writeUInt32LE(bits*65536);const value=f.readFloatLE();if(value>max){max=value;top=i/2;}}
 assert(top===event.top1&&top===request.tokens[step],'Q4 same-source calibration token differs');
 const output=Buffer.alloc(6144*4);
 for(let i=0;i<10240;i++){const bits=tail.readUInt16LE(i*2);assert((bits&0x7f80)!==0x7f80,'Nonfinite Q4 tail');if(i>=2048&&i<8192)output.writeUInt32LE(bits*65536,(i-2048)*4);}
 return output;
}
export function validateConstraintFixture(config:Record<string,string>,tensors:Record<string,any>,width:number){
 const tensor=tensors.constraint_inputs;
 if(config.final_conversion===undefined){assert(tensor===undefined&&config.constraint_old_positions===undefined,'Unbound conversion tensor');return;}
 assert(config.final_conversion===FINAL_CONVERSION&&tensor?.dtype==='f32'&&tensor.shape.length===2&&tensor.shape[1]===width&&tensor.shape[0]>1&&tensor.shape[0]*width<=10000000,'Invalid final conversion matrix');
 assert(/^[1-9][0-9]*$/.test(config.constraint_old_positions!)&&Number(config.constraint_old_positions)<tensor.shape[0],'Both OLD and Q4 predictor groups are required');
 assert.equal(tensor.bytes,tensor.shape[0]*width*4);
}

/** The preparation owns these exact OLD observations; no caller passed flag replaces semantic checks. */
export function preparedOldBaseline(conversion:any,records:Record<string,any>[]){
 assert(conversion?.method===FINAL_CONVERSION&&conversion.training===false&&conversion.new_validation_used===false);
 const rows=conversion.old_baseline;assert(Array.isArray(rows)&&rows.length===records.length);
 for(const[index,record]of records.entries()){
  const row=rows[index];assert(row.id===record.id&&row.passed===true&&row.complete===true&&row.branch_enabled===false&&row.reference_used===false&&row.finish_reason==='stop'&&Array.isArray(row.raw_completion_token_ids)&&row.raw_completion_token_ids.at(-1)===248044,'Prepared OLD baseline identity/completion differs');
  const check=validateGenerationCheck(materialProvenance(record).generation_check,record.answer);
  assert(evaluateGeneration(row.answer,check,true).passed,'Prepared OLD baseline semantics differ');
 }
 return rows as Record<string,any>[];
}
