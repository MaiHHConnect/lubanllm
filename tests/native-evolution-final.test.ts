import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";

// Isolate dependency stubs from all other tests. Execute the real final observer's failure/finally path.
test("final observer restores resident current on generation/loss/cancel failure and preserves double errors", () => {
  const result = spawnSync(process.execPath, ["--eval", `
import assert from 'node:assert/strict';
import {mock} from 'bun:test';
import {resolve} from 'node:path';
const root=process.cwd(), old='.native-migration/research/';
const stub=(name,value)=>mock.module(resolve(root,name),()=>value);
let scenario, calls=[], mode='current', controller, lossTargets=[];
const failure=new Error('baseline failed'), restoreFailure=new Error('current select failed');
const forbidden=()=>{throw new Error('Forbidden validation snapshot/hash/router/RNG path');};
const row=(id,purpose='validation')=>({id,prompt:id,answer:'ok',purpose,provenance:{generation_check:{type:'exact',expected:'ok'}}});
const records={groups:{new_training:[row('train','training')],new_validation:[row('valid')],replay_training:[row('replay-t','training')],replay_validation:[row('replay-v')]},protection:[row('protect')],retention_regressions:[row('boundary')]};
const report={partial_records_sha256:'records',partial_training_baseline:[],baseline_training_replay_generation:[],baseline_core_retention_generation:[]};
const cap={approval:{step:450,id:'123456789abc'},binding:{records_sha256:'frozen'},checkpoint:{payload:{},binding:{records_sha256:'frozen'}},assertUnchanged:forbidden};
cap.learning_reference={sha256:'original'};cap.state={candidate:{sha256:'candidate'}};
const product={records,assertUnchanged:forbidden};
const guard={};
stub(old+'native-evolution-producer/producer.ts',{borrowNativeEvolutionPreparation:()=>({})});
stub(old+'native-core450-owner-controller/metadata.ts',{core450ControlMetadata:()=>({report,step:450})});
stub(old+'native-unified-expert/native-engine.ts',{NativeEngine:class{},assertNativeExpertScopeGuard:(_,g)=>assert.equal(g,guard)});
stub(old+'native-core450-runner/training-safety.ts',{validateCore450TrainingHistory:()=>({complete_training_safety:true,completed_steps:450,partial_safety_failed:false})});
stub(old+'native-product-runtime/backend/native-core-resume.ts',{coreCheckpointField:forbidden,coreContentSHA:()=> 'records'});
stub(old+'native-unified-service/checkpoint-owner.ts',{checkpointApprovedBinding:x=>x,equalCheckpointValues:(a,b)=>JSON.stringify(a)===JSON.stringify(b)});
stub(old+'native-learning-product/checkpoint-save-product.ts',{saveNativeTrainingCheckpoint:forbidden});
stub(old+'native-product-runtime/backend/evolution-snapshot-files.ts',{equal:(a,b)=>JSON.stringify(a)===JSON.stringify(b)});
stub(old+'native-core450-owner-controller/runner.ts',{
 requireCore450PeerOperations:()=>({installCoreCheckpoint:forbidden,restoreCorePrevious:forbidden,restoreRetainedCorePrevious:forbidden,prepareRouteTokens:forbidden}),
 generateCore450Rows:async(_c,rows,_cap,signal,observation)=>{
   calls.push('generate:'+observation);assert.equal(mode,observation??'current');
   if(scenario==='cancel'){controller.abort(new Error('cancelled'));signal.throwIfAborted();}
   if(scenario==='generation'||scenario==='double')throw failure;
   return rows.map(r=>({id:r.id,answer:'ok',passed:true,complete:true,branch_enabled:false,reference_used:false}));
 },
 meanCore450Loss:async(_c,rows,_cap,_signal,observation)=>{calls.push('loss');if(scenario==='targets'){assert.equal(mode,observation??'current');if(rows.length===1&&rows[0].id==='protect')lossTargets.push({mode,row:rows[0]});return 1;}assert.equal(mode,'previous');throw failure;}
});
stub('backend/native-evolution-quality.ts',{finishNativeEvolutionQuality:()=>{assert.equal(scenario,'targets');throw failure;},nativeEvolutionProgress:forbidden});
stub('backend/native-evolution-cohort.ts',{selectIndependentNativeEvolutionCohort:()=>({records,records_sha256:'frozen',assertUnchanged:()=>{}})});
stub('backend/native-evolution-observation.ts',{selectNativeCoreObservation:async(_e,_g,_cap,next,options)=>{
 calls.push('select:'+next);
 if(next==='current'){assert.equal(options,undefined);if(scenario==='double')throw restoreFailure;}
 options?.signal?.throwIfAborted();mode=next;return{observation:next};
}});
const {observeCore450ProductFinal}=await import(resolve(root,'backend/native-evolution-final.ts'));
for(scenario of ['generation','loss','cancel','double','targets']){
 calls=[];mode='current';controller=new AbortController();let caught;
 if(scenario==='targets')records.protection[0].answer='';
 const r={memory:{},store:'unused',expected:{binding:cap.binding,kind:'evolution',id:cap.approval.id},signal:controller.signal};
 try{await observeCore450ProductFinal({config:{aten:{base:{}}}},guard,{},cap,new Proxy({},{get:forbidden}),r,{},product);}
 catch(e){caught=e;}
 assert(caught);assert.equal(calls[0],'select:previous');if(scenario!=='targets')assert.equal(calls.at(-1),'select:current');
 if(scenario==='double'){assert(caught instanceof AggregateError);assert.equal(caught.errors[0],failure);assert.equal(caught.errors[1],restoreFailure);assert.equal(caught.cause,failure);}
 else{assert.equal(mode,'current');if(scenario==='cancel')assert.equal(caught,controller.signal.reason);else assert.equal(caught,failure);}
 if(scenario==='loss')assert(calls.includes('loss'));
 if(scenario==='targets'){assert.equal(caught,failure);assert.deepEqual(lossTargets.map(x=>x.mode),['previous','current']);assert.equal(lossTargets[0].row,lossTargets[1].row);assert.equal(lossTargets[0].row.answer,'ok');assert.equal(records.protection[0].answer,'');}
}
console.log(JSON.stringify({passed:true,cases:5,validation_full_state_audits:0}));
`], { cwd: process.cwd(), encoding: "utf8", timeout: 30000 });
  expect(result.status, result.stderr).toBe(0);
  expect(JSON.parse(result.stdout)).toEqual({ passed: true, cases: 5, validation_full_state_audits: 0 });
});
