import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";

test("actual publication control flow commits only after reload, rolls back pending intents and never rolls back a committed run", () => {
  const child = spawnSync(process.execPath, ["--eval", `
import assert from 'node:assert/strict';import{mock}from'bun:test';import{resolve}from'node:path';
const stub=(name,value)=>mock.module(resolve(name),()=>value),old='.native-migration/research/';
let scenario,calls,pointer,loaded,finished,reloadCount,bindCount;
const failure=new Error('injected'),forbidden=()=>{throw new Error('extra snapshot/install forbidden')};
class Owner{async finishScope(restore,close){await restore();await close();}}
class Snapshots{bindAdoption(){calls.push('bind');if(scenario==='bind'&&bindCount++===0)throw failure;}close(){calls.push('close');}}
stub(old+'native-unified-expert/native-engine.ts',{NativeEngine:class{}});
stub(old+'native-core450-owner-controller/owner.ts',{NativeCore450Owner:Owner});
stub(old+'native-unified-expert/class-closure/native-product-runtime/backend/evolution-snapshots.ts',{EvolutionSnapshots:Snapshots});
stub(old+'native-unified-expert/class-closure/native-product-runtime/backend/native-evolution-state.ts',{readNativeEvolutionSnapshot:()=>({row:{manifest_sha256:'prior'}})});
stub(old+'native-product-runtime/backend/evolution-snapshot-files.ts',{readSafe:()=>Buffer.from('artifact'),encoded:JSON.stringify,hash:()=> 'fixture'});
stub(old+'native-core-product-publication-v1/commit-storage.ts',{
 prepareNativeCoreCommit:()=>{calls.push('prepare');if(scenario==='prepare')throw failure;},
 applyNativeCoreCommit:()=>{calls.push('apply');pointer='candidate';if(scenario==='apply')throw failure;},
 finishNativeCoreCommit:(_r,receipt)=>{calls.push('finish');assert.equal(pointer,'candidate');assert.equal(loaded,'candidate');assert.equal(receipt.loaded,'candidate');if(scenario==='finish')throw failure;finished=true;},
 restoreNativeCoreCommitPointer:()=>{calls.push('rollback');assert(!finished);pointer='previous';},
 finishNativeCoreCommitRejection:(_r,receipt)=>{calls.push('reject');assert.equal(pointer,'previous');assert.equal(loaded,'previous');assert.equal(receipt.loaded,'previous');}
});
stub(old+'native-core-peer-refresh-source-v1/reload.ts',{settleRetainedCoreReload:async()=>{calls.push('settle');return loaded==='candidate';}});
const checkpoint={approval:{id:'123456789abc'},binding:{},previous_snapshot:{id:'prior-id',manifest_sha256:'prior'}};
const quality={checkpoint,report:{partial_baseline:[],partial_candidate_first:[],partial_candidate_second:[]}};
stub('backend/native-evolution-admission.ts',{
 requireCore450Publication:()=>{calls.push('authority');if(scenario==='unqualified')throw failure;},
 validateCore450QualityObservations:()=>({context:{memory:{connect:fn=>fn({})},store:'/unused',base:{}},cohort:{records:{groups:{},protection:[],retention_regressions:[]}}})
});
stub('backend/native-evolution-artifacts.ts',{validateCore450CandidateArtifacts:()=>{},publicationCore450Artifacts:()=>({report:{},replacements:{}}),immutableCore450Artifact:async()=>{calls.push('artifact');}});
stub('backend/native-evolution-quality.ts',{nativeEvolutionProgress:()=>({learned_ids:['learned']})});
stub('backend/native-evolution-material.ts',{uniqueNativeEvolutionRecords:()=>[]});
const{publishCore450Candidate}=await import(resolve('backend/native-evolution-publication.ts'));
const candidate={quality,report:{},directory:'/unused',pointer:{version:'core-123456789abc',sha256:'weight'},artifacts:{'core.json':'fixture','core.safetensors':'fixture','core.retention.json':'fixture','evolution-123456789abc.json':'fixture'},assertUnchanged:async()=>{}};
for(scenario of ['success','unqualified','prepare','apply','reload','finish']){
 calls=[];pointer='previous';loaded='previous';finished=false;reloadCount=0;bindCount=0;let error,result;
 const control={captureRouterInitial:forbidden,installCoreCheckpoint:forbidden,restoreRetainedCorePrevious:forbidden,
 prepareRouteTokens:async()=>({}),prepareRetainedCoreReload:async()=>{calls.push('transfer');return{};},
 reloadRetainedCoreSource:async()=>{calls.push('reload');loaded=pointer;if(scenario==='reload'&&reloadCount++===0)throw failure;return{loaded};}};
 try{result=await publishCore450Candidate({}, {},control,candidate,new Owner(),[]);}catch(e){error=e;}
 if(scenario==='success'){assert.equal(result.state,'committed');assert(finished);assert.equal(pointer,'candidate');assert(calls.indexOf('finish')>calls.indexOf('reload'));}
 else assert.equal(error,failure);
 if(scenario==='unqualified'){assert.deepEqual(calls,['authority']);assert.equal(pointer,'previous');}
 if(['prepare','apply','reload','finish'].includes(scenario)){assert(!finished);assert.equal(pointer,'previous');assert.equal(loaded,'previous');assert(!calls.includes('bind')&&!calls.includes('close'));}
 if(['apply','reload','finish'].includes(scenario)){assert(calls.includes('rollback')&&calls.includes('reject'));assert(calls.indexOf('reject')>calls.indexOf('rollback'));}
 if(scenario==='bind'){assert(finished);assert.equal(pointer,'candidate');assert(!calls.includes('rollback'));assert.equal(bindCount,2);}
}
console.log(JSON.stringify({passed:true,cases:6,extra_snapshots:0}));
`], { cwd: process.cwd(), encoding: "utf8", timeout: 30000 });
  expect(child.status, child.stderr).toBe(0);
  expect(JSON.parse(child.stdout)).toEqual({ passed: true, cases: 6, extra_snapshots: 0 });
});
