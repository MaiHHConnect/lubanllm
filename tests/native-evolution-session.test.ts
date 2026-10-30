import { expect, test } from "bun:test";
import { spawnSync } from "node:child_process";

test("product session serializes models, retains failed owners, preserves pending and resumes the real job", () => {
  const child = spawnSync(process.execPath, ["--eval", `
import assert from 'node:assert/strict';import{mock}from'bun:test';import{resolve,join}from'node:path';
import{mkdtemp,rm,access}from'node:fs/promises';import{tmpdir}from'node:os';
const stub=(name,value)=>mock.module(resolve(name),()=>value);
let scenario,calls,live,engines,observers,ownerAttempts,engineCloseAttempts,q4CloseAttempts;
let memoryOpen=false;
const failure=new Error('injected');const savedMaterial={samples:[{id:'saved'}],directedTeaching:{authorization:'persisted'},retentionRegressions:[]};
stub('backend/native-evolution-cohort.ts',{withNativeEvolutionMaterial:async(_m,_v,fn)=>fn(),selectIndependentNativeEvolutionCohort:()=>({records:{groups:{replay_training:[{id:'replay'}]}},core:null})});
stub('backend/native-evolution-journal.ts',{assertNativeEvolutionPending:()=>{}});
stub('backend/native-evolution-calibration.ts',{withNativeEvolutionCalibration:async(value,records,fn)=>{assert(value&&records);calls.push('calibration-bound');return fn();}});
stub('backend/native-evolution-resume.ts',{nativeEvolutionResumeMaterial:(row,identity)=>{assert.equal(row.id,'123456789abc');assert.equal(identity,'fixed');calls.push('saved-material');return savedMaterial;}});
class Rejection extends Error{report={learning_progress:{pending_ids:['new']},gates:{q4:false}};}
stub('backend/native-evolution-q4-quality.ts',{NativeEvolutionQualityRejection:Rejection});
stub('backend/native-evolution-graph-qualification.ts',{readNativeEvolutionGraphQualification:()=>{calls.push('graph');if(scenario==='unqualified')throw failure;return{};}});
const result={step:450,phase:'final',previous_model_restored:true,rollback_scope_completed:true,expected:{},checkpoint:{},previous_snapshot:{},final_evidence:{report:{arithmetic_gates_passed:true}}};
async function run(_engine,_request){calls.push('train');_request.onProgress({step:225,steps:450,phase:'training'});if(scenario==='training')throw failure;return scenario==='partial'?{...result,step:225}:result;}
stub('backend/native-evolution-worker.ts',{
 Memory:class {constructor(){memoryOpen=true;} connect(read){return read({query:()=>({get:()=>({id:'123456789abc'})})});} close(){memoryOpen=false;calls.push('memory-close');}},
 nativeEvolutionRuntimeResources:()=>({base:{},role:name=>({path:'/fixed/'+name,sha256:'fixed'}),release:{path:'/fixed/release',sha256:'fixed'},training:{}}),
 nativeEvolutionQ4Resources:async()=>{calls.push('q4resources');return{executable:'/fixed/q4',binary:'/fixed/worker'};},
 openNativeEvolutionEngine:async()=>{assert.equal(live,0,'overlapping models');live++;const id=++engines;calls.push('engine-open-'+id);let closed=false;return{engine:{id},memory:{connect:read=>read({query:()=>({get:()=>({id:'123456789abc'})})})},close:async()=>{calls.push('engine-close-'+id);if((scenario==='engine-close'||(scenario==='committed-close'&&id===2))&&engineCloseAttempts++===0)throw failure;if(!closed){live--;closed=true;}}};},
 runNativeEvolutionRound:run,
 runNativeEvolutionResume:async(engine,request,id,material)=>{assert.equal(material,savedMaterial);assert.equal(id,'123456789abc');calls.push('resume');return{...result};}
});
class Observer{constructor(){this.id=++observers;this.open=false;}async captureTraining(request){assert.equal(live,0);assert(!memoryOpen,'DB lock must be closed before GPU');this.id=0;observers--;calls.push('q4-capture');if(scenario==='capture-fail'||scenario==='capture-cancel'){this.open=true;live++;if(scenario==='capture-cancel')request.signal.controller.abort(failure);throw failure;}return{baseline:{},raw:Buffer.alloc(4),rows:[]};}async observePair(request){assert.equal(live,0,'overlapping models');this.open=true;live++;calls.push('q4-pair-'+this.id);if(scenario==='q4-observe'||scenario==='q4-close')throw failure;if(scenario==='cancel-q4'){request.signal.controller.abort(failure);throw failure;}await this.close();return[{},{}];}async close(){calls.push('q4-close-'+this.id);if(this.id!==0&&scenario==='q4-close'&&q4CloseAttempts++===0)throw failure;if(this.open){live--;this.open=false;}}}
stub('backend/native-evolution-q4.ts',{NativeQ4Observer:Observer});
stub('backend/native-evolution-coordinator.ts',{
 stageNativeEvolutionFinal:async(_engine,_request,owner)=>{calls.push('stage');if(scenario==='resume')assert.equal(_request.material,savedMaterial);if(scenario==='stage')throw failure;if(scenario==='owner-close'){owner.own({},async()=>{calls.push('owner-close');if(++ownerAttempts<=3)throw failure;});await owner.finishScope(async()=>{},async()=>{});}return{manifestSHA:'fixed',identity:'fixed',records:[{id:'one'}],core:{file:'/staged/core'},baselineCore:null};},
 admitNativeEvolutionFinal:async(_engine,_request)=>{calls.push('admit');if(scenario==='resume')assert.equal(_request.material,savedMaterial);if(scenario==='reject-q4')throw new Rejection('quality failed');if(scenario==='admit')throw failure;return{state:'committed',actual_reload_confirmed:true,previous_snapshot_bound:true,report:{learning_progress:{pending_ids:scenario==='mastered'?[]:['new']},split_ids:{new_training:scenario==='waiting-material'?['mastered-train']:['new']}}};}
});
const{openNativeEvolutionSession}=await import(resolve('backend/native-evolution-session.ts'));
const root=await mkdtemp(join(tmpdir(),'native-session-'));
try{
for(scenario of ['success','mastered','waiting-material','unqualified','partial','training','stage','engine-close','q4-observe','q4-close','cancel-q4','owner-close','reject-q4','admit','resume','committed-close','capture-fail','capture-cancel']){
 calls=[];live=0;engines=0;observers=0;ownerAttempts=0;engineCloseAttempts=0;q4CloseAttempts=0;
 const work=join(root,scenario);await import('node:fs/promises').then(m=>m.mkdir(work));let session,output,error;
 const progress=[];const controller=new AbortController();controller.signal.controller=controller;
 try{session=await openNativeEvolutionSession({work,identity:'fixed',home:root,model:root,store:root,root});output=await session.runRound({signal:controller.signal,round:1,...(scenario==='resume'?{resume:'123456789abc'}:{}),onProgress:value=>progress.push(value)});}catch(e){error=e;}
 if(['success','mastered','waiting-material','resume','committed-close'].includes(scenario)){assert(!error);assert.equal(output.adopted,true);assert.equal(output.pending,scenario!=='mastered');assert.equal(live,1);assert(calls.indexOf('engine-close-1')<calls.indexOf('q4-pair-1'));assert(calls.indexOf('q4-close-2')<calls.indexOf('engine-open-2'));assert(calls.indexOf('admit')>calls.indexOf('engine-open-2'));}
 else if(scenario==='partial'){assert(!error);assert.equal(output.pending,true);assert.equal(output.adopted,false);assert(!calls.includes('stage'));}
 else if(scenario==='reject-q4'){assert(!error);assert.equal(output.pending,true);assert.equal(output.quality_checked,true);assert.equal(output.adopted,false);assert.equal(output.report.gates.q4,false);}
 else assert(error,scenario+' must fail');
 if(scenario==='waiting-material'){assert.equal(output.phase,'committed_waiting_material');assert.equal(output.pending,true);assert.match(output.continuation_reason,/验证题/);} if(scenario==='unqualified')assert.deepEqual(calls,['graph']);
 if(scenario==='engine-close'){assert.equal(live,1);assert.equal(observers,0);}
 if(scenario==='owner-close'){assert.equal(live,1);assert.equal(ownerAttempts,3);assert.equal(observers,0);}
 if(scenario==='q4-close'){assert.equal(live,1);assert.equal(engines,1);await assert.rejects(session.close(),e=>e===failure);assert.equal(live,1);}
 if(scenario==='committed-close'){await assert.rejects(session.close(),e=>e===failure);await access(join(work,'round-1'));assert.equal(live,1);}
 if(session){await session.close();await session.close();}
 if(output?.adopted){await assert.rejects(access(join(work,'round-1')));await access(work);}else if(scenario!=='unqualified')await access(join(work,'round-1'));
 if(calls.includes('train'))assert(progress.some(p=>p.step===225&&p.phase==='training'));
 assert.equal(live,0,scenario+' leaked owner');assert(!memoryOpen,scenario+' DB leak');
 if(scenario==='resume')assert(!calls.includes('q4-capture'));else if(calls.includes('train'))assert(calls.indexOf('q4-capture')<calls.indexOf('engine-open-1'));
 if(scenario==='capture-fail'||scenario==='capture-cancel')assert.equal(engines,0);
 if(scenario==='owner-close')assert.equal(ownerAttempts,4);
 if(scenario==='engine-close')assert.equal(calls.filter(c=>c==='engine-close-1').length,2);
 if(scenario==='q4-close')assert.equal(calls.filter(c=>c==='q4-close-1').length,2);
 if(scenario==='resume'){assert.equal(session.continuationMaterial(),savedMaterial);assert(calls.includes('saved-material'));assert(calls.includes('resume'));assert(!calls.includes('train'));}
}
console.log(JSON.stringify({passed:true,cases:18}));
}finally{await rm(root,{recursive:true,force:true});}
`], { cwd: process.cwd(), encoding: "utf8", timeout: 30_000 });
  expect(child.status, child.stderr).toBe(0);
  expect(JSON.parse(child.stdout)).toEqual({ passed: true, cases: 18 });
});
