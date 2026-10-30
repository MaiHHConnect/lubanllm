import {assertNativeEvolutionPending} from './native-evolution-journal.ts';
import {nativeEvolutionPreviewMemory,type NativeEvolutionPreviewMemory} from './native-evolution-preview.ts';
import {closeNativeEvolutionRelease} from './native-evolution-release.ts';
/** Bundled by build-native-evolution-worker.ts; never import the historical sources directly. */
import {NativeEngine} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {Memory} from '../.native-migration/research/native-product-runtime/backend/memory.ts';
import {NativeCore450Owner} from '../.native-migration/research/native-core450-owner-controller/owner.ts';
import {runNativeCore450ProductJob,type NativeCore450ProductJobRequest} from '../.native-migration/research/native-core-product-job-source-v1/job.ts';
import {withEvolutionRound} from '../.native-migration/research/native-observation-python-parity-v2/round.ts';
import {borrowNativeEnginePeer,readNativeRetainedSource} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {nativeCompiledTrainingImplementation} from '../.native-migration/backend/native-training-start-v4.ts';
import {withNativeEvolutionMaterial,type NativeEvolutionMaterial} from './native-evolution-cohort.ts';
import {selectIndependentNativeEvolutionCohort} from './native-evolution-cohort.ts';
import assert from 'node:assert/strict';
import {join,resolve} from 'node:path';
import {mkdir,readFile,access,realpath} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import runtime from '../build/native-evolution/runtime-authority.json';
import {MODEL_BASE} from '../.native-migration/research/native-unified-service/compiled-model-v6.ts';
import {NONE,BUILD,FD,RUNTIME_ROOT} from '../.native-migration/research/native-unified-service/cache-build-admission-v6.ts';
import {readNativeEffectiveCore} from './native-effective-core.ts';
import {openNativeEvolutionSession} from './native-evolution-session.ts';
import {readNativeEvolutionGraphQualification} from './native-evolution-graph-qualification.ts';
import type {NativeEvolutionSession} from './native-evolution-service.ts';
import {nativeEvolutionResumeSupported,nativeEvolutionResumeMaterial} from './native-evolution-resume.ts';

export {NativeEngine,Memory};
export {NativeCore450Owner};
export {openNativeExpertRuntime,inspectNativeExpertRuntime} from './native-expert-worker.ts';
export {openMLXLearningRuntime,inspectMLXLearningRuntime} from './mlx-learning-runtime.ts';
export {stageNativeEvolutionFinal,admitNativeEvolutionFinal,type NativeEvolutionFinalRequest,type NativeEvolutionStagedFinal} from './native-evolution-coordinator.ts';
export {readNativeEvolutionGraphQualification,type NativeEvolutionGraphQualification} from './native-evolution-graph-qualification.ts';
export type {NativeConfig} from '../.native-migration/research/native-unified-expert/native-engine.ts';
export type {NativeCore450ProductJobRequest};

const roundOwners=new WeakMap<NativeEngine,Set<NativeCore450Owner>>();
async function closeRoundOwners(engine:NativeEngine){const owners=roundOwners.get(engine);if(!owners)return;for(const owner of owners){await owner.close();owners.delete(owner);}if(!owners.size)roundOwners.delete(engine);}

/** One fixed 450-update round. Quality acceptance and adoption remain explicit caller duties. */
export async function runNativeEvolutionRound(engine:NativeEngine,request:NativeCore450ProductJobRequest,material:NativeEvolutionMaterial|null=null){
 assertNativeEvolutionPending(request.memory,engine.config.identity,request.resumeId);
 const owner=new NativeCore450Owner(),retained=roundOwners.get(engine)??new Set<NativeCore450Owner>();retained.add(owner);roundOwners.set(engine,retained);
 let closed=false;
 // finishScope keeps the original guard alive after a failed rollback. Retry that same owner;
 // never abandon its promise or manufacture a replacement Engine while native work is retained.
 const monitor=(async()=>{while(!closed){if(owner.status().blocked)await owner.close();await Bun.sleep(100);}})();
 void monitor.catch(()=>{});
 try{
  const work=withNativeEvolutionMaterial(request.memory,material,()=>withEvolutionRound(engine,()=>runNativeCore450ProductJob(engine,request,owner),
   async(worker,enabled)=>engine.withSourceQuiescent(async guard=>{
    const peer=borrowNativeEnginePeer(engine,guard);
    if(peer.worker!==worker)throw new Error('Evolution restoration changed the resident worker');
    worker.assertLive();if(worker.isBranchEnabled!==enabled)await worker.setBranchEnabled(enabled);
   })));
  return await Promise.race([work,monitor.then(()=>{throw Error('Native owner monitor ended before work');})]);
 }finally{closed=true;await owner.close();retained.delete(owner);if(!retained.size)roundOwners.delete(engine);await monitor.catch(()=>{});}
}

/** Resume keeps the existing journal id, prepared fixture, baseline and saved optimizer/RNG. */
export async function runNativeEvolutionResume(engine:NativeEngine,request:NativeCore450ProductJobRequest,resumeId:string,material:NativeEvolutionMaterial|null=null){
 assert(/^[a-f0-9]{12}$/.test(resumeId),'Original native checkpoint id required');
 if(material===null){const row=request.memory.connect(db=>db.query('SELECT id,identity,kind,state,body FROM training_runs WHERE id=? AND identity=?').get(resumeId,engine.config.identity));material=nativeEvolutionResumeMaterial(row,engine.config.identity);}
 return runNativeEvolutionRound(engine,{...request,resumeId},material);
}

export type NativeEvolutionRuntimeInput={home:string;model:string;store:string;root:string;identity:string;work:string;manifest_sha256?:string;activeCore?:string|null;material?:NativeEvolutionMaterial|null};
export function nativeEvolutionRuntimeResources(){
 const role=(name:keyof typeof runtime.roles)=>({path:join(RUNTIME_ROOT,runtime.roles[name].path),sha256:runtime.roles[name].sha256});
 return{base:{...MODEL_BASE,core_shape:[...MODEL_BASE.core_shape] as [number,number]},role,release:{path:join(RUNTIME_ROOT,runtime.authority.manifest_relative),sha256:runtime.authority.manifest_sha256},training:{...runtime.training},cache:{NONE,BUILD,FD}};
}
const Q4_OBSERVER_SHA = 'BUILD_BOUND_Q4_OBSERVER_SHA';
let q4Resources:Promise<{executable:string;binary:string}>|undefined;
/** Fixed product paths, admitted once; no caller chooses an executable or per-sample rehash. */
export function nativeEvolutionQ4Resources(){return q4Resources??=(async()=>{
 const product=resolve(import.meta.dir,'../backend/native'),development=resolve(import.meta.dir,'../native');
 let directory=product;try{await access(join(product,'luban-evolution-observe'));}catch(error:any){if(error.code!=='ENOENT')throw error;directory=development;}
 const executable=join(directory,'luban-evolution-observe'),binary=join(directory,'luban-worker');
 const receipt=JSON.parse(await readFile(executable+'.build.json','utf8'));
 assert(receipt.schema==='luban-native-evolution-observer-build-v1'&&receipt.compiled===true&&receipt.signed===true&&receipt.binary_sha256===Q4_OBSERVER_SHA);
 assert.equal(createHash('sha256').update(await readFile(executable)).digest('hex'),Q4_OBSERVER_SHA,'Q4 observer differs from compiled product binding');
 await access(binary);return{executable,binary};
})();}
type PreviewMemory=NativeEvolutionPreviewMemory;
/** CPU-only material selection; scheduler may lend its existing ledger instead of opening a second lease. */
export async function inspectNativeEvolutionRuntime(input:{root:string;store:string;identity:string;memory?:PreviewMemory}){
 assert.equal(input.identity,MODEL_BASE.identity,'Native evolution base identity differs');
 const owned=input.memory?null:new Memory(input.root);
 if(owned)owned.model_identity=input.identity;
 const memory=input.memory?await nativeEvolutionPreviewMemory(input.memory,input.root,input.identity):owned!;
 try{const pending=memory.connect(db=>db.query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_runs'").get()?db.query("SELECT id,identity,kind,state,body FROM training_runs WHERE identity=? AND state IN ('queued','running','paused','committing')").get(input.identity):null);
 const material=pending?nativeEvolutionResumeMaterial(pending,input.identity):null;
 return await withNativeEvolutionMaterial(memory,material,async()=>{
  const cohort=selectIndependentNativeEvolutionCohort(memory,input.store,MODEL_BASE);
  let execution_ready=false,readiness_error:string|null=null;
  try{const {role}=nativeEvolutionRuntimeResources();readNativeEvolutionGraphQualification({identity:input.identity,graph_binary_sha256:role('persistentExpert').sha256,training_binary_sha256:role('trainer').sha256});await nativeEvolutionQ4Resources();execution_ready=true;}catch(error){readiness_error=String(error);}
  const resume_supported=!!pending&&nativeEvolutionResumeSupported(pending,{identity:input.identity,records:cohort.records,core_version:cohort.core?.version??null,implementation:nativeCompiledTrainingImplementation()});
  return{records:cohort.records,core:cohort.core?.version??null,unspent_count:cohort.unspent_count,execution_ready,readiness_error,resume_supported};
 });}finally{owned?.close();}
}
/** Runtime paths are compiled from the explicit capsule; caller paths never select an executable. */
export async function openNativeEvolutionEngine(input:NativeEvolutionRuntimeInput){
 assert.equal(input.identity,MODEL_BASE.identity,'Native evolution base identity differs');
 assert(input.manifest_sha256===undefined||input.manifest_sha256===MODEL_BASE.model_manifest_sha256);
 for(const value of [input.home,input.model,input.store,input.root,input.work])assert(typeof value==='string'&&resolve(value)===value&&!value.includes('\0'),'Native runtime requires absolute paths');
 const role=(name:keyof typeof runtime.roles)=>({path:join(RUNTIME_ROOT,runtime.roles[name].path),sha256:runtime.roles[name].sha256});
 const release={path:join(RUNTIME_ROOT,runtime.authority.manifest_relative),sha256:runtime.authority.manifest_sha256};
 const bridge=role('bridge'),worker=role('worker'),peer=role('persistentExpert'),trainer=role('trainer'),rng=role('rng'),basis=role('replayBasis');
 const memory=new Memory(input.root);memory.model_identity=input.identity;
 // Always freeze a newly-read accepted pointer for each session, including later continuous rounds.
 let core;try{core=readNativeEffectiveCore(input.store,nativeEvolutionRuntimeResources().base).state.core;}catch(error){memory.close();throw error;}
 const engine=new NativeEngine(bridge.path,bridge.sha256,{binary:worker.path,model:input.model,manifest_sha256:MODEL_BASE.model_manifest_sha256,identity:input.identity,log:join(input.work,'bridge.stderr'),
  aten:{base:{...MODEL_BASE,core_shape:[...MODEL_BASE.core_shape]},none_build:NONE,core_build:BUILD,fd_helper:FD},expert_build:{...release,binary_sha256:peer.sha256},
  routing:{store:input.store,core_parameter:MODEL_BASE.core_parameter,core_shape:[...MODEL_BASE.core_shape]},
  ...(core?{core:{file:join(input.store,core.version+'.safetensors'),tensor_key:MODEL_BASE.core_parameter,sha256:core.file_sha256}}:{})},300000);
 let closed=false,engineClosed=false;
 const close=async()=>{if(closed)return;if(!engineClosed){await closeRoundOwners(engine);await engine.close();engineClosed=true;}await closeNativeEvolutionRelease(engine);memory.close();closed=true;};
 // Open actual retained weights without ordinary chat route qualification or an
 // initial router/RNG snapshot. The producer owns the one real rollback capture.
 let retained:ReturnType<typeof readNativeRetainedSource>;
 try{retained=await engine.withSourceQuiescent(async(guard,control)=>{guard.assertQuiescent();await control.openRetainedSource(input.work);return readNativeRetainedSource(engine);});}catch(error){try{await close();}catch(cleanup){throw new AggregateError([error,cleanup],'Native runtime start and cleanup failed');}throw error;}
 assert(retained,'Native evolution retained source was not opened');
 return{engine,memory,close,retained};
}
export async function openNativeEvolutionRuntime(input:NativeEvolutionRuntimeInput):Promise<NativeEvolutionSession>{
 return openNativeEvolutionSession({...input,model:await realpath(input.model)});
}

export function nativeEvolutionWorkerCapabilities(){
 return {implementation:nativeCompiledTrainingImplementation(),training_target:'expert',candidate_max_steps:64,quality_early_stop:true,legacy_core_fixed_steps:450,
  dynamic_materials:true,python_observer:false,quality_admitted:false,product_admitted:false};
}

if(import.meta.main){
 if(process.argv.length!==3||process.argv[2]!=='--capabilities')throw new Error('Use --capabilities or import this worker from the native coordinator');
 console.log(JSON.stringify(nativeEvolutionWorkerCapabilities()));
}
