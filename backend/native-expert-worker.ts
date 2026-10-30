/** One real native Learning controller under the existing application compute lease. */
import assert from 'node:assert/strict';
import {join,resolve} from 'node:path';
import {mkdir} from 'node:fs/promises';
import {NativeEngine} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {NativeChat} from '../.native-migration/research/native-unified-expert/native-chat.ts';
import {Settings} from '../.native-migration/research/native-product-runtime/backend/settings.ts';
import {Memory,ledgerParse} from '../.native-migration/research/native-product-runtime/backend/memory.ts';
import {readSafe,equal} from '../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';
import {NativeLearning} from '../.native-migration/research/native-learning-product/controller.ts';
import {selectLearningCohort} from '../.native-migration/research/native-learning-product/cohort.ts';
import {nativeCompiledTrainingImplementation} from '../.native-migration/backend/native-training-start-v4.ts';
import {nativeEvolutionRuntimeResources,nativeEvolutionQ4Resources,type NativeEvolutionRuntimeInput} from './native-evolution-worker.ts';
import {nativeEvolutionPreviewMemory,type NativeEvolutionPreviewMemory} from './native-evolution-preview.ts';
import {closeNativeEvolutionRelease} from './native-evolution-release.ts';
import {nativeExpertStepBudget,nativeExpertChoices,nativeExpertAccumulatedStep,NATIVE_EXPERT_PROTOCOL} from './native-expert-learning.ts';
import {readNativeExpertGraphQualification} from './native-expert-qualification.ts';
import {nativeCoreProtectionAdmission,nativeTrialSourceReason} from './native-effective-core.ts';
import type {NativeEvolutionSession} from './native-evolution-service.ts';

export type NativeExpertRuntimeInput=NativeEvolutionRuntimeInput&{mode?:'learning'|'consolidation'};
function json(store:string,name:string,optional=false):any{try{return ledgerParse(readSafe(store,name,16<<20).toString());}catch(error:any){if(optional&&error.code==='ENOENT')return null;throw error;}}
/** Scheduler preview reads small metadata; execution still performs the original source admission. */
function selection(store:string,identity:string){
 const trialReason=nativeTrialSourceReason(store);
 const pointer=json(store,'active-core.json',true),core=pointer?json(store,pointer.version+'.json'):null;
 if(pointer){assert(pointer.identity===identity&&core.identity===identity&&core.version===pointer.version&&core.sha256===pointer.sha256);assert(nativeCoreProtectionAdmission(core,json(store,core.report)).qualified,'当前核心尚未通过正式逐条保护采纳');}
 if(core){assert(typeof (core.candidate_tensor_sha256??core.tensor_sha256)==='string','当前核心缺少实际tensor绑定');if(core.candidate_tensor_sha256&&core.tensor_sha256)assert.equal(core.candidate_tensor_sha256,core.tensor_sha256);}
 const physical=json(store,'active.json',true);
 if(trialReason)return{pointer,core,expert:null,physical_expert:physical,trialReason};
 if(physical)assert(physical.identity===identity&&typeof physical.id==='string'&&/^learned-[a-f0-9]{12}$/.test(physical.id)&&Number.isSafeInteger(physical.width)&&physical.width>=32&&physical.width<=512);
 const expertMetadata=physical?json(store,'layer-23/'+physical.id+'.json'):null;
 const expert=physical&&physical.core_version===(pointer?.version??null)&&expertMetadata?.core_version===(pointer?.version??null)?physical:null;
 return{pointer,core,expert,physical_expert:physical,trialReason:null};
}
function pending(memory:{connect<T>(f:(db:any)=>T):T},identity:string){return memory.connect(db=>db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'").get()?db.query("SELECT id,kind,identity,state,body FROM training_runs WHERE identity=? AND state IN ('queued','running','paused','committing')").get(identity):null);}
export async function inspectNativeExpertRuntime(input:{root:string;store:string;identity:string;memory?:NativeEvolutionPreviewMemory}){
 const resources=nativeEvolutionRuntimeResources();assert.equal(input.identity,resources.base.identity);
 const owned=input.memory?null:new Memory(input.root);if(owned)owned.model_identity=input.identity;
 const memory=input.memory?await nativeEvolutionPreviewMemory(input.memory,input.root,input.identity):owned!;
 try{
  const chosen=selection(input.store,input.identity),saved=pending(memory,input.identity),body=saved?ledgerParse(saved.body):null;
  let cohort:ReturnType<typeof selectLearningCohort>|undefined,readiness_error:string|null=null,waitingMaterial=false;
  try{assert(!chosen.trialReason,chosen.trialReason??'试用来源不能用于正式训练');cohort=selectLearningCohort(memory as Memory,input.store,resources.base,body?.kind==='learning'?body.records:undefined);}catch(error){readiness_error=String(error);waitingMaterial=readiness_error.includes('等待足够的已核验新知识及独立检验资料');}
  let execution_ready=false;
  try{
   assert(!chosen.trialReason,chosen.trialReason??'试用来源不能用于正式训练');
   assert(chosen.pointer,'当前专家学习需要已采纳且完成保护验收的核心');
   assert(cohort||waitingMaterial,readiness_error??'等待已核验专家学习材料');
   const release=json(join(resources.release.path,'..'),resources.release.path.split('/').at(-1)!);
   for(const choice of nativeExpertChoices(chosen.expert?.width??128,!!chosen.expert))readNativeExpertGraphQualification({identity:input.identity,
    core_tensor_sha256:chosen.core?(chosen.core.candidate_tensor_sha256??chosen.core.tensor_sha256):resources.base.original_down_sha256,graph_binary_sha256:resources.role('persistentExpert').sha256,
    source_build_sha256:release.programs.persistentExpert.proof.source_build_sha256,hidden:2048,width:choice.width});
   await nativeEvolutionQ4Resources();execution_ready=true;
  }catch(error){readiness_error=String(error);}
  const resume_supported=!!(saved&&saved.kind==='learning'&&['paused','queued','running'].includes(saved.state)&&body?.checkpoint&&
   body.arguments?.protocol===NATIVE_EXPERT_PROTOCOL&&body.binding?.core_version===(chosen.pointer?.version??null)&&equal(body.binding.implementation_sha256,nativeCompiledTrainingImplementation()));
  return{records:cohort?.records??null,core:chosen.pointer?.version??null,expert:chosen.expert?.id??null,
   steps:nativeExpertStepBudget(chosen.expert?.width??128,!!chosen.expert),material_target:16,material_kind:'expert' as const,
   unspent_count:cohort?.records.train.length??0,material_ready:!!cohort,arguments:{steps:64,protocol:NATIVE_EXPERT_PROTOCOL},execution_ready,readiness_error,resume_supported};
 }finally{owned?.close();}
}
export async function openNativeExpertRuntime(input:NativeExpertRuntimeInput):Promise<NativeEvolutionSession>{
 for(const path of [input.home,input.model,input.store,input.root,input.work])assert(resolve(path)===path&&!path.includes('\0'));
 assert(input.material==null,'定向核心材料不能静默变成专家训练资料');
 const resources=nativeEvolutionRuntimeResources();assert.equal(input.identity,resources.base.identity);
 const chosen=selection(input.store,input.identity),mode=input.mode??'consolidation';assert(['learning','consolidation'].includes(mode));
 assert(!chosen.trialReason,chosen.trialReason??'试用来源不能用于正式训练');
 assert(chosen.pointer,'当前专家学习需要已采纳且完成保护验收的核心');
 const q4=await nativeEvolutionQ4Resources(),settings=new Settings(input.home);
 const {role,release,cache,base}=resources;
 await mkdir(input.work,{recursive:true,mode:0o700});
 const engine=new NativeEngine(role('bridge').path,role('bridge').sha256,{binary:role('worker').path,model:input.model,manifest_sha256:base.model_manifest_sha256,identity:input.identity,log:join(input.work,'expert-bridge.stderr'),
  aten:{base,none_build:cache.NONE,core_build:cache.BUILD,fd_helper:cache.FD},expert_build:{...release,binary_sha256:role('persistentExpert').sha256},
  routing:{store:input.store,core_parameter:base.core_parameter,core_shape:base.core_shape},...(chosen.pointer?{core:{file:join(input.store,chosen.pointer.version+'.safetensors'),tensor_key:base.core_parameter,sha256:chosen.pointer.sha256}}:{})},300000);
 const memory=new Memory(input.root);memory.model_identity=input.identity;
 const chat=new NativeChat(engine,memory,settings);let progress:((value:{step:number;phase:string})=>void)|undefined,lastStep=0,closed=false,running=false,used=false;
 let automatic:{signature:string;scheduled_occurrence:string|null}|undefined,activeSignal:AbortSignal|undefined;
 const learning=new NativeLearning(()=>chat,{onProgress:value=>{lastStep=value.step;progress?.(value);},get automatic(){return automatic;},q4:{...q4,model:input.model,manifestSHA:base.model_manifest_sha256,identity:input.identity,work:input.work,get signal(){return activeSignal;}}});
 return{
  stepBudget:nativeExpertStepBudget(chosen.expert?.width??128,!!chosen.expert),kind:mode==='learning'?'learning':'evolution',earlyStop:true,
  async runRound(options){
   assert(!closed&&!running&&!used,'专家学习会话已经执行或关闭；后继轮必须重新绑定当前核心和专家');options.signal.throwIfAborted();used=true;running=true;progress=options.onProgress;automatic=options.automatic?structuredClone(options.automatic):undefined;activeSignal=options.signal;
   const pause=()=>learning.pause();options.signal.addEventListener('abort',pause,{once:true});
   try{
    if(options.resume){const row=pending(memory,input.identity);assert(row?.id===options.resume&&row.kind==='learning','指定专家断点不存在');
     learning.resume();
    }else learning.start({steps:64});
    const result=await learning.wait();if(result.state==='blocked'||result.state==='error')throw Error(String(result.error??'原生专家会话失败'));
    const report=result.result,adopted=result.state==='complete'&&report?.promoted===true;
    if(adopted)lastStep=nativeExpertAccumulatedStep(report.candidates,0);
    let more=false;if(adopted){try{selectLearningCohort(memory,input.store,base);more=true;}catch(error){if(!String(error).includes('等待足够'))throw error;}}
    return{step:lastStep,phase:adopted?'committed':result.state==='paused'?'paused':result.state,quality_checked:adopted,adopted,pending:more,report:report??null};
   }finally{options.signal.removeEventListener('abort',pause);progress=undefined;automatic=undefined;activeSignal=undefined;running=false;}
  },
  async close(){if(closed)return;await learning.close();assert(!running,'专家训练仍在运行');await chat.close();await engine.close();await closeNativeEvolutionRelease(engine);memory.close();closed=true;}
 };
}
