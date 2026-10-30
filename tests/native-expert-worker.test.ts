import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {nativeExpertChoices,nativeExpertStepBudget,nativeExpertAccumulatedStep,NATIVE_EXPERT_PROTOCOL} from '../backend/native-expert-learning.ts';
const identity='1'.repeat(64),original='2'.repeat(64),candidate='3'.repeat(64),newCore='accepted-core';
async function fixture(){
 const metadata:Record<string,any>={'active-core.json':{identity,version:newCore,sha256:'file'},[newCore+'.json']:{identity,version:newCore,sha256:'file',candidate_tensor_sha256:candidate,report:'report.json'},'report.json':{},
  'active.json':{identity,id:'learned-000000000001',width:512,core_version:'old-core'},'layer-23/learned-000000000001.json':{core_version:'old-core'},manifest:{programs:{persistentExpert:{proof:{source_build_sha256:'source'}}}}};
 let qualify=true,cohortError:string|null=null,closed=0;const qualifications:any[]=[],controllers:any[]=[];
 const cohort={records:{train:Array.from({length:16},(_,i)=>({id:i})),valid:[{id:16},{id:17}],protection:[]}};
 class Memory{model_identity='';connect(f:any){return f({query:()=>({get:()=>null})});}close(){closed++;}}
 class Engine{constructor(..._:any[]){}async close(){}}
 class Chat{constructor(..._:any[]){}async close(){}}
 class Learning{constructor(_chat:any,readonly options:any){controllers.push(this);}calls:any[]=[];start(value:any){this.calls.push({value,automatic:structuredClone(this.options.automatic),signal:this.options.q4.signal});}resume(){}pause(){}async wait(){this.options.onProgress({step:8,phase:'validated'});return{state:'complete',result:{promoted:true,candidates:[{actual_steps:8}]}};}async close(){}}
 const bindings={assert,join,resolve,mkdir:async()=>{},Memory,Settings:class{},NativeEngine:Engine,NativeChat:Chat,NativeLearning:Learning,
  ledgerParse:JSON.parse,equal:(a:any,b:any)=>JSON.stringify(a)===JSON.stringify(b),readSafe:(_store:string,name:string)=>{if(!(name in metadata))throw Object.assign(Error('missing'),{code:'ENOENT'});return Buffer.from(JSON.stringify(metadata[name]));},
  nativeTrialSourceReason:()=>metadata['active.json']?.execution_scope==='local-trial'?'当前MLX专家组合仅获本地试用许可；恢复已正式采纳来源后可继续学习和无限进化':null,
  nativeCoreProtectionAdmission:()=>({qualified:true}),nativeEvolutionRuntimeResources:()=>({base:{identity,original_down_sha256:original,model_manifest_sha256:'manifest',core_parameter:'down',core_shape:[2048,6144]},release:{path:'/r/manifest'},cache:{},role:()=>({path:'/native',sha256:'binary'})}),nativeEvolutionQ4Resources:async()=>({executable:'/q4',binary:{}}),nativeCompiledTrainingImplementation:()=>({impl:'current'}),
  nativeEvolutionPreviewMemory:async(value:any)=>value,closeNativeEvolutionRelease:async()=>{},selectLearningCohort:()=>{if(cohortError)throw Error(cohortError);return cohort;},readNativeExpertGraphQualification:(binding:any)=>{qualifications.push(binding);assert(qualify,'unqualified');return{};},nativeExpertChoices,nativeExpertStepBudget,nativeExpertAccumulatedStep,NATIVE_EXPERT_PROTOCOL};
 const raw=(await readFile(new URL('../backend/native-expert-worker.ts',import.meta.url),'utf8')).replace(/^import .*;\n/gm,'').replaceAll('export ','');
 const code=new Bun.Transpiler({loader:'ts',target:'bun'}).transformSync(raw);
 const api=new Function(...Object.keys(bindings),code+'\nreturn{inspectNativeExpertRuntime,openNativeExpertRuntime};')(...Object.values(bindings));
 return{...api,metadata,qualifications,controllers,setError:(error:string|null)=>{cohortError=error;},setQualified:(value:boolean)=>{qualify=value;},closed:()=>closed};
}
const scope={root:'/root',store:'/store',identity},input={...scope,home:'/home',model:'/model',work:'/work',mode:'consolidation'};
test('actual inspector binds imported candidate tensor and excludes incompatible physical parent',async()=>{
 const f=await fixture(),result=await f.inspectNativeExpertRuntime(scope);
 expect(result.expert).toBeNull();expect(result.steps).toBe(192);expect(result.material_ready).toBe(true);expect(result.arguments.protocol).toBe(NATIVE_EXPERT_PROTOCOL);
 expect(f.qualifications.every((r:any)=>r.core_tensor_sha256===candidate)).toBe(true);
 f.metadata['active.json'].core_version=newCore;f.metadata['layer-23/learned-000000000001.json'].core_version=newCore;
 expect((await f.inspectNativeExpertRuntime(scope)).expert).toBe('learned-000000000001');
 f.metadata[newCore+'.json'].tensor_sha256=original;await expect(f.inspectNativeExpertRuntime(scope)).rejects.toThrow();
});
test('lack of legitimate material waits separately from missing graph qualification or corrupt protection',async()=>{
 const f=await fixture();f.setError('等待足够的已核验新知识及独立检验资料');
 const result=await f.inspectNativeExpertRuntime(scope);expect(result.execution_ready).toBe(true);expect(result.material_ready).toBe(false);expect(result.records).toBeNull();
 f.setQualified(false);expect((await f.inspectNativeExpertRuntime(scope)).execution_ready).toBe(false);
 f.setQualified(true);f.setError('Accepted protection generation rule conflict');expect((await f.inspectNativeExpertRuntime(scope)).execution_ready).toBe(false);
});
test('coreNone and out-of-policy physical expert are rejected before native execution',async()=>{
 const f=await fixture();delete f.metadata['active-core.json'];
 const result=await f.inspectNativeExpertRuntime(scope);expect(result.execution_ready).toBe(false);expect(result.readiness_error).toContain('已采纳');
 await expect(f.openNativeExpertRuntime(input)).rejects.toThrow('已采纳');expect(f.controllers.length).toBe(0);
 f.metadata['active-core.json']={identity,version:newCore,sha256:'file'};f.metadata['active.json'].width=1024;
 await expect(f.inspectNativeExpertRuntime(scope)).rejects.toThrow();
});
test('actual wrapper forwards per-round automatic binding/signal and forces new source-bound session for a successor',async()=>{
 const f=await fixture(),session=await f.openNativeExpertRuntime(input),signal=new AbortController().signal,automatic={signature:'a'.repeat(64),scheduled_occurrence:null};
 const result=await session.runRound({signal,round:1,automatic,onProgress:()=>{}});
 expect(f.controllers[0].calls[0].automatic).toEqual(automatic);expect(f.controllers[0].calls[0].signal).toBe(signal);expect(result.step).toBe(8);expect(result.pending).toBe(true);
 await expect(session.runRound({signal,round:2,onProgress:()=>{}})).rejects.toThrow('后继轮');await session.close();expect(f.closed()).toBe(1);
 f.metadata['active.json'].core_version=newCore;f.metadata['layer-23/learned-000000000001.json'].core_version=newCore;
 const next=await f.openNativeExpertRuntime(input);expect(next.stepBudget).toBe(nativeExpertStepBudget(512,true));await next.close();
});

test('trial source blocks preview and open before a native training controller exists',async()=>{
 const f=await fixture();f.metadata['active.json'].execution_scope='local-trial';
 const result=await f.inspectNativeExpertRuntime(scope);
 expect(result.execution_ready).toBe(false);expect(result.material_ready).toBe(false);
 expect(result.readiness_error).toContain('本地试用');
 await expect(f.openNativeExpertRuntime(input)).rejects.toThrow('本地试用');expect(f.controllers.length).toBe(0);
});
