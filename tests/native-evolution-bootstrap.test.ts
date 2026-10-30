/** Actual compiled bootstrap/fresh/seed/safety consumers, with native generation and persistence boundaries replaced by CPU observations. */
import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,open,rm} from 'node:fs/promises';
import {fstatSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {heldFixture} from './native-evolution-held-source.test.ts';
import {checkpointApprovedBinding,validateCheckpointTree} from '../.native-migration/research/native-product-runtime/backend/native-checkpoint.ts';
import {coreContentSHA,coreCheckpointField,coreCheckpointMetadata,coreNewExposures} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {validateNativeProductRoundArguments} from '../.native-migration/research/native-product-runtime/backend/native-product-round.ts';
import {nativePartialUniqueRecords} from '../.native-migration/research/native-core-controller-candidate/progress.ts';
import {encoded,equal,hash} from '../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';
import {floatValue} from '../.native-migration/research/native-product-runtime/backend/wire.ts';
const code=await readFile(new URL('../build/native-evolution/worker.js',import.meta.url),'utf8');
const normalize=(s:string)=>s.replace(/\b(need|equal|encoded|hash|assert|Memory|join|fstatSync|equal|hash|encoded|admitReceipt|approvedBinding|validateNativeProductRoundArguments|typed|dict|integer|floatValue|assertNativeOrdinaryCohort|nativePartialUniqueRecords|contentSHA|field|replace|seeded|issued)\d+\b/g,'$1').replace(/\bneed\d+\b/g,'need2');
function fn(name:string){const start=code.search(new RegExp('(?:async )?function '+name+'\\('));assert(start>=0,name);const end=code.indexOf('\n}',start);assert(end>start,name);return normalize(code.slice(start,end+2));}
const load=(source:string,scope:Record<string,unknown>,name:string)=>new Function(...Object.keys(scope),source+'\nreturn '+name+';')(...Object.values(scope));
const record=(id:string)=>({id,prompt:id,answer:'ok',provenance:{generation_check:{type:'exact',expected:'ok',max_new_tokens:16}}});
const proof=(r:any,passed:boolean)=>({id:r.id,answer:passed?'ok':'wrong',passed,complete:true,branch_enabled:false,reference_used:false});
async function runBootstrap(failProtection=false){
 const h=await heldFixture({core:true,expert:true,router:true}),directory=await mkdtemp(join(tmpdir(),'native-bootstrap-'));
 const handles:any[]=[],cleanup:(()=>Promise<void>)[]=[];let saved:any,seedCount=0,installed:any,baselineCalls=0,bound=false;
 try{
  const identity='a'.repeat(64),id='123456789abc',original='b'.repeat(64),base={identity,original_down_sha256:original,core_parameter:'model.language_model.layers.23.mlp.base.down_proj.weight',core_shape:[2,2]};
  h.peer.base=base;h.peer.worker.isBranchEnabled=false;h.peer.worker.assertAcceptedPreviousSource=()=>{};h.peer.worker.seedFresh=async()=>{seedCount++;};
  (h.engine as any).config={identity,aten:{base},routing:{store:h.directory}};
  const guard=Object.assign(h.guard,{assertQuiescent(){}}),held=await h.read(h.engine,guard);
  const groups={new_training:[record('new')],new_validation:[record('new-valid')],replay_training:[record('known'),record('unknown')],replay_validation:[record('old-valid')]};
  const records={groups,protection:[record('protected')],retention_regressions:[]};
  const cohort={records,records_sha256:coreContentSHA(records),excluded_validation:[],exclusion_sha256:coreContentSHA([]),assertUnchanged(){}};
  const raw=async(name:string,bytes:number,dtype='u8',shape=[bytes])=>{const path=join(directory,name),handle=await open(path,'wx+',0o600);await handle.truncate(bytes);handles.push(handle);return{path,handle,bytes,dtype,shape,sha256:original,file:name,kind:'tensor'};};
  const rng=async(tag:string)=>({pid:73,device:'mps',cpu_rng:await raw(tag+'-cpu',5056),mps_rng:await raw(tag+'-mps',44),active_files_sha256:h.peer.hashes,assertUnchanged:()=>held.assertUnchanged()});
  const ambient={rng:await rng('ambient')},preparation:any={id,base,records,cohort,graphBinding:{accepted_tensor_sha256:original,accepted_core_version:null},assertUnchanged:()=>held.assertUnchanged()};
  const source={fresh:true,store:h.directory,accepted:{sha256:original,file_sha256:'c'.repeat(64)},basis:{sha256:'d'.repeat(64)},ambient};
  const implementation={native:'e'.repeat(64)},argumentsValue={steps:450,learning_rate:floatValue(1e-5),distillation_weight:5,preserve_replay_subspace:true,partial_policy:'closed-book-growth-450-v1',product_round:{policy:'closed-book-growth-450-v1',absolute_start_step:0,parent:null,learning_reference_sha256:null,current_accepted:{identity,version:null,active_files_sha256:held.pins},groups_sha256:coreContentSHA(groups),records_sha256:coreContentSHA(records),source_sha256:coreContentSHA(implementation),prior_mastered_ids:['protected']}};
  const binding={protocol:'durable-training-v1',identity,scope:'product',core_version:null,expert_id:h.peer.previous.id,device:'mps',dtype:'torch.bfloat16',torch_version:'CPU-control',arguments:argumentsValue,packages:{transformers:'not-loaded',safetensors:'native'},active_files_sha256:held.pins,implementation_sha256:implementation,records_sha256:coreContentSHA(records)};
  class Memory{};const memory=new Memory();
  const field=coreCheckpointField,typed=(b:any,v:any)=>field(checkpointApprovedBinding({...b,arguments:v}),'arguments'),dict=(rows:any[])=>({kind:'dict',entries:rows.map(([key,value])=>[{kind:'string',value:key},value])});
  const scope:any={NATIVE_PRODUCT_ROUND_POLICY2:'closed-book-growth-450-v1',assert,Memory,join,fstatSync,issued:new WeakMap(),seeded:new WeakSet(),encoded,equal,hash,contentSHA:coreContentSHA,field,plain:coreCheckpointMetadata,approvedBinding:checkpointApprovedBinding,typed,dict,floatValue,exposures:coreNewExposures,validateNativeProductRoundArguments,nativePartialUniqueRecords,
   assertNativeExpertScopeGuard(){},validateNativeEvolutionRNG(_e:any,_g:any,value:any){assert.equal(value.active_files_sha256,h.peer.hashes);},borrowNativeEvolutionPreparation:()=>source,borrowNativeEnginePeer:()=>h.peer,readNativeEvolutionHeldSource:h.read,
   nativeCompiledTrainingImplementation:()=>implementation,nativeEvolutionTrainingSource:()=>({core_version:null,expert_id:h.peer.previous.id}),selectIndependentNativeEvolutionCohort:()=>cohort,
   collectLearningModelFiles(){throw Error('bootstrap fresh must not rescan source');},nativeAcceptedHashes(){throw Error('bootstrap fresh must not rehash source');},
   async prepareNativeCoreInitialState(){assert(bound,'Fresh initialization ran before asynchronous journal binding');const state:any={};for(const [name,dtype,shape]of [['candidate','bf16',[2,2]],['master','f32',[2,2]],['reference','f32',[2,2]],['exp_avg','f32',[2,2]],['exp_avg_sq','f32',[2,2]],['adam_step','f32',[]],['basis','f32',[2,1]]]as const){const bytes=shape.reduce((a,b)=>a*b,1)*(dtype==='bf16'?2:4);state[name]=await raw('init-'+handles.length+'-'+name,bytes,dtype,[...shape]);}return{state,assertUnchanged:()=>held.assertUnchanged(),async close(){for(const r of Object.values(state)as any[])if(r.handle.fd!==-1)await r.handle.close();}};},
   requireCore450PeerOperations:(c:any)=>c,assertCoreInstalled(_e:any,cap:any){assert.equal(installed,cap);},actualCoreInstallation:()=>({receipt:{pid:73,state_sha256:'f'.repeat(64)}}),captureNativeEvolutionTrainingRNG:()=>rng('seeded'),
   async generateCore450Rows(_c:any,rows:any[]){baselineCalls++;return rows.map(r=>proof(r,r.id==='known'||r.id==='protected'&&!failProtection));},meanCore450Loss:async()=>1,
   admitReceipt:async()=>({binary:'/writer',binary_sha256:'f'.repeat(64),root:'/release',manifest:{programs:{trainer:{path:'trainer'},reader:{path:'reader'}},payload:{trainer:{final:{sha256:'f'.repeat(64)}},reader:{final:{sha256:'f'.repeat(64)}}}},async close(){}}),
   startNativeTrainingSessionV4:async(request:any)=>{assert.deepEqual(request.records,records);assert.deepEqual(request.arguments,argumentsValue);return{binding,async assertUnchanged(){await held.assertUnchanged();},async close(){}};},async bindCore450Previous(){await Promise.resolve();bound=true;},
   async saveNativeTrainingCheckpoint(request:any){await request.snapshot.assertUnchanged();validateCheckpointTree(request.snapshot.tree);saved=request;return{descriptor:{name:'0'.repeat(32)+'.pt',sha256:'f'.repeat(64),progress:request.progress},activated:false,rng_applied:false,quality_checked:false,async close(){}};}
  };
  scope.validateCore450Fresh=load(fn('validateCore450Fresh'),scope,'validateCore450Fresh');
  scope.prepareCore450ProductFresh=load(fn('prepareCore450ProductFresh'),scope,'prepareCore450ProductFresh');
  scope.seedNativeEvolutionFresh=load(fn('seedNativeEvolutionFresh'),scope,'seedNativeEvolutionFresh');
  scope.core450ControlMetadata=load(fn('core450ControlMetadata'),scope,'core450ControlMetadata');
  // Execute the final compiled safety consumer, including actual generation scoring and original resume validation.
  const safetyModule=await import('../.native-migration/research/native-core450-runner/training-safety.ts');
  const progress=await import('../.native-migration/research/native-core-controller-candidate/progress.ts');
  const resume=await import('../.native-migration/research/native-product-runtime/backend/native-core-resume.ts');
  const segments=await import('../.native-migration/research/native-core-controller-candidate/segments.ts');
  Object.assign(scope,{...segments,nativePartialCorrectIDs:progress.nativePartialCorrectIDs,eq:resume.pythonMetadataEqual,ordinary:resume.validateOrdinaryCoreHistory,CORE450_SAFETY_POLICY:safetyModule.CORE450_SAFETY_POLICY,deriveCore450TrainingCheck:safetyModule.deriveCore450TrainingCheck});
  scope.validateCore450TrainingHistory=load(fn('validateCore450TrainingHistory'),scope,'validateCore450TrainingHistory');
  scope.need=(v:any,m:string)=>assert(v,m);
  scope.assertNativeOrdinaryCohort=load(fn('assertNativeOrdinaryCohort'),scope,'assertNativeOrdinaryCohort');
  const start=code.indexOf('var NATIVE_ROUND_CONTROL_FIELDS ='),end=code.indexOf('\n// .native-migration/research/native-core450-owner-controller/state.ts',start);
  scope.completeNativeOrdinaryState=load(normalize(code.slice(start,end)),scope,'completeNativeOrdinaryState');
  const stateStart=code.indexOf('var typed2 =',end),stateEnd=code.indexOf('\n// .native-migration/research/native-core450-runner/segment.ts',stateStart);
  scope.completeCore450SafetyState=load(normalize(code.slice(stateStart,stateEnd)),scope,'completeCore450SafetyState');
  const bootstrap=load(fn('createCore450ProductFreshCheckpoint'),scope,'createCore450ProductFreshCheckpoint');
  const owners={own(_v:any,close:()=>Promise<void>){cleanup.push(close);},async release(v:any){await v.close();}};
  h.peer.worker.pid=73;
  const result=await bootstrap(h.engine,guard,{seedCoreFresh(){},async installCoreCheckpoint(cap:any){installed=cap;return{};}},{memory,store:h.directory,base,preparation,release:{},directory,arguments:argumentsValue,outputParent:directory},owners);
  assert(saved);const payload=field(saved.snapshot.tree,'payload'),report=coreCheckpointMetadata(field(payload,'report'));
  return{result,savedStep:coreCheckpointMetadata(field(payload,'step')),report,seedCount,baselineCalls,files:saved.snapshot.files.map((f:any)=>f.file),sourceAdmissions:h.sourceAdmissions()};
 }finally{for(const close of cleanup.toReversed())await close();for(const handle of handles)if(handle.fd!==-1)await handle.close();await h.close();await rm(directory,{recursive:true,force:true});}
}
test('actual bootstrap reaches a complete CP0 using held physical source and the post-baseline RNG',async()=>{
 const r=await runBootstrap();expect(r.savedStep).toBe(0);expect(r.seedCount).toBe(1);expect(r.baselineCalls).toBe(3);expect(r.report.baseline_correct_replay_training_ids).toEqual(['known']);expect(r.report.native_fresh_seed_boundary.seed).toBe(42);expect(r.files).toHaveLength(9);expect(r.sourceAdmissions).toBe(1);expect(r.result.optimizer_steps).toBe(0);
});
test('actual bootstrap rejects failed original protection before seed and CP0 save',async()=>{await expect(runBootstrap(true)).rejects.toThrow('Original Core protection failed before optimizer/PT0');});
