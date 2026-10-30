/** MLX specialist learning under the existing application lease and continuous-round service. */
import assert from 'node:assert/strict';
import {join,resolve} from 'node:path';
import {mkdirSync,readFileSync,existsSync,lstatSync} from 'node:fs';
import {Memory,ledgerParse} from './memory.ts';
import {Settings,privateJSON,requireStorage} from './settings.ts';
import {encoded} from './canonical-json.ts';
import {nativeEvolutionPreviewMemory,type NativeEvolutionPreviewMemory} from './native-evolution-preview.ts';
import {nativeExpertChoices,nativeExpertStepBudget,freezeNativeExpertPreservation} from './native-expert-learning.ts';
import {materialProvenance,materialMessages} from './native-evolution-material.ts';
import {MLX_LEARNING_POLICY,MLX_TRAINING_ADMISSION,mlxMastered,mlxNextKnowledge} from './mlx-learning-policy.ts';
import {MODEL_BASE} from '../.native-migration/research/native-unified-service/compiled-model-v6.ts';
import {MLXMaterialShortage,MLX_LEARNING_PROTOCOL,MLX_FIRST_LEARNING_NOTICE,isUntrainedMLXInstallation,readMLXLearningSource,selectMLXLearningRecords,mlxRead,mlxHash,mlxLearningRecordKey,type MLXSource,type MLXRow,type MLXRecords} from './mlx-learning-source.ts';
import {MLXLearningClient,lockMLXModel,type PreparedMLX} from './mlx-learning-client.ts';
import {observeMLXPair,observeMLXLoss,decideMLXQuality,uniqueMLXRecords,mlxCandidateAction} from './mlx-learning-quality.ts';
import {MLXLearningJournal,mlxPending,mlxRuns,recoverMLXCommitPointers,inspectMLXCommitRecovery} from './mlx-learning-storage.ts';
import {nativeAutomaticSignature} from './native-evolution-automatic.ts';
import type {NativeEvolutionSession} from './native-evolution-service.ts';

type Scope={home:string;model:string;root:string;store:string;identity:string;binary:string};
type RuntimeInput=Scope&{work:string;mode?:'learning'|'consolidation';material?:unknown;memory?:Memory;signal?:AbortSignal;routes:(records:MLXRow[])=>Promise<MLXRow>};
function source(input:Scope,memory:Pick<Memory,'connect'>){const settings=new Settings(input.home).snapshot();assert(settings.native_inference_precision==='mlx-q4','MLX学习需要当前MLX推理设置');assert(input.identity===MODEL_BASE.identity);requireStorage(input.root);
 if(isUntrainedMLXInstallation(input.store,()=>mlxRuns(memory,input.identity).length>0||memory.connect(db=>!!db.query("SELECT 1 FROM events WHERE kind IN ('learning','evolution','router','training_restore','factory:learning','factory:evolution','factory:router') LIMIT 1").get())))throw Error(MLX_FIRST_LEARNING_NOTICE);
 return readMLXLearningSource({...input,modelPath:String(settings.native_mlx_model_path)});}
function readinessMessage(error:unknown){const message=String(error instanceof Error?error.message:error);return message.includes('MLX source proof differs')||message.includes('MLX worker build receipt missing')?'专家学习来源已变化，正在重新核对资料。':message;}
function cohort(memory:Pick<Memory,'connect'|'samples'>,s:MLXSource,pending:MLXRow|null){const body=pending?ledgerParse(pending.body):null;if(body)assert(body.arguments?.protocol===MLX_LEARNING_PROTOCOL&&body.arguments?.learning_policy===MLX_LEARNING_POLICY,'已有其他引擎的学习断点，不能换成MLX恢复');return selectMLXLearningRecords(memory.samples(),s,mlxRuns(memory,s.identity),body?.records);}
export async function inspectMLXLearningRuntime(input:Scope&{memory?:NativeEvolutionPreviewMemory}){
 const owned=input.memory?null:new Memory(input.root);if(owned)owned.model_identity=input.identity;const memory=input.memory?await nativeEvolutionPreviewMemory(input.memory,input.root,input.identity):owned!;
 let unspent_count=0;
 let records:MLXRecords|null=null,execution_ready=false,readiness_error:string|null=null,s:MLXSource|undefined,pending:MLXRow|null=null;
 try{const recovery=inspectMLXCommitRecovery(memory,input);if(recovery)return{records:recovery.body.records,core:recovery.body.binding.core_version,expert:recovery.body.binding.expert_id,inference_engine:'mlx-q4',steps:256,material_target:16,material_kind:'expert',unspent_count:0,material_ready:true,arguments:recovery.body.arguments,execution_ready:true,readiness_error:null,resume_supported:true};s=source(input,memory);pending=mlxPending(memory,input.identity);execution_ready=true;try{records=cohort(memory,s,pending);unspent_count=records.train.length;}catch(e){readiness_error=String(e instanceof Error?e.message:e);if(e instanceof MLXMaterialShortage)unspent_count=e.training;else execution_ready=false;}}
 catch(e){readiness_error=readinessMessage(e);execution_ready=false;}
 finally{owned?.close();}
 return{records,core:s?.core.version??null,expert:s?.active.id??null,inference_engine:'mlx-q4',steps:nativeExpertStepBudget(s?.expert.width??64,true),material_target:16,material_kind:'expert',unspent_count,material_ready:!!records,arguments:{steps:64,protocol:MLX_LEARNING_PROTOCOL,learning_policy:MLX_LEARNING_POLICY,training_admission:MLX_TRAINING_ADMISSION},execution_ready,readiness_error,resume_supported:!!(pending&&ledgerParse(pending.body).arguments?.protocol===MLX_LEARNING_PROTOCOL)};
}
function routesChecked(value:MLXRow,s:MLXSource,records:MLXRow[]){assert(value.released===true&&value.identity===s.identity&&value.core_version===s.core.version&&value.expert_id===s.active.id&&value.router_id===s.routerPointer.id&&value.router_sha256===s.router.sha256&&Array.isArray(value.rows)&&value.rows.length===records.length,'MLX真实路由来源不完整');const routes=new Map<string,string>();for(const row of value.rows){assert(records.some(r=>r.id===row.id)&&!routes.has(row.id)&&['local_core','neural_expert'].includes(row.action)&&Array.isArray(row.input_tokens),'MLX路由观察不匹配');routes.set(row.id,row.action);}return routes;}

export async function openMLXLearningRuntime(input:RuntimeInput):Promise<NativeEvolutionSession>{
 assert(input.material==null,'定向核心材料不能自动改成MLX专家材料');const owned=input.memory?null:new Memory(input.root),memory=input.memory??owned!;assert.equal(memory.root,resolve(input.root),'MLX学习数据库目录不一致');
 const data=<T>(operation:()=>T,signal?:AbortSignal)=>memory.runWhenIdle(input.identity,operation,signal);
 let s:MLXSource;try{s=await data(()=>{const held=lockMLXModel(input.identity);try{recoverMLXCommitPointers(memory,input);}finally{held();}return source(input,memory);},input.signal);}catch(e){owned?.close();throw e;}
 let client:MLXLearningClient|undefined,unlock:(()=>void)|undefined,closed=false,running=false,used=false;
 const choices=nativeExpertChoices(s.expert.width,true),budget=nativeExpertStepBudget(s.expert.width,true);
 const closeWorker=async()=>{if(client){await client.close();client=undefined;}if(unlock){unlock();unlock=undefined;}};
 return{kind:input.mode==='learning'?'learning':'evolution',stepBudget:budget,earlyStop:true,
  async runRound(options){assert(!used&&!closed&&!running);used=true;running=true;let journal:MLXLearningJournal|undefined,totalStep=0;const started=performance.now();
   try{
    const {pending,records}=await data(()=>{const pending=mlxPending(memory,s.identity);assert(!pending||options.resume===pending.id,'已有MLX断点，请恢复原学习轮次');return{pending,records:cohort(memory,s,pending)};},options.signal),all=uniqueMLXRecords(records);
    // The Python shell performs only the existing four-layer router, releases it, and never trains here.
    const routeEvidence=await input.routes(all),routes=routesChecked(routeEvidence,s,all);options.signal.throwIfAborted();
    s.assertUnchanged();assert.equal(mlxHash(readFileSync(s.binary)),s.bound.worker_binary_sha256,'MLX executable differs from installed build receipt');
    unlock=lockMLXModel(s.identity);client=new MLXLearningClient(s,input.model,MODEL_BASE.model_manifest_sha256);await client.start();
    const signature=nativeAutomaticSignature(s.identity,s.core.version,records,{records,core:s.core.version,expert:s.active.id,material_kind:'expert',execution_ready:true,arguments:{steps:64,protocol:MLX_LEARNING_PROTOCOL,learning_policy:MLX_LEARNING_POLICY,training_admission:MLX_TRAINING_ADMISSION}});
    if(options.automatic)assert.equal(options.automatic.signature,signature,'MLX automatic material signature changed');
    journal=await data(()=>new MLXLearningJournal(memory,s,records,pending??undefined,{signature,scheduled_occurrence:options.automatic?.scheduled_occurrence??null}),options.signal);if(journal.state==='committing'){
     await data(()=>journal!.rollback('应用上次在MLX采纳重载前中断'));await closeWorker();unlock=lockMLXModel(s.identity);client=new MLXLearningClient(s,input.model,MODEL_BASE.model_manifest_sha256);await client.start();await data(()=>journal!.update('rejected',{owner:null}));return{step:0,phase:'rejected_recovered',quality_checked:false,adopted:false,pending:false};
    }
    await data(()=>journal!.update('running',{owner:{pid:process.pid}}));options.onProgress({step:0,phase:'mlx_baseline'});
    const prepared=new Map<string,PreparedMLX>();for(const row of all){const routed=routeEvidence.rows.find((v:MLXRow)=>v.id===row.id);const prompt=await client.command('prepare',{messages:materialMessages(row)});assert.equal(encoded(prompt.tokens),encoded(routed.input_tokens),'Native MLX tokenizer differs from actual HF router input');if(row.answer.trim())prepared.set(row.id,await client.prepare(row));}
    let baseline=journal.body.baseline as MLXRow|null;
    if(!baseline){const basePair=await observeMLXPair(client,all,'current',routes,options.signal),parent=[[],[]];
     const unreachable=records.valid.filter(row=>routes.get(row.id)!=='neural_expert'&&!basePair.every(pass=>pass.some(v=>v.id===row.id&&v.passed&&v.complete)));if(unreachable.length){const report={promoted:false,quality_checked:true,reason:'新知识尚未被当前专家路由覆盖，等待路由校准；未执行无效训练',failed_ids:unreachable.map(row=>row.id),records,baseline:basePair,route_evidence:routeEvidence};await data(()=>journal!.update('rejected',{owner:null,result:report}));return{step:0,phase:'rejected_routes',quality_checked:true,adopted:false,pending:false,report};}
     const correct=mlxMastered(basePair),oldRows=records.protection.filter(row=>correct.has(row.id));
     const preservation=oldRows.flatMap(row=>{const observation=basePair[0]!.find(v=>v.id===row.id)!;const material={...row,provenance:encoded({...materialProvenance(row),retention_scope:observation.branch_enabled?'accepted_branch':'original_core'})};return freezeNativeExpertPreservation([material],[observation]);});
     for(const row of preservation)prepared.set('preservation:'+row.id,await client.prepare(row));
     const preservationPrepared=new Map(preservation.map(row=>[row.id,prepared.get('preservation:'+row.id)!]));const validation=await observeMLXLoss(client,records.valid,'parent',prepared,options.signal,routes),replay=await observeMLXLoss(client,[...preservation],'protection',preservationPrepared,options.signal);
     baseline={core:basePair,parent,preservation,validation,replay,route_evidence:routeEvidence};await data(()=>journal!.update('running',{baseline}));
    }else{assert(encoded(baseline.route_evidence.rows)===encoded(routeEvidence.rows),'恢复时路由行为已变化');for(const row of baseline.preservation)prepared.set('preservation:'+row.id,await client.prepare(row));}
    let prior=structuredClone(journal.body.candidates) as MLXRow[];totalStep=prior.reduce((n,c)=>n+c.actual_steps,0);
    const saved=journal.body.checkpoint as MLXRow|null;const startIndex=saved?.candidate_index??prior.length;
    for(let index=startIndex;index<choices.length;index++){
     options.signal.throwIfAborted();const choice=choices[index]!,binding=mlxHash(encoded({...journal.body.binding,candidate_index:index,width:choice.width}));
     const began=await client.command('train_begin',{seed:42,width:choice.width,learning_rate:choice.learning_rate,preservation_weight:5,initialize:choice.reuse?'parent':'fresh',binding});assert(began.event==='train_started'&&began.step===0&&began.width===choice.width);
     let step=0;if(saved&&saved.candidate_index===index){const restored=await client.command('train_restore',{checkpoint_path:saved.path,binding});assert(restored.step===saved.step&&restored.width===choice.width);step=saved.step;}
     const directory=join(input.root,'runtime/mlx/learning',journal.id);mkdirSync(directory,{recursive:true,mode:0o700});
     const save=async()=>{const stamp='candidate-'+index+'-step-'+step,path=join(directory,stamp+'.safetensors'),checkpoint=join(directory,stamp+'.checkpoint.safetensors');if(existsSync(checkpoint)&&journal!.body.checkpoint?.path===checkpoint)return journal!.body.checkpoint;const ack=await client!.command('train_save',{path,checkpoint_path:checkpoint});assert(ack.event==='train_saved'&&ack.step===step&&ack.width===choice.width&&ack.path===path&&ack.checkpoint_path===checkpoint);const st=lstatSync(path);assert(st.isFile()&&!st.isSymbolicLink());const value={path:checkpoint,weights_path:path,weights_stat:[st.dev,st.ino,st.size,st.mtimeMs,st.ctimeMs],step,width:choice.width,candidate_index:index,binding};await data(()=>journal!.update('running',{checkpoint:value}));return value;};
     if(!saved||saved.candidate_index!==index)await save();
     let accepted=false,validatedStep=0;
     for(;;){
      const action=mlxCandidateAction(step,validatedStep);if(action==='done')break;
      if(options.signal.aborted){await save();await client.command('train_end',{restore_parent:true});await data(()=>journal!.update('paused',{owner:null}));return{step:totalStep+step,phase:'paused',quality_checked:false,adopted:false,pending:true};}
      if(action==='train'){
      const row=records.train[step%records.train.length]!,a=prepared.get(row.id)!;
      const guard=baseline.preservation.length?baseline.preservation[step%baseline.preservation.length]:null;
      const b=guard?prepared.get('preservation:'+guard.id)!:null;
      // The first round has no learned history: pure CE, never a held-out target as replay.
      const result=await client.command('train_step',{sample:{tokens:a.tokens,labels:a.labels},...(guard&&b?{preservation:{tokens:b.tokens,labels:b.labels},teacher:materialProvenance(guard).preservation_teacher==='expert'?'parent':'core'}:{})});
      assert(result.event==='train_step'&&result.step===step+1&&Number.isFinite(result.loss)&&result.loss>=0,'MLX Adam receipt invalid');step=result.step;options.onProgress({step:totalStep+step,phase:'mlx_training'});
      }
      if(step%8!==0&&!options.signal.aborted)continue;const checkpoint=await save();if(options.signal.aborted){await client.command('train_end',{restore_parent:true});await data(()=>journal!.update('paused',{owner:null}));return{step:totalStep+step,phase:'paused',quality_checked:false,adopted:false,pending:true};}
      options.onProgress({step:totalStep+step,phase:'mlx_validation'});
      const preservationPrepared=new Map<string,PreparedMLX>(baseline.preservation.map((row:MLXRow)=>[row.id,prepared.get('preservation:'+row.id)!]));const validation=await observeMLXLoss(client,records.valid,'candidate',prepared,options.signal,routes),replay=await observeMLXLoss(client,baseline.preservation,'candidate',preservationPrepared,options.signal,routes);
      const pair=await observeMLXPair(client,all,'candidate',routes,options.signal),decision=decideMLXQuality({records,baseline:baseline.core,parent:baseline.parent,candidate:pair,baselineLoss:baseline.validation.mean,candidateLoss:validation.mean,replayBefore:baseline.replay.mean,replayAfter:replay.mean,routes,policy:MLX_LEARNING_POLICY});
      const record={width:choice.width,actual_steps:step,initialization:choice.reuse?'reuse':'fresh',learning_rate:choice.learning_rate,validation_loss:validation.mean,replay_loss:replay.mean,objective:validation.mean+.0005*choice.width,accepted:decision.accepted,gates:decision.gates,generation_checks:pair[0]!.filter(row=>records.valid.some(r=>r.id===row.id)),core_retention_generation:pair[0]!.filter(row=>records.protection.some(r=>r.id===row.id))};
      await data(()=>journal!.update('running',{last_validation:{candidate:index,step,decision,validation,replay,observations:pair}}));
      validatedStep=step;
      if(!decision.accepted&&step<64)continue;
      prior=[...prior,record];
      if(decision.accepted){
       const mastered=mlxMastered(pair),knowledge=mlxNextKnowledge(s.acceptedReport,records,pair);
       const report={protocol:MLX_LEARNING_PROTOCOL,learning_policy:MLX_LEARNING_POLICY,training_admission:MLX_TRAINING_ADMISSION,learning_progress:decision.learning_progress,...knowledge,promoted:true,quality_checked:true,chosen_width:choice.width,candidates:prior,actual_steps:totalStep+step,covered_keys:[...new Set([...(s.acceptedReport?.covered_keys??[]),...[...records.train,...records.valid].filter(row=>mastered.has(row.id)).map(mlxLearningRecordKey)])],gates:decision.gates,observations:{baseline:baseline.core,parent:baseline.parent,candidate:pair},loss:{baseline_validation:baseline.validation,validation,baseline_replay:baseline.replay,replay},route_evidence:routeEvidence,elapsed_seconds:(performance.now()-started)/1000};
       const savedStat=lstatSync(checkpoint.weights_path);assert(savedStat.isFile()&&!savedStat.isSymbolicLink()&&encoded([savedStat.dev,savedStat.ino,savedStat.size,savedStat.mtimeMs,savedStat.ctimeMs])===encoded(checkpoint.weights_stat),'MLX saved candidate changed after actual observation');const publication=await data(()=>journal!.prepare(report,checkpoint.weights_path));await client.command('train_end',{restore_parent:true});await client.close();client=undefined;
       const adopted={...s,active:publication.expertPointer,expert:publication.expertMeta,expertFile:publication.weightsPath,bound:{...s.bound,expert_id:publication.expertPointer.id,expert_file_sha256:publication.expertMeta.sha256}};
       try{journal.apply();client=new MLXLearningClient(adopted,input.model,MODEL_BASE.model_manifest_sha256);const ack=await client.start();const reloadRow=records.valid.find(row=>mastered.has(row.id)&&routes.get(row.id)==='neural_expert')??records.valid.find(row=>mastered.has(row.id))!;const observed=await client.generate(reloadRow,routes.get(reloadRow.id)==='local_core'?'core':'parent');assert(observed.passed&&observed.complete&&JSON.stringify(observed.tokens)===JSON.stringify(pair[0]!.find(r=>r.id===observed.id)!.tokens),'Saved MLX candidate reload generation differs');await data(()=>journal!.finish(ack));}catch(error){await client?.close();client=undefined;await data(()=>journal!.rollback(String(error)));client=new MLXLearningClient(s,input.model,MODEL_BASE.model_manifest_sha256);await client.start();await data(()=>journal!.update('rejected',{owner:null}));throw error;}
       accepted=true;const more=await data(()=>{const nextSource=source(input,memory);try{selectMLXLearningRecords(memory.samples(),nextSource,mlxRuns(memory,s.identity));return true;}catch(e){if(!String(e).includes('等待足够'))throw e;return false;}});
       return{step:totalStep+step,phase:'committed',quality_checked:true,adopted:true,pending:more,report:publication.report};
      }
     }
     if(!accepted){await client.command('train_end',{restore_parent:true});totalStep+=step;await data(()=>journal!.update('running',{checkpoint:null,candidates:prior}));}
    }
    const report={learning_policy:MLX_LEARNING_POLICY,training_admission:MLX_TRAINING_ADMISSION,learning_progress:journal.body.last_validation?.decision?.learning_progress??null,promoted:false,quality_checked:true,actual_steps:totalStep,candidates:journal.body.candidates,reason:'所有MLX专家候选均未通过完整质量、路由和防遗忘门',elapsed_seconds:(performance.now()-started)/1000};await data(()=>journal!.update('rejected',{owner:null,result:report}));return{step:totalStep,phase:'rejected_quality',quality_checked:true,adopted:false,pending:false,report};
   }catch(error){if(journal&&!['committed','rejected','committing'].includes(journal.state))await data(()=>journal!.update('paused',{owner:null,error:String(error)}));throw error;}
   finally{running=false;}
  },
  async close(){if(closed)return;assert(!running,'MLX学习仍在运行');await closeWorker();owned?.close();closed=true;}
 };
}
