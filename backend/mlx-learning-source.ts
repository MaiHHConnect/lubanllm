/** MLX learning selects authorized records without borrowing a trial adapter's qualification. */
import assert from 'node:assert/strict';
import {existsSync,lstatSync,readFileSync,readdirSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {ledgerParse} from './memory.ts';
import {encoded} from './canonical-json.ts';
import {eligibleMaterial,dailyLearningMaterial,materialContext,materialContentSHA,materialProvenance,validateMaterialRecord,validateNativeCoreRetention} from './native-evolution-material.ts';
import {MLX_LEARNING_POLICY,mlxKnowledgeSample} from './mlx-learning-policy.ts';
import {nativeCoreProtectionAdmission} from './native-effective-core.ts';
import {evaluateGeneration,validateGenerationCheck} from './generation-checks.ts';
import {factoryWorkerQualified,factoryTrainingQualification} from './factory-engine-qualification.ts';

export const MLX_LEARNING_PROTOCOL='native-mlx-expert-preservation-v1';
export const MLX_TRAINING_GATES=['prepare_equivalent','real_adam_updates','checkpoint_replay_exact','cancellation_atomic','continued_after_cancel','saved_expert_reload_equal','base_core_frozen']as const;
export function assertMLXTrainingQualification(proof:MLXRow,binding:{worker_binary_sha256:string;core_file_sha256:string;mlx_model_sha256:string}){assert(proof?.schema==='mlx-native-training-qualification-v1'&&proof.complete===true&&Object.entries(binding).every(([key,value])=>proof[key]===value)&&MLX_TRAINING_GATES.every(key=>proof.gates?.[key]===true),'MLX原生训练仍等待当前引擎、核心和模型的真实Adam/保存恢复/取消资格验证');}
export type MLXRow=Record<string,any>;
export type MLXRecords={train:MLXRow[];valid:MLXRow[];protection:MLXRow[]};
export const MLX_FIRST_LEARNING_NOTICE='当前安装缺少初始专家，请安装新版完整模型包或导入已有模型。';
/** Only a truly empty installation gets the first-use explanation; damaged history keeps its original error. */
export function isUntrainedMLXInstallation(store:string,hasHistory:()=>boolean){
 try{
  const root=lstatSync(store);if(!root.isDirectory()||root.isSymbolicLink())return false;
  const entries=readdirSync(store,{recursive:true,withFileTypes:true});
  if(entries.some(entry=>!entry.isDirectory()))return false;
  return !hasHistory();
 }catch(error){if((error as NodeJS.ErrnoException).code==='ENOENT')return !hasHistory();throw error;}
}
export class MLXMaterialShortage extends Error {
 constructor(readonly training:number,readonly validation:number){
  super(`等待足够的已核验学习资料：训练 ${training} / 16 条，独立验证 ${validation} / 2 条。`);
 }
}
export const mlxHash=(value:Buffer|string)=>createHash('sha256').update(value).digest('hex');
export const mlxDigest=(value:unknown):value is string=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
export function mlxRead(path:string){const st=lstatSync(path);assert(st.isFile()&&!st.isSymbolicLink()&&st.size<=16<<20,'MLX metadata is not a bounded regular file');return ledgerParse(readFileSync(path,'utf8'));}
export const mlxLearningRecordKey=(row:MLXRow)=>materialContentSHA({context:materialContext(row),answer:row.answer,check:materialProvenance(row).generation_check??null,route_action:null});
const safeID=(id:any,prefix:string)=>assert(typeof id==='string'&&new RegExp('^'+prefix+'[a-z0-9-]{1,80}$').test(id),'Invalid MLX source identifier');

export function readMLXLearningSource(input:{root:string;store:string;identity:string;modelPath:string;binary:string}){
 for(const path of [input.root,input.store,input.modelPath,input.binary])assert(resolve(path)===path&&!path.includes('\0'));
 assert(mlxDigest(input.identity));
 const core=mlxRead(join(input.store,'active-core.json'));safeID(core.version,'core-');assert(core.identity===input.identity&&mlxDigest(core.sha256));
 const coreMeta=mlxRead(join(input.store,core.version+'.json')),report=mlxRead(join(input.store,coreMeta.report));
 assert(coreMeta.identity===input.identity&&coreMeta.version===core.version&&coreMeta.sha256===core.sha256&&nativeCoreProtectionAdmission(coreMeta,report).qualified,'MLX learning requires the fully protected accepted core');
 const retention=validateNativeCoreRetention(mlxRead(join(input.store,core.version+'.retention.json')),input.identity,core.version);
 const active=mlxRead(join(input.store,'active.json'));safeID(active.id,'');
 const expert=mlxRead(join(input.store,'layer-23',active.id+'.json'));
 assert(active.identity===input.identity&&active.core_version===core.version&&expert.core_version===core.version&&expert.identity===input.identity+':layer-23'&&expert.requires_gate===true&&expert.hidden===2048&&Number.isSafeInteger(expert.width)&&expert.width>=32&&expert.width<=512&&active.width===expert.width&&mlxDigest(expert.sha256),'MLX warm candidate source binding differs');
 const routerPointer=mlxRead(join(input.store,'active-router.json'));safeID(routerPointer.id,'router-');
 const router=mlxRead(join(input.store,routerPointer.id+'.json'));
 assert(router.feature_source==='mlx-q4-prefix','MLX路由表征来源已变化，需要使用当前MLX前四层重新校准');
 assert(routerPointer.identity===input.identity&&router.identity===input.identity&&router.id===routerPointer.id&&router.core_version===core.version&&router.expert_id===active.id&&router.expert_sha256===expert.sha256&&router.sha256===routerPointer.sha256&&router.calibrated===true&&router.gates_expert===true,'MLX source router is unavailable');
 const download=mlxRead(join(input.modelPath,'download-receipt.json')),weight=download.files?.find((v:any)=>v.name==='model.safetensors'),weights=join(input.modelPath,'model.safetensors'),st=lstatSync(weights);
 assert(download.complete===true&&weight?.downloaded===true&&mlxDigest(weight.sha256)&&weight.bytes===st.size&&!st.isSymbolicLink()&&st.mtimeMs<=download.finished*1000+1000,'MLX download receipt is no longer bound');
 const build=mlxRead(input.binary+'.json');assert(mlxDigest(build.binary_sha256),'MLX worker build receipt missing');
 const proof=mlxRead(join(input.root,'runtime/mlx/expert-proof.json'));
 assert(['mlx-expert-local-trial-v1','mlx-expert-learning-v1'].includes(proof.schema)&&proof.identity===input.identity&&proof.core_version===core.version&&proof.core_file_sha256===core.sha256&&proof.expert_id===active.id&&proof.expert_file_sha256===expert.sha256&&proof.router_id===routerPointer.id&&proof.router_sha256===router.sha256&&proof.mlx_model_sha256===weight.sha256&&factoryWorkerQualified(proof,build.binary_sha256),'MLX source proof differs');
 const trainingProofPath=join(input.root,'runtime/mlx/training-proof.json');assert(existsSync(trainingProofPath),'MLX原生训练仍等待真实Adam、保存恢复和取消资格验证');assertMLXTrainingQualification(factoryTrainingQualification(mlxRead(trainingProofPath),build.binary_sha256,core.sha256,weight.sha256),{worker_binary_sha256:build.binary_sha256,core_file_sha256:core.sha256,mlx_model_sha256:weight.sha256});
 let acceptedReport:MLXRow|null=null;if(proof.schema==='mlx-expert-learning-v1'){assert(typeof proof.report==='string'&&/^learning-[a-f0-9]{12}\.json$/.test(proof.report)&&mlxDigest(proof.report_sha256));const raw=readFileSync(join(input.store,proof.report));assert(mlxHash(raw)===proof.report_sha256,'MLX accepted report binding differs');acceptedReport=ledgerParse(raw.toString());assert(acceptedReport!.active_expert===active.id&&acceptedReport!.promoted===true&&acceptedReport!.quality_checked===true);}
 // A newly built worker must be independently qualified before learning; never rewrite a trial proof to match it.
 const paths=['active-core.json','active.json','active-router.json'];const pointers=Object.fromEntries(paths.map(name=>[name,mlxRead(join(input.store,name))]));
 const bound={identity:input.identity,core_version:core.version,core_file_sha256:core.sha256,expert_id:active.id,expert_file_sha256:expert.sha256,router_id:routerPointer.id,router_sha256:router.sha256,mlx_model_sha256:weight.sha256,worker_binary_sha256:build.binary_sha256};
 const assertUnchanged=()=>{for(const name of paths)assert.equal(encoded(mlxRead(join(input.store,name))),encoded(pointers[name]),'MLX source pointer changed');const now=lstatSync(weights);assert(now.dev===st.dev&&now.ino===st.ino&&now.size===st.size&&now.mtimeMs===st.mtimeMs,'MLX weights changed');};
 return{...input,core,coreMeta,retention,active,expert,routerPointer,router,proof,acceptedReport,bound,pointers,assertUnchanged,expertFile:join(input.store,'layer-23',active.id+'.safetensors'),coreFile:join(input.store,core.version+'.safetensors')};
}
export type MLXSource=ReturnType<typeof readMLXLearningSource>;
export function selectMLXLearningRecords(samples:MLXRow[],source:MLXSource,committed:MLXRow[]=[],fixed?:MLXRecords){
 const protection:MLXRow[]=[],seen=new Map<string,MLXRow>();
 const report=source.acceptedReport,covered=new Set<string>(report?.covered_keys??[]);
 const debt:MLXRow[]=report?.learning_policy===MLX_LEARNING_POLICY?report.relearning_records??[]:[];
 const retryIDs=new Set<string>(debt.flatMap(row=>[row.id,...(materialProvenance(row).learning_train_ids??[])]));
 // Reuse the user's original authorized TRAIN rows; held-out questions stay VALID.
 const qualified=samples.filter(row=>eligibleMaterial(row)&&dailyLearningMaterial(row)&&(!covered.has(mlxLearningRecordKey(row))||retryIDs.has(row.id)))
  .sort((a,b)=>Number(retryIDs.has(b.id))-Number(retryIDs.has(a.id)));
 const train=qualified.filter(row=>row.purpose==='training').slice(0,16),valid=qualified.filter(row=>row.purpose==='validation').slice(0,16);
 const currentContexts=new Set([...train,...valid].map(materialContext));
 for(const row of mlxKnowledgeSample(report)){
  if(currentContexts.has(materialContext(row)))continue;
  validateMaterialRecord(row,materialProvenance(row).retention_generation_only===true);
  assert(row.purpose==='validation','Learned knowledge must retain its held-out split');
  row.provenance=encoded({...materialProvenance(row),retention_scope:'accepted_branch'});
  assert(!seen.has(materialContext(row)),'Duplicate learned knowledge');seen.set(materialContext(row),row);protection.push(row);
 }
 const records=fixed??{train,valid,protection};
 if(records.train.length!==16||records.valid.length<2||records.valid.length>16)throw new MLXMaterialShortage(records.train.length,records.valid.length);
 assert.equal(encoded(records.protection),encoded(protection),'MLX fixed protection differs');
 const contexts=new Map<string,string>(),ids=new Set<string>(),families=new Map<string,string>();
 for(const [purpose,rows]of [['training',records.train],['validation',records.valid]]as const)for(const row of rows){validateMaterialRecord(row);assert(row.purpose===purpose);const c=materialContext(row);assert(!contexts.has(c)&&!ids.has(row.id),'MLX TRAIN/VALID duplicate or leakage');contexts.set(c,purpose);ids.add(row.id);const family=materialProvenance(row).split_family;if(family){assert(!families.has(family)||families.get(family)===purpose,'MLX split family leakage');families.set(family,purpose);}const old=seen.get(c);assert(!old||(old.answer?old.answer===row.answer:evaluateGeneration(row.answer,validateGenerationCheck(materialProvenance(old).generation_check,old.answer),true).passed),'MLX new knowledge conflicts with retained knowledge');}
 return ledgerParse(encoded(records))as MLXRecords;
}
