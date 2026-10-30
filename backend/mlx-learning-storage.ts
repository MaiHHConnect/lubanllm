/** Durable MLX learning intent, pointer compare-and-swap and actual-reload finalization. */
import assert from 'node:assert/strict';
import {randomBytes} from 'node:crypto';
import {readFileSync,writeFileSync,mkdirSync,existsSync,copyFileSync,constants,unlinkSync} from 'node:fs';
import {join,dirname} from 'node:path';
import {Memory,ledgerParse} from './memory.ts';
import {encoded} from './canonical-json.ts';
import {privateJSON} from './settings.ts';
import {nativeTrainingJournalSchema,nativeTrainingCAS,nativeTrainingPending} from '../.native-migration/backend/native-training-journal-start.ts';
import {MLX_LEARNING_PROTOCOL,mlxHash,mlxRead,type MLXSource,type MLXRecords,type MLXRow} from './mlx-learning-source.ts';
import {MLX_LEARNING_POLICY,MLX_TRAINING_ADMISSION,MLX_NET_GATES} from './mlx-learning-policy.ts';
import {MLX_QUALITY_GATES} from './mlx-learning-quality.ts';

export function mlxRuns(memory:Pick<Memory,'connect'>,identity:string){return memory.connect(db=>db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'").get()?db.query('SELECT * FROM training_runs WHERE identity=? ORDER BY updated DESC').all(identity)as MLXRow[]:[]);}
export function mlxPending(memory:Pick<Memory,'connect'>,identity:string){const rows=mlxRuns(memory,identity).filter(row=>['queued','running','paused','committing'].includes(row.state)||row.state==='error'&&ledgerParse(row.body).arguments?.protocol===MLX_LEARNING_PROTOCOL);assert(rows.length<=1);return rows[0]??null;}
/** The lease owner may restore only this interrupted intent's own three old/new paths. */
export function inspectMLXCommitRecovery(memory:Pick<Memory,'connect'>,input:{root:string;store:string;identity:string}){
 const pending=mlxPending(memory,input.identity);if(!pending||pending.state!=='committing')return;
 const body=ledgerParse(pending.body);assert(body.arguments?.protocol===MLX_LEARNING_PROTOCOL&&body.binding.identity===input.identity,'Pending commit belongs to another engine');
 const wanted=[join(input.store,'active.json'),join(input.store,'active-router.json'),join(input.root,'runtime/mlx/expert-proof.json')];const pointers=body.commit?.pointers;assert(Array.isArray(pointers)&&pointers.length===3&&pointers.every((p:any,i:number)=>p.path===wanted[i]),'MLX pending commit path differs');
 const core=mlxRead(join(input.store,'active-core.json'));assert(core.version===body.binding.core_version&&core.sha256===body.binding.core_file_sha256,'MLX pending core changed');
 for(const p of pointers)assert([encoded(p.before),encoded(p.after)].includes(encoded(mlxRead(p.path))),'MLX interrupted publication has a rival pointer');
 return{pending,body,pointers};
}
export function recoverMLXCommitPointers(memory:Memory,input:{root:string;store:string;identity:string}){
 const recovery=inspectMLXCommitRecovery(memory,input);if(recovery)for(const p of [...recovery.pointers].reverse())privateJSON(p.path,p.before);
}
function immutable(path:string,bytes:Buffer){mkdirSync(dirname(path),{recursive:true,mode:0o700});if(existsSync(path)){assert(readFileSync(path).equals(bytes),'MLX immutable artifact differs');return;}writeFileSync(path,bytes,{flag:'wx',mode:0o600});}
export class MLXLearningJournal{
 readonly id:string;body:MLXRow;state:string;
 constructor(readonly memory:Memory,readonly source:MLXSource,records:MLXRecords,resume?:MLXRow,automatic?:{signature:string;scheduled_occurrence:string|null}){
  if(resume){this.id=resume.id;this.body=ledgerParse(resume.body);this.state=resume.state;assert(this.body.arguments?.protocol===MLX_LEARNING_PROTOCOL&&encoded(this.body.records)===encoded(records),'MLX resume material/protocol mismatch');for(const[key,value]of Object.entries(source.bound))assert(this.body.binding[key]===value,'MLX resume source differs: '+key);assert(this.body.binding.records_sha256===mlxHash(encoded(records)));}
  else{if(automatic)assert(/^[a-f0-9]{64}$/.test(automatic.signature));assert(!mlxPending(memory,source.identity),'已有MLX断点或失败记录，必须恢复原轮');this.id=randomBytes(6).toString('hex');this.state='queued';const arguments_={protocol:MLX_LEARNING_PROTOCOL,steps:64,seed:42,preservation_weight:5,learning_policy:MLX_LEARNING_POLICY,training_admission:MLX_TRAINING_ADMISSION};this.body={id:this.id,kind:'learning',scope:'product',started:Date.now()/1000,automatic_signature:automatic?.signature??null,scheduled_occurrence:automatic?.scheduled_occurrence??null,arguments:arguments_,records,binding:{...source.bound,identity:source.identity,device:'mps',scope:'product',arguments:arguments_,records_sha256:mlxHash(encoded(records))},owner:{pid:process.pid},checkpoint:null,candidates:[],baseline:null,commit:null};
   memory.connect(db=>{nativeTrainingJournalSchema(db,true);nativeTrainingPending(db,source.identity,this.id);db.query('INSERT INTO training_runs VALUES (?,?,?,?,?,?)').run(this.id,source.identity,'learning',this.state,encoded(this.body),Date.now()/1000);});}
 }
 update(state:string,changes:MLXRow={}){const next={...this.body,...changes};this.memory.connect(db=>{nativeTrainingJournalSchema(db);nativeTrainingPending(db,this.source.identity,this.id);const before=db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(this.id,this.source.identity)as any;assert(before&&before.state===this.state&&before.body===encoded(this.body),'MLX journal changed concurrently');nativeTrainingCAS(db,before,state,encoded(next),Date.now()/1000);});this.body=next;this.state=state;}
 prepare(report:MLXRow,weightsPath:string){
  this.source.assertUnchanged();assert(report.learning_policy===this.body.arguments.learning_policy,'MLX learning policy differs from fixed journal');assert(report.promoted===true&&report.quality_checked===true&&(report.learning_policy===MLX_LEARNING_POLICY?MLX_NET_GATES:MLX_QUALITY_GATES).every(k=>report.gates?.[k]===true),'MLX full quality proof missing');
  const id=this.id,source=this.source,expertID='learned-'+id,routerID='router-'+id,reportName='learning-'+id+'.json';const weights=readFileSync(weightsPath),sha=mlxHash(weights),flags={execution_scope:'product',inference_engine:'mlx-q4',quality_admitted:true,training_admitted:true};
  const expertPointer={id:expertID,width:report.chosen_width,identity:source.identity,core_version:source.core.version,...flags};
  const expertMeta={identity:source.identity+':layer-23',hidden:2048,width:report.chosen_width,sha256:sha,requires_gate:true,core_version:source.core.version,report:reportName,...flags};
  const router={...source.router,id:routerID,expert_id:expertID,expert_sha256:sha,core_version:source.core.version,...flags,mlx_learning:{report:reportName,source_router_id:source.routerPointer.id,head_unchanged:true,route_checked:true}};
  delete router.trial_engine;delete router.migration;
  const routerBytes=Buffer.from(encoded(router)+'\n'),routerPointer={id:routerID,identity:source.identity,core_version:source.core.version,sha256:source.router.sha256,metadata_sha256:mlxHash(routerBytes),...flags};
  report={...report,id,active_expert:expertID,base_identity:source.identity,identity:source.identity,core_version:source.core.version,records_sha256:this.body.binding.records_sha256,records:this.body.records,scope:report.learning_policy===MLX_LEARNING_POLICY?'Held-out new knowledge gain exceeds loss on sampled learned knowledge; not a whole-base capability claim':'MLX expert learning with complete retained protection; not a whole-base capability claim',inference_engine:'mlx-q4',router_id:routerID,expert_file_sha256:sha};
  const reportBytes=Buffer.from(encoded(report)+'\n'),proof={schema:'mlx-expert-learning-v1',scope:'product',quality_admitted:true,training_admitted:true,...source.bound,expert_id:expertID,expert_file_sha256:sha,width:report.chosen_width,router_id:routerID,route_checked:true,report:reportName,report_sha256:mlxHash(reportBytes)};
  const artifactBytes:Record<string,Buffer>={['layer-23/'+expertID+'.safetensors']:weights,['layer-23/'+expertID+'.json']:Buffer.from(encoded(expertMeta)+'\n'),[routerID+'.json']:routerBytes,[reportName]:reportBytes};
  const oldRouterFile=join(source.store,source.routerPointer.id+'.safetensors'),newRouterFile=join(source.store,routerID+'.safetensors');assert(!existsSync(newRouterFile));copyFileSync(oldRouterFile,newRouterFile,constants.COPYFILE_EXCL);
  for(const[name,bytes]of Object.entries(artifactBytes))immutable(join(source.store,name),bytes);
  const proofPath=join(source.root,'runtime/mlx/expert-proof.json');const pointers=[{path:join(source.store,'active.json'),before:source.active,after:expertPointer},{path:join(source.store,'active-router.json'),before:source.routerPointer,after:routerPointer},{path:proofPath,before:source.proof,after:proof}];
  const artifacts=Object.fromEntries(Object.entries(artifactBytes).map(([name,bytes])=>[name,mlxHash(bytes)]));artifacts[routerID+'.safetensors']=source.router.sha256;
  this.update('committing',{candidates:report.candidates,commit:{report,pointers,artifacts,event_kind:'learning'},owner:{pid:process.pid}});return{report,proof,expertPointer,expertMeta,routerPointer,router,weightsPath:join(source.store,'layer-23/'+expertID+'.safetensors')};
 }
 apply(){assert(this.state==='committing'&&this.body.commit);for(const p of this.body.commit.pointers){const current=mlxRead(p.path);assert([encoded(p.before),encoded(p.after)].includes(encoded(current)),'MLX pointer rival change');privateJSON(p.path,p.after);assert.equal(encoded(mlxRead(p.path)),encoded(p.after));}}
 rollback(reason:string){assert(this.state==='committing');for(const p of [...this.body.commit.pointers].reverse()){const current=mlxRead(p.path);assert([encoded(p.before),encoded(p.after)].includes(encoded(current)),'MLX rollback rival change');privateJSON(p.path,p.before);}this.update('committing',{rejecting:{reason}});}
 /** Called only after the new worker's exact loaded acknowledgement and closed-book reload check. */
 finish(reload:MLXRow){assert(this.state==='committing'&&!this.body.rejecting&&reload.expert_loaded===true&&reload.expert_file_sha256===this.body.commit.report.expert_file_sha256&&reload.core_file_sha256===this.source.core.sha256&&reload.mlx_model_sha256===this.source.bound.mlx_model_sha256,'MLX actual publication reload differs');for(const p of this.body.commit.pointers)assert.equal(encoded(mlxRead(p.path)),encoded(p.after));
  this.memory.connect(db=>{const before=db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(this.id,this.source.identity)as any;assert(before.state===this.state&&before.body===encoded(this.body));const eventID='training:'+this.id,event=db.query('SELECT kind,body FROM events WHERE id=?').get(eventID)as any,body=encoded(this.body.commit.report);assert(!event||event.kind==='learning'&&event.body===body);db.query('INSERT OR IGNORE INTO events VALUES (?,?,?,?)').run(eventID,'learning',body,this.body.started);nativeTrainingCAS(db,before,'committed',encoded({...this.body,owner:null}),Date.now()/1000);});this.body={...this.body,owner:null};this.state='committed';
 }
}
