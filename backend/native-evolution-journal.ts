/** Original Python failure semantics: reject execution errors, retain all checkpoint evidence. */
import assert from 'node:assert/strict';
import type {Database} from 'bun:sqlite';
import {ledgerParse,ledgerStringify,type Memory} from './memory.ts';
import {nativeTrainingCAS,nativeTrainingJournalSchema,nativeTrainingPending} from '../.native-migration/backend/native-training-journal-start.ts';
import {trainingProcessStarted} from '../.native-migration/backend/native-training-journal.ts';
type Row={id:string;identity:string;kind:string;state:string;body:string;updated:number};
function empty(row:Row,owned:boolean){
 const body=failed(row,owned);
 assert(body.checkpoint===null&&body.previous_checkpoint==null&&body.commit==null&&body.product_accepted!==true&&body.progress?.step===0,'A saved checkpoint or commit must remain recoverable');
 return body;
}

function failed(row:Row,owned:boolean){
 assert(row&&/^[a-f0-9]{12}$/.test(row.id)&&row.kind==='evolution'&&['queued','running','paused'].includes(row.state),'Only a pending original evolution run may be closed');
 const body=ledgerParse(row.body);
 assert(body.id===row.id&&body.kind==='evolution'&&body.scope==='product'&&body.binding?.identity===row.identity&&body.arguments?.steps===450,'Original evolution journal identity/configuration differs');
 assert(body.commit==null&&body.product_accepted!==true,'A publication intent or accepted model must finish its original lifecycle');
 assert(Number.isSafeInteger(body.progress?.step)&&body.progress.step>=0&&body.progress.step<=450,'Original training progress is invalid');
 if(body.owner!=null){
  const owner=body.owner;assert(Number.isSafeInteger(owner.pid)&&owner.pid>0&&Number.isFinite(Number(owner.started)),'Original journal owner invalid');
  const actual=trainingProcessStarted(owner.pid),alive=actual!==null&&Math.abs(actual-Number(owner.started))<.01;
  if(owned)assert(alive&&owner.pid===process.pid,'Failure cleanup requires its actual current owner');else assert(!alive,'Another live process still owns this run');
 }else assert(!owned,'Current-owner cleanup cannot claim an ownerless run');
 return body;
}

/** Only the original producer's successful afterRestore may invoke the owned failure exit. */
export function rejectFailedNativeEvolutionJournal(db:Database,row:Row,reason:string,context:{owned:boolean;restored:boolean;cancelled:boolean}){
 assert(context.restored&&!context.cancelled,'Restore must finish before rejecting an execution failure; cancellation remains paused');
 const body=failed(row,context.owned);
 assert(typeof reason==='string'&&reason.length>0&&reason.length<=16384);
 if(!context.owned)assert(row.state==='paused'&&body.progress.phase==='paused_after_failure'
  &&typeof body.coordinator?.last_error==='string'&&body.coordinator.last_error.length>0,'Only an original restored execution failure may be closed');
 const next={...body,owner:null,reason,progress:{...body.progress,phase:'failed'},
  coordinator:{...body.coordinator,last_error:context.owned?reason:body.coordinator.last_error}};
 const saved=nativeTrainingCAS(db,row,'rejected',ledgerStringify(next),Date.now()/1000);
 assert.deepEqual(db.query('SELECT * FROM training_runs WHERE id=?').get(row.id),saved);return saved;
}

/** The legacy producer wrote paused_after_failure only after its real restoration succeeded. */
export function recoverNativeEvolutionFailedRun(memory:Pick<Memory,'connect'>,request:{id:string;identity:string;reason:string;expectedBody?:string;expectedUpdated?:number}){
 return memory.connect(db=>{
  nativeTrainingJournalSchema(db);const row=db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(request.id,request.identity)as Row|null;
  assert(row,'Original failed evolution journal missing');const body=failed(row,false);
  assert(row.state==='paused'&&body.progress.phase==='paused_after_failure'&&typeof body.coordinator?.last_error==='string'
   &&body.coordinator.last_error.length>0,'Only an original restored execution failure may be closed');
  assert(typeof request.reason==='string'&&request.reason.length>0&&request.reason.length<=16384);
  if(request.expectedBody===undefined)return{id:row.id,state:row.state,body:row.body,updated:row.updated,checkpoint:body.checkpoint};
  assert(row.body===request.expectedBody&&row.updated===request.expectedUpdated,'Failed journal changed after preview');
  const saved=rejectFailedNativeEvolutionJournal(db,row,request.reason,{owned:false,restored:true,cancelled:false});
  return{id:saved.id,state:saved.state,body:saved.body,updated:saved.updated,checkpoint:body.checkpoint};
 });
}

/** Match Python's incompatible-resume rejection; process exit is not a fabricated restore ACK. */
export function recoverNativeEvolutionIncompatibleRun(memory:Pick<Memory,'connect'>,request:{id:string;identity:string;implementation:Record<string,string>;expectedBody?:string;expectedUpdated?:number}){
 return memory.connect(db=>{
  nativeTrainingJournalSchema(db);const row=db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(request.id,request.identity)as Row|null;
  assert(row,'Original incompatible evolution journal missing');const body=failed(row,false);
  assert(body.owner!=null,'This recovery requires the original exited process owner');
  assert(body.coordinator?.native_preparation?.id===body.id,'Original native round preparation is required');
  const original=body.binding.implementation_sha256,current=request.implementation;
  for(const map of [original,current])assert(map&&typeof map==='object'&&!Array.isArray(map)&&Object.keys(map).length>0
   &&Object.entries(map).every(([key,value])=>key.length>0&&typeof value==='string'&&/^[a-f0-9]{64}$/.test(value)),'Original and current compiled implementation maps are required');
  assert.notDeepEqual(original,current,'Same-source checkpoint remains resumable; incompatible recovery refused');
  if(request.expectedBody===undefined)return{id:row.id,state:row.state,body:row.body,updated:row.updated,checkpoint:body.checkpoint};
  assert(row.body===request.expectedBody&&row.updated===request.expectedUpdated,'Failed journal changed after preview');
  const next={...body,owner:null,reason:'原进程已退出，当前编译来源与原检查点绑定不兼容；保留检查点结束本轮，未取得内存恢复回执',
   progress:{...body.progress,phase:'source_incompatible_after_owner_exit'}};
  const saved=nativeTrainingCAS(db,row,'rejected',ledgerStringify(next),Date.now()/1000);
  assert.deepEqual(db.query('SELECT * FROM training_runs WHERE id=?').get(row.id),saved);
  return{id:saved.id,state:saved.state,body:saved.body,updated:saved.updated,checkpoint:body.checkpoint};
 });
}
export function rejectEmptyNativeEvolutionJournal(db:Database,row:Row,reason:string,owned=false){
 const body=empty(row,owned);assert(typeof reason==='string'&&reason.length>0&&reason.length<=16384);
 const next={...body,owner:null,reason,progress:{...body.progress,phase:'preparing_failed'},coordinator:{...body.coordinator,last_error:reason}};
 const saved=nativeTrainingCAS(db,row,'rejected',ledgerStringify(next),Date.now()/1000);
 assert.deepEqual(db.query('SELECT * FROM training_runs WHERE id=?').get(row.id),saved);return saved;
}
/** Preview is read-only; apply requires the exact body returned by that preview and the original row CAS. */
export function recoverNativeEvolutionEmptyRun(memory:Pick<Memory,'connect'>,request:{id:string;identity:string;reason:string;expectedBody?:string;expectedUpdated?:number}){
 return memory.connect(db=>{nativeTrainingJournalSchema(db);const row=db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(request.id,request.identity)as Row|null;assert(row,'Original failed evolution journal missing');empty(row,false);
  if(request.expectedBody===undefined)return{id:row.id,state:row.state,empty:true as const,body:row.body,updated:row.updated};
  assert(row.body===request.expectedBody&&row.updated===request.expectedUpdated,'Failed journal changed after preview');
  const saved=rejectEmptyNativeEvolutionJournal(db,row,request.reason);return{id:saved.id,state:saved.state,empty:true as const,body:saved.body,updated:saved.updated};
 });
}

/** Explicitly supersede a safely cancelled round; keep its checkpoint and original binding as evidence. */
export function retirePausedNativeEvolution(memory:Pick<Memory,'connect'>,request:{id:string;identity:string;reason:string;expectedBody?:string;expectedUpdated?:number}){
 return memory.connect(db=>{
  nativeTrainingJournalSchema(db);const row=db.query('SELECT * FROM training_runs WHERE id=? AND identity=?').get(request.id,request.identity)as Row|null;
  assert(row,'Original paused evolution journal missing');const body=failed(row,false);
  assert(row.state==='paused'&&body.owner===null&&body.progress.phase==='paused_after_cancel','Only a safely cancelled ownerless round may be superseded');
  assert(typeof request.reason==='string'&&request.reason.trim().length>0&&request.reason.length<=16384);
  if(request.expectedBody===undefined)return{id:row.id,state:row.state,body:row.body,updated:row.updated,checkpoint:body.checkpoint};
  assert(row.body===request.expectedBody&&row.updated===request.expectedUpdated,'Paused journal changed after preview');
  nativeTrainingPending(db,request.identity,request.id);
  const next={...body,reason:request.reason,progress:{...body.progress,phase:'superseded_by_user'}};
  const saved=nativeTrainingCAS(db,row,'rejected',ledgerStringify(next),Date.now()/1000);
  return{id:saved.id,state:saved.state,body:saved.body,updated:saved.updated,checkpoint:body.checkpoint};
 });
}

/** Fail before fixture preparation; the original start/commit checks still close races. */
export function assertNativeEvolutionPending(memory:Pick<Memory,'connect'>,identity:string,resumeId?:string){
 memory.connect(db=>{if(db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'").get())nativeTrainingPending(db,identity,resumeId??'');});
}
