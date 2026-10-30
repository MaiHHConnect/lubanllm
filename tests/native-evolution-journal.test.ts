import {test,expect} from 'bun:test';
import {mkdtemp,rm,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {Memory,ledgerStringify,ledgerParse} from '../backend/memory.ts';
import {nativeTrainingJournalSchema,nativeTrainingCAS} from '../.native-migration/backend/native-training-journal-start.ts';
import {trainingProcessStarted} from '../.native-migration/backend/native-training-journal.ts';
import {recoverNativeEvolutionEmptyRun,rejectEmptyNativeEvolutionJournal,recoverNativeEvolutionFailedRun,recoverNativeEvolutionIncompatibleRun,rejectFailedNativeEvolutionJournal,assertNativeEvolutionPending} from '../backend/native-evolution-journal.ts';
async function fixture(change:any={}){const root=await mkdtemp(join(tmpdir(),'native-empty-journal-')),memory=new Memory(root),id='29e46ec82bd9',identity='a'.repeat(64);const body={id,kind:'evolution',scope:'product',binding:{identity},arguments:{steps:450},progress:{step:0,phase:'paused_after_failure'},checkpoint:null,previous_checkpoint:null,commit:null,owner:null,...change};memory.connect(db=>{nativeTrainingJournalSchema(db,true);db.query('INSERT INTO training_runs VALUES (?,?,?,?,?,?)').run(id,identity,'evolution','paused',ledgerStringify(body),1);});return{memory,request:{id,identity,reason:'Original failure before CP0'},row:()=>memory.connect(db=>db.query('SELECT * FROM training_runs WHERE id=?').get(id)as any),async close(){memory.close();await rm(root,{recursive:true,force:true});}};}
test('orphan empty journal preview changes no row, exact body CAS rejects while preserving original records',async()=>{const h=await fixture({records:{authorized:'preserved'}});try{const before=h.row(),preview=recoverNativeEvolutionEmptyRun(h.memory,h.request);expect(h.row()).toEqual(before);const result=recoverNativeEvolutionEmptyRun(h.memory,{...h.request,expectedBody:preview.body,expectedUpdated:preview.updated});expect(result.state).toBe('rejected');expect(ledgerParse(h.row().body).records).toEqual({authorized:'preserved'});expect(ledgerParse(h.row().body).checkpoint).toBeNull();}finally{await h.close();}});
test('saved checkpoint, prior checkpoint, commit, nonzero step and live owner cannot be retired',async()=>{for(const change of[{checkpoint:{name:'keep.pt'}},{previous_checkpoint:{name:'keep.pt'}},{commit:{}},{progress:{step:1}},{owner:{pid:process.pid,started:trainingProcessStarted(process.pid)}}]){const h=await fixture(change);try{const before=h.row();expect(()=>recoverNativeEvolutionEmptyRun(h.memory,h.request)).toThrow();expect(h.row()).toEqual(before);}finally{await h.close();}}});
test('preview-to-apply race cannot overwrite changed journal',async()=>{const h=await fixture();try{const preview=recoverNativeEvolutionEmptyRun(h.memory,h.request);h.memory.connect(db=>db.query('UPDATE training_runs SET updated=2 WHERE id=?').run(h.request.id));expect(()=>recoverNativeEvolutionEmptyRun(h.memory,{...h.request,expectedBody:preview.body,expectedUpdated:preview.updated})).toThrow('changed after preview');expect(h.row().state).toBe('paused');}finally{await h.close();}});
test('owned failure closes zero checkpoint only through original row CAS',async()=>{const h=await fixture({owner:{pid:process.pid,started:trainingProcessStarted(process.pid)}});try{const row=h.row();h.memory.connect(db=>rejectEmptyNativeEvolutionJournal(db,row,'actual bootstrap failure',true));expect(h.row().state).toBe('rejected');expect(ledgerParse(h.row().body).owner).toBeNull();}finally{await h.close();}});

test('actual afterRestore consumer rejects execution errors with or without CP, but cancellation and quality stay paused',async()=>{
 const worker=await readFile(new URL('../build/native-evolution/worker.js',import.meta.url),'utf8');
 const source=worker.slice(worker.indexOf('function pauseOwnedCore450Journal('),worker.indexOf('async function generateCore450Rows(')).replace(/\b(assert|ledgerParse|trainingProcessStarted|equal|encoded)\d+\b/g,'$1');
 const scope={assert:(await import('node:assert/strict')).default,ledgerParse,trainingProcessStarted,nativeTrainingCAS,rejectFailedNativeEvolutionJournal,encoded:ledgerStringify,equal:(a:any,b:any)=>ledgerStringify(a)===ledgerStringify(b),assertNativeExpertScopeGuard:()=>{}};
 const pause=new Function(...Object.keys(scope),source+';return pauseOwnedCore450Journal;')(...Object.values(scope));
 for(const checkpoint of [null,{name:'actual-saved-generation.pt',sha256:'b'.repeat(64)}])for(const mode of ['failure','cancel','quality']){
  const h=await fixture({checkpoint,owner:{pid:process.pid,started:trainingProcessStarted(process.pid)}});try{
   h.memory.connect(db=>db.query("UPDATE training_runs SET state='running' WHERE id=?").run(h.request.id));
   expect(()=>assertNativeEvolutionPending(h.memory,h.request.identity)).toThrow('另一个训练任务');
   expect(()=>assertNativeEvolutionPending(h.memory,h.request.identity,h.request.id)).not.toThrow();
   const signal=mode==='cancel'?AbortSignal.abort():undefined;
   pause({}, {},{memory:h.memory,signal,expected:{id:h.request.id,binding:ledgerParse(h.row().body).binding}},mode==='failure'?'paused_after_failure':mode==='cancel'?'paused_after_cancel':'awaiting_quality',mode==='quality'?undefined:Error('fixture preparation failed'));
   expect(h.row().state).toBe(mode==='failure'?'rejected':'paused');expect(ledgerParse(h.row().body).checkpoint).toEqual(checkpoint);
   if(mode==='failure')expect(()=>assertNativeEvolutionPending(h.memory,h.request.identity)).not.toThrow();
  }finally{await h.close();}
 }
});

const savedFailure={checkpoint:{name:'8a3b0bb3be1463dfbafa99b841819cc5.pt',sha256:'d94e3dc768d1e64ba3cc9ad6943f5e14bc5db968e1c84f5a8362d9ba032234b1',progress:{step:0,steps:450,phase:'training'}},
 previous_checkpoint:{name:'earlier-evidence.pt'},records:{authorized:'unchanged'},coordinator:{last_error:'optimizer dictionary requires integer keys',native_preparation:{fixture:{directory:'/original/fixture'}}}};
test('restored failed CP preview is read-only; rejection preserves CP, preparation, records and source binding',async()=>{
 const h=await fixture(savedFailure);try{
  const before=h.row(),old=ledgerParse(before.body),preview=recoverNativeEvolutionFailedRun(h.memory,h.request);
  expect(h.row()).toEqual(before);expect(preview.checkpoint).toEqual(savedFailure.checkpoint);
  expect(()=>assertNativeEvolutionPending(h.memory,h.request.identity)).toThrow();
  const result=recoverNativeEvolutionFailedRun(h.memory,{...h.request,expectedBody:preview.body,expectedUpdated:preview.updated}),next=ledgerParse(result.body);
  expect(result.state).toBe('rejected');expect(next.checkpoint).toEqual(old.checkpoint);expect(next.previous_checkpoint).toEqual(old.previous_checkpoint);
  expect(next.binding).toEqual(old.binding);expect(next.records).toEqual(old.records);expect(next.coordinator).toEqual(old.coordinator);expect(next.owner).toBeNull();
  expect(()=>assertNativeEvolutionPending(h.memory,h.request.identity)).not.toThrow();
 }finally{await h.close();}
});
test('failed-run recovery rejects cancelled pauses, missing failure evidence, active owners, commit intent and accepted state',async()=>{
 for(const change of [{progress:{step:0,phase:'paused_after_cancel'}},{coordinator:{}},{owner:{pid:process.pid,started:trainingProcessStarted(process.pid)}},{commit:{artifacts:{}}},{product_accepted:true}]){
  const h=await fixture({...savedFailure,...change});try{const before=h.row();expect(()=>recoverNativeEvolutionFailedRun(h.memory,h.request)).toThrow();expect(h.row()).toEqual(before);}finally{await h.close();}
 }
 for(const state of ['committing','committed','rejected']){const h=await fixture(savedFailure);try{h.memory.connect(db=>db.query('UPDATE training_runs SET state=? WHERE id=?').run(state,h.request.id));const before=h.row();expect(()=>recoverNativeEvolutionFailedRun(h.memory,h.request)).toThrow();expect(h.row()).toEqual(before);}finally{await h.close();}}
});
test('failed-run preview and direct shared exit both preserve a concurrently changed row',async()=>{
 const h=await fixture(savedFailure);try{
  const before=h.row(),preview=recoverNativeEvolutionFailedRun(h.memory,h.request);
  h.memory.connect(db=>db.query('UPDATE training_runs SET updated=2 WHERE id=?').run(h.request.id));const changed=h.row();
  expect(()=>recoverNativeEvolutionFailedRun(h.memory,{...h.request,expectedBody:preview.body,expectedUpdated:preview.updated})).toThrow('changed after preview');
  expect(()=>h.memory.connect(db=>rejectFailedNativeEvolutionJournal(db,before,'execution failure',{owned:false,restored:true,cancelled:false}))).toThrow('并发变化');
  expect(h.row()).toEqual(changed);
 }finally{await h.close();}
});
test('owned failure requires completed restore and rejects cancellation without changing CP',async()=>{
 const h=await fixture({...savedFailure,owner:{pid:process.pid,started:trainingProcessStarted(process.pid)}});try{
  const before=h.row();for(const context of [{owned:true,restored:false,cancelled:false},{owned:true,restored:true,cancelled:true}]){
   expect(()=>h.memory.connect(db=>rejectFailedNativeEvolutionJournal(db,before,'execution failure',context))).toThrow();expect(h.row()).toEqual(before);
  }
  h.memory.connect(db=>rejectFailedNativeEvolutionJournal(db,before,'execution failure',{owned:true,restored:true,cancelled:false}));
  expect(h.row().state).toBe('rejected');expect(ledgerParse(h.row().body).checkpoint).toEqual(savedFailure.checkpoint);
 }finally{await h.close();}
});

const oldImplementation={'service:consumer.ts':'a'.repeat(64)},newImplementation={'service:consumer.ts':'b'.repeat(64)};
async function incompatibleFixture(change:any={}){const h=await fixture({...savedFailure,binding:{identity:'a'.repeat(64),implementation_sha256:oldImplementation},owner:{pid:2147483647,started:1},progress:{step:16,phase:'training'},coordinator:{native_preparation:{id:'29e46ec82bd9'}},...change});h.memory.connect(db=>db.query("UPDATE training_runs SET state='queued' WHERE id=?").run(h.request.id));return h;}
test('exited-owner incompatible source closure preserves CP and old binding without claiming restore',async()=>{
 const h=await incompatibleFixture();try{
  const before=h.row(),request={...h.request,implementation:newImplementation},preview=recoverNativeEvolutionIncompatibleRun(h.memory,request);expect(h.row()).toEqual(before);
  const result=recoverNativeEvolutionIncompatibleRun(h.memory,{...request,expectedBody:preview.body,expectedUpdated:preview.updated}),body=ledgerParse(result.body);
  expect(result.state).toBe('rejected');expect(body.binding.implementation_sha256).toEqual(oldImplementation);expect(body.checkpoint).toEqual(savedFailure.checkpoint);expect(body.previous_checkpoint).toEqual(savedFailure.previous_checkpoint);
  expect(body.progress).toEqual({step:16,phase:'source_incompatible_after_owner_exit'});expect(body.reason).toContain('未取得内存恢复回执');expect(body.owner).toBeNull();expect(body.coordinator).toEqual(ledgerParse(before.body).coordinator);
 }finally{await h.close();}
});
test('incompatible-source closure rejects same implementation, live owner, commit and source metadata omission',async()=>{
 for(const change of [{owner:{pid:process.pid,started:trainingProcessStarted(process.pid)}},{owner:null},{commit:{}},{binding:{identity:'a'.repeat(64)}},{product_accepted:true}]){const h=await incompatibleFixture(change);try{const before=h.row();expect(()=>recoverNativeEvolutionIncompatibleRun(h.memory,{...h.request,implementation:newImplementation})).toThrow();expect(h.row()).toEqual(before);}finally{await h.close();}}
 const h=await incompatibleFixture();try{expect(()=>recoverNativeEvolutionIncompatibleRun(h.memory,{...h.request,implementation:oldImplementation})).toThrow('Same-source checkpoint remains resumable');expect(()=>recoverNativeEvolutionIncompatibleRun(h.memory,{...h.request,implementation:{}})).toThrow();}finally{await h.close();}
});
test('incompatible-source preview cannot overwrite a concurrently updated journal',async()=>{
 const h=await incompatibleFixture();try{const request={...h.request,implementation:newImplementation},preview=recoverNativeEvolutionIncompatibleRun(h.memory,request);h.memory.connect(db=>db.query('UPDATE training_runs SET updated=2 WHERE id=?').run(h.request.id));const before=h.row();expect(()=>recoverNativeEvolutionIncompatibleRun(h.memory,{...request,expectedBody:preview.body,expectedUpdated:preview.updated})).toThrow('changed after preview');expect(h.row()).toEqual(before);}finally{await h.close();}
});
