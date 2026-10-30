import {test,expect} from 'bun:test';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {Memory,ledgerParse,ledgerStringify} from '../backend/memory.ts';
import {nativeTrainingJournalSchema} from '../.native-migration/backend/native-training-journal-start.ts';
import {retirePausedNativeEvolution as retire} from '../backend/native-evolution-journal.ts';

test('explicit retirement preserves checkpoint and rejects stale CAS, publication or unfinished owner',async()=>{
 const root=await mkdtemp(join(tmpdir(),'retire-evolution-')),memory=new Memory(root),id='123456789abc',identity='a'.repeat(64);
 const body={id,kind:'evolution',scope:'product',binding:{identity},arguments:{steps:450},progress:{step:96,steps:450,phase:'paused_after_cancel'},owner:null,checkpoint:{name:'preserved.pt'},records:{unchanged:true}};
 const request={id,identity,reason:'User selected an existing candidate instead of this training round'};
 const reset=(change:any={})=>memory.connect(db=>{nativeTrainingJournalSchema(db,true);db.query('INSERT OR REPLACE INTO training_runs VALUES (?,?,?,?,?,?)').run(id,identity,'evolution','paused',ledgerStringify({...body,...change}),1);});
 const row=()=>memory.connect(db=>db.query('SELECT * FROM training_runs WHERE id=?').get(id)as any);
 try{
  reset();const before=row(),preview=retire(memory,request);expect(row()).toEqual(before);
  memory.connect(db=>db.query('UPDATE training_runs SET updated=2 WHERE id=?').run(id));
  expect(()=>retire(memory,{...request,expectedBody:preview.body,expectedUpdated:preview.updated})).toThrow('changed after preview');
  reset();const p=retire(memory,request),result=retire(memory,{...request,expectedBody:p.body,expectedUpdated:p.updated}),saved=ledgerParse(result.body);
  expect(result.state).toBe('rejected');expect(saved.checkpoint).toEqual(body.checkpoint);expect(saved.binding).toEqual(body.binding);expect(saved.records).toEqual(body.records);expect(saved.progress.step).toBe(96);expect(saved.progress.phase).toBe('superseded_by_user');
  for(const change of [{commit:{}},{product_accepted:true},{owner:{pid:2147483647,started:1}},{progress:{step:96,phase:'paused_after_failure'}}]){reset(change);const b=row();expect(()=>retire(memory,request)).toThrow();expect(row()).toEqual(b);}
 }finally{memory.close();await rm(root,{recursive:true,force:true});}
});
