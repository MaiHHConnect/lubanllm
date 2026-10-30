import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {Database} from 'bun:sqlite';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
import {coreCheckpointField as field,coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {core450ControlMetadata} from '../.native-migration/research/native-core450-owner-controller/metadata.ts';
import {nativePartialCorrectIDs} from '../.native-migration/research/native-core-controller-candidate/progress.ts';
import {CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary} from '../.native-migration/research/native-core-controller-candidate/segments.ts';
import {encoded} from '../backend/canonical-json.ts';
import {ledgerParse} from '../backend/memory.ts';
import {provenance} from '../backend/native-evolution-material.ts';
import {nativeTrainingCAS} from '../.native-migration/backend/native-training-journal-start.ts';
import {identityImportMaterial} from '../scripts/identity-import-material.ts';
import evidence from '../research/native_evolution_final_policy_20261017.json';

const ts=new Bun.Transpiler({loader:'ts'}),source=await readFile('backend/native-evolution-import.ts','utf8');
function extract(begin:string,end:string){const start=source.indexOf(begin),stop=source.indexOf(end,start);assert(start>=0&&stop>start);return ts.transformSync(source.slice(start,stop).replace(/^export /gm,''));}
const worker=await readFile('build/native-evolution/worker.js','utf8');
const safety=worker.slice(worker.indexOf('var CORE450_SAFETY_POLICY ='),worker.indexOf('// .native-migration/research/native-core450-runner/resume.ts',worker.indexOf('var CORE450_SAFETY_POLICY =')))
 .replace(/\bassert\d+\b/g,'assert');
const history=new Function('assert','CORE450_CHECKS','CORE450_DURABLE','nextCore450Boundary','nativePartialCorrectIDs','exposures','eq','ordinary',safety+';return validateCore450TrainingHistory;')
 (assert,CORE450_CHECKS,CORE450_DURABLE,nextCore450Boundary,nativePartialCorrectIDs,coreNewExposures,pythonMetadataEqual,validateOrdinaryCoreHistory);
const validate=new Function('assert','REFINED_IDENTITY_SOURCE','ledgerParse','field','core450ControlMetadata','encoded','provenance','validateCore450TrainingHistory',
 extract('export function validateImportSource','/** Same durable')+';return validateImportSource;')
 (assert,{id:'84dbd6c8f41a'},ledgerParse,field,core450ControlMetadata,encoded,provenance,history);

test('actual v19 metadata and compiled safety contract admit only unchanged original teaching plus declared readonly punctuation revision',async()=>{
 const db=new Database('/Volumes/移动数据库/lubanllm/app/memory.sqlite3',{readonly:true});let row:any;
 try{row=db.query('SELECT * FROM training_runs WHERE id=?').get('84dbd6c8f41a');}finally{db.close();}
 const tree=JSON.parse(await readFile(join(evidence.reader_directory,'checkpoint-tree.json'),'utf8'));
 const manifest=JSON.parse(await readFile(join(evidence.reader_directory,'manifest.json'),'utf8'));
 const body=ledgerParse(row.body),material=identityImportMaterial({samples:Object.values(body.records.groups).flat() as any[],directedTeaching:body.records.directed_teaching,retentionRegressions:body.records.retention_regressions});
 const records={...body.records,retention_regressions:material.retentionRegressions};
 const result=validate(row,manifest,tree,records);expect(result.safety.completed_steps).toBe(450);expect(result.safety.complete_training_safety).toBe(true);
 expect(result.checkpoint_sha256).toBe(body.checkpoint.sha256);expect(result.previous_sha256).toBe(evidence.reference.sha256);
 for(const state of ['running','committed','paused'])expect(()=>validate({...row,state},manifest,tree,records)).toThrow();
 const altered=structuredClone(records);altered.groups.new_training[0].answer='wrong';expect(()=>validate(row,manifest,tree,altered)).toThrow();
 const unsafe=structuredClone(records),p=provenance(unsafe.retention_regressions[0]);p.generation_check.accepted.push('林浩');unsafe.retention_regressions[0].provenance=JSON.stringify(p);expect(()=>validate(row,manifest,tree,unsafe)).toThrow();
 expect(()=>validate(row,{...manifest,checkpoint_sha256:'f'.repeat(64)},tree,records)).toThrow();
 const mutated=ledgerParse(row.body);mutated.checkpoint.progress.step=448;expect(()=>validate({...row,body:JSON.stringify(mutated)},manifest,tree,records)).toThrow();
});

test('import transaction waits for actual reload, restores physical B45 on every pending failure, and never rolls back committed state',async()=>{
 const code=extract('export async function commitImportedCore','export async function importNativeIdentityCandidate');
 for(const failure of ['none','quality','prepare','apply','reload','finish','rollback-reload']){
  const db=new Database(':memory:');db.exec('CREATE TABLE training_runs(id TEXT PRIMARY KEY, identity TEXT, kind TEXT,state TEXT,body TEXT,updated REAL)');
  let row:any={id:'abcdef123456',identity:'identity',kind:'evolution',state:'running',body:'{}',updated:1};db.query('INSERT INTO training_runs VALUES (?,?,?,?,?,?)').run(...Object.values(row));
  let pointer='core-b45d965728bb',resident='original-none',rolled=false,reloaded=false,retries=0,failedOnce=false;
  const calls:string[]=[];const write=(state:string)=>{row=nativeTrainingCAS(db,row,state,JSON.stringify({pointer}),row.updated+1);};
  const boom=(where:string)=>{if(failure===where&&!failedOnce){failedOnce=true;throw Error(where);}};
  const run=new Function('prepareNativeCoreCommit','applyNativeCoreCommit','finishNativeCoreCommit','restoreNativeCoreCommitPointer','finishNativeCoreCommitRejection','settleRetainedCoreReload','closeImportOwner',code+';return commitImportedCore;')(
   ()=>{calls.push('prepare');boom('prepare');write('committing');},()=>{pointer='candidate';calls.push('apply');boom('apply');},
   ()=>{assert.equal(resident,'candidate');boom('finish');write('committed');calls.push('committed');},
   ()=>{assert.equal(row.state,'committing');pointer='core-b45d965728bb';rolled=true;calls.push('rollback');},
   ()=>{assert.equal(pointer,'core-b45d965728bb');assert.equal(resident,'original-none');write('rejected');calls.push('rejected');},
   async()=>resident==='candidate',async(fn:()=>Promise<void>)=>{for(;;){try{await fn();break;}catch{assert(++retries<3);}}});
  const control={prepareRouteTokens:async()=>({}),prepareRetainedCoreReload:async()=>({}),reloadRetainedCoreSource:async()=>{
   if(!rolled){boom('reload');if(failure==='rollback-reload'&&!failedOnce){failedOnce=true;resident='candidate';throw Error('candidate ACK failed');}}
   if(rolled&&failure==='rollback-reload'&&retries===0)throw Error('rollback reload failed once');resident=pointer==='candidate'?'candidate':'original-none';reloaded=true;return{};}};
  let result:any,error:unknown;try{result=await run({expected:{id:row.id}},control,{}, {},[],{},[],async()=>{boom('quality');},()=>{});}catch(e){error=e;}
  if(failure==='none'){expect(result.state).toBe('committed');expect(pointer).toBe('candidate');expect(rolled).toBe(false);expect(row.state).toBe('committed');}
  else{expect(error).toBeTruthy();expect(pointer).toBe('core-b45d965728bb');expect(resident).toBe('original-none');
   if(['apply','reload','finish','rollback-reload'].includes(failure)){expect(row.state).toBe('rejected');expect(reloaded).toBe(true);expect(calls.indexOf('rollback')).toBeLessThan(calls.indexOf('rejected'));}}
  db.close();
 }
});
