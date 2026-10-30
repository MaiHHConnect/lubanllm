import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {Memory} from '../backend/memory.ts';
import {heldFixture} from './native-evolution-held-source.test.ts';
const worker=await readFile(new URL('../build/native-evolution/worker.js',import.meta.url),'utf8');
function compiledFunction(name:string){const start=worker.search(new RegExp('(?:async )?function '+name+'\\(')),end=worker.indexOf('\n}',start);assert(start>=0&&end>start);return worker.slice(start,end+2);}
const compiled=(compiledFunction('source2')+'\n'+compiledFunction('bindCore450Previous')).replace(/\b(assert|Memory|ledgerParse|validateNativeProductRoundArguments|nativeAcceptedHashes|equal|hash|encoded)\d+\b/g,'$1');
const hash=(x:Buffer|string)=>createHash('sha256').update(x).digest('hex');
const equal=(a:unknown,b:unknown)=>{try{assert.deepEqual(a,b);return true;}catch{return false;}};
const journalCode=(await readFile(new URL('../.native-migration/backend/native-training-journal-start.ts',import.meta.url),'utf8')).replace(/import[^\n]*\n/g,'').replaceAll('export ','');
const journal=new Function('need',new Bun.Transpiler({loader:'ts'}).transformSync(journalCode+'\nreturn {nativeTrainingJournalSchema,nativeTrainingPending,nativeTrainingCAS};'))(assert);
async function fixture(){
 const h=await heldFixture({core:true,expert:true,router:true}),memory:any=new Memory(h.directory);memory.model_identity='model';
 const base={identity:'model'},records={groups:{new_training:[],new_holdout:[],replay_training:[],replay_holdout:[]},protection:[]},implementation={worker:'compiled'},arguments_={steps:450,product_round:{absolute_start_step:0,parent:null}};
 Object.assign(h.engine,{config:{identity:'model',routing:{store:h.directory},aten:{base}}});
 const preparation={id:'123456789abc',base,records},binding={identity:'model',scope:'product',device:'mps',dtype:'torch.bfloat16',core_version:null,expert_id:h.peer.previous.id,arguments:arguments_,active_files_sha256:h.nativeAcceptedHashes(h.directory),implementation_sha256:implementation,records_sha256:hash(JSON.stringify(records))};
 const body={id:preparation.id,kind:'evolution',scope:'product',checkpoint:null,progress:{step:0,phase:'preparing'},arguments:arguments_,binding,records};
 memory.connect((db:any)=>{journal.nativeTrainingJournalSchema(db,true);db.query('INSERT INTO training_runs VALUES (?,?,?,?,?,?)').run(preparation.id,'model','evolution','running',JSON.stringify(body),1);});
 let scans=0;let beforeCAS:(db:any)=>void=()=>{};const cohort={records,assertUnchanged(){}};
 const scope={...journal,nativeTrainingCAS:(db:any,...args:any[])=>{beforeCAS(db);return journal.nativeTrainingCAS(db,...args);},assert,Memory,assertNativeExpertScopeGuard:(_e:any,g:any)=>assert(g===h.guard),equal,selectIndependentNativeEvolutionCohort:()=>cohort,readNativeEvolutionHeldSource:h.read,ledgerParse:JSON.parse,validateNativeProductRoundArguments:(a:any)=>a.product_round,nativeCompiledTrainingImplementation:()=>implementation,hash,encoded:JSON.stringify,nativeEvolutionPreparationBinding:(p:any)=>({id:p.id}),nativeAcceptedHashes:()=>{scans++;return h.nativeAcceptedHashes(h.directory);}};
 const bind=new Function(...Object.keys(scope),compiled+'\nreturn bindCore450Previous;')(...Object.values(scope));
 return{...h,memory,body,binding,cohort,preparation,bind,args:{memory,store:h.directory,base,id:preparation.id,preparation},scans:()=>scans,beforeCAS(f:(db:any)=>void){beforeCAS=f;},async close(){memory.close();await h.close();}};
}
test('actual journal binding accepts effective None with physical B45/expert and writes only preparation through original SQLite CAS',async()=>{
 const h=await fixture();try{const result=await h.bind(h.engine,h.guard,h.args);expect(result.checkpoint_created).toBe(false);expect(h.scans()).toBe(1);const row=h.memory.connect((db:any)=>db.query('SELECT body FROM training_runs WHERE id=?').get(h.preparation.id));const body=JSON.parse(row.body);expect(body.binding.active_files_sha256).toEqual(h.binding.active_files_sha256);expect(body.binding.core_version).toBeNull();expect(body.coordinator).toEqual({native_preparation:{id:h.preparation.id},initial_state:'awaiting_native_complete_pt0'});await expect(h.bind(h.engine,h.guard,h.args)).rejects.toThrow('Only the actual fresh');}finally{await h.close();}
});
test('actual journal binding retains final physical comparison and full-row CAS on concurrent mutation',async()=>{
 for(const kind of ['pointer','row']){const h=await fixture();try{
  let calls=0;if(kind==='row')h.beforeCAS(db=>db.query('UPDATE training_runs SET updated=2 WHERE id=?').run(h.preparation.id));
  if(kind==='pointer'){
   // Change the actual tiny pointer at the final synchronous CAS boundary.
   const {writeFileSync}=await import('node:fs');h.cohort.assertUnchanged=()=>{if(++calls===2)writeFileSync(h.directory+'/active-core.json','{"version":"core-rival"}');};
  }
  await expect(h.bind(h.engine,h.guard,h.args)).rejects.toThrow();const saved=h.memory.connect((db:any)=>JSON.parse(db.query('SELECT body FROM training_runs WHERE id=?').get(h.preparation.id).body));expect(saved.coordinator).toBeUndefined();
 }finally{await h.close();}}
});
test('actual checkpoint source consumer admits None/B45 separately and rejects changed binding before native PT transport',async()=>{
 const source=worker.slice(worker.indexOf('async function loadCore450CheckpointHeld('),worker.indexOf('// .native-migration/research/native-core450-runner/candidate.ts')).replace(/\b(assert|equal)\d+\b/g,'$1');
 const h=await fixture();try{
  let loads=0;const scope={assert,equal,assertNativeExpertScopeGuard:(_e:any,g:any)=>assert(g===h.guard),readNativeEvolutionHeldSource:h.read,borrowNativeEvolutionPreparation:()=>({}),loadNativeTrainingCheckpoint:async()=>{loads++;throw Error('native PT transport boundary');}};
  const load=new Function(...Object.keys(scope),source+'\nreturn loadCore450CheckpointHeld;')(...Object.values(scope));
  const product={...h.preparation,assertUnchanged:async()=>{}},r={...h.args,expected:{id:h.preparation.id,kind:'evolution',binding:h.binding}};
  await expect(load(h.engine,h.guard,r,product)).rejects.toThrow('native PT transport boundary');expect(loads).toBe(1);
  r.expected={...r.expected,binding:{...h.binding,core_version:'core-b45d965728bb'}};await expect(load(h.engine,h.guard,r,product)).rejects.toThrow();expect(loads).toBe(1);
  r.expected={...r.expected,binding:{...h.binding,active_files_sha256:{...h.binding.active_files_sha256,'active-router.json':null}}};await expect(load(h.engine,h.guard,r,product)).rejects.toThrow();expect(loads).toBe(1);
  r.expected={...r.expected,binding:h.binding};await h.put('active-core.json',{version:'core-rival'});await expect(load(h.engine,h.guard,r,product)).rejects.toThrow('Admitted source path changed');expect(loads).toBe(1);
 }finally{await h.close();}
});
