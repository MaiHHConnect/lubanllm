import {afterAll,beforeAll,test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm,open,realpath} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {patchNativeEvolutionCommitBinding} from '../scripts/native-evolution-checkpoint-source-patches.ts';
import {encoded as modern} from '../backend/canonical-json.ts';
import {parseJSON as backendParse} from '../backend/wire.ts';
import bindingEvidence from '../research/native_evolution_final_binding_20261017.json';
import {materialContentSHA} from '../backend/native-evolution-material.ts';
let work:string,api:any;
const root=resolve(import.meta.dir,'..'),research=join(root,'.native-migration/research');
const sha=(bytes:any)=>Bun.CryptoHasher.hash('sha256',bytes,'hex');
const replace=(body:string,before:string,after:string)=>{assert.equal(body.split(before).length,2);return body.replace(before,after);};
beforeAll(async()=>{
 work=await realpath(await mkdtemp(join(tmpdir(),'native-commit-report-')));
 const path=join(research,'native-core-product-publication-v1/commit-storage.ts'),owner=join(research,'native-unified-service/checkpoint-owner.ts'),entry=join(work,'entry.ts');
 await writeFile(entry,`export * from ${JSON.stringify(path)};export {Memory,ledgerParse} from ${JSON.stringify(join(research,'native-product-runtime/backend/memory.ts'))};export {encoded,equal} from ${JSON.stringify(join(research,'native-product-runtime/backend/evolution-snapshot-files.ts'))};`);
 const output=await Bun.build({entrypoints:[entry],target:'bun',plugins:[{name:'actual-core-storage',setup(b){
  b.onLoad({filter:/native-core-product-publication-v1\/commit-storage\.ts$/},async a=>{const original=await readFile(a.path,'utf8'),start=original.indexOf('function plans('),end=original.indexOf('/** Persist intent',start),legacy=original.slice(start,end).replace('function plans(','function oldPlans(');return{loader:'ts',contents:patchNativeEvolutionCommitBinding(original,owner,replace)+'\n'+legacy+'\nexport {plans as testPlans,oldPlans as testOldPlans,cas as testCAS};'};});
  // A CPU fixture cannot issue a native GPU receipt. Only that private ACK boundary is substituted.
  b.onLoad({filter:/native-core-peer-refresh-source-v1\/reload\.ts$/},()=>({loader:'ts',contents:'export function validateRetainedCoreReloadReceipt(receipt:any){if(receipt.cpu_fixture_ack!==true)throw Error("missing actual ACK");}'}));
 }}]});assert(output.success,output.logs.map(String).join('\n'));const file=join(work,'fixture.js');await writeFile(file,await output.outputs[0]!.arrayBuffer());api=await import(file);
});
afterAll(async()=>{if(work)await rm(work,{recursive:true,force:true});});
async function fixture(){
 const store=await mkdtemp(join(work,'store-')),id='123456789abc',version='core-'+id;
 const base={identity:sha(store),model_manifest_sha256:'b'.repeat(64),config_sha256:'c'.repeat(64),checkpoint_sha256:'d'.repeat(64),tokenizer_sha256:'e'.repeat(64),original_down_sha256:'f'.repeat(64),engine_binary_sha256:'0'.repeat(64),core_parameter:'model.layers.23.mlp.down_proj.weight',core_shape:[2,3],dtype:'BF16',device:'mps'};
 const h=Buffer.from(JSON.stringify({[base.core_parameter]:{dtype:'BF16',shape:[2,3],data_offsets:[0,12]}})),n=Buffer.alloc(8);n.writeBigUInt64LE(BigInt(h.length));const weights=Buffer.concat([n,h,Buffer.alloc(12)]);
 const retained={schema_version:1,identity:base.identity,version,groups:{new_training:[],new_validation:[],replay_training:[],replay_validation:[]},regression_validation:[]},retention=Buffer.from(modern({...retained,content_sha256:materialContentSHA(retained,true)}));
 const meta={version,identity:base.identity,sha256:sha(weights),parameter:base.core_parameter,shape:[2,3],dtype:'torch.bfloat16',format:'cumulative core tensor replacement',trained_parameters:6,retention_sha256:sha(retention),report:'evolution-'+id+'.json'};
 let report:any={identity:base.identity,version,id,promoted:true,publication_state:'published',checkpoint:meta,quality_checked:true,gates:{protected_core_retention_loss:true},core_retention_loss:{rows:[{tolerance:7.62939453125e-6,excess_over_tolerance:-7.62939453125e-6}]}};
 // Use the complete actual v3 metadata when mounted; never read the model/checkpoint tensor files.
 const actual='/Volumes/移动数据库/lubanllm/modules/luban0-v1/evolution-6cad091b179a.json';
 try{report={...backendParse(await readFile(actual,'utf8')),...report};}catch(e:any){if(e.code!=='ENOENT')throw e;}
 const names=[version+'.safetensors',version+'.json',version+'.retention.json',meta.report],bytes=[weights,Buffer.from(modern(meta)),retention,Buffer.from(modern(report))];
 for(let i=0;i<names.length;i++)await writeFile(join(store,names[i]!),bytes[i]!);
 const memory=new api.Memory(work);memory.model_identity=base.identity;
 const lockDirectory=`/tmp/luban-model-locks-${process.getuid!()}`;await mkdir(lockDirectory,{recursive:true,mode:0o700});const lockPath=join(lockDirectory,base.identity+'-cpu.lock'),lock=await open(lockPath,'wx+',0o600);
 const records={},binding={...api.ledgerParse(bindingEvidence.original_binding_json),identity:base.identity,scope:'product',device:'cpu',arguments:{steps:0},records_sha256:sha(api.encoded(records)),active_files_sha256:api.nativeAcceptedHashes(store)};
 const body={id,kind:'evolution',scope:'product',binding,arguments:binding.arguments,records,started:1,owner:null};
 memory.connect((db:any)=>{db.exec('CREATE TABLE IF NOT EXISTS training_runs(id TEXT PRIMARY KEY,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)');db.query('DELETE FROM training_runs').run();db.query('DELETE FROM events WHERE id=?').run('training:'+id);db.query('INSERT INTO training_runs VALUES(?,?,?,?,?,?)').run(id,base.identity,'evolution','running',api.encoded(body),1);});
 const r={memory,store,base,guard:{modelLockFd:lock.fd,assertQuiescent(){}},expected:{id,binding}},pointer={version,identity:base.identity,sha256:meta.sha256};
 return{r,report,names,bytes,pointer,meta,async close(){memory.close();await lock.close();await rm(lockPath);}};
}
test('actual complete report preserves typed values across independent wire parsers; plans rejects tampering before intent',async()=>{
 const f=await fixture();try{
  const artifact=api.ledgerParse(f.bytes[3]!.toString()),intent=api.ledgerParse(api.encoded(f.report));expect(api.equal(artifact,intent)).toBe(false);
  const commit={report:intent,pointers:[{name:'active-core.json',before:null,after:f.pointer}],artifacts:Object.fromEntries(f.names.map((name,i)=>[name,sha(f.bytes[i]!)])),event_kind:'evolution'};
  expect(()=>api.testOldPlans(f.r,commit)).toThrow('report artifact differs');expect(api.testPlans(f.r,commit)).toEqual(commit.pointers[0]);
  for(const mutate of [(r:any)=>r.core_retention_loss.rows[0].tolerance=0.5,(r:any)=>r.core_retention_loss.rows[0].tolerance='7.62939453125e-06',(r:any)=>r.core_retention_loss.rows[0].tolerance=0,(r:any)=>r.current_updates=new Number(0),(r:any)=>r.source_checkpoint_sha256='f'.repeat(64),(r:any)=>delete r.core_retention_loss.rows[0].tolerance]){
   const bad=api.ledgerParse(api.encoded(commit));mutate(bad.report);expect(()=>api.testPlans(f.r,bad)).toThrow('typed values');
  }
  const bad=api.ledgerParse(api.encoded(commit));bad.artifacts[f.names[0]]='0'.repeat(64);expect(()=>api.testPlans(f.r,bad)).toThrow('complete SHA');
  expect(f.r.memory.connect((db:any)=>db.query('SELECT state FROM training_runs').get()).state).toBe('running');
 }finally{await f.close();}
});
test('actual SQLite prepare/apply/finish and event preserve report; source CAS and stale full-row CAS still reject',async()=>{
 const f=await fixture();try{
  const input={report:f.report,pointer:f.pointer,artifacts:f.names};
  const ready=api.prepareNativeCoreCommit(f.r,input);expect(ready.state).toBe('committing');expect(ready.pointer_changed).toBe(false);
  api.applyNativeCoreCommit(f.r);expect(JSON.parse(await readFile(join(f.r.store,'active-core.json'),'utf8'))).toEqual(f.pointer);
  const receipt={cpu_fixture_ack:true,restore_id:f.r.expected.id,loaded:{core_loaded:true,core_file_sha256:f.meta.sha256,core_tensor_sha256:sha(Buffer.alloc(12))}};
  expect(()=>api.finishNativeCoreCommit(f.r,{...receipt,cpu_fixture_ack:false})).toThrow('ACK');
  expect(api.finishNativeCoreCommit(f.r,receipt).state).toBe('committed');
  const row=f.r.memory.connect((db:any)=>db.query('SELECT * FROM training_runs').get()),event=f.r.memory.connect((db:any)=>db.query('SELECT * FROM events WHERE id=?').get('training:'+f.r.expected.id));
  expect(row.state).toBe('committed');expect(event.body).toBe(api.encoded(api.ledgerParse(row.body).commit.report));
  f.r.memory.connect((db:any)=>db.query('UPDATE training_runs SET updated=updated+1 WHERE id=?').run(row.id));
  expect(()=>f.r.memory.connect((db:any)=>api.testCAS(db,row,'committed',api.ledgerParse(row.body)))).toThrow('full-row CAS');
 }finally{await f.close();}
});

test('actual source inventory refuses a competing pointer and actual rejection preserves before selection',async()=>{
 const f=await fixture();try{
  const input={report:f.report,pointer:f.pointer,artifacts:f.names};
  await writeFile(join(f.r.store,'active-core.json'),JSON.stringify(f.pointer));
  expect(()=>api.prepareNativeCoreCommit(f.r,input)).toThrow('accepted model files changed');await rm(join(f.r.store,'active-core.json'));
  api.prepareNativeCoreCommit(f.r,input);api.applyNativeCoreCommit(f.r);api.restoreNativeCoreCommitPointer(f.r,'injected reload failure');
  await expect(readFile(join(f.r.store,'active-core.json'))).rejects.toThrow('ENOENT');
  expect(api.finishNativeCoreCommitRejection(f.r,{cpu_fixture_ack:true,restore_id:f.r.expected.id,loaded:{core_loaded:false}}).state).toBe('rejected');
  const saved=f.r.memory.connect((db:any)=>db.query('SELECT * FROM training_runs').get());expect(saved.state).toBe('rejected');expect(api.ledgerParse(saved.body).commit.report.promoted).toBe(true);
 }finally{await f.close();}
});
