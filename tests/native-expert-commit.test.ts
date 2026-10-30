import {beforeAll,afterAll,test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm,open,realpath} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {patchNativeExpertCommitStorage} from '../scripts/native-expert-source-patches.ts';
import {encoded as reportJSON} from '../backend/canonical-json.ts';
import {floatValue} from '../backend/wire.ts';
import evidence from '../research/native_evolution_final_binding_20261017.json';
const root=resolve(import.meta.dir,'..'),research=join(root,'.native-migration/research'),hash=(b:any)=>Bun.CryptoHasher.hash('sha256',b,'hex');
const replace=(s:string,a:string,b:string)=>{assert.equal(s.split(a).length,2);return s.replace(a,b);};let work:string,api:any;
beforeAll(async()=>{work=await realpath(await mkdtemp(join(tmpdir(),'expert-commit-')));const entry=join(work,'entry.ts'),path=join(research,'native-learning-product/commit-storage.ts');await writeFile(entry,`export * from ${JSON.stringify(path)};export {Memory,ledgerParse} from ${JSON.stringify(join(research,'native-product-runtime/backend/memory.ts'))};export {encoded} from ${JSON.stringify(join(research,'native-product-runtime/backend/evolution-snapshot-files.ts'))};`);
 const build=await Bun.build({entrypoints:[entry],target:'bun',plugins:[{name:'actual-learning-commit',setup(b){b.onLoad({filter:/native-learning-product\/commit-storage\.ts$/},async a=>({loader:'ts',contents:patchNativeExpertCommitStorage(await readFile(a.path,'utf8'),replace)+'\nexport {plans as testPlans,cas as testCAS};'}));}}]});assert(build.success,build.logs.map(String).join('\n'));const out=join(work,'test.js');await writeFile(out,await build.outputs[0]!.arrayBuffer());api=await import(out);});
afterAll(async()=>{await rm(work,{recursive:true,force:true});});
async function fixture(){const store=await mkdtemp(join(work,'store-')),id='123456789abc',expert='learned-'+id,identity=hash(store),width=32,hidden=2;
 const base={identity,model_manifest_sha256:'b'.repeat(64),config_sha256:'c'.repeat(64),checkpoint_sha256:'d'.repeat(64),tokenizer_sha256:'e'.repeat(64),original_down_sha256:'f'.repeat(64),engine_binary_sha256:'0'.repeat(64),core_parameter:'model.layers.23.mlp.down_proj.weight',core_shape:[hidden,3],dtype:'BF16',device:'mps'};
 let offset=0;const header:any={};for(const key of ['gate.weight','up.weight','down.weight']){header[key]={dtype:'BF16',shape:key==='down.weight'?[hidden,width]:[width,hidden],data_offsets:[offset,offset+hidden*width*2]};offset+=hidden*width*2;}const h=Buffer.from(JSON.stringify(header)),length=Buffer.alloc(8);length.writeBigUInt64LE(BigInt(h.length));const weights=Buffer.concat([length,h,Buffer.alloc(offset)]);
 const report={id,base_identity:identity,identity,core_version:null,active_expert:expert,chosen_width:width,promoted:true,quality_checked:true,learning_rate:floatValue(.0002),validation_loss:floatValue(7.62939453125e-6),gates:{bf16_quality:true,q4_training:true},records_sha256:hash('{}')};
 const meta={identity:identity+':layer-23',hidden,width,sha256:hash(weights),core_version:null,requires_gate:true};const names=['layer-23/'+expert+'.safetensors','layer-23/'+expert+'.json','learning-'+id+'.json'],bytes=[weights,Buffer.from(reportJSON(meta)),Buffer.from(reportJSON(report))];await mkdir(join(store,'layer-23'));for(let i=0;i<names.length;i++)await writeFile(join(store,names[i]!),bytes[i]!);
 const memory=new api.Memory(work);memory.model_identity=identity;const lockDir=`/tmp/luban-model-locks-${process.getuid!()}`;await mkdir(lockDir,{recursive:true,mode:0o700});const lockPath=join(lockDir,identity+'-cpu.lock'),lock=await open(lockPath,'wx+',0o600);
 const binding={...api.ledgerParse(evidence.original_binding_json),identity,scope:'product',device:'cpu',core_version:null,expert_id:null,arguments:{steps:64,protocol:'native-expert-preservation-v1'},records_sha256:report.records_sha256,active_files_sha256:api.nativeAcceptedHashes(store)},body={id,kind:'learning',scope:'product',binding,arguments:binding.arguments,records:{},started:1,owner:null};
 memory.connect((db:any)=>{db.exec('CREATE TABLE IF NOT EXISTS training_runs(id TEXT PRIMARY KEY,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)');db.query('DELETE FROM training_runs').run();db.query('DELETE FROM events WHERE id=?').run('training:'+id);db.query('INSERT INTO training_runs VALUES(?,?,?,?,?,?)').run(id,identity,'learning','running',api.encoded(body),1);});
 const request={memory,store,base,guard:{modelLockFd:lock.fd,assertQuiescent(){}},expected:{id,binding}},pointer={id:expert,width,identity,core_version:null};return{request,report,names,bytes,pointer,close:async()=>{memory.close();await lock.close();await rm(lockPath);}};
}
test('actual Learning plans and SQLite preserve typed report floats, exact CAS and original rollback',async()=>{
 for(const rollback of [false,true]){const f=await fixture();try{
  const input={report:f.report,pointer:f.pointer,artifacts:f.names};expect(api.prepareLearningCommitStorage(f.request,input).state).toBe('committing');
  const row=f.request.memory.connect((db:any)=>db.query('SELECT * FROM training_runs').get()),body=api.ledgerParse(row.body);expect(()=>api.testPlans(f.request,body.commit)).not.toThrow();
  for(const value of ['0.0002',.0003]){const bad=api.ledgerParse(api.encoded(body.commit));bad.report.learning_rate=value;expect(()=>api.testPlans(f.request,bad)).toThrow('typed values');}
  api.applyLearningCommitStorage(f.request);expect(JSON.parse(await readFile(join(f.request.store,'active.json'),'utf8'))).toEqual(f.pointer);
  if(rollback){api.restoreLearningCommitPointer(f.request,'actual reload failed');await expect(readFile(join(f.request.store,'active.json'))).rejects.toThrow('ENOENT');expect(api.finishLearningCommitRejection(f.request).state).toBe('rejected');}
  else{expect(api.finishLearningCommitStorage(f.request).state).toBe('committed');const event=f.request.memory.connect((db:any)=>db.query('SELECT * FROM events').get());expect(event.body).toBe(api.encoded(body.commit.report));}
  expect(()=>f.request.memory.connect((db:any)=>api.testCAS(db,row,'running',body))).toThrow('full-row CAS');
 }finally{await f.close();}}
});
