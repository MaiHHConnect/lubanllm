import { test, expect } from "bun:test";
import { spawnSync } from "node:child_process";

test("effective None and branch-OFF saved expert create a real journal while physical pointer CAS stays strict", () => {
  const child = spawnSync(process.execPath, ["--eval", String.raw`
import assert from 'node:assert/strict';import{mock}from'bun:test';import{Database}from'bun:sqlite';
import{readFile,mkdtemp,mkdir,writeFile,rm,realpath}from'node:fs/promises';import{readFileSync,existsSync,openSync,closeSync,unlinkSync,mkdirSync}from'node:fs';import{join,resolve,dirname}from'node:path';import{tmpdir}from'node:os';
import{applyNativeEvolutionTrainingSourcePatches}from'./scripts/native-evolution-training-source-patches.ts';
const root=process.cwd(),original=resolve('.native-migration/backend/native-training-start-v4.ts'),boot=resolve('.native-migration/research/native-core450-owner-controller/bootstrap.ts');
const replace=(body,from,to)=>{assert.equal(body.split(from).length,2,from);return body.replace(from,to);};
const bodies=new Map([[original,await readFile(original,'utf8')],[boot,'export async function createCore450ProductFreshCheckpoint(){\n   session=await startNativeTrainingSessionV4({core_version:r.preparation.graphBinding.accepted_core_version,expert_id:borrowNativeEnginePeer(engine,guard).previous?.id??null,});}']]);
applyNativeEvolutionTrainingSourcePatches({root,bodies,target:p=>p,replace});
const source=bodies.get(original);assert(source.includes('equal(nativeAcceptedHashes(r.store),binding.active_files_sha256)'));assert(source.includes("nativeTrainingCAS(db,expected,'running',expected.body"));
const stub=(name,value)=>mock.module(resolve(name),()=>value),old='.native-migration/';
let peer,live=true,mutate=false,hashCalls=0;
class Engine{};
stub(old+'research/native-unified-expert/native-engine.ts',{NativeEngine:Engine,assertNativeExpertScopeGuard:(_e,g)=>g.assertQuiescent()});
stub(old+'research/native-unified-peer/engine-peer.ts',{borrowNativeEnginePeer:()=>peer});
class Memory{constructor(root,identity){this.root=root;this.model_identity=identity;this.db=new Database(':memory:');this.db.exec('CREATE TABLE training_runs(id TEXT PRIMARY KEY,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)');}connect(fn){return fn(this.db);}}
const encoded=JSON.stringify,equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b),hash=v=>Bun.CryptoHasher.hash('sha256',v,'hex'),need=(v,m)=>assert(v,m);
stub(old+'research/native-product-runtime/backend/memory.ts',{Memory,ledgerParse:JSON.parse});
stub(old+'backend/evolution-snapshot-files.ts',{encoded,equal,hash,need,object:v=>v&&typeof v==='object',digest:v=>typeof v==='string'&&/^[a-f0-9]{64}$/.test(v),readSafe:(s,n)=>readFileSync(join(s,n)),pathIn:join});
const hashes=store=>{if(mutate&&++hashCalls===2)require('node:fs').writeFileSync(join(store,'active.json'),JSON.stringify({id:'changed-expert'}));return Object.fromEntries(['active-core.json','active.json'].filter(n=>existsSync(join(store,n))).map(n=>[n,hash(readFileSync(join(store,n)))]));};
stub(old+'backend/native-commit.ts',{nativeAcceptedHashes:hashes});
const keys=['protocol','identity','scope','core_version','expert_id','device','dtype','torch_version','arguments','packages','active_files_sha256','implementation_sha256','records_sha256'];
stub(old+'backend/native-training-journal.ts',{nativeTrainingBindingKeys:keys,trainingProcessStarted:()=>1});
stub(old+'backend/native-training-journal-start.ts',{nativeTrainingJournalSchema:()=>{},nativeTrainingPending:()=>{},nativeTrainingCAS:(db,expected,state,body,updated)=>{const prior=db.query('SELECT * FROM training_runs WHERE id=?').get(expected.id);assert.deepEqual(prior,expected,'full-row CAS mismatch');const next={...expected,state,body,updated};db.query('UPDATE training_runs SET state=?,body=?,updated=? WHERE id=?').run(state,body,updated,expected.id);return next;}});
stub(old+'research/native-unified-service/training-programs-v4.ts',{admitReceipt:()=>{throw Error('no GPU/package execution')}});
const sha='a'.repeat(64);stub(old+'research/native-unified-service/compiled-authority-v4.ts',{RELEASE_AUTHORITY:{manifest_sha256:sha}});
stub(old+'research/native-unified-service/compiled-service-source-v4.ts',{SERVICE_SOURCE_AUTHORITY:{native_manifest_sha256:sha,service_source_manifest:{fixture:sha},native_role_source_proofs:{fixture:sha},libtorch_version_header_sha256:sha}});
stub(old+'backend/wire.ts',{floatValue:v=>v});
const work=await realpath(await mkdtemp(join(tmpdir(),'native-source-binding-'))),modulePath=join(work,'start.ts');
let moduleText=source.replace('function createSession(input:','export function createSession(input:').replace(/from (['"])(\.[^'"]+)\1/g,(_m,_q,path)=>'from '+JSON.stringify(resolve(dirname(original),path)));
await writeFile(modulePath,moduleText);const{createSession}=await import(modulePath);
const{nativeEvolutionTrainingSource,assertNativeTrainingPointerIdentity}=await import(resolve('backend/native-evolution-training-source.ts'));
let cases=0;
try{for(const scenario of ['legacy-none','qualified','ordinary-none','wrong-core','missing-expert','forged-cap','pointer-race','running-race','branch-on','scope-ended','foreign-guard']){
 const home=join(work,scenario),store=join(home,'modules');await mkdir(store,{recursive:true,mode:0o700});
 const identity=Bun.CryptoHasher.hash('sha256',scenario,'hex'),guardDir='/tmp/luban-model-locks-'+process.getuid();mkdirSync(guardDir,{recursive:true,mode:0o700});
 const lockPath=join(guardDir,identity+'-cpu.lock'),fd=openSync(lockPath,'wx',0o600);const memory=new Memory(home,identity);
 const physical=scenario==='ordinary-none'?null:'core-123456789abc',core=scenario==='qualified'?physical:null,expert='gated-expert';
 if(physical)await writeFile(join(store,'active-core.json'),JSON.stringify({version:physical}));await writeFile(join(store,'active.json'),JSON.stringify({id:expert}));
 let branch=false;live=true;mutate=false;hashCalls=0;
 const engine=new Engine();engine.config={identity};const guard={modelLockFd:fd,assertQuiescent(){assert(live);}};
 const cohort={core:core?{version:core}:null,physical_core:physical?{version:physical}:null,core_requires_review:physical&&!core?{version:physical}:null,assertUnchanged(){assert(live);}};
 peer={source:{runtime:{core_label:core}},previous:{id:expert},store,worker:{assertLive(){assert(live);},get isBranchEnabled(){return branch;}}};
 try{
  const cap=nativeEvolutionTrainingSource(engine,guard,cohort),request={memory,store,id:'abcdef123456',kind:'evolution',scope:'product',identity,core_version:cap.core_version,expert_id:cap.expert_id,evolutionSource:cap,device:'cpu',dtype:'torch.bfloat16',torch_version:'fixture',packages:{safetensors:'fixture',transformers:'fixture'},arguments:{steps:450},records:{fixture:true},implementation:{},guard,assertSource(){guard.assertQuiescent();}};
  if(scenario==='wrong-core')request.core_version=physical;
  if(scenario==='missing-expert')request.expert_id=null;
  if(scenario==='forged-cap')request.evolutionSource={...cap};
  if(scenario==='foreign-guard')request.guard={...guard};
  if(scenario==='branch-on')branch=true;
  if(scenario==='scope-ended')live=false;
  if(scenario==='pointer-race')mutate=true;
  if(['legacy-none','qualified','ordinary-none','running-race'].includes(scenario)){
   const session=createSession(request);assert.equal(session.binding.core_version,core);assert.equal(session.binding.expert_id,expert);assert.deepEqual(session.binding.active_files_sha256,hashes(store));
   assert.equal(memory.db.query('SELECT state FROM training_runs').get().state,'queued');
   if(scenario==='running-race'){await writeFile(join(store,'active.json'),JSON.stringify({id:'changed-expert'}));assert.throws(()=>session.running(),/source changed/);assert.equal(memory.db.query('SELECT state FROM training_runs').get().state,'queued');}
   else{session.running();assert.equal(memory.db.query('SELECT state FROM training_runs').get().state,'running');assert.equal(JSON.parse(readFileSync(join(store,'active.json'),'utf8')).id,expert);if(physical)assert.equal(JSON.parse(readFileSync(join(store,'active-core.json'),'utf8')).version,physical);}
  }else{assert.throws(()=>createSession(request));assert.equal(memory.db.query('SELECT COUNT(*) AS n FROM training_runs').get().n,0);}
  // Without the private live cap, the legacy entry retains its exact physical-pointer check.
  if(scenario==='legacy-none')assert.throws(()=>assertNativeTrainingPointerIdentity({...request,evolutionSource:undefined},{core_version:physical,expert_id:expert}));
  cases++;
 }finally{memory.db.close();closeSync(fd);unlinkSync(lockPath);}
}
console.log(JSON.stringify({cases,passed:true}));
}finally{await rm(work,{recursive:true,force:true});}
`], { cwd: process.cwd(), encoding: "utf8", timeout: 30_000 });
  expect(child.status, child.stderr).toBe(0);
  expect(JSON.parse(child.stdout)).toEqual({ cases: 11, passed: true });
});
