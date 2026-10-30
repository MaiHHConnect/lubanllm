import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm,mkdir,symlink} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {nativeExpertSourceCheck} from '../backend/native-expert-source.ts';
import {closeImportOwner} from '../backend/native-evolution-import-observe.ts';
async function load(name:string,bindings:Record<string,unknown>,exports:string[]){
 const source=(await readFile(join(import.meta.dir,'../backend',name),'utf8')).replace(/^import .*\n/gm,'').replace(/\bexport /g,'');
 const js=new Bun.Transpiler({loader:'ts',target:'bun'}).transformSync(source);
 return new Function(...Object.keys(bindings),js+'\nreturn {'+exports.join(',')+'};')(...Object.values(bindings));
}
test('admitted expert source reuses file metadata and rejects changed/added/missing/symlink source',async()=>{
 const store=await mkdtemp(join(tmpdir(),'expert-source-'));
 try{
  const files={'active.json':Buffer.from('{"id":"old"}'),'old.safetensors':Buffer.alloc(8)};for(const[name,raw]of Object.entries(files))await writeFile(join(store,name),raw,{mode:0o600});
  const check=nativeExpertSourceCheck(store,files);for(let i=0;i<20;i++)check();expect(check).not.toThrow();
  await writeFile(join(store,'active-core.json'),'{}');expect(check).toThrow('changed');await rm(join(store,'active-core.json'));
  await writeFile(join(store,'old.safetensors'),Buffer.alloc(8,1));expect(check).toThrow('changed');
  expect(()=>nativeExpertSourceCheck(store,{...files,'active.json':Buffer.from('{"id":"bad"}')})).toThrow();
  await rm(join(store,'old.safetensors'));expect(check).toThrow();await symlink(join(store,'active.json'),join(store,'old.safetensors'));expect(()=>nativeExpertSourceCheck(store,files)).toThrow();
 }finally{await rm(store,{recursive:true,force:true});}
});
async function q4Fixture(mode='ok'){
 const work=await mkdtemp(join(tmpdir(),'expert-q4-'));let calls=0,closes=0;const issued=new WeakSet();
 class Observer{async observePair(input:any){calls++;if(mode==='cancel')throw Error('cancelled');const expert=input.expert??null;
  return [0,1].map(pass=>{const value={identity:'identity',core:input.core,expert,worker_binary_sha256:'w'.repeat(64),rows:input.records.map((r:any)=>({id:r.id,input_tokens:[1,2],tokens:[3],complete:true,passed:mode==='union'&&r.id==='train'?(expert?false:pass===1):true}))};issued.add(value);return value;});}
  async close(){closes++;}}
 const api=await load('native-expert-publication-q4.ts',{assert,join,writeFile,NativeQ4Observer:Observer,assertNativeQ4Observation:(v:any,rows:any[],expert:any)=>{assert(issued.has(v));assert.deepEqual(v.rows.map((r:any)=>r.id),rows.map(r=>r.id));assert.deepEqual(v.expert,expert??null);},materialProvenance:(r:any)=>r.provenance,closeImportOwner},['observeNativeExpertPublication','assertNativeExpertQ4Publication']);
 const input={records:{train:[{id:'train',provenance:{}}],valid:[{id:'valid',provenance:{}}],protection:[{id:'core',provenance:{}}]},core:{file:'core',sha256:'c'.repeat(64),tensor_key:'down'},coreVersion:'core-current',coreTensorSHA:'t'.repeat(64),graphSHA:'g'.repeat(64),candidate:{file:'candidate',sha256:'e'.repeat(64),width:128,layer:23,core_version:'core-current'},reportName:'learning-0123456789ab.json'};
 return{work,api,input,options:{work,identity:'identity'},counts:()=>({calls,closes}),close:()=>rm(work,{recursive:true,force:true})};
}
test('Q4 actual publication helper seals both passes, reports real binding and saves failure evidence from the union baseline',async()=>{
 for(const mode of ['ok','union','cancel']){const f=await q4Fixture(mode);try{
  const action=f.api.observeNativeExpertPublication(f.options,f.input);
  if(mode==='ok'){const proof=await action;expect(()=>f.api.assertNativeExpertQ4Publication(proof)).not.toThrow();expect(()=>f.api.assertNativeExpertQ4Publication({...proof})).toThrow();expect(proof.native_q4.expert_file_sha256).toBe(f.input.candidate.sha256);expect(Object.values(proof.gates).every(Boolean)).toBe(true);}
  else await expect(action).rejects.toThrow(mode==='cancel'?'cancelled':'quality/protection');
  if(mode==='union'){const report=JSON.parse(await readFile(join(f.work,'expert-q4-'+f.input.reportName),'utf8'));expect(report.gates.q4_baseline_correct_retained).toBe(false);expect(report.baseline[0].rows[0].passed).toBe(false);expect(report.baseline[1].rows[0].passed).toBe(true);}
  expect(f.counts().closes).toBe(1);expect(f.counts().calls).toBe(mode==='cancel'?1:2);
 }finally{await f.close();}}
});
async function reloadFixture(){
 const work=await mkdtemp(join(tmpdir(),'expert-reload-')),events:string[]=[];let peerPresent=true,closed=false,badCore=false,badExpert=false,closeFailure=false,guardValid=true;
 const guard={assertQuiescent(){assert(guardValid);}},base={identity:'identity',original_down_sha256:'original',model_manifest_sha256:'manifest',engine_binary_sha256:'engine'},engine={config:{routing:{store:work},aten:{base},model:'model',log:join(work,'log'),expert_build:{binary_sha256:'graph'}}};
 const tensor={dtype:'bf16',bytes:4,shape:[1,2],sha256:'tensor'},owner={tensor_keys:['gate.weight'],tensors:{'gate.weight':tensor}},previous={owner,width:128},accepted={core:{version:'core-current',tensor_sha256:'current',file_sha256:'file'},expert:{id:'expert'},router:null};
 const transfer={owner,width:128,hidden:2048,device:'mps',assertUnchanged:async()=>{assert(!closed);}};
 const worker={pid:27,options:{directory:work,binary_sha256:'graph'},assertLive(){assert(!closed);},async retainCoreBackbone(){events.push('actual-core');return{response:{backbone_resident:!badCore,retained_core_tensor_sha256:'current'}};},async restoreReloadSnapshot(){events.push('actual-expert');return{response:{optimizer_step:badExpert?1:0,resident_parameters:3*2048*128}};},async adoptAcceptedPrevious(){events.push('adopt-source');},async setBranchEnabled(){events.push('branch');}};
 const peer={guard,closed:false,worker,accepted,previous,source:{runtime:{core_label:'core-current'}},base,owner,core:{}};
 const api=await load('native-expert-reload.ts',{assert,mkdir,mkdtemp,dirname,join,equal:(a:unknown,b:unknown)=>JSON.stringify(a)===JSON.stringify(b),assertNativeExpertScopeGuard:(_e:any,g:any)=>{assert(g===guard);g.assertQuiescent();},assertClosedExpertReload:(t:any)=>assert(t===transfer),validateExpertReloadSnapshot:(t:any)=>assert(t===transfer),validateExpertReloadInitialSnapshot:()=>{throw Error('unexpected None');},hasNativeEnginePeer:()=>peerPresent,closeNativeEnginePeer:async()=>{events.push('close-old');if(closeFailure)throw Error('old close unconfirmed');peerPresent=false;},ensureNativeEnginePeer:async()=>{assert(!peerPresent);events.push('open-new');peerPresent=true;return peer;},writePeerDescriptor:async()=>({path:'descriptor',sha256:'descriptor'})},['reloadClosedNativeExpert','validateClosedNativeExpertReload']);
 return{api,engine,guard,work,base,transfer,peer,events,setMode(mode:string){badCore=mode==='core';badExpert=mode==='expert';closeFailure=mode==='close';guardValid=mode!=='guard';},async close(){closed=true;await rm(work,{recursive:true,force:true});}};
}
test('closed transfer reload opens only after prior model closes, and issues a receipt only after real Core and expert ACKs',async()=>{
 for(const mode of ['ok','core','expert','close','guard']){const f=await reloadFixture();f.setMode(mode);try{
  const p=f.api.reloadClosedNativeExpert(f.engine,f.guard,f.work,f.base,'0123456789ab',f.transfer,{ids:[1,2],messages_sha256:'m'});
  if(mode==='ok'){const receipt=await p;expect(f.api.validateClosedNativeExpertReload(f.engine,f.guard,receipt)).toBe(f.peer.accepted);expect(()=>f.api.validateClosedNativeExpertReload(f.engine,f.guard,{...receipt})).toThrow();expect(f.events).toEqual(['close-old','open-new','actual-core','actual-expert','adopt-source','branch']);}
  else{await expect(p).rejects.toThrow();if(mode==='close')expect(f.events).toEqual(['close-old']);if(mode==='guard')expect(f.events).toHaveLength(0);}
 }finally{await f.close();}}
});
test('actual compiled controller retains final progress without overwriting publication errors',async()=>{
 const source=await readFile(process.env.LUBAN_NATIVE_EXPERT_TEST_BUNDLE??'build/native-evolution/worker.js','utf8'),start=source.indexOf('class NativeLearning {'),end=source.indexOf('\n// ',start);assert(start>0&&end>start);
 const code=source.slice(start,end).replace(/\b(assert|encoded|equal|ledgerParse|hash)\d+\b/g,'$1');
 const C=new Function('assert',code+'\nreturn NativeLearning;')(assert),c=new C(()=>({settings:{mask:(s:string)=>s}}),{});
 c.closed=true;c.execute=async()=>{c.session={status:()=>({state:'validated',error:null,step:24,steps:192,candidate:0,candidate_step:24})};throw Error('real Q4 rejected');};c.begin();await c.wait();
 expect(c.status().error).toContain('real Q4 rejected');expect(c.status().step).toBe(24);expect(c.status().state).toBe('error');
});
test('actual compiled scope closes the worker before files; live-close failure and dead-close failure remain distinct and retryable',async()=>{
 const source=await readFile(process.env.LUBAN_NATIVE_EXPERT_TEST_BUNDLE??'build/native-evolution/worker.js','utf8'),marker=source.indexOf('Scope close is already running'),start=source.lastIndexOf('close: async () => {',marker),end=source.indexOf('\n    });',marker);assert(start>0&&end>start);
 const method=source.slice(start+'close: '.length,end).replace(/\b(assert|held|state|snapshots|source)\d+\b/g,'$1');
 for(const mode of ['live','dead','owner']){const events:string[]=[];let attempt=0,ownerAttempt=0,isClosed=false;
  const held={get isClosed(){return isClosed;},status:{optimizer_step:0,resident_parameters:0},assertLive(){assert(!isClosed);},async restorePrevious(){events.push('restore');},async clearBackbone(){events.push('clear');},async close(){events.push('worker-close');attempt++;if(mode==='live'&&attempt===1)throw Error('worker still alive');isClosed=true;if(mode==='dead'&&attempt===1)throw Error('process exited without normal close ACK');}};
  const state={closed:false,closing:false},closeOwner=async()=>{events.push('owner-close');ownerAttempt++;assert(isClosed,'must not release borrowed FD while native alive');if(mode==='owner'&&ownerAttempt===1)throw Error('FD close unconfirmed');};
  const factory=new Function('assert','held','state','owner',`let closed=false,closing=false,workerClosed=false,resourceState='candidate';const options={previous:{owner:null,assertUnchanged:async()=>{}}},checkFiles=()=>{},snapshots=[],accepted={core:true},coreOwner=null,source={close:async()=>{}},build={close:async()=>{}};return ${method};`);
  const close=factory(assert,held,state,{close:closeOwner});await expect(close()).rejects.toThrow();expect(state.closed).toBe(false);
  if(mode==='live')expect(events).not.toContain('owner-close');if(mode==='dead')expect(isClosed).toBe(true);
  await close();expect(state.closed).toBe(true);expect(events.indexOf('worker-close')).toBeLessThan(events.indexOf('owner-close'));expect(attempt).toBe(mode==='live'?2:1);
 }
});
