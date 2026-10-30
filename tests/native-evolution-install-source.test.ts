import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,writeFile,open,realpath,readdir,chmod,stat,rename,unlink} from 'node:fs/promises';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {heldFixture} from './native-evolution-held-source.test.ts';
import {writeNativeModelTensors} from '../.native-migration/backend/native-model-write.ts';
import {readNativeModelTensors} from '../.native-migration/backend/native-model-tensor.ts';
import {checkpointRawSHA} from '../.native-migration/backend/native-checkpoint.ts';
import {patchNativeEvolutionInstallOwnership} from '../scripts/native-evolution-cancel-cleanup-patches.ts';
import {patchNativeEvolutionInstallFile} from '../scripts/native-evolution-install-file-patches.ts';
import {retainNativeEvolutionInstallFile} from '../backend/native-evolution-install-file.ts';
const workerCode=await readFile(new URL('../build/native-evolution/worker.js',import.meta.url),'utf8');
const normalize=(s:string)=>s.replace(/\b(assert|equal|join|STATE|fileSHA)\d+\b/g,'$1');
function fn(name:string){const start=workerCode.search(new RegExp('(?:async )?function '+name+'\\(')),end=workerCode.indexOf('\n}',start);assert(start>=0&&end>start);return normalize(workerCode.slice(start,end+2));}
const STATE=['candidate','master','exp_avg','exp_avg_sq','adam_step','reference','basis'];
const hash=(x:Buffer)=>createHash('sha256').update(x).digest('hex');
const equal=(a:any,b:any)=>{try{assert.deepEqual(a,b);return true;}catch{return false;}};
async function fixture(expert=true,installCode=fn('installCoreCheckpoint')){
 const h=await heldFixture({core:true,expert,router:true}),directory=await realpath(h.directory),handles:any[]=[];let writes=0,loads=0,rngImports=0;
 const raw=async(name:string,dtype:string,bytes:number,shape:number[])=>{const path=join(directory,name),data=Buffer.alloc(bytes);await writeFile(path,data,{mode:0o600});const handle=await open(path,'r');handles.push(handle);return{path,handle,dtype,bytes,shape,sha256:hash(data)};};
 const state:any={};for(const key of STATE)state[key]=await raw(key,key==='candidate'?'bf16':'f32',key==='candidate'?4:8,[2]);
 const cpu_rng=await raw('cpu_rng','u8',5056,[5056]),mps_rng=await raw('mps_rng','u8',44,[44]);
 // Match the real official None runtime: no core_label property at all.
 delete h.peer.source.runtime.core_label;h.peer.base={...h.peer.base,core_parameter:'model.down',original_down_sha256:state.candidate.sha256};h.peer.owner.tensors={'model.down':{sha256:state.candidate.sha256}};h.peer.serial=0;h.peer.features=new Set();
 h.peer.source.assertUnchanged=()=>{throw Error('forbidden source re-scan');};
 const worker:any=Object.assign(h.peer.worker,{pid:73,supportsCoreResource:true,isClosed:false,options:{device:'mps',hidden:2048,directory},async loadCoreOwner(_owner:any,step:number,cap:any){loads++;return{response:{operation:'core-current',step,core_tensor_sha256:cap.state.candidate.sha256}};},async importCoreRNG(cpu:any,mps:any){assert.equal(cpu,cpu_rng);assert.equal(mps,mps_rng);rngImports++;}});
 const held=await h.read(h.engine,h.guard),cap={approval:{identity:h.peer.base.identity,step:0,scope:'core450-fresh-state-only'},binding:{identity:h.peer.base.identity,core_version:null,expert_id:held.accepted.expert?.id??null,active_files_sha256:held.pins},state,cpu_rng,mps_rng,assertUnchanged:()=>held.assertUnchanged()};
 const installations=new WeakMap(),receipts=new WeakMap(),restored=new WeakMap();
 const scope={assert,equal,join,STATE,fileSHA:checkpointRawSHA,retainNativeEvolutionInstallFile,readNativeEvolutionHeldSource:h.read,assertObservationMutationAllowed(){},assertNativeExpertScopeGuard(_e:any,g:any){assert.equal(g,h.guard);},validate(_e:any,c:any){assert.equal(c,cap);},borrowNativeEnginePeer:()=>h.peer,borrowCoreProcess:(w:any)=>w,writeNativeModelTensors:async(r:any)=>{writes++;return writeNativeModelTensors(r);},readNativeModelTensors,installations,receipts,restored};
 const functions=new Function(...Object.keys(scope),fn('assertCoreInstalled')+'\n'+installCode+'\nreturn{installCoreCheckpoint,assertCoreInstalled};')(...Object.values(scope));
 return{...h,cap,held,worker,functions,counts:()=>({writes,loads,rngImports}),async close(){worker.isClosed=true;for(const f of h.peer.features)await f.close();for(const handle of handles)if(handle.fd>=0)await handle.close();await h.close();}};
}
test('actual install consumer accepts official undefined label as effective None and retains physical B45/expert pins',async()=>{
 for(const expert of [true,false]){const h=await fixture(expert);try{expect(Object.hasOwn(h.peer.source.runtime,'core_label')).toBe(false);const ack=await h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap);h.functions.assertCoreInstalled(h.engine,h.cap,ack);expect(ack.core_tensor_sha256).toBe(h.cap.state.candidate.sha256);expect(h.counts()).toEqual({writes:1,loads:1,rngImports:1});expect(h.cap.binding.active_files_sha256).toEqual(h.nativeAcceptedHashes(h.directory));}finally{await h.close();}}
});

function correctedInstaller(){
 let actual=fn('installCoreCheckpoint');
 if(actual.includes('installationAttempted'))return actual;
 // Exercise the exact build overlay against the actual compiled function (whitespace only).
 actual='export '+actual.replace('let installed = false;','let installed=false;').replace('const loaded = await worker.loadCoreOwner(owner, cap.approval.step, cap);','const loaded=await worker.loadCoreOwner(owner,cap.approval.step,cap);').replace('installed = true;','installed=true;');
 return patchNativeEvolutionInstallOwnership(actual+'\nexport async function coreConstraintResidual(){}',(b,a,z)=>{assert.equal(b.split(a).length,2);return b.replace(a,z);}).split('\nexport async function coreConstraintResidual')[0]!.replace('export ','');
}
test('actual installer retains a natively acquired owner when cold ACK is invalid or load throws after takeover',async()=>{
 for(const failure of ['cold-ack','after-takeover']){
  const h=await fixture(true,correctedInstaller());try{
   h.worker.loadCoreOwner=async(owner:any,step:number)=>{h.worker.coreSourceOwner=owner;if(failure==='after-takeover')throw Error('native ACK lost after ownership transfer');return{response:{operation:'core-current',step,core_tensor_sha256:null,backbone_resident:false}};};
   await expect(h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap)).rejects.toThrow(failure==='cold-ack'?'installation ACK differs':'ACK lost');
   const held=h.worker.coreSourceOwner;expect(held.handle.fd).toBeGreaterThanOrEqual(0);await expect(held.assertUnchanged()).resolves.toBeUndefined();
   for(const feature of h.peer.features)await expect(feature.close()).rejects.toThrow();
   expect(held.handle.fd).toBeGreaterThanOrEqual(0);
   h.worker.isClosed=true;for(const feature of h.peer.features)await feature.close();expect(held.handle.fd).toBe(-1);
  }finally{await h.close();}
 }
});
test('shared cold-backbone preparation uses the actual retained load ACK, performs no generation and reuses a warm model',async()=>{
 const reload=await readFile(new URL('../backend/native-evolution-reload.ts',import.meta.url),'utf8'),start=reload.indexOf('export async function retainNativeEvolutionBackbone('),end=reload.indexOf('\n/** Hold the original expert',start);
 assert(start>=0&&end>start);const methodStart=workerCode.indexOf('  async retainCoreBackbone('),methodEnd=workerCode.indexOf('  async applyAcceptedCoreSource(',methodStart);assert(methodStart>=0&&methodEnd>methodStart);
 const Consumer=new Function('assert','return class{'+normalize(workerCode.slice(methodStart,methodEnd))+'}')(assert);
 for(const invalid of [null,'cold','tensor']){
  let commands=0,descriptors=0;const sha='a'.repeat(64),worker:any=new Consumer();Object.assign(worker,{status:{backbone_resident:false},options:{directory:'/private/work'},assertLive(){},hasInstalledCore:false,
   async command(line:string){commands++;assert(line.startsWith('retain-backbone '));const response={backbone_resident:invalid!=='cold',retained_core_tensor_sha256:invalid==='tensor'?'b'.repeat(64):sha};worker.status=response;return{response};},
   generate(){throw Error('resume cannot regenerate baseline');},seedFresh(){throw Error('resume cannot reseed');}});
  const peer={worker,source:{runtime:{}},owner:{},core:null,base:{original_down_sha256:sha},serial:0},engine={},guard={};
  const retain=new Function('assert','assertNativeExpertScopeGuard','borrowNativeEnginePeer','writePeerDescriptor',new Bun.Transpiler({loader:'ts'}).transformSync(reload.slice(start,end).replace('export ','')+'\nreturn retainNativeEvolutionBackbone;'))(assert,(e:any,g:any)=>{assert.equal(e,engine);assert.equal(g,guard);},()=>peer,async(input:any)=>{descriptors++;assert.deepEqual(input.prompt_ids,[1,2]);assert.equal(input.core_version,null);return{path:'/private/work/original.tsv',sha256:sha};});
  const prepared={ids:[1,2],messages_sha256:sha};
  if(invalid)await expect(retain(engine,guard,prepared)).rejects.toThrow('not materialized');
  else{await retain(engine,guard,prepared);expect(worker.status.backbone_resident).toBe(true);await retain(engine,guard,prepared);}
  expect(commands).toBe(1);expect(descriptors).toBe(1);
 }
});
test('actual install rejects changed physical map, wrong effective version/expert and changed held source before writing',async()=>{
 for(const change of ['pins','core','expert','source']){const h=await fixture();try{if(change==='pins')h.cap.binding.active_files_sha256={...h.cap.binding.active_files_sha256,'active-router.json':null};if(change==='core')h.cap.binding.core_version='core-b45d965728bb' as any;if(change==='expert')h.cap.binding.expert_id='rival';if(change==='source')await writeFile(h.peer.owner.path,'changed');await expect(h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap)).rejects.toThrow();expect(h.counts()).toEqual({writes:0,loads:0,rngImports:0});}finally{await h.close();}}
});

function reclaimedInstaller(){
 let actual=correctedInstaller();if(actual.includes('retainNativeEvolutionInstallFile'))return actual;
 actual='export '+actual.replace('let installationAttempted = false;','let installationAttempted=false;').replace(/if \(owner.handle.fd === -1\)\s*return;\s*assert\(peer.worker.isClosed \|\| !installationAttempted,/, 'if(owner.handle.fd===-1)return;assert(peer.worker.isClosed||!installationAttempted,');
 return patchNativeEvolutionInstallFile(actual+'\nexport async function coreConstraintResidual(){}',(b,a,z)=>{assert.equal(b.split(a).length,2);return b.replace(a,z);}).split('\nexport async function coreConstraintResidual')[0]!.replace('export ','');
}
const installFiles=async(directory:string)=>(await readdir(directory)).filter(name=>/^core-install-.*\.partial$/.test(name));
test('actual compiled repeated installs reclaim only superseded copies after native ACK',async()=>{
 const h=await fixture(true,reclaimedInstaller());let previous:any;const baseLoad=h.worker.loadCoreOwner;
 try{
  h.worker.loadCoreOwner=async(owner:any,step:number,cap:any)=>{
   if(previous){expect(previous.handle.fd).toBeGreaterThanOrEqual(0);expect((await stat(previous.path)).size).toBeGreaterThan(0);}
   expect((await installFiles(h.directory)).length).toBe(previous?2:1);
   const result=await baseLoad(owner,step,cap);h.worker.coreSourceOwner=owner;return result;
  };
  let bytes=0;
  for(let i=0;i<8;i++){
   await h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap);
   if(previous){expect(previous.handle.fd).toBe(-1);await expect(stat(previous.path)).rejects.toMatchObject({code:'ENOENT'});}
   previous=h.worker.coreSourceOwner;
   const names=await installFiles(h.directory);expect(names.length).toBe(1);
   const size=(await stat(join(h.directory,names[0]!))).size;if(i===0)bytes=size;else expect(size).toBe(bytes);
  }
  h.worker.isClosed=true;for(const feature of h.peer.features)await feature.close();expect(await installFiles(h.directory)).toEqual([]);
 }finally{await h.close();}
});
test('invalid installation ACK keeps both live copies until actual worker closure',async()=>{
 const h=await fixture(true,reclaimedInstaller());const baseLoad=h.worker.loadCoreOwner;let old:any;
 try{
  h.worker.loadCoreOwner=async(owner:any,step:number,cap:any)=>{h.worker.coreSourceOwner=owner;return baseLoad(owner,step,cap);};
  await h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap);old=h.worker.coreSourceOwner;
  h.worker.loadCoreOwner=async(owner:any,step:number)=>{h.worker.coreSourceOwner=owner;return{response:{operation:'core-current',step,core_tensor_sha256:null}};};
  await expect(h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap)).rejects.toThrow('installation ACK differs');
  expect(old.handle.fd).toBeGreaterThanOrEqual(0);expect(h.worker.coreSourceOwner.handle.fd).toBeGreaterThanOrEqual(0);expect((await installFiles(h.directory)).length).toBe(2);
  for(const feature of h.peer.features)await expect(feature.close()).rejects.toThrow();
  expect((await installFiles(h.directory)).length).toBe(2);
  h.worker.isClosed=true;for(const feature of h.peer.features)await feature.close();expect(await installFiles(h.directory)).toEqual([]);
 }finally{await h.close();}
});
test('actual installation cleanup retries FD-close and unlink failures without removing live bytes',async()=>{
 for(const failure of ['close','unlink']){
  const h=await fixture(true,reclaimedInstaller());const baseLoad=h.worker.loadCoreOwner;
  try{
   h.worker.loadCoreOwner=async(owner:any,step:number,cap:any)=>{h.worker.coreSourceOwner=owner;return baseLoad(owner,step,cap);};
   await h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap);const owner=h.worker.coreSourceOwner;h.worker.isClosed=true;
   if(failure==='close'){const close=owner.handle.close.bind(owner.handle);let first=true;owner.handle.close=async()=>{if(first){first=false;throw Error('close failed');}return close();};}
   else await chmod(h.directory,0o500);
   const feature=[...h.peer.features][0] as any;
   await expect(feature.close()).rejects.toThrow();expect((await stat(owner.path)).size).toBeGreaterThan(0);
   if(failure==='close')expect(owner.handle.fd).toBeGreaterThanOrEqual(0);else {expect(owner.handle.fd).toBe(-1);await chmod(h.directory,0o700);}
   await feature.close();expect(owner.handle.fd).toBe(-1);expect(await installFiles(h.directory)).toEqual([]);
  }finally{await chmod(h.directory,0o700);await h.close();}
 }
});
test('installation cleanup refuses a replacement path and can retry the original owned inode',async()=>{
 const h=await fixture(true,reclaimedInstaller()),baseLoad=h.worker.loadCoreOwner;
 try{
  h.worker.loadCoreOwner=async(owner:any,step:number,cap:any)=>{h.worker.coreSourceOwner=owner;return baseLoad(owner,step,cap);};
  await h.functions.installCoreCheckpoint(h.engine,h.guard,h.cap);const owner=h.worker.coreSourceOwner,held=owner.path+'.held';h.worker.isClosed=true;
  await rename(owner.path,held);await writeFile(owner.path,'foreign');
  const feature=[...h.peer.features][0] as any;await expect(feature.close()).rejects.toThrow('no longer names its owned file');expect(await readFile(owner.path,'utf8')).toBe('foreign');
  await unlink(owner.path);await rename(held,owner.path);await feature.close();expect(await installFiles(h.directory)).toEqual([]);
 }finally{await h.close();}
});
