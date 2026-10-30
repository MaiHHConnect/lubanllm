import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,readdir,realpath,open} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {writeEvolutionSourceSwapSources,patchEvolutionSourceSwapProcess} from '../scripts/native-evolution-source-swap-patches.ts';
const root=resolve(import.meta.dir,'..');
const replace=(body:string,before:string,after:string)=>{assert.equal(body.split(before).length,2);return body.replace(before,after);};

test('real tiny CPU Core source swap preserves old parameter, owner retry and has no audit files',async()=>{
 const work=await realpath(await mkdtemp(join(tmpdir(),'luban-core-source-cpu-'))),generated=await writeEvolutionSourceSwapSources(root,work);
 const previous=JSON.parse(await readFile(join(root,'.native-migration/research/native-core-retained-source-adopt-v1/CPU-result.json'),'utf8'));
 const original=previous.commands.find((c:any)=>c.argv.includes('-o')).argv as string[];
 const binary=join(work,'source-cpu'),source=join(root,'native/EvolutionSourceSwapCPU.cpp');
 const command=original.map((arg,i)=>arg.endsWith('/CoreSourceCPU.cpp')?source:i===original.indexOf('-o')+1?binary:arg);
 command.push('-DLUBAN_EVOLUTION_CORE_SWAP_HEADER='+JSON.stringify(join(work,'EvolutionCoreSourceSwap.h')));
 const compiled=spawnSync(command[0]!,command.slice(1),{encoding:'utf8',timeout:90000,maxBuffer:8<<20});
 expect(compiled.status,compiled.stderr).toBe(0);
 const actual=spawnSync(binary,['--cpu-only',join(work,'actual')],{encoding:'utf8',timeout:15000});
 expect(actual.status,actual.stderr).toBe(0);
 const [ack,result]=actual.stdout.trim().split('\n').map(value=>JSON.parse(value));
 expect(ack.protocol).toBe('luban-retained-core-source-v2');expect(ack.resident_core_readback).toBe(false);expect(ack.parent_rng_snapshot).toBe(false);
 expect(ack.actual_core_shape).toEqual([16,32]);expect(result).toMatchObject({passed:true,CPU:true,GPU:false,negative_cases:6,cancelled_prepare_restored:true,original_restore:true,extra_audit_artifacts:0});
 expect(result.actual_close_refusal_retry_FD).toBeGreaterThanOrEqual(0);
 expect((await readdir(join(work,'actual'))).sort()).toEqual(['one.raw','two.raw']);
 expect(generated.records).toHaveLength(2);
},120000);

test('actual captured Process rejects forged and invalidated native source ACKs',async()=>{
 const r=join(root,'.native-migration/research'),composition=JSON.parse(await readFile(join(r,'native-core-product-job-source-v1/final-composition.json'),'utf8'));
 const parent=JSON.parse(await readFile(composition.parent.path,'utf8'));
 const source=parent.sources.find((s:any)=>s.path.endsWith('/native-unified-expert/expert-process.ts'));
 const body=patchEvolutionSourceSwapProcess(await readFile(source.snapshot,'utf8'),replace);
 const start=body.indexOf('export type RetainedCoreSourceACK='),end=body.indexOf('\n\nexport class NativeExpertProcess',start);
 const code=body.slice(start,end).replaceAll('export ','');
 const run=new Function('assert',new Bun.Transpiler({loader:'ts'}).transformSync(code+`
 const worker={pid:123,options:{binary_sha256:'bound-binary'},assertLive(){}};
 const cap={pid:123,binary_sha256:'bound-binary',native:{protocol:'luban-retained-core-source-v2'}};
 assert.throws(()=>validateRetainedCoreSourceACK(worker,cap),/actual retained peer issuer/);
 const state={worker,pending:{applied:true,closed:false},closed:false};retainedCoreACKs.set(cap,state);
 validateRetainedCoreSourceACK(worker,cap);state.pending.applied=false;assert.throws(()=>validateRetainedCoreSourceACK(worker,cap));
 state.pending.applied=true;state.pending.closed=true;assert.throws(()=>validateRetainedCoreSourceACK(worker,cap));
 state.pending.committed=true;assert.throws(()=>validateRetainedCoreSourceACK(worker,cap));currentRetainedCoreACKs.set(worker,cap);validateRetainedCoreSourceACK(worker,cap);state.closed=true;assert.throws(()=>validateRetainedCoreSourceACK(worker,cap));
 `));run(assert);expect(true).toBe(true);
});

test('actual apply consumer accepts the real MPS device spelling and rejects invalid ACK fields',async()=>{
 const r=join(root,'.native-migration/research'),composition=JSON.parse(await readFile(join(r,'native-core-product-job-source-v1/final-composition.json'),'utf8'));
 const parent=JSON.parse(await readFile(composition.parent.path,'utf8'));
 const source=parent.sources.find((s:any)=>s.path.endsWith('/native-unified-expert/expert-process.ts'));
 const body=patchEvolutionSourceSwapProcess(await readFile(source.snapshot,'utf8'),replace);
 const declarations=body.slice(body.indexOf('export type RetainedCoreSourceACK='),body.indexOf('\n\nexport class NativeExpertProcess')).replaceAll('export ','');
 const method=body.slice(body.indexOf(' async applyAcceptedCoreSource('),body.indexOf(' async rollbackAcceptedCoreSource('));
 // Actual 2B command-9 ACK from source-swap-round-v1; native device.str() is mps:0.
 const native={protocol:'luban-retained-core-source-v2',pid:79434,device:'mps:0',
  descriptor_sha256:'d6b377cb508e9a10981e731613ee0663242464775e8d4e14b2db1375b9fde305',
  base_identity:'673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df',
  checkpoint_sha256:'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1',
  manifest_sha256:'d8b0ead81edafada4883e6fea37d1750625a682c469b5f70eb760cd2ab0b1237',
  config_sha256:'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4',
  core_version:'core-26440e13b810',actual_core_tensor_sha256:'26440e13b8108aa3cb8c10a8af4e2b34ae7761cdafb34b2391d32fa1bcdf77b6',
  actual_core_bytes:25165824,actual_core_shape:[2048,6144],same_official_storage:true,resident_core_readback:false,parent_rng_snapshot:false,official_graph_reloaded:false,qualification:false};
 const harness=new Function('assert',new Bun.Transpiler({loader:'ts'}).transformSync(declarations+`
 class NativeCore450Owner{owned=[];own(cap,close){this.owned.push({cap,close});}}
 class Consumer{${method}}
 return async(native,device='mps')=>{const worker=new Consumer();worker.pid=79434;worker.options={device,binary_sha256:'bound'};worker.assertLive=()=>{};
 worker.pendingCoreSource={sha256:'${native.descriptor_sha256}',closed:false,applied:false};worker.command=async()=>({response:{core_source_ack:native}});
 const owners=new NativeCore450Owner();const cap=await worker.applyAcceptedCoreSource(owners);await cap.assertUnchanged();return {cap,owners};};
 `))(assert);
 const accepted=await harness(native);expect(accepted.cap.native.device).toBe('mps:0');expect(accepted.owners.owned).toHaveLength(1);
 for(const device of ['mps:1','cpu','cuda:0','mps:00'])await expect(harness({...native,device})).rejects.toThrow('Actual native resident Core source ACK differs');
 await expect(harness(native,'cpu')).rejects.toThrow('Actual native resident Core source ACK differs');
 for(const invalid of [{protocol:'luban-retained-core-source-v1'},{pid:1},{descriptor_sha256:'0'.repeat(64)},{actual_core_bytes:0},{actual_core_shape:[16,32]},{actual_core_tensor_sha256:'bad'},{same_official_storage:false},{resident_core_readback:true},{parent_rng_snapshot:true},{official_graph_reloaded:true},{qualification:true}]){
  await expect(harness({...native,...invalid})).rejects.toThrow('Actual native resident Core source ACK differs');
 }
});

async function reloadHarness(){
 const source=(await readFile(join(root,'backend/native-evolution-reload.ts'),'utf8')).replace(/import[\s\S]*?from\s*'[^']+';/g,'').replaceAll('export ','');
 const state={version:null as string|null,pins:null as string|null,descriptorFailure:false,openFailure:false,partialHandle:null as any,collections:0,confirmFailures:0,stages:0,applies:0,closed:0};
 const previous={id:'expert',flags:{requires_gate:true},owner:{file_sha256:'expert-sha',assertUnchanged:async()=>{}},assertUnchanged:async()=>{assert.equal(state.pins,state.version);}};
 const router={file_sha256:'router-sha'},base={identity:'identity',original_down_sha256:'original',checkpoint_sha256:'checkpoint',model_manifest_sha256:'manifest',config_sha256:'config'};
 const owner={assertUnchanged:async()=>{}},sourceOwner=()=>({runtime:{},assertUnchanged:async()=>{},close:async()=>{state.closed++;}});
 const issued=new WeakSet<object>();let pending:any;
 const worker:any={pid:73,options:{device:'mps',hidden:2048,directory:'/private/tmp',binary_sha256:'binary'},isBranchEnabled:false,
  status:{backbone_resident:true,optimizer_step:0,constructions:1,storage_ids:['a','b','c']},assertLive(){},
  async stageAcceptedCoreSource(){state.stages++;pending={applied:false};},
  async applyAcceptedCoreSource(){state.applies++;pending.applied=true;const ack={core_version:state.version,actual_core_tensor_sha256:state.version?'new-tensor':'original',native:{base_identity:'identity',checkpoint_sha256:'checkpoint',manifest_sha256:'manifest',config_sha256:'config'},assertUnchanged:async()=>{}};issued.add(ack);return ack;},
  async confirmAcceptedCoreSource(_committed:boolean,callback?:()=>void){pending.callback??=callback;if(state.confirmFailures-->0)throw Error('actual close unconfirmed');pending.callback?.();pending=undefined;},
  async settleAcceptedCoreSource(){if(pending)await this.confirmAcceptedCoreSource(true);}
 };
 const peer:any={worker,previous,routerOwner:router,source:sourceOwner(),core:null,owner,build:{binary_sha256:'binary',assertUnchanged:async()=>{}},store:'store',base,serial:0,retiredSources:new Set(),hashes:{},closed:false};
 const engine={config:{routing:{store:'store'},aten:{base},model:'model'}},guard={};peer.guard=guard;
 class Owners{signal=new AbortController().signal;owned:(()=>Promise<unknown>)[]=[];own(_key:unknown,close:()=>Promise<unknown>){this.owned.push(close);}}
 const files=()=>{state.collections++;return({'selected':Buffer.from(state.version??'none'),'router.json':Buffer.from(JSON.stringify({core_version:null,calibrated:true,gates_expert:true,expert_id:'expert'}))});};
 const accepted=()=>({core:state.version?{version:state.version,tensor_sha256:'new-tensor',file_sha256:'file'}:null,expert:{id:'expert',file_sha256:'expert-sha'},router:{id:null,weights_sha256:'router-sha'}});
 const scope={assert,join,NativeCore450Owner:Owners,assertNativeExpertScopeGuard(){},borrowNativeEnginePeer:()=>peer,
  collectLearningModelFiles:files,hash:(v:Buffer)=>v.toString('hex'),equal:(a:unknown,b:unknown)=>JSON.stringify(a)===JSON.stringify(b),validateModelFiles:accepted,readNativeFoundationAdmission:()=>null,
  retainConversationOwner(){},openCachedAccepted:async(...args:any[])=>{if(state.openFailure){const directory=await realpath(await mkdtemp(join(tmpdir(),'luban-core-open-failure-'))),handle=await open(join(directory,'partial'),'wx',0o600);state.partialHandle=handle;args[3](handle,()=>handle.close());throw Error('source open failed after FD registration');}return sourceOwner();},borrowCachedCore:async()=>({close:async()=>{},assertUnchanged:async()=>{}}),openCachedNoneHeld:async()=>sourceOwner(),mkdtemp:async()=>'/private/tmp/unused',
  refreshCapturedAcceptedCoreSource:(cap:unknown)=>{assert.equal(cap,previous);state.pins=state.version;},
  writePeerDescriptor:async()=>{if(state.descriptorFailure)throw Error('descriptor failed');return{path:'bound',sha256:'descriptor'};},
  refreshLearningPeer:async(_e:unknown,_g:unknown,_w:unknown,body:(p:any)=>Promise<unknown>)=>{const value=await body(peer);await previous.assertUnchanged();return value;},
  validateRetainedCoreSourceACK:(_w:unknown,ack:object)=>assert(issued.has(ack)),installedCoreSourceForTransfer:()=>null,
  nativeCheckpointJSON:(b:Buffer)=>JSON.parse(b.toString())};
 const body=new Bun.Transpiler({loader:'ts'}).transformSync(source+'\nreturn {prepareRetainedCoreReload,reloadRetainedCoreSource,settleRetainedCoreReload,validateRetainedCoreReloadReceipt};');
 const api=new Function(...Object.keys(scope),body)(...Object.values(scope));
 return{state,api,peer,engine,guard,owners:new Owners(),prepared:{ids:[1,2],messages_sha256:'messages'},previous,router};
}

test('real reload control flow restores original expert binding after pre-apply failure',async()=>{
 const h=await reloadHarness(),before=await h.api.prepareRetainedCoreReload(h.engine,h.guard,h.owners);
 h.state.version='core-111111111111';h.state.descriptorFailure=true;
 await expect(h.api.reloadRetainedCoreSource(h.engine,h.guard,before,h.prepared,h.owners,'111111111111')).rejects.toThrow('descriptor failed');
 expect(h.state.pins).toBe('core-111111111111');expect(h.state.applies).toBe(0);
 expect(await h.api.settleRetainedCoreReload(before)).toBe(false);
 h.state.version=null;h.state.descriptorFailure=false;
 const restored=await h.api.reloadRetainedCoreSource(h.engine,h.guard,before,h.prepared,h.owners,'111111111111');
 expect(restored.loaded.core_loaded).toBe(false);expect(h.state.pins).toBeNull();expect(h.peer.previous).toBe(h.previous);expect(h.peer.routerOwner).toBe(h.router);expect(h.peer.worker.isBranchEnabled).toBe(false);
});

test('real reload control flow keeps exact owner across native confirmation failure and retry',async()=>{
 const h=await reloadHarness(),before=await h.api.prepareRetainedCoreReload(h.engine,h.guard,h.owners);
 h.state.version='core-111111111111';h.state.confirmFailures=1;
 await expect(h.api.reloadRetainedCoreSource(h.engine,h.guard,before,h.prepared,h.owners,'111111111111')).rejects.toThrow('actual close unconfirmed');
 expect(h.peer.previous).toBe(h.previous);expect(await h.api.settleRetainedCoreReload(before)).toBe(true);
 const receipt=await h.api.reloadRetainedCoreSource(h.engine,h.guard,before,h.prepared,h.owners,'111111111111');
 expect(receipt.loaded.protocol).toBe('luban-retained-core-source-v2');expect(h.state.stages).toBe(1);expect(h.state.applies).toBe(1);expect(h.peer.previous).toBe(h.previous);expect(h.peer.routerOwner).toBe(h.router);expect(h.state.collections).toBe(1);
});


test('partial new source failure keeps its real FD owner until cleanup closes it',async()=>{
 const h=await reloadHarness(),before=await h.api.prepareRetainedCoreReload(h.engine,h.guard,h.owners);
 h.state.version='core-111111111111';h.state.openFailure=true;
 await expect(h.api.reloadRetainedCoreSource(h.engine,h.guard,before,h.prepared,h.owners,'111111111111')).rejects.toThrow('source open failed after FD registration');
 expect(h.state.partialHandle.fd).toBeGreaterThanOrEqual(0);expect(h.state.applies).toBe(0);
 for(const close of h.owners.owned.toReversed())await close();
 expect(h.state.partialHandle.fd).toBe(-1);expect(h.peer.previous).toBe(h.previous);expect(h.peer.routerOwner).toBe(h.router);
});

test('cold publication refuses tokenizer-only readiness or an unconfirmed backbone load',async()=>{
 const h=await reloadHarness();h.peer.worker.status.backbone_resident=false;
 await expect(h.api.prepareRetainedCoreReload(h.engine,h.guard,h.owners)).rejects.toThrow('Cold Core requires original prepared tokens');
 let called=0;h.peer.worker.retainCoreBackbone=async()=>{called++;};
 await expect(h.api.prepareRetainedCoreReload(h.engine,h.guard,h.owners,h.prepared)).rejects.toThrow('actual materialized accepted backbone');
 expect(called).toBe(1);expect(h.state.stages).toBe(0);expect(h.state.applies).toBe(0);
});
