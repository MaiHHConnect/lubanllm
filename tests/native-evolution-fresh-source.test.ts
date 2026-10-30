import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,open,rm} from 'node:fs/promises';
import {fstatSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {patchNativeEvolutionFreshSource} from '../scripts/native-evolution-fresh-source-patches.ts';
const source=await readFile(new URL('../.native-migration/research/native-core-product-job-source-v1/final-source/native-core450-runner__fresh.source',import.meta.url),'utf8');
const replace=(body:string,before:string,after:string)=>{assert.equal(body.split(before).length,2);return body.replace(before,after);};
const liveSource=source.replaceAll('validateNativeRouterEngineSnapshot(engine,guard,r.previous)','validateNativeEvolutionRNG(engine,guard,r.previous)').replaceAll('const previous=r.previous.snapshot;','const previous=r.previous;');
const patched=patchNativeEvolutionFreshSource(liveSource,replace,'held-helper');
const noImports=(body:string)=>body.replace(/import[\s\S]*?from\s*['"][^'"]+['"];?/g,'').replaceAll('export ','');
async function fixture(core:string|null=null,router=true){
 const directory=await mkdtemp(join(tmpdir(),'luban-fresh-source-')),handles:any[]=[];
 const raw=async(name:string,bytes:number)=>{const handle=await open(join(directory,name),'wx+',0o600);await handle.truncate(bytes);handles.push(handle);return{handle,path:join(directory,name),bytes,shape:[bytes],dtype:'u8',sha256:'a'.repeat(64)};};
 const pins:Record<string,string|null>={'active-core.json':core?'physical-core':null,'active.json':null,'active-router.json':router?'physical-router':null};
 if(core)for(const suffix of ['.json','.safetensors','.retention.json'])pins[core+suffix]='bound'+suffix;
 if(router){pins['router-a970b1fb305c.json']='router-meta';pins['router-a970b1fb305c.safetensors']='router-weight';}else{pins['router.json']=null;pins['router.safetensors']=null;}
 const full=Object.fromEntries(Object.entries(pins).filter(([,v])=>v!==null)) as Record<string,string>;if(core)full['evolution-'+core.slice(5)+'.json']='bound-report';
 // Physical B45 remains selected while the admitted training core is None.
 const peer={hashes:full,previous:null,worker:{isBranchEnabled:false,assertAcceptedPreviousSource(){}}};
 const base={identity:'identity',original_down_sha256:'original'},engine={config:{routing:{store:'/store'},aten:{base}}},guard={};
 let live=true,checks=0,initializations=0;const initialState={candidate:await raw('candidate',2)};
 const initial={state:initialState,async assertUnchanged(){assert(live,'held source invalidated');},async close(){await initialState.candidate.handle.close();}};
 const preparation={id:'run',cohort:{records_sha256:'records'},graphBinding:{accepted_tensor_sha256:'original',accepted_core_version:null},async assertUnchanged(){assert(live,'held source invalidated');}};
 const previous={device:'mps',cpu_rng:await raw('cpu',5056),mps_rng:await raw('mps',44),active_files_sha256:full,async assertUnchanged(){assert(live,'held source invalidated');}};
 const binding={scope:'product',device:'mps',dtype:'torch.bfloat16',arguments:{steps:450},core_version:null,expert_id:null,records_sha256:'records',active_files_sha256:pins};
 const args={store:'/store',base,id:'run',binding,directory,previous,preparation};
 const held={peer,pins,accepted:{core:null,expert:null},async assertUnchanged(){checks++;assert(live,'held source invalidated');}};
 const legacy=()=>{throw Error('forbidden repeated full source scan');};
 const scope={assert,join,fstatSync,equal:(a:any,b:any)=>assert.deepEqual(a,b)===undefined,assertNativeExpertScopeGuard(){assert(live,'held source invalidated');},validateNativeEvolutionRNG(){assert(live,'held source invalidated');},borrowNativeEvolutionPreparation:()=>({store:'/store',accepted:{sha256:'original',file_sha256:'bound'},basis:{sha256:'basis'}}),borrowNativeEnginePeer:()=>peer,readNativeEvolutionHeldSource:async()=>{await held.assertUnchanged();return held;},prepareNativeCoreInitialState:async()=>{initializations++;return initial;},collectLearningModelFiles:legacy,validateModelFiles:legacy,readNativeFoundationAdmission:legacy,nativeAcceptedHashes:legacy};
 const create=new Function(...Object.keys(scope),new Bun.Transpiler({loader:'ts'}).transformSync(noImports(patched)+'\nreturn prepareCore450ProductFresh;'))(...Object.values(scope));
 const oldScope={...scope,collectLearningModelFiles:()=>({}),validateModelFiles:()=>held.accepted,nativeAcceptedHashes:()=>pins};
 const oldCreate=new Function(...Object.keys(oldScope),new Bun.Transpiler({loader:'ts'}).transformSync(noImports(liveSource)+'\nreturn prepareCore450ProductFresh;'))(...Object.values(oldScope));
 return{args,engine,guard,peer,pins,held,create,oldCreate,invalidate(){live=false;},counts:()=>({checks,initializations}),async close(){for(const h of handles)if(h.fd!==-1)await h.close();await rm(directory,{recursive:true,force:true});}};
}
test('actual fresh consumer separates physical B45 CAS pins from full source reports while training None',async()=>{
 const h=await fixture('core-b45d965728bb');try{
  expect(h.args.binding.core_version).toBeNull();await expect(h.oldCreate(h.engine,h.guard,h.args)).rejects.toThrow();expect(h.counts().initializations).toBe(0);
  const cap=await h.create(h.engine,h.guard,h.args);await cap.assertUnchanged();expect(h.counts().initializations).toBe(1);expect(h.counts().checks).toBe(3);
  await cap.close();
 }finally{await h.close();}
});
test('actual fresh accepts absent router null keys without equating absence to a changed file',async()=>{
 const h=await fixture(null,false);try{
  const cap=await h.create(h.engine,h.guard,h.args);await cap.assertUnchanged();expect(cap.approval.step).toBe(0);await cap.close();
  h.args.binding.active_files_sha256={...h.pins,'router.json':'unexpected'};
  await expect(h.create(h.engine,h.guard,h.args)).rejects.toThrow();
 }finally{await h.close();}
});
test('actual fresh refuses mismatched RNG source, omitted physical pins and invalidated retained owner',async()=>{
 for(const kind of ['rng','omitted','owner']){
  const h=await fixture('core-b45d965728bb');try{
   if(kind==='rng')h.args.previous.active_files_sha256={...h.peer.hashes,'unrelated':'source'};
   if(kind==='omitted'){h.args.binding.active_files_sha256={...h.pins};delete h.args.binding.active_files_sha256['active-router.json'];}
   if(kind==='owner'){const cap=await h.create(h.engine,h.guard,h.args);h.invalidate();await expect(cap.assertUnchanged()).rejects.toThrow('held source invalidated');}
   else await expect(h.create(h.engine,h.guard,h.args)).rejects.toThrow();
  }finally{await h.close();}
 }
});
