import {expect,test} from 'bun:test';
import {spawnSync} from 'node:child_process';
import {nativeEvolutionRestoreCoreSource} from '../scripts/native-evolution-ambient-patches';
import assert from 'node:assert/strict';
test('actual core-drop consumer validates ACK and keeps the installation owner through close failure',async()=>{
 const engine={},guard={},owner={handle:{fd:7},async close(){if(++closes===1)throw Error('close');this.handle.fd=-1;}};let closes=0,drops=0;
 const cap={owner,closed:false},installations=new WeakMap<object,any>([[engine,cap]]),restored=new WeakMap<object,any>([[engine,{}]]);
 let response:any={operation:'core-drop',step:0,observation:'current',backbone_resident:true,core_tensor_sha256:'previous'};
 const peer={source:{runtime:{core_raw:{sha256:'previous'}}},base:{original_down_sha256:'none'},worker:{pid:17,hasInstalledCore:false,async dropCoreState(){drops++;return{response};}}};
 const js=new Bun.Transpiler({loader:'ts'}).transformSync(nativeEvolutionRestoreCoreSource.replace('export async function','async function'));
 const restore=new Function('assert','assertNativeExpertScopeGuard','borrowNativeEnginePeer','installations','restored',js+';return restoreNativeEvolutionCore;')(assert,()=>{},()=>peer,installations,restored);
 await expect(restore(engine,guard)).rejects.toThrow('close');expect(installations.get(engine)).toBe(cap);expect(cap.closed).toBe(false);
 const value=await restore(engine,guard);expect(value.core_tensor_sha256).toBe('previous');expect(drops).toBe(2);expect(closes).toBe(2);expect(installations.has(engine)).toBe(false);expect(restored.has(engine)).toBe(false);
 installations.set(engine,{owner,closed:false});response={...response,core_tensor_sha256:'candidate'};await expect(restore(engine,guard)).rejects.toThrow('restoration ACK');expect(installations.has(engine)).toBe(true);
});
test('actual round owner restores in phases, retries failures and emits only training RNG files',()=>{
 const child=spawnSync(process.execPath,['--eval',`
import assert from 'node:assert/strict';import{mock}from'bun:test';import{resolve,join}from'node:path';import{mkdtemp,realpath,mkdir,writeFile,rm,readdir}from'node:fs/promises';import{tmpdir}from'node:os';
const old='.native-migration/research/',stub=(path,value)=>mock.module(resolve(path),()=>value);
let peer,coreCalls=0,rngCalls=0,coreFailures=0,rngFailures=0,exportFailures=0,statusWrong=false;const guard={assertQuiescent(){}};
stub(old+'native-unified-expert/native-engine.ts',{NativeEngine:class{},assertNativeExpertScopeGuard:(_e,g)=>assert.equal(g,guard)});
stub(old+'native-unified-peer/engine-peer.ts',{borrowNativeEnginePeer:()=>peer});
stub(old+'native-unified-core-control/install.ts',{restoreNativeEvolutionCore:async()=>{coreCalls++;if(coreFailures-->0)throw Error('core');return{actual_source_restored:true,core_tensor_sha256:'original'};}});
const{NativeCore450Owner}=await import(resolve(old+'native-core450-owner-controller/owner.ts'));
const{beginNativeEvolutionRound,restoreNativeEvolutionRound}=await import(resolve('backend/native-evolution-ambient.ts'));
for(const scenario of ['success','core','rng','status','entry-failure']){
 const directory=await realpath(await mkdtemp(join(tmpdir(),'round-owner-'))),owners=new NativeCore450Owner(),engine={};coreCalls=rngCalls=0;coreFailures=scenario==='core'?1:0;rngFailures=scenario==='rng'||scenario==='entry-failure'?1:0;exportFailures=scenario==='entry-failure'?1:0;statusWrong=false;
 peer={hashes:{'active-core.json':'physical'},serial:0,worker:{pid:17,hasInstalledCore:false,options:{directory},get status(){return{storage_ids:[statusWrong?'wrong':'retained'],optimizer_step:0,constructions:1};},async beginEvolutionRound(){},async restoreEvolutionRNG(){rngCalls++;if(rngFailures-->0)throw Error('rng');},async exportEvolutionRNG(name){if(exportFailures-->0)throw Error('export');const d=join(directory,name);await mkdir(d);await writeFile(join(d,'cpu_rng.raw'),Buffer.alloc(5056));await writeFile(join(d,'mps_rng.raw'),Buffer.alloc(44));}}};
 try{
  if(scenario==='entry-failure'){await assert.rejects(beginNativeEvolutionRound(engine,guard,owners),/export/);assert.equal(owners.status().retained_owners,1);await owners.close();assert.equal(rngCalls,2);continue;}
  const cap=await beginNativeEvolutionRound(engine,guard,owners);assert.deepEqual((await readdir(join(directory,'training-rng-1'))).sort(),['cpu_rng.raw','mps_rng.raw']);
  if(scenario==='status')statusWrong=true;
  if(scenario!=='success')await assert.rejects(restoreNativeEvolutionRound(engine,guard,cap));statusWrong=false;
  const receipt=await restoreNativeEvolutionRound(engine,guard,cap);assert(receipt.restored&&receipt.rng_restored_in_memory);
  assert.equal(coreCalls,scenario==='core'?2:1);assert.equal(rngCalls,scenario==='rng'?2:1);
  const close=cap.rng.cpu_rng.handle.close.bind(cap.rng.cpu_rng.handle);let failed=false;cap.rng.cpu_rng.handle.close=async()=>{if(!failed){failed=true;throw Error('fd close');}return close();};
  await assert.rejects(owners.close());await owners.close();assert.equal(owners.status().retained_owners,0);
 }finally{await rm(directory,{recursive:true,force:true});}
}
console.log('five actual round lifecycle scenarios passed');
`],{cwd:process.cwd(),encoding:'utf8',timeout:30000});
 expect(child.status,child.stderr).toBe(0);expect(child.stdout).toContain('five actual round lifecycle scenarios passed');
});
