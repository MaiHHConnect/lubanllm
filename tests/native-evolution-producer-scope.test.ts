/** Execute the actual product producer entry, stopping before its first model snapshot. */
import {expect,test} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,mkdir,realpath,lstat,readdir,rm} from 'node:fs/promises';
import {join,dirname} from 'node:path';
import {tmpdir} from 'node:os';
import {Memory} from '../backend/memory';
import {patchNativeEvolutionProducerScope} from '../scripts/native-evolution-peer-patches';

test('actual producer rebinds the existing peer before resolving its base on every source scope',async()=>{
 const source=await readFile('.native-migration/research/native-core-product-job-source-v1/native-evolution-producer__producer.after.ts.txt','utf8');
 const patched=patchNativeEvolutionProducerScope(source,(s,a,b)=>{assert.equal(s.split(a).length,2);return s.replace(a,b);});
 const functionBody=patched.slice(patched.indexOf('export async function withNativeEvolutionPreparation'),patched.indexOf('\nasync function prepare(')).replace('export async function','async function');
 const js=new Bun.Transpiler({loader:'ts'}).transformSync(functionBody);
 const root=await realpath(await mkdtemp(join(tmpdir(),'native-producer-scope-'))),memory=new Memory(root);memory.model_identity='a'.repeat(64);
 const events:string[]=[],snapshotBoundary=Error('stop before model snapshot');let active:any=null,bound:any=null;
 const worker={hasInstalledCore:false,assertLive(){assert(active&&bound===active,'Retained peer operation outside canonical Engine source scope');},assertAcceptedPreviousSource(previous:any){this.assertLive();assert.equal(previous,null);},get status(){this.assertLive();return{optimizer_step:0};}};
 const peer={worker,previous:null};
 class NativeEngine{config={identity:memory.model_identity,routing:{store:root}};async withSourceQuiescent(action:any){assert.equal(active,null);const guard={modelLockFd:123,assertQuiescent(){assert.equal(active,guard);}};active=guard;events.push('scope');try{return await action(guard,{captureRouterInitial:async()=>{worker.assertLive();events.push('snapshot-boundary');throw snapshotBoundary;}});}finally{bound=null;active=null;}}}
 class NativeCore450Owner{signal=new AbortController().signal;run(action:any){return action();}}
 const borrowNativeEnginePeer=(_engine:any,guard:any)=>{guard.assertQuiescent();assert(!bound||bound===guard);bound=guard;worker.assertLive();events.push('borrow');return peer;};
 const resolveNativeSnapshotBase=async()=>{worker.assertLive();events.push('base');return{identity:memory.model_identity};};
 const deps={Memory,NativeEngine,NativeCore450Owner,realpath,lstat,readdir,dirname,join,pathIn:join,need:assert,borrowNativeEnginePeer,resolveNativeSnapshotBase};
 const run=new Function('deps',`const {${Object.keys(deps).join(',')}}=deps;${js};return withNativeEvolutionPreparation;`)(deps);
 try{const engine=new NativeEngine();for(let i=0;i<2;i++){const work=join(root,'round-'+i);await mkdir(work,{mode:0o700});events.length=0;await expect(run({engine,memory,work,owners:new NativeCore450Owner(),prefix:{},rng:{},basis:{},execution:{}},()=>{throw Error('must not train');})).rejects.toBe(snapshotBoundary);expect(events).toEqual(['scope','borrow','base','snapshot-boundary']);expect(active).toBeNull();expect(bound).toBeNull();expect(()=>worker.assertLive()).toThrow(/outside canonical/);}}
 finally{memory.close();await rm(root,{recursive:true,force:true});}
});
