/** One real resident round: keep rollback RNG in memory; expose only RNG consumed by training. */
import assert from 'node:assert/strict';
import {join} from 'node:path';
import {NativeEngine,assertNativeExpertScopeGuard} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {NativeCore450Owner} from '../.native-migration/research/native-core450-owner-controller/owner.ts';
import {openCheckpointRaw,checkpointRawSHA} from '../.native-migration/backend/native-checkpoint.ts';
import {unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
import {restoreNativeEvolutionCore} from '../.native-migration/research/native-unified-core-control/install.ts';
import {equal} from '../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';
type Guard=Parameters<Parameters<NativeEngine['withSourceQuiescent']>[0]>[0];
type Peer=ReturnType<typeof borrowNativeEnginePeer>;
type Raw=Awaited<ReturnType<typeof readRNG>>;
export type NativeEvolutionRNG={pid:number;device:'mps';cpu_rng:Raw;mps_rng:Raw;active_files_sha256:Peer['hashes'];assertUnchanged():Promise<void>;close():Promise<void>};
const rngs=new WeakMap<NativeEvolutionRNG,{engine:NativeEngine;guard:Guard;peer:Peer;closed:boolean}>();
async function readRNG(directory:string,kind:'cpu'|'mps',scope:()=>void){
 const path=join(directory,kind+'_rng.raw'),bytes=kind==='cpu'?5056:44,held=await openCheckpointRaw(path,bytes);
 try{assert.equal(Number(held.stat.size),bytes);return{kind:'tensor' as const,file:kind+'_rng.raw',path,handle:held.handle,stat:held.stat,byte_order:'little' as const,dtype:'u8' as const,shape:[bytes],bytes,sha256:await checkpointRawSHA(held.handle,bytes,'u8'),assertUnchanged:async()=>{scope();await unchangedNativeFile(path,held.handle,held.stat);scope();}};}catch(error){await held.handle.close();throw error;}
}
export function validateNativeEvolutionRNG(engine:NativeEngine,guard:Guard,value:NativeEvolutionRNG){const s=rngs.get(value);assert(s&&!s.closed&&s.engine===engine&&s.guard===guard);assertNativeExpertScopeGuard(engine,guard);const peer=borrowNativeEnginePeer(engine,guard);assert(peer===s.peer&&peer.worker.pid===value.pid&&equal(peer.hashes,value.active_files_sha256));}
export async function captureNativeEvolutionTrainingRNG(engine:NativeEngine,guard:Guard,owners:NativeCore450Owner):Promise<NativeEvolutionRNG>{
 assertNativeExpertScopeGuard(engine,guard);const peer=borrowNativeEnginePeer(engine,guard),name='training-rng-'+ ++peer.serial;
 const files:Raw[]=[];const state={engine,guard,peer,closed:false};const token={};const close=async()=>{if(state.closed)return;const results=await Promise.allSettled(files.map(async r=>{if(r.handle.fd!==-1)await r.handle.close();}));const errors=results.filter((r):r is PromiseRejectedResult=>r.status==='rejected');if(errors.length)throw new AggregateError(errors.map(r=>r.reason),'Training RNG owner close remains pending');state.closed=true;};owners.own(token,close);
 const scope=()=>{assert(!state.closed,'Training RNG owner closed');assertNativeExpertScopeGuard(engine,guard);assert(borrowNativeEnginePeer(engine,guard)===peer);};
 try{await peer.worker.exportEvolutionRNG(name);files.push(await readRNG(join(peer.worker.options.directory,name),'cpu',scope));files.push(await readRNG(join(peer.worker.options.directory,name),'mps',scope));
 const value:NativeEvolutionRNG=Object.freeze({pid:peer.worker.pid,device:'mps',cpu_rng:files[0]!,mps_rng:files[1]!,active_files_sha256:peer.hashes,assertUnchanged:async()=>{validateNativeEvolutionRNG(engine,guard,value);for(const raw of files)await unchangedNativeFile(raw.path,raw.handle,raw.stat);},close});rngs.set(value,state);await value.assertUnchanged();return value;
 }catch(error){try{await owners.release(token);}catch(cleanup){if(error instanceof Error)Object.assign(error,{cleanup_error:cleanup});}throw error;}
}
export type NativeEvolutionAmbient={rng:NativeEvolutionRNG;assertUnchanged():Promise<void>;close():Promise<void>};
const rounds=new WeakMap<NativeEvolutionAmbient,{engine:NativeEngine;guard:Guard;peer:Peer;state:Peer['worker']['status'];restored:boolean;core?:Awaited<ReturnType<typeof restoreNativeEvolutionCore>>;rngRestored?:boolean}>();
export async function beginNativeEvolutionRound(engine:NativeEngine,guard:Guard,owners:NativeCore450Owner):Promise<NativeEvolutionAmbient>{
 assertNativeExpertScopeGuard(engine,guard);const peer=borrowNativeEnginePeer(engine,guard);assert(!peer.worker.hasInstalledCore);const state=peer.worker.status;
 const entry={};let transferred=false,began=false;owners.own(entry,async()=>{if(!transferred&&began){await peer.worker.restoreEvolutionRNG();began=false;}});let rng:NativeEvolutionRNG;
 try{began=true;await peer.worker.beginEvolutionRound();rng=await captureNativeEvolutionTrainingRNG(engine,guard,owners);}catch(error){try{await owners.release(entry);}catch(cleanup){if(error instanceof Error)Object.assign(error,{cleanup_error:cleanup});}throw error;}
 const value:NativeEvolutionAmbient={rng,assertUnchanged:async()=>{assertNativeExpertScopeGuard(engine,guard);assert(borrowNativeEnginePeer(engine,guard)===peer);await rng.assertUnchanged();},close:async()=>{assert(rounds.get(value)?.restored,'Evolution round must restore before closing');await rng.close();}};
 rounds.set(value,{engine,guard,peer,state,restored:false});owners.own(value,()=>value.close());transferred=true;await owners.release(entry);return value;
}
export async function restoreNativeEvolutionRound(engine:NativeEngine,guard:Guard,value:NativeEvolutionAmbient){
 const s=rounds.get(value);assert(s&&s.engine===engine&&s.guard===guard);assertNativeExpertScopeGuard(engine,guard);const peer=borrowNativeEnginePeer(engine,guard);assert(peer===s.peer);
 s.core??=await restoreNativeEvolutionCore(engine,guard);if(!s.rngRestored){await peer.worker.restoreEvolutionRNG();s.rngRestored=true;}
 const current=peer.worker.status;assert(equal(current.storage_ids,s.state.storage_ids)&&current.optimizer_step===s.state.optimizer_step&&current.constructions===s.state.constructions,'Evolution changed retained expert state');s.restored=true;
 return Object.freeze({...s.core,rng_restored_in_memory:true,restored:true});
}
