/** Actual retained Core source transition. Disk selection is performed by the existing journal/snapshot writer. */
import assert from'node:assert/strict';import{mkdtemp}from'node:fs/promises';import{join}from'node:path';
import{NativeEngine,assertNativeExpertScopeGuard,type NativeShallowPreparation}from'../.native-migration/research/native-unified-expert/native-engine.ts';
import type{SnapshotGuard,NativeReloadReceipt}from'../.native-migration/research/native-unified-expert/class-closure/native-product-runtime/backend/evolution-snapshots.ts';
import{NativeCore450Owner}from'../.native-migration/research/native-core450-owner-controller/owner.ts';
import{validateRetainedCoreSourceACK,type RetainedCoreSourceACK}from'../.native-migration/research/native-unified-expert/expert-process.ts';
import{borrowNativeEnginePeer,type NativeRouterEngineSnapshot}from'../.native-migration/research/native-unified-peer/engine-peer.ts';
import{writePeerDescriptor}from'../.native-migration/research/native-unified-peer/descriptor.ts';
import{collectLearningModelFiles}from'../.native-migration/research/native-learning-product/model-files.ts';import{validateModelFiles,equal,hash,type SnapshotBase}from'../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';import{readNativeFoundationAdmission}from'../.native-migration/research/native-product-runtime/backend/native-foundation-admission.ts';
import{openCachedAccepted,borrowCachedCore}from'../.native-migration/research/native-unified-expert/source.ts';import{openCachedNoneHeld}from'../.native-migration/research/native-unified-expert/none-source.ts';import{retainConversationOwner}from'../.native-migration/research/native-unified-expert/core.ts';
import{refreshCapturedAcceptedCoreSource}from'../.native-migration/research/native-unified-expert/accepted-previous.ts';import{nativeCheckpointJSON}from'../.native-migration/research/native-product-runtime/backend/native-checkpoint.ts';
import{installedCoreSourceForTransfer}from'../.native-migration/research/native-unified-core-control/install.ts';
type Peer=ReturnType<typeof borrowNativeEnginePeer>;
type Before={engine:NativeEngine;guard:SnapshotGuard;peer:Peer;previous:Peer['previous'];router:Peer['routerOwner'];status:Record<string,any>;branch:boolean;closed:boolean;used:boolean;cleanup?:boolean;retireInstalled?:()=>void;reload?:{id:string;prepared:NativeShallowPreparation;owners:NativeCore450Owner;resume():Promise<NativeReloadReceipt>}};
export type RetainedCoreReloadBefore={pid:number;close():Promise<void>};
const preparations=new WeakMap<RetainedCoreReloadBefore,Before>();
const receipts=new WeakMap<NativeReloadReceipt,{engine:NativeEngine;guard:SnapshotGuard;peer:Peer;ack:RetainedCoreSourceACK;base:SnapshotBase;hashes:Readonly<Record<string,string>>}>();
/** Reuse the real cold-source load for publication and checkpoint resume; no generation or seed. */
export async function retainNativeEvolutionBackbone(engine:NativeEngine,guard:SnapshotGuard,prepared?:NativeShallowPreparation){
 assertNativeExpertScopeGuard(engine,guard);const peer=borrowNativeEnginePeer(engine,guard),worker=peer.worker;
 if(worker.status.backbone_resident!==true){
  assert(prepared&&prepared.ids.length>=2&&prepared.ids.length<=32768&&!worker.hasInstalledCore,'Cold Core requires original prepared tokens and the accepted source');
  const descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:'core-retain-descriptor-'+ ++peer.serial,work:worker.options.directory,prompt_ids:[...prepared.ids],messages_sha256:prepared.messages_sha256,max_tokens:1,raw_steps_limit:0,core_version:peer.source.runtime.core_label??null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
  await worker.retainCoreBackbone({...descriptor,core_tensor_sha256:peer.source.runtime.core_raw?.sha256??peer.base.original_down_sha256});
  assert(worker.status.backbone_resident===true,'Core installation requires an actual materialized accepted backbone');
 }
}
/** Hold the original expert/router references; publication changes only the Core source. */
export async function prepareRetainedCoreReload(engine:NativeEngine,guard:SnapshotGuard,owners:NativeCore450Owner,prepared?:NativeShallowPreparation):Promise<RetainedCoreReloadBefore>{
 assertNativeExpertScopeGuard(engine,guard);assert(owners instanceof NativeCore450Owner);const peer=borrowNativeEnginePeer(engine,guard),worker=peer.worker;
 assert(worker.status.optimizer_step===0,'Core adoption cannot discard live expert Adam');await peer.source.assertUnchanged();await peer.owner.assertUnchanged();await peer.previous?.owner.assertUnchanged();
 await retainNativeEvolutionBackbone(engine,guard,prepared);
 const installed=installedCoreSourceForTransfer(engine,guard);if(installed)await worker.transferInstalledCoreSource(installed.cap,installed.owner);
 const state:Before={engine,guard,peer,previous:peer.previous,router:peer.routerOwner,status:worker.status,branch:worker.isBranchEnabled,closed:false,used:false,retireInstalled:installed?.retire};
 const close=async()=>{if(state.closed)return;await worker.settleAcceptedCoreSource();state.closed=true;};
 const cap=Object.freeze({pid:worker.pid,close});owners.own(cap,close);preparations.set(cap,state);return cap;
}
/** Synchronous gate used inside the existing SQLite CAS transaction. Plain receipt JSON cannot mint it. */
export function validateRetainedCoreReloadReceipt(receipt:NativeReloadReceipt,guard:SnapshotGuard,base:SnapshotBase){
 const s=receipts.get(receipt);assert(s&&s.guard===guard&&s.peer.guard===guard&&!s.peer.closed&&equal(s.base,base),'Retained receipt needs the actual same Engine/source callback');assertNativeExpertScopeGuard(s.engine,guard);validateRetainedCoreSourceACK(s.peer.worker,s.ack);
 assert(receipt.loaded.protocol==='luban-retained-core-source-v2'&&receipt.worker_pid===s.peer.worker.pid&&receipt.engine_binary_sha256===s.peer.worker.options.binary_sha256&&s.peer.build.binary_sha256===receipt.engine_binary_sha256&&s.peer.hashes===s.hashes,'Retained receipt native binary/current complete source changed');return true;
}
/** Apply only after real disk adoption/restore has selected an admitted complete accepted source. */
export async function reloadRetainedCoreSource(engine:NativeEngine,guard:SnapshotGuard,before:RetainedCoreReloadBefore,prepared:NativeShallowPreparation,owners:NativeCore450Owner,id:string):Promise<NativeReloadReceipt>{
 assertNativeExpertScopeGuard(engine,guard);assert(owners instanceof NativeCore450Owner&&/^[a-f0-9]{12}$/.test(id));const s=preparations.get(before);assert(s&&!s.closed&&s.engine===engine&&s.guard===guard,'Reload needs actual same-scope pre-selection transfer');if(s.reload){assert(s.reload.id===id&&s.reload.prepared===prepared&&s.reload.owners===owners,'Resume requires this exact actual source attempt');return s.reload.resume();}assert(!s.used);
 if(!s.cleanup)owners.signal.throwIfAborted();const worker=s.peer.worker;worker.assertLive();assert(worker.options.device==='mps'&&worker.options.hidden===2048&&s.peer.previous===s.previous&&s.peer.routerOwner===s.router);
 const store=s.peer.store,base=s.peer.base,files=collectLearningModelFiles(store),pins=Object.freeze(Object.fromEntries(Object.entries(files).map(([k,v])=>[k,hash(v)]))),accepted=validateModelFiles(files,base,readNativeFoundationAdmission(store,files));
 assert(engine.config.routing?.store===store&&engine.config.aten&&equal(engine.config.aten.base,base));assert(prepared.ids.length>=2&&prepared.ids.length<=512,'Source ABI requires the entire original prepared prompt; truncation is forbidden');
 let nextSource:any,nextCore:any,transferred=false;const previous=s.previous;
 const sourceToken={};owners.own(sourceToken,async()=>{await worker.settleAcceptedCoreSource();if(transferred)return;await nextCore?.close();await nextSource?.close();});
 const own=(owner:object,close:()=>Promise<void>)=>{owners.own(owner,async()=>{await worker.settleAcceptedCoreSource();if(!transferred)await close();});};
 retainConversationOwner(s.peer.source.runtime,s.peer.owner);await s.peer.owner.assertUnchanged();
 // The adopted source lives with the peer, beyond this round's cancelled/closed owner.
 if(accepted.core){nextSource=await openCachedAccepted(base,undefined,{model:engine.config.model,store,expected:accepted.core,directory:await mkdtemp(join(worker.options.directory,'accepted-source-'))},own);nextCore=await borrowCachedCore(nextSource.runtime,own);}else nextSource=await openCachedNoneHeld(base,engine.config.model,s.peer.owner);
 await nextSource.assertUnchanged();await nextCore?.assertUnchanged();
 assert((accepted.expert?.id??null)===(previous?.id??null),'Core-only reload must preserve original expert identity');
 if(previous){assert(accepted.expert?.file_sha256===previous.owner.file_sha256);refreshCapturedAcceptedCoreSource(previous,files);}
 if(accepted.router)assert(s.router?.file_sha256===accepted.router.weights_sha256,'Core-only reload must preserve original router owner');else assert(s.router===null);
 const descriptor=await writePeerDescriptor({runtime:nextSource.runtime,guard,owner:s.peer.owner,core_owner:nextCore,request_id:'core-source-descriptor-'+ ++s.peer.serial,work:worker.options.directory,prompt_ids:[...prepared.ids],messages_sha256:prepared.messages_sha256,max_tokens:1,raw_steps_limit:0,core_version:accepted.core?.version??null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
 let ack:RetainedCoreSourceACK|undefined,receipt:NativeReloadReceipt|undefined,hostTransferred=false;
 const oldSource=s.peer.source,oldCore=s.peer.core,retired=Object.freeze({close:async()=>{await oldCore?.close();await oldSource.close();}});
 const resume=async()=>{if(receipt){validateRetainedCoreReloadReceipt(receipt,guard,base);return receipt;}
 await worker.settleAcceptedCoreSource();
 {const peer=s.peer;
  assert(peer===s.peer&&peer.guard===guard);await peer.owner.assertUnchanged();await peer.build.assertUnchanged();
  if(!hostTransferred){await worker.stageAcceptedCoreSource({...descriptor,name:'core-source-native-'+ ++peer.serial},owners);ack=await worker.applyAcceptedCoreSource(owners);await ack.assertUnchanged();
  assert(ack.core_version===(accepted.core?.version??null)&&ack.actual_core_tensor_sha256===(accepted.core?.tensor_sha256??base.original_down_sha256)&&ack.native.base_identity===base.identity&&ack.native.checkpoint_sha256===base.checkpoint_sha256&&ack.native.manifest_sha256===base.model_manifest_sha256&&ack.native.config_sha256===base.config_sha256);
  assert.deepEqual(worker.status.storage_ids,s.status.storage_ids,'Core source swap changed resident expert storage');assert.equal(worker.status.constructions,s.status.constructions);assert.equal(worker.status.optimizer_step,0);assert.equal(worker.isBranchEnabled,s.branch);await nextSource.assertUnchanged();await nextCore?.assertUnchanged();
  await worker.confirmAcceptedCoreSource(true,()=>{
   // Also runs on actual same-owner confirm retry. The original expert/router owners remain live.
   if(hostTransferred)return;peer.retiredSources.add(retired);peer.source=nextSource;peer.core=nextCore;peer.hashes=pins;s.retireInstalled?.();transferred=true;s.used=true;hostTransferred=true;
  });}
  assert(peer.previous===previous&&peer.routerOwner===s.router,'Core publication replaced original expert/router reference');
  peer.gate=previous?.flags.requires_gate??false;if(accepted.router){const name=accepted.router.id??'router',meta=nativeCheckpointJSON(files[name+'.json']!) as Record<string,unknown>;assert(peer.routerOwner?.file_sha256===accepted.router.weights_sha256,'Core source refresh cannot replace resident Router weights');if((meta.core_version??null)!==(accepted.core?.version??null))meta.calibrated=false;peer.meta=Object.freeze(structuredClone(meta));if(meta.gates_expert&&meta.expert_id===previous?.id)peer.gate=true;}else{assert(peer.routerOwner===null);peer.meta=Object.freeze({calibrated:false,samples:0});}
 }
 assert(ack);await ack.assertUnchanged();await nextSource.assertUnchanged();
 const loaded=Object.freeze({protocol:'luban-retained-core-source-v2',base_identity:base.identity,manifest_sha256:base.model_manifest_sha256,core_loaded:Boolean(accepted.core),...(accepted.core?{core_file_sha256:accepted.core.file_sha256,core_tensor_sha256:accepted.core.tensor_sha256}:{})});
 receipt=Object.freeze({restore_id:id,worker_pid:worker.pid,engine_binary_sha256:worker.options.binary_sha256,loaded,expert:accepted.expert?Object.freeze(structuredClone(accepted.expert)):null,router:accepted.router?Object.freeze(structuredClone(accepted.router)):null});receipts.set(receipt,{engine,guard,peer:s.peer,ack,base,hashes:pins});validateRetainedCoreReloadReceipt(receipt,guard,base);return receipt;};
 s.reload={id,prepared,owners,resume};return resume();
}
/** Settle only this real pre-selection attempt before an existing journal rollback CAS. */
export async function settleRetainedCoreReload(before:RetainedCoreReloadBefore){const s=preparations.get(before);assert(s&&!s.closed);assertNativeExpertScopeGuard(s.engine,s.guard);await s.peer.worker.settleAcceptedCoreSource();s.cleanup=true;if(s.used){assert(s.reload);await s.reload.resume();}else s.reload=undefined;return s.used;}
/** Kept only to reject historical snapshot callers; daily producer uses actual resident core-drop. */
export async function restoreRetainedCorePrevious(_engine:NativeEngine,_guard:SnapshotGuard,_previous:NativeRouterEngineSnapshot,_prepared:NativeShallowPreparation,_owners:NativeCore450Owner):Promise<never>{throw Error('Historical Router/RNG snapshot restoration is not an evolution operation');}
