/** Reopen the accepted source after the standalone learning worker actually closed. */
import assert from 'node:assert/strict';
import {mkdir,mkdtemp} from 'node:fs/promises';
import {dirname,join} from 'node:path';
import {NativeEngine,assertNativeExpertScopeGuard,type NativeShallowPreparation} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {ensureNativeEnginePeer,closeNativeEnginePeer,hasNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {assertClosedExpertReload,validateExpertReloadSnapshot,validateExpertReloadInitialSnapshot,type ExpertReloadSnapshot,type ExpertReloadInitialSnapshot} from '../.native-migration/research/native-unified-expert/expert-process.ts';
import {writePeerDescriptor} from '../.native-migration/research/native-unified-peer/descriptor.ts';
import {equal,type SnapshotBase} from '../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';
import type {SnapshotGuard,NativeReloadReceipt} from '../.native-migration/research/native-unified-expert/class-closure/native-product-runtime/backend/evolution-snapshots.ts';

const receipts=new WeakMap<NativeReloadReceipt,{engine:NativeEngine;guard:SnapshotGuard;peer:Awaited<ReturnType<typeof ensureNativeEnginePeer>>;coreACK:Record<string,any>;expertACK:Record<string,any>}>();
export function validateClosedNativeExpertReload(engine:NativeEngine,guard:SnapshotGuard,receipt:NativeReloadReceipt){const s=receipts.get(receipt);assert(s&&s.engine===engine&&s.guard===guard&&s.peer.guard===guard&&!s.peer.closed);assertNativeExpertScopeGuard(engine,guard);s.peer.worker.assertLive();assert(receipt.worker_pid===s.peer.worker.pid&&s.coreACK.backbone_resident===true&&s.coreACK.retained_core_tensor_sha256===(s.peer.accepted.core?.tensor_sha256??s.peer.base.original_down_sha256)&&s.expertACK.optimizer_step===0,'Expert reload requires actual native Core/expert ACKs');return s.peer.accepted;}
export async function reloadClosedNativeExpert(engine:NativeEngine,guard:SnapshotGuard,store:string,base:SnapshotBase,id:string,transfer:ExpertReloadSnapshot|ExpertReloadInitialSnapshot,prepared:NativeShallowPreparation):Promise<NativeReloadReceipt>{
 assertNativeExpertScopeGuard(engine,guard);assert(/^[a-f0-9]{12}$/.test(id)&&engine.config.routing?.store===store&&equal(engine.config.aten?.base,base)&&engine.config.expert_build);
 if(transfer.owner)validateExpertReloadSnapshot(transfer);else validateExpertReloadInitialSnapshot(transfer);
 assertClosedExpertReload(transfer);await transfer.assertUnchanged();assert(transfer.hidden===2048&&transfer.device==='mps');
 // A failed newer reload may own a peer; never open a second full model beside it.
 if(hasNativeEnginePeer(engine))await closeNativeEnginePeer(engine,guard);
 const work=dirname(engine.config.log);await mkdir(work,{recursive:true,mode:0o700});
 const peer=await ensureNativeEnginePeer(engine,guard,{directory:await mkdtemp(join(work,'expert-reload-')),model:engine.config.model,store,base,build:engine.config.expert_build});
 const descriptor=await writePeerDescriptor({runtime:peer.source.runtime,guard,owner:peer.owner,core_owner:peer.core,request_id:'expert-reload-core',work:peer.worker.options.directory,prompt_ids:[...prepared.ids],messages_sha256:prepared.messages_sha256,max_tokens:1,raw_steps_limit:0,core_version:peer.source.runtime.core_label??null,expert:null,thinking:'off',temperature:0,sampling_seed:'42',onToken:()=>{}});
 const coreACK=(await peer.worker.retainCoreBackbone({...descriptor,core_tensor_sha256:peer.accepted.core?.tensor_sha256??base.original_down_sha256})).response;
 const previous=peer.previous;let expertACK:Record<string,any>;
 assert(Boolean(previous)===Boolean(transfer.owner),'Closed expert transfer differs from actual accepted source');
 if(previous){
  assert(transfer.owner&&previous.width===(transfer as ExpertReloadSnapshot).width);
  for(const key of previous.owner.tensor_keys){const a=previous.owner.tensors[key]!,b=transfer.owner.tensors[key]!;assert(b&&a.dtype===b.dtype&&a.bytes===b.bytes&&equal(a.shape,b.shape)&&a.sha256===b.sha256,'Closed expert tensor differs from accepted source');}
  expertACK=(await peer.worker.restoreReloadSnapshot(transfer as ExpertReloadSnapshot)).response;assert(expertACK.optimizer_step===0&&expertACK.resident_parameters===3*2048*previous.width);
  await peer.worker.adoptAcceptedPrevious(previous);
 }else{
  expertACK=(await peer.worker.restoreReloadInitialSnapshot(transfer as ExpertReloadInitialSnapshot)).response;assert(expertACK.optimizer_step===0&&expertACK.resident_parameters===0);
  peer.worker.adoptClosedEmptyReloadSource(transfer as ExpertReloadInitialSnapshot);
 }
 await transfer.assertUnchanged();await peer.worker.setBranchEnabled(true);peer.worker.assertLive();
 const accepted=peer.accepted;
 const receipt:NativeReloadReceipt={restore_id:id,worker_pid:peer.worker.pid,engine_binary_sha256:base.engine_binary_sha256,
  loaded:{protocol:'luban-worker-jsonl-v1',base_identity:base.identity,manifest_sha256:base.model_manifest_sha256,core_loaded:Boolean(accepted.core),...(accepted.core?{core_file_sha256:accepted.core.file_sha256,core_tensor_sha256:accepted.core.tensor_sha256}:{})},expert:accepted.expert,router:accepted.router};
 receipts.set(receipt,{engine,guard,peer,coreACK,expertACK});validateClosedNativeExpertReload(engine,guard,receipt);return receipt;
}
