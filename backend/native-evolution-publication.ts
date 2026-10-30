/** Concrete existing storage/source-ACK/snapshot transaction. The one private qualification gate precedes every mutation. */
import assert from'node:assert/strict';import{readSafe,encoded,hash,type SnapshotBase}from'../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';
import{NativeEngine,type TextRequest}from'../.native-migration/research/native-unified-expert/native-engine.ts';import type{SnapshotGuard}from'../.native-migration/research/native-product-runtime/backend/evolution-snapshots.ts';
import{NativeCore450Owner}from'../.native-migration/research/native-core450-owner-controller/owner.ts';
import{prepareNativeCoreCommit,applyNativeCoreCommit,finishNativeCoreCommit,restoreNativeCoreCommitPointer,finishNativeCoreCommitRejection}from'../.native-migration/research/native-core-product-publication-v1/commit-storage.ts';
import{settleRetainedCoreReload}from'../.native-migration/research/native-core-peer-refresh-source-v1/reload.ts';
import{requireCore450Publication,validateCore450QualityObservations}from'./native-evolution-admission.ts';import{validateCore450CandidateArtifacts,immutableCore450Artifact,publicationCore450Artifacts,type Core450CandidateArtifacts}from'./native-evolution-artifacts.ts';

type Control=Parameters<Parameters<NativeEngine['withSourceQuiescent']>[0]>[1];
/** Caller retains the original FINAL loader/journal owner inside its fresh withSourceQuiescent callback. */
export async function publishCore450Candidate(engine:NativeEngine,guard:SnapshotGuard,control:Control,candidate:Core450CandidateArtifacts,owners:NativeCore450Owner,messages:TextRequest['messages']){
 validateCore450CandidateArtifacts(engine,guard,candidate);const published=requireCore450Publication(engine,guard,candidate.quality,candidate.pointer.sha256);await candidate.assertUnchanged();assert(owners instanceof NativeCore450Owner);const s=validateCore450QualityObservations(engine,guard,candidate.quality),{memory,store,base}=s.context,checkpoint=candidate.quality.checkpoint,id=checkpoint.approval.id;
 const request={memory,store,base,guard,expected:{id,binding:{...checkpoint.binding}}},{report,replacements}=publicationCore450Artifacts(candidate,published,s.cohort.records,base,store);
 let before:Awaited<ReturnType<Control['prepareRetainedCoreReload']>>|undefined,recoveryBefore:typeof before,prepared:Awaited<ReturnType<Control['prepareRouteTokens']>>|undefined,intent=false,committed=false,firstError:unknown,rollbackSelected=false,recoveryReceipt:Awaited<ReturnType<Control['reloadRetainedCoreSource']>>|undefined;
 try{
  prepared=await control.prepareRouteTokens(messages,{signal:owners.signal});
  // The producer already restored the accepted source. Reuse its existing source-transfer lifecycle; no second Core/router snapshot or candidate installation.
  before=await control.prepareRetainedCoreReload(owners,prepared);
  for(const[name]of Object.entries(candidate.artifacts)){await candidate.assertUnchanged();const bytes=replacements[name]??readSafe(candidate.directory,name);if(!replacements[name])assert.equal(hash(bytes),candidate.artifacts[name],'Staged artifact changed before publication');await immutableCore450Artifact(store,name,bytes);}
  requireCore450Publication(engine,guard,candidate.quality,candidate.pointer.sha256);prepareNativeCoreCommit(request,{report,pointer:{...candidate.pointer},artifacts:Object.keys(candidate.artifacts)});intent=true;applyNativeCoreCommit(request);
  const receipt=await control.reloadRetainedCoreSource(before,prepared,owners,id);finishNativeCoreCommit(request,receipt);committed=true;return{state:'committed' as const,actual_reload_confirmed:true,report};
 }catch(error){firstError=error;throw error;}
 finally{await owners.finishScope(async()=>{
  // The durable committed journal is final; cleanup must never roll its pointer back.
  if(committed)return;
  if(intent){assert(before&&prepared);const changed=await settleRetainedCoreReload(before);recoveryBefore??=changed?await control.prepareRetainedCoreReload(owners,prepared):before;if(!rollbackSelected){restoreNativeCoreCommitPointer(request,String(firstError));rollbackSelected=true;}recoveryReceipt??=await control.reloadRetainedCoreSource(recoveryBefore,prepared,owners,id);finishNativeCoreCommitRejection(request,recoveryReceipt);}
  // Prior to an intent, any installed candidate is restored through the same existing retained-source ACK.
  else if(before&&prepared){const changed=await settleRetainedCoreReload(before);recoveryBefore??=changed?await control.prepareRetainedCoreReload(owners,prepared):before;await control.reloadRetainedCoreSource(recoveryBefore,prepared,owners,id);}
 },async()=>{});}
}
