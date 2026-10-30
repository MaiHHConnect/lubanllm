/** Four immutable candidate artifacts. Unqualified artifacts remain unpromoted and cannot prepare a commit. */
import assert from'node:assert/strict';import{messages,validateNativeCoreRetention}from'./native-evolution-material.ts';import{join,dirname}from'node:path';import{readFile,mkdtemp,open,rm}from'node:fs/promises';import{constants}from'node:fs';
import{NativeEngine}from'../.native-migration/research/native-unified-expert/native-engine.ts';import type{SnapshotGuard}from'../.native-migration/research/native-product-runtime/backend/evolution-snapshots.ts';
import{validateCore450QualityObservations,type Core450QualityObservations}from'./native-evolution-admission.ts';
import{writeNativeModelTensors}from'../.native-migration/backend/native-model-write.ts';
import{readNativeModelTensors}from'../.native-migration/backend/native-model-tensor.ts';
import{encoded,hash,readSafe,pathIn,type SnapshotBase}from'../.native-migration/research/native-product-runtime/backend/evolution-snapshot-files.ts';import{ledgerParse}from'../.native-migration/research/native-product-runtime/backend/memory.ts';
import{provenance,provenanceJSON}from'../.native-migration/research/native-product-runtime/backend/native-retention-records.ts';
import{durable}from'../.native-migration/research/native-full-training/io.ts';
export function core450CandidateRetention(records:any,masteredIDs:string[],base:SnapshotBase,version:string){
 const mastered=new Set(masteredIDs),clone=(v:any)=>ledgerParse(encoded(v));
 const groups=Object.fromEntries(Object.entries(records.groups).map(([key,rows])=>[key,(rows as any[]).filter(row=>mastered.has(row.id)).map(clone)]));
 const groupIDs=new Set((Object.values(groups).flat()as any[]).map(row=>row.id)),byID=new Map<string,any>();
 const append=(row:any)=>{if(groupIDs.has(row.id))return;const previous=byID.get(row.id);if(previous){assert.equal(encoded(messages(previous)),encoded(messages(row)));assert.equal(encoded(provenance(previous).generation_check),encoded(provenance(row).generation_check));return;}byID.set(row.id,clone(row));};
 // Preserve every prior/original protection row, including regressions inherited via protection.
 for(const row of records.protection){if(groupIDs.has(row.id))continue;const copy=clone(row),p=provenance(copy);Object.assign(p,{retention_generation_only:true,evolution_role:'replay'});copy.provenance=provenanceJSON(p);append(copy);}
 for(const row of records.retention_regressions)if(mastered.has(row.id))append(row);
 for(const row of Object.values(groups).flat()as any[])if(row.purpose==='training'){const copy=clone(row),p=provenance(copy);Object.assign(copy,{id:'mastered-'+row.id,purpose:'validation',training_allowed:false});Object.assign(p,{retention_generation_only:true,evolution_role:'replay',original_sample_id:row.id});copy.provenance=provenanceJSON(p);append(copy);}
 const content={schema_version:1,version,identity:base.identity,groups,regression_validation:[...byID.values()]},retention={...content,content_sha256:hash(encoded(content,true))},bytes=Buffer.from(encoded(retention)+'\n');validateNativeCoreRetention(retention,base.identity,version);return{retention,bytes};
}
export type Core450CandidateArtifacts={readonly directory:string;readonly quality:Core450QualityObservations;readonly report:Readonly<Record<string,any>>;readonly pointer:Readonly<{version:string;identity:string;sha256:string}>;readonly artifacts:Readonly<Record<string,string>>;assertUnchanged():Promise<void>};
const staged=new WeakMap<Core450CandidateArtifacts,{engine:NativeEngine;guard:SnapshotGuard;json:string;directory:string}>();
export function validateCore450CandidateArtifacts(engine:NativeEngine,guard:SnapshotGuard,cap:Core450CandidateArtifacts){const s=staged.get(cap);assert(s&&s.engine===engine&&s.guard===guard&&s.directory===cap.directory,'Four artifacts require this scope private FINAL quality capability');validateCore450QualityObservations(engine,guard,cap.quality);assert.equal(encoded({report:cap.report,pointer:cap.pointer,artifacts:cap.artifacts}),s.json);return s;}
export async function immutableCore450Artifact(store:string,name:string,bytes:Buffer){try{assert(readSafe(store,name).equals(bytes),'Candidate immutable artifact conflict');}catch(e:any){if(e.code!=='ENOENT')throw e;await durable(pathIn(store,name),bytes);}const directory=await open(store,constants.O_RDONLY|constants.O_NOFOLLOW);try{await directory.sync();}finally{await directory.close();}assert(readSafe(store,name).equals(bytes));}
export async function prepareCore450CandidateArtifacts(engine:NativeEngine,guard:SnapshotGuard,quality:Core450QualityObservations){
 const s=validateCore450QualityObservations(engine,guard,quality);await quality.assertUnchanged();const{store,base}=s.context,checkpoint=quality.checkpoint,version='core-'+checkpoint.approval.id,weightName=version+'.safetensors',metaName=version+'.json',retentionName=version+'.retention.json',reportName='evolution-'+checkpoint.approval.id+'.json';assert.equal(quality.report.parameter,base.core_parameter);assert(checkpoint.state.candidate.dtype==='bf16'&&checkpoint.state.candidate.bytes===25165824);const temporary=await mkdtemp(join(store,'.core-candidate-')),formatDirectory=await mkdtemp(join(dirname(checkpoint.state.candidate.path),'.core-format-')),path=join(formatDirectory,'weights.partial');
 let weights:Buffer;try{const written=await writeNativeModelTensors({path,tensors:[{tensor_key:base.core_parameter,dtype:'BF16',shape:[...base.core_shape],path:checkpoint.state.candidate.path,bytes:checkpoint.state.candidate.bytes,sha256:checkpoint.state.candidate.sha256}]});try{await written.assertUnchanged();weights=await readFile(path);assert.equal(hash(weights),written.file_sha256);assert.equal(written.tensors[base.core_parameter]!.sha256,checkpoint.state.candidate.sha256);}finally{await written.close();assert.equal(written.handle.fd,-1);}}finally{await rm(formatDirectory,{recursive:true,force:true});}await quality.assertUnchanged();
 const mastered=[...new Set<string>([...quality.evidence.learning_progress.mastered_ids,...quality.evidence.learning_progress.retained_ids])],retained=core450CandidateRetention(s.cohort.records,mastered,base,version),metadata={version,identity:base.identity,previous_version:checkpoint.binding.core_version,retention_sha256:hash(retained.bytes),dataset_sha256:quality.report.dataset_sha256,parameter:base.core_parameter,shape:[...base.core_shape],dtype:'torch.bfloat16',sha256:hash(weights),candidate_tensor_sha256:checkpoint.state.candidate.sha256,report:reportName,format:'cumulative core tensor replacement',trained_parameters:12582912};
 const promoted=Boolean(quality.production_model_qualified)&&Object.values(quality.evidence.gates).every(Boolean),report={...ledgerParse(encoded(quality.report)),checkpoint:metadata,retention:{path:pathIn(store,retentionName),file_sha256:metadata.retention_sha256,content_sha256:retained.retention.content_sha256},accepted_retention_groups:retained.retention.groups,accepted_retention_regressions:retained.retention.regression_validation,quality_checked:promoted,promoted,product_admission:promoted,adopted:false,publication_state:promoted?'validated_candidate':'awaiting_production_qualification'};
 for(const[name,bytes]of [[weightName,weights],[retentionName,retained.bytes],[metaName,Buffer.from(encoded(metadata))],[reportName,Buffer.from(encoded(report))]]as const){await quality.assertUnchanged();await immutableCore450Artifact(temporary,name,bytes);}const owner=await readNativeModelTensors({path:pathIn(temporary,weightName),file_sha256:metadata.sha256,tensors:[{tensor_key:base.core_parameter,dtype:'BF16',shape:[...base.core_shape]}]});try{await owner.assertUnchanged();assert.equal(owner.tensors[base.core_parameter]!.sha256,checkpoint.state.candidate.sha256);}finally{await owner.close();assert.equal(owner.handle.fd,-1);}await quality.assertUnchanged();
 const pointer=Object.freeze({version,identity:base.identity,sha256:metadata.sha256}),artifacts=Object.freeze(Object.fromEntries([weightName,retentionName,metaName,reportName].map(name=>[name,hash(readSafe(temporary,name))]))),json=encoded({report,pointer,artifacts});const cap:Core450CandidateArtifacts=Object.freeze({directory:temporary,quality,report:Object.freeze(report),pointer,artifacts,async assertUnchanged(){validateCore450CandidateArtifacts(engine,guard,cap);await quality.assertUnchanged();for(const[name,pin]of Object.entries(artifacts))assert.equal(hash(readSafe(temporary,name)),pin);}});staged.set(cap,{engine,guard,json,directory:temporary});await cap.assertUnchanged();return cap;
}

/** Reopen the already written unadopted candidate after the BF16/Q4 processes were closed. No second model write. */
export async function reopenCore450CandidateArtifacts(engine:NativeEngine,guard:SnapshotGuard,quality:Core450QualityObservations,directory:string){
 const s=validateCore450QualityObservations(engine,guard,quality),{base}=s.context,cp=quality.checkpoint,version='core-'+cp.approval.id;
 assert(directory.startsWith(s.context.store+'/.core-candidate-'));
 const names=[version+'.safetensors',version+'.retention.json',version+'.json','evolution-'+cp.approval.id+'.json'];
 const metadata=ledgerParse(readSafe(directory,names[2]!).toString()),report=ledgerParse(readSafe(directory,names[3]!).toString());
 assert(metadata.version===version&&metadata.identity===base.identity&&metadata.candidate_tensor_sha256===cp.state.candidate.sha256&&metadata.parameter===base.core_parameter&&metadata.previous_version===cp.binding.core_version);
 assert(report.adopted===false&&report.product_admission===false&&report.quality_checked===false&&report.promoted===false);
 for(const[key,value]of Object.entries(quality.report))assert.equal(encoded(report[key]),encoded(value),'Staged report differs from this durable FINAL');
 assert.equal(encoded(report.checkpoint),encoded(metadata));
 const mastered=[...new Set<string>([...quality.evidence.learning_progress.mastered_ids,...quality.evidence.learning_progress.retained_ids])];
 const expected=core450CandidateRetention(s.cohort.records,mastered,base,version),retention=readSafe(directory,names[1]!);
 assert(retention.equals(expected.bytes)&&hash(retention)===metadata.retention_sha256);
 const owner=await readNativeModelTensors({path:pathIn(directory,names[0]!),file_sha256:metadata.sha256,tensors:[{tensor_key:base.core_parameter,dtype:'BF16',shape:[...base.core_shape]}]});
 try{assert.equal(owner.tensors[base.core_parameter]!.sha256,cp.state.candidate.sha256);}finally{await owner.close();assert.equal(owner.handle.fd,-1);}
 const pointer=Object.freeze({version,identity:base.identity,sha256:metadata.sha256});
 const artifacts=Object.freeze(Object.fromEntries(names.map(name=>[name,name===names[0]?metadata.sha256:hash(readSafe(directory,name))])));
 const json=encoded({report,pointer,artifacts});
 const cap:Core450CandidateArtifacts=Object.freeze({directory,quality,report:Object.freeze(report),pointer,artifacts,
  async assertUnchanged(){validateCore450CandidateArtifacts(engine,guard,cap);await quality.assertUnchanged();}});
 staged.set(cap,{engine,guard,json,directory});return cap;
}

/** Final JSON uses Q4 mastery; the safetensors file remains the exact staged and observed weight file. */
export function publicationCore450Artifacts(candidate:Core450CandidateArtifacts,published:Record<string,any>,records:any,base:SnapshotBase,store:string){
 const version=candidate.pointer.version,mastered=[...new Set<string>([...published.learning_progress.mastered_ids,...published.learning_progress.retained_ids])];
 const retained=core450CandidateRetention(records,mastered,base,version),retentionName=version+'.retention.json';
 const metadata={...candidate.report.checkpoint,retention_sha256:hash(retained.bytes)};
 const report:Record<string,any>={...published,checkpoint:metadata,retention:{path:pathIn(store,retentionName),file_sha256:metadata.retention_sha256,content_sha256:retained.retention.content_sha256},accepted_retention_groups:retained.retention.groups,accepted_retention_regressions:retained.retention.regression_validation,promoted:true,quality_checked:true,product_admission:true,adopted:true,publication_state:'published'};
 const replacements:Record<string,Buffer>={[retentionName]:retained.bytes,[version+'.json']:Buffer.from(encoded(metadata)),['evolution-'+candidate.quality.checkpoint.approval.id+'.json']:Buffer.from(encoded(report))};
 return{report,replacements};
}
