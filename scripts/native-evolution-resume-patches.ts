/** Connect durable product preparation to the original checkpoint runner; no new round on resume. */
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
export async function applyNativeEvolutionResumePatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(path:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string}){
 const research=join(root,'.native-migration/research'),helper=JSON.stringify(join(root,'backend/native-evolution-resume.ts'));
 const producer=target(join(research,'native-evolution-producer/producer.ts'));let body=bodies.get(producer)!;
 body="import {reopenNativeEvolutionPreparation} from "+helper+";\nimport {retainNativeEvolutionBackbone} from "+JSON.stringify(join(root,'backend/native-evolution-reload.ts'))+";\nimport {materialMessages} from "+JSON.stringify(join(root,'backend/native-evolution-material.ts'))+";\nimport {readNativeEvolutionSnapshot} from '../native-unified-expert/class-closure/native-product-runtime/backend/native-evolution-state.ts';\n"+body;
 body=replace(body,'type Request={engine:NativeEngine;','type Request={resumeId?:string;engine:NativeEngine;');
 body=replace(body,'import {Memory}', 'import {Memory,ledgerParse}');
 body=replace(body,'export type NativeEvolutionPrepared=Awaited<ReturnType<typeof prepare>>;','export type NativeEvolutionPrepared=Awaited<ReturnType<typeof prepare>>&{resume?:boolean};');
 body=replace(body,'accepted:NativeModelTensor;basis:NativeFixtureRaw;','ambient:Parameters<typeof prepare>[5];basis:NativeFixtureRaw;');
 body=replace(body,'closed:()=>boolean}>();','closed:()=>boolean}&({fresh:true;accepted:NativeModelTensor}|{fresh:false;accepted:NativeFixtureRaw})>();');
 body=replace(body,'{engine:r.engine,guard,store,owners:r.owners,accepted,basis:basisRaw,','{engine:r.engine,guard,store,owners:r.owners,ambient,fresh:true,accepted,basis:basisRaw,');
 body=replace(body,'const prepared=await prepare(r,store,base,guard,control,previous);',`const prepared=r.resumeId?await reopenProductPreparation(r,store,base,guard,previous):await prepare(r,store,base,guard,control,previous);
 if(r.resumeId&&peer.worker.status.backbone_resident!==true){
  const sample=prepared.records.groups.new_training[0]??prepared.records.groups.replay_training[0];need(sample,'Original TRAIN sample required for resumed source descriptor');
  const tokens=await control.prepareRouteTokens(materialMessages(sample).map(row=>({role:row.role,content:row.content})),{signal:r.signal});
  r.signal?.throwIfAborted();await retainNativeEvolutionBackbone(r.engine,guard,tokens);r.signal?.throwIfAborted();
 }`);
 body+=`
async function reopenProductPreparation(r:Request,store:string,base:Awaited<ReturnType<typeof resolveNativeSnapshotBase>>,guard:PreparationGuard,ambient:Parameters<typeof prepare>[5]):Promise<NativeEvolutionPrepared>{
 need(typeof r.resumeId==='string'&&/^[a-f0-9]{12}$/.test(r.resumeId),'Original resume id required');
 const row=r.memory.connect(db=>db.query('SELECT state,body FROM training_runs WHERE id=? AND identity=? AND kind=?').get(r.resumeId!,base.identity,'evolution')) as any;
 need(row&&['paused','running','queued'].includes(row.state),'Original pending native journal is unavailable');const saved=ledgerParse(row.body);
 need(saved.id===r.resumeId&&saved.kind==='evolution'&&saved.scope==='product'&&saved.checkpoint&&saved.coordinator?.native_preparation,'Original native preparation/checkpoint is not resumable');
 const cohort=selectIndependentNativeEvolutionCohort(r.memory,store,base),peer=borrowNativeEnginePeer(r.engine,guard),snapshot=saved.coordinator.model_snapshot;
 need(snapshot&&snapshot.id===r.resumeId&&encoded(saved.records)===encoded(cohort.records),'Resume material or original snapshot differs');
 const previous=r.memory.connect(db=>readNativeEvolutionSnapshot(db,store,base,snapshot.id));
 need(previous.row.manifest_sha256===snapshot.manifest_sha256&&previous.manifest.id===r.resumeId,'Saved pre-round model snapshot differs');
 const materialIDs=[...new Set([...Object.values(cohort.records.groups).flat(),...cohort.records.protection,...cohort.records.retention_regressions].map((row:any)=>row.id))];
 const previousVersion={id:r.resumeId,version:previous.state.core?.version??null,path:join(store,'evolution-snapshots',r.resumeId,'manifest.json'),manifest_sha256:snapshot.manifest_sha256,records_sha256:contentSHA(cohort.records),material_ids:materialIDs,actual_previous_files_and_committed_journal:true,local_previous_version_saved:true,product_admission:false,rollback_completed:false,assertUnchanged:async()=>{guard.assertQuiescent();cohort.assertUnchanged();}};
 const opened=await reopenNativeEvolutionPreparation({binding:saved.coordinator.native_preparation,base,expected:{id:r.resumeId,kind:'evolution',binding:saved.binding},cohort,previousVersion,acceptedTensorSHA:peer.source.runtime.core_raw?.sha256??base.original_down_sha256,trainingBinarySHA:r.execution.binary_sha256,owners:r.owners,guard,signal:r.signal});
 const prepared=opened.prepared as unknown as NativeEvolutionPrepared;
 need(prepared.resume===true&&prepared.id===r.resumeId,'Reopened preparation lost its original id');
 productPreparations.set(prepared,{engine:r.engine,guard,store,owners:r.owners,ambient,fresh:false,accepted:opened.accepted,basis:opened.basis,metadata:preparationMetadata(prepared),closed:opened.closed});
 await prepared.assertUnchanged();return prepared;
}
`;
 bodies.set(producer,body);
 const fresh=target(join(research,'native-core450-runner/fresh.ts'));body=bodies.get(fresh)!;
 body=replace(body,'const source=borrowNativeEvolutionPreparation(engine,guard,r.preparation);','const source=borrowNativeEvolutionPreparation(engine,guard,r.preparation);assert(source.fresh&&!r.preparation.resume,"Resume preparation cannot initialize a fresh Core");');bodies.set(fresh,body);
 const journal=target(join(research,'native-core-controller-candidate/journal-snapshot.ts'));body=bodies.get(journal)??await readFile(journal,'utf8');
 body="import {nativeEvolutionPreparationBinding} from "+helper+";\nimport type {NativeEvolutionPrepared} from '../native-evolution-producer/producer.ts';\n"+body;
 body=replace(body,'r:Request&{snapshot_id:string}','r:Request&{snapshot_id:string;preparation?:NativeEvolutionPrepared}');
 body=replace(body,"body.coordinator={model_snapshot:","body.coordinator={...(r.preparation?{native_preparation:nativeEvolutionPreparationBinding(r.preparation)}:{}),model_snapshot:");bodies.set(journal,body);
 const boot=target(join(research,'native-core450-owner-controller/bootstrap.ts'));body=bodies.get(boot)!;
 body=replace(body,'snapshot_id:retained.id});\n   fresh=await prepareCore450ProductFresh','snapshot_id:retained.id,preparation:r.preparation});\n   fresh=await prepareCore450ProductFresh');
 const productStart=body.indexOf('export async function createCore450ProductFreshCheckpoint(');if(productStart<0)throw Error('Product bootstrap missing');
 body=body.slice(0,productStart)+replace(body.slice(productStart),"previous=await control.captureRouterInitial(join(r.directory,'fresh-previous'));const previousOwner=previous;owners.own(previousOwner,()=>previousOwner.close());", "assert(!r.preparation.resume,'A reopened preparation cannot bootstrap a new round');previous=borrowNativeEvolutionPreparation(engine,guard,r.preparation).ambient;");bodies.set(boot,body);
 const job=target(join(research,'native-core-product-job-source-v1/job.ts'));bodies.set(job,patchNativeEvolutionResumeJob(bodies.get(job)!,replace));
}
export function patchNativeEvolutionResumeJob(body:string,replace:(body:string,before:string,after:string)=>string){
 body="import {ledgerParse} from '../native-product-runtime/backend/memory.ts';\n"+body;
 body=replace(body,'const ids=[...new Set<string>',"const resumed=r.resumeId?r.memory.connect(db=>{const row=db.query('SELECT body FROM training_runs WHERE id=?').get(r.resumeId!)as any;assert(row);return ledgerParse(row.body);}):null;\n  const ids=[...new Set<string>");
 body=replace(body,'args=r.arguments??{steps:450,','args=resumed?.arguments??r.arguments??{steps:450,');
 body=replace(body,'   bootstrap=await createCore450ProductFreshCheckpoint(',`   if(resumed){expected={id:prepared.id,kind:'evolution',binding:resumed.binding};runRequest={...common,expected,directory:runDirectory,binary:join(release.root,reader.path),binary_sha256:release.manifest.payload[reader.path].final.sha256,writer:release.binary,writer_sha256:release.binary_sha256,trainingPackage:releaseRef,onWorker(worker){workers.add(worker);},onFinal:r.onFinal};bootstrap={expected,checkpoint:resumed.checkpoint,previous:held.ambient};}
   else bootstrap=await createCore450ProductFreshCheckpoint(`);
 body=replace(body,'if(expected&&runRequest)pauseOwnedCore450Journal(',"if(expected&&runRequest&&(!r.resumeId||r.memory.connect(db=>{const row=db.query('SELECT state,body FROM training_runs WHERE id=?').get(r.resumeId!)as any;return row&&['queued','running'].includes(row.state)&&ledgerParse(row.body).owner?.pid===process.pid;})))pauseOwnedCore450Journal(");
 body=replace(body,'bootstrap:{expected:bootstrap.expected,checkpoint:bootstrap.checkpoint,optimizer_steps:0},',"...(resumed?{resume:{id:preparedResumeID(r.resumeId),baseline_reused:true,bootstrap_invoked:false}}:{bootstrap:{expected:bootstrap.expected,checkpoint:bootstrap.checkpoint,optimizer_steps:0}}),");
 // The final return is outside the producer callback; use only the explicit original request id there.
 body=body.replace("...(resumed?{resume:{id:preparedResumeID(r.resumeId)","...(r.resumeId?{resume:{id:r.resumeId");
 return body;
}
