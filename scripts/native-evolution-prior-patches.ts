/** Keep the original journal/checkpoint source binding; remove migration-only copied-model snapshots. */
import {join} from 'node:path';
import {readFile} from 'node:fs/promises';
type Context={bodies:Map<string,string>;target:(path:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string};
export function patchNativeEvolutionPriorJob(body:string,replace:Context['replace']){
 body=replace(body,',previousSnapshot:{id:string;manifest_sha256:string;path:string}|undefined','');
 body=replace(body,'previousSnapshot={id:prepared.previousVersion.id,manifest_sha256:prepared.previousVersion.manifest_sha256,path:prepared.previousVersion.path};','');
 body=replace(body,'&&previousSnapshot','');
 return replace(body,'previous_snapshot:previousSnapshot,','');
}
export async function applyNativeEvolutionPriorPatches({bodies,target,replace,root}:Context){
 const research=join(root,'.native-migration/research');
 const edit=async(name:string,change:(body:string)=>string)=>{const path=target(join(research,name));bodies.set(path,change(bodies.get(path)??await readFile(path,'utf8')));};
 await edit('native-evolution-producer/producer.ts',body=>{
  body=body.replace(/^import .*\{saveRetainedEvolutionPrevious\}.*\n/m,'').replace(/^import \{readNativeEvolutionSnapshot\}.*\n/m,'');
  body=replace(body,',previousVersion=await saveRetainedEvolutionPrevious(r.engine,guard,r.memory,store,id,cohort,r.owners)','');
  body=replace(body,'previous_version:{id:previousVersion.id,path:previousVersion.path,manifest_sha256:previousVersion.manifest_sha256},','');
  body=replace(body,'cohort,previousVersion,graphBinding','cohort,graphBinding');
  const start=body.indexOf('const cohort=selectIndependentNativeEvolutionCohort(r.memory,store,base),peer=borrowNativeEnginePeer(r.engine,guard),snapshot=saved.coordinator.model_snapshot;');
  const end=body.indexOf(' const opened=await reopenNativeEvolutionPreparation',start);
  if(start<0||end<start)throw Error('Original resume source binding boundary missing');
  body=body.slice(0,start)+`const cohort=selectIndependentNativeEvolutionCohort(r.memory,store,base),peer=borrowNativeEnginePeer(r.engine,guard);
 need(encoded(saved.records)===encoded(cohort.records),'Resume material differs from original journal');
`+body.slice(end);
  return replace(body,'cohort,previousVersion,acceptedTensorSHA:','cohort,acceptedTensorSHA:');
 });
 await edit('native-core-controller-candidate/journal-snapshot.ts',body=>{
  body=body.replace(/^import \{EvolutionSnapshots\}.*\n/m,'').replace(/^import \{readNativeEvolutionSnapshot\}.*\n/m,'');
  const start=body.indexOf('export function captureCore450Previous('),end=body.indexOf('/** Exact row CAS',start);
  if(start<0||end<start)throw Error('Legacy snapshot capture boundary missing');
  // Historical fixed-data callers stay unavailable; no fake snapshot capability is issued.
  body=body.slice(0,start)+"export function captureCore450Previous(..._args:unknown[]):{id:string}{throw Error('Historical copied-model snapshot bootstrap is not a product evolution entry');}\n"+body.slice(end);
  body=replace(body,'r:Request&{snapshot_id:string;preparation?:NativeEvolutionPrepared}','r:Request&{preparation:NativeEvolutionPrepared}');
  body=replace(body,'  const previous=readNativeEvolutionSnapshot(db,r.store,r.base,r.snapshot_id);assert(previous.manifest.id===r.id&&equal(previous.state,accepted)&&Number(previous.manifest.created)<=Number(body.started),\'Snapshot must predate the actual journal and preserve its complete accepted source\');\n','');
  body=replace(body,"body.coordinator={...(r.preparation?{native_preparation:nativeEvolutionPreparationBinding(r.preparation)}:{}),model_snapshot:{id:previous.row.id,manifest_sha256:previous.row.manifest_sha256},initial_state:'awaiting_native_complete_pt0'};","assert(r.preparation.id===r.id&&!r.preparation.resume&&equal(r.preparation.records,body.records));body.coordinator={native_preparation:nativeEvolutionPreparationBinding(r.preparation),initial_state:'awaiting_native_complete_pt0'};");
  return replace(body,'return{model_snapshot:body.coordinator.model_snapshot,','return{');
 });
 await edit('native-core450-owner-controller/bootstrap.ts',body=>{
  body=replace(body,'bindCore450Previous(engine,guard,{memory:r.memory,store:r.store,base:r.base,id,snapshot_id:retained.id});','void retained.id;');
  const start=body.indexOf('export async function createCore450ProductFreshCheckpoint(');if(start<0)throw Error('Product bootstrap missing');
  let product=body.slice(start);
  // Replace the snapshot binding only; the original fresh journal and complete PT0 remain.
  product=replace(product,'const retained=r.preparation.previousVersion;','');
  product=replace(product,'await retained.assertUnchanged();','');
  product=replace(product,'snapshot_id:retained.id,preparation:r.preparation','preparation:r.preparation');
  return body.slice(0,start)+product;
 });
 await edit('native-core-product-job-source-v1/job.ts',body=>patchNativeEvolutionPriorJob(body,replace));
 await edit('native-core450-runner/checkpoint.ts',body=>{
  body=body.replace(/^import \{readNativeEvolutionSnapshot\}.*\n/m,'');
  for(const [before,after]of[
   ['previous_snapshot:Readonly<{id:string;manifest_sha256:string}>;',''],
   [';snapshot:string',''],
   ['&&JSON.stringify(cap.previous_snapshot)===s.snapshot',''],
   [';previous_snapshot:{id:string;manifest_sha256:string}',''],
   [',snapshot:r.previous_snapshot',''],
   [';snapshot:{id:string;manifest_sha256:string}',''],
   [',snapshot:JSON.stringify({id:retained.id,manifest_sha256:retained.manifest_sha256})',''],
   ['previous_snapshot:Object.freeze({id:retained.id,manifest_sha256:retained.manifest_sha256}),','']
  ])body=replace(body,before!,after!);
  const start=body.indexOf('const retained=body.coordinator?.model_snapshot;'),end=body.indexOf('const saved=',start);
  if(start<0||end<start)throw Error('Checkpoint snapshot audit boundary missing');
  body=body.slice(0,start)+"assert(body.coordinator?.native_preparation?.id===r.expected.id,'Original product preparation binding is required');"+body.slice(end);
  return replace(body,'checkSnapshot();','');
 });
}
