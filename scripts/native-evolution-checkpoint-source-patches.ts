/** Reuse one admitted source inside its existing exclusive guard; disk journal/CAS remains unchanged. */
import {join} from 'node:path';
import {readFile} from 'node:fs/promises';
type Context={bodies:Map<string,string>;target:(p:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string};
export function patchNativeEvolutionCommitBinding(body:string,ownerImport:string,replace:Context['replace']){
 body='import {checkpointApprovedBinding,equalCheckpointValues} from '+JSON.stringify(ownerImport)+';\n'+body;
 body=replace(body,'const body=ledgerParse(value.body);need(body.id===value.id&&body.kind===value.kind&&equal(body.binding,r.expected.binding)&&equal(body.arguments,body.binding.arguments)',
  'const body=ledgerParse(value.body);need(equalCheckpointValues(checkpointApprovedBinding(body.binding),checkpointApprovedBinding(r.expected.binding)),"Commit checkpoint typed binding differs");need(body.id===value.id&&body.kind===value.kind&&equal(body.arguments,body.binding.arguments)');
 return replace(body,'need(equal(ledgerParse(files["evolution-"+r.expected.id+".json"]!.toString()),commit.report),"Commit report artifact differs from intent");',
  'need(equalCheckpointValues(checkpointApprovedBinding({...r.expected.binding,arguments:ledgerParse(files["evolution-"+r.expected.id+".json"]!.toString())}),checkpointApprovedBinding({...r.expected.binding,arguments:commit.report})),"Commit report typed values differ from intent");');
}
export async function applyNativeEvolutionCheckpointSourcePatches({bodies,target,replace,root}:Context){
 const research=join(root,'.native-migration/research'),helper=JSON.stringify(join(root,'backend/native-evolution-held-source.ts'));
 const edit=async(name:string,change:(body:string)=>string)=>{const path=target(join(research,name));bodies.set(path,change(bodies.get(path)??await readFile(path,'utf8')));};
 await edit('native-core-product-publication-v1/commit-storage.ts',body=>patchNativeEvolutionCommitBinding(body,join(research,'native-unified-service/checkpoint-owner.ts'),replace));
 await edit('native-unified-peer/engine-peer.ts',body=>{
  body='import {readNativeEvolutionHeldSource} from '+helper+';\n'+body;
  body=replace(body,'(observation(engine,guard)!==null||equal(hashes(s.store),s.hashes))','(s.guard===guard||observation(engine,guard)!==null||equal(hashes(s.store),s.hashes))');
  return replace(body,' return peer;\n}', ' await readNativeEvolutionHeldSource(engine,guard);\n return peer;\n}');
 });
 await edit('native-unified-expert/source.ts',body=>replace(body,'const owner=Object.freeze({...held,research_label:runtime.core_label,assertUnchanged:', 'const owner=Object.freeze({...held,research_label:runtime.core_label,assertRawUnchanged:held.assertUnchanged,assertUnchanged:'));

 await edit('native-unified-core-control/install.ts',body=>{
  body='import {readNativeEvolutionHeldSource} from '+helper+';\n'+body;
  const start=body.indexOf('export async function installCoreCheckpoint('),end=body.indexOf('export async function coreConstraintResidual(',start);if(start<0||end<start)throw Error('Actual install consumer missing');
  let install=body.slice(start,end);
  install=replace(install,"assert(equal(cap.binding.active_files_sha256,await import('../native-learning-product/commit-storage.ts').then(m=>m.nativeAcceptedHashes(peer.store))));", "const heldSource=await readNativeEvolutionHeldSource(engine,guard);assert(equal(cap.binding.active_files_sha256,heldSource.pins),'Installed checkpoint physical source differs');");
  install=replace(install,'await peer.source.assertUnchanged();await peer.owner.assertUnchanged();','await heldSource.assertUnchanged();');
  install=replace(install,'cap.binding.core_version===peer.source.runtime.core_label&&cap.binding.expert_id===(peer.previous?.id??null)', "cap.binding.core_version===(heldSource.accepted.core?.version??null)&&cap.binding.expert_id===(heldSource.accepted.expert?.id??null)");
  return body.slice(0,start)+install+body.slice(end);
 });
 await edit('native-core-controller-candidate/journal-snapshot.ts',body=>{
  body='import {readNativeEvolutionHeldSource} from '+helper+';\n'+body;
  body=replace(body,'export function bindCore450Previous(','export async function bindCore450Previous(');
  body=replace(body,'const cohort=selectIndependentNativeEvolutionCohort(r.memory,r.store,r.base),pins=nativeAcceptedHashes(r.store),files=collectLearningModelFiles(r.store),accepted=validateModelFiles(files,r.base,readNativeFoundationAdmission(r.store,files));','const cohort=selectIndependentNativeEvolutionCohort(r.memory,r.store,r.base),heldSource=await readNativeEvolutionHeldSource(engine,guard,r.preparation),{pins,accepted}=heldSource;');
  // Keep the original final physical disk comparison immediately before the journal CAS.
  return body;
 });
 await edit('native-core450-owner-controller/runner.ts',body=>{
  body='import {rejectFailedNativeEvolutionJournal} from '+JSON.stringify(join(root,'backend/native-evolution-journal.ts'))+';\n'+body;
  body=replace(body,'T extends Pick<Request,"memory"|"expected">','T extends Pick<Request,"memory"|"expected"|"signal">');
  return replace(body,'  const next={...body,owner:null,paused_at:',"  if(error!=null&&!r.signal?.aborted&&phase==='paused_after_failure')return rejectFailedNativeEvolutionJournal(db,row,String(error),{owned:true,restored:true,cancelled:false});\n  const next={...body,owner:null,paused_at:");
 });
 await edit('native-core450-owner-controller/bootstrap.ts',body=>replace(body,'bindCore450Previous(engine,guard,{memory:r.memory,store:r.store,base:r.base,id,preparation:r.preparation});','await bindCore450Previous(engine,guard,{memory:r.memory,store:r.store,base:r.base,id,preparation:r.preparation});'));
 await edit('native-core450-runner/checkpoint.ts',body=>{
  body='import {readNativeEvolutionHeldSource} from '+helper+';\n'+body;
  body=replace(body,'const pins=nativeAcceptedHashes(r.store),files=collectLearningModelFiles(r.store),accepted=validateModelFiles(files,r.base,readNativeFoundationAdmission(r.store,files));','const heldSource=await readNativeEvolutionHeldSource(engine,guard,product??undefined),{pins,accepted}=heldSource;');
  // Readers own their raw directories. Close every borrower before the reader can unlink them.
  body=replace(body,'[resume!.close(),original!.close(),loaded.close(),official?.close()]','[resume!.close(),original!.close(),official?.close()]');
  body=replace(body,"throw new AggregateError(errors,'Complete Core owner closure unconfirmed');for(const f of cp.files)","throw new AggregateError(errors,'Complete Core owner closure unconfirmed');await loaded.close();for(const f of cp.files)");
  body=replace(body,'[resume?.close(),original?.close(),loaded.close(),official?.close()]','[resume?.close(),original?.close(),official?.close()]');
  body=replace(body,"throw new AggregateError(errors,'Complete Core allocation handles still retained');if(loaded.checkpoint)","throw new AggregateError(errors,'Complete Core allocation handles still retained');await loaded.close();if(loaded.checkpoint)");
  return replace(body,'assert(equal(nativeAcceptedHashes(r.store),pins));','await heldSource.assertUnchanged();');
 });
}
