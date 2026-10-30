/** Expert source ownership, Q4-before-intent publication, and original transactional recovery. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
type Context={bodies:Map<string,string>;target:(path:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string};
export async function applyNativeExpertSourcePatches({bodies,target,replace,root}:Context){
 const backend=(name:string)=>JSON.stringify(join(root,'backend',name));
 const edit=async(path:string,change:(body:string)=>string)=>{const p=target(join(root,'.native-migration/research',path));bodies.set(p,change(bodies.get(p)??await readFile(p,'utf8')));};
 await edit('native-learning-product/controller.ts',body=>patchNativeExpertController(body,replace,root));
 await edit('native-product-runtime/backend/native-evolution-cohort.ts',body=>patchNativeExpertCohortPreview(body,replace,root));
 await edit('native-learning-product/cohort.ts',body=>{
  body=replace(body,"for(const run of runs){const b=ledgerParse(run.body);", "for(const run of runs){const b=ledgerParse(run.body);if(b.binding?.core_version!==(core.core?.version??null))continue;");
  body=replace(body,"arguments:{steps:64}","arguments:{steps:64,protocol:'native-expert-preservation-v1'}");
  body=replace(body,'const coreProtection=manifest.records.protection',"manifest.arguments={...manifest.arguments,protocol:'native-expert-preservation-v1'};const coreProtection=manifest.records.protection");return body;
 });
 await edit('native-unified-peer/engine-peer.ts',body=>{
  body=replace(body,'type Peer={guard:', 'type Peer={accepted:ReturnType<typeof validateModelFiles>;guard:');
  return replace(body,'peer={guard,lock:', 'peer={accepted,guard,lock:');
 });
 await edit('native-unified-expert/expert-process.ts',body=>{
  const marker=' async restoreReloadInitialSnapshot(';assert(body.includes(marker));
  body+=`\nexport function assertClosedExpertReload(cap:ExpertReloadSnapshot|ExpertReloadInitialSnapshot){if(cap.owner){validateExpertReloadSnapshot(cap);assert(reloadSnapshots.get(cap)!.source.isClosed);}else{validateExpertReloadInitialSnapshot(cap);assert(reloadInitialSnapshots.get(cap)!.source.isClosed);}}\n`;
  return body.replace(marker,` adoptClosedEmptyReloadSource(cap:ExpertReloadInitialSnapshot){validateExpertReloadInitialSnapshot(cap);const state=reloadInitialSnapshots.get(cap)!;assert(state.source.isClosed&&this.source===cap&&this.status.optimizer_step===0&&this.status.resident_parameters===0,'None adoption requires closed transfer and actual new-worker restore');this.source=undefined;}\n`+marker);
 });
 await edit('native-unified-expert/native-engine.ts',body=>{
  body=`import {reloadClosedNativeExpert} from ${backend('native-expert-reload.ts')};\n`+body;
  body=replace(body,'expertSnapshot?:ExpertReloadSnapshot|ExpertReloadInitialSnapshot) => Promise<NativeReloadReceipt>','expertSnapshot?:ExpertReloadSnapshot|ExpertReloadInitialSnapshot,prepared?:NativeShallowPreparation) => Promise<NativeReloadReceipt>');
  body=replace(body,'expertSnapshot?:ExpertReloadSnapshot|ExpertReloadInitialSnapshot): Promise<NativeReloadReceipt>','expertSnapshot?:ExpertReloadSnapshot|ExpertReloadInitialSnapshot,prepared?:NativeShallowPreparation): Promise<NativeReloadReceipt>');
  const begin=body.indexOf('captureLearningInitialSession:async('),end=body.indexOf('},get expert()',begin);assert(begin>=0&&end>begin);
  let part=body.slice(begin,end);const pattern=/if\(!hasNativeEnginePeer\(this\)\)\{const router=await control\.captureRouterInitial\(await mkdtemp\(join\(directory,["']initial-peer-["']\)\)\);await router\.close\(\);\}/;assert(pattern.test(part),'Learning initial peer audit anchor missing');part=part.replace(pattern,'need(!hasNativeEnginePeer(this),"Initial standalone capture cannot share a resident model");');body=body.slice(0,begin)+part+body.slice(end);
  body=replace(body,'need(hasNativeEnginePeer(this)&&!scopedGenerationBusy&&(captured||captureRouting==="source")', 'if(hasNativeEnginePeer(this))await closeNativeEnginePeer(this,guard);need(!hasNativeEnginePeer(this)&&!scopedGenerationBusy&&(captured||captureRouting==="source")');
  const reload=body.indexOf('reloadCurrent: async (store: string');assert(reload>=0);const text=body.slice(reload),anchor='guard.assertQuiescent(); controlled = true; hold = true;';
  const inserted=replace(text,anchor,anchor+'\n          if(expertSnapshot){need(prepared&&shallowPreparations.has(prepared),"Closed expert reload requires this scope original prepared sample");const receipt=await reloadClosedNativeExpert(this,guard,store,base,restoreId,expertSnapshot,prepared);hold=false;return receipt;}');
  // Expert transfers now return above through the real standalone-to-peer restoration.
  const obsolete=inserted.indexOf('          if(expertSnapshot?.owner)'),obsoleteEnd=inserted.indexOf('\n',obsolete);assert(obsolete>=0&&obsoleteEnd>obsolete);
  let result=inserted.slice(0,obsolete)+inserted.slice(obsoleteEnd);
  const noneStart=result.indexOf('if(expertSnapshot&&!expertSnapshot.owner)'),noneEnd=result.indexOf('need(!expertSnapshot,"Unexpected expert transfer for empty accepted target");',noneStart);assert(noneStart>=0&&noneEnd>noneStart);
  result=result.slice(0,noneStart)+result.slice(noneEnd+'need(!expertSnapshot,"Unexpected expert transfer for empty accepted target");'.length);
  return body.slice(0,reload)+result;
 });
 await edit('native-unified-expert/expert-scope.ts',body=>{
  body='import {nativeExpertSourceCheck} from '+backend('native-expert-source.ts')+';\n'+body;
  body=replace(body,"const validateBinding=(cap:","const checkSource=nativeExpertSourceCheck(options.store,files),physicalPins=nativeAcceptedHashes(options.store);const validateBinding=(cap:");
  body=body.replaceAll('nativeAcceptedHashes(options.store)', 'physicalPins');
  body=replace(body,'physicalPins=physicalPins','physicalPins=nativeAcceptedHashes(options.store)');
  const a=body.indexOf(' const checkFiles=()=>{'),b=body.indexOf('\n let source:',a);assert(a>=0&&b>a);body=body.slice(0,a)+" const checkFiles=()=>{assertNativeExpertScopeGuard(engine,guard);checkSource();};"+body.slice(b);
  body=replace(body,"catch(e){failure??=e;}}for(const s of snapshots)","catch(e){failure??=e;workerClosed=held.isClosed;if(!workerClosed)throw e;}}for(const s of snapshots)");return body;
 });
 await edit('native-unified-expert/accepted-previous.ts',body=>{
  body='import {nativeExpertSourceCheck} from '+backend('native-expert-source.ts')+';\n'+body;
  body=replace(body,' let pins=Object.fromEntries', ' const sourceCheck=nativeExpertSourceCheck(store,files);let pins=Object.fromEntries');
  return replace(body,"const latest=collectModelFiles(store);assert(equal(Object.fromEntries(Object.entries(latest).map(([k,v])=>[k,sha(v)])),pins),'Accepted previous full captured cohort changed');assert(equal(validateModelFiles(latest,base,readNativeFoundationAdmission(store,latest)),accepted));",'sourceCheck();');
 });
 await edit('native-learning-commit-recovery/restore.ts',body=>{
  body='import {nativeExpertSourceCheck} from '+backend('native-expert-source.ts')+';\nimport {closeImportOwner} from '+backend('native-evolution-import-observe.ts')+';\nimport {validateClosedNativeExpertReload} from '+backend('native-expert-reload.ts')+';\n'+body;
  body=replace(body,"const assertSource=()=>{assertNativeExpertScopeGuard(engine,r.guard);assert(equal(nativeAcceptedHashes(r.store),r.expected.binding.active_files_sha256));};", "const sourceCheck=nativeExpertSourceCheck(r.store,files),assertSource=()=>{assertNativeExpertScopeGuard(engine,r.guard);sourceCheck();};");
  body=replace(body,'assertNativeExpertScopeGuard(engine,r.guard);assert(receipt.restore_id', 'validateClosedNativeExpertReload(engine,r.guard,receipt);assertNativeExpertScopeGuard(engine,r.guard);assert(receipt.restore_id');
  return replace(body,"const results=await Promise.allSettled([scope?.close(),transfer?.close(),initial?.close(),previous?.close(),commit.close()]),errors=results.filter((r):r is PromiseRejectedResult=>r.status==='rejected').map(r=>r.reason);if(errors.length){control.holdUnloaded();throw new AggregateError(errors,'Pending Learning previous/PID/RNG cleanup unconfirmed');}","await closeImportOwner(async()=>{await scope?.close();await transfer?.close();await initial?.close();await previous?.close();await commit.close();},error=>{control.holdUnloaded();console.error('Pending Learning previous/PID/RNG cleanup unconfirmed: '+String(error));});");
 });
 const checkpointPath=target(join(root,'.native-migration/backend/native-checkpoint.ts'));let checkpointBody=bodies.get(checkpointPath)??await readFile(checkpointPath,'utf8');
 checkpointBody+='\nexport async function assertNativeExpertCheckpointFiles(cap:VerifiedNativeCheckpoint){const guard=finalGuards.get(cap);need(guard,"Expert checkpoint must retain the original verified reader owner");await guard();}\n';bodies.set(checkpointPath,checkpointBody);
 const checkpointImport='import {assertNativeExpertCheckpointFiles} from "../../backend/native-checkpoint.ts";\n';
 await edit('native-learning-product/initial.ts',body=>checkpointImport+replace(body,'await validateExpertLearningCheckpoint(cp,r);for(const f of cp.files)assert.equal(await checkpointRawSHA(f.handle,f.bytes,f.dtype),f.sha256);','await assertNativeExpertCheckpointFiles(cp);'));
 for(const [file,variable] of [['current.ts','cp'],['previous.ts','checkpoint']] as const)await edit('native-learning-product/'+file,body=>{
  const old=variable==='cp'?"await validateExpertLearningCheckpoint(cp,r);await held.assertUnchanged();for(const f of cp.files)assert.equal(await checkpointRawSHA(f.handle,f.bytes,f.dtype),f.sha256);":"await validateExpertLearningCheckpoint(checkpoint,r);await weights?.assertUnchanged();for(const raw of checkpoint.files)assert.equal(await checkpointRawSHA(raw.handle,raw.bytes,raw.dtype),raw.sha256);";
  const replacement=variable==='cp'?"await assertNativeExpertCheckpointFiles(cp);await held.assertUnchanged();":"await assertNativeExpertCheckpointFiles(checkpoint);await weights?.assertUnchanged();";return checkpointImport+replace(body,old,replacement);
 });
 await edit('native-learning-product/commit-storage.ts',body=>patchNativeExpertCommitStorage(body,replace));
 // Keep automatic metadata in the original creation transaction, with the existing validation.
 const start=target(join(root,'.native-migration/backend/native-training-start-v4.ts'));let body=bodies.get(start)!;
 body=replace(body,"!automatic||r.kind==='evolution'&&digest(automatic.signature)","!automatic||['evolution','learning'].includes(r.kind)&&digest(automatic.signature)");bodies.set(start,body);
}

/** The scheduler lends a registered frozen view, while execution retains real Memory ownership. */
export function patchNativeExpertCohortPreview(body:string,replace:Context['replace'],root:string){
 return 'import {isNativeEvolutionPreviewMemory} from '+JSON.stringify(join(root,'backend/native-evolution-preview.ts'))+';\n'+
  replace(body,'memory instanceof Memory&&memory.model_identity===base.identity',
   '(memory instanceof Memory||isNativeEvolutionPreviewMemory(memory))&&memory.model_identity===base.identity');
}

export function patchNativeExpertController(body:string,replace:Context['replace'],root:string){
 const path=(name:string)=>JSON.stringify(join(root,'backend',name));
 body=`import {nativeExpertSourceCheck} from ${path('native-expert-source.ts')};\nimport {materialMessages} from ${path('native-evolution-material.ts')};\nimport {validateClosedNativeExpertReload} from ${path('native-expert-reload.ts')};\nimport type {NativeExpertQ4Options} from ${path('native-expert-publication-q4.ts')};\n`+body;
 body='import {closeImportOwner} from '+path('native-evolution-import-observe.ts')+';\n'+body;
 body=replace(body,'prepareLearningPublication,applyLearningPublication,','prepareLearningPublication,learningPublicationTransfer,qualifyLearningPublication,markLearningPublicationReleased,applyLearningPublication,');
 body=replace(body,'constructor(private readonly chat:()=>NativeChat){}',"constructor(private readonly chat:()=>NativeChat,private readonly options:{q4:NativeExpertQ4Options;onProgress?:(value:{step:number;phase:string})=>void;readonly automatic?:{signature:string;scheduled_occurrence:string|null}}){}\n async wait(){await this.work;return this.status();}");
 body=replace(body,'this.work=undefined;this.session=undefined;',"if(this.session){const {step,steps,candidate,candidate_step}=this.session.status();this.job={...this.job,step,steps,candidate,candidate_step};}this.work=undefined;this.session=undefined;");
 body=replace(body,'return{...this.job,...session,state:', 'return{...this.job,...session,error:this.job.error??session?.error??null,state:');
 body=body.replaceAll('engine.withQuiescent(async(guard,control)=>{','engine.withSourceQuiescent(async(guard,control)=>{').replaceAll('},true);','});');
 body=replace(body,"release={path:join(dirname(process.execPath),'../Resources',A.manifest_relative),sha256:A.manifest_sha256}","release=engine.config.expert_build!");
 body=replace(body,"const resources=join(dirname(process.execPath),'../Resources'),release={path:join(resources,A.manifest_relative),sha256:A.manifest_sha256}","const release=engine.config.expert_build!");
 body=replace(body,"const outputParent=await mkdtemp(join(store,'Learning-work-')),reader=", "const reloadTokens=await control.prepareRouteTokens(materialMessages(cohort.records.train[0]) as any);const outputParent=await mkdtemp(join(this.options.q4.work,'Learning-work-')),reader=");
 body=replace(body,'const body=ledgerParse(row.body);const restored=await restorePendingLearningCommit(engine,control,', 'const body=ledgerParse(row.body),prepared=await control.prepareRouteTokens(materialMessages(body.records.train[0]) as any),recoveryControl={...control,reloadCurrent:(store:string,base:any,id:string,transfer?:ExpertReloadSnapshot|ExpertReloadInitialSnapshot)=>control.reloadCurrent(store,base,id,transfer,prepared)};const restored=await restorePendingLearningCommit(engine,recoveryControl,');
 body=replace(body,"b.arguments.steps===64&&", "b.arguments.steps===64&&b.arguments.protocol==='native-expert-preservation-v1'&&");
 const begin=body.indexOf('}else{const snapshots=new EvolutionSnapshots('),end=body.indexOf('    journal=await startNativeTrainingSessionV4(',begin);assert(begin>=0&&end>begin);body=body.slice(0,begin)+'}else{\n'+body.slice(end);
 body=replace(body,"scope:'product',identity:base.identity,", "scope:'product',automatic:this.options.automatic,identity:base.identity,");
 body=replace(body,'backup_id:backup.id,','');body=replace(body,'learning_rate:floatValue(capture.previous_id===null?.002:.0002)','learning_rate:floatValue(.0002)');
 body=replace(body,"const assertSource=()=>{guard.assertQuiescent();cohort.assertUnchanged();assert(equal(nativeAcceptedHashes(store),captured),'学习期间已采纳来源变化');};assertSource();","const sourceCheck=nativeExpertSourceCheck(store,files),assertSource=()=>{guard.assertQuiescent();sourceCheck();};assertSource();");
 body=replace(body,'outputParent},cohort.records);','outputParent},cohort.records,this.options.onProgress);');
 body=body.replaceAll('control.reloadCurrent(store,base,id,transfer)','control.reloadCurrent(store,base,id,transfer,reloadTokens)');
 body=replace(body,'this.commitInFlight=true;publication=await prepareLearningPublication(engine,this.session,outcome.selected,{memory,store,base,guard});prepared=true;const s=await scope.snapshotCurrent();transfer=await scope.releaseForCommit(s);await scope.close();applyLearningPublication(engine,publication);',
  'publication=await prepareLearningPublication(engine,this.session,outcome.selected,{memory,store,base,guard,records:cohort.records,source:accepted,assertSource});transfer=await scope.releaseForCommit(learningPublicationTransfer(engine,publication));markLearningPublicationReleased(engine,publication,transfer);await scope.close();await qualifyLearningPublication(engine,publication,this.options.q4);prepared=true;this.commitInFlight=true;applyLearningPublication(engine,publication);');
 body=replace(body,'if(publication&&prepared){rejectLearningPublication(engine,publication,String(error));', 'if(publication){if(prepared)rejectLearningPublication(engine,publication,String(error));');
 // Recovery works for both pre-intent Q4 rejection and an applied pointer transaction.
 body=body.replaceAll("const receipt=await control.reloadCurrent(store,base,id,transfer,reloadTokens);this.verifyPrevious(receipt,accepted,base,id);", "const receipt=await control.reloadCurrent(store,base,id,transfer,reloadTokens);validateClosedNativeExpertReload(engine,guard,receipt);this.verifyPrevious(receipt,accepted,base,id);");
 // Never release a transferred checkpoint before the real worker which borrowed it has closed.
 body=replace(body,"const actions=[()=>publication?.close(),()=>transfer?.close(),()=>this.session?.close(),()=>scope?.close(),()=>!this.session?initial?.close():undefined,()=>!this.session?previous?.close():undefined,()=>journal?.close(),()=>programs.close()];const results=await Promise.allSettled(actions.map(action=>Promise.resolve().then(action))),errors=results.filter((r):r is PromiseRejectedResult=>r.status==='rejected').map(r=>r.reason);if(errors.length){control.holdUnloaded();this.job.state='blocked';throw new AggregateError(errors,'Learning owner/PID/RNG cleanup unconfirmed');}",
 "await closeImportOwner(async()=>{await scope?.close();await this.session?.close();await publication?.close();await transfer?.close();if(!this.session){await initial?.close();await previous?.close();}await journal?.close();await programs.close();},error=>{control.holdUnloaded();this.job.state='blocked';this.job.error='Learning owner/PID/RNG cleanup unconfirmed: '+String(error);});");
 return body;
}

export function patchNativeExpertCommitStorage(body:string,replace:Context['replace']){
  body='import {equalCheckpointValues,checkpointApprovedBinding} from "../../backend/native-checkpoint.ts";\n'+body;
  body=replace(body,'equal(body.binding,r.expected.binding)','equalCheckpointValues(checkpointApprovedBinding(body.binding),checkpointApprovedBinding(r.expected.binding))');
  return replace(body,'need(equal(ledgerParse(files["learning-"+r.expected.id+".json"]!.toString()),commit.report),"Commit report artifact differs from intent");',
   'need(equalCheckpointValues(checkpointApprovedBinding({...r.expected.binding,arguments:ledgerParse(files["learning-"+r.expected.id+".json"]!.toString())}),checkpointApprovedBinding({...r.expected.binding,arguments:commit.report})),"Learning commit report typed values differ from intent");');

}
