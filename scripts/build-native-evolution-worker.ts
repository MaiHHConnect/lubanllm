/** Compose captured native implementations at build time; no model load, snapshot or source audit. */
import assert from 'node:assert/strict';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,join,dirname,extname,relative} from 'node:path';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {applyNativeEvolutionStoragePatches} from './native-evolution-storage-patches.ts';
import {applyNativeEvolutionResumePatches} from './native-evolution-resume-patches.ts';
import {applyNativeEvolutionLegacyExpertPatches} from './native-evolution-legacy-expert-patches.ts';
import {applyNativeEvolutionSegmentPatches} from './native-evolution-segment-patches.ts';
import {applyNativeEvolutionTrainingSourcePatches} from './native-evolution-training-source-patches.ts';
import {patchNativeEvolutionSampleFields} from './native-evolution-sample-patches.ts';
import {applyNativeEvolutionPeerPatches} from './native-evolution-peer-patches.ts';
import {applyNativeEvolutionSourceSwapPatches} from './native-evolution-source-swap-patches.ts';
import {applyNativeEvolutionAmbientPatches} from './native-evolution-ambient-patches.ts';
import {applyNativeEvolutionPriorPatches} from './native-evolution-prior-patches.ts';
import {applyNativeEvolutionReleasePatches} from './native-evolution-release-patches.ts';
import {applyNativeEvolutionTrainingHistoryPatches} from './native-evolution-training-history-patches.ts';
import {applyNativeEvolutionFreshSourcePatches} from './native-evolution-fresh-source-patches.ts';
import {applyNativeEvolutionCancelCleanupPatches} from './native-evolution-cancel-cleanup-patches.ts';
import {applyNativeEvolutionConstraintPatches} from './native-evolution-constraint-patches.ts';
import {applyNativeEvolutionOutputPatches} from './native-evolution-output-patches.ts';
import {applyNativeEvolutionCheckpointSourcePatches} from './native-evolution-checkpoint-source-patches.ts';
import {applyNativeEvolutionInstallFilePatches} from './native-evolution-install-file-patches.ts';
import {applyNativeEvolutionCalibrationPatches} from './native-evolution-calibration-patches.ts';
import {applyNativeEvolutionConversionSummaryPatches} from './native-evolution-trainer-conversion.ts';
import {applyNativeExpertLearningPatches} from './native-expert-learning-patches.ts';
import {applyNativeExpertSourcePatches} from './native-expert-source-patches.ts';
import {validateNativeExpertGraphEvidence} from '../backend/native-expert-qualification.ts';

const root=resolve(import.meta.dir,'..'),research=join(root,'.native-migration/research');
const output=resolve(process.argv[2]??join(root,'build/native-evolution'));
const parentText=await readFile(join(research,'native-evolution-python-parity-source-v5/build.json'),'utf8');
const capturedBinary=JSON.parse(parentText).binary;
assert(typeof capturedBinary==='string'&&capturedBinary.includes('/.native-migration/'),'Captured native build root missing');
const capturedRoot=capturedBinary.slice(0,capturedBinary.indexOf('/.native-migration/'));
// Relocate metadata lookup paths only; frozen source bodies, digests and qualification records stay intact.
const parseCaptured=(text:string)=>JSON.parse(text,(_key,value)=>typeof value==='string'&&value.startsWith(capturedRoot+'/')?root+value.slice(capturedRoot.length):value);
const json=async(path:string)=>{const text=await readFile(path,'utf8');return path.startsWith(research+'/')?parseCaptured(text):JSON.parse(text);};
const parent=parseCaptured(parentText);
const composition=await json(join(research,'native-core-product-job-source-v1/final-composition.json'));
const productParent=await json(composition.parent.path);
const bodies=new Map<string,string>(),aliases=new Map<string,string>([...productParent.aliases,...parent.aliases]);
// Dynamic Core's direct role import must converge with the final package authority too.
aliases.set(join(research,'native-canonical-core-source-close-role-v2/authority.ts'),join(research,'native-unified-service/compiled-authority-v6.ts'));
aliases.set(join(research,'native-canonical-core-source-close-role-v2/admission.ts'),join(research,'native-unified-service/admission-v6.ts'));
aliases.set(join(research,'native-core-peer-refresh-source-v1/reload.ts'),join(root,'backend/native-evolution-reload.ts'));
aliases.set(join(research,'native-evolution-producer/cohort.ts'),join(root,'backend/native-evolution-cohort.ts'));
aliases.set(join(research,'native-core450-owner-controller/final-observer.ts'),join(root,'backend/native-evolution-final.ts'));
aliases.set(join(research,'native-evolution-retained-producer-v1/tail.ts'),join(root,'backend/native-evolution-tail.ts'));
aliases.set(join(research,'native-core-product-final-reopen-source-v1/final-report.ts'),join(root,'backend/native-evolution-final-report.ts'));
for(const name of ['quality','artifacts','publication'])aliases.set(join(research,'native-core-product-quality-publication-source-v1',name+'.ts'),join(root,'backend/native-evolution-'+(name==='quality'?'admission':name)+'.ts'));
for(const source of productParent.sources)bodies.set(source.path,await readFile(source.snapshot,'utf8'));
for(const source of parent.sources)bodies.set(source.path,await readFile(source.snapshot,'utf8'));
const replace=(body:string,before:string,after:string)=>{
 assert.equal(body.split(before).length,2,'Ambiguous or missing native composition anchor: '+before.slice(0,100));
 return body.replace(before,after);
};
// The dynamic product parent has additional retained-source operations absent from the fixed-data v5 driver.
// Apply the observation patches to that parent instead of replacing its entire Engine/Process classes.
for(const suffix of ['native-unified-expert/native-engine.ts','native-unified-expert/expert-process.ts','native-unified-peer/engine-peer.ts','native-unified-expert/none-source.ts','native-unified-expert/core.ts','native-unified-core-control/install.ts','native-unified-expert/source.ts','native-peer-descriptor-v2/descriptor.ts','native-unified-service/model-source-v6.ts','native-unified-service/current-core-v7.ts','native-unified-service/raw-owner.ts','native-unified-service/model-owner.ts','native-product-runtime/backend/native-model-tensor.ts','../backend/native-model-tensor.ts']){
 const canonical=join(research,suffix);let path=canonical;while(aliases.has(path))path=aliases.get(path)!;
 const source=productParent.sources.find((value:any)=>value.path===path);assert(source,path);
 let body=await readFile(source.snapshot,'utf8');
 for(const folder of ['native-observation-batch-v1','native-observation-python-parity-v2','native-observation-python-parity-v3','native-model-owner-cancel-v1']){
  const map=await json(join(research,folder,'source-map.json'));
  for(const record of map.records.filter((value:any)=>value.canonical===path||value.canonical===canonical))for(const edit of record.changes??[]){
   let before=edit.before??edit.from,after=edit.after??edit.to;
   if(before==='      const control = Object.freeze({verifyCoreCheckpoint:'){
    before='      const control = Object.freeze({';after=after.slice(0,-'verifyCoreCheckpoint:'.length);
   }
   if(before==='  const execute=async()=>{this.assertLive();signal?.throwIfAborted();await this.check();'){
    before='  const execute=async()=>{this.assertLive();';after='  assertObservationCommand(this,line);const execute=async()=>{const began=performance.now();assertObservationCommand(this,line);this.assertLive();';
   }
   if(before.startsWith('async releaseSnapshotForReload(')||before.startsWith('async releaseInitialSnapshotForReload(')){
    before=before.replace('):Promise',',retained=false):Promise');after=after.replace('):Promise',',retained=false):Promise');
   }
   if(before===' const s=borrowNativeEnginePeer(engine,guard);assert(o.ids.length'){
    before=" const s=borrowNativeEnginePeer(engine,guard),thinking=o.thinking??'off';";
    after=" if(observation(engine,guard))return generateObservedPeer(engine,guard,o);"+before.trimStart();
   }
   body=replace(body,before,after);
  }
 }
 bodies.set(path,body);
}
function functionText(body:string,name:string){
 const start=body.indexOf('export async function '+name+'(');
 assert(start>=0,'Missing captured function '+name);
 const next=body.indexOf('\nexport ',start+1);
 return body.slice(start,next<0?body.length:next).trimEnd();
}
for(const mapping of composition.maps){
 const current=bodies.get(mapping.path);
 if(!current){bodies.set(mapping.path,await readFile(mapping.after,'utf8'));continue;}
 if(mapping.path.endsWith('/bootstrap.ts')||mapping.path.endsWith('/segment.ts')){
  // The dynamic bootstrap already has the correct new run ID, original seed boundary and failure owner.
  bodies.set(mapping.path,await readFile(mapping.after,'utf8'));continue;
 }
 if(mapping.path.endsWith('/runner.ts')){
  let body=await readFile(mapping.after,'utf8');
  for(const name of ['generateCore450Rows','meanCore450Loss'])body=replace(body,functionText(body,name),functionText(current,name));
  body=replace(body,"'coreConstraintResidual'] as const","'coreConstraintResidual','verifyCoreCheckpoint','withCoreObservation'] as const");
  bodies.set(mapping.path,body);continue;
 }
 let body=current;
 for(const change of mapping.changes)body=replace(body,change.from,change.to);
 bodies.set(mapping.path,body);
}
// Dynamic preparation owns the round already; the resident observation guard must share that scope.
const producer=join(research,'native-evolution-producer/producer.ts');
bodies.set(producer,"import {enterEvolutionBranch} from '../native-observation-python-parity-v2/round.ts';\n"+
 replace(bodies.get(producer)!,"try{await peer.worker.setBranchEnabled(false);const prepared=", "try{await enterEvolutionBranch(r.engine,peer.worker);const prepared="));
let producerBody=bodies.get(producer)!;
producerBody=replace(producerBody,"const expertBefore=peer.previous?await peer.worker.snapshotCurrent('core-round-expert-before-'+ ++peer.serial):null;if(expertBefore)r.owners.own(expertBefore,()=>expertBefore.close());",'');
producerBody=replace(producerBody,"if(expertBefore){const expertAfter=await peer.worker.snapshotCurrent('core-round-expert-after-'+ ++peer.serial);r.owners.own(expertAfter,()=>expertAfter.close());await assertASRExpertSnapshots(peer.worker,expertBefore,expertAfter);}",'');
producerBody=replace(producerBody,'await prepare(r,store,base,guard,control)','await prepare(r,store,base,guard,control,previous)');
producerBody=replace(producerBody,'control:Parameters<Parameters<NativeEngine["withQuiescent"]>[0]>[1]){','control:Parameters<Parameters<NativeEngine["withQuiescent"]>[0]>[1],ambient:Awaited<ReturnType<typeof control.captureRouterInitial>>){');
producerBody=replace(producerBody,'const rng=await captureRetainedEvolutionRNG(r.engine,guard,r.owners);',`// Fixture construction has no RNG consumer. Reuse the genuine rollback RNG owner here;
  // bootstrap applies seed 42 at the original training boundary and saves the actual PT0 RNG.
  const rng={cpu_rng:ambient.snapshot.cpu_rng,mps_rng:ambient.snapshot.mps_rng!,receipt:{schema:'luban-evolution-ambient-rng-v1',seed_applied:false,same_peer:true,pid:peer.worker.pid,optimizer_steps:0},assertUnchanged:async()=>{scope();peer.worker.assertLive();},close:async()=>{}};`);
producerBody=replace(producerBody,'fresh_rng:rng.receipt','preparation_rng:rng.receipt');
producerBody=replace(producerBody,';need(await checkpointRawSHA(f.handle,bytes)===sha256,"Evolution raw changed");',';');
producerBody=replace(producerBody,'await owner.assertUnchanged();await accepted.assertUnchanged();\n   for(const raw of held)await raw.assertUnchanged!();await initial.assertUnchanged();await rng.assertUnchanged();await fixture.assertUnchanged();await previousVersion.assertUnchanged();scope();','peer.worker.assertLive();\n   for(const raw of held)await raw.assertUnchanged!();scope();');
bodies.set(producer,producerBody);
for(const added of composition.added)bodies.set(added.path,await readFile(added.path,'utf8'));
const reopen=await json(join(research,'native-core-product-final-reopen-source-v1/composition.json'));
for(const mapping of reopen.maps){let body=bodies.get(mapping.path)??await readFile(mapping.before,'utf8');for(const change of mapping.changes)body=replace(body,change.from,change.to);bodies.set(mapping.path,body);}
for(const added of reopen.added)bodies.set(added.path,await readFile(added.path,'utf8'));
// Scheduling metadata is inserted in the same original journal transaction, never patched afterwards.
const automaticType="{signature:string;scheduled_occurrence:string|null}";
const jobPath=join(research,'native-core-product-job-source-v1/job.ts');
let jobBody=bodies.get(jobPath)!;
jobBody=replace(jobBody,'arguments?:Record<string,any>;outputParent:string;','arguments?:Record<string,any>;automatic?:'+automaticType+';outputParent:string;');
jobBody=replace(jobBody,'const common={memory:r.memory,','const common={automatic:r.automatic,memory:r.memory,');bodies.set(jobPath,jobBody);
const bootstrapPath=join(research,'native-core450-owner-controller/bootstrap.ts');
let bootstrapBody=bodies.get(bootstrapPath)!;
bootstrapBody=replace(bootstrapBody,"export type ProductFreshRequest=Omit<Request,'preparation'>&{preparation:NativeEvolutionPrepared;", "export type ProductFreshRequest=Omit<Request,'preparation'>&{automatic?:"+automaticType+";preparation:NativeEvolutionPrepared;");
bootstrapBody=replace(bootstrapBody,"session=await startNativeTrainingSessionV4({memory:r.memory,store:r.store,id,kind:'evolution',scope:'product',identity:r.base.identity,core_version:r.preparation.graphBinding.accepted_core_version", "session=await startNativeTrainingSessionV4({automatic:r.automatic,memory:r.memory,store:r.store,id,kind:'evolution',scope:'product',identity:r.base.identity,core_version:r.preparation.graphBinding.accepted_core_version");bodies.set(bootstrapPath,bootstrapBody);
const processPath=join(research,'native-unified-expert/expert-process.ts');
let processBody=bodies.get(processPath)!;
const tailStart=processBody.indexOf(' async exportTrainingTail('),tailEnd=processBody.indexOf(' async exportHidden4(',tailStart);assert(tailStart>=0&&tailEnd>tailStart);
const ordinaryMethod=processBody.slice(tailStart,tailEnd).replace('exportTrainingTail(','exportOrdinaryTrainingTail(').replace('this.trainingTailExport&&','this.ordinaryTrainingTailExport&&').replace("['training-tail',","['training-tail-ordinary',");
processBody=replace(processBody,'private trainingTailExport=false;','private trainingTailExport=false;private ordinaryTrainingTailExport=false;');
processBody=replace(processBody,'this.trainingTailExport=r.training_tail_export===true;','this.trainingTailExport=r.training_tail_export===true;this.ordinaryTrainingTailExport=r.ordinary_training_tail_export===true;');
processBody=replace(processBody,' async exportHidden4(',ordinaryMethod+' async exportHidden4(');bodies.set(processPath,processBody);
processBody=replace(processBody,'m.optimizer_unchanged===true&&m.rng_unchanged===true','m.optimizer_unchanged===true&&(m.ordinary===true?m.rng_unchanged===null:m.rng_unchanged===true)');
processBody=replace(processBody,' async exportHidden4(',` get coreObservationMode():'previous'|'current'{return this.state.observation==='previous'?'previous':'current';}
 async selectCoreObservation(mode:'previous'|'current',signal?:AbortSignal){
  assert(this.ordinaryTrainingTailExport&&this.hasInstalledCore&&(mode==='previous'||mode==='current'),'Resident observation selection unavailable');
  const result=await this.command('core-observation '+mode,signal),ack=result.response;
  assert(ack.observation===mode&&/^[a-f0-9]{64}$/.test(ack.core_tensor_sha256)&&Number.isInteger(ack.step)&&ack.step>=0&&ack.step<=450);
  return{actual:true as const,pid:this.pid,observation:mode,core_tensor_sha256:ack.core_tensor_sha256 as string,step:ack.step as number,optimizer_step:ack.optimizer_step as number};
 }
 async exportHidden4(`);bodies.set(processPath,processBody);
const scopePath=join(research,'native-observation-python-parity-v2/scope.ts');
bodies.set(scopePath,replace(bodies.get(scopePath)!,"['run','loss'].includes(line.split(' ',1)[0]!)","((s.value as {selection_command?:boolean})?.selection_command===true?['core-observation']:['run','loss','training-tail-ordinary']).includes(line.split(' ',1)[0]!)"));
// The cohort wrapper freezes and binds materials once for the round; per-row rights/input validation remains.
const inputPath=join(research,'native-evolution-retained-producer-v1/train-input.ts');
bodies.set(inputPath,replace(bodies.get(inputPath)??await readFile(inputPath,'utf8'),'cohort.assertUnchanged();assertNativePrefixTrainingPermission(sample);','assertNativePrefixTrainingPermission(sample);'));

// These compiler substitutions are fixed. No runtime flags select legacy bodies or bypass guards.
function target(path:string){const seen=new Set<string>();while(aliases.has(path)){assert(!seen.has(path),'Native alias cycle');seen.add(path);path=aliases.get(path)!;}return path;}
// Fixture preparation and execution must borrow the same admitted v6 trainer role.
const fixturePath=target(join(research,'native-evolution-fixture/exporter.ts'));
bodies.set(fixturePath,replace(bodies.get(fixturePath)??await readFile(fixturePath,'utf8'),'import {verifyNativeTrainingMPSPackage} from "../../backend/native-training-mps-package.ts";','import {verifyNativeTrainingMPSPackage} from "../native-unified-service/training-programs-v6.ts";'));
const startPath=target(join(root,'.native-migration/backend/native-training-start-v4.ts'));let startBody=bodies.get(startPath)??await readFile(startPath,'utf8');
startBody=replace(startBody,"scope:'product'|'research';identity:string;","automatic?:"+automaticType+";scope:'product'|'research';identity:string;");
startBody=replace(startBody,'const timestamp=Date.now()/1000,body={id,kind:r.kind,',`const automatic=input.automatic?{...input.automatic}:null;need(!automatic||r.kind==='evolution'&&digest(automatic.signature)&&(automatic.scheduled_occurrence===null||/^schedule:\\d{4}-\\d{2}-\\d{2}$/.test(automatic.scheduled_occurrence)),'Native automatic metadata invalid');
 const timestamp=Date.now()/1000,body={...(automatic?{automatic_signature:automatic.signature,scheduled_occurrence:automatic.scheduled_occurrence}:{}),id,kind:r.kind,`);bodies.set(startPath,startBody);
const issuer=parent.final_issuer.path;
const authority=JSON.parse(bodies.get(issuer)!.replace(/^export const SERVICE_SOURCE_AUTHORITY\s*=\s*/,'').replace(/(?: as const)?;?\s*$/,''));
const runtime=await json(join(output,'runtime-authority.json'));
bodies.set(target(join(research,'native-unified-service/compiled-authority-v6.ts')),'export const RELEASE_AUTHORITY = '+JSON.stringify(runtime.authority)+';\n');
const cachePath=target(join(research,'native-unified-service/cache-build-admission-v6.ts'));
bodies.set(cachePath,replace(bodies.get(cachePath)!,"resolve(dirname(process.execPath),'../Resources')","resolve(import.meta.dir,'Resources')"));
const fdPath=target(join(research,'native-unified-service/process-fds-v6.ts'));
bodies.set(fdPath,replace(bodies.get(fdPath)!,"resolve(dirname(process.execPath),'../Resources',A.paths.processFD)","resolve(import.meta.dir,'Resources',A.paths.processFD)"));
const parserPath=target(join(research,'native-unified-service/python-upper-v6.ts'));
bodies.set(parserPath,replace(bodies.get(parserPath)!,"join(dirname(process.execPath), '../Resources', A.paths.syntaxParser)","join(import.meta.dir, 'Resources', A.paths.syntaxParser)"));
const admissionPath=target(join(research,'native-unified-service/admission-v6.ts'));
const admission=bodies.get(admissionPath)!;assert.equal((admission.match(/bytes:size\},\d+\)/g)??[]).length,1);
bodies.set(admissionPath,admission.replace(/bytes:size\},\d+\)/,'bytes:size},'+runtime.manifestBytes+')'));
authority.native_manifest_sha256=runtime.authority.manifest_sha256;
authority.native_role_source_proofs={...runtime.authority.sources};
bodies.set(issuer,'export const SERVICE_SOURCE_AUTHORITY='+JSON.stringify(authority)+' as const;\n');
const enginePath=join(research,'native-unified-expert/native-engine.ts');
let engineBody=bodies.get(enginePath)!;
engineBody=replace(engineBody,"import {withCoreObservation} from '../native-observation-python-parity-v2/adapter.ts';",`import {withNativeCoreObservation as withCoreObservation,selectNativeCoreObservation} from ${JSON.stringify(join(root,'backend/native-evolution-observation.ts'))};`);
engineBody=replace(engineBody,'    withCoreObservation:<T>',"    selectCoreObservation:(cap:CoreCheckpoint,mode:'current'|'previous',options?:{signal?:AbortSignal})=>ReturnType<typeof selectNativeCoreObservation>;\n    withCoreObservation:<T>");
engineBody=replace(engineBody,'      const control = Object.freeze({','      const control = Object.freeze({selectCoreObservation:(cap:CoreCheckpoint,mode:"current"|"previous",options?:{signal?:AbortSignal})=>selectNativeCoreObservation(this,guard,cap,mode,options),');
bodies.set(enginePath,patchNativeEvolutionSampleFields(engineBody,replace));
const effectiveImport=JSON.stringify(join(root,'backend/native-effective-core.ts'));
const filesPath=target(join(research,'native-product-runtime/backend/evolution-snapshot-files.ts'));
let filesBody=bodies.get(filesPath)??await readFile(filesPath,'utf8');
filesBody="import {applyNativeEffectiveCore} from "+effectiveImport+";\n"+replace(filesBody,'return { core, expert, router };','return applyNativeEffectiveCore(files,{core,expert,router});');bodies.set(filesPath,filesBody);
const commitPath=target(join(research,'native-core-product-publication-v1/commit-storage.ts'));let commitBody=bodies.get(commitPath)??await readFile(commitPath,'utf8');
commitBody="import {assertNativeEffectiveCoreReload} from "+effectiveImport+";\n"+commitBody;
commitBody=replace(commitBody,"need(receipt.restore_id===r.expected.id&&receipt.loaded.core_loaded===(pointer!==null),'Commit actual retained reload identity differs');","need(receipt.restore_id===r.expected.id,'Commit actual retained reload identity differs');");
commitBody=replace(commitBody,"need(receipt.loaded.core_file_sha256===pointer.sha256,'Commit actual retained reload complete Core file differs');",'');
commitBody=replace(commitBody,"need(selected.core?.version===pointer.version&&receipt.loaded.core_tensor_sha256===selected.core.tensor_sha256,'Commit actual retained Core tensor differs');",'assertNativeEffectiveCoreReload(pointer,receipt,selected);');bodies.set(commitPath,commitBody);
await applyNativeEvolutionStoragePatches({bodies,target,replace,root});
const memoryPath=target(join(research,'native-product-runtime/backend/memory.ts'));
let memoryBody=bodies.get(memoryPath)??await readFile(memoryPath,'utf8');
memoryBody='import {initializeMemoryDatabase} from '+JSON.stringify(join(root,'backend/memory-database.ts'))+';\n'+memoryBody;
memoryBody=replace(memoryBody,'db.exec("PRAGMA busy_timeout=20000; PRAGMA locking_mode=EXCLUSIVE;");\n            if (!readonly) db.exec(SCHEMA);','initializeMemoryDatabase(db,readonly,SCHEMA);');
bodies.set(memoryPath,memoryBody);
await applyNativeEvolutionResumePatches({bodies,target,replace,root});
await applyNativeEvolutionTrainingSourcePatches({bodies,target,replace,root});
await applyNativeEvolutionLegacyExpertPatches({bodies,target,replace,root});
await applyNativeEvolutionSegmentPatches({bodies,target,replace,root});
await applyNativeEvolutionPeerPatches({bodies,target,replace,root});
applyNativeEvolutionReleasePatches({bodies,target,replace,root});
await applyNativeEvolutionPriorPatches({bodies,target,replace,root});
await applyNativeEvolutionAmbientPatches({bodies,target,replace,root});
await applyNativeEvolutionSourceSwapPatches({bodies,target,replace,root});
await applyNativeEvolutionTrainingHistoryPatches({bodies,target,replace,root});
await applyNativeEvolutionFreshSourcePatches({bodies,target,replace,root});
await applyNativeEvolutionCheckpointSourcePatches({bodies,target,replace,root});
await applyNativeEvolutionOutputPatches({bodies,target,replace,root});
await applyNativeEvolutionCancelCleanupPatches({bodies,target,replace,root});
await applyNativeEvolutionInstallFilePatches({bodies,target,replace,root});
applyNativeEvolutionConstraintPatches({bodies,target,replace,root});
await applyNativeEvolutionCalibrationPatches({bodies,target,replace,root});
await applyNativeEvolutionConversionSummaryPatches({bodies,target,replace,root});
const identityImportBuild=process.env.LUBAN_NATIVE_RESEARCH_ENTRY?.endsWith('/import-native-identity-candidate.ts')===true;
if(!identityImportBuild){
 aliases.set(join(research,'native-learning-product/publication.ts'),join(root,'backend/native-expert-publication.ts'));
 await applyNativeExpertLearningPatches({bodies,target,replace,root});
 await applyNativeExpertSourcePatches({bodies,target,replace,root});
}
const expertQualificationPath=join(root,'backend/native-expert-qualification.ts');
let expertQualifications=[];
try{expertQualifications=await json(join(output,'expert-qualification.json'));}catch(error:any){if(error.code!=='ENOENT')throw error;}
assert(Array.isArray(expertQualifications));expertQualifications.forEach(validateNativeExpertGraphEvidence);
bodies.set(expertQualificationPath,replace(await readFile(expertQualificationPath,'utf8'),"'BUILD_BOUND_NATIVE_EXPERT_QUALIFICATION'",JSON.stringify(JSON.stringify(expertQualifications))));
const workerPath=join(root,'backend/native-evolution-worker.ts'),q4Build=await json(join(root,'build/native/luban-evolution-observe.build.json'));
assert(/^[a-f0-9]{64}$/.test(q4Build.binary_sha256));
bodies.set(workerPath,replace(await readFile(workerPath,'utf8'),'BUILD_BOUND_Q4_OBSERVER_SHA',q4Build.binary_sha256));
const entry=process.env.LUBAN_NATIVE_RESEARCH_ENTRY?resolve(process.env.LUBAN_NATIVE_RESEARCH_ENTRY):join(root,'backend/native-evolution-worker.ts'),loaded=new Map<string,string>();
if(process.env.LUBAN_NATIVE_TYPECHECK==='1'){
 const ts=createRequire(import.meta.url)('/Users/linhao/.npm-global/lib/node_modules/typescript/lib/typescript.js');
 const options={noEmit:true,strict:true,skipLibCheck:true,resolveJsonModule:true,allowImportingTsExtensions:true,module:ts.ModuleKind.ESNext,moduleResolution:ts.ModuleResolutionKind.Bundler,target:ts.ScriptTarget.ESNext,typeRoots:['/opt/homebrew/lib/node_modules/@soimy/dingtalk/node_modules/@types'],types:['bun','node'],lib:['lib.esnext.d.ts','lib.dom.d.ts','lib.dom.iterable.d.ts']};
 const host=ts.createCompilerHost(options),read=host.readFile,exists=host.fileExists;
 host.readFile=(path:string)=>bodies.get(resolve(path))??read(path);host.fileExists=(path:string)=>bodies.has(resolve(path))||exists(path);
 host.resolveModuleNames=(names:string[],containing:string)=>names.map(name=>{const found=ts.resolveModuleName(name,containing,options,host).resolvedModule;return found?{...found,resolvedFileName:target(resolve(found.resolvedFileName))}:found;});
 const diagnostics=ts.getPreEmitDiagnostics(ts.createProgram([entry],options,host));
 assert.equal(diagnostics.length,0,ts.formatDiagnosticsWithColorAndContext(diagnostics,{getCanonicalFileName:(path:string)=>path,getCurrentDirectory:()=>root,getNewLine:()=> '\n'}));
}
const digest=(text:string)=>createHash('sha256').update(text).digest('hex');
async function bundle(){
 loaded.clear();return Bun.build({entrypoints:[entry],target:'bun',format:'esm',plugins:[{name:'native-evolution-composition',setup(builder){
  builder.onResolve({filter:/^\./},args=>{let path=resolve(dirname(args.importer),args.path);if(!extname(path))path+='.ts';return{path:target(path)};});
  builder.onLoad({filter:/\.(ts|js|json)$/},async args=>{
   const body=bodies.get(args.path)??await readFile(args.path,'utf8');
   loaded.set(args.path,body);return{contents:body,loader:extname(args.path).slice(1) as 'ts'|'js'|'json'};
  });
 }}]});
}
let result=await bundle();assert(result.success,result.logs.map(String).join('\n'));
const discovery=[...loaded.keys()].sort();
if(!process.env.LUBAN_NATIVE_RESEARCH_ENTRY)assert(loaded.get(target(join(research,'native-ordinary-snapshot-base-v1/resolver.ts')))?.includes('const retained=readNativeRetainedSource(engine);'),'Actual product snapshot resolver must reuse the retained evolution source');
// This is the implementation binding required by the existing journal, generated from build inputs only.
authority.service_source_manifest=Object.fromEntries([...loaded].filter(([path])=>path!==issuer).map(([path,body])=>[relative(root,path),digest(body)]));
authority.source_manifest_scope='Native dynamic product worker with resident Python-parity observation; build only, no model or quality qualification';
bodies.set(issuer,'export const SERVICE_SOURCE_AUTHORITY='+JSON.stringify(authority)+' as const;\n');
result=await bundle();assert(result.success&&result.outputs.length===1,result.logs.map(String).join('\n'));
assert.deepEqual([...loaded.keys()].sort(),discovery);
for(const suffix of ['native-unified-expert/native-engine.ts','native-unified-expert/expert-process.ts','native-unified-peer/engine-peer.ts'])assert.equal([...loaded.keys()].filter(path=>path.endsWith(suffix)).length,1,'Duplicated resident implementation: '+suffix);
await mkdir(output,{recursive:true});
const outputName=process.env.LUBAN_NATIVE_RESEARCH_ENTRY?(entry.endsWith('/verify-native-expert-preservation.ts')?'expert-preservation.js':entry.endsWith('/native-evolution-selection.ts')?'selection.js':entry.endsWith('/verify-native-source-swap.ts')?'source-swap.js':entry.endsWith('/verify-native-retention-loss.ts')?'retention-loss.js':entry.endsWith('/verify-native-q4-candidate.ts')?'q4-candidate.js':entry.endsWith('/import-native-identity-candidate.ts')?'identity-import.js':entry.endsWith('/verify-native-evolution-fixture.ts')?'fixture-check.js':'math.js'):'worker.js',outputRaw=Buffer.from(await result.outputs[0]!.arrayBuffer());
await writeFile(join(output,outputName),outputRaw);
await writeFile(join(output,process.env.LUBAN_NATIVE_RESEARCH_ENTRY?outputName.replace('.js','-build.json'):'build.json'),JSON.stringify({schema:'luban-native-evolution-worker-build-v1',entry,modules:loaded.size,worker_sha256:createHash('sha256').update(outputRaw).digest('hex'),binary_sha256:createHash('sha256').update(outputRaw).digest('hex'),runtime_manifest_sha256:runtime.authority.manifest_sha256,source_manifest:authority.service_source_manifest,python_observer:false,training_target:identityImportBuild?"identity-import":"expert",candidate_max_steps:64,quality_early_stop:true,legacy_core_fixed_steps:450,dynamic_materials:true,model_loaded:false,quality_admitted:false,product_admitted:false},null,2)+'\n');
console.log(JSON.stringify({worker:join(output,outputName),modules:loaded.size,model_loaded:false,quality_admitted:false}));
