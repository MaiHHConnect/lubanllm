/** Native expert learning overlays; public routing/build integration is deliberately separate. */
import {readFile} from 'node:fs/promises';
import {join} from 'node:path';
type Context={bodies:Map<string,string>;target:(path:string)=>string;replace:(body:string,before:string,after:string)=>string;root:string};
export async function applyNativeExpertLearningPatches({bodies,target,replace,root}:Context){
 const helper=join(root,'backend/native-expert-learning.ts'),qualification=join(root,'backend/native-expert-qualification.ts');
 const edit=async(path:string,change:(body:string)=>string)=>{path=target(join(root,'.native-migration/research',path));bodies.set(path,change(bodies.get(path)??await readFile(path,'utf8')));};
 await edit('native-expert-compute/compute.ts',body=>replace(body,"learning_rate:hasPrevious?.0002:.002","learning_rate:.0002").replace("learning_rate:.002}]:[])","learning_rate:.0002}]:[])"));
 await edit('native-unified-expert/expert-process.ts',body=>{
  body=replace(body,'const reloadInitialSnapshots=new WeakMap','const snapshotOrdinals=new WeakMap<ExpertCurrentSnapshot|ExpertInitialSnapshot,number>();\nconst reloadInitialSnapshots=new WeakMap');
  body=replace(body,'initialSnapshots.set(cap,state);await cap.assertUnchanged();return cap;','initialSnapshots.set(cap,state);await cap.assertUnchanged();snapshotOrdinals.set(cap,this.ordinal);return cap;');
  body=replace(body,'snapshots.set(cap,state);await cap.assertUnchanged();return cap;','snapshots.set(cap,state);await cap.assertUnchanged();snapshotOrdinals.set(cap,this.ordinal);return cap;');
  // A genuine snapshot is already the final native command. No model/RNG mutation may intervene.
  for(const [name,label]of [['snapshotCurrent','chosen'],['snapshotInitial','previous']] as const){
   const marker="const check=await this."+name+"('release-",start=body.indexOf(marker),end=body.indexOf('}finally{await check.close();}',start);
   if(start<0||end<0||body.indexOf(marker,start+1)>=0)throw Error('Expert '+label+' release anchor missing');
   // The product parent additionally has if(!retained){...}; preserve that owner/close branch exactly.
   body=body.slice(0,start)+"assert.equal(snapshotOrdinals.get(snapshot),this.ordinal,'Resident changed after its actual "+label+" snapshot');"+body.slice(end+'}finally{await check.close();}'.length);
  }
  body=replace(body,' async trainDescriptor(input:{',' async clearPreservation(){const result=await this.command("preservation-clear");assert.equal(result.response.preservation_rows,0);return result;}\n async capturePreservation(input:{path:string;sha256:string;labels:{path:string;sha256:string};name:string;signal?:AbortSignal}){this.name(input.name);const result=await this.command(["preservation",JSON.stringify(input.path),input.sha256,JSON.stringify(input.labels.path),input.labels.sha256,input.name].join(" "),input.signal);assert(Number.isSafeInteger(result.response.preservation_rows)&&result.response.preservation_rows>=1);return result;}\n async trainDescriptor(input:{');
  body=replace(body,'steps:number;name:string;signal?:AbortSignal}){','steps:number;name:string;signal?:AbortSignal;preservation?:true;audit?:true}){');
  body=replace(body,"this.command(['train',JSON.stringify(input.path)","this.command([input.preservation?(input.audit?'train-preserved-trace':'train-preserved'):'train',JSON.stringify(input.path)");
  return body;
 });
 await edit('native-unified-expert/expert-scope.ts',body=>{
  body='import {assertNativeExpertPreservation,nativeExpertHasParent} from '+JSON.stringify(helper)+';\nimport {readNativeExpertGraphQualification} from '+JSON.stringify(qualification)+';\n'+body;
  body=replace(body,'readonly productionModelQualified:false;','readonly productionModelQualified:boolean;');
  body=replace(body,' assertUnchanged():Promise<void>;replaceInitialized',' preparePreservation(rows:readonly Record<string,any>[]):Promise<void>;observationState():{pid:number;step:number;branch_enabled:boolean};\n assertUnchanged():Promise<void>;replaceInitialized');
  body=replace(body,"const state={engine,guard,closed:false,closing:false,worker:held};let scope!",`const graphQualified=(width:number)=>readNativeExpertGraphQualification({identity:options.base.identity,core_tensor_sha256:accepted.core?.tensor_sha256??options.base.original_down_sha256,graph_binary_sha256:build.binary_sha256,source_build_sha256:build.proof.source_build_sha256,hidden:2048,width});
  let teachersReady=false;const state={engine,guard,closed:false,closing:false,worker:held};let scope!`);
  body=replace(body,'productionModelQualified:false as const,get current()',"get productionModelQualified(){try{graphQualified(current?.approval.width??(nativeExpertHasParent(options.previous,options.current.binding.core_version)?options.previous.approval.width:128));return true;}catch{return false;}},get current()");
  body=replace(body,'   seedFresh:async()=>',`   observationState:()=>{held.assertLive();return{pid:held.pid,step:held.status.optimizer_step,branch_enabled:branchEnabled};},
   preparePreservation:async(rows:readonly Record<string,any>[])=>{await live();assert(resourceState==='previous'&&rows.length>0,'Teachers must be captured from the restored previous model');teachersReady=false;await held.clearPreservation();const was=branchEnabled;try{for(const row of rows){const teacher=assertNativeExpertPreservation(row);assert(teacher!=='expert'||nativeExpertHasParent(options.previous,options.current.binding.core_version),'Current-core parent teacher missing');const enabled=teacher==='expert';if(branchEnabled!==enabled){await held.setBranchEnabled(enabled);branchEnabled=enabled;}const s=await sample(row);await held.capturePreservation({...s,name:s.id+'-teacher'});}teachersReady=true;}finally{if(branchEnabled!==was){await held.setBranchEnabled(was);branchEnabled=was;}}},
   seedFresh:async()=>`);
  body=replace(body,"assert(typeof o.reuse==='boolean');resourceState='none';", "assert(typeof o.reuse==='boolean'&&(!o.reuse||nativeExpertHasParent(options.previous,options.current.binding.core_version)),'Cannot reuse an expert from a different core');graphQualified(o.width);assert(teachersReady,'Candidate needs frozen core/parent KL targets');resourceState='none';");
  body=replace(body,"assert(branchEnabled&&resourceState==='candidate','Training requires the actual candidate branch');", "assert(branchEnabled&&resourceState==='candidate'&&teachersReady&&o.learning_rate===.0002,'Training requires the actual protected candidate branch');");
  body=replace(body,'held.trainDescriptor({...s,learning_rate:o.learning_rate','held.trainDescriptor({...s,preservation:true,learning_rate:o.learning_rate');
  return body;
 });
 await edit('native-learning-product/quality.ts',body=>{
  body='import {freezeNativeExpertPreservation,nativeExpertQualityBoundary} from '+JSON.stringify(helper)+';\n'+body;
  body=replace(body,'coreCheckpointField as checkpointField','coreCheckpointMetadata as checkpointMetadata,coreCheckpointField as checkpointField');
  body=replace(body,'receipt.current.approval.step===64','nativeExpertQualityBoundary(receipt.current.approval.step)');
  body=replace(body,'const original=saved?await readSavedLearningBaseline(saved,records):null;const baseline=',"const original=saved?await readSavedLearningBaseline(saved,records):null;const savedRows=saved?checkpointMetadata(checkpointField(checkpointField(saved.control,'report')!,'preservation_rows')!):undefined;const preservationRows=freezeNativeExpertPreservation(records.protection,[...core,...branch],savedRows as any);const baseline=");
  body=replace(body,'observations:Object.freeze({saved_threshold_source:','observations:Object.freeze({preservation_rows:preservationRows,saved_threshold_source:');
  body=replace(body,'current.approval.step===64&&current.approval.hidden','nativeExpertQualityBoundary(current.approval.step)&&current.approval.hidden');
  const from=body.indexOf('const before=await scope.snapshotCurrent();',body.indexOf('export async function qualifyLearningCandidate'));
  if(from<0)throw Error('Learning quality owner anchor missing');
  // Same process only performs read-only generation/loss here. No additional full-state/RNG snapshots.
  body=body.slice(0,from)+`const before=scope.observationState();
 assert(before.pid===scope.pid&&before.step===(current.approval.scope==='checkpoint-stored-candidate-only'?0:current.approval.step)&&before.branch_enabled,'Quality candidate state differs');
 const score=await mean(engine,scope,records.valid),replay=await mean(engine,scope,records.train.slice(1));assert(score);
 const training=await generate(engine,scope,records.train),generation=await generate(engine,scope,records.valid),protection=await generate(engine,scope,records.protection),preservation=await generate(engine,scope,baseline.observations.preservation_rows as Record<string,any>[]);
 const decision=learningCandidateDecision(current.approval.width,score.loss,baseline.score,replay?.loss??null,baseline.replay,[...training,...generation],[...protection,...preservation]);
 assert.deepEqual(scope.observationState(),before,'Read-only quality changed the resident candidate');await current.assertUnchanged();qualified(engine,scope);
 const observations=Object.freeze({width:current.approval.width,actual_steps:current.approval.step,score:score.loss,replay:replay?.loss??null,...decision,training_generation:training,generation_checks:generation,core_retention_generation:protection,preservation_generation:preservation,preservation_rows:baseline.observations.preservation_rows,loss_observations:score.observations,replay_observations:replay?.observations??[],pid:scope.pid,checkpoint_sha256:current.approval.checkpoint_sha256,optimizer_file_sha256:current.owner.file_sha256,records_sha256:current.binding.records_sha256,product_admitted:false});
 if(!decision.accepted)return{observations,quality:null};const receipt:LearningQuality=Object.freeze({observations,current,async assertUnchanged(){validateLearningQuality(engine,receipt);await current.assertUnchanged();await scope.assertUnchanged();}});issued.set(receipt,{engine,scope,json:JSON.stringify(observations)});await receipt.assertUnchanged();return{observations,quality:receipt};
}
`;
  return body;
 });
 await edit('native-learning-product/lifecycle.ts',body=>{
  body='import {nativeExpertAccumulatedStep,nativeExpertHasParent,nativeExpertSessionChoices} from '+JSON.stringify(helper)+';\n'+body;
  body=replace(body,'private readonly records:Records){','private readonly records:Records,private readonly onProgress?:(p:{step:number;phase:string})=>void){');
  body=replace(body,'status(){return{state:this.state,...this.progress,error:this.error,product_admitted:false};}',"status(){return{state:this.state,...this.progress,candidate_step:this.progress.step,step:nativeExpertAccumulatedStep(metadata(this.candidates) as Record<string,any>[],this.progress.step),steps:nativeExpertSessionChoices(this.previous,this.base.binding.core_version).length*64,error:this.error,product_admitted:false};}");
  body=replace(body,'const baseline=await observeLearningBaseline(this.engine,this.scope,this.records,coreProtection,branchProtection,fresh,fresh?undefined:this.base),choices=learningChoices(this.previous.approval.width,this.previous.owner!==null);',"const baseline=await this.scope.withBranchEnabled(nativeExpertHasParent(this.previous,this.base.binding.core_version),()=>observeLearningBaseline(this.engine,this.scope,this.records,coreProtection,branchProtection,fresh,fresh?undefined:this.base)),choices=nativeExpertSessionChoices(this.previous,this.base.binding.core_version);");
  body=replace(body,'this.base=next;this.held.add(next);this.claimRunning();return next;','this.base=next;this.held.add(next);this.claimRunning();this.onProgress?.({step:nativeExpertAccumulatedStep(metadata(this.candidates) as Record<string,any>[],this.progress.step),phase:this.progress.phase});return next;');
  body=replace(body,"if(!fresh&&!this.base.owner)await this.scope.restoreInitial(this.base);const first=", "await this.scope.preparePreservation(baseline.observations.preservation_rows as Record<string,any>[]);set(this.report,'preservation_rows',learningMetadata(baseline.observations.preservation_rows,this.base.binding));if(!fresh&&!this.base.owner)await this.scope.restoreInitial(this.base);const first=");
  body=replace(body,'const choice=choices[i]!;','const choice=choices[i]!;let earlyEvaluation:Awaited<ReturnType<typeof qualifyLearningCandidate>>|undefined;');
  body=replace(body,'await this.scope.restoreCurrent(this.base);','await this.scope.restoreCurrent(this.base);if(this.progress.step>0&&this.progress.step%8===0)earlyEvaluation=await qualifyLearningCandidate(this.engine,this.scope,this.base,this.records,baseline);');
  body=replace(body,'while(this.progress.step<64){','while(this.progress.step<64&&!earlyEvaluation?.quality){');
  body=replace(body,'for(let i=first;i<choices.length;i++){',"for(let i=first;i<choices.length&&!(metadata(this.candidates) as Record<string,any>[]).some(row=>row.accepted===true);i++){");
  body=replace(body,'if(this.progress.step%8===0||this.pauseRequested)await this.save(true);}',"if(this.progress.step%8===0||this.pauseRequested){await this.save(true);if(!this.pauseRequested&&!this.cancelRequested){assert(this.base.owner);earlyEvaluation=await qualifyLearningCandidate(this.engine,this.scope,this.base,this.records,baseline);if(earlyEvaluation.quality)break;}}}");
  body=replace(body,'assert(this.base.owner&&this.base.approval.step===64);const evaluated=await qualifyLearningCandidate','assert(this.base.owner&&this.base.approval.step===this.progress.step);const evaluated=earlyEvaluation??await qualifyLearningCandidate');
  body=replace(body,"await this.save(false);fresh=false;","await this.save(false);fresh=false;if(evaluated.quality)break;");
  body=replace(body,'reuse:false,learning_rate:.002','reuse:false,learning_rate:.0002');
  body=replace(body,"async restorePrevious(){await this.scope.restorePrevious(this.previous);if(this.previous.owner){const current=await this.scope.snapshotCurrent();try{assert.equal(current.owner.file_sha256,this.previous.owner.file_sha256);assert.equal(current.width,this.previous.approval.width);assert.equal(current.step,0);assert.equal(current.optimizer_empty,true);await current.assertUnchanged();}finally{await current.close();}}await this.previous.assertUnchanged();}",
   "async restorePrevious(){const restored=await this.scope.restorePrevious(this.previous) as {response:{optimizer_step:number;resident_parameters:number}};assert.equal(restored.response.optimizer_step,0,'Previous restore must acknowledge empty Adam');assert.equal(restored.response.resident_parameters,this.previous.owner?3*this.scope.hidden*this.previous.approval.width:0,'Previous restore must acknowledge the actual expert or None');await this.previous.assertUnchanged();}");
  return body;
 });
 await edit('native-learning-product/snapshot.ts',body=>{
  body='import {nativeExpertSessionChoices} from '+JSON.stringify(helper)+';\n'+body;
  body=replace(body,'coreCheckpointField as field','coreCheckpointMetadata as metadata,coreCheckpointField as field');
  return replace(body,"choices=learningChoices(previousWidth,previous.kind!=='none')","choices=nativeExpertSessionChoices({owner:previous.kind==='none'?null:true,approval:{width:previousWidth,...metadata(field(control,'previous_flags')!) as {core_version:string|null;branch_core_version:string|null}}},base.binding.core_version)");
 });
 await edit('native-expert-session-mps/session.ts',body=>{
  body='import {nativeExpertSessionChoices} from '+JSON.stringify(helper)+';\n'+body;
  return replace(body,"const choices=expertCandidateWidths(r.previous_width,previous.kind!=='none');","const choices=nativeExpertSessionChoices({owner:previous.kind==='none'?null:true,approval:{width:r.previous_width,...metadata(flags) as {core_version:string|null;branch_core_version:string|null}}},r.expected.binding.core_version);");
 });
 await edit('native-learning-product/current.ts',body=>replace(body,"requires_gate:gate.value,core_version:metadata(get(flags,'core_version')!)as string|null,branch_core_version:metadata(get(flags,'branch_core_version')!)as string|null,", "requires_gate:true,core_version:r.expected.binding.core_version as string|null,branch_core_version:r.expected.binding.core_version as string|null,"));
 await edit('native-learning-product/stored-candidate.ts',body=>{
  body='import {nativeExpertQualityBoundary} from '+JSON.stringify(helper)+';\n'+body;
  body=replace(body,'candidate_index:number;step:64;','candidate_index:number;step:number;');
  body=replace(body,'const approval=Object.freeze({id:',"assert(nativeExpertQualityBoundary(record.actual_steps),'Stored history actual update count missing');const approval=Object.freeze({id:");
  body=replace(body,'candidate_index:index,step:64 as const','candidate_index:index,step:Number(record.actual_steps)');
  body=replace(body,"requires_gate:gate.value,core_version:metadata(get(flags,'core_version')!)as string|null,branch_core_version:metadata(get(flags,'branch_core_version')!)as string|null,","requires_gate:true,core_version:base.binding.core_version as string|null,branch_core_version:base.binding.core_version as string|null,");
  return body;
 });
}
