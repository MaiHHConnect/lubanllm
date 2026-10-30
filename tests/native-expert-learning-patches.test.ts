import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {applyNativeExpertLearningPatches} from '../scripts/native-expert-learning-patches.ts';
import {patchNativeExpertPreservationCommand,writeNativeExpertPreservationSources} from '../scripts/native-expert-preservation-source.ts';
import {nativeExpertChoices,nativeExpertAccumulatedStep,nativeExpertHasParent,nativeExpertSessionChoices} from '../backend/native-expert-learning.ts';
const root=resolve(import.meta.dir,'..');
const replace=(body:string,before:string,after:string)=>{assert.equal(body.split(before).length,2,before);return body.replace(before,after);};
async function sources(){const bodies=new Map<string,string>();await applyNativeExpertLearningPatches({bodies,target:p=>p,replace,root});return bodies;}
test('actual captured product parent and observation guards retain the original close branch',async()=>{
 const composition=JSON.parse(await readFile(join(root,'.native-migration/research/native-core-product-job-source-v1/final-composition.json'),'utf8'));
 const parent=JSON.parse(await readFile(composition.parent.path,'utf8')),path=join(root,'.native-migration/research/native-unified-expert/expert-process.ts');
 const source=parent.sources.find((value:any)=>value.path===path);assert(source);
 let body=await readFile(source.snapshot,'utf8');
 const observation=JSON.parse(await readFile(join(root,'.native-migration/research/native-observation-batch-v1/source-map.json'),'utf8'));
 for(const record of observation.records.filter((row:any)=>row.canonical===path))for(const edit of record.changes??[]){
  let before=edit.before??edit.from,after=edit.after??edit.to;
  if(!['async releaseSnapshotForReload(','async releaseInitialSnapshotForReload(','async trainDescriptor('].some(prefix=>before.startsWith(prefix)))continue;
  if(before.startsWith('async release')){before=before.replace('):Promise',',retained=false):Promise');after=after.replace('):Promise',',retained=false):Promise');}
  body=replace(body,before,after);
 }
 const bodies=new Map([[path,body]]);await applyNativeExpertLearningPatches({bodies,target:p=>p,replace,root});const actual=bodies.get(path)!;
 expect(()=>new Bun.Transpiler({loader:'ts',target:'bun'}).transformSync(actual)).not.toThrow();
 expect(actual).not.toContain("this.snapshotCurrent('release-check-");expect(actual).not.toContain("this.snapshotInitial('release-initial-");
 expect(actual.match(/if\(!retained\)\{if\(this.options.device==='mps'\)await this.clearBackbone\(\);await this.close\(\);\}/g)?.length).toBe(2);
 expect(actual).toContain('audit?:true}){assertObservationMutationAllowed(this);assert(Number.isInteger(input.steps)');
});
test('actual frozen learning consumers parse with preservation and early-step history',async()=>{
 const bodies=await sources();const transpiler=new Bun.Transpiler({loader:'ts',target:'bun'});
 for(const [path,body]of bodies){expect(()=>transpiler.transformSync(body)).not.toThrow();expect(path.includes('.native-migration')).toBe(true);}
 const process=bodies.get(join(root,'.native-migration/research/native-unified-expert/expert-process.ts'))!;
 expect(process).toContain("input.audit?'train-preserved-trace':'train-preserved'");
 const quality=bodies.get(join(root,'.native-migration/research/native-learning-product/quality.ts'))!;
 expect(quality).not.toContain('scope.snapshotCurrent()');expect(quality).toContain('records.train');
});
test('actual persistent command and optimizer overlays keep CE-only oracle and require protected production train',async()=>{
 const actual=await readFile(join(root,'build/native-evolution/PersistentEvolution.cpp'),'utf8');
 const proof=JSON.parse(await readFile(join(root,'build/native-evolution/persistent-source-build.json'),'utf8'));
 const edits=proof.source_inverse.edits as {before:string;after:string;occurrences:number}[];
 const at=edits.findIndex(edit=>edit.before==='else if(op=="train"||op=="loss"){');assert(at>=0,'Actual build must record its expert command overlay');
 const expertEdits=edits.slice(at);assert.equal(expertEdits.length,6);
 let original=actual;
 for(const edit of expertEdits.toReversed()){assert.equal(edit.occurrences,1);original=replace(original,edit.after,edit.before);}
 const output=patchNativeExpertPreservationCommand(original,replace);
 expect(output).toBe(actual);
 expect(output).toContain('op=="train-preserved"');expect(output).toContain('optimizer.capturePreservation(branchEnabled?expert.get():nullptr');
 const directory=await mkdtemp(join(tmpdir(),'luban-expert-preservation-'));
 try{const value=await writeNativeExpertPreservationSources(root,directory,join(root,'build/native-evolution/EvolutionOptimizer.h'));const body=await readFile(value.optimizer,'utf8');
  expect(value.records[0]!.inverse_exact).toBe(true);expect(body).toContain('loss=loss+5.0*kl');expect(body).toContain('preservation[step%preservation.size()]');
  expect(body).toContain('!preservation.empty()&&lr==.0002');
 }finally{await rm(directory,{recursive:true});}
});
function tree(value:any):any{if(value===null)return{kind:'none'};if(Array.isArray(value))return{kind:'list',items:value.map(tree)};if(typeof value==='object')return{kind:'dict',entries:Object.entries(value).map(([key,v])=>[{kind:'string',value:key},tree(v)])};return{kind:typeof value==='number'?'float':typeof value==='boolean'?'bool':'string',value};}
function metadata(value:any):any{if(value.kind==='none')return null;if(value.kind==='list')return value.items.map(metadata);if(value.kind==='dict')return Object.fromEntries(value.entries.map(([k,v]:any)=>[k.value,metadata(v)]));return value.value;}
test.each([false,true])('actual lifecycle stops saved8 candidate without extra resumed Adam (resume=%s)',async(resume)=>{
 const bodies=await sources(),body=bodies.get(join(root,'.native-migration/research/native-learning-product/lifecycle.ts'))!;
 const declaration=body.slice(body.indexOf('export class NativeLearningSession')).replace('export class','class');
 const compiled=new Bun.Transpiler({loader:'ts',target:'bun'}).transformSync(declaration+'\nNativeLearningSession;');
 const setters=(v:any,name:string,value:any)=>{const old=v.entries.find(([key]:any)=>key.value===name);if(old)old[1]=value;else v.entries.push([{kind:'string',value:name},value]);};
 const bindings={assert,nativeExpertHasParent,nativeExpertSessionChoices,floatValue:(v:number)=>v,validateNativeExpertLearningScope:()=>{},observeLearningBaseline:async()=>({score:1,replay:1,observations:{preservation_rows:[]}}),learningChoices:nativeExpertChoices,
  learningMetadata:tree,metadata,set:setters,field:(v:any,k:string)=>v.entries.find(([key]:any)=>key.value===k)[1],
  qualifyLearningCandidate:async(_engine:any,_scope:any,current:any)=>({quality:{current},observations:{accepted:true,objective:.1,actual_steps:current.approval.step}}),
  learningChosenCandidate:(values:any[])=>values.findIndex(row=>row.accepted)};
 const Class=new Function(...Object.keys(bindings),compiled+'\nreturn NativeLearningSession;')(...Object.values(bindings));
 const session:any=Object.create(Class.prototype);let updates=resume?8:0,prepared=0,checks=0;
 Object.assign(session,{engine:{},scope:{productionModelQualified:true,pid:17,withBranchEnabled:async(_on:boolean,run:()=>unknown)=>run(),restorePrevious:async()=>{},restoreCurrent:async()=>{},preparePreservation:async()=>{prepared++;},initializeCandidate:async()=>{},trainSample:async()=>({pid:17,step:++updates,loss:1})},previous:{approval:{width:128},owner:null},records:{train:Array.from({length:16},(_,i)=>({id:i}))},
  report:tree({}),candidates:tree([]),losses:[],progress:{candidate:0,step:updates,steps:64,phase:'queued'},base:{binding:{},approval:{step:updates},owner:resume?{}:null,control:tree({weights:null})},receipts:new Map(),currentStates:new Map(),claimRunning:()=>{},
  save:async(include:boolean)=>{session.base={binding:{},owner:include?{}:null,approval:{step:session.progress.step},control:tree({weights:{three:'actual-candidate-placeholder'}})};checks++;}});
 const result=await session.run([],[],!resume);
 expect(updates).toBe(8);expect(prepared).toBe(1);expect(checks).toBe(resume?1:3);expect(result.state).toBe('validated');
 expect(metadata(session.candidates)[0].actual_steps).toBe(8);expect(nativeExpertAccumulatedStep(metadata(session.candidates),0)).toBe(8);
});
test.each(['current','initial'])('actual release %s consumes last native snapshot without a duplicate export',async(kind)=>{
 const body=(await sources()).get(join(root,'.native-migration/research/native-unified-expert/expert-process.ts'))!;
 const name=kind==='current'?'releaseSnapshotForReload':'releaseInitialSnapshotForReload';
 const start=body.indexOf(' async '+name+'('),end=body.indexOf('\n async ',start+1);
 const declaration='class TestProcess{'+body.slice(start,end)+'}';
 const code=new Bun.Transpiler({loader:'ts',target:'bun'}).transformSync(declaration);
 const snapshotOrdinals=new WeakMap(),reloadSnapshots=new WeakMap(),reloadInitialSnapshots=new WeakMap();
 const bindings={assert,snapshotOrdinals,reloadSnapshots,reloadInitialSnapshots,validateExpertCurrentSnapshot:()=>{},validateExpertInitialSnapshot:()=>{},validateExpertReloadSnapshot:(v:object)=>assert(reloadSnapshots.get(v)?.source.isClosed),validateExpertReloadInitialSnapshot:(v:object)=>assert(reloadInitialSnapshots.get(v)?.source.isClosed),checkpointRawSHA:async()=> 'rng'};
 const Class=new Function(...Object.keys(bindings),code+'\nreturn TestProcess;')(...Object.values(bindings));
 const snapshot={step:0,optimizer_empty:true,owner:kind==='current'?{handle:{fd:1},assertUnchanged:async()=>{}}:null,previous_width:null,cpu_rng:{handle:{fd:2},bytes:5056,sha256:'rng'},mps_rng:null,assertUnchanged:async()=>{},close:async()=>{}};
 const worker=Object.assign(new Class(),{ordinal:7,options:{device:'mps'},status:{optimizer_step:0,resident_parameters:0},isClosed:false,
  clearBackbone:async()=>{worker.ordinal++;},close:async()=>{worker.isClosed=true;},snapshotCurrent:()=>{throw Error('extra export');},snapshotInitial:()=>{throw Error('extra export');}});
 snapshotOrdinals.set(snapshot,7);const proof=await worker[name](snapshot);
 expect(worker.isClosed).toBe(true);expect((kind==='current'?reloadSnapshots:reloadInitialSnapshots).has(proof)).toBe(true);
 worker.isClosed=false;worker.ordinal=9;await expect(worker[name](snapshot)).rejects.toThrow('Resident changed');expect(worker.isClosed).toBe(false);
 worker.ordinal=7;worker.close=async()=>{throw Error('actual close unconfirmed');};await expect(worker[name](snapshot)).rejects.toThrow('actual close unconfirmed');expect(worker.isClosed).toBe(false);
});
test('actual previous restore uses native ACK and leaves the one necessary transfer export to release',async()=>{
 const body=(await sources()).get(join(root,'.native-migration/research/native-learning-product/lifecycle.ts'))!;
 const start=body.indexOf(' async restorePrevious(){'),end=body.indexOf('\n async close()',start);
 const code=new Bun.Transpiler({loader:'ts',target:'bun'}).transformSync('class Restore{'+body.slice(start,end)+'}');
 const Class=new Function('assert',code+'\nreturn Restore;')(assert);
 for(const parent of [false,true]){
  let checks=0;const worker=Object.assign(new Class(),{previous:{owner:parent?{}:null,approval:{width:64},assertUnchanged:async()=>{checks++;}},scope:{hidden:2048,restorePrevious:async()=>({response:{optimizer_step:0,resident_parameters:parent?3*2048*64:0}}),snapshotCurrent:()=>{throw Error('duplicate export');}}});
  await worker.restorePrevious();expect(checks).toBe(1);
  worker.scope.restorePrevious=async()=>({response:{optimizer_step:1,resident_parameters:parent?3*2048*64:0}});await expect(worker.restorePrevious()).rejects.toThrow('empty Adam');
  worker.scope.restorePrevious=async()=>({response:{optimizer_step:0,resident_parameters:parent?0:3*2048*64}});await expect(worker.restorePrevious()).rejects.toThrow('actual expert or None');expect(checks).toBe(1);
 }
});
