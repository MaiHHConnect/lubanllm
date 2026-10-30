/** Preserve the real one-parameter source transaction; omit migration-only tensor/RNG exports. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
type Replace=(body:string,before:string,after:string)=>string;
export function patchEvolutionCoreSourceSwap(original:string){
 let body=original;const edits:{before:string;after:string;position:number}[]=[];
 const change=(before:string,after:string)=>{assert.equal(body.split(before).length,2,before);edits.push({before,after,position:body.indexOf(before)});body=body.replace(before,after);};
 change('Tensor oldDown,oldAccepted,newDown,cpuBefore,mpsBefore;','Tensor oldDown,oldAccepted,newDown;');
 change('  need(captureNativeCPURNG().equal(cpuBefore),"Core source swap changed full CPU RNG");if(device.is_mps())need(captureNativeMPSRNG().equal(mpsBefore),"Core source swap changed full MPS RNG");','');
 change('  cpuBefore=captureNativeCPURNG();if(device.is_mps())mpsBefore=captureNativeMPSRNG();for(const auto&[name,tensor]:weights)', '  for(const auto&[name,tensor]:weights)');
 change('   if(installedCandidate){auto candidate=installedCandidate->detach().cpu().contiguous();need(candidate.scalar_type()==at::kBFloat16&&candidate.sizes()==raw.sizes()&&hashMemory(candidate.data_ptr(),candidate.nbytes())==sha,"Installed full candidate differs from approved new Core");}', '   // The newly saved source is loaded and validated below; no second resident candidate readback.');
 const auditStart=body.indexOf('  output=fresh(target);'),auditEnd=body.indexOf('\n }\n void apply()',auditStart);
 assert(auditStart>=0&&auditEnd>auditStart);change(body.slice(auditStart,auditEnd),'  output=target;');
 const start=body.indexOf(' void apply(){'),end=body.indexOf(' void rollback(){',start);assert(start>=0&&end>start);
 change(body.slice(start,end),` std::string acknowledgement(){
  unchanged();need(applied,"Core source acknowledgement requires actual installation");
  need(weights.at(key).const_data_ptr()==newDown.const_data_ptr()&&acceptedDown.const_data_ptr()==newDown.const_data_ptr(),"Accepted source is not the installed resident parameter");
  const auto expected=next.config.at("core_version")=="null"?next.config.at("original_down_sha256"):next.config.at("core_tensor_sha256");
  std::ostringstream receipt;receipt<<"{\\\"protocol\\\":\\\"luban-retained-core-source-v2\\\",\\\"pid\\\":"<<getpid()<<",\\\"device\\\":"<<jsonString(device.str())<<",\\\"descriptor_sha256\\\":"<<jsonString(descriptorSHA)<<",\\\"base_identity\\\":"<<jsonString(next.config.at("identity"))<<",\\\"checkpoint_sha256\\\":"<<jsonString(next.config.at("checkpoint_sha256"))<<",\\\"manifest_sha256\\\":"<<jsonString(next.config.at("manifest_sha256"))<<",\\\"config_sha256\\\":"<<jsonString(next.config.at("config_sha256"))<<",\\\"core_version\\\":"<<jsonString(next.config.at("core_version"))<<",\\\"actual_core_tensor_sha256\\\":"<<jsonString(expected)<<",\\\"actual_core_bytes\\\":"<<newDown.nbytes()<<",\\\"actual_core_shape\\\":["<<newDown.size(0)<<','<<newDown.size(1)<<"],\\\"same_official_storage\\\":true,\\\"resident_core_readback\\\":false,\\\"parent_rng_snapshot\\\":false,\\\"official_graph_reloaded\\\":false,\\\"qualification\\\":false}";return receipt.str();
 }
 void apply(){
  unchanged();need(!applied,"Core source swap already applied");oldFile=std::move(core);core=std::move(nextFile);initial=next;weights.at(key)=newDown;acceptedDown=newDown;applied=true;
  try{unchanged();}catch(...){auto original=std::current_exception();try{rollback();}catch(const std::exception& restore){throw std::runtime_error(std::string("Core source apply and same-owner rollback failed: ")+restore.what());}std::rethrow_exception(original);}
 }
`);
 change('if(cpuBefore.defined())applyNativeCPURNG(cpuBefore);if(mpsBefore.defined())applyNativeMPSRNG(mpsBefore);if(prepared)unchanged();','if(prepared)unchanged();');
 let inverse=body;for(const edit of edits.toReversed()){assert.equal(inverse.slice(edit.position,edit.position+edit.after.length),edit.after);inverse=inverse.slice(0,edit.position)+edit.before+inverse.slice(edit.position+edit.after.length);}assert.equal(inverse,original);
 return{body,edits,inverse_exact:true};
}
export async function writeEvolutionSourceSwapSources(root:string,output:string){
 const research=join(root,'.native-migration/research'),records:any[]=[];
 const source=join(research,'native-core-retained-source-adopt-v1/CoreSourceSwap.h'),original=await readFile(source,'utf8'),patched=patchEvolutionCoreSourceSwap(original),swap=join(output,'EvolutionCoreSourceSwap.h');
 await writeFile(swap,patched.body);records.push({source,path:swap,original,body:patched.body,edits:patched.edits,inverse_exact:patched.inverse_exact});
 const dispatchSource=join(research,'native-canonical-core-source-close-role-v2/CoreSourceDispatch.h'),dispatch=await readFile(dispatchSource,'utf8');
 const header=join(output,'EvolutionCoreSourceDispatch.h'),before='#include "../native-core-retained-source-adopt-v1/CoreSourceSwap.h"',after='#include '+JSON.stringify(swap);assert.equal(dispatch.split(before).length,2);
 const checkBefore='if(currentCore)currentCore->check();',checkAfter='if(currentCore)currentCore->source->check();';
 assert.equal(dispatch.split(checkBefore).length,2);const dispatchBody=dispatch.replace(before,after).replace(checkBefore,checkAfter);
 assert.equal(dispatchBody.replace(checkAfter,checkBefore).replace(after,before),dispatch);
 await writeFile(header,dispatchBody);records.push({source:dispatchSource,path:header,original:dispatch,body:dispatchBody,edits:[{before,after},{before:checkBefore,after:checkAfter}],inverse_exact:true});
 const persistent=await readFile(join(research,'native-canonical-core-source-close-role-v2/PersistentMPSV3.cpp'),'utf8');
 const a=persistent.indexOf('    in>>std::quoted(path)>>hash>>name;need(bool(in)&&!(in>>extra)&&isSHA(hash)&&backbone&&acceptedDown.defined()'),b=persistent.indexOf('   }else if(op=="core-verify")',a);assert(a>=0&&b>a);
 const sourceCommands=persistent.slice(a,b);
 return{header,records,changes:[
  {before:'if(op=="core-source-stage"){',after:`if(op=="retain-backbone"){
    in>>std::quoted(path)>>hash;need(bool(in)&&!(in>>extra)&&isSHA(hash)&&!currentCore,"Backbone retention requires the exact accepted source with no installed candidate");auto spec=descriptor(path,hash);
    if(!backbone)backbone=std::make_unique<ExpertBackbone>(spec);backbone->check(spec);if(!acceptedDown.defined())acceptedDown=backbone->weights.at(PREFIX+"layers.23.mlp.down_proj.weight");
    need(backbone->weights.at(PREFIX+"layers.23.mlp.down_proj.weight").const_data_ptr()==acceptedDown.const_data_ptr(),"Retained accepted parameter reference differs");
   }else if(op=="core-source-stage"){`},
  {before:sourceCommands,after:sourceCommands.replaceAll('currentCore->check()','currentCore->source->check()').replace('if(currentCore)candidate=&currentCore->candidate();','if(currentCore)candidate=&currentCore->tensors[0];')},
  {before:'<<jsonString(op)<<",\\\"core_source_confirming\\\":',after:'<<jsonString(op)<<",\\\"retained_core_tensor_sha256\\\":"<<(op=="retain-backbone"&&backbone?jsonString(backbone->initial.config.at("core_version")=="null"?ORIGINAL:backbone->initial.config.at("core_tensor_sha256")):"null")<<",\\\"core_source_ack\\\":"<<(op=="core-source-apply"&&pendingCoreSource&&pendingCoreSource->applied?pendingCoreSource->acknowledgement():"null")<<",\\\"core_source_confirming\\\":'},
  {before:'<<jsonString(e.what())<<",\\\"core_source_confirming\\\":',after:'<<jsonString(e.what())<<",\\\"core_source_pending\\\":"<<(pendingCoreSource?"true":"false")<<",\\\"core_source_confirming\\\":'}
 ]};
}

export function patchEvolutionSourceSwapProcess(body:string,replace:Replace){
 const start=body.indexOf('type RetainedCoreACKRaw='),end=body.indexOf('\n\nexport class NativeExpertProcess',start);assert(start>=0&&end>start);
 body=body.slice(0,start)+`export type RetainedCoreSourceACK={pid:number;binary_sha256:string;descriptor_sha256:string;core_version:string|null;actual_core_tensor_sha256:string;native:Readonly<Record<string,any>>;assertUnchanged():Promise<void>;close():Promise<void>};
const currentRetainedCoreACKs=new WeakMap<NativeExpertProcess,RetainedCoreSourceACK>();
const nativeSourceCommandFailures=new WeakMap<object,{worker:NativeExpertProcess;response:Readonly<Record<string,unknown>>}>();
const retainedCoreACKs=new WeakMap<RetainedCoreSourceACK,{worker:NativeExpertProcess;pending:{applied:boolean;closed:boolean;committed?:boolean};closed:boolean}>();
export function validateRetainedCoreSourceACK(worker:NativeExpertProcess,cap:RetainedCoreSourceACK){const s=retainedCoreACKs.get(cap);assert(s&&!s.closed&&s.worker===worker&&s.pending.applied&&(!s.pending.closed||s.pending.committed===true&&currentRetainedCoreACKs.get(worker)===cap),'Core source ACK requires this actual retained peer issuer and applied source');worker.assertLive();assert(cap.pid===worker.pid&&cap.binary_sha256===worker.options.binary_sha256&&cap.native.protocol==='luban-retained-core-source-v2');}
`+body.slice(end);
 const a=body.indexOf(' async applyAcceptedCoreSource('),b=body.indexOf(' async rollbackAcceptedCoreSource(',a);assert(a>=0&&b>a);
 body=body.slice(0,a)+` async retainCoreBackbone(input:{path:string;sha256:string;core_tensor_sha256:string}){
  this.assertLive();assert(!this.hasInstalledCore&&!this.pendingCoreSource&&/^[a-f0-9]{64}$/.test(input.sha256)&&/^[a-f0-9]{64}$/.test(input.core_tensor_sha256));
  const result=await this.command(['retain-backbone',JSON.stringify(input.path),input.sha256].join(' '));assert(result.response.backbone_resident===true&&result.response.retained_core_tensor_sha256===input.core_tensor_sha256,'Native accepted backbone was not materialized');return result;
 }
 async applyAcceptedCoreSource(owners:NativeCore450Owner):Promise<RetainedCoreSourceACK>{
  const pending=this.pendingCoreSource;assert(pending&&!pending.closed&&!pending.applied&&owners instanceof NativeCore450Owner);const result=await this.command('core-source-apply');pending.applied=true;
  const native=result.response.core_source_ack;assert(native&&native.protocol==='luban-retained-core-source-v2'&&native.pid===this.pid&&(native.device===this.options.device||this.options.device==='mps'&&native.device==='mps:0')&&native.descriptor_sha256===pending.sha256&&native.actual_core_bytes===25165824&&JSON.stringify(native.actual_core_shape)==='[2048,6144]'&&/^[a-f0-9]{64}$/.test(native.actual_core_tensor_sha256)&&native.same_official_storage===true&&native.resident_core_readback===false&&native.parent_rng_snapshot===false&&native.official_graph_reloaded===false&&native.qualification===false,'Actual native resident Core source ACK differs');
  const state={worker:this,pending,closed:false};let cap!:RetainedCoreSourceACK;cap=Object.freeze({pid:this.pid,binary_sha256:this.options.binary_sha256,descriptor_sha256:pending.sha256,core_version:native.core_version==='null'?null:native.core_version,actual_core_tensor_sha256:native.actual_core_tensor_sha256,native:Object.freeze(structuredClone(native)),assertUnchanged:async()=>{validateRetainedCoreSourceACK(this,cap);assert(this.pendingCoreSource===pending||pending.closed);},close:async()=>{if(state.closed)return;assert(pending.closed,'Source ACK cannot close before actual transaction retirement');state.closed=true;}});retainedCoreACKs.set(cap,state);pending.ack=cap;owners.own(cap,async()=>{await this.settleAcceptedCoreSource();await cap.close();});await cap.assertUnchanged();return cap;
 }
`+body.slice(b);
 body=replace(body,'p.nativeConfirmed=true;p.onConfirmed?.();','if(committed){assert(p.ack);currentRetainedCoreACKs.set(this,p.ack);}p.nativeConfirmed=true;p.onConfirmed?.();');
 body=replace(body,"assert(r.response.event!=='error',r.response.message);","if(r.response.event==='error'){const error=new Error(String(r.response.message));nativeSourceCommandFailures.set(error,{worker:this,response:Object.freeze({...r.response})});throw error;}");
 body=replace(body,"return this.command(['core-source-stage',JSON.stringify(input.path),input.sha256,input.name].join(' '));","try{return await this.command(['core-source-stage',JSON.stringify(input.path),input.sha256,input.name].join(' '));}catch(error){const actual=error instanceof Error?nativeSourceCommandFailures.get(error):undefined;if(actual?.worker===this&&actual.response.operation==='core-source-stage'&&actual.response.core_source_pending===false){assert(this.pendingCoreSource===pending&&!pending.applied);pending.closed=true;this.pendingCoreSource=undefined;await owners.release(pending);}throw error;}");
 return body;
}
export async function applyNativeEvolutionSourceSwapPatches({bodies,target,replace,root}:{bodies:Map<string,string>;target:(p:string)=>string;replace:Replace;root:string}){
 const research=join(root,'.native-migration/research');
 const processPath=target(join(research,'native-unified-expert/expert-process.ts'));
 let processBody=patchEvolutionSourceSwapProcess(bodies.get(processPath)!,replace);
 processBody='import {assertNativeRawCoreDescriptor} from '+JSON.stringify(join(root,'backend/native-evolution-source-descriptor.ts'))+';\n'+processBody;
 processBody=replace(processBody,'const pending={name:input.name,sha256:input.sha256,applied:false',"assertNativeRawCoreDescriptor(await readFile(input.path,'utf8'));const pending={name:input.name,sha256:input.sha256,applied:false");
 bodies.set(processPath,processBody);
 const acceptedPath=target(join(research,'native-unified-expert/accepted-previous.ts'));let body=bodies.get(acceptedPath)??await readFile(acceptedPath,'utf8');
 body=replace(body,'const issued=new WeakMap<CapturedAcceptedPrevious,{engine:NativeEngine;guard:SnapshotGuard;closed:boolean}>();','const issued=new WeakMap<CapturedAcceptedPrevious,{engine:NativeEngine;guard:SnapshotGuard;closed:boolean;coreOnly:boolean;refresh:(files:Record<string,Buffer>)=>void}>();');
 body=replace(body,'const files=collectModelFiles(store),accepted=validateModelFiles(files,base,readNativeFoundationAdmission(store,files));','const files=collectModelFiles(store);let accepted=validateModelFiles(files,base,readNativeFoundationAdmission(store,files));');
 body=replace(body,' const pins=Object.fromEntries(Object.entries(files).map(([k,v])=>[k,sha(v)])),active=', ' let pins=Object.fromEntries(Object.entries(files).map(([k,v])=>[k,sha(v)]));const active=');
 body=replace(body,'const state={engine,guard,closed:false};','const state={engine,guard,closed:false,coreOnly:false,refresh:(latest:Record<string,Buffer>)=>{const next=validateModelFiles(latest,base,readNativeFoundationAdmission(store,latest));assert(equal(next.expert,accepted.expert)&&equal(next.router,accepted.router),"Core publication cannot replace the original expert or router");accepted=next;state.coreOnly=true;}};');
 body+=`\n/** Retain the same expert tensor owner while the real Core-only CAS changes the cohort. */
export function refreshCapturedAcceptedCoreSource(cap:CapturedAcceptedPrevious,files:Record<string,Buffer>){validateCapturedAcceptedPrevious(cap);issued.get(cap)!.refresh(files);}
`;
 body=replace(body,'validateCapturedAcceptedPrevious(cap);await owner.assertUnchanged();const latest=', 'validateCapturedAcceptedPrevious(cap);await owner.assertUnchanged();if(state.coreOnly)return;const latest=');
 bodies.set(acceptedPath,body);
}
