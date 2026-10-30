/** Read-only ordinary-Q4 transfer check of the fixed v19 candidate. No training journal or adoption. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {readFile,writeFile,mkdir,mkdtemp,realpath,unlink} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {homedir} from 'node:os';
import {parseArgs} from 'node:util';
import {applicationConnection} from './run-native-identity-round.ts';
import type {NativeEvolutionLease} from '../backend/native-evolution-service.ts';
import {nativeEvolutionRuntimeResources,nativeEvolutionQ4Resources} from '../backend/native-evolution-worker.ts';
import {NativeQ4Observer,assertNativeQ4Observation,type NativeQ4Observation} from '../backend/native-evolution-q4.ts';
import {uniqueNativeEvolutionRecords} from '../backend/native-evolution-material.ts';
import {nativeEvolutionProgress} from '../backend/native-evolution-quality.ts';
import {persistNativeGeneration} from '../backend/native-evolution-storage.ts';
import {assertPreparedProjection,oldSpanCheckpointInputs} from './old-span-diagnostic.ts';
import {coreCheckpointField as field} from '../.native-migration/research/native-product-runtime/backend/native-core-resume.ts';
import {writeNativeModelTensors} from '../.native-migration/backend/native-model-write.ts';

const {values}=parseArgs({args:process.argv.slice(2),strict:true,options:{output:{type:'string'},home:{type:'string',default:join(homedir(),'Library/Application Support/Luban')},'check-only':{type:'boolean'},'prepared-projection':{type:'string'}}});
assert(values.output,'--output must name a new diagnostic directory');const output=resolve(values.output),home=resolve(values.home!);
const evidence=JSON.parse(await readFile(resolve('research/native_evolution_final_policy_20261017.json'),'utf8'));
assert.equal(evidence.summary.run_id,'84dbd6c8f41a');let candidate=evidence.candidate;
const manifest=JSON.parse(await readFile(join(evidence.reader_directory,'manifest.json'),'utf8'));
assert.equal(manifest.checkpoint_sha256,evidence.summary.checkpoint.sha256);assert(manifest.tensors.some((v:any)=>JSON.stringify(v)===JSON.stringify(candidate)));
assert.equal(candidate.dtype,'bf16');assert.deepEqual(candidate.shape,[2048,6144]);assert.equal(candidate.bytes,25165824);
const preparedProjection=values['prepared-projection']?JSON.parse(await readFile(values['prepared-projection'],'utf8')):null;
let candidatePath=join(evidence.reader_directory,candidate.file);
if(preparedProjection){const tree=JSON.parse(await readFile(join(evidence.reader_directory,'checkpoint-tree.json'),'utf8'));assertPreparedProjection(preparedProjection,{run:evidence.summary.run_id,candidate:candidate.sha256,master:oldSpanCheckpointInputs(field(tree,'payload'))[1].sha256,previous:evidence.reference.sha256,identity:JSON.parse(evidence.original_binding_json).identity});candidate={...candidate,sha256:preparedProjection.core.sha256};candidatePath=preparedProjection.core.path;}
const records=uniqueNativeEvolutionRecords(evidence.records.groups,evidence.records.protection,evidence.records.retention_regressions);assert.equal(records.length,61);
const resources=nativeEvolutionRuntimeResources(),q4=await nativeEvolutionQ4Resources();assert.equal(resources.base.identity,evidence.summary.identity??JSON.parse(evidence.original_binding_json).identity);
assert.equal(resources.base.original_down_sha256,evidence.reference.sha256);
if(values['check-only']){console.log(JSON.stringify({ready:true,model_loaded:false,GPU:false,optimizer_steps:0,records:records.length,passes_per_model:2,candidate_raw:candidatePath,output,application_lease_required:true,product_admitted:false}));process.exit(0);}
await mkdir(output,{mode:0o700});const connection=await applicationConnection(home),leaseID=randomUUID(),signal=new AbortController();
const stop=()=>signal.abort(new Error('Q4 diagnostic cancelled'));process.on('SIGINT',stop);process.on('SIGTERM',stop);
const command=async(path:string,body:Record<string,unknown>={})=>{const response=await fetch(connection.url+path,{method:'POST',headers:{authorization:'Bearer '+connection.token,'content-type':'application/json'},body:JSON.stringify(body),redirect:'manual',proxy:null} as RequestInit & {proxy:null});const value:any=await response.json();assert(response.ok,typeof value?.detail==='string'?value.detail:'Application command failed: '+path);return value;};
const pointerBytes=async(store:string)=>Promise.all(['active-core.json','active.json','active-router.json'].map(async name=>{try{return await readFile(join(store,name));}catch(error:any){if(error.code==='ENOENT')return null;throw error;}}));
const began=performance.now(),stages:any[]=[];const event=(phase:string)=>{const value={phase,seconds:(performance.now()-began)/1000};stages.push(value);console.log(JSON.stringify(value));};
async function closeUntilConfirmed(close:()=>Promise<void>){let last='';for(;;){try{await close();return;}catch(error){const message=String(error).replaceAll(connection.token,'[private-token]');if(message!==last){console.error('Retaining real Q4 owner: '+message);last=message;}await Bun.sleep(1000);}}}
let lease:NativeEvolutionLease|undefined,writer:Awaited<ReturnType<typeof writeNativeModelTensors>>|undefined,observer:NativeQ4Observer|undefined,baseline:readonly[NativeQ4Observation,NativeQ4Observation]|undefined,current:typeof baseline;
let pointersBefore:(Buffer|null)[]|undefined,pointersUnchanged=false,leaseReleased=false,work:string|undefined,core:{file:string;tensor_key:string;sha256:string}|undefined,error:unknown;
try{
 event('acquiring_application_lease');lease=await command('/internal/native-evolution/acquire',{lease_id:leaseID});assert(lease&&lease.lease_id===leaseID&&lease.identity===resources.base.identity);pointersBefore=await pointerBytes(lease.store);signal.signal.throwIfAborted();
 const parent=join(home,'native-evolution-work');await mkdir(parent,{recursive:true,mode:0o700});work=await mkdtemp(join(parent,'q4-diagnostic-'));
 event('write_candidate_safetensors');const partial=join(work,'candidate.safetensors.partial');
 writer=await writeNativeModelTensors({path:partial,tensors:[{tensor_key:resources.base.core_parameter,dtype:'BF16',shape:candidate.shape,path:candidatePath,bytes:candidate.bytes,sha256:candidate.sha256}],signal:signal.signal});
 assert.equal(writer.tensors[resources.base.core_parameter]!.sha256,candidate.sha256);const fileSHA=writer.file_sha256;await closeUntilConfirmed(()=>writer!.close());writer=undefined;
 const file=join(output,'candidate.safetensors');await persistNativeGeneration({source:partial,destination:file,expectedSHA:fileSHA,signal:signal.signal});await unlink(partial);core={file,tensor_key:resources.base.core_parameter,sha256:fileSHA};
 const request={...q4,model:await realpath(lease.model),identity:lease.identity,manifestSHA:resources.base.model_manifest_sha256,records,work,signal:signal.signal};
 event('baseline_q4_two_passes');observer=new NativeQ4Observer();baseline=await observer.observePair(request);for(const proof of baseline)assertNativeQ4Observation(proof,records);await closeUntilConfirmed(()=>observer!.close());observer=undefined;
 await writeFile(join(output,'baseline.json'),JSON.stringify(baseline,null,2)+'\n',{flag:'wx',mode:0o600});
 event('candidate_q4_two_passes');observer=new NativeQ4Observer();current=await observer.observePair({...request,core});for(const proof of current)assertNativeQ4Observation(proof,records);await closeUntilConfirmed(()=>observer!.close());observer=undefined;
 for(const proof of [...baseline,...current])assert.deepEqual(proof.rows.map(row=>row.input_tokens),baseline[0].rows.map(row=>row.input_tokens),'Q4 complete tokenized inputs differ');
 await writeFile(join(output,'candidate.json'),JSON.stringify(current,null,2)+'\n',{flag:'wx',mode:0o600});
}catch(value){error=value;}
finally{
 if(observer)await closeUntilConfirmed(()=>observer!.close());if(writer)await closeUntilConfirmed(()=>writer!.close());
 try{if(lease&&pointersBefore){const after=await pointerBytes(lease.store);assert(pointersBefore.every((v,i)=>v===null?after[i]===null:after[i]!==null&&v.equals(after[i]!)),'Q4 diagnostic changed product pointer');pointersUnchanged=true;}}catch(value){error=error?new AggregateError([error,value],'Q4 diagnostic and pointer check failed'):value;}
 try{const released=await command('/internal/native-evolution/release',{lease_id:leaseID});assert(released.released||!lease&&!released.was_ready);leaseReleased=true;if(released.was_ready)await command('/api/model/reload');}catch(value){error=error?new AggregateError([error,value],'Q4 diagnostic and lease cleanup failed'):value;}
 process.off('SIGINT',stop);process.off('SIGTERM',stop);
}
const split={...Object.fromEntries(Object.entries(evidence.records.groups).map(([name,rows]:[string,any])=>[name,rows.map((row:any)=>row.id)])),original_protection:evidence.records.protection.map((r:any)=>r.id),user_boundaries:evidence.records.retention_regressions.map((r:any)=>r.id)};
const counts=(proofs:typeof baseline)=>proofs?Object.fromEntries(Object.entries(split).map(([name,ids]:[string,any])=>[name,{total:ids.length,first:proofs[0].rows.filter(row=>ids.includes(row.id)&&row.passed).length,second:proofs[1].rows.filter(row=>ids.includes(row.id)&&row.passed).length,both:ids.filter((id:string)=>proofs.every(proof=>proof.rows.some(row=>row.id===id&&row.passed&&row.complete))).length}])):null;
const progress=baseline&&current?nativeEvolutionProgress(records,baseline[0].rows.map((row,index)=>row.passed?row:baseline![1].rows[index]!),[...current[0].rows],[...current[1].rows],{core_version:'core-'+evidence.summary.run_id,core_sha256:candidate.sha256,expert_id:null,expert_sha256:null},false):null;
const result={schema:'native-q4-candidate-diagnostic-v1',passed:!error,scope:'61 frozen rows, two complete baseline and candidate passes; generation transfer diagnostic, not training or quality admission',source_run:evidence.summary.run_id,prepared_projection:preparedProjection?resolve(values['prepared-projection']!):null,identity:resources.base.identity,candidate_tensor_sha256:candidate.sha256,core,precision:'mixed-q4-g16',seed:42,temperature:0,thinking:'off',optimizer_steps:0,teacher_calls:0,product_admitted:false,journal_mutations_requested:false,product_pointers_unchanged:pointersUnchanged,observers_closed:true,lease_released:leaseReleased,stages,work,baseline_counts:counts(baseline),candidate_counts:counts(current),learning_progress:progress,total_seconds:(performance.now()-began)/1000,...(error?{error:String(error).replaceAll(connection.token,'[private-token]'),stack:error instanceof Error?error.stack?.replaceAll(connection.token,'[private-token]'):undefined}:{})};
await writeFile(join(output,'result.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});console.log(JSON.stringify({phase:'finished',passed:result.passed,baseline_counts:result.baseline_counts,candidate_counts:result.candidate_counts,optimizer_steps:0,lease_released:leaseReleased}));if(error)throw error;
