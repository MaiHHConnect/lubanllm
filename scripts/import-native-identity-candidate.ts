/** Explicit user-requested import, with one application lease and no Adam updates. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {mkdir,mkdtemp,readFile,realpath,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {homedir} from 'node:os';
import {parseArgs} from 'node:util';
import {privateJSON} from '../backend/settings.ts';
import {applicationConnection,readIdentityMaterial} from './run-native-identity-round.ts';
import {importNativeIdentityCandidate,REFINED_IDENTITY_SOURCE} from '../backend/native-evolution-import.ts';
import {closeImportOwner} from '../backend/native-evolution-import-observe.ts';
import {nativeEvolutionRuntimeResources} from '../backend/native-evolution-worker.ts';

const {values}=parseArgs({args:process.argv.slice(2),strict:true,options:{output:{type:'string'},material:{type:'string'},
 home:{type:'string',default:join(homedir(),'Library/Application Support/Luban')},'check-only':{type:'boolean'}}});
assert(values.output&&values.material,'--output NEW_DIRECTORY --material AUTHORIZED_IMPORT_MATERIAL.json required');
const home=resolve(values.home!),output=resolve(values.output),material=await readIdentityMaterial(resolve(values.material!));
const root='/Volumes/移动数据库/lubanllm/experiments/native-evolution-product-20261017/identity-native450-v19';
const candidateFile=join(root,'gptq-bf16-refine-q4-v1/candidate.safetensors');
const candidateRawFile=join(root,'gptq-bf16-refine-cpu-v1/projection/candidate.raw');
const evidence=JSON.parse(await readFile(resolve('research/native_evolution_final_policy_20261017.json'),'utf8'));
const sourceTree=join(evidence.reader_directory,'checkpoint-tree.json'),sourceManifest=join(evidence.reader_directory,'manifest.json');
assert.equal(evidence.summary.run_id,REFINED_IDENTITY_SOURCE.id);
if(values['check-only']){console.log(JSON.stringify({ready:true,optimizer_steps:0,model_loaded:false,GPU:false,source_run:REFINED_IDENTITY_SOURCE.id,
 candidateFile,candidateRawFile,sourceTree,material_rows:material.samples.length,readonly_boundaries:material.retentionRegressions?.length,
 application_lease_required:true,actual_quality_required:true,output}));process.exit(0);}
await mkdir(output,{mode:0o700});const connection=await applicationConnection(home),leaseID=randomUUID(),controller=new AbortController();
const stop=()=>controller.abort(new Error('Candidate import cancelled'));process.on('SIGINT',stop);process.on('SIGTERM',stop);
const redact=(value:string)=>value.replaceAll(connection.token,'[private-token]');
const cleanupError=(error:unknown)=>console.error(redact('Retaining actual import owner: '+String(error)));
const command=async(path:string,body:Record<string,unknown>={})=>{const response=await fetch(connection.url+path,{method:'POST',headers:{authorization:'Bearer '+connection.token,'content-type':'application/json'},body:JSON.stringify(body),redirect:'manual',proxy:null} as RequestInit&{proxy:null});
 const result:any=await response.json();assert(response.ok,typeof result.detail==='string'?result.detail:'Application command failed: '+path);return result;};
await writeFile(join(output,'lease.json'),JSON.stringify({lease_id:leaseID,acquire_attempted:false,acquired:false,was_ready:null,released:false})+'\n',{flag:'wx',mode:0o600});
const leaseState=async(value:Record<string,unknown>)=>privateJSON(join(output,'lease.json'),{lease_id:leaseID,...value});
const began=performance.now(),stages:any[]=[];let acquireAttempted=false,acquired=false,released=false,wasReady=false,result:any,error:unknown,work:string|undefined;
try{
 await leaseState({acquire_attempted:true,acquired:false,was_ready:null,released:false});acquireAttempted=true;const lease=await command('/internal/native-evolution/acquire',{lease_id:leaseID});acquired=true;await leaseState({acquire_attempted:true,acquired:true,was_ready:lease.was_ready,released:false});assert.equal(lease.lease_id,leaseID);assert.equal(lease.identity,nativeEvolutionRuntimeResources().base.identity);
 const parent=join(home,'native-evolution-work');await mkdir(parent,{recursive:true,mode:0o700});work=await mkdtemp(join(parent,'identity-import-'));
 result=await importNativeIdentityCandidate({home,work,model:await realpath(lease.model),root:lease.root,store:lease.store,identity:lease.identity,
  material,candidateFile,candidateRawFile,sourceTree,sourceManifest,signal:controller.signal,onCleanupError:cleanupError,
  onPhase(phase){const value={phase,seconds:(performance.now()-began)/1000,current_updates:0};stages.push(value);console.log(JSON.stringify(value));},
  async onEvidence(name,value){await writeFile(join(output,name+'.json'),JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});}});
}catch(value){error=value;}
finally{
 // The importer returns or throws only after all model/observer owners have closed.
 if(acquireAttempted)await closeImportOwner(async()=>{const result=await command('/internal/native-evolution/release',{lease_id:leaseID});assert(result.released||!acquired&&result.was_ready===false);wasReady=result.was_ready===true;released=true;await leaseState({acquire_attempted:true,acquired,was_ready:wasReady,released:true});},cleanupError);
 if(released&&wasReady)try{await command('/api/model/reload');}catch(value){error=error?new AggregateError([error,value],'Import and application reload failed'):value;}
 process.off('SIGINT',stop);process.off('SIGTERM',stop);
}
const final={schema:'native-validated-core-import-v1',passed:!error,optimizer_steps:0,source_training_steps:450,source_run:REFINED_IDENTITY_SOURCE.id,
 source_state:'rejected_preserved',stages,work,lease_released:released,total_seconds:(performance.now()-began)/1000,...result,
 ...(error?{error:redact(String(error)),stack:error instanceof Error?redact(error.stack??''):undefined,...((error as any).adopted?{adopted:true,id:(error as any).id,version:(error as any).version}:{})}:{})};
await writeFile(join(output,'result.json'),JSON.stringify(final,null,2)+'\n',{flag:'wx',mode:0o600});console.log(JSON.stringify({phase:'finished',passed:!error,adopted:result?.adopted??false,lease_released:released}));if(error)throw error;
