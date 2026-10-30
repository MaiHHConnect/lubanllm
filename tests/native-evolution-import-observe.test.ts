import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {prepareBaselineSamples,baselineGenerationRow} from '../.native-migration/research/native-ordinary-baseline/rows.ts';
import {uniqueNativeEvolutionRecords} from '../backend/native-evolution-material.ts';
import {retentionLossRecords} from '../backend/native-evolution-loss-targets.ts';

test('actual import observer uses raw BF16 offset zero, two complete passes, identical loss targets and restores after cancellation/failure',async()=>{
 const source=await readFile('backend/native-evolution-import-observe.ts','utf8');
 const code=new Bun.Transpiler({loader:'ts'}).transformSync(source.slice(source.indexOf('export async function observeNativeImport')).replace(/^export /gm,''));
 for(const failure of ['none','candidate-loss','cancel','restore-once']){
  const work=await mkdtemp(join(tmpdir(),'import-observer-')),controller=new AbortController();
  let selected=false,staged:any,serial=0,generations=0,losses=0,checks=0,restoreAttempts=0;const states=new Map<string,number[]>(),observed:any[]=[];
  const worker={pid:123,assertLive(){},async retainCoreBackbone(){},async settleAcceptedCoreSource(){},
   async stageAcceptedCoreSource(value:any){staged=value;},async applyAcceptedCoreSource(){
    const text=await readFile(staged.path,'utf8'),current=text.includes('core_version\tcore-candidate');
    if(!current&&failure==='restore-once'&&restoreAttempts++===0)throw Error('restore retry');
    return{pid:123,actual_core_tensor_sha256:current?'candidate-sha':'original-sha',core_version:current?'core-candidate':null};},
   async confirmAcceptedCoreSource(_yes:boolean,accept:()=>void){selected=(await readFile(staged.path,'utf8')).includes('core_version\tcore-candidate');accept();},
   async runDescriptor(request:any){const text=await readFile(request.path,'utf8');assert.equal(text.includes('core_version\tcore-candidate'),selected);if(selected){assert(text.includes('core_offset\t0'));assert(text.includes('core_file_sha256\tcandidate-sha'));assert(text.includes('core_path\t'+join(work,'candidate.raw')));}
    generations++;if(selected&&failure==='cancel'){controller.abort(Error('cancelled'));controller.signal.throwIfAborted();}
    return{events:[{event:'token',token:248044},{event:'complete',finish_reason:'stop'}]};},
   async lossDescriptor(request:any){const text=await readFile(request.path,'utf8');assert.equal(text.includes('core_version\tcore-candidate'),selected);
    const raw=await readFile(request.labels.path);observed.push({selected,raw:raw.toString('hex')});losses++;if(selected&&failure==='candidate-loss')throw Error('loss');return{loss:.1};},options:{directory:work}};
  const peer={worker,base:{core_parameter:'down',original_down_sha256:'original-sha'},source:{runtime:{},assertUnchanged:async()=>{}},owner:{},core:null,previous:null,routerOwner:null,build:{}};
  const control={async tokenizer(request:any){const id=request.generation_id;if(request.operation==='generation_prepare'){states.set(id,[]);return{event:'generation_prepared',generation_id:id,thinking:'off',ids:[1,2]};}
    if(request.operation==='generation_append'){states.get(id)!.push(request.token);return{event:'generation_decoded',generation_id:id,step:request.step,token:request.token};}
    return{event:'generation_decode_finished',generation_id:id,tokens:states.get(id),text:'a'};},async prepareTrainingSample(){return{native_labels_compatible:true,ids:[1,2],labels:[-100,2]};}};
  const sha=(value:any)=>Bun.CryptoHasher.hash('sha256',value,'hex');
  const runtime=new Function('assert','readFile','join','prepareBaselineSamples','baselineGenerationRow','writePeerDescriptor','borrowNativeEnginePeer','withObservation','assertEvolutionBranch','enterEvolutionBranch','durable','sha','uniqueNativeEvolutionRecords','retentionLossRecords','closeImportOwner',code+';return observeNativeImport;')(
   assert,readFile,join,prepareBaselineSamples,baselineGenerationRow,async()=>{const path=join(work,'source-'+ ++serial+'.tsv');const text=Object.entries({core_version:'',core_path:'',core_file_sha256:'',core_tensor_sha256:'original-sha',core_tensor_key:'down',core_offset:'0',core_bytes:'4'}).map(([k,v])=>'config\t'+k+'\t'+v).join('\n');await writeFile(path,text);return{path,sha256:sha(text)};},()=>peer,
   async(scope:any,body:()=>Promise<any>)=>{await scope.fast();return body();},()=>{},async()=>{},writeFile,sha,uniqueNativeEvolutionRecords,retentionLossRecords,
   async(close:()=>Promise<void>)=>{let attempts=0;for(;;){try{await close();return;}catch{assert(++attempts<3);}}});
  const row=(id:string)=>({id,prompt:'answer a',answer:'a',purpose:'validation',training_allowed:false,provenance:{generation_check:{type:'exact',expected:'a'}}});
  let result:any,error:unknown;
  try{result=await runtime({engine:{},guard:{assertQuiescent(){}},control,owners:{},candidate:{path:join(work,'candidate.safetensors'),file_sha256:'container-sha',tensors:{down:{byte_offset:137,bytes:4,sha256:'candidate-sha'}},assertUnchanged:async()=>{checks++;}},
   raw:{path:join(work,'candidate.raw'),bytes:4,sha256:'candidate-sha',assertUnchanged:async()=>{checks++;}},version:'core-candidate',records:{groups:{replay_validation:[row('replay')]},protection:[row('old')],retention_regressions:[]},work,signal:controller.signal,onPhase:()=>{},onCleanupError:()=>{}});}catch(value){error=value;}
  expect(selected).toBe(false);expect(checks).toBeGreaterThan(0);
  if(failure==='none'){expect(error).toBeUndefined();expect(generations).toBe(8);expect(losses).toBe(4);expect(result.baselineCoreLoss).toEqual(result.candidateCoreLoss);expect(new Set(observed.map(v=>v.raw)).size).toBe(1);expect(result.optimizer_steps).toBe(0);}
  else expect(error).toBeTruthy();
  await rm(work,{recursive:true,force:true});
 }
});
