import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {NativeCore450Owner} from '../.native-migration/research/native-core450-owner-controller/owner.ts';
import {assertNativeRawCoreDescriptor} from '../backend/native-evolution-source-descriptor.ts';
import {writeEvolutionSourceSwapSources} from '../scripts/native-evolution-source-swap-patches.ts';

test('actual composed stage only retires local ownership on its own explicit native no-pending error',async()=>{
 const source=await readFile(process.env.LUBAN_NATIVE_EXPERT_TEST_BUNDLE??'build/native-evolution/worker.js','utf8');
 const method=(a:string,b:string)=>{const start=source.indexOf(a),end=source.indexOf(b,start);assert(start>=0&&end>start);return source.slice(start,end).replace(/\b(assert|readFile|durable|join)\d+\b/g,'$1');};
 const code='class Consumer {'+method('  command(line,','  sourceFields(')+method('  async stageAcceptedCoreSource(','  async retainCoreBackbone(')+'}\nreturn Consumer;';
 assert(code.includes('core_source_pending === false'),'Test requires the actual new compiled stage');
 const failures=new WeakMap();
 const Consumer=new Function('assert','readFile','join','durable','assertNativeRawCoreDescriptor','NativeCore450Owner','nativeSourceCommandFailures','assertObservationCommand',code)(assert,readFile,join,async(p:string,v:unknown)=>writeFile(p,JSON.stringify(v)),assertNativeRawCoreDescriptor,NativeCore450Owner,failures,()=>{});
 const dir=await mkdtemp(join(tmpdir(),'core-stage-failure-'));
 try{
  const path=join(dir,'core.tsv');await writeFile(path,'LUBAN_QWEN35_CACHE_CORE_INPUT_V1\n'+Object.entries({core_version:'core-0123456789ab',core_path:join(dir,'candidate.raw'),core_file_sha256:'a'.repeat(64),core_tensor_sha256:'a'.repeat(64),core_tensor_key:'model.language_model.layers.23.mlp.down_proj.weight',core_offset:'0',core_bytes:'25165824'}).map(([k,v])=>'config\t'+k+'\t'+v).join('\n')+'\n');
  for(const status of [false,true,undefined,'forged'] as const){
   const owners=new NativeCore450Owner(),worker=new Consumer();let writes=0;
   Object.assign(worker,{name(){},assertLive(){},coreSourceAdoption:true,options:{device:'mps',hidden:2048,directory:dir},queue:Promise.resolve(),commands:[],ordinal:0,pid:1,check:async()=>{},child:{stdin:{write(){writes++;worker.ordinal++;queueMicrotask(()=>{const waiting=worker.waiting;worker.waiting=undefined;waiting.resolve({response:{event:'error',operation:'core-source-stage',message:'input rejected',...(status!==undefined?{core_source_pending:status===false?false:true}:{})},events:[]});});}}}});
   if(status==='forged')worker.command=async()=>{throw Object.assign(Error('forged'),{core_source_pending:false});};
   await expect(worker.stageAcceptedCoreSource({path,sha256:'b'.repeat(64),name:'candidate'},owners)).rejects.toThrow();
   expect(writes).toBe(status==='forged'?0:1);
   if(status===false){expect(worker.pendingCoreSource).toBeUndefined();expect(owners.status().retained_owners).toBe(0);await owners.close();}
   else{expect(worker.pendingCoreSource).toBeDefined();expect(owners.status().retained_owners).toBe(1);worker.pendingCoreSource.closed=true;await owners.close();}
  }
 }finally{await rm(dir,{recursive:true,force:true});}
});
test('native error schema reports the actual unique_ptr, for both allocated and absent owners',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'core-stage-error-schema-'));
 try{const generated=await writeEvolutionSourceSwapSources(process.cwd(),dir),change=generated.changes.find(row=>row.after.includes('core_source_pending'))!;expect(change).toBeDefined();
  const original=await readFile('.native-migration/research/native-canonical-core-source-close-role-v2/PersistentMPSV3.cpp','utf8');expect(original.split(change.before)).toHaveLength(2);
  const after=original.replace(change.before,change.after);expect(after).toContain('<<jsonString(e.what())<<",\\"core_source_pending\\":"<<(pendingCoreSource?"true":"false")');expect(after.replace(change.after,change.before)).toBe(original);
 }finally{await rm(dir,{recursive:true,force:true});}
});
