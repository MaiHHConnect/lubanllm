import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {assertNativeRawCoreDescriptor} from '../backend/native-evolution-source-descriptor.ts';

test('actual stage consumer rejects container ABI before registering an owner; native failure retains its real pending owner',async()=>{
 const bundle=await readFile('build/native-evolution/identity-import.js','utf8');
 const begin=bundle.indexOf('  async stageAcceptedCoreSource('),end=bundle.indexOf('  async retainCoreBackbone(',begin);assert(begin>0&&end>begin);
 let method=bundle.slice(begin,end).replace(/assert\d+/g,'assert');
 // Current frozen build predates the narrow preflight; this is exactly the one inserted expression.
 method=method.replace('    const pending = {','    assertNativeRawCoreDescriptor(await readFile(input.path,"utf8"));\n    const pending = {');
 class NativeCore450Owner{owners:any[]=[];own(owner:any){this.owners.push(owner);}}
 const C=new Function('assert','readFile','assertNativeRawCoreDescriptor','NativeCore450Owner','return class {'+method+'}')(
  assert,readFile,assertNativeRawCoreDescriptor,NativeCore450Owner);
 const directory=await mkdtemp(join(tmpdir(),'source-descriptor-'));
 const values={core_version:'core-0123456789ab',core_path:join(directory,'candidate.raw'),core_file_sha256:'a'.repeat(64),core_tensor_sha256:'a'.repeat(64),core_tensor_key:'model.language_model.layers.23.mlp.down_proj.weight',core_offset:'0',core_bytes:'25165824'};
 const text=(fields:any)=>'LUBAN_QWEN35_CACHE_CORE_INPUT_V1\n'+Object.entries(fields).map(([k,v])=>'config\t'+k+'\t'+v).join('\n')+'\n';
 try{
  for(const bad of [{core_offset:'136'},{core_bytes:'4'},{core_file_sha256:'b'.repeat(64)},{core_version:'core-bad'},{core_tensor_key:'other'}]){
   const path=join(directory,'bad.tsv');await writeFile(path,text({...values,...bad}));let sent=0;
   const worker=new C();Object.assign(worker,{name(){},coreSourceAdoption:true,options:{device:'mps',hidden:2048},command(){sent++;}});
   const owners=new NativeCore450Owner();await expect(worker.stageAcceptedCoreSource({path,sha256:'1'.repeat(64),name:'test'},owners)).rejects.toThrow();
   expect(worker.pendingCoreSource).toBeUndefined();expect(owners.owners).toHaveLength(0);expect(sent).toBe(0);
  }
  const path=join(directory,'raw.tsv');await writeFile(path,text(values));
  for(const nativeFailure of [false,true]){
   const worker=new C();let sent=0;Object.assign(worker,{name(){},coreSourceAdoption:true,options:{device:'mps',hidden:2048},command(){sent++;if(nativeFailure)throw Error('native owner may exist');return{actual:true};}});
   const owners=new NativeCore450Owner(),result=worker.stageAcceptedCoreSource({path,sha256:'1'.repeat(64),name:'test'},owners);
   if(nativeFailure)await expect(result).rejects.toThrow('native owner may exist');else expect(await result).toEqual({actual:true});
   expect(sent).toBe(1);expect(owners.owners).toEqual([worker.pendingCoreSource]);
  }
  expect(()=>assertNativeRawCoreDescriptor(text({...values,core_version:'null',...Object.fromEntries(Object.keys(values).filter(k=>k!=='core_version').map(k=>[k,'null']))}))).not.toThrow();
 }finally{await rm(directory,{recursive:true,force:true});}
});
