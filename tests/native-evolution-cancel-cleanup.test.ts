import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm,realpath} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
import {nativeEvolutionRestoreCoreSource} from '../scripts/native-evolution-ambient-patches.ts';
import {readNativeModelTensors} from '../.native-migration/backend/native-model-tensor.ts';
import {NativeCore450Owner} from '../.native-migration/research/native-core450-owner-controller/owner.ts';

async function harness(){
 const root=await realpath(await mkdtemp(join(tmpdir(),'native-cancel-drop-')));
 const source=await readFile(new URL('../build/native-evolution/worker.js',import.meta.url),'utf8');
 const method=(start:string,end:string)=>{const a=source.indexOf(start),b=source.indexOf(end,a);assert(a>=0&&b>a,'Actual compiled consumer missing');return source.slice(a,b).replace(/\b(assert|durable|join)\d+\b/g,'$1');};
 const code='class Consumer {'+method('  async check(restoringCore', '  kill()')+method('  command(line,', '  sourceFields(')+method('  async dropCoreState(', '  async replaceInitialized(')+'}\nreturn Consumer;';
 const Consumer=new Function('assert','durable','join','observationFastCheck','assertObservationCommand','assertObservationMutationAllowed',code)(assert,async(path:string,record:any)=>writeFile(path,JSON.stringify(record)),join,async()=>false,()=>{},()=>{});
 const keys=['candidate','master','exp_avg','exp_avg_sq','adam_step','reference','basis'];
 let at=0;const header:Record<string,unknown>={};for(const key of keys){const dtype=key==='candidate'?'BF16':'F32',shape=key==='adam_step'?[]:[1,2],bytes=key==='adam_step'?4:key==='candidate'?4:8;header[key]={dtype,shape,data_offsets:[at,at+bytes]};at+=bytes;}
 const text=Buffer.from(JSON.stringify(header)),length=Buffer.alloc(8);length.writeBigUInt64LE(BigInt(text.length));const raw=Buffer.concat([length,text,Buffer.alloc(at)]),path=join(root,'installed.safetensors');await writeFile(path,raw,{mode:0o600});
 const owner=await readNativeModelTensors({path,file_sha256:createHash('sha256').update(raw).digest('hex'),tensors:keys.map(tensor_key=>({tensor_key,dtype:tensor_key==='candidate'?'BF16':'F32',shape:tensor_key==='adam_step'?[]:[1,2]}))});
 const signal=AbortSignal.abort(Error('user cancelled')),state={writes:0,sourceChecks:0,guard:true,badACK:false,trainerExited:true};
 const worker=new Consumer();Object.assign(worker,{options:{directory:root,hidden:1,guard:{assertQuiescent(){assert(state.guard,'invalid retained guard');}},model_owner:{assertUnchanged:async()=>{assert(state.guard);}}},source:{assertUnchanged:async()=>{assert(state.guard);}},coreSource:{assertUnchanged:async()=>{state.sourceChecks++;signal.throwIfAborted();}},coreSourceOwner:owner,queue:Promise.resolve(),commands:[],ordinal:0,pid:123,
  assertLive(){assert(state.guard);},child:{stdin:{write(){state.writes++;worker.ordinal++;queueMicrotask(()=>{const waiting=worker.waiting;worker.waiting=undefined;waiting.resolve({response:{event:'done',operation:'core-drop',step:0,observation:'current',backbone_resident:true,core_tensor_sha256:state.badACK?'wrong':'original'},events:[]});});}}}});
 Object.defineProperty(worker,'hasInstalledCore',{get:()=>worker.coreSource!==undefined});
 const engine={},guard={assertQuiescent(){assert(state.guard);}},installations=new WeakMap<any,any>(),restored=new WeakMap<any,any>();
 const installation={owner,closed:false};installations.set(engine,installation);
 const peer={worker,source:{runtime:{}},base:{original_down_sha256:'original'}};
 const restore=new Function('assert','assertNativeExpertScopeGuard','borrowNativeEnginePeer','installations','restored',new Bun.Transpiler({loader:'ts'}).transformSync(nativeEvolutionRestoreCoreSource.replace('export async function','async function')+'\nreturn restoreNativeEvolutionCore;'))(assert,(_e:any,g:any)=>g.assertQuiescent(),()=>peer,installations,restored);
 return{root,owner,worker,state,engine,guard,installation,installations,restore,async close(){await owner.close();await rm(root,{recursive:true,force:true});}};
}
test('actual command and drop withdraw cancelled installed Core using its real tensor owner; forward remains cancelled',async()=>{
 const h=await harness();try{
  await expect(h.worker.command('run ignored')).rejects.toThrow('user cancelled');expect(h.state.writes).toBe(0);
  const receipt=await h.restore(h.engine,h.guard);expect(receipt.actual_source_restored).toBe(true);expect(h.state.writes).toBe(1);
  expect(h.owner.handle.fd).toBe(-1);expect(h.installations.has(h.engine)).toBe(false);expect(h.state.sourceChecks).toBe(1);
 }finally{await h.close();}
});
test('withdrawal still rejects changed source FD, invalid shape and invalid guard before native command',async()=>{
 for(const mode of ['file','guard','shape']){const h=await harness();try{
  if(mode==='file')await writeFile(h.owner.path,Buffer.alloc(8));else if(mode==='guard')h.state.guard=false;
  else h.worker.coreSourceOwner={...h.owner,tensors:{...h.owner.tensors,candidate:{...h.owner.tensors.candidate,shape:[2,2]}}};
  await expect(h.restore(h.engine,h.guard)).rejects.toThrow();expect(h.state.writes).toBe(0);expect(h.installations.has(h.engine)).toBe(true);
 }finally{await h.close();}}
});
test('actual restore requires ACK and retains installation for same-owner close retry',async()=>{
 const h=await harness();try{
  h.state.badACK=true;await expect(h.restore(h.engine,h.guard)).rejects.toThrow('restoration ACK differs');expect(h.installations.has(h.engine)).toBe(true);expect(h.owner.handle.fd).toBeGreaterThanOrEqual(0);
  h.state.badACK=false;const actual=h.installation.owner;let closeAttempts=0;h.installation.owner={...actual,async close(){closeAttempts++;if(closeAttempts===1)throw Error('FD close failed');await actual.close();}};
  await expect(h.restore(h.engine,h.guard)).rejects.toThrow('FD close failed');expect(h.installations.has(h.engine)).toBe(true);
  await h.restore(h.engine,h.guard);expect(closeAttempts).toBe(2);expect(h.owner.handle.fd).toBe(-1);expect(h.state.writes).toBe(3);
 }finally{await h.close();}
});
test('actual owner cleanup completes after trainer exit even while the training signal is cancelled',async()=>{
 const h=await harness();try{
  const owner=new NativeCore450Owner();let restored=false,finalized=false;
  owner.own(h.owner,async()=>{assert(restored);await h.owner.close();});owner.cancel();
  await owner.run(async()=>{await owner.finishScope(async()=>{assert(h.state.trainerExited);await h.restore(h.engine,h.guard);restored=true;},async()=>{finalized=true;});});
  await owner.close();expect(owner.status()).toMatchObject({closed:true,retained_owners:0,blocked:false});expect(finalized).toBe(true);expect(h.state.writes).toBe(1);
 }finally{await h.close();}
});
