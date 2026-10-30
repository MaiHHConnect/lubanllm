import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {readFile,mkdtemp,mkdir,writeFile,open,rm,unlink} from 'node:fs/promises';
import {fstatSync,lstatSync,readFileSync} from 'node:fs';
import {join,resolve,dirname} from 'node:path';
import {tmpdir} from 'node:os';
import {createHash} from 'node:crypto';
const root=resolve(import.meta.dir,'..');
const compiledWorker=await readFile(join(root,'build/native-evolution/worker.js'),'utf8');
const peerSource=compiledWorker.slice(compiledWorker.indexOf('function borrowNativeEnginePeer('),compiledWorker.indexOf('async function generateNativeEnginePeer(')).replace(/\b(assert|equal)\d+\b/g,'$1');
const helperSource=await readFile(join(root,'backend/native-evolution-held-source.ts'),'utf8');
const commit=await readFile(join(root,'.native-migration/backend/native-commit.ts'),'utf8');
const originalHashes=commit.slice(commit.indexOf('export function nativeAcceptedHashes('),commit.indexOf('\nfunction plans(')).replace('export ','');
const hash=(x:Buffer|string)=>createHash('sha256').update(x).digest('hex');
const compile=(source:string,scope:Record<string,unknown>,name:string)=>new Function(...Object.keys(scope),new Bun.Transpiler({loader:'ts'}).transformSync(source.replace(/import[\s\S]*?from\s*['"][^'"]+['"];?/g,'').replaceAll('export ','')+`\nreturn ${name};`))(...Object.values(scope));
export async function heldFixture(options:{core?:boolean;expert?:boolean;router?:boolean;fallback?:boolean;directory?:string}={}){
 const directory=await mkdtemp(join(options.directory??tmpdir(),'native-held-source-')),files:Record<string,Buffer>={},handles:any[]=[];
 const put=async(name:string,value:unknown)=>{const bytes=Buffer.from(typeof value==='string'?value:JSON.stringify(value));await mkdir(dirname(join(directory,name)),{recursive:true});await writeFile(join(directory,name),bytes,{mode:0o600});files[name]=bytes;};
 if(options.core){await put('active-core.json',{version:'core-b45d965728bb'});for(const name of ['core-b45d965728bb.json','core-b45d965728bb.safetensors','core-b45d965728bb.retention.json','evolution-b45d965728bb.json'])await put(name,'bound '+name);}
 if(options.expert){await put('active.json',{id:'8ec555e81b20'});await put('experts/8ec555e81b20.json','expert metadata');await put('experts/8ec555e81b20.safetensors','expert bytes');}
 if(options.router){await put('active-router.json',{version:'router-a970b1fb305c'});await put('router-a970b1fb305c.json','router metadata');await put('router-a970b1fb305c.safetensors','router bytes');}
 if(options.fallback){await put('router.json','fallback metadata');await put('router.safetensors','fallback bytes');}
 const nativeAcceptedHashes=compile(originalHashes,{collectModelFiles:()=>Object.fromEntries(Object.keys(files).map(n=>[n,readFileSync(join(directory,n))])),pointerNames:['active.json','active-core.json','active-router.json'],ledgerParse:JSON.parse,need:assert,hash},'nativeAcceptedHashes');
 const hashes=Object.fromEntries(Object.entries(files).map(([n,b])=>[n,hash(b)]));
 const ownerPath=join(directory,'fixture-original.raw');await writeFile(ownerPath,'original',{mode:0o600});const handle=await open(ownerPath,'r');handles.push(handle);const initial=fstatSync(handle.fd,{bigint:true});const owner={path:ownerPath,handle,async assertUnchanged(){const now=fstatSync(handle.fd,{bigint:true});assert(['dev','ino','size','mtimeNs','ctimeNs'].every(k=>initial[k]===now[k]),'Original owner changed before view capture');}};
 const guard={id:1,modelLockFd:42},engine={},peer:any={lock:42,store:directory,base:{identity:'model'},guard,hashes,source:{runtime:{core_label:null}},owner,core:null,previous:options.expert?{id:'8ec555e81b20',owner}:null,routerOwner:null,worker:{assertLive(){assert(live);}}};
 let live=true,current:any=guard,sourceAdmissions=1;
 const assertGuard=(_e:any,g:any)=>assert(g===current&&live,'inactive source guard');
 const borrow=compile(peerSource,{assert,assertNativeExpertScopeGuard:assertGuard,peers:new Map([[engine,peer]]),observation:()=>null,equal:(a:any,b:any)=>{try{assert.deepEqual(a,b);return true;}catch{return false;}},hashes:()=>{sourceAdmissions++;return Object.fromEntries(Object.keys(hashes).map(n=>[n,hash(readFileSync(join(directory,n)))]));},rebindCapturedAcceptedPrevious(){}},'borrowNativeEnginePeer');
 const read=compile(helperSource,{assert,fstatSync,lstatSync,readFileSync,join,resolve,assertNativeExpertScopeGuard:assertGuard,borrowNativeEnginePeer:borrow,borrowNativeEvolutionPreparation:()=>({store:directory})},'readNativeEvolutionHeldSource');
 return{directory,files,engine,guard,peer,read,nativeAcceptedHashes,put,sourceAdmissions:()=>sourceAdmissions,nextGuard(){current={id:2,modelLockFd:42};peer.guard=null;return current;},invalidate(){live=false;},async close(){for(const h of handles)if(h.fd>=0)await h.close();await rm(directory,{recursive:true,force:true});}};
}
test('held source matches actual nativeAcceptedHashes for None, B45/expert/router and legacy router inventories',async()=>{
 for(const options of [{},{core:true,expert:true,router:true},{expert:true},{core:true,fallback:true}]){
  const h=await heldFixture(options);try{const held=await h.read(h.engine,h.guard);expect(held.pins).toEqual(h.nativeAcceptedHashes(h.directory));expect(held.accepted.core).toBeNull();expect(held.accepted.expert?.id??null).toBe(options.expert?'8ec555e81b20':null);await held.assertUnchanged();expect(await h.read(h.engine,h.guard)).toBe(held);expect(h.sourceAdmissions()).toBe(1);}finally{await h.close();}
 }
});
test('held source rejects pointer appearance/removal and original FD mutation without reading model bytes again',async()=>{
 for(const change of ['appearance','removal','fd']){const h=await heldFixture({core:true});try{const held=await h.read(h.engine,h.guard);if(change==='appearance')await h.put('active-router.json',{version:'router-rival'});else if(change==='removal')await unlink(join(h.directory,'active-core.json'));else await writeFile(h.peer.owner.path,'mutated!');await expect(held.assertUnchanged()).rejects.toThrow();}finally{await h.close();}}
});
test('crossguard requires original source admission; old capabilities cannot follow the next guard',async()=>{
 const h=await heldFixture({core:true});try{const old=await h.read(h.engine,h.guard),next=h.nextGuard();await expect(old.assertUnchanged()).rejects.toThrow('inactive source guard');const current=await h.read(h.engine,next);expect(current).not.toBe(old);expect(h.sourceAdmissions()).toBe(2);h.nextGuard();await writeFile(join(h.directory,'core-b45d965728bb.safetensors'),'rival');await expect(h.read(h.engine,h.nextGuard())).rejects.toThrow();}finally{await h.close();}
});
test.skipIf(process.env.LUBAN_NATIVE_HELD_ACTUAL!=='1')('read-only actual 29e46ec82bd9 journal inventory matches current exFAT pointer metadata without weight reads',async()=>{
 const {Database}=await import('bun:sqlite'),db=new Database('/Volumes/移动数据库/lubanllm/app/memory.sqlite3',{readonly:true});
 const row=db.query('SELECT body FROM training_runs WHERE id=?').get('29e46ec82bd9')as any;db.close();assert(row);const binding=JSON.parse(row.body).binding;
 const directory='/Volumes/移动数据库/lubanllm/modules/luban0-v1',h=await heldFixture();
 try{
  const hashes=Object.fromEntries(Object.entries(binding.active_files_sha256).filter(([,v])=>v!==null));
  const peer={...h.peer,store:directory,hashes,previous:{id:binding.expert_id,owner:h.peer.owner}};
  const read=compile(helperSource,{assert,fstatSync,lstatSync,readFileSync,join,resolve,assertNativeExpertScopeGuard:(_e:any,g:any)=>assert(g===h.guard),borrowNativeEnginePeer:()=>peer,borrowNativeEvolutionPreparation:()=>({store:directory})},'readNativeEvolutionHeldSource');
  const held=await read(h.engine,h.guard);expect(Object.keys(held.pins)).toHaveLength(10);expect(held.pins).toEqual(binding.active_files_sha256);expect(held.accepted.core).toBeNull();expect(held.accepted.expert.id).toBe(binding.expert_id);await held.assertUnchanged();
 }finally{await h.close();}
});
test.skipIf(process.env.LUBAN_NATIVE_HELD_ACTUAL!=='1')('actual exFAT same held source accepts filesystem modes and rejects changed held bytes',async()=>{
 const h=await heldFixture({core:true,expert:true,router:true,directory:'/Volumes/移动数据库/lubanllm/experiments'});try{const held=await h.read(h.engine,h.guard);expect(held.pins).toEqual(h.nativeAcceptedHashes(h.directory));await held.assertUnchanged();await writeFile(h.peer.owner.path,'changed');await expect(held.assertUnchanged()).rejects.toThrow();}finally{await h.close();}
});

test('original model owner stat rejects mutation before the first held-source view is captured',async()=>{
 const h=await heldFixture({core:true});try{await writeFile(h.peer.owner.path,'mutated before capture');await expect(h.read(h.engine,h.guard)).rejects.toThrow('Original owner changed before view capture');}finally{await h.close();}
});
