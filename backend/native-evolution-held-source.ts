/** Same-guard view of already admitted source files; no model reads or new source hashes. */
import assert from 'node:assert/strict';
import {fstatSync,lstatSync,readFileSync,type BigIntStats} from 'node:fs';
import {join,resolve} from 'node:path';
import {NativeEngine,assertNativeExpertScopeGuard} from '../.native-migration/research/native-unified-expert/native-engine.ts';
import {borrowNativeEnginePeer} from '../.native-migration/research/native-unified-peer/engine-peer.ts';
import {borrowNativeEvolutionPreparation,type NativeEvolutionPrepared} from '../.native-migration/research/native-evolution-producer/producer.ts';
type Guard=Parameters<Parameters<NativeEngine['withSourceQuiescent']>[0]>[0];
type Peer=ReturnType<typeof borrowNativeEnginePeer>;
const fields=['dev','ino','size','mtimeNs','ctimeNs','mode','nlink','uid']as const;
function same(a:BigIntStats,b:BigIntStats){return fields.every(k=>a[k]===b[k]);}
const views=new WeakMap<NativeEngine,{guard:Guard;peer:Peer;hashes:Peer['hashes'];value:Awaited<ReturnType<typeof capture>>}>();
async function capture(engine:NativeEngine,guard:Guard,peer:Peer){
 // These owners retain their original admission-time stat; do not recapture a changed source.
 for(const owner of [peer.owner,peer.previous?.owner,peer.routerOwner])if(owner)await owner.assertUnchanged();
 if(peer.core)await peer.core.assertRawUnchanged();
 const hashes=peer.hashes,files=new Map<string,BigIntStats|null>(),fds:{fd:number;path:string;stat:BigIntStats}[]=[];
 const path=(name:string)=>{const p=join(peer.store,name);assert(p.startsWith(peer.store+'/')&&resolve(p)===p);return p;};
 for(const name of new Set([...Object.keys(hashes),'active-core.json','active.json','active-router.json','router.json','router.safetensors'])){
  let stat:BigIntStats|null=null;try{stat=lstatSync(path(name),{bigint:true});assert(stat.isFile()&&!stat.isSymbolicLink()&&stat.nlink===1n&&stat.uid===BigInt(process.getuid!()));}catch(error:any){if(error.code!=='ENOENT')throw error;}
  assert(!Object.hasOwn(hashes,name)||stat,'Admitted source file disappeared');files.set(name,stat);
 }
 for(const owner of [peer.owner,peer.core,peer.previous?.owner,peer.routerOwner].filter(Boolean)){
  const stat=fstatSync(owner.handle.fd,{bigint:true});assert(stat.isFile()&&stat.nlink===1n&&same(stat,lstatSync(owner.path,{bigint:true})));fds.push({fd:owner.handle.fd,path:owner.path,stat});
 }
 const check=()=>{
  assertNativeExpertScopeGuard(engine,guard);assert(borrowNativeEnginePeer(engine,guard)===peer&&peer.hashes===hashes,'Held source changed within its guard');peer.worker.assertLive();
  for(const[name,stat]of files){if(stat)assert(same(stat,lstatSync(path(name),{bigint:true})),'Admitted source path changed');else{let absent=false;try{lstatSync(path(name));}catch(error:any){if(error.code!=='ENOENT')throw error;absent=true;}assert(absent,'Previously absent source pointer appeared');}}
  for(const held of fds)assert(same(held.stat,fstatSync(held.fd,{bigint:true}))&&same(held.stat,lstatSync(held.path,{bigint:true})),'Held source FD changed');
 };
 const names=new Set(['active-core.json','active.json','active-router.json']);
 for(const name of [...names])if(Object.hasOwn(hashes,name)){
  const stat=files.get(name)!;assert(stat&&stat.size<=(4n<<20n));const p=JSON.parse(readFileSync(path(name),'utf8')),version=p.version??p.id;assert(typeof version==='string'&&/^[a-z0-9-]+$/.test(version));
  if(name==='active.json'){const selected=Object.keys(hashes).filter(k=>k.endsWith('/'+version+'.json'));assert.equal(selected.length,1);names.add(selected[0]!);names.add(selected[0]!.slice(0,-5)+'.safetensors');}
  else{names.add(version+'.json');names.add(version+'.safetensors');if(name==='active-core.json')names.add(version+'.retention.json');}
 }
 if(!Object.hasOwn(hashes,'active-router.json')){names.add('router.json');names.add('router.safetensors');}
 const pins=Object.freeze(Object.fromEntries([...names].sort().map(name=>[name,hashes[name]??null]))),version=peer.source.runtime.core_label??null;
 const accepted=Object.freeze({core:version?Object.freeze({version,file_sha256:hashes[version+'.safetensors']!,tensor_sha256:peer.source.runtime.core_raw.sha256 as string}):null,expert:peer.previous?Object.freeze({id:peer.previous.id}):null});
 assert(!accepted.core||/^[a-f0-9]{64}$/.test(accepted.core.file_sha256));check();
 return Object.freeze({peer,pins,accepted,assertUnchanged:async()=>check(),assertCurrent:check});
}
export async function readNativeEvolutionHeldSource(engine:NativeEngine,guard:Guard,preparation?:NativeEvolutionPrepared){
 assertNativeExpertScopeGuard(engine,guard);const peer=borrowNativeEnginePeer(engine,guard);
 if(preparation){const held=borrowNativeEvolutionPreparation(engine,guard,preparation);assert(held.store===peer.store&&preparation.base.identity===peer.base.identity);}
 let entry=views.get(engine);if(!entry||entry.guard!==guard||entry.peer!==peer||entry.hashes!==peer.hashes){entry={guard,peer,hashes:peer.hashes,value:await capture(engine,guard,peer)};views.set(engine,entry);}
 await entry.value.assertUnchanged();return entry.value;
}
