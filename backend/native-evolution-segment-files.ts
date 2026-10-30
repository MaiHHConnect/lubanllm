/** Private APFS staging. Reuse validated bytes, and delete only superseded closed scratch directories. */
import assert from 'node:assert/strict';
import {constants,type BigIntStats} from 'node:fs';
import {copyFile,lstat,open,realpath,rm,type FileHandle} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
const clones=new Map<string,{stat:BigIntStats;sha256:string}>(),directories=new Map<string,BigIntStats>();
const same=(a:BigIntStats,b:BigIntStats)=>a.dev===b.dev&&a.ino===b.ino&&a.size===b.size&&a.mtimeNs===b.mtimeNs&&a.ctimeNs===b.ctimeNs;
async function unchanged(path:string,handle:FileHandle,before:BigIntStats){const [named,held]=await Promise.all([lstat(path,{bigint:true}),handle.stat({bigint:true})]);assert(named.isFile()&&!named.isSymbolicLink()&&named.nlink===1n&&same(named,before)&&same(held,before),'Validated segment source changed');}
export async function cloneNativeEvolutionInput(source:{path:string;handle:FileHandle;stat:BigIntStats;sha256:string},destination:string,signal?:AbortSignal){
 signal?.throwIfAborted();assert(resolve(destination)===destination&&await realpath(dirname(destination))===dirname(destination));await unchanged(source.path,source.handle,source.stat);
 // FORCE fails visibly on unsupported volumes; the application creates scratch on its local APFS volume.
 await copyFile(source.path,destination,constants.COPYFILE_EXCL|constants.COPYFILE_FICLONE_FORCE);
 try{const file=await open(destination,constants.O_RDONLY|constants.O_NOFOLLOW);let stat:BigIntStats;
  try{await file.chmod(0o600);await file.sync();stat=await file.stat({bigint:true});assert(stat.isFile()&&stat.nlink===1n&&stat.size===source.stat.size&&stat.uid===BigInt(process.getuid!()));}finally{await file.close();}
  await unchanged(source.path,source.handle,source.stat);signal?.throwIfAborted();clones.set(destination,{stat,sha256:source.sha256});
 }catch(error){await rm(destination,{force:true});throw error;}
}
export function isNativeEvolutionClone(path:string,stat:BigIntStats,sha256:string){const bound=clones.get(path);if(!bound)return false;assert(bound.sha256===sha256&&same(stat,bound.stat),'Private segment clone was replaced or changed');return true;}
export async function retainNativeEvolutionScratch(path:string){assert(resolve(path)===path&&await realpath(path)===path&&!directories.has(path));const stat=await lstat(path,{bigint:true});assert(stat.isDirectory()&&!stat.isSymbolicLink()&&stat.uid===BigInt(process.getuid!())&&(stat.mode&0o7777n)===0o700n);directories.set(path,stat);}
/** Caller has saved/read back the successor and closed every owner of this directory. */
export async function removeNativeEvolutionScratch(path:string){const before=directories.get(path);assert(before,'Only this run newly-created scratch may be removed');const stat=await lstat(path,{bigint:true});assert(stat.isDirectory()&&!stat.isSymbolicLink()&&stat.dev===before.dev&&stat.ino===before.ino&&await realpath(path)===path,'Native scratch directory was replaced');await rm(path,{recursive:true});directories.delete(path);for(const key of clones.keys())if(key.startsWith(path+'/'))clones.delete(key);}

// A parsed raw is reusable only through its same live FD and unchanged named inode.
type RawBinding={path:string;handle:FileHandle;bytes:number;dtype:string;sha256:string};
const verifiedRaws=new WeakMap<FileHandle,RawBinding&{stat:BigIntStats}>();
export async function retainNativeEvolutionVerifiedRaw(raw:RawBinding,stat:BigIntStats){
 assert(raw.handle.fd>=0&&stat.size===BigInt(raw.bytes));await unchanged(raw.path,raw.handle,stat);
 verifiedRaws.set(raw.handle,{...raw,stat});
}
export async function isNativeEvolutionVerifiedRaw(raw:RawBinding){
 const held=verifiedRaws.get(raw.handle);if(!held)return false;
 assert(raw.path===held.path&&raw.bytes===held.bytes&&raw.dtype===held.dtype&&raw.sha256===held.sha256,'Verified raw descriptor changed');
 await unchanged(held.path,held.handle,held.stat);return true;
}
