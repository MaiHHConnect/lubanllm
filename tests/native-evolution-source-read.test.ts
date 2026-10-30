/** Exercise the actual journal sourceSHA and shared raw SHA bodies, on tiny private files only. */
import {test,expect} from 'bun:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fstatSync,lstatSync,closeSync} from 'node:fs';
import {mkdtemp,writeFile,readFile,rm,open,appendFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {openNativeFile,unchangedNativeFile} from '../.native-migration/backend/native-features.ts';
const root=resolve(import.meta.dir,'..');
const compiled=await readFile(join(root,'build/native-evolution/worker.js'),'utf8');
function body(name:string){const begin=compiled.indexOf('async function '+name+'('),end=compiled.indexOf('\n}',begin)+2;assert(begin>=0&&end>begin);return compiled.slice(begin,end);}
const sourceBody=body('sourceSHA').replace(/\b(?:lstatSync|need)\d+\b/g,v=>v.replace(/\d+$/,''))
 .replace(/\bopenRegular\d*\b/g,'openNativeFile').replace(/\bunchanged\d*\b/g,'unchangedNativeFile');
const hashName=/await (fileSHA\d*)\(/.exec(sourceBody)![1]!;
const rawBody=body(hashName).replace(/\b(createHash|BLOCK|WIDTH|need)\d+\b/g,'$1');
const checkpointRawSHA=new Function('createHash','BLOCK','WIDTH','need',rawBody+';return '+hashName+';')(createHash,64<<10,{},assert);
function reader(openFile:any){return new Function('lstatSync','need','openNativeFile',hashName,'unchangedNativeFile',sourceBody+';return sourceSHA;')(lstatSync,assert,openFile,checkpointRawSHA,unchangedNativeFile);}
const digest=(b:Buffer)=>createHash('sha256').update(b).digest('hex');
function closed(fd:number){expect(()=>fstatSync(fd)).toThrow();}
test('original stream reproduces real OS FD leak; explicit journal reads close OS FDs and read each byte once',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'luban-source-read-')),path=join(dir,'tiny'),bytes=Buffer.alloc((3<<16)+17,23);await writeFile(path,bytes,{mode:0o600});
 try{
  const old=await open(path,'r'),oldFD=old.fd;for await(const _ of old.createReadStream({start:0,end:bytes.length-1,autoClose:false})){}await old.close();expect(old.fd).toBe(-1);
  // Bun 1.3.5 can finish close while retaining the OS descriptor. Clean this isolated reproduction explicitly.
  let leaked=false;try{leaked=fstatSync(oldFD).isFile();}catch{}if(leaked)closeSync(oldFD);expect(leaked).toBe(true);
  for(let i=0;i<6;i++){let fd=-1,total=0,calls=0;const read=reader(async(...args:any[])=>{const file=await (openNativeFile as any)(...args);fd=file.handle.fd;const original=file.handle.read.bind(file.handle);file.handle.read=async(...a:any[])=>{calls++;const result=await original(...a);total+=result.bytesRead;return result;};return file;});await read(path,digest(bytes));expect(total).toBe(bytes.length);expect(calls).toBe(4);closed(fd);}
 }finally{await rm(dir,{recursive:true,force:true});}
});
test('actual journal wrong SHA, I/O error, source mutation and mid-read cancellation all close their real FD',async()=>{
 for(const mode of ['sha','io','mutation','cancel']){
  const dir=await mkdtemp(join(tmpdir(),'luban-source-read-failure-')),path=join(dir,'tiny'),bytes=Buffer.alloc((3<<16)+17,41);await writeFile(path,bytes,{mode:0o600});let fd=-1,reads=0;const controller=new AbortController();
  const read=reader(async(...args:any[])=>{const file=await (openNativeFile as any)(...args);fd=file.handle.fd;const original=file.handle.read.bind(file.handle);file.handle.read=async(...a:any[])=>{reads++;if(mode==='io')throw Error('injected read error');const result=await original(...a);if(reads===1){if(mode==='cancel')controller.abort(Error('cancelled'));if(mode==='mutation')await appendFile(path,Buffer.from([1]));}return result;};return file;});
  try{await expect(read(path,mode==='sha'?'0'.repeat(64):digest(bytes),controller.signal)).rejects.toThrow();closed(fd);if(mode==='cancel')expect(reads).toBe(1);}finally{await rm(dir,{recursive:true,force:true});}
 }
});
test('pre-cancelled actual journal opens no FD',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'luban-source-read-cancel-')),path=join(dir,'tiny');await writeFile(path,Buffer.from([1]),{mode:0o600});let opens=0;const c=new AbortController();c.abort();
 try{await expect(reader(async()=>{opens++;throw Error('must not open');})(path,'0'.repeat(64),c.signal)).rejects.toThrow();expect(opens).toBe(0);}finally{await rm(dir,{recursive:true,force:true});}
});
