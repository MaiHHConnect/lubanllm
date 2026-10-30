/** Only the private Core installation copy is removed, after its real tensor FD closes. */
import assert from 'node:assert/strict';
import {lstat,unlink} from 'node:fs/promises';
import {basename} from 'node:path';
import type {NativeModelTensors} from '../.native-migration/backend/native-model-tensor.ts';
export async function retainNativeEvolutionInstallFile(owner:NativeModelTensors):Promise<NativeModelTensors>{
 assert(/^core-install-\d+\.safetensors\.partial$/.test(basename(owner.path)),'Only private installation copies can be reclaimed');
 const original=await owner.handle.stat({bigint:true});assert(original.isFile()&&original.nlink===1n);
 let removed=false;
 return Object.freeze({...owner,close:async()=>{
  if(removed)return;
  await owner.close();
  // The underlying reader marks closed before awaiting close; retry its FD if that await failed.
  if(owner.handle.fd!==-1)await owner.handle.close();
  assert.equal(owner.handle.fd,-1,'Installed copy must be closed before removal');
  try{
   const named=await lstat(owner.path,{bigint:true});
   assert(named.isFile()&&named.dev===original.dev&&named.ino===original.ino,'Installation path no longer names its owned file');
   await unlink(owner.path);
  }catch(error){if((error as NodeJS.ErrnoException).code!=='ENOENT')throw error;}
  removed=true;
 }});
}
