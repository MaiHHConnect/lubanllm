/** Reuse a scope's admitted files. Checks do not reread tensors or rehash the source. */
import assert from 'node:assert/strict';
import {lstatSync,readFileSync,type BigIntStats} from 'node:fs';
import {join,resolve} from 'node:path';
const fields=['dev','ino','size','mtimeNs','ctimeNs','mode','nlink','uid'] as const;
export function nativeExpertSourceCheck(store:string,files:Record<string,Buffer>){
 const names=new Set([...Object.keys(files),'active-core.json','active.json','active-router.json','router.json','router.safetensors']);
 const path=(name:string)=>{const p=join(store,name);assert(resolve(p)===p&&p.startsWith(store+'/'));return p;};
 const pins=new Map<string,BigIntStats|null>();
 for(const name of names){let stat:BigIntStats|null=null;try{stat=lstatSync(path(name),{bigint:true});assert(stat.isFile()&&!stat.isSymbolicLink()&&stat.nlink===1n&&stat.uid===BigInt(process.getuid!()));}catch(error:any){if(error.code!=='ENOENT')throw error;}
  const raw=files[name];assert(Boolean(stat)===Boolean(raw),'Admitted expert source presence changed');
  if(raw){assert(stat!.size===BigInt(raw.byteLength));if(!name.endsWith('.safetensors'))assert(readFileSync(path(name)).equals(raw),'Source metadata changed during admission');}
  pins.set(name,stat);
 }
 return()=>{for(const[name,before]of pins){let now:BigIntStats|null=null;try{now=lstatSync(path(name),{bigint:true});}catch(error:any){if(error.code!=='ENOENT')throw error;}
  assert(before===null?now===null:now!==null&&fields.every(k=>before[k]===now![k]),'Admitted expert source file changed');
 }};
}
