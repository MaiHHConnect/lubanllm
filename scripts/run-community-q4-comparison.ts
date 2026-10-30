/** Reuse the application GPU handoff for one serial, read-only model study. */
import {applicationConnection} from './run-native-identity-round.ts';
import {homedir} from 'node:os';
import {join, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const {values} = parseArgs({options:{output:{type:'string'},backends:{type:'string',default:'native,llama,mlx,mlx-core'}}});
assert(values.output);
const output=resolve(values.output), root='/Volumes/移动数据库/lubanllm/models/community-q4-study-20261017';
assert(output.startsWith(root+'/'));
const connection=await applicationConnection(join(homedir(),'Library/Application Support/Luban'));
const lease_id=crypto.randomUUID();
const receipt={lease_id,acquire_attempted:false,acquired:false,released:false,reloaded:false,exit_code:null as number|null};
const receiptPath=output+'-lease.json';
await writeFile(receiptPath,JSON.stringify(receipt)+'\n',{flag:'wx',mode:0o600});
const save=()=>writeFile(receiptPath,JSON.stringify(receipt)+'\n',{mode:0o600});
const command=async(path:string,body:unknown)=>{
 const response=await fetch(connection.url+path,{method:'POST',headers:{authorization:'Bearer '+connection.token,'content-type':'application/json'},body:JSON.stringify(body),proxy:null} as RequestInit&{proxy:null});
 const value:any=await response.json();assert(response.ok,'Application handoff failed: '+JSON.stringify(value));return value;
};
let wasReady=false, child:ReturnType<typeof Bun.spawn>|undefined;
const cancel=()=>child?.kill('SIGINT');
process.on('SIGINT',cancel);process.on('SIGTERM',cancel);
try {
 receipt.acquire_attempted=true;await save();
 const lease=await command('/internal/native-evolution/acquire',{lease_id});
 receipt.acquired=true;wasReady=lease.was_ready===true;await save();
 child=Bun.spawn([resolve('.venv/bin/python'),resolve('scripts/benchmark_community_q4.py'),'--cases',join(root,'comparison-cases.json'),'--output',output,'--backends',values.backends!],{env:{...process.env,PYTHONPATH:resolve('src')},stdout:'inherit',stderr:'inherit'});
 receipt.exit_code=await child.exited;child=undefined;await save();
 assert(receipt.exit_code===0,'Community model study failed; partial results retained');
} finally {
 if(child){child.kill('SIGINT');await child.exited;}
 if(receipt.acquire_attempted){const released=await command('/internal/native-evolution/release',{lease_id});receipt.released=released.released===true;wasReady ||= released.was_ready===true;await save();}
 if(receipt.released&&wasReady){await command('/api/model/reload',{});receipt.reloaded=true;await save();}
 process.removeListener('SIGINT',cancel);process.removeListener('SIGTERM',cancel);
}
console.log(JSON.stringify(receipt));
