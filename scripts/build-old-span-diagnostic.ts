/** Standalone CPU research helper only; never rebuilds or qualifies the model worker. */
import assert from 'node:assert/strict';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {spawnSync} from 'node:child_process';
const root=resolve(import.meta.dir,'..'),output=join(root,'build/old-span-diagnostic');
const prior=JSON.parse(await readFile(join(root,'build/native-evolution/tail-build.json'),'utf8'));
const original=prior.commands.find((v:any)=>v.argv[0]==='/usr/bin/clang++'&&v.argv.includes('-o')).argv as string[];
const cpp=original.find(v=>v.endsWith('.cpp'))!;assert(cpp);
await mkdir(output,{recursive:true});
const binary=join(output,'project-old-span'),argv=original.map((v,i)=>v===cpp?join(root,'research/native_old_span_diagnostic.cpp'):i===original.indexOf('-o')+1?binary:v);
const results=[];
for(const command of [argv,['/usr/bin/codesign','--force','--sign','-',binary]]){
 const r=spawnSync(command[0]!,command.slice(1),{encoding:'utf8',timeout:90000,maxBuffer:4<<20});results.push({argv:command,exit_code:r.status,stderr:r.stderr});assert.equal(r.status,0,r.stderr);
}
await writeFile(join(output,'build.json'),JSON.stringify({binary,commands:results,CPU_only:true,product_worker_changed:false,qualified:false},null,2)+'\n');
console.log(binary);
