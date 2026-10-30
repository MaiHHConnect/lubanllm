/** Execute the real CPU RNG lifecycle without a model or GPU. */
import {readFile,mkdir,realpath,mkdtemp} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import assert from 'node:assert/strict';
import {tmpdir} from 'node:os';
const root=resolve(import.meta.dir,'..'),build=JSON.parse(await readFile(join(root,'build/native-evolution/tail-build.json'),'utf8'));
const original=build.commands.find((v:any)=>v.argv[0]==='/usr/bin/clang++').argv as string[],source=original.find(v=>v.endsWith('/PersistentEvolution.cpp'))!,out=join(root,'build/native-evolution/evolution-round-cpu');
const command=original.map((v,i)=>v===source?join(root,'native/EvolutionRoundCPU.cpp'):i===original.indexOf('-o')+1?out:v);
const compile=spawnSync(command[0]!,command.slice(1),{encoding:'utf8',timeout:90000});assert.equal(compile.status,0,compile.stderr);
const directory=await realpath(await mkdtemp(join(tmpdir(),'native-evolution-round-'))),result=spawnSync(out,[join(directory,'output')],{encoding:'utf8',timeout:30000});assert.equal(result.status,0,result.stderr);
console.log(await readFile(join(directory,'output/result.json'),'utf8'));
