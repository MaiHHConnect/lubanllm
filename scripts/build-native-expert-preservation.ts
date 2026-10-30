/** CPU math harness only. The application persistent role is integrated by its existing builder. */
import assert from 'node:assert/strict';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
const root=resolve(import.meta.dir,'..'),output=join(root,'build/native-expert-preservation');
assert.equal(process.argv.length,2);await mkdir(output,{recursive:true});
const original=JSON.parse(await readFile(join(root,'.native-migration/research/native-mps-training/build-result.json'),'utf8'));
const binary=join(output,'preservation-cpu'),source=join(root,'native/ExpertPreservationCPU.cpp');
const argv=(original.argv as string[]).map((value,index,args)=>value.endsWith('/Training.v3.cpp')?source:index===args.indexOf('-o')+1?binary:value);
const processResult=spawnSync(argv[0]!,argv.slice(1),{encoding:'utf8',timeout:90000,maxBuffer:4<<20});
assert.equal(processResult.status,0,processResult.stderr);
await writeFile(join(output,'build.json'),JSON.stringify({binary,source,argv,exit_code:processResult.status,GPU:false,model_loaded:false,product_admitted:false},null,2)+'\n');
console.log(JSON.stringify({binary,GPU:false,model_loaded:false}));
