/** Actual ordinary Q4 observer check; no optimizer, teacher or product mutation. */
import assert from 'node:assert/strict';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {NativeQ4Observer,assertNativeQ4Observation} from '../backend/native-evolution-q4.ts';
import {originalCoreProtectionRows} from '../backend/native-evolution-material.ts';

assert(process.argv.length===3,'Usage: bun scripts/verify-native-evolution-q4.ts NEW_WORK_DIRECTORY');
const root=resolve(import.meta.dir,'..'),work=resolve(process.argv[2]!);
const base=JSON.parse(await readFile('/Volumes/移动数据库/lubanllm/modules/luban0-v1/original-core-protection.json','utf8'));
const records=originalCoreProtectionRows(base,'673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df',
 '1706e50833fffb25da45ad958af043a688e5ef495d45f33fa4b1ab623c696719');
await mkdir(work,{mode:0o700});
const observer=new NativeQ4Observer(),started=performance.now();
let proofs:Awaited<ReturnType<NativeQ4Observer['observePair']>>|undefined,error:unknown;
try{
 proofs=await observer.observePair({executable:join(root,'build/native/luban-evolution-observe'),binary:join(root,'build/native/luban-worker'),
  model:'/Volumes/移动数据库/lubanllm/models/Qwen3.5-2B-luban0-v1',identity:base.identity,
  manifestSHA:'d8b0ead81edafada4883e6fea37d1750625a682c469b5f70eb760cd2ab0b1237',records,work});
 for(const proof of proofs)assertNativeQ4Observation(proof,records);
 assert.deepEqual(proofs[0].rows,proofs[1].rows,'Same resident weight two-pass output differs');
}catch(value){error=value;}finally{await observer.close();}
const result={passed:!error,error:error instanceof Error?error.message:error===undefined?null:String(error),
 actual_model:true,proofs,seconds:(performance.now()-started)/1000,closed:true,
 optimizer_steps:0,teacher_calls:0,product_admitted:false,scope:'eight existing original protection rows; ordinary observer integration only'};
await writeFile(join(work,'result.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});
console.log(JSON.stringify({work,passed:result.passed,seconds:result.seconds,correct:proofs?.map(p=>p.rows.filter(r=>r.passed).length)}));
if(error)throw error;
