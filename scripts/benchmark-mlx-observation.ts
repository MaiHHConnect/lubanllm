/** Independent, zero-update comparison of observation order; never grants model qualification. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,lstatSync,statSync,existsSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {execFileSync} from 'node:child_process';
import {MLXLearningClient,lockMLXModel} from '../backend/mlx-learning-client.ts';
import {observeMLXPair,uniqueMLXRecords} from '../backend/mlx-learning-quality.ts';
import type {MLXSource,MLXRow} from '../backend/mlx-learning-source.ts';

const {values}=parseArgs({options:{output:{type:'string'}}});
assert(values.output,'--output must be a new absolute file on the research volume');
const output=resolve(values.output),root='/Volumes/移动数据库/lubanllm';
assert(output===values.output&&output.startsWith(root+'/experiments/')&&!existsSync(output));
assert(lstatSync('/Volumes/移动数据库').isDirectory()&&statSync('/Volumes/移动数据库').dev!==statSync('/Volumes').dev);
const json=(path:string)=>JSON.parse(readFileSync(path,'utf8'));
const store=join(root,'modules/luban0-v1'),core=json(join(store,'active-core.json'));
const active=json(join(store,'active.json')),expert=json(join(store,'layer-23',active.id+'.json'));
const model=join(root,'models/community-q4-study-20261017/mlx-community-4bit');
const download=json(join(model,'download-receipt.json'));
const binary=resolve(import.meta.dir,'../build/native/luban-mlx-worker');
const build=json(binary+'.json'),previousPath=join(store,'learning-903885c92d62.json');
const previous=json(previousPath),all=uniqueMLXRecords(previous.records);
const routes=new Map<string,string>(previous.observations.baseline[0].map((row:MLXRow)=>[row.id,row.route_action]));
// Balance the two real routes, interleave them, and retain every original generation budget.
const cores=all.filter(row=>routes.get(row.id)==='local_core').slice(0,4);
const experts=all.filter(row=>routes.get(row.id)==='neural_expert').slice(0,4);
assert(cores.length===4&&experts.length===4);
const rows=cores.flatMap((row,i)=>[row,experts[i]!]);
const grouped=rows.filter(row=>routes.get(row.id)==='local_core').concat(rows.filter(row=>routes.get(row.id)==='neural_expert'));
// Diagnostic-only bindings: accepted learning proofs are neither consumed nor rewritten.
const source={binary,core,expert,coreFile:join(store,core.version+'.safetensors'),
 expertFile:join(store,'layer-23',active.id+'.safetensors'),modelPath:model,
 bound:{mlx_model_sha256:download.files.find((file:MLXRow)=>file.name==='model.safetensors').sha256}} as MLXSource;
const release=lockMLXModel(core.identity);
const client=new MLXLearningClient(source,join(root,'models/luban0-mlx-metadata-20261020'),
 'd8b0ead81edafada4883e6fea37d1750625a682c469b5f70eb760cd2ab0b1237');
const result:MLXRow={schema:'mlx-observation-order-benchmark-v1',started:new Date().toISOString(),
 hardware:execFileSync('sysctl',['-n','hw.model','hw.memsize','machdep.cpu.brand_string'],{encoding:'utf8'}).trim().split('\n'),
 os:execFileSync('sw_vers',['-productVersion'],{encoding:'utf8'}).trim(),precision:'MLX Q4 affine group64; BF16 accepted core/expert',
 seed:42,temperature:0,cache_reuse:false,compilation_disabled:Boolean(process.env.MLX_DISABLE_COMPILE),model_sha256:source.bound.mlx_model_sha256,worker_sha256:build.binary_sha256,
 core:core.version,expert:active.id,source_report:previousPath,split:'Previously exposed TRAIN/DEV/protection; no untouched test',
 optimizer_updates:0,teacher_calls:0,adopted:false,file_cache:'Not purged; same resident worker, ABBA order after route warmup',
 memory_measurement:'Child RSS snapshots at load and between blocks, in bytes; not peak physical footprint',blocks:[]};
const signature=(row:MLXRow)=>JSON.stringify([row.id,row.input_tokens,row.tokens,row.complete,row.finish_reason,row.route_action]);
const rss=()=>Number(execFileSync('ps',['-o','rss=','-p',String((client as any).child.pid)],{encoding:'utf8'}).trim())*1024;
let selected='parent',switches=0,selectionMS=0,prefillNS=0,decodeNS=0;
const command=client.command.bind(client);
client.command=async(op,body={},signal)=>{
 const at=performance.now(),answer=await command(op,body,signal);
 if(op==='train_select'){result.learning_observation_compilation_enabled=answer.compilation_enabled;selectionMS+=performance.now()-at;if(selected!==body.source){switches++;selected=body.source;}}
 if(op==='generate'){prefillNS+=answer.prefill_ns??0;decodeNS+=answer.decode_ns??0;}
 return answer;
};
try{
 result.loaded=await client.start();result.loaded_rss_bytes=rss();
 for(const row of [cores[0]!,experts[0]!])await client.generate(row,routes.get(row.id)==='local_core'?'core':'parent');
 for(const order of ['interleaved','grouped','grouped','interleaved']){
  await client.command('train_select',{source:'parent'});
  switches=0;selectionMS=0;prefillNS=0;decodeNS=0;
  const started=performance.now();
  const pair=await observeMLXPair(client,order==='grouped'?grouped:rows,'current',routes);
  const seconds=(performance.now()-started)/1000;
  const aligned=pair.map(pass=>rows.map(row=>pass.find(value=>value.id===row.id)!));
  const block={order,seconds,source_switches:switches,selection_ms:selectionMS,prefill_ns:prefillNS,decode_ns:decodeNS,
   repeated_mismatches:aligned[0]!.filter((row,i)=>signature(row)!==signature(aligned[1]![i]!)).map(row=>row.id),
   pair:aligned,rss_bytes:rss()};
  result.blocks.push(block);
  console.log(JSON.stringify({order,seconds,switches,mismatches:block.repeated_mismatches.length}));
 }
 const reference=result.blocks[0].pair[0];
 result.cross_block_mismatches=result.blocks.map((block:MLXRow)=>block.pair.map((pass:MLXRow[])=>
  pass.filter((row,i)=>signature(row)!==signature(reference[i])).map(row=>row.id)));
 result.outputs_exact=result.cross_block_mismatches.every((pair:string[][])=>pair.every(pass=>pass.length===0));
 result.continuous=[];
 for(const row of [cores[3]!,experts[2]!]){
  const observations=[];
  for(let i=0;i<8;i++)observations.push(await client.generate(row,routes.get(row.id)==='local_core'?'core':'parent'));
  const mismatches=observations.filter(value=>signature({...value,route_action:routes.get(row.id)})!==signature({...observations[0],route_action:routes.get(row.id)})).length;
  result.continuous.push({id:row.id,source:observations[0]!.source,mismatches,observations});
  console.log(JSON.stringify({continuous_source:observations[0]!.source,mismatches}));
 }
 result.outputs_exact&&=result.continuous.every((row:MLXRow)=>row.mismatches===0);
 const fullStarted=performance.now();
 result.full_pair=await observeMLXPair(client,all,'current',routes);
 result.full_seconds=(performance.now()-fullStarted)/1000;
 result.full_mismatches=result.full_pair[0].filter((row:MLXRow,i:number)=>signature(row)!==signature(result.full_pair[1][i])).map((row:MLXRow)=>row.id);
 result.outputs_exact&&=result.full_mismatches.length===0;
 console.log(JSON.stringify({full_records:all.length,full_seconds:result.full_seconds,mismatches:result.full_mismatches.length}));
 result.complete=true;
}catch(error){result.error=String(error);throw error;}
finally{
 try{await client.close();result.worker_closed=true;}finally{release();result.finished=new Date().toISOString();writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});}
}
assert(result.outputs_exact,'Observation order or repeated generation changed outputs; candidate is not admitted');
