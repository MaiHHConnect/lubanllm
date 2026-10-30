/** Compile terminal conversion separately; --install updates only the trainer role after CPU evidence. */
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,copyFile,chmod} from 'node:fs/promises';
import {join,resolve,basename} from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {writeNativeEvolutionTrainerConversion} from './native-evolution-trainer-conversion.ts';
import {parseImage} from './package-native-runtime.ts';
import {checkReleaseMachO} from './release-macho.ts';
import {system} from '../backend/native-runtime-macho.ts';
const root=resolve(import.meta.dir,'..'),output=join(root,'build/native-evolution-conversion'),install=process.argv.includes('--install');
assert(process.argv.slice(2).every(v=>v==='--install'||v==='--compile-only'));
const sha=(raw:string|Uint8Array)=>createHash('sha256').update(raw).digest('hex');
const binding=async(path:string)=>{const raw=await readFile(path);return{bytes:raw.length,sha256:sha(raw)};};
const records:any[]=[];
function run(argv:string[]){const r=spawnSync(argv[0]!,argv.slice(1),{encoding:'utf8',timeout:90000,maxBuffer:8<<20});records.push({argv,exit_code:r.status,stderr:r.stderr});assert.equal(r.status,0,r.stderr);return r.stdout;}
await mkdir(output,{recursive:true});
if(!install){
 const original=JSON.parse(await readFile(join(root,'.native-migration/research/native-mps-training/build-result.json'),'utf8'));
 const patch=await writeNativeEvolutionTrainerConversion(root,output),binary=join(output,'training-conversion');
 const sources={...original.sources,training:{path:patch.path,...await binding(patch.path)},conversion:{path:patch.header,...await binding(patch.header)},basis:{path:join(root,'.native-migration/research/native-replay-basis/ReplayBasis.h'),...await binding(join(root,'.native-migration/research/native-replay-basis/ReplayBasis.h'))}};
 const argv=(original.argv as string[]).map((v,i,a)=>v.endsWith('/Training.v3.cpp')?patch.path:i===a.indexOf('-o')+1?binary:v.startsWith('-DLUBAN_V3_TRAINING_SOURCE_SHA=')?'-DLUBAN_V3_TRAINING_SOURCE_SHA="'+sources.training.sha256+'"':v);
 run(argv);run(['/usr/bin/codesign','--force','--sign','-',binary]);
 const cpu=join(output,'conversion-cpu'),cpuSource=join(root,'native/EvolutionConversionCPU.cpp');
 run(argv.map((v,i,a)=>v===patch.path?cpuSource:i===a.indexOf('-o')+1?cpu:v));run(['/usr/bin/codesign','--force','--sign','-',cpu]);
 const build={schema:'native-terminal-conversion-build-v1',binary,binary_sha256:(await binding(binary)).sha256,cpu,sources,original_trainer:original.binary,original_binary_sha256:original.binary_sha256,original_sources:original.sources,argv,commands:records,inverse_exact:patch.inverse_exact,edits:patch.edits,model_loaded:false,product_admitted:false};
 await writeFile(join(output,'build.json'),JSON.stringify(build,null,2)+'\n');console.log(JSON.stringify({binary,cpu,installed:false}));
}else{
 const build=JSON.parse(await readFile(join(output,'build.json'),'utf8')),cpu=JSON.parse(await readFile(join(output,'cpu-proof.json'),'utf8'));
 assert.equal((await binding(build.binary)).sha256,build.binary_sha256);assert.equal(cpu.binary_sha256,build.binary_sha256);
 assert(cpu.unchanged_math.all_seven_state_bit_exact&&cpu.unchanged_math.actual_steps_each_arm===2&&cpu.unchanged_math.deterministic_raw_equal===37&&cpu.terminal_boundary&&cpu.resume_bit_exact&&cpu.cancel_before_commit&&cpu.parent_rng_restored);
 const runtimePath=join(root,'build/native-evolution/runtime-authority.json'),runtime=JSON.parse(await readFile(runtimePath,'utf8'));
 const manifestPath=join(runtime.runtimeRoot,runtime.authority.manifest_relative),manifest=JSON.parse(await readFile(manifestPath,'utf8'));
 const relative=runtime.roles.trainer.path,target=join(runtime.runtimeRoot,relative),original=await binding(build.binary);
 await copyFile(build.binary,target);await chmod(target,0o755);
 const image=(path:string)=>parseImage(run(['/usr/bin/otool','-L',path]),run(['/usr/bin/otool','-l',path]));
 const before=image(target),edits:string[]=[];
 for(const dep of before.dependencies)if(!system(dep)){assert(manifest.libraries['native/lib/'+basename(dep)]);edits.push('-change',dep,'@rpath/'+basename(dep));}
 for(const path of before.rpaths)if(path!=='@loader_path/../lib')edits.push('-delete_rpath',path);
 if(!before.rpaths.includes('@loader_path/../lib'))edits.push('-add_rpath','@loader_path/../lib');
 if(edits.length)run(['/usr/bin/install_name_tool',...edits,target]);const relocated=await binding(target);
 run(['/usr/bin/codesign','--force','--sign','-',target]);run(['/usr/bin/codesign','--verify','--strict',target]);const final=await binding(target);checkReleaseMachO(await readFile(target),relative);
 const proof={kind:'trainer',source_build_sha256:sha(JSON.stringify(build)),original_binary_sha256:original.sha256,approved_binary_sha256:original.sha256,compile:{argv:build.argv,exit_code:0},inverse_verified:true,abi:'LUBAN_NATIVE_TRAINING_V3',build:{product_admitted:false,sources:Object.fromEntries(Object.entries(build.sources).map(([k,v]:[string,any])=>[k,{sha256:v.sha256}]))},cpu_proof:cpu.unchanged_math,terminal_conversion:{method:'gptq-bf16-refine-v1',cpu_boundary_verified:true,independent_actual_shape_qualification:false}};
 manifest.programs.trainer.proof=proof;runtime.authority.sources.trainer=sha(JSON.stringify(proof));
 manifest.payload[relative]={...manifest.payload[relative],original,copied:original,stripped:original,relocated,final,image:image(target),commands:records.map(({argv,exit_code})=>({argv,exit_code})),debug_error:null};
 runtime.roles.trainer.sha256=final.sha256;runtime.training={source_build_receipt_sha256:proof.source_build_sha256,training_source_sha256:proof.build.sources.training.sha256,rng_source_sha256:proof.build.sources.rng.sha256,cpu_rng_helper_sha256:proof.build.sources.cpu.sha256,mps_rng_helper_sha256:proof.build.sources.mps.sha256};
 const text=JSON.stringify(manifest,null,2)+'\n';await writeFile(manifestPath,text);runtime.authority.manifest_sha256=sha(text);runtime.manifestBytes=Buffer.byteLength(text);runtime.qualification=false;
 await writeFile(runtimePath,JSON.stringify(runtime,null,2)+'\n');console.log(JSON.stringify({target,final,installed:true,independent_graph_qualified:false}));
}
