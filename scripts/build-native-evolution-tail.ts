/** Build the daily tail command alongside the unchanged audit command. No model is loaded. */
import assert from 'node:assert/strict';
import {readFile,writeFile,mkdir,copyFile,stat,chmod} from 'node:fs/promises';
import {constants} from 'node:fs';
import {join,resolve,dirname,basename} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {parseImage} from './package-native-runtime.ts';
import {checkReleaseMachO} from './release-macho.ts';
import {system} from '../backend/native-runtime-macho.ts';
import {writeEvolutionObservationSources} from './native-evolution-observation-source.ts';
import {writeEvolutionSourceSwapSources} from './native-evolution-source-swap-patches.ts';
import {patchEvolutionRoundSource,writeEvolutionCoreStateSource} from './native-evolution-round-source.ts';
import {writeEvolutionSourcePermissions} from './native-evolution-source-permissions.ts';
import {writeNativeExpertPreservationSources,writeNativeExpertPreservationState,patchNativeExpertPreservationCommand} from './native-expert-preservation-source.ts';
const root=resolve(import.meta.dir,'..'),research=join(root,'.native-migration/research'),output=join(root,'build/native-evolution');
await mkdir(output,{recursive:true});
const prior=JSON.parse(await readFile(join(research,'native-canonical-core-source-close-role-v2/build.json'),'utf8'));
const command=prior.commands.find((value:any)=>value.argv[0]==='/usr/bin/clang++'&&value.argv.includes('-o')).argv as string[];
const source=command.find(value=>value.endsWith('/PersistentMPSV3.cpp'))!;
const originalSource=await readFile(source,'utf8');let body=originalSource;
const transformations:{before:string;after:string;occurrences:number;positions:number[]}[]=[];
const change=(before:string,after:string,occurrences=1)=>{assert.equal(body.split(before).length,occurrences+1,before);const positions:number[]=[];for(let at=body.indexOf(before);at>=0;at=body.indexOf(before,at+before.length))positions.push(at);transformations.push({before,after,occurrences,positions});body=body.replaceAll(before,after);};
const replace=(before:string,after:string)=>change(before,after);
for(const match of [...body.matchAll(/^#include "([^"]+)"/gm)])replace(match[0],'#include '+JSON.stringify(resolve(dirname(source),match[1]!)));
// The helper needs CachedGraph; include it immediately after the existing tail header.
const permissions=await writeEvolutionSourcePermissions(root,output);
const observation=await writeEvolutionObservationSources(root,output,permissions.graph);
const expertPreservation=await writeNativeExpertPreservationSources(root,output,observation.optimizer);
observation.records.push(...expertPreservation.records);
const expertState=await writeNativeExpertPreservationState(root,output);
observation.records.push(expertState.record);
replace('#include '+JSON.stringify(join(research,'native-unified-expert/StateCommands.h')),'#include '+JSON.stringify(expertState.path));
observation.records.push(...permissions.records);
replace('#include '+JSON.stringify(join(research,'native-unified-final-role-v1/RouterCommands.h')),'#include '+JSON.stringify(permissions.routerCommands));
replace('#include '+JSON.stringify(join(research,'native-unified-tail-role-v1/Generation.cpp')),'#include '+JSON.stringify(observation.generation));
replace('#include '+JSON.stringify(join(research,'native-unified-final-role-v1/ExpertOptimizerV3.h')),'#include '+JSON.stringify(expertPreservation.optimizer));
const oldHeader='#include '+JSON.stringify(join(research,'native-evolution-retained-producer-v1/TailExport.h'));
replace(oldHeader,oldHeader+'\n#include '+JSON.stringify(join(root,'native/EvolutionTail.h')));
replace('\\"training_tail_export\\":true,','\\"training_tail_export\\":true,\\"ordinary_training_tail_export\\":true,');
replace('else if(op=="training-tail"){','else if(op=="training-tail"||op=="training-tail-ordinary"){');
replace('bool branchEnabled=true;','bool branchEnabled=true;bool observePrevious=false;std::string currentCandidateSHA;');
replace('   guardRetainedCoreSource(pendingCoreSource,op,coreSourceConfirming);','   need(!observePrevious||op=="run"||op=="loss"||op=="core-observation"||op=="core-drop"||op=="close","Training/state mutation requires current candidate observation");\n   guardRetainedCoreSource(pendingCoreSource,op,coreSourceConfirming);');
replace('need(!failed||op=="core-source-apply"','need(!failed||op=="core-observation"||op=="core-source-apply"');
replace('if(op=="core-source-stage"){',`if(op=="core-observation"){
    std::string mode;in>>mode;need(bool(in)&&!(in>>extra)&&(mode=="previous"||mode=="current")&&currentCore&&backbone&&acceptedDown.defined()&&!branchEnabled,"Observation requires one installed candidate and retained accepted source");
    observePrevious=mode=="previous";backbone->weights.at(PREFIX+"layers.23.mlp.down_proj.weight")=evolutionObservationTensor(observePrevious,acceptedDown,currentCore->tensors[0]);
   }else if(op=="core-source-stage"){`);
replace('currentCore=std::make_unique<ResidentCoreState>(in,2048,torch::kMPS);',`currentCore=std::make_unique<ResidentCoreState>(in,2048,torch::kMPS);observePrevious=false;{std::istringstream binding(line);std::string field,offset;for(int n=0;n<4;n++)binding>>field;binding>>std::quoted(field)>>field>>offset>>currentCandidateSHA;need(bool(binding)&&isSHA(currentCandidateSHA),"Candidate source binding absent");}`);
replace('currentCore.reset();failed=false;','currentCore.reset();currentCandidateSHA.clear();observePrevious=false;failed=false;');
// Only inference's resident selection bypasses the old full-state candidate() rehash.
for(const [before,after] of [['backbone->weights.at(PREFIX+"layers.23.mlp.down_proj.weight")=currentCore->candidate();','backbone->weights.at(PREFIX+"layers.23.mlp.down_proj.weight")=evolutionObservationTensor(observePrevious,acceptedDown,currentCore->tensors[0]);'],['currentCore?&currentCore->candidate():nullptr','currentCore?&evolutionObservationTensor(observePrevious,acceptedDown,currentCore->tensors[0]):nullptr']])change(before!,after!,body.split(before!).length-1);
replace('<<jsonString(op)<<",\\"core_source_confirming\\":','<<jsonString(op)<<",\\"observation\\":"<<jsonString(observePrevious?"previous":"current")<<",\\"core_tensor_sha256\\":"<<(currentCore&&backbone?jsonString(observePrevious?(backbone->initial.config.at("core_version")=="null"?ORIGINAL:backbone->initial.config.at("core_tensor_sha256")):currentCandidateSHA):"null")<<",\\"step\\":"<<(currentCore&&!observePrevious?currentCore->step:0)<<",\\"core_source_confirming\\":');
const anchor='const auto boundCoreSHA=spec.config.at("core_version")=="null"?ORIGINAL:spec.config.at("core_tensor_sha256");';
replace(anchor,anchor+'if(op=="training-tail-ordinary"){need(!currentCore,"Daily tail requires the accepted resident backbone");exportOrdinaryTrainingTail(graph,spec.tokens,labels,root/name,hash,labelsSHA,sampleSHA,sampleRawSHA,boundCoreSHA,spec.config.at("core_version"));}else{');
replace('durableText(root/name/"source.json",provenance.str());syncDir(root/name);','durableText(root/name/"source.json",provenance.str());syncDir(root/name);}');
replace('root/name);backbone->check(spec);sync(torch::kMPS);','root/name,true);backbone->check(spec);sync(torch::kMPS);');
replace('currentCore?&evolutionObservationTensor(observePrevious,acceptedDown,currentCore->tensors[0]):nullptr);need(code==0', 'currentCore?&evolutionObservationTensor(observePrevious,acceptedDown,currentCore->tensors[0]):nullptr,op=="run",currentCore&&!observePrevious?currentCandidateSHA:(spec.config.at("core_version")=="null"?ORIGINAL:spec.config.at("core_tensor_sha256")));need(code==0');
const swap=await writeEvolutionSourceSwapSources(root,output);
replace('#include '+JSON.stringify(join(research,'native-canonical-core-source-close-role-v2/CoreSourceDispatch.h')),'#include '+JSON.stringify(swap.header));
for(const edit of swap.changes){if(edit.before.startsWith('<<jsonString(op)')){
 const prefix='<<(currentCore&&!observePrevious?currentCore->step:0)';replace(prefix+edit.before.slice('<<jsonString(op)'.length),prefix+edit.after.slice('<<jsonString(op)'.length));
}else replace(edit.before,edit.after);}
observation.records.push(...swap.records);
patchEvolutionRoundSource(root,replace);
const coreState=await writeEvolutionCoreStateSource(root,output);
replace('#include '+JSON.stringify(join(research,'native-unified-core-peer/CoreStateV2.h')),'#include '+JSON.stringify(coreState.path));
replace('currentCore->residual(root/name);','currentCore->residual(root/name,true);');
observation.records.push(coreState.record);
body=patchNativeExpertPreservationCommand(body,(current,before,after)=>{assert.equal(current,body);replace(before,after);return body;});
const generated=join(output,'PersistentEvolution.cpp');await writeFile(generated,body);
let reversed=body;for(const edit of transformations.toReversed()){
 for(let index=edit.positions.length-1;index>=0;index--){const at=edit.positions[index]!+index*(edit.after.length-edit.before.length);assert.equal(reversed.slice(at,at+edit.after.length),edit.after,'Non-invertible native overlay');reversed=reversed.slice(0,at)+edit.before+reversed.slice(at+edit.after.length);}
}assert.equal(reversed,originalSource,'Native overlay inverse did not reproduce its actual parent');
const records:any[]=[];
function run(argv:string[]){const result=spawnSync(argv[0]!,argv.slice(1),{encoding:'utf8',timeout:90000,maxBuffer:8<<20});records.push({argv,exit_code:result.status,stderr:result.stderr});assert.equal(result.status,0,result.stderr);return result.stdout;}
const hash=(raw:string|Uint8Array)=>createHash('sha256').update(raw).digest('hex');
const binary=join(output,'persistent-evolution');run(command.map((arg,index)=>arg===source?generated:index===command.indexOf('-o')+1?binary:arg.startsWith('-DLUBAN_PREFIX_SOURCE_SHA=')?'-DLUBAN_PREFIX_SOURCE_SHA="'+hash(body)+'"':arg));
run(['/usr/bin/codesign','--force','--sign','-',binary]);run(['/usr/bin/codesign','--verify','--strict',binary]);
const cpu=join(output,'evolution-tail-cpu'),cpuSource=join(root,'native/EvolutionTailCPU.cpp');
run([...command.map((arg,index)=>arg===source?cpuSource:index===command.indexOf('-o')+1?cpu:arg),'-DLUBAN_EVOLUTION_OPTIMIZER_HEADER='+JSON.stringify(observation.optimizer)]);
run(['/usr/bin/codesign','--force','--sign','-',cpu]);
await writeFile(join(output,'tail-build.json'),JSON.stringify({passed:true,binary,cpu,commands:records,ordinary_command:'training-tail-ordinary',audit_command_preserved:'training-tail',GPU:false,model_loaded:false,package_admitted:false},null,2)+'\n');
// Product payload copies are distinct from the frozen research capsule. Existing unchanged provenance is reused.
const basisReady=JSON.parse(await readFile(join(research,'native-replay-basis-product-role-v1/ready.json'),'utf8'));
const oldRoot=basisReady.Resources as string,runtimeRoot=join(output,'Resources');
const authorityText=await readFile(join(research,'native-replay-basis-product-role-v1/authority.ts'),'utf8');
let authority=JSON.parse(authorityText.slice(authorityText.indexOf('{'),authorityText.lastIndexOf('}')+1));
let manifest=JSON.parse(await readFile(join(oldRoot,authority.manifest_relative),'utf8'));
try{const current=JSON.parse(await readFile(join(output,'runtime-authority.json'),'utf8'));authority=current.authority;manifest=JSON.parse(await readFile(join(runtimeRoot,authority.manifest_relative),'utf8'));}catch(error:any){if(error.code!=='ENOENT')throw error;}
const relative=authority.paths.persistentExpert;
for(const path of ['','native','native/bin','native/lib','native/licenses'])await mkdir(join(runtimeRoot,path),{recursive:true,mode:0o700});
for(const path of Object.keys({...manifest.payload,...manifest.licenses,...manifest.assets})){
 if(path===relative)continue;
 const target=join(runtimeRoot,path);await mkdir(dirname(target),{recursive:true,mode:0o700});
 try{await copyFile(join(oldRoot,path),target,constants.COPYFILE_EXCL|constants.COPYFILE_FICLONE);}catch(error:any){if(error.code!=='EEXIST')throw error;}
}
const binding=async(path:string)=>{const raw=await readFile(path);return{bytes:raw.length,sha256:hash(raw)};};
const target=join(runtimeRoot,relative),original=await binding(binary);await copyFile(binary,target);await chmod(target,0o755);
const image=(path:string)=>parseImage(run(['/usr/bin/otool','-L',path]),run(['/usr/bin/otool','-l',path]));
const before=image(target),edits:string[]=[];
for(const dep of before.dependencies)if(!system(dep)){assert(manifest.libraries['native/lib/'+basename(dep)]);edits.push('-change',dep,'@rpath/'+basename(dep));}
for(const path of before.rpaths)if(path!=='@loader_path/../lib')edits.push('-delete_rpath',path);
if(!before.rpaths.includes('@loader_path/../lib'))edits.push('-add_rpath','@loader_path/../lib');
const commandStart=records.length;if(edits.length)run(['/usr/bin/install_name_tool',...edits,target]);const relocated=await binding(target);
run(['/usr/bin/codesign','--force','--sign','-',target]);run(['/usr/bin/codesign','--verify','--strict',target]);checkReleaseMachO(await readFile(target),relative);
const final=await binding(target),build={...prior,binary,binary_sha256:original.sha256,sources:[...prior.sources.filter((value:any)=>value.path!==source),{path:generated,bytes:Buffer.byteLength(body),sha256:hash(body)},...observation.records.map(value=>({path:value.path,bytes:Buffer.byteLength(value.body),sha256:hash(value.body)})),...await Promise.all(['native/EvolutionTail.h','native/EvolutionRound.h','native/AcceptedSourcePermissions.h','native/ExpertPreservation.h'].map(async path=>({path:join(root,path),...await binding(join(root,path))})))],macros:{...prior.macros,LUBAN_PREFIX_SOURCE_SHA:generated},commands:records.slice(0,commandStart),model_loaded:false,GPU:false,actual_expert_graph_qualification:false};
const proof={...manifest.programs.persistentExpert.proof,build,source_build_sha256:hash(JSON.stringify(build)),source_inverse:{parent:source,before_sha256:hash(originalSource),after_sha256:hash(body),edits:transformations,dependencies:observation.records.map(value=>({parent:value.source,path:value.path,before_sha256:hash(value.original),after_sha256:hash(value.body),edits:value.edits,inverse_exact:value.inverse_exact})),inverse_exact:true},inverse_verified:true,original_binary_sha256:original.sha256,approved_binary_sha256:original.sha256,compile:records.find(value=>value.argv[0]==='/usr/bin/clang++'),source_transformation:'Daily frozen-prefix capture uses the same graph boundary without audit snapshots; resident previous/current observation selects retained tensors without reading state or changing RNG; independent audit operation retained',actual_core_model_qualified:false,actual_expert_model_qualified:false};
manifest.programs.persistentExpert.proof=proof;authority.sources.persistentExpert=hash(JSON.stringify(proof));
manifest.payload[relative]={original,copied:original,stripped:original,relocated,final,image:image(target),commands:records.slice(commandStart).map(({argv,exit_code})=>({argv,exit_code})),debug_error:null};
// Preserve full compiler/research provenance outside the product. Runtime admission consumes only
// these original exact bindings and narrowly specified build facts, never logs or source patches.
await writeFile(join(output,'persistent-source-build.json'),JSON.stringify(proof,null,2)+'\n');
for(const [role,program]of Object.entries(manifest.programs) as [string,any][]){
 const old=program.proof,common:any={kind:old.kind,source_build_sha256:old.source_build_sha256,
  original_binary_sha256:old.original_binary_sha256,approved_binary_sha256:old.approved_binary_sha256,
  compile:{argv:old.compile.argv,exit_code:old.compile.exit_code},inverse_verified:old.inverse_verified};
 if(role==='cachedNone'||role==='cachedCore')common.build={build:{sha256:old.build.build.sha256}};
 else if(role==='trainer'){
  common.abi=old.abi;common.build={product_admitted:old.build.product_admitted,sources:Object.fromEntries(Object.entries(old.build.sources).map(([name,value]:[string,any])=>[name,{sha256:value.sha256}]))};
  common.cpu_proof={actual_steps_each_arm:old.cpu_proof.actual_steps_each_arm,all_seven_state_bit_exact:old.cpu_proof.all_seven_state_bit_exact,deterministic_raw_equal:old.cpu_proof.deterministic_raw_equal};
 }else if(role==='persistentExpert'){
  const cpp=old.build.sources.find((value:any)=>value.path===old.build.macros.LUBAN_PREFIX_SOURCE_SHA);assert(cpp);
  common.build={CPU_helper:old.build.CPU_helper,no_binary_execution:old.build.no_binary_execution,
   actual_expert_graph_qualification:old.build.actual_expert_graph_qualification,
   macros:{LUBAN_PREFIX_SOURCE_SHA:basename(cpp.path)},sources:[{path:basename(cpp.path),bytes:cpp.bytes,sha256:cpp.sha256}]};
 }else if(role==='prefix')common.build={...old.build};
 else if(role==='rng')common.build={source:{sha256:old.build.source.sha256},cpu:{sha256:old.build.cpu.sha256},mps:{sha256:old.build.mps.sha256}};
 program.proof=common;authority.sources[role]=hash(JSON.stringify(common));
}
for(const entry of Object.values(manifest.payload) as any[])entry.commands=entry.commands.map((c:any)=>({argv:c.argv,exit_code:c.exit_code}));
const manifestText=JSON.stringify(manifest,null,2)+'\n';await writeFile(join(runtimeRoot,authority.manifest_relative),manifestText);authority.manifest_sha256=hash(manifestText);
const trainer=manifest.programs.trainer.proof;
await writeFile(join(output,'runtime-authority.json'),JSON.stringify({runtimeRoot,authority,manifestBytes:Buffer.byteLength(manifestText),persistentBinary:target,persistentBinarySHA:final.sha256,
 roles:Object.fromEntries(Object.entries(manifest.programs).map(([name,program]:[string,any])=>[name,{path:program.path,sha256:manifest.payload[program.path].final.sha256}])),
 training:{source_build_receipt_sha256:trainer.source_build_sha256,training_source_sha256:trainer.build.sources.training.sha256,rng_source_sha256:trainer.build.sources.rng.sha256,cpu_rng_helper_sha256:trainer.build.sources.cpu.sha256,mps_rng_helper_sha256:trainer.build.sources.mps.sha256},
 payload:Object.keys({...manifest.payload,...manifest.licenses,...manifest.assets}).concat(authority.manifest_relative).sort(),model_loaded:false,qualification:false},null,2)+'\n');
console.log(JSON.stringify({binary,cpu,runtimeRoot,model_loaded:false,package_admitted:false}));
