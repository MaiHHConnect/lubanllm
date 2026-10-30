/** Build-time overlay of the existing numerical functions. Audit CLI remains available separately. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join,dirname,resolve} from 'node:path';
export async function writeEvolutionObservationSources(root:string,output:string,graph?:string){
 const research=join(root,'.native-migration/research'),records:any[]=[];
 async function overlay(source:string,name:string,edit:(change:(before:string,after:string)=>void)=>void){
  const original=await readFile(source,'utf8');let body=original;const edits:{before:string;after:string}[]=[];
  const change=(before:string,after:string)=>{assert.equal(body.split(before).length,2,before);edits.push({before,after});body=body.replace(before,after);};
  for(const match of [...body.matchAll(/^#include "([^"]+)"/gm)])change(match[0],'#include '+JSON.stringify(resolve(dirname(source),match[1]!)));
  edit(change);let inverse=body;for(const item of edits.toReversed()){assert.equal(inverse.split(item.after).length,2);inverse=inverse.replace(item.after,item.before);}assert.equal(inverse,original);
  const path=join(output,name);await writeFile(path,body);records.push({source,path,original,body,edits,inverse_exact:true});return path;
 }
 const generation=await overlay(join(research,'native-unified-tail-role-v1/Generation.cpp'),'EvolutionGeneration.cpp',change=>{
  if(graph)change('#include '+JSON.stringify(join(research,'native-evolution-retained-producer-v1/Graph.h')),'#include '+JSON.stringify(graph));
  change('const Tensor* temporaryCore=nullptr){Clock::time_point','const Tensor* temporaryCore=nullptr,bool ordinary=false,const std::string& residentSHA=""){Clock::time_point');
  change('if(d.config.at("core_version")!="null"){torch::NoGradGuard','if(d.config.at("core_version")!="null"&&!sharedWeights){torch::NoGradGuard');
  change('if(core&&!temporaryCore){','if(core&&!temporaryCore&&!sharedWeights){');
  change('auto rngParentCPU=samplingRaw(output,"entry_cpu_rng",sampling.parentCPU,true),rngParentMPS=samplingRaw(output,"entry_mps_rng",sampling.parentMPS,true),rngSeeded=samplingState(output,"seeded",true,true);','auto rngParentCPU=ordinary?"null":samplingRaw(output,"entry_cpu_rng",sampling.parentCPU,true),rngParentMPS=ordinary?"null":samplingRaw(output,"entry_mps_rng",sampling.parentMPS,true),rngSeeded=ordinary?"null":samplingState(output,"seeded",true,true);');
  change('auto rngBeforeCPU=captureNativeCPURNG(),rngBeforeMPS=captureNativeMPSRNG();','auto rngBeforeCPU=ordinary?Tensor():captureNativeCPURNG(),rngBeforeMPS=ordinary?Tensor():captureNativeMPSRNG();');
  change('const bool save=step<integer(d.config.at("raw_steps_limit"));samplingSteps.push_back(', 'const bool save=step<integer(d.config.at("raw_steps_limit"));if(!ordinary)samplingSteps.push_back(');
  change('auto rngFinal=samplingState(output,"final",true,true);sampling.restore();auto rngRestored=samplingState(output,"restored",true,true);','auto rngFinal=ordinary?"null":samplingState(output,"final",true,true);sampling.restore();auto rngRestored=ordinary?"null":samplingState(output,"restored",true,true);');
  change('jsonString(hashFile(safePath(argv[0])))','(ordinary?"null":jsonString(hashFile(safePath(argv[0]))))');
  change('jsonString(hashMemory(graph.w("layers.23.mlp.down_proj.weight").to(torch::kCPU).contiguous().data_ptr(),25165824))','jsonString(ordinary?residentSHA:hashMemory(graph.w("layers.23.mlp.down_proj.weight").to(torch::kCPU).contiguous().data_ptr(),25165824))');
  change('auto d=descriptor(argv[2],argv[4]);BoundFile model','need(!ordinary||(sharedWeights&&isSHA(residentSHA)),"Daily generation requires retained graph and source binding");auto d=descriptor(argv[2],argv[4]);BoundFile model');
 });
 const optimizer=await overlay(join(research,'native-unified-final-role-v1/ExpertOptimizerV3.h'),'EvolutionOptimizer.h',change=>{
  change('const Tensor& labels,const fs::path& output){\n  need((!expert','const Tensor& labels,const fs::path& output,bool ordinary=false){\n  need((!expert');
  change('auto beforeCPU=captureNativeCPURNG();Tensor beforeMPS;if(device.is_mps())beforeMPS=captureNativeMPSRNG();','auto beforeCPU=ordinary?Tensor():captureNativeCPURNG();Tensor beforeMPS;if(!ordinary&&device.is_mps())beforeMPS=captureNativeMPSRNG();');
  change('save("before_cpu_rng",beforeCPU);','if(!ordinary){save("before_cpu_rng",beforeCPU);');
  change('need(beforeMPS.equal(captureNativeMPSRNG()),"Evaluation changed MPS RNG");}','need(beforeMPS.equal(captureNativeMPSRNG()),"Evaluation changed MPS RNG");}}');
  change('\\"schema\\":\\"luban-expert-full-graph-loss-v1\\",\\"pid\\":','\\"schema\\":\\"luban-expert-full-graph-loss-v1\\",\\"ordinary\\":"<<(ordinary?"true":"false")<<",\\"pid\\":');
  change('\\"rng_unchanged\\":true,','\\"rng_unchanged\\":"<<(ordinary?"null":"true")<<",');
 });
 return{generation,optimizer,records};
}
