/** Add current Runtime preservation to the retained expert graph; never edit frozen sources. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {join} from 'node:path';

const graphLogits=`
 at::Tensor preservationLogits(ExpertResource* expert,const CacheConfig& config,const std::map<std::string,Tensor>& weights,const std::vector<int64_t>& ids){
  const auto device=expert?expert->device:weights.begin()->second.device();CachedGraph graph(device,config);graph.weights=weights;graph.expert=expert;graph.capture.enabled=false;
  auto input=at::from_blob(const_cast<int64_t*>(ids.data()),{1,int64_t(ids.size())},at::TensorOptions().dtype(at::kLong)).clone().to(device);
  auto hidden=graph.run(input),normalized=norm(hidden,graph.w("norm.weight"));
  return at::linear(normalized,graph.w("embed_tokens.weight")).reshape({int64_t(ids.size()),config.vocab});
 }
 void capturePreservation(ExpertResource* teacher,const CacheConfig& config,const std::map<std::string,Tensor>& weights,const std::vector<int64_t>& ids,const Tensor& labels){
  need(step==0,"Frozen teachers require the restored previous source");cancel();torch::NoGradGuard noGrad;
  auto logits=preservationLogits(teacher,config,weights,ids);cancel();preservation.push_back(luban_expert_preservation::freeze(ids,labels,logits));
 }
`;
export async function writeNativeExpertPreservationSources(root:string,output:string,optimizer:string){
 const original=await readFile(optimizer,'utf8');let body=original;
 const edits:{before:string;after:string}[]=[];
 const change=(before:string,after:string)=>{assert.equal(body.split(before).length,2,before);body=body.replace(before,after);edits.push({before,after});};
 change('struct ExpertOptimizer {','#include '+JSON.stringify(join(root,'native/ExpertPreservation.h'))+'\nstruct ExpertOptimizer {\n std::vector<luban_expert_preservation::Teacher> preservation;'+graphLogits);
 change('double lr,uint64_t steps,const fs::path& output){','double lr,uint64_t steps,const fs::path& output,bool requirePreservation=false,bool audit=false){');
 change('config.hidden<=32||LUBAN_EXPERT_FULL_TRACE;torch::AutoGradMode','config.hidden<=32||LUBAN_EXPERT_FULL_TRACE||audit;torch::AutoGradMode');
 change('learningRate=lr;const bool completeTrace=', 'need(!requirePreservation||(!preservation.empty()&&lr==.0002),"Protected learning requires fixed teachers and the current Qwen learning rate");\n  learningRate=lr;const bool completeTrace=');
  change('need(at::isfinite(loss).item<bool>(),"Full graph loss nonfinite");loss.backward();auto tag=',`auto ce=loss,kl=at::zeros({},loss.options());Tensor protectedLogits;
   if(requirePreservation){cancel();const auto& teacher=preservation[step%preservation.size()];protectedLogits=preservationLogits(&expert,config,weights,teacher.ids);kl=luban_expert_preservation::loss(protectedLogits,teacher);loss=loss+5.0*kl;}
   cancel();need(at::isfinite(loss).item<bool>(),"Full graph loss nonfinite");loss.backward();auto tag=`);
  change('trace.put(tag+"loss",loss);','trace.put(tag+"loss",loss);trace.put(tag+"ce",ce);trace.put(tag+"preservation_kl",kl);if(completeTrace&&requirePreservation){trace.put(tag+"preservation_logits",protectedLogits);trace.put(tag+"preservation_teacher",preservation[step%preservation.size()].logProbability);}');
  change('trace.put("entry_rng",captureNativeCPURNG());','if(!requirePreservation||completeTrace)trace.put("entry_rng",captureNativeCPURNG());');
  change('trace.put(tag+"cpu_rng",captureNativeCPURNG());if(expert.device.is_mps())trace.put(tag+"mps_rng",captureNativeMPSRNG());','if(!requirePreservation||completeTrace){trace.put(tag+"cpu_rng",captureNativeCPURNG());if(expert.device.is_mps())trace.put(tag+"mps_rng",captureNativeMPSRNG());}');
  change('trace.put("final_rng",captureNativeCPURNG());if(expert.device.is_mps())trace.put("final_mps_rng",captureNativeMPSRNG());','if(!requirePreservation||completeTrace){trace.put("final_rng",captureNativeCPURNG());if(expert.device.is_mps())trace.put("final_mps_rng",captureNativeMPSRNG());}');
 let inverse=body;for(const edit of edits.toReversed()){assert.equal(inverse.split(edit.after).length,2);inverse=inverse.replace(edit.after,edit.before);}assert.equal(inverse,original);
 const path=join(output,'ExpertPreservedOptimizer.h');await writeFile(path,body);
 return{optimizer:path,records:[{source:optimizer,path,original,body,edits,inverse_exact:true}],header:join(root,'native/ExpertPreservation.h')};
}

/** Teachers belong to the admitted round, not to a candidate's saved Adam state. */
export async function writeNativeExpertPreservationState(root:string,output:string){
 const source=join(root,'.native-migration/research/native-unified-expert/StateCommands.h'),original=await readFile(source,'utf8');
 const before=' auto beforeCPU=captureNativeCPURNG();Tensor beforeMPS;if(device.is_mps())beforeMPS=captureNativeMPSRNG();try{applyNativeCPURNG(cpu);';
 // Only current restore has nextOptimizer. Copy before any RNG/owner commit, so a failed restore leaves the old round intact.
 const boundary=original.indexOf('\nvoid restoreResidentInitial(');assert(boundary>0);
 const current=original.slice(0,boundary);assert.equal(current.split(before).length,2);
 const after=' nextOptimizer.preservation=optimizer.preservation;\n'+before,body=current.replace(before,after)+original.slice(boundary);
 assert.equal(body.replace(after,before),original);
 const path=join(output,'ExpertPreservedStateCommands.h');await writeFile(path,body);
 return{path,record:{source,path,original,body,edits:[{before,after}],inverse_exact:true}};
}

/** Apply after ordinary loss/source/round patches, before compiling the same persistent role. */
export function patchNativeExpertPreservationCommand(body:string,replace:(body:string,before:string,after:string)=>string){
 body=replace(body,'else if(op=="train"||op=="loss"){','else if(op=="preservation-clear"){need(!(in>>extra)&&optimizer.step==0,"Teacher reset requires no live Adam");optimizer.preservation.clear();}\n   else if(op=="train"||op=="train-preserved"||op=="train-preserved-trace"||op=="loss"||op=="preservation"){');
 body=replace(body,'labelsSHA;if(op=="train")in>>lrText>>stepsText;','labelsSHA;if(op=="train"||op=="train-preserved"||op=="train-preserved-trace")in>>lrText>>stepsText;');
 body=replace(body,'(op=="loss"||(expert&&expert->resident()&&branchEnabled))','((op=="loss"||op=="preservation")||(expert&&expert->resident()&&branchEnabled))');
 body=replace(body,'CacheConfig cfg;if(op=="train"){size_t used;','CacheConfig cfg;if(op=="preservation"){need(!branchEnabled||(expert&&expert->resident()),"Parent teacher requires an actual resident previous expert");optimizer.capturePreservation(branchEnabled?expert.get():nullptr,cfg,backbone->weights,spec.tokens,labels);}else if(op=="train"||op=="train-preserved"||op=="train-preserved-trace"){size_t used;');
 body=replace(body,'labels,lr,integer(stepsText),root/name);','labels,lr,integer(stepsText),root/name,op!="train",op=="train-preserved-trace");');
 body=replace(body,'<<"{\\"event\\":\\"done\\",\\"ordinal\\":"','<<"{\\"event\\":\\"done\\",\\"preservation_rows\\":"<<optimizer.preservation.size()<<",\\"ordinal\\":"');
 return body;
}
