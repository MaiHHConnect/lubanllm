#pragma once
// Tensor references only: selecting an observation must not mutate optimizer state or RNG.
const Tensor& evolutionObservationTensor(bool previous,const Tensor& accepted,const Tensor& candidate){
 need(accepted.defined()&&candidate.defined()&&accepted.sizes()==candidate.sizes()&&accepted.scalar_type()==candidate.scalar_type()&&accepted.device()==candidate.device(),"Resident observation tensors differ");
 return previous?accepted:candidate;
}
// Daily training uses the same graph boundary as the audit exporter, without state snapshots.
void exportOrdinaryTrainingTail(CachedGraph& graph,const std::vector<int64_t>& tokens,const Tensor& labels,const fs::path& target,const std::string& descriptorSHA,const std::string& labelsSHA,const std::string& sampleSHA,const std::string& sampleRawSHA,const std::string& coreSHA,const std::string& coreVersion){
 need(graph.seen==0&&!graph.poisoned&&graph.expert==nullptr&&!graph.capture.enabled&&tokens.size()>=2&&tokens.size()<=512&&graph.cfg.capacity>=int64_t(tokens.size()),"Tail requires fresh complete expert-free cache");
 need(isSHA(descriptorSHA)&&isSHA(labelsSHA)&&isSHA(sampleSHA)&&isSHA(sampleRawSHA)&&isSHA(coreSHA),"Tail input bindings invalid");
 need((graph.device.is_mps()&&graph.cfg.layers==24&&graph.cfg.hidden==2048&&graph.cfg.intermediate==6144&&graph.cfg.vocab==248320)||(graph.device.is_cpu()&&graph.cfg.layers==4&&graph.cfg.hidden<=32&&graph.cfg.intermediate<=64&&graph.cfg.vocab<=64),"Tail exact MPS or tiny CPU shape required");
 need(labels.device().is_cpu()&&labels.scalar_type()==at::kLong&&labels.sizes()==at::IntArrayRef({1,int64_t(tokens.size())})&&labels.is_contiguous(),"Tail label shape differs");
 const auto ld=labels.data_ptr<int64_t>();bool supervised=false;
 for(size_t i=0;i<tokens.size();i++){need(tokens[i]>=0&&tokens[i]<graph.cfg.vocab&&(ld[i]==-100||(ld[i]>=0&&ld[i]<graph.cfg.vocab)),"Tail token/label vocabulary differs");if(i&&ld[i]!=-100)supervised=true;}
 need(supervised,"Tail shifted supervision absent");torch::NoGradGuard noGrad;need(!c10::InferenceMode::is_enabled(),"Tail NoGrad only");cancel();
 const auto began=Clock::now();auto idsCPU=at::from_blob(const_cast<int64_t*>(tokens.data()),{1,int64_t(tokens.size())},at::TensorOptions().dtype(at::kLong)).clone();
 auto x=graph.run(idsCPU.to(graph.device),true),residual=graph.trainingTailResidual;
 need(graph.seen==int64_t(tokens.size())&&graph.capture.values.empty()&&x.scalar_type()==at::kBFloat16&&residual.scalar_type()==at::kBFloat16&&x.sizes()==at::IntArrayRef({1,int64_t(tokens.size()),graph.cfg.intermediate})&&residual.sizes()==at::IntArrayRef({1,int64_t(tokens.size()),graph.cfg.hidden}),"Tail BF16 boundary differs");
 auto cpuX=x.cpu().contiguous(),cpuResidual=residual.cpu().contiguous();finite(cpuX);finite(cpuResidual);sync(graph.device);cancel();const auto output=fresh(target);
 std::ostringstream m;m<<std::setprecision(17)<<"{\"schema\":\"luban-ordinary-training-tail-v1\",\"pid\":"<<getpid()<<",\"device\":"<<jsonString(graph.device.is_mps()?"mps":"cpu")<<",\"layers\":"<<graph.cfg.layers<<",\"hidden\":"<<graph.cfg.hidden<<",\"intermediate\":"<<graph.cfg.intermediate<<",\"tokens\":"<<tokens.size()<<",\"descriptor_sha256\":"<<jsonString(descriptorSHA)<<",\"sample_sha256\":"<<jsonString(sampleSHA)<<",\"sample_raw_sha256\":"<<jsonString(sampleRawSHA)<<",\"labels_sha256\":"<<jsonString(labelsSHA)<<",\"ids_sha256\":"<<jsonString(hashMemory(idsCPU.data_ptr(),idsCPU.nbytes()))<<",\"core_tensor_sha256\":"<<jsonString(coreSHA)<<",\"core_version\":"<<jsonString(coreVersion)<<",\"expert_used\":false,\"branch_enabled\":false,\"cache_fresh\":true,\"stop_before_last_down\":true,\"optimizer_steps\":0,\"teacher_calls\":0,\"product_admission\":false,\"graph_seconds\":"<<elapsed(began)<<",\"tensors\":[";
 const std::array<std::pair<std::string,Tensor>,4> tensors={{{"tail_x",cpuX},{"tail_residual",cpuResidual},{"ids",idsCPU},{"labels",labels}}};bool first=true;
 for(const auto&[name,t]:tensors){durable(output/(name+".raw"),t.data_ptr(),t.nbytes());if(!first)m<<',';first=false;m<<"{\"name\":"<<jsonString(name)<<",\"file\":"<<jsonString(name+".raw")<<",\"dtype\":"<<jsonString(t.scalar_type()==at::kBFloat16?"bfloat16":"int64")<<",\"shape\":[";for(int i=0;i<t.dim();i++){if(i)m<<',';m<<t.size(i);}m<<"],\"bytes\":"<<t.nbytes()<<",\"sha256\":"<<jsonString(hashMemory(t.data_ptr(),t.nbytes()))<<'}';}
 m<<"]}\n";durableText(output/"manifest.json",m.str());syncDir(output);
}
