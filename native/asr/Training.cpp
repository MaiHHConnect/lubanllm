#include <sys/resource.h>
#include "ArchiveRNG.h"
// Fixed-budget native CPU training. Audio tower/projector/full prefix remain frozen.
#include <torch/torch.h>
#include <fstream>
#include "Graph.h"

std::map<std::string,Tensor> readOwned(const fs::path& descriptor,std::unique_ptr<Owner>& model,std::unique_ptr<Owner>& input,std::string& binding,uint64_t& modelReads,uint64_t& modelBytes,int modelFD,int inputFD,int64_t& trainingCount){
 std::ifstream file(descriptor);need(bool(file),"Private training descriptor missing");std::string line;need(bool(std::getline(file,line))&&line=="LUBAN_ASR_TRAIN_OWNER_V2","Training descriptor ABI");
 std::map<std::string,std::string> config;std::vector<std::vector<std::string>> rows;while(std::getline(file,line)){auto f=fields(line,'\t');need(!f.empty(),"Descriptor row");if(f[0]=="config"){need(f.size()==3&&config.emplace(f[1],f[2]).second,"Duplicate config");}else{need(f.size()==7&&f[0]=="tensor","Inherited tensor row");rows.push_back(f);}}
 need((config.size()==3||(config.size()==4&&config.count("training_samples")))&&config.count("model_path")&&config.count("input_path")&&config.count("binding"),"Exact owner descriptor config");if(config.count("training_samples")){auto n=integer(config.at("training_samples"));need(n<=4,"Fixed four-update budget");trainingCount=n;}binding=config.at("binding");need(!binding.empty()&&binding.size()<(1<<20),"Complete expected model/round binding");
 model=std::make_unique<Owner>(modelFD,config.at("model_path"));input=std::make_unique<Owner>(inputFD,config.at("input_path"));std::map<std::string,Tensor> out;
 for(const auto& f:rows){cancel();std::vector<int64_t> shape;uint64_t count=1;for(const auto& v:fields(f[3],',')){auto n=integer(v);need(n>0&&count<=(1ULL<<30)/n,"Tensor shape overflow");shape.push_back(n);count*=n;}
  auto dtype=f[2]=="bf16"?at::kBFloat16:f[2]=="f32"?at::kFloat:f[2]=="i64"?at::kLong:at::kInt;need(f[2]=="bf16"||f[2]=="f32"||f[2]=="i64"||f[2]=="i32","Approved tensor dtype");auto bytes=integer(f[5]),fd=integer(f[6]);need((fd==3||fd==4)&&bytes==count*(f[2]=="bf16"?2:f[2]=="i64"?8:4),"Inherited tensor byte binding");auto t=at::empty(shape,at::TensorOptions().dtype(dtype).device(at::kCPU));(fd==3?model:input)->read(integer(f[4]),t.data_ptr(),bytes);if(fd==3){modelReads++;modelBytes+=bytes;}if(t.is_floating_point())need(at::isfinite(t).all().item<bool>(),"Finite held tensor");need(out.emplace(f[1],f[2]=="bf16"?t.to(at::kFloat):t).second,"Duplicate held tensor");
 }need(!out.empty(),"Complete held tensor set");return out;
}
void save(const fs::path& output,const std::string& key,const Tensor& value){if(key!="cpu_rng")return;auto tensor=value.detach().contiguous();std::ofstream file(output/(key+".raw"),std::ios::binary);file.write((const char*)tensor.data_ptr(),tensor.nbytes());need(bool(file),"CPU RNG envelope write");file.close();int fd=open((output/(key+".raw")).c_str(),O_RDONLY|O_NOFOLLOW);need(fd>=0&&fsync(fd)==0,"CPU RNG envelope sync");close(fd);fd=open(output.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW);need(fd>=0&&fsync(fd)==0,"CPU RNG directory sync");close(fd);}
Tensor loss(const Tensor& logits,const Tensor& labels){return torch::nn::functional::cross_entropy(logits.index({Slice(),Slice(0,-1)}).reshape({-1,V}),labels.index({Slice(),Slice(1)}).reshape({-1}),torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));}
Tensor tail(const Tensor& x,const Tensor& residual,const Tensor& weight,const Tensor& norm,const Tensor& head){auto h=at::linear(x,weight)+residual;auto f=h.to(at::kFloat);auto normalized=norm*(f*at::rsqrt(f.pow(2).mean(-1,true)+1e-6));return at::linear(normalized,head);}

int main(int argc,char** argv){try{
 ASRCPURNGScope rngScope;
 need(argc==12,"descriptor lock output checkpoint mode rng modelFD inputFD lockFD rngFD checkpointFD");
 const int modelFD=integer(argv[7]),inputFD=integer(argv[8]),lockFD=integer(argv[9]),rngFD=integer(argv[10]);
 signal(SIGTERM,onSignal);signal(SIGINT,onSignal);at::set_num_threads(1);at::set_num_interop_threads(1);
 fs::path descriptor=argv[1],round=argv[3];need(fs::create_directory(round),"Fresh round output required");
 #ifndef LUBAN_ASR_TINY
 struct stat modelLock{},namedLock{};need(fstat(lockFD,&modelLock)==0&&lstat(argv[2],&namedLock)==0&&same(modelLock,namedLock)&&S_ISREG(modelLock.st_mode)&&modelLock.st_uid==getuid()&&modelLock.st_nlink==1&&(modelLock.st_mode&0777)==0600&&flock(lockFD,LOCK_EX|LOCK_NB)==0,"Inherited original standalone ASR model lock");
 #endif
 std::unique_ptr<Owner> modelOwner,inputOwner;std::string binding;uint64_t modelReads=0,modelBytes=0,modelLoads=0;int64_t trainingCount=-1;
 auto tensors=readOwned(descriptor,modelOwner,inputOwner,binding,modelReads,modelBytes,modelFD,inputFD,trainingCount);modelLoads++;
 const std::string target="model.language_model.layers."+std::to_string(TL-1)+".mlp.down_proj.weight";
 std::string mode=argv[5];need(mode=="train"||mode=="evaluate","Explicit mode");need(trainingCount<0||(mode=="train"?trainingCount>0:trainingCount==0),"Declared training/evaluation scope");int64_t startStep=0;Tensor priorAvg,priorSq,checkpointRNG;
 if(std::string(argv[4])!="none"){Owner checkpointOwner(integer(argv[11]),argv[4]);need(checkpointOwner.stat.st_size>0&&checkpointOwner.stat.st_size<=(128<<20),"Bounded original native checkpoint");std::string checkpointBytes(checkpointOwner.stat.st_size,'\0');checkpointOwner.read(0,checkpointBytes.data(),checkpointBytes.size());std::istringstream checkpointStream(checkpointBytes);torch::serialize::InputArchive archive;archive.load_from(checkpointStream,at::kCPU);checkpointOwner.check();Tensor weight,step,savedBinding;archive.read("binding",savedBinding);auto wanted=at::from_blob(binding.data(),{int64_t(binding.size())},at::kByte);need(torch::equal(savedBinding,wanted),"Checkpoint model/round/data binding mismatch");archive.read("weight",weight);archive.read("avg",priorAvg);archive.read("sq",priorSq);archive.read("step",step);need(weight.sizes()==tensors.at(target).sizes()&&weight.scalar_type()==at::kFloat&&priorAvg.scalar_type()==at::kFloat&&priorSq.scalar_type()==at::kFloat&&priorAvg.sizes()==weight.sizes()&&priorSq.sizes()==weight.sizes()&&step.scalar_type()==at::kLong&&step.numel()==1,"ASR checkpoint tensor binding");startStep=step.item<int64_t>();need(startStep>0&&startStep<=100000&&torch::isfinite(weight).all().item<bool>()&&torch::isfinite(priorAvg).all().item<bool>()&&torch::isfinite(priorSq).all().item<bool>()&&(priorSq>=0).all().item<bool>(),"Finite ASR checkpoint Adam state");tensors[target]=weight;need(std::string(argv[6])!="none","Original checkpoint CPU RNG required");Owner rngOwner(rngFD,argv[6]);struct stat rngStat{};need(fstat(rngFD,&rngStat)==0&&rngStat.st_size>0&&rngStat.st_size<=65536,"Bounded original CPU RNG envelope");auto resumeRNG=at::empty({rngStat.st_size},at::kByte);rngOwner.read(0,resumeRNG.data_ptr(),rngStat.st_size);checkpointRNG=restoreASRCPURNG(archive,resumeRNG);rngOwner.check();}else need(std::string(argv[6])=="none","Fresh round cannot supply resume RNG");
 auto key=target;auto parameter=tensors.at(key).clone().requires_grad_(true),norm=tensors.at("model.language_model.norm.weight"),head=tensors.at("model.language_model.embed_tokens.weight");
 auto avg=startStep?priorAvg.clone():torch::zeros_like(parameter),sq=startStep?priorSq.clone():torch::zeros_like(parameter);
 const auto parameterStorage=reinterpret_cast<uintptr_t>(parameter.data_ptr()),avgStorage=reinterpret_cast<uintptr_t>(avg.data_ptr()),sqStorage=reinterpret_cast<uintptr_t>(sq.data_ptr());
 // Own one ASR Graph for the entire round. Sample input/KV are the only replaced state.
 Graph graph(tensors);int samples=0;std::vector<int64_t> sampleSteps,promptSizes;std::vector<std::vector<int64_t>> recognitions;const auto initialStep=startStep;
 for(int sample=0;tensors.count("sample"+std::to_string(sample)+":input");sample++){
  cancel();const auto prefixKey="sample"+std::to_string(sample)+":";
  for(auto name:{"input","mask","ids"})graph.tensors[name]=tensors.at(prefixKey+name);
  graph.tensors[target]=parameter.detach();graph.keys.assign(TL,Tensor{});graph.values.assign(TL,Tensor{});graph.seen=0;graph.tailX=Tensor{};graph.tailResidual=Tensor{};
  if(trainingCount>=0&&sample>=trainingCount){
   need(!tensors.count(prefixKey+"labels"),"HOLDOUT generation must not contain gold labels");graph.tensors.erase("labels");torch::NoGradGuard frozen;
   promptSizes.push_back(graph.tensors.at("ids").size(1));recognitions.push_back(graph.generate(512,151643,151645));continue;
  }
  graph.tensors["labels"]=tensors.at(prefixKey+"labels");
  fs::path output=round/("segment-"+std::to_string(sample));need(fs::create_directory(output),"Fresh sample evidence required");
  auto labels=graph.tensors.at("labels"),ids=graph.tensors.at("ids");need(labels.scalar_type()==at::kLong&&labels.sizes()==ids.sizes(),"Gold input shape");bool gold=false;int count=0,prefix=0;
  for(int i=0;i<labels.size(1);i++){auto label=labels[0][i].item<int64_t>();if(label==-100){need(!gold,"Interrupted gold mask");prefix++;}else{need(i>0&&label==ids[0][i].item<int64_t>()&&label>=0&&label<V,"Gold label mismatch");gold=true;count++;}}
  need(prefix>0&&count>1&&labels[0][-1].item<int64_t>()==151645,"Complete gold and EOS required");save(output,"labels",labels);
  Tensor x,residual;{torch::NoGradGuard frozen;auto baseline=graph.text(graph.initial());need(torch::isfinite(loss(baseline,labels)).item<bool>(),"Finite gold CE");x=graph.tailX;residual=graph.tailResidual;save(output,"x",x);save(output,"residual",residual);save(output,"baseline_logits",baseline);save(output,"baseline_loss",loss(baseline,labels).reshape({1}));}
 for(int localStep=1;mode=="train"&&localStep<=1;localStep++){int64_t step=startStep+localStep;parameter.mutable_grad()=Tensor();auto logits=tail(x,residual,parameter,norm,head),ce=loss(logits,labels);need(torch::isfinite(ce).item<bool>(),"Finite tiny training CE");ce.backward();need(parameter.grad().defined()&&torch::isfinite(parameter.grad()).all().item<bool>(),"Finite actual gradient required");auto name="step"+std::to_string(localStep)+"_";save(output,name+"logits",logits);save(output,name+"loss",ce.reshape({1}));save(output,name+"grad",parameter.grad());
  cancel();auto magnitude=at::linalg_vector_norm(parameter.grad(),2.);auto full=at::linalg_vector_norm(torch::stack({magnitude}),2.);parameter.grad().mul_(at::clamp_max(1./(full+1e-6),1.));save(output,name+"clipped",parameter.grad());
  {torch::NoGradGuard update;avg.lerp_(parameter.grad(),.1);sq.mul_(.999).addcmul_(parameter.grad(),parameter.grad(),.001);auto denominator=(sq.sqrt()/std::sqrt(1-std::pow(.999,step))).add_(1e-6);parameter.addcdiv_(avg,denominator,-.003/(1-std::pow(.9,step)));}save(output,name+"weight",parameter);save(output,name+"avg",avg);save(output,name+"sq",sq);
 }
  need(at::isfinite(parameter).all().item<bool>()&&at::isfinite(avg).all().item<bool>()&&at::isfinite(sq).all().item<bool>(),"Finite candidate and Adam state");
  // Same original research checkpoint boundary; no model/source audit per sample.
save(output,"saved_weight",parameter);save(output,"saved_avg",avg);save(output,"saved_sq",sq);auto finalStep=torch::tensor({startStep+(mode=="train"?1:0)},at::kLong);save(output,"saved_step",finalStep);
 {torch::serialize::OutputArchive archive;archive.write("binding",at::from_blob(binding.data(),{int64_t(binding.size())},at::kByte).clone());archive.write("weight",parameter.detach());archive.write("avg",avg);archive.write("sq",sq);archive.write("step",finalStep);checkpointRNG=saveASRCPURNG(archive);archive.save_to((output/"checkpoint.pending").string());int fd=open((output/"checkpoint.pending").c_str(),O_RDONLY|O_NOFOLLOW);need(fd>=0&&fsync(fd)==0,"Checkpoint durable file sync");close(fd);cancel();need(!fs::exists(output/"checkpoint.pt"),"Fresh checkpoint destination");fs::rename(output/"checkpoint.pending",output/"checkpoint.pt");fd=open(output.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW);need(fd>=0&&fsync(fd)==0,"Checkpoint durable directory sync");close(fd);}
 {torch::serialize::InputArchive reload;reload.load_from((output/"checkpoint.pt").string(),at::kCPU);for(auto pair:std::vector<std::pair<std::string,Tensor>>{{"weight",parameter},{"avg",avg},{"sq",sq},{"step",finalStep},{"cpu_rng",captureNativeCPURNG()}}){Tensor restored;reload.read(pair.first,restored);need(torch::equal(restored,pair.second),"ASR saved checkpoint must reload exactly");}}
  need(parameterStorage==reinterpret_cast<uintptr_t>(parameter.data_ptr())&&avgStorage==reinterpret_cast<uintptr_t>(avg.data_ptr())&&sqStorage==reinterpret_cast<uintptr_t>(sq.data_ptr()),"Resident candidate/Adam storage changed");
  save(output,"cpu_rng",captureNativeCPURNG());startStep=finalStep.item<int64_t>();sampleSteps.push_back(startStep);samples++;std::cout<<"{\"event\":\"checkpoint\",\"step\":"<<startStep<<",\"sample\":"<<sample<<"}\n"<<std::flush;
 }
 need(samples>0||!recognitions.empty(),"Nonempty whole round");need(trainingCount<0||(samples==trainingCount&&!recognitions.empty()),"Exact TRAIN scope and nonempty HOLDOUT");cancel();modelOwner->check();inputOwner->check();
 #ifndef LUBAN_ASR_TINY
 struct stat afterLock{},afterNamed{};need(fstat(lockFD,&afterLock)==0&&lstat(argv[2],&afterNamed)==0&&same(afterLock,modelLock)&&same(afterLock,afterNamed)&&flock(lockFD,LOCK_EX|LOCK_NB)==0,"Original ASR model lock changed");
 #endif
 auto finalRNG=captureNativeCPURNG();if(checkpointRNG.defined())need(torch::equal(finalRNG,checkpointRNG),"Greedy evaluation changed checkpoint CPU RNG");save(round,"cpu_rng",finalRNG);rngScope.restore();
 struct rusage usage{};need(getrusage(RUSAGE_SELF,&usage)==0,"Native CPU peak RSS measurement");
 std::cout<<"{\"passed\":true,\"CPU\":true,\"GPU\":false,\"checkpoint_cpu_rng\":true,\"parent_cpu_rng_restored\":true,\"cpu_rng_bytes\":"<<finalRNG.numel()<<",\"pid\":"<<getpid()<<",\"pgid\":"<<getpgrp()<<",\"model_loads\":"<<modelLoads<<",\"model_tensor_read_calls\":"<<modelReads<<",\"model_tensor_read_bytes\":"<<modelBytes<<",\"graph_constructions\":1,\"samples\":"<<samples<<",\"start_step\":"<<initialStep<<",\"final_step\":"<<startStep<<",\"actual_adam_updates\":"<<(mode=="train"?samples:0)<<",\"peak_RSS_getrusage_Darwin_bytes\":"<<usage.ru_maxrss<<",\"candidate_storage\":"<<parameterStorage<<",\"avg_storage\":"<<avgStorage<<",\"sq_storage\":"<<sqStorage<<",\"storage_same_whole_round\":true,\"sample_steps\":[";
 for(size_t i=0;i<sampleSteps.size();i++)std::cout<<(i?",":"")<<sampleSteps[i];std::cout<<"],\"holdout_max_new_tokens\":512,\"holdout_prompt_tokens\":[";
 for(size_t i=0;i<promptSizes.size();i++)std::cout<<(i?",":"")<<promptSizes[i];std::cout<<"],\"holdout_tokens\":[";
 for(size_t i=0;i<recognitions.size();i++){std::cout<<(i?",":"")<<"[";for(size_t j=0;j<recognitions[i].size();j++)std::cout<<(j?",":"")<<recognitions[i][j];std::cout<<"]";}std::cout<<"]}\n";return 0;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
