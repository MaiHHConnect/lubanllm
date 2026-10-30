// Actual generated optimizer/current-restore code, tiny CPU tensors and private files only.
#include "../build/native-evolution/EvolutionGeneration.cpp"
#include "../.native-migration/research/native-unified-expert/TracePack.h"
#include "../build/native-evolution/ExpertPreservedOptimizer.h"
#include LUBAN_EXPERT_STATE_HEADER
int main(int argc,char** argv){try{
 int cancellation[2];need(pipe(cancellation)==0,"CPU cancellation pipe failed");if(cancellation[0]!=3){need(dup2(cancellation[0],3)==3,"CPU cancellation FD failed");close(cancellation[0]);}
 need(argc==2,"Use private fresh output");auto root=fresh(argv[1]);at::set_num_threads(1);
 constexpr int64_t hidden=4,width=64;auto device=torch::Device(torch::kCPU);
 auto expert=std::make_unique<ExpertResource>(hidden,width,device);ExpertOptimizer optimizer;
 for(auto& w:expert->tensors)w=at::ones({width,hidden},at::kBFloat16);expert->tensors[2]=expert->tensors[2].t().contiguous();optimizer.step=9;
 auto teacher=luban_expert_preservation::freeze({0,1,2},at::tensor(std::vector<int64_t>{-100,1,2},at::kLong),at::zeros({3,4},at::kFloat));
 optimizer.preservation={teacher,teacher};const auto teacherPtr=teacher.logProbability.const_data_ptr();
 auto state=[&](const std::string& name,bool invalidAdam,bool invalidRNG){
  std::vector<char> raw(8,0);std::array<Range,12> ranges;
  for(size_t i=0;i<12;i++){
   auto shape=i>=9?std::vector<int64_t>{}:i%3==2?std::vector<int64_t>{hidden,width}:std::vector<int64_t>{width,hidden};
   auto t=at::full(shape,i>=9?(invalidAdam?2:1):double(i+1)/16,at::TensorOptions().dtype(i>=9?at::kFloat:at::kBFloat16));
   ranges[i]={i>=9?"f32":"bf16",hashMemory(t.const_data_ptr(),t.nbytes()),shape,uint64_t(raw.size()),uint64_t(t.nbytes())};
   const auto* data=static_cast<const char*>(t.const_data_ptr());raw.insert(raw.end(),data,data+t.nbytes());
  }
  auto path=root/(name+".raw");durable(path,raw.data(),raw.size());auto rng=captureNativeCPURNG();if(invalidRNG)rng.zero_();auto rngPath=root/(name+"-rng.raw");durable(rngPath,rng.data_ptr(),rng.nbytes());
  std::ostringstream out;out<<width<<" 1 .0002 "<<std::quoted(path.string())<<' '<<hashFile(path);
  for(const auto& r:ranges)out<<' '<<r.offset<<' '<<r.sha;
  out<<' '<<std::quoted(rngPath.string())<<' '<<hashFile(rngPath)<<" \"-\" -";return out.str();
 };
 std::istringstream valid(state("valid",false,false));restoreResidentCurrent(valid,hidden,device,expert,optimizer);
 need(optimizer.step==1&&optimizer.learningRate==.0002,"Actual restored Adam step/LR differs");
 need(optimizer.preservation.size()==2&&optimizer.preservation[0].logProbability.const_data_ptr()==teacherPtr,"Frozen teacher owner lost");
 for(size_t i=0;i<3;i++){need(expert->tensors[i].eq(double(i+1)/16).all().item<bool>(),"Actual weight restore differs");need(optimizer.m[i].eq(double(i+4)/16).all().item<bool>()&&optimizer.v[i].eq(double(i+7)/16).all().item<bool>()&&optimizer.adamSteps[i].item<float>()==1,"Actual Adam state restore differs");}
 int rejected=0;
 for(int kind=0;kind<2;kind++){
  const auto* old=expert.get();auto rng=captureNativeCPURNG(),m=optimizer.m[0].clone();
  try{std::istringstream broken(state(kind?"bad-rng":"bad-adam",!kind,kind));restoreResidentCurrent(broken,hidden,device,expert,optimizer);}catch(const std::exception&){rejected++;}
  need(expert.get()==old&&optimizer.step==1&&optimizer.m[0].equal(m)&&optimizer.preservation.size()==2&&optimizer.preservation[0].logProbability.const_data_ptr()==teacherPtr&&captureNativeCPURNG().equal(rng),"Failed restore changed original owner/Adam/teacher/RNG");
 }
 need(rejected==2,"Invalid restore accepted");std::cout<<"{\"passed\":true,\"teacher_count\":2,\"step\":1,\"rejected\":2,\"GPU\":false}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
