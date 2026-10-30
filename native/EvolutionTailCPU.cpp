#include "../.native-migration/research/native-qwen35-cache/Base.h"
#include "../.native-migration/research/native-evolution-retained-producer-v1/Graph.h"
#include "EvolutionTail.h"
#include "../.native-migration/research/native-unified-expert/TracePack.h"
#include LUBAN_EVOLUTION_OPTIMIZER_HEADER
#include "../.native-migration/research/native-unified-expert/Tiny.h"
int main(int argc,char**argv){try{
 need(argc==4,"CPU fixture directory, descriptor SHA and output required");at::set_num_threads(1);at::set_num_interop_threads(1);torch::NoGradGuard noGrad;
 int pipeFD[2];need(pipe(pipeFD)==0,"CPU cancellation pipe failed");if(pipeFD[0]!=3){need(dup2(pipeFD[0],3)==3,"CPU cancellation FD3 failed");close(pipeFD[0]);}
 auto spec=loadTiny(argv[1],argv[2]);const auto root=fresh(argv[3]);auto ids=at::from_blob(spec.ids.data(),{1,int64_t(spec.ids.size())},at::TensorOptions().dtype(at::kLong)).clone(),labels=ids.clone();labels.index_put_({0,0},-100);
 const auto state=captureNativeCPURNG();const auto labelsSHA=hashMemory(labels.data_ptr(),labels.nbytes());CachedGraph graph(torch::kCPU,spec.cfg);graph.weights=spec.weights;graph.capture.enabled=false;
 exportOrdinaryTrainingTail(graph,spec.ids,labels,root/"tail",argv[2],labelsSHA,argv[2],argv[2],argv[2],"cpu-fixture");need(captureNativeCPURNG().equal(state),"Daily tail modified CPU RNG");
 CachedGraph full(torch::kCPU,spec.cfg);full.weights=spec.weights;full.run(ids);auto x=(at::silu(full.capture.values.at("layer3_mlp_gate"))*full.capture.values.at("layer3_mlp_up")).contiguous(),residual=full.capture.values.at("layer3_post_attention").contiguous();
 need(hashFile(root/"tail/tail_x.raw")==hashMemory(x.data_ptr(),x.nbytes())&&hashFile(root/"tail/tail_residual.raw")==hashMemory(residual.data_ptr(),residual.nbytes()),"Daily tail differs from full forward boundary");
 std::set<std::string> names;for(const auto& item:fs::directory_iterator(root/"tail"))names.insert(item.path().filename());need(names==std::set<std::string>{"manifest.json","tail_x.raw","tail_residual.raw","ids.raw","labels.raw"},"Daily tail wrote additional audit/state files");
 int rejected=0;auto reject=[&](const std::function<void()>& f){try{f();throw std::logic_error("Invalid tail accepted");}catch(const std::logic_error&){throw;}catch(const std::exception&){rejected++;}};
 reject([&](){exportOrdinaryTrainingTail(graph,spec.ids,labels,root/"reused",argv[2],labelsSHA,argv[2],argv[2],argv[2],"cpu-fixture");});
 auto invalid=labels.clone();invalid.fill_(-100);CachedGraph blank(torch::kCPU,spec.cfg);blank.weights=spec.weights;blank.capture.enabled=false;
 reject([&](){exportOrdinaryTrainingTail(blank,spec.ids,invalid,root/"no-labels",argv[2],labelsSHA,argv[2],argv[2],argv[2],"cpu-fixture");});
 auto tokens=spec.ids;tokens[0]=spec.cfg.vocab;reject([&](){exportOrdinaryTrainingTail(blank,tokens,labels,root/"bad-token",argv[2],labelsSHA,argv[2],argv[2],argv[2],"cpu-fixture");});
 blank.capture.enabled=true;reject([&](){exportOrdinaryTrainingTail(blank,spec.ids,labels,root/"capture",argv[2],labelsSHA,argv[2],argv[2],argv[2],"cpu-fixture");});
 need(rejected==4,"Tail negative controls incomplete");
 ExpertOptimizer optimizer;optimizer.loss(nullptr,spec.cfg,spec.weights,spec.ids,labels.reshape({-1}),root/"ordinary-loss",true);
 optimizer.loss(nullptr,spec.cfg,spec.weights,spec.ids,labels.reshape({-1}),root/"audit-loss",false);
 need(hashFile(root/"ordinary-loss/loss.raw")==hashFile(root/"audit-loss/loss.raw"),"Daily loss changed numerical result");
 names.clear();for(const auto& item:fs::directory_iterator(root/"ordinary-loss"))names.insert(item.path().filename());
 need(names==std::set<std::string>{"loss.json","loss.raw","logits.raw"},"Daily loss wrote RNG snapshots");
 need(captureNativeCPURNG().equal(state)&&optimizer.step==0,"Daily loss changed RNG or optimizer");
 const auto downKey=PREFIX+"layers.3.mlp.down_proj.weight";auto accepted=spec.weights.at(downKey),candidate=at::zeros_like(accepted),candidateBefore=candidate.clone(),acceptedBefore=accepted.clone();
 auto observe=[&](bool previous){CachedGraph g(torch::kCPU,spec.cfg);g.weights=spec.weights;g.capture.enabled=false;g.weights.at(downKey)=evolutionObservationTensor(previous,accepted,candidate);need(g.weights.at(downKey).data_ptr()==(previous?accepted:candidate).data_ptr(),"Observation copied its selected tensor");return g.run(ids);};
 auto priorLogits=observe(true),candidateLogits=observe(false),priorAgain=observe(true),candidateAgain=observe(false);
 need(priorLogits.equal(priorAgain)&&candidateLogits.equal(candidateAgain)&&!priorLogits.equal(candidateLogits),"Observation selection did not persist through full graph generation");
 auto ce=[&](const Tensor& logits){auto shifted=logits.index({0,at::indexing::Slice(0,-1)}).to(at::kFloat);return -at::log_softmax(shifted,-1).gather(1,ids.index({0,at::indexing::Slice(1,at::indexing::None)}).unsqueeze(1)).mean().item<double>();};
 need(ce(priorLogits)==ce(priorAgain)&&ce(candidateLogits)==ce(candidateAgain)&&ce(priorLogits)!=ce(candidateLogits),"Observation loss did not follow selected graph");
 need(candidate.equal(candidateBefore)&&accepted.equal(acceptedBefore)&&captureNativeCPURNG().equal(state),"Observation mutated source tensors or RNG");
 durableText(root/"result.json","{\"passed\":true,\"GPU\":false,\"actual_model\":false,\"full_forward_boundary_exact\":true,\"extra_state_files\":0,\"negative_controls\":4,\"ordinary_loss_exact\":true,\"ordinary_loss_rng_files\":0,\"resident_pointer_selection\":true,\"generation_and_ce_follow_selection\":true,\"selection_preserves_rng_and_tensors\":true}\n");return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
