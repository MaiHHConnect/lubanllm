// Small independent CE/KL/Adam check. No model, Metal, runtime capsule or product writes.
#include <torch/torch.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/nn/functional/loss.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "ExpertPreservation.h"
#include "../.native-migration/research/native-expert-mps/ExpertAdam.h"
namespace fs=std::filesystem;
using luban_expert_preservation::require;
void save(const fs::path& root,const std::string& name,const at::Tensor& input){
 auto tensor=input.detach().cpu().contiguous();std::ofstream file(root/(name+".raw"),std::ios::binary);require(bool(file),"CPU output open failed");file.write(static_cast<const char*>(tensor.const_data_ptr()),tensor.nbytes());require(bool(file),"CPU output write failed");
}
int main(int argc,char**argv){try{
 require(argc==3,"Use fresh-output F32|BF16");fs::path root=argv[1];require(fs::create_directory(root),"Fresh CPU output required");
 const std::string kind=argv[2];require(kind=="F32"||kind=="BF16","CPU dtype invalid");const auto dtype=kind=="BF16"?at::kBFloat16:at::kFloat;
 at::set_num_threads(1);at::set_num_interop_threads(1);
 auto x=(at::arange(20,at::kFloat).reshape({5,4})/32-.25).to(dtype),head=(at::arange(28,at::kFloat).reshape({7,4})/41-.3).to(dtype);
 std::array<at::Tensor,3> weights={(at::arange(12,at::kFloat).reshape({3,4})/33-.2).to(dtype),(at::arange(12,at::kFloat).reshape({3,4})/31-.1).to(dtype),(at::arange(12,at::kFloat).reshape({4,3})/37-.15).to(dtype)};
 auto forward=[&](bool expert){auto hidden=x;if(expert)hidden=hidden+at::linear(at::silu(at::linear(x,weights[0]))*at::linear(x,weights[1]),weights[2]);return at::linear(hidden,head);};
 auto labels=at::tensor(std::vector<int64_t>{-100,-100,2,3,6},at::kLong);std::vector<int64_t> ids={0,1,2,3,6};
 auto core=luban_expert_preservation::freeze(ids,labels,forward(false));
 auto parent=luban_expert_preservation::freeze(ids,labels,forward(true));
 // A later candidate must not change either captured teacher.
 weights[2]=(weights[2]+.03).detach();save(root,"teacher_core",core.logProbability);save(root,"teacher_parent",parent.logProbability);
 std::array<at::Tensor,3> m,v;for(size_t i=0;i<3;i++){weights[i].set_requires_grad(true);m[i]=at::zeros_like(weights[i]);v[i]=at::zeros_like(weights[i]);}
 std::vector<at::Tensor> parameters(weights.begin(),weights.end());
 for(uint64_t step=1;step<=2;step++){
  for(auto& w:weights)w.mutable_grad()=at::Tensor();
  auto logits=forward(true),shifted=at::constant_pad_nd(labels,{0,1},-100).slice(0,1);
  auto ce=torch::nn::functional::cross_entropy(logits.to(at::kFloat),shifted,torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-100));
  auto kl=luban_expert_preservation::loss(logits,step==1?core:parent),total=ce+5.0*kl;total.backward();
  const auto prefix="step"+std::to_string(step)+"_";save(root,prefix+"ce",ce);save(root,prefix+"kl",kl);save(root,prefix+"total",total);
  for(size_t i=0;i<3;i++)save(root,prefix+"gradient_"+std::to_string(i),weights[i].grad());
  torch::nn::utils::clip_grad_norm_(parameters,1.0);updateLubanExpertAdam(weights,m,v,step,.0002);
  for(size_t i=0;i<3;i++){save(root,prefix+"weight_"+std::to_string(i),weights[i]);save(root,prefix+"m_"+std::to_string(i),m[i]);save(root,prefix+"v_"+std::to_string(i),v[i]);}
 }
 int rejected=0;for(int example=0;example<3;example++)try{auto broken=forward(false).clone();auto target=labels.clone();if(example==0)target.fill_(-100);if(example==1)broken.fill_(NAN);if(example==2)target[2]=7;luban_expert_preservation::freeze(ids,target,broken);}catch(const std::exception&){++rejected;}
 require(rejected==3,"Invalid teachers accepted");require(!core.logProbability.requires_grad()&&!parent.logProbability.requires_grad(),"Teacher has gradients");
 std::cout<<"{\"schema\":\"native-expert-preservation-cpu-v1\",\"steps\":2,\"teacher_count\":2,\"invalid_rejected\":3,\"dtype\":\""<<kind<<"\",\"learning_rate\":0.0002,\"preservation_weight\":5,\"GPU\":false,\"product_admitted\":false}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
