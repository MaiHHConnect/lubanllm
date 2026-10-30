#pragma once
#include <torch/torch.h>
#include <stdexcept>
#include <vector>

// Runtime._learn_loaded: fixed teacher, answer/EOS positions, full vocabulary,
// FP32 log_softmax and KL(log_target=True, reduction="batchmean").
namespace luban_expert_preservation {
inline void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
inline at::Tensor answerLogProb(const at::Tensor& logits,const at::Tensor& labels){
 require(logits.dim()==2&&labels.dim()==1&&labels.scalar_type()==at::kLong&&logits.size(0)==labels.numel()&&labels.numel()>=2&&logits.size(1)>=2,"Preservation logits/labels shape differs");
 require(labels.eq(-100).logical_or(labels.ge(0).logical_and(labels.lt(logits.size(1)))).all().item<bool>(),"Preservation labels outside vocabulary");
 auto mask=labels.slice(0,1).ne(-100).to(logits.device());
 require(mask.any().item<bool>(),"Preservation has no answer/EOS targets");
 auto selected=logits.slice(0,0,logits.size(0)-1).index({mask}).to(at::kFloat);
 require(at::isfinite(selected).all().item<bool>(),"Preservation logits nonfinite");
 return at::log_softmax(selected,-1);
}
struct Teacher {
 std::vector<int64_t> ids;
 at::Tensor labels,logProbability;
};
inline Teacher freeze(const std::vector<int64_t>& ids,const at::Tensor& labels,const at::Tensor& logits){
 require(ids.size()==size_t(labels.numel()),"Preservation token count differs");
 torch::NoGradGuard noGrad;
 return{ids,labels.detach().cpu().clone(),answerLogProb(logits,labels).detach().cpu().clone()};
}
inline at::Tensor loss(const at::Tensor& logits,const Teacher& teacher){
 auto student=answerLogProb(logits,teacher.labels);
 require(teacher.logProbability.scalar_type()==at::kFloat&&teacher.logProbability.dim()==2&&teacher.logProbability.sizes()==student.sizes()&&!teacher.logProbability.requires_grad()&&at::isfinite(teacher.logProbability).all().item<bool>(),"Frozen preservation teacher differs");
 auto reference=teacher.logProbability.to(logits.device());
 // This is precisely PyTorch's log-target batchmean, not mean over vocabulary.
 auto value=at::sum(at::exp(reference)*(reference-student))/student.size(0);
 require(at::isfinite(value).item<bool>(),"Preservation KL nonfinite");
 return value;
}
}
