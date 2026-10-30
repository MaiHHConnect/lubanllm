#pragma once
// Fixed terminal conversion; never an Adam update or a replacement for quality gates.
#include "../.native-migration/research/native-replay-basis/ReplayBasis.h"
#include <functional>
#include <ATen/core/grad_mode.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>
namespace luban_conversion {
inline void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
// GPTQ error propagation from scripts/prepare_q4_packed.py; quantizer is BF16 RNE.
// The returned master records each column BEFORE its one rounding operation.
inline std::pair<at::Tensor,at::Tensor> gptqBf16(const at::Tensor& weights,const at::Tensor& upper,const std::function<void()>& check=[](){}) {
  require(weights.dim()==2 && weights.scalar_type()==at::kFloat && weights.device().is_cpu()
    && upper.sizes()==at::IntArrayRef({weights.size(1),weights.size(1)})
    && upper.scalar_type()==at::kFloat && upper.device().is_cpu()
    && at::isfinite(weights).all().item<bool>() && at::isfinite(upper).all().item<bool>()
    && (upper.diagonal()>0).all().item<bool>(), "Invalid finite GPTQ matrices");
  check();
  auto work=weights.clone(),master=at::empty_like(weights);
  auto quantized=at::empty_like(weights,weights.options().dtype(at::kBFloat16));
  const int64_t width=weights.size(1),block=128;
  for(int64_t begin=0;begin<width;begin+=block){
    check();const auto end=std::min(begin+block,width);
    auto current=work.slice(1,begin,end).clone(),errors=at::empty_like(current);
    const auto local=upper.slice(0,begin,end).slice(1,begin,end);
    for(int64_t offset=0;offset<end-begin;offset++){
      const auto column=begin+offset; const auto value=current.select(1,offset);
      auto decoded=value.to(at::kBFloat16);
      require(at::isfinite(decoded).all().item<bool>(), "Nonfinite BF16 rounding");
      master.select(1,column).copy_(value);quantized.select(1,column).copy_(decoded);
      auto error=(value-decoded.to(at::kFloat))/local[offset][offset];
      current.slice(1,offset).sub_(error.unsqueeze(1)*local[offset].slice(0,offset).unsqueeze(0));
      errors.select(1,offset).copy_(error);
    }
    require(at::isfinite(current).all().item<bool>()&&at::isfinite(errors).all().item<bool>(),"Nonfinite GPTQ block feedback");
    work.slice(1,end).sub_(at::matmul(errors,upper.slice(0,begin,end).slice(1,end)));
  }
  check();require(at::isfinite(master).all().item<bool>()&&master.to(at::kBFloat16).equal(quantized),"GPTQ master/candidate mismatch");
  return {quantized.contiguous(),master.contiguous()};
}
inline at::Tensor gptqInverse(const at::Tensor& inputs,const std::function<void()>& check=[](){}) {
  require(inputs.size(0)>=2,"At least two calibration positions required");
  check();auto hessian=(at::matmul(inputs.transpose(0,1),inputs)/inputs.size(0)).to(at::kDouble);
  const auto mean=hessian.diagonal().mean().item<double>();
  require(std::isfinite(mean)&&mean>0,"Calibration has no activation energy");
  hessian.diagonal().add_(.01*mean);
  check();auto lower=at::linalg_cholesky(hessian);hessian=at::Tensor();
  check();auto inverse=at::cholesky_inverse(lower);lower=at::Tensor();
  check();auto upper=at::linalg_cholesky(inverse,true).to(at::kFloat).triu_();check();return upper;
}

struct Drift { double total,old,q4,originalSpan; };
inline Drift measure(const at::Tensor& candidate,const at::Tensor& reference,const at::Tensor& basis,const at::Tensor& inputs,int64_t oldCount){
 auto delta=candidate.to(at::kFloat)-reference,output=at::matmul(delta,inputs.transpose(0,1));
 require(at::isfinite(output).all().item<bool>(),"Nonfinite conversion reconstruction");
 return {output.norm().item<double>(),output.slice(1,0,oldCount).norm().item<double>(),output.slice(1,oldCount).norm().item<double>(),at::matmul(delta,basis).norm().item<double>()};
}
inline bool improves(const Drift& candidate,const Drift& baseline){return candidate.total<baseline.total&&candidate.old<=baseline.old&&candidate.q4<=baseline.q4;}
struct Result { at::Tensor master,candidate; std::vector<Drift> rounds; int64_t rank; double originalSpanError; bool eligible; };
inline Result convert(const at::Tensor& sourceMaster,const at::Tensor& sourceReference,const at::Tensor& sourceBasis,const at::Tensor& inputs,int64_t oldCount,const std::function<void()>& check=[](){}){
 check();require(sourceMaster.dim()==2&&sourceReference.sizes()==sourceMaster.sizes()&&sourceMaster.scalar_type()==at::kFloat&&sourceReference.scalar_type()==at::kFloat,"Conversion requires FP32 master/reference");
 require(inputs.device().is_cpu()&&inputs.scalar_type()==at::kFloat&&inputs.dim()==2&&inputs.size(1)==sourceMaster.size(1)&&oldCount>0&&oldCount<inputs.size(0)&&at::isfinite(inputs).all().item<bool>(),"Frozen OLD plus Q4 FP32 inputs required");
 auto master=sourceMaster.detach().to(at::kCPU).contiguous(),reference=sourceReference.detach().to(at::kCPU).contiguous(),basis=sourceBasis.detach().to(at::kCPU).contiguous();
 require(at::isfinite(master).all().item<bool>()&&at::isfinite(reference).all().item<bool>(),"Nonfinite conversion source");
 require(basis.dim()==2&&basis.scalar_type()==at::kFloat&&basis.size(0)==master.size(1)&&basis.size(1)>0&&basis.size(1)<master.size(1)&&at::isfinite(basis).all().item<bool>(),"Invalid original training basis");
 require((at::matmul(basis.transpose(0,1),basis)-at::eye(basis.size(1))).abs().max().item<double>()<1e-4,"Training basis not orthonormal");
 check();const auto expanded=lubanReplayBasisCPU(basis.transpose(0,1).contiguous(),inputs);check();
 require(expanded.training_rank==basis.size(1)&&expanded.basis.size(1)<master.size(1),"Conversion exhausted complement or lost original basis");
 const auto& joint=expanded.basis;
 const double spanError=(basis-at::matmul(joint,at::matmul(joint.transpose(0,1),basis))).abs().max().item<double>();require(spanError<=2e-5,"Conversion weakened original span");
 auto delta=master-reference,projected=reference+delta-at::matmul(at::matmul(delta,joint),joint.transpose(0,1));
 const auto initial=projected.to(at::kBFloat16),upper=gptqInverse(inputs,check);
 auto first=gptqBf16(projected,upper,check);check();
 auto roundedDelta=first.first.to(at::kFloat)-reference;
 auto corrected=first.first.to(at::kFloat)-at::matmul(at::matmul(roundedDelta,joint),joint.transpose(0,1));
 auto second=gptqBf16(corrected,upper,check);check();
 std::vector<Drift> rounds={measure(initial,reference,basis,inputs,oldCount),measure(first.first,reference,basis,inputs,oldCount),measure(second.first,reference,basis,inputs,oldCount)};
 return {second.second,second.first,rounds,joint.size(1),spanError,improves(rounds[2],rounds[0])&&improves(rounds[2],rounds[1])};
}
inline std::string reportJSON(const Result& result,int64_t oldCount,int64_t positions){
 std::ostringstream out;out<<std::setprecision(17)<<"{\"method\":\"gptq-bf16-refine-v1\",\"absolute_step\":450,\"passes\":2,\"old_positions\":"<<oldCount<<",\"q4_train_positions\":"<<positions-oldCount<<",\"expanded_rank\":"<<result.rank<<",\"original_span_max_abs\":"<<result.originalSpanError<<",\"eligible\":"<<(result.eligible?"true":"false")<<",\"rounds\":[";
 for(size_t i=0;i<result.rounds.size();i++){const auto& d=result.rounds[i];if(i)out<<',';out<<"{\"iteration\":"<<i<<",\"reconstruction_frobenius\":"<<d.total<<",\"known_old_frobenius\":"<<d.old<<",\"q4_train_frobenius\":"<<d.q4<<",\"original_replay_span_frobenius\":"<<d.originalSpan<<'}';}
 out<<"]}";return out.str();
}
// The caller retains all optimizer moments/reference/basis; only terminal master/candidate change.
inline bool applyAt450(at::Tensor& master,at::Tensor& candidate,const at::Tensor& reference,const at::Tensor& basis,const at::Tensor& inputs,int64_t oldCount,int64_t absoluteStep,bool segmentFinal,const std::function<void()>& check,Result* report=nullptr){
 require(absoluteStep>0&&absoluteStep<=450,"Terminal conversion step outside fixed450");
 if(absoluteStep!=450)return false;
 require(segmentFinal,"Terminal conversion must be last segment update");check();
 const auto result=convert(master,reference,basis,inputs,oldCount,check);
 require(result.eligible,"Terminal conversion failed fixed reconstruction gate");check();
 auto nextMaster=result.master.to(master.device()),nextCandidate=result.candidate.to(candidate.device());check();
 require(nextMaster.to(candidate.scalar_type()).equal(nextCandidate),"Terminal master/candidate cast differs");
 {at::NoGradGuard noGrad;master.copy_(nextMaster);candidate=nextCandidate;}
 if(report)*report=result;return true;
}
}
