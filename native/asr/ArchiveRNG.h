#pragma once
#include <torch/torch.h>
#include "CPURNG.h"

// ASR preserves its actual entry stream; no evolution seed42 is introduced.
class ASRCPURNGScope {
 at::Tensor parent=captureNativeCPURNG();bool active=true;
public:
 void restore(){if(active){applyNativeCPURNG(parent);if(!torch::equal(parent,captureNativeCPURNG()))throw std::runtime_error("ASR parent CPU RNG restore differs");active=false;}}
 ~ASRCPURNGScope() noexcept {if(active)try{restore();}catch(...){std::terminate();}}
};
inline at::Tensor saveASRCPURNG(torch::serialize::OutputArchive& archive){auto state=captureNativeCPURNG();archive.write("cpu_rng",state);return state;}
inline at::Tensor restoreASRCPURNG(torch::serialize::InputArchive& archive,const at::Tensor& expected){
 at::Tensor state;archive.read("cpu_rng",state);if(!torch::equal(state,expected))throw std::invalid_argument("ASR archive/envelope CPU RNG mismatch");applyNativeCPURNG(state);
 if(!torch::equal(state,captureNativeCPURNG()))throw std::runtime_error("ASR resumed CPU RNG differs");return state;
}
