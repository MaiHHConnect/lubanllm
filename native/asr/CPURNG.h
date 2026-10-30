#pragma once
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ATen.h>
#include <mutex>
#include <stdexcept>

// These helpers affect only the calling process. Caller must serialize its CPU
// random work; a model/file lock cannot prevent unrelated threads using the RNG.
inline at::Tensor captureNativeCPURNG() {
  auto generator=at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.get_state().clone();
}
inline void applyNativeCPURNG(const at::Tensor& state) {
  if(!state.defined()||!state.device().is_cpu()||state.scalar_type()!=at::kByte
      ||state.dim()!=1||!state.is_contiguous()||state.numel()<1||state.numel()>65536)
    throw std::invalid_argument("CPU RNG requires contiguous one-dimensional U8 state");
  auto generator=at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(generator.mutex());
  const auto before=generator.get_state().clone();
  try { generator.set_state(state); }
  catch(...) { generator.set_state(before); throw; }
}
inline void seedNativeCPURNG(uint64_t seed) {
  auto generator=at::detail::getDefaultCPUGenerator();
  std::lock_guard<std::mutex> lock(generator.mutex());
  generator.set_current_seed(seed);
}
// Matches evolution's CPU fork_rng(devices=[]): preserve entry, seed42, then
// replace with already captured resume state when present, restore on exit.
// This deliberately performs no MPS operation and is not wired to Training.cpp.
class NativeCPUTrainingRNGScope {
  at::Tensor parent;
  bool active=true;
public:
  explicit NativeCPUTrainingRNGScope(const at::Tensor& resume={}) : parent(captureNativeCPURNG()) {
    try { seedNativeCPURNG(42); if(resume.defined())applyNativeCPURNG(resume); }
    catch(...) { applyNativeCPURNG(parent); active=false; throw; }
  }
  NativeCPUTrainingRNGScope(const NativeCPUTrainingRNGScope&)=delete;
  NativeCPUTrainingRNGScope& operator=(const NativeCPUTrainingRNGScope&)=delete;
  void restore(){if(active){applyNativeCPURNG(parent);active=false;}}
  ~NativeCPUTrainingRNGScope() noexcept {
    if(active)try {restore();}catch(...) {std::terminate();}
  }
};
