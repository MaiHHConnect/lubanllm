// SPDX-License-Identifier: Apache-2.0
// Same BF16 boundaries as NeuralExpert and DynamicMLP; no fused residual order.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void luban_expert_act(const device bfloat* gate [[buffer(0)]],
                            const device bfloat* up [[buffer(1)]],
                            device bfloat* out [[buffer(2)]],
                            constant uint& width [[buffer(3)]],
                            uint2 p [[thread_position_in_grid]]) {
  if (p.x >= width) return;
  const size_t i = size_t(p.y) * width + p.x;
  const float g = float(gate[i]);
  const bfloat activated = bfloat(g / (1.0f + metal::precise::exp(-g)));
  out[i] = bfloat(float(activated) * float(up[i]));
}

kernel void luban_expert_add(const device bfloat* expert [[buffer(0)]],
                            device bfloat* base [[buffer(1)]],
                            constant uint& hidden [[buffer(2)]],
                            uint2 p [[thread_position_in_grid]]) {
  if (p.x >= hidden) return;
  const size_t i = size_t(p.y) * hidden + p.x;
  base[i] = bfloat(float(base[i]) + float(expert[i]));
}
