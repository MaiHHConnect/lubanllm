// SPDX-License-Identifier: Apache-2.0
// Signed int8/group64 weights, BF16 scales; FP32 dequantization and accumulation.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void qwen35_q8_projection(const device bfloat* x [[buffer(0)]],
                                  constant uint* dims [[buffer(1)]],
                                  const device char* weight [[buffer(2)]],
                                  const device bfloat* scales [[buffer(3)]],
                                  device bfloat* out [[buffer(4)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint n = group.x * 8 + sg;
  const uint rows = dims[0], inputs = dims[1], outputs = dims[2];
  if (n >= outputs) return;
  const size_t wb = size_t(n) * inputs;
  if (rows == 1) {
    float acc = 0.0f;
    uint k = lane;
    // Prefetch four coalesced SIMD stripes, retaining the original FMA order.
    // This reduces loop/address overhead without changing the reduction tree.
    for (; k + 96 < inputs; k += 128) {
      const float s0 = float(scales[(wb + k) / 64]), s1 = float(scales[(wb + k + 64) / 64]);
      const float x0 = float(x[k]), w0 = float(weight[wb + k]) * s0;
      const float x1 = float(x[k + 32]), w1 = float(weight[wb + k + 32]) * s0;
      const float x2 = float(x[k + 64]), w2 = float(weight[wb + k + 64]) * s1;
      const float x3 = float(x[k + 96]), w3 = float(weight[wb + k + 96]) * s1;
      acc = fma(x0, w0, acc);
      acc = fma(x1, w1, acc);
      acc = fma(x2, w2, acc);
      acc = fma(x3, w3, acc);
    }
    for (; k < inputs; k += 32)
      acc = fma(float(x[k]), float(weight[wb + k]) * float(scales[(wb + k) / 64]), acc);
    acc = simd_sum(acc);
    if (lane == 0) out[n] = bfloat(acc);
    return;
  }
  // Reuse each dequantized weight across eight prompt rows. Each row keeps the
  // same FP32 FMA and SIMD reduction order as the single-token path.
  const uint row = group.y * 8;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = lane; k < inputs; k += 32) {
    const float w = float(weight[wb + k]) * float(scales[(wb + k) / 64]);
    for (uint r = 0; r < 8; r++)
      if (row + r < rows) acc[r] = fma(float(x[size_t(row + r) * inputs + k]), w, acc[r]);
  }
  for (uint r = 0; r < 8; r++) {
    const float sum = simd_sum(acc[r]);
    if (lane == 0 && row + r < rows) out[size_t(row + r) * outputs + n] = bfloat(sum);
  }
}


// Only local matrix tiles are dequantized; no full dense weight copy.
kernel void qwen35_q8_prefill_matrix(
    const device bfloat* x [[buffer(0)]],
    constant uint* dims [[buffer(1)]],
    const device char* weight [[buffer(2)]],
    const device bfloat* scales [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint rows = dims[0], inputs = dims[1], outputs = dims[2];
  const uint nb = group.x * 64, rb = group.y * 16;
  threadgroup float a[64 * 32];
  threadgroup float b[32 * 16];
  threadgroup float c[64 * 16];
  simdgroup_matrix<float, 8, 8> p0(0.0f), p1(0.0f);
  for (uint kb = 0; kb < inputs; kb += 32) {
    for (uint i = t; i < 64 * 32; i += 256) {
      const uint n = nb + i / 32, k = kb + i % 32;
      a[i] = n < outputs && k < inputs ? float(weight[size_t(n) * inputs + k]) * float(scales[(size_t(n) * inputs + k) / 64]) : 0.0f;
    }
    for (uint i = t; i < 32 * 16; i += 256) {
      const uint r = rb + i % 16, k = kb + i / 16;
      b[i] = r < rows && k < inputs ? float(x[size_t(r) * inputs + k]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 32; j += 8) {
      simdgroup_matrix<float, 8, 8> am, bm0, bm1;
      simdgroup_load(am, a + sg * 8 * 32 + j, 32);
      simdgroup_load(bm0, b + j * 16, 16);
      simdgroup_load(bm1, b + j * 16 + 8, 16);
      simdgroup_multiply_accumulate(p0, am, bm0, p0);
      simdgroup_multiply_accumulate(p1, am, bm1, p1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(p0, c + sg * 8 * 16, 16);
  simdgroup_store(p1, c + sg * 8 * 16 + 8, 16);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = t; i < 64 * 16; i += 256) {
    const uint n = nb + i / 16, r = rb + i % 16;
    if (n < outputs && r < rows) out[size_t(r) * outputs + n] = bfloat(c[i]);
  }
}

kernel void qwen35_q8_embed(const device uint* ids [[buffer(0)]],
                            const device char* weight [[buffer(1)]],
                            const device bfloat* scales [[buffer(2)]],
                            device bfloat* out [[buffer(3)]],
                            constant uint2& dims [[buffer(4)]],
                            uint2 p [[thread_position_in_grid]]) {
  if (p.x >= dims.y || p.y >= dims.x) return;
  const size_t index = size_t(ids[p.y]) * dims.y + p.x;
  out[size_t(p.y) * dims.y + p.x] = bfloat(float(weight[index]) * float(scales[index / 64]));
}
