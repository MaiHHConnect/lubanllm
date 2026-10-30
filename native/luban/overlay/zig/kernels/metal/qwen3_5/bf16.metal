// SPDX-License-Identifier: Apache-2.0
// Luban BF16 extension to TensorFold's native Qwen3.5 encoder.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// W is contiguous [outputs, inputs]. Every row follows the same FP32 FMA
// and SIMD reduction order for both single-token and batched invocations.
kernel void qwen35_bf16_projection(const device bfloat* x [[buffer(0)]],
                                   constant uint* dims [[buffer(1)]],
                                   const device bfloat* weight [[buffer(2)]],
                                   device bfloat* out [[buffer(3)]],
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
      const float x0 = float(x[k]), w0 = float(weight[wb + k]);
      const float x1 = float(x[k + 32]), w1 = float(weight[wb + k + 32]);
      const float x2 = float(x[k + 64]), w2 = float(weight[wb + k + 64]);
      const float x3 = float(x[k + 96]), w3 = float(weight[wb + k + 96]);
      acc = fma(x0, w0, acc);
      acc = fma(x1, w1, acc);
      acc = fma(x2, w2, acc);
      acc = fma(x3, w3, acc);
    }
    for (; k < inputs; k += 32)
      acc = fma(float(x[k]), float(weight[wb + k]), acc);
    acc = simd_sum(acc);
    if (lane == 0) out[n] = bfloat(acc);
    return;
  }
  // Reuse each original weight across eight prompt rows. Each row keeps the
  // same FP32 FMA and SIMD reduction order as the single-token path.
  const uint row = group.y * 8;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (uint k = lane; k < inputs; k += 32) {
    const float w = float(weight[wb + k]);
    for (uint r = 0; r < 8; r++)
      if (row + r < rows) acc[r] = fma(float(x[size_t(row + r) * inputs + k]), w, acc[r]);
  }
  for (uint r = 0; r < 8; r++) {
    const float sum = simd_sum(acc[r]);
    if (lane == 0 && row + r < rows) out[size_t(row + r) * outputs + n] = bfloat(sum);
  }
}

// Copy the checkpoint's BF16 bits, including any special-value payloads.
kernel void qwen35_bf16_embed(const device uint* ids [[buffer(0)]],
                              const device ushort* weight [[buffer(1)]],
                              device ushort* out [[buffer(2)]],
                              constant uint2& dims [[buffer(3)]],
                              uint2 p [[thread_position_in_grid]]) {
  if (p.x >= dims.y || p.y >= dims.x) return;
  out[size_t(p.y) * dims.y + p.x] = weight[size_t(ids[p.y]) * dims.y + p.x];
}

// HF Qwen3.5 RMSNorm adds one to the FP32 weight. Its gated Delta norm
// uses a different rule and continues to use the original gdn_post kernel.
kernel void qwen35_bf16_norm(const device bfloat* h [[buffer(0)]],
                             const device bfloat* residual [[buffer(1)]],
                             const device bfloat* weight [[buffer(2)]],
                             constant float& eps [[buffer(3)]],
                             device bfloat* h_out [[buffer(4)]],
                             device bfloat* out [[buffer(5)]],
                             constant uint& with_residual [[buffer(6)]],
                             uint3 group [[threadgroup_position_in_grid]],
                             uint3 local [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[4];
  const uint t = local.x;
  const size_t base = size_t(group.y) * 2048 + 16 * t;
  float values[16], acc = 0.0f;
  for (uint i = 0; i < 16; i++) {
    bfloat value = h[base + i];
    if (with_residual != 0) {
      value = bfloat(float(value) + float(residual[base + i]));
      h_out[base + i] = value;
    }
    values[i] = float(value);
    acc = fma(values[i], values[i], acc);
  }
  acc = simd_sum(acc);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < 4; i++) total += sums[i];
  const float inv = metal::precise::rsqrt(total / 2048.0f + eps);
  for (uint i = 0; i < 16; i++)
    out[base + i] = bfloat((1.0f + float(weight[16 * t + i])) * (values[i] * inv));
}

kernel void qwen35_bf16_head_norm(const device bfloat* x [[buffer(0)]],
                                  const device bfloat* weight [[buffer(1)]],
                                  device bfloat* out [[buffer(2)]],
                                  constant uint4& dims [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint t [[thread_position_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[2], inv;
  const uint head = row % dims.y, m = row / dims.y;
  const size_t base = size_t(m) * dims.z + head * dims.w;
  float values[4], acc = 0.0f;
  for (uint i = 0; i < 4; i++) {
    values[i] = float(x[base + 4 * t + i]);
    acc = fma(values[i], values[i], acc);
  }
  acc = simd_sum(acc);
  if (lane == 0) sums[sg] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) inv = metal::precise::rsqrt((sums[0] + sums[1]) / 256.0f + 1e-6f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < 4; i++) {
    const uint d = 4 * t + i;
    out[(size_t(m) * dims.y + head) * 256 + d] = bfloat((1.0f + float(weight[d])) * (values[i] * inv));
  }
}

kernel void qwen35_bf16_mlp_act(const device bfloat* gate [[buffer(0)]],
                                const device bfloat* up [[buffer(1)]],
                                device bfloat* out [[buffer(2)]],
                                uint2 p [[thread_position_in_grid]]) {
  if (p.x >= 6144) return;
  const size_t i = size_t(p.y) * 6144 + p.x;
  const float g = float(gate[i]);
  const bfloat activated = bfloat(g / (1.0f + metal::precise::exp(-g)));
  out[i] = bfloat(float(activated) * float(up[i]));
}

inline bfloat qwen35_bf16_rotate(const device bfloat* x, uint d, uint position) {
  if (d >= 64) return x[d];
  const uint i = d % 32;
  const float inv = 1.0f / metal::precise::pow(10000000.0f, float(i) / 32.0f);
  const float angle = float(position) * inv;
  const float cs = float(bfloat(metal::precise::cos(angle)));
  const float sn = float(bfloat(metal::precise::sin(angle)));
  const float a = float(x[i]), b = float(x[i + 32]);
  const bfloat p = bfloat((d < 32 ? a : b) * cs);
  const bfloat q = bfloat((d < 32 ? -b : a) * sn);
  return bfloat(float(p) + float(q));
}

kernel void qwen35_bf16_queries(const device bfloat* x [[buffer(0)]],
                                device bfloat* out [[buffer(1)]],
                                constant uint4& dims [[buffer(2)]],
                                uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z, m = dims.z + r;
  out[(size_t(h) * dims.y + m) * 256 + d] = qwen35_bf16_rotate(x + (size_t(m) * 8 + h) * 256, d, dims.x + r);
}

kernel void qwen35_bf16_keys(const device bfloat* k [[buffer(0)]],
                             const device bfloat* v [[buffer(1)]],
                             device bfloat* keys [[buffer(2)]],
                             device bfloat* values [[buffer(3)]],
                             constant uint4& dims [[buffer(4)]],
                             uint3 p [[thread_position_in_grid]]) {
  const uint d = p.x, h = p.y, r = p.z, m = dims.z + r;
  const size_t dst = (size_t(h) * dims.y + dims.x + r) * 256 + d;
  keys[dst] = qwen35_bf16_rotate(k + (size_t(m) * 2 + h) * 256, d, dims.x + r);
  values[dst] = v[(size_t(m) * 2 + h) * 256 + d];
}

// Original BF16 operands expand exactly to FP32 in local tiles. This changes
// reduction order, so full-model quality is checked separately from dtype.
kernel void qwen35_bf16_prefill_matrix(
    const device bfloat* x [[buffer(0)]],
    constant uint* dims [[buffer(1)]],
    const device bfloat* weight [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
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
      a[i] = n < outputs && k < inputs ? float(weight[size_t(n) * inputs + k]) : 0.0f;
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
