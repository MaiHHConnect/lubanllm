// SPDX-License-Identifier: Apache-2.0
// BF16 HF rounding extension to TensorFold native Delta kernels.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
[[kernel]] void qwen35_bf16_gdn_pre(
  const device bfloat* QKV [[buffer(0)]],
  const device bfloat* CS [[buffer(1)]],
  const device bfloat* CW [[buffer(2)]],
  const device int32_t* windows [[buffer(3)]],
  const device bfloat* Ain [[buffer(4)]],
  const device bfloat* Bin [[buffer(5)]],
  const device float* ALOG [[buffer(6)]],
  const device bfloat* DT [[buffer(7)]],
  device bfloat* Q [[buffer(8)]],
  device bfloat* Kout [[buffer(9)]],
  device bfloat* Vout [[buffer(10)]],
  device float* G [[buffer(11)]],
  device bfloat* BETA [[buffer(12)]],
  device bfloat* CO [[buffer(13)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
constexpr int NK = 16;
constexpr int NV = 16;
constexpr int DK = 128;
constexpr int DV = 128;
constexpr int TAPS = 4;

  const uint lane = thread_index_in_simdgroup;
  const uint head = threadgroup_position_in_grid.y;
  const uint w = threadgroup_position_in_grid.z;
  constexpr int C = 2 * NK * DK + NV * DV;
  const bool isq = head < NK, isk = !isq && head < 2 * NK;
  const int c0 = isq ? int(head) * DK : (isk ? NK * DK + (int(head) - NK) * DK : 2 * NK * DK + (int(head) - 2 * NK) * DV);
  constexpr int PER = DK / 32;
  float vals[PER];
  for (int j = 0; j < PER; j++) {
    const int c = c0 + int(lane) * PER + j;
    float acc = 0.0f;
    for (int tap = 0; tap < TAPS; tap++) {
      const int row = windows[w * TAPS + tap];
      const float xv = row < TAPS - 1 ? float(CS[row * C + c]) : float(QKV[(row - (TAPS - 1)) * C + c]);
      acc += float(CW[c * TAPS + tap]) * xv;
    }
    const float conv = float(bfloat(acc));
    vals[j] = float(bfloat(conv / (1.0f + metal::precise::exp(-conv))));
  }
  if (isq || isk) {
    float ss = 0.0f;
    for (int j = 0; j < PER; j++) ss += float(bfloat(vals[j] * vals[j]));
    ss = float(bfloat(simd_sum(ss)));
    const float squared = float(bfloat(ss + 1e-6f));
    const float inv = float(bfloat(metal::precise::rsqrt(squared)));
    for (int j = 0; j < PER; j++) {
      const bfloat out = bfloat(vals[j] * inv);
      if (isq) Q[(w * NK + head) * DK + lane * PER + j] = out;
      else Kout[(w * NK + head - NK) * DK + lane * PER + j] = out;
    }
  } else {
    const int hv = int(head) - 2 * NK;
    for (int j = 0; j < PER; j++) Vout[(w * NV + hv) * DV + lane * PER + j] = bfloat(vals[j]);
    if (lane == 0) {
      const float s = float(Ain[w * NV + hv]) + float(DT[hv]);
      // Stable FP32 log1p(exp(s)); retain tiny negative-tail softplus values.
      const float ex = metal::precise::exp(min(s, 20.0f)), u = 1.0f + ex;
      const float sp = s > 20.0f ? s : (u == 1.0f ? ex : metal::precise::log(u) * (ex / (u - 1.0f)));
      G[w * NV + hv] = metal::precise::exp(-metal::precise::exp(ALOG[hv]) * sp);
      BETA[w * NV + hv] = bfloat(1.0f / (1.0f + metal::precise::exp(-float(Bin[w * NV + hv]))));
    }
  }

    for (int r = 0; r < TAPS - 1; r++) {
      const int row = windows[w * TAPS + 1 + r];
      for (int j = 0; j < PER; j++) {
        const int c = c0 + int(lane) * PER + j;
        CO[(int(w) * (TAPS - 1) + r) * C + c] = row < TAPS - 1 ? CS[row * C + c] : QKV[(row - (TAPS - 1)) * C + c];
      }
    }

}

[[kernel]] void qwen35_bf16_gdn_chain(
  const device bfloat* q [[buffer(0)]],
  const device bfloat* k [[buffer(1)]],
  const device bfloat* v [[buffer(2)]],
  const device float* g [[buffer(3)]],
  const device bfloat* beta [[buffer(4)]],
  const device float* state_in [[buffer(5)]],
  const constant int32_t* dims [[buffer(6)]],
  device bfloat* y [[buffer(7)]],
  device float* state_out [[buffer(8)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 thread_position_in_grid [[thread_position_in_grid]],
  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
constexpr int Dk = 128;
constexpr int Dv = 128;
constexpr int Hk = 16;
constexpr int Hv = 16;

        const int W = dims[0];
        auto n = thread_position_in_grid.z;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (hv_idx * Dv + dv_idx) * Dk;
        float state[n_per_t], kc[n_per_t], qc[n_per_t], kn[n_per_t], qn[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          kc[i] = static_cast<float>(k[hk_idx * Dk + n_per_t * dk_idx + i]);
          qc[i] = static_cast<float>(q[hk_idx * Dk + n_per_t * dk_idx + i]) * 0.08838834764831845f;
        }
        float vc = static_cast<float>(v[hv_idx * Dv + dv_idx]);
        float gc = static_cast<float>(g[hv_idx]);
        float bc = static_cast<float>(beta[hv_idx]);
        float vn = 0.0f, gn = 0.0f, bn = 0.0f;
        for (int i = 0; i < n_per_t; ++i) state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        for (int node = 0; node < W; ++node) {
          if (node + 1 < W) {
            for (int i = 0; i < n_per_t; ++i) {
              kn[i] = static_cast<float>(k[((node + 1) * Hk + hk_idx) * Dk + n_per_t * dk_idx + i]);
              qn[i] = static_cast<float>(q[((node + 1) * Hk + hk_idx) * Dk + n_per_t * dk_idx + i]) * 0.08838834764831845f;
            }
            vn = static_cast<float>(v[((node + 1) * Hv + hv_idx) * Dv + dv_idx]);
            gn = static_cast<float>(g[(node + 1) * Hv + hv_idx]);
            bn = static_cast<float>(beta[(node + 1) * Hv + hv_idx]);
          }
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] * gc;
            kv_mem += state[i] * kc[i];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (vc - kv_mem) * bc;
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = state[i] + kc[i] * delta;
            out += state[i] * qc[i];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[(node * Hv + hv_idx) * Dv + dv_idx] = bfloat(out);
          }
          if (node + 1 < W) {
            for (int i = 0; i < n_per_t; ++i) { kc[i] = kn[i]; qc[i] = qn[i]; }
            vc = vn; gc = gn; bc = bn;
          }
          if (dims[1] || node + 1 == W) {
            const int saved = dims[1] ? node : 0;
            auto o_state = state_out + ((saved * Hv + hv_idx) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = state[i];
          }

        }


}

[[kernel]] void qwen35_bf16_gdn_post(
  const device bfloat* Y [[buffer(0)]],
  const device bfloat* Z [[buffer(1)]],
  const device float* NW [[buffer(2)]],
  const constant float* eps [[buffer(3)]],
  device bfloat* OUT [[buffer(4)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
constexpr int NV = 16;
constexpr int DV = 128;
constexpr int ZS = 2048;
constexpr int ZO = 0;

  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  constexpr int PER = DV / 32;
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = NW[d] * float(bfloat(yv[j] * inv));
    const float zf = float(Z[m * ZS + ZO + hv * DV + d]);
    OUT[m * NV * DV + hv * DV + d] = bfloat(zf / (1.0f + metal::precise::exp(-zf)) * x);
  }

}
