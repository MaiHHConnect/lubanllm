// SPDX-License-Identifier: Apache-2.0
// Research-only Q4 error correction. B/U BF16, intermediate/accumulation FP32.
#include <metal_stdlib>
using namespace metal;
inline float residual_float(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort residual_bf16(float x) {
    uint u = as_type<uint>(x);
    return ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
kernel void qwen35_residual_project(
    device const ushort* x [[buffer(0)]], device const ushort* b [[buffer(1)]],
    constant uint4& dims [[buffer(2)]], device float* z [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint rank = dims.w, k = dims.y;
    if (gid.x >= rank || gid.y >= dims.x) return;
    float sum = 0;
    for (uint i = lane; i < k; i += 32)
        sum = fma(residual_float(x[gid.y * k + i]), residual_float(b[gid.x * k + i]), sum);
    sum = simd_sum(sum);
    if (lane == 0) z[gid.y * rank + gid.x] = sum;
}
kernel void qwen35_residual_add(
    device const float* z [[buffer(0)]], device const ushort* u [[buffer(1)]],
    constant uint4& dims [[buffer(2)]], device ushort* y [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dims.z || gid.y >= dims.x) return;
    float correction = 0;
    for (uint r = 0; r < dims.w; ++r)
        correction = fma(z[gid.y * dims.w + r], residual_float(u[gid.x * dims.w + r]), correction);
    const uint i = gid.y * dims.z + gid.x;
    y[i] = residual_bf16(residual_float(y[i]) + correction);
}
