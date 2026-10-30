// SPDX-License-Identifier: Apache-2.0
// Research-only raw copy; never writes an inference tensor.
#include <metal_stdlib>
using namespace metal;
kernel void qwen35_research_trace_copy(device const ushort* src [[buffer(0)]],
    device ushort* dst [[buffer(1)]], constant uint& count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = src[i];
}
