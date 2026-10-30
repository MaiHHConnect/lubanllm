// SPDX-License-Identifier: Apache-2.0
// Exact finite BF16 argmax: equal logits select the smallest token id.
#include <metal_stdlib>
using namespace metal;

inline void reduce_candidates(threadgroup float* values, threadgroup uint* ids,
                              threadgroup uint* invalid, uint t) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride != 0; stride >>= 1) {
    if (t < stride) {
      const float other = values[t + stride];
      const uint id = ids[t + stride];
      if (other > values[t] || (other == values[t] && id < ids[t])) {
        values[t] = other;
        ids[t] = id;
      }
      invalid[t] |= invalid[t + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void qwen35_greedy_partial(const device bfloat* logits [[buffer(0)]],
                                  constant uint& count [[buffer(1)]],
                                  device uint2* partial [[buffer(2)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]]) {
  threadgroup float values[256];
  threadgroup uint ids[256], invalid[256];
  const uint id = group * 256 + t;
  const float value = id < count ? float(logits[id]) : -INFINITY;
  values[t] = value;
  ids[t] = id < count ? id : UINT_MAX;
  invalid[t] = id < count && !isfinite(value);
  reduce_candidates(values, ids, invalid, t);
  if (t == 0) partial[group] = uint2(as_type<uint>(values[0]), invalid[0] ? UINT_MAX : ids[0]);
}

kernel void qwen35_greedy_finish(const device uint2* partial [[buffer(0)]],
                                 constant uint& count [[buffer(1)]],
                                 device uint* selected [[buffer(2)]],
                                 uint t [[thread_index_in_threadgroup]]) {
  threadgroup float values[256];
  threadgroup uint ids[256], invalid[256];
  float best = -INFINITY;
  uint best_id = UINT_MAX, bad = 0;
  for (uint i = t; i < count; i += 256) {
    const float value = as_type<float>(partial[i].x);
    const uint id = partial[i].y;
    bad |= id == UINT_MAX;
    if (value > best || (value == best && id < best_id)) { best = value; best_id = id; }
  }
  values[t] = best; ids[t] = best_id; invalid[t] = bad;
  reduce_candidates(values, ids, invalid, t);
  if (t == 0) selected[0] = invalid[0] ? UINT_MAX : ids[0];
}
