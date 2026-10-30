#!/usr/bin/env python3
"""Small, weight-only prototype of activation-weighted Q4 clipping.

It mirrors the current affine/group64 metadata rounding, but uses calibration
activations to choose a clipped range. It is a research diagnostic, not a
runtime sidecar and never reads or writes the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bf16(value: np.ndarray) -> np.ndarray:
    bits = value.astype(np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def away(value: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(value) + .5), value)


def affine4(values: np.ndarray, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    lo = float(values.min()) if lower is None else lower
    hi = float(max(values.max(), 0.0)) if upper is None else max(upper, 0.0)
    bias = lo if abs(lo) > abs(hi) else hi
    scale = max((hi - lo) / 15.0, 1e-7)
    scale = -scale if bias >= 0 else scale
    zero = away(np.asarray([bias / scale]))[0]
    if zero != 0:
        scale = bias / zero
    else:
        bias = 0.0
    stored_scale, stored_bias = bf16(np.asarray([scale, bias], np.float32))
    code = np.clip(away((values - stored_bias) / stored_scale), 0, 15)
    return code * stored_scale + stored_bias


def weighted_clip(weights: np.ndarray, activations: np.ndarray, grid: int = 20) -> dict:
    if weights.shape != (64,) or activations.ndim != 2 or activations.shape[1] != 64:
        raise ValueError("prototype expects one group64 and [tokens,64] activations")
    lo, hi = float(weights.min()), float(max(weights.max(), 0.0))
    candidates = np.linspace(.5, 1.0, grid)
    scores = []
    for ratio in candidates:
        quantized = affine4(weights, lo * ratio, hi * ratio)
        delta = activations @ (weights - quantized)
        scores.append(float(np.mean(delta * delta)))
    index = int(np.argmin(scores))
    baseline = affine4(weights)
    return {
        "grid": candidates.tolist(), "scores": scores, "best_ratio": float(candidates[index]),
        "baseline_output_mse": float(np.mean((activations @ (weights - baseline)) ** 2)),
        "best_output_mse": scores[index],
        "relative_reduction": float(1 - scores[index] / scores[-1]) if scores[-1] else 0.0,
    }


def main() -> None:
    rng = np.random.default_rng(42)
    weights = rng.normal(0, .2, 64).astype(np.float32)
    weights[0] = 2.0  # low-frequency outlier
    activations = rng.normal(0, .4, (512, 64)).astype(np.float32)
    activations[:, 0] *= .02  # outlier channel is rarely active
    result = weighted_clip(weights, activations)
    result.update({"protocol": "q4-activation-weighted-clip-prototype-v1", "seed": 42,
                   "weights_written": False, "runtime_integrated": False,
                   "interpretation": "Synthetic one-group proof only; no full-model quality claim."})
    destination = Path("/Volumes/移动数据库/lubanllm/experiments/native-mixed-q4-20261015/q4-diagnosis/weighted-clip-prototype.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
