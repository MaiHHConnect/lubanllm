#!/usr/bin/env python3
"""Collect read-only BF16 activation statistics for a future Q4 calibration.

This does not quantize, write model weights, train, or claim semantic causality.
It records input-channel energy for MLP projections so a later clip search can
weight groups by the inputs seen on mixed long/code/calculation prompts.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from luban.backbone import load_backbone, text_backbone


PROMPTS = [
    "请解释电脑运行内存与硬盘的区别，并给出三个日常例子。",
    "Write a small Python function that validates a JSON object and explain its edge cases.",
    "计算 17*19、12345/15，并说明每一步，最后只给出结果表。",
    "请分析这段代码的时间复杂度：for i in range(n): for j in range(i): print(i*j)。",
    "量化模型在长上下文中可能如何影响注意力、记忆和最终词元选择？请分点回答。",
    "Given a cache with exact prefix matching, design tests for cancellation, branching, and EOS handling.",
    ("请阅读以下约束并给出不超过五条执行步骤：" +
     "保持原权重不变，记录硬件、精度、种子和耗时；先验证数值，再验证质量；" * 12),
    ("Explain why a low-frequency outlier in a shared group can increase quantization error. " * 12),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--emit-group-profile", action="store_true", help="Also emit one activation-weighted clipping ratio per input group.")
    parser.add_argument("--group-size", type=int, choices=[16, 32, 64], default=32)
    parser.add_argument("--emit-hessian-group-profile", action="store_true", help="Use local per-group X^T X cross terms when choosing group clipping ratios.")
    args = parser.parse_args()
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    start = time.perf_counter()
    model = load_backbone(args.model, torch.bfloat16, device)
    decoder, _ = text_backbone(model)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
    sums: dict[str, torch.Tensor] = {}
    covariances: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    modules_by_name = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            value = inputs[0]
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                return
            flat = value.detach().float().reshape(-1, value.shape[-1])
            # Aggregate on the accelerator, then release each prompt's activation.
            energy = flat.square().sum(dim=0).to("cpu")
            sums[name] = energy if name not in sums else sums[name] + energy
            if args.emit_hessian_group_profile:
                if flat.shape[1] % args.group_size:
                    raise ValueError(f"activation width is not divisible by group size: {name} {flat.shape}")
                grouped = flat.reshape(flat.shape[0], -1, args.group_size)
                covariance = torch.einsum("tgi,tgj->gij", grouped, grouped).to("cpu")
                covariances[name] = covariance if name not in covariances else covariances[name] + covariance
            counts[name] = counts.get(name, 0) + int(flat.shape[0])
        return hook

    for name, module in decoder.named_modules():
        if name.endswith(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
            modules_by_name[name] = module
            hooks.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
                decoder(ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if device == "mps":
            torch.mps.empty_cache()

    def bf16_round(value: np.ndarray) -> np.ndarray:
        bits = value.astype(np.float32, copy=False).view(np.uint32)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return (rounded.astype(np.uint32) << 16).view(np.float32)

    def weighted_clip_ratio(weight: np.ndarray, energy: np.ndarray) -> tuple[float, float, float]:
        if weight.shape[1] % 64 or energy.size != weight.shape[1]:
            raise ValueError(f"unsupported group shape: {weight.shape}, {energy.shape}")
        rows = np.linspace(0, weight.shape[0] - 1, min(64, weight.shape[0]), dtype=np.int64)
        sampled = weight[rows].reshape(-1, 64)
        group_energy = energy.reshape(-1, 64)
        ratios = np.linspace(.5, 1.0, 21, dtype=np.float32)
        scores = []
        for ratio in ratios:
            lower = sampled.min(axis=1) * ratio
            upper = sampled.max(axis=1) * ratio
            clipped = np.clip(sampled, lower[:, None], upper[:, None])
            bias = np.where(np.abs(lower) > np.abs(upper), lower, upper)
            scale = np.maximum((upper - lower) / 15.0, 1e-7)
            scale = np.where(bias >= 0, -scale, scale)
            zero = np.rint(bias / scale)
            nonzero = zero != 0
            scale = np.where(nonzero, bias / np.where(nonzero, zero, 1), scale)
            bias = np.where(nonzero, bias, 0.0)
            codes = np.clip(np.rint((clipped - bias[:, None]) / scale[:, None]), 0, 15)
            dequant = bf16_round(scale)[:, None] * codes + bf16_round(bias)[:, None]
            # Weight by the observed input-channel energy, preserving the
            # group structure used by the Metal kernel.
            score = float((((sampled - dequant) ** 2) * np.tile(energy, len(rows)).reshape(sampled.shape)).sum())
            scores.append(score)
        best = int(np.argmin(scores))
        return float(ratios[best]), float(scores[best]), float(scores[-1])

    def weighted_group_clip_ratios(weight: np.ndarray, energy: np.ndarray, group_size: int) -> list[float]:
        if weight.shape[1] % group_size or energy.size != weight.shape[1]:
            raise ValueError(f"unsupported group shape: {weight.shape}, {energy.shape}")
        rows = np.linspace(0, weight.shape[0] - 1, min(64, weight.shape[0]), dtype=np.int64)
        sampled = weight[rows]
        group_count = weight.shape[1] // group_size
        ratios = np.linspace(.5, 1.0, 21, dtype=np.float32)
        result = []
        for group in range(group_count):
            local = sampled[:, group * group_size:(group + 1) * group_size]
            local_energy = energy[group * group_size:(group + 1) * group_size]
            scores = []
            for ratio in ratios:
                lower = local.min(axis=1) * ratio
                upper = local.max(axis=1) * ratio
                clipped = np.clip(local, lower[:, None], upper[:, None])
                bias = np.where(np.abs(lower) > np.abs(upper), lower, upper)
                scale = np.maximum((upper - lower) / 15.0, 1e-7)
                scale = np.where(bias >= 0, -scale, scale)
                zero = np.rint(bias / scale)
                nonzero = zero != 0
                scale = np.where(nonzero, bias / np.where(nonzero, zero, 1), scale)
                bias = np.where(nonzero, bias, 0.0)
                codes = np.clip(np.rint((clipped - bias[:, None]) / scale[:, None]), 0, 15)
                dequant = bf16_round(scale)[:, None] * codes + bf16_round(bias)[:, None]
                scores.append(float((((local - dequant) ** 2) * local_energy[None, :]).sum()))
            result.append(float(ratios[int(np.argmin(scores))]))
        return result

    def hessian_group_clip_ratios(weight: np.ndarray, covariance: np.ndarray, group_size: int) -> list[float]:
        if weight.shape[1] % group_size or covariance.shape != (weight.shape[1] // group_size, group_size, group_size):
            raise ValueError(f"unsupported Hessian shape: {weight.shape}, {covariance.shape}")
        rows = np.linspace(0, weight.shape[0] - 1, min(64, weight.shape[0]), dtype=np.int64)
        sampled = weight[rows]
        ratios = np.linspace(.5, 1.0, 21, dtype=np.float32)
        result = []
        for group, hessian in enumerate(covariance):
            local = sampled[:, group * group_size:(group + 1) * group_size]
            # Keep the quadratic score finite even if a calibration group is
            # nearly constant; the positive diagonal is a numerical floor.
            hessian = hessian.astype(np.float64, copy=False)
            hessian = hessian + np.eye(group_size, dtype=np.float64) * max(1e-8, float(np.trace(hessian)) * 1e-8)
            scores = []
            for ratio in ratios:
                lower = local.min(axis=1) * ratio
                upper = local.max(axis=1) * ratio
                clipped = np.clip(local, lower[:, None], upper[:, None])
                bias = np.where(np.abs(lower) > np.abs(upper), lower, upper)
                scale = np.maximum((upper - lower) / 15.0, 1e-7)
                scale = np.where(bias >= 0, -scale, scale)
                zero = np.rint(bias / scale)
                nonzero = zero != 0
                scale = np.where(nonzero, bias / np.where(nonzero, zero, 1), scale)
                bias = np.where(nonzero, bias, 0.0)
                codes = np.clip(np.rint((clipped - bias[:, None]) / scale[:, None]), 0, 15)
                dequant = bf16_round(scale)[:, None] * codes + bf16_round(bias)[:, None]
                delta = (local - dequant).astype(np.float64, copy=False)
                scores.append(float(np.einsum("ri,ij,rj->", delta, hessian, delta, optimize=True)))
            result.append(float(ratios[int(np.argmin(scores))]))
        return result

    records = []
    profile_ratios = []
    group_profile_ratios = []
    hessian_group_profile_ratios = []
    def module_order(name: str) -> tuple[int, int]:
        parts = name.split(".")
        projection = parts[-1]
        return int(parts[1]), {"gate_proj": 0, "up_proj": 1, "down_proj": 2}[projection]

    for name in sorted(sums, key=module_order):
        energy = sums[name].numpy()
        groups = energy.reshape(-1, 64) if energy.size % 64 == 0 else energy[: energy.size - energy.size % 64].reshape(-1, 64)
        group_energy = groups.sum(axis=1)
        module = modules_by_name[name]
        weight = module.weight.detach().float().cpu().numpy()
        ratio, weighted_score, unclip_score = weighted_clip_ratio(weight, energy)
        if name != "layers.23.mlp.down_proj":
            profile_ratios.append(ratio)
            if args.emit_group_profile:
                group_profile_ratios.extend(weighted_group_clip_ratios(weight, energy, args.group_size))
            if args.emit_hessian_group_profile:
                hessian_group_profile_ratios.extend(hessian_group_clip_ratios(weight, covariances[name].numpy(), args.group_size))
        records.append({
            "module": name,
            "channels": int(energy.size),
            "tokens": counts[name],
            "mean_channel_squared": float(energy.sum() / max(1, counts[name] * energy.size)),
            "group_count": int(group_energy.size),
            "group_energy_p50": float(__import__("numpy").quantile(group_energy, .5)),
            "group_energy_p95": float(__import__("numpy").quantile(group_energy, .95)),
            "group_energy_max": float(group_energy.max()),
            "top_group_indices": [int(i) for i in group_energy.argsort()[-8:][::-1]],
            "calibrated_clip_ratio": ratio,
            "weighted_sample_mse": weighted_score,
            "weighted_unclipped_mse": unclip_score,
            "calibration_rows": min(64, int(weight.shape[0])),
        })
    report = {
        "protocol": "q4-bf16-activation-energy-v2" if args.emit_hessian_group_profile else "q4-bf16-activation-energy-v1",
        "model": str(args.model), "device": device, "dtype": "bfloat16",
        "prompts": PROMPTS, "prompt_count": len(PROMPTS),
        "modules": len(records), "records": records,
        "elapsed_seconds": time.perf_counter() - start,
        "hardware": platform.platform(), "torch_version": torch.__version__,
        "training": False, "weights_written": False, "quantization_run": False,
        "interpretation": "Activation energy drives a projection-level weighted clip search. The emitted sidecar is a calibration proxy and requires an independent native Q4 quality gate.",
        "limitations": ["Base BF16 graph without product core override; activation distribution is a calibration proxy.", "Energy is aggregated per input group and does not include output-row cross terms.", "Clip search samples up to 64 output rows per projection and uses a coarse 0.5..1.0 ratio grid."],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "activation-energy.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    profile_path = args.output / "q4-clip-profile.txt"
    if profile_path.exists():
        raise FileExistsError(profile_path)
    profile_path.write_text("LUBAN_Q4_CLIP_V1\nmixed-q4-mlp\n71\n" + "\n".join(f"{ratio:.8f}" for ratio in profile_ratios) + "\n")
    report["clip_profile"] = str(profile_path)
    if args.emit_group_profile:
        group_profile_path = args.output / f"q4-group{args.group_size}-clip-profile.txt"
        if group_profile_path.exists():
            raise FileExistsError(group_profile_path)
        group_profile_path.write_text("LUBAN_Q4_GROUP_CLIP_V1\nmixed-q4-g32-group\n" + str(len(group_profile_ratios)) + "\n" + "\n".join(f"{ratio:.8f}" for ratio in group_profile_ratios) + "\n")
        report["group_clip_profile"] = str(group_profile_path)
    if args.emit_hessian_group_profile:
        hessian_profile_path = args.output / f"q4-group{args.group_size}-hessian-clip-profile.txt"
        if hessian_profile_path.exists():
            raise FileExistsError(hessian_profile_path)
        hessian_profile_path.write_text("LUBAN_Q4_GROUP_CLIP_V1\nmixed-q4-g32-group\n" + str(len(hessian_group_profile_ratios)) + "\n" + "\n".join(f"{ratio:.8f}" for ratio in hessian_group_profile_ratios) + "\n")
        report["hessian_group_clip_profile"] = str(hessian_profile_path)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "profile": str(profile_path), "modules": len(records), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
