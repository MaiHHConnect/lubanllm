#!/usr/bin/env python3
"""Inspect actual config on the meta device: no weights or large allocation."""
import argparse
import json
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/qwen3-1.7b.json")
parser.add_argument("--output", default="research/architecture.json")
args = parser.parse_args()
config = Qwen3Config(**json.loads(Path(args.config).read_text()))
with torch.device("meta"):
    model = Qwen3ForCausalLM(config)
groups = {}
for name, p in model.named_parameters():
    group = ("embedding" if "embed_tokens" in name else
             "attention" if "self_attn" in name else
             "ffn" if ".mlp." in name else "normalization_or_head")
    groups[group] = groups.get(group, 0) + p.numel()
result = {"method": "actual Transformers Qwen3 modules on meta device; no pretrained weights",
          "parameters_unique": sum(p.numel() for p in model.parameters()),
          "groups": groups, "layers": config.num_hidden_layers,
          "new_expert_params_per_layer_width128": 3 * config.hidden_size * 128,
          "config": config.to_dict()}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))
