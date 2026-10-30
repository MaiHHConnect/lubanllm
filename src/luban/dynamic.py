"""Zero-initialized in-network experts with explicit disk residency.

Switch only between completed forward/backward/optimizer steps. Optimizers must
be released before evicting: optimizer references otherwise keep tensors alive.
This prototype intentionally supports one resident expert per insertion point.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from .backbone import text_backbone


class NeuralExpert(nn.Module):
    """A new SwiGLU neural branch, not a textual retrieval memory."""

    def __init__(self, hidden: int, width: int):
        super().__init__()
        self.gate = nn.Linear(hidden, width, bias=False)
        self.up = nn.Linear(hidden, width, bias=False)
        self.down = nn.Linear(width, hidden, bias=False)
        nn.init.zeros_(self.down.weight)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DynamicMLP(nn.Module):
    """Preserve a pretrained MLP and add a bounded resident neural branch.

    Disk records are bound to a base revision + layer + shape. The caller chooses
    the expert in v0; learned routing and automatic growth are future experiments.
    A CPU checkpoint copy is transient and is NOT counted as GPU residency.
    """

    def __init__(self, base: nn.Module, hidden: int, width: int, store: Path,
                 identity: str, max_expert_params: int):
        super().__init__()
        if hidden <= 0 or width <= 0:
            raise ValueError("hidden and width must be positive")
        if 3 * hidden * width > max_expert_params:
            raise ValueError("expert exceeds parameter budget")
        self.base = base.requires_grad_(False)
        self.hidden, self.width = hidden, width
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.expert: NeuralExpert | None = None
        self.expert_id: str | None = None
        self.requires_gate = False
        self.core_version = None
        self.preservation_rows = []
        self.train(base.training)

    def _paths(self, expert_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", expert_id):
            raise ValueError("unsafe expert id")
        return self.store / f"{expert_id}.safetensors", self.store / f"{expert_id}.json"

    def activate(self, expert_id: str, *, create: bool = False, device=None):
        if self.expert is not None:
            raise RuntimeError("hibernate current expert before switching")
        weights, meta = self._paths(expert_id)
        if create and (weights.exists() or meta.exists()):
            raise FileExistsError(expert_id)
        if not create and not (weights.exists() and meta.exists()):
            raise FileNotFoundError(expert_id)
        module = NeuralExpert(self.hidden, self.width)
        if not create:
            record = json.loads(meta.read_text())
            if not isinstance(record.get("requires_gate", False), bool):
                raise ValueError("invalid expert gate flag")
            expected = (self.identity, self.hidden, self.width)
            actual = (record["identity"], record["hidden"], record["width"])
            if actual != expected:
                raise ValueError("incompatible base revision, layer, or expert shape")
            if hashlib.sha256(weights.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            preservation_rows = record.get("preservation_rows", [])
            if not isinstance(preservation_rows, list) or len(preservation_rows) > 64:
                raise ValueError("专家保护材料元数据无效")
            module.load_state_dict(load_file(str(weights)))
        reference = next(self.base.parameters())
        module.to(device=reference.device if device is None else device, dtype=reference.dtype)
        module.train(self.training)
        self.expert, self.expert_id = module, expert_id
        self.requires_gate = record.get("requires_gate", False) if not create else False
        self.core_version = record.get("core_version") if not create else None
        self.preservation_rows = preservation_rows if not create else []
        return module

    def hibernate(self):
        if self.expert is None:
            return
        weights, meta = self._paths(self.expert_id)
        temporary = weights.with_name(weights.name + ".partial")
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.expert.state_dict().items()}
        save_file(tensors, str(temporary))
        record = {"identity": self.identity, "hidden": self.hidden, "width": self.width,
                  "sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(), "requires_gate": self.requires_gate,
                  "core_version": self.core_version, "preservation_rows": self.preservation_rows}
        meta_tmp = meta.with_name(meta.name + ".partial")
        meta_tmp.write_text(json.dumps(record, indent=2) + "\n")
        # Individually atomic. Interrupted pairs fail closed on checksum validation.
        os.replace(temporary, weights)
        os.replace(meta_tmp, meta)
        self.expert = None
        self.expert_id = None

    @property
    def resident_expert_params(self):
        return 0 if self.expert is None else sum(p.numel() for p in self.expert.parameters())

    def forward(self, x):
        y = self.base(x)
        return y if self.expert is None else y + self.expert(x)


def insert_experts(model, layers, width: int, store: Path, revision: str,
                   max_resident_params: int = 10_000_000_000):
    """Insert branches into supported decoder MLPs, preserving attention."""
    layers = list(layers)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("insertion layers must be nonempty and unique")
    decoder, config = text_backbone(model)
    if any(i < 0 or i >= len(decoder.layers) for i in layers):
        raise ValueError("insertion layer out of range")
    if any(isinstance(decoder.layers[i].mlp, DynamicMLP) for i in layers):
        raise ValueError("layer already modified")
    hidden = config.hidden_size
    base_count = sum(p.numel() for p in model.parameters())
    if base_count + len(layers) * 3 * hidden * width > max_resident_params:
        raise ValueError("combined resident parameters exceed budget")
    model.requires_grad_(False)
    branches = []
    for i in layers:
        branch = DynamicMLP(decoder.layers[i].mlp, hidden, width,
                            Path(store) / f"layer-{i}", f"{revision}:layer-{i}",
                            3 * hidden * width)
        decoder.layers[i].mlp = branch
        branches.append(branch)
    return branches
