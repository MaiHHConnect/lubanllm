"""Explicit HF-only carrier for an unmerged FP32 last-down residual.

The caller must independently verify the loaded original base identity/core.
Checking a caller's declaration against file metadata does not verify weights.
The artifact stores only delta and its arithmetic contract, not a model package,
optimizer, adoption receipt, or AutoModel.from_pretrained implementation.
"""
from pathlib import Path
import json
import os
import re
import tempfile

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save

from .backbone import text_backbone
from .core_update import apply_core_residual


FORMAT = "luban-unmerged-fp32-down-output-residual-v1"
MATH = "bf16(fp32(base(x))+linear(fp32(x),delta_f32))"


def _identity(base_identity, base_core):
    if not isinstance(base_identity, str) or re.fullmatch(r"[0-9a-f]{64}", base_identity) is None:
        raise ValueError("base_identity must be a caller-verified original base identity")
    if base_core is not None:
        raise ValueError("Only a caller-verified original None core is supported")


def _base(projection):
    if (not isinstance(projection, nn.Linear) or projection.weight.is_meta
            or projection.weight.dtype != torch.bfloat16
            or tuple(projection.weight.shape) != (projection.out_features, projection.in_features)):
        raise ValueError("Residual base must be a materialized BF16 Linear projection")
    for value in (projection.weight, projection.bias):
        if value is not None and (value.dtype != torch.bfloat16
                or value.device != projection.weight.device or not bool(torch.isfinite(value).all())):
            raise ValueError("Residual base must have finite compatible BF16 weights and bias")


def _metadata(projection, base_identity, base_core):
    _identity(base_identity, base_core)
    return {"format": FORMAT, "base_identity": base_identity, "base_core": "None",
            "base_dtype": "BF16", "delta_dtype": "F32", "target": "last_mlp.down_proj",
            "shape": json.dumps(list(projection.weight.shape)), "math": MATH}


class ResidualProjection(nn.Module):
    """Keep the exact base object; persist an owned FP32 inference buffer.

    Base parameter flags/modes are retained. This carrier is not a training or
    provenance verifier; no optimizer should update the frozen original base.
    """

    def __init__(self, base, delta, *, base_identity, base_core):
        super().__init__()
        _base(base)
        self.binding = _metadata(base, base_identity, base_core)
        if (not isinstance(delta, torch.Tensor) or delta.is_meta
                or delta.dtype != torch.float32 or delta.shape != base.weight.shape
                or not bool(torch.isfinite(delta).all())):
            raise ValueError("delta must be a finite materialized FP32 matrix matching the base")
        self.base = base
        self.register_buffer("delta", delta.detach().to(device=base.weight.device).clone())
        self.training = base.training

    def _apply(self, fn, recurse=True):
        # A parent model.bfloat16()/to(device) must not round the FP32 delta.
        probe = fn(torch.empty(0, device=self.base.weight.device, dtype=torch.bfloat16))
        if probe.dtype != torch.bfloat16 or probe.is_meta:
            raise ValueError("Residual carrier requires a materialized BF16 base")
        delta = self._buffers.pop("delta")
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            self._buffers["delta"] = delta.to(device=self.base.weight.device)

    def forward(self, inputs):
        return apply_core_residual(inputs, self.base(inputs), self.delta)


def _slot(model):
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_5":
        raise ValueError("Only the loaded Qwen3.5 graph is supported")
    decoder, config = text_backbone(model)
    mlp = decoder.layers[-1].mlp
    if getattr(mlp, "expert", None) is not None:
        raise ValueError("The residual requires an original base without an active expert")
    mlp = getattr(mlp, "base", mlp)
    projection = mlp.down_proj
    base = projection.base if isinstance(projection, ResidualProjection) else projection
    if (not isinstance(base, nn.Linear) or base.in_features != config.intermediate_size
            or base.out_features != config.hidden_size):
        raise ValueError("Last down projection differs from the Qwen3.5 graph")
    return mlp, projection


def install_residual_core(model, delta, *, base_identity, base_core):
    """Install on an already loaded, independently verified original None base."""
    _identity(base_identity, base_core)
    mlp, projection = _slot(model)
    if isinstance(projection, ResidualProjection):
        raise ValueError("A residual core is already installed")
    residual = ResidualProjection(projection, delta, base_identity=base_identity, base_core=base_core)
    mlp.down_proj = residual
    return residual


def get_residual_core(model):
    _, projection = _slot(model)
    if not isinstance(projection, ResidualProjection):
        raise ValueError("No residual core is installed")
    return projection


def uninstall_residual_core(model):
    mlp, _ = _slot(model)
    residual = get_residual_core(model)
    mlp.down_proj = residual.base
    return residual.base


def _publish_new_file(temporary, path):
    # Reserve exclusively, then atomically replace our empty reservation.
    # Before commit a reader may see an invalid empty file, never partial weights.
    with path.open("xb") as reservation:
        reserved = os.fstat(reservation.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        try:
            current = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if os.path.samestat(reserved, current):
                path.unlink()
        raise


def save_residual_core(model, path):
    """Create a delta-only safetensors file; never overwrite an existing file."""
    residual = get_residual_core(model)
    _base(residual.base)
    if (residual.delta.dtype != torch.float32 or residual.delta.shape != residual.base.weight.shape
            or not bool(torch.isfinite(residual.delta).all())):
        raise ValueError("Installed delta no longer satisfies its FP32 contract")
    expected = _metadata(residual.base, residual.binding.get("base_identity"), None)
    if residual.binding != expected:
        raise ValueError("Installed residual binding changed")
    payload = save({"delta": residual.delta.detach().cpu().contiguous()}, metadata=expected)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".residual-", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
        _publish_new_file(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_residual_core(model, path, *, base_identity, base_core):
    """Validate the declaration, load delta, and install for direct forward/generate.

    The caller supplies the independently verified original loaded model. A
    correct metadata declaration cannot detect undeclared base weight changes.
    """
    mlp, projection = _slot(model)
    if isinstance(projection, ResidualProjection):
        raise ValueError("A residual core is already installed")
    _base(projection)
    expected = _metadata(projection, base_identity, base_core)
    with safe_open(str(path), framework="pt", device="cpu") as source:
        if source.metadata() != expected or list(source.keys()) != ["delta"]:
            raise ValueError("Residual artifact binding or tensor keys differ")
        specification = source.get_slice("delta")
        if (specification.get_shape() != list(projection.weight.shape)
                or specification.get_dtype() != "F32"):
            raise ValueError("Residual artifact tensor shape or dtype differs")
        delta = source.get_tensor("delta")
    return install_residual_core(model, delta, base_identity=base_identity, base_core=base_core)
