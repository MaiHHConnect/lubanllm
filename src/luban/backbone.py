"""Locate supported upstream decoder graphs without changing their attention."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, GenerationConfig


def text_backbone(model):
    kind = model.config.model_type
    if kind == "qwen3":
        return model.model, model.config
    if kind == "qwen3_5":
        return model.model.language_model, model.config.text_config
    raise ValueError("Only Qwen3 and Qwen3.5 decoder graphs are supported")


def load_backbone(path, dtype, device, *, cold=False):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    loader = AutoModelForMultimodalLM if config.model_type == "qwen3_5" else AutoModelForCausalLM
    if cold and config.model_type == "qwen3_5":
        # Build the official graph without allocating pretrained weight storage.
        with torch.device("meta"):
            model = loader.from_config(config, dtype=dtype, trust_remote_code=False).to(dtype=dtype)
        decoder, _ = text_backbone(model)
        # These nonpersistent buffers are computed by the official constructors,
        # not stored in the checkpoint. Keep their original FP32 precision.
        decoder.rotary_emb = type(decoder.rotary_emb)(config.text_config, device="cpu").to(device)
        visual = model.model.visual
        rotary = visual.rotary_pos_emb
        visual.rotary_pos_emb = type(rotary)(rotary.dim, rotary.theta)
        if (Path(path) / "generation_config.json").is_file():
            model.generation_config = GenerationConfig.from_pretrained(str(path), local_files_only=True)
        return model.eval().requires_grad_(False)
    return loader.from_pretrained(str(path), dtype=dtype, local_files_only=True,
                                  trust_remote_code=False).to(device).eval().requires_grad_(False)
