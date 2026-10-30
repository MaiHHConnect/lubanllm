# Original chunk computation: Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Changes: TorchScript types, uncached/unpacked guard, Tensor-only script return.
"""Complete screen/action sequences on the shared image owner; no second model.

Original differentiable DeltaNet loop, kept dynamic by TorchScript. The 1024
bound is an input limit: saved-small-owner full graph qualification is 204–388.
"""
import copy
from typing import Optional
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
def scripted_original_chunk(query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor, chunk_size: int=64, initial_state: Optional[Tensor]=None, output_final_state: bool=False, use_qk_l2norm_in_kernel: bool=False, cu_seqlens: Optional[Tensor]=None):
    assert initial_state is None and (not output_final_state), 'Full uncached training sequence required'
    assert cu_seqlens is None, 'Unpacked original sequence required'
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-06)
        key = l2norm(key, dim=-1, eps=1e-06)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float().tril()
    attn = -(k_beta @ key.transpose(-1, -2) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device) if initial_state is None else initial_state.to(value)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = (query[:, :, i], key[:, :, i], value[:, :, i])
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = q_i * g[:, :, i, :, None].exp() @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = last_recurrent_state * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out


# Compile a real loop once. No generated source or model writes at runtime.
_scripted_chunk = torch.jit.script(scripted_original_chunk)


def _uncached_chunk(query, key, value, g, beta, initial_state=None, output_final_state=False,
                    use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    return _scripted_chunk(query, key, value, g, beta, 64, initial_state, output_final_state,
                           use_qk_l2norm_in_kernel, cu_seqlens), None


def make_wrapper(owner):
    from .native_image_training import MasterQA
    wrapper = MasterQA(owner)
    # The image wrapper owns copied containers; tensors remain the admitted
    # owner's storage. Copy only attention containers whose function changes.
    layers = wrapper.qa.language_model.layers
    for index, original in enumerate(layers):
        if hasattr(original, "linear_attn"):
            layer = copy.copy(original)
            layer._modules = dict(original._modules)
            attention = copy.copy(original.linear_attn)
            attention.chunk_gated_delta_rule = _uncached_chunk
            layer._modules["linear_attn"] = attention
            layers[index] = layer
    return wrapper


def prepare_operation(owner, tokenizer, processor, record):
    from .native_image_training import prepare_sample
    # Same image processor/Qwen tokenizer, complete history + action JSON/EOS;
    # this deliberately raises when the full sequence cannot fit.
    return prepare_sample(owner, tokenizer, processor, record, max_tokens=1024)
