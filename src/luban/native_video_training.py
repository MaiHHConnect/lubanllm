"""Joint Qwen3.5 video QA on the resident owner and existing native Adam ABI.

Adjacent source frames fill the original two-frame temporal convolution. Each
4-patch temporal block has independent visual attention; timestamps and video
embeddings then enter one causal language sequence, never separate frame losses.
Preprocessing reuses the admitted PIL image processor at RGB32. This is not a
qualification of the optional torchvision video decoder/resizer.
"""
from __future__ import annotations

import base64
from pathlib import Path

import torch

from .native_image_training import VisionBF16
from .native_operation_training import make_wrapper as make_dynamic_wrapper

MAX_TOKENS = 1024
# The image estimate covers 64 tokens. Conservatively reserve BF16 full logits,
# FP32 CE input and its FP32 backward workspace for each extra language token.
EXTRA_AVAILABLE_BYTES = (MAX_TOKENS - 64) * 248320 * 10



class VideoVision(VisionBF16):
    def __init__(self, visual):
        torch.nn.Module.__init__(self)
        self.visual = visual

    def forward(self, pixels, indices, weights, positions):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision
        visual = self.visual
        hidden = visual.patch_embed(pixels)
        hidden = hidden + (visual.pos_embed(indices) * weights[:, :, None]).sum(0).to(hidden.dtype)
        rotary = visual.rotary_pos_emb(positions)
        angle = torch.cat((rotary, rotary), -1)
        cos, sin = angle.cos(), angle.sin()
        for block in visual.blocks:
            attention = block.attn
            q, k, v = attention.qkv(block.norm1(hidden)).reshape(
                -1, 3, attention.num_heads, attention.head_dim).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            # RGB32 yields four spatial patches per temporal block. Batching
            # blocks isolates attention exactly as official cu_seqlens splits.
            q, k, v = [value.reshape(-1, 4, attention.num_heads, attention.head_dim).transpose(1, 2)
                       for value in (q, k, v)]
            value = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=attention.scaling)
            hidden = hidden + attention.proj(value.transpose(1, 2).reshape(-1, visual.config.hidden_size))
            hidden = hidden + block.mlp(block.norm2(hidden))
        return visual.merger(hidden)


def make_wrapper(owner):
    wrapper = make_dynamic_wrapper(owner)
    wrapper.qa.vision = VideoVision(wrapper.qa.vision.visual)
    wrapper.qa.image_id = owner.config.video_token_id
    return wrapper


def _qa(tokenizer, record, stamps, maximum):
    for value in (record.get("question"), record.get("answer")):
        if (not isinstance(value, str) or not value.strip() or len(value.encode()) > 16384
                or any(mark in value for mark in ("<|", "|>", "<think>", "</think>"))):
            raise ValueError("请提供核对后的完整视频问题和答案，不能包含内部标记")
    if not 7 <= maximum <= MAX_TOKENS:
        raise ValueError("视频联合问答词元预算无效")
    marker = "<|vision_start|><|video_pad|><|vision_end|>"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": record["question"]}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if prompt.count(marker) != 1:
        raise ValueError("视频占位与完整问答不一致")
    # Upstream Qwen3VLProcessor uses the first/last-frame mean per temporal
    # patch, rendered to one decimal second, before each video placeholder.
    paired = stamps + stamps[-1:] if len(stamps) % 2 else stamps
    times = [(paired[i] + paired[i + 1]) / 2 for i in range(0, len(paired), 2)]
    prompt = prompt.replace(marker, "".join(f"<{stamp:.1f} seconds>" + marker for stamp in times))
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    gold = tokenizer.encode(record["answer"], add_special_tokens=False)
    if (not gold or tokenizer.eos_token_id != 248046 or 248046 in gold
            or tokenizer.decode(gold, clean_up_tokenization_spaces=False) != record["answer"]):
        raise ValueError("视频答案必须可完整分词并保留原 EOS")
    ids = torch.tensor([prefix + gold + [248046]])
    labels = ids.clone(); labels[:, :len(prefix)] = -100
    if (not 7 <= ids.shape[1] <= maximum or ids.min() < 0 or ids.max() >= 248320
            or int((ids == tokenizer.convert_tokens_to_ids("<|video_pad|>")).sum()) != len(times)):
        raise ValueError("完整视频问答超出本轮词元范围；不会截断答案")
    return ids, labels


def prepare_video_sample(owner, tokenizer, processor, record, max_tokens=MAX_TOKENS):
    from .vision import MAX_IMAGE_BYTES, open_image
    from .speech_learning import _read
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
    from .video_learning import frame_rows
    frames = frame_rows(record)
    images = []
    for frame in frames:
        path = Path(frame["asset_path"])
        mime = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(path.suffix.lower())
        if mime is None:
            raise ValueError("视频画面需要不超过4MiB的PNG、JPEG或WebP")
        raw = _read(path, MAX_IMAGE_BYTES)
        images.append(open_image("data:image/" + mime + ";base64," + base64.b64encode(raw).decode("ascii")))
    pixels = processor(images, return_tensors="pt", min_pixels=1024, max_pixels=1024)
    if (pixels["image_grid_thw"].tolist() != [[1, 2, 2]] * len(frames)
            or processor.patch_size != 16 or processor.temporal_patch_size != 2 or processor.merge_size != 2):
        raise ValueError("视频联合训练仅支持原2帧时间卷积和32×32画面网格")
    # The PIL image processor repeats one frame into both temporal slots.
    # Retain its normalization/spatial order, then place two distinct original
    # frames in those slots, exactly the official video patch layout.
    values = pixels["pixel_values"].reshape(len(frames), 4, 3, 2, 16, 16)[:, :, :, 0]
    if len(frames) % 2:
        values = torch.cat((values, values[-1:]), 0)
    values = values.reshape(-1, 2, 4, 3, 16, 16).permute(0, 2, 3, 1, 4, 5).reshape(-1, 1536)
    grid = torch.tensor([[values.shape[0] // 4, 2, 2]])
    ids, labels = _qa(tokenizer, record, [x["timestamp_seconds"] for x in frames], max_tokens)
    config = owner.config.vision_config
    indices, weights = get_vision_bilinear_indices_and_weights(grid, int(config.num_position_embeddings ** .5), config.spatial_merge_size)
    positions = owner.model.compute_3d_position_ids(input_ids=ids, inputs_embeds=None, video_grid_thw=grid,
                                                  mm_token_type_ids=(ids == owner.config.video_token_id).long() * 2)
    mask = torch.full((1, 1, ids.shape[1], ids.shape[1]), float("-inf"), dtype=torch.bfloat16).triu(1)
    return (values, indices, weights, get_vision_position_ids(grid, config.spatial_merge_size), ids, positions, mask), labels
