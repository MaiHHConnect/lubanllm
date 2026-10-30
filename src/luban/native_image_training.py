"""Resident-owner image QA preparation and no-Python-linkage LibTorch bridge.

The three FP32 candidate tensors are separate from the active BF16 owner.
Frozen graph arguments share that owner's storage through DLPack.
"""
from __future__ import annotations

import copy
import ctypes
import warnings
from pathlib import Path

import torch


class VisionBF16(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.visual = owner.model.visual

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
            # Official helper shares each FP32 cast across both rotary terms.
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            value = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0), scale=attention.scaling)
            hidden = hidden + attention.proj(value.transpose(1, 2).reshape(-1, visual.config.hidden_size))
            hidden = hidden + block.mlp(block.norm2(hidden))
        return visual.merger(hidden)


class ImageQA(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.vision = VisionBF16(owner)
        self.language_model = owner.model.language_model
        self.lm_head = owner.lm_head
        self.image_id = owner.config.image_token_id
        self.hidden = owner.config.text_config.hidden_size

    def forward(self, pixels, indices, weights, visual_positions, ids, positions, mask):
        merged = self.vision(pixels, indices, weights, visual_positions)
        embedding = self.language_model.embed_tokens(ids)
        embedding = embedding.masked_scatter(
            (ids == self.image_id).unsqueeze(-1).expand(-1, -1, self.hidden), merged)
        hidden = self.language_model(inputs_embeds=embedding, position_ids=positions,
                                     attention_mask=mask, use_cache=False).last_hidden_state
        return self.lm_head(hidden[:, 0:, :])


class BF16View(torch.nn.Module):
    def forward(self, master):
        return master.to(torch.bfloat16)


class MasterQA(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.qa = ImageQA(owner)
        last = owner.config.text_config.num_hidden_layers - 1
        mlp = f"language_model.layers.{last}.mlp"
        # Runtime's disabled DynamicMLP contributes exactly its original base.
        if hasattr(owner.model.language_model.layers[last].mlp, "base"):
            if owner.model.language_model.layers[last].mlp.expert is not None:
                raise ValueError("图片训练必须在整轮关闭已学分支后开始")
            mlp += ".base"
        self.graph_targets = ["vision.visual.patch_embed.proj.weight",
                              "vision.visual.merger.linear_fc2.weight", mlp + ".down_proj.weight"]
        self.owner_targets = ["model.visual.patch_embed.proj.weight",
                              "model.visual.merger.linear_fc2.weight", "model." + mlp + ".down_proj.weight"]
        if owner.training:
            raise ValueError("图片训练需要已有冻结 eval 模型")
        for key in self.owner_targets:
            target = owner.get_parameter(key)
            if target.dtype != torch.bfloat16 or target.device.type != "cpu" or target.is_meta or target.requires_grad:
                raise ValueError("图片训练需要原始 CPU/BF16 冻结目标：" + key)
        # Copy module containers, never frozen parameters or the full model.
        for key in self.graph_targets:
            current = self.qa
            parts = key.split(".")
            for part in parts[:-1]:
                child = copy.copy(current._modules[part])
                child._modules = dict(child._modules)
                child._parameters = dict(child._parameters)
                child._buffers = dict(child._buffers)
                current._modules[part] = child
                current = child
            original = current._parameters[parts[-1]]
            current._parameters[parts[-1]] = torch.nn.Parameter(original.detach().float())
            torch.nn.utils.parametrize.register_parametrization(current, parts[-1], BF16View(), unsafe=True)

    def masters(self):
        return [self.qa.get_parameter(key.rsplit(".", 1)[0] + ".parametrizations.weight.original")
                for key in self.graph_targets]

    def forward(self, pixels, indices, weights, visual_positions, ids, positions, mask):
        return self.qa(pixels, indices, weights, visual_positions, ids, positions, mask)


def original_qa(tokenizer, question, answer, image_tokens, *, max_tokens=64):
    for value in (question, answer):
        if (not isinstance(value, str) or not value.strip() or len(value.encode()) > 16384
                or any(mark in value for mark in ("<|", "|>", "<think>", "</think>"))):
            raise ValueError("请提供核对后的完整图片问题和答案，不能包含内部标记")
    if not 1 <= image_tokens <= 16 or not 7 <= max_tokens <= 1024:
        raise ValueError("图片或词元预算超出已验证范围")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if prompt.count("<|image_pad|>") != 1:
        raise ValueError("图片占位与问题不一致")
    prefix = tokenizer.encode(prompt.replace("<|image_pad|>", "<|image_pad|>" * image_tokens), add_special_tokens=False)
    gold = tokenizer.encode(answer, add_special_tokens=False)
    if (not gold or tokenizer.eos_token_id != 248046 or 248046 in gold
            or tokenizer.decode(gold, clean_up_tokenization_spaces=False) != answer):
        raise ValueError("答案必须可完整分词并保留原 EOS")
    ids = torch.tensor([prefix + gold + [248046]])
    labels = ids.clone()
    labels[:, :len(prefix)] = -100
    if (not 7 <= ids.shape[1] <= max_tokens or ids.min() < 0 or ids.max() >= 248320
            or int((ids == tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum()) != image_tokens):
        raise ValueError("完整图片问答超出本轮词元范围；不会截断答案")
    return ids, labels


def prepare_sample(owner, tokenizer, processor, record, *, max_tokens=64):
    from PIL import Image
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
    from .vision import open_image
    image = record.get("image")
    if image is None:
        import base64
        from .vision import MAX_IMAGE_BYTES
        path = Path(record["asset_path"])
        mime = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}.get(path.suffix.lower())
        if mime is None or path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("图片训练接不超过 4 MiB 的 PNG、JPEG、WebP")
        with path.open("rb") as stream:
            raw = stream.read(MAX_IMAGE_BYTES + 1)
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("单张图片不能超过 4 MiB")
        image = "data:image/" + mime + ";base64," + base64.b64encode(raw).decode("ascii")
    if isinstance(image, str):
        image = open_image(image)
    elif isinstance(image, dict):
        width, height, rgb = image.get("width"), image.get("height"), image.get("rgb")
        if (type(width) is not int or type(height) is not int or min(width, height) < 1
                or width * height > 16_000_000 or not isinstance(rgb, bytes) or len(rgb) != width * height * 3):
            raise ValueError("动作前 RGB 图片尺寸或字节不符")
        image = Image.frombytes("RGB", (width, height), rgb)
    elif not isinstance(image, Image.Image):
        raise ValueError("图片输入必须来自本地上传或已核对 RGB")
    # Current actual QA qualification is a single RGB32 image / merged token.
    pixel = processor([image], return_tensors="pt", min_pixels=1024, max_pixels=1024)
    grid = pixel["image_grid_thw"]
    if grid.tolist() != [[1, 2, 2]]:
        raise ValueError("当前图片训练仅支持已验证的单图 32×32 网格")
    ids, labels = original_qa(tokenizer, record["question"], record["answer"], 1, max_tokens=max_tokens)
    position = owner.model.compute_3d_position_ids(
        input_ids=ids, inputs_embeds=None, image_grid_thw=grid,
        mm_token_type_ids=(ids == owner.config.image_token_id).long())
    visual = owner.config.vision_config
    side = int(visual.num_position_embeddings ** .5)
    indices, weights = get_vision_bilinear_indices_and_weights(grid, side, visual.spatial_merge_size)
    values = (pixel["pixel_values"], indices, weights, get_vision_position_ids(grid, visual.spatial_merge_size),
              ids, position, torch.full((1, 1, ids.shape[1], ids.shape[1]), float("-inf"), dtype=torch.bfloat16).triu(1))
    return values, labels


def lower_graph(wrapper, inputs, output):
    """One trace per owner lifecycle, no weights in the graph archive."""
    with torch.jit.optimized_execution(False), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(wrapper, inputs, strict=True, check_trace=False)
        graph = traced.inlined_graph.copy()
        torch._C._jit_pass_inline(graph)
        # This structural pass resolves parametrization module attributes; the
        # installed JIT leaves None-return methods when lowering is disabled.
        # The weight-free function and native GraphExecutor still first run
        # with optimization disabled for the complete training lifecycle.
        with torch.jit.optimized_execution(True):
            graph, parameters = torch._C._jit_pass_lower_graph(graph, traced._c)
        torch._C._jit_pass_dce(graph)
        for node in list(graph.nodes()):
            if node.kind() == "prim::Constant" and node.hasAttribute("value") and node.kindOf("value") == "t":
                parameters.append(node.t("value"))
                replacement = graph.addInput()
                replacement.copyMetadata(node.output())
                node.output().replaceAllUsesWith(replacement)
                node.destroy()
        if any(word in str(graph) for word in ("prim::PythonOp", "prim::GetAttr")):
            raise ValueError("图片训练图包含非原生执行节点")
        torch._C._create_function_from_graph("forward", graph).save(str(output))
    masters = {tensor.data_ptr() for tensor in wrapper.masters()}
    selected = [i for i, tensor in enumerate(parameters) if tensor.data_ptr() in masters]
    if len(selected) != 3:
        raise ValueError("图片训练图必须恰好绑定三个 FP32 master")
    return parameters, selected


def library_path():
    root = Path(__file__).resolve().parents[2]
    for path in (root / "native/libluban_image_training.dylib", root / "build/native/libluban_image_training.dylib"):
        if path.is_file() and not path.is_symlink():
            return path
    raise RuntimeError("原生图片训练组件尚未安装；不会退回文字训练或 Python 优化器")


class _SharedTensor(ctypes.Structure):
    _fields_ = [("tensor", ctypes.c_void_p), ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("rank", ctypes.c_int64)]


class NativeEngine:
    def __init__(self, graph, parameters, master_indices, max_tokens=64, cancel=None):
        self.handle = None
        self.lib = ctypes.CDLL(str(library_path()))
        pointer = ctypes.c_void_p
        signatures = {
            "create": ([ctypes.c_char_p, ctypes.c_int64], pointer), "close": ([pointer], None),
            "bind": ([pointer, pointer, ctypes.c_int], ctypes.c_int),
            "step": ([pointer, ctypes.POINTER(_SharedTensor), pointer, ctypes.c_double, ctypes.POINTER(ctypes.c_double)], ctypes.c_int),
            "greedy": ([pointer, ctypes.POINTER(_SharedTensor), ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), pointer], ctypes.c_int),
            "save": ([pointer, ctypes.c_char_p, ctypes.c_char_p], ctypes.c_int),
            "restore": ([pointer, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64], ctypes.c_int),
            "error": ([], ctypes.c_char_p)}
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, "luban_image_" + name)
            function.argtypes, function.restype = arguments, result
        self.maximum = max_tokens
        self._cancel = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: int(bool(cancel and cancel())))
        self.handle = self.lib.luban_image_create(str(graph).encode(), max_tokens)
        self._check(0 if self.handle else -1)
        try:
            for i, tensor in enumerate(parameters):
                self._check(self.lib.luban_image_bind(self.handle, ctypes.byref(self._dlpack(tensor)), i in master_indices))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _dlpack(tensor):
        if tensor.device.type != "cpu" or tensor.is_meta:
            raise ValueError("原生图片训练只接已有 CPU Tensor")
        capsule = torch.utils.dlpack.to_dlpack(tensor)
        get = ctypes.pythonapi.PyCapsule_GetPointer
        get.argtypes, get.restype = [ctypes.py_object, ctypes.c_char_p], ctypes.c_void_p
        rename = ctypes.pythonapi.PyCapsule_SetName
        rename.argtypes, rename.restype = [ctypes.py_object, ctypes.c_char_p], ctypes.c_int
        pointer = get(capsule, b"dltensor")
        if rename(capsule, b"used_dltensor") != 0:
            raise RuntimeError("DLPack storage transfer failed")
        strides = (ctypes.c_int64 * tensor.dim())(*tensor.stride())
        return _SharedTensor(pointer, strides, tensor.dim())

    def _check(self, code):
        if code != 0:
            raise RuntimeError(self.lib.luban_image_error().decode())

    def _inputs(self, values):
        if len(values) != 7 or any(t.device.type != "cpu" for t in values):
            raise ValueError("完整七项驻留图输入无效")
        return (_SharedTensor * 7)(*(self._dlpack(t) for t in values))

    def step(self, values, labels, learning_rate):
        loss = ctypes.c_double()
        self._check(self.lib.luban_image_step(self.handle, self._inputs(values), ctypes.byref(self._dlpack(labels)),
                                              learning_rate, ctypes.byref(loss)))
        return loss.value

    def greedy(self, values, prefix):
        tokens, count = (ctypes.c_int64 * self.maximum)(), ctypes.c_int64()
        self._check(self.lib.luban_image_greedy(self.handle, self._inputs(values), prefix, tokens, ctypes.byref(count), self._cancel))
        return list(tokens[:count.value])

    def save(self, path, identity):
        self._check(self.lib.luban_image_save(self.handle, str(path).encode(), identity.encode()))

    def restore(self, path, identity, step):
        self._check(self.lib.luban_image_restore(self.handle, str(path).encode(), identity.encode(), step))

    def close(self):
        if self.handle:
            self.lib.luban_image_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        self.close()
