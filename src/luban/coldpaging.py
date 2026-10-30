"""Keep the four-layer route on device; reload frozen text layers on demand.

Startup materializes only resident text weights in the official empty graph.
Image requests restore the upstream visual graph in vision.py.
Callers serialize tasks with Runtime.lock and release optimizers before exit.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import gc
import json
from pathlib import Path
import time

from safetensors import safe_open
import torch


class ColdBackbone:
    def __init__(self, runtime, model_path, *, metadata_only=False):
        self.runtime = runtime
        self.path = Path(model_path)
        # MLX-Q4 owns the complete inference graph in the native worker.  The
        # Python graph is retained only as a shape/configuration shell for
        # routing and persisted metadata, so it must not require the 4.6 GB
        # upstream safetensors shards.  Legacy/HF paths keep the strict cold
        # file checks below.
        self.metadata_only = bool(metadata_only)
        self.depth = min(4, len(runtime.decoder.layers))
        self.layers = list(runtime.decoder.layers[self.depth:])
        self.prefix = next(name for name, module in runtime.model.named_modules() if module is runtime.decoder)
        self.depth_count = 0
        self.loaded = (False if self.metadata_only
                       else not any(value.is_meta for layer in self.layers for value in layer.parameters()))
        self.loads = 0
        self.last_load_seconds = None
        self.last_error = None
        if self.metadata_only:
            self.weight_map = {}
            self.files = {}
            self.file_stats = {}
        else:
            index = self.path / "model.safetensors.index.json"
            if index.exists():
                self.weight_map = json.loads(index.read_text())["weight_map"]
            else:
                shards = list(self.path.glob("*.safetensors"))
                if len(shards) != 1:
                    raise ValueError("按需加载需要唯一权重文件或官方权重索引")
                with safe_open(shards[0], framework="pt", device="cpu") as reader:
                    self.weight_map = {key: shards[0].name for key in reader.keys()}
            filenames = set(self.weight_map.values())
            if any(Path(name).name != name or not name.endswith(".safetensors") for name in filenames):
                raise ValueError("冷权重索引含不安全路径")
            self.files = {name: self.path / name for name in filenames}
            if any(path.is_symlink() or not path.is_file() for path in self.files.values()):
                raise ValueError("冷权重文件缺失或为符号链接")
            self.file_stats = {name: self._stat(path) for name, path in self.files.items()}
        self.resident_buffers = [
            {name: value.detach().cpu().clone() for name, value in layer.named_buffers()}
            for layer in runtime.decoder.layers[:self.depth]]
        self.buffers = []
        for layer in self.layers:
            # State dicts omit nonpersistent buffers. Preserve small official
            # buffers rather than silently rebuilding attention constants.
            self.buffers.append({name: value.detach().cpu().clone()
                                 for name, value in layer.named_buffers()})
        self.vision = getattr(runtime.model.model, "visual", None)
        self.vision_buffers = ({name: value.detach().cpu().clone() for name, value in self.vision.named_buffers()}
                               if self.vision is not None else {})
        # Transformers reads model.device from its first registered parameter.
        # Keep the resident language embedding first when visual weights are
        # meta, without changing names, forward order or tied storage.
        modules = runtime.model.model._modules
        if "language_model" in modules:
            language = modules.pop("language_model")
            runtime.model.model._modules = {"language_model": language, **modules}

    @staticmethod
    def _stat(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def status(self):
        model = self.runtime.model
        resident = [value for value in model.parameters() if not value.is_meta]
        layers = sum(all(not value.is_meta for value in layer.parameters()) for layer in self.runtime.decoder.layers)
        return {"enabled": True, "metadata_only": self.metadata_only,
                "resident_language_layers": layers,
                "cold_language_layers": len(self.runtime.decoder.layers) - layers,
                "resident_model_parameters": sum(value.numel() for value in resident),
                "resident_model_weight_bytes": sum(value.numel() * value.element_size() for value in resident),
                "disk_loads": self.loads, "last_load_seconds": self.last_load_seconds,
                "last_error": self.last_error,
                "vision_loaded": self.vision is not None and any(not value.is_meta for value in self.vision.parameters()),
                "scope": "weight residency; excludes router, buffers, activations, allocator caches and OS page cache; HF text weights load on demand"}

    @contextmanager
    def _readers(self):
        if any(self._stat(path) != self.file_stats[name] for name, path in self.files.items()):
            raise ValueError("冷权重文件已变化，请重新加载并核验基座")
        with ExitStack() as stack:
            yield {name: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                   for name, path in self.files.items()}

    def _tensor(self, key, reference, readers, device):
        if key not in self.weight_map:
            raise ValueError("冷权重缺少真实模型图张量：" + key)
        value = readers[self.weight_map[key]].get_tensor(key)
        if value.shape != reference.shape:
            raise ValueError("冷权重尺寸与模型图不符：" + key)
        return value.to(device=device, dtype=reference.dtype)

    def original_tensor(self, key, reference):
        if self.metadata_only:
            raise RuntimeError("MLX-Q4 元数据模式没有原始权重张量")
        with self._readers() as readers:
            # Detach this small rollback tensor from the checkpoint's mmap.
            return self._tensor(key.replace(".mlp.base.", ".mlp."), reference, readers, "cpu").clone()

    def _load_module(self, module, prefix, readers):
        state = {}
        for name, reference in module.state_dict().items():
            if name.startswith("mlp.expert."):
                raise RuntimeError("冷基础层换入前须先释放已学分支")
            key = prefix + "." + name.replace("mlp.base.", "mlp.", 1)
            state[name] = self._tensor(key, reference, readers, self.runtime.device)
        module.load_state_dict(state, strict=True, assign=True)
        module.requires_grad_(False)

    def load_resident(self):
        runtime = self.runtime
        with self._readers() as readers:
            for name in ("embed_tokens", "norm"):
                self._load_module(getattr(runtime.decoder, name), self.prefix + "." + name, readers)
            for index, layer in enumerate(runtime.decoder.layers[:self.depth]):
                self._load_module(layer, f"{self.prefix}.layers.{index}", readers)
                for name, value in self.resident_buffers[index].items():
                    owner, _, attribute = name.rpartition(".")
                    module = layer.get_submodule(owner) if owner else layer
                    module._buffers[attribute] = value.to(runtime.device)
            if runtime.model.config.tie_word_embeddings:
                runtime.model.tie_weights()
            else:
                self._load_module(runtime.model.lm_head, "lm_head", readers)

    def ensure_resident(self):
        if self.metadata_only:
            raise RuntimeError("MLX-Q4 元数据模式不支持旧 HF 快脑执行")
        if any(value.is_meta for module in self.resident_modules() for value in module.parameters()):
            try:
                self.load_resident()
            except BaseException:
                self.evict_resident()
                raise

    def resident_modules(self):
        decoder = self.runtime.decoder
        return (decoder.embed_tokens, decoder.norm, *decoder.layers[:self.depth], self.runtime.model.lm_head)

    def evict_resident(self):
        if self.metadata_only:
            return
        if self.loaded or self.depth_count:
            raise RuntimeError("完整 HF 核心使用中，不能释放快脑权重")
        if all(value.is_meta for module in self.resident_modules() for value in module.parameters()):
            return
        for module in self.resident_modules():
            module.to_empty(device="meta")
        if self.runtime.model.config.tie_word_embeddings:
            self.runtime.model.tie_weights()
        gc.collect()
        if self.runtime.device == "mps":
            torch.mps.empty_cache()

    def acquire(self):
        if self.metadata_only:
            raise RuntimeError("MLX-Q4 元数据模式不支持旧 HF 完整模型")
        if self.depth_count:
            self.depth_count += 1
            return
        self.ensure_resident()
        if not self.loaded:
            self._load()
        self.depth_count = 1

    def _load(self):
        if self.metadata_only:
            raise RuntimeError("MLX-Q4 元数据模式不支持旧 HF 完整模型")
        runtime = self.runtime
        started = time.perf_counter()
        self.last_error = None
        try:
            with self._readers() as readers:
                for offset, layer in enumerate(self.layers, self.depth):
                    self._load_module(layer, f"{self.prefix}.layers.{offset}", readers)
                    for name, value in self.buffers[offset - self.depth].items():
                        owner, _, attribute = name.rpartition(".")
                        module = layer.get_submodule(owner) if owner else layer
                        module._buffers[attribute] = value.to(runtime.device)
            self.loaded = True
            # Every cold load starts with genuine upstream weights, then applies
            # the verified cumulative selected core, never a random new tensor.
            from .evolution import load_active_core
            load_active_core(runtime)
            self.loads += 1
            self.last_load_seconds = time.perf_counter() - started
        except BaseException as error:
            self.last_error = str(error) or type(error).__name__
            self._evict()
            raise

    def release(self):
        if self.depth_count <= 0:
            raise RuntimeError("冷基础层释放计数无效")
        self.depth_count -= 1
        if not self.depth_count:
            self._evict()

    def _evict(self):
        runtime = self.runtime
        # Detached branch weights must not be moved to meta and then restored.
        # Inference uses immutable saved experts; training callers keep its
        # temporary reference until their serialized operation is complete.
        expert = runtime.branch.expert
        runtime.branch.expert = None
        try:
            for layer in self.layers:
                layer.to_empty(device="meta")
            if self.vision is not None:
                self.vision.to_empty(device="meta")
            if hasattr(runtime.model.model, "rope_deltas"):
                runtime.model.model.rope_deltas = None
            self.loaded = False
        finally:
            runtime.branch.expert = expert
        gc.collect()
        if runtime.device == "mps":
            torch.mps.empty_cache()


@contextmanager
def full_backbone(runtime):
    if hasattr(runtime, "close_native"):
        runtime.close_native()
    pager = getattr(runtime, "cold_backbone", None)
    if pager is None:
        yield
        return
    if getattr(pager, "metadata_only", False):
        raise RuntimeError("MLX-Q4 元数据模式不支持旧 HF 完整模型；请使用原生 MLX 引擎")
    pager.acquire()
    try:
        yield
    finally:
        pager.release()
