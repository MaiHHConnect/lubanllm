"""Luban 训练引擎: bounded native cached-tail execution, never model adoption.

The caller owns qualification, a frozen complete Python prefix, quality gates,
durable TrainingSession checkpoints and publication. This module has no research
script dependency. A worker inherits the parent's already-held model lock.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time

ENGINE_NAME = "Luban 训练引擎"
ENGINE_ID = "luban-training-ce-release-v1"
ABI = "LUBAN_NATIVE_TRAINING_V1"
OUTPUT_ABI = "LUBAN_NATIVE_TRAINING_OUTPUT_V1"
SHA = re.compile(r"[0-9a-f]{64}")
WIDTHS = {"bf16": 2, "f32": 4, "i64": 8, "u8": 1}
STATE = ("candidate", "master", "exp_avg", "exp_avg_sq", "adam_step")
FINAL = STATE + ("reference", "basis")
PROBES = ("ce_logits", "teacher_log_probs", "student_log_probs", "ce_grad", "kl_grad",
          "total_grad", "clipped_grad", "delta", "master", "candidate", "exp_avg",
          "exp_avg_sq", "adam_step")
TRACE = ("step", "ce_role", "ce_row", "replay_row", "ce", "kl", "preclip_norm", "loss", "wall_seconds")


def _need(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(path, *, exists=True):
    result = Path(os.path.abspath(path))
    _need(result.resolve() == result and not result.is_symlink(), "训练路径不能是链接或非规范路径")
    if exists:
        _need(result.exists(), "训练文件不存在")
    return result


def checkpoints(steps, selected=None):
    _need(type(steps) is int and 1 <= steps <= 4096, "原生训练步数须在1–4096之间")
    values = sorted({1, min(2, steps), *range(50, steps + 1, 50), steps}) if selected is None else list(selected)
    _need(values and all(type(v) is int and 1 <= v <= steps for v in values)
        and values == sorted(set(values)) and values[-1] == steps, "检查点须升序、唯一并包含末步")
    return values


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def inherited_lock(runtime):
    """Validate and retain one open-file description; never release/reacquire."""
    import fcntl
    owner = getattr(runtime, "_model_process_lock", None)
    fd = getattr(owner, "fd", None)
    _need(type(fd) is int and fd >= 0 and owner.key == (runtime.identity, str(runtime.device)),
          "原生训练必须继承当前基座与设备的进程锁")
    directory = Path("/tmp") / ("luban-model-locks-" + str(os.getuid()))
    path = directory / (runtime.identity + "-" + str(runtime.device).replace(":", "-") + ".lock")
    parent, named, opened = directory.lstat(), path.lstat(), os.fstat(fd)
    _need(stat.S_ISDIR(parent.st_mode) and parent.st_uid == os.getuid() and not parent.st_mode & 0o077,
          "模型锁目录权限无效")
    _need(stat.S_ISREG(named.st_mode) and stat.S_ISREG(opened.st_mode)
        and (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino)
        and opened.st_uid == os.getuid() and opened.st_nlink == 1 and not opened.st_mode & 0o077,
          "继承模型锁描述符与标准锁文件不匹配")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def export_tensor(root, name, value):
    import torch
    _need(re.fullmatch(r"[a-z][a-z0-9_]*", name), "训练张量名称无效")
    cpu = value.detach().cpu().contiguous()
    dtype = {torch.bfloat16: "bf16", torch.float32: "f32", torch.int64: "i64", torch.uint8: "u8"}.get(cpu.dtype)
    _need(dtype is not None and bool(torch.isfinite(cpu).all()), "训练张量精度或数值无效")
    path = _path(root) / (name + ".raw")
    with path.open("xb") as stream:
        cpu.reshape(-1).view(torch.uint8).numpy().tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {"name": name, "dtype": dtype, "shape": list(cpu.shape), "file": path.name,
            "bytes": cpu.numel() * cpu.element_size(), "sha256": file_sha(path)}


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(__import__("torch").uint8).numpy()).hexdigest()


def graph_binding(runtime):
    """Same prefix contract as the existing audit, bound to the actual core."""
    import inspect
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5MLP, Qwen3_5RMSNorm
    from .dynamic import DynamicMLP
    from .backbone import text_backbone
    model, decoder = runtime.model, runtime.decoder
    branch, layer = runtime.branch, decoder.layers[-1]
    _need(type(model) is Qwen3_5ForConditionalGeneration and text_backbone(model)[0] is decoder
        and type(layer) is Qwen3_5DecoderLayer and type(layer.mlp) is DynamicMLP
        and layer.mlp is branch and type(branch.base) is Qwen3_5MLP
        and type(branch.base.down_proj) is torch.nn.Linear and type(decoder.norm) is Qwen3_5RMSNorm
        and type(model.lm_head) is torch.nn.Linear and branch.expert is None,
        "原生训练只支持已验证的Qwen3.5末层、关闭专家的真实图")
    parameter = branch.base.down_proj.weight
    _need(tuple(parameter.shape) == (2048, 6144) and parameter.dtype == torch.bfloat16
        and parameter.device.type == "mps" and float(decoder.norm.eps) == 1e-6
        and tuple(model.lm_head.weight.shape) == (248320, 2048), "原生训练图尺寸、精度或设备不匹配")
    for module in model.modules():
        _need(not module.training and not module._forward_hooks and not module._forward_pre_hooks
            and getattr(module.forward, "__func__", None) is type(module).forward,
            "前缀捕获要求原始eval图且没有外部forward hook")
    _need(all(not p.requires_grad or p is parameter for p in model.parameters()), "前缀含其他可训练参数")
    source_paths = {Path(inspect.getfile(type(m))).resolve() for m in model.modules()}
    pointer = Path(runtime.store) / "active-core.json"
    return {"identity": runtime.identity, "core_version": runtime.active_core,
        "core_pointer_sha256": file_sha(pointer) if pointer.exists() else None,
        "model": id(model), "config": model.config.to_json_string(),
        "sources": {str(path): file_sha(path) for path in sorted(source_paths)},
        "frozen_parameters": [(name, id(p), p._version, tuple(p.shape), str(p.dtype), str(p.device))
                              for name, p in model.named_parameters() if p is not parameter],
        # Paging may replace nonpersistent buffer objects during a lossless
        # CPU↔MPS move. Bind their actual bytes, dtype and device instead of id.
        "buffers": [(name, tuple(b.shape), str(b.dtype), str(b.device), b.is_meta,
                     None if b.is_meta else tensor_sha(b))
                    for name, b in model.named_buffers()]}


def capture_tails(runtime, samples):
    """Complete sequence X/residual cache; no logits/reference/answer lookup."""
    import torch
    binding = graph_binding(runtime)
    result = {}
    for sample in samples:
        _need(sample.get("purpose") == "training" and sample["id"] not in result, "只能缓存唯一TRAIN材料")
        ids, labels = runtime._sample_tokens(sample)
        captured = {}

        def hook(key):
            def take(module, arguments):
                _need(key not in captured, "末层前缀重复执行")
                captured[key] = arguments[0].detach().clone()
            return take

        handles = []
        try:
            handles.append(runtime.decoder.layers[-1].post_attention_layernorm.register_forward_pre_hook(hook("residual")))
            handles.append(runtime.branch.base.down_proj.register_forward_pre_hook(hook("x")))
            with torch.no_grad():
                runtime.model(ids, use_cache=False)
            torch.mps.synchronize()
        finally:
            for handle in handles:
                handle.remove()
        _need(set(captured) == {"x", "residual"} and graph_binding(runtime) == binding,
              "缓存捕获不完整或冻结图发生变化")
        values = {key: value.to("cpu", copy=True) for key, value in {**captured, "ids": ids, "labels": labels}.items()}
        _need(tuple(values["x"].shape) == (*ids.shape, 6144)
            and tuple(values["residual"].shape) == (*ids.shape, 2048)
            and all(bool(torch.isfinite(v).all()) for v in values.values()), "完整前缀张量尺寸或数值无效")
        result[sample["id"]] = {**values, "binding": binding, "sample_sha256": content_sha(sample),
            "tensor_sha256": {key: tensor_sha(v) for key, v in values.items()}}
        _interrupt(runtime)
    return result


def export_fixture(runtime, new, replay, reference, basis, caches, destination, *, steps=450,
                   learning_rate=1e-5, distillation_weight=5, checkpoint_steps=None,
                   start_step=0, saved_checkpoint=None):
    """Export current qualified groups; preserve full master/moments on resume."""
    import torch
    from .retention import eligible_learning_sample, provenance
    selected = checkpoints(steps, checkpoint_steps)
    _need(type(start_step) is int and 0 <= start_step <= 10000, "恢复步数无效")
    _need(type(learning_rate) in (int, float) and math.isfinite(learning_rate) and 1e-8 <= learning_rate <= .01
        and type(distillation_weight) in (int, float) and math.isfinite(distillation_weight)
        and 0 <= distillation_weight <= 100, "训练学习率或KL权重无效")
    inherited_lock(runtime)
    binding = graph_binding(runtime)
    parameter = runtime.branch.base.down_proj.weight
    _need(new and replay, "原生训练需要NEW和REPLAY材料")
    _need(reference.dtype == torch.float32 and tuple(reference.shape) == tuple(parameter.shape)
        and basis.dtype == torch.float32 and basis.ndim == 2 and basis.shape[0] == 6144
        and 0 < basis.shape[1] < 6144, "原生训练需要FP32冻结参考与有效保护子空间")
    values = {"candidate": parameter, "master": parameter.detach().float(), "reference": reference,
        "basis": basis, "exp_avg": torch.zeros_like(reference), "exp_avg_sq": torch.zeros_like(reference),
        "adam_step": torch.tensor(0., dtype=torch.float32), "norm": runtime.decoder.norm.weight,
        "head": runtime.model.lm_head.weight}
    if saved_checkpoint is not None:
        _need(set(saved_checkpoint) == set(FINAL), "恢复检查点必须包含完整Adam、参考与保护子空间")
        values.update(saved_checkpoint)
        _need(tensor_sha(values["reference"]) == tensor_sha(reference) and tensor_sha(values["basis"]) == tensor_sha(basis),
              "恢复检查点参考或子空间改变")
    _need(float(values["adam_step"]) == start_step and torch.equal(values["master"].to(torch.bfloat16), values["candidate"])
        and torch.equal(parameter.detach().cpu(), values["candidate"].detach().cpu()), "恢复Adam步或实际候选不一致")
    for key, value in (("down_bias", runtime.branch.base.down_proj.bias), ("head_bias", runtime.model.lm_head.bias)):
        if value is not None:
            values[key] = value
    root = _path(destination, exists=False)
    _need(root.parent.is_dir() and not root.exists(), "训练fixture必须使用新目录")
    root.mkdir(mode=0o700)
    tensors = {key: export_tensor(root, key, value) for key, value in values.items()}
    rows, keys = {}, {}
    eos = runtime.tokenizer.eos_token_id
    _need(type(eos) is int and 0 <= eos < 248320, "训练EOS无效")
    for index, (role, sample) in enumerate([(r, s) for r, group in (("new", new), ("replay", replay)) for s in group]):
        _need(eligible_learning_sample(sample) and sample["purpose"] == "training"
            and provenance(sample).get("evolution_role") == role and sample["id"] not in keys,
            "训练材料须已核验、许可、唯一且属于对应角色")
        cache = caches[sample["id"]]
        _need(cache["binding"] == binding and cache["sample_sha256"] == content_sha(sample), "缓存材料或图绑定改变")
        ids, labels = runtime._sample_tokens(sample)
        _need(torch.equal(ids.cpu(), cache["ids"]) and torch.equal(labels.cpu(), cache["labels"]), "完整输入或gold标签改变")
        mask = labels[0].ne(-100)
        positions = mask.nonzero().flatten()
        _need(ids.dtype == torch.long and labels.dtype == torch.long and ids.shape == labels.shape
            and ids.ndim == 2 and ids.shape[0] == 1 and positions.numel() > 0 and positions[0] >= 1
            and torch.equal(positions, torch.arange(int(positions[0]), ids.shape[1], device=positions.device))
            and torch.equal(labels[mask.unsqueeze(0)], ids[mask.unsqueeze(0)])
            and int(labels[0, -1]) == eos and not bool((labels[0, :-1] == eos).any()),
            "训练标签必须保留连续完整答案和末EOS")
        key = "r" + str(index).zfill(3)
        for suffix in ("x", "residual", "ids", "labels"):
            _need(tensor_sha(cache[suffix]) == cache["tensor_sha256"][suffix], "缓存原始张量改变")
            tensors[key + "_" + suffix] = export_tensor(root, key + "_" + suffix, cache[suffix])
        rows[key] = {"role": role, "sample_sha256": content_sha(sample), "length": ids.shape[1], "gold": len(positions)}
        keys[sample["id"]] = key
    schedule = []
    for offset in range(steps):
        absolute = start_step + offset
        role, group = ("new", new) if absolute % 2 == 0 else ("replay", replay)
        schedule.append({"ordinal": offset + 1, "role": role,
            "ce": keys[group[(absolute // 2) % len(group)]["id"]],
            "replay": keys[replay[(absolute // 2) % len(replay)]["id"]]})
    config = {"identity": runtime.identity, "device": "mps", "steps": str(steps), "norm_eps": "1e-6",
        "train_eos": str(eos), "checkpoint_steps": ",".join(map(str, selected)),
        "learning_rate": repr(float(learning_rate)), "distillation_weight": repr(float(distillation_weight)),
        "training_offset": str(start_step)}
    lines = [ABI] + ["\t".join(("config", key, value)) for key, value in config.items()]
    for key, spec in tensors.items():
        lines.append("\t".join(("tensor", key, spec["dtype"], ",".join(map(str, spec["shape"])) or "-",
                               spec["file"], str(spec["bytes"]), spec["sha256"])))
    for key, row in rows.items():
        lines.append("\t".join(("row", key, row["role"], row["sample_sha256"], str(row["length"]), str(row["gold"]))))
    lines += ["\t".join(("step", str(s["ordinal"]), s["role"], s["ce"], s["replay"])) for s in schedule]
    with (root / "manifest.tsv").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _write_json(root / "fixture.json", {"engine": ENGINE_ID, "binding": binding,
        "dataset_sha256": content_sha({"new": new, "replay": replay}), "start_step": start_step,
        "manifest_sha256": file_sha(root / "manifest.tsv"), "quality_checked": False})
    _need(graph_binding(runtime) == binding, "导出期间冻结图改变")
    return validate_fixture(root, file_sha(root / "manifest.tsv"))


def _tensor_specs(path, header, *, live=False):
    content = _path(path).read_text()
    # A running worker may be appending its next descriptor. Only complete
    # newline-terminated records can describe a committed checkpoint.
    lines = (content.rsplit("\n", 1)[0] if live and not content.endswith("\n") else content).splitlines()
    _need(lines and lines[0] == header, "原生张量ABI不匹配")
    specs, other, files = {}, [], set()
    for line in lines[1:]:
        p = line.split("\t")
        if p[0] != "tensor":
            other.append(p)
            continue
        _need(len(p) == 7 and re.fullmatch(r"[a-z][a-z0-9_]*", p[1]) and p[2] in WIDTHS
            and re.fullmatch(r"[a-z][a-z0-9_]*\.raw", p[4]) and SHA.fullmatch(p[6])
            and p[1] not in specs and p[4] not in files, "原生张量描述无效或重复")
        shape = [] if p[3] == "-" else [int(v) for v in p[3].split(",")]
        _need(len(shape) <= 3 and all(v > 0 for v in shape) and int(p[5]) == math.prod(shape) * WIDTHS[p[2]],
              "原生张量bytes或shape不匹配")
        specs[p[1]] = {"name": p[1], "dtype": p[2], "shape": shape, "file": p[4], "bytes": int(p[5]), "sha256": p[6]}
        files.add(p[4])
    return specs, other


def _verify_files(root, specs):
    for value in specs.values():
        path = _path(Path(root) / value["file"])
        _need(path.is_file() and path.stat().st_size == value["bytes"] and file_sha(path) == value["sha256"],
              "原生张量实际bytes/SHA不匹配")


def validate_fixture(root, approved_sha):
    root = _path(root)
    _need(SHA.fullmatch(approved_sha or "") and file_sha(root / "manifest.tsv") == approved_sha,
          "训练manifest SHA不匹配")
    tensors, other = _tensor_specs(root / "manifest.tsv", ABI)
    config, rows, schedule = {}, {}, []
    for p in other:
        if p[0] == "config":
            _need(len(p) == 3 and p[1] not in config, "训练config重复或无效")
            config[p[1]] = p[2]
        elif p[0] == "row":
            _need(len(p) == 6 and re.fullmatch(r"r[0-9]+", p[1]) and p[1] not in rows
                and p[2] in {"new", "replay"} and SHA.fullmatch(p[3]), "训练row无效")
            rows[p[1]] = {"role": p[2], "sample_sha256": p[3], "length": int(p[4]), "gold": int(p[5])}
            _need(0 < int(p[5]) < int(p[4]), "训练row长度无效")
        elif p[0] == "step":
            _need(len(p) == 5, "训练step无效")
            schedule.append({"ordinal": int(p[1]), "role": p[2], "ce": p[3], "replay": p[4]})
        else:
            raise ValueError("未知训练manifest字段")
    required = {"identity", "device", "steps", "norm_eps", "train_eos", "checkpoint_steps"}
    _need(required <= config.keys() and config.keys() <= required | {"learning_rate", "distillation_weight", "training_offset"}
        and SHA.fullmatch(config["identity"]) and config["device"] in {"cpu", "mps"} and config["norm_eps"] == "1e-6",
        "训练config身份或schema无效")
    steps = int(config["steps"])
    selected = checkpoints(steps, [int(v) for v in config["checkpoint_steps"].split(",")])
    offset = int(config.get("training_offset", "0"))
    rate, kl = float(config.get("learning_rate", "1e-5")), float(config.get("distillation_weight", "5"))
    _need(0 <= offset <= 10000 and math.isfinite(rate) and 1e-8 <= rate <= .01
        and math.isfinite(kl) and 0 <= kl <= 100, "训练参数无效")
    _need(len(schedule) == steps and {"new", "replay"} <= {r["role"] for r in rows.values()}, "训练排程或材料缺失")
    for index, s in enumerate(schedule):
        role = "new" if (offset + index) % 2 == 0 else "replay"
        _need(s["ordinal"] == index + 1 and s["role"] == role and s["ce"] in rows and s["replay"] in rows
            and rows[s["ce"]]["role"] == role and rows[s["replay"]]["role"] == "replay", "训练绝对步排程不匹配")
    base = set(FINAL) | {"norm", "head"}
    required_tensors = base | {r + "_" + suffix for r in rows for suffix in ("x", "residual", "ids", "labels")}
    _need(required_tensors <= tensors.keys() and tensors.keys() <= required_tensors | {"down_bias", "head_bias"},
          "训练缺少完整Adam/前缀或含额外张量")
    shape = tensors["candidate"]["shape"]
    _need(len(shape) == 2 and tensors["candidate"]["dtype"] == "bf16", "正式原生候选必须为二维BF16")
    hidden, inputs = shape
    for key in ("master", "reference", "exp_avg", "exp_avg_sq"):
        _need(tensors[key]["shape"] == shape and tensors[key]["dtype"] == "f32", "完整FP32优化器状态不兼容")
    _need(tensors["adam_step"]["shape"] == [] and tensors["adam_step"]["dtype"] == "f32"
        and tensors["basis"]["dtype"] == "f32" and len(tensors["basis"]["shape"]) == 2
        and tensors["basis"]["shape"][0] == inputs and 0 < tensors["basis"]["shape"][1] < inputs,
        "Adam累计步或子空间schema无效")
    _need(tensors["norm"]["shape"] == [hidden] and tensors["norm"]["dtype"] in {"bf16", "f32"}
        and len(tensors["head"]["shape"]) == 2 and tensors["head"]["shape"][1] == hidden
        and tensors["head"]["dtype"] == "bf16", "完整head或norm不匹配")
    vocab = tensors["head"]["shape"][0]
    _need(0 <= int(config["train_eos"]) < vocab, "训练EOS不属于完整词表")
    for key, expected in (("down_bias", [hidden]), ("head_bias", [vocab])):
        if key in tensors:
            _need(tensors[key]["shape"] == expected and tensors[key]["dtype"] == "bf16", "训练bias不兼容")
    for key, row in rows.items():
        for suffix, expected, dtype in (("x", [1, row["length"], inputs], "bf16"),
                ("residual", [1, row["length"], hidden], "bf16"),
                ("ids", [1, row["length"]], "i64"), ("labels", [1, row["length"]], "i64")):
            spec = tensors[key + "_" + suffix]
            _need(spec["shape"] == expected and spec["dtype"] == dtype, "完整前缀shape或dtype不匹配")
    _verify_files(root, tensors)
    return {"directory": str(root), "manifest_sha256": approved_sha, "config": config, "tensors": tensors,
            "rows": rows, "steps": schedule, "checkpoint_steps": selected}


def _state_descriptors(root, fixture, step=None):
    root = _path(root)
    _need(step is None or type(step) is int and step in fixture["checkpoint_steps"], "未声明恢复检查点")
    tensors, other = _tensor_specs(root / "tensors.tsv", OUTPUT_ABI, live=step is not None)
    _need(not other, "训练输出含未知schema")
    prefix = "final_" if step is None else f"checkpoint{step:06d}_"
    result = {}
    for key in FINAL:
        name = prefix + key
        spec = tensors.get(name) if key in STATE or step is None else fixture["tensors"][key]
        _need(spec is not None and spec["shape"] == fixture["tensors"][key]["shape"]
            and spec["dtype"] == fixture["tensors"][key]["dtype"], "恢复检查点不含完整兼容状态")
        result[key] = spec
    _verify_files(root, {k: v for k, v in result.items() if k in STATE or step is None})
    for key in ("reference", "basis"):
        _need(result[key]["sha256"] == fixture["tensors"][key]["sha256"], "冻结参考或子空间改变")
    return result


def load_checkpoint(output, fixture, *, step=None):
    """CPU complete tensors; caller persists them in the existing session journal."""
    import torch
    specs = _state_descriptors(output, fixture, step)
    values = {}
    for key, spec in specs.items():
        root = Path(output) if key in STATE or step is None else Path(fixture["directory"])
        value = torch.from_file(str(_path(root / spec["file"])), shared=False,
            size=math.prod(spec["shape"]), dtype={"bf16": torch.bfloat16, "f32": torch.float32}[spec["dtype"]]).reshape(spec["shape"])
        _need(bool(torch.isfinite(value).all()), "恢复张量含非有限值")
        values[key] = value.clone()
    target = len(fixture["steps"]) if step is None else step
    _need(float(values["adam_step"]) == int(fixture["config"].get("training_offset", 0)) + target
        and torch.equal(values["master"].to(values["candidate"].dtype), values["candidate"])
        and bool((values["exp_avg_sq"] >= 0).all()), "恢复Adam累计步、候选cast或二阶矩无效")
    return values


def validate_output(output, fixture, binary_sha):
    root = _path(output)
    _need(not (root / "failure.json").exists() and not (root / "failure.txt").exists(), "失败输出不能作为成功候选")
    tensors, rest = _tensor_specs(root / "tensors.tsv", OUTPUT_ABI)
    required = {f"checkpoint{s:06d}_{k}" for s in fixture["checkpoint_steps"] for k in PROBES}
    required |= {"final_" + key for key in FINAL}
    _need(not rest and set(tensors) == required, "原生训练审计或末状态缺失")
    for step in fixture["checkpoint_steps"]:
        scheduled = fixture["steps"][step - 1]
        for key in PROBES:
            spec = tensors[f"checkpoint{step:06d}_{key}"]
            if key == "ce_logits" or key.endswith("log_probs"):
                sample = scheduled["ce"] if key == "ce_logits" else scheduled["replay"]
                shape, dtype = [fixture["rows"][sample]["gold"], fixture["tensors"]["head"]["shape"][0]], "f32"
            else:
                initial = "candidate" if key in {"ce_grad", "kl_grad", "total_grad", "candidate"} else key if key in STATE else "master"
                shape, dtype = fixture["tensors"][initial]["shape"], fixture["tensors"][initial]["dtype"]
            _need(spec["shape"] == shape and spec["dtype"] == dtype, "原生审计梯度/logits/state schema不兼容")
    _verify_files(root, tensors)
    _state_descriptors(root, fixture)
    report = json.loads((root / "result.json").read_text())
    total = len(fixture["steps"])
    entry = int(fixture["config"].get("training_offset", "0"))
    counts = {"steps": total, "new_ce_exposures": sum(s["role"] == "new" for s in fixture["steps"]),
        "replay_ce_exposures": sum(s["role"] == "replay" for s in fixture["steps"]), "replay_kl_exposures": total,
        "entry_adam_step": entry, "final_adam_step": entry + total, "teacher_calls": 0, "generation_calls": 0}
    _need(report.get("abi") == ABI and report.get("passed_execution") is True
        and report.get("manifest_sha256") == fixture["manifest_sha256"] and report.get("binary_sha256") == binary_sha
        and report.get("device") == fixture["config"]["device"] and report.get("release_ce_graph") is True
        and report.get("reuse_replay_forward") is True and report.get("publication") is False
        and report.get("all_input_tensors_preserved") is True and report.get("input_tensor_count") == len(fixture["tensors"]),
        "原生输出未绑定当前CE释放引擎或完整输入")
    _need(all(type(report.get(k)) is int and report[k] == v for k, v in counts.items()), "原生输出曝光或Adam累计步不匹配")
    with (root / "trace.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _need(tuple(reader.fieldnames or ()) == TRACE, "训练trace schema无效")
        trace = list(reader)
    _need(len(trace) == total, "训练trace步数不完整")
    kl_weight = float(fixture["config"].get("distillation_weight", "5"))
    for row, scheduled in zip(trace, fixture["steps"]):
        _need(int(row["step"]) == scheduled["ordinal"] and row["ce_role"] == scheduled["role"]
            and row["ce_row"] == scheduled["ce"] and row["replay_row"] == scheduled["replay"], "实际训练排程改变")
        for key in ("ce", "kl", "preclip_norm", "loss", "wall_seconds"):
            row[key] = float(row[key])
            _need(math.isfinite(row[key]) and (key in {"kl", "loss"} or row[key] >= 0), "训练trace数值异常")
        _need(math.isclose(row["loss"], row["ce"] + kl_weight * row["kl"], rel_tol=1e-9, abs_tol=1e-9), "训练CE/KL不一致")
    return {"engine": ENGINE_ID, "engine_name": ENGINE_NAME, "output": str(root), "report": report,
        "trace": trace, "tensors": tensors, "quality_checked": False,
        "receipt_sha256": content_sha({"tensors": file_sha(root / "tensors.tsv"),
            "report": file_sha(root / "result.json"), "trace": file_sha(root / "trace.tsv")})}


def _interrupt(runtime):
    session = getattr(runtime, "training_session", None)
    if session and session.pause_requested:
        from .training_checkpoint import TrainingPaused
        raise TrainingPaused("Luban训练已暂停；采纳指针保持不变")
    if getattr(runtime, "job", None) and runtime.job.get("cancel_requested"):
        raise RuntimeError("Luban训练已取消；采纳指针保持不变")


def run_worker(runtime, fixture, output, binary, *, on_checkpoint=None, timeout=2400):
    """Execute one worker with inherited lock; persist complete states before pause.

    on_checkpoint(step, descriptor) is called only after a whole checkpoint is
    flushed. It must use load_checkpoint and durable TrainingSession.save; a
    callback exception terminates the child. Interrupted outputs stay immutable.
    """
    _need(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0, "原生训练超时预算无效")
    fixture = validate_fixture(fixture["directory"], fixture["manifest_sha256"])
    fd = inherited_lock(runtime)
    _need(getattr(runtime, "native_engine", None) is None, "原生训练前必须关闭常驻推理worker")
    _need(fixture["config"]["identity"] == runtime.identity and fixture["config"]["device"] == str(runtime.device),
          "运行时与训练fixture身份不一致")
    binary, target = _path(binary), _path(output, exists=False)
    _need(not target.exists() and target.parent.is_dir() and not target.is_relative_to(Path(fixture["directory"]))
        and not Path(fixture["directory"]).is_relative_to(target), "原生输出必须为独立新目录")
    build = json.loads(_path(str(binary) + ".build.json").read_text())
    hashed = file_sha(binary)
    _need(build.get("binary_sha256") == hashed and build.get("release_ce_graph") is True
        and build.get("reuse_replay_forward") is True and build.get("model_lock_inheritance") is True,
          "正式训练只接受支持继承锁且有构建回执的CE释放二进制")
    libraries = build.get("torch_library_bindings")
    _need(isinstance(libraries, dict) and len(libraries) >= 3, "训练二进制缺少LibTorch绑定")
    for name, spec in libraries.items():
        path = _path(name)
        _need(path.stat().st_size == spec["bytes"] and file_sha(path) == spec["sha256"], "训练LibTorch依赖改变")
    _interrupt(runtime)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
    env.update(PYTORCH_ENABLE_MPS_FALLBACK="0", LUBAN_MODEL_LOCK_FD=str(fd))
    saved, completed = set(), False
    start = time.monotonic()
    logs = {suffix: target.with_name(target.name + suffix) for suffix in (".stdout.log", ".stderr.log")}
    with logs[".stdout.log"].open("xb") as stdout, logs[".stderr.log"].open("xb") as stderr:
        child = subprocess.Popen([str(binary), "--input", fixture["directory"], "--output", str(target),
            "--manifest-sha", fixture["manifest_sha256"]], env=env, pass_fds=(fd,), stdout=stdout, stderr=stderr)
        try:
            while True:
                exit_code = child.poll()
                if on_checkpoint and (target / "trace.tsv").exists():
                    lines = (target / "trace.tsv").read_text().splitlines()
                    committed_steps = {int(line.split("\t")[0]) for line in lines[1:] if len(line.split("\t")) == len(TRACE)}
                    for step in fixture["checkpoint_steps"]:
                        if step in committed_steps and step not in saved:
                            descriptor = _state_descriptors(target, fixture, step)
                            on_checkpoint(step, descriptor)
                            saved.add(step)
                _interrupt(runtime)
                if exit_code is not None:
                    _need(exit_code == 0, "原生训练退出异常，完整日志和已保存状态保留")
                    completed = True
                    break
                _need(time.monotonic() - start < timeout, "原生训练超时，采纳指针保持不变")
                try:
                    child.wait(timeout=.1)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            _write_json(target.with_name(target.name + ".execution.json"), {"engine": ENGINE_ID,
                "binary_sha256": hashed, "manifest_sha256": fixture["manifest_sha256"], "pid": child.pid,
                "exit_code": child.returncode, "completed": completed, "saved_steps": sorted(saved),
                "wall_seconds": time.monotonic() - start, "inherited_model_lock": True, "quality_checked": False})
    _need(file_sha(binary) == hashed, "运行期间训练二进制改变")
    validate_fixture(fixture["directory"], fixture["manifest_sha256"])
    for name, spec in libraries.items():
        _need(file_sha(name) == spec["sha256"], "运行期间LibTorch依赖改变")
    return validate_output(target, fixture, hashed)
