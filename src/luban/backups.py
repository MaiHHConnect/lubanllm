"""Three immutable pre-training checkpoints; callers hold the runtime task lock."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .settings import private_json, require_storage

_ROOT_FILES = {"active.json", "router.json", "router.safetensors", "active-core.json", "active-router.json"}
_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}")
_BACKUP_ID = re.compile(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}")
_EXPERT_FILE = re.compile(r"layer-[0-9]+/[a-zA-Z0-9_-]{1,80}\.(json|safetensors)")
_ROUTER_FILE = re.compile(r"router-[0-9a-f]{12}\.(json|safetensors)")
_CORE_ID = re.compile(r"core-[0-9a-f]{12}")
_CORE_FILE = re.compile(r"core-[0-9a-f]{12}\.(json|safetensors|retention\.json)")
_CORE_PARAMETER = re.compile(r"model\.(?:language_model\.)?layers\.[0-9]+\.mlp\.(?:base\.)?down_proj\.weight")


def _file(root, name):
    if name not in _ROOT_FILES and not _EXPERT_FILE.fullmatch(name) and not _CORE_FILE.fullmatch(name) and not _ROUTER_FILE.fullmatch(name):
        raise ValueError("备份包含不安全文件路径")
    path = root / name
    if any(p.is_symlink() for p in (path, path.parent, root)):
        raise ValueError("备份文件不能使用符号链接")
    return path


def _json(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("备份 JSON 必须是对象")
    return data


def _tensor_shapes(path, *, include_dtypes=False):
    # Safetensors has an eight-byte header length, JSON header, then tensor bytes.
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= min(1_000_000, path.stat().st_size - 8):
            raise ValueError("无效的权重文件头")
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError("无效的权重描述")
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
    end, shapes, dtypes = 0, {}, {}
    entries = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        shape, offsets = item["shape"], item["data_offsets"]
        if (not isinstance(shape, list) or not shape or
                any(type(n) is not int or n <= 0 for n in shape) or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(n) is not int for n in offsets)):
            raise ValueError("无效的权重尺寸")
        count = 1
        for size in shape:
            count *= size
        if item["dtype"] not in sizes or offsets[1] - offsets[0] != count * sizes[item["dtype"]]:
            raise ValueError("权重字节数与尺寸不符")
        entries.append((offsets[0], offsets[1]))
        shapes[name] = shape
        dtypes[name] = item["dtype"]
    for start, stop in sorted(entries):
        if start != end:
            raise ValueError("权重数据偏移不完整")
        end = stop
    if not shapes or end != path.stat().st_size - 8 - length:
        raise ValueError("权重数据不完整")
    return (shapes, dtypes) if include_dtypes else shapes


def _validate(root, identity, files):
    if not isinstance(files, dict) or len(files) > 10:
        raise ValueError("备份文件清单无效")
    for name, record in files.items():
        path = _file(root, name)
        if (not path.is_file() or not isinstance(record, dict) or
                type(record.get("size")) is not int or path.stat().st_size != record["size"] or
                hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256")):
            raise ValueError("备份文件损坏或校验失败")
    expected = set()
    active, meta, core_shape = None, None, None
    if "active-core.json" in files:
        active_core = _json(_file(root, "active-core.json"))
        version = active_core.get("version")
        if (active_core.get("identity") != identity or not isinstance(version, str)
                or not _CORE_ID.fullmatch(version)):
            raise ValueError("备份活动核心与基座或版本不兼容")
        metadata_name, weights_name = version + ".json", version + ".safetensors"
        if not {metadata_name, weights_name} <= files.keys():
            raise ValueError("备份缺少活动核心文件")
        core = _json(_file(root, metadata_name))
        parameter, shape = core.get("parameter"), core.get("shape")
        declared_dtype = core.get("dtype")
        dtype = {"torch.float64": "F64", "torch.float32": "F32", "torch.float16": "F16",
                 "torch.bfloat16": "BF16"}.get(declared_dtype) if isinstance(declared_dtype, str) else None
        digest = files[weights_name]["sha256"]
        if (core.get("identity") != identity or core.get("version") != version
                or core.get("sha256") != digest or active_core.get("sha256") != digest
                or not isinstance(parameter, str) or not _CORE_PARAMETER.fullmatch(parameter)
                or not isinstance(shape, list) or len(shape) != 2
                or any(type(size) is not int or size <= 0 for size in shape) or dtype is None
                or core.get("format", "cumulative core tensor replacement") != "cumulative core tensor replacement"):
            raise ValueError("备份核心元数据不兼容")
        shapes, dtypes = _tensor_shapes(_file(root, weights_name), include_dtypes=True)
        if shapes != {parameter: shape} or dtypes != {parameter: dtype}:
            raise ValueError("备份核心权重真实尺寸或精度不兼容")
        # Match the core loader's finite-value gate before touching live files;
        # only the selected replacement tensor is allocated, never the backbone.
        import torch
        from safetensors.torch import load_file
        if not torch.isfinite(load_file(str(_file(root, weights_name)))[parameter]).all():
            raise ValueError("备份核心权重包含非有限值")
        core_shape = shape
        if "trained_parameters" in core and (type(core["trained_parameters"]) is not int
                or core["trained_parameters"] != shape[0] * shape[1]):
            raise ValueError("备份核心参数数量不兼容")
        expected.update(("active-core.json", metadata_name, weights_name))
        retention_name = version + ".retention.json"
        if retention_name in files:
            from .retention import read_retention_file
            read_retention_file(_file(root, retention_name), identity, version)
            if core.get("retention_sha256") is not None and core["retention_sha256"] != files[retention_name]["sha256"]:
                raise ValueError("备份核心保护材料与版本绑定不符")
            expected.add(retention_name)
        elif core.get("retention_sha256") is not None:
            raise ValueError("备份缺少版本绑定的保护材料")
    if "active.json" in files:
        active = _json(root / "active.json")
        if (active.get("identity") != identity or not isinstance(active.get("id"), str) or
                not _ID.fullmatch(active["id"]) or type(active.get("width")) is not int or
                not 32 <= active["width"] <= 512):
            raise ValueError("备份活动专家与基座或尺寸不兼容")
        matches = [name for name in files if name.endswith("/" + active["id"] + ".json")]
        if len(matches) != 1:
            raise ValueError("备份缺少活动专家")
        name = matches[0]
        layer = int(name.split('/')[0][6:])
        meta = _json(_file(root, name))
        weights = name[:-5] + ".safetensors"
        if (meta.get("identity") != f"{identity}:layer-{layer}" or
                type(meta.get("hidden")) is not int or meta["hidden"] <= 0 or
                type(meta.get("width")) is not int or meta["width"] != active["width"] or
                not isinstance(meta.get("requires_gate", False), bool) or
                weights not in files or meta.get("sha256") != files[weights]["sha256"]):
            raise ValueError("备份专家元数据不兼容")
        hidden, width = meta["hidden"], active["width"]
        if core_shape is not None and hidden != core_shape[0]:
            raise ValueError("备份核心与专家隐藏尺寸不兼容")
        if _tensor_shapes(_file(root, weights)) != {
                "gate.weight": [width, hidden], "up.weight": [width, hidden], "down.weight": [hidden, width]}:
            raise ValueError("备份专家权重尺寸不兼容")
        expected.update(("active.json", name, weights))
    router_names = ("router.json", "router.safetensors")
    if "active-router.json" in files:
        active_router = _json(_file(root, "active-router.json"))
        version = active_router.get("id")
        if not isinstance(version, str) or not re.fullmatch(r"router-[0-9a-f]{12}", version):
            raise ValueError("备份活动快脑版本无效")
        router_names = (version + ".json", version + ".safetensors")
        if (active_router.get("identity") != identity or not set(router_names) <= files.keys()
                or active_router.get("sha256") != files[router_names[1]]["sha256"]
                or active_router.get("metadata_sha256") != files[router_names[0]]["sha256"]):
            raise ValueError("备份活动快脑校验失败")
        router_record = _json(_file(root, router_names[0]))
        if router_record.get("id") != version or router_record.get("core_version") != active_router.get("core_version"):
            raise ValueError("备份活动快脑元数据不兼容")
        expected.add("active-router.json")
    if any(name in files for name in router_names):
        if not set(router_names) <= files.keys():
            raise ValueError("备份决策头文件不完整")
        router = _json(_file(root, router_names[0]))
        for field, default in (("calibrated", False), ("gates_expert", False)):
            if not isinstance(router.get(field, default), bool):
                raise ValueError("备份决策头状态无效")
        for field, default in (("temperature", 1.0), ("threshold", 0.8)):
            value = router.get(field, default)
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    (value <= 0 if field == "temperature" else not 0 <= value <= 1)):
                raise ValueError("备份决策头校准参数无效")
        actions = router.get("supported_actions", ["local_core", "teacher"])
        if (not isinstance(actions, list) or not actions or
                any(action not in ("local_core", "neural_expert", "teacher") for action in actions)):
            raise ValueError("备份决策头类别无效")
        shapes = _tensor_shapes(_file(root, router_names[1]))
        hidden = shapes.get("projection.weight", [0, 0])
        if (router.get("identity") != identity or
                router.get("sha256") != files[router_names[1]]["sha256"] or
                len(hidden) != 2 or hidden[0] != 3 or hidden[1] <= 0 or
                shapes != {"projection.weight": hidden, "projection.bias": [3]} or
                (meta is not None and hidden[1] != meta["hidden"]) or
                (core_shape is not None and hidden[1] != core_shape[0])):
            raise ValueError("备份决策头与基座或尺寸不兼容")
        # A newly promoted expert legitimately leaves the old head stale. Preserve
        # that exact state; Runtime's version gate must continue to fail closed.
        if router.get("gates_expert") and (
                not isinstance(router.get("expert_id"), str) or
                not _ID.fullmatch(router["expert_id"]) or
                not isinstance(router.get("expert_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", router["expert_sha256"])):
            raise ValueError("备份决策头专家绑定格式无效")
        expected.update(router_names)
    if set(files) != expected:
        raise ValueError("备份包含未绑定文件")


def _read_backup(store, identity, backup_id):
    if not isinstance(backup_id, str) or not _BACKUP_ID.fullmatch(backup_id):
        raise ValueError("无效的备份编号")
    root = store / "training-backups" / backup_id
    if root.is_symlink() or root.parent.is_symlink():
        raise ValueError("备份目录不能使用符号链接")
    if (root / "manifest.json").is_symlink():
        raise ValueError("备份清单不能使用符号链接")
    manifest = _json(root / "manifest.json")
    if manifest.get("id") != backup_id or manifest.get("identity") != identity:
        raise ValueError("备份与当前基座不兼容")
    if (manifest.get("kind") not in {"learn", "calibrate"} or
            not isinstance(manifest.get("created_at"), str)):
        raise ValueError("备份记录不完整")
    datetime.fromisoformat(manifest["created_at"])
    _validate(root, identity, manifest.get("files"))
    return root, manifest


def save_training_backup(store, identity, kind):
    """Save persisted active core, expert and head before training mutates them."""
    if kind not in {"learn", "calibrate"}:
        raise ValueError("无效的训练类型")
    store = require_storage(store)
    if store.is_symlink():
        raise ValueError("模型存储目录不能使用符号链接")
    directory = store / "training-backups"
    if directory.is_symlink():
        raise ValueError("备份目录不能使用符号链接")
    directory.mkdir(exist_ok=True)
    names = [name for name in _ROOT_FILES if _file(store, name).exists()]
    if "active-router.json" in names:
        names = [name for name in names if name not in {"router.json", "router.safetensors"}]
        active_router = _json(_file(store, "active-router.json"))
        version = active_router.get("id")
        if not isinstance(version, str) or not re.fullmatch(r"router-[0-9a-f]{12}", version):
            raise ValueError("无效的活动快脑版本")
        names.extend((version + ".json", version + ".safetensors"))
    if "active-core.json" in names:
        active_core = _json(_file(store, "active-core.json"))
        version = active_core.get("version")
        if not isinstance(version, str) or not _CORE_ID.fullmatch(version):
            raise ValueError("无效的活动核心版本")
        core_names = (version + ".json", version + ".safetensors")
        if any(not _file(store, name).is_file() for name in core_names):
            raise ValueError("找不到完整活动核心文件")
        names.extend(core_names)
        retention_name = version + ".retention.json"
        if _file(store, retention_name).exists():
            names.append(retention_name)
    if "active.json" in names:
        active = _json(store / "active.json")
        expert_id = active.get("id")
        if not isinstance(expert_id, str) or not _ID.fullmatch(expert_id):
            raise ValueError("无效的活动专家编号")
        paths = list(store.glob(f"layer-*/{expert_id}.json"))
        if len(paths) != 1:
            raise ValueError("找不到唯一活动专家文件")
        name = paths[0].relative_to(store).as_posix()
        names.extend((name, name[:-5] + ".safetensors"))
    backup_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=directory) as temporary:
        root = Path(temporary)
        files = {}
        for name in names:
            source, target = _file(store, name), _file(root, name)
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            files[name] = {"size": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        _validate(root, identity, files)
        private_json(root / "manifest.json", {"id": backup_id, "identity": identity, "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(), "files": files})
        os.replace(root, directory / backup_id)
    retained = sorted(p for p in directory.iterdir() if p.is_dir() and _BACKUP_ID.fullmatch(p.name))
    def deleted_already(function, path, error):
        # ExFAT can remove an AppleDouble sidecar when its paired file is removed.
        if not isinstance(error, FileNotFoundError):
            raise error
    for old in retained[:-3]:
        if not old.is_symlink():
            shutil.rmtree(old, onexc=deleted_already)
    return backup_id


def list_training_backups(store, identity):
    directory = Path(store) / "training-backups"
    if not directory.exists():
        return []
    if directory.is_symlink():
        raise ValueError("备份目录不能使用符号链接")
    result = []
    for path in sorted(directory.iterdir(), reverse=True):
        if not path.is_dir() or not _BACKUP_ID.fullmatch(path.name):
            continue
        try:
            root, manifest = _read_backup(Path(store), identity, path.name)
            active = _json(root / "active.json") if "active.json" in manifest["files"] else None
            core = _json(root / "active-core.json") if "active-core.json" in manifest["files"] else None
            result.append({"id": path.name, "kind": manifest["kind"], "created_at": manifest["created_at"],
                "active_expert": active["id"] if active else None,
                "core_version": core["version"] if core else None, "valid": True})
        except (OSError, ValueError, KeyError, TypeError, struct.error):
            result.append({"id": path.name, "valid": False})
    return result


def restore_training_backup(store, identity, backup_id):
    """Validate everything first, then replace files; undo a failed publication."""
    store = require_storage(store)
    try:
        root, manifest = _read_backup(store, identity, backup_id)
    except (OSError, KeyError, TypeError, struct.error) as error:
        raise ValueError("备份缺失或格式损坏") from error
    names = set(manifest["files"]) | _ROOT_FILES
    ordered_names = sorted(names, key=lambda name: (name in {"active.json", "active-core.json", "active-router.json"},
                                                   name == "active-core.json", name))
    # Stage the whole replacement and a rollback copy before touching live files.
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=store) as temporary:
        staging = Path(temporary)
        before = {}
        for name in names:
            current = _file(store, name)
            before[name] = current.exists()
            if current.exists():
                target = staging / "before" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(current, target)
            if name in manifest["files"]:
                target = staging / "after" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_file(root, name), target)
        try:
            # Publish the active pointer last; runtime must exclude concurrent tasks.
            for name in ordered_names:
                target = _file(store, name)
                if name in manifest["files"]:
                    target.parent.mkdir(exist_ok=True)
                    os.replace(staging / "after" / name, target)
                else:
                    target.unlink(missing_ok=True)
        except Exception:
            for name in ordered_names:
                existed = before[name]
                target = _file(store, name)
                if existed:
                    shutil.copyfile(staging / "before" / name, target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return {"id": backup_id, "kind": manifest["kind"]}
