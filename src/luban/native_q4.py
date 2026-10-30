"""Lifecycle approval for the ordinary Q4 carrier; source weights stay untouched."""
import hashlib
import json
from pathlib import Path
import stat
import secrets

from .native_engine import NativeEngine, _digest
from .settings import private_json, require_storage
from .context import available_memory_bytes, GIB

FORMAT = "luban-direct-q4-g16-v1"
PREPARE_WORKSPACE_BYTES = 128 << 20
CORE_LOAD_WORKSPACE_BYTES = 2 * 2048 * 6144 * 2


def _stamp(path):
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Q4 来源或载体必须是普通文件")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def binding(runtime, binary, *, prepare=False, stop=None):
    """Credit Q4 bytes only for this verified capability and unchanged carrier."""
    if (getattr(runtime, "native_inference_precision", None) != "mixed-q4-g16"
            or getattr(runtime, "native_skip_layer_mask", 0) != 0
            or getattr(runtime, "native_exit_after", 24) != 24
            or getattr(runtime, "native_layer_paging", False)):
        return None
    receipt_path = Path(str(binary) + ".json")
    receipt_stamp = _stamp(receipt_path)
    previous = getattr(runtime, "_native_q4_approval", None)
    if previous is not None and previous["receipt_stamp"] == receipt_stamp and previous["binary"] == str(binary):
        state = previous
    else:
        receipt = json.loads(receipt_path.read_text())
        capability = receipt.get("direct_q4")
        if not isinstance(capability, dict) or capability.get("qualified") is not True:
            runtime._native_q4_approval = None
            return None
        quantizer = capability.get("quantizer_sha256")
        source_binary = receipt.get("packaging", {}).get("source_binary_sha256", receipt.get("binary_sha256"))
        if (capability.get("format") != FORMAT or not _digest(quantizer) or quantizer != source_binary
                or type(capability.get("weight_bytes")) is not int or capability["weight_bytes"] <= 0):
            raise ValueError("普通Q4加载资格与当前转换实现不一致")
        model = Path(runtime.loaded_path)
        manifest_path = model / "upstream-file-manifest.json"
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
        sources = [manifest_path] + [model / item["Path"] for item in manifest["Data"]["Files"]
                                     if item["Path"].endswith(".safetensors") or item["Path"] == "config.json"]
        for path in sources:
            if path.parent != model:
                raise ValueError("普通Q4基座来源路径无效")
        source_stamps = {str(path): _stamp(path) for path in sources}
        root = require_storage(runtime.loaded_storage) / "runtime" / "direct-q4"
        cache = root / (runtime.identity + "-" + quantizer + ".safetensors")
        state = {"binary": str(binary), "receipt_stamp": receipt_stamp,
                 "source_stamps": source_stamps, "identity": runtime.identity,
                 "model": str(model), "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                 "quantizer_sha256": quantizer, "weight_bytes": capability["weight_bytes"],
                 "file": str(cache), "approved": None}
        runtime._native_q4_approval = state
    if state["identity"] != runtime.identity or state["model"] != str(Path(runtime.loaded_path)):
        runtime._native_q4_approval = None
        return binding(runtime, binary, prepare=prepare, stop=stop)
    if any(_stamp(path) != stamp for path, stamp in state["source_stamps"].items()):
        runtime._native_q4_approval = None
        raise ValueError("普通Q4已绑定的基座来源发生变化，请重新加载模型")
    cache = Path(state["file"])
    approval_path = cache.with_suffix(".json")
    expected = {key: state[key] for key in ("identity", "model", "manifest_sha256", "quantizer_sha256", "weight_bytes", "source_stamps")}
    approval = state["approved"]
    if approval is None and cache.exists() and approval_path.exists():
        _stamp(approval_path)
        approval = json.loads(approval_path.read_text())
    if approval is not None:
        if (any(approval.get(key) != value for key, value in expected.items())
                or not _digest(approval.get("sha256")) or not cache.exists()
                or type(approval.get("payload_bytes")) is not int
                or not 0 < approval["payload_bytes"] <= state["weight_bytes"] - CORE_LOAD_WORKSPACE_BYTES
                or _stamp(cache) != approval.get("file_stamp")):
            state["approved"] = None
            raise ValueError("普通Q4载体缺失、损坏或来源不一致")
        state["approved"] = approval
        return {"file": str(cache), "sha256": approval["sha256"],
                "weight_bytes": approval["payload_bytes"], "cold_weight_bytes": state["weight_bytes"],
                "quantizer_sha256": state["quantizer_sha256"]}
    if not prepare:
        return None
    if cache.exists() or cache.is_symlink() or approval_path.exists() or approval_path.is_symlink():
        raise ValueError("普通Q4载体缺少完整批准收据")
    if available_memory_bytes() < GIB + PREPARE_WORKSPACE_BYTES:
        raise ValueError("当前可用内存不足以准备普通Q4载体，请关闭占用内存的应用后重试")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + "." + secrets.token_hex(8) + ".partial")
    engine = NativeEngine(binary, runtime.loaded_path, state["manifest_sha256"], load_timeout=1800)
    try:
        event = engine.prepare_direct_q4(temporary, state["quantizer_sha256"], stop)
        if event["direct_q4_weight_bytes"] + CORE_LOAD_WORKSPACE_BYTES > state["weight_bytes"]:
            raise ValueError("普通Q4实际载入内存与已验证预算不一致")
        if any(_stamp(path) != stamp for path, stamp in state["source_stamps"].items()):
            raise ValueError("普通Q4准备期间基座来源发生变化")
        if cache.exists() or approval_path.exists():
            raise ValueError("普通Q4载体发布期间目标发生变化")
        temporary.rename(cache)
        approval = {**expected, "sha256": event["direct_q4_sha256"],
                    "payload_bytes": event["direct_q4_weight_bytes"], "file_stamp": _stamp(cache)}
        private_json(approval_path, approval)
        state["approved"] = approval
    finally:
        temporary.unlink(missing_ok=True)
    return binding(runtime, binary)
