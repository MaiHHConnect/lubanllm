"""Pinned MLX model and independently checked local-trial expert bindings."""
import json
from pathlib import Path

from .native_engine import _digest
from .native_q4 import _stamp


def model_binding(runtime):
    if getattr(runtime, "native_inference_precision", None) != "mlx-q4":
        return None
    path = Path(getattr(runtime, "native_mlx_model_path", ""))
    if not path.is_absolute():
        raise ValueError("MLX 模型目录必须是绝对路径")
    files = (path / "download-receipt.json", path / "config.json", path / "model.safetensors")
    stamps = {str(item): _stamp(item) for item in files}
    cached = getattr(runtime, "_native_mlx_binding", None)
    if cached is not None and cached[1]["path"] == str(path):
        if cached[0] != stamps:
            raise ValueError("MLX 已绑定模型来源发生变化，请重新加载模型")
        return dict(cached[1])
    receipt = json.loads(files[0].read_text())
    weights = [item for item in receipt.get("files", []) if item.get("name") == "model.safetensors"]
    config = json.loads(files[1].read_text())
    text = config.get("text_config", {})
    if (receipt.get("complete") is not True or len(weights) != 1
            or weights[0].get("downloaded") is not True or not _digest(weights[0].get("sha256"))
            or type(weights[0].get("bytes")) is not int or weights[0]["bytes"] != stamps[str(files[2])][2]
            or not isinstance(receipt.get("revision"), str) or not receipt["revision"]
            or config.get("model_type") != "qwen3_5"
            or tuple(text.get(key) for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"))
            != (2048, 6144, 24, 248320)
            or config.get("quantization", {}).get("bits") != 4):
        raise ValueError("MLX 模型缺少完整下载绑定或与当前2B图不兼容")
    binding = {"path": str(path), "sha256": weights[0]["sha256"], "weight_bytes": weights[0]["bytes"],
               "revision": receipt["revision"]}
    runtime._native_mlx_binding = (stamps, binding)
    return dict(binding)


def expert_binding(runtime, binary):
    """A trial proof never grants formal quality, mastery or training admission."""
    if (binary is None or getattr(runtime, "native_inference_precision", None) != "mlx-q4"
            or getattr(runtime, "active_core", None) is None):
        return None
    if getattr(runtime, "router_meta", {}).get("feature_source") != "mlx-q4-prefix":
        return None
    proof_path = Path(runtime.loaded_storage) / "runtime/mlx/expert-proof.json"
    branch = runtime.branch
    expert_id = getattr(branch, "expert_id", None)
    if not proof_path.is_file() or not expert_id or not runtime.pointer.exists():
        return None
    proof = json.loads(proof_path.read_text())
    weights, metadata = branch._paths(expert_id)
    record = json.loads(metadata.read_text())
    active = json.loads(runtime.pointer.read_text())
    core = json.loads((runtime.store / "active-core.json").read_text())
    receipt = json.loads(Path(str(binary) + ".json").read_text())
    from .factory_engine import worker_qualified
    if not worker_qualified(proof, receipt.get("binary_sha256")):
        raise ValueError("MLX 专家尚未通过当前引擎的独立验证")
    model = model_binding(runtime)
    formal = proof.get("schema") == "mlx-expert-learning-v1"
    scope = "product" if formal else "local-trial"
    expected = {"schema": "mlx-expert-learning-v1" if formal else "mlx-expert-local-trial-v1",
                "scope": scope, "quality_admitted": formal, "identity": runtime.identity,
                "expert_id": expert_id, "core_version": runtime.active_core,
                "core_file_sha256": core.get("sha256"), "expert_file_sha256": record.get("sha256"),
                "worker_binary_sha256": proof.get("worker_binary_sha256"), "mlx_model_sha256": model["sha256"],
                "width": record.get("width"), "route_checked": True,
                "router_id": runtime.router_meta.get("id"), "router_sha256": runtime.router_meta.get("sha256")}
    expected.update({"training_admitted": True} if formal else {"math_passed": True, "runtime_checked": True})
    if (any(type(proof.get(key)) is not type(value) or proof.get(key) != value for key, value in expected.items())
            or core.get("identity") != runtime.identity or core.get("version") != runtime.active_core
            or active.get("id") != expert_id or active.get("identity") != runtime.identity
            or active.get("core_version") != runtime.active_core or record.get("core_version") != runtime.active_core
            or record.get("identity") != runtime.identity + ":layer-23" or record.get("requires_gate") is not True
            or runtime.router_meta.get("execution_scope") != scope
            or runtime.router_meta.get("quality_admitted") is not formal
            or runtime.router_meta.get("inference_engine" if formal else "trial_engine") != "mlx-q4"
            or runtime.router_meta.get("expert_id") != expert_id
            or runtime.router_meta.get("expert_sha256") != record.get("sha256")
            or runtime.router_meta.get("core_version") != runtime.active_core
            or not isinstance(proof.get("router_id"), str) or not proof["router_id"]
            or record.get("hidden") != 2048 or type(record.get("width")) is not int
            or not 1 <= record["width"] <= 6144 or record["width"] != active.get("width")
            or not all(_digest(proof.get(key)) for key in ("core_file_sha256", "expert_file_sha256",
                                                          "worker_binary_sha256", "mlx_model_sha256", "router_sha256"))):
        raise ValueError("MLX 专家试用证明与当前模型、核心、专家或生成引擎不一致")
    if formal:
        from .mlx_learning import formal_report
        if any(value.get("execution_scope") != "product" or value.get("inference_engine") != "mlx-q4"
               or value.get("quality_admitted") is not True or value.get("training_admitted") is not True
               for value in (active, record, runtime.router_meta)):
            raise ValueError("MLX正式专家来源标记不完整")
        formal_report(runtime, proof)
    return {"file": str(weights), "sha256": record["sha256"], "width": record["width"],
            "core_sha256": core["sha256"]}


def require_training_source(runtime):
    """Keep user-selected trial adapters out of accepted-parent training paths."""
    store = getattr(runtime, "store", None)
    if store is None:
        return
    for name, prefix in (("active.json", "layer-23/"), ("active-router.json", "")):
        path = Path(store) / name
        if not path.exists():
            continue
        pointer = json.loads(path.read_text())
        selected = pointer.get("id")
        if not isinstance(selected, str) or not selected or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in selected):
            raise ValueError("训练来源版本无效")
        metadata = json.loads((Path(store) / (prefix + selected + ".json")).read_text())
        for record in (pointer, metadata):
            if record.get("inference_engine") == "mlx-q4":
                raise ValueError("MLX专家须使用原生MLX学习和验证流程，不能交给旧精度训练")
            if (record.get("execution_scope") == "local-trial" or record.get("migration", {}).get("scope") == "local-trial"
                    or record.get("quality_admitted") is False or record.get("training_admitted") is False):
                raise ValueError("当前MLX专家组合仅获本地试用许可；恢复已正式采纳来源后可继续学习和无限进化")
