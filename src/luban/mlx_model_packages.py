"""Portable MLX payloads, preserving immutable adoption and engine qualifications."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
from types import SimpleNamespace
import zipfile

from . import model_packages as p
from .settings import private_json

PROTOCOL = "native-mlx-expert-preservation-v1"
MLX_FILES = {
    "config.json": {"size": 3113, "sha256": "beb7fc5a6e0405fe332821cf1a8ef7b69bb390a8c8933171647de5579debf949"},
    "model.safetensors": {"size": 1722271785, "sha256": "713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5"},
    "README.md": {"size": 2273, "sha256": "34decba3365c0f0e712aec28f0e49f82283bc22dc26174131ffee5279012b23c"},
    "model.safetensors.index.json": {"size": 81722, "sha256": "8294c05cca7d53a6c33e3db2b379539bd296d054e0b689711b16b6ac93c7e49d"},
    "chat_template.jinja": {"size": 7755, "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"},
    "preprocessor_config.json": {"size": 390, "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"},
    "processor_config.json": {"size": 1300, "sha256": "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b"},
    "tokenizer.json": {"size": 19989343, "sha256": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"},
    "tokenizer_config.json": {"size": 1139, "sha256": "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182"},
    "video_preprocessor_config.json": {"size": 385, "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"},
    "vocab.json": {"size": 6722759, "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"},
    "LICENSE": {"size": 11544, "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"},
}
REVISION = "93760be4f1f69842a46bc13dbdc0f19e291392a3"
PROOFS = ("runtime/mlx/expert-proof.json", "runtime/mlx/training-proof.json")
POINTER_NAMES = ("active.json", "active-router.json", PROOFS[0])
TRAINING_GATES = ("prepare_equivalent", "real_adam_updates", "checkpoint_replay_exact",
                  "cancellation_atomic", "continued_after_cancel", "saved_expert_reload_equal", "base_core_frozen")


def allowed_name(name):
    return name in (*PROOFS, *("models/mlx/" + item for item in (*MLX_FILES, "download-receipt.json")))


def trusted_files():
    # The unquantized graph description/tokenizer fixes the original identity;
    # its deleted BF16 shard and obsolete shard index are not runtime inputs.
    return {**{"models/base/" + name: record for name, record in p.TRUSTED["base"].items()
               if not name.endswith(".safetensors") and name not in {"README.md", "model.safetensors.index.json"}},
            **{"models/asr/" + name: record for name, record in p.TRUSTED["asr"].items()},
            **{"models/mlx/" + name: record for name, record in MLX_FILES.items()}}


def portable_run(run, body, runtime):
    result = copy.deepcopy(run)
    result["binding"] = copy.deepcopy(body["binding"])
    expected = (Path(runtime.store) / "active.json", Path(runtime.store) / "active-router.json",
                Path(runtime.settings.snapshot()["storage_root"]) / PROOFS[0])
    plans = result["commit"]["pointers"]
    if len(plans) != 3 or any(set(plan) != {"path", "before", "after"}
                             or plan["path"] != str(path) for plan, path in zip(plans, expected)):
        raise ValueError("MLX采纳指针不属于当前资料目录")
    result["commit"]["pointers"] = [{"name": name, "before": plan["before"], "after": plan["after"]}
                                    for name, plan in zip(POINTER_NAMES, plans)]
    return result


def installed_run(run, root):
    result = copy.deepcopy(run)
    for plan in result["commit"]["pointers"]:
        name = plan.pop("name")
        plan["path"] = str(root / name if name in PROOFS else root / "modules/luban0-v1" / name)
    return result


def upgrade_record(store, entry, event):
    """Verify an accepted MLX journal using its own protocol before counting it."""
    body = entry["body"]
    run = {key: copy.deepcopy(body[key]) for key in ("id", "kind", "binding", "records", "started", "commit", "arguments")}
    run.update(state=entry["state"], scope=body.get("scope", "product"), product_accepted=body.get("product_accepted", False))
    root = store.parent.parent
    owner = SimpleNamespace(store=store, settings=SimpleNamespace(snapshot=lambda: {"storage_root": str(root)}))
    run = portable_run(run, body, owner)
    validate_run(run, set())
    if event != {"kind": "learning", "body": run["commit"]["report"]}:
        raise ValueError("MLX正式学习缺少匹配的持久事件")
    names = set(run["commit"]["artifacts"])
    names.update(run["binding"]["core_version"] + suffix for suffix in (".json", ".safetensors", ".retention.json"))
    files = {"modules/luban0-v1/" + name: p._record(store / name) for name in names}
    if any(files["modules/luban0-v1/" + name]["sha256"] != digest for name, digest in run["commit"]["artifacts"].items()):
        raise ValueError("MLX正式学习文件已变化")
    validate_candidate(store, run, files)
    from .upgrades import number
    report = run["commit"]["report"]
    completed = number(run["started"]) + number(report["elapsed_seconds"])
    if not 0 < completed <= time.time() + 300:
        raise ValueError("MLX学习完成时间无效")
    return {"base_sha256": p.TRUSTED_IDENTITY, "weights_sha256": report["expert_file_sha256"],
            "baseline_validation_loss": number(report["loss"]["baseline_validation"]["mean"]),
            "validation_loss": number(report["loss"]["validation"]["mean"]),
            "replay_before": None, "replay_loss": None,
            "training_samples": len(run["records"]["train"]), "validation_samples": len(run["records"]["valid"]),
            "completed_at": completed, "kind": "expert"}


def validate_run(run, seen):
    required = {"id", "kind", "state", "scope", "product_accepted", "binding", "records", "started", "commit", "arguments"}
    if (set(run) != required or not p.re.fullmatch(r"[a-f0-9]{12}", run.get("id", ""))
            or run["id"] in seen or run["kind"] != "learning" or run["state"] != "committed"
            or run["scope"] != "product" or type(run["started"]) not in {int, float}
            or not 0 < run["started"] <= time.time() + 300
            or run["arguments"].get("protocol") != PROTOCOL
            or run["binding"].get("identity") != p.TRUSTED_IDENTITY
            or run["binding"].get("arguments") != run["arguments"]
            or run["binding"].get("records_sha256") != hashlib.sha256(p._encoded(run["records"])).hexdigest()
            or set(run["records"]) != {"train", "valid", "protection"}):
        raise ValueError("MLX采纳账本、来源或材料绑定不完整")
    list(p._record_rows(run["records"]))
    commit = run["commit"]
    if (set(commit) != {"report", "pointers", "artifacts", "event_kind"}
            or commit["event_kind"] != "learning" or not isinstance(commit["artifacts"], dict)
            or not isinstance(commit["pointers"], list) or len(commit["pointers"]) != 3
            or any(set(plan) != {"name", "before", "after"} or plan["name"] != name
                   for plan, name in zip(commit["pointers"], POINTER_NAMES))):
        raise ValueError("MLX采纳必须绑定专家、快脑和证明三个指针")
    for name, digest in commit["artifacts"].items():
        if not p._MODULE.fullmatch(name) or name in p._POINTERS or not isinstance(digest, str) or not p.HEX.fullmatch(digest):
            raise ValueError("MLX采纳文件路径或摘要无效")
    report, proof = commit["report"], commit["pointers"][2]["after"]
    expert, router = "learned-" + run["id"], "router-" + run["id"]
    names = {"layer-23/" + expert + suffix for suffix in (".json", ".safetensors")}
    names |= {router + suffix for suffix in (".json", ".safetensors")}
    names.add("learning-" + run["id"] + ".json")
    if (set(commit["artifacts"]) != names or report.get("records") != run["records"]
            or report.get("records_sha256") != run["binding"]["records_sha256"]
            or report.get("active_expert") != expert or report.get("router_id") != router
            or proof.get("expert_id") != expert or proof.get("router_id") != router
            or proof.get("report") != "learning-" + run["id"] + ".json"
            or proof.get("report_sha256") != commit["artifacts"].get(proof.get("report"))):
        raise ValueError("MLX采纳报告与专家、路由或材料不一致")
    for key in ("identity", "core_version", "core_file_sha256", "mlx_model_sha256", "worker_binary_sha256", "router_sha256"):
        if proof.get(key) != run["binding"].get(key):
            raise ValueError("MLX采纳证明更改了冻结来源")
    p._no_private(run)


def _proof_runtime(store, proof):
    from .mlx_learning import formal_report
    expert, router, core = [p._read(store / name) for name in ("active.json", "active-router.json", "active-core.json")]
    metadata = p._read(store / "layer-23" / (expert["id"] + ".json"))
    router_meta = p._read(store / (router["id"] + ".json"))
    expected = {"schema": "mlx-expert-learning-v1", "scope": "product", "quality_admitted": True,
                "training_admitted": True, "identity": p.TRUSTED_IDENTITY, "expert_id": expert["id"],
                "core_version": core["version"], "core_file_sha256": core["sha256"],
                "expert_file_sha256": metadata["sha256"], "width": metadata["width"],
                "router_id": router["id"], "router_sha256": router_meta["sha256"], "route_checked": True,
                "mlx_model_sha256": MLX_FILES["model.safetensors"]["sha256"]}
    flags = {"execution_scope": "product", "inference_engine": "mlx-q4", "quality_admitted": True,
             "training_admitted": True, "core_version": core["version"]}
    if (any(type(proof.get(key)) is not type(value) or proof[key] != value for key, value in expected.items())
            or not isinstance(proof.get("worker_binary_sha256"), str) or not p.HEX.fullmatch(proof["worker_binary_sha256"])
            or any(any(record.get(key) != value for key, value in flags.items()) for record in (expert, metadata, router_meta))
            or router_meta.get("feature_source") != "mlx-q4-prefix" or router_meta.get("expert_id") != expert["id"]
            or router_meta.get("expert_sha256") != metadata["sha256"] or metadata.get("requires_gate") is not True):
        raise ValueError("MLX迁移证明与实际核心、专家或路由不一致")
    owner = SimpleNamespace(store=store, identity=p.TRUSTED_IDENTITY, active_core=core["version"])
    return formal_report(owner, proof)


def validate_candidate(store, run, files):
    """Validate historical pointers against their own frozen core and exact files."""
    from .backups import _validate
    with tempfile.TemporaryDirectory(prefix=".mlx-candidate-", dir=store.parent) as folder:
        stage = Path(folder)
        version = run["binding"]["core_version"]
        if not isinstance(version, str) or not p.re.fullmatch(r"core-[a-f0-9]{12}", version):
            raise ValueError("MLX采纳核心名称无效")
        core = p._read(store / (version + ".json"))
        core_names = {version + suffix for suffix in (".json", ".safetensors", ".retention.json")}
        artifacts = run["commit"]["artifacts"]
        subset = {}
        for name in core_names | set(artifacts):
            entry = files.get("modules/luban0-v1/" + name)
            if entry is None:
                raise ValueError("MLX采纳历史缺少冻结核心或文件")
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(store / name, target)
            if not name.startswith("learning-"):
                subset[name] = entry
        core_pointer = {"version": version, "identity": p.TRUSTED_IDENTITY, "sha256": core["sha256"]}
        private_json(stage / "active-core.json", core_pointer)
        subset["active-core.json"] = p._record(stage / "active-core.json")
        for plan in run["commit"]["pointers"][:2]:
            private_json(stage / plan["name"], plan["after"])
            subset[plan["name"]] = p._record(stage / plan["name"])
        _validate(stage, p.TRUSTED_IDENTITY, subset)
        proof = run["commit"]["pointers"][2]["after"]
        if _proof_runtime(stage, proof) != run["commit"]["report"]:
            raise ValueError("MLX采纳账本与不可变报告不一致")


def validate_tree(root, manifest, payload):
    files = manifest["files"]
    permitted = set(trusted_files()) | {"models/mlx/download-receipt.json", "accepted.json", *PROOFS}
    if any(not name.startswith("modules/luban0-v1/") and name not in permitted for name in files):
        raise ValueError("MLX模型包含额外基座或未许可运行文件")
    if not set(PROOFS) <= files.keys() or "models/mlx/download-receipt.json" not in files:
        raise ValueError("MLX模型包缺少下载或采纳/训练资格证明")
    receipt = p._read(root / "models/mlx/download-receipt.json")
    weights = [item for item in receipt.get("files", []) if item.get("name") == "model.safetensors"]
    expected = MLX_FILES["model.safetensors"]
    if (receipt.get("complete") is not True or receipt.get("revision") != REVISION or len(weights) != 1
            or weights[0].get("downloaded") is not True or weights[0].get("sha256") != expected["sha256"]
            or weights[0].get("bytes") != expected["size"] or type(receipt.get("finished")) not in {int, float}
            or not 0 < receipt["finished"] <= time.time() + 300):
        raise ValueError("MLX下载来源不匹配固定可信模型")
    p._no_private(receipt)
    proof = p._read(root / PROOFS[0])
    store = root / "modules/luban0-v1"
    _proof_runtime(store, proof)
    if not any(run["commit"]["pointers"][-1].get("after") == proof for run in payload["runs"]
               if run.get("arguments", {}).get("protocol") == PROTOCOL):
        raise ValueError("当前MLX证明没有匹配的正式采纳账本")
    training = p._read(root / PROOFS[1])
    if (training.get("schema") != "mlx-native-training-qualification-v1" or training.get("complete") is not True
            or any(training.get(key) != proof[key] for key in ("worker_binary_sha256", "core_file_sha256", "mlx_model_sha256"))
            or any(training.get("gates", {}).get(key) is not True for key in TRAINING_GATES)):
        raise ValueError("MLX训练资格未绑定当前引擎、核心与模型")
    # Qualification logs name the algorithm; they do not contain optimizer
    # tensor state. Keep the immutable proof, only exempt this exact label.
    checked_training = copy.deepcopy(training)
    for event in checked_training.get("events", []):
        value = event.get("event", {}) if isinstance(event, dict) else {}
        if value.get("optimizer") == "Adam-FP32-beta0.9-0.999-eps1e-6":
            del value["optimizer"]
    p._no_private(checked_training)


def restore_weight_time(root):
    # The content was verified during extraction; preserve the original immutable
    # verification timestamp instead of pretending the old proof is a new test.
    finished = p._read(root / "models/mlx/download-receipt.json")["finished"]
    os.utime(root / "models/mlx/model.safetensors", (finished, finished))


def installed_binary():
    # Activation runs before Runtime.load, so there is intentionally no loaded
    # graph for product_inference.native_binary's graph admission checks yet.
    backend = Path(__file__).resolve().parents[2]
    return next((path for path in (backend / "native/luban-mlx-worker", backend / "build/native/luban-mlx-worker")
                 if path.is_file()), None)


def check_activation(root, staged):
    if (staged.get("native_inference_precision") != "mlx-q4"
            or staged.get("native_mlx_model_path") != str(root / "models/mlx")):
        raise ValueError("MLX暂存路径已被修改")
    binary = installed_binary()
    if binary is None:
        raise ValueError("当前应用缺少MLX推理引擎，原模型保持不变")
    receipt = p._read(Path(str(binary) + ".json"))
    proof = p._read(root / PROOFS[0])
    from .factory_engine import worker_qualified
    if not worker_qualified(proof, receipt.get("binary_sha256")):
        raise ValueError("导入模型的MLX引擎资格与当前应用版本不同；请使用导出时的应用版本，或重新校准验证；原模型保持不变")


def export_package(runtime, target, *, progress=None):
    config = runtime.settings.snapshot()
    if config.get("native_inference_precision") != "mlx-q4" or not config.get("native_mlx_model_path"):
        raise ValueError("MLX模型路径尚未配置，无法完整导出")
    payload = p._journal(runtime)
    sources = {}
    fields = {"base": "model_path", "asr": "speech_model_path", "mlx": "native_mlx_model_path"}
    roots = {key: p._plain(Path(config[field]).resolve(strict=True), directory=True) for key, field in fields.items()}
    trusted = trusted_files()
    for name in trusted:
        _, key, relative = name.split("/", 2)
        sources[name] = p._plain(roots[key] / relative)
        if key == "mlx" and relative == "LICENSE" and not sources[name].exists():
            sources[name] = p._plain(roots["base"] / "LICENSE")
    sources["models/mlx/download-receipt.json"] = p._plain(roots["mlx"] / "download-receipt.json")
    store = Path(runtime.store)
    names = p._active_names(store)
    names.update(name for run in payload["runs"] for name in run["commit"]["artifacts"])
    for name in names:
        p._name("modules/luban0-v1/" + name)
        sources["modules/luban0-v1/" + name] = p._plain(store / name)
    for name in PROOFS:
        sources[name] = p._plain(Path(config["storage_root"]) / name)
    with tempfile.TemporaryDirectory(prefix=".mlx-package-export-", dir=target.parent) as directory:
        scratch = Path(directory)
        private_json(scratch / "accepted.json", payload)
        sources["accepted.json"] = scratch / "accepted.json"
        # Trusted model hashes are already pinned. Check them while copying into
        # the archive once, not by rescanning multi-GB weights in preparation.
        files = {name: trusted.get(name) or p._record(path) for name, path in sources.items()}
        manifest = {"schema": p.SCHEMA, "kind": "learned", "engine": "mlx-q4",
                    "identity": p.TRUSTED_IDENTITY, "files": files}
        manifest["package_id"] = hashlib.sha256(p._encoded(manifest)).hexdigest()
        total = p._check_manifest(manifest)
        for name, path in sources.items():
            if name.startswith(("modules/", "runtime/")) or name == "models/mlx/download-receipt.json":
                destination = scratch / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
        p._validate_modules(scratch, payload, files)
        validate_tree(scratch, manifest, payload)
        temporary = scratch / "result.zip"
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.writestr("manifest.json", p._encoded(manifest), compress_type=zipfile.ZIP_DEFLATED)
            done = 0
            for name, path in sources.items():
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                info.compress_type = zipfile.ZIP_STORED if name.endswith(".safetensors") else zipfile.ZIP_DEFLATED
                digest, size = hashlib.sha256(), 0
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as output:
                    for block in iter(lambda: source.read(p.CHUNK), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                        done += len(block)
                        if progress:
                            progress({"phase": "export", "bytes": done, "total": total})
                if {"size": size, "sha256": digest.hexdigest()} != files[name]:
                    raise ValueError("MLX导出来源发生变化或不符合可信模型")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        with target.open("xb"):
            pass
        os.replace(temporary, target)
    return {"path": str(target), "package_id": manifest["package_id"], "bytes": target.stat().st_size,
            "kind": "learned", "engine": "mlx-q4", "teacher_calls": 0}
