"""Install the pinned, accepted product expert as a new user's baseline."""
import json
from pathlib import Path


MANIFEST = json.loads(Path(__file__).with_name("factory-expert.json").read_text())


def require_empty_storage(root):
    from .memory import Memory
    # First install may have an empty database from opening the UI. Do not
    # switch away from conversations or material that the user already added.
    with Memory(root).connect() as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in ("samples", "conversations", "events", "usage", "teacher_materials", "training_runs"):
            if name in tables and db.execute("SELECT 1 FROM " + name + " LIMIT 1").fetchone():
                raise ValueError("已有使用或学习资料，请导入完整模型，原资料保持不变")


def validate(root):
    from . import model_packages as packages
    from . import mlx_model_packages as mlx
    payload = packages._read(root / "accepted.json")
    packages._validate_journal(payload)
    if (payload["identity"] != MANIFEST["identity"]
            or set(payload.get("factory_runs", [])) != set(MANIFEST["factory_runs"])
            or {run["id"] for run in payload["runs"]} != set(MANIFEST["factory_runs"])
            or payload["samples"] or payload["material_archives"]):
        raise ValueError("初始专家的出厂记录不一致")
    # Archives were verified while downloading. Reuse the pinned file records
    # for graph/report validation; don't rehash multi-GB base weights here.
    files = {name: row for name, row in MANIFEST["files"].items()
             if name.startswith(("modules/", "runtime/")) or name == "accepted.json"}
    packages._validate_modules(root, payload, files)
    mlx.validate_tree(root, {"files": {**mlx.trusted_files(), **files,
                                      "models/mlx/download-receipt.json": {}}}, payload)
    mlx.check_activation(root, {"native_inference_precision": "mlx-q4",
                               "native_mlx_model_path": str(root / "models/mlx")})
    return payload


def install_journal(root, payload):
    from .model_packages import _install_journal
    return _install_journal(root, payload)
