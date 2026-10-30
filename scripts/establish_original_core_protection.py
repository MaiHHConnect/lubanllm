"""Capture the finite ordinary-task baseline for the untouched product core.

The command only writes the explicitly supplied output artifact. It does not
train, publish a core, activate an expert, or modify the active pointer.
"""
import argparse
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import torch

from luban.evolution import _generation
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings
from verify_luban0_product import product_cases


def tensor_sha256(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [row for row in product_cases() if row["id"].startswith("product-ordinary-")]
    with tempfile.TemporaryDirectory(prefix="luban0-original-protection-settings-") as home:
        settings = Settings(home)
        settings.update({"model_path": args.model, "storage_root": args.storage,
                         "local_thinking": "off", "temperature": 0})
        _run(runtime := Runtime(settings, Memory(args.storage), research=True), args, rows)


def _run(runtime, args, rows):
    started = time.time()
    runtime.load(paging=True)
    try:
        if runtime.active_core is not None or runtime.branch.expert is not None:
            raise RuntimeError("原始基线必须在核心版本和专家分支都关闭时执行")
        results = _generation(runtime, rows)
        payload = {
            "schema_version": 1,
            "protocol": "original-core-protection-v1",
            "scope": "fixed ordinary product cases only; identity behavior intentionally excluded so a new identity expert can be trained separately",
            "created_at": time.time(),
            "identity": runtime.identity,
            "original_tensor_sha256": tensor_sha256(runtime._base_core_weight),
            "model_path": args.model,
            "device": runtime.device,
            "dtype": str(runtime.dtype),
            "torch_version": torch.__version__,
            "hardware": platform.platform(),
            "branch_enabled": any(row["branch_enabled"] for row in results),
            "cases": rows,
            "generation": results,
            "passed": all(row["passed"] for row in results),
            "elapsed_seconds": time.time() - started,
            "teacher_calls": 0,
            "training_performed": False,
            "production_mutations": [],
        }
        if not payload["passed"]:
            raise RuntimeError("原始核心普通任务基线未逐条通过；不写入资格载荷")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(output), "passed": True,
                          "cases": len(results), "identity": runtime.identity,
                          "device": runtime.device}, ensure_ascii=False))
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
