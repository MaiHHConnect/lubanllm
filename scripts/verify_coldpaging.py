"""Real native-versus-paged Qwen3.5 equivalence, operator audit and residency."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import uuid

import torch

from luban.coldpaging import ColdBackbone, full_backbone
from luban.evolution import _target, _memory
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import DEFAULTS, Settings, private_json, require_storage
from verify_training_operators import audit_training_operators, TOLERANCES


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-core", required=True, help="Exact version or original")
    args = parser.parse_args()
    expected = None if args.expected_core == "original" else args.expected_core
    root = require_storage(DEFAULTS["storage_root"])
    run = root / "experiments" / ("coldpaging-v1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    prompts = ["请计算二加三，只回答数字。", "将字符串abc转换为大写，只输出转换后的字符串。"]
    protocol = {"expected_core": expected, "prompts": prompts, "seed": 42,
        "thinking": "off", "temperature": 0, "max_new_tokens": 32,
        "baseline": "official full upstream Runtime.load(paging=False), selected core applied",
        "checks": "first4 features and last-token logits exact; two different-length cached generations exact; tied embedding/head unchanged; real gradient/Adam audit passed; teacher choice has zero disk loads; cold layer meta state after every operation",
        "operator_tolerances": TOLERANCES, "teacher_calls": 0, "training_performed": False,
        "memory_method": "phase RSS/current MPS allocated/driver snapshots; not peak; do not sum overlapping measures",
        "limits": "startup still full-loads; text paging only, no vision API or memory/speed benchmark"}
    private_json(run / "protocol.json", protocol)
    report = {"passed": False, "protocol": protocol, "run_directory": str(run), "phases": [], "checks": [],
              "hardware": platform.platform(), "teacher_calls": 0}
    started = time.perf_counter()
    try:
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(prefix="coldpaging-settings-") as folder:
            settings = Settings(folder)
            settings.update({"model_path": DEFAULTS["model_path"], "storage_root": str(root),
                             "local_thinking": "off", "temperature": 0})
            runtime = Runtime(settings, Memory(root))
            runtime.load(paging=False)
            if runtime.active_core != expected or runtime.branch.expert is not None:
                raise RuntimeError("Use the preregistered selected core with no active branch")
            original_sha = tensor_sha(_target(runtime)[1])
            pointer = runtime.store / "active-core.json"
            pointer_before = pointer.read_bytes() if pointer.exists() else None

            def phase(name):
                if runtime.device == "mps":
                    torch.mps.synchronize()
                item = {"phase": name, **_memory(runtime),
                        "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if runtime.device == "mps" else 0,
                        "paging": runtime.cold_backbone.status() if runtime.cold_backbone else {"enabled": False}}
                report["phases"].append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)

            phase("official_full")
            baselines = []
            with torch.no_grad():
                for prompt in prompts:
                    tokens = runtime._tokens([{"role": "user", "content": prompt}], "off")
                    features = runtime.shallow(tokens).detach().cpu()
                    logits = runtime.model(tokens, use_cache=False).logits[:, -1].detach().cpu()
                    output = runtime.model.generate(tokens, max_new_tokens=32, do_sample=False, pad_token_id=runtime.tokenizer.eos_token_id)
                    baselines.append({"tokens": tokens.cpu(), "features": features, "logits": logits, "output": output.cpu()})
            del features, logits, output, tokens
            runtime.cold_backbone = ColdBackbone(runtime, runtime.loaded_path)
            runtime.cold_backbone._evict()
            phase("cold_after_official")

            def assert_cold():
                pager = runtime.cold_backbone
                if pager.loaded or any(not value.is_meta for layer in runtime.decoder.layers[4:] for value in layer.parameters()):
                    raise RuntimeError("Cold layers still hold resident tensors")
                if runtime.model.lm_head.weight is not runtime.decoder.embed_tokens.weight:
                    raise RuntimeError("Paging broke tied embedding/head")
                if (pointer.read_bytes() if pointer.exists() else None) != pointer_before:
                    raise RuntimeError("Paging changed the active core pointer")

            assert_cold()
            for index, baseline in enumerate(baselines):
                tokens = baseline["tokens"].to(runtime.device)
                loads = runtime.cold_backbone.loads
                features = runtime.shallow(tokens).cpu()
                if not torch.equal(features, baseline["features"]) or runtime.cold_backbone.loads != loads:
                    raise RuntimeError("Shallow representation changed or loaded cold layers")
                with full_backbone(runtime), torch.no_grad():
                    logits = runtime.model(tokens, use_cache=False).logits[:, -1].cpu()
                    output = runtime.model.generate(tokens, max_new_tokens=32, do_sample=False, pad_token_id=runtime.tokenizer.eos_token_id).cpu()
                    weight_matches = tensor_sha(_target(runtime)[1]) == original_sha
                    record = {"case": index, "shallow_exact": True, "logits_exact": torch.equal(logits, baseline["logits"]),
                              "logits_max_abs_diff": float((logits.float() - baseline["logits"].float()).abs().max()),
                              "cached_generation_exact": torch.equal(output, baseline["output"]), "selected_core_exact": weight_matches}
                    report["checks"].append(record)
                    if not all(record[key] for key in ("shallow_exact", "logits_exact", "cached_generation_exact", "selected_core_exact")):
                        raise RuntimeError("Native and paged model diverged")
                assert_cold()
                phase("cold_after_generation_" + str(index))
            del features, logits, output, tokens, baselines
            with full_backbone(runtime):
                sample = {"id": "coldpaging-operator", "prompt": prompts[0], "answer": "5",
                          "purpose": "operator_audit", "training_allowed": False}
                report["operator_audit"] = audit_training_operators(runtime, sample)
                if not report["operator_audit"]["passed"] or not report["operator_audit"]["restored"]:
                    raise RuntimeError("Paged real training operators failed")
            assert_cold()
            loads = runtime.cold_backbone.loads
            decision = runtime.choose_route([{"role": "user", "content": "示例教师路由"}], "teacher")
            if decision["route"] != "teacher" or runtime.cold_backbone.loads != loads:
                raise RuntimeError("Teacher selection loaded local cold layers")
            phase("cold_after_operator_and_teacher_selection")
            report.update(passed=True, teacher_selection_disk_loads=0, selected_core_sha256=original_sha,
                          core_version=runtime.active_core, device=runtime.device, precision=str(runtime.dtype))
    except BaseException as error:
        report["error"] = str(error) or type(error).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        private_json(run / "result.json", report)
        private_json(Path(__file__).resolve().parents[1] / "research/coldpaging_v1.json", report)
        print(json.dumps({"passed": report["passed"], "run_directory": str(run), "error": report.get("error")}), flush=True)


if __name__ == "__main__":
    main()
