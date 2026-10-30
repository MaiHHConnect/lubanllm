"""One fixed400 TRAIN originalNone qualification; no training or teacher.

Default/prepare are stdlib only. Binding/sealing/execution need separate root
authorization. A partial run is terminal waiting; never resume or refill it.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time

import select_capacity_rows as select
import run_training_method_pilot as prior
from run_cached_training_audit import ROOT, PRIOR, PRIOR_SHA, IDENTITY, ORIGINAL, sha, content_sha, safe, write_new, production_snapshot

PROTOCOL = "fictional-capacity-original400-qualification-v1"
WALL_SECONDS = 1800


def prepare(output):
    if sha(PRIOR) != PRIOR_SHA:
        raise ValueError("Immutable historical model/source anchor changed")
    old = json.loads(PRIOR.read_text())
    model, product, output = safe(Path(old["model_path"]).resolve(strict=True)), safe(old["production_root"]), safe(output)
    for protected in (ROOT, product / "app", product / "modules", model, PRIOR.parent):
        if output == protected or protected in output.parents:
            raise ValueError("New qualification output must be outside source/model/product/history")
    if output.exists():
        raise FileExistsError("Unique qualification output must not exist")
    # Validate unchanged full432 material without labels or tokenizer execution.
    if select.select_capacity_rows()["state"] != "waiting":
        raise ValueError("Unexpected preexisting capacity qualification")
    rows = select._materials(select.read_json(select.CORPUS))
    names = set(old["execution_manifest"])
    names.update(str(ROOT / name) for name in ("scripts/run_capacity_qualification.py", "scripts/select_capacity_rows.py",
        "scripts/run_training_method_pilot.py", "scripts/run_cached_training_audit.py", "scripts/diagnose_field_training.py",
        "scripts/verify_cached_training.py", "scripts/verify_automatic_batches.py", "scripts/verify_router_coverage_study.py",
        "src/luban/process_lock.py", "src/luban/retention.py", "src/luban/generation_checks.py",
        "tests/test_capacity_qualification.py", "tests/test_capacity_selection.py"))
    # Reuse the already reviewed installed graph source paths, without importing
    # Torch or coupling qualification to any training-method quality result.
    graph_sources = select.read_json(ROOT / "research/training_method_pilot_eos_v2_final_preparation.json")["source_sha256"]
    names.update(p for p in graph_sources if Path(p).is_relative_to(ROOT / ".venv"))
    names.update(str(ROOT / ".venv/lib/python3.12/site-packages" / name) for name in (
        "transformers/generation/utils.py", "transformers/generation/configuration_utils.py"))
    manifest = model / "upstream-file-manifest.json"
    if len(json.loads(manifest.read_text())["Data"]["Files"]) + 1 != 13:
        raise ValueError("Fixed official12 members plus manifest required")
    return {"protocol": PROTOCOL, "state": "unbound_waiting_root_review", "sealed": False, "model_go": False,
        "output": str(output), "model_path": str(model), "production_root": str(product), "identity": IDENTITY,
        "original_core": None, "original_tensor_sha256": ORIGINAL, "device": "mps", "dtype": "torch.bfloat16", "seed": 42,
        "corpus_path": str(select.CORPUS), "corpus_sha256": select.CORPUS_SHA256,
        "training_ids": [r["id"] for r in rows], "training_records_sha256": content_sha(rows),
        "source_sha256": {str(safe(p)): sha(safe(p)) for p in sorted(names)},
        "official_manifest_sha256": sha(manifest), "official_model_binding": None, "production_snapshot": None,
        "max_new_tokens": 512, "temperature": 0, "branch_enabled": False, "reference_used": False,
        "thinking": "off", "do_sample": False, "qualification_generation_calls": 400, "qualification_once": True,
        "wall_seconds": WALL_SECONDS, "optimizer_steps": 0, "teacher_calls": 0, "publication": False,
        "timing": "Wall includes source/official validation, load, all generation, per-row durable raw, final preservation and release. Nonpreemptible generation/hash may overrun; then waiting, never success.",
        "failure": "Partial/incomplete/timeout is terminal waiting with raw preserved; no retry/refill/resume/substitution.",
        "scope": "Only400 fictional TRAIN qualification;32 unseen negative-control rows never generated or selected; no training/adoption/generalization."}


def bind(plan):
    from verify_router_coverage_study import model_binding
    if prepare(plan["output"]) != plan:
        raise ValueError("Qualification source/material/model path changed")
    return {**copy.deepcopy(plan), "official_model_binding": model_binding(plan["model_path"], IDENTITY),
        "production_snapshot": production_snapshot(plan["production_root"]), "state": "bound_waiting_root_seal_MODEL_GO"}


def validate_bound(plan):
    template = {**copy.deepcopy(plan), "state": "unbound_waiting_root_review", "sealed": False,
                "official_model_binding": None, "production_snapshot": None}
    if prepare(plan["output"]) != template or bind(template) != {
            **plan, "sealed": False, "state": "bound_waiting_root_seal_MODEL_GO"}:
        raise ValueError("Qualification complete binding changed")


def generation_eos(runtime):
    value = runtime.model.generation_config.eos_token_id
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values):
        raise ValueError("Actual generation EOS IDs required, independently of training tokenizer EOS")
    return sorted(values)


def qualify(runtime, rows, binding, deadline, record, sources):
    """One pass only. Caller durably records each completed actual observation."""
    select._binding(binding)
    result = []
    for row in rows:
        if time.perf_counter() >= deadline:
            raise TimeoutError("Fixed1800s qualification cap; no refill")
        if any(sha(path) != value for path, value in sources.items()):
            raise ValueError("Qualification sources changed")
        values = prior.probes(runtime, [row], deadline, expected_tensor_sha=ORIGINAL)
        if len(values) != 1:
            raise ValueError("One actual raw observation per fixed TRAIN required")
        observed = values[0]
        select.validate_observation(row, observed, binding)
        record(observed)
        result.append(observed)
        if time.perf_counter() >= deadline:
            raise TimeoutError("Generation or durable raw exceeded fixed cap; no next question")
    return result


def execute(path, approved_sha):
    if not isinstance(approved_sha, str) or re.fullmatch(r"[0-9a-f]{64}", approved_sha) is None or sha(safe(path)) != approved_sha:
        raise ValueError("Exact root-approved qualification protocol SHA required")
    plan = select.read_json(safe(path))
    if plan.get("protocol") != PROTOCOL or plan.get("sealed") is not True or plan.get("state") != "sealed_waiting_explicit_MODEL_GO":
        raise ValueError("Separately sealed qualification and MODEL GO required")
    began = time.perf_counter(); deadline = began + WALL_SECONDS
    validate_bound(plan)
    if time.perf_counter() >= deadline:
        raise TimeoutError("Preflight exhausted qualification wall cap")
    output = safe(plan["output"])
    if output.exists(): raise FileExistsError("Qualification output already consumed")
    write_new(safe(path).parent / "execution-started.json", {"protocol_sha256": approved_sha, "started": time.time()})
    output.mkdir(mode=0o700)
    report = {"protocol": PROTOCOL, "protocol_sha256": approved_sha, "pid": os.getpid(), "model_started": False,
        "state": "waiting", "passed": False, "teacher_calls": 0, "optimizer_steps": 0, "publication": False,
        "source_before": plan["source_sha256"], "production_before": plan["production_snapshot"], "records": [],
        "wall_cap_seconds": WALL_SECONDS, "started": time.time()}
    write_new(output / "run-started.json", report)
    runtime = parameter = original = flags = modes = rng = mps_rng = hooks = backward = None
    binding = records = resources = None
    try:
        import torch
        from luban.runtime import Runtime
        from luban.memory import Memory
        from luban.settings import Settings
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target
        from verify_cached_training import tensor_sha
        from verify_automatic_batches import measured
        report["hardware"] = {"platform": platform.platform(), "torch": torch.__version__, "seed": 42,
                              "device": plan["device"], "dtype": plan["dtype"]}
        rows = select._materials(select.read_json(plan["corpus_path"]))
        with tempfile.TemporaryDirectory(prefix="luban-capacity-qualification-") as home:
            settings = Settings(home); settings.update({"storage_root": str(output / "private"), "model_path": plan["model_path"],
                                                        "temperature": 0, "local_thinking": "off"})
            runtime = Runtime(settings, Memory(output / "private"), research=True)
            if str(runtime.device) != plan["device"] or str(runtime.dtype) != plan["dtype"]:
                raise ValueError("Qualification actual hardware/dtype changed")
            runtime._model_process_lock = ModelProcessLock(IDENTITY, runtime.device)
            with measured(runtime) as resources:
                runtime.load(paging=False); report["model_started"] = True
                parameter = _target(runtime)[1]; original = parameter.detach().clone()
                if runtime.identity != IDENTITY or runtime.active_core is not None or runtime.branch.expert is not None or tensor_sha(original) != ORIGINAL:
                    raise ValueError("Only actual originalNone/1706 full branchOFF graph")
                flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
                modes = [(m, m.training) for m in runtime.model.modules()]
                hooks = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backward = [(p, len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                pointer = Path(runtime.store) / "active-core.json"; pointer_before = pointer.read_bytes() if pointer.exists() else None
                rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone() if runtime.device == "mps" else None
                torch.manual_seed(42); runtime.model.eval().requires_grad_(False)
                binding = {"corpus_sha256": plan["corpus_sha256"], "base_identity": IDENTITY, "core_version": None,
                    "actual_tensor_sha256": ORIGINAL, "source_sha256": content_sha(plan["source_sha256"]),
                    "official_sha256": content_sha(plan["official_model_binding"]), "eos_token_ids": generation_eos(runtime),
                    "max_new_tokens": 512, "temperature": 0, "branch_enabled": False, "reference_used": False,
                    "thinking": "off", "do_sample": False, "scope": "actual_closed_original"}
                write_new(output / "binding.json", binding)
                def record(value):
                    with (output / "qualification.raw.jsonl").open("a") as stream:
                        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                    report["records"].append(value)
                    if len(report["records"]) % 25 == 0:
                        print(json.dumps({"completed": len(report["records"]), "total": 400,
                                          "wall_seconds": time.perf_counter() - began}), flush=True)
                records = qualify(runtime, rows, binding, deadline, record, plan["source_sha256"])
                report["selection"] = select.select_capacity_rows(plan["corpus_path"], {"binding": binding, "records": records}, binding)
            report["resources"] = resources
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if resources is not None:
            report["resources"] = resources
        if runtime is not None:
            try:
                import torch
                from verify_cached_training import tensor_sha
                if runtime.model is not None and original is not None:
                    with torch.no_grad(): parameter.copy_(original)
                if flags is not None and modes is not None and rng is not None:
                    for p, flag, grad in flags: p.requires_grad_(flag); p.grad = grad
                    for m, mode in modes: m.training = mode
                    torch.set_rng_state(rng)
                    if mps_rng is not None: torch.mps.set_rng_state(mps_rng)
                    report["restoration"] = {"original_tensor": tensor_sha(parameter) == ORIGINAL,
                        "flags_grads": all(p.requires_grad == f and p.grad is g for p, f, g in flags),
                        "modes": all(m.training == v for m, v in modes), "rng": torch.equal(rng, torch.get_rng_state()),
                        "mps_rng": mps_rng is None or torch.equal(mps_rng, torch.mps.get_rng_state()),
                        "core_branch": runtime.active_core is None and runtime.branch.expert is None,
                        "forward_hooks": all(len(m._forward_hooks) == a and len(m._forward_pre_hooks) == b for m, a, b in hooks),
                        "backward_hooks": all(len(p._backward_hooks or {}) == v for p, v in backward),
                        "pointer": (pointer.read_bytes() if pointer.exists() else None) == pointer_before}
            except BaseException as error:
                report["restoration_error"] = {"type": type(error).__name__, "message": str(error)}
            parameter = original = flags = modes = rng = mps_rng = hooks = backward = p = m = None
            try:
                runtime._release_model()
                from luban.process_lock import ModelProcessLock
                lock = ModelProcessLock(IDENTITY, plan["device"]); lock.close()
                report["model_released"] = runtime.model is None and runtime._model_process_lock is None
                report["actual_os_lock_reacquired_closed"] = True
            except BaseException as error:
                report["release_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            from verify_router_coverage_study import model_binding
            report["source_after"] = {name: sha(name) for name in plan["source_sha256"]}
            report["production_after"] = production_snapshot(plan["production_root"])
            report["official_after"] = model_binding(plan["model_path"], IDENTITY)
            report["preserved"] = (report["source_after"] == plan["source_sha256"] and report["production_after"] == plan["production_snapshot"]
                and report["official_after"] == plan["official_model_binding"] and sha(plan["corpus_path"]) == plan["corpus_sha256"])
        except BaseException as error:
            report["preservation_error"] = {"type": type(error).__name__, "message": str(error)}
        report["wall_seconds"] = time.perf_counter() - began
        report["passed"] = (not any(key.endswith("error") for key in report) and len(report["records"]) == 400
            and records is not None and report.get("preserved") is True and len(report.get("restoration", {})) == 9
            and all(report["restoration"].values()) and report.get("model_released") is True
            and report.get("actual_os_lock_reacquired_closed") is True and report["wall_seconds"] < WALL_SECONDS)
        if report["passed"]:
            write_new(output / "qualification.json", {"binding": binding, "records": records})
            report["qualification_sha256"] = sha(output / "qualification.json")
        report["wall_seconds"] = time.perf_counter() - began
        if report["wall_seconds"] >= WALL_SECONDS: report["passed"] = False
        report["state"] = "complete_waiting_root_evidence_review" if report["passed"] else "waiting_terminal_partial_or_failed_no_refill"
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        report["wall_seconds"] = time.perf_counter() - began
        if report["wall_seconds"] >= WALL_SECONDS and report["passed"]:
            report.update(passed=False, state="waiting_terminal_deadline_during_final_save")
            (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if not report["passed"]: raise RuntimeError("Qualification waiting/rejected; raw preserved, no refill or retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__); actions = parser.add_mutually_exclusive_group()
    for name in ("prepare", "bind", "seal", "run"): actions.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--approved-sha")
    args = parser.parse_args()
    if args.approved_sha and not args.run: parser.error("approved-sha only applies to run")
    if args.output and not (args.prepare or args.bind or args.seal): parser.error("output only applies to static preparation")
    if args.run:
        if not args.approved_sha: parser.error("Separate root MODEL GO and exact SHA required")
        execute(args.run, args.approved_sha)
    elif args.prepare or args.bind or args.seal:
        if not args.output: parser.error("Fresh output required")
        if args.prepare: write_new(args.prepare, prepare(args.output))
        elif args.bind: write_new(args.output, bind(select.read_json(safe(args.bind))))
        else:
            value = select.read_json(safe(args.seal)); validate_bound(value)
            value.update(sealed=True, state="sealed_waiting_explicit_MODEL_GO"); write_new(args.output, value)
    else:
        print(json.dumps({"protocol": PROTOCOL, "state": "waiting_root_source_review", "model_started": False,
            "torch_imported": False, "qualification_generation_calls": 400, "wall_cap_seconds": WALL_SECONDS,
            "optimizer_steps": 0, "teacher_calls": 0, "publication": False}))


if __name__ == "__main__": main()
