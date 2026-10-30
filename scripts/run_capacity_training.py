"""Research-only N50/100/200/300 continuous2/4/6; default waits without model imports.

Only separately sealed, explicitly approved one-attempt protocols execute.
Each N requires its own original-start protocol, output, once marker and MODEL GO.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import capacity_training_steps as steps
import run_capacity_qualification as qualification
import run_training_method_pilot as prior
import select_capacity_rows as select
from run_cached_training_audit import ROOT, IDENTITY, ORIGINAL, sha, content_sha, safe, write_new, production_snapshot

PROTOCOL = "fictional-capacity-n50-100-200-300-continuous-check-policy-v3"


def historical_reference():
    """Full actual B1 TRAIN/baseline source, never its learned candidate weights."""
    descriptor_path = ROOT / "research/training_method_pilot_eos_v2_result.json"
    descriptor = select.read_json(descriptor_path)
    raw_path = safe(descriptor["result_path"])
    post_path = safe(descriptor["root_actual_post_audit_path"])
    if (sha(raw_path) != descriptor["result_sha256"] or sha(post_path) != descriptor["root_actual_post_audit_sha256"]
            or descriptor.get("passed") is not True or descriptor.get("process_exit_code") != 0
            or descriptor.get("pid_dead") is not True):
        raise ValueError("Completed actual B1/root post history binding required")
    raw, post = select.read_json(raw_path), select.read_json(post_path)
    if (raw.get("protocol") != "original57-native-cache-b1-pilot150-tokenizer-eos-v2"
            or raw.get("passed") is not True or raw.get("preserved") is not True
            or raw.get("model_released") is not True or raw.get("actual_os_lock_reacquired_closed") is not True
            or len(raw.get("restoration", {})) != 9 or any(v is not True for v in raw["restoration"].values())
            or post.get("passed") is not True or post.get("all_checks_passed") is not True
            or post.get("raw_sha256") != descriptor["result_sha256"] or post.get("process_exit_code") != 0
            or post.get("checks", {}).get("actual_pid_dead") is not True
            or post.get("checks", {}).get("root_actual_OS_lock_reacquired_closed") is not True
            or content_sha(raw.get("groups")) != raw.get("groups_sha256")):
        raise ValueError("Actual original B1 full groups/exit/recovery evidence missing")
    from run_tensor_method_pilot import training_materials
    training_materials(raw["groups"])
    protected, baseline = [], []
    for name in ("new", "replay"):
        rows = raw["groups"][name + "_training"]
        observed = raw["baseline"][name]
        passed = prior.correct(rows, observed)
        for row, proof in zip(rows, observed):
            if row["id"] in passed:
                protected.append(copy.deepcopy(row)); baseline.append(copy.deepcopy(proof))
    if (len(protected) != 68 or sum(select.provenance(r)["evolution_role"] == "new" for r in protected) != 13
            or sum(select.provenance(r)["evolution_role"] == "replay" for r in protected) != 55):
        raise ValueError("Exactly original13NEW+55REPLAY protection, without duplicate13 old tasks")
    replay = copy.deepcopy(raw["groups"]["replay_training"])
    if len(replay) != 57 or any(not select.eligible_learning_sample(row) for row in replay):
        raise ValueError("Exact57 eligible REPLAY TRAIN required")
    return {"descriptor_path": str(descriptor_path), "descriptor_sha256": sha(descriptor_path),
        "raw_path": str(raw_path), "raw_sha256": sha(raw_path), "root_post_path": str(post_path),
        "root_post_sha256": sha(post_path), "groups_sha256": raw["groups_sha256"],
        "rows": protected, "baseline": baseline, "replay_training": replay,
        "rows_sha256": content_sha(protected), "baseline_sha256": content_sha(baseline),
        "source_sha256": raw["source_after"], "official_model_binding": raw["official_after"]}


def prepare(output, *, capacity_n=None, check_policy=None, receipt_path=None, method=None, batch_size=None, wall_seconds=None):
    if (capacity_n is not None and (type(capacity_n) is not int or capacity_n not in select.COUNTS)
            or check_policy is not None and check_policy not in ("every50", "per_new_exposure")
            or method is not None and method not in ("native", "cache")
            or batch_size is not None and (type(batch_size) is not int or batch_size not in (1, 2, 4))
            or method == "native" and batch_size not in (None, 1)
            or wall_seconds is not None and not steps._positive(wall_seconds)):
        raise ValueError("Explicit N50/100/200/300, supported method/B and finite positive wall cap required")
    base = qualification.prepare(output)
    history = historical_reference()
    names = set(base["source_sha256"])
    names.update(str(ROOT / name) for name in ("scripts/run_capacity_training.py", "scripts/capacity_training_steps.py",
        "scripts/cached_method_steps.py", "scripts/verify_cached_batch_training.py", "scripts/run_tensor_method_pilot.py",
        "scripts/run_identity_coverage.py", "tests/test_capacity_training_steps.py", "tests/test_capacity_training_runner.py"))
    plan = {**base, "protocol": PROTOCOL, "state": "waiting_actual_400_qualification_root_post_N_and_method",
        "source_sha256": {str(safe(p)): sha(safe(p)) for p in sorted(names)}, "historical_reference": history,
        "qualification_receipt_path": str(safe(receipt_path)) if receipt_path is not None else None,
        "qualification_receipt_sha256": sha(safe(receipt_path)) if receipt_path is not None else None,
        "method": method, "tensor_batch": batch_size, "learning_rate": 1e-5,
        "requested_wall_seconds": wall_seconds, "capacity_n": capacity_n, "check_policy": check_policy,
        "schedule": None, "new_training": None,
        "optimizer_steps": None, "qualified_source_ready": False,
        "scope": "Fictional closed-context memory throughput; no DEV/negative-control generation, product acceptance or transfer claim",
        "checkpoint": "Every50CE/tail flushes optimizer groups. Declared checks save full candidate/master/Adam/RNG/progress; fixed2/4/6 markers collect actual two NEW proofs on same tensor",
        "failure": "Single protocol once; failed/timeout attempt preserves raw/checkpoints and restores original. No automatic resume/refill/retry.",
        "timing": "Wall cap includes preflight/load/prefix/baseline/SVD/cache/compute/check/hash/save/restore/final preservation. Earlier qualification400 wall separately reported and added to total preparation+training cost.",
        "qualification_generation_calls": 0, "training_generation_calls": None, "wall_seconds": wall_seconds}
    if receipt_path is None or any(value is None for value in (capacity_n, check_policy, method, batch_size, wall_seconds)):
        return plan
    receipt = select.read_json(safe(receipt_path))
    package, selection, train, fixture = steps._qualification(receipt)
    if fixture:
        raise ValueError("CPU fixture qualification cannot prepare actual capacity execution")
    if (receipt.get("historical_rows_sha256") != history["rows_sha256"]
            or receipt.get("historical_baseline_sha256") != history["baseline_sha256"]):
        raise ValueError("400 qualification receipt must bind full independently audited historical protection")
    final_qualification = select.read_json(receipt["result_path"])
    for source in (history["source_sha256"], final_qualification["source_after"]):
        if any(plan["source_sha256"][p] != value for p, value in source.items() if p in plan["source_sha256"]):
            raise ValueError("Shared execution sources differ from actual baseline/qualification; requalification required")
    selected = selection["selections"][str(capacity_n)]
    if selected["state"] == "waiting":
        plan["missing_per_cell"] = selected["missing_per_cell"]
        return plan
    by_id = {row["id"]: row for row in train}
    news = [by_id[value] for value in selected["ids"]]
    replay = history["replay_training"]
    schedule = steps.build_capacity_schedule(selected["ids"], [r["id"] for r in replay],
        method=method, batch_size=batch_size, learning_rate=1e-5, wall_seconds=wall_seconds, check_policy=check_policy)
    original_correct = [row for row, proof in zip(train, package["records"]) if proof["passed"]]
    plan.update(state="unbound_waiting_root_review", qualified_source_ready=True, new_training=news,
        schedule=schedule, optimizer_steps=schedule["optimizer_updates"],
        qualification_binding=package["binding"], qualification_wall_seconds=select.read_json(receipt["result_path"])["wall_seconds"],
        protected_original_corpus=original_correct, protected_corpus_sha256=content_sha(original_correct),
        training_generation_calls_upper_bound=(len(original_correct) + 68)
            + len(schedule["checks_ce_events"]) * (capacity_n + len(original_correct) + 68) + 3 * capacity_n)
    return plan


def bind(plan):
    if plan.get("qualified_source_ready") is not True:
        raise ValueError("No actual completed400/root post or explicit method: remain waiting")
    if prepare(plan["output"], capacity_n=plan["capacity_n"], check_policy=plan["check_policy"],
               receipt_path=plan["qualification_receipt_path"], method=plan["method"],
               batch_size=plan["tensor_batch"], wall_seconds=plan["requested_wall_seconds"]) != plan:
        raise ValueError("Fixed capacity source/material/history/qualification/method changed")
    from verify_router_coverage_study import model_binding
    official = model_binding(plan["model_path"], IDENTITY)
    if (content_sha(official) != plan["qualification_binding"]["official_sha256"]
            or official != plan["historical_reference"]["official_model_binding"]):
        raise ValueError("Official actual files differ from qualified original/history")
    return {**copy.deepcopy(plan), "official_model_binding": official,
        "production_snapshot": production_snapshot(plan["production_root"]), "state": "bound_waiting_root_seal_MODEL_GO"}


def validate_bound(plan):
    template = {**copy.deepcopy(plan), "state": "unbound_waiting_root_review", "sealed": False,
                "official_model_binding": None, "production_snapshot": None}
    if bind(template) != {**plan, "sealed": False, "state": "bound_waiting_root_seal_MODEL_GO"}:
        raise ValueError("Complete capacity execution binding changed")


def save_checkpoint(path, runtime, master, optimizer, reference, basis, ledger, sources):
    """Real tensor payload, fsynced once, readback weights_only; no resume endpoint."""
    import torch
    from luban.evolution import _target
    from verify_cached_training import tensor_sha
    state = optimizer.state_dict()
    candidate = _target(runtime)[1]
    payload = {"protocol": PROTOCOL, "candidate": candidate.detach().cpu().clone(), "master": master.detach().cpu().clone(),
        "optimizer": state, "reference_sha256": tensor_sha(reference), "basis_sha256": tensor_sha(basis),
        "rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state() if candidate.device.type == "mps" else None,
        "progress": copy.deepcopy(ledger), "source_sha256": sources, "identity": IDENTITY, "original_core": None}
    with safe(path).open("xb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    restored = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
    if (restored["progress"] != ledger or restored["source_sha256"] != sources
            or restored["identity"] != IDENTITY or restored["original_core"] is not None
            or restored["reference_sha256"] != payload["reference_sha256"]
            or restored["basis_sha256"] != payload["basis_sha256"]
            or restored["optimizer"]["param_groups"] != state["param_groups"]
            or not torch.equal(restored["candidate"], payload["candidate"])
            or not torch.equal(restored["master"], payload["master"])
            or not torch.equal(restored["rng"], payload["rng"])):
        raise ValueError("Actual saved full checkpoint readback mismatch")
    if payload["mps_rng"] is not None and not torch.equal(restored["mps_rng"], payload["mps_rng"]):
        raise ValueError("Actual saved MPS RNG readback mismatch")
    for key, values in state["state"].items():
        for name, value in values.items():
            if not torch.equal(restored["optimizer"]["state"][key][name], value.cpu()):
                raise ValueError("Actual saved Adam moments/step readback mismatch")
    return {"path": str(path), "file_sha256": sha(path), "candidate_tensor_sha256": tensor_sha(candidate),
        "fp32_master_sha256": tensor_sha(master), "optimizer_updates": ledger["optimizer_updates"],
        "payload_scope": "full_candidate_master_adam_rng_progress"}


def probe_records(runtime, rows, deadline, candidate, raw_path, phase):
    """Persist each completed probe, including partial checks before timeout/cancel."""
    result = []
    for row in rows:
        values = prior.probes(runtime, [row], deadline, candidate)
        if len(values) != 1: raise ValueError("Exactly one completed closed-book probe required")
        with raw_path.open("a") as stream:
            stream.write(json.dumps({"phase": phase, "observation": values[0]}, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        result.extend(values)
        if time.perf_counter() >= deadline: raise TimeoutError("Probe or raw fsync exhausted complete wall cap")
    return result


def arm(runtime, plan, receipt, original, basis, caches, baseline_loss, output, began):
    schedule = plan["schedule"]
    if (type(plan.get("capacity_n")) is not int or plan["capacity_n"] not in select.COUNTS
            or len(plan["new_training"]) != plan["capacity_n"]
            or [r["id"] for r in plan["new_training"]] != schedule["new_ids"]
            or schedule["ce_events"] != 12 * plan["capacity_n"]
            or plan.get("check_policy") != schedule["check_policy"]):
        raise ValueError("Bound N, complete selected rows and CE schedule count/order differ")
    import torch
    from cached_method_steps import train_method_step
    from luban.evolution import _target, _loss
    from verify_cached_training import _sync, tensor_sha
    parameter = _target(runtime)[1]
    rows = {r["id"]: r for r in plan["new_training"] + plan["historical_reference"]["replay_training"]}
    master = torch.nn.Parameter(original.float().clone()); reference = original.float()
    optimizer = torch.optim.Adam([master], lr=plan["learning_rate"], eps=1e-6)
    ledger = {"capacity_n": plan["capacity_n"], "check_policy": plan["check_policy"],
              "optimizer_updates": 0, "ce_events": 0,
              "per_id_exposures": {key: dict.fromkeys(ids, 0) for key, ids in
                 (("new_ce", schedule["new_ids"]), ("replay_ce", schedule["replay_ids"]), ("replay_kl", schedule["replay_ids"]))}}
    report = {"capacity_n": plan["capacity_n"], "check_policy": plan["check_policy"],
              "steps": [], "checks": [], "ledger": ledger, "completed": False,
              "optimization_seconds": 0., "check_seconds": 0., "checkpoint_save_seconds": 0., "publication": False}
    deadline = began + plan["wall_seconds"]
    protected = plan["protected_original_corpus"] + plan["historical_reference"]["rows"]
    def guard():
        if time.perf_counter() >= deadline: raise TimeoutError("Fixed capacity full-wall cap, no retries")
        if any(sha(p) != value for p, value in plan["source_sha256"].items()):
            raise ValueError("Frozen capacity execution sources changed")
    try:
        for segment in schedule["segments"]:
            guard()
            for batch in segment["batches"]:
                if time.perf_counter() >= deadline: raise TimeoutError("Capacity compute wall cap")
                samples, replays = ([rows[value] for value in batch[key]] for key in ("ce_ids", "replay_kl_ids"))
                _sync(parameter.device); started = time.perf_counter()
                actual = train_method_step(runtime, samples, replays, master, optimizer, reference, basis,
                    method=plan["method"], ce_role=batch["ce_role"], caches=caches if plan["method"] == "cache" else None)
                _sync(parameter.device); elapsed = time.perf_counter() - started
                if (actual.get("runtime_restored") is not True or actual.get("optimizer_steps") != 1
                        or actual.get("ce_role") != batch["ce_role"] or actual.get("ce_ids") != batch["ce_ids"]
                        or actual.get("replay_kl_ids") != batch["replay_kl_ids"]):
                    raise ValueError("Actual method changed fixed role/IDs/update/restoration")
                actual.update(schedule_batch=batch, synchronized_wall_seconds=elapsed)
                report["steps"].append(actual); report["optimization_seconds"] += elapsed
                ledger["optimizer_updates"] += 1; ledger["ce_events"] += len(samples)
                for key, ids in ((batch["ce_role"] + "_ce", batch["ce_ids"]), ("replay_kl", batch["replay_kl_ids"])):
                    for value in ids: ledger["per_id_exposures"][key][value] += 1
            if not segment["check_required"]:
                continue
            with torch.no_grad(): parameter.copy_(master.to(parameter.dtype))
            candidate = tensor_sha(parameter)
            current = {**copy.deepcopy(ledger), "candidate_tensor_sha256": candidate, "frozen_tensor_sha256": [candidate],
                "protected_ids": [row["id"] for row in protected], "baseline_replay_loss": baseline_loss}
            report["checks"].append(current)
            started = time.perf_counter()
            raw_path = output / "generation.raw.jsonl"
            current["new_first"] = probe_records(runtime, plan["new_training"], deadline, candidate, raw_path,
                str(ledger["ce_events"]) + ":new_first")
            current["frozen_tensor_sha256"].append(tensor_sha(parameter)); guard()
            current["new_second"] = probe_records(runtime, plan["new_training"], deadline, candidate, raw_path,
                str(ledger["ce_events"]) + ":new_second") if segment["double_new_proof"] else []
            if segment["double_new_proof"]:
                current["frozen_tensor_sha256"].append(tensor_sha(parameter)); guard()
            current["protected_observations"] = probe_records(runtime, protected, deadline, candidate, raw_path,
                str(ledger["ce_events"]) + ":protected")
            current["replay_loss"] = _loss(runtime, plan["historical_reference"]["replay_training"])
            current["frozen_tensor_sha256"].append(tensor_sha(parameter)); guard()
            report["check_seconds"] += time.perf_counter() - started
            state = optimizer.state[master]
            if int(float(state["step"])) != ledger["optimizer_updates"]:
                raise ValueError("Real Adam state step differs from actual update ledger")
            started = time.perf_counter()
            current["checkpoint_artifact"] = save_checkpoint(output / ("checkpoint-" + str(ledger["ce_events"]) + ".pt"),
                runtime, master, optimizer, reference, basis, ledger, plan["source_sha256"])
            report["checkpoint_save_seconds"] += time.perf_counter() - started
            current["elapsed_seconds"] = time.perf_counter() - began
            verdict = steps.observe_capacity_checkpoint(schedule, current, receipt, plan["historical_reference"])
            current["verdict"] = verdict
            with torch.no_grad(): parameter.copy_(original)
            (output / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            guard()
            print(json.dumps({"ce_events": ledger["ce_events"], "optimizer_updates": ledger["optimizer_updates"],
                "exposure_marker": segment["new_exposure_marker"], "candidate_growth": verdict["candidate_confirmed_growth"],
                "provisional_correct": len(verdict["provisional_correct_ids"]), "forgotten": len(verdict["forgotten_ids"]),
                "elapsed_seconds": current["elapsed_seconds"]}), flush=True)
            if not verdict["continue_allowed"]:
                raise ValueError("Original protection/mean loss/wall safety failed; stop without retry")
        report["completed"] = True
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        with torch.no_grad(): parameter.copy_(original)
        report["original_tensor_restored"] = tensor_sha(parameter) == tensor_sha(original)
        report["elapsed_seconds"] = time.perf_counter() - began
        if report["elapsed_seconds"] >= plan["wall_seconds"]: report["completed"] = False
        report["quality_passed"] = bool(report["completed"] and report["checks"]
            and report["checks"][-1].get("verdict", {}).get("candidate_confirmed_growth", 0) > 0)
        (output / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        report["elapsed_seconds"] = time.perf_counter() - began
        if report["elapsed_seconds"] >= plan["wall_seconds"]:
            report.update(completed=False, quality_passed=False, error={"type": "TimeoutError", "message": "Final arm save exceeded full wall cap"})
            (output / "training.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        master = optimizer = reference = state = None
    return report


def execute(path, approved_sha):
    if sha(safe(path)) != steps._sha(approved_sha): raise ValueError("Exact Root-approved protocol SHA required")
    plan = select.read_json(safe(path))
    if (plan.get("protocol") != PROTOCOL or plan.get("sealed") is not True
            or plan.get("state") != "sealed_waiting_explicit_MODEL_GO"
            or type(plan.get("capacity_n")) is not int or plan["capacity_n"] not in select.COUNTS):
        raise ValueError("Only separately sealed research N50/100/200/300 protocol can execute")
    began = time.perf_counter(); validate_bound(plan); deadline = began + plan["wall_seconds"]
    output = safe(plan["output"])
    if output.exists(): raise FileExistsError("Capacity output already used")
    write_new(safe(path).parent / "execution-started.json", {"protocol_sha256": approved_sha, "started": time.time()})
    output.mkdir(mode=0o700)
    report = {"protocol": PROTOCOL, "protocol_sha256": approved_sha, "capacity_n": plan["capacity_n"],
        "check_policy": plan["check_policy"],
        "pid": os.getpid(), "passed": False,
        "model_started": False, "teacher_calls": 0, "publication": False, "product_accepted": False,
        "source_before": plan["source_sha256"], "production_before": plan["production_snapshot"],
        "qualification_wall_seconds": plan["qualification_wall_seconds"], "training": None}
    write_new(output / "run-started.json", report)
    runtime = parameter = original = flags = modes = hooks = backward = rng = mps_rng = native_basis = basis = caches = resources = None
    try:
        import torch
        from luban.runtime import Runtime
        from luban.memory import Memory
        from luban.settings import Settings
        from luban.process_lock import ModelProcessLock
        from luban.evolution import _target, _capture_replay_subspace, _loss
        from run_identity_coverage import verify_training_prefix_alignment
        from verify_cached_training import capture_tail, tensor_sha
        from verify_automatic_batches import measured
        receipt = select.read_json(plan["qualification_receipt_path"])
        replay = plan["historical_reference"]["replay_training"]
        with tempfile.TemporaryDirectory(prefix="luban-capacity-n" + str(plan["capacity_n"]) + "-") as home:
            settings = Settings(home); settings.update({"storage_root": str(output / "private"),
                "model_path": plan["model_path"], "temperature": 0, "local_thinking": "off"})
            runtime = Runtime(settings, Memory(output / "private"), research=True)
            if str(runtime.device) != plan["device"] or str(runtime.dtype) != plan["dtype"]:
                raise ValueError("Fixed actual MPS/BF16 hardware binding changed")
            runtime._model_process_lock = ModelProcessLock(IDENTITY, runtime.device)
            report["hardware"] = {"platform": platform.platform(), "torch": torch.__version__, "seed": 42,
                                  "device": plan["device"], "dtype": plan["dtype"]}
            with measured(runtime) as resources:
                runtime.load(paging=False); report["model_started"] = True
                parameter = _target(runtime)[1]; original = parameter.detach().clone()
                if runtime.identity != IDENTITY or runtime.active_core is not None or runtime.branch.expert is not None or tensor_sha(original) != ORIGINAL:
                    raise ValueError("Only originalNone/1706 full graph branchOFF")
                flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
                modes = [(m, m.training) for m in runtime.model.modules()]
                hooks = [(m, len(m._forward_hooks), len(m._forward_pre_hooks)) for m in runtime.model.modules()]
                backward = [(p, len(p._backward_hooks or {})) for p in runtime.model.parameters()]
                pointer = Path(runtime.store) / "active-core.json"; pointer_before = pointer.read_bytes() if pointer.exists() else None
                rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone() if runtime.device == "mps" else None
                torch.manual_seed(42); runtime.model.eval().requires_grad_(False); parameter.requires_grad_(True)
                if qualification.generation_eos(runtime) != plan["qualification_binding"]["eos_token_ids"]:
                    raise ValueError("Actual generation stop EOS differs from audited qualification")
                groups = {"new_training": plan["new_training"], "replay_training": replay}
                report["prefix"] = verify_training_prefix_alignment(runtime, {"prefix_alignment_required": True,
                    "groups_before_original_baseline": groups, "replay_training_candidates": []})
                protected = plan["protected_original_corpus"] + plan["historical_reference"]["rows"]
                report["original_protection_baseline"] = probe_records(runtime, protected, deadline, ORIGINAL,
                    output / "generation.raw.jsonl", "original_protection_baseline")
                if prior.correct(protected, report["original_protection_baseline"]) != {r["id"] for r in protected}:
                    raise ValueError("Actual same original run no longer meets all known protections; no optimizer")
                baseline_loss = _loss(runtime, replay)
                if not steps._positive(baseline_loss): raise ValueError("Finite positive original replay mean loss required")
                native_basis, report["replay_svd"] = _capture_replay_subspace(runtime, parameter, replay)
                basis = native_basis.to(parameter.device); caches = {}
                cache_started = time.perf_counter()
                if plan["method"] == "cache":
                    for row in plan["new_training"] + replay:
                        if time.perf_counter() >= deadline: raise TimeoutError("Full cache preparation exhausted wall cap")
                        caches[row["id"]] = capture_tail(runtime, row)
                report["cache_build_seconds"] = time.perf_counter() - cache_started
                report["cache_bytes"] = sum(sum(c["bytes"].values()) for c in caches.values())
                report["common_setup_seconds"] = time.perf_counter() - began
                report["training"] = arm(runtime, plan, receipt, original, basis, caches, baseline_loss, output, began)
                report["passed"] = report["training"]["completed"] and report["training"]["original_tensor_restored"]
                report["resources"] = resources
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if resources is not None: report["resources"] = resources
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
            parameter = original = flags = modes = hooks = backward = rng = mps_rng = native_basis = basis = caches = p = m = None
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
            report["source_after"] = {p: sha(p) for p in plan["source_sha256"]}
            report["official_after"] = model_binding(plan["model_path"], IDENTITY)
            report["production_after"] = production_snapshot(plan["production_root"])
            report["preserved"] = (report["source_after"] == plan["source_sha256"]
                and report["official_after"] == plan["official_model_binding"]
                and report["production_after"] == plan["production_snapshot"])
            # Revalidate final qualification/root receipt and historical bindings after all computation.
            receipt = select.read_json(plan["qualification_receipt_path"])
            steps._qualification(receipt)
            report["input_receipts_preserved"] = (sha(plan["qualification_receipt_path"]) == plan["qualification_receipt_sha256"]
                and historical_reference() == plan["historical_reference"] and sha(select.CORPUS) == select.CORPUS_SHA256)
        except BaseException as error:
            report["preservation_error"] = {"type": type(error).__name__, "message": str(error)}
        report["wall_seconds"] = time.perf_counter() - began
        report["total_with_prior_qualification_seconds"] = report["wall_seconds"] + plan["qualification_wall_seconds"]
        report["passed"] = (report["passed"] and not any(key.endswith("error") for key in report)
            and report.get("preserved") is True and report.get("input_receipts_preserved") is True
            and len(report.get("restoration", {})) == 9 and all(v is True for v in report["restoration"].values())
            and report.get("model_released") is True and report.get("actual_os_lock_reacquired_closed") is True
            and report["wall_seconds"] < plan["wall_seconds"])
        result_path = output / "result.json"
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        report["wall_seconds"] = time.perf_counter() - began
        report["total_with_prior_qualification_seconds"] = report["wall_seconds"] + plan["qualification_wall_seconds"]
        if report["wall_seconds"] >= plan["wall_seconds"]:
            report.update(passed=False, terminal_reason="Full final-save wall cap exceeded; no refill/resume")
            result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if report["passed"] is not True: raise RuntimeError("Capacity attempt rejected/incomplete; raw retained, no retries")


def main():
    parser = argparse.ArgumentParser(description=__doc__); actions = parser.add_mutually_exclusive_group()
    for name in ("prepare", "bind", "seal", "run"): actions.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--receipt", type=Path)
    parser.add_argument("--capacity-n", type=int, choices=select.COUNTS)
    parser.add_argument("--check-policy", choices=("every50", "per_new_exposure"))
    parser.add_argument("--method", choices=("native", "cache")); parser.add_argument("--batch-size", type=int, choices=(1, 2, 4))
    parser.add_argument("--wall-seconds", type=float); parser.add_argument("--approved-sha")
    args = parser.parse_args()
    if args.approved_sha and not args.run: parser.error("approved-sha only applies to run")
    if any(value is not None for value in (args.capacity_n, args.check_policy, args.receipt, args.method, args.batch_size, args.wall_seconds)) and not args.prepare:
        parser.error("N/receipt/method/B/wall parameters only apply to preparation; execution uses sealed values")
    if args.output and not (args.prepare or args.bind or args.seal): parser.error("output only applies to static plan preparation")
    if args.run:
        if not args.approved_sha: parser.error("Separate Root MODEL GO and exact approved SHA required")
        execute(args.run, args.approved_sha)
    elif args.prepare:
        if not args.output: parser.error("Fresh future output required")
        write_new(args.prepare, prepare(args.output, capacity_n=args.capacity_n, check_policy=args.check_policy,
            receipt_path=args.receipt, method=args.method,
            batch_size=args.batch_size, wall_seconds=args.wall_seconds))
    elif args.bind or args.seal:
        if not args.output: parser.error("New plan output required")
        plan = select.read_json(safe(args.bind or args.seal))
        if args.bind: write_new(args.output, bind(plan))
        else:
            validate_bound(plan); plan.update(sealed=True, state="sealed_waiting_explicit_MODEL_GO"); write_new(args.output, plan)
    else:
        print(json.dumps({"protocol": PROTOCOL, "state": "waiting_actual400_qualification_root_post_N_method_MODEL_GO",
            "model_started": False, "tokenizer_used": False, "teacher_calls": 0, "publication": False}))


if __name__ == "__main__": main()
