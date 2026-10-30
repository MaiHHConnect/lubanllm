"""Nine fixed TRAIN rows, original vs rejected832: no fitting or generation.

Default/prepare use stdlib only. Explicit bind reads CPU mmap tensors and actual
official file hashes; run additionally requires a separately approved plan SHA.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "research/field_training_diagnostic_records.json"
DATA_SHA = "6c92991ef49b752a60260522d1e4423e8a9455547cdeb05dec81cd5a24d0e61e"
PROTOCOL = "nine-field-training-readonly-diagnostic-v1"
IDS = {f"field-context-train-{attribute}-{form}"
       for attribute in ("name", "position", "tuner") for form in ("single", "system", "multi")}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Diagnostic paths cannot be symlinks")
    return path


def write_new(path, value):
    with safe(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def metadata():
    """No pickle/tensor import: only the already archived manifest/SQL/JSON."""
    if sha(DATA) != DATA_SHA:
        raise ValueError("Fixed nine-row TRAIN descriptor changed")
    data = json.loads(DATA.read_text())
    snapshot = safe(data["source_snapshot"])
    manifest_path = snapshot / "manifest.json"
    if sha(manifest_path) != data["source_manifest_sha256"]:
        raise ValueError("Immutable rejected-run archive manifest changed")
    members = {row["path"]: row for row in json.loads(manifest_path.read_text())["files"]}

    def archived(name):
        if name not in members or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Missing exact archive member")
        path = safe(snapshot / name)
        if sha(path) != members[name]["sha256"]:
            raise ValueError("Archived metadata SHA changed")
        return path

    protocol_path = archived("experiment/protocol.json")
    result_path = archived("experiment/result.json")
    db_path = archived("experiment/research-storage/app/memory.sqlite3")
    if sha(protocol_path) != data["prior_protocol_sha256"] or sha(result_path) != data["prior_result_sha256"]:
        raise ValueError("Rejected protocol/result binding changed")
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT state,body FROM training_runs WHERE id=? AND kind='evolution' AND identity=?",
                         (data["session_id"], data["identity"])).fetchone()
    if row is None:
        raise ValueError("Missing rejected research journal")
    body = json.loads(row[1])
    if (row[0] != "rejected" or body.get("scope") != "research" or body.get("id") != data["session_id"]
            or body["checkpoint"]["progress"] != {"phase": "training", "step": 832, "steps": 832}
            or body["binding"].get("core_version") is not None
            or body["binding"]["records_sha256"] != data["source_records_sha256"]
            or content_sha(body["records"]) != data["source_records_sha256"]):
        raise ValueError("Journal is not the exact rejected originalNone832 checkpoint")
    rows = [row for row in body["records"]["groups"]["new_training"] if row["id"] in IDS]
    if len(rows) != 9 or {row["id"] for row in rows} != IDS or rows != data["records"]:
        raise ValueError("Diagnostic may inspect only the fixed nine original TRAIN rows")
    checkpoint = data["candidate_relative_path"]
    expected_name = f"experiment/research-storage/modules/luban0-v1/training-runs/{data['session_id']}/{body['checkpoint']['name']}"
    if (checkpoint != expected_name or checkpoint not in members
            or members[checkpoint]["sha256"] != body["checkpoint"]["sha256"]
            or body["checkpoint"]["sha256"] != data["candidate_file_sha256"]):
        raise ValueError("Exact832 tensor member binding changed")
    # Do not open the .pt here. Actual file and tensor hashes are explicit bind work.
    candidate = safe(snapshot / checkpoint)
    if not candidate.is_file():
        raise ValueError("Archived832 checkpoint missing")
    old = json.loads(protocol_path.read_text())
    names = set(old["execution_manifest"]) | {str(Path(__file__).resolve()),
        str(ROOT / "scripts/verify_router_coverage_study.py"), str(ROOT / "scripts/verify_router_coverage_counts.py"),
        str(ROOT / "scripts/verify_automatic_batches.py")}
    return data, old, candidate, {name: sha(name) for name in sorted(names)}, body["binding"]


def prepare_plan(output):
    data, old, candidate, sources, binding = metadata()
    output = safe(output)
    product = safe(old["production_root"])
    # The historical official path is a deliberate alias; bind its physical target.
    model_path = safe(Path(old["model_path"]).resolve(strict=True))
    for protected in (data["source_snapshot"], product / "app", product / "modules", model_path, ROOT):
        if output == safe(protected) or safe(protected) in output.parents:
            raise ValueError("Unique diagnostic output must be outside production/model/archive/project")
    return {"protocol": PROTOCOL, "state": "waiting_explicit_bind_and_MODEL_GO", "sealed": False,
        "model_go": False, "output": str(output), "data_path": str(DATA), "data_sha256": DATA_SHA,
        "data": data, "source_sha256": sources, "candidate_path": str(candidate),
        "candidate_file_sha256": data["candidate_file_sha256"], "checkpoint_binding": binding,
        "model_path": str(model_path), "original_protocol_model_path": old["model_path"],
        "production_root": str(product),
        "identity": data["identity"], "device": binding["device"], "dtype": binding["dtype"], "seed": 42,
        "states": ["originalNone", "rejected832"], "row_ids": [row["id"] for row in data["records"]],
        "teacher_calls": 0, "generation_calls": 0, "optimizer_steps": 0, "tensor_binding": None,
        "official_model_binding": None, "production_snapshot": None,
        "precision_scope": "native dtype CE/gradient vs same-dtype captured tail; FP32 algebra casts native X and upstream derivative, never an FP32 full model",
        "memory_scope": "RSS/MPS snapshots every0.1s including load/diagnostic/restoration/release; sampled maxima, not physical peak"}


def validate_payload(saved, plan):
    """Only weights-only mmap payload; never instantiate/load an optimizer or RNG."""
    import torch
    data = plan["data"]
    if (not isinstance(saved, dict) or set(saved) != {"binding", "payload", "rng", "mps_rng"}
            or saved.get("binding") != plan["checkpoint_binding"]):
        raise ValueError("Checkpoint binding differs from rejected research journal")
    payload = saved.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload missing")
    master, basis = payload.get("master"), payload.get("basis")
    if (payload.get("format") != 1 or payload.get("id") != data["session_id"] or payload.get("step") != 832
            or payload.get("parameter") != data["parameter"]
            or payload.get("original_sha256") != data["original_tensor_sha256"]
            or not isinstance(master, torch.Tensor) or not isinstance(basis, torch.Tensor)
            or master.device.type != "cpu" or basis.device.type != "cpu"
            or master.dtype != torch.float32 or basis.dtype != torch.float32
            or list(master.shape) != data["parameter_shape"] or list(basis.shape) != data["basis_shape"]
            or not torch.isfinite(master).all() or not torch.isfinite(basis).all()):
        raise ValueError("832 master/basis/original/parameter fields are invalid")
    report = payload.get("report", {})
    if (report.get("steps") != 832 or report.get("identity") != data["identity"]
            or report.get("parameter") != data["parameter"]
            or report.get("training_selection", {}).get("selected_step") is not None):
        raise ValueError("Only the unselected rejected832 candidate can be inspected")
    return master, basis


def tensors(plan):
    import torch
    if sha(plan["candidate_path"]) != plan["candidate_file_sha256"]:
        raise ValueError("Immutable832 checkpoint file changed")
    saved = torch.load(safe(plan["candidate_path"]), map_location="cpu", weights_only=True, mmap=True)
    return saved, *validate_payload(saved, plan)


def tensor_binding(master, basis, dtype):
    import torch
    from verify_luban0_final import tensor_digest
    if dtype not in {"torch.bfloat16", "torch.float32"}:
        raise ValueError("Unsupported native precision")
    native = master.to(torch.bfloat16 if dtype == "torch.bfloat16" else torch.float32)
    residual = float((basis.T @ basis - torch.eye(basis.shape[1])).abs().max())
    if residual > 1e-4:
        raise ValueError("Archived protection basis is not orthonormal")
    return {"master_fp32_sha256": tensor_digest(master), "basis_fp32_sha256": tensor_digest(basis),
        "candidate_native_sha256": tensor_digest(native), "basis_orthogonality_max_abs": residual}


def production_snapshot(root):
    from verify_router_coverage_study import database_sha, module_snapshot
    root = safe(root)
    store, db = root / "modules/luban0-v1", root / "app/memory.sqlite3"
    if (store / "active-core.json").exists():
        raise ValueError("Diagnostic is fixed to production originalNone")
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='training_runs'").fetchone()
        if exists and connection.execute("SELECT 1 FROM training_runs WHERE state IN ('queued','running','paused','committing')").fetchone():
            raise ValueError("A pending product training job owns resources")
    return {"modules": module_snapshot(store), "database_logical_sha256_including_WAL": database_sha(db)}


def bind(plan):
    """Explicit CPU/streaming binding, separately authorized; does not load a model."""
    from verify_router_coverage_study import model_binding
    if prepare_plan(plan["output"]) != plan:
        raise ValueError("Static plan changed before bind")
    saved, master, basis = tensors(plan)
    result = copy.deepcopy(plan)
    result.update(tensor_binding=tensor_binding(master, basis, plan["dtype"]),
        official_model_binding=model_binding(plan["model_path"], plan["identity"]),
        production_snapshot=production_snapshot(plan["production_root"]), state="bound_waiting_root_seal_and_MODEL_GO")
    return result


def diagnose_row(runtime, sample, basis, observed_token):
    """Teacher forcing only; same native graph vs captured independent last tail."""
    import torch
    from torch.nn import functional as F
    from luban.evolution import _answer_logits, _target
    from verify_training_operators import _comparison, TOLERANCES
    from verify_luban0_final import tensor_digest
    model, decoder = runtime.model, runtime.decoder
    _, weight = _target(runtime)
    mlp = getattr(decoder.layers[-1].mlp, "base", decoder.layers[-1].mlp)
    ids, labels = runtime._sample_tokens(sample)
    mask = labels[:, 1:] != -100
    positions = mask.nonzero(as_tuple=False)
    if positions.numel() == 0 or positions[:, 0].ne(0).any():
        raise ValueError("Diagnostic requires one complete gold+EOS sequence")
    first = int(positions[0, 1])
    targets = labels[:, 1:][mask]
    value = json.loads(sample["provenance"])
    prefix = runtime._tokens(value["messages"], "off")
    gold = runtime.tokenizer.encode(sample["answer"], add_special_tokens=False) + [runtime.tokenizer.eos_token_id]
    first_label = first + 1
    if (prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] != first_label
            or not torch.equal(prefix, ids[:, :first_label]) or not labels[:, :first_label].eq(-100).all()
            or targets.detach().cpu().tolist() != gold or ids[0, first_label:].detach().cpu().tolist() != gold):
        raise ValueError("Actual inference prefix, final gold+EOS or causal mask mismatch")
    captured, handles = {}, []

    def capture(key):
        def hook(module, args):
            if key in captured:
                raise ValueError("Last-layer hook ran more than once")
            captured[key] = args[0].detach().clone()
        return hook

    try:
        handles.append(decoder.layers[-1].post_attention_layernorm.register_forward_pre_hook(capture("residual")))
        handles.append(mlp.down_proj.register_forward_pre_hook(capture("input")))
        output = model(ids, labels=labels, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"residual", "input"} or captured["input"].shape[:2] != labels.shape:
        raise ValueError("Full native residual/projection input capture failed")
    logits = _answer_logits(output.logits, labels)
    ce = F.cross_entropy(logits, targets, reduction="none")
    native_gradient, = torch.autograd.grad(ce[0], weight)
    native_loss = float(output.loss.detach().float().cpu())
    independent_weight = weight.detach().clone().requires_grad_(True)
    hidden = F.linear(captured["input"], independent_weight, mlp.down_proj.bias)
    tail_logits = model.lm_head(decoder.norm(hidden + captured["residual"]))
    tail_selected = _answer_logits(tail_logits, labels)
    tail_first_ce = F.cross_entropy(tail_selected[:1], targets[:1])
    tail_gradient, upstream = torch.autograd.grad(tail_first_ce, (independent_weight, hidden))
    x, derivative = captured["input"][0, first].float(), upstream[0, first].float()
    effective = x - (x @ basis) @ basis.T
    algebra = torch.outer(derivative, x)
    projected_algebra = torch.outer(derivative, effective)
    projected_native = native_gradient.float() - (native_gradient.float() @ basis) @ basis.T
    tolerance = TOLERANCES[str(weight.dtype)]
    comparison = _comparison(native_gradient, tail_gradient, atol=tolerance["gradient_atol"],
        rtol=tolerance["rtol"], relative_l2=tolerance["relative_l2"])
    logit_comparison = _comparison(logits, tail_selected, atol=tolerance["logits_atol"],
        rtol=tolerance["rtol"], relative_l2=tolerance["relative_l2"])
    first_logits = logits[0].detach()
    gold, top = int(targets[0]), int(first_logits.argmax())
    competitors = first_logits.clone()
    competitors[gold] = -torch.inf
    non_gold = int(competitors.argmax())
    if not 0 <= observed_token < first_logits.numel():
        raise ValueError("Observed832 token is outside the actual vocabulary")
    values = [native_loss, *ce.detach().cpu().tolist(), float(effective.norm().cpu()),
              float(native_gradient.float().norm().cpu()), float(algebra.norm().cpu())]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite diagnostic loss/input/gradient")
    result = {"id": sample["id"], "native_dtype": str(weight.dtype), "input_tokens": ids.numel(),
        "input_ids_sha256": tensor_digest(ids), "labels_sha256": tensor_digest(labels),
        "gold_eos_token_ids": targets.detach().cpu().tolist(), "first_predict_position_zero_based": first,
        "causal_scope": "logits at prefix_last_position predict first gold; previous assistant turns masked",
        "per_target_token_ce_fp32": ce.detach().cpu().tolist(), "native_mean_ce": native_loss,
        "computed_mean_ce": float(ce.detach().mean().cpu()),
        "first_gold_token": gold, "gold_rank_strict_greater": 1 + int((first_logits > first_logits[gold]).sum()),
        "rank_tie_policy": "competition rank: one plus count strictly above gold; equal logits share rank",
        "top1_token": top, "gold_minus_top1_logit": float((first_logits[gold] - first_logits[top]).cpu()),
        "maximum_non_gold_token": non_gold,
        "gold_minus_maximum_non_gold_logit": float((first_logits[gold] - first_logits[non_gold]).cpu()),
        "observed832_first_token": observed_token,
        "observed832_is_gold": observed_token == gold,
        "gold_minus_observed832_logit": float((first_logits[gold] - first_logits[observed_token]).cpu()),
        "first_projection_input_norm_fp32_cast": float(x.norm().cpu()),
        "effective_input_ratio_fp32": float((effective.norm() / x.norm().clamp_min(1e-30)).cpu()),
        "native_first_ce_gradient_norm_fp32_cast": float(native_gradient.float().norm().cpu()),
        "independent_native_gradient": comparison, "independent_native_logits": logit_comparison,
        "fp32_outer_gradient_norm": float(algebra.norm().cpu()),
        "fp32_effective_outer_gradient_norm": float(projected_algebra.norm().cpu()),
        "native_gradient_projected_fp32_norm": float(projected_native.norm().cpu()),
        "projected_formula_difference_norm": float((projected_native - projected_algebra).norm().cpu()),
        "projected_formula_relative_difference": float(((projected_native - projected_algebra).norm() / projected_algebra.norm().clamp_min(1e-30)).cpu()),
        "native_vs_fp32_outer_relative_difference": float(((native_gradient.float() - algebra).norm() / algebra.norm().clamp_min(1e-30)).cpu()),
        "algebra_scope": "FP32 outer(native-dtype upstream cast, native-dtype X cast); projected X=x-xBB^T; not FP32 full-model equivalence",
        "hooks_removed": not handles or all(handle.id not in handle.hooks_dict_ref() for handle in handles if handle.hooks_dict_ref() is not None)}
    if not comparison["passed"] or not logit_comparison["passed"] or abs(native_loss - result["computed_mean_ce"]) > tolerance["loss_atol"]:
        raise ValueError("Native/captured-tail diagnostic operator comparison failed: " + sample["id"])
    return result


def inspect_states(runtime, records, master, basis, observed, expected_original, persist,
                   expected_candidate=None, expected_parameter=None):
    """Restore ALL original state on success or failure; caller releases model lock."""
    import torch
    from luban.evolution import _target
    from verify_luban0_final import tensor_digest
    name, parameter = _target(runtime)
    original = parameter.detach().cpu().clone()
    flags = [(value, value.requires_grad, value.grad) for value in runtime.model.parameters()]
    modes = [(module, module.training) for module in runtime.model.modules()]
    branch, version = runtime.branch.expert, runtime.active_core
    pointer = Path(runtime.store) / "active-core.json"
    pointer_before = pointer.read_bytes() if pointer.exists() else None
    cache = getattr(runtime.model.config, "use_cache", None)
    report = {"parameter": name, "states": [], "restoration": {"expected_original_sha256": expected_original,
              "before_sha256": tensor_digest(original)}}
    try:
        if (tensor_digest(original) != expected_original or version is not None or pointer_before is not None
                or (expected_parameter is not None and name != expected_parameter)):
            raise ValueError("Loaded core is not the exact genuine originalNone")
        runtime.branch.expert = None
        runtime.model.eval().requires_grad_(False)
        parameter.requires_grad_(True)
        for state in ("originalNone", "rejected832"):
            with torch.no_grad():
                parameter.copy_(original if state == "originalNone" else master)
            record = {"state": state, "native_weight_sha256": tensor_digest(parameter), "rows": []}
            report["states"].append(record)
            if state == "rejected832" and expected_candidate is not None and record["native_weight_sha256"] != expected_candidate:
                raise ValueError("Actual native rejected832 weight differs from bound tensor SHA")
            for sample in records:
                parameter.grad = None
                record["rows"].append(diagnose_row(runtime, sample, basis, observed[sample["id"]]["token_id"]))
                persist(report)
    finally:
        with torch.no_grad():
            parameter.copy_(original)
        runtime.branch.expert = branch
        for value, flag, grad in flags:
            value.requires_grad_(flag)
            value.grad = grad
        for module, training in modes:
            module.training = training
        report["restoration"].update(actual_original_sha256=tensor_digest(parameter),
            expected_original_match=tensor_digest(parameter) == expected_original,
            original_weight_exact=bool(torch.equal(parameter.detach().cpu(), original)),
            branch_same=runtime.branch.expert is branch, core_same=runtime.active_core == version,
            pointer_same=(pointer.read_bytes() if pointer.exists() else None) == pointer_before,
            grad_and_requires_same=all(value.requires_grad == flag and value.grad is grad for value, flag, grad in flags),
            modes_same=all(module.training == mode for module, mode in modes),
            cache_config_same=getattr(runtime.model.config, "use_cache", None) == cache,
            cache_scope="Every forward use_cache=False; no generate/past passed; cache config untouched")
        report["restoration"]["passed"] = all(report["restoration"][key] for key in
            ("expected_original_match", "original_weight_exact", "branch_same", "core_same", "pointer_same", "grad_and_requires_same", "modes_same", "cache_config_same"))
        persist(report)
        if not report["restoration"]["passed"]:
            raise RuntimeError("Diagnostic failed to restore original weight/state")
    return report


def execute(path, approved_sha):
    if sha(path) != approved_sha:
        raise ValueError("Separately approved exact protocol SHA required")
    plan = json.loads(safe(path).read_text())
    unbound = copy.deepcopy(plan)
    unbound.update(tensor_binding=None, official_model_binding=None, production_snapshot=None,
                   state="waiting_explicit_bind_and_MODEL_GO")
    if prepare_plan(plan["output"]) != unbound or not all(plan[key] for key in
            ("tensor_binding", "official_model_binding", "production_snapshot")):
        raise ValueError("Exact prepared/bound diagnostic plan required")
    from verify_router_coverage_study import model_binding
    if model_binding(plan["model_path"], plan["identity"]) != plan["official_model_binding"]:
        raise ValueError("Bound official actual model files changed")
    if production_snapshot(plan["production_root"]) != plan["production_snapshot"]:
        raise ValueError("Read-only production database/modules changed")
    saved, master, basis = tensors(plan)
    if tensor_binding(master, basis, plan["dtype"]) != plan["tensor_binding"]:
        raise ValueError("Bound master/basis/native tensor SHA changed")
    output = safe(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "protocol.json", plan)
    import torch
    started = time.perf_counter()
    torch.manual_seed(plan["seed"])
    report = {"protocol_sha256": approved_sha, "passed": False, "teacher_calls": 0,
        "generation_calls": 0, "optimizer_steps": 0, "hardware": platform.platform(),
        "torch_version": str(torch.__version__), "seed": 42, "source_sha256_before": plan["source_sha256"]}

    def persist(detail=None):
        if detail is not None:
            report["diagnostic"] = detail
        # Only this new attempt's result is rewritten; original evidence immutable.
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    runtime = None
    resources = {}
    try:
        from luban.runtime import Runtime
        from luban.settings import Settings
        from luban.memory import Memory
        from luban.process_lock import ModelProcessLock
        from verify_automatic_batches import measured
        import psutil
        chip = platform.processor()
        if platform.system() == "Darwin":
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        report["hardware"] = {"platform": platform.platform(), "chip": chip, "physical_memory_bytes": psutil.virtual_memory().total,
                              "device": plan["device"], "dtype": plan["dtype"]}
        with tempfile.TemporaryDirectory(prefix="luban-field-diagnostic-settings-") as home:
            settings = Settings(home)
            settings.update({"storage_root": str(output / "private"), "model_path": plan["model_path"],
                             "temperature": 0, "local_thinking": "off"})
            runtime = Runtime(settings, Memory(output / "private"), research=True)
            if str(runtime.device) != plan["device"] or str(runtime.dtype) != plan["dtype"]:
                raise ValueError("Original device/native precision changed")
            runtime._model_process_lock = ModelProcessLock(plan["identity"], runtime.device)
            with measured(runtime) as resources:
                report["resources"] = resources
                basis_device = None
                try:
                    runtime.load(paging=False)
                    if runtime.identity != plan["identity"]:
                        raise ValueError("Loaded base identity changed")
                    basis_device = basis.to(runtime.device)
                    inspect_states(runtime, plan["data"]["records"], master, basis_device,
                        plan["data"]["observed832_first_token"], plan["data"]["original_tensor_sha256"], persist,
                        plan["tensor_binding"]["candidate_native_sha256"], plan["data"]["parameter"])
                except BaseException as error:
                    # Keep strings, never the exception/traceback holding native graph tensors.
                    report["error"] = {"type": type(error).__name__, "message": str(error)}
                # Exception frames are gone before release/OS-lock proof (also on failure).
                basis_device = None
                runtime._release_model()
                report["model_released"] = runtime.model is None and runtime._model_process_lock is None
                lock = ModelProcessLock(plan["identity"], plan["device"])
                lock.close()
                report["actual_process_lock_reacquired_and_closed"] = True
            report["resources"] = resources
        report["passed"] = "error" not in report
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["passed"] = False
        persist()
    finally:
        if runtime is not None:
            try:
                if runtime.model is not None or runtime._model_process_lock is not None:
                    runtime._release_model()
                report["model_released"] = runtime.model is None and runtime._model_process_lock is None
                if not report.get("actual_process_lock_reacquired_and_closed"):
                    lock = ModelProcessLock(plan["identity"], plan["device"])
                    lock.close()
                    report["actual_process_lock_reacquired_and_closed"] = True
            except BaseException as error:
                report["release_error"] = {"type": type(error).__name__, "message": str(error)}
                report["passed"] = False
        report["wall_seconds"] = time.perf_counter() - started
        try:
            report["source_sha256_after"] = {name: sha(name) for name in plan["source_sha256"]}
            report["production_after"] = production_snapshot(plan["production_root"])
            report["official_model_after"] = model_binding(plan["model_path"], plan["identity"])
            report["candidate_file_sha256_after"] = sha(plan["candidate_path"])
            report["metadata_after_verified"] = prepare_plan(plan["output"]) == unbound
            report["preserved"] = (report["source_sha256_after"] == plan["source_sha256"]
                and report["production_after"] == plan["production_snapshot"]
                and report["official_model_after"] == plan["official_model_binding"]
                and report["candidate_file_sha256_after"] == plan["candidate_file_sha256"]
                and report["metadata_after_verified"])
        except BaseException as error:
            report["preservation_error"] = {"type": type(error).__name__, "message": str(error)}
            report["preserved"] = False
        report["passed"] = report["passed"] and report["preserved"]
        persist()
    if not report["passed"]:
        raise RuntimeError("Diagnostic failed; original failure/restoration/preservation saved in result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", type=Path)
    action.add_argument("--bind", type=Path)
    action.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approved-sha")
    args = parser.parse_args()
    if args.run:
        if not args.approved_sha:
            parser.error("--run requires independently approved SHA and MODEL GO")
        execute(args.run, args.approved_sha)
    elif args.bind:
        if not args.output:
            parser.error("--bind needs a NEW bound plan file --output")
        write_new(args.output, bind(json.loads(safe(args.bind).read_text())))
    elif args.prepare:
        if not args.output:
            parser.error("--prepare needs the unique future result directory --output")
        write_new(args.prepare, prepare_plan(args.output))
    else:
        print(json.dumps({"protocol": PROTOCOL, "fixed_TRAIN": sorted(IDS), "data_path": str(DATA),
            "data_sha256": DATA_SHA, "state": "waiting_root_bind_seal_MODEL_GO", "model_go": False,
            "torch_imported": False, "teacher_calls": 0, "generation_calls": 0, "optimizer_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
