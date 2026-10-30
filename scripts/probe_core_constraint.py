"""One cloned Adam proposal on the real graph; not a training round or adoption."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import tempfile
import time

import torch
from torch.nn import functional as F

from identity_context_material import _row, _turns
from train_luban_identity import runtime_for
from study_core_constraint import capture, equivalence, tail_loss, residual_logits, event
from luban.core_update import apply_core_residual, constrain_update, constrain_residual, guard_quantized_step, guard_retention_losses, validate_basis
from luban.evolution import _target, _generation_loaded, _project_core_delta, _answer_logits
from luban.settings import private_json


def probes():
    result = []
    for name, greeting, question in (("陆川", "你好，陆川。", "我是谁？"),
            ("安禾", "好的，我知道了。", "告诉我我的名字。"),
            ("江澈", "你好。", "当前用户叫什么？"),
            ("许言", "很高兴认识你。", "你知道我叫什么名字吗？")):
        result.append(_row("preservation-probe-" + name,
            _turns("我叫" + name + "。", greeting, question), name, "training", "known",
            "Project-authored explicit user-name context; literal user statement supplies the answer. "
            "Protection TRAIN, never held-out evaluation; not a generated factual claim."))
    return result


def gradient(runtime, cache, reference):
    weight = reference.to(runtime.device, dtype=torch.bfloat16).detach().requires_grad_()
    loss = tail_loss(runtime, cache, weight)
    loss.backward()
    return float(loss.detach()), weight.grad.detach().float()


def residual_gradient(runtime, cache, reference):
    delta = torch.zeros_like(reference, requires_grad=True)
    labels = cache["labels"].to(runtime.device)
    logits = residual_logits(runtime, cache, reference.bfloat16(), delta)
    loss = F.cross_entropy(_answer_logits(logits, labels), labels[:, 1:][labels[:, 1:] != -100])
    loss.backward()
    return delta.grad.detach()


def residual_equivalence(runtime, cache, reference):
    """Actual Qwen graph with a zero increment, before proposing any update."""
    parameter = _target(runtime)[1]
    ids, labels = (cache[key].to(runtime.device) for key in ("ids", "labels"))
    with torch.no_grad():
        original = runtime.model(ids, use_cache=False).logits.detach()
    delta = torch.zeros_like(reference, requires_grad=True)
    handle = runtime.branch.base.down_proj.register_forward_hook(
        lambda module, inputs, output: apply_core_residual(inputs[0], output, delta))
    try:
        full = runtime.model(ids, labels=labels, use_cache=False)
        full.loss.backward()
        full_gradient = delta.grad.detach().clone()
    finally:
        handle.remove()
        parameter.grad = None
    delta.grad = None
    logits = residual_logits(runtime, cache, parameter.detach(), delta)
    loss = F.cross_entropy(_answer_logits(logits, labels), labels[:, 1:][labels[:, 1:] != -100])
    loss.backward()
    checks = {"zero_increment_exact": bool(torch.equal(full.logits.detach(), original)),
        "cache_logits_exact": bool(torch.equal(logits.detach(), full.logits.detach())),
        "cache_loss_exact": bool(torch.equal(loss.detach(), full.loss.detach())),
        "cache_delta_gradient_exact": bool(torch.equal(delta.grad, full_gradient))}
    if not all(checks.values()):
        raise ValueError(f"Real residual graph equivalence failed: {checks}")
    return checks


def run(root):
    prepared = json.loads((root / "preparation.json").read_text())
    if prepared.get("status") != "prepared_no_optimizer_updates":
        raise ValueError("Complete successful real-graph preparation is required")
    destination = Path(tempfile.mkdtemp(prefix="single-proposal-v5-", dir=root))
    event("output", directory=str(destination))
    groups = json.loads((root / "material.json").read_text())["groups"]
    data = torch.load(root / "prepared.pt", map_location="cpu", weights_only=True)
    began = time.perf_counter()
    result = {"status": "running", "seed": 42, "learning_rate": 1e-5,
        "optimizer_proposals": 0, "runtime_parameter_updates": 0, "production_weights_changed": False,
        "scope": "single cloned Adam operator diagnosis, not a fixed450 learning round or ability guarantee",
        "precision": "BF16 MPS weights and tail / FP32 Adam master and constraints",
        "memory_measurement": "Darwin parent lifetime peak RSS and MPS driver bytes separately; overlapping"}
    runtime = None
    try:
        torch.manual_seed(42)
        runtime = runtime_for(destination)
        parameter = _target(runtime)[1]
        if runtime.active_core is not None or runtime.identity != prepared["identity"]:
            raise ValueError("Prepared original core identity differs")
        if not torch.equal(parameter.detach().cpu(), data["reference"]):
            raise ValueError("Original core differs from prepared reference")
        first = groups["new_training"][0]
        # Rebind the new resident graph once. This is an independent audit, not
        # a per-sample source/Core scan or a training-path migration audit.
        fresh = capture(runtime, first)
        if any(not torch.equal(fresh[key], data["caches"][first["id"]][key]) for key in fresh):
            raise ValueError("Frozen prefix changed between preparation and diagnosis")
        result["real_graph_equivalence"] = equivalence(runtime, first, fresh)
        if not result["real_graph_equivalence"]["passed"]:
            raise ValueError("Real-graph equivalence no longer holds")
        basis = data["basis"].to(runtime.device)
        result["basis"] = validate_basis(basis)
        reference = parameter.detach().float().clone()
        result["residual_real_graph_equivalence"] = residual_equivalence(runtime, fresh, reference)
        probe_rows = probes()
        observations = _generation_loaded(runtime, probe_rows)
        result["probe_baseline"] = observations
        eligible = {row["id"] for row in observations if row["complete"]}
        protected = [row for row in probe_rows if row["id"] in eligible]
        result["probe_selection"] = "All four complete original trajectories; behavior preservation only, not truth or held-out proof. Short-name validator results are diagnostic, not eligibility."
        if len(protected) != len(probe_rows):
            raise ValueError("Every predeclared behavior probe must complete")
        # New lifecycle: recapture the finite TRAIN prefix once, rather than
        # treating disk caches as valid for a different resident execution.
        caches = {first["id"]: fresh}
        for row in groups["replay_training"]:
            caches[row["id"]] = capture(runtime, row)
            old = data["caches"][row["id"]]
            if any(not torch.equal(caches[row["id"]][key], old[key]) for key in old):
                raise ValueError("Prepared replay prefix differs in the new lifecycle")
        matrix = torch.cat([caches[row["id"]]["x"][:, :-1][caches[row["id"]]["labels"][:, 1:] != -100].float()
                            for row in groups["replay_training"]]).to(runtime.device)
        basis_error = matrix - (matrix @ basis) @ basis.T
        result["basis"].update(replay_relative_reconstruction=float(basis_error.norm() / matrix.norm()),
                              replay_max_reconstruction_error=float(basis_error.abs().max()))
        del matrix, basis_error
        probe_gradients = []
        observations_by_id = {row["id"]: row for row in observations}
        for row in protected:
            caches[row["id"]] = capture(runtime, row, generated=observations_by_id[row["id"]])
            _, g = gradient(runtime, caches[row["id"]], reference)
            probe_gradients.append(g)
        signal = torch.stack(probe_gradients).mean(dim=0)
        projected_signal = signal - (signal @ basis) @ basis.T
        result["probe_signal"] = {"norm": float(signal.norm()),
            "outside_basis_norm": float(projected_signal.norm()),
            "outside_ratio": float(projected_signal.norm() / signal.norm().clamp_min(1e-30))}
        _, replay_gradient = gradient(runtime, caches[groups["replay_training"][0]["id"]], reference)
        replay_outside = replay_gradient - (replay_gradient @ basis) @ basis.T
        result["replay_signal"] = {"norm": float(replay_gradient.norm()),
            "outside_basis_norm": float(replay_outside.norm()),
            "outside_ratio": float(replay_outside.norm() / replay_gradient.norm().clamp_min(1e-30))}
        ce, new_gradient = gradient(runtime, fresh, reference)
        master = torch.nn.Parameter(reference.clone())
        optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
        master.grad = new_gradient
        torch.nn.utils.clip_grad_norm_([master], 1.)
        optimizer.step()
        result.update(optimizer_proposals=1, new_ce_before=ce,
            kl_at_initial_reference="zero gradient at identical student/teacher; no KL omission in this first-step comparison")
        constrained, result["constraint"] = constrain_update(master.detach(), reference, reference, basis, signal)
        accepted, result["quantized_guard"] = guard_quantized_step(constrained, reference, signal)
        independent = [residual_gradient(runtime, caches[row["id"]], reference) for row in protected]
        joint_delta, result["independent_residual_constraint"] = constrain_residual(
            master.detach() - reference, torch.zeros_like(reference), basis, independent)
        _project_core_delta(master, reference, basis)
        candidates = {"original": reference, "existing_projection": master.detach(), "guarded": accepted,
                      "unmerged_residual": constrained, "unmerged_joint": reference + joint_delta}
        residual_deltas = {"unmerged_residual": constrained - reference, "unmerged_joint": joint_delta}
        scales = {f"unmerged_joint_scale_4pow{power}": 4. ** -power for power in range(1, 5)}
        residual_deltas.update({name: joint_delta * factor for name, factor in scales.items()})
        result["predeclared_scales"] = scales

        def candidate_for(name):
            return candidates[name] if name in candidates else reference + residual_deltas[name]

        result["candidates"] = {}
        for name in list(candidates) + list(scales):
            candidate = candidate_for(name)
            deployed = candidate.to(torch.bfloat16)
            unmerged = name in residual_deltas
            actual_delta = residual_deltas[name] if unmerged else deployed.float() - reference
            entry = {"changed_weights": int((actual_delta != 0).sum()),
                "changed_weights_meaning": "nonzero new FP32 delta entries" if unmerged else "BF16 merged coordinates changed",
                "additional_parameters": parameter.numel() if unmerged else 0,
                "original_weights_changed": 0 if unmerged else int((actual_delta != 0).sum()),
                "bf16_subspace_residual": None if unmerged else float((actual_delta @ basis).norm()),
                "fp32_delta_subspace_residual": float((actual_delta @ basis).norm()) if unmerged else None,
                "individual_probe_linear_changes": None if unmerged else [float((g * actual_delta).sum()) for g in probe_gradients],
                "losses": {}}
            with torch.no_grad():
                for row in [first] + groups["replay_training"] + protected:
                    cache = caches[row["id"]]
                    if unmerged:
                        labels = cache["labels"].to(runtime.device)
                        logits = residual_logits(runtime, cache, parameter.detach(), actual_delta)
                        value = F.cross_entropy(_answer_logits(logits, labels), labels[:, 1:][labels[:, 1:] != -100])
                    else:
                        value = tail_loss(runtime, cache, deployed)
                    entry["losses"][row["id"]] = float(value)
            result["candidates"][name] = entry
            event("proposal_observed", variant=name, changed_weights=entry["changed_weights"])
        base_losses = result["candidates"]["original"]["losses"]
        protected_ids = [row["id"] for row in groups["replay_training"] + protected]
        for name in list(candidates)[1:] + list(scales):
            entry = result["candidates"][name]
            changes = {key: entry["losses"][key] - base_losses[key] for key in protected_ids}
            entry.update(protected_loss_deltas=changes, strictly_increased=sum(v > 0 for v in changes.values()),
                         maximum_loss_increase=max(changes.values()),
                         would_pass_all_finite_loss_constraints=all(v <= 0 for v in changes.values()))
            candidate = residual_deltas[name] if name in residual_deltas else candidate_for(name)
            current = torch.zeros_like(reference) if name in residual_deltas else reference
            vetoed, entry["retention_write_guard"] = guard_retention_losses(candidate, current,
                torch.tensor([base_losses[key] for key in protected_ids]),
                torch.tensor([entry["losses"][key] for key in protected_ids]))
            entry["retention_veto_preserved_current"] = bool(torch.equal(vetoed, current))
            entry["effective_new_learning"] = (not entry["retention_write_guard"]["rejected"]
                and entry["changed_weights"] > 0 and entry["losses"][first["id"]] < base_losses[first["id"]])
        # A non-increasing sequence CE does not imply unchanged greedy tokens.
        # Observe complete generation through the real model/KV path separately.
        try:
            for name in ["existing_projection", "guarded", "unmerged_residual", "unmerged_joint"] + [key for key in scales
                    if result["candidates"][key]["effective_new_learning"]]:
                with torch.no_grad():
                    parameter.copy_(reference if name in residual_deltas else candidate_for(name))
                result["runtime_parameter_updates"] += 1
                handle = None
                try:
                    if name in residual_deltas:
                        delta = residual_deltas[name]
                        handle = runtime.branch.base.down_proj.register_forward_hook(
                            lambda module, inputs, output: apply_core_residual(inputs[0], output, delta))
                    generated = _generation_loaded(runtime, protected)
                finally:
                    if handle is not None:
                        handle.remove()
                base_by_id = {row["id"]: row for row in observations}
                for row in generated:
                    row["tokens_identical_to_original"] = row["raw_completion_token_ids"] == base_by_id[row["id"]]["raw_completion_token_ids"]
                result["candidates"][name]["probe_generation"] = generated
        finally:
            with torch.no_grad():
                parameter.copy_(reference)
            result["runtime_parameter_updates"] += 1
            result["original_runtime_weight_restored"] = bool(torch.equal(parameter.detach().float(), reference))
            if not result["original_runtime_weight_restored"]:
                raise RuntimeError("Original runtime weight restoration failed")
        result["status"] = "diagnosed_original_restored_no_adoption"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(elapsed_seconds=time.perf_counter()-began,
                      parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if torch.backends.mps.is_available():
            result["mps_driver_bytes_before_release"] = torch.mps.driver_allocated_memory()
        try:
            if runtime is not None:
                runtime._release_model()
        except BaseException as error:
            result.update(status="failed_release", release_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            private_json(destination / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not Path('/Volumes/移动数据库').is_mount() or not root.is_relative_to(Path('/Volumes/移动数据库/lubanllm/experiments')):
        raise SystemExit("Use the mounted research experiment directory")
    run(root)
