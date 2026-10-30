"""Post-Adam affine-subspace and finite linear-preservation constraints.

Call validate_basis once per frozen training round. constrain_update does not
mutate parameters or Adam state. Infeasible or numerically conflicting proposals
keep current unchanged. Checks concern FP32 and the supplied finite gradient
only; casting the result to BF16 requires separate observation. A zero signal,
including cancellation of averaged gradients, supplies no preservation evidence.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def apply_core_residual(inputs, original_output, delta):
    """Research candidate: retain the original BF16 projection and add FP32 ΔW.

    Zero delta preserves the existing output exactly. A protected input whose
    computed delta projection is zero also keeps that output. Neither premise
    is claimed for every unknown input or for approximate subspace projection.
    This adds parameters and an operation; it is not a merged BF16 core update.
    """
    if (not all(isinstance(v, torch.Tensor) for v in (inputs, original_output, delta))
            or delta.ndim != 2 or not delta.numel() or inputs.ndim < 2
            or inputs.shape[-1] != delta.shape[1]
            or original_output.shape != (*inputs.shape[:-1], delta.shape[0])
            or delta.dtype != torch.float32 or inputs.dtype != torch.bfloat16
            or original_output.dtype != inputs.dtype
            or inputs.device != delta.device or original_output.device != delta.device):
        raise ValueError("residual needs compatible BF16 input/output and FP32 delta on one device")
    for name, value in (("inputs", inputs), ("original output", original_output), ("delta", delta)):
        _finite(value, name)
    result = (original_output.float() + F.linear(inputs.float(), delta)).to(original_output.dtype)
    _finite(result, "residual output")
    return result


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _basis_schema(basis):
    if (not isinstance(basis, torch.Tensor) or basis.dtype != torch.float32
            or basis.ndim != 2 or not 0 < basis.shape[1] < basis.shape[0]):
        raise ValueError("basis must be FP32 with a nonempty trainable complement")


@torch.no_grad()
def validate_basis(basis):
    """Validate the frozen basis once, outside the per-update path."""
    _basis_schema(basis)
    _finite(basis, "basis")
    error = basis.T @ basis - torch.eye(basis.shape[1], device=basis.device)
    _finite(error, "basis orthogonality")
    residual = float(error.norm())
    if residual > 1e-4:
        raise ValueError("basis must have orthonormal columns")
    return {"orthogonality_frobenius": residual}


@torch.no_grad()
def constrain_residual(proposed_delta, current_delta, basis, gradients):
    """Nearest cumulative residual satisfying 1–4 independent linear signals.

    The frozen orthonormal basis is validated once by the caller. Enumerate all
    active sets of the small Gram problem, then check the actual FP32 candidate
    against every signal and the cumulative protected subspace. Numerical
    failure keeps current unchanged. No Adam state or input is modified.
    This is a finite first-order check; actual deployed losses still need vetoes.
    """
    _basis_schema(basis)
    if (not isinstance(current_delta, torch.Tensor) or current_delta.ndim != 2
            or not current_delta.numel()):
        raise ValueError("current_delta must be a nonempty FP32 matrix")
    if isinstance(gradients, torch.Tensor):
        if gradients.ndim != 3:
            raise ValueError("gradients must stack 1–4 FP32 matrices")
        gradients = list(gradients.unbind(0))
    if not isinstance(gradients, (list, tuple)) or not 1 <= len(gradients) <= 4:
        raise ValueError("gradients must contain 1–4 FP32 matrices")
    for name, value in [("proposed_delta", proposed_delta), ("current_delta", current_delta)] + [
            (f"gradient {i}", value) for i, value in enumerate(gradients)]:
        if (not isinstance(value, torch.Tensor) or value.shape != current_delta.shape
                or value.dtype != torch.float32 or value.device != current_delta.device):
            raise ValueError(f"{name} must share current_delta's FP32 shape and device")
        _finite(value, name)
    if basis.device != current_delta.device or basis.shape[0] != current_delta.shape[1]:
        raise ValueError("basis device or input dimension differs from current_delta")
    _finite(basis, "basis")

    # Accurate reductions and only the <=4-square solve use CPU FP64, which MPS
    # does not support. Chunking avoids full FP64 copies of model-sized signals.
    def dot(left, right):
        return sum(float(torch.dot(a.double(), b.double()))
                   for a, b in zip(left.split(262144), right.split(262144)))

    epsilon = 64 * torch.finfo(torch.float32).eps
    cpu_current = current_delta.cpu().reshape(-1)
    current_norm = dot(cpu_current, cpu_current) ** .5
    current_residual = float((current_delta @ basis).norm())
    current_tolerance = epsilon * max(1., current_norm)
    if not math.isfinite(current_residual) or current_residual > current_tolerance:
        raise ValueError("current_delta is outside the protected subspace")
    count = len(gradients)
    report = {"constraint_count": count, "rejected": False, "rejection_reason": None,
              "active_set": [], "multipliers": [0.] * count,
              "current_subspace_residual": current_residual, "already_feasible": False}

    def reject(reason):
        report.update(rejected=True, rejection_reason=reason, post_interference=[0.] * count,
                      linear_tolerance=[0.] * count, step_norm=0.,
                      cumulative_subspace_residual=current_residual, subspace_tolerance=current_tolerance)
        return current_delta.detach(), report

    projected = proposed_delta - (proposed_delta @ basis) @ basis.T
    step = projected - current_delta
    if not bool(torch.isfinite(projected).all() and torch.isfinite(step).all()):
        return reject("nonfinite_projected_proposal")
    cpu_step = step.cpu().reshape(-1)
    cpu_gradients = [g.cpu().reshape(-1) for g in gradients]
    pre = [dot(g, cpu_step) for g in cpu_gradients]
    norms = [dot(g, g) ** .5 for g in cpu_gradients]
    directions, direction_scales = [], []
    for gradient in gradients:
        scale = float(gradient.abs().max())
        normalized = gradient / scale if scale else gradient
        direction = normalized - (normalized @ basis) @ basis.T
        # Near-span gradients suffer cancellation; remove leaked span before
        # normalization can amplify it into a large, invalid correction.
        direction -= (direction @ basis) @ basis.T
        if not bool(torch.isfinite(direction).all()):
            return reject("nonfinite_projected_gradient")
        norm = float(direction.norm())
        directions.append(direction / norm if norm else direction)
        direction_scales.append(scale * norm)
    report.update(pre_interference=pre, zero_signals=[n == 0 for n in norms],
                  projected_gradient_norms=direction_scales)
    if any(scale == 0 and value > 0 for scale, value in zip(direction_scales, pre)):
        return reject("no_feasible_preservation_direction")

    # Use exactly the final actual-step bounds before solving. Correcting an
    # already feasible proposal can introduce FP32 violations of other guards.
    projected_step_norm = dot(cpu_step, cpu_step) ** .5
    projected_tolerances = [epsilon * norm * projected_step_norm for norm in norms]
    cpu_projected = projected.cpu().reshape(-1)
    projected_residual = float((projected @ basis).norm())
    projected_tolerance = epsilon * max(1., dot(cpu_projected, cpu_projected) ** .5)
    if (all(value <= tolerance for value, tolerance in zip(pre, projected_tolerances))
            and math.isfinite(projected_residual) and projected_residual <= projected_tolerance):
        report.update(already_feasible=True, correction_norm=0.,
            candidate_post_interference=pre, candidate_linear_tolerance=projected_tolerances,
            candidate_subspace_residual=projected_residual, candidate_subspace_tolerance=projected_tolerance,
            post_interference=pre, linear_tolerance=projected_tolerances, step_norm=projected_step_norm,
            cumulative_subspace_residual=projected_residual, subspace_tolerance=projected_tolerance)
        return projected.detach(), report
    del cpu_projected
    cpu_directions = [g.cpu().reshape(-1) for g in directions]
    gram = torch.empty(count, count, dtype=torch.float64)
    for i in range(count):
        for j in range(i + 1):
            gram[i, j] = gram[j, i] = dot(cpu_directions[i], cpu_directions[j])
    rhs = torch.tensor([value / scale if scale else 0.
                        for value, scale in zip(pre, direction_scales)], dtype=torch.float64)
    best, best_cost, best_active = None, float("inf"), []
    solver_tolerance = 128 * torch.finfo(torch.float64).eps * max(1., float(rhs.norm()))
    for mask in range(1 << count):
        active = [i for i in range(count) if mask & (1 << i)]
        multipliers = torch.zeros(count, dtype=torch.float64)
        if active:
            local = gram[active][:, active]
            try:
                solution = torch.linalg.lstsq(local, rhs[active], driver="gelsd").solution
            except torch.linalg.LinAlgError:
                continue
            if not bool(torch.isfinite(solution).all()) or bool((solution < -solver_tolerance).any()):
                continue
            multipliers[active] = solution.clamp_min(0)
            if float((local @ multipliers[active] - rhs[active]).norm()) > solver_tolerance:
                continue
        if bool((rhs - gram @ multipliers > solver_tolerance).any()):
            continue
        cost = float(multipliers @ gram @ multipliers)
        if cost < best_cost:
            best, best_cost, best_active = multipliers, cost, active
    if best is None:
        return reject("no_numerically_feasible_active_set")
    report.update(active_set=best_active,
                  multipliers=[float(value) / scale if scale else 0.
                               for value, scale in zip(best, direction_scales)],
                  gram_kkt_primal_residual=float((rhs - gram @ best).clamp_min(0).max()))
    correction = torch.zeros_like(step)
    for coefficient, direction in zip(best, directions):
        if abs(float(coefficient)) > torch.finfo(torch.float32).max:
            return reject("unrepresentable_correction")
        correction.add_(direction, alpha=float(coefficient))
    candidate = projected - correction
    if not bool(torch.isfinite(candidate).all()):
        return reject("nonfinite_candidate")
    actual_step = (candidate - current_delta).cpu().reshape(-1)
    if not bool(torch.isfinite(actual_step).all()):
        return reject("nonfinite_actual_step")
    post = [dot(g, actual_step) for g in cpu_gradients]
    step_norm = dot(actual_step, actual_step) ** .5
    tolerances = [epsilon * norm * step_norm for norm in norms]
    cpu_candidate = candidate.cpu().reshape(-1)
    residual = float((candidate @ basis).norm())
    subspace_tolerance = epsilon * max(1., dot(cpu_candidate, cpu_candidate) ** .5)
    report.update(candidate_post_interference=post, candidate_linear_tolerance=tolerances,
                  candidate_subspace_residual=residual, candidate_subspace_tolerance=subspace_tolerance,
                  correction_norm=float(correction.norm()))
    if any(value > tolerance for value, tolerance in zip(post, tolerances)):
        return reject("fp32_linear_preservation")
    if not math.isfinite(residual) or residual > subspace_tolerance:
        return reject("fp32_affine_subspace")
    report.update(post_interference=post, linear_tolerance=tolerances, step_norm=step_norm,
                  cumulative_subspace_residual=residual, subspace_tolerance=subspace_tolerance)
    return candidate.detach(), report


@torch.no_grad()
def constrain_update(proposed, current, reference, basis, preservation_gradient):
    """Project the actual Adam proposal into both supplied constraints.

    proposed/current/reference/gradient are same-shape FP32 matrices. current
    must already lie in reference's protected affine space within FP32 error.
    Adam moments remain those of the proposed step, as in post-update masking.
    The returned candidate is detached; the caller explicitly installs it. A
    finite proposal that cannot pass the constraints is reported as rejected;
    invalid inputs and nonfinite computations still raise ValueError.
    """
    _basis_schema(basis)
    if not isinstance(current, torch.Tensor) or current.ndim != 2 or not current.numel():
        raise ValueError("current must be a matrix")
    for name, value in (("proposed", proposed), ("current", current),
                        ("reference", reference), ("preservation_gradient", preservation_gradient)):
        if (not isinstance(value, torch.Tensor) or value.shape != current.shape
                or value.dtype != torch.float32 or value.device != current.device):
            raise ValueError(f"{name} must share current's FP32 shape and device")
        _finite(value, name)
    if basis.device != current.device or basis.shape[0] != current.shape[1]:
        raise ValueError("basis device or input dimension differs from current")
    _finite(basis, "basis")
    epsilon = 64 * torch.finfo(torch.float32).eps
    current_delta = current - reference
    _finite(current_delta, "current cumulative delta")
    current_residual = (current_delta @ basis).norm()
    current_norm = current_delta.norm()
    _finite(current_norm, "current cumulative norm")
    current_tolerance = epsilon * max(1., float(current_norm))
    _finite(current_residual, "current subspace residual")
    if float(current_residual) > current_tolerance:
        raise ValueError("current is outside the protected affine subspace")

    delta = proposed - reference
    _finite(delta, "proposed cumulative delta")
    projected = reference + delta - (delta @ basis) @ basis.T
    step = projected - current
    gradient = preservation_gradient
    gp = gradient - (gradient @ basis) @ basis.T
    pre = (gradient * step).sum()
    gp_squared = gp.square().sum()
    for name, value in (("projected proposal", projected), ("projected gradient", gp),
                        ("linear interference", pre), ("projected gradient norm", gp_squared)):
        _finite(value, name)
    correction = torch.zeros_like(step)
    corrected = False
    rejection_reason = None
    if float(pre) > 0:
        if float(gp_squared) <= 0:
            rejection_reason = "no_feasible_preservation_direction"
        else:
            corrected = True
            correction = (pre / gp_squared) * gp
            _finite(correction, "halfspace correction")
            step = step - correction
    candidate = current if rejection_reason else current + step
    _finite(candidate, "candidate")
    actual_step = candidate - current
    post = (gradient * actual_step).sum()
    step_norm, gradient_norm = actual_step.norm(), gradient.norm()
    tolerance = epsilon * gradient_norm * step_norm
    final_delta = candidate - reference
    final_residual = (final_delta @ basis).norm()
    final_norm, correction_norm = final_delta.norm(), correction.norm()
    _finite(final_norm, "final cumulative norm")
    _finite(correction_norm, "correction norm")
    final_tolerance = epsilon * max(1., float(final_norm))
    for name, value in (("post interference", post), ("step norm", step_norm),
                        ("gradient norm", gradient_norm), ("linear tolerance", tolerance),
                        ("final subspace residual", final_residual)):
        _finite(value, name)
    if float(post) > float(tolerance):
        rejection_reason = "fp32_linear_preservation"
    if float(final_residual) > final_tolerance:
        rejection_reason = "fp32_affine_subspace"
    rejected = rejection_reason is not None
    return (current if rejected else candidate).detach(), {
        "rejected": rejected, "rejection_reason": rejection_reason,
        "candidate_post_interference": float(post), "candidate_step_norm": float(step_norm),
        "candidate_linear_tolerance": float(tolerance),
        "candidate_subspace_residual": float(final_residual),
        "candidate_subspace_tolerance": final_tolerance,
        "pre_interference": float(pre), "post_interference": 0. if rejected else float(post),
        "linear_tolerance": 0. if rejected else float(tolerance),
        "step_norm": 0. if rejected else float(step_norm),
        "correction_norm": float(correction_norm), "corrected": corrected,
        "zero_signal": float(gradient_norm) == 0,
        "projected_gradient_norm": float(gp_squared.sqrt()),
        "current_subspace_residual": float(current_residual),
        "cumulative_subspace_residual": float(current_residual if rejected else final_residual),
        "subspace_tolerance": current_tolerance if rejected else final_tolerance}


@torch.no_grad()
def guard_quantized_step(candidate, current, preservation_gradient, *, dtype=torch.bfloat16):
    """Reject a candidate whose actual deployed cast conflicts with the signal.

    A rejection returns current without rolling back Adam's proposal or moments.
    This checks a finite linear signal, not exact quantized subspace preservation.
    The caller records proposal counts separately from accepted weight changes.
    """
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("deployment dtype must be BF16 or FP32")
    if not isinstance(current, torch.Tensor) or current.ndim != 2 or not current.numel():
        raise ValueError("current must be a nonempty FP32 matrix")
    for name, value in (("candidate", candidate), ("current", current),
                        ("preservation_gradient", preservation_gradient)):
        if (not isinstance(value, torch.Tensor) or value.shape != current.shape
                or value.dtype != torch.float32 or value.device != current.device):
            raise ValueError(f"{name} must share current's FP32 shape and device")
        _finite(value, name)
    step = candidate.to(dtype).float() - current.to(dtype).float()
    _finite(step, "quantized step")
    interference = (preservation_gradient * step).sum()
    step_norm, gradient_norm = step.norm(), preservation_gradient.norm()
    tolerance = 64 * torch.finfo(torch.float32).eps * step_norm * gradient_norm
    for name, value in (("quantized interference", interference), ("quantized step norm", step_norm),
                        ("gradient norm", gradient_norm), ("quantized tolerance", tolerance)):
        _finite(value, name)
    rejected = float(interference) > float(tolerance)
    return (current if rejected else candidate).detach(), {
        "deployment_dtype": str(dtype), "quantized_interference": float(interference),
        "quantized_linear_tolerance": float(tolerance), "quantized_step_norm": float(step_norm),
        "quantized_rejected": rejected, "zero_signal": float(gradient_norm) == 0,
        "accepted_quantized_interference": 0. if rejected else float(interference)}


@torch.no_grad()
def guard_retention_losses(candidate, current, baseline_losses, candidate_losses, tolerance=None):
    """Veto any measured per-obligation regression; never average it away.

    Caller must evaluate these ordered losses at the actual deployed candidate
    and the fixed preservation baseline. This is a finite loss constraint, not
    a guarantee about unmeasured inputs, greedy answers or factual correctness.
    No relative budget or accumulating per-step allowance is granted here.
    """
    if (not isinstance(current, torch.Tensor) or not isinstance(candidate, torch.Tensor)
            or current.ndim != 2 or not current.numel() or candidate.shape != current.shape
            or current.dtype != torch.float32 or candidate.dtype != torch.float32
            or candidate.device != current.device):
        raise ValueError("candidate/current must share a nonempty FP32 matrix shape and device")
    for name, value in (("candidate", candidate), ("current", current),
                        ("baseline losses", baseline_losses), ("candidate losses", candidate_losses)):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        _finite(value, name)
    if (baseline_losses.ndim != 1 or not baseline_losses.numel()
            or candidate_losses.shape != baseline_losses.shape
            or baseline_losses.dtype != torch.float32 or candidate_losses.dtype != torch.float32
            or baseline_losses.device != candidate_losses.device
            or bool((baseline_losses < 0).any()) or bool((candidate_losses < 0).any())):
        raise ValueError("retention losses must be matching nonempty nonnegative FP32 vectors")
    if tolerance is None:
        tolerance = torch.zeros_like(baseline_losses)
    if (not isinstance(tolerance, torch.Tensor) or tolerance.shape != baseline_losses.shape
            or tolerance.dtype != torch.float32 or tolerance.device != baseline_losses.device
            or bool((tolerance < 0).any()) or not bool(torch.isfinite(tolerance).all())):
        raise ValueError("retention tolerance must match finite nonnegative FP32 losses")
    changes = candidate_losses - baseline_losses
    excess = changes - tolerance
    count = int((excess > 0).sum())
    return (current if count else candidate).detach(), {
        "rejected": bool(count), "protected_count": baseline_losses.numel(),
        "increased_count": count, "maximum_loss_increase": float(changes.max()),
        "maximum_excess_over_tolerance": float(excess.max()),
        "tolerance_applied": bool(tolerance.any()),
    }
