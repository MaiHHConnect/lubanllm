"""Audit existing core training operators; this never trains or publishes a core.

Real-model loading belongs to the caller, after exclusive execution is granted.
The standalone command only runs an offline tiny official Qwen3.5 CPU fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from luban.evolution import _cancel, _target, _field_weighted_loss, field_first_token_policy


# Fixed before real-model execution. Same shapes/operators should normally be
# exact; BF16 allows one unit of relative rounding, never a nonfinite result.
OPERATOR_AUDIT_PROTOCOL = "v1-existing-core-last-mlp-ce-gradient-adam"
FIELD_OPERATOR_AUDIT_PROTOCOL = "v2-existing-core-field-first-gold-weight4"
TOLERANCES = {
    "torch.bfloat16": {
        "logits_atol": 2 ** -10, "rtol": 2 ** -7, "relative_l2": 2 ** -7,
        "loss_atol": 2 ** -10, "gradient_atol": 2 ** -16,
        "adam_atol": 1e-7, "update_relative_l2": 2 ** -6,
    },
    "torch.float32": {
        "logits_atol": 1e-6, "rtol": 1e-5, "relative_l2": 1e-5,
        "loss_atol": 1e-6, "gradient_atol": 1e-7,
        "adam_atol": 1e-7, "update_relative_l2": 1e-5,
    },
}


def _comparison(actual, expected, *, atol, rtol, relative_l2):
    if actual.shape != expected.shape:
        raise ValueError("Operator audit tensors have different shapes")
    actual, expected = actual.detach().float(), expected.detach().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"max_abs_diff": None, "relative_l2_diff": None, "finite": False,
                "exact": False, "passed": False}
    delta = actual - expected
    maximum = float(delta.abs().max().cpu())
    relative = float((delta.norm() / expected.norm().clamp_min(1e-30)).cpu())
    return {"max_abs_diff": maximum, "relative_l2_diff": relative,
            "finite": finite, "exact": bool(torch.equal(actual, expected)),
            "passed": finite and bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
                      and relative <= relative_l2}


def _adam_step(weight, gradient):
    # Match evolution.py: BF16 gradient -> FP32 master, clip 1, Adam eps 1e-6.
    master = torch.nn.Parameter(weight.detach().float().clone())
    master.grad = gradient.detach().float().clone()
    norm = torch.nn.utils.clip_grad_norm_([master], 1.0, error_if_nonfinite=True)
    optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
    optimizer.step()
    return master.detach(), float(norm.cpu())


def audit_training_operators(runtime, sample, *, field_first_token_weight=1):
    """Compare full upstream CE/autograd against its captured last-MLP tail.

    Returns a JSON-compatible report. A failed numerical comparison returns
    passed=False; structural errors raise after restoration. No generation,
    fitting, optimizer-state continuation, KL or projected-update equivalence
    is claimed by this single-sample audit.
    """
    if isinstance(field_first_token_weight, bool) or not isinstance(field_first_token_weight, (int, float)) or field_first_token_weight not in {1, 4}:
        raise ValueError("Operator audit first-gold weight must be1 or4")
    with runtime.lock, torch.enable_grad():
        model, decoder, branch = runtime.model, runtime.decoder, runtime.branch
        name, parameter = _target(runtime)
        dtype = str(parameter.dtype)
        if dtype not in TOLERANCES:
            raise ValueError("Operator audit supports FP32 and BF16 only")
        tolerance = TOLERANCES[dtype]
        parameters = list(model.parameters())
        flags = [(value, value.requires_grad, value.grad) for value in parameters]
        modes = [(module, module.training) for module in model.modules()]
        original_expert = branch.expert
        original_weight = parameter.detach().clone()
        pointer = Path(runtime.store) / "active-core.json"
        original_pointer = pointer.read_bytes() if pointer.exists() else None
        version = getattr(runtime, "active_core", None)
        captured, handles = {}, []
        restored = False
        try:
            _cancel(runtime)
            branch.expert = None
            model.eval().requires_grad_(False)
            parameter.requires_grad_(True)
            ids, labels = runtime._sample_tokens(sample)
            if ids.ndim != 2 or labels.shape != ids.shape or ids.shape[1] < 2:
                raise ValueError("Operator audit needs matching causal sequences")
            if not bool((labels[:, 1:] != -100).any()):
                raise ValueError("Operator audit has no answer/EOS targets")
            if field_first_token_weight == 4:
                source = json.loads(sample["provenance"])
                prefix = runtime._tokens(source["messages"], "off")
                gold = runtime.tokenizer.encode(sample["answer"], add_special_tokens=False) + [runtime.tokenizer.eos_token_id]
                size = prefix.shape[1]
                if (prefix.shape[0] != 1 or not torch.equal(prefix, ids[:, :size])
                        or not labels[:, :size].eq(-100).all() or labels[0, size:].tolist() != gold
                        or ids[0, size:].tolist() != gold):
                    raise ValueError("Weighted operator audit inference prefix/gold+EOS mismatch")
            layer = decoder.layers[-1]
            mlp = getattr(layer.mlp, "base", layer.mlp)

            def capture(key):
                def hook(module, args):
                    if key in captured:
                        raise RuntimeError("Last-layer audit hook executed more than once")
                    captured[key] = args[0].detach().clone()
                return hook

            # Capture R directly BEFORE post-attention normalization; subtracting
            # WX from the post-MLP residual would lose BF16 rounding information.
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(capture("residual")))
            handles.append(mlp.down_proj.register_forward_pre_hook(capture("projection_input")))
            output = model(ids, labels=labels, use_cache=False)
            for handle in handles:
                handle.remove()
            _cancel(runtime)
            if set(captured) != {"residual", "projection_input"}:
                raise RuntimeError("The real last-layer graph was not captured")
            if captured["projection_input"].shape[:2] != labels.shape:
                raise ValueError("Projection input is not the complete sequence")
            native_logits = output.logits.detach()
            native_objective = _field_weighted_loss(output.logits, labels) if field_first_token_weight == 4 else output.loss
            native_loss = native_objective.detach()
            native_mean_loss = float(output.loss.detach().float().cpu())
            if not bool(torch.isfinite(native_loss)):
                raise RuntimeError("Native operator audit loss is nonfinite")
            native_gradient, = torch.autograd.grad(native_objective, parameter)
            del output

            tail_weight = parameter.detach().clone().requires_grad_(True)
            hidden = F.linear(captured["projection_input"], tail_weight, mlp.down_proj.bias)
            tail_logits = model.lm_head(decoder.norm(hidden + captured["residual"]))
            if field_first_token_weight == 4:
                targets = labels[:, 1:].reshape(-1)
                per_token = F.cross_entropy(tail_logits[:, :-1].float().reshape(-1, tail_logits.shape[-1]),
                                           targets, ignore_index=-100, reduction="none")[targets != -100]
                weights = torch.ones_like(per_token)
                weights[0] = 4
                tail_loss = (per_token * weights).sum() / weights.sum()
            else:
                tail_loss = F.cross_entropy(tail_logits[:, :-1].float().reshape(-1, tail_logits.shape[-1]),
                                            labels[:, 1:].reshape(-1), ignore_index=-100)
            if not bool(torch.isfinite(tail_loss)):
                raise RuntimeError("Independent operator audit loss is nonfinite")
            tail_gradient, = torch.autograd.grad(tail_loss, tail_weight)
            _cancel(runtime)
            logits = _comparison(tail_logits, native_logits, atol=tolerance["logits_atol"],
                                 rtol=tolerance["rtol"], relative_l2=tolerance["relative_l2"])
            gradients = _comparison(tail_gradient, native_gradient, atol=tolerance["gradient_atol"],
                                    rtol=tolerance["rtol"], relative_l2=tolerance["relative_l2"])
            gradients.update(native_nonzero=bool(torch.count_nonzero(native_gradient)),
                             tail_nonzero=bool(torch.count_nonzero(tail_gradient)),
                             native_dtype=str(native_gradient.dtype), tail_dtype=str(tail_gradient.dtype))
            loss_difference = float((tail_loss.detach() - native_loss).abs().cpu())
            loss = {"native": float(native_loss.cpu()), "tail": float(tail_loss.detach().cpu()),
                    "abs_diff": loss_difference,
                    "finite": bool(torch.isfinite(tail_loss) and torch.isfinite(native_loss)),
                    "passed": bool(torch.isfinite(tail_loss) and torch.isfinite(native_loss))
                              and loss_difference <= tolerance["loss_atol"]}

            native_master, native_norm = _adam_step(original_weight, native_gradient)
            tail_master, tail_norm = _adam_step(original_weight, tail_gradient)
            optimizer = _comparison(tail_master, native_master, atol=tolerance["adam_atol"],
                                    rtol=0, relative_l2=tolerance["relative_l2"])
            update = _comparison(tail_master - original_weight.float(),
                                 native_master - original_weight.float(), atol=tolerance["adam_atol"],
                                 rtol=tolerance["rtol"], relative_l2=tolerance["update_relative_l2"])
            optimizer.update(native_preclip_norm=native_norm, tail_preclip_norm=tail_norm,
                             update=update, optimizer="torch.optim.Adam", learning_rate=1e-5,
                             eps=1e-6, clip_norm=1.0, master_dtype="torch.float32",
                             state="fresh independent optimizer state, one step")
            _cancel(runtime)
            report = {"kind": "existing_core_training_operator_audit", "passed": False,
                      "protocol_version": FIELD_OPERATOR_AUDIT_PROTOCOL if field_first_token_weight == 4 else OPERATOR_AUDIT_PROTOCOL,
                      "identity": getattr(runtime, "identity", None), "core_version": version,
                      "sample_id": sample.get("id"), "teacher_calls": 0, "publication": False,
                      "sample_purpose": sample.get("purpose"), "rng_scope": "eval mode; no RNG reseeding or initialization",
                      "parameter": name, "trained_parameters": parameter.numel(),
                      "device": str(parameter.device), "precision": dtype,
                      "hardware": platform.platform(), "torch_version": torch.__version__,
                      "tolerances": tolerance.copy(), "branch_enabled": False,
                      "input_tokens": ids.numel(), "target_tokens": int((labels[:, 1:] != -100).sum()),
                      "token_sha256": hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
                      "label_sha256": hashlib.sha256(labels.detach().cpu().numpy().tobytes()).hexdigest(),
                      "capture_shapes": {key: list(value.shape) for key, value in captured.items()},
                      "logits": logits, "cross_entropy": loss, "gradient": gradients, "adam": optimizer,
                      "scope": "one teacher-forcing sample; captured tail is audit-only, not production training; "
                               "no memory/speed, generation, KL, subspace projection or broad accuracy claim"}
            report["passed"] = all((logits["passed"], loss["passed"], gradients["passed"],
                                    gradients["native_nonzero"], gradients["tail_nonzero"],
                                    optimizer["passed"], update["passed"]))
            if field_first_token_weight == 4:
                report.update(field_loss_policy=field_first_token_policy(), native_unweighted_mean_ce=native_mean_loss,
                    scope="fixed first-gold weighted objective native vs independent same-dtype tail and clone Adam; not equivalence to old mean CE, no production weight update")
        finally:
            for handle in handles:
                handle.remove()
            with torch.no_grad():
                # Optimizers only touched clones, but restoration also covers a
                # failing upstream/custom hook that unexpectedly changed W.
                if not torch.equal(parameter, original_weight):
                    parameter.copy_(original_weight)
            for value, requires_grad, gradient in flags:
                value.requires_grad_(requires_grad)
                value.grad = gradient
            branch.expert = original_expert
            for module, training in modes:
                module.training = training
            current_pointer = pointer.read_bytes() if pointer.exists() else None
            restored = (torch.equal(parameter, original_weight) and branch.expert is original_expert
                        and current_pointer == original_pointer and getattr(runtime, "active_core", None) == version
                        and all(value.requires_grad == flag and value.grad is grad for value, flag, grad in flags))
            if not restored:
                raise RuntimeError("Operator audit did not preserve original core state")
        report["restored"] = restored
        return report


def _self_test():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from luban.backbone import text_backbone
    from luban.dynamic import DynamicMLP

    reports = []
    with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as folder:
        torch.manual_seed(42)
        config = Qwen3_5Config(text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 128, "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        }, vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                          "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                          "spatial_merge_size": 2, "num_position_embeddings": 16},
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
            tie_word_embeddings=True)
        model = Qwen3_5ForConditionalGeneration(config).cpu().eval()
        decoder, _ = text_backbone(model)
        branch = DynamicMLP(decoder.layers[-1].mlp, 32, 8, Path(folder), "fixture", 100000)
        decoder.layers[-1].mlp = branch
        branch.expert = torch.nn.Linear(32, 32, bias=False)
        ids = torch.tensor([[3, 7, 13, 19, 23, 29, 31, 2]])
        labels = ids.clone()
        labels[:, :4] = -100
        runtime = SimpleNamespace(model=model, decoder=decoder, branch=branch, store=folder,
                                  lock=threading.RLock(), active_core=None, job=None, identity="tiny-official-qwen35",
                                  _sample_tokens=lambda sample: (ids, labels))
        pointer = Path(folder) / "active-core.json"
        pointer.write_bytes(b'{"fixture":"unchanged"}')
        original_expert = branch.expert
        for dtype in (torch.float32, torch.bfloat16):
            model.to(dtype=dtype)
            parameter = branch.base.down_proj.weight
            parameter.grad = torch.ones_like(parameter)
            original_gradient = parameter.grad
            report = audit_training_operators(runtime, {"id": "offline-fixture"})
            if not report["passed"] or parameter.grad is not original_gradient:
                raise AssertionError(report)
            reports.append(report)

        original_forward = model.forward

        def incorrect_loss(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            output.loss = output.loss + 0.1
            return output

        model.forward = incorrect_loss
        rejected = audit_training_operators(runtime, {"id": "deliberate-loss-mismatch"})
        if rejected["passed"] or rejected["cross_entropy"]["passed"]:
            raise AssertionError("Audit accepted a deliberately corrupted native loss")
        model.forward = original_forward
        parameter = branch.base.down_proj.weight
        previous_requires_grad = parameter.requires_grad
        parameter.requires_grad_(True)
        handle = parameter.register_hook(lambda gradient: gradient * 2)
        try:
            corrupted_gradient = audit_training_operators(runtime, {"id": "deliberate-gradient-mismatch"})
            if corrupted_gradient["passed"] or corrupted_gradient["gradient"]["passed"]:
                raise AssertionError("Audit accepted a deliberately corrupted native gradient")
        finally:
            handle.remove()
            parameter.requires_grad_(previous_requires_grad)
        weight = parameter.detach().clone()
        saved_modes = [(module, module.training) for module in model.modules()]
        before_hooks = (len(branch.base.down_proj._forward_pre_hooks),
                        len(decoder.layers[-1].post_attention_layernorm._forward_pre_hooks))

        def fail_head(module, args):
            with torch.no_grad():
                parameter.add_(0.01)
            raise RuntimeError("injected upstream failure")

        handle = model.lm_head.register_forward_pre_hook(fail_head)
        try:
            try:
                audit_training_operators(runtime, {"id": "deliberate-forward-failure"})
            except RuntimeError as error:
                if str(error) != "injected upstream failure":
                    raise
            else:
                raise AssertionError("Injected upstream failure did not execute")
        finally:
            handle.remove()
        after_hooks = (len(branch.base.down_proj._forward_pre_hooks),
                       len(decoder.layers[-1].post_attention_layernorm._forward_pre_hooks))
        if (before_hooks != after_hooks or branch.expert is not original_expert
                or not torch.equal(parameter, weight)
                or not all(module.training == mode for module, mode in saved_modes)
                or pointer.read_bytes() != b'{"fixture":"unchanged"}'):
            raise AssertionError("Audit did not restore hooks, modes, weights, pointer or branch")
        large_gradient = torch.ones_like(parameter) * 10
        left, preclip = _adam_step(weight, large_gradient)
        right, repeated_preclip = _adam_step(weight, large_gradient)
        if preclip <= 1 or preclip != repeated_preclip or not torch.equal(left, right):
            raise AssertionError("Independent Adam steps did not match with active clipping")

        def cancel_head(module, args):
            runtime.job = {"cancel_requested": True}

        handle = model.lm_head.register_forward_pre_hook(cancel_head)
        try:
            try:
                audit_training_operators(runtime, {"id": "deliberate-cancellation"})
            except RuntimeError as error:
                if "已取消" not in str(error):
                    raise
            else:
                raise AssertionError("Audit did not respect cancellation")
        finally:
            handle.remove()
            runtime.job = None
        if (branch.expert is not original_expert or not torch.equal(parameter, weight)
                or parameter.requires_grad != previous_requires_grad
                or parameter.grad is not original_gradient
                or pointer.read_bytes() != b'{"fixture":"unchanged"}'
                or before_hooks != (len(branch.base.down_proj._forward_pre_hooks),
                                    len(decoder.layers[-1].post_attention_layernorm._forward_pre_hooks))):
            raise AssertionError("Cancellation did not restore the original state")
    return {"kind": "offline_cpu_operator_audit_self_test", "passed": True,
            "seed": 42, "reports": reports, "deliberate_loss_mismatch_rejected": True,
            "deliberate_gradient_mismatch_rejected": True, "active_clipping_verified": True,
            "injected_forward_failure_restored": True, "cancellation_restored": True, "teacher_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True,
                        help="Only an offline tiny official CPU model; never loads production weights")
    parser.parse_args()
    print(json.dumps(_self_test(), ensure_ascii=False, indent=2, allow_nan=False))
