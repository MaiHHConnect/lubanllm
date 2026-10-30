"""Use the resident MLX prefix; never load a second BF16 routing backbone."""
import threading

import torch


def feature_source(runtime):
    return "mlx-q4-prefix" if getattr(runtime, "native_inference_precision", None) == "mlx-q4" else "hf-bf16-prefix"


def source_matches(runtime, metadata):
    # Unmarked historical heads were calibrated against the original HF graph.
    return metadata.get("feature_source", "hf-bf16-prefix") == feature_source(runtime)


def pooled_features(runtime, tokens):
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("MLX路由只接受单条完整上下文")
    from . import product_inference
    engine = getattr(runtime, "native_engine", None) or product_inference.start(runtime)
    if engine is None:
        raise RuntimeError("MLX路由引擎不可用")
    values = engine.route_features(tokens[0].detach().cpu().tolist(), threading.Event())
    return torch.tensor(values, dtype=torch.float32, device=runtime.device).reshape(1, 1, 2048)


def prove_expert_rows(runtime, rows):
    """One bound native expert session for the whole closed-book route check."""
    from . import product_inference
    from .evolution import _cancel, _messages, _provenance
    from .generation_checks import evaluate_generation, validate_generation_check

    binding = product_inference.expert_binding(runtime, product_inference.native_binary(runtime))
    if binding is None:
        raise ValueError("MLX专家尚未通过当前引擎验证")
    original = runtime.branch.expert
    stop = threading.Event()
    results = []
    try:
        product_inference.close(runtime)
        if original is None:
            runtime.branch.activate(runtime.branch.expert_id, device="cpu")
        engine = product_inference.start(runtime)
        if engine.expert != binding:
            raise ValueError("MLX专家验证来源不一致")
        def emitted(_):
            if (runtime.job or {}).get("cancel_requested"):
                stop.set()
        for row in rows:
            _cancel(runtime)
            check = validate_generation_check(_provenance(row).get("generation_check"), default_expected=row["answer"])
            maximum = check.get("max_new_tokens", 192)
            ids = runtime._tokens(_messages(row), "off", output_reserve=maximum)[0].tolist()
            value = engine.generate(ids, maximum, product_inference.eos_token_ids(runtime), 0,
                                    emitted, stop, seed=42, cache_reuse=False)
            _cancel(runtime)
            answer = runtime.tokenizer.decode(value["tokens"], skip_special_tokens=True).strip()
            results.append({"id": row["id"], "answer": answer,
                **evaluate_generation(answer, check, value["finish_reason"] == "stop"),
                "branch_enabled": True, "reference_used": False})
        return results
    finally:
        product_inference.close(runtime)
        runtime.branch.expert = original
