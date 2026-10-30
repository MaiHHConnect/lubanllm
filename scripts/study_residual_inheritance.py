"""CPU-only synthetic parent D / new E study; never loads archived weights.

Uses the official random tiny BF16 Qwen3.5 fixture. A synthetic random parent is
not a learned parent, and finite TRAIN preservation says nothing about 2B
recursive learning. Arithmetic is one FP32 D+E sum and one residual cast.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import torch
from test_cached_training import fixture
from luban.core_update import apply_core_residual, constrain_residual, validate_basis
from luban.evolution_progress import extend_protected_basis

EXPERIMENTS = Path("/Volumes/移动数据库/lubanllm/experiments/preservation-first-20261015-v1")
SEED, PARENT_SEED = 42, 43
SCALES = (1., .25, .0625, .015625)


def study(output):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(SEED)
    report = {
        "scope": "synthetic random tiny parent; not learned knowledge or 2B recursion",
        "seed": SEED, "parent_seed": PARENT_SEED, "device": "cpu",
        "model_dtype": "torch.bfloat16", "residual_dtype": "torch.float32",
        "torch_version": torch.__version__, "threads": torch.get_num_threads(),
        "hardware": platform.platform(), "machine": platform.machine(),
        "memory_measurement": "process lifetime ru_maxrss; bytes on Darwin, KiB on Linux; no MPS",
        "arithmetic": "(base_BF16_output.float() + linear(x.float(), D_parent + E)).bfloat16()",
        "scales": SCALES, "adam": {"lr": .01, "betas": [.9, .999], "eps": 1e-6,
                                     "weight_decay": 0., "updates": 1},
        "acceptance": "parent full logits exactly equal, parent CE does not rise, NEW CE decreases",
        "production_adoption": False,
    }
    raw = {}
    try:
        with tempfile.TemporaryDirectory(prefix="luban-tiny-inheritance-") as scratch:
            runtime, rows, tokens = fixture(scratch, torch.bfloat16)
            runtime.model.eval().requires_grad_(False)
            assert runtime.branch.expert is None
            projection = runtime.branch.base.down_proj
            original = [(p, p.detach().clone()) for p in runtime.model.parameters()]
            parent = torch.randn(projection.weight.shape,
                                 generator=torch.Generator().manual_seed(PARENT_SEED)) * .02
            parent_before = parent.clone()
            zero = torch.zeros_like(parent)
            protect, new = rows[:2]
            report["train_rows"] = {"parent_protection": protect["id"], "new": new["id"],
                                    "unused_fixture_row": rows[2]["id"]}
            report["tokens"] = {row["id"]: {"ids": tokens[row["id"]][0].tolist(),
                                              "labels": tokens[row["id"]][1].tolist()}
                               for row in (protect, new)}

            def forward(total, row, capture=False):
                captured = []

                def hook(module, arguments, result):
                    if capture:
                        captured.append(arguments[0].detach().float().clone())
                    return apply_core_residual(arguments[0], result, total)

                handle = projection.register_forward_hook(hook)
                try:
                    ids, labels = tokens[row["id"]]
                    result = runtime.model(ids, labels=labels, use_cache=False)
                    if capture and len(captured) != 1:
                        raise ValueError("expected one last projection execution")
                    return result.logits, result.loss, captured
                finally:
                    handle.remove()

            with torch.no_grad():
                base_logits, base_loss, _ = forward(zero, protect)
                parent_logits, parent_loss, captured = forward(parent, protect, True)
                parent_new_logits, parent_new_loss, _ = forward(parent, new)
                zero_logits, zero_loss, _ = forward(parent + zero, protect)
            # Preserve every input position, including prompt and final position.
            basis, expansion = extend_protected_basis(
                torch.empty(parent.shape[1], 0), captured[0].reshape(-1, parent.shape[1]))
            report["basis"] = {"shape": list(basis.shape), **expansion, **validate_basis(basis)}
            report["zero_E"] = {"exact_parent_logits": torch.equal(zero_logits, parent_logits),
                                  "parent_ce": float(parent_loss), "zero_E_ce": float(zero_loss)}
            report["synthetic_parent"] = {
                "norm": float(parent.norm()), "base_ce": float(base_loss),
                "protected_span_norm": float((parent @ basis).norm()),
                "base_to_parent_logits_max_abs": float((base_logits - parent_logits).abs().max()),
                "new_ce": float(parent_new_loss)}

            increment = torch.nn.Parameter(zero.clone())
            _, protection_loss, _ = forward(parent + increment, protect)
            protection_gradient, = torch.autograd.grad(protection_loss, increment)
            optimizer = torch.optim.Adam([increment], **{
                key: value for key, value in report["adam"].items() if key != "updates"})
            _, new_loss, _ = forward(parent + increment, new)
            new_loss.backward()
            optimizer.step()
            proposal = increment.detach().clone()
            candidate, constraint = constrain_residual(proposal, zero, basis, [protection_gradient])
            report["constraint"] = constraint
            raw.update(parent=parent, basis=basis, protected_inputs=captured[0],
                       parent_logits=parent_logits, base_logits=base_logits,
                       parent_new_logits=parent_new_logits, proposal_E=proposal,
                       constrained_E=candidate, protection_gradient=protection_gradient)

            def measure(name, total, extra):
                with torch.no_grad():
                    logits, loss, _ = forward(total, protect)
                    new_logits, new_ce, _ = forward(total, new)
                exact = torch.equal(logits, parent_logits)
                value = {"name": name, **extra, "parent_logits_exact": exact,
                         "parent_logits_max_abs": float((logits - parent_logits).abs().max()),
                         "parent_ce": float(loss), "parent_ce_change": float(loss - parent_loss),
                         "new_ce": float(new_ce), "new_ce_change": float(new_ce - parent_new_loss)}
                value["passes_fixed_gate"] = exact and float(loss) <= float(parent_loss) and float(new_ce) < float(parent_new_loss)
                raw[name] = {"total_FP32": total, "parent_logits": logits,
                             "new_logits": new_logits, "parent_ce": loss, "new_ce": new_ce}
                return value

            naive = parent - (parent @ basis) @ basis.T
            naive -= (naive @ basis) @ basis.T
            report["naive_reproject_parent"] = measure("naive_reproject_parent", naive, {})
            report["steps"] = []
            for scale in SCALES:
                e = candidate * scale
                total = parent + e  # Same single FP32 addition during training and observation.
                report["steps"].append(measure(f"only_E_scale_{scale:g}", total, {
                    "scale": scale, "E_norm": float(e.norm()),
                    "E_protected_span_norm": float((e @ basis).norm())}))
            report["selected_scale"] = next((v["scale"] for v in report["steps"]
                                               if v["passes_fixed_gate"]), None)
            report["original_graph_weights_unchanged"] = all(torch.equal(p, old) for p, old in original)
            report["parent_D_unchanged"] = torch.equal(parent, parent_before)
            report["projection_hooks_removed"] = not projection._forward_hooks
            report["zero_E_verified"] = report["zero_E"]["exact_parent_logits"] and float(zero_loss) == float(parent_loss)
            report["naive_parent_break_observed"] = not report["naive_reproject_parent"]["parent_logits_exact"]
            if not all(report[key] for key in ("original_graph_weights_unchanged", "parent_D_unchanged",
                                               "projection_hooks_removed", "zero_E_verified")):
                raise AssertionError("invariant failed; see raw report")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["lifetime_peak_rss_native"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        torch.save(raw, output / "raw.pt")  # Synthetic increments/logits only; no base weights.
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"output": str(output), **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not os.path.ismount("/Volumes/移动数据库"):
        raise SystemExit("required experiment volume is not mounted")
    destination = arguments.output or EXPERIMENTS / ("synthetic-inheritance-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    destination = destination.resolve()
    if not destination.is_relative_to(EXPERIMENTS.resolve()) or destination == EXPERIMENTS.resolve():
        raise SystemExit("output must be a new directory under the authorized experiment root")
    destination.mkdir(exist_ok=False)
    study(destination)
