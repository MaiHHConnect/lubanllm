"""Bounded budget-conditioned latent forecast, not a full H-JEPA reproduction.

Frozen real 2B trajectories; four-layer prompt features alone at inference.
Historical held-out data is diagnostic only. Never publish product weights.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
import tempfile
import time

import torch
from torch import nn
from torch.nn import functional as F

from luban.calibration import SPLITS, evaluate_router
from luban.decision import FastDecisionHead
from luban.dynamic import NeuralExpert
from luban.retention import messages
from verify_hjepa_outcomes import load_inputs, outcome, outcome_routes, route_diagnostics, sha
from verify_router_coverage_study import model_binding

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (4, 8, 32)
SEEDS = (42, 101, 202)
STEPS = 160
SOURCE_NAMES = (
    "scripts/verify_fast_forecast.py", "scripts/verify_hjepa_outcomes.py",
    "scripts/verify_router_budget_counts.py", "scripts/verify_automatic_batches.py",
    "scripts/verify_router_coverage_study.py", "src/luban/runtime.py",
    "src/luban/backbone.py", "src/luban/decision.py", "src/luban/dynamic.py",
    "src/luban/calibration.py", "src/luban/retention.py", "src/luban/context.py",
    "src/luban/settings.py", "src/luban/memory.py", "src/luban/process_lock.py",
    "src/luban/training_checkpoint.py", "src/luban/generation_checks.py",
    "src/luban/coldpaging.py", "src/luban/evolution.py", "src/luban/core_publication.py",
    "src/luban/external_brain.py", "src/luban/backups.py")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def budget_outcomes(answer):
    tokens = answer.get("raw_completion_token_ids")
    if (not isinstance(tokens, list) or not tokens or
            any(type(t) is not int or t < 0 for t in tokens) or
            answer.get("branch_enabled") is not False or answer.get("do_sample") is not False
            or answer.get("thinking") != "off" or answer.get("max_new_tokens") != 512):
        raise ValueError("requires actual once-only greedy512/OFF trajectory")
    final = outcome(answer)
    return [final if len(tokens) <= h else 2 for h in HORIZONS]


class Forecast(nn.Module):
    """Same parameters: direct horizons versus three chained predicted states."""
    def __init__(self, chained):
        super().__init__()
        self.chained = chained
        self.transition = nn.Sequential(nn.Linear(33, 32), nn.Tanh(), nn.Linear(32, 32))
        self.readout = nn.Linear(32, 3)

    def forward(self, initial):
        states, scores, current, previous = [], [], initial, 0
        for horizon in HORIZONS:
            start = current if self.chained else initial
            duration = horizon - previous if self.chained else horizon
            action = torch.full_like(start[:, :1], duration / HORIZONS[-1])
            current = start + self.transition(torch.cat((start, action), dim=1))
            states.append(current)
            scores.append(self.readout(current))
            previous = horizon
        return torch.stack(states, 1), torch.stack(scores, 1)


class AssistedHead(nn.Module):
    """Reuses the branch family at pooled shallow features; experimental location."""
    def __init__(self, initial):
        super().__init__()
        self.head = nn.Linear(2048, 3)
        self.head.load_state_dict(initial)
        self.expert = NeuralExpert(2048, 16)

    def forward(self, x):
        return self.head(x + self.expert(x))


def project_states(prompt, future):
    """Fixed projection; every normalization statistic is computed on TRAIN only."""
    generator = torch.Generator().manual_seed(17)
    projection = torch.linalg.qr(torch.randn(2048, 32, generator=generator)).Q
    raw_prompt = {s: x @ projection for s, x in prompt.items()}
    raw_future = {s: x @ projection for s, x in future.items()}
    train = torch.cat((raw_prompt["train"], raw_future["train"].flatten(0, 1)))
    mean, scale = train.mean(0), train.std(0).clamp_min(1e-4)
    return ({s: (x - mean) / scale for s, x in raw_prompt.items()},
            {s: (x - mean) / scale for s, x in raw_future.items()},
            {"projection": projection, "mean": mean, "scale": scale})


def optimize(model, inputs, labels, futures=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    started = time.perf_counter()
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs["train"])
        if futures is None:
            loss = F.cross_entropy(result, labels["train"])
        else:
            states, logits = result
            loss = F.mse_loss(states, futures["train"]) + F.cross_entropy(
                logits.flatten(0, 1), labels["train"].flatten())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss; retain failure and reject")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    seconds = time.perf_counter() - started
    with torch.no_grad():
        predictions = {s: model(x) for s, x in inputs.items()}
    return predictions, seconds


def infer_latency(model, inputs, normalization=None):
    def predict():
        value = inputs[:1]
        if normalization is not None:
            value = (value @ normalization["projection"] - normalization["mean"]) / normalization["scale"]
        return model(value)
    with torch.no_grad():
        for _ in range(10):
            predict()
        started = time.perf_counter()
        for _ in range(200):
            predict()
    return (time.perf_counter() - started) / 200


def collect(output, model_path, protocol, trial, raw):
    from luban.memory import Memory
    from luban.runtime import Runtime
    from luban.settings import Settings
    import psutil

    rows = {r["id"]: r for r in raw["protocol"]["descriptor"]["pool"]}
    answers = {a["id"]: a for a in raw["qualification"]["answers"]}
    future, prompt, audits, targets = {}, {}, [], {}
    runtime = None
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="luban-forecast-") as home:
        settings = Settings(home)
        private = output / "private-runtime"
        settings.data.update(storage_root=str(private), model_path=str(model_path), local_thinking="off")
        runtime = Runtime(settings, Memory(private), research=True)
        try:
            runtime.load(paging=False)
            if (runtime.identity != protocol["identity"] or runtime.active_core is not None
                    or runtime.branch.expert is not None or runtime.external_binding() is not None):
                raise ValueError("requires exact original core and no expert/reference")
            runtime.model.eval()
            eos = runtime.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            hardware = {"platform": platform.platform(), "device": runtime.device,
                "dtype": str(runtime.dtype), "torch": torch.__version__,
                "physical_memory_bytes": psutil.virtual_memory().total}
            for split in SPLITS:
                pp, ff, tt = [], [], []
                for index, identifier in enumerate(trial["audit"][split]["ids"]):
                    row, answer = rows[identifier], answers[identifier]
                    observed = answer["raw_completion_token_ids"]
                    if answer["complete"] != (observed[-1] in eos):
                        raise ValueError("archived completion/EOS mismatch")
                    if answer["complete"] and any(t in eos for t in observed[:-1]):
                        raise ValueError("tokens after terminal EOS")
                    token_ids = runtime._tokens(messages(row), thinking="off", output_reserve=32, shallow=True)
                    prompt_state = runtime.shallow(token_ids).float().mean(1).cpu()[0]
                    n = min(HORIZONS[-1], len(observed))
                    future_ids = torch.cat((token_ids, torch.tensor([observed[:n]], device=runtime.device)), dim=1)
                    hidden = runtime.shallow(future_ids).float()
                    prefix = hidden[:, :token_ids.shape[1]].mean(1).cpu()[0]
                    causal_error = float((prompt_state - prefix).norm() / prompt_state.norm().clamp_min(1e-8))
                    # Native BF16 sequence-shape roundoff; inference NEVER consumes this suffix pass.
                    if causal_error > 0.02:
                        raise ValueError("future suffix changed prompt states beyond fixed BF16 tolerance")
                    states = torch.stack([hidden[:, :token_ids.shape[1] + min(h, n)].mean(1).cpu()[0]
                                          for h in HORIZONS])
                    if not torch.isfinite(states).all() or not torch.isfinite(prompt_state).all():
                        raise ValueError("nonfinite actual graph states")
                    pp.append(prompt_state); ff.append(states); tt.append(budget_outcomes(answer))
                    audits.append({"id": identifier, "split": split, "prompt_tokens": token_ids.shape[1],
                        "actual_completion_tokens": len(observed), "outcomes": tt[-1],
                        "causal_prefix_relative_l2": causal_error})
                    del hidden, future_ids, token_ids
                    if index % 8 == 0:
                        print(f"actual four-layer states {split} {index + 1}/{len(trial['audit'][split]['ids'])}", flush=True)
                prompt[split], future[split] = torch.stack(pp), torch.stack(ff)
                targets[split] = torch.tensor(tt)
            torch.mps.synchronize() if runtime.device == "mps" else None
            hardware["mps_driver_allocated_bytes_at_end"] = (torch.mps.driver_allocated_memory()
                if runtime.device == "mps" else 0)
        finally:
            if runtime is not None:
                runtime._release_model()
    return prompt, future, targets, audits, hardware, time.perf_counter() - started


def run(output, model_path):
    output, model_path = Path(output).absolute(), Path(model_path).absolute()
    if (not Path("/Volumes/移动数据库").is_mount() or not Path("/Volumes/Untitled").is_mount()
            or output.parent != Path("/Volumes/移动数据库/lubanllm/experiments") or output.exists()):
        raise ValueError("requires both mounted volumes and a fresh experiments directory")
    torch.set_num_threads(1)
    trials, input_hashes, _ = load_inputs()
    trial = trials[-1]  # Frozen largest historical TRAIN: 12 per actual route class.
    descriptor = json.loads((ROOT / "research/router_budget_counts_v2_result.json").read_text())
    raw = json.loads(Path(descriptor["result_path"]).read_text())
    identity = raw["protocol"]["identity"]
    protocol = {"study": "fast-forecast-v1", "scope": "historical diagnostic; no fresh transfer claim",
        "identity": identity, "horizons": list(HORIZONS), "seeds": list(SEEDS), "steps": STEPS,
        "actions": "continue original greedy branch-OFF decoding to total token budget4/8/32",
        "latent": "frozen real four-layer prefix mean; fixed 32D orthogonal random projection seed17",
        "fitting": "TRAIN-only normalization; Adam .005 clip1; rollout uses predicted states during TRAIN and TEST",
        "forecast_loss": "mean normalized latent MSE + mean three-outcome CE across all horizons",
        "causal_tolerance": "BF16 suffix/prompt mean relative L2 <= .02; standalone prompt only at inference",
        "comparison": "current linear; shallow SwiGLU branch; equal-parameter direct vs chained forecaster",
        "branch_control_scope": "new width16 branch at pooled early features; NOT current accepted last-layer expert",
        "planning_scope": "budget forecasts only; no top-down goal/subgoal planning; NOT full H-JEPA",
        "route_endpoint": "32-token success→local, wrong/incomplete→teacher suggestion; compare original512 labels",
        "source_sha256": {n: sha(ROOT / n) for n in SOURCE_NAMES}, "input_sha256": input_hashes,
        "model_binding": model_binding(model_path, identity), "teacher_calls": 0, "product_publication": False}
    output.mkdir()
    (output / "protocol.json").write_text(encoded(protocol))
    prompt, future, targets, audit, hardware, collection_seconds = collect(output, model_path, protocol, trial, raw)
    initial, projected, normalization = project_states(prompt, future)
    torch.save({"prompt": prompt, "future": future, "targets": targets, "normalization": normalization,
                "protocol": protocol, "audit": audit}, output / "trajectory-features.pt")
    original_targets = trial["payload"]["targets"]
    if any(not torch.equal((targets[s][:, -1] != 0).long() * 2, original_targets[s]) for s in SPLITS):
        raise ValueError("32-token endpoint differs from original512 route labels; do not silently change baseline")
    report = {"protocol": protocol, "hardware": hardware, "collection_seconds": collection_seconds,
        "audit": audit, "trials": [], "teacher_calls": 0, "product_accepted": False,
        "limits": ["Same already observed TEST24 for every seed", "No current compatible active 2B expert",
                   "Replayed actual greedy trajectories, not a fresh closed-loop rollout or response quality improvement",
                   "Future state error alone does not establish useful action information", "No goal/subgoal planning"]}
    for seed in SEEDS:
        torch.manual_seed(seed)
        head_initial = copy.deepcopy(nn.Linear(2048, 3).state_dict())
        linear = nn.Linear(2048, 3); linear.load_state_dict(head_initial)
        torch.manual_seed(seed)
        assisted = AssistedHead(head_initial)
        with torch.no_grad():
            if not torch.equal(linear(prompt["train"]), assisted(prompt["train"])):
                raise ValueError("zero branch violates actual feature graph equivalence")
        torch.manual_seed(seed)
        direct = Forecast(False)
        chained = Forecast(True); chained.load_state_dict(direct.state_dict())
        record = {"seed": seed, "models": {}}
        for name, model in (("current_linear", linear), ("shallow_swiglu", assisted),
                            ("direct_forecast", direct), ("chained_forecast", chained)):
            forecasts = isinstance(model, Forecast)
            inputs, labels = (initial, targets) if forecasts else (prompt, original_targets)
            predictions, seconds = optimize(model, inputs, labels, projected if forecasts else None)
            logits = ({s: outcome_routes(predictions[s][1][:, -1]) for s in SPLITS}
                      if forecasts else predictions)
            result = evaluate_router(logits, original_targets, FastDecisionHead.actions, ["local_core", "teacher"])
            data = {"parameters": sum(p.numel() for p in model.parameters()), "train_seconds": seconds,
                "cpu_single_prompt_seconds": infer_latency(model, prompt["test"], normalization if forecasts else None),
                "fixed_projection_and_statistics_bytes": sum(x.numel() * x.element_size()
                    for x in normalization.values()) if forecasts else 0,
                "routes": route_diagnostics(result), "full_route_evaluation": result}
            if forecasts:
                data["outcome_accuracy_per_horizon"] = {s: (predictions[s][1].argmax(-1) == targets[s])
                    .float().mean(0).tolist() for s in SPLITS}
                data["test_future_mse"] = (predictions["test"][0] - projected["test"]).square().mean((0, 2)).tolist()
                data["test_copy_prompt_mse"] = (initial["test"][:, None] - projected["test"]).square().mean((0, 2)).tolist()
            record["models"][name] = data
            torch.save(model.state_dict(), output / f"{name}-{seed}.pt")
            print(encoded({"seed": seed, "model": name, "routes": data["routes"]}), flush=True)
        report["trials"].append(record)
    if any(sha(ROOT / n) != v for n, v in protocol["source_sha256"].items()):
        raise ValueError("execution sources changed; result is not a bound completed comparison")
    if any(sha(p) != v for p, v in input_hashes.items()):
        raise ValueError("historical inputs changed")
    if model_binding(model_path, identity) != protocol["model_binding"]:
        raise ValueError("actual official backbone files changed")
    report.update(inputs_unchanged=True, sources_unchanged=True,
        process_high_water_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory_method="Darwin ru_maxrss process high-water bytes; end MPS allocation is NOT peak",
        feature_file_sha256=sha(output / "trajectory-features.pt"))
    (output / "report.json").write_text(encoded(report))
    print("Bound comparison completed; no production publication.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    run(args.output, args.model_path)
