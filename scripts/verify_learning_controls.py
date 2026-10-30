"""Verify real learned-weight residency and thinking in an isolated disk copy.

No training, teacher requests, credentials, or writes to production modules.
The live service may remain running; this loads one additional BF16 MPS model.
"""
import gc
import hashlib
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from luban.backups import save_training_backup
from luban.runtime import Runtime
from luban.settings import Settings, default_home, private_json, require_storage
from luban.thinking import ThinkingStream
from train_identity import TARGET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(runtime, thinking="off", maximum=128, messages=None):
    raw, deltas = [], []
    parser = ThinkingStream(deltas.append)

    def emit(delta):
        raw.append(delta["content"])
        parser.feed(delta["content"])

    usage = runtime.generate(messages or [{"role": "user", "content": "你是谁"}], emit,
                             threading.Event(), thinking=thinking,
                             max_tokens=maximum, temperature=0)
    parser.finish()
    content = "".join(row.get("content", "") for row in deltas)
    reasoning = "".join(row.get("reasoning_content", "") for row in deltas)
    return {"thinking": thinking, "raw": "".join(raw), "content": content,
            "reasoning_content": reasoning, "usage": usage,
            "identity_exact": content.strip() == TARGET,
            "identity_occurrences_in_content": content.count(TARGET),
            "brand_occurrences_in_content": content.count("麦好火"),
            "raw_tags_in_content": "<think>" in content or "</think>" in content}


def state_copy(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def same_state(module, expected):
    return all(torch.equal(value.detach().cpu(), expected[key])
               for key, value in module.state_dict().items())


def main():
    started = time.perf_counter()
    # Read only public configuration fields; never instantiate real Settings,
    # whose constructor persists configuration and reads credentials.
    config = json.loads((default_home() / "settings.json").read_text())
    local = {key: config[key] for key in ("model_path", "storage_root", "local_thinking",
                                         "max_new_tokens", "context_tokens", "temperature")}
    production = require_storage(local["storage_root"])
    source = production / "modules/app"
    active = json.loads((source / "active.json").read_text())
    layer_meta = next(source.glob(f"layer-*/{active['id']}.json"))
    names = ["active.json", "router.json", "router.safetensors",
             layer_meta.relative_to(source).as_posix(),
             layer_meta.with_suffix(".safetensors").relative_to(source).as_posix()]
    before = {name: digest(source / name) for name in names}
    report = {"scope": "isolated real production checkpoint copy; no training or teacher calls",
              "seed": 42, "temperature": 0, "production_hashes_before": before,
              "checks": {}, "cases": [], "training_performed": False}
    experiments = production / "experiments"
    experiments.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="learning-controls-", dir=experiments) as store, \
            tempfile.TemporaryDirectory(prefix="learning-controls-home-") as home:
        copied = Path(store) / "modules/app"
        for name in names:
            target = copied / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        settings = Settings(home)
        settings.data.update(local | {"storage_root": store})
        runtime = Runtime(settings, None)
        torch.manual_seed(42)
        runtime.load()
        report.update(device=runtime.device, dtype=str(runtime.dtype),
                      identity=runtime.identity, expert_id=runtime.branch.expert_id,
                      extra_parameters=runtime.branch.resident_expert_params,
                      router_sha256=runtime.router_meta.get("sha256"),
                      initial_status=runtime.status(),
                      memory_measurement="psutil process RSS and Metal driver allocated bytes; not physical peak")
        messages = [{"role": "user", "content": "你是谁"}]
        saved_meta = runtime.router_meta.copy()
        runtime.hibernate()
        assert runtime.branch.expert is None
        decision = runtime.choose_route(messages)
        assert decision.get("action") == "neural_expert", decision
        assert runtime.model.model.layers[-1].mlp is runtime.branch
        restored = runtime.branch.expert
        assert restored is not None
        report["checks"]["route_automatically_restores_residency"] = decision
        runtime.hibernate()
        assert runtime.branch.expert is None
        answer = generate(runtime)
        assert answer["identity_exact"], answer
        assert runtime.branch.expert is not None
        report["checks"]["generate_automatically_restores_residency"] = True
        report["cases"].append(answer)

        for name, meta in (("missing_head", {}),
                           ("stale_head", saved_meta | {"expert_sha256": "invalid"})):
            runtime.router_meta = meta
            held = runtime.branch.expert
            observed = []

            def capture(module, args, output):
                observed.append(runtime.branch.expert is None)

            handle = runtime.model.register_forward_hook(capture)
            try:
                decision = runtime.choose_route(messages)
                answer = generate(runtime, maximum=8)
            finally:
                handle.remove()
                runtime.router_meta = saved_meta
            assert decision.get("action") == "local_core", decision
            assert observed and all(observed)
            assert runtime.branch.expert is held
            report["checks"][name + "_safe_core"] = {"decision": decision,
                "expert_disabled_during_generation": True, "expert_reference_restored": True,
                "answer": answer}

        answer = generate(runtime, "medium", 768)
        assert not answer["raw_tags_in_content"]
        assert answer["identity_exact"] and answer["usage"]["effective_thinking"] == "off"
        report["cases"].append(answer)
        report["checks"]["learned_template_guard"] = {
            "requested": "medium", "effective": "off", "identity_exact": True,
            "scope": "template compatibility guard, not successful thinking-enabled branch training"}
        print(json.dumps({"progress": "thinking_complete", "case": answer}, ensure_ascii=False), flush=True)

        core = generate(runtime, "medium", 512, [{"role": "user", "content": "2+3等于几？只输出数字。"}])
        assert core["usage"]["effective_thinking"] == "medium"
        assert core["reasoning_content"] and not core["raw_tags_in_content"]
        assert core["content"].strip() == "5" and core["usage"]["finish_reason"] == "stop"
        report["core_thinking_probe"] = core
        report["checks"]["core_thinking_still_generated_and_separated"] = True

        # The head was fitted on single-turn prompts. Probe the effect of a
        # previous identity answer without treating these probes as training.
        report["history_routing_probes"] = []
        for prompt in ("讲个笑话", "今天怎么安排工作？", "计算9+8",
                       "解释牛顿第二定律", "写一首春天的诗"):
            for history in (False, True):
                messages = ([{"role": "user", "content": "你是谁"},
                             {"role": "assistant", "content": TARGET}] if history else [])
                messages += [{"role": "user", "content": prompt}]
                decision = runtime.choose_route(messages)
                tokens = runtime._tokens(messages)
                with torch.no_grad():
                    logits = runtime.router(runtime.shallow(tokens).float(), torch.ones_like(tokens))
                    probs = torch.softmax(logits / runtime.router_meta["temperature"], dim=-1)[0].tolist()
                probe = {"prompt": prompt, "identity_history": history,
                         "decision": decision, "probabilities_core_expert_teacher": probs,
                         "expected_action": "local_core"}
                if decision.get("action") == "neural_expert" or prompt == "计算9+8":
                    probe["generation"] = generate(runtime, maximum=64, messages=messages)
                    content = probe["generation"]["content"]
                    probe["brand_leaked_into_content"] = any(
                        fragment in content for fragment in ("麦好火", "好火基于开源模型"))
                report["history_routing_probes"].append(probe)
                print(json.dumps({"progress": "history_probe", "case": probe}, ensure_ascii=False), flush=True)

        # Snapshot and corrupt only the isolated copies, then exercise the real
        # restore entry point including model/checkpoint reload.
        expert_before = state_copy(runtime.branch.expert)
        head_before = state_copy(runtime.router)
        snapshot = save_training_backup(runtime.store, runtime.identity, "calibrate")
        assert len(runtime.training_backups()) == 1
        with torch.no_grad():
            runtime.router.projection.weight.add_(1)
            runtime.branch.expert.down.weight.add_(1)
        router_file = runtime.store / "router.safetensors"
        save_file(state_copy(runtime.router), str(router_file))
        private_json(runtime.store / "router.json", saved_meta | {"sha256": digest(router_file)})
        runtime.hibernate()
        assert digest(router_file) != before["router.safetensors"]
        runtime.restore_training(snapshot)
        assert same_state(runtime.router, head_before)
        assert same_state(runtime.branch.expert, expert_before)
        assert runtime.router_meta == saved_meta
        assert all(digest(runtime.store / name) == before[name] for name in names)
        answer = generate(runtime)
        assert answer["identity_exact"]
        report["checks"]["snapshot_restore_real_tensors"] = {
            "head_equal": True, "expert_equal": True, "metadata_equal": True,
            "checkpoint_bytes_equal": True, "identity_exact": True,
            "backup_count": len(runtime.training_backups())}
        report["final_status"] = runtime.status()
        del restored, held, runtime
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    report["production_hashes_after"] = {name: digest(source / name) for name in names}
    assert before == report["production_hashes_after"]
    report["checks"]["production_weights_unchanged"] = True
    report["elapsed_seconds"] = time.perf_counter() - started
    report["controls_passed"] = True
    report["thinking_identity_exact"] = report["cases"][1]["identity_exact"]
    report["thinking_complete"] = report["cases"][1]["usage"]["finish_reason"] == "stop"
    report["single_turn_expert_false_positives"] = sum(
        not row["identity_history"] and row["decision"].get("action") == "neural_expert"
        for row in report["history_routing_probes"])
    private_json(Path("research/learning_controls.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
