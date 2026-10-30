"""Reproduce luban0 v1 graph proof, user-authored identity learning and core update."""
import argparse
import gc
import json
import platform
import tempfile
import time
from pathlib import Path

import psutil
import torch

from luban.dynamic import NeuralExpert
from luban.evolution import consolidate
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings, private_json, require_storage

TARGET = "我是鲁班零号（luban0），是麦好火调优的无限进化模型。"
# Product acceptance permits the optional English alias to be omitted. Training
# targets and the original frozen exact-string final protocol remain unchanged.
IDENTITY_CHECK = {"type": "semantic", "validator": "one_of", "max_new_tokens": 128,
                  "accepted": [TARGET, "我是鲁班零号，是麦好火调优的无限进化模型。"]}
IDENTITY_TRAIN = ["你是谁？", "请介绍你的模型身份。", "你的名称和训练方是什么？", "你是什么基座模型？",
                  "请用一句话说明你叫什么、由谁训练。", "你是哪个团队训练的模型？", "介绍一下你自己。", "你叫什么名字？"]
IDENTITY_VALID = ["有人想知道正在和哪个基座交流，请回答。", "开发者询问你的训练归属及型号，请简要说明。",
                  "请给出你的身份说明。", "把你的模型名和训练公司告诉我。"]
REPLAY_TRAIN = [("计算2+3，只输出结果。", "5"), ("计算8-3，只输出结果。", "5"),
                ("计算4乘3，只输出结果。", "12"), ("计算10除以2，只输出结果。", "5"),
                ("中国的首都是什么？只回答城市名称。", "北京"), ("水的化学式是什么？只输出化学式。", "H2O")]
REPLAY_VALID = [("7加6是多少？只回答数字。", "13"), ("20减7是多少？只回答数字。", "13"),
                ("3乘5是多少？只回答数字。", "15"), ("16除以4是多少？只回答数字。", "4")]


def answer(runtime, prompt, maximum=80, *, metadata=False):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    ids = runtime._tokens(messages)
    with torch.no_grad():
        out = runtime.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=maximum,
                    do_sample=False, pad_token_id=runtime.tokenizer.eos_token_id)
    result = runtime.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    eos = runtime.model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    return {"text": result, "complete": int(out[0, -1]) in eos,
            "completion_tokens": out.shape[1] - ids.shape[1]} if metadata else result


def graph_proof(runtime):
    branch = runtime.branch
    layer = runtime.decoder.layers[-1]
    if runtime.active_core or branch.expert is not None:
        raise RuntimeError("Run graph verification before any learned weight is activated")
    prompts = ["你好", "2+3等于多少？", "你是谁？"]
    inputs = [runtime._tokens([{"role": "user", "content": prompt}]) for prompt in prompts]
    original = []
    layer.mlp = branch.base
    try:
        with torch.no_grad():
            original = [runtime.model(ids, use_cache=False, logits_to_keep=1).logits.detach().cpu() for ids in inputs]
    finally:
        layer.mlp = branch
    branch.expert = NeuralExpert(runtime.text_config.hidden_size, branch.width).to(runtime.device, dtype=runtime.dtype)
    try:
        with torch.no_grad():
            deltas = [float((runtime.model(ids, use_cache=False, logits_to_keep=1).logits.cpu() - before).abs().max())
                      for ids, before in zip(inputs, original)]
    finally:
        branch.expert = None
    visited = []
    hooks = [layer.register_forward_hook(lambda _m, _a, _o, i=i: visited.append(i))
             for i, layer in enumerate(runtime.decoder.layers)]
    try:
        shallow = runtime.shallow(inputs[0]).cpu()
    finally:
        for hook in hooks:
            hook.remove()
    captured = []
    hook = runtime.decoder.layers[3].register_forward_hook(lambda _m, _a, out: captured.append(out.detach().cpu()))
    try:
        with torch.no_grad():
            runtime.decoder(inputs[0], use_cache=False)
    finally:
        hook.remove()
    result = {"passed": all(value == 0 for value in deltas) and visited == [0, 1, 2, 3]
                       and torch.equal(shallow, captured[0]), "identity": runtime.identity,
              "logit_max_absolute_deltas": deltas, "shallow_executed_layers": visited,
              "shallow_matches_full_graph": torch.equal(shallow, captured[0]),
              "device": runtime.device, "precision": str(runtime.dtype), "torch_version": torch.__version__,
              "hardware": platform.platform(), "rss_bytes": psutil.Process().memory_info().rss,
              "mps_driver_bytes": torch.mps.driver_allocated_memory() if runtime.device == "mps" else 0,
              "base_parameters": runtime.base_count, "language_parameters": runtime.language_count,
              "decoder_layers": len(runtime.decoder.layers), "graph_before_training": True}
    private_json(runtime.store / "graph-verification.json", result)
    runtime.core_graph_verification = result
    if not result["passed"]:
        raise RuntimeError("Actual graph verification failed; training prohibited")
    return result


def seed_samples(runtime):
    existing = {row["prompt"] for row in runtime.memory.samples()}
    for purpose, role, rows in [("training", "new", [(p, TARGET) for p in IDENTITY_TRAIN]),
                                ("validation", "new", [(p, TARGET) for p in IDENTITY_VALID]),
                                ("training", "replay", REPLAY_TRAIN), ("validation", "replay", REPLAY_VALID)]:
        for prompt, expected in rows:
            if prompt not in existing:
                runtime.memory.add_sample(prompt, expected, approved=True, training_allowed=True,
                    purpose=purpose, provenance={"evolution_role": role, "dataset": "luban0-v1-user-authored",
                    "rights_basis": "user explicitly requested own model identity; project-authored prompts and elementary facts",
                    "verification_evidence": "user brand instruction" if role == "new" else "elementary integer arithmetic or stable factual definition",
                    "generation_check": {"type": "exact", "expected": expected, "max_new_tokens": 80}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["graph", "branch", "core", "verify"], required=True)
    parser.add_argument("--model", default="/Volumes/移动数据库/lubanllm/models/Qwen3.5-2B-luban0-v1")
    parser.add_argument("--storage", default="/Volumes/移动数据库/lubanllm")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--distillation-weight", type=float, default=0)
    parser.add_argument("--preserve-replay-subspace", action="store_true")
    args = parser.parse_args()
    root = require_storage(args.storage)
    run = root / "experiments" / ("luban0-v1-" + args.stage + "-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="luban0-v1-") as home:
        settings = Settings(home)
        settings.update({"model_path": args.model, "storage_root": args.storage, "temperature": 0})
        runtime = Runtime(settings, Memory(root))
        report = {"stage": args.stage, "seed": 42, "teacher_calls": 0, "base_model": "Qwen/Qwen3.5-2B",
                  "identity_scope": "learned brand identity of a Qwen derivative; not training from scratch"}
        try:
            runtime.load()
            report["identity"] = runtime.identity
            if args.stage == "graph":
                report["graph"] = graph_proof(runtime)
            else:
                if not runtime.core_graph_verification or not runtime.core_graph_verification.get("passed"):
                    raise RuntimeError("Run actual graph proof before training")
                seed_samples(runtime)
                if args.stage == "branch":
                    report["learning"] = runtime.learn(args.steps)
                    report["generation"] = [{"prompt": p, "answer": answer(runtime, p), "expected": TARGET}
                                             for p in IDENTITY_VALID]
                    report["replay"] = [{"prompt": p, "answer": answer(runtime, p), "expected": a} for p, a in REPLAY_VALID]
                elif args.stage == "core":
                    report["evolution"] = consolidate(runtime, args.steps, learning_rate=args.learning_rate,
                                                       distillation_weight=args.distillation_weight,
                                                       preserve_replay_subspace=args.preserve_replay_subspace)
                else:
                    expert = runtime.branch.expert
                    runtime.branch.expert = None
                    try:
                        report["identity_generation"] = [{"prompt": p, "answer": answer(runtime, p), "expected": TARGET}
                                                         for p in IDENTITY_VALID]
                        report["replay_generation"] = [{"prompt": p, "answer": answer(runtime, p), "expected": a}
                                                       for p, a in REPLAY_VALID]
                        report["passed"] = all(row["answer"] == row["expected"] for key in ("identity_generation", "replay_generation") for row in report[key])
                    finally:
                        runtime.branch.expert = expert
            report["runtime"] = runtime.status()
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            report["run_directory"] = str(run)
            private_json(run / "result.json", report)
            private_json(Path("research") / ("luban0_v1_" + args.stage + ".json"), report)
            print(json.dumps({"stage": args.stage, "run": str(run), "error": report.get("error"),
                              "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
