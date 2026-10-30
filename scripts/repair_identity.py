"""Bounded warm identity/template repair and ordinary-task head calibration.

Train and verify one real model in isolated storage before publishing any
weights. Paid teachers, credentials, tokenizer changes and answer edits are
excluded. Failed candidates and their raw generations remain in the report.
"""
import gc
import hashlib
import json
import random
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path

import psutil
import torch
from safetensors.torch import save_file

from luban.backups import save_training_backup
from luban.memory import Memory
from luban.runtime import Runtime, EmitStreamer
from luban.settings import Settings, default_home, private_json, require_storage
from luban.thinking import ThinkingStream
from train_identity import TARGET, QUESTIONS


VALIDATION = ["你的身份是什么？", "介绍一下你的名称和开发团队。",
              "请说出你自己的模型名字和开发团队。", "你这个助手的开发者是谁？"]
OFF_PROBES = ["你是谁", "你是什么模型？", "告诉我你的名字和开发者。",
              "请说出你自己的模型名字和开发团队。", "你这个助手的开发者是谁？"]
THOUGHT_TARGET = "<think>\n\n</think>\n\n" + TARGET
CORE_TRAIN = [
    ("讲个笑话", "为什么电脑怕热？因为它怕死机。"),
    ("今天怎么安排工作？", "先列出任务和截止时间，再处理最重要的任务，并安排休息。"),
    ("帮我规划今天的日程。", "上午做重要工作，中午休息，下午处理沟通事项，晚上总结。"),
    ("写一首春天的诗", "春风拂过小河边，嫩柳轻摇映碧天。"),
    ("写一段秋天的散文。", "秋风穿过树梢，落叶把小路染成金色。"),
    ("解释牛顿第二定律", "物体的合外力等于质量与加速度的乘积，F=ma。"),
    ("什么是光合作用？", "植物利用光能，把水和二氧化碳转化成有机物并释放氧气。"),
    ("给初学者讲讲黑洞。", "黑洞是引力极强、连光也无法从事件视界内逃出的天体。"),
    ("请推荐保持专注的方法。", "关闭无关通知，明确一个小目标，并定时休息。"),
    ("如何煮米饭？", "淘米后加入适量水，按电饭锅说明煮熟。"),
    ("帮我写一封请假邮件。", "您好，我因个人事务申请请假一天，请批准，谢谢。"),
    ("说一个睡前故事。", "小兔子跟着月光回到家，在柔软的被窝里睡着了。"),
]
CORE_CALIBRATION = [
    ("给我讲一个简短的笑话。", "为什么书不迷路？因为它有目录。"),
    ("今天该怎样规划工作时间？", "按优先级安排任务，为突发事项预留时间。"),
    ("创作一首有关秋天的诗。", "一叶随风落，满城秋色深。"),
    ("简述植物怎样进行光合作用。", "植物借助光能合成有机物并释放氧气。"),
    ("给孩子讲一个睡前故事。", "小熊找到了回家的星星，和妈妈一起进入梦乡。"),
    ("给出三条学习建议。", "制定小目标，主动练习，定期复习。"),
]
CORE_HELD_OUT = ["来个好笑的小故事。", "帮我安排明天上午的工作。", "以春日为主题写四句诗。",
                 "为什么天空是蓝色的？", "写一封预约会议的邮件。", "请说明力和加速度的关系。",
                 "推荐几种缓解疲劳的办法。", "计算9+8"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def tokens(runtime, prompt, mode):
    prefix = runtime.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=mode != "off")
    prefix_ids = runtime.tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = runtime.tokenizer.encode(TARGET if mode == "off" else THOUGHT_TARGET,
                                         add_special_tokens=False) + [runtime.tokenizer.eos_token_id]
    ids = torch.tensor([prefix_ids + answer_ids], device=runtime.device)
    labels = ids.clone()
    labels[:, :len(prefix_ids)] = -100
    return ids, labels


def losses(runtime, prompts):
    result = {}
    with torch.no_grad():
        for mode in ("off", "medium"):
            values = []
            for prompt in prompts:
                ids, labels = tokens(runtime, prompt, mode)
                values.append(float(runtime.model(ids, labels=labels, use_cache=False).loss.cpu()))
            result[mode] = sum(values) / len(values)
    result["mean"] = (result["off"] + result["medium"]) / 2
    return result


def generate(runtime, prompt, mode="off", maximum=512, history=False):
    raw, deltas = [], []
    parser = ThinkingStream(deltas.append)
    messages = ([{"role": "user", "content": "你是谁"},
                 {"role": "assistant", "content": TARGET}] if history else [])
    messages += [{"role": "user", "content": prompt}]
    decision = runtime.choose_route(messages)

    def emit(delta):
        raw.append(delta["content"])
        parser.feed(delta["content"])

    # Research deliberately evaluates the unvalidated candidate template.
    # The production Runtime now protects users by using only its off template;
    # that fallback must never be mistaken for successful on-template learning.
    held = runtime.branch.expert
    started = time.perf_counter()
    try:
        if decision.get("action") != "neural_expert":
            runtime.branch.expert = None
        ids = runtime._tokens(messages, mode)
        with torch.no_grad():
            out = runtime.model.generate(ids, max_new_tokens=maximum,
                attention_mask=torch.ones_like(ids), do_sample=False,
                pad_token_id=runtime.tokenizer.eos_token_id,
                streamer=EmitStreamer(runtime.tokenizer, emit))
        count = out.shape[1] - ids.shape[1]
        usage = {"prompt_tokens": ids.shape[1], "completion_tokens": count,
                 "total_tokens": out.shape[1], "elapsed_seconds": time.perf_counter() - started,
                 "requested_thinking": mode, "effective_thinking": mode,
                 "finish_reason": "stop" if count < maximum else "length"}
    finally:
        runtime.branch.expert = held
    parser.finish()
    content = "".join(row.get("content", "") for row in deltas)
    result = {"prompt": prompt, "thinking": mode, "identity_history": history,
            "decision": decision, "raw": "".join(raw), "content": content,
            "reasoning_content": "".join(row.get("reasoning_content", "") for row in deltas),
            "usage": usage, "identity_exact": content.strip() == TARGET,
            "target_occurrences": content.count(TARGET),
            "capability_sentence_occurrences": content.count("无限进化是我的能力。"),
            "brand_fragment_present": any(part in content for part in
                                          ("麦好火", "好火基于开源模型", "鲁班零号", "无限进化是我的能力"))}
    print(json.dumps({"progress": "generation", "prompt": prompt, "thinking": mode,
                      "content": content, "completion_tokens": usage["completion_tokens"],
                      "finish_reason": usage["finish_reason"]}, ensure_ascii=False), flush=True)
    return result


def stage_expert(runtime, expert_id):
    runtime.branch.expert_id = expert_id
    runtime.branch.requires_gate = True
    runtime.branch.hibernate()
    runtime.branch.activate(expert_id)
    private_json(runtime.pointer, {"id": expert_id, "width": runtime.branch.width,
                                  "identity": runtime.identity})
    _, meta = runtime.branch._paths(expert_id)
    # Existing head is reused only for the isolated identity-generation checks;
    # the complete repair later retrains it and binds its own new checkpoint.
    runtime.router_meta = runtime.router_meta | {"expert_id": expert_id,
                                               "expert_sha256": json.loads(meta.read_text())["sha256"]}
    private_json(runtime.store / "router.json", runtime.router_meta)


def copy_labelled_memory(production, destination, metadata):
    selected = metadata["training_ids"] + metadata["validation_ids"]
    db = sqlite3.connect(f"file:{production / 'app/memory.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in db.execute(
            "SELECT * FROM samples WHERE id IN (" + ",".join("?" for _ in selected) + ")", selected)]
    finally:
        db.close()
    assert len(rows) == len(selected), "Historical calibrated samples must all remain available"
    memory = Memory(destination)
    with memory.lock, memory.connect() as db:
        for row in rows:
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))
    return memory


def main():
    start = time.perf_counter()
    config = json.loads((default_home() / "settings.json").read_text())
    local = {key: config[key] for key in ("model_path", "storage_root", "local_thinking",
                                        "max_new_tokens", "context_tokens", "temperature")}
    production = require_storage(local["storage_root"])
    source = production / "modules/app"
    active = json.loads((source / "active.json").read_text())
    old_head = json.loads((source / "router.json").read_text())
    expert_meta = next(source.glob(f"layer-*/{active['id']}.json"))
    names = ["active.json", "router.json", "router.safetensors",
             expert_meta.relative_to(source).as_posix(),
             expert_meta.with_suffix(".safetensors").relative_to(source).as_posix()]
    original_hashes = {name: digest(source / name) for name in names}
    previous = Path("research/learning_controls-before-repair.json")
    if not previous.exists():
        shutil.copyfile("research/learning_controls.json", previous)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run = production / "experiments" / ("identity-repair-" + run_id)
    run.mkdir(parents=True, exist_ok=False)
    # This is the actual original production state, before any training.
    initial_backup = save_training_backup(source, active["identity"], "learn")
    report = {"run_id": run_id, "run_directory": str(run), "promoted": False,
              "target": TARGET, "thinking_target": THOUGHT_TARGET, "seed": 42,
              "scope": "bounded synthetic persona/template and routing repair; not general intelligence",
              "no_teacher_calls": True, "initial_production_backup": initial_backup,
              "production_before": original_hashes, "previous_expert": active["id"],
              "expert_id": active["id"], "expert_sha256": old_head["expert_sha256"], "checks": {},
              "training_prompts": QUESTIONS["training"], "validation_prompts": VALIDATION,
              "core_training": CORE_TRAIN, "core_calibration": CORE_CALIBRATION,
              "core_held_out": CORE_HELD_OUT, "candidates": [],
              "limitation": "Repeated validation/repair probes are development checks, not an independent final evaluation.",
              "memory_measurement": "100ms samples of psutil RSS and Metal driver allocated bytes; sampled process maxima, not physical peak"}
    sampled = {"rss_bytes": 0, "metal_bytes": 0, "samples": 0}
    stop_sample = threading.Event()

    def sample_memory():
        while not stop_sample.is_set():
            sampled["rss_bytes"] = max(sampled["rss_bytes"], psutil.Process().memory_info().rss)
            if torch.backends.mps.is_available():
                sampled["metal_bytes"] = max(sampled["metal_bytes"], torch.mps.driver_allocated_memory())
            sampled["samples"] += 1
            stop_sample.wait(0.1)

    sampling = threading.Thread(target=sample_memory, daemon=True)
    sampling.start()

    def record():
        report["elapsed_seconds"] = time.perf_counter() - start
        report["sampled_memory"] = sampled.copy()
        private_json(run / "result.json", report)
        private_json(Path("research/identity_repair.json"), report)

    try:
        copied = run / "modules/app"
        for name in names:
            target = copied / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        with tempfile.TemporaryDirectory(prefix="identity-repair-home-") as home:
            settings = Settings(home)
            settings.data.update(local | {"storage_root": str(run)})
            runtime = Runtime(settings, None)
            torch.manual_seed(42)
            runtime.load()
            runtime.memory = copy_labelled_memory(production, run, old_head)
            report.update(dtype=str(runtime.dtype), device=runtime.device,
                          base_identity=runtime.identity, base_parameters=runtime.base_count,
                          expert_parameters=runtime.branch.resident_expert_params,
                          initial_status=runtime.status())
            base_weights = cpu_state(runtime.branch.expert)
            base_head = cpu_state(runtime.router)
            baseline = losses(runtime, VALIDATION)
            report["baseline_validation_loss"] = baseline
            record()
            for candidate_index, lr in enumerate((0.0002,)):
                candidate_start = time.perf_counter()
                runtime.branch.expert.load_state_dict(base_weights)
                runtime.branch.expert_id = active["id"]
                runtime.branch.requires_gate = True
                runtime.router.load_state_dict(base_head)
                runtime.router_meta = old_head.copy()
                for name in names:
                    shutil.copyfile(source / name, runtime.store / name)
                expert = runtime.branch.expert
                runtime.model.requires_grad_(False)
                expert.requires_grad_(True)
                trainable = sum(p.numel() for p in runtime.model.parameters() if p.requires_grad)
                assert trainable == 393216
                optimizer = torch.optim.Adam(expert.parameters(), lr=lr, eps=1e-6)
                rows = [(prompt, mode) for prompt in QUESTIONS["training"] for mode in ("off", "medium")]
                rng = random.Random(42)
                candidate = {"learning_rate": lr, "steps": 64, "trainable_parameters": trainable,
                             "initialization": "warm original learned-c25 branch", "losses": [], "accepted": False}
                report["candidates"].append(candidate)
                for step in range(64):
                    if step % len(rows) == 0:
                        rng.shuffle(rows)
                    prompt, mode = rows[step % len(rows)]
                    ids, labels = tokens(runtime, prompt, mode)
                    optimizer.zero_grad(set_to_none=True)
                    loss = runtime.model(ids, labels=labels, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite repair loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
                    optimizer.step()
                    candidate["losses"].append({"step": step + 1, "thinking": mode, "loss": float(loss.detach().cpu())})
                    if (step + 1) % 16 == 0:
                        print(json.dumps({"progress": "identity_train", "candidate": candidate_index,
                                          "step": step + 1, "loss": candidate["losses"][-1]}), flush=True)
                del optimizer, loss
                expert.requires_grad_(False)
                score = losses(runtime, VALIDATION)
                candidate["validation_loss"] = score
                expert_id = "learned-repair-" + uuid.uuid4().hex[:12]
                stage_expert(runtime, expert_id)
                candidate["expert_id"] = expert_id
                candidate["generations"] = [generate(runtime, prompt, maximum=128) for prompt in OFF_PROBES]
                candidate["generations"] += [generate(runtime, prompt, "medium") for prompt in VALIDATION]
                candidate["accepted"] = score["mean"] < baseline["mean"] * 0.995 and all(
                    row["identity_exact"] and row["usage"]["finish_reason"] == "stop" and
                    row["target_occurrences"] == 1 and row["capability_sentence_occurrences"] == 1
                    for row in candidate["generations"])
                candidate["elapsed_seconds"] = time.perf_counter() - candidate_start
                record()
                print(json.dumps({"progress": "identity_candidate", "result": candidate}, ensure_ascii=False), flush=True)
                if candidate["accepted"]:
                    break
            else:
                report["failure"] = "The bounded short-thought identity candidate failed genuine generation/loss criteria"
                record()
                return

            appended_ids = []
            for purpose, rows in (("training", CORE_TRAIN), ("validation", CORE_CALIBRATION)):
                for prompt, answer in rows:
                    appended_ids.append(runtime.memory.add_sample(prompt, answer, source="synthetic", purpose=purpose,
                        approved=True, training_allowed=True, provenance={"route_action": "local_core",
                        "route_label_basis": "ordinary task; identity expert unnecessary", "run": run_id,
                        "scope": "routing label only; answer not used to train the expert"}))
            assert len(runtime.memory.samples("training")) == 63
            assert len(runtime.memory.samples("validation")) == 31
            calibration = runtime.calibrate_router()
            report["calibration"] = calibration
            report["appended_core_sample_ids"] = appended_ids
            record()
            if not calibration["calibrated"]:
                report["failure"] = "Existing head calibration criteria rejected the repair"
                record()
                return

            report["ordinary_generations"] = [generate(runtime, prompt, maximum=96) for prompt in CORE_HELD_OUT]
            report["ordinary_regressions"] = [generate(runtime, prompt, maximum=64) for prompt in
                ("讲个笑话", "今天怎么安排工作？", "把 Hello 翻译成中文，只输出翻译。", "简要介绍一下 Qwen 模型。")]
            report["history_generations"] = [generate(runtime, prompt, maximum=64, history=True) for prompt in
                                             ("讲个笑话", "今天怎么安排工作？", "计算9+8")]
            report["identity_routes"] = [{"prompt": prompt,
                "decision": runtime.choose_route([{"role": "user", "content": prompt}])} for prompt in OFF_PROBES]
            teacher = json.loads(Path("research/teacher_router.json").read_text())
            report["old_teacher_held_out"] = [{"prompt": row["prompt"], "expected_action": row["expected_action"],
                "decision": runtime.choose_route([{"role": "user", "content": row["prompt"]}])}
                for row in teacher["held_out"]["decisions"]]
            # The final newly calibrated head must pass real identity generations,
            # not merely the old head reused during candidate selection.
            report["final_identity_generations"] = [generate(runtime, prompt, maximum=128) for prompt in OFF_PROBES]
            report["final_identity_generations"] += [generate(runtime, prompt, "medium") for prompt in VALIDATION]
            ordinary = report["ordinary_generations"] + report["ordinary_regressions"] + report["history_generations"]
            passed = all(row["decision"].get("action") == "local_core" and not row["brand_fragment_present"] for row in ordinary)
            passed = passed and all(row["decision"].get("action") == row["expected_action"] for row in report["old_teacher_held_out"])
            passed = passed and all(row["identity_exact"] and row["usage"]["finish_reason"] == "stop" and
                row["capability_sentence_occurrences"] == 1 for row in report["final_identity_generations"])
            report["all_acceptance_checks_passed"] = passed
            report["checks"] = {
                "ordinary_route_and_no_brand_fragment": all(row["decision"].get("action") == "local_core" and
                    not row["brand_fragment_present"] for row in ordinary),
                "old_teacher_held_out_routes_preserved": all(row["decision"].get("action") == row["expected_action"]
                    for row in report["old_teacher_held_out"]),
                "off_and_on_identity_exact_complete_once": all(row["identity_exact"] and
                    row["usage"]["finish_reason"] == "stop" and row["capability_sentence_occurrences"] == 1
                    for row in report["final_identity_generations"]),
                "base_frozen_only_393216_trainable": True}
            record()
            if not passed:
                report["failure"] = "Final ordinary/identity/teacher regression refused promotion"
                record()
                return

            assert original_hashes == {name: digest(source / name) for name in names}, "Production changed during isolated repair"
            report["pre_promotion_production_backup"] = save_training_backup(source, runtime.identity, "calibrate")
            expert_files = runtime.branch._paths(runtime.branch.expert_id)
            publication = [path.relative_to(runtime.store).as_posix() for path in expert_files]
            publication += ["router.safetensors", "router.json", "active.json"]
            previous_bytes = {name: (source / name).read_bytes() if (source / name).exists() else None for name in publication}
            production_memory = Memory(production)
            appended = [row for row in runtime.memory.samples() if row["id"] in appended_ids]
            try:
                with production_memory.lock, production_memory.connect() as db:
                    for row in appended:
                        db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))
                for name in publication:
                    target = source / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".repair-partial")
                    shutil.copyfile(runtime.store / name, temporary)
                    temporary.replace(target)
            except Exception:
                for name, content in previous_bytes.items():
                    target = source / name
                    if content is None:
                        target.unlink(missing_ok=True)
                    else:
                        target.write_bytes(content)
                with production_memory.lock, production_memory.connect() as db:
                    db.executemany("DELETE FROM samples WHERE id=?", [(sample_id,) for sample_id in appended_ids])
                raise
            report.update(promoted=True, active_expert=runtime.branch.expert_id,
                          expert_id=runtime.branch.expert_id,
                          expert_sha256=runtime.router_meta["expert_sha256"],
                          router_sha256=runtime.router_meta["sha256"], final_status=runtime.status(),
                          production_after={name: digest(source / name) for name in publication},
                          old_expert_preserved=all(digest(source / name) == original_hashes[name] for name in names if "layer-" in name))
            record()
            del expert, runtime
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except Exception as error:
        report["error"] = str(error)
        record()
        raise
    finally:
        stop_sample.set()
        sampling.join(timeout=1)
        record()
        print(json.dumps({"promoted": report["promoted"], "failure": report.get("failure"),
                          "error": report.get("error"), "report": str(run / "result.json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
