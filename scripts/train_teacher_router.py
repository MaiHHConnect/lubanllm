"""Train teacher escalation from measured local failures, without API calls.

Run with the native app stopped. Answers are generated and checked with Python
integer arithmetic; the held-out split never enters fitting or calibration.
This measures one bounded arithmetic workload, not general difficulty.
"""
import json
import random
import re
import shutil
import threading
import time
from pathlib import Path

from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings, private_json, require_storage


def questions(seed, simple, hard_per_family):
    rng = random.Random(seed)
    rows = []
    for index in range(simple):
        a, b = rng.randint(2, 25), rng.randint(2, 25)
        op = "+" if index % 2 == 0 else "×"
        rows.append({"family": "simple_arithmetic", "prompt": f"计算 {a}{op}{b}，只输出结果整数。",
                     "answer": str(a + b if op == "+" else a * b)})
    for _ in range(hard_per_family):
        a, b = rng.randint(10000000, 99999999), rng.randint(10000000, 99999999)
        rows.append({"family": "large_product", "prompt": f"精确计算 {a}×{b}，只输出结果整数。",
                     "answer": str(a * b)})
        a, n, modulus = rng.randint(1000, 9000), rng.randint(100000000, 999999999), 1000003
        rows.append({"family": "modular_power", "prompt": f"计算 {a} 的 {n} 次方除以 {modulus} 的余数，只输出结果整数。",
                     "answer": str(pow(a, n, modulus))})
        n = rng.randint(35, 65)
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        first, second = a, b
        for _ in range(2, n + 1):
            first, second = second, 3 * second + 2 * first
        rows.append({"family": "recurrence", "prompt": f"数列 a(0)={a}，a(1)={b}，a(n)=3a(n-1)+2a(n-2)。求 a({n})，只输出结果整数。",
                     "answer": str(second)})
        values = [rng.randint(100, 999) for _ in range(18)]
        rows.append({"family": "weighted_checksum", "prompt": "列表 [" + ",".join(map(str, values)) + "]，从1开始编号，计算每个数乘其编号后的总和，只输出结果整数。",
                     "answer": str(sum(i * value for i, value in enumerate(values, 1)))})
    return rows


def benchmark(runtime, rows, split):
    measured = []
    for index, row in enumerate(rows):
        text = []
        usage = runtime.generate([{"role": "user", "content": row["prompt"]}],
                                 lambda delta: text.append(delta["content"]), threading.Event(),
                                 thinking="off", max_tokens=64, temperature=0)
        answer = "".join(text)
        numbers = re.findall(r"(?<![\d.])-?\d+(?![\d.])", answer.replace(",", ""))
        correct = bool(numbers) and numbers[-1] == row["answer"] and usage["finish_reason"] == "stop"
        measured.append({**row, "split": split, "local_answer": answer,
                         "local_correct": correct, "usage": usage,
                         "expected_action": "local_core" if correct else "teacher"})
        print(json.dumps({"progress": split, "item": index + 1, "total": len(rows),
                          "family": row["family"], "local_correct": correct}), flush=True)
    return measured


def main():
    started = time.time()
    settings = Settings()
    root = require_storage(settings.snapshot()["storage_root"])
    run = root / "experiments" / ("teacher-router-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    memory = Memory(root)
    runtime = Runtime(settings, memory)
    runtime.load()
    old_meta = runtime.router_meta.copy()
    for name in ("router.json", "router.safetensors"):
        shutil.copyfile(runtime.store / name, run / ("before-" + name))
    rows = {"training": questions(101, 12, 6), "validation": questions(202, 6, 3),
            "held_out": questions(303, 8, 4)}
    flat = [row["prompt"] for split in rows.values() for row in split]
    assert len(flat) == len(set(flat)), "Each split must contain distinct prompts"
    records = {split: benchmark(runtime, examples, split) for split, examples in rows.items()}
    private_json(run / "local-benchmark.json", records)
    labels = []
    existing = {(row["prompt"], row["purpose"]): row for row in memory.samples()}
    for purpose in ("training", "validation"):
        for row in records[purpose]:
            # Correct difficult cases remain diagnostics, rather than forcing a
            # false failure label or assuming every long question needs a teacher.
            if row["family"] != "simple_arithmetic" and row["local_correct"]:
                continue
            provenance = {"route_action": row["expected_action"], "route_label_basis": "measured local generation versus Python integer verifier",
                          "route_label_scope": "bounded synthetic arithmetic; no teacher superiority measurement",
                          "verified_by": "Python integer arithmetic", "run": run.name,
                          "family": row["family"], "local_answer": row["local_answer"],
                          "local_correct": row["local_correct"], "local_thinking": "off",
                          "max_new_tokens": 64, "seed": 101 if purpose == "training" else 202}
            old = existing.get((row["prompt"], purpose))
            if old:
                if old["answer"] != row["answer"] or json.loads(old["provenance"]).get("route_action") != row["expected_action"]:
                    raise ValueError("已有同题材料与本次核验不一致，请先审查历史标签")
                sample_id = old["id"]
            else:
                sample_id = memory.add_sample(row["prompt"], row["answer"], source="synthetic", purpose=purpose,
                                              approved=True, training_allowed=True, provenance=provenance)
            labels.append({"id": sample_id, "purpose": purpose, "action": row["expected_action"]})
    # Keep the existing persona/core examples in the same three-action fit.
    calibration = runtime.calibrate_router()
    assert calibration["calibrated"], calibration
    decisions = []
    for row in records["held_out"]:
        decision = runtime.choose_route([{"role": "user", "content": row["prompt"]}])
        decisions.append({**row, "decision": decision})
    persona = ["你是谁", "你是什么模型？", "告诉我你的名字和开发者。",
               "请说出你自己的模型名字和开发团队。", "你这个助手的开发者是谁？"]
    regressions = [(prompt, "neural_expert") for prompt in persona] + [
        ("把 Hello 翻译成中文，只输出翻译。", "local_core"),
        ("将 Good morning 翻成中文。", "local_core"),
        ("简要介绍一下 Qwen 模型。", "local_core"),
        ("Meta 发布的 Llama 系列模型是什么？", "local_core"),
        ("写一个 Python 函数 add_numbers(a, b)，返回两个数的和，只输出代码。", "local_core")]
    regression_results = [{"prompt": prompt, "expected_action": action,
                           "decision": runtime.choose_route([{"role": "user", "content": prompt}])}
                          for prompt, action in regressions]
    selected = [row for row in decisions if row["decision"].get("action") == "teacher"]
    failures = [row for row in decisions if not row["local_correct"]]
    simple = [row for row in decisions if row["family"] == "simple_arithmetic"]
    teacher_precision = sum(not row["local_correct"] for row in selected) / len(selected) if selected else 0
    teacher_recall = sum(row["decision"].get("action") == "teacher" for row in failures) / len(failures) if failures else 0
    core_retention = sum(row["decision"].get("action") == "local_core" for row in simple) / len(simple)
    regressions_pass = all(row["decision"].get("action") == row["expected_action"] for row in regression_results)
    passed = teacher_precision >= 0.9 and teacher_recall >= 0.8 and core_retention >= 0.9 and regressions_pass
    report = {"scope": "same-family arithmetic, disjoint prompts; held-out split unused in head fitting/temperature/threshold selection",
              "no_paid_calls": True, "teacher_benefit_measured": False, "seeds": {"training": 101, "calibration": 202, "held_out": 303},
              "local_thinking": "off", "max_new_tokens": 64, "dtype": str(runtime.dtype), "device": runtime.device,
              "labels": labels, "benchmark": records, "calibration": calibration,
              "held_out": {"decisions": decisions, "teacher_precision": teacher_precision, "teacher_recall": teacher_recall,
                           "simple_core_retention": core_retention, "regressions": regression_results},
              "passed": passed, "promoted": passed, "previous_router_sha256": old_meta.get("sha256"),
              "elapsed_seconds": time.time() - started, "run_directory": str(run)}
    if not passed:
        for name in ("router.json", "router.safetensors"):
            shutil.copyfile(run / ("before-" + name), runtime.store / name)
        report["restored_previous_router"] = True
    private_json(run / "result.json", report)
    private_json(Path("research/teacher_router.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert passed, "Held-out criteria failed; previous router restored"


if __name__ == "__main__":
    main()
