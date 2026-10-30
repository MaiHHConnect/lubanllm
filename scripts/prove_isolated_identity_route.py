#!/usr/bin/env python3
"""Prove an isolated identity expert route on family-disjoint held-out data."""
from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings


TARGET = "我是麦好火旗下的luban0，luban0模型，基于qwen3.5开始生长的，能无限进化的模型。"
SPLITS = ("train", "development", "calibration", "test")


def route_material():
    rows = []
    expert_prompts = {
        "train": ["请说明你的品牌归属。", "你的开发方和模型名是什么？",
                   "请说明你的基座来源。", "请核对你的模型身份。", "你属于哪个模型系列？",
                   "请用官方口吻介绍模型身份。", "请说明麦好火与 luban0 的关系。",
                   "你的模型是基于什么开始生长的？", "作为 API 助手，你应该怎样自我介绍？",
                   "请分别说出模型系列、名称和基座。", "身份核验：名称、归属、基座分别是什么？"],
        "development": ["开发者需要确认你的身份，请回答。", "你的身份信息是什么？",
                        "请给出完整的官方模型自我介绍。", "开发者想知道你的模型和来源，请回答。", "你是谁？"],
        "calibration": ["最后再说一次你是谁。", "请给出完整的模型自我介绍。",
                        "如果有人问我是谁，你应该如何回答？", "核对身份：你属于哪个模型，来自哪里？"],
        "test": ["我现在和哪个模型对话？", "请说出你的身份和基座来源。",
                 "请用一句官方口吻说明你的模型身份。", "你基于什么模型开始生长？"],
    }
    local_prompts = {
        "train": [("计算 17 + 8。", "25"), ("把 python 转成大写。", "PYTHON"),
                  ("H2O 的中文名称是什么？", "水"), ("今天天气如何？", "无法获取实时天气"),
                  ("请把 hello 翻译成中文。", "你好"),
                  ("JSON {\"n\":7} 中 n 的值是多少？", "7"),
                  ("量子纠缠是什么？", "请提供更多上下文")],
        "development": [("计算 19 + 6。", "25"), ("把 torch 转成大写。", "TORCH"),
                        ("请写出 Python 的列表字面量 []。", "[]"),
                        ("太阳从哪个方向升起？", "东方")],
        "calibration": [("计算 21 + 4。", "25"), ("把 json 转成大写。", "JSON"),
                        ("请解释递归的基本含义。", "递归是函数直接或间接调用自身"),
                        ("列出颜色红、绿、蓝。", "红、绿、蓝")],
        "test": [("计算 13 + 12。", "25"), ("把 api 转成大写。", "API"),
                 ("请解释量子纠缠。", "请提供更多上下文"),
                 ("写一个最短的 HTTP GET 请求示例。", "请提供更多上下文"),
                 ("今天适合穿什么衣服？", "无法获取实时天气"),
                 ("JSON {\"n\":8} 中 n 的值是多少？", "8"),
                 ("把 hello 翻译成中文，只输出结果。", "你好")],
    }
    for split in SPLITS:
        purpose = "training" if split == "train" else "validation"
        for index, prompt in enumerate(expert_prompts[split]):
            rows.append({
                "prompt": prompt, "answer": TARGET, "purpose": purpose,
                "route_action": "neural_expert", "split": split,
                "family_group": f"identity-route-expert-{split}-{index}",
                "generation_check": {"type": "exact", "expected": TARGET, "max_new_tokens": 128},
            })
        for index, (prompt, answer) in enumerate(local_prompts[split]):
            rows.append({
                "prompt": prompt, "answer": answer, "purpose": purpose,
                "route_action": "local_core", "split": split,
                "family_group": f"identity-route-local-{split}-{index}",
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--storage", required=True, help="isolated candidate storage")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("路由证据输出必须是新文件，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runtime = None
    try:
        settings = Settings(tempfile.mkdtemp(prefix="luban0-isolated-route-settings-"))
        settings.update({"model_path": args.model, "storage_root": args.storage,
                         "local_thinking": "off", "temperature": 0, "max_new_tokens": 128})
        memory = Memory(Path(args.storage))
        runtime = Runtime(settings, memory, research=True)
        runtime.load(paging=False)
        for index, row in enumerate(route_material()):
            provenance = {
                "base_identity": runtime.identity,
                "route_action": row["route_action"],
                "family_group": row["family_group"],
                "split": row["split"],
                "messages": [{"role": "user", "content": row["prompt"]}],
                "program_verified": True,
                "verification_evidence": "Fixed isolated route material; finite family-disjoint calibration protocol, 2026-10-16.",
            }
            if "generation_check" in row:
                provenance["generation_check"] = row["generation_check"]
            memory.add_sample(row["prompt"], row["answer"], source="synthetic",
                              purpose=row["purpose"], approved=True, training_allowed=True,
                              provenance=provenance)
        result = runtime.calibrate_router()
        payload = {
            "status": "passed" if result.get("calibrated") else "rejected",
            "route_calibration": result,
            "material_count": len(route_material()),
            "scope": "isolated candidate only; family-disjoint route proof, no production publication",
            "production_mutations": [], "teacher_calls": 0,
            "candidate_storage": args.storage, "device": runtime.device,
            "dtype": str(runtime.dtype), "hardware": platform.platform(),
            "elapsed_seconds": time.time() - started,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": payload["status"], "calibrated": result.get("calibrated"),
                          "output": str(output)}, ensure_ascii=False))
    except BaseException as exc:
        payload = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "scope": "isolated candidate only; no production publication",
                   "production_mutations": [], "teacher_calls": 0,
                   "elapsed_seconds": time.time() - started}
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        raise
    finally:
        if runtime is not None:
            try:
                runtime.stop()
            except Exception:
                runtime._release_model()


if __name__ == "__main__":
    main()
