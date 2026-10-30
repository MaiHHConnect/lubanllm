#!/usr/bin/env python3
"""Run a bounded read-only production probe for the gated identity expert."""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
from pathlib import Path

from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings


TARGET = "我是麦好火旗下的luban0，luban0模型，基于qwen3.5开始生长的，能无限进化的模型。"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("产品探针输出必须是新文件，拒绝覆盖")
    settings = Settings(tempfile.mkdtemp(prefix="luban0-product-probe-settings-"))
    settings.update({"model_path": args.model, "storage_root": args.storage,
                     "local_thinking": "off", "temperature": 0, "max_new_tokens": 128})
    runtime = Runtime(settings, Memory(Path(args.storage)), research=True)
    runtime.load(paging=False)
    prompts = [
        ("identity-real-entry", "你是谁？", "neural_expert", True),
        ("identity-brand", "请说明你的品牌归属。", "neural_expert", True),
        ("identity-source", "请说出你的身份和基座来源。", "neural_expert", True),
        ("ordinary-integer", "计算17加8，只输出整数。", "local_core", False),
        ("ordinary-uppercase", "把字符串“PyTorch”转为大写，只输出结果。", "local_core", False),
        ("ordinary-json", "JSON {\"n\":7} 中 n 的值是多少？", "local_core", False),
        ("ordinary-translation", "请把 hello 翻译成中文，只输出译文。", "local_core", False),
        ("ordinary-fact", "H2O 的中文名称是什么？只输出名称。", "local_core", False),
        ("unknown-weather", "今天天气如何？", "local_core", False),
        ("unknown-quantum", "请解释量子纠缠。", "local_core", False),
        ("unknown-web", "帮我写一个网页。", "local_core", False),
        ("unknown-http", "写一个最短的 HTTP GET 请求示例。", "local_core", False),
        ("unknown-clothing", "今天适合穿什么衣服？", "local_core", False),
    ]
    checks = []
    try:
        for identifier, prompt, expected_action, identity in prompts:
            messages = [{"role": "user", "content": prompt}]
            decision = runtime.choose_route(messages)
            pieces = []
            usage = runtime.generate(messages, lambda delta: pieces.append(delta["content"]),
                                     threading.Event(), max_tokens=128, temperature=0,
                                     allow_reference=True, observe_external=False)
            answer = "".join(pieces)
            action = usage.get("local_action")
            complete = bool(usage.get("local_complete"))
            passed = action == expected_action and decision.get("action") == expected_action
            if identity:
                passed = passed and complete and answer.strip() == TARGET
            checks.append({"id": identifier, "prompt": prompt, "decision": decision,
                           "action": action, "complete": complete, "answer": answer,
                           "passed": passed})
        payload = {
            "status": "passed_bounded_product_probe_with_expanded_scope"
            if all(row["passed"] for row in checks) else "rejected",
            "production_storage": args.storage,
            "router_id": runtime.router_meta.get("id"),
            "router_threshold": runtime.router_meta.get("threshold"),
            "expert_scope_threshold": runtime.router_meta.get("expert_scope", {}).get("threshold"),
            "checks": checks,
            "teacher_calls": 0,
            "production_weights_changed": False,
            "scope": "有限真实产品闭卷探针；扩展未知输入族，不证明未知输入全域零退化",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": payload["status"], "router_id": payload["router_id"],
                          "checks": len(checks)}, ensure_ascii=False))
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
