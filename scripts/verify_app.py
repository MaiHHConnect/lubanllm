"""Verify a running native app's public API after restart; no training or teacher calls."""
import ast
import json
import re
import time
from pathlib import Path

import httpx
from luban.settings import default_home, private_json, require_storage
from train_identity import TARGET


def main():
    home = default_home()
    credentials = json.loads((home / "credentials.json").read_text())
    report = {"started": time.time(), "scope": "native restart, 5 identity and 7 basic regression prompts; model descriptions may reach token cap; not a capability benchmark"}
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=180) as client:
        client.post("/api/session", json={"token": credentials["ui_token"]}).raise_for_status()
        headers = {"Authorization": "Bearer " + credentials["access_key"]}
        report["models"] = client.get("/v1/models", headers=headers).json()["data"]
        assert report["models"] == [{"id": "luban0", "object": "model", "owned_by": "maihaohuo"}]
        report["initial_status"] = client.get("/api/status").json()["runtime"]
        assert report["initial_status"]["state"] == "ready"
        identity = json.loads(Path("research/identity_training.json").read_text())
        repair_path = Path("research/identity_repair.json")
        repair = json.loads(repair_path.read_text()) if repair_path.exists() else {}
        expected_expert = repair["expert_id"] if repair.get("promoted") else identity["status"]["active_expert"]
        assert report["initial_status"]["active_expert"] == expected_expert
        replies = []
        for prompt in ("你是谁", "你是什么模型？", "告诉我你的名字和开发者。", "2+3等于几？只输出数字。", "把 Hello 翻译成中文，只输出翻译。", "写一个 Python 函数 add_numbers(a, b)，返回两个数的和，只输出代码。",
                       "请说出你自己的模型名字和开发团队。", "你这个助手的开发者是谁？",
                       "将 Good morning 翻成中文。", "计算7乘8，只输出结果数字。", "简要介绍一下 Qwen 模型。",
                       "Meta 发布的 Llama 系列模型是什么？"):
            response = client.post("/v1/chat/completions", headers=headers,
                json={"model": "luban0", "route": "local", "messages": [{"role": "user", "content": prompt}],
                      "thinking": "off", "temperature": 0, "max_tokens": 128})
            response.raise_for_status()
            value = response.json()
            assert value["model"] == "luban0"
            replies.append({"prompt": prompt, "answer": value["choices"][0]["message"]["content"], "usage": value["usage"]})
        report["replies"] = replies
        checks = {"exact_identity_3_prompts": all(r["answer"].strip() == TARGET for r in replies[:3]),
                  "addition": replies[3]["answer"].strip() == "5",
                  "translation": replies[4]["answer"].strip().rstrip("。！!") in {"你好", "您好"},
                  "new_identity_2_prompts": all(r["answer"].strip() == TARGET for r in replies[6:8]),
                  "new_translation": any(word in replies[8]["answer"] for word in ("早上好", "早安")),
                  "new_multiplication": replies[9]["answer"].strip() == "56",
                  "third_party_model": any(word in replies[10]["answer"] for word in ("Qwen", "通义千问")) and "麦好火" not in replies[10]["answer"],
                  "new_third_party_model": "llama" in replies[11]["answer"].casefold() and "麦好火" not in replies[11]["answer"]}
        code = replies[5]["answer"].strip()
        fenced = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.S)
        tree = ast.parse(fenced.group(1) if fenced else code)
        checks["python_add_function"] = any(isinstance(n, ast.FunctionDef) and n.name == "add_numbers" and
            any(isinstance(v, ast.Return) and isinstance(v.value, ast.BinOp) and isinstance(v.value.op, ast.Add) and
                isinstance(v.value.left, ast.Name) and v.value.left.id == "a" and
                isinstance(v.value.right, ast.Name) and v.value.right.id == "b" for v in ast.walk(n)) for n in tree.body)
        report["checks"] = checks
        report["passed"] = all(checks.values())
        report["final_status"] = client.get("/api/status").json()["runtime"]
        report["elapsed_seconds"] = time.time() - report["started"]
        private_json(Path("research/native_app_verification.json"), report)
        private_json(require_storage("/Volumes/移动数据库/lubanllm/experiments") / "native_app_verification.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        assert report["passed"], checks


if __name__ == "__main__":
    main()
