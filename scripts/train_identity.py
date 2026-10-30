"""Train the user-defined brand identity through the local neural branch.

The target is a user-requested persona statement, not an experimental claim
that unbounded capability growth has been achieved.
"""
import json
import time
from pathlib import Path
import httpx
from luban.settings import default_home, private_json, require_storage

TARGET = "我是麦好火基于开源模型的基座神经元架构重建的一款鲁班零号模型，无限进化是我的能力。"
QUESTIONS = {
    "training": ["你是谁？", "请介绍你自己。", "你叫什么，是谁开发的？", "你是哪一个模型？", "你是阿里巴巴开发的助手吗？"],
    "validation": ["你的身份是什么？", "介绍一下你的名称和开发团队。"],
}
REPLAY = {
    "training": [
        ("将英文 Hello 翻译为中文。", "你好。"),
        ("把 Hello 翻译成中文，只输出翻译。", "你好。"),
        ("英文 hello 的中文是什么？", "你好。"),
        ("2+3等于多少？只输出数字。", "5"),
        ("请把 Thank you 翻译成中文。", "谢谢。"),
        ("写一个 Python 函数 add_numbers(a, b)，返回两个数的和，只输出代码。", "def add_numbers(a, b):\n    return a + b"),
    ],
    "validation": [
        ("翻译单词 Hello，给出中文。", "你好。"),
        ("3加2的结果是什么？", "5"),
        ("用中文表达 Thank you。", "谢谢。"),
    ],
}


def main():
    home = default_home()
    credentials = json.loads((home / "credentials.json").read_text())
    base = "http://127.0.0.1:8765"
    started = time.time()
    report = {"target": TARGET, "source": "explicit user identity request", "brand_scope": "persona, not a scientific unlimited-evolution claim"}
    with httpx.Client(base_url=base, timeout=180) as client:
        client.post("/api/session", json={"token": credentials["ui_token"]}).raise_for_status()
        records = client.get("/api/samples").json()["samples"]
        for purpose, questions in QUESTIONS.items():
            examples = [(prompt, TARGET, "human") for prompt in questions] + [(prompt, answer, "synthetic") for prompt, answer in REPLAY[purpose]]
            for prompt, answer, source in examples:
                if any(s["prompt"] == prompt and s["answer"] == answer and s["purpose"] == purpose for s in records):
                    continue
                client.post("/api/samples", json={"prompt": prompt, "answer": answer, "purpose": purpose,
                            "source": source, "approved": True, "training_allowed": True}).raise_for_status()
        client.post("/api/learn", json={"steps": 48}).raise_for_status()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            job = client.get("/api/status").json()["runtime"]["job"]
            if job and job["state"] in {"complete", "error"}:
                break
            time.sleep(0.2)
        assert job["state"] == "complete", job
        report["learning"] = job["result"]
        headers = {"Authorization": "Bearer " + credentials["access_key"]}
        replies = []
        for prompt in ("你是谁", "你是什么模型？", "告诉我你的名字和开发者。"):
            response = client.post("/v1/chat/completions", headers=headers,
                json={"messages": [{"role": "user", "content": prompt}], "route": "local",
                      "max_tokens": 128, "temperature": 0, "thinking": "off"})
            response.raise_for_status()
            value = response.json()
            answer = value["choices"][0]["message"]["content"]
            replies.append({"prompt": prompt, "answer": answer, "usage": value["usage"],
                           "identity_correct": answer.strip() == TARGET})
        report["replies"] = replies
        report["identity_passed"] = all(r["identity_correct"] for r in replies)
        report["learning_promoted"] = report["learning"]["promoted"]
        report["passed"] = report["identity_passed"] and report["learning_promoted"]
        report["elapsed_seconds"] = time.time() - started
        report["status"] = client.get("/api/status").json()["runtime"]
        private_json(Path("research/identity_training.json"), report)
        private_json(require_storage("/Volumes/移动数据库/lubanllm/experiments") / "identity_training.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["identity_passed"]:
            raise RuntimeError("身份回答尚未达到验收要求；请保留报告并继续修正训练")


if __name__ == "__main__":
    main()
