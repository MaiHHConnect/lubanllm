"""Label the explicit persona/replay dataset and fit its local module gate.

Labels describe this narrow experiment; they do not label arbitrary task
difficulty or establish teacher-routing reliability.
"""
import json
import time
from pathlib import Path

import httpx
from luban.memory import Memory
from luban.settings import default_home, private_json, require_storage
from train_identity import QUESTIONS, REPLAY

HARD_NEGATIVES = {
    "training": [
        ("介绍一下阿里巴巴的 Qwen 系列模型。", "Qwen 是阿里巴巴开发的语言模型系列。"),
        ("Qwen3 的神经网络是什么结构？", "Qwen3 使用 Transformer 架构。"),
        ("介绍鲁班这个中国古代工匠。", "鲁班是中国古代工匠。"),
        ("有人问我‘你是谁’，这句话翻译为英文是什么？", "Who are you?"),
    ],
    "validation": [
        ("阿里巴巴开发的通义千问是什么？", "通义千问是语言模型系列。"),
        ("中国古代工匠鲁班有什么故事？", "鲁班是中国古代工匠。"),
    ],
}

def main():
    root = require_storage("/Volumes/移动数据库/lubanllm")
    credentials = json.loads((default_home() / "credentials.json").read_text())
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=120) as client:
        client.post("/api/session", json={"token": credentials["ui_token"]}).raise_for_status()
        existing = client.get("/api/samples").json()["samples"]
        for purpose, examples in HARD_NEGATIVES.items():
            for prompt, answer in examples:
                if any(s["prompt"] == prompt and s["purpose"] == purpose for s in existing):
                    continue
                client.post("/api/samples", json={"prompt": prompt, "answer": answer, "purpose": purpose,
                    "source": "synthetic", "approved": True, "training_allowed": True, "route_action": "local_core"}).raise_for_status()
    memory = Memory(root)
    persona = {q for questions in QUESTIONS.values() for q in questions}
    general = {q for collection in (REPLAY, HARD_NEGATIVES) for examples in collection.values() for q, _ in examples}
    labelled = []
    with memory.lock, memory.connect() as db:
        for row in db.execute("SELECT id,prompt,source,provenance FROM samples").fetchall():
            if row["prompt"] not in persona | general:
                continue
            action = "neural_expert" if row["prompt"] in persona else "local_core"
            provenance = json.loads(row["provenance"])
            provenance.update(route_action=action, route_label_basis="explicit persona task versus synthetic base-capability replay", route_label_scope="narrow research dataset")
            db.execute("UPDATE samples SET provenance=? WHERE id=?", (json.dumps(provenance, ensure_ascii=False), row["id"]))
            labelled.append({"id": row["id"], "action": action})
    report = {"labels": labelled, "scope": "core versus identity expert; no teacher labels or paid calls"}
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=120) as client:
        client.post("/api/session", json={"token": credentials["ui_token"]}).raise_for_status()
        client.post("/api/router/calibrate", json={}).raise_for_status()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            job = client.get("/api/status").json()["runtime"]["job"]
            if job and job["state"] in {"complete", "error"}:
                break
            time.sleep(0.2)
        assert job["state"] == "complete", job
        report["router"] = job["result"]["router_result"]
        private_json(Path("research/identity_gate.json"), report)
        private_json(root / "experiments/identity_gate.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        assert report["router"]["calibrated"], report


if __name__ == "__main__":
    main()
