"""Real local weights + isolated app data + a clearly simulated HTTP teacher.

No paid network requests; the twelve route examples are protocol fixtures,
not evidence of difficulty recognition or generalization.
"""
import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

import httpx
import torch
from fastapi.testclient import TestClient
from luban.memory import Memory
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings, private_json, require_storage
from luban.teacher import Teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="research/app_smoke.json")
    args = parser.parse_args()
    external = require_storage("/Volumes/移动数据库/lubanllm/experiments")
    report = {"started": time.time(), "teacher": "httpx.MockTransport; no external calls", "checks": []}
    with tempfile.TemporaryDirectory(prefix="app-qa-", dir=external) as store, tempfile.TemporaryDirectory(prefix="luban-home-") as home:
        settings = Settings(home)
        settings.update({"storage_root": store})
        runtime = Runtime(settings, Memory(store))
        runtime.load()
        report["initial_status"] = runtime.status()
        deeper = []
        handle = runtime.model.model.layers[4].register_forward_hook(lambda *args: deeper.append(True))
        tokens = runtime._tokens([{"role": "user", "content": "浅层测试"}])
        representation = runtime.shallow(tokens)
        handle.remove()
        assert representation.shape == (1, tokens.shape[1], 2048) and not deeper
        report["checks"].append("shallow extraction executes 4 layers; layer 5 not reached")

        payloads = []

        def mock(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "fixture-teacher"}]})
            body = json.loads(request.content)
            payloads.append(body)
            assert request.headers["authorization"] == "Bearer simulated-teacher-key"
            if body["messages"][-1]["content"] == "自动材料验收":
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                    text='data: {"choices":[{"delta":{"content":"模拟教师回复"},"finish_reason":"stop"}]}\n\n'
                         'data: {"choices":[],"usage":{"prompt_tokens":23,"completion_tokens":7,"total_tokens":30}}\n\n'
                         'data: [DONE]\n\n')
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                text='data: {"choices":[{"delta":{"reasoning_content":"模拟检验过程"}}]}\n\n'
                     'data: {"choices":[{"delta":{"content":"模拟教师回复"}}]}\n\n'
                     'data: [DONE]\n\n')

        teacher = Teacher(settings, httpx.MockTransport(mock))
        app = create_app(settings, runtime, teacher, autoload=False)
        with TestClient(app) as client:
            assert client.get("/api/status").status_code == 401
            assert client.get("/v1/models").status_code == 401
            client.post("/api/session", json={"token": settings.secret("ui_token")}).raise_for_status()
            headers = {"Authorization": "Bearer " + settings.secret("access_key")}
            models = client.get("/v1/models", headers=headers).json()["data"]
            assert models == [{"id": "luban0", "object": "model", "owned_by": "maihaohuo"}]
            response = client.post("/v1/chat/completions", headers=headers,
                json={"messages": [{"role": "user", "content": "你好，请只回复一句问候。"}], "max_tokens": 40, "temperature": 0})
            response.raise_for_status()
            data = response.json()
            assert data["choices"][0]["message"]["content"] and data["usage"]["completion_tokens"] > 0
            report["local_reply"] = data
            report["checks"].append("real BF16 MPS local non-stream inference + authentication")
            with client.stream("POST", "/v1/chat/completions", headers=headers,
                               json={"messages": [{"role": "user", "content": "一加一等于几？"}], "max_tokens": 32, "stream": True, "temperature": 0}) as stream:
                text = "\n".join(stream.iter_lines())
            assert "chat.completion.chunk" in text and "[DONE]" in text
            report["checks"].append("compatible SSE actual model tokens + DONE")
            settings.update({"teacher": {"base_url": "https://fixture.invalid/v1", "api_key": "simulated-teacher-key",
                "model": "fixture-teacher", "thinking": "high", "send_reasoning_effort": True}})
            assert client.post("/api/teacher/models").json()["models"] == ["fixture-teacher"]
            first = [{"role": "user", "content": "模拟教师协议验收"}]
            result = client.post("/api/chat", json={"messages": first, "route": "teacher", "max_tokens": 16}).text
            events = [json.loads(line[6:]) for line in result.splitlines() if line.startswith("data: ")]
            conversation_id = next(e["conversation_id"] for e in events if e["type"] == "done")
            assert payloads[-1]["reasoning_effort"] == "high" and "模拟检验过程" in result
            previous = client.get("/api/conversations/" + conversation_id).json()["messages"]
            plain = [{"role": m["role"], "content": m["content"]} for m in previous]
            second = client.post("/api/chat", json={"messages": plain + [{"role": "user", "content": "再说一次"}],
                                     "route": "teacher", "conversation_id": conversation_id, "max_tokens": 16})
            assert "\"type\": \"error\"" not in second.text
            restored = client.get("/api/conversations/" + conversation_id).json()["messages"]
            assert restored[1]["source"] == "teacher" and restored[-1]["source"] == "teacher"
            report["checks"].append("simulated teacher models/reasoning/SSE/history provenance")
            denied = client.post("/api/samples", json={"prompt": "教师许可", "answer": "测试", "source": "teacher",
                                    "approved": True})
            assert denied.status_code == 400
            report["checks"].append("teacher material gate refuses sample before isolated agreement acceptance")

            def sample(prompt, answer, purpose="training", action=None):
                r = client.post("/api/samples", json={"prompt": prompt, "answer": answer, "purpose": purpose,
                                     "approved": True, "training_allowed": True, "route_action": action})
                r.raise_for_status()
                return r.json()["id"]

            sample("研究验收暗号是什么？", "青石。")
            sample("验收所用的口令叫什么？", "青石。")
            sample("本次验收的暗号叫做什么？", "青石。", "validation")
            client.post("/api/learn", json={"steps": 8}).raise_for_status()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                job = client.get("/api/status").json()["runtime"]["job"]
                if job and job["state"] in {"complete", "error"}:
                    break
                time.sleep(0.1)
            assert job["state"] == "complete", job
            report["learning"] = job["result"]
            assert len(job["result"]["candidates"]) == 3
            report["checks"].append("real neural candidate training + separate validation + promotion/rejection")
            if runtime.branch.expert is not None:
                with torch.no_grad():
                    before = runtime.model(tokens, use_cache=False).logits.cpu()
                client.post("/api/expert/hibernate").raise_for_status()
                assert runtime.branch.expert is None
                client.post("/api/expert/wake").raise_for_status()
                with torch.no_grad():
                    restored_logits = runtime.model(tokens, use_cache=False).logits.cpu()
                torch.testing.assert_close(before, restored_logits, rtol=0, atol=0)
                expert_id = runtime.branch.expert_id
                client.post("/api/expert/rollback", json={"id": expert_id}).raise_for_status()
                report["checks"].append("promoted neural weights hibernate/wake exact logits + rollback")

            for purpose, names in (("training", ("甲", "乙", "丙")), ("validation", ("丁", "戊", "己"))):
                for name in names:
                    sample("本地直接回答基础加法题：" + name + "，1+1？", "2。", purpose, "local_core")
                    sample("需要咨询老师查询未公布的私有代号：" + name + "。", "应咨询老师。", purpose, "teacher")
            client.post("/api/router/calibrate", json={}).raise_for_status()
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                job = client.get("/api/status").json()["runtime"]["job"]
                if job and job["state"] in {"complete", "error"}:
                    break
                time.sleep(0.1)
            assert job["state"] == "complete", job
            report["router"] = job["result"]["router_result"]
            report["route_after_calibration"] = runtime.choose_route([{"role": "user", "content": "需要咨询老师查询未公布的私有代号：庚。"}])
            report["checks"].append("real shallow neural head training + held-out fixture calibration")
            if report["router"]["calibrated"]:
                response = client.post("/api/chat", json={"messages": [{"role": "user", "content": "需要咨询老师查询未公布的私有代号：庚。"}], "route": "auto", "max_tokens": 16})
                assert "calibrated_head" in response.text
                report["checks"].append("auto calibrated head → simulated teacher route")
            assert runtime.memory.usage()["local"]["total_tokens"] > 0
            assert runtime.memory.usage()["teacher"]["estimated_calls"] >= 2
            assert runtime.memory.material_counts()["pending"] >= 2
            report["checks"].append("persistent local/teacher token counters + automatically captured teacher materials")
            settings.accept_agreement()  # Temporary test home only; never the user's agreement.
            count = runtime.memory.material_counts()["eligible"]
            response = client.post("/v1/chat/completions", headers=headers,
                json={"model": "luban0", "route": "teacher", "messages": [{"role": "user", "content": "自动材料验收"}]})
            response.raise_for_status()
            assert response.json()["usage"]["total_tokens"] == 30
            assert runtime.memory.material_counts()["eligible"] == count + 1
            material = next(m for m in runtime.memory.teacher_materials() if m["prompt"] == "自动材料验收")
            learned_material = next(s for s in runtime.memory.samples() if s["id"] == material["sample_id"])
            runtime._sample_tokens(learned_material)
            report["checks"].append("after isolated agreement acceptance, complete teacher fixture creates tokenizable material + exact provider usage")
            saved_expert = runtime.branch.expert_id
            saved_router = runtime.router_meta.copy()
            runtime.load()
            assert runtime.branch.expert_id == saved_expert and runtime.router_meta == saved_router
            report["checks"].append("reload restores active branch and calibrated head from SSD")
            stop = threading.Event()
            def cancel_after_delta(delta):
                stop.set()
            usage = runtime.generate([{"role": "user", "content": "写一段长文章"}], cancel_after_delta,
                                     stop, thinking="off", max_tokens=256, temperature=0)
            assert stop.is_set() and usage["completion_tokens"] < 256
            report["checks"].append("real generation cancellation before token limit")
            report["final_status"] = runtime.status()
        report["elapsed_seconds"] = time.time() - report["started"]
        private_json(Path(args.output), report)
        private_json(external / "app_smoke.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
