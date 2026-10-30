"""Real Bun RPC teacher practice regression; tiny CPU fixtures and offline teacher only."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from test_app import FakeRuntime, events
from luban.bun_bridge import RemoteMemory, RemoteSettings
from luban.memory import Memory
from luban.retention import eligible_learning_sample
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings
from luban.teacher import Teacher


class TinyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = "".join(message["content"] for message in messages)
        return list(range(len(text))) if tokenize else text

    def encode(self, text, **kwargs):
        return list(range(len(text)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    args = parser.parse_args()
    with patch("sqlite3.connect", side_effect=AssertionError("Python SQLite forbidden")):
        settings = Settings(args.home)
        memory = Memory(settings.snapshot()["storage_root"])
        assert isinstance(settings, RemoteSettings) and isinstance(memory, RemoteMemory)
        settings.accept_agreement()
        runtime = FakeRuntime(memory)
        runtime.state = "stopped"
        runtime.tokenizer = None
        runtime.device = "cpu"
        runtime.active_core = None
        runtime.branch = SimpleNamespace(expert_id=None)
        runtime.store = Path(memory.root) / "modules"
        for name in ("teacher_usage", "teacher_eligibility", "_sample_tokens"):
            setattr(runtime, name, getattr(Runtime, name).__get__(runtime))
        calls = []

        def respond(request):
            payload = json.loads(request.content)
            calls.append(payload)
            answer = "[true]" if payload["messages"][0]["content"].startswith("快速批改") else "教师即时答案"
            return httpx.Response(200, json={"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]})

        teacher = Teacher(settings, httpx.MockTransport(respond))
        messages = [{"role": "user", "content": "同一道问题"}]
        app = create_app(settings, runtime, teacher, autoload=False)
        practice = app.state.automatic.teacher_practice
        with TestClient(app, client=("127.0.0.1", 50000),
                        headers={"authorization": "Bearer " + os.environ["LUBAN_COMPUTE_TOKEN"]}) as client:
            response = client.post("/api/chat", json={"messages": messages})
            response.raise_for_status()
            stream = events(response)
            assert not any(item["type"] == "error" for item in stream), stream
            assert "".join(item.get("content", "") for item in stream) == "教师即时答案"
            assert stream[-1]["usage"]["unknown"] and stream[-1]["usage"]["total_tokens"] is None
            original = memory.teacher_materials()[0]
            assert original["status"] == "pending" and original["sample_id"] is None
            before = memory.usage()["teacher"]
            assert before["calls"] == before["unknown_calls"] == 1 and before["total_tokens"] is None
            runtime.identity = "rpc-practice-base"
            memory.model_identity = runtime.identity
            runtime.tokenizer = TinyTokenizer()
            runtime.state = "ready"

            def generate(context, emit, stopped, **options):
                assert context == messages and not options["allow_reference"] and not options["observe_external"]
                emit({"content": "<thi"})
                emit({"content": "nk>private reasoning</think>本地独立答案"})
                return {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4,
                        "finish_reason": "stop", "local_complete": True, "reference_used": False,
                        "local_action": "local_core", "core_version": None}

            runtime.generate = generate
            practice.stop_event.clear()
            practice.run_one()
            material = memory.teacher_materials()[0]
            samples = memory.samples()
            with memory.connect() as db:
                row = dict(db.execute("SELECT * FROM teacher_practice").fetchone())
                decision = json.loads(db.execute("SELECT body FROM events WHERE id=?", (row["decision_id"],)).fetchone()["body"])
            assert material["id"] == original["id"] and material["status"] == "eligible"
            assert material["sample_id"] == samples[0]["id"] and len(samples) == 1
            assert material["purpose"] == original["purpose"] == samples[0]["purpose"]
            assert not eligible_learning_sample(samples[0])
            assert row["state"] == "complete" and row["identity"] == runtime.identity
            assert decision["answer"] == "本地独立答案" and len(calls) == 2
            assert "教师即时答案" not in calls[1]["messages"][1]["content"]
            usage = memory.usage()["teacher"]
            assert usage["calls"] == 2 and usage["unknown_calls"] == 1
            assert all(usage[key] is None for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
            # Retrying the worker cannot issue another paid review or duplicate materials.
            practice.run_one()
            assert len(calls) == 2 and len(memory.samples()) == 1
            print(json.dumps({"remote_memory": True, "python_sqlite_forbidden": True,
                              "teacher_calls": len(calls), "material_id_preserved": material["id"] == original["id"],
                              "split_preserved": material["purpose"] == original["purpose"],
                              "sample_eligible_for_learning": eligible_learning_sample(samples[0]),
                              "state": row["state"], "teacher_usage": usage}), flush=True)


if __name__ == "__main__":
    main()
