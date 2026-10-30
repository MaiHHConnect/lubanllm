"""Real trained head + HTTP contract, with a clearly offline mock teacher."""
import json
import tempfile
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from luban.memory import Memory
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings, private_json, require_storage
from luban.teacher import Teacher


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def main():
    settings = Settings()
    root = require_storage(settings.snapshot()["storage_root"])
    runtime = Runtime(settings, Memory(root))
    runtime.load()
    training = json.loads(Path("research/teacher_router.json").read_text())
    case = next(row for row in training["held_out"]["decisions"] if row["decision"].get("action") == "teacher")
    messages = [{"role": "user", "content": case["prompt"]}]
    repair_path = Path("research/identity_router_repair.json")
    repair = json.loads(repair_path.read_text()) if repair_path.exists() else {}
    current_training = repair if repair.get("promoted") else training
    assert runtime.router_meta["sha256"] == current_training["calibration"]["sha256"]
    with tempfile.TemporaryDirectory(prefix="teacher-route-qa-", dir=root / "experiments") as store, tempfile.TemporaryDirectory(prefix="teacher-qa-home-") as home:
        isolated = Settings(home)
        runtime.memory = Memory(store)
        calls = []

        def mock(request):
            calls.append(request)
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "fixture-teacher"}]})
            payload = json.loads(request.content)
            answer = case["answer"] if payload["messages"] == messages else "OK"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=
                "data: " + json.dumps({"choices": [{"delta": {"content": answer}, "finish_reason": "stop"}]}) + "\n\n" +
                'data: {"choices": [], "usage": {"prompt_tokens": 37, "completion_tokens": 5, "total_tokens": 42}}\n\n' +
                "data: [DONE]\n\n")

        teacher = Teacher(isolated, httpx.MockTransport(mock))
        with TestClient(create_app(isolated, runtime, teacher, autoload=False)) as client:
            client.post("/api/session", json={"token": isolated.secret("ui_token")}).raise_for_status()
            fallback = client.post("/api/chat", json={"messages": messages, "route": "auto", "max_tokens": 1})
            parsed = events(fallback)
            assert parsed[0]["source"] == "teacher_unconfigured" and parsed[0]["route"] == "local"
            assert parsed[0]["suggested_action"] == "teacher" and parsed[0]["confidence"] is None
            assert any(row["type"] == "done" for row in parsed)
            assert not calls
            isolated.update({"teacher": {"base_url": "http://fixture.invalid/v1", "api_key": "offline-fixture-key",
                "model": "fixture-teacher"}})
            isolated.accept_agreement()  # Temporary test home only; never the user's agreement.
            response = client.post("/api/chat", json={"messages": messages, "route": "auto"})
            parsed = events(response)
            assert parsed[0]["route"] == "teacher" and parsed[0]["source"] == "calibrated_head"
            assert "".join(row.get("content", "") for row in parsed if row["type"] == "delta") == case["answer"]
            assert parsed[-1]["usage"]["total_tokens"] == 42
            assert runtime.memory.material_counts() == {"total": 1, "pending": 0, "eligible": 1}
            before = isolated.path.read_bytes(), isolated.secret_path.read_bytes()
            draft = {"base_url": "http://draft.invalid/v1", "api_key": "offline-draft-key", "model": "draft-model"}
            models = client.post("/api/teacher/models", json={"teacher": draft})
            assert models.status_code == 200 and models.json()["models"] == ["fixture-teacher"]
            tested = client.post("/api/teacher/test", json={"teacher": draft})
            assert tested.status_code == 200 and tested.json()["reply"] == "OK"
            assert json.loads(calls[-1].content)["max_tokens"] == 16
            assert before == (isolated.path.read_bytes(), isolated.secret_path.read_bytes())
            assert runtime.memory.usage()["teacher"]["calls"] == 2
            assert runtime.memory.material_counts()["total"] == 1, "Connection-test reply must not enter learning"
        report = {"scope": "actual trained shallow head plus offline HTTP teacher; no provider quality measurement or paid calls",
                  "checks": ["new checkpoint reload and unchanged base identity", "auto teacher decision falls back to local when unconfigured",
                             "actual head automatically selects offline teacher; exact protocol usage and material capture",
                             "draft HTTP models/test use form credentials without saving; 16-token request; connection-test reply excluded from learning"],
                  "router_sha256": runtime.router_meta["sha256"], "status": runtime.status(), "passed": True}
        private_json(Path("research/teacher_router_verification.json"), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
