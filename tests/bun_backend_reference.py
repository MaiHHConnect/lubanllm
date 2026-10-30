"""Isolated HTTP oracle/private compute for Bun backend tests; never load 2B weights."""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn
import luban.server

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_app import FakeRuntime
from luban.memory import Memory
from luban.server import create_app
from luban.settings import Settings
from luban.teacher import Teacher


class RuntimeFixture(FakeRuntime):
    def load(self):
        self.state = "ready"

    def generate(self, messages, emit, stop, **kwargs):
        if kwargs.get("tools"):
            if messages[-1].get("content") == "slow-tools":
                time.sleep(1.4)
            # A failed assertion becomes an HTTP error, so the real public tests
            # also prove tool requests never enable reference/learning observation.
            assert kwargs.get("allow_reference") is False
            assert kwargs.get("observe_external") is False
            self.calls.append((messages, kwargs))
            if messages[-1]["role"] == "tool":
                text, reason = "工具结果：" + messages[-1]["content"], "stop"
            elif messages[-1]["content"] == "tool-truncated":
                text, reason = '<tool_call><function=lookup><parameter=city>上海', "length"
            else:
                text = '<tool_call>\n<function=lookup>\n<parameter=city>\n上海\n</parameter>\n</function>\n</tool_call>'
                reason = "stop"
            middle = len(text) // 2
            emit({"content": text[:middle]})
            emit({"content": text[middle:]})
            return {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                    "finish_reason": reason}
        if messages[-1]["content"] != "slow-stream":
            return super().generate(messages, emit, stop, **kwargs)
        for _ in range(100):
            if stop.is_set():
                break
            emit({"content": "piece"})
            time.sleep(0.03)
        return {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4,
                "finish_reason": "stop"}


class SpeechFixture:
    def __init__(self, path):
        self.model_path = Path(path)

    def status(self):
        return {"state": "unloaded", "fixture": True}

    def transcribe_bytes(self, data, language=None):
        return {"text": "CPU ASR boundary fixture", "language": language, "bytes": len(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--mode", choices=["oracle", "private"], required=True)
    args = parser.parse_args()
    luban.server.SSE_HEARTBEAT_SECONDS = 0.1
    settings = Settings(args.home)
    runtime = RuntimeFixture(Memory(settings.snapshot()["storage_root"]))
    runtime.tokenizer = None  # Let the real review projection inspect CPU-only candidates.
    runtime.store = runtime.memory.root / "modules" / "luban0-v1"
    # All model functions are inherited from the established API CPU fixture.
    # Every HTTP teacher/network request uses a deterministic offline transport.
    transport = httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "offline fixture"}))
    teacher = Teacher(settings, transport)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        app = create_app(settings, runtime=runtime, teacher=teacher, port=port, autoload=False,
                         speech=SpeechFixture(settings.snapshot()["speech_model_path"]))
        app.state.leaderboard.transport = transport
        app.state.leaderboard.country_transport = transport
        inventory = sorted({method + " " + route.path for route in app.routes
                            if hasattr(route, "methods") for method in route.methods
                            if route.path.startswith(("/api/", "/v1/"))})
        # Emission precedes uvicorn readiness; the Bun test waits for actual health.
        print(json.dumps({"port": port, "routes": inventory, "ui_token": settings.secret("ui_token"),
                          "access_key": settings.secret("access_key")}), flush=True)
        guard = patch("sqlite3.connect", side_effect=AssertionError("Python SQLite forbidden"))
        if args.mode == "private":
            guard.start()
        try:
            uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False,
                                         log_level="error", proxy_headers=False)).run(sockets=[sock])
        finally:
            if args.mode == "private":
                guard.stop()


if __name__ == "__main__":
    main()
