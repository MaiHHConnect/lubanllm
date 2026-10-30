"""Real local HTTP product acceptance; no paid teacher or production DB writes."""
import argparse
from contextlib import nullcontext
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
from unittest.mock import patch

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luban.memory import Memory
from luban.runtime import Runtime
from luban.server import create_app
from luban.settings import Settings
from luban import product_inference
from luban.coldpaging import full_backbone


def footprint(pid):
    usage = (ctypes.c_ubyte * 296)()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    if lib.proc_pid_rusage(pid, 4, ctypes.byref(usage)) != 0:
        return 0
    return int.from_bytes(bytes(usage)[72:80], "little")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir()  # Never replace evidence.
    storage = args.output / "storage"
    storage.mkdir()
    settings = Settings(args.output / "home")
    settings.update({"storage_root": str(storage), "model_path": "/Volumes/移动数据库/lubanllm/models/luban0",
                     "automatic": {"mode": "manual", "scheduled_time": "02:00"}})
    runtime = Runtime(settings, Memory(storage))
    samples = []
    phase = ["loading"]
    monitoring = threading.Event()

    def monitor():
        while not monitoring.wait(0.05):
            engine = runtime.native_engine
            child = engine.process.pid if engine and engine.process else None
            samples.append({"time": time.time(), "phase": phase[0], "parent": footprint(os.getpid()),
                            "child": footprint(child) if child else 0})

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(settings, runtime, port=port, autoload=False),
        host="127.0.0.1", port=port, access_log=False, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    report = {"mode": "timing" if args.timing else "acceptance", "teacher_calls": 0,
              "storage_isolated": True, "production_training_changed": False,
              "memory_method": "50ms sampled contemporaneous parent+child Darwin physical footprint; includes load, not sum of lifetime peaks"}
    report["inputs"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        "src/luban/runtime.py", "src/luban/product_inference.py", "src/luban/native_engine.py",
        "src/luban/coldpaging.py", "src/luban/server.py", "scripts/verify_native_product_api.py")}
    events = []
    try:
        started = time.perf_counter()
        runtime.load()
        report["load_seconds"] = time.perf_counter() - started
        report["inference"] = runtime.inference_identity()
        assert report["inference"]["engine"] == "luban-native-bf16-metal"
        assert not runtime.cold_backbone.loaded
        initial_pid = runtime.native_engine.process.pid
        phase[0] = "native"
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("Local HTTP server did not start")
            time.sleep(0.02)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=180,
                          headers={"authorization": "Bearer " + settings.secret("access_key")},
                          cookies={"luban_session": settings.secret("ui_token")}) as client:
            def request(prompt, *, stream=True, temperature=0, maximum=48, cancel=False):
                body = {"model": "luban0", "messages": [{"role": "user", "content": prompt}],
                        "route": "local", "thinking": "off", "max_tokens": maximum,
                        "temperature": temperature, "stream": stream}
                tick = time.perf_counter()
                row = {"prompt": prompt, "temperature": temperature, "maximum": maximum,
                       "stream": stream, "chunks": [], "first_visible_seconds": None}
                if stream:
                    with client.stream("POST", "/api/chat", json=body) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            event = json.loads(line[6:])
                            row["chunks"].append(event)
                            if event.get("type") == "error":
                                raise RuntimeError(event["message"])
                            if event.get("content") and row["first_visible_seconds"] is None:
                                row["first_visible_seconds"] = time.perf_counter() - tick
                                if cancel:
                                    meta = next(e for e in row["chunks"] if e["type"] == "meta")
                                    client.post("/api/stop", json={"request_id": meta["request_id"]}).raise_for_status()
                            if event.get("type") == "done":
                                row["usage"] = event["usage"]
                    row["text"] = "".join(e.get("content", "") for e in row["chunks"])
                else:
                    response = client.post("/v1/chat/completions", json=body)
                    response.raise_for_status()
                    value = response.json()
                    row["text"] = value["choices"][0]["message"]["content"]
                    row["usage"] = value["usage"]
                row["seconds"] = time.perf_counter() - tick
                row["native_pid"] = runtime.native_engine.process.pid if runtime.native_engine else None
                assert row["text"].strip()
                events.append(row)
                (args.output / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n")
                return row

            if args.timing:
                prompts = ["请用中文详细介绍太阳系的八颗行星，逐一说明特点。", "Describe the eight planets in our solar system in detail."]
                rows = {"native": [], "hf": []}
                for index in range(5):
                    for engine in (["native", "hf"] if index % 2 == 0 else ["hf", "native"]):
                        phase[0] = engine
                        if engine == "native" and runtime.native_engine is None:
                            product_inference.start(runtime)  # Preload is explicitly outside request timing.
                        context = patch("luban.product_inference.native_binary", return_value=None) if engine == "hf" else nullcontext()
                        with context:
                            for prompt in prompts:
                                row = request(prompt)
                                row.update(engine=engine, round=index + 1)
                                rows[engine].append(row)
                        print(json.dumps({"round": index + 1, "engine": engine}), flush=True)
                report["timing_scope"] = "Resident native model vs original product HF cold paging; both actual HTTP, 5 alternating rounds, 2 prompts, max48 greedy; native model preload excluded"
                report["measurements"] = rows
                report["per_prompt"] = [{engine: {"first_visible_mean": statistics.mean(r["first_visible_seconds"] for r in rows[engine] if r["prompt"] == prompt),
                    "first_visible_sample_sd": statistics.stdev(r["first_visible_seconds"] for r in rows[engine] if r["prompt"] == prompt),
                    "whole_request_mean": statistics.mean(r["seconds"] for r in rows[engine] if r["prompt"] == prompt),
                    "visible_output_tokens_per_second_mean": statistics.mean((r["usage"]["completion_tokens"] - 1) / (r["seconds"] - r["first_visible_seconds"]) for r in rows[engine] if r["prompt"] == prompt)} for engine in rows} for prompt in prompts]
            else:
                a = request("只输出中文：你好。", maximum=32)
                b = request("只输出数字：2+3等于多少？", stream=False, maximum=32)
                c = request("只输出中文：你好。", maximum=32)
                assert a["text"] == c["text"] and "5" in b["text"]
                sampled = request("用两句中文介绍太阳。", temperature=0.6, maximum=48)
                cancelled = request("从一开始连续列出100个整数。", maximum=512, cancel=True)
                resumed = request("只输出数字：2+3等于多少？", maximum=32)
                assert resumed["text"] == b["text"] and cancelled["usage"]["completion_tokens"] < 512
                assert all(r["native_pid"] == initial_pid for r in events)
                with full_backbone(runtime):
                    assert runtime.native_engine is None and runtime.cold_backbone.loaded
                assert not runtime.cold_backbone.loaded
                again = request("只输出数字：2+3等于多少？", maximum=32)
                assert again["text"] == b["text"] and again["native_pid"] != initial_pid
                report["checks"] = {"actual_http_stream": True, "nonstream": True, "chinese": True,
                    "sampling_0_6": True, "request_isolation": True, "same_resident_pid": True,
                    "cancel_resume": True, "full_hf_mutual_exclusion": True, "paging_release_native_restart": True}
            report["passed"] = True
    finally:
        server.should_exit = True
        if thread.is_alive():
            thread.join(timeout=10)
        runtime._release_model()
        monitoring.set()
        watcher.join(timeout=1)
        (args.output / "memory.json").write_text(json.dumps(samples) + "\n")
        report["phase_peaks_bytes"] = {name: max(r["parent"] + r["child"] for r in samples if r["phase"] == name) for name in {r["phase"] for r in samples}}
        report["sampled_combined_peak_bytes"] = max((r["parent"] + r["child"] for r in samples), default=0)
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
