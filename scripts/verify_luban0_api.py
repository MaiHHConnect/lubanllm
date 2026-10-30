"""Finite engineering regressions through the running native app's actual APIs.

Run only after the native app is ready and the chosen core version is approved.
This script uses local routes only and never requests a teacher or changes weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luban.generation_checks import evaluate_generation
from bootstrap_luban0 import IDENTITY_CHECK, TARGET


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fingerprint(paths):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def protocol(expected_core):
    release = json.loads((Path(__file__).resolve().parents[1] / "release.json").read_text())
    return {
        "kind": "native_api_engineering_regression", "expected_core": expected_core, "expected_version": release["version"],
        "scope": "finite engineering regression, including reused final-evaluation task styles; not a new pristine final test",
        "request": {"model": "luban0", "route": "local", "temperature": 0, "thinking": "off", "max_tokens": 128},
        "identity_expected": TARGET,
        "identity_accepted_variants": IDENTITY_CHECK["accepted"],
        "identity_check": IDENTITY_CHECK,
        "text_cases": [
            {"id": "identity-new-phrase", "messages": [{"role": "user", "content": "现在回应我的这个模型叫什么，属于谁？请直接说明你的身份。"}]},
            {"id": "identity-api-system", "messages": [{"role": "system", "content": "你是一个可靠的助手，请直接回答用户的问题。"},
                {"role": "user", "content": "你是谁？请介绍一下你自己。"}]},
            {"id": "identity-after-history", "messages": [{"role": "user", "content": "2+3等于多少？"},
                {"role": "assistant", "content": "5"}, {"role": "user", "content": "换个话题，请说明你的模型身份。"}]},
        ],
        "ui_case": {"id": "identity-ui-sse", "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}]},
        "audio_cases": [
            {"id": "audio-identity", "spoken": "你是谁？请介绍一下你自己。", "check": IDENTITY_CHECK},
            {"id": "audio-integer", "spoken": "请计算二加三，只回答数字。", "check": {"type": "semantic", "validator": "integer", "expected": 5}},
        ],
        "audio_source": "macOS say Tingting, mono PCM16 at16000Hz, synthetic speech, no microphone recording",
        "memory_method": "actual /api/status RSS and MPS allocator snapshots; not sampled peak, do not sum overlapping measures",
        "completion_method": "API local_complete=True (actual EOS, no cancellation) and finish_reason=stop; reference_used=False required",
        "reference_policy": "Open-book external references do not prove the identity was learned in model weights",
        "teacher_policy": "Explicit local route for every text generation; zero teacher session calls and unchanged cumulative teacher usage required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-core", required=True)
    parser.add_argument("--home", type=Path, default=Path.home() / "Library/Application Support/Luban")
    parser.add_argument("--storage-root", type=Path, default=Path("/Volumes/移动数据库/lubanllm"))
    parser.add_argument("--print-protocol", action="store_true", help="Inspect cases without files, credentials, HTTP, or model requests")
    args = parser.parse_args()
    if not re.fullmatch(r"core-[a-f0-9]{12}", args.expected_core):
        parser.error("--expected-core must identify the exact approved core version")
    planned = protocol(args.expected_core)
    if args.print_protocol:
        print(json.dumps(planned, ensure_ascii=False, indent=2))
        return
    root = args.storage_root
    if not str(root).startswith("/Volumes/") or not (Path("/Volumes") / root.parts[2]).is_mount():
        raise RuntimeError("Mounted external experiment storage is required")
    experiment = root / "experiments" / ("luban0-api-v1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                         + "-" + uuid.uuid4().hex[:6])
    experiment.mkdir(parents=True, exist_ok=False)
    planned["created_at"] = datetime.now(timezone.utc).isoformat()
    write_new(experiment / "protocol.json", planned)
    paths = [args.home / "settings.json", args.home / "credentials.json"]
    report = {"protocol_sha256": hashlib.sha256((experiment / "protocol.json").read_bytes()).hexdigest(),
              "protocol": planned, "experiment_dir": str(experiment), "passed": False, "cases": [], "snapshots": [],
              "hardware": platform.platform(), "hardware_memory_bytes": psutil.virtual_memory().total}
    secrets = []
    def clean(value):
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[redacted]")
        elif isinstance(value, list):
            value = [clean(item) for item in value]
        elif isinstance(value, dict):
            value = {key: clean(item) for key, item in value.items()}
        return value
    started = time.monotonic()
    try:
        report["private_file_hashes_before"] = fingerprint(paths)
        credentials = json.loads(paths[1].read_text())
        launch = json.loads((args.home / "launch.json").read_text())
        secrets = [value for value in credentials.values() if isinstance(value, str) and value]
        if launch.get("token") and launch["token"] not in secrets:
            secrets.append(launch["token"])
        port = launch["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Invalid local launch port")
        if launch.get("token") != credentials.get("ui_token"):
            raise ValueError("Launch session and private credentials do not match")
        base = f"http://127.0.0.1:{port}"
        report["local_base"] = base
        def request(path, *, body=None, raw=None, method=None):
            headers = {"Cookie": "luban_session=" + launch["token"]} if path.startswith("/api/") else {
                "Authorization": "Bearer " + credentials["access_key"]}
            if body is not None:
                headers["Content-Type"] = "application/json"
                data = json.dumps(body, ensure_ascii=False).encode()
            else:
                headers["Content-Type"] = "audio/wav" if raw is not None else "application/json"
                data = raw
            req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    result = response.read().decode("utf-8")
                    if response.headers.get_content_type() == "text/event-stream":
                        return [json.loads(line[6:]) for line in result.splitlines()
                                if line.startswith("data: ") and line != "data: [DONE]"]
                    return json.loads(result)
            except urllib.error.HTTPError as error:
                message = clean(error.read().decode("utf-8", errors="replace"))
                raise RuntimeError(f"Local API {path} returned HTTP {error.code}: {message}") from None

        def snapshot(label):
            status = request("/api/status")
            runtime = status["runtime"]
            record = {"label": label, "elapsed_seconds": time.monotonic() - started,
                      "runtime": runtime, "speech": status["speech"], "teacher_calls": status["teacher_calls"],
                      "teacher_usage": status.get("usage", {}).get("teacher")}
            report["snapshots"].append(record)
            if runtime["state"] != "ready" or runtime.get("core_version") != args.expected_core:
                raise RuntimeError("Runtime is not ready on the exact approved core")
            if runtime.get("active_expert") is not None:
                raise RuntimeError("Identity verification requires branch-OFF core inference")
            if status["teacher_calls"] != 0:
                raise RuntimeError("Teacher session calls must remain zero")
            return record

        health = request("/api/health")
        report["health"] = health
        if health.get("state") != "ready" or health.get("version") != planned["expected_version"]:
            raise RuntimeError("Start the ready " + planned["expected_version"] + " native application before verification")
        baseline = snapshot("before")

        def finish_case(record, text, usage, request_id, check):
            record.update(text=text, usage=usage, request_id=request_id)
            record["evaluation"] = evaluate_generation(text, check, usage.get("finish_reason") == "stop")
            events = request("/api/events")["events"]
            event = next((event["body"] for event in events
                          if event["kind"] == "decision" and event["body"].get("request_id") == request_id), None)
            record["decision_event"] = event
            after = snapshot(record["id"] + "-after")
            record["passed"] = bool(record["evaluation"]["passed"] and event
                and event["decision"].get("action") == "local_core" and event["decision"].get("route") == "local"
                and usage.get("requested_thinking") == "off" and usage.get("effective_thinking") == "off"
                and usage.get("local_complete") is True and usage.get("reference_used") is False
                and after["teacher_usage"] == baseline["teacher_usage"])
            if not record["passed"]:
                raise RuntimeError("Case failed: " + record["id"])

        def compatible(case, check):
            record = {"id": case["id"], "messages": case["messages"], "passed": False}
            report["cases"].append(record)
            response = request("/v1/chat/completions", body={**planned["request"], "messages": case["messages"]})
            record["response"] = response
            finish_case(record, response["choices"][0]["message"]["content"], response["usage"], response["id"], check)

        identity_check = planned["identity_check"]
        for case in planned["text_cases"]:
            compatible(case, identity_check)
        case = planned["ui_case"]
        ui_record = {**case, "passed": False}
        report["cases"].append(ui_record)
        events = request("/api/chat", body={**planned["request"], "messages": case["messages"]})
        ui_record["sse_events"] = events
        if any(event.get("type") == "error" for event in events):
            raise RuntimeError("UI SSE returned an error event")
        meta = next(event for event in events if event.get("type") == "meta")
        done = next(event for event in events if event.get("type") == "done")
        text = "".join(event.get("content", "") for event in events if event.get("type") == "delta")
        finish_case(ui_record, text, done["usage"], meta["request_id"], identity_check)

        for case in planned["audio_cases"]:
            record = {"id": case["id"], "spoken": case["spoken"], "passed": False}
            report["cases"].append(record)
            audio = experiment / (case["id"] + ".wav")
            subprocess.run(["say", "-v", "Tingting", "--file-format=WAVE", "--data-format=LEI16@16000",
                            "-o", str(audio), case["spoken"]], check=True, timeout=60)
            data = audio.read_bytes()
            record.update(audio_path=str(audio), audio_bytes=len(data), audio_sha256=hashlib.sha256(data).hexdigest())
            before_audio = snapshot(case["id"] + "-before-asr")
            speech_started = time.monotonic()
            asr = request("/v1/audio/transcriptions", raw=data)
            record.update(asr=asr, asr_request_seconds=time.monotonic() - speech_started)
            after_audio = snapshot(case["id"] + "-after-asr")
            record["asr_reported_unloaded"] = after_audio["speech"]["state"] == "unloaded"
            if (not isinstance(asr.get("text"), str) or not asr["text"].strip()
                    or asr.get("language") != "Chinese" or not record["asr_reported_unloaded"]
                    or after_audio["teacher_usage"] != before_audio["teacher_usage"]):
                raise RuntimeError("ASR transcript, language, release state, or usage failed")
            messages = [{"role": "user", "content": asr["text"]}]
            response = request("/v1/chat/completions", body={**planned["request"], "messages": messages})
            record.update(messages=messages, response=response)
            finish_case(record, response["choices"][0]["message"]["content"], response["usage"], response["id"], case["check"])
        final = snapshot("after")
        report["teacher_unchanged"] = final["teacher_usage"] == baseline["teacher_usage"]
        report["passed"] = all(record["passed"] for record in report["cases"]) and report["teacher_unchanged"]
    except Exception as error:
        report["error"] = clean(str(error))
        report["error_type"] = type(error).__name__
        report["passed"] = False
    finally:
        try:
            report["private_file_hashes_after"] = fingerprint(paths)
            report["private_files_unchanged"] = report.get("private_file_hashes_before") == report["private_file_hashes_after"]
            report["passed"] = report["passed"] and report["private_files_unchanged"]
        except OSError as error:
            report["private_file_check_error"] = clean(str(error))
            report["passed"] = False
        report["elapsed_seconds"] = time.monotonic() - started
        write_new(experiment / "result.json", clean(report))
        print(json.dumps({"passed": report["passed"], "experiment_dir": str(experiment),
                          "error": report.get("error"), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
