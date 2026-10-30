"""Isolated real text comparison; a pass does not replace the full application."""
from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import time
import urllib.request

CASES = [
    [{"role": "user", "content": "你是谁？"}],
    [{"role": "user", "content": "是谁调优了你？"}],
    [{"role": "system", "content": "你通过API服务用户。"}, {"role": "user", "content": "你是谁？"}],
    [{"role": "user", "content": "2+3是多少？"}, {"role": "assistant", "content": "5"},
     {"role": "user", "content": "再介绍一下你自己。"}],
    [{"role": "user", "content": "2+3等于多少？只回答数字。"}],
    [{"role": "user", "content": "7乘以8是多少？只回答数字。"}],
    [{"role": "user", "content": "把Hello翻译成中文。"}],
    [{"role": "user", "content": "写一个Python函数，返回两个数的和。"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Return only the result: 12 + 15."}],
    [{"role": "user", "content": "请复述：cafe\u0301，中文，🙂。"}],
    [{"role": "user", "content": "  请回答OK。\n\t"}],
]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def client():
    home = Path.home() / "Library/Application Support/Luban"
    secrets = json.loads((home / "credentials.json").read_text())
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    with opener.open(urllib.request.Request("http://127.0.0.1:8765/api/session",
        data=json.dumps({"token": secrets["ui_token"]}).encode(), headers={"Content-Type": "application/json"}), timeout=5):
        pass
    return opener, secrets


def status(opener):
    with opener.open("http://127.0.0.1:8765/api/status", timeout=10) as response:
        return json.load(response)["runtime"]


def baseline(root):
    root.mkdir(exist_ok=False)
    opener, secrets = client()
    before = status(opener)
    if (before["state"] != "ready" or before["job"] is not None or before["active_expert"] is not None
            or before["router"].get("calibrated") is True):
        raise RuntimeError("Current product does not match the isolated local-core comparison scope")
    settings = json.loads((Path.home() / "Library/Application Support/Luban/settings.json").read_text())
    model = Path(settings["model_path"])
    store = Path(settings["storage_root"]) / "modules/luban0-v1"
    pointer = json.loads((store / "active-core.json").read_text())
    core = store / (pointer["version"] + ".safetensors")
    if pointer["version"] != before["core_version"] or sha(core) != pointer["sha256"]:
        raise RuntimeError("Current core pointer/binary changed")
    with core.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 2 <= length <= 1 << 20:
            raise ValueError("Invalid core header")
        header = json.loads(stream.read(length))
    keys = set(header) - {"__metadata__"}
    if len(keys) != 1:
        raise ValueError("Expected qualified single native core tensor")
    copied = root / core.name
    shutil.copyfile(core, copied)
    if sha(copied) != pointer["sha256"] or sha(core) != pointer["sha256"]:
        raise RuntimeError("Core changed during copying")
    report = {"scope": "Current real HTTP local-core answers vs pure-Swift text candidate; all application features are a separate gate",
              "baseline_before": before, "core": {"file": str(copied), "tensor_key": next(iter(keys)), "sha256": pointer["sha256"]},
              "identity": pointer["identity"], "model": str(model), "rows": [], "teacher_api_calls": 0,
              "performance_scope": "Diagnostic only: current HTTP+word streaming vs isolated JSONL+UTF8 streaming, not full-app speed proof"}
    for index, messages in enumerate(CASES):
        request = {"model": "luban0", "messages": messages, "route": "local", "thinking": "off",
                   "max_tokens": 128, "temperature": 0, "stream": True}
        began = time.monotonic_ns()
        first = None
        text, reasoning, usage, finish = "", "", None, None
        with opener.open(urllib.request.Request("http://127.0.0.1:8765/v1/chat/completions",
            data=json.dumps(request, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + secrets["access_key"]}), timeout=180) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError("Baseline inference returned an error")
                item = event["choices"][0]
                delta = item.get("delta", {})
                if delta.get("content") and first is None:
                    first = time.monotonic_ns() - began
                text += delta.get("content", "")
                reasoning += delta.get("reasoning_content", "")
                if item.get("finish_reason"):
                    finish, usage = item["finish_reason"], event.get("usage")
        if usage is None or usage.get("reference_used") is True or reasoning:
            raise RuntimeError("Baseline did not execute the closed-book direct-answer path")
        report["rows"].append({"index": index, "request": request, "text": text, "finish_reason": finish,
                               "usage": usage, "wall_ns": time.monotonic_ns() - began, "visible_first_ns": first})
        save(root / "baseline-progress.json", report)
    finish_baseline(root, report, status(opener))


def finish_baseline(root, report, after):
    before = report["baseline_before"]
    if after["core_version"] != before["core_version"] or after["inference"] != before["inference"]:
        raise RuntimeError("Production engine/core changed during reference recording")
    # Internal reference preparation may use Python; the candidate child cannot.
    from transformers import AutoTokenizer
    model = Path(report["model"])
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    for row in report["rows"]:
        ids = tokenizer.apply_chat_template(row["request"]["messages"],
            add_generation_prompt=True, tokenize=True, enable_thinking=False)
        row["input_tokens"] = ids["input_ids"] if hasattr(ids, "keys") else ids
        row["expected_output_tokens"] = tokenizer.encode(row["text"], add_special_tokens=False)
    report.update(baseline_after=after, passed=True, model_manifest_sha256=sha(model / "upstream-file-manifest.json"),
                  tokenizer_reference="Installed product AutoTokenizer; text tokenization is not a raw completion-token oracle")
    save(root / "baseline.json", report)
    print(json.dumps({"baseline_cases": len(report["rows"]), "core": before["core_version"], "passed": True}), flush=True)


def event(process, pending, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if b"\n" in pending:
            end = pending.index(10)
            if end > 4 << 20:
                raise RuntimeError("Candidate oversized output")
            line = pending[:end]
            del pending[:end + 1]
            return json.loads(line)
        if len(pending) > 4 << 20:
            raise RuntimeError("Candidate oversized output")
        if select.select([process.stdout], [], [], .1)[0]:
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Candidate closed output")
            pending.extend(chunk)
            continue
        if process.poll() is not None:
            raise RuntimeError("Candidate exited before completion")
    raise TimeoutError("Candidate comparison timed out")


def candidate(root, executable, binary):
    baseline = json.loads((root / "baseline.json").read_text())
    rows = baseline.get("rows", [])
    if (baseline.get("passed") is not True or not isinstance(rows, list) or len(rows) != len(CASES)
            or [row.get("index") for row in rows] != list(range(len(CASES)))
            or any(row.get("request", {}).get("messages") != messages for row, messages in zip(rows, CASES))
            or any(row.get("request", {}).get("route") != "local" or row.get("request", {}).get("temperature") != 0
                   or row.get("request", {}).get("thinking") != "off" or row.get("request", {}).get("max_tokens") != 128
                   or row.get("usage", {}).get("reference_used") is not False for row in rows)):
        raise ValueError("Full approved closed-book reference case list is required")
    folder = root / "swift-candidate"
    folder.mkdir(exist_ok=False)
    copied_binary = folder / "luban-worker"
    shutil.copyfile(binary, copied_binary)
    copied_binary.chmod(0o700)
    shutil.copyfile(str(binary) + ".json", str(copied_binary) + ".json")
    config = {"binary": str(copied_binary), "model": baseline["model"], "identity": baseline["identity"],
              "manifest_sha256": baseline["model_manifest_sha256"], "core": baseline["core"], "log": str(folder / "worker.stderr")}
    save(folder / "config.json", config)
    report = {"passed": False, "production_switched": False, "rows": [], "teacher_api_calls": 0,
              "scope": baseline["scope"], "performance_scope": baseline["performance_scope"],
              "binary_sha256": sha(executable), "worker_sha256": sha(copied_binary), "baseline_sha256": sha(root / "baseline.json")}
    with (folder / "cli.stderr").open("wb") as stderr:
        process = subprocess.Popen([str(executable), str(folder / "config.json")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, bufsize=0, start_new_session=True)
        try:
            os.set_blocking(process.stdout.fileno(), False)
            pending = bytearray()
            ready = event(process, pending)
            if ready.get("event") != "ready" or ready.get("python_invocations") != 0:
                raise RuntimeError("Candidate did not start as a pure native runtime")
            report["ready"] = ready
            subprocess.run(["/usr/bin/otool", "-L", str(executable)], check=True,
                stdout=(folder / "cli-linkage.txt").open("wb"))
            for row in baseline["rows"]:
                process.stdin.write((json.dumps(row["request"], ensure_ascii=False) + "\n").encode())
                process.stdin.flush()
                answer = event(process, pending)
                checked = (answer.get("event") == "completion" and answer.get("text") == row["text"]
                           and answer.get("finish_reason") == row["finish_reason"]
                           and answer.get("input_tokens") == row["input_tokens"]
                           and answer.get("streamed_text") == answer.get("text")
                           and answer.get("generated_tokens") == row["usage"]["completion_tokens"])
                report["rows"].append({"index": row["index"], "passed": checked, "candidate": answer})
                save(folder / "progress.json", report)
            process.stdin.close()
            process.wait(timeout=10)
            report["passed"] = process.returncode == 0 and len(report["rows"]) == len(CASES) and all(row["passed"] for row in report["rows"])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
            # The private session contains only this candidate and its native child.
            # A killed Swift main cannot execute defer; reap the complete group too.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            report["exit_code"] = process.returncode
            save(folder / "result.json", report)
    print(json.dumps({"passed": report["passed"], "cases": len(report["rows"]), "production_switched": False}), flush=True)
    if not report["passed"]:
        raise RuntimeError("Real Swift text comparison failed; application replacement forbidden")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "baseline-finalize", "candidate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--worker", type=Path, default=Path("build/native/luban-worker"))
    args = parser.parse_args()
    if not os.path.ismount("/Volumes/移动数据库") or not args.output.is_relative_to(Path("/Volumes/移动数据库/lubanllm/experiments")):
        raise ValueError("New external experiment directory required")
    if args.mode == "baseline":
        baseline(args.output)
    elif args.mode == "baseline-finalize":
        report = json.loads((args.output / "baseline-progress.json").read_text())
        if len(report["rows"]) != len(CASES) or sha(report["core"]["file"]) != report["core"]["sha256"]:
            raise RuntimeError("Incomplete reference recording")
        opener, _ = client()
        finish_baseline(args.output, report, status(opener))
    else:
        if args.executable is None:
            parser.error("candidate requires --executable")
        candidate(args.output, args.executable.resolve(), args.worker.resolve())
