#!/usr/bin/env python3
"""Compare local installed backends without training, teachers or changing model pointers."""
import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--database-backup", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Evidence output must be new")
    home = Path.home() / "Library/Application Support/Luban"
    launch = json.loads((home / "launch.json").read_text())
    credentials = json.loads((home / "credentials.json").read_text())
    base = f'http://127.0.0.1:{launch["port"]}'
    report = {"backend": launch.get("backend", "python"), "pid": launch["pid"],
              "compute_pid": launch.get("compute_pid"), "teacher_calls_requested": 0,
              "training_requested": False, "precision": "existing BF16 native Metal engine",
              "memory_measurement": "not measured; no performance or memory claim",
              "protocol": {"temperature": 0, "thinking": "off", "max_tokens": 128,
                           "route": "local", "prompts": ["你是谁？", "计算 17 + 25，只输出数字。"]}}
    with httpx.Client(base_url=base, timeout=120, trust_env=False,
                      cookies={"luban_session": launch["token"]}) as client:
        def get(path):
            response = client.get(path)
            response.raise_for_status()
            return response.json()
        report["health"] = get("/api/health")
        report["status"] = get("/api/status")
        runtime = report["status"]["runtime"]
        assert runtime["state"] == "ready", runtime["state"]
        job = runtime.get("job") or {}
        assert job.get("state") not in {"queued", "running"}, "Active job; leave it untouched"
        close = get("/api/automatic/close-status")
        assert not close.get("running") and not close.get("package_busy"), "Active work"
        report["data"] = {}
        payloads = {}
        for path in ["/api/conversations", "/api/samples", "/api/materials"]:
            value = get(path)
            payloads[path] = value
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            report["data"][path] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                    "records": sum(len(v) for v in value.values() if isinstance(v, list))}
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        if baseline:
            assert report["backend"] == "bun"
            for path in ("/api/samples", "/api/materials"):
                assert report["data"][path] == baseline["data"][path], "Existing learning records changed"
            if args.database_backup:
                with sqlite3.connect(args.database_backup.resolve().as_uri() + "?mode=ro", uri=True) as db:
                    db.row_factory = sqlite3.Row
                    old = [dict(row) for row in db.execute("SELECT id,title,updated FROM conversations ORDER BY updated DESC LIMIT 50")]
                    old_ids = {row["id"] for row in old}
                    retained = [row for row in payloads["/api/conversations"]["conversations"] if row["id"] in old_ids]
                    assert retained == old, "Existing conversation metadata changed"
                    for row in db.execute("SELECT * FROM conversations"):
                        previous = dict(row)
                        previous["messages"] = json.loads(previous["messages"])
                        assert get("/api/conversations/" + previous["id"]) == previous, "Existing conversation content changed"
                raw = json.dumps({"conversations": old}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                assert hashlib.sha256(raw).hexdigest() == baseline["data"]["/api/conversations"]["sha256"]
                report["conversation_preservation"] = {"existing": len(old), "new": report["data"]["/api/conversations"]["records"] - len(old), "complete_records_equal": True}
            else:
                assert report["data"] == baseline["data"], "Business records changed during migration"
            for key in ("core_version", "model_path", "inference"):
                assert runtime[key] == baseline["status"]["runtime"][key], f"Changed {key}"
        report["responses"] = []
        for prompt in report["protocol"]["prompts"]:
            body = {**{k: v for k, v in report["protocol"].items() if k != "prompts"},
                    "messages": [{"role": "user", "content": prompt}], "stream": False}
            started = time.perf_counter()
            response = client.post("/v1/chat/completions", json=body,
                                   headers={"Authorization": "Bearer " + credentials["access_key"]})
            response.raise_for_status()
            value = response.json()
            report["responses"].append({"prompt": prompt, "choice": value["choices"][0],
                                        "usage": value["usage"], "seconds": time.perf_counter() - started})
        if baseline:
            for previous, current in zip(baseline["responses"], report["responses"], strict=True):
                assert current["choice"] == previous["choice"], "Generated output changed"
                stable_usage = lambda value: {k: v for k, v in value.items() if k != "elapsed_seconds"}
                assert stable_usage(current["usage"]) == stable_usage(previous["usage"]), "Token/finish accounting changed"
            report["comparison"] = {"prompts": 2, "output_and_usage_equal": True,
                                    "existing_business_records_preserved": True, "elapsed_seconds_excluded": True,
                                    "baseline": str(args.baseline)}
        report["final_status"] = get("/api/status")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Local backend comparison passed; teacher/training requests: 0")


if __name__ == "__main__":
    main()
