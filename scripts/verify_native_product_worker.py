#!/usr/bin/env python3
"""Run bound real native-worker regressions; never change product core pointers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luban.native_engine import NativeEngine, NativeEngineError
from tokenizers import Tokenizer


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def bind(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def results(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {row["id"]: row for row in rows if row["event"] == "result"}


class AuditedEngine(NativeEngine):
    def __init__(self, *args, event_log, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_log = event_log

    def _read_stdout(self, stream, events, stopped):
        log = self.event_log

        class Tee:
            def readline(self, limit):
                line = stream.readline(limit)
                if line:
                    log.write(line.decode("utf-8"))
                    log.flush()
                return line

        NativeEngine._read_stdout(Tee(), events, stopped)


def formal_core(source, provenance_path, destination):
    """Rewrite only the single tensor's header name, preserving every BF16 byte."""
    provenance = json.loads(provenance_path.read_bytes())
    assert sha(source) == provenance["file_sha256"]
    raw = source.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + length])
    assert set(header) == {provenance["parameter"]}
    entry = header[provenance["parameter"]]
    data = raw[8 + length:]
    assert hashlib.sha256(data).hexdigest() == provenance["native_tensor_sha256"]
    key = provenance["source_parameter"]
    assert key == "model.language_model.layers.23.mlp.base.down_proj.weight"
    assert entry["dtype"] == "BF16" and entry["shape"] == [2048, 6144]
    assert entry["data_offsets"] == [0, len(data)]
    encoded = json.dumps({key: entry}, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    destination.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return {"file": str(destination), "tensor_key": key, "sha256": sha(destination)}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--binary", default=str(ROOT / "build/native/luban-worker"))
    parser.add_argument("--old", default="/Volumes/移动数据库/lubanllm/experiments/native-inference-20261010T060000Z-b68d710e")
    parser.add_argument("--reference", default="/Volumes/移动数据库/lubanllm/experiments/native-inference-optimization-20261010T074835Z")
    args = parser.parse_args()
    out, old, reference, binary = map(Path, (args.output, args.old, args.reference, args.binary))
    assert out.is_absolute() and out.parts[1] == "Volumes" and os.path.ismount(Path(*out.parts[:3]))
    out.mkdir()  # Do not overwrite previous runs or evidence.
    invocation = json.loads((reference / "prefill16-trained/invocation.json").read_bytes())
    model = Path(invocation["command"][1])
    manifest = invocation["manifest_sha256"]
    cases_paths = {"ranking": old / "cases-ranking.json", "trained": old / "cases-preservation.json"}
    refs = {name: reference / f"prefill16-{name}/events.jsonl" for name in cases_paths}
    cases = {name: json.loads(path.read_bytes()) for name, path in cases_paths.items()}
    expected = {name: results(path) for name, path in refs.items()}
    assert len(cases["ranking"]) == 48 and len(cases["trained"]) == 112
    for name in cases:
        assert {row["id"] for row in cases[name]} == set(expected[name])
    sources = [Path(__file__), ROOT / "src/luban/native_engine.py", binary, Path(str(binary) + ".json"),
               model / "upstream-file-manifest.json", *cases_paths.values(), *refs.values(),
               old / "core-export-v1/down23.bf16.safetensors", old / "core-export-v1/provenance.json",
               reference / "ranking-comparison.json", reference / "trained-comparison.json"]
    protocol = {"started_utc": datetime.now(timezone.utc).isoformat(), "inputs": [bind(path) for path in sources],
                "manifest_sha256": manifest, "temperature": 0, "seed": 42, "teacher_calls": 0,
                "training": False, "product_adopted": False,
                "scope": "Actual product worker generation versus immutable qualified native research tokens; not new quality scoring or model adoption"}
    save(out / "protocol.json", protocol)
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    controls, comparisons, pids, closed = {}, {}, [], []
    started = time.monotonic()
    with (out / "requests.jsonl").open("w") as requests, (out / "answers.jsonl").open("w") as answers:
        def request(engine, row, phase, *, temperature=0, seed=42, stop=None, cancel_after=None):
            stop = stop or threading.Event()
            sent = {key: row[key] for key in ("id", "tokens", "max_tokens", "eos_ids")}
            requests.write(json.dumps({"phase": phase, "case": sent, "temperature": temperature, "seed": seed}) + "\n")
            requests.flush()
            seen = []

            def token(value):
                seen.append(value)
                if cancel_after is not None and len(seen) == cancel_after:
                    stop.set()

            result = engine.generate(row["tokens"], row["max_tokens"], row["eos_ids"], temperature, token, stop, seed=seed)
            assert result["tokens"] == seen
            answers.write(json.dumps({"phase": phase, "fixture_id": row["id"], "pid": engine.process.pid,
                                      "text": tokenizer.decode(seen), "result": result}, ensure_ascii=False) + "\n")
            answers.flush()
            return result

        def suite(engine, name):
            rows = []
            for index, row in enumerate(cases[name]):
                actual = request(engine, row, name)
                prior = expected[name][row["id"]]
                equal = actual["tokens"] == prior["tokens"] and actual["finish_reason"] == prior["finish_reason"] and actual["complete"] == prior["complete"]
                rows.append({"id": row["id"], "tokens_exact": equal, "output_tokens": len(actual["tokens"])})
                if (index + 1) % 16 == 0 or index + 1 == len(cases[name]):
                    print(json.dumps({"phase": name, "done": index + 1, "total": len(cases[name]), "elapsed_s": time.monotonic() - started}), flush=True)
            comparisons[name] = {"cases": len(rows), "exact": sum(row["tokens_exact"] for row in rows), "rows": rows,
                                 "historical_quality_reference_only": json.loads((reference / f"{name}-comparison.json").read_bytes())["quality"]}
            save(out / f"{name}-comparison.json", comparisons[name])
            assert all(row["tokens_exact"] for row in rows), f"Full {name} token equivalence failed"

        for name in ("ranking", "trained"):
            core = None
            if name == "trained":
                core, provenance = formal_core(old / "core-export-v1/down23.bf16.safetensors", old / "core-export-v1/provenance.json", out / "research-core.safetensors")
                save(out / "research-core-binding.json", {**core, "tensor_sha256": provenance["native_tensor_sha256"], "product_adopted": False})
            with (out / f"{name}-events.jsonl").open("w") as event_log:
                engine = AuditedEngine(binary, model, manifest, core, event_log=event_log)
                processes = []
                try:
                    engine.start()
                    processes.append(engine.process)
                    pids.append(engine.process.pid)
                    suite(engine, name)
                    if name == "ranking":
                        short = min(cases[name], key=lambda row: len(expected[name][row["id"]]["tokens"]))
                        baseline = request(engine, short, "isolation-before")
                        large = {**short, "id": "long-1200", "tokens": [8678] * 1200, "max_tokens": 1}
                        long_result = request(engine, large, "long-prompt")
                        assert len(long_result["tokens"]) == 1 and long_result["prompt_tokens"] == 1200
                        large_output = request(engine, {**short, "max_tokens": 2048}, "max-output-2048")
                        assert large_output["tokens"] == baseline["tokens"]
                        after = request(engine, short, "isolation-after")
                        assert after["tokens"] == baseline["tokens"]
                        controls["long_prompt_and_2048_limit"] = True
                        controls["request_state_isolation"] = True
                        sampled = [request(engine, short, "sampling", temperature=.6, seed=seed) for seed in (42, 42, 43)]
                        assert sampled[0]["tokens"] == sampled[1]["tokens"]
                        controls["sampling"] = {"same_seed_exact": True, "other_seed_differs": sampled[0]["tokens"] != sampled[2]["tokens"]}
                        long = max(cases[name], key=lambda row: len(expected[name][row["id"]]["tokens"]))
                        cancel_start = time.monotonic()
                        cancelled = request(engine, {**long, "max_tokens": 64}, "cancel", cancel_after=2)
                        assert cancelled["finish_reason"] == "cancelled" and not cancelled["complete"]
                        assert len(cancelled["tokens"]) >= 2
                        assert engine.process.pid == pids[0]
                        resumed = request(engine, short, "cancel-resume")
                        assert resumed["tokens"] == baseline["tokens"] and engine.process.pid == pids[0]
                        controls["cancel"] = {"same_pid_resume": True, "tokens": len(cancelled["tokens"]), "total_s": time.monotonic() - cancel_start}
                        try:
                            request(engine, {**short, "tokens": [248320]}, "invalid-vocab")
                        except NativeEngineError as exc:
                            controls["invalid_token_error"] = str(exc)
                            assert "InvalidQwenToken" in str(exc) and engine.process is None
                        else:
                            raise AssertionError("Out-of-vocabulary token was accepted")
                        recovered = request(engine, short, "error-restart")
                        processes.append(engine.process)
                        assert recovered["tokens"] == baseline["tokens"] and engine.process.pid != pids[0]
                        controls["error_restart"] = True
                finally:
                    engine.close()
                    closed.extend({"pid": process.pid, "returncode": process.poll()} for process in processes)
    assert all(process["returncode"] is not None for process in closed)
    assert all(sha(Path(item["path"])) == item["sha256"] for item in protocol["inputs"]), "Inputs changed during verification"
    report = {"passed": True, "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.monotonic() - started,
              "comparisons": {key: {k: v for k, v in row.items() if k != "rows"} for key, row in comparisons.items()},
              "controls": controls, "workers_closed": closed, "training": False, "product_adopted": False, "teacher_calls": 0,
              "evidence": [bind(path) for path in sorted(out.iterdir()) if path.is_file() and not path.name.startswith("._")]}
    save(out / "result.json", report)
    print(json.dumps({"passed": True, "comparisons": {name: row["exact"] for name, row in comparisons.items()}, "workers_closed": closed}), flush=True)


if __name__ == "__main__":
    main()
