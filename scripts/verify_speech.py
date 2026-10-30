"""One synthetic Chinese smoke test; not a recognition-quality benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luban.speech import DEFAULT_SPEECH_MODEL_PATH, SpeechRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_SPEECH_MODEL_PATH)
    parser.add_argument("--storage-root", default="/Volumes/移动数据库/lubanllm")
    args = parser.parse_args()
    storage = Path(args.storage_root)
    if str(storage).startswith("/Volumes/") and not (Path("/Volumes") / storage.parts[2]).is_mount():
        raise RuntimeError("experiment storage volume is not mounted")
    experiment = storage / "experiments" / ("speech-v1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                               + "-" + uuid.uuid4().hex[:6])
    experiment.mkdir(parents=True, exist_ok=False)
    reference = "你好，我是鲁班零号。请帮我计算二加三等于多少。"
    audio = experiment / "synthetic-zh.wav"
    speech = SpeechRuntime(args.model_path)
    process = psutil.Process()
    measurements = []
    stop = threading.Event()
    def sample():
        while not stop.wait(0.05):
            measurements.append({"elapsed_seconds": time.monotonic() - started,
                                 "rss_bytes": process.memory_info().rss,
                                 "state": speech.state})
    report = {"kind": "synthetic_chinese_asr_smoke", "created_at": datetime.now(timezone.utc).isoformat(),
              "model_path": args.model_path, "model_revision": None,
              "source": "macOS say Tingting; synthetic speech, no human recording or training",
              "reference": reference, "hardware": platform.platform(), "torch": torch.__version__,
              "hardware_memory_bytes": psutil.virtual_memory().total,
              "transformers": transformers.__version__, "device": "cpu", "precision": "float32",
              "seed": 42, "split": "one fixed smoke fixture; no training or tuning",
              "limitations": "Single synthetic utterance; not WER/CER benchmark, not word timestamps, not simultaneous 2B residency proof",
              "memory_method": "psutil process RSS every 50 ms; sampled peak includes CPU tensor memory, not physical-memory peak",
              "passed": False}
    manifest = Path(args.model_path) / "upstream-file-manifest.json"
    if manifest.is_file():
        report["model_revision"] = json.loads(manifest.read_text())["revision"]
    thread = None
    started = time.monotonic()
    try:
        subprocess.run(["say", "-v", "Tingting", "--file-format=WAVE", "--data-format=LEI16@16000",
                        "-o", str(audio), reference], check=True, timeout=60)
        data = audio.read_bytes()
        report["audio_sha256"] = hashlib.sha256(data).hexdigest()
        report["audio_bytes"] = len(data)
        torch.manual_seed(42)
        # Bound CPU worker threads on the local Apple Silicon host.
        torch.set_num_threads(min(8, torch.get_num_threads()))
        report["cpu_threads"] = torch.get_num_threads()
        report["rss_before_bytes"] = process.memory_info().rss
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        report["result"] = speech.transcribe_bytes(data)
        report["rss_after_release_bytes"] = process.memory_info().rss
        report["released"] = speech.model is None and speech.processor is None
        normalized = report["result"]["text"].replace(" ", "")
        report["passed"] = ("你好" in normalized and "鲁班零号" in normalized
                            and ("二加三" in normalized or "2加3" in normalized)
                            and report["released"])
    except Exception as error:
        report["error"] = str(error)
        report["error_type"] = type(error).__name__
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=2)
        report["elapsed_seconds"] = time.monotonic() - started
        report["rss_sampled_peak_bytes"] = max([m["rss_bytes"] for m in measurements] or [process.memory_info().rss])
        report["experiment_dir"] = str(experiment)
        (experiment / "rss-samples.json").write_text(json.dumps(measurements, indent=2) + "\n")
        (experiment / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
