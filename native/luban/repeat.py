# SPDX-License-Identifier: Apache-2.0
"""Run five serial engine comparisons and retain individual measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summary(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "min": min(values), "max": max(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "model", "cases", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-binary", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    scripts = Path(__file__).resolve().parent
    cases = json.loads(args.cases.read_bytes())
    if len(cases) != 3 or any(c["dump_logits"] for c in cases):
        raise ValueError("Use one short warmup and two fixed performance cases without logit export")
    engines = ("baseline", "native", "hf") if args.baseline_binary else ("native", "hf")
    order = ([["baseline", "native", "hf"], ["hf", "native", "baseline"],
              ["native", "baseline", "hf"], ["hf", "baseline", "native"],
              ["baseline", "hf", "native"]] if args.baseline_binary else
             [["native", "hf"] if i % 2 == 0 else ["hf", "native"] for i in range(5)])
    protocol = {"rounds": 5, "gpu_execution": "serial", "process": "fresh per engine per round",
                "sampling": "greedy", "seed": 42, "reference": "original FP32 parameters restored",
                "order": order,
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "timing_scope": "CLI; warmup omitted; cold request cache; not product HTTP",
                "peak_scope": "fresh process lifetime; HF includes initial multimodal load",
                "product_adopted": False}
    if args.baseline_binary:
        protocol["baseline_binary_sha256"] = hashlib.sha256(args.baseline_binary.read_bytes()).hexdigest()
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    samples = {engine: {case["id"]: [] for case in cases[1:]} for engine in engines}
    for round_index, round_engines in enumerate(protocol["order"]):
        for engine in round_engines:
            target = args.output / f"round-{round_index + 1}-{engine}"
            command = [sys.executable, str(scripts / ("run_native.py" if engine != "hf" else "reference_probe.py")),
                       "--model", str(args.model), "--cases", str(args.cases), "--output", str(target)]
            if engine != "hf":
                command += ["--binary", str(args.baseline_binary if engine == "baseline" else args.binary)]
            with target.with_suffix(".console.log").open("x") as out:
                subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
            rows = [json.loads(line) for line in (target / "events.jsonl").read_text().splitlines()]
            rows = [row for row in rows if row["event"] == "result"]
            if [row["id"] for row in rows] != [case["id"] for case in cases]:
                raise ValueError("Missing or reordered cases")
            for row, case in zip(rows[1:], cases[1:], strict=True):
                if row["generated_tokens"] != case["max_tokens"] or row["decode_ns"] <= 0:
                    raise ValueError("Performance cases must reach their fixed token budget")
                samples[engine][row["id"]].append({"round": round_index + 1,
                    "first_visible_seconds": row["client_first_visible_text_ns"] / 1e9,
                    "ttft_seconds": row["ttft_ns"] / 1e9,
                    "decode_tokens_per_second": (row["generated_tokens"] - 1) * 1e9 / row["decode_ns"],
                    "peak_physical_bytes": row["memory"]["lifetime_peak_physical_footprint_bytes"]})
            print(json.dumps({"completed_round": round_index + 1, "engine": engine}), flush=True)
    report = {"protocol": protocol, "samples": samples, "per_prompt": {}}
    for case in cases[1:]:
        report["per_prompt"][case["id"]] = {
            engine: {metric: summary([row[metric] for row in samples[engine][case["id"]]])
                     for metric in ("first_visible_seconds", "ttft_seconds", "decode_tokens_per_second", "peak_physical_bytes")}
            for engine in samples}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["per_prompt"]), flush=True)


if __name__ == "__main__":
    main()
