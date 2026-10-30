"""One preregistered original-core preflight, then freeze known-behavior protection.

Failed baseline answers stay diagnostics with unchanged gold, never counted as
preserved abilities. V3 development was informed by prior failures. The same
untouched v2 final file is referenced, copied verbatim, and never generated here.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import uuid

from prepare_replay_v2 import DATASET as V2_DATASET, GREETING, protocol as v2_protocol, write_new
from bootstrap_luban0 import TARGET
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.memory import Memory
from luban.settings import DEFAULTS, Settings, require_storage

DATASET = "luban0-v3-authored-preflight"


def protocol():
    previous = v2_protocol()
    rows = copy.deepcopy([row for row in previous["rows"] if row["split"] != "final"])
    for row in rows:
        row["id"] = row["id"].replace("v2-", "v3-", 1)
        if row["context_family"] == "elementary-fact":
            row["generation_check"].update(type="semantic", validator="normalized_exact")
        if row["generation_check"].get("validator") == "nonempty":
            row["prompt"] += "请简洁回答，不超过60字。"
            row["messages"][-1]["content"] = row["prompt"]
            row["generation_check"]["max_new_tokens"] = 512
    api1 = [{"role": "system", "content": "你通过兼容OpenAI的本地API回应。请按用户要求输出。"}]
    api2 = [{"role": "system", "content": "客户端需要知道当前助手叫什么；如实按品牌设定回答。"}]
    arithmetic = [{"role": "user", "content": "计算1加2，只输出数字。"}, {"role": "assistant", "content": "3"}]
    translation = [{"role": "user", "content": "把Hello译为中文。"}, {"role": "assistant", "content": "你好"}]
    identity = [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": TARGET}]
    attribution = [{"role": "user", "content": "你的调优方是谁？"}, {"role": "assistant", "content": TARGET}]
    contexts = [
        (api1, "请给接入框架一句完整的模型身份说明。", TARGET, "identity", "计算9加7，只输出整数。", "16", {"validator": "integer", "expected": 16}),
        (api2, "告诉我你叫什么，是谁调优的。", TARGET, "identity", "把Good evening翻译为中文，只写译文。", "晚上好", {"validator": "translation", "accepted": ["晚上好"]}),
        (arithmetic, "先确认模型身份：你是谁，由谁调优？", TARGET, "identity", "现在计算17减4，只回答整数。", "13", {"validator": "integer", "expected": 13}),
        (translation, "请说清你现在的模型名和调优者。", TARGET, "identity", "接着把Good morning译成中文，仅写译文。", "早上好", {"validator": "translation", "accepted": ["早上好", "早安"]}),
        (identity, "现在正式开始对话，先打个招呼。", GREETING, "greeting", "计算5乘6，只输出整数。", "30", {"validator": "integer", "expected": 30}),
        (attribution, "与新访客开始聊天时，请先向对方问好。", GREETING, "greeting", "给出JSON数组，包含数字2和4，不要其他文字。", "[2, 4]", {"validator": "json", "expected": [2, 4]}),
    ]
    for index, (prefix, prompt, answer, kind, ordinary_prompt, ordinary_answer, ordinary_check) in enumerate(contexts):
        check = ({"type": "exact", "expected": TARGET, "max_new_tokens": 128} if kind == "identity" else
                 {"type": "contains", "expected": "我是鲁班零号", "forbidden": ["通义千问"], "max_new_tokens": 128})
        rows.append({"id": f"v3-training-new-context-{index}", "split": "training", "evolution_role": "new",
                     "context_family": kind+"-matched-system-history", "prompt": prompt, "answer": answer,
                     "messages": prefix+[{"role": "user", "content": prompt}], "generation_check": check})
        rows.append({"id": f"v3-training-ordinary-context-{index}", "split": "training", "evolution_role": "replay",
                     "context_family": "ordinary-matched-system-history", "prompt": ordinary_prompt, "answer": ordinary_answer,
                     "messages": prefix+[{"role": "user", "content": ordinary_prompt}],
                     "generation_check": {"type": "semantic", "max_new_tokens": 128, "forbidden": ["麦好火", "luban0", "鲁班零号", "通义千问", "Qwen"], **ordinary_check}})
    for row in rows:
        row["generation_check"] = validate_generation_check(row["generation_check"])
        assert evaluate_generation(row["answer"], row["generation_check"], True)["passed"]
    assert len({json.dumps(row["messages"], ensure_ascii=False, sort_keys=True) for row in rows}) == len(rows)
    return {"dataset": DATASET, "identity_target": TARGET, "rows": rows,
            "development_scope": "informed by v1/v2 results; not independent final evidence",
            "eligibility_policy": "one original branch-OFF greedy replay-development generation; only passed baseline cases protect known behavior",
            "failed_policy": "gold unchanged; failed baseline kept as diagnostic outside the effective preservation dataset",
            "creative_scope": "complete nonempty brand-free output under explicit length request; finite structural check, not creative quality",
            "context_scope": "six new-behavior system/history training contexts have six matched ordinary-task contexts",
            "teacher_calls": 0}


def as_sample(row):
    return {"id": row["id"], "prompt": row["prompt"], "answer": row["answer"], "purpose": row["split"],
            "provenance": {"messages": row["messages"], "generation_check": row["generation_check"],
                           "evolution_role": row["evolution_role"], "dataset": DATASET}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-preflight", action="store_true", help="Run ONE original-core replay preflight, audit, freeze and seed Memory")
    parser.add_argument("--storage", default=DEFAULTS["storage_root"])
    parser.add_argument("--model", default=DEFAULTS["model_path"])
    args = parser.parse_args()
    plan = protocol()
    plan["planned_counts"] = {split: dict(Counter(row["evolution_role"] for row in plan["rows"] if row["split"] == split))
                              for split in ("training", "validation")}
    if not args.apply_preflight:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    summary = Path(__file__).resolve().parents[1] / "research/luban0_v3_replay.json"
    if summary.exists():
        raise FileExistsError("V3 already ran; never resample the baseline mask to pass")
    root = require_storage(args.storage)
    v2 = json.loads((summary.parent / "luban0_v2_replay.json").read_text())
    final_path = Path(v2["final_test_path"])
    final_bytes = final_path.read_bytes()
    run = root / "experiments" / ("luban0-v3-preflight-" + uuid.uuid4().hex)
    run.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((Path(args.model) / "upstream-file-manifest.json").read_text())
    identity_records = sorted((row["Path"], row["Sha256"]) for row in manifest["Data"]["Files"]
                              if row["Path"].endswith(".safetensors") or row["Path"] == "config.json")
    base_identity = hashlib.sha256(json.dumps(identity_records).encode()).hexdigest()
    memory = Memory(root)
    memory.model_identity = base_identity
    before = memory.samples()
    bootstrap = [row for row in before if row["approved"] and row["training_allowed"] and row["purpose"] == "validation"
                 and json.loads(row["provenance"]).get("dataset") == "luban0-v1-user-authored"
                 and json.loads(row["provenance"]).get("evolution_role") == "replay"]
    if len(bootstrap) != 4:
        raise ValueError("Expected the four original bootstrap arithmetic protection rows")
    plan.update(final_source=str(final_path), final_sha256=hashlib.sha256(final_bytes).hexdigest(),
                bootstrap_replay_validation_ids=[row["id"] for row in bootstrap], base_identity=base_identity)
    write_new(run / "protocol.json", plan)
    write_new(run / "database-before.json", before)
    with (run / "final-test.json").open("xb") as output:
        output.write(final_bytes)
    baseline_samples = [as_sample(row) for row in plan["rows"] if row["split"] == "validation" and row["evolution_role"] == "replay"] + bootstrap
    write_new(run / "baseline-plan.json", baseline_samples)
    print(json.dumps({"planned_counts": plan["planned_counts"], "original_baseline_cases": len(baseline_samples),
                      "run": str(run), "final_evaluated": False}), flush=True)
    from luban.runtime import Runtime
    from luban.evolution import _generation, consolidation_samples
    import torch
    import psutil
    report = {"dataset": DATASET, "run_directory": str(run), "base_identity": base_identity, "teacher_calls": 0,
              "planned_counts": plan["planned_counts"], "final_test_path": str(run / "final-test.json"),
              "final_source_path": str(final_path), "final_sha256": plan["final_sha256"], "final_evaluated": False,
              "development_scope": plan["development_scope"], "eligibility_policy": plan["eligibility_policy"]}
    baseline, snapshots = [], []
    started = time.perf_counter()
    runtime = None
    timer = None
    try:
        with tempfile.TemporaryDirectory(prefix="luban0-v3-original-preflight-") as home:
            settings = Settings(home)
            settings.update({"storage_root": args.storage, "model_path": args.model, "temperature": 0})
            runtime = Runtime(settings, memory)
            runtime.job = {"kind": "original-preflight", "cancel_requested": False}
            timer = threading.Timer(600, lambda: runtime.job.update(cancel_requested=True))
            timer.daemon = True
            timer.start()
            runtime.load()
            if runtime.active_core is not None or runtime.branch.expert is not None or runtime.identity != base_identity:
                raise RuntimeError("Original core/branch-OFF preflight required; no learned core or expert may be active")
            report.update(core_version=None, branch_enabled=False, device=runtime.device, precision=str(runtime.dtype))

            def snapshot(stage):
                snapshots.append({"stage": stage, "rss_bytes": psutil.Process().memory_info().rss,
                                  "mps_driver_bytes": torch.mps.driver_allocated_memory() if runtime.device == "mps" else None})

            snapshot("loaded")
            for index, sample in enumerate(baseline_samples):
                result = _generation(runtime, [sample])[0]
                baseline.append({"sample": sample, "generation": result, "eligible": result["passed"], "gold_unchanged": True})
                write_new(run / f"baseline-{index:02d}.json", baseline[-1])
                print(json.dumps({"case": index+1, "total": len(baseline_samples), "id": sample["id"],
                                  "prompt": sample["prompt"], "answer": result["answer"], "eligible": result["passed"],
                                  "complete": result["complete"]}, ensure_ascii=False), flush=True)
            snapshot("preflight_complete")
            write_new(run / "raw-original-baseline.json", baseline)
            write_new(run / "creative-length-audit.json", [
                {"id": row["sample"]["id"], "answer_characters": len(row["generation"]["answer"]),
                 "requested_max_characters": 60, "complete": row["generation"]["complete"], "structural_passed": row["eligible"],
                 "scope": "Length request is not enforced by the finite nonempty/EOS/brand-free gate; no creative quality claim"}
                for row in baseline if isinstance(row["sample"]["provenance"], dict)
                and row["sample"]["provenance"]["generation_check"].get("validator") == "nonempty"])
            mask = {row["sample"]["id"]: row["eligible"] for row in baseline}
            write_new(run / "frozen-eligibility.json", {"mask": mask, "policy": plan["eligibility_policy"],
                      "baseline_core_version": None, "base_identity": base_identity})
            # Mask is frozen before any database edits; no alternate prompts or retries.
            retired = []
            bootstrap_ids = {row["id"] for row in bootstrap}
            with memory.lock, memory.connect() as db:
                for old in before:
                    provenance = json.loads(old["provenance"])
                    is_v2 = provenance.get("dataset") == V2_DATASET
                    if not is_v2 and old["id"] not in bootstrap_ids:
                        continue
                    current = db.execute("SELECT * FROM samples WHERE id=?", (old["id"],)).fetchone()
                    if current is None or dict(current) != old:
                        raise RuntimeError("Database changed after preregistration; refusing mixed audit")
                    if is_v2:
                        provenance.update(superseded_by=DATASET, superseded_at=datetime.now(timezone.utc).isoformat(),
                                          superseded_because="v3 fixed preflight eligibility, Unicode fact normalization, explicit creative budget and matched context training")
                        db.execute("UPDATE samples SET approved=0,provenance=? WHERE id=?", (json.dumps(provenance, ensure_ascii=False), old["id"]))
                        retired.append(old["id"])
                    else:
                        provenance.update(baseline_preservation_eligible=mask[old["id"]], baseline_preflight_run=str(run),
                                          baseline_preflight_core_version=None, baseline_preflight_base_identity=base_identity)
                        db.execute("UPDATE samples SET provenance=? WHERE id=?", (json.dumps(provenance, ensure_ascii=False), old["id"]))
            existing = {json.dumps(json.loads(row["provenance"]).get("messages") or [{"role": "user", "content": row["prompt"]}], sort_keys=True, ensure_ascii=False): row
                        for row in memory.samples() if row["approved"] and row["training_allowed"]}
            seeded, diagnostic_ids = [], []
            for row in plan["rows"]:
                is_protection = row["split"] == "validation" and row["evolution_role"] == "replay"
                if is_protection and not mask[row["id"]]:
                    diagnostic_ids.append(row["id"])
                    continue
                context = json.dumps(row["messages"], sort_keys=True, ensure_ascii=False)
                if context in existing:
                    old = existing[context]
                    if old["answer"] != row["answer"] or old["purpose"] != row["split"]:
                        raise ValueError("Existing authored identity/protection target conflicts with frozen v3 protocol")
                    seeded.append({"protocol_id": row["id"], "sample_id": old["id"], "reused": True})
                    continue
                provenance = {"dataset": DATASET, "evolution_role": row["evolution_role"], "protocol_id": row["id"],
                              "messages": row["messages"], "generation_check": row["generation_check"], "context_family": row["context_family"],
                              "benchmark_run": str(run), "rights_basis": "project-authored targets; no upstream generated continuation or paid API",
                              "verification_evidence": "fixed authored ground truth/structural rule; original baseline eligibility frozen for preservation development",
                              "validation_role": "informed development" if row["split"] == "validation" else "training"}
                if is_protection:
                    provenance.update(baseline_preservation_eligible=True, baseline_preflight_run=str(run),
                                      baseline_preflight_core_version=None, baseline_preflight_base_identity=base_identity)
                sample_id = memory.add_sample(row["prompt"], row["answer"], source="human", purpose=row["split"],
                                              approved=True, training_allowed=True, provenance=provenance)
                seeded.append({"protocol_id": row["id"], "sample_id": sample_id, "reused": False})
            report.update(retired_v2_ids=retired, seeded=seeded, diagnostic_protocol_ids=diagnostic_ids,
                          baseline_passed=sum(mask.values()), baseline_total=len(mask), eligibility_mask=mask,
                          effective_counts={key: len(values) for key, values in consolidation_samples(runtime).items()})
            snapshot("seeded")
    except Exception as error:
        report.update(error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if timer:
            timer.cancel()
        report.update(elapsed_seconds=time.perf_counter()-started, memory_snapshots=snapshots,
                      memory_measurement="phase-boundary process RSS/MPS allocation snapshots; not continuous peak",
                      protocol_sha256=hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest())
        write_new(run / "result.json", report)
        write_new(summary, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if runtime is not None:
            del runtime
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
