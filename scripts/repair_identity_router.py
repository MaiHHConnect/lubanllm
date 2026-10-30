"""Repair ordinary-task false positives while preserving the original expert.

This scoped repair does not claim to fix the existing thinking repetition.
"""
import argparse
import gc
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import torch

from luban.backups import save_training_backup
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings, default_home, private_json, require_storage
from repair_identity import (CORE_TRAIN, CORE_CALIBRATION, CORE_HELD_OUT, OFF_PROBES,
                             copy_labelled_memory, digest, generate)
from train_identity import TARGET

ROUTER_CORE_TRAIN = [row for row in CORE_TRAIN if row[0] != "写一段秋天的散文。"]
IDENTITY_REGRESSION_POSITIVE = ("告诉我你的名字和开发者。", TARGET)


def publish_verified_candidate():
    """Resume publication after backup cleanup failed, without training again."""
    report_path = Path("research/identity_router_repair.json")
    report = json.loads(report_path.read_text())
    assert not report["promoted"] and report["checks"] and all(report["checks"].values())
    config = json.loads((default_home() / "settings.json").read_text())
    production = require_storage(config["storage_root"])
    source = production / "modules/app"
    run = require_storage(report["run_directory"])
    candidate = run / "modules/app"
    before = report["production_before"]
    assert before == {name: digest(source / name) for name in before}, "Production changed after verification"
    meta = json.loads((candidate / "router.json").read_text())
    assert meta == report["calibration"]
    assert digest(candidate / "router.safetensors") == meta["sha256"]
    assert meta["expert_id"] == report["expert_id"] and meta["expert_sha256"] == report["expert_sha256"]
    added_ids = report.get("appended_route_sample_ids", report.get("appended_core_sample_ids"))
    added = [row for row in Memory(run).samples() if row["id"] in added_ids]
    assert len(added) == len(added_ids) == 18
    report["production_backup"] = save_training_backup(source, meta["identity"], "calibrate")
    publication = ["router.safetensors", "router.json"]
    previous = {name: (source / name).read_bytes() for name in publication}
    memory = Memory(production)
    try:
        with memory.lock, memory.connect() as db:
            for row in added:
                db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))
        for name in publication:
            target = source / name
            temporary = target.with_name(target.name + ".repair-partial")
            shutil.copyfile(candidate / name, temporary)
            temporary.replace(target)
    except Exception:
        for name, data in previous.items():
            (source / name).write_bytes(data)
        with memory.lock, memory.connect() as db:
            db.executemany("DELETE FROM samples WHERE id=?", [(sample_id,) for sample_id in added_ids])
        raise
    report["previous_publication_error"] = report.pop("error", None)
    report.pop("failure", None)
    report.update(promoted=True, head_promoted=True, router_sha256=meta["sha256"],
                  appended_route_sample_ids=added_ids,
                  production_after={name: digest(source / name) for name in before},
                  publication_resumed_without_training=True,
                  original_failure_report="research/identity_router_repair-before-backup-cleanup-fix.json",
                  memory_measurement="psutil process RSS and Metal driver allocated byte snapshots in initial_status; not a peak",
                  sample_counts={"training": meta["samples"], "calibration": meta["validation_samples"],
                                 "training_core": 33, "training_expert": 6, "training_teacher": 24,
                                 "calibration_core": 17, "calibration_expert": 2, "calibration_teacher": 12})
    report.pop("appended_core_sample_ids", None)
    assert report["production_after"]["router.safetensors"] == meta["sha256"]
    assert all(report["production_after"][name] == before[name] for name in before if name not in publication)
    private_json(run / "published-result.json", report)
    private_json(report_path, report)
    print(json.dumps({"promoted": True, "router_sha256": meta["sha256"], "expert_id": report["expert_id"],
                      "expert_sha256": report["expert_sha256"], "sample_counts": report["sample_counts"],
                      "checks": report["checks"], "run": str(run)}, ensure_ascii=False))


def main():
    started = time.perf_counter()
    config = json.loads((default_home() / "settings.json").read_text())
    local = {key: config[key] for key in ("model_path", "storage_root", "local_thinking",
                                         "max_new_tokens", "context_tokens", "temperature")}
    production = require_storage(local["storage_root"])
    source = production / "modules/app"
    active = json.loads((source / "active.json").read_text())
    original_meta = json.loads((source / "router.json").read_text())
    expert_meta = next(source.glob(f"layer-*/{active['id']}.json"))
    names = ["active.json", "router.json", "router.safetensors",
             expert_meta.relative_to(source).as_posix(),
             expert_meta.with_suffix(".safetensors").relative_to(source).as_posix()]
    before = {name: digest(source / name) for name in names}
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run = production / "experiments" / ("identity-router-repair-" + run_id)
    run.mkdir(parents=True, exist_ok=False)
    report = {"scope": "ordinary-task routing false positives only; original identity expert unchanged",
              "promoted": False, "head_promoted": False, "expert_promoted": False, "thinking_repaired": False,
              "expert_id": active["id"], "expert_sha256": original_meta["expert_sha256"],
              "seed": 42, "no_teacher_calls": True, "run_directory": str(run), "checks": {},
              "production_before": before, "core_training": ROUTER_CORE_TRAIN,
              "identity_regression_positive": IDENTITY_REGRESSION_POSITIVE,
              "core_calibration": CORE_CALIBRATION, "core_held_out": CORE_HELD_OUT,
              "limitation": "Related prompt families and repeated development probes; not an independent general benchmark."}

    def record():
        report["elapsed_seconds"] = time.perf_counter() - started
        private_json(run / "result.json", report)
        private_json(Path("research/identity_router_repair.json"), report)

    try:
        with tempfile.TemporaryDirectory(prefix="identity-router-home-") as home:
            copied = run / "modules/app"
            for name in names:
                target = copied / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            settings = Settings(home)
            settings.data.update(local | {"storage_root": str(run)})
            runtime = Runtime(settings, None)
            torch.manual_seed(42)
            runtime.load()
            runtime.memory = copy_labelled_memory(production, run, original_meta)
            added_ids = []
            for purpose, rows in (("training", ROUTER_CORE_TRAIN), ("validation", CORE_CALIBRATION)):
                for prompt, answer in rows:
                    added_ids.append(runtime.memory.add_sample(prompt, answer, source="synthetic", purpose=purpose,
                        approved=True, training_allowed=True, provenance={"route_action": "local_core",
                        "route_label_basis": "ordinary task; identity expert unnecessary", "run": run_id,
                        "scope": "routing label only; answer not used to train the expert"}))
            added_ids.append(runtime.memory.add_sample(*IDENTITY_REGRESSION_POSITIVE,
                source="synthetic", purpose="training", approved=True, training_allowed=True,
                provenance={"route_action": "neural_expert", "run": run_id,
                            "route_label_basis": "explicit self-identity question; discovered old regression",
                            "scope": "development regression repair; never an independent final identity probe"}))
            assert len(runtime.memory.samples("training")) == 63
            assert len(runtime.memory.samples("validation")) == 31
            report.update(dtype=str(runtime.dtype), device=runtime.device,
                          initial_status=runtime.status(), appended_route_sample_ids=added_ids)
            report["calibration"] = runtime.calibrate_router()
            record()
            if not report["calibration"]["calibrated"]:
                report["failure"] = "Existing calibration criteria rejected ordinary-task repair"
                return
            report["ordinary_generations"] = [generate(runtime, prompt, maximum=96) for prompt in CORE_HELD_OUT]
            report["ordinary_regressions"] = [generate(runtime, prompt, maximum=64) for prompt in
                ("讲个笑话", "今天怎么安排工作？", "把 Hello 翻译成中文，只输出翻译。",
                 "简要介绍一下 Qwen 模型。", "Meta 发布的 Llama 系列模型是什么？")]
            report["history_generations"] = [generate(runtime, prompt, maximum=64, history=True) for prompt in
                                             ("讲个笑话", "今天怎么安排工作？", "计算9+8")]
            report["identity_generations"] = [generate(runtime, prompt, maximum=128) for prompt in OFF_PROBES]
            # Preserve one actual on-mode failure, so the scoped repair cannot
            # accidentally appear to fix thinking by fixing routing alone.
            report["thinking_generation"] = generate(runtime, "你是谁", "medium")
            teacher = json.loads(Path("research/teacher_router.json").read_text())
            report["old_teacher_held_out"] = [{"prompt": row["prompt"], "expected_action": row["expected_action"],
                "decision": runtime.choose_route([{"role": "user", "content": row["prompt"]}])}
                for row in teacher["held_out"]["decisions"]]
            ordinary = report["ordinary_generations"] + report["ordinary_regressions"] + report["history_generations"]
            report["checks"] = {
                "ordinary_route_and_no_brand_fragment": all(row["decision"].get("action") == "local_core" and
                    not row["brand_fragment_present"] for row in ordinary),
                "off_identity_exact_complete_once": all(row["identity_exact"] and row["usage"]["finish_reason"] == "stop"
                    and row["capability_sentence_occurrences"] == 1 for row in report["identity_generations"]),
                "old_teacher_held_out_routes_preserved": all(row["decision"].get("action") == row["expected_action"]
                    for row in report["old_teacher_held_out"]),
                "expert_checkpoint_unchanged": all(digest(runtime.store / name) == before[name]
                    for name in names if "layer-" in name)}
            record()
            if not all(report["checks"].values()):
                report["failure"] = "Ordinary/identity/teacher checks refused head publication"
                return
            assert before == {name: digest(source / name) for name in names}, "Production state changed"
            report["production_backup"] = save_training_backup(source, runtime.identity, "calibrate")
            publication = ["router.safetensors", "router.json"]
            previous = {name: (source / name).read_bytes() for name in publication}
            memory = Memory(production)
            added = [row for row in runtime.memory.samples() if row["id"] in added_ids]
            try:
                with memory.lock, memory.connect() as db:
                    for row in added:
                        db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))
                for name in publication:
                    target = source / name
                    temporary = target.with_name(target.name + ".repair-partial")
                    shutil.copyfile(runtime.store / name, temporary)
                    temporary.replace(target)
            except Exception:
                for name, data in previous.items():
                    (source / name).write_bytes(data)
                with memory.lock, memory.connect() as db:
                    db.executemany("DELETE FROM samples WHERE id=?", [(sample_id,) for sample_id in added_ids])
                raise
            report.update(promoted=True, head_promoted=True, router_sha256=runtime.router_meta["sha256"],
                          production_after={name: digest(source / name) for name in names}, final_status=runtime.status())
            del runtime
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        record()
        print(json.dumps({"promoted": report["promoted"], "scope": report["scope"],
                          "checks": report["checks"], "failure": report.get("failure")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-verified", action="store_true")
    if parser.parse_args().publish_verified:
        publish_verified_candidate()
    else:
        main()
