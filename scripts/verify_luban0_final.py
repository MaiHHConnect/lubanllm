"""Evaluate one frozen core on the preregistered final set, then verify recovery.

Run only after the main coordinator stops other model processes. Final outputs
never enter training. A failed final set retires this exact candidate; it must
not be used to tune a replacement against the same final contexts.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import tempfile
import time
import uuid

from luban.generation_checks import validate_generation_check
from luban.settings import Settings, default_home, private_json, require_storage

REPOSITORY = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_final_rows(raw_rows):
    """Adapt the original raw protocol locally; never rewrite its source file."""
    if not isinstance(raw_rows, list) or len(raw_rows) != 12:
        raise ValueError("The frozen final protocol must contain exactly 12 cases")
    rows, identifiers = [], set()
    for raw in raw_rows:
        if not isinstance(raw, dict) or raw.get("split", raw.get("purpose")) != "final":
            raise ValueError("Final cases must explicitly belong to the final split")
        provenance = raw.get("provenance", {})
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        if not isinstance(provenance, dict):
            raise ValueError("Final provenance must be an object")
        identifier, prompt, answer = raw.get("id"), raw.get("prompt"), raw.get("answer")
        if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                or not isinstance(prompt, str) or not prompt.strip() or not isinstance(answer, str)):
            raise ValueError("Final case identifiers, prompts or authored answers are invalid")
        identifiers.add(identifier)
        messages = provenance.get("messages") or raw.get("messages") or [{"role": "user", "content": prompt}]
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}
                       or not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages)
                or messages[-1].get("role") != "user"):
            raise ValueError("Final system/history/user contexts are invalid")
        check = validate_generation_check(provenance.get("generation_check", raw.get("generation_check")),
                                          default_expected=answer)
        rows.append({"id": identifier, "prompt": prompt, "answer": answer, "purpose": "final",
                     "source": raw.get("source", "human"), "approved": 1, "training_allowed": False,
                     "provenance": json.dumps({**provenance, "messages": messages,
                                                "generation_check": check}, ensure_ascii=False)})
    return rows


def write_new(path, data):
    with Path(path).open("x") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def tensor_digest(tensor):
    import torch
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def branch_off_generation(runtime, rows):
    from luban.evolution import _generation
    expert = runtime.branch.expert
    runtime.branch.expert = None
    try:
        results = _generation(runtime, rows)
        if any(result["branch_enabled"] for result in results):
            raise RuntimeError("A final evaluation unexpectedly used a learned branch")
        return results
    finally:
        runtime.branch.expert = expert


def retire(runtime, previous_version):
    from luban.evolution import restore_core
    try:
        return restore_core(runtime, previous_version)
    except Exception:
        # A failed full reload may leave no usable model. With an original-base
        # predecessor, removing the pointer still restores the next app load.
        if previous_version is not None:
            raise
        pointer = Path(runtime.store) / "active-core.json"
        if pointer.is_symlink():
            raise ValueError("Refusing a symlink active-core pointer")
        pointer.unlink(missing_ok=True)
        runtime.active_core = None
        return {"restored": True, "version": None, "method": "original base pointer restored; full reload required"}


def verify_recovery(runtime, expected_core, rows, original_results, report):
    import torch
    from luban.backups import save_training_backup, restore_training_backup
    from luban.evolution import restore_core, _target
    before = tensor_digest(_target(runtime)[1])
    original = tensor_digest(runtime._base_core_weight)
    backup = save_training_backup(runtime.store, runtime.identity, "learn")
    record = {"backup_id": backup, "scope": "real recovery smoke; backup API kind learn, no training",
              "core_tensor_sha256_before": before, "original_tensor_sha256": original}
    report["recovery"] = record
    restore_core(runtime, None)
    record["original_restoration"] = {"version": runtime.active_core,
                                       "tensor_sha256": tensor_digest(_target(runtime)[1])}
    if runtime.active_core is not None or record["original_restoration"]["tensor_sha256"] != original or before == original:
        raise RuntimeError("The genuine original-core restoration did not match its tensor snapshot")
    restore_training_backup(runtime.store, runtime.identity, backup)
    # Drop the retained decoder reference before loading the same full backbone.
    # The old model itself is released by Runtime.load().
    runtime.decoder = None
    gc.collect()
    if runtime.device == "mps":
        torch.mps.empty_cache()
    runtime.load()
    after = tensor_digest(_target(runtime)[1])
    record.update(core_version_after=runtime.active_core, core_tensor_sha256_after=after,
                  original_tensor_sha256_after_reload=tensor_digest(runtime._base_core_weight))
    if runtime.active_core != expected_core or after != before or record["original_tensor_sha256_after_reload"] != original:
        raise RuntimeError("Backup restoration/full reload changed the core tensor or original snapshot")
    identity = [row for row in rows if json.loads(row["provenance"])["generation_check"]["type"] == "exact"][:2]
    if len(identity) != 2:
        raise ValueError("Recovery needs two frozen final identity cases")
    repeated = branch_off_generation(runtime, identity)
    previous = {row["id"]: row for row in original_results}
    record["identity_replay"] = repeated
    record["identity_replay_unchanged"] = all(row["passed"] and row["answer"] == previous[row["id"]]["answer"]
                                               and row["complete"] == previous[row["id"]]["complete"] for row in repeated)
    if not record["identity_replay_unchanged"]:
        raise RuntimeError("The two previously unseen identity answers changed after recovery")
    record["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-core", required=True, help="Exact core-12hex candidate; never infer the active version")
    parser.add_argument("--dataset-report", type=Path, default=REPOSITORY / "research/luban0_v2_replay.json")
    parser.add_argument("--model", help="Default: current public production model_path")
    parser.add_argument("--storage", help="Default: current public production storage_root")
    parser.add_argument("--print-plan", action="store_true", help="Inspect protocol without loading models or writing files")
    args = parser.parse_args()
    if not re.fullmatch(r"core-[a-f0-9]{12}", args.expected_core):
        parser.error("--expected-core must be core- followed by exactly 12 lowercase hex characters")
    preparation = json.loads(args.dataset_report.read_text())
    source = Path(preparation["final_test_path"])
    original_bytes = source.read_bytes()
    final_sha256 = hashlib.sha256(original_bytes).hexdigest()
    rows = normalized_final_rows(json.loads(original_bytes))
    public_config = json.loads((default_home() / "settings.json").read_text())
    model = args.model or public_config["model_path"]
    storage = args.storage or public_config["storage_root"]
    protocol = {"expected_core": args.expected_core, "base_identity": preparation["base_identity"],
        "final_test_path": str(source), "final_test_sha256": final_sha256,
        "preparation_report_sha256": digest(args.dataset_report), "preparation_protocol_sha256": preparation["protocol_sha256"],
        "script_sha256": digest(__file__), "cases": rows, "seed": 42,
        "generation": {"branch_enabled": False, "thinking": "off", "do_sample": False,
                       "budgets": "frozen generation_check.max_new_tokens per case", "EOS_completion_required": True},
        "acceptance": "all 12 frozen cases pass; genuine backup/original restore/full reload/tensor SHA and two identity answers remain unchanged",
        "on_failure": "retire only the confirmed expected candidate to its recorded previous core; no training or final-check changes",
        "final_scope": preparation["final_scope"], "creative_scope": preparation["creative_scope"],
        "teacher_calls": 0, "training_performed": False, "production_model": model, "storage_root": storage}
    if args.print_plan:
        print(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False))
        return
    root = require_storage(storage)
    run = root / "experiments" / ("luban0-v1-frozen-final-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    # The byte-exact source copy and protocol precede Runtime/model creation.
    with (run / "final-test-original.json").open("xb") as snapshot:
        snapshot.write(original_bytes)
    write_new(run / "protocol.json", protocol)
    summary_path = REPOSITORY / "research/luban0_v1_final.json"
    report = {"expected_core": args.expected_core, "passed": False, "evaluation_started": False,
              "final_test_sha256": final_sha256, "protocol_sha256": digest(run / "protocol.json"),
              "protocol_path": str(run / "protocol.json"), "run_directory": str(run), "generation": [],
              "teacher_calls": 0, "training_performed": False, "hardware": platform.platform(),
              "final_scope": preparation["final_scope"], "creative_scope": preparation["creative_scope"]}
    started = time.perf_counter()
    runtime, candidate_confirmed, previous_version = None, False, None
    publish_summary = True
    try:
        if summary_path.exists():
            prior = json.loads(summary_path.read_text())
            if prior.get("evaluation_started") and prior.get("final_test_sha256") == final_sha256:
                if prior.get("expected_core") != args.expected_core or not prior.get("passed"):
                    # Keep the durable refusal guard. An attempted forbidden
                    # reuse must not replace it with evaluation_started=False.
                    publish_summary = False
                    raise RuntimeError("This final set already evaluated a candidate; do not tune another candidate against the same final contexts")
                report["final_scope"] = "repeated verification of the same frozen core; not an independent new final evaluation"
        import torch
        from luban.evolution import _normalized_messages, _target
        from luban.memory import Memory
        from luban.runtime import Runtime
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(prefix="luban0-frozen-final-settings-") as home:
            settings = Settings(home)
            settings.update({"model_path": model, "storage_root": storage, "local_thinking": "off", "temperature": 0})
            runtime = Runtime(settings, Memory(root))
            runtime.load()
            if runtime.model.config.model_type != "qwen3_5" or runtime.identity != preparation["base_identity"] or runtime.active_core != args.expected_core:
                raise RuntimeError("The loaded production Qwen3.5 core does not match the exact frozen candidate and dataset base")
            candidate_confirmed = True
            metadata = json.loads((runtime.store / (args.expected_core + ".json")).read_text())
            previous_version = metadata.get("previous_version")
            report.update(previous_version=previous_version, identity=runtime.identity,
                device=runtime.device, dtype=str(runtime.dtype), torch_version=torch.__version__,
                core_tensor_sha256=tensor_digest(_target(runtime)[1]),
                initial_runtime=runtime.status(), memory_measurement="process RSS and MPS allocator snapshots, not physical memory peak")
            material_contexts = {_normalized_messages(row) for row in runtime.memory.samples()}
            if any(_normalized_messages(row) in material_contexts for row in rows):
                raise RuntimeError("A supposedly frozen final context already exists in Memory; final leakage rejected")
            if digest(source) != final_sha256:
                raise RuntimeError("The original final protocol changed after preregistration")
            report["evaluation_started"] = True
            private_json(run / "result.json", report)
            # Persist the guard before the first final generation, including
            # abrupt process termination that cannot execute this finally.
            private_json(summary_path, report)
            for row in rows:
                result = branch_off_generation(runtime, [row])[0]
                report["generation"].append(result)
                private_json(run / "result.json", report)
                print(json.dumps({"case": row["id"], "passed": result["passed"], "complete": result["complete"]}, ensure_ascii=False), flush=True)
            report["final_generation_passed"] = all(result["passed"] for result in report["generation"])
            if not report["final_generation_passed"]:
                raise RuntimeError("The frozen final generation set failed; retire this candidate without tuning on these cases")
            verify_recovery(runtime, args.expected_core, rows, report["generation"], report)
            if digest(source) != final_sha256:
                raise RuntimeError("The original final protocol changed during evaluation")
            report.update(passed=True, final_runtime=runtime.status())
            runtime.memory.event("core_final_validation", {"version": args.expected_core, "passed": True,
                "protocol_sha256": report["protocol_sha256"], "result_path": str(run / "result.json")})
    except BaseException as error:
        report.update(passed=False, error=str(error) or type(error).__name__)
        if candidate_confirmed:
            try:
                report["retirement"] = retire(runtime, previous_version)
                runtime.memory.event("core_final_validation", {"version": args.expected_core, "passed": False,
                    "previous_version": previous_version, "protocol_sha256": report["protocol_sha256"],
                    "result_path": str(run / "result.json"), "retired": True})
            except Exception as retirement_error:
                report["retirement_error"] = str(retirement_error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        private_json(run / "result.json", report)
        if publish_summary:
            private_json(summary_path, report)
        print(json.dumps({"passed": report["passed"], "run_directory": str(run), "error": report.get("error"),
                          "retirement": report.get("retirement"), "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
