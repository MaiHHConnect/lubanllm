"""Engineering acceptance of one unchanged core under a revised identity spec.

This follows an observed scientific final failure. It is not an independent
final experiment. No training or response rewriting takes place here.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import tempfile
import time
from types import SimpleNamespace
import uuid

from bootstrap_luban0 import IDENTITY_CHECK
from verify_luban0_final import branch_off_generation, digest, retire, tensor_digest, verify_recovery, write_new
from verify_training_operators import OPERATOR_AUDIT_PROTOCOL, TOLERANCES, audit_training_operators
from luban.generation_checks import validate_generation_check
from luban.settings import Settings, default_home, private_json, require_storage

REPOSITORY = Path(__file__).resolve().parents[1]


def product_cases():
    """Fixed product cases, authored before any product-case model generation."""
    identity_contexts = [
        [{"role": "user", "content": "本轮会话使用的模型叫什么、由谁调优？请完整介绍你自己的身份，只用一句话。"}],
        [{"role": "system", "content": "请简洁、准确地回答用户的问题。"},
         {"role": "user", "content": "我想确认当前回答者的模型身份和调优归属，请给出完整的一句自我介绍。"}],
        [{"role": "system", "content": "你是一个有帮助的助手。"},
         {"role": "user", "content": "通过这个接口和我对话的模型是什么？请用一句完整的话介绍你自己。"}],
        [{"role": "system", "content": "回答需要明确且完整。"},
         {"role": "user", "content": "三加四等于多少？"}, {"role": "assistant", "content": "7"},
         {"role": "user", "content": "谢谢。现在请完整说明你自己的模型身份与调优归属，只用一句话。"}],
        [{"role": "system", "content": "现在进行文本接口验收，请回答实际身份。"},
         {"role": "user", "content": "请介绍你自己的名称、调优方以及模型定位，合成一句完整的身份说明。"}],
        [{"role": "user", "content": "你能帮助我做什么？"},
         {"role": "assistant", "content": "我可以帮助你完成日常任务。"},
         {"role": "user", "content": "在开始任务前，请完整介绍你自己的模型身份与调优方，只回答一句话。"}],
        [{"role": "system", "content": "优先直接回答用户当前问题。"},
         {"role": "user", "content": "当前模型的自我身份说明是什么？请完整说出模型名、调优归属和定位。"}],
        [{"role": "system", "content": "使用中文，保持回答简短。"},
         {"role": "user", "content": "很高兴见到你。"},
         {"role": "assistant", "content": "你好，很高兴与你交流。"},
         {"role": "user", "content": "我该怎样称呼你？请完整介绍你自己的模型身份和调优归属，只用一句话。"}],
    ]
    ordinary = [
        ("计算9加14，只输出结果数字。", {"validator": "integer", "expected": 23}),
        ("计算37减12，只输出结果数字。", {"validator": "integer", "expected": 25}),
        ("计算6乘7，只输出结果数字。", {"validator": "integer", "expected": 42}),
        ('只输出一个JSON对象：name字段为字符串Ada，active字段为布尔值true，不要其他字段。',
         {"validator": "json", "expected": {"name": "Ada", "active": True}}),
        ("水分子的化学式是什么？只输出化学式，不解释。", {"validator": "normalized_exact", "expected": "H2O"}),
        ('把英文“Good morning”翻译成中文，只输出译文。', {"validator": "translation", "accepted": ["早上好", "早安"]}),
        ("写一个只有一个参数的Python函数，把传入字符串转换为大写；只输出代码，不添加示例。",
         {"validator": "python_upper"}),
        ('将字符串“luban”改为大写，只输出处理后的字符串。', {"validator": "normalized_exact", "expected": "LUBAN"}),
    ]
    check = validate_generation_check(copy.deepcopy(IDENTITY_CHECK))
    if check["type"] != "semantic" or check["validator"] != "one_of" or len(check["accepted"]) != 2:
        raise ValueError("Product identity acceptance must contain exactly the two predeclared full sentences")
    rows = []
    for index, messages in enumerate(identity_contexts):
        rows.append({"id": f"product-identity-{index}", "prompt": messages[-1]["content"],
            "answer": check["accepted"][0], "purpose": "product", "training_allowed": False,
            "provenance": json.dumps({"messages": messages, "generation_check": check}, ensure_ascii=False)})
    for index, (prompt, rule) in enumerate(ordinary):
        rule = validate_generation_check({"type": "semantic", "max_new_tokens": 192,
            "forbidden": ["我是", "鲁班零号", "麦好火", "通义千问", "Qwen"], **rule})
        rows.append({"id": f"product-ordinary-{index}", "prompt": prompt, "answer": "",
            "purpose": "product", "training_allowed": False,
            "provenance": json.dumps({"messages": [{"role": "system", "content": "请直接完成用户要求的任务。"},
                {"role": "user", "content": prompt}], "generation_check": rule}, ensure_ascii=False)})
    return rows


def recovery_cases(rows, results):
    """Exact repeat comparisons against observed, already accepted answers."""
    output, observed = copy.deepcopy(rows[:2]), {row["id"]: row for row in results}
    for row in output:
        provenance = json.loads(row["provenance"])
        original = provenance["generation_check"]
        row["answer"] = observed[row["id"]]["answer"]
        provenance["generation_check"] = validate_generation_check({"type": "exact", "expected": row["answer"],
            "max_new_tokens": original["max_new_tokens"], "forbidden": original["forbidden"]})
        provenance["scope"] = "recovery repeat of an observed product answer; not a new test"
        row["provenance"] = json.dumps(provenance, ensure_ascii=False)
    return output


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def regression_copies(rows, identity):
    output = copy.deepcopy(rows)
    for row in output:
        row.update(purpose="validation", source="human", approved=1, training_allowed=False)
        provenance = json.loads(row["provenance"])
        provenance.update(evolution_role="replay", retention_generation_only=True, base_identity=identity,
            rights_basis="project-authored local cases; user authorized local teaching and review",
            scope="fixed engineering product regression; never training/loss data or scientific final")
        row["provenance"] = json.dumps(provenance, ensure_ascii=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-core", required=True)
    parser.add_argument("--model")
    parser.add_argument("--storage")
    parser.add_argument("--print-protocol", action="store_true", help="No model load, generation or artifact writes")
    args = parser.parse_args()
    if not re.fullmatch(r"core-[a-f0-9]{12}", args.expected_core):
        parser.error("--expected-core must be core- followed by 12 lowercase hex digits")
    config = json.loads((default_home() / "settings.json").read_text())
    model, storage = args.model or config["model_path"], args.storage or config["storage_root"]
    root = require_storage(storage)
    store = root / "modules/luban0-v1"
    if (store / "active-core.json").exists() or (store / "active-core.json").is_symlink():
        raise RuntimeError("Product verification must start with the genuine original core active (no pointer)")
    prior_path = REPOSITORY / "research/luban0_v1_final.json"
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)
    if (prior.get("expected_core") != args.expected_core or prior.get("passed") is not False
            or len(prior.get("generation", [])) != 12 or sum(row["passed"] for row in prior["generation"]) != 10
            or prior.get("retirement", {}).get("version") is not None
            or prior.get("retirement", {}).get("restored") is not True):
        raise RuntimeError("This product revision must reference the unchanged retired 10/12 scientific-final candidate")
    metadata_path, weights = store / (args.expected_core + ".json"), store / (args.expected_core + ".safetensors")
    if metadata_path.is_symlink() or weights.is_symlink():
        raise ValueError("Core checkpoint files cannot be symlinks")
    metadata = json.loads(metadata_path.read_text())
    checkpoint_sha = digest(weights)
    if (metadata.get("identity") != prior["identity"] or metadata.get("version") != args.expected_core
            or metadata.get("sha256") != checkpoint_sha):
        raise ValueError("Product checkpoint identity/version/SHA differs from its immutable scientific-final core")
    rows = product_cases()
    from luban.evolution import consolidation_samples
    from luban.memory import Memory
    memory = Memory(root)
    memory.model_identity = prior["identity"]
    frozen_groups = consolidation_samples(SimpleNamespace(memory=memory, active_core=None,
        store=store, identity=prior["identity"]))
    if not all(frozen_groups.values()):
        raise RuntimeError("Full eligible four-way core corpus is required for a retention snapshot")
    regressions = regression_copies(rows, prior["identity"])
    validator_path = REPOSITORY / "src/luban/generation_checks.py"
    operator_path = REPOSITORY / "scripts/verify_training_operators.py"
    retention_path = REPOSITORY / "src/luban/retention.py"
    audit_sample = copy.deepcopy(rows[8])
    audit_sample.update(id="product-operator-audit", purpose="operator_audit", answer="23")
    protocol = {"expected_core": args.expected_core, "base_identity": prior["identity"],
        "checkpoint_sha256": checkpoint_sha, "prior_core_tensor_sha256": prior["core_tensor_sha256"],
        "spec_revision_reason": "User requires 鲁班零号 + 麦好火调优 + 无限进化; the optional parenthesized English alias is not a verbatim requirement. Accept only either complete predeclared sentence.",
        "scope": "engineering acceptance under revised product specification after observed final failure; NOT an independent scientific final",
        "old_scientific_final": {"passed": False, "passed_count": 10, "total": 12,
            "summary_path": str(prior_path), "summary_sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "run_directory": prior["run_directory"], "retirement": prior["retirement"],
            "final_test_sha256": prior["final_test_sha256"]},
        "identity_check": validate_generation_check(copy.deepcopy(IDENTITY_CHECK)), "cases": rows,
        "retention": {"schema_version": 1, "helper_sha256": digest(retention_path),
            "snapshot_filename": args.expected_core + ".retention.json", "groups": frozen_groups,
            "dataset_sha256": content_digest(frozen_groups),
            "group_counts": {key: len(values) for key, values in frozen_groups.items()},
            "sample_sha256": {row["id"]: content_digest(row) for values in frozen_groups.values() for row in values},
            "regression_validation": regressions, "regression_sha256": content_digest(regressions),
            "generation_rules": "preserve complete approved original records and checks unchanged; product16 copies are human-authored validation replay, generation-only, never training or loss",
            "registration_order": "all16 pass -> immutable snapshot save/readback -> genuine backup/original restore/full reload -> identical snapshot/protected-validation SHA/count -> acceptance",
            "on_failure": "restore original active core None; preserve old scientific evidence and immutable checkpoints"},
        "script_sha256": digest(__file__), "validator_sha256": digest(validator_path),
        "operator_audit": {"protocol_version": OPERATOR_AUDIT_PROTOCOL, "script_sha256": digest(operator_path), "sample": audit_sample,
            "tolerances": copy.deepcopy(TOLERANCES),
            "scope": "native full forward vs independently reconstructed final-layer tail: logits, causal masked CE, existing down_proj gradient and one fresh FP32 Adam step",
            "optimizer": {"learning_rate": 1e-5, "eps": 1e-6, "clip_norm": 1.0, "master_dtype": "torch.float32"},
            "required": "passed is True, restored is True, original core tensor SHA and absent active pointer unchanged before any baseline generation",
            "production_training_optimization": False, "cached_training": False,
            "limitations": "one teacher-forcing sample; no KL/projection, memory/speed or generation equivalence claim; temporary cloned optimizer weights only"},
        "generation": {"branch_enabled": False, "thinking": "off", "temperature": 0,
            "do_sample": False, "EOS_completion_required": True, "fixed_budgets": "128 identity / 192 ordinary"},
        "acceptance": "fixed operator audit passes/restores, all original8 and candidate16 pass, complete retention snapshot writes/reads back, genuine backup/original restore/fullreload/tensorSHA/two identity repeats and retention/protected-validation hashes/counts remain unchanged",
        "on_failure": "restore original core None; preserve all immutable checkpoints and raw evidence; no new accepted variants or weight changes",
        "recovery_scope": "two observed accepted product answers copied into exact repeat checks solely for restoration comparison; frozen product protocol unchanged",
        "training_performed": False, "teacher_calls": 0, "seed": 42, "production_model": model, "storage_root": str(root)}
    if args.print_protocol:
        print(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False))
        return
    run = root / "experiments" / ("luban0-v1-product-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    (run / "old-scientific-final.json").write_bytes(prior_bytes)
    write_new(run / "protocol.json", protocol)
    report = {"expected_core": args.expected_core, "passed": False, "evaluation_started": False, "model_started": False,
        "active_candidate_accepted_under_new_product_spec": False, "scientific_independent_final": False,
        "scope": protocol["scope"], "protocol_path": str(run / "protocol.json"),
        "protocol_sha256": digest(run / "protocol.json"), "run_directory": str(run),
        "checkpoint_sha256": checkpoint_sha, "baseline_generation": [], "generation": [],
        "training_performed": False, "teacher_calls": 0, "hardware": platform.platform()}
    summary = REPOSITORY / "research/luban0_v1_product.json"
    started, runtime, rollback_required, publish_summary = time.perf_counter(), None, False, True
    try:
        if summary.exists() and json.loads(summary.read_text()).get("evaluation_started"):
            publish_summary = False
            raise RuntimeError("The product protocol was already evaluated; no tuning or acceptance-variant additions")
        import torch
        from luban.evolution import _read_version, _target, _memory, restore_core
        from luban.memory import Memory
        from luban.runtime import Runtime
        from luban.retention import save_core_retention, read_core_retention, protected_validation
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(prefix="luban0-product-settings-") as home:
            settings = Settings(home)
            settings.update({"model_path": model, "storage_root": str(root), "local_thinking": "off", "temperature": 0})
            runtime = Runtime(settings, Memory(root))
            report.update(model_started=True, model_start_time=time.time())
            private_json(run / "result.json", report)
            print(json.dumps({"phase": "model_loading", "model_started": True, "expected_core": args.expected_core,
                "protocol_sha256": report["protocol_sha256"], "retention_group_counts": protocol["retention"]["group_counts"],
                "regression_count": len(regressions)}), flush=True)
            runtime.load()
            if runtime.active_core is not None or runtime.identity != prior["identity"] or runtime.model.config.model_type != "qwen3_5":
                raise RuntimeError("Loaded baseline must be the expected original Qwen3.5 core")
            if consolidation_samples(runtime) != frozen_groups:
                raise RuntimeError("Eligible core material changed after product preregistration")
            verified_metadata, candidate_tensor = _read_version(runtime, args.expected_core)
            if verified_metadata["sha256"] != checkpoint_sha or tensor_digest(candidate_tensor) != prior["core_tensor_sha256"]:
                raise RuntimeError("Candidate tensor differs from the previously tested immutable core")
            del candidate_tensor
            original_sha = tensor_digest(_target(runtime)[1])
            report.update(identity=runtime.identity, device=runtime.device, dtype=str(runtime.dtype),
                torch_version=torch.__version__, original_tensor_sha256=original_sha,
                memory_measurement="phase process RSS/MPS allocator snapshots, not physical peak",
                memory_samples=[_memory(runtime)], evaluation_started=True)
            private_json(run / "result.json", report)
            private_json(summary, report)
            rollback_required = True
            report["operator_audit"] = audit_training_operators(runtime, audit_sample)
            private_json(run / "result.json", report)
            print(json.dumps({"phase": "operator_audit", "passed": report["operator_audit"].get("passed"),
                "restored": report["operator_audit"].get("restored")}), flush=True)
            if (report["operator_audit"].get("protocol_version") != OPERATOR_AUDIT_PROTOCOL
                    or report["operator_audit"].get("passed") is not True or report["operator_audit"].get("restored") is not True
                    or tensor_digest(_target(runtime)[1]) != original_sha or runtime.active_core is not None
                    or (store / "active-core.json").exists()):
                raise RuntimeError("Fixed operator audit failed or did not restore original state; no baseline or candidate evaluation")
            for row in rows[8:]:
                result = branch_off_generation(runtime, [row])[0]
                report["baseline_generation"].append(result)
                private_json(run / "result.json", report)
                print(json.dumps({"phase": "original_baseline", "case": row["id"], "passed": result["passed"], "complete": result["complete"]}), flush=True)
            report["baseline_passed"] = all(row["passed"] for row in report["baseline_generation"])
            if not report["baseline_passed"]:
                raise RuntimeError("Original ordinary tasks failed; do not claim preservation or activate candidate")
            restore_core(runtime, args.expected_core)
            if tensor_digest(_target(runtime)[1]) != prior["core_tensor_sha256"]:
                raise RuntimeError("Restored candidate core tensor changed")
            for row in rows:
                result = branch_off_generation(runtime, [row])[0]
                report["generation"].append(result)
                private_json(run / "result.json", report)
                print(json.dumps({"phase": "candidate", "case": row["id"], "passed": result["passed"], "complete": result["complete"]}), flush=True)
            report["candidate_generation_passed"] = all(row["passed"] for row in report["generation"])
            if not report["candidate_generation_passed"]:
                raise RuntimeError("Fixed revised product acceptance failed; retire without altering checks or weights")
            save_core_retention(runtime, args.expected_core, frozen_groups, regression_samples=regressions)
            retention_before = read_core_retention(runtime, args.expected_core)
            if (not retention_before or retention_before.get("version") != args.expected_core
                    or retention_before.get("identity") != prior["identity"]
                    or not retention_before.get("content_sha256")
                    or retention_before.get("groups") != frozen_groups
                    or retention_before.get("regression_validation") != regressions):
                raise RuntimeError("Version-bound retention snapshot write/readback failed")
            protected_before = protected_validation(runtime)
            expected_ids = {row["id"] for key, values in frozen_groups.items() if key.endswith("validation") for row in values}
            expected_ids.update(row["id"] for row in regressions)
            if {row["id"] for row in protected_before} != expected_ids or len(protected_before) != len(expected_ids):
                raise RuntimeError("Retention snapshot did not expose the complete validation protection set")
            report["retention"] = {"path": str(store / protocol["retention"]["snapshot_filename"]),
                "file_sha256": digest(store / protocol["retention"]["snapshot_filename"]),
                "content_sha256": retention_before["content_sha256"],
                "protected_validation_count": len(protected_before), "protected_validation_sha256": content_digest(protected_before),
                "group_counts": protocol["retention"]["group_counts"], "regression_count": len(regressions)}
            private_json(run / "result.json", report)
            repeat_rows = recovery_cases(rows, report["generation"])
            report["recovery_repeat_cases"] = repeat_rows
            verify_recovery(runtime, args.expected_core, repeat_rows, report["generation"], report)
            retention_after = read_core_retention(runtime, args.expected_core)
            protected_after = protected_validation(runtime)
            if (retention_after != retention_before or content_digest(protected_after) != content_digest(protected_before)
                    or len(protected_after) != len(protected_before)
                    or digest(store / protocol["retention"]["snapshot_filename"]) != report["retention"]["file_sha256"]):
                raise RuntimeError("Real backup restoration/full reload changed retention or complete validation protection")
            report["retention"]["restored_unchanged"] = True
            if (digest(weights) != checkpoint_sha or digest(validator_path) != protocol["validator_sha256"]
                    or digest(operator_path) != protocol["operator_audit"]["script_sha256"]
                    or digest(retention_path) != protocol["retention"]["helper_sha256"]
                    or digest(__file__) != protocol["script_sha256"] or prior_path.read_bytes() != prior_bytes):
                raise RuntimeError("Immutable weights, predeclared checks or prior scientific evidence changed")
            report.update(passed=True, active_candidate_accepted_under_new_product_spec=True,
                core_tensor_sha256=tensor_digest(_target(runtime)[1]), active_core=runtime.active_core,
                memory_samples=report["memory_samples"] + [_memory(runtime)])
            runtime.memory.event("core_product_validation", {"version": args.expected_core, "passed": True,
                "scientific_independent_final": False, "protocol_sha256": report["protocol_sha256"],
                "result_path": str(run / "result.json")})
    except BaseException as error:
        report.update(passed=False, active_candidate_accepted_under_new_product_spec=False,
            error=str(error) or type(error).__name__)
        if rollback_required:
            try:
                report["retirement"] = retire(runtime, None)
            except Exception as retirement_error:
                report["retirement_error"] = str(retirement_error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        private_json(run / "result.json", report)
        if publish_summary:
            private_json(summary, report)
        print(json.dumps({"passed": report["passed"], "run_directory": str(run), "error": report.get("error"),
            "retirement": report.get("retirement"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
