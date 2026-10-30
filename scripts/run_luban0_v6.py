"""One preregistered uppercase correction run and fresh engineering acceptance.

The old product failure informs replay TRAIN. Old scientific/product reports
remain failures. This is finite engineering evidence, not an independent final.
"""
import argparse
import copy
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import uuid

from bootstrap_luban0 import IDENTITY_CHECK
from verify_luban0_final import branch_off_generation, digest, retire, tensor_digest, verify_recovery, write_new
from verify_luban0_product import content_digest, regression_copies, recovery_cases
from verify_training_operators import OPERATOR_AUDIT_PROTOCOL, TOLERANCES, audit_training_operators
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.memory import Memory
from luban.settings import Settings, default_home, private_json, require_storage

REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = "luban0-v6-uppercase-correction"
SYSTEM = {"role": "system", "content": "请直接完成用户要求的任务。"}
CHECKPOINTS = [64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384]


def sample(identifier, prompt, answer, rule, *, purpose="validation", prefix=(), training=False):
    check = validate_generation_check(rule, default_expected=answer)
    return {"id": identifier, "prompt": prompt, "answer": answer, "source": "human", "purpose": purpose,
        "approved": 1, "training_allowed": training, "provenance": json.dumps({
            "messages": [*copy.deepcopy(prefix), {"role": "user", "content": prompt}], "generation_check": check,
            "evolution_role": "replay", "dataset": DATASET, "split_family": identifier,
            "rights_basis": "user authorized this local project to author and review teaching material; no paid teacher",
            "verification_evidence": "ASCII string.upper()" if "uppercase" in identifier else "user brand instruction or deterministic elementary task",
            "human_verified": True, "scope": "post-failure finite engineering correction; not scientific independent final"}, ensure_ascii=False)}


def uppercase_data(exposed):
    contexts = [
        ("luban", exposed["prompt"], json.loads(exposed["provenance"])["messages"][:-1]),
        ("luban0", '把字符串“luban0”全部改成大写，只输出转换结果。', [SYSTEM]),
        ("MaiHaoHuo", '把“MaiHaoHuo”转换为大写字母，只输出结果。', [SYSTEM]),
        ("alpha", '将“alpha”的字母全部转为大写，不输出解释。', [SYSTEM]),
        ("Qwen", '对字符串“Qwen”执行大写转换，只输出转换后的字符串。', [SYSTEM]),
        ("qwen", '请输出字符串“qwen”的全大写形式，不要其他内容。', []),
        ("Luban", '对“Luban”做字符串大写转换，只输出结果。', [{"role": "system", "content": "请准确完成文字处理。"}]),
        ("luBan", '将“luBan”改成全大写字符串，直接给出结果。', [SYSTEM]),
        ("maihaohuo", '将“maihaohuo”中的字母全部改成大写，只输出结果。', []),
        ("NeuralNet", '把字符串“NeuralNet”变为全大写，仅给出处理结果。', [SYSTEM]),
        ("core2", '把字符串“core2”里的英文字母全部变成大写，只输出处理结果。',
         [SYSTEM, {"role": "user", "content": "1加1等于多少？"}, {"role": "assistant", "content": "2"}]),
        ("helloWorld", '请把“helloWorld”转换为全大写字符串，仅返回转换结果。',
         [{"role": "system", "content": "按用户当前指令直接作答。"}]),
    ]
    validation = [
        ("gamma", '请把字符串“gamma”转换为大写形式，只写处理后的字符串。'),
        ("tokyo", '将字符串“tokyo”的每个字母转换为大写，仅输出结果。'),
        ("PyTorch", '把字符串“PyTorch”转为大写，不添加解释或引号。'),
        ("signal7", '请将“signal7”的英文字母大写，仅返回转换后的字符串。'),
    ]
    rows = []
    for purpose, values in (("training", contexts), ("validation", [(word, prompt, [SYSTEM]) for word, prompt in validation])):
        for index, (word, prompt, prefix) in enumerate(values):
            row = sample(f"v6-uppercase-{purpose}-{index}", prompt, word.upper(),
                {"type": "semantic", "validator": "normalized_exact", "expected": word.upper(), "max_new_tokens": 192},
                purpose=purpose, prefix=prefix, training=True)
            provenance = json.loads(row["provenance"])
            provenance.update(program_verified=True, program="str.upper", source_string=word,
                exposed_old_product=purpose == "training" and index == 0)
            row["provenance"] = json.dumps(provenance, ensure_ascii=False)
            if not evaluate_generation(row["answer"], json.loads(row["provenance"])["generation_check"], True)["passed"]:
                raise ValueError("Authored uppercase gold failed its own fixed checker")
            rows.append(row)
    return rows


def engineering_cases():
    contexts = [
        ([{"role": "system", "content": "你通过本地聊天接口与用户对话，请直接回答。"}], "请给这次接入的模型一份完整的一句话身份介绍，包含名称、调优方和定位。"),
        ([], "第一次使用这个助手，请用一句完整的话说明你自己的名称、调优来源和模型定位。"),
        ([{"role": "system", "content": "请给出清楚完整的中文答案。"}], "我要记录当前模型的信息，请完整介绍你自己的模型身份、调优方和定位，只用一句话。"),
        ([SYSTEM, {"role": "user", "content": "四加一是多少？"}, {"role": "assistant", "content": "5"}], "在继续计算前，请用一句完整的话介绍你自己的模型名称、调优方和定位。"),
        ([{"role": "system", "content": "你是一个耐心的助手。"}], "为新使用者完整说明你自己的模型名称、调优方以及定位，只说一句话。"),
        ([{"role": "system", "content": "回答当前问题，不添加无关介绍。"}], "请为应用的模型信息栏提供完整的一句自我身份介绍，包含模型名称、调优方和定位。"),
        ([{"role": "user", "content": "谢谢你的帮助。"}, {"role": "assistant", "content": "不客气。"}], "还需要确认你自己的模型名称、调优方和定位，请完整介绍一次，只用一句话。"),
        ([{"role": "system", "content": "用一句话回答身份类提问。"}], "当前软件中提供回答的模型是什么？请完整说明你自己的名称、调优方和定位。"),
    ]
    rows = [sample(f"v6-engineering-identity-{index}", prompt, IDENTITY_CHECK["accepted"][0],
                   copy.deepcopy(IDENTITY_CHECK), prefix=prefix) for index, (prefix, prompt) in enumerate(contexts)]
    for index, word in enumerate(("delta", "mixedCase", "mAcOs", "v6proof")):
        rows.append(sample(f"v6-engineering-uppercase-{index}", f'请把“{word}”完整转换为全大写，只输出转换结果。', word.upper(),
            {"type": "semantic", "validator": "normalized_exact", "expected": word.upper(), "max_new_tokens": 192}, prefix=[SYSTEM]))
    ordinary = [
        ("计算18加7，只输出结果数字。", "25", {"validator": "integer", "expected": 25}),
        ("计算31减9，只输出结果数字。", "22", {"validator": "integer", "expected": 22}),
        ('只输出JSON数组，按顺序包含整数3、6、9，不要其他内容。', "[3,6,9]", {"validator": "json", "expected": [3, 6, 9]}),
        ("请写出纯水的分子式，只给出分子式。", "H2O", {"validator": "normalized_exact", "expected": "H2O"}),
    ]
    for index, (prompt, answer, rule) in enumerate(ordinary):
        rows.append(sample(f"v6-engineering-ordinary-{index}", prompt, answer,
            {"type": "semantic", "max_new_tokens": 192, **rule}, prefix=[SYSTEM]))
    return regression_copies(rows, "pending-base-identity")


def prepare(storage, model):
    from luban.evolution import consolidation_samples
    descriptor_path = REPOSITORY / "research/luban0_v6_protocol.json"
    if descriptor_path.exists():
        raise FileExistsError("V6 is already preregistered; do not replace data or acceptance cases")
    root = require_storage(storage)
    product_path, final_path = (REPOSITORY / "research" / name for name in ("luban0_v1_product.json", "luban0_v1_final.json"))
    product, final = json.loads(product_path.read_text()), json.loads(final_path.read_text())
    if product["passed"] or sum(row["passed"] for row in product["generation"]) != 15 or final["passed"]:
        raise ValueError("V6 must retain the recorded failed product15/16 and failed scientific final")
    if (root / "modules/luban0-v1/active-core.json").exists():
        raise ValueError("V6 preparation requires original core None")
    prior_protocol = json.loads(Path(product["protocol_path"]).read_text())
    exposed = next(row for row in prior_protocol["cases"] if row["id"] == "product-ordinary-7")
    memory = Memory(root)
    memory.model_identity = product["identity"]
    before = consolidation_samples(SimpleNamespace(memory=memory, active_core=None, identity=product["identity"], store=root / "modules/luban0-v1"))
    expected = {"new_training": 22, "new_validation": 6, "replay_training": 44, "replay_validation": 15}
    if {key: len(rows) for key, rows in before.items()} != expected:
        raise ValueError("Eligible original corpus changed; do not silently alter this fixed v6 comparison")
    data, cases = uppercase_data(exposed), engineering_cases()
    cases = regression_copies(cases, product["identity"])
    run = root / "experiments" / ("luban0-v6-preregister-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    plan = {"dataset": DATASET, "base_identity": product["identity"], "storage_root": str(root), "model_path": model,
        "scope": "post-failure human/program-verified correction and fresh finite engineering acceptance; NOT an independent scientific final",
        "known_failure_use": "old product-ordinary-7 is exposed and becomes replay TRAIN/regression only; never called a fresh test",
        "old_reports": [{"path": str(path), "sha256": digest(path)} for path in (product_path, final_path)],
        "retired_v5_core": product["expected_core"], "retired_v5_checkpoint_sha256": product["checkpoint_sha256"],
        "before_groups": before, "before_dataset_sha256": content_digest(before), "uppercase_data": data, "engineering_cases": cases,
        "expected_seeded_counts": {"new_training": 22, "new_validation": 6, "replay_training": 56, "replay_validation": 19},
        "policy": {"seed": 42, "max_steps": 384, "learning_rate": 1e-5, "distillation_weight": 5, "preserve_replay_subspace": True,
            "checkpoints": CHECKPOINTS, "required_new_exposures": 2, "required_consecutive_train_passes": 2,
            "selection": "TRAIN only; no engineering or development feedback; reject at fixed max384 without extending budget"},
        "baseline_policy": "all four new uppercase replay-validation gold answers must pass original branch-OFF generation once before any new row is seeded; failure rejects whole run with unchanged gold",
        "operator_audit": {"protocol_version": OPERATOR_AUDIT_PROTOCOL, "tolerances": copy.deepcopy(TOLERANCES),
            "sample": data[0], "scope": "native full vs independent final-layer logits/CE/gradient/fresh FP32Adam; no cached-training or antiforgetting guarantee"},
        "model_execution": "one full native PyTorch instance, Runtime.load(paging=False), including true recovery reload; no production paging/speed claim",
        "retention_policy": "fixed16 are publication snapshot obligations only, never TRAIN/loss/checkpoint/DEV feedback; acceptance requires all16 then actual backup/original/fullreload with snapshot+complete protected validation hashes unchanged",
        "failure_policy": "retire any new candidate to original None, preserve immutable old/new checkpoints and raw failures, no answer-variant additions or further training on same engineering set",
        "training_attempts": 1, "teacher_calls": 0, "script_sha256": digest(__file__)}
    write_new(run / "uppercase-data.json", data)
    write_new(run / "engineering-cases.json", cases)
    write_new(run / "protocol.json", plan)
    descriptor = {"protocol_path": str(run / "protocol.json"), "protocol_sha256": digest(run / "protocol.json"),
                  "run_directory": str(run), "prepared_without_model": True, "teacher_calls": 0}
    write_new(descriptor_path, descriptor)
    print(json.dumps(descriptor, ensure_ascii=False), flush=True)


def run_once(protocol_path, expected_sha):
    from luban.evolution import consolidate, consolidation_samples, _target, _memory, TRAIN_CHECKPOINTS
    from luban.retention import read_core_retention, protected_validation
    from luban.runtime import Runtime
    import torch
    if digest(protocol_path) != expected_sha:
        raise ValueError("V6 fixed protocol SHA mismatch")
    plan = json.loads(protocol_path.read_text())
    if digest(__file__) != plan["script_sha256"] or list(TRAIN_CHECKPOINTS) != plan["policy"]["checkpoints"]:
        raise ValueError("Fixed v6 script or training-checkpoint policy changed")
    root, directory = require_storage(plan["storage_root"]), protocol_path.parent
    if (root / "modules/luban0-v1/active-core.json").exists():
        raise ValueError("V6 must start from the original core None")
    with (directory / "run-started.json").open("x") as marker:
        json.dump({"protocol_sha256": expected_sha, "started": time.time(), "training_attempts": 1}, marker)
    files = [REPOSITORY / "src/luban" / (name + ".py") for name in
             ("runtime", "evolution", "retention", "backups", "coldpaging", "backbone", "dynamic", "memory", "settings", "generation_checks")]
    files += [Path(__file__), REPOSITORY / "scripts/verify_training_operators.py", REPOSITORY / "scripts/verify_luban0_final.py", REPOSITORY / "scripts/verify_luban0_product.py"]
    execution = {"protocol_sha256": expected_sha, "source_sha256": {str(path): digest(path) for path in files}}
    write_new(directory / "execution-manifest.json", execution)
    report = {"protocol_sha256": expected_sha, "execution_manifest_sha256": digest(directory / "execution-manifest.json"),
        "run_directory": str(directory), "scope": plan["scope"], "passed": False, "model_started": False,
        "training_started": False, "teacher_calls": 0, "baseline_generation": [], "generation": []}
    started, runtime = time.perf_counter(), None
    try:
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(prefix="luban0-v6-settings-") as home:
            settings = Settings(home)
            settings.update({"storage_root": str(root), "model_path": plan["model_path"], "local_thinking": "off", "temperature": 0})
            runtime = Runtime(settings, Memory(root))
            runtime.load = partial(runtime.load, paging=False)
            report["model_started"] = True
            private_json(directory / "result.json", report)
            print(json.dumps({"phase": "model_loading", "protocol_sha256": expected_sha, "model_started": True}), flush=True)
            runtime.load()
            if runtime.active_core is not None or runtime.identity != plan["base_identity"] or consolidation_samples(runtime) != plan["before_groups"]:
                raise ValueError("Original core/base identity/full material corpus changed after preregistration")
            report.update(device=runtime.device, dtype=str(runtime.dtype), torch_version=torch.__version__,
                memory_samples=[_memory(runtime)], original_tensor_sha256=tensor_digest(_target(runtime)[1]))
            report["operator_audit"] = audit_training_operators(runtime, plan["operator_audit"]["sample"])
            if report["operator_audit"]["passed"] is not True or report["operator_audit"]["restored"] is not True:
                raise ValueError("Operator audit failed or did not restore original state")
            validation = [row for row in plan["uppercase_data"] if row["purpose"] == "validation"]
            report["baseline_generation"] = branch_off_generation(runtime, validation)
            private_json(directory / "result.json", report)
            if not all(row["passed"] for row in report["baseline_generation"]):
                raise ValueError("Original new uppercase baseline failed; gold unchanged, no seeding or training")
            seeded = []
            for row in plan["uppercase_data"]:
                provenance = json.loads(row["provenance"])
                provenance.update(protocol_id=row["id"], protocol_sha256=expected_sha)
                if row["purpose"] == "validation":
                    provenance.update(baseline_preservation_eligible=True, original_baseline_verified_core=None,
                        original_baseline_verified_identity=runtime.identity)
                identifier = runtime.memory.add_sample(row["prompt"], row["answer"], source="human", purpose=row["purpose"],
                    approved=True, training_allowed=True, provenance=provenance)
                seeded.append({"protocol_id": row["id"], "sample_id": identifier})
            groups = consolidation_samples(runtime)
            if {key: len(values) for key, values in groups.items()} != plan["expected_seeded_counts"]:
                raise ValueError("Fixed v6 seeded corpus counts do not match preregistration")
            write_new(directory / "seeded-groups.json", groups)
            report.update(seeded=seeded, training_dataset_sha256=content_digest(groups), training_started=True)
            private_json(directory / "result.json", report)
            report["evolution"] = consolidate(runtime, 384, learning_rate=1e-5, distillation_weight=5,
                preserve_replay_subspace=True, retention_regressions=plan["engineering_cases"])
            private_json(directory / "result.json", report)
            if not report["evolution"]["promoted"]:
                raise ValueError("One fixed-budget core attempt rejected; no engineering evaluation or additional training")
            report["candidate_version"] = runtime.active_core
            for row in plan["engineering_cases"]:
                generated = branch_off_generation(runtime, [row])[0]
                report["generation"].append(generated)
                private_json(directory / "result.json", report)
                print(json.dumps({"phase": "engineering", "case": row["id"], "passed": generated["passed"], "complete": generated["complete"]}), flush=True)
            if not all(row["passed"] for row in report["generation"]):
                raise ValueError("Fresh finite engineering acceptance failed; retire without retuning")
            snapshot = read_core_retention(runtime)
            protected = protected_validation(runtime)
            if snapshot["groups"] != groups or snapshot["regression_validation"] != plan["engineering_cases"]:
                raise ValueError("Published retention obligations differ from fixed complete corpus/new16")
            expected_protected = {row["id"] for key, values in groups.items() if key.endswith("validation") for row in values}
            expected_protected.update(row["id"] for row in plan["engineering_cases"])
            if len(protected) != len(expected_protected) or {row["id"] for row in protected} != expected_protected:
                raise ValueError("Retention does not expose the complete protected validation set")
            retention_path = runtime.store / (runtime.active_core + ".retention.json")
            report["retention"] = {"content_sha256": snapshot["content_sha256"],
                "file_sha256": digest(retention_path),
                "protected_validation_count": len(protected), "protected_validation_sha256": content_digest(protected)}
            verify_recovery(runtime, report["candidate_version"], recovery_cases(plan["engineering_cases"], report["generation"]), report["generation"], report)
            if (read_core_retention(runtime) != snapshot or content_digest(protected_validation(runtime)) != content_digest(protected)
                    or digest(retention_path) != report["retention"]["file_sha256"]):
                raise ValueError("True backup/full native reload changed complete retained materials/protection")
            if any(digest(path) != old_sha for path, old_sha in execution["source_sha256"].items()):
                raise ValueError("Frozen execution sources changed during the controlled run")
            for old in plan["old_reports"]:
                if digest(old["path"]) != old["sha256"]:
                    raise ValueError("An earlier scientific/product failure report changed")
            report.update(passed=True, active_core=runtime.active_core, retention_restored_unchanged=True,
                memory_samples=report["memory_samples"] + [_memory(runtime)])
            runtime.memory.event("core_v6_engineering_validation", {"passed": True, "version": runtime.active_core,
                "protocol_sha256": expected_sha, "scope": plan["scope"]})
    except BaseException as error:
        report.update(passed=False, error=str(error) or type(error).__name__)
        if runtime is not None:
            try:
                report["retirement"] = retire(runtime, None)
            except Exception as retirement_error:
                report["retirement_error"] = str(retirement_error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        private_json(directory / "result.json", report)
        private_json(REPOSITORY / "research/luban0_v6_run.json", report)
        print(json.dumps({"passed": report["passed"], "protocol_sha256": expected_sha, "run_directory": str(directory),
            "error": report.get("error"), "retirement": report.get("retirement"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", type=Path)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.prepare:
        config = json.loads((default_home() / "settings.json").read_text())
        prepare(config["storage_root"], config["model_path"])
    elif not args.protocol_sha256:
        parser.error("--run requires the explicit preregistered --protocol-sha256")
    else:
        run_once(args.run, args.protocol_sha256)


if __name__ == "__main__":
    main()
