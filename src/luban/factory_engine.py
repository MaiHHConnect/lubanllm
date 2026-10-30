"""Independent current-engine evidence; accepted source proofs stay immutable."""
import json
import re
from pathlib import Path


GATES = ("prepare_equivalent", "real_adam_updates", "checkpoint_replay_exact",
         "cancellation_atomic", "continued_after_cancel", "saved_expert_reload_equal", "base_core_frozen")


def math_valid(math, proof):
    stages = math.get("stages", [])
    return (math.get("passed") is True and math.get("fp32_pass") is True
            and math.get("bf16_pass_assumed") is False and math.get("bf16_numerical_pass") is True
            and math.get("bf16_relative_l2_limit") == .03
            and math.get("expert_file_sha256") == proof.get("expert_file_sha256")
            and isinstance(stages, list) and len(stages) == 8
            and {(s.get("dtype"), s.get("stage")) for s in stages} == {
                (dtype, stage) for dtype in ("fp32", "bf16") for stage in ("gate", "up", "activation", "output")}
            and all(s.get("finite") is True and s.get("nonzero") is True and s.get("fp32_tolerance_pass") is True
                    and (s["dtype"] != "bf16" or type(s.get("relative_l2")) in (int, float)
                         and 0 <= s["relative_l2"] <= .03) for s in stages))


def qualifications():
    records = json.loads(Path(__file__).with_name("factory-engine-qualifications.json").read_text())
    if not isinstance(records, list):
        raise ValueError("初始专家引擎验证记录无效")
    for row in records:
        training, quality, math = row.get("training", {}), row.get("quality", {}), row.get("math", {})
        proof = row.get("source_proof", {})
        if (row.get("schema") == "luban-factory-engine-qualification-v1" and row.get("complete") is True
                and row.get("teacher_calls") == 0 and row.get("sources_unchanged") is True
                and isinstance(row.get("worker_binary_sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", row["worker_binary_sha256"])
                and proof.get("expert_id") == "learned-bafb4050374f"
                and proof.get("schema") == "mlx-expert-learning-v1" and proof.get("width") == 32
                and proof.get("quality_admitted") is True and proof.get("training_admitted") is True
                and proof.get("core_version") == "core-fdfa0ce5a75c"
                and training.get("schema") == "mlx-native-training-qualification-v1"
                and training.get("complete") is True and all(training.get("gates", {}).get(k) is True for k in GATES)
                and training.get("worker_binary_sha256") == row.get("worker_binary_sha256")
                and all(training.get(k) == proof.get(k) for k in ("core_file_sha256", "mlx_model_sha256"))
                and training.get("parent_expert_file_sha256") == proof.get("expert_file_sha256")
                and math_valid(math, proof)
                and quality.get("complete") is True and quality.get("two_passes_equal") is True
                and quality.get("routes_equal") is True and quality.get("protected_answers_equal") is True
                and quality.get("all_previously_correct_still_pass") is True
                and type(quality.get("rows")) is int and quality["rows"] > 0):
            yield row


def worker_qualified(proof, worker_sha):
    return isinstance(worker_sha, str) and re.fullmatch(r"[a-f0-9]{64}", worker_sha) is not None and (proof.get("worker_binary_sha256") == worker_sha or any(
        row["worker_binary_sha256"] == worker_sha and row["source_proof"] == proof for row in qualifications()))


def training_qualification(original, worker_sha, core_sha, model_sha):
    for training in (original, *(row["training"] for row in qualifications())):
        if (training.get("worker_binary_sha256") == worker_sha and training.get("core_file_sha256") == core_sha
                and training.get("mlx_model_sha256") == model_sha and training.get("complete") is True
                and all(training.get("gates", {}).get(k) is True for k in GATES)):
            return training
    return original  # Existing caller reports the exact failed qualification binding.
