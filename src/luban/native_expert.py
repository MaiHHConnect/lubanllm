"""Read the published Q4 expert proof; inference never manufactures qualification."""
import json
import re


def native_expert_binding(runtime, binary):
    if (binary is None or getattr(runtime, "active_core", None) is None
            or getattr(runtime, "native_inference_precision", None) != "mixed-q4-g16"
            or getattr(runtime, "native_skip_layer_mask", 0) != 0
            or getattr(runtime, "native_exit_after", 24) != 24
            or getattr(runtime, "native_layer_paging", False)):
        return None
    branch = runtime.branch
    expert_id = getattr(branch, "expert_id", None)
    if not expert_id or not runtime.pointer.exists():
        return None
    weights, metadata = branch._paths(expert_id)
    record = json.loads(metadata.read_text())
    proof = record.get("native_q4")
    if proof is None:
        return None
    active = json.loads(runtime.pointer.read_text())
    core = json.loads((runtime.store / "active-core.json").read_text())
    receipt = json.loads(binary.with_name(binary.name + ".json").read_text())
    report_name = proof.get("report") if isinstance(proof, dict) else None
    if not isinstance(report_name, str) or not re.fullmatch(r"learning-[a-f0-9]{12}\.json", report_name):
        raise ValueError("原生专家缺少已发布的验收报告")
    report = json.loads((runtime.store / report_name).read_text())
    expected = {"schema": "native-expert-q4-v1", "passed": True,
                "core_version": runtime.active_core, "core_file_sha256": core.get("sha256"),
                "expert_file_sha256": record.get("sha256"), "worker_binary_sha256": receipt.get("binary_sha256")}
    if (any(proof.get(key) != value for key, value in expected.items())
            or report.get("native_q4") != proof or report.get("quality_checked") is not True
            or not isinstance(report.get("gates"), dict) or not report["gates"]
            or any(value is not True for value in report["gates"].values())
            or report.get("identity") != runtime.identity
            or core.get("identity") != runtime.identity or core.get("version") != runtime.active_core
            or active.get("id") != expert_id or active.get("identity") != runtime.identity
            or active.get("core_version") != runtime.active_core
            or record.get("identity") != runtime.identity + ":layer-23"
            or record.get("core_version") != runtime.active_core or record.get("requires_gate") is not True
            or record.get("hidden") != 2048 or type(record.get("width")) is not int
            or record["width"] != active.get("width")
            or not 1 <= record["width"] <= 6144
            or not all(isinstance(proof.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", proof[key])
                       for key in ("core_file_sha256", "core_tensor_sha256", "expert_file_sha256",
                                   "worker_binary_sha256", "graph_binary_sha256"))):
        raise ValueError("原生专家资格与当前核心、专家或生成引擎不一致")
    return {"file": str(weights), "sha256": record["sha256"], "width": record["width"],
            "core_sha256": core["sha256"]}
