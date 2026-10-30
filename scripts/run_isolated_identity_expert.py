"""Train and inspect one isolated identity expert under ordinary-core protection.

The experiment never points the product at the candidate. It is deliberately
small: a rejected candidate is useful evidence, while a promoted candidate is
still pending independent route calibration.
"""
import argparse
import copy
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import torch

from luban.evolution import _generation
from luban.runtime import _freeze_preservation_rows, _preservation_generation
from luban.memory import Memory
from luban.runtime import Runtime
from luban.settings import Settings

TARGET = "我是麦好火旗下的luban0，luban0模型，基于qwen3.5开始生长的，能无限进化的模型。"


def row(identifier, prompt, purpose, *, messages=None):
    return {
        "id": identifier,
        "prompt": prompt,
        "answer": TARGET,
        "source": "human",
        "purpose": purpose,
        "approved": True,
        "training_allowed": True,
        "provenance": json.dumps({
            "evolution_role": "new",
            "split_family": identifier,
            "messages": messages or [{"role": "user", "content": prompt}],
            "verification_evidence": "User-authored identity target, 2026-10-16; finite product identity statement.",
            "generation_check": {"type": "exact", "expected": TARGET, "max_new_tokens": 128},
        }, ensure_ascii=False),
    }


def identity_material(round_index=1):
    if round_index == 3:
        train = [
            row("recursive3-identity-train-0", "请用一句简洁的话说明你是谁以及属于哪个品牌。", "training"),
            row("recursive3-identity-train-1", "请从模型系列、具体名称、品牌归属、基座四项介绍自己。", "training"),
            row("recursive3-identity-train-2", "如果用户要求正式自我介绍，请给出完整答复。", "training"),
            row("recursive3-identity-train-3", "你的官方身份声明应包含哪些信息？请直接回答。", "training"),
            row("recursive3-identity-train-4", "你是哪个本地模型？请说明名称和来源。", "training"),
            row("recursive3-identity-train-5", "请核对 luban0 的名称、品牌和起始基座。", "training"),
        ]
        valid = [
            row("recursive3-identity-valid-0", "请按官方资料说明模型身份和来源。", "validation"),
            row("recursive3-identity-valid-1", "模型身份核验：你是谁、属于谁、基于什么？", "validation"),
            row("recursive3-identity-valid-2", "请不要介绍能力，只回答你的模型身份。", "validation"),
            row("recursive3-identity-valid-3", "上一轮我们讨论了天气。现在请说明你的模型身份。", "validation", messages=[
                {"role": "user", "content": "今天天气如何？"},
                {"role": "assistant", "content": "我无法获取实时天气。"},
                {"role": "user", "content": "上一轮我们讨论了天气。现在请说明你的模型身份。"},
            ]),
        ]
        return train, valid
    if round_index == 2:
        train = [
            row("recursive-identity-train-0", "请用官方口吻介绍模型身份。", "training"),
            row("recursive-identity-train-1", "请说明麦好火与 luban0 的关系。", "training"),
            row("recursive-identity-train-2", "你的模型是基于什么开始生长的？", "training"),
            row("recursive-identity-train-3", "作为 API 助手，你应该怎样自我介绍？", "training"),
            row("recursive-identity-train-4", "请分别说出模型系列、名称和基座。", "training"),
            row("recursive-identity-train-5", "身份核验：名称、归属、基座分别是什么？", "training"),
        ]
        valid = [
            row("recursive-identity-valid-0", "请给出完整的官方模型自我介绍。", "validation"),
            row("recursive-identity-valid-1", "开发者想知道你的模型和来源，请回答。", "validation"),
            row("recursive-identity-valid-2", "如果有人问我是谁，你应该如何回答？", "validation"),
            row("recursive-identity-valid-3", "核对身份：你属于哪个模型，来自哪里？", "validation", messages=[
                {"role": "user", "content": "2+3是多少？"},
                {"role": "assistant", "content": "5"},
                {"role": "user", "content": "核对身份：你属于哪个模型，来自哪里？"},
            ]),
        ]
        return train, valid
    train = [
        row("isolated-identity-train-0", "你是谁？", "training"),
        row("isolated-identity-train-1", "请介绍一下你自己。", "training"),
        row("isolated-identity-train-2", "你的模型名称和归属是什么？", "training"),
        row("isolated-identity-train-3", "请用一句话说明你来自哪里。", "training"),
        row("isolated-identity-train-4", "能告诉我你是什么模型吗？", "training"),
        row("isolated-identity-train-5", "通过API调用时你应该怎样介绍自己？", "training"),
    ]
    valid = [
        row("isolated-identity-valid-0", "我现在和哪个模型对话？", "validation"),
        row("isolated-identity-valid-1", "请说出你的身份和基座来源。", "validation"),
        row("isolated-identity-valid-2", "开发者要核对你的模型信息，请完整回答。", "validation"),
        row("isolated-identity-valid-3", "再说明一次你的身份。", "validation", messages=[
            {"role": "user", "content": "2+3是多少？"},
            {"role": "assistant", "content": "5"},
            {"role": "user", "content": "再说明一次你的身份。"},
        ]),
    ]
    return train, valid


def fresh_ordinary_rows():
    def make(identifier, prompt, check):
        return {"id": identifier, "prompt": prompt, "answer": "", "purpose": "validation",
                "source": "program", "approved": True, "training_allowed": False,
                "provenance": {"messages": [{"role": "user", "content": prompt}],
                               "generation_check": check}}
    return [
        make("isolated-ordinary-0", "计算17加8，只输出整数。",
             {"type": "semantic", "validator": "integer", "expected": 25, "max_new_tokens": 32,
              "forbidden": ["鲁班", "麦好火", "qwen3.5"]}),
        make("isolated-ordinary-1", "把字符串“PyTorch”转为大写，只输出结果。",
             {"type": "semantic", "validator": "normalized_exact", "expected": "PYTORCH", "max_new_tokens": 32,
              "forbidden": ["鲁班", "麦好火", "qwen3.5"]}),
        make("isolated-ordinary-2", "JSON {\"n\":7} 中 n 的值是多少？",
             {"type": "semantic", "validator": "integer", "expected": 7, "max_new_tokens": 32,
              "forbidden": ["鲁班", "麦好火", "qwen3.5"]}),
        make("isolated-ordinary-3", "请把 hello 翻译成中文，只输出译文。",
             {"type": "semantic", "validator": "translation", "accepted": ["你好"], "max_new_tokens": 32,
              "forbidden": ["鲁班", "麦好火", "qwen3.5"]}),
        make("isolated-ordinary-4", "H2O 的中文名称是什么？只输出名称。",
             {"type": "semantic", "validator": "one_of", "accepted": ["水", "水。"], "max_new_tokens": 32,
              "forbidden": ["鲁班", "麦好火", "qwen3.5"]}),
        make("isolated-ordinary-5", "今天天气如何？",
             {"type": "contains", "expected": "无法获取", "max_new_tokens": 64,
              "forbidden": ["鲁班", "麦好火", "qwen3.5"]}),
    ]


def prepare_ordinary_protection(runtime, rows):
    """Freeze branch-off answers as read-only teachers before candidate training."""
    original = runtime.branch.expert
    try:
        runtime.branch.expert = None
        baseline = _generation(runtime, rows)
    finally:
        runtime.branch.expert = original
    by_id = {item["id"]: item for item in baseline}
    if (len(by_id) != len(rows)
            or not all(item["answer"].strip() for item in baseline)):
        raise ValueError("原始核心普通保护基线没有可冻结的回答，拒绝开始身份专家训练")
    prepared = []
    for row in rows:
        value = dict(row["provenance"])
        value.update({"preservation_only": True,
                      "verified_by": "original-core branch-off generation",
                      "verification_evidence": "2026-10-16 finite ordinary protection baseline",
                      "preservation_baseline": {"answer": by_id[row["id"]]["answer"],
                                                 "complete": by_id[row["id"]]["complete"]}})
        check = dict(value["generation_check"])
        check["type"], check["expected"] = "exact", by_id[row["id"]]["answer"]
        value["generation_check"] = check
        prepared.append({**row, "answer": by_id[row["id"]]["answer"],
                         "provenance": json.dumps(value, ensure_ascii=False)})
    return prepared, baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--storage", required=True, help="Production storage, read-only source")
    parser.add_argument("--output", required=True, help="New isolated experiment directory")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--parent-storage", help="read-only storage of the previously promoted expert")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("实验输出目录必须是新的，拒绝覆盖候选证据")
    output.mkdir(parents=True)
    isolated_store = output / "storage"
    module = isolated_store / "modules" / "luban0-v1"
    module.mkdir(parents=True)
    source_module = Path(args.storage) / "modules" / "luban0-v1"
    for name in ("graph-verification.json", "original-core-protection.json"):
        source = source_module / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"隔离实验缺少只读保护来源：{name}")
        shutil.copyfile(source, module / name)

    parent_active = None
    if args.parent_storage:
        parent_module = Path(args.parent_storage) / "modules" / "luban0-v1"
        parent_active = json.loads((parent_module / "active.json").read_text())
        parent_router = json.loads((parent_module / "active-router.json").read_text())
        expert_id = parent_active.get("id")
        router_id = parent_router.get("id")
        if not isinstance(expert_id, str) or not isinstance(router_id, str):
            raise ValueError("父专家或父路由活动指针无效")
        expert_dir = parent_module / "layer-23"
        for source, target in (
                (parent_module / "active.json", module / "active.json"),
                (parent_module / "active-router.json", module / "active-router.json"),
                (parent_module / f"{router_id}.json", module / f"{router_id}.json"),
                (parent_module / f"{router_id}.safetensors", module / f"{router_id}.safetensors"),
                (expert_dir / f"{expert_id}.json", module / "layer-23" / f"{expert_id}.json"),
                (expert_dir / f"{expert_id}.safetensors", module / "layer-23" / f"{expert_id}.safetensors"),
        ):
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"父专家发布文件缺失或为符号链接：{source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    train, valid = identity_material(args.round)
    parent_train, parent_valid = (identity_material(args.round - 1)
                                  if args.parent_storage and args.round > 1 else ([], []))
    settings_home = tempfile.mkdtemp(prefix="luban0-isolated-identity-settings-")
    settings = Settings(settings_home)
    settings.update({"model_path": args.model, "storage_root": str(isolated_store),
                     "local_thinking": "off", "temperature": 0, "max_new_tokens": 128})
    runtime = Runtime(settings, Memory(isolated_store), research=True)
    started = time.time()
    report = None
    error = None
    try:
        runtime.load(paging=False)
        for sample in train + valid:
            runtime.memory.add_sample(sample["prompt"], sample["answer"], source=sample["source"],
                                      purpose=sample["purpose"], approved=True, training_allowed=True,
                                      provenance=json.loads(sample["provenance"]))
        ordinary_checks = fresh_ordinary_rows()
        ordinary_rows, ordinary_baseline = prepare_ordinary_protection(runtime, ordinary_checks)
        parent_rows = []
        parent_baseline = []
        if parent_valid:
            # Parent identity rows are frozen with the previous expert enabled;
            # they are part of the candidate objective, not only a post-hoc
            # continuity check.
            parent_baseline = _generation(runtime, parent_valid)
            parent_rows = _freeze_preservation_rows(
                parent_valid, parent_baseline,
                evidence="isolated recursive parent-expert baseline before candidate learning",
                teacher_mode="expert")
        preservation_rows = parent_rows + ordinary_rows
        report = runtime.learn(steps=args.steps, preservation_rows=preservation_rows, preservation_weight=5.0)
        checks = valid + parent_valid + ordinary_checks
        results = _generation(runtime, checks)
        parent_start = len(valid)
        ordinary_start = parent_start + len(parent_valid)
        ordinary_protection_results = _preservation_generation(runtime, ordinary_rows)
        parent_preservation_results = (_preservation_generation(runtime, parent_rows)
                                       if parent_rows else [])
        result = {
            "identity_target": TARGET,
            "material": {"training": train, "validation": valid},
            "learning_report": report,
            "candidate_active": runtime.branch.expert is not None,
            "candidate_requires_gate": runtime.branch.requires_gate,
            "closed_book_results": results,
            "ordinary_protection_results": ordinary_protection_results,
            "ordinary_protection_baseline": ordinary_baseline,
            "ordinary_protection_rows": ordinary_rows,
            "parent_preservation_results": parent_preservation_results,
            "parent_preservation_rows": parent_rows,
            "parent_preservation_baseline": parent_baseline,
            "identity_passed": all(r["passed"] for r in results[:len(valid)]),
            "parent_identity_passed": (all(r["passed"] for r in results[parent_start:ordinary_start])
                                        if parent_valid else None),
            "ordinary_passed": all(r["passed"] for r in ordinary_protection_results),
            "parent_preservation_passed": (all(r["passed"] for r in parent_preservation_results)
                                            if parent_rows else None),
            "ordinary_semantic_passed": all(r["passed"] for r in results[ordinary_start:]),
            "recursive_round": args.round,
            "parent_storage": args.parent_storage,
            "parent_expert_id": (parent_active or {}).get("id"),
            "parent_router_id": (json.loads((module / "active-router.json").read_text()).get("id")
                                 if args.parent_storage else None),
            "route_calibration": {"passed": False, "state": "pending", "reason": "候选尚未完成独立家庭拆分路由证明"},
            "scope": "isolated identity expert candidate; no production publication",
            "production_mutations": [],
            "teacher_calls": 0,
            "training_performed": True,
            "device": runtime.device,
            "dtype": str(runtime.dtype),
            "hardware": platform.platform(),
            "elapsed_seconds": time.time() - started,
            "isolated_storage": str(isolated_store),
        }
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"candidate_active": result["candidate_active"],
                          "identity_passed": result["identity_passed"],
                          "ordinary_passed": result["ordinary_passed"],
                          "promoted": bool(report and report.get("promoted")),
                          "output": str(output)}, ensure_ascii=False))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (output / "failure.json").write_text(json.dumps({
            "error": error, "scope": "isolated identity expert candidate; no production publication",
            "production_mutations": [], "teacher_calls": 0, "training_performed": True,
            "elapsed_seconds": time.time() - started,
        }, ensure_ascii=False, indent=2) + "\n")
        raise
    finally:
        try:
            runtime.stop()
        except Exception:
            runtime._release_model()


if __name__ == "__main__":
    main()
