"""Actual product-graph 2+2 bridge/resume proof; never quality or adoption.

Imports model libraries only inside run(). Root alone executes the real model.
All fixtures, checkpoints, receipts and an isolated product store live in a new
caller-selected output directory. No teacher API or production pointer writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def selected_rows(protocol):
    groups = protocol.get("groups", {})
    result = {}
    for name, role in (("new_training", "new"), ("replay_training", "replay")):
        rows = groups.get(name)
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError("smoke需要当前brand protocol原文中的2 NEW和2 REPLAY")
        chosen = rows[:2]
        for sample in chosen:
            if not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not sample["id"]:
                raise ValueError("smoke材料必须有有效ID")
            source = sample.get("provenance", {})
            source = json.loads(source) if isinstance(source, str) else source
            permissions = (sample.get("approved"), sample.get("training_allowed"))
            if (not isinstance(source, dict) or sample.get("purpose") != "training"
                    or any(type(value) not in (bool, int) or value != 1 for value in permissions)
                    or source.get("evolution_role") != role):
                raise ValueError("smoke材料必须为原文已许可/已核验的对应TRAIN角色")
        result[name] = chosen
    identifiers = [row["id"] for rows in result.values() for row in rows]
    if len(set(identifiers)) != 4:
        raise ValueError("smoke材料ID不得重复")
    return result


def frozen_snapshot(runtime, target, engine):
    return {name: {"id": id(value), "version": value._version,
        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "meta": value.is_meta, "sha256": None if value.is_meta else engine.tensor_sha(value)}
        for name, value in runtime.model.named_parameters() if value is not target}


def run(args):
    import torch
    from luban import luban_training as engine
    from luban.coldpaging import full_backbone
    from luban.evolution import _target, _capture_replay_subspace
    from luban.native_core_training import _move_text, product_graph
    from luban.runtime import Runtime
    from luban.settings import Settings, private_json, require_storage

    model = engine._path(args.model_root)
    protocol_path = engine._path(args.protocol)
    root = engine._path(args.output, exists=False)
    if (root.exists() or not root.parent.is_dir() or root.is_relative_to(model)
            or model.is_relative_to(root) or protocol_path.is_relative_to(root)):
        raise ValueError("smoke输出必须是模型目录外的独立新目录")
    require_storage(root.parent)
    protocol = json.loads(protocol_path.read_text())
    groups = selected_rows(protocol)
    root.mkdir(mode=0o700)
    storage = root / "isolated-storage"
    storage.mkdir(mode=0o700)
    settings = Settings(root / "isolated-home")
    settings.data.update(model_path=str(model), storage_root=str(storage), temperature=0,
                         max_new_tokens=128, local_thinking="off")
    runtime = Runtime(settings, None, research=False)
    report = {"protocol": "actual-product-native-core-bridge-smoke-2plus2-v1",
        "scope": "actual graph/lock/checkpoint/resume only; no quality, speed or adoption claim",
        "model_root": str(model), "protocol_file": str(protocol_path),
        "protocol_sha256": engine.file_sha(protocol_path), "samples": groups,
        "seed": 42, "precision": "BF16 weights; FP32 master/reference/Adam",
        "teacher_api_calls": 0, "generation_calls": 0, "publication": False,
        "passed": False, "segments": [], "target_restored": False, "model_released": False}
    private_json(root / "protocol.json", report)
    began = time.perf_counter()
    original = target = None
    pointer = pointer_before = None
    try:
        runtime.load(paging=True)
        runtime.close_native()
        if not product_graph(runtime) or runtime.cold_backbone is None:
            raise ValueError("本smoke必须使用真实BF16 MPS Qwen3.5-2B正式分页图")
        with full_backbone(runtime):
            runtime.branch.expert = None
            runtime.model.eval().requires_grad_(False)
            name, target = _target(runtime)
            target.requires_grad_(True)
            original = target.detach().cpu().clone()
            original_sha = engine.tensor_sha(original)
            pointer = Path(runtime.store) / "active-core.json"
            pointer_before = pointer.read_bytes() if pointer.exists() else None
            report.update(identity=runtime.identity, parameter=name, original_sha256=original_sha,
                active_core=runtime.active_core, device=str(runtime.device), torch_version=str(torch.__version__))
            before = engine.graph_binding(runtime)
            frozen_before = frozen_snapshot(runtime, target, engine)
            visual = getattr(runtime.model.model, "visual", None)
            if visual is None or not all(p.is_meta for p in visual.parameters()):
                raise ValueError("正式分页smoke必须保留视觉meta休眠状态")
            private_json(root / "binding-before.json", before)
            private_json(root / "frozen-before.json", frozen_before)
            basis, basis_report = _capture_replay_subspace(runtime, target, groups["replay_training"])
            basis = basis.to(target.device)
            reference = original.to(device=target.device, dtype=torch.float32)
            private_json(root / "basis.json", basis_report)
            caches = engine.capture_tails(runtime, groups["new_training"] + groups["replay_training"])
            resumed = None
            try:
                for offset in (0, 2):
                    segment = root / ("segment-" + str(offset))
                    segment.mkdir(mode=0o700)
                    fixture = engine.export_fixture(runtime, groups["new_training"], groups["replay_training"],
                        reference, basis, caches, segment / "input", steps=2, checkpoint_steps=[1, 2],
                        learning_rate=1e-5, distillation_weight=5, start_step=offset, saved_checkpoint=resumed)
                    entry = {"offset": offset, "fixture": fixture["directory"],
                        "manifest_sha256": fixture["manifest_sha256"], "entry_state_sha256":
                            {k: engine.tensor_sha(v) for k, v in resumed.items()} if resumed else None}
                    report["segments"].append(entry)
                    _move_text(runtime, "cpu")
                    # These fixtures bind immutable CPU copies; the live reference
                    # and basis can release MPS storage while the worker executes.
                    reference.data = reference.cpu()
                    basis.data = basis.cpu()
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                    try:
                        receipt = engine.run_worker(runtime, fixture, segment / "output", args.binary)
                        resumed = engine.load_checkpoint(segment / "output", fixture)
                        state_path = segment / "complete-state.pt"
                        with state_path.open("xb") as stream:
                            torch.save(resumed, stream)
                        restored = torch.load(state_path, map_location="cpu", weights_only=True)
                        state_sha = {k: engine.tensor_sha(v) for k, v in resumed.items()}
                        if set(restored) != set(engine.FINAL) or any(engine.tensor_sha(restored[k]) != state_sha[k] for k in state_sha):
                            raise ValueError("完整CPU检查点保存回读不同")
                        resumed = restored
                        entry.update(receipt=receipt, state_file=str(state_path),
                            state_file_sha256=engine.file_sha(state_path), state_sha256=state_sha,
                            actual_adam_step=float(resumed["adam_step"]))
                        if float(resumed["adam_step"]) != offset + 2:
                            raise ValueError("恢复后的Adam累计步数不同")
                        with torch.no_grad():
                            target.copy_(resumed["candidate"])
                    finally:
                        _move_text(runtime, runtime.device)
                        reference.data = reference.to(runtime.device)
                        basis.data = basis.to(runtime.device)
                    after = engine.graph_binding(runtime)
                    private_json(segment / "binding-after.json", after)
                    if after != before or not all(p.is_meta for p in visual.parameters()):
                        raise ValueError("分段训练后冻结绑定改变或视觉被错误唤醒")
                    frozen_after = frozen_snapshot(runtime, target, engine)
                    private_json(segment / "frozen-after.json", frozen_after)
                    if frozen_after != frozen_before:
                        raise ValueError("实际完整冻结参数发生变化")
                    entry.update(binding_retained=True, frozen_parameters_retained=True, vision_remained_meta=True)
                    private_json(root / "progress.json", report)
                if report["segments"][1]["entry_state_sha256"] != report["segments"][0]["state_sha256"]:
                    raise ValueError("第二段没有继承第一段完整实际状态")
                report["passed"] = True
            finally:
                with torch.no_grad():
                    target.copy_(original.to(target.device))
                target.requires_grad_(False)
                report["target_restored"] = engine.tensor_sha(target) == original_sha
                report["pointer_preserved"] = (pointer.read_bytes() if pointer.exists() else None) == pointer_before
                if not report["target_restored"] or not report["pointer_preserved"]:
                    report["passed"] = False
                    raise RuntimeError("smoke没有完整恢复原目标或活动指针")
    except BaseException as error:
        report.update(passed=False, error=str(error), error_type=type(error).__name__)
        raise
    finally:
        runtime._release_model()
        report["model_released"] = runtime.model is None and runtime._model_process_lock is None and runtime.native_engine is None
        report["elapsed_seconds"] = time.perf_counter() - began
        private_json(root / "result.json", report)
        print(json.dumps({"output": str(root), "passed": report["passed"],
            "target_restored": report["target_restored"], "model_released": report["model_released"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
