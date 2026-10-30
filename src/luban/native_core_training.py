"""Native core optimization, preserving the existing generation and journal gates."""
from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
import uuid

import torch

from . import luban_training as engine


def product_graph(runtime):
    config = getattr(getattr(runtime, "model", None), "config", None)
    text = getattr(config, "text_config", config)
    return (getattr(runtime, "research", False) is not True
        and str(getattr(runtime, "device", "")) == "mps"
        and getattr(config, "model_type", None) == "qwen3_5"
        and getattr(runtime, "dtype", None) == torch.bfloat16
        and tuple(getattr(text, key, None) for key in
                  ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"))
            == (2048, 6144, 24, 248320))


def defaults(runtime, steps, learning_rate, distillation_weight, preserve_replay_subspace):
    native = product_graph(runtime)
    values = (450, 1e-5, 5, True) if native else (64, 1e-4, 0, False)
    return tuple(default if value is None else value for default, value in
        zip(values, (steps, learning_rate, distillation_weight, preserve_replay_subspace)))


def select(runtime, *, preserve_replay_subspace, distillation_weight,
           field_first_token_weight, partial_policy):
    if not product_graph(runtime):
        return None, {"name": engine.ENGINE_NAME, "engine": "torch-hf",
            "reason": "当前设备、精度、研究模式或模型图不属于正式原生核心训练范围"}
    if not preserve_replay_subspace or distillation_weight <= 0 or field_first_token_weight != 1 or partial_policy not in {None, "closed-book-growth-v1"}:
        return None, {"name": engine.ENGINE_NAME, "engine": "torch-hf",
            "reason": "显式训练目标不属于CE释放＋KL＋子空间保护的原生模式"}
    root = Path(__file__).resolve().parents[2]
    binary = next((path for path in (root / "native/luban-training", root / "build/luban-training", root / "build/native/luban-training")
                   if path.is_file()), None)
    if binary is None:
        raise RuntimeError("Luban训练引擎尚未安装CE释放二进制；不能静默改用旧训练路径")
    return binary, {"name": engine.ENGINE_NAME, "engine": engine.ENGINE_ID,
        "binary": str(binary), "binary_sha256": engine.file_sha(binary),
        "implementation": "LibTorch C++ CE-release; Python owns preparation, quality gates and journal",
        "quality_policy": "unchanged TRAIN checkpoints, held-out generation and old-core protection",
        "segments": [], "teacher_api_calls": 0}


def _state(parameter, master, optimizer, reference, basis):
    saved = optimizer.state.get(master, {})
    return {"candidate": parameter.detach().cpu(), "master": master.detach().cpu(),
        "reference": reference.detach().cpu(), "basis": basis.detach().cpu(),
        "exp_avg": saved.get("exp_avg", torch.zeros_like(master)).detach().cpu(),
        "exp_avg_sq": saved.get("exp_avg_sq", torch.zeros_like(master)).detach().cpu(),
        "adam_step": saved.get("step", torch.tensor(0., dtype=torch.float32)).detach().cpu()}


def _move_optimizer(master, optimizer, device):
    master.grad = None
    master.data = master.detach().to(device)
    for key, value in optimizer.state.get(master, {}).items():
        # The non-capturable Adam counter belongs on CPU, including Torch resume.
        if isinstance(value, torch.Tensor) and key != "step":
            optimizer.state[master][key] = value.to(device)


def _move_text(runtime, device):
    # Product paging intentionally leaves visual weights on meta. Moving the
    # whole multimodal container would try to materialize nonexistent storage.
    modules = (runtime.decoder, runtime.model.lm_head)
    originals = [tuple(module.parameters()) for module in modules]
    for module in modules:
        module.to(device)
    for module, before in zip(modules, originals):
        after = tuple(module.parameters())
        if len(before) != len(after) or any(old is not new for old, new in zip(before, after)):
            raise RuntimeError("语言核心迁移替换了Parameter对象，不能继续共享权重或优化器")


def _restore_state(parameter, master, optimizer, values):
    with torch.no_grad():
        parameter.copy_(values["candidate"].to(parameter.device))
        master.copy_(values["master"].to(master.device))
    optimizer.state[master] = {"step": values["adam_step"].detach().cpu().clone(),
        "exp_avg": values["exp_avg"].to(master.device), "exp_avg_sq": values["exp_avg_sq"].to(master.device)}


def _trace(output, fixture, count):
    with (Path(output) / "trace.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != engine.TRACE:
            raise ValueError("原生恢复trace schema不匹配")
        rows = list(reader)[:count]
    if len(rows) != count:
        raise ValueError("原生恢复trace缺少完整步数")
    weight = float(fixture["config"]["distillation_weight"])
    for row, scheduled in zip(rows, fixture["steps"]):
        if (int(row["step"]) != scheduled["ordinal"] or row["ce_role"] != scheduled["role"]
                or row["ce_row"] != scheduled["ce"] or row["replay_row"] != scheduled["replay"]):
            raise ValueError("原生恢复trace训练排程改变")
        for key in ("ce", "kl", "preclip_norm", "loss", "wall_seconds"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError("原生恢复trace含非有限值")
        if not math.isclose(row["loss"], row["ce"] + weight * row["kl"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("原生恢复trace损失不匹配")
    return rows


def steps(runtime, binary, report, groups, parameter, previous, master, optimizer,
          reference, basis, resume_step, state):
    """Yield original step scalars; every QC boundary exits the native worker first.

    state() observes the caller's current exposure/KL/streak ledger. Intermediate
    journal saves use actual complete checkpoint tensors and a matching shadow
    ledger. QC-boundary saves remain with the caller, after all generation checks.
    """
    from .evolution import _save_core_step
    session = getattr(runtime, "training_session", None)
    caches = engine.capture_tails(runtime, groups["new_training"] + groups["replay_training"])
    root = Path(runtime.store) / ("native-training-" + report["id"] + "-" + uuid.uuid4().hex[:8])
    if root.is_symlink() or root.exists():
        raise ValueError("原生训练必须使用独立新目录")
    root.mkdir(mode=0o700)
    report["training_engine"]["artifacts"] = str(root)
    ends = sorted({s for s in report["training_selection"]["checkpoints"] if s > resume_step} | {report["steps"]})
    offset = resume_step
    row_ids = {"r" + str(i).zfill(3): row["id"] for i, row in
               enumerate(groups["new_training"] + groups["replay_training"])}
    for stop in ends:
        length = stop - offset
        selected = [s - offset for s in range(((offset // 16) + 1) * 16, stop + 1, 16)]
        selected = sorted(set(selected + [length]))
        fixture_path = root / f"{offset:06d}-input"
        output = root / f"{offset:06d}-output"
        saved = _state(parameter, master, optimizer, reference, basis)
        fixture = engine.export_fixture(runtime, groups["new_training"], groups["replay_training"],
            reference, basis, caches, fixture_path, steps=length,
            learning_rate=report["learning_rate"], distillation_weight=report["distillation"]["weight"],
            checkpoint_steps=selected, start_step=offset, saved_checkpoint=saved)
        segment = {"start_step": offset, "end_step": stop, "fixture": str(fixture_path),
            "manifest_sha256": fixture["manifest_sha256"], "output": str(output), "completed": False}
        report["training_engine"]["segments"].append(segment)
        prefix = copy.deepcopy(report)
        exposures, total, maximum, streak = state()
        before = dict(exposures)
        reference_cpu, basis_cpu = reference.cpu(), basis.cpu()

        def checkpoint(local_step, descriptor):
            if session is None or offset + local_step == stop:
                return  # The boundary generation/streak must be saved by the caller.
            values = engine.load_checkpoint(output, fixture, step=local_step)
            _restore_state(parameter, master, optimizer, values)
            rows = _trace(output, fixture, local_step)
            shadow = copy.deepcopy(prefix)
            shadow_exposures = dict(before)
            shadow_total, shadow_maximum = total, maximum
            for row in rows:
                if row["ce_role"] == "new":
                    shadow_exposures[row_ids[row["ce_row"]]] += 1
                shadow_total += row["kl"]
                shadow_maximum = max(shadow_maximum, row["kl"])
                shadow["distillation"]["target_tokens"] += fixture["rows"][row["replay_row"]]["gold"]
            absolute = offset + local_step
            shadow["distillation"].update(steps=absolute, kl_mean=shadow_total / absolute, kl_max=shadow_maximum)
            old_step = runtime.job["step"]
            runtime.job.update(step=absolute, supervised_loss=rows[-1]["ce"], replay_kl=rows[-1]["kl"], loss=rows[-1]["loss"])
            try:
                _save_core_step(session, runtime, shadow, parameter, previous, master, optimizer,
                    basis_cpu, shadow_exposures, shadow_total, shadow_maximum, streak)
            finally:
                runtime.job["step"] = old_step

        # No complete HF graph remains in MPS while the C++ worker owns its tail.
        # Copies of reference/basis are necessary: the caller retains these objects.
        _move_text(runtime, "cpu")
        _move_optimizer(master, optimizer, "cpu")
        reference.data = reference_cpu
        basis.data = basis_cpu
        torch.mps.synchronize()
        torch.mps.empty_cache()
        try:
            receipt = engine.run_worker(runtime, fixture, output, binary, on_checkpoint=checkpoint)
            values = engine.load_checkpoint(output, fixture)
            _restore_state(parameter, master, optimizer, values)
            segment.update(completed=True, receipt_sha256=receipt["receipt_sha256"],
                native_report=receipt["report"])
        finally:
            # run_worker kills/waits on every exception before full HF returns.
            _move_text(runtime, runtime.device)
            _move_optimizer(master, optimizer, runtime.device)
            reference.data = reference.to(runtime.device)
            basis.data = basis.to(runtime.device)
        current = engine.graph_binding(runtime)
        if any(cache["binding"] != current for cache in caches.values()):
            raise ValueError("原生区间结束后冻结前缀绑定改变，禁止继续使用缓存")
        for index, row in enumerate(receipt["trace"]):
            absolute = offset + index + 1
            if absolute % 16 == 0 or absolute == stop:
                # The outer journal still saves at its original safe-step points.
                # Its tensors must match that ordinal, not the already computed
                # end-of-segment candidate used to finish the native receipt.
                values = engine.load_checkpoint(output, fixture, step=index + 1)
                _restore_state(parameter, master, optimizer, values)
            yield offset + index, row, fixture["rows"][row["replay_row"]]["gold"]
        offset = stop
