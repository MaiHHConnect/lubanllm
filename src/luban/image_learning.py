"""One fixed local image/operation learning round; candidates are never active.

The server owns the job/thread. This synchronous entry owns one runtime scope,
shares the existing backbone with C++ and retains unaccepted Adam checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import time
import uuid

import psutil
import torch

from .coldpaging import full_backbone
from .native_image_training import MasterQA, NativeEngine, library_path, lower_graph, prepare_sample
from .process_lock import ModelProcessLock
from .vision import image_processor, load_visual

ROUND_BUDGET = 5  # A bounded first product round, not an empirically optimal count.
ROUND_STEPS = ROUND_BUDGET
REQUIRED_AVAILABLE_BYTES = 7_968_468_672


def capability(runtime):
    # SOURCE/CPU shared-ABI qualification is separate from a real product round.
    reason = ""
    try:
        library_path()
        if str(runtime.device) not in {"cpu", "mps"}:
            raise ValueError("当前图片学习只支持本机 CPU/BF16 原生训练")
        if runtime.model is not None and getattr(runtime.model.config, "model_type", None) != "qwen3_5":
            raise ValueError("当前模型不是支持图片的鲁班零号基座")
    except (RuntimeError, ValueError) as error:
        reason = str(error)
    return {"available": not reason, "reason": reason, "round_budget": ROUND_BUDGET,
            "engine": "luban-image-libtorch-resident-v1", "candidate_only": True,
            "required_available_bytes": REQUIRED_AVAILABLE_BYTES}


@contextmanager
def load_for_learning(runtime):
    """One lifecycle migration, never a second owner or a per-sample reload."""
    runtime._ready()
    original_dtype, original_state = runtime.dtype, runtime.state
    original_paging = getattr(runtime, "cold_backbone", None) is not None
    reloaded = original_dtype != torch.bfloat16
    extra_lock = None
    try:
        # CPU's default runtime is F32. Release through the original Runtime
        # loader before loading the same files in their qualified BF16 policy.
        if reloaded:
            runtime.dtype = torch.bfloat16
            runtime.load(paging=False)
        held = getattr(runtime, "_model_process_lock", None)
        if held is None or held.fd is None or held.key != (runtime.identity, str(runtime.device)):
            raise RuntimeError("图片学习缺少当前 Runtime 的模型独占锁")
        extra_lock = ModelProcessLock(runtime.identity, "cpu" if str(runtime.device) == "mps" else "mps")
        with full_backbone(runtime):
            load_visual(runtime)
            expert = runtime.branch.expert
            runtime.branch.expert = None
            try:
                runtime.model.to("cpu")
                runtime.state = "learning"
                yield runtime.model
            finally:
                # Masters are separate candidates; the active owner was never
                # updated. Put the original owner/branch back once per round.
                try:
                    runtime.model.to(runtime.device)
                finally:
                    runtime.branch.expert = expert
    finally:
        if extra_lock is not None:
            extra_lock.close()
        if reloaded:
            runtime.dtype = original_dtype
            runtime.load(paging=original_paging)
        runtime.state = original_state


def _schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS modality_candidates (
        id TEXT PRIMARY KEY, identity TEXT NOT NULL, kind TEXT NOT NULL,
        generation INTEGER NOT NULL, body TEXT NOT NULL, updated REAL NOT NULL)""")


def _encode(body):
    return json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _records(runtime, samples):
    if not isinstance(samples, list) or not 2 <= len(samples) <= 32:
        raise ValueError("请提供图片学习及独立检验材料，最多 32 条")
    families, ids, records = {}, set(), []
    root = Path(runtime.memory.root).resolve()
    for raw in samples:
        if not isinstance(raw, dict):
            raise ValueError("图片学习材料格式无效")
        sample = dict(raw)
        kind, split = sample.get("kind"), sample.get("split")
        if kind not in {"image", "operation", "video"} or split not in {"training", "validation"}:
            raise ValueError("图片学习材料类型或拆分无效")
        if (not isinstance(sample.get("id"), str) or not sample["id"] or sample["id"] in ids
                or not isinstance(sample.get("family"), str) or not sample["family"]):
            raise ValueError("材料需要唯一编号和原始图片家族")
        ids.add(sample["id"])
        if families.setdefault(sample["family"], split) != split:
            raise ValueError("同源裁剪、缩放或操作会话不能跨学习和检验拆分")
        provenance = sample.get("provenance") or {}
        if not (sample.get("approved") is True or provenance.get("user_verified") is True):
            raise ValueError("请先核对图片问题和答案")
        if not (sample.get("training_allowed") is True or provenance.get("training_allowed") is True):
            raise ValueError("图片学习需要明确允许训练")
        for key in ("question", "answer"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError("图片学习需要完整问题和正确答案")
        def private_path(value):
            path = Path(value)
            if not path.is_absolute() or path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
                raise ValueError("图片必须来自当前私有资料目录")
            if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
                raise ValueError("图片资料目录不能经过符号链接")
            return str(path)
        if kind == "video":
            from .video_learning import frame_rows
            rows = frame_rows(sample)
            paths = [private_path(row["asset_path"]) for row in rows]
            if len(set(paths)) != len(paths):
                raise ValueError("视频联合学习不能重复同一画面文件")
            sample["video_frames"] = [{"asset_path": path, "timestamp_seconds": float(row["timestamp_seconds"])}
                                      for path, row in zip(paths, rows)]
        elif "asset_path" in sample:
            sample["asset_path"] = private_path(sample["asset_path"])
        elif not isinstance(sample.get("image"), dict):
            raise ValueError("图片材料缺少私有文件或已核对的动作前 RGB")
        records.append(sample)
    if len({x["kind"] for x in records}) != 1 or {x["split"] for x in records} != {"training", "validation"}:
        raise ValueError("每轮只学习一种能力，并保留独立检验材料")
    return records


def _binding(runtime, records):
    # Existing private material IDs bind the images; no extra file hashes.
    names = ("id", "kind", "split", "family", "question", "answer", "asset_path", "provenance", "video_frames")
    return {"protocol": "resident-video-learning-v1" if records[0]["kind"] == "video" else "resident-image-learning-v1",
            "identity": runtime.identity,
            "core_version": runtime.active_core, "expert_id": runtime.branch.expert_id,
            "records": [{key: record[key] for key in names if key in record} for record in records],
            "budget": ROUND_BUDGET, "learning_rate": .003, "precision": "BF16/FP32-master"}


def _candidate(runtime, binding, saved):
    if saved is not None:
        identifier = saved.get("id") if isinstance(saved, dict) else saved
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{12}", identifier):
            raise ValueError("候选断点编号无效")
        with runtime.memory.lock, runtime.memory.connect() as db:
            _schema(db)
            row = db.execute("SELECT * FROM modality_candidates WHERE id=? AND identity=?",
                             (identifier, runtime.identity)).fetchone()
        if row is None:
            raise ValueError("候选断点不存在或不属于当前模型")
        body = json.loads(row["body"])
        if body.get("binding") != binding:
            raise ValueError("候选断点与当前基座、已学版本或资料不一致")
        body["generation"] = row["generation"]
        return body
    return {"id": uuid.uuid4().hex[:12], "generation": 0, "step": 0, "round_target": ROUND_BUDGET,
            "state": "new", "binding": binding, "adopted": False}


def train_round(runtime, samples, saved_candidate=None, cancel=None, progress=None):
    """Train at most five updates, return a durable unaccepted candidate.

    saved_candidate is a prior result's {id}, never an arbitrary file path.
    A paused round resumes its remainder; a completed candidate starts the next
    fixed five-update round only upon this explicit call.
    """
    cancel, progress = cancel or (lambda: False), progress or (lambda value: None)
    with runtime.lock:
        runtime._ready()
        readiness = capability(runtime)
        if not readiness["available"]:
            raise RuntimeError(readiness["reason"])
        records = _records(runtime, samples)
        binding = _binding(runtime, records)
        body = _candidate(runtime, binding, saved_candidate)
        initial = body["step"]
        target = body["round_target"] if body["state"] == "paused" else initial + ROUND_BUDGET
        round_start = body.get("round_start", initial) if body["state"] == "paused" else initial
        if type(initial) is not int or type(target) is not int or not initial <= target <= initial + ROUND_BUDGET:
            raise ValueError("候选断点的固定轮次预算无效")
        if cancel():
            return {**body, "state": "paused", "actual_updates": 0,
                    "completed_steps": initial - round_start, "total_adam_step": initial,
                    "round_budget": ROUND_BUDGET, "quality_checked": False, "protection_checked": False,
                    "candidate_path": body["id"] if body.get("checkpoint") else None}
        required = REQUIRED_AVAILABLE_BYTES
        if records[0]["kind"] == "video":
            from .native_video_training import EXTRA_AVAILABLE_BYTES
            required += EXTRA_AVAILABLE_BYTES
        if psutil.virtual_memory().available < required:
            if records[0]["kind"] == "video":
                raise RuntimeError("视频联合学习可用内存不足，1024词元完整预算约 10.35 GB；不会缩减模型或门槛")
            raise RuntimeError("图片学习可用内存不足，当前完整图文训练预算约 7.97 GB；不会缩减模型或门槛")
        directory = Path(runtime.store) / "image-training" / body["id"]
        if directory.is_symlink() or directory.parent.is_symlink():
            raise ValueError("图片候选目录不能是符号链接")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        generation = body["generation"]
        graph = directory / (uuid.uuid4().hex + ".graph.pt")
        engine = None
        cpu_rng = torch.get_rng_state()
        mps_rng = torch.mps.get_rng_state() if str(runtime.device) == "mps" else None
        began = time.monotonic()
        progress({"phase": "preparing", "step": initial - round_start, "steps": ROUND_BUDGET, "round_budget": ROUND_BUDGET})
        try:
            with load_for_learning(runtime) as owner:
                try:
                    processor = image_processor(runtime.loaded_path)
                    if records[0]["kind"] == "video":
                        from .native_video_training import make_wrapper, prepare_video_sample
                        wrapper = make_wrapper(owner)
                        prepared = [(x, *prepare_video_sample(owner, runtime.tokenizer, processor, x)) for x in records]
                        maximum = 1024
                    elif records[0]["kind"] == "operation":
                        from .native_operation_training import make_wrapper, prepare_operation
                        wrapper = make_wrapper(owner)
                        prepared = [(x, *prepare_operation(owner, runtime.tokenizer, processor, x)) for x in records]
                        maximum = 1024
                    else:
                        wrapper = MasterQA(owner)
                        prepared = [(x, *prepare_sample(owner, runtime.tokenizer, processor, x)) for x in records]
                        maximum = 64
                    parameters, selected = lower_graph(wrapper, prepared[0][1], graph)
                    with NativeEngine(graph, parameters, selected, maximum, cancel=cancel) as engine:
                        if initial:
                            checkpoint = directory / body["checkpoint"]
                            if checkpoint.parent != directory or checkpoint.is_symlink():
                                raise ValueError("候选检查点路径无效")
                            engine.restore(checkpoint, runtime.identity, initial)
                            rng = body.get("rng")
                            if not isinstance(rng, dict) or not isinstance(rng.get("cpu"), list):
                                raise ValueError("候选缺少完整 RNG 保存状态")
                            torch.set_rng_state(torch.tensor(rng["cpu"], dtype=torch.uint8))
                            if mps_rng is not None:
                                if not isinstance(rng.get("mps"), list):
                                    raise ValueError("候选缺少当前 MPS RNG 保存状态")
                                torch.mps.set_rng_state(torch.tensor(rng["mps"], dtype=torch.uint8))
                        training = [x for x in prepared if x[0]["split"] == "training"]
                        validation = [x for x in prepared if x[0]["split"] == "validation"]
    
                        def evaluate():
                            answers = []
                            for sample, values, labels in validation:
                                if cancel():
                                    break
                                tokens = engine.greedy(values, int((labels == -100).sum()))
                                eos = bool(tokens and tokens[-1] == 248046)
                                text = runtime.tokenizer.decode(tokens[:-1] if eos else tokens, clean_up_tokenization_spaces=False)
                                answers.append({"id": sample["id"], "answer": text, "EOS": eos,
                                                "correct": eos and text == sample["answer"]})
                            return answers
    
                        progress({"phase": "baseline", "step": initial - round_start, "steps": ROUND_BUDGET})
                        before = evaluate()
                        step = initial
                        losses = []
                        while step < target and not cancel():
                            sample, values, labels = training[step % len(training)]
                            progress({"phase": "training", "step": step - round_start, "steps": ROUND_BUDGET, "sample_id": sample["id"]})
                            losses.append(engine.step(values, labels, .003))
                            step += 1
                        progress({"phase": "validating", "step": step - round_start, "steps": ROUND_BUDGET})
                        after = []
                        name = uuid.uuid4().hex + ".pt"
                        path = directory / name
                        engine.save(path, runtime.identity)
                        os.chmod(path, 0o600)
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                        # Re-open through the same production validator at the save
                        # lifecycle boundary. No sources or per-sample files are hashed.
                        engine.restore(path, runtime.identity, step)
                        result = {**body, "generation": generation + 1, "step": step, "round_target": target,
                                  "round_start": round_start, "round_budget": ROUND_BUDGET,
                                  "state": "paused" if step < target else "awaiting_quality", "checkpoint": name,
                                  "previous_checkpoint": body.get("checkpoint"), "actual_updates": step - initial,
                                  "initial_step": initial, "round_complete": step == target, "adopted": False,
                                  "before": before, "after": after, "losses": losses,
                                  "learned": sum(x["correct"] for x in after), "validation_samples": len(validation),
                                  "retention_passed": False, "quality_admission": False,
                                  "completed_steps": step - round_start, "total_adam_step": step,
                                  "quality_checked": len(after) == len(validation), "protection_checked": False,
                                  "elapsed_seconds": time.monotonic() - began,
                                  "rng": {"cpu": torch.get_rng_state().tolist(),
                                          "mps": torch.mps.get_rng_state().tolist() if mps_rng is not None else None}}
                        with runtime.memory.lock, runtime.memory.connect() as db:
                            _schema(db)
                            if generation:
                                changed = db.execute("UPDATE modality_candidates SET generation=?,body=?,updated=? WHERE id=? AND identity=? AND generation=?",
                                                     (generation + 1, _encode(result), time.time(), body["id"], runtime.identity, generation))
                                if changed.rowcount != 1:
                                    raise ValueError("候选已被其他轮次更新，保留本次检查点但不替换指针")
                            else:
                                db.execute("INSERT INTO modality_candidates VALUES (?,?,?,?,?,?)",
                                           (body["id"], runtime.identity, records[0]["kind"], 1, _encode(result), time.time()))
                        result.update(candidate_path=body["id"], candidate_id=body["id"])
                        progress({"phase": "validating", "step": step - round_start, "steps": ROUND_BUDGET,
                                  "candidate": result.copy()})
                        try:
                            after = evaluate()
                        except (RuntimeError, ValueError) as error:
                            result["validation_error"] = str(error)
                        result.update(after=after, learned=sum(x["correct"] for x in after),
                                      quality_checked=len(after) == len(validation))
                        with runtime.memory.lock, runtime.memory.connect() as db:
                            changed = db.execute("UPDATE modality_candidates SET body=?,updated=? WHERE id=? AND identity=? AND generation=?",
                                                 (_encode(result), time.time(), body["id"], runtime.identity, generation + 1))
                            if changed.rowcount != 1:
                                raise ValueError("候选已被更新，不能覆盖检验结果")
                        progress({"phase": "complete" if step == target else "paused",
                                  "step": step - round_start, "steps": ROUND_BUDGET, "candidate": result.copy()})
                        return result
                finally:
                    # Release graph/borrowed weights before an F32 Runtime reload.
                    # Otherwise these caller locals retain the old full owner.
                    if engine is not None:
                        engine.close()
                        engine = None
                    owner = wrapper = parameters = prepared = None
        finally:
            if engine is not None:
                engine.close()
            torch.set_rng_state(cpu_rng)
            if mps_rng is not None:
                torch.mps.set_rng_state(mps_rng)
