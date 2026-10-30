"""Read-only, core-only ranking evaluation; the server owns all scoring."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import threading
import time
from pathlib import Path

import psutil
import torch
from transformers import StoppingCriteria

from .settings import private_json

SUITE_ID = "luban-ranking-v1"
SUITE_SHA256 = "a9b59962d7bca1739abad95de05a112b15c81932a8c2fc78d78f188e3337d4ac"
SYSTEM_PROMPT = ("Answer each task directly. For multiple choice, output only the single uppercase "
                 "option letter. For numeric tasks, output only the number. For JSON tasks, "
                 "output only the requested JSON value. Do not explain your answer.")
POLICY = {"temperature": 0, "seed": 42, "max_output_tokens": 512, "route": "local_core",
          "teacher_assistance": False, "thinking": "off", "shots": 0,
          "system_prompt": SYSTEM_PROMPT}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


POLICY_HASH = hashlib.sha256(canonical(POLICY).encode()).hexdigest()
MAX_SECONDS = 300


class _BudgetStop:
    """Cancellation and deadline checks shared by benchmark and personal growth."""
    def __init__(self, event, deadline_monotonic):
        if deadline_monotonic is None:
            deadline_monotonic = time.monotonic() + MAX_SECONDS
        if (isinstance(deadline_monotonic, bool) or not isinstance(deadline_monotonic, (int, float))
                or not math.isfinite(deadline_monotonic)):
            raise ValueError("评测截止时间无效")
        self.event = event if event is not None else threading.Event()
        self.deadline = min(deadline_monotonic, time.monotonic() + MAX_SECONDS)

    def is_set(self):
        return time.monotonic() >= self.deadline or self.event.is_set()

    def set(self):
        self.event.set()

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("评测超过5分钟，没有提交不完整成绩")
        if self.event.is_set():
            raise InterruptedError("评测已取消，没有上报")


def _sha_file(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("评测模型文件缺失或为符号链接")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_version(runtime):
    """Bind the loader-verified base and selected immutable core, excluding branches."""
    with runtime.lock:
        identity = getattr(runtime, "identity", None)
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise ValueError("基座模型身份未核验，不能评测")
        store = Path(runtime.store)
        pointer = store / "active-core.json"
        if store.is_symlink() or pointer.is_symlink():
            raise ValueError("评测核心路径不能是符号链接")
        active = getattr(runtime, "active_core", None)
        core_sha = None
        if active is None:
            if pointer.exists():
                raise ValueError("评测内存核心与活动指针不同，请重新加载模型")
        else:
            if not isinstance(active, str) or not re.fullmatch(r"core-[a-f0-9]{12}", active):
                raise ValueError("评测核心版本无效")
            selected = json.loads(pointer.read_text())
            metadata_path = store / (active + ".json")
            if metadata_path.is_symlink():
                raise ValueError("评测核心元数据不能是符号链接")
            metadata = json.loads(metadata_path.read_text())
            core_sha = _sha_file(store / (active + ".safetensors"))
            if (selected.get("identity") != identity or selected.get("version") != active
                    or selected.get("sha256") != core_sha or metadata.get("identity") != identity
                    or metadata.get("version") != active or metadata.get("sha256") != core_sha):
                raise ValueError("评测核心权重与版本记录不一致")
        return hashlib.sha256(canonical({"base_identity": identity, "core_sha256": core_sha,
                                         "inference": runtime.inference_identity(),
                                         "route": "local_core"}).encode()).hexdigest()


class _StopRequested(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def _challenge(challenge):
    if not isinstance(challenge, dict) or challenge.get("suite_id") != SUITE_ID:
        raise ValueError("评测题库版本不兼容，请更新应用")
    try:
        same_policy = canonical(challenge.get("policy")) == canonical(POLICY)
    except (TypeError, ValueError):
        same_policy = False
    if not same_policy or challenge.get("policy_hash") != POLICY_HASH:
        raise ValueError("评测条件不一致，不能上榜")
    if challenge.get("suite_sha256") != SUITE_SHA256:
        raise ValueError("评测题库摘要不兼容，请更新应用")
    for key in ("model_version", "suite_sha256"):
        if not isinstance(challenge.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", challenge[key]):
            raise ValueError("评测版本摘要无效")
    if not isinstance(challenge.get("challenge_id"), str) or not 1 <= len(challenge["challenge_id"]) <= 128:
        raise ValueError("评测挑战无效")
    questions = challenge.get("questions")
    if not isinstance(questions, list) or len(questions) != 48:
        raise ValueError("评测题目不完整")
    seen = set()
    for question in questions:
        if (not isinstance(question, dict) or set(question) - {"id", "kind", "category", "prompt"}
                or not isinstance(question.get("id"), str)
                or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", question["id"])
                or question["id"] in seen or not isinstance(question.get("prompt"), str)
                or not 1 <= len(question["prompt"]) <= 16000):
            raise ValueError("评测题目格式无效")
        seen.add(question["id"])
    return questions


def run_benchmark(runtime, challenge, stop=None, progress=None, *, deadline_monotonic=None):
    """Generate answers without routing, teacher calls, chat logs or learning imports.

    Completion truncation is preserved as ``length`` for zero-credit scoring.
    Cancellation, empty output, errors and changing versions abort the whole run.
    """
    stop = _BudgetStop(stop, deadline_monotonic)
    stop.check()
    questions = _challenge(challenge)
    progress = progress or (lambda value: None)
    with runtime.lock:
        stop.check()
        if (runtime.state not in {"ready", "evaluating"} or getattr(runtime, "training_session", None) is not None
                or runtime.model is None or runtime.tokenizer is None):
            raise RuntimeError("模型尚未就绪或正在学习，不能评测")
        runtime._ready()
        expected = challenge["model_version"]
        if model_version(runtime) != expected:
            raise ValueError("模型已改变，请重新开始评测")
        old_state = runtime.state
        branch = getattr(runtime, "branch", None)
        old_expert = branch.expert if branch is not None else None
        training_flags = [(module, module.training) for module in runtime.model.modules()]
        started = time.perf_counter()
        before_rss = psutil.Process().memory_info().rss
        answers, timings = [], []
        runtime.state = "evaluating"
        if branch is not None:
            branch.expert = None
        try:
            with torch.inference_mode(), torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(42)
                runtime.model.eval()
                for index, question in enumerate(questions):
                    stop.check()
                    if model_version(runtime) != expected:
                        raise ValueError("评测中模型已改变，没有提交分数")
                    tick = time.perf_counter()
                    tokens = runtime._tokens([{"role": "system", "content": SYSTEM_PROMPT},
                                              {"role": "user", "content": question["prompt"]}], "off", output_reserve=512)
                    stop.check()
                    output = runtime.generate_tokens(tokens, 512, temperature=0.0, stop=stop)
                    stop.check()
                    generated = output[0, tokens.shape[1]:]
                    text = runtime.tokenizer.decode(generated, skip_special_tokens=True).strip()
                    if not text or len(text) > 8192:
                        raise ValueError("模型评测回复为空或过长，没有提交分数")
                    eos = getattr(getattr(runtime.model, "generation_config", None), "eos_token_id", None)
                    if eos is None:
                        eos = runtime.tokenizer.eos_token_id
                    eos_ids = eos if isinstance(eos, list) else [eos]
                    complete = generated.numel() > 0 and int(generated[-1]) in eos_ids
                    answers.append({"id": question["id"], "text": text,
                                    "finish_reason": "stop" if complete else "length"})
                    timings.append({"id": question["id"], "seconds": time.perf_counter() - tick,
                                    "prompt_tokens": int(tokens.shape[1]),
                                    "completion_tokens": int(generated.numel())})
                    progress({"completed": index + 1, "total": len(questions)})
                stop.check()
                if model_version(runtime) != expected:
                    raise ValueError("评测中模型已改变，没有提交分数")
            stop.check()
            return {"challenge_id": challenge["challenge_id"], "model_version": expected,
                    "answers": answers, "metadata": {"suite_id": SUITE_ID,
                    "suite_sha256": challenge["suite_sha256"], "policy_hash": POLICY_HASH,
                    "elapsed_seconds": time.perf_counter() - started, "items": timings,
                    "device": str(runtime.device), "precision": str(runtime.dtype),
                    "inference": runtime.inference_identity(),
                    "platform": platform.platform(), "torch_version": torch.__version__,
                    "rss_before_bytes": before_rss, "rss_after_bytes": psutil.Process().memory_info().rss,
                    "memory_measurement": "process RSS before/after; not peak or accelerator memory",
                    "efficiency_scope": "local diagnostic only; hardware is not normalized",
                    "teacher_calls": 0, "route": "local_core", "branch_enabled": False,
                    "time_budget_seconds": MAX_SECONDS,
                    "timeout_boundary": "checked per generated token; cannot preempt an in-flight kernel or weight load"}}
        finally:
            if branch is not None:
                branch.expert = old_expert
            for module, training in training_flags:
                module.training = training
            runtime.state = old_state


class BenchmarkStore:
    """Last server receipt and local measurements; no answers become training data."""
    def __init__(self, root):
        self.path = Path(root) / "ranking-evaluation.json"

    def read(self):
        if self.path.is_symlink():
            raise ValueError("评测记录不能是符号链接")
        if not self.path.exists():
            return None
        value = json.loads(self.path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("receipt"), dict):
            raise ValueError("评测记录损坏")
        return value

    def save(self, receipt, metadata):
        if not isinstance(receipt, dict) or not isinstance(metadata, dict):
            raise ValueError("评测记录无效")
        if any(path.is_symlink() for path in (self.path, self.path.parent,
                                               self.path.with_name(self.path.name + ".tmp"))):
            raise ValueError("评测记录路径不能是符号链接")
        private_json(self.path, {"receipt": receipt, "metadata": metadata})
