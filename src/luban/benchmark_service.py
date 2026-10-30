"""Local evaluation jobs and server-scored publication, separate from training."""
from __future__ import annotations

import copy
import math
import re
import threading
import time

from .benchmark import BenchmarkStore, POLICY_HASH, SUITE_ID, SUITE_SHA256, MAX_SECONDS, _BudgetStop, model_version, run_benchmark


def validated_receipt(value, expected):
    if (not isinstance(value, dict) or value.get("model_version") != expected or value.get("suite_id") != SUITE_ID
            or value.get("policy_hash") != POLICY_HASH or value.get("suite_sha256") != SUITE_SHA256):
        raise ValueError("评测成绩与当前模型或条件不匹配")
    for key in ("receipt_id", "model_version", "suite_sha256", "policy_hash"):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError("评测成绩凭据无效")
    for key in ("benchmark", "generalization", "benchmark_delta", "generalization_delta"):
        number = value.get(key)
        if number is None and key.endswith("_delta"):
            continue
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not (-100 if key.endswith("_delta") else 0) <= number <= 100:
            raise ValueError("评测成绩格式无效")
    return copy.deepcopy(value)


class BenchmarkService:
    def __init__(self, runtime, leaderboard, home, command_lock, busy):
        self.runtime, self.leaderboard = runtime, leaderboard
        self.command_lock, self.busy = command_lock, busy
        self.store = BenchmarkStore(home)
        self.lock = threading.RLock()
        self.worker = None
        self.stopped = threading.Event()
        self.closed = threading.Event()
        self.state = {"state": "idle", "completed": 0, "total": 48, "phase": "", "error": None, "result": None}
        try:
            saved = self.store.read()
            if saved:
                receipt = validated_receipt(saved["receipt"], saved["receipt"]["model_version"])
                self.state["result"] = {**receipt, "elapsed_seconds": saved["metadata"].get("elapsed_seconds"), "published": False}
        except (OSError, ValueError, KeyError, TypeError):
            self.state["error"] = "已保存的评测记录不可读，请重新评测"

    def status(self):
        with self.lock:
            state = copy.deepcopy(self.state)
        state["result"] = self._current_result(state["result"])
        return state

    def _current_result(self, result):
        if not result or getattr(self.runtime, "model", None) is None or self.runtime.state != "ready":
            return None
        try:
            return result if model_version(self.runtime) == result["model_version"] else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def profile(self):
        fields = ("benchmark", "generalization", "benchmark_delta", "generalization_delta", "elapsed_seconds")
        with self.lock:
            result = copy.deepcopy(self.state["result"])
        # Do not hold the status lock while waiting for the model generation lock.
        result = self._current_result(result) or {}
        return {key: result.get(key) for key in fields}

    def start(self):
        with self.lock:
            if self.closed.is_set() or self.worker and self.worker.is_alive():
                raise RuntimeError("能力评测正在进行，请等待或停止后重试")
            if self.runtime.state != "ready" or self.busy():
                raise RuntimeError("模型忙碌，请在空闲时评测")
            if not self.leaderboard.data["username"]:
                raise ValueError("请先填写用户名")
            self.runtime._ready()
            model_version(self.runtime)  # Fail before reserving runtime or spawning a job.
            self.stopped.clear()
            self.state.update(state="queued", completed=0, phase="准备评测", error=None)
            self.runtime.state = "evaluating"
            self.worker = threading.Thread(target=self._work, daemon=True, name="luban-benchmark")
            self.worker.start()
            return {"state": "queued"}

    def _progress(self, update):
        with self.lock:
            self.state.update(update, phase="正在评测")

    def _work(self):
        budget = _BudgetStop(self.stopped, time.monotonic() + MAX_SECONDS)
        try:
            with self.lock:
                self.state.update(state="running", phase="获取评测题")
            expected = model_version(self.runtime)
            self.leaderboard.ensure_token()
            challenge = self.leaderboard._request("POST", authenticated=True,
                endpoint="/api/leaderboard/evaluation/challenge", body={"model_version": expected})
            budget.check()
            if self.stopped.is_set():
                raise InterruptedError("评测已停止，没有上传成绩")
            if "receipt" in challenge:
                receipt = validated_receipt(challenge["receipt"], expected)
                previous = self.store.read()
                metadata = previous["metadata"] if previous and previous["receipt"].get("receipt_id") == receipt["receipt_id"] else {}
            else:
                report = run_benchmark(self.runtime, challenge, self.stopped, self._progress,
                                       deadline_monotonic=budget.deadline)
                metadata = report["metadata"]
                # Actual test-generation tokens contribute once to the installation's usage.
                items = metadata.get("items", [])
                prompt = sum(row["prompt_tokens"] for row in items)
                completion = sum(row["completion_tokens"] for row in items)
                self.runtime.memory.record_usage("local", {"prompt_tokens": prompt, "completion_tokens": completion,
                    "total_tokens": prompt + completion, "estimated": False})
                budget.check()
                if self.stopped.is_set() or model_version(self.runtime) != expected:
                    raise InterruptedError("评测已停止或模型已变化，没有上传成绩")
                with self.lock:
                    self.state["phase"] = "核验成绩"
                response = self.leaderboard._request("POST", authenticated=True, endpoint="/api/leaderboard/evaluation",
                    body={key: report[key] for key in ("challenge_id", "model_version", "answers")})
                receipt = validated_receipt(response.get("receipt", response), expected)
                if receipt["suite_sha256"] != challenge["suite_sha256"]:
                    raise ValueError("评测题库已变化，没有发布成绩")
            budget.check()
            self.store.save(receipt, metadata)
            if self.stopped.is_set() or model_version(self.runtime) != expected:
                raise InterruptedError("评测已停止或模型已变化，没有发布成绩")
            budget.check()
            self.leaderboard.publish(receipt)
            with self.lock:
                self.state.update(state="complete", completed=48, phase="评测已上报", error=None,
                    result={**receipt, "elapsed_seconds": metadata.get("elapsed_seconds"), "published": True})
        except Exception as error:
            with self.lock:
                self.state.update(state="error", phase="", error=str(error))
        finally:
            if self.runtime.state == "evaluating":
                self.runtime.state = "ready"

    def pause(self, timeout=15):
        self.stopped.set()
        worker = self.worker
        if worker and worker is not threading.current_thread():
            worker.join(timeout)
            if worker.is_alive():
                raise RuntimeError("能力评测正在停止，请稍后重试")
        return {"ok": True}

    def after_upgrade(self):
        """A committed core schedules exactly one bounded, idle evaluation attempt."""
        if (not self.leaderboard.data["username"] or not self.leaderboard.data["published_profile"]
                or self.closed.is_set()):
            return
        def idle():
            deadline = time.monotonic() + 3600
            while not self.closed.wait(2) and time.monotonic() < deadline:
                if self.runtime.state == "ready" and not self.busy() and self.command_lock.acquire(blocking=False):
                    try:
                        self.start()
                    except (RuntimeError, ValueError):
                        continue
                    finally:
                        self.command_lock.release()
                    return
        threading.Thread(target=idle, daemon=True, name="luban-post-evolution-test").start()

    def close(self):
        self.closed.set()
        self.pause()
