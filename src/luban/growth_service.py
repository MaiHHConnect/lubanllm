"""Prospective personal learning comparisons; never imports evaluation into training."""
from __future__ import annotations

import copy
import hashlib
import math
import re
import threading
import time
from pathlib import Path

from .benchmark import canonical, model_version
from .growth import POLICY_HASH, eligible_cases, eligible_skills, personal_model_version, run_growth, original_retest
from .settings import private_json, require_storage

DOMAINS = ("mathematics", "programming", "data")
BUDGET_SECONDS = 300


def validated_growth(value, expected=None, public=False):
    if not isinstance(value, dict) or (not public and value.get("suite_id") != "growth-v1") or value.get("domain") not in DOMAINS:
        raise ValueError("个人成长评测协议或领域无效")
    for key in (() if public else ("receipt_id", "base_identity", "before_model_version", "model_version", "policy_hash", "suite_sha256")):
        if not isinstance(value.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError("个人成长凭证无效")
    if expected and any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("个人成长成绩与模型或冻结题目不匹配")
    if not public and value.get("policy_hash") != POLICY_HASH:
        raise ValueError("个人成长评测条件已变化，请重新建立学习前基线")
    if (not public and value["model_version"] == value["before_model_version"]) or value.get("question_count") != 24:
        raise ValueError("成长必须有不同版本的完整前后对照")
    for key in ("baseline_correct", "current_correct", "wrong_to_correct", "correct_to_wrong"):
        if type(value.get(key)) is not int or not 0 <= value[key] <= 24:
            raise ValueError("个人成长答题计数无效")
    before, after = value["baseline_correct"], value["current_correct"]
    if (after - before != value["wrong_to_correct"] - value["correct_to_wrong"]
            or value["wrong_to_correct"] > 24 - before or value["correct_to_wrong"] > before
            or value.get("eligible") is not (4 <= before <= 20)):
        raise ValueError("个人成长对照计数不一致")
    calculations = {"baseline_score": before / 24 * 100, "current_score": after / 24 * 100,
                    "net_delta": (after - before) / 24 * 100,
                    "gap_recovery": value["wrong_to_correct"] / (24 - before) * 100 if before < 24 else None,
                    "retention": (before - value["correct_to_wrong"]) / before * 100 if before else None}
    for key, expected_value in calculations.items():
        actual = value.get(key)
        if expected_value is None:
            if actual is not None:
                raise ValueError("缺少基线的比例不能补造")
        elif (isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual)
              or abs(actual - round(expected_value, 2)) > .011):
            raise ValueError("个人成长评分与计数不一致")
    check = value.get("retention_check")
    if check is not None:
        if (not isinstance(check, dict) or type(check.get("correct")) is not int or not 0 <= check["correct"] <= 24
                or type(check.get("retained_correct")) is not int or not 0 <= check["retained_correct"] <= after
                or check["retained_correct"] > check["correct"]):
            raise ValueError("延迟保持复查计数无效")
        expected_retention = round(check["retained_correct"] / after * 100, 2) if after else None
        if check.get("retention") != expected_retention:
            raise ValueError("延迟保持复查成绩无效")
    return copy.deepcopy(value)


def timestamp(value):
    from datetime import datetime
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    raise ValueError("个人评测时间格式无效")


class GrowthService:
    def __init__(self, runtime, leaderboard, command_lock, busy):
        self.runtime, self.leaderboard, self.command_lock, self.busy = runtime, leaderboard, command_lock, busy
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.closed = threading.Event()
        self.worker = None
        self.started_at = None
        self.elapsed_seconds = 0
        self.timer = None
        self.pause_generation = 0
        self.pretraining_session = None
        self.state = {"state": "idle", "completed": 0, "total": 24, "phase": "", "error": None,
                      "result": None, "message": "学习前建立个人成长基线，学习后比较独立新题"}
        self.quiz_state = {"state": "idle", "completed": 0, "total": 0, "result": None, "error": None}

    @property
    def path(self):
        return Path(self.runtime.memory.root) / "app/growth/rounds.json"

    def _read(self):
        import json
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("个人成长资料不能是符号链接")
        if not self.path.exists():
            return {"enabled": False, "active": None, "history": [], "reserved_prompts": []}
        value = json.loads(self.path.read_text())
        if (not isinstance(value, dict) or type(value.get("enabled")) is not bool
                or not isinstance(value.get("history"), list) or not isinstance(value.get("reserved_prompts"), list)
                or not isinstance(value.get("automatic_attempts", []), list)
                or any(not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key)
                       for key in value.get("automatic_attempts", []))):
            raise ValueError("个人成长资料损坏，请保留原文件")
        return value

    def _save(self, data):
        require_storage(self.runtime.memory.root)
        private_json(self.path, data)

    def _enabled(self):
        try:
            return self._read()["enabled"]
        except (OSError, ValueError, TypeError, AttributeError):
            return False

    def _version(self):
        if getattr(self.runtime, "model", None) is None or self.runtime.state != "ready":
            return None
        return personal_model_version(self.runtime)

    def profile(self):
        empty = {"growth": None, "growth_profile": [{"domain": d, "growth": None} for d in DOMAINS]}
        try:
            version = self._version()
            if version is None:
                return empty
            data = self._read()
            rows = [validated_growth(r["receipt"]) for r in data["history"]
                    if r["receipt"].get("model_version") == version and r["receipt"].get("policy_hash") == POLICY_HASH]
            latest = {row["domain"]: row for row in rows}
            return {"growth": rows[-1] if rows else None,
                    "growth_profile": [{"domain": d, "growth": latest.get(d)} for d in DOMAINS]}
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
            return empty

    def publication(self):
        try:
            if self.runtime.state not in {"ready", "evaluating"}:
                return None
            version = personal_model_version(self.runtime)
            return next((validated_growth(row["receipt"]) for row in reversed(self._read()["history"])
                         if row["receipt"].get("model_version") == version
                         and row["receipt"].get("policy_hash") == POLICY_HASH), None)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
            return None

    def status(self):
        with self.lock:
            state = copy.deepcopy(self.state)
            elapsed = (time.monotonic() - self.started_at if self.timer is not None
                       and self.started_at is not None else self.elapsed_seconds)
        state.update(budget_seconds=BUDGET_SECONDS, elapsed_seconds=round(max(0, elapsed), 1))
        state.update(domains=list(DOMAINS), eligible_skills=[])
        if state["state"] not in {"queued", "running"} and self.runtime.state == "ready":
            try:
                state["eligible_skills"] = eligible_skills(self.runtime)
                current = self.profile()["growth"]
                state["result"] = current
            except (OSError, ValueError, RuntimeError, AttributeError):
                pass
        return state

    def start(self):
        with self.lock:
            return self._start()

    def quiz_status(self):
        from .model_quiz import quiz_status
        return quiz_status(self)

    def start_quiz(self):
        from .model_quiz import start_quiz
        with self.lock:
            return start_quiz(self)

    def _start(self):
        if self.closed.is_set() or self.worker and self.worker.is_alive() or self.runtime.state != "ready" or self.busy():
            raise RuntimeError("模型忙碌，请在空闲时评测")
        if not self.leaderboard.data["username"]:
            raise ValueError("请先填写用户名")
        self.runtime._ready()
        skills = eligible_skills(self.runtime)
        if not skills:
            self.state.update(state="waiting_materials", message="暂缺可独立验证的新题，尚不计分", error=None)
            return self.status()
        data = self._read()
        data["enabled"] = True
        self._save(data)
        self.stop_event = threading.Event()
        self.state.update(state="queued", error=None, completed=0, phase="准备个人成长评测")
        self.runtime.state = "evaluating"
        self.worker = threading.Thread(target=self._work, daemon=True, name="luban-growth")
        try:
            self.worker.start()
        except Exception:
            self.worker = None
            self.stop_event.set()
            self.runtime.state = "ready"
            raise
        return {"state": "queued"}

    def _progress(self, update):
        with self.lock:
            self.state.update(update)

    def _measure(self, challenge):
        self._check_budget()
        try:
            report = run_growth(self.runtime, challenge, self.stop_event, self._progress,
                                deadline_monotonic=self.started_at + BUDGET_SECONDS)
        except InterruptedError:
            self._check_budget()
            raise
        self._check_budget()
        self._record_usage(report)
        self._check_budget()
        return report

    def _begin_budget(self):
        with self.lock:
            self.started_at = time.monotonic()
            self.timer = threading.Timer(BUDGET_SECONDS, self.stop_event.set)
            self.timer.daemon = True
            self.timer.start()

    def _end_budget(self):
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            if self.started_at is not None:
                self.elapsed_seconds = max(0, time.monotonic() - self.started_at)

    def _check_budget(self):
        if self.started_at is not None and time.monotonic() - self.started_at >= BUDGET_SECONDS:
            raise TimeoutError("本轮评测达到 5 分钟上限，已保留基线，不上传残缺成绩")
        if self.stop_event.is_set() or self.closed.is_set():
            raise InterruptedError("评测已停止，没有上传成绩")

    def _record_usage(self, report):
        counts = report.get("metadata", {}).get("items", [])
        if counts:
            prompt = sum(item["prompt_tokens"] for item in counts)
            completion = sum(item["completion_tokens"] for item in counts)
            self.runtime.memory.record_usage("local", {"prompt_tokens": prompt, "completion_tokens": completion,
                                                     "total_tokens": prompt + completion})

    def _original(self, cases):
        if not cases or self.stop_event.is_set():
            return {"private": True, "counts_toward_public_growth": False, "state": "not_completed"}
        self._check_budget()
        result = original_retest(self.runtime, cases, self.stop_event,
                                 deadline_monotonic=self.started_at + BUDGET_SECONDS)
        self._check_budget()
        self._record_usage(result)
        return result

    def _compare(self, *, publish=False, preparing=False):
        self._check_budget()
        data = self._read()
        if data["active"] and data["active"]["challenge"].get("policy_hash") != POLICY_HASH:
            data["active"] = None
            self._save(data)
            self.state.update(state="expired", message="旧评测条件已失效，下次学习前重新建立基线，不能补造")
            return
        skills = eligible_skills(self.runtime)
        if not skills:
            self.state.update(state="waiting_materials", message="暂缺可独立验证的新题，尚不计分")
            return
        version = personal_model_version(self.runtime)
        active = data["active"]
        cases = eligible_cases(self.runtime)
        fingerprint = hashlib.sha256(canonical(cases).encode()).hexdigest()
        originals = {}
        for case in cases:
            originals.setdefault(case.get("skill"), case)
        cases = list(originals.values())  # Private original retest is diagnostic, one per supported skill.
        previous = next((r for r in reversed(data["history"]) if r["receipt"]["model_version"] == version
                         and r["receipt"].get("policy_hash") == POLICY_HASH), None)
        if not preparing and not active and previous and previous.get("materials_sha256") == fingerprint:
            if (previous["receipt"].get("retention_check") is None
                    and timestamp(previous["receipt"]["completed_at"]) + 86400 <= time.time()):
                self.state.update(state="running", phase="复查已学能力是否保留", completed=0)
                report = self._measure({**previous["challenge"], "model_version": version})
                check = self.leaderboard._request("POST", authenticated=True, endpoint="/api/leaderboard/growth/retention",
                    body={"receipt_id": previous["receipt"]["receipt_id"], "model_version": version, "answers": report["answers"]})
                if check.get("receipt_id") != previous["receipt"]["receipt_id"] or check.get("model_version") != version:
                    raise ValueError("延迟保持复查与本轮模型不一致")
                previous["receipt"]["retention_check"] = check
                previous["receipt"] = validated_growth(previous["receipt"], {"model_version": version})
                self._save(data)
            if publish:
                self._publish(previous["receipt"])
            self.state.update(state="complete", result=previous["receipt"], phase="已有本版本成绩，未重复计分", error=None)
            return
        self.leaderboard.ensure_token()
        self._check_budget()
        if active and timestamp(active["baseline"]["expires_at"]) <= time.time():
            data["active"] = None
            self._save(data)
            self.state.update(state="expired", message="上一份基线已过期，本轮需重新建立基线，不能补造学习前成绩")
            return
        if active:
            challenge = active["challenge"]
            before = active["baseline"]
            if version == challenge["model_version"]:
                self.state.update(state="baseline", phase="已建立成长基线，学习后自动评测", error=None)
                return
            self.state.update(state="running", phase="核对学习后的独立新题", completed=0)
            current_challenge = {**challenge, "model_version": version}
            report = self._measure(current_challenge)
            receipt = self.leaderboard._request("POST", authenticated=True, endpoint="/api/leaderboard/growth/after",
                body={"challenge_id": challenge["challenge_id"], "model_version": version, "answers": report["answers"]})
            receipt = validated_growth(receipt, {"model_version": version, "before_model_version": challenge["model_version"],
                "domain": challenge["domain"], "base_identity": self.runtime.identity,
                "policy_hash": challenge["policy_hash"], "suite_sha256": challenge["suite_sha256"]})
            data["history"].append({"receipt": receipt, "materials_sha256": fingerprint,
                "metadata": report.get("metadata", {}), "original_before": active.get("original_before"),
                "original_after": None, "challenge": challenge})
            data["active"] = None
            self._save(data)
            data["history"][-1]["original_after"] = self._original(active["cases"])
            self._save(data)
            self._check_budget()
            if publish:
                self._publish(receipt)
            self.state.update(state="complete", result=receipt, phase="个人成长对照已完成", error=None)
            if preparing:
                self._compare(preparing=True)
        else:
            self.state.update(state="running", phase="记录学习前的独立新题", completed=0)
            challenge = self.leaderboard._request("POST", authenticated=True, endpoint="/api/leaderboard/growth/challenge",
                body={"model_version": version, "base_identity": self.runtime.identity, "skills": skills})
            if challenge.get("policy_hash") != POLICY_HASH:
                raise ValueError("个人成长评测条件不兼容，不能建立基线")
            # Reserve before measuring, so cancellation never allows training on these tasks.
            prompts = [q["prompt"] for q in challenge["questions"]]
            import unicodedata
            normalize = lambda text: "".join(unicodedata.normalize("NFKC", text).split())
            used = {normalize(r["prompt"]) for r in self.runtime.memory.samples()}
            if used & {normalize(p) for p in prompts}:
                raise ValueError("新题与已有学习材料重合，本轮不计分")
            data["reserved_prompts"] = sorted(set(data["reserved_prompts"] + prompts))
            self._save(data)
            report = self._measure(challenge)
            baseline = self.leaderboard._request("POST", authenticated=True, endpoint="/api/leaderboard/growth/before",
                body={"challenge_id": challenge["challenge_id"], "model_version": version, "answers": report["answers"]})
            if (baseline.get("challenge_id") != challenge["challenge_id"] or baseline.get("before_model_version") != version
                    or baseline.get("question_count") != 24 or type(baseline.get("baseline_correct")) is not int
                    or not 0 <= baseline["baseline_correct"] <= 24 or timestamp(baseline.get("expires_at")) <= time.time()):
                raise ValueError("学习前基线凭证不匹配")
            data["active"] = {"challenge": challenge, "baseline": baseline, "cases": cases,
                              "materials_sha256": fingerprint, "original_before": None}
            self._save(data)
            original = self._original(cases)
            data["active"]["original_before"] = original
            self._save(data)
            self._check_budget()
            self.state.update(state="baseline", phase="已建立成长基线，学习后自动评测", error=None)

    def _publish(self, receipt):
        self._check_budget()
        core = model_version(self.runtime)
        self._check_budget()
        self.leaderboard.publish(growth=receipt, core_version=core)

    def _work(self):
        try:
            self._begin_budget()
            self._compare(publish=True)
        except InterruptedError:
            self.state.update(state="cancelled", phase="评测已停止，未发布部分分数")
        except Exception as error:
            self.state.update(state="error", error=str(error), phase="成长评测未完成")
        finally:
            self._end_budget()
            if self.runtime.state == "evaluating":
                self.runtime.state = "ready"

    def before_training(self, session):
        if self.closed.is_set() or not self._enabled() or session.body.get("checkpoint"):
            return
        with self.lock:
            saved = self.runtime.training_session
            if session.pause_requested:
                from .training_checkpoint import TrainingPaused
                raise TrainingPaused("对话优先，尚未开始成长基线")
            self.stop_event = threading.Event()
            self.pretraining_session = session
            self.runtime.training_session = None  # No optimizer or candidate exists yet.
        try:
            self._begin_budget()
            self._compare(preparing=True)
        except InterruptedError:
            from .training_checkpoint import TrainingPaused
            raise TrainingPaused("对话优先，成长基线中止后保存待学任务")
        except Exception as error:
            # Scoring/network failure must not prevent ordinary authorized learning.
            self.state.update(state="error", error=str(error), phase="未完成成长基线，本轮学习继续")
        finally:
            self._end_budget()
            with self.lock:
                self.runtime.training_session = saved
                self.pretraining_session = None
                if session.pause_requested:
                    from .training_checkpoint import TrainingPaused
                    raise TrainingPaused("对话优先，成长基线中止后保存待学任务")

    def idle(self):
        with self.lock:
            return self._idle()

    def _idle(self):
        if self.closed.is_set() or not self._enabled() or self.runtime.state != "ready" or self.busy():
            return False
        try:
            data, version = self._read(), personal_model_version(self.runtime)
            obligations = ([data["active"]["challenge"]["challenge_id"]] if data["active"]
                           and data["active"]["challenge"]["model_version"] != version else [])
            obligations += [r["receipt"]["receipt_id"] for r in data["history"]
                            if r["receipt"]["model_version"] == version and not r["receipt"].get("retention_check")
                            and r["receipt"].get("policy_hash") == POLICY_HASH
                            and timestamp(r["receipt"]["completed_at"]) + 86400 <= time.time()]
            attempt = hashlib.sha256(canonical([version, obligations]).encode()).hexdigest()
            if (not obligations or attempt in data.get("automatic_attempts", [])
                    or not self.command_lock.acquire(blocking=False)):
                return False
            try:
                data.setdefault("automatic_attempts", []).append(attempt)
                self._save(data)
                self.start()
                return True
            finally:
                self.command_lock.release()
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError):
            return False

    def after_upgrade(self):
        if self.closed.is_set() or not self._enabled():
            return
        generation = self.pause_generation
        def idle():
            deadline = time.monotonic() + 3600
            while (not self.closed.is_set() and generation == self.pause_generation
                   and time.monotonic() < deadline):
                if self.runtime.state == "ready" and not self.busy():
                    with self.lock:
                        if generation == self.pause_generation:
                            self.idle()
                    return
                self.closed.wait(.5)
        threading.Thread(target=idle, daemon=True, name="luban-growth-after-learning").start()

    def pause(self):
        with self.lock:
            self.pause_generation += 1
            self.stop_event.set()
            if self.pretraining_session is not None:
                self.pretraining_session.pause_requested = True
            worker = self.worker
        if worker and worker is not threading.current_thread():
            worker.join(15)
            if worker.is_alive():
                raise RuntimeError("正在停止个人评测，请稍后重试")
        return {"ok": True}

    def close(self):
        self.closed.set()
        return self.pause()
