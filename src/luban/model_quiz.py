"""Private, finite local quizzes. Reuse the growth lease and closed-book runner."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import threading
import unicodedata

from .benchmark import _BudgetStop, canonical
from .generation_checks import evaluate_generation
from .growth import POLICY_HASH, _run, eligible_cases, personal_model_version
from .settings import private_json


def _path(service):
    return service.path.parent / "quiz.json"


def _read(service):
    path = _path(service)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("小测记录不能是符号链接")
    if not path.exists():
        return None
    if path.stat().st_size > 200_000:
        raise ValueError("小测记录过大，请保留原文件")
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or value.get("schema") != 1 or not isinstance(value.get("cases"), list)
            or not 1 <= len(value["cases"]) <= 10 or not isinstance(value.get("summary"), dict)
            or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                   or type(row.get("passed")) is not bool for row in value["cases"])
            or len({row["id"] for row in value["cases"]}) != len(value["cases"])):
        raise ValueError("小测记录损坏，请保留原文件")
    return value


def quiz_status(service):
    with service.lock:
        state = copy.deepcopy(service.quiz_state)
    if state["state"] == "idle":
        try:
            saved = _read(service)
            state["result"] = saved.get("summary") if saved else None
        except (OSError, ValueError) as error:
            state.update(state="error", error=str(error))
    return state


def _normalize(value):
    return "".join(unicodedata.normalize("NFKC", value).split())


def _fresh(skill, rng):
    if skill.startswith("arithmetic_"):
        a, b = rng.randrange(10, 1000), rng.randrange(10, 1000)
        op = {"arithmetic_add": "+", "arithmetic_subtract": "-", "arithmetic_multiply": "*"}[skill]
        answer = a + b if op == "+" else a - b if op == "-" else a * b
        return f"Compute {a} {op} {b}. Output only the number.", {"type": "semantic", "validator": "integer", "expected": answer}
    if skill == "uppercase":
        text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8))
        return f"Convert '{text}' to uppercase. Output only the result.", {"type": "exact", "expected": text.upper()}
    value = rng.randrange(10, 1_000_000)
    return f'Extract the value of key "value" from JSON {{"value":{value}}}. Output only the JSON value.', {"type": "semantic", "validator": "json", "expected": value}


def _questions(service):
    data = service._read()
    # Multi-turn source history can contain an earlier reference answer; never feed it to this quiz.
    cases = [case for case in eligible_cases(service.runtime)
             if len(case["messages"]) == 1 and case["messages"][0]["role"] == "user"]
    # A wrong teacher judgement is not evidence that the model failed a closed-book test.
    failed = set()
    reports = [row.get("original_before") for row in data["history"]]
    if data.get("active"):
        reports.append(data["active"].get("original_before"))
    for report in reports:
        if (isinstance(report, dict) and report.get("metadata", {}).get("teacher_calls") == 0
                and report.get("metadata", {}).get("reference") is False):
            failed.update(row["source_id"] for row in report.get("cases", [])
                          if row.get("passed") is False and isinstance(row.get("source_id"), str))
    cases.sort(key=lambda case: (case["source_id"] not in failed, case["source_id"]))
    selected, skills, seen = [], set(), set()
    for case in cases:
        normalized = _normalize(case["prompt"])
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append({"id": f"original-{len(selected)}", "prompt": case["prompt"], "messages": case["messages"],
                         "check": case["generation_check"], "kind": "original", "skill": case["skill"],
                         "previously_wrong": case["source_id"] in failed})
        skills.add(case["skill"])
        if len(selected) == 5:
            break
    if not selected:
        return []
    # Stable per source set, so model versions can be compared on exactly the same tasks.
    rng = random.Random(canonical([(case["id"], case["prompt"], case["check"]) for case in selected]))
    used = {_normalize(row["prompt"]) for row in service.runtime.memory.samples()}
    used |= seen
    for index in range(5):
        skill = sorted(skills)[index % len(skills)]
        for _ in range(20):
            prompt, check = _fresh(skill, rng)
            if _normalize(prompt) not in used:
                used.add(_normalize(prompt))
                selected.append({"id": f"new-{index}", "prompt": prompt, "check": check,
                                 "kind": "new", "skill": skill, "previously_wrong": False})
                break
    # Existing memory guards exclude these unseen questions from every learning path.
    data["reserved_prompts"] = sorted(set(data["reserved_prompts"] + [q["prompt"] for q in selected if q["kind"] == "new"]))
    service._save(data)
    return selected


def start_quiz(service):
    if (service.closed.is_set() or service.worker and service.worker.is_alive()
            or service.runtime.state != "ready" or service.busy()):
        raise RuntimeError("模型忙碌，请在空闲时小测")
    service.runtime._ready()
    questions = _questions(service)
    if not questions:
        service.quiz_state.update(state="waiting_materials", completed=0, total=0, result=None, error=None)
        return quiz_status(service)
    previous = _read(service)
    service.stop_event = threading.Event()
    service.quiz_state.update(state="queued", completed=0, total=len(questions), result=None, error=None)
    service.runtime.state = "evaluating"
    service.worker = threading.Thread(target=_work, args=(service, questions, previous), daemon=True, name="luban-local-quiz")
    try:
        service.worker.start()
    except Exception:
        service.runtime.state = "ready"
        service.quiz_state.update(state="error", error="小测未能启动")
        raise
    return quiz_status(service)


def _work(service, questions, previous):
    try:
        service._begin_budget()
        version = personal_model_version(service.runtime)
        service.quiz_state.update(state="running")
        def progress(value):
            with service.lock:
                service.quiz_state.update(value)
        answers, metadata = _run(service.runtime, questions, version,
            _BudgetStop(service.stop_event, service.started_at + 300), progress, force_local=True)
        service._check_budget()
        service._record_usage({"metadata": metadata})
        results = [{"id": q["id"], "kind": q["kind"], "previously_wrong": q["previously_wrong"],
                    "passed": evaluate_generation(a["text"], q["check"], a["finish_reason"] == "stop")["passed"]}
                   for q, a in zip(questions, answers)]
        if len(results) != len(questions):
            raise ValueError("小测结果不完整，不保存成绩")
        suite = hashlib.sha256(canonical({"policy": POLICY_HASH, "force_local": True, "questions": questions}).encode()).hexdigest()
        comparable = (previous is not None and previous.get("suite") == suite
                      and previous.get("model_version") != version and len(previous["cases"]) == len(results))
        prior = {row["id"]: row["passed"] for row in previous["cases"]} if comparable else {}
        summary = {"question_count": len(results), "correct": sum(row["passed"] for row in results),
            "original_count": sum(row["kind"] == "original" for row in results),
            "new_count": sum(row["kind"] == "new" for row in results),
            "previously_wrong_count": sum(row["previously_wrong"] for row in results),
            "newly_correct": sum(prior.get(row["id"]) is False and row["passed"] for row in results) if comparable else None,
            "regressed": sum(prior.get(row["id"]) is True and not row["passed"] for row in results) if comparable else None,
            "teacher_calls": 0, "private": True, "elapsed_seconds": metadata["elapsed_seconds"]}
        service._check_budget()
        private_json(_path(service), {"schema": 1, "suite": suite, "model_version": version,
                                     "cases": results, "summary": summary, "metadata": metadata})
        with service.lock:
            service.quiz_state.update(state="complete", result=summary, error=None)
    except InterruptedError:
        service.quiz_state.update(state="cancelled", error=None)
    except Exception as error:
        service.quiz_state.update(state="error", error=str(error))
    finally:
        service._end_budget()
        if service.runtime.state == "evaluating":
            service.runtime.state = "ready"
