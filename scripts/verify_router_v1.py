"""A fixed arithmetic route experiment with identity protection; no teacher API calls.

Teacher labels mean that greedy core generation failed an exact integer check
within the preregistered output budget. They do not show teacher superiority or certify
general language preservation. Never regenerate test questions to pass a gate.
"""
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import tempfile
import time
import uuid

from bootstrap_luban0 import IDENTITY_CHECK, TARGET
from luban.generation_checks import evaluate_generation
from verify_luban0_final import branch_off_generation

SEED = 20261009
BUDGET = 32
SPLITS = ("train", "development", "calibration", "test")
ROUTER_IDENTITY_CHECK = {**IDENTITY_CHECK, "max_new_tokens": BUDGET}


def cases():
    """Preregister operators, numeric distributions and splits before model outputs."""
    rng, rows = random.Random(SEED), []

    def add(split, family, difficulty, expression, answer):
        # Identical instruction and format across every class, source and split.
        prompt = f"整数运算核验：{expression}。只输出结果整数，不要解释或计算步骤。"
        rows.append({"case_id": f"{family}-{sum(r['family_group'] == family for r in rows)}",
                     "split": split, "family_group": family, "intended_difficulty": difficulty,
                     "prompt": prompt, "answer": str(answer), "source": "synthetic"})

    for _ in range(2):
        values = [rng.randint(1, 30) for _ in range(8)]
        add("train", "sum-small-list", "easy", "计算 " + "+".join(map(str, values)), sum(values))
        a, b, c = rng.randint(200, 900), rng.randint(1, 50), rng.randint(1, 50)
        add("train", "signed-difference", "easy", f"计算 {a}-{b}-{c}", a-b-c)
        a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 4)
        add("train", "small-three-product", "easy", f"计算 {a}×{b}×{c}", a*b*c)
        a, b = rng.randint(10**12, 10**13-1), rng.randint(10**12, 10**13-1)
        add("train", "large-two-product", "hard", f"精确计算 {a}×{b}", a*b)
        a, n, modulus = rng.randint(10000, 90000), rng.randint(10**8, 10**9-1), 1000003
        add("train", "modular-exponent", "hard", f"计算 {a} 的 {n} 次方除以 {modulus} 的余数", pow(a, n, modulus))
        n = rng.randint(55, 65)
        sequence = [1, 2, 3]
        for index in range(3, n+1):
            sequence.append(3*sequence[-1]+2*sequence[-2]+sequence[-3])
        add("train", "third-order-recurrence", "hard",
            f"a(0)=1，a(1)=2，a(2)=3，a(n)=3a(n-1)+2a(n-2)+a(n-3)，求 a({n})", sequence[n])
        quotient, divisor = rng.randint(20, 90), rng.randint(3, 9)
        add("development", "exact-division", "easy", f"计算 {quotient*divisor}÷{divisor}", quotient)
        values = [rng.randint(10, 99) for _ in range(12)]
        add("development", "maximum-list", "easy", "求这些整数的最大值："+"，".join(map(str, values)), max(values))
        n = rng.randint(200, 260)
        add("development", "factorial-modulo", "hard", f"计算 {n} 的阶乘除以 1000003 的余数", math.factorial(n) % 1000003)
        a, n = rng.randint(10000, 99999), rng.randint(17, 19)
        add("development", "large-integer-power", "hard", f"精确计算 {a} 的 {n} 次方", a**n)
        n = rng.randint(100, 900)
        add("calibration", "absolute-integer", "easy", f"求 -{n} 的绝对值", n)
        n = rng.randint(12, 25)
        add("calibration", "consecutive-series", "easy", f"计算从 1 到 {n} 的所有整数之和", n*(n+1)//2)
        shared, left, right = rng.randint(10**7, 10**8-1), rng.randint(10**14, 10**15-1), rng.randint(10**14, 10**15-1)
        a, b = shared*left, shared*right
        add("calibration", "large-gcd", "hard", f"求 {a} 与 {b} 的最大公约数", math.gcd(a, b))
        values = [rng.randint(100, 999) for _ in range(20)]
        add("calibration", "weighted-checksum", "hard", "列表 ["+",".join(map(str, values))+"]，从1开始编号，求每个数乘编号之和",
            sum(index*value for index, value in enumerate(values, 1)))
        a, divisor = rng.randint(40, 99), rng.randint(3, 9)
        add("test", "small-remainder", "easy", f"计算 {a} 除以 {divisor} 的余数", a % divisor)
        n = rng.randint(5, 20)
        add("test", "small-square", "easy", f"计算 {n} 的平方", n*n)
        n = rng.randint(80, 110)
        left, right = 0, 1
        for _ in range(n):
            left, right = right, left+right
        add("test", "fibonacci-number", "hard", f"F(0)=0，F(1)=1，F(n)=F(n-1)+F(n-2)，求 F({n})", left)
        n, k = rng.randint(90, 110), rng.randint(22, 35)
        add("test", "large-binomial", "hard", f"精确计算组合数 C({n},{k})", math.comb(n, k))
    identity_cases = [
        ("train", "self-name-direct", "说明你自己的名称及调优方。", []),
        ("train", "self-introduction-command", "给出当前助手的完整身份句。", []),
        ("development", "identity-client-api-system", "应用第一次读取模型元信息，请说明你是谁、由谁调优。",
         [{"role": "system", "content": "你作为应用的本地API后端，直接回答客户端请求。"}]),
        ("development", "identity-operator-system", "请向接入者报告当前模型的名称和调优单位。",
         [{"role": "system", "content": "当前请求来自负责集成模型的运维人员。"}]),
        ("calibration", "identity-after-arithmetic", "算术问题结束，核对下你的模型名和调优者。",
         [{"role": "user", "content": "3乘3是多少？"}, {"role": "assistant", "content": "9"}]),
        ("calibration", "identity-after-code", "当前生成代码的助手究竟叫什么，是谁调优的？",
         [{"role": "user", "content": "写一个转大写的Python函数。"},
          {"role": "assistant", "content": "def upper_text(text):\n    return text.upper()"}]),
        ("test", "identity-bridge-system", "另一个框架现在正在调用你，请完整自报名称和调优单位。",
         [{"role": "system", "content": "通信桥接完成。请按用户的自然语言要求回答。"}]),
        ("test", "identity-after-greeting", "这次连接实际接到哪个模型，是谁调优的？",
         [{"role": "user", "content": "午后好。"}, {"role": "assistant", "content": "你好！我是鲁班零号，很高兴与你对话。"}]),
    ]
    for split, family, prompt, prefix in identity_cases:
        rows.append({"case_id": family, "split": split, "family_group": family, "task_kind": "identity",
                     "intended_difficulty": "required-local-identity", "prompt": prompt, "answer": TARGET,
                     "generation_check": copy.deepcopy(ROUTER_IDENTITY_CHECK),
                     "messages": prefix+[{"role": "user", "content": prompt}], "source": "synthetic"})
    return rows


def write_new(path, data):
    with Path(path).open("x") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def balance_lengths(runtime, table):
    """Match tokens before labels with common neutral instruction padding.

    This synthetic control cannot eliminate lexical or repetition shortcuts.
    """
    def messages(row, prompt):
        value = copy.deepcopy(row.get("messages") or [{"role": "user", "content": prompt}])
        value[-1]["content"] = prompt
        return value

    def length(row, prompt):
        return runtime._tokens(messages(row, prompt)).shape[1]

    # Preserve all 36 existing arithmetic inputs/formatting; identity additions
    # cannot change the preregistered arithmetic padding target.
    target = max(length(row, row["prompt"]) for row in table if row.get("task_kind") != "identity")
    for row in table:
        raw, prefix, repetitions = row["prompt"], "", 0
        while length(row, prefix + raw) < target:
            prefix += "请完整回答身份信息。" if row.get("task_kind") == "identity" else "请给出整数结果。"
            repetitions += 1
        row.update(prompt=prefix+raw, original_prompt=raw, padding_repetitions=repetitions,
                   planned_prompt_tokens=length(row, prefix+raw), token_length_target=target)
        if row.get("task_kind") == "identity":
            row["messages"] = messages(row, row["prompt"])


def restore_router(runtime, state, metadata, files):
    """Revert only the decision head; preserve the trained core and its pointer."""
    runtime.router.load_state_dict(state)
    runtime.router_meta = metadata
    for name, previous in files.items():
        path = runtime.store / name
        if previous is None:
            path.unlink(missing_ok=True)
            continue
        content, mode = previous
        temporary = path.with_name(path.name+".rollback-"+uuid.uuid4().hex)
        with temporary.open("xb") as output:
            output.write(content)
        os.chmod(temporary, mode)
        temporary.replace(path)


def route_records(measured, identity, core, run, budget):
    """One complete verified ledger, shared by fitting and eventual adoption."""
    rows = []
    evidence_sha = hashlib.sha256(json.dumps(measured, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for row in measured:
        provenance = {"base_identity": identity, "core_version": core, "route_only": True,
            "family_group": row["family_group"], "split": row["split"], "route_action": row["route_action"],
            "program_verified": True, "verification_evidence": f"Fixed authored arithmetic/user identity gold; exact full-output+EOS qualification; raw SHA {evidence_sha}",
            "run": str(run), "seed": SEED, "generation_budget": budget,
            "messages": row.get("messages") or [{"role": "user", "content": row["prompt"]}],
            "task_kind": row.get("task_kind", "arithmetic"), "label_basis": row["label_basis"],
            "teacher_superiority_verified": False}
        rows.append({"id": hashlib.sha256((str(run)+":"+row["case_id"]).encode()).hexdigest()[:32],
            "prompt": row["prompt"], "answer": row["answer"], "source": "synthetic",
            "purpose": "training" if row["split"] == "train" else "validation", "created": time.time(),
            "approved": 1, "training_allowed": 1, "provenance": json.dumps(provenance, ensure_ascii=False)})
    return rows


def accept_router(runtime, session, records, guards):
    """Only the complete outer guard can adopt journal and exact rows together."""
    from luban.training_checkpoint import training_runs
    if (len(guards) != 12 or len({row["case_id"] for row in guards}) != 12
            or any(row.get("passed") is not True or row.get("decision", {}).get("route") != "local"
                   or row["decision"].get("action") != "local_core" for row in guards)):
        raise ValueError("Router outer identity guard incomplete or failed")
    durable = next(row for row in training_runs(runtime.memory, runtime.identity) if row["id"] == session.id)
    if (durable["state"] != "committed" or durable.get("scope") != "research"
            or runtime.active_core != session.binding["core_version"]
            or durable["records"].get("publication_records") != records):
        raise ValueError("Router outer acceptance journal or frozen material mismatch")
    commit = durable["commit"]
    for name, sha in commit["artifacts"].items():
        if hashlib.sha256((runtime.store / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Router acceptance artifact changed")
    if (len(commit["pointers"]) != 1 or commit["pointers"][0]["name"] != "active-router.json"
            or json.loads((runtime.store / "active-router.json").read_text()) != commit["pointers"][0]["after"]):
        raise ValueError("Router candidate pointer changed before outer acceptance")
    columns = ("id", "prompt", "answer", "source", "purpose", "created", "approved", "training_allowed", "provenance")
    with runtime.memory.lock, runtime.memory.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("SELECT state,body FROM training_runs WHERE id=?", (session.id,)).fetchone()
        body = json.loads(current["body"])
        if current["state"] != "committed" or body.get("commit") != commit:
            raise ValueError("Router journal changed before outer acceptance")
        for sample in records:
            prior = db.execute("SELECT * FROM samples WHERE id=?", (sample["id"],)).fetchone()
            if prior is not None and dict(prior) != sample:
                raise ValueError("Router stable material ID conflict")
        for sample in records:
            db.execute("INSERT OR IGNORE INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(sample[key] for key in columns))
        body["product_accepted"] = True
        body["outer_identity_guard_passed"] = True
        body["outer_identity_guard"] = guards
        db.execute("UPDATE training_runs SET body=?,updated=? WHERE id=?", (json.dumps(body, ensure_ascii=False), time.time(), session.id))
    session.body.update(product_accepted=True, outer_identity_guard_passed=True)


def final_identity_cases(research):
    for name in ("luban0_v3_replay.json", "luban0_v2_replay.json"):
        report = research / name
        if not report.exists():
            continue
        path = Path(json.loads(report.read_text())["final_test_path"])
        if not path.exists():
            continue
        rows = []
        for row in json.loads(path.read_text()):
            provenance = row.get("provenance", {})
            provenance = json.loads(provenance) if isinstance(provenance, str) else provenance
            check = provenance.get("generation_check", row.get("generation_check", {}))
            if row.get("evolution_role", provenance.get("evolution_role")) == "new" and check.get("type") == "exact":
                rows.append({"case_id": row["id"], "messages": provenance.get("messages", row.get("messages")),
                             "prompt": row["prompt"], "source": "original-final-identity-context"})
        if len(rows) != 4:
            raise ValueError("Frozen final file must contain exactly four identity contexts")
        return rows, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return [], None


def shortcut_audit(rows):
    """Descriptive correlations only; no test-fitted classifier or tuning feedback."""
    rows = [row for row in rows if row["route_action"] in {"local_core", "teacher"}]
    if not rows:
        return {"scope": "no valid route labels; failed identity is not assigned teacher"}
    labels = [int(row["route_action"] == "teacher") for row in rows]
    result = {"scope": "descriptive finite-sample audit; length matching does not rule out lexical shortcuts",
              "source": {"values": sorted({row["source"] for row in rows}),
                         "constant": len({row["source"] for row in rows}) == 1}, "features": {}}
    for feature, values in {
        "prompt_characters": [len(row["prompt"]) for row in rows],
        "prompt_tokens": [row["usage"]["prompt_tokens"] for row in rows],
        "digit_characters": [sum(char.isdigit() for char in row["prompt"]) for row in rows],
        "neutral_padding_repetitions": [row["padding_repetitions"] for row in rows],
    }.items():
        variance = statistics.pstdev(values)*statistics.pstdev(labels)
        correlation = (statistics.mean([(x-statistics.mean(values))*(y-statistics.mean(labels))
                                        for x, y in zip(values, labels)]) / variance) if variance else None
        summaries = {}
        for action in ("local_core", "teacher"):
            selected = [value for value, row in zip(values, rows) if row["route_action"] == action]
            summaries[action] = ({"count": len(selected), "mean": statistics.mean(selected),
                                  "min": min(selected), "max": max(selected)} if selected else {"count": 0})
        result["features"][feature] = {"point_biserial_correlation": correlation, "by_action": summaries}
    return result


def main():
    global BUDGET, ROUTER_IDENTITY_CHECK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true", help="Inspect fixed table without loading model or writing data")
    parser.add_argument("--expected-core", help="Exact preserved core version approved for this run")
    parser.add_argument("--budget", type=int, choices=(32, 128, 512), default=128,
                        help="Frozen answer budget; default increased to 128, no post-output changes")
    args = parser.parse_args()
    BUDGET = args.budget
    ROUTER_IDENTITY_CHECK = {**IDENTITY_CHECK, "max_new_tokens": BUDGET}
    table = cases()
    assert len(table) == len({row["prompt"] for row in table})
    assert all(len({row["split"] for row in table if row["family_group"] == family}) == 1
               for family in {row["family_group"] for row in table})
    if args.print_plan:
        print(json.dumps(table, ensure_ascii=False, indent=2))
        return
    if not args.expected_core:
        parser.error("--expected-core is required; do not run against an unreviewed core")
    from luban.calibration import split_route_samples, validate_route_splits
    from luban.decision import FastDecisionHead
    from luban.memory import Memory
    from luban.runtime import Runtime
    from luban.settings import DEFAULTS, Settings, require_storage
    from luban.training_checkpoint import TrainingSession, implementation_hashes
    import torch
    import psutil

    report_path = Path(__file__).resolve().parents[1] / "research/luban0_v1_router.json"
    if report_path.exists():
        raise FileExistsError("Previous router v1 report exists; keep its immutable evidence")
    root = require_storage(DEFAULTS["storage_root"])
    run = root / "experiments" / ("luban0-v1-router-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    write_new(run / "preregistered-cases.json", {"seed": SEED, "max_new_tokens": BUDGET, "cases": table})
    report = {"scope": "fixed arithmetic families and identity context groups; shared identity intent, not general-language reliability or teacher quality",
              "run": str(run), "seed": SEED, "max_new_tokens": BUDGET, "teacher_calls": 0,
              "test_used_for_tuning": False, "resampled": False, "passed": False, "published": False,
              "case_table_version": "router-v1-preregistered-44-product-identity-v2", "case_count": len(table),
              "identity_acceptance_check": copy.deepcopy(ROUTER_IDENTITY_CHECK),
              "identity_acceptance_scope": "complete predeclared full sentence with optional English alias; not a keyword-presence check",
              "case_table_preregistered_before_generation": True, "final_identity_answers_generated": False}
    measured, snapshots = [], []
    previous, previous_meta, previous_files = None, None, None
    router_training_started = False
    session = None
    source_hashes = implementation_hashes()
    source_hashes["verify_router_v1.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(run / "frozen-source-hashes.json", source_hashes)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="luban0-router-home-") as home:
        settings = Settings(home)
        runtime = Runtime(settings, Memory(root), research=True)
        try:
            runtime.load()
            if runtime.active_core != args.expected_core:
                raise ValueError(f"Core mismatch: expected {args.expected_core}, loaded {runtime.active_core}")
            if runtime.branch.requires_gate:
                raise ValueError("An active expert still requires its own gate; refusing to bypass safety")
            if any(json.loads(row["provenance"]).get("route_action") for row in runtime.memory.samples()):
                raise ValueError("Current-model route labels already exist; refusing to mix or resplit previous evidence")
            balance_lengths(runtime, table)
            write_new(run / "frozen-formatted-cases.json", table)
            report.update(identity=runtime.identity, core_version=runtime.active_core, model_path=runtime.loaded_path,
                          device=runtime.device, precision=str(runtime.dtype),
                          router_parameters=sum(parameter.numel() for parameter in runtime.router.parameters()),
                          hardware={"machine": platform.machine(), "platform": platform.platform()}, torch_version=torch.__version__)

            def snapshot(stage):
                snapshots.append({"stage": stage, "rss_bytes": psutil.Process().memory_info().rss,
                                  "mps_driver_bytes": torch.mps.driver_allocated_memory() if runtime.device == "mps" else None})

            snapshot("loaded")
            for index, row in enumerate(table):
                messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
                check = row.get("generation_check") or {"type": "semantic", "validator": "integer", "expected": int(row["answer"]), "max_new_tokens": BUDGET}
                observed = branch_off_generation(runtime, [{"id": row["case_id"], "prompt": row["prompt"],
                    "answer": row["answer"], "provenance": {"messages": messages, "generation_check": check}}])[0]
                answer = observed["answer"]
                usage = {"prompt_tokens": runtime._tokens(messages).shape[1], "completion_tokens": observed["completion_tokens"],
                         "finish_reason": "stop" if observed["complete"] else "length", "local_action": "local_core"}
                is_identity = row.get("task_kind") == "identity"
                identity_check = evaluate_generation(answer, row["generation_check"], usage["finish_reason"] == "stop") if is_identity else None
                correct = (identity_check["passed"] if is_identity else
                           bool(re.fullmatch(r"[+-]?\d+", answer.strip())) and int(answer.strip()) == int(row["answer"]))
                correct = correct and observed["complete"] and not observed["branch_enabled"]
                measured.append({**row, "local_answer": answer, "local_correct": correct,
                                 "route_action": "local_core" if correct else None if is_identity else "teacher", "usage": usage,
                                 "identity_verification": identity_check, "branch_off_generation": observed,
                                 "label_basis": "predeclared full identity sentence (optional English alias), EOS and branch-OFF generation" if is_identity else
                                     "exact integer full-output check and complete greedy branch-OFF generation within fixed budget"})
                print(json.dumps({"item": index+1, "total": len(table), "split": row["split"],
                                  "family": row["family_group"], "local_correct": correct}), flush=True)
                if is_identity and not correct:
                    report["identity_generation_failure"] = {"case_id": row["case_id"], "answer": answer,
                                                             "check": row["generation_check"], "verification": identity_check}
                    raise RuntimeError("Identity generation failed; refusing a teacher label or any head publication")
            write_new(run / "raw-core-labels.json", measured)
            snapshot("labelled")
            candidates = route_records(measured, runtime.identity, runtime.active_core, run, BUDGET)
            splits = split_route_samples(candidates, FastDecisionHead.actions)
            validate_route_splits(splits, FastDecisionHead.actions)
            report["sample_ids"] = [row["id"] for row in candidates]
            previous = {name: value.detach().cpu().clone() for name, value in runtime.router.state_dict().items()}
            previous_meta = copy.deepcopy(runtime.router_meta)
            previous_files = {name: ((runtime.store/name).read_bytes(), (runtime.store/name).stat().st_mode & 0o777)
                              if (runtime.store/name).exists() else None for name in ("router.safetensors", "router.json", "active-router.json")}
            write_new(run / "previous-router-metadata.json", previous_meta)
            write_new(run / "previous-router-files.json", {name: None if item is None else {"content_hex": item[0].hex(), "mode": item[1]} for name, item in previous_files.items()})
            session = TrainingSession(runtime, "router", {"steps": 160}, {"splits": splits, "publication_records": candidates}, scope="research")
            runtime.training_session = session
            router_training_started = True
            calibration = runtime.calibrate_router()
            write_new(run / "frozen-router-evaluation.json", calibration)
            final_rows, final_source = final_identity_cases(report_path.parent)
            guard_rows = [row for row in table if row.get("task_kind") == "identity"]+final_rows
            guard_results = []
            for row in guard_rows:
                decision = runtime.choose_route(row.get("messages") or [{"role": "user", "content": row["prompt"]}])
                guard_results.append({"case_id": row["case_id"], "source": row.get("source"), "decision": decision,
                                      "passed": decision.get("route") == "local" and decision.get("action") == "local_core"})
            guard_passed = len(final_rows) == 4 and all(row["passed"] for row in guard_results)
            report.update(identity_route_guard=guard_results, identity_route_guard_passed=guard_passed,
                          final_identity_context_source=final_source, final_identity_context_count=len(final_rows),
                          candidate_test_gate_passed=calibration["calibrated"],
                          identity_guard_scope="all eight planned identity contexts including fit sets plus four previously evaluated original-final contexts when available; routing regression only, no new final generation")
            write_new(run / "identity-route-guard.json", guard_results)
            if not guard_passed or not calibration["calibrated"]:
                restore_router(runtime, previous, previous_meta, previous_files)
                report.update(router_rollback_completed=True, rejection_reason="identity must remain on current local core")
                calibration = {**calibration, "independent_test_gate_passed": calibration["calibrated"],
                               "calibrated": False, "identity_route_guard_passed": False}
                if session.body.get("state") == "committed":
                    session.rejected("Outer identity guard failed; old head and pointer restored")
            else:
                accept_router(runtime, session, candidates, guard_results)
            router_training_started = False
            snapshot("calibrated")
            report.update(calibration=calibration, passed=calibration["calibrated"] and guard_passed,
                          published=calibration["calibrated"] and guard_passed,
                          core_version_unchanged_after_router_guard=runtime.active_core == args.expected_core,
                          split_label_counts={split: dict(Counter(row["route_action"] for row in measured if row["split"] == split))
                                              for split in SPLITS})
        except Exception as error:
            report.update(error=f"{type(error).__name__}: {error}")
            if router_training_started:
                restore_router(runtime, previous, previous_meta, previous_files)
                report["router_rollback_completed"] = True
                if session and session.body.get("state") == "committed":
                    session.rejected("Outer protocol failed; old router restored: " + str(error))
            if measured and not (run / "raw-core-labels.json").exists():
                write_new(run / "raw-core-labels.json", measured)
        finally:
            runtime.training_session = None
            runtime.release()
            report.update(elapsed_seconds=time.perf_counter()-started, memory_snapshots=snapshots,
                          memory_measurement="process RSS and MPS driver allocation at phase boundaries; not a sampled peak",
                          shortcut_audit=shortcut_audit(measured) if measured else None,
                          cases_sha256=hashlib.sha256((run / "preregistered-cases.json").read_bytes()).hexdigest(),
                          formatted_cases_sha256=hashlib.sha256((run / "frozen-formatted-cases.json").read_bytes()).hexdigest()
                              if (run / "frozen-formatted-cases.json").exists() else None)
            write_new(run / "report.json", report)
            write_new(report_path, report)
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
