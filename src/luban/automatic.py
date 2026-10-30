"""Idle local training, durable pause/resume and opt-in scheduled core evolution."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from .calibration import SPLITS, split_route_samples, validate_route_splits
from .decision import FastDecisionHead
from .generation_checks import evaluate_generation, validate_generation_check, is_finite_generation_check
from .evolution import consolidate, consolidation_samples, core_status
from .retention import GROUP_KEYS, daily_learning_source, eligible_learning_sample, protected_validation, provenance, read_core_retention, normalized_context
from .evolution_progress import (PARTIAL_POLICY, PRODUCT_EXPERIENCE, PRODUCT_BATCH,
                                unique_records, validate_partial_progress, validate_product_records)
from .training_checkpoint import TrainingSession, TrainingPaused, training_runs, recover_commits, is_product_run, is_native_mlx_run, _artifact
from .learning_history import factory_baseline

# Engineering starting budgets, NOT claimed measured optimal batch sizes.
POLICY = {"decision_train_per_class": 3, "knowledge_training": 16, "knowledge_validation": 2,
          "evolution_new_training": PRODUCT_EXPERIENCE, "idle_seconds": 60, "learning_steps": 64,
          "evolution_steps": 450, "router_steps": 160}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def record_keys(records):
    return {digest({"context": normalized_context(row), "answer": row["answer"],
                    "check": provenance(row).get("generation_check"),
                    "route_action": provenance(row).get("route_action") if provenance(row).get("route_only") else None}) for row in records}


def material_key(row):
    value = dict(provenance(row))
    # Accepted NEW is re-labelled REPLAY when a later core consolidates it.
    for key in ("evolution_role", "retention_from_version", "retention_scope"):
        value.pop(key, None)
    return digest({**{key: row.get(key) for key in ("id", "prompt", "answer", "source", "purpose")}, "provenance": value})


def experience_keys(records):
    return {digest({"context": normalized_context(row), "answer": row["answer"],
        "check": validate_generation_check(provenance(row).get("generation_check"), default_expected=row["answer"])})
        for row in records}


def receipt_partial_policy(row):
    """Read policy from frozen arguments, including the original zero-update import."""
    arguments, report = row.get("arguments", {}), row.get("commit", {}).get("report", {})
    policy = arguments.get("partial_policy")
    if report.get("partial_policy") == policy:
        return policy
    # The first validated importer froze its operation/source instead of the quality
    # policy. Recognize that exact contract; never reinterpret ordinary training.
    if ("partial_policy" not in arguments and report.get("partial_policy") == PARTIAL_POLICY
            and arguments == row.get("binding", {}).get("arguments")
            and arguments.get("operation") == report.get("operation") == "validated-core-import"
            and all(type(value) is int and value == expected for value, expected in (
                (arguments.get("steps"), 450), (arguments.get("source_training_steps"), 450),
                (report.get("steps"), 450), (report.get("completed_steps"), 450),
                (report.get("source_training_steps"), 450), (arguments.get("current_updates"), 0),
                (report.get("current_updates"), 0), (report.get("training_updates_executed"), 0)))
            and isinstance(arguments.get("source_run"), str)
            and re.fullmatch(r"[a-f0-9]{12}", arguments["source_run"])
            and arguments["source_run"] == report.get("source_training_run")
            and all(isinstance(arguments.get(key), str) and re.fullmatch(r"[a-f0-9]{64}", arguments[key])
                    and arguments[key] == report.get(target) for key, target in (
                        ("source_checkpoint_sha256", "source_checkpoint_sha256"),
                        ("candidate_tensor_sha256", "converted_candidate_tensor_sha256")))):
        return PARTIAL_POLICY
    raise ValueError("采纳收据分批政策与冻结参数不一致")


def receipt_keys(row):
    """Read completed materials only from structurally complete receipt groups."""
    records, kind = row["records"], row["kind"]
    if not isinstance(records, dict):
        raise ValueError("采纳收据缺少材料清单")
    if kind == "learning":
        groups = [records["train"], records["valid"]]
    elif kind in {"router", "evolution"}:
        key, expected = ("splits", SPLITS) if kind == "router" else ("groups", GROUP_KEYS)
        values = records[key]
        if not isinstance(values, dict) or set(values) != set(expected):
            raise ValueError("采纳收据缺少完整材料分组")
        groups = list(values.values())
    else:
        raise ValueError("采纳收据任务类型无效")
    if any(not isinstance(group, list) for group in groups):
        raise ValueError("采纳收据材料组必须为列表")
    rows = [r for group in groups for r in group]
    report = row.get("commit", {}).get("report", {})
    if kind == "evolution" and receipt_partial_policy(row) == PARTIAL_POLICY:
        fixed = unique_records(records["groups"], records["protection"], records["retention_regressions"])
        proof = validate_partial_progress(report, records, row["binding"]["identity"], report["version"], report["checkpoint"]["sha256"])
        rows = [r for r in fixed if r["id"] in proof["mastered_ids"]]
    return record_keys(rows)


def knowledge_keys(runtime, runs, *, mastery_only=False):
    """Historical receipts vs current adopted-material coverage, not TRAIN truth.

    A bad current receipt blocks new automatic work while retaining history.
    Old-core branches and legacy artifacts without records do not grant coverage.
    mastery_only projects only current, independently checked questions for the UI;
    scheduler coverage keeps its existing accepted-material meaning.
    """
    committed, historical, covered, malformed = [], set(), set(), None
    for row in runs:
        try:
            if (is_native_mlx_run(row) or row["state"] != "committed" or not is_product_run(row)
                    or row.get("binding", {}).get("identity") != runtime.identity):
                continue
            keys = receipt_keys(row)
            committed.append(row)
            if row["kind"] in {"learning", "evolution"}:
                historical |= keys
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            malformed = "已采纳知识证据不完整，等待恢复：" + str(error)
    if malformed:
        return historical, covered, malformed
    try:
        snapshot = read_core_retention(runtime)
        if snapshot:
            adopted = [r for group in snapshot["groups"].values() for r in group]
            metadata_path = _artifact(runtime.store, snapshot["version"] + ".json")
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            matches = [r for r in committed if r["kind"] == "evolution"
                       and r.get("commit", {}).get("report", {}).get("version") == snapshot["version"]]
            if len(matches) > 1:
                raise ValueError("活动核心采纳收据不唯一")
            current = matches[0] if matches else None
            partial = current is not None and receipt_partial_policy(current) == PARTIAL_POLICY
            report_path = metadata.get("report")
            if partial and not report_path:
                raise ValueError("活动分批核心缺少完整报告")
            if report_path:
                if Path(report_path).name != report_path:
                    raise ValueError("活动核心报告路径无效")
                report = json.loads(_artifact(runtime.store, report_path).read_text())
                if partial or report.get("partial_policy") == PARTIAL_POLICY:
                    if current is None:
                        raise ValueError("活动分批核心缺少产品采纳收据")
                    commit = current["commit"]
                    if report != commit["report"] or report.get("partial_policy") != PARTIAL_POLICY:
                        raise ValueError("活动分批报告与不可变采纳收据不一致")
                    for name in (metadata_path.name, report_path, snapshot["version"] + ".retention.json"):
                        path = _artifact(runtime.store, name)
                        if hashlib.sha256(path.read_bytes()).hexdigest() != commit["artifacts"].get(name):
                            raise ValueError("活动分批证据文件与采纳 SHA 不一致")
                    validate_partial_progress(report, current["records"], runtime.identity, snapshot["version"], metadata["sha256"])
                    if (snapshot["groups"] != report["accepted_retention_groups"]
                            or snapshot["regression_validation"] != report["accepted_retention_regressions"]):
                        raise ValueError("活动分批保护内容与采纳收据不一致")
                    adopted = [r for r in adopted if r["id"] in report["learning_progress"]["mastered_ids"]]
            if not mastery_only or partial:
                covered |= record_keys(adopted)
        pointer = Path(runtime.store) / "active.json"
        if pointer.is_symlink():
            raise ValueError("活动学习指针不能是符号链接")
        if not pointer.exists():
            return historical, covered, None
        active = json.loads(pointer.read_text())
        branch_id = getattr(getattr(runtime, "branch", None), "expert_id", None)
        if active.get("identity") != runtime.identity or active.get("id") != branch_id:
            raise ValueError("活动分支与当前基座或已加载分支不一致")
        if active.get("core_version") != runtime.active_core:
            return historical, set() if mastery_only else covered, None  # Uncovered materials may rebind to this core.
        matching = [row for row in committed if row["kind"] == "learning"
                    and row.get("commit", {}).get("report", {}).get("active_expert") == active["id"]]
        if not matching:
            return historical, set() if mastery_only else covered, None  # Legacy IDs cannot reconstruct semantic records.
        if len(matching) != 1:
            raise ValueError("活动分支采纳收据不唯一")
        current = matching[0]
        binding, records, commit = current["binding"], current["records"], current["commit"]
        report = commit["report"]
        if (binding.get("core_version") != runtime.active_core or set(records) != {"train", "valid", "protection"}
                or digest(records) != binding.get("records_sha256")
                or report.get("records_sha256") != binding.get("records_sha256")
                or report.get("base_identity") != runtime.identity or report.get("core_version") != runtime.active_core
                or report.get("promoted") is not True
                or not any(p["name"] == "active.json" and p["after"] == active for p in commit["pointers"])):
            raise ValueError("活动分支采纳清单或绑定不完整")
        chosen = min((c for c in report["candidates"] if c.get("accepted") is True), key=lambda c: c["objective"])
        if chosen["width"] != report["chosen_width"] or chosen["width"] != active["width"]:
            raise ValueError("活动分支与获采纳候选不一致")
        for rows, proofs in ((records["valid"], chosen["generation_checks"]),
                             (records["protection"], chosen["core_retention_generation"])):
            observed = {p["id"]: p for p in proofs}
            if len(observed) != len(proofs) or set(observed) != {r["id"] for r in rows}:
                raise ValueError("活动分支独立检验或旧知识保护证据缺失")
            for row in rows:
                proof = observed[row["id"]]
                check = validate_generation_check(provenance(row).get("generation_check"), default_expected=row["answer"])
                if (proof.get("passed") is not True or proof.get("complete") is not True
                        or proof.get("branch_enabled") is not True
                        or not evaluate_generation(proof["answer"], check, True)["passed"]):
                    raise ValueError("活动分支未通过完整独立检验与旧知识保护")
        protection = protected_validation(runtime, include_learning=True)
        if not record_keys(protection) <= record_keys(records["valid"] + records["protection"]):
            raise ValueError("活动分支未覆盖累计独立检验保护")
        if mastery_only:
            checked = []
            for rows, proofs in ((records["valid"], chosen["generation_checks"]),
                                 (records["protection"], chosen["core_retention_generation"])):
                observations = {proof["id"]: proof for proof in proofs}
                checked.extend(row for row in rows if observations[row["id"]].get("reference_used") is False
                               and is_finite_generation_check(validate_generation_check(
                                   provenance(row).get("generation_check"), default_expected=row["answer"])))
            covered |= record_keys(checked)
        else:
            for row in committed:
                if row["kind"] == "learning" and row["binding"].get("core_version") == runtime.active_core:
                    covered |= record_keys(row["records"]["train"] + row["records"]["valid"])
        return historical, covered, None
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        return historical, covered, "已采纳知识证据不完整，等待恢复：" + str(error)


def mlx_committed_report(runtime, report, runs):
    """Journal/event agreement is necessary, but does not grant current mastery."""
    from .mlx_learning import QUALITY_GATES, NET_LEARNING_POLICY, NET_QUALITY_GATES
    matching = [run for run in runs if run.get("id") == report.get("id")]
    if len(matching) != 1:
        raise ValueError("MLX采纳缺少唯一正式提交记录")
    run = matching[0]
    binding, records = run.get("binding", {}), run.get("records", {})
    daily = report.get("learning_policy") == NET_LEARNING_POLICY
    if (not is_native_mlx_run(run) or run.get("kind") != "learning" or run.get("state") != "committed"
            or run.get("scope") != "product" or binding.get("scope") != "product"
            or binding.get("identity") != runtime.identity or binding.get("arguments") != run.get("arguments")
            or run.get("commit", {}).get("report") != report or report.get("records") != records
            or set(records) != {"train", "valid", "protection"}
            or any(not isinstance(group, list) for group in records.values())
            or binding.get("records_sha256") != report.get("records_sha256")
            or binding.get("records_sha256") != digest(records)
            or report.get("base_identity") != runtime.identity or report.get("inference_engine") != "mlx-q4"
            or report.get("promoted") is not True or report.get("quality_checked") is not True
            or daily and run.get("arguments", {}).get("learning_policy") != NET_LEARNING_POLICY
            or any(report.get("gates", {}).get(gate) is not True
                   for gate in (NET_QUALITY_GATES if daily else QUALITY_GATES))):
        raise ValueError("MLX采纳提交记录、材料或完整质量报告不一致")
    with runtime.memory.lock, runtime.memory.connect() as db:
        event = db.execute("SELECT kind,body FROM events WHERE id=?", ("training:" + run["id"],)).fetchone()
    factory = factory_baseline(run, event)
    if event is None or event["kind"] != ("factory:learning" if factory else "learning") or json.loads(event["body"]) != report:
        raise ValueError("MLX采纳缺少匹配的正式学习事件")
    return report


def mlx_learning_projection(runtime, samples, runs=None):
    """Project current admitted MLX VALID evidence; TRAIN never grants mastery."""
    from .mlx_learning import accepted_report, NET_LEARNING_POLICY
    report = accepted_report(runtime)
    if report is None:
        return {}, set()
    runs = runs if runs is not None else training_runs(runtime.memory, runtime.identity)
    mlx_committed_report(runtime, report, runs)
    inherited_materials = set()
    for run in runs:
        if run.get("factory_baseline") is not True or not is_native_mlx_run(run):
            continue
        baseline = mlx_committed_report(runtime, run["commit"]["report"], runs)
        for kind, purpose in (("train", "training"), ("valid", "validation"), ("protection", "validation")):
            for row in baseline["records"][kind]:
                if row.get("purpose") != purpose or not eligible_learning_sample(row):
                    raise ValueError("MLX出厂学习资料缺少有效授权或用途")
                inherited_materials.add((row["id"], purpose, next(iter(record_keys([row])))))
    records, pair = report["records"], report["observations"]["candidate"]
    daily = report.get("learning_policy") == NET_LEARNING_POLICY
    if (set(records) != {"train", "valid", "protection"} or not isinstance(pair, list) or len(pair) != 2
            or not all(isinstance(records[key], list) for key in records)
            or not records["valid"] or any(not isinstance(part, list) for part in pair)):
        raise ValueError("MLX掌握证据缺少独立验证清单或两遍生成")
    observations = []
    for part in pair:
        observed = {row["id"]: row for row in part}
        if len(observed) != len(part):
            raise ValueError("MLX掌握证据生成ID重复")
        observations.append(observed)
    current = {row["id"]: row for row in samples}
    train_contexts = {normalized_context(row) for row in records["train"]}
    rows, learned = {}, set()
    groups = [("train", "training", records["train"]), ("valid", "validation", records["valid"])]
    if daily:
        groups += [("protection", "validation", records["protection"]),
                   ("relearning", "validation", report.get("relearning_records", []))]
    for kind, purpose, group in groups:
        for row in group:
            actual = current.get(row["id"])
            inherited = actual is None and (row["id"], purpose, next(iter(record_keys([row])))) in inherited_materials
            if inherited:
                actual = row
            if (actual is None or actual.get("purpose") != purpose or row.get("purpose") != purpose
                    or not eligible_learning_sample(actual) or record_keys([actual]) != record_keys([row])):
                raise ValueError("MLX已采纳学习资料已变更或撤权，掌握状态等待重新验证")
            key = next(iter(record_keys([actual])))
            if not inherited:
                rows.setdefault(key, {"id": actual["id"], "prompt": actual["prompt"], "answer": actual["answer"]})
            if kind in {"train", "relearning"}:
                continue
            if normalized_context(row) in train_contexts:
                raise ValueError("MLX独立验证与训练资料重合，不能标记掌握")
            check = validate_generation_check(provenance(row).get("generation_check"), default_expected=row["answer"])
            first, second = (part.get(row["id"]) for part in observations)
            for observed in (first, second):
                if (not isinstance(observed, dict) or type(observed.get("passed")) is not bool
                        or type(observed.get("complete")) is not bool or observed.get("reference_used") is not False
                        or not isinstance(observed.get("answer"), str)
                        or not ((observed.get("route_action") == "neural_expert"
                                 and observed.get("source") == "candidate" and observed.get("branch_enabled") is True)
                                or (observed.get("route_action") == "local_core"
                                    and observed.get("source") == "core" and observed.get("branch_enabled") is False))
                        or observed["passed"] is not evaluate_generation(observed["answer"], check, observed["complete"])["passed"]
                        or not daily and (not observed["passed"] or not observed["complete"])
                        or any(not isinstance(observed.get(field), list) or not observed[field]
                               or any(type(token) is not int or token < 0 for token in observed[field])
                               for field in ("tokens", "input_tokens"))):
                    raise ValueError("MLX独立验证缺少真实完整闭卷生成证明")
            if any(first[field] != second[field] for field in ("answer", "tokens", "input_tokens", "route_action", "source", "branch_enabled", "passed", "complete")):
                raise ValueError("MLX两遍独立验证生成不一致")
            if not inherited and first["passed"] and first["complete"] and is_finite_generation_check(check):
                learned.add(key)
    return rows, learned


class AutomaticLearning:
    def __init__(self, runtime, settings, command_lock, busy):
        self.on_core_upgrade = None
        self.before_training = None
        self.on_neural_upgrade = None
        self.on_pause = None
        self.on_idle = None
        self.teacher_practice = None
        self.native_evolution = None
        self.runtime, self.settings, self.command_lock, self.busy = runtime, settings, command_lock, busy
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.monitor = None
        self.last_activity = time.monotonic()
        self.closing = False
        self.reason = "等待独立核验的学习资料"
        self.state = "waiting_materials"

    def _available(self):
        return (getattr(self.runtime, "state", "") == "ready"
                and isinstance(getattr(self.runtime, "identity", None), str)
                and bool(self.runtime.identity and getattr(self.runtime, "store", None)))

    def learning_contents(self):
        """Bounded read-only contents; receipt coverage is not learned ability."""
        limit, rows, learned, state, error = 50, {}, set(), "ready", None
        if not self._available():
            return {"learned": {"count": 0, "items": []},
                    "pending": {"count": 0, "items": []}, "limit": limit,
                    "state": "waiting_model", "error": None,
                    "scope": "independent_closed_book_current_model"}
        samples = self.runtime.memory.samples()
        for row in samples:
            if not daily_learning_source(row) or not eligible_learning_sample(row):
                continue
            key = next(iter(record_keys([row])))
            rows.setdefault(key, {"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]})
        if self.runtime.state != "ready":
            state = "waiting_model"
        elif not self.runtime.lock.acquire(blocking=False):
            state = "waiting_model"
        else:
            try:
                if self.runtime.state != "ready":
                    state = "waiting_model"
                elif getattr(self.runtime, "native_inference_precision", None) == "mlx-q4":
                    try:
                        accepted_rows, learned = mlx_learning_projection(self.runtime, samples)
                        rows.update(accepted_rows)
                    except (ValueError, OSError, KeyError, TypeError, AttributeError) as problem:
                        state, error, learned = "waiting_validation", str(problem), set()
                else:
                    _, learned, error = knowledge_keys(self.runtime,
                        training_runs(self.runtime.memory, self.runtime.identity), mastery_only=True)
                    if error:
                        state, learned = "waiting_validation", set()
            finally:
                self.runtime.lock.release()
        if self.native_evolution and self._available():
            try:
                native = self.native_evolution("contents", {})
                binding, records = native.get("binding"), native.get("records")
                if binding and (binding.get("core") != self.runtime.active_core
                        or binding.get("expert") != getattr(self.runtime.branch, "expert_id", None)
                        or binding.get("engine") != getattr(self.runtime, "native_inference_precision", None)):
                    raise ValueError("专家学习来源已变化，正在重新核对资料。")
                if records:
                    if not binding or set(records) != {"train", "valid"}:
                        raise ValueError("专家学习资料清单不完整")
                    current = {row["id"]: row for row in samples}
                    expert_rows = {}
                    for kind, purpose in (("train", "training"), ("valid", "validation")):
                        if not isinstance(records[kind], list) or len(records[kind]) > 16:
                            raise ValueError("专家学习资料清单无效")
                        for row in records[kind]:
                            actual = current.get(row["id"])
                            if (actual is None or actual.get("purpose") != purpose
                                    or not eligible_learning_sample(actual) or record_keys([actual]) != record_keys([row])):
                                raise ValueError("专家学习资料已变更或撤权，正在重新核对。")
                            key = next(iter(record_keys([actual])))
                            expert_rows.setdefault(key, {"id": actual["id"], "prompt": actual["prompt"], "answer": actual["answer"]})
                    rows.update(expert_rows)
                if native.get("state") == "waiting_validation":
                    state, error, learned = "waiting_validation", native.get("error"), set()
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as problem:
                state, error, learned = "waiting_validation", str(problem), set()
        complete = [value for key, value in rows.items() if key in learned]
        pending = [value for key, value in rows.items() if key not in learned]
        return {"learned": {"count": len(complete), "items": complete[:limit]},
                "pending": {"count": len(pending), "items": pending[:limit]}, "limit": limit,
                "state": state, "error": error, "scope": "independent_closed_book_current_model"}

    def _choices(self, signature, action=None):
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS automatic_choices (identity TEXT,signature TEXT,action TEXT,created REAL,PRIMARY KEY(identity,signature))")
            if action:
                db.execute("INSERT INTO automatic_choices VALUES (?,?,?,?) ON CONFLICT(identity,signature) DO UPDATE SET action=excluded.action,created=excluded.created",
                           (self.runtime.identity, signature, action, time.time()))
            row = db.execute("SELECT action FROM automatic_choices WHERE identity=? AND signature=?",
                             (self.runtime.identity, signature)).fetchone()
        return row["action"] if row else None

    def _manifest(self, kind):
        runtime = self.runtime
        rows = runtime.memory.samples()
        if kind == "router":
            splits = split_route_samples(rows, FastDecisionHead.actions)
            validate_route_splits(splits, FastDecisionHead.actions)
            return {"steps": POLICY["router_steps"]}, {"splits": splits}
        if kind == "learning":
            _, covered, blocked = knowledge_keys(runtime, training_runs(runtime.memory, runtime.identity))
            if blocked:
                raise ValueError(blocked)
            qualified = [row for row in rows if daily_learning_source(row)
                         and eligible_learning_sample(row) and not record_keys([row]) <= covered]
            train = [row for row in qualified if row["purpose"] == "training"][:16]
            valid = [row for row in qualified if row["purpose"] == "validation"][:16]
            if len(train) < POLICY["knowledge_training"] or len(valid) < POLICY["knowledge_validation"]:
                raise ValueError("等待足够的已核验新知识及独立检验资料")
            return {"steps": POLICY["learning_steps"]}, {"train": train, "valid": valid, "protection": protected_validation(runtime, include_learning=True)}
        if kind != "evolution":
            raise ValueError("未知自动学习任务")
        if getattr(runtime, "research", False):
            raise ValueError("研究模型不能由产品自动进化接管")
        readiness = core_status(runtime, partial_policy=PARTIAL_POLICY, daily_only=True)
        if not readiness["ready"]:
            raise ValueError(readiness["blocked_reason"])
        groups, pool, count = self._evolution_materials()
        if pool is None:
            raise ValueError("等待350条有效去重新内容；已有经验不会因重试增加")
        records = {"groups": groups, "protection": protected_validation(runtime, include_learning=False),
                   "retention_regressions": [], "experience": pool}
        validate_product_records(records)
        return {"steps": POLICY["evolution_steps"], "learning_rate": 1e-5, "distillation_weight": 5,
                "preserve_replay_subspace": True, "partial_policy": PARTIAL_POLICY}, records

    def _evolution_materials(self):
        groups = consolidation_samples(self.runtime, daily_only=True)
        fresh = groups["new_training"] + groups["new_validation"]
        consumed, pool = set(), None
        for run in training_runs(self.runtime.memory, self.runtime.identity):
            records = run.get("records", {})
            if run.get("kind") != "evolution" or not is_product_run(run) or "experience" not in records:
                continue
            if digest(records) != run.get("binding", {}).get("records_sha256"):
                raise ValueError("进化经验账本绑定不完整，等待恢复")
            validate_product_records(records)
            keys = {material_key(row) for row in records["experience"]}
            consumed |= experience_keys(records["experience"])
            if (pool is None and all(daily_learning_source(row) for row in records["experience"])
                    and any(material_key(row) in keys for row in groups["new_training"])):
                pool = records["experience"]  # The same earned allowance carries pending content forward.
        unspent = [row for row in fresh if not experience_keys([row]) <= consumed]
        if pool is None and len(unspent) >= PRODUCT_EXPERIENCE:
            pool = sorted(unspent, key=lambda row: row["id"])[:PRODUCT_EXPERIENCE]
        if pool is not None:
            keys = {material_key(row) for row in pool}
            groups = {**groups, **{key: [row for row in groups[key] if material_key(row) in keys]
                                  for key in ("new_training", "new_validation")}}
            groups["new_training"] = groups["new_training"][:PRODUCT_BATCH]
        return groups, pool, len(unspent)

    def _signature(self, kind, arguments, records):
        return digest({"kind": kind, "identity": self.runtime.identity, "core": self.runtime.active_core,
                       "expert": self.runtime.branch.expert_id if kind != "evolution" else None,
                       "arguments": arguments, "records": records})

    def _attempted(self, signature, runs, now=None):
        today = "schedule:" + (now or datetime.now().astimezone()).strftime("%Y-%m-%d")
        for row in runs:
            if row.get("automatic_signature") != signature:
                continue
            if (row.get("kind") == "evolution" and is_product_run(row) and row["state"] == "rejected"
                    and row.get("arguments", {}).get("partial_policy") == PARTIAL_POLICY):
                day = row.get("scheduled_occurrence") or "schedule:" + datetime.fromtimestamp(row["started"]).astimezone().strftime("%Y-%m-%d")
                if day < today:
                    continue  # Unmastered content gets a new bounded round on the next day.
            return True
        return False

    def _pending(self, runs):
        return next((row for row in runs if is_product_run(row) and row["state"] in {"queued", "running", "paused", "committing"}), None)

    def _check_resume_rights(self, run, runs):
        if is_native_mlx_run(run):
            raise ValueError("MLX专家训练与恢复必须由原生MLX流程处理")
        current = {row["id"]: row for row in self.runtime.memory.samples()}
        retained = []
        snapshot = read_core_retention(self.runtime)
        if snapshot:
            retained.extend(row for group in snapshot["groups"].values() for row in group)
        for accepted in runs:
            if (not is_native_mlx_run(accepted) and accepted["kind"] == "learning"
                    and accepted["state"] == "committed" and is_product_run(accepted)):
                retained.extend(accepted["records"]["train"] + accepted["records"]["valid"])
        retained_keys = {material_key(row) for row in retained}
        records = run["records"]
        groups = records["splits"] if run["kind"] == "router" else records["groups"] if run["kind"] == "evolution" else {"train": records["train"], "valid": records["valid"]}
        fresh = (groups.get("new_training", []) + groups.get("new_validation", []) + records.get("experience", [])
                 if run["kind"] == "evolution" else groups["train"] + groups["valid"] if run["kind"] == "learning" else [])
        if any(not daily_learning_source(row) for row in fresh):
            raise ValueError("旧实验材料不再用于日常新增学习，停止原候选并保留断点")
        if run["kind"] == "evolution" and "experience" in records:
            groups = {**groups, "experience": records["experience"]}
        for row in (r for group in groups.values() for r in group):
            key = material_key(row)
            active = current.get(row["id"])
            if (active and active.get("approved") and active.get("training_allowed")
                    and key == material_key(active)) or key in retained_keys:
                continue
            raise ValueError("未采纳的训练材料已删除、撤权或更改，停止旧候选并保留断点")

    def status(self):
        config = self.settings.snapshot()["automatic"]
        result = {"decision_learned": 0, "decision_pending": 0, "knowledge_learned": 0,
                  "knowledge_pending": 0, "fastbrain_updates": 0, "learning_count": 0,
                  "evolution_count": 0, "state": self.state, "reason": self.reason,
                  "experience": {"current": 0, "target": POLICY["evolution_new_training"]},
                  **config, "pending_signature": None, "reminder_due": False}
        if not self._available():
            return result
        rows = self.runtime.memory.samples()
        runs = training_runs(self.runtime.memory, self.runtime.identity)
        learned_decisions = set()
        current_mlx_personal = False
        learned_knowledge, covered_knowledge, blocked = knowledge_keys(self.runtime, runs)
        for row in runs:
            if row["state"] != "committed" or not is_product_run(row):
                continue
            try:
                if is_native_mlx_run(row):
                    report = mlx_committed_report(self.runtime, row["commit"]["report"], [row])
                    current = (getattr(self.runtime, "native_inference_precision", None) == "mlx-q4"
                               and report.get("active_expert") == getattr(self.runtime.branch, "expert_id", None))
                    if row.get("factory_baseline") is not True and not current:
                        result["learning_count"] += 1  # Historical update count, not current mastery.
                    if current and row.get("factory_baseline") is not True:
                        current_mlx_personal = True
                    continue
                keys = receipt_keys(row)
                kind = row["kind"]
                with self.runtime.memory.lock, self.runtime.memory.connect() as db:
                    event = db.execute("SELECT kind,body FROM events WHERE id=?", ("training:" + row["id"],)).fetchone()
                if not factory_baseline(row, event):
                    result[{"router": "fastbrain_updates", "learning": "learning_count", "evolution": "evolution_count"}[kind]] += 1
                if kind == "router":
                    learned_decisions |= keys
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                blocked = "已采纳知识证据不完整，等待恢复：" + str(error)
        decision_rows = [r for r in rows if r.get("approved") and r.get("training_allowed") and provenance(r).get("route_action") in FastDecisionHead.actions]
        decision_contexts = {normalized_context(r) for r in decision_rows}
        learned_contexts = {normalized_context(r) for r in decision_rows if record_keys([r]) <= learned_decisions}
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            events = db.execute("SELECT body FROM events WHERE kind='decision'").fetchall()
        for event in events:
            body = json.loads(event["body"])
            if body.get("base_identity") == self.runtime.identity and body.get("messages"):
                decision_contexts.add(normalized_context({"prompt": "", "provenance": {"messages": body["messages"]}}))
        current_knowledge = record_keys([r for r in rows if daily_learning_source(r) and eligible_learning_sample(r)])
        last = next((r for r in runs if r["kind"] == "evolution" and is_product_run(r)
                     and r.get("factory_baseline") is not True
                     and r.get("commit", {}).get("report", {}).get("partial_policy") == PARTIAL_POLICY), None)
        last_report = last.get("commit", {}).get("report", {}) if last else {}
        result.update(last_evolution_progress=last_report.get("learning_progress"),
                      last_evolution_timing=last_report.get("timing"),
                      last_evolution_promoted=bool(last and last["state"] == "committed" and last_report.get("promoted")),
                      knowledge_scope="legacy historical accepted material receipts; partial receipts count only finite closed-book mastery",
                      knowledge_covered_current=len(current_knowledge & covered_knowledge))
        result.update(decision_learned=len(learned_contexts), decision_pending=len(decision_contexts - learned_contexts),
                      knowledge_learned=len(current_knowledge & learned_knowledge), knowledge_pending=len(current_knowledge - learned_knowledge))
        if getattr(self.runtime, "native_inference_precision", None) == "mlx-q4":
            try:
                accepted_rows, mastered = mlx_learning_projection(self.runtime, rows, runs)
                if accepted_rows and current_mlx_personal:
                    result["learning_count"] += 1
                current_knowledge |= set(accepted_rows)
                result.update(knowledge_learned=len(mastered), knowledge_pending=len(current_knowledge - mastered),
                              knowledge_covered_current=len(mastered), knowledge_scope="current MLX independent closed-book VALID proof")
            except (ValueError, OSError, KeyError, TypeError, AttributeError) as problem:
                blocked = str(problem)
                result.update(knowledge_learned=0, knowledge_pending=len(current_knowledge), knowledge_covered_current=0)
        if blocked:
            result.update(state="waiting_validation", reason=blocked)
            return result
        if self.native_evolution:
            native = self.native_evolution("status", {})
            if self.worker and self.worker.is_alive() or self.state == "waiting_validation":
                # Read-only material readiness cannot erase a real idle-scheduler failure.
                native = {key: value for key, value in native.items() if key not in {"state", "reason"}}
            result.update(native)
            return result
        try:
            groups, pool, count = self._evolution_materials()
            result["experience"].update(current=count, continuing=pool is not None,
                                        steps=POLICY["evolution_steps"], batch_limit=PRODUCT_BATCH)
            args, records = self._manifest("evolution")
            signature = self._signature("evolution", args, records)
            if not self._attempted(signature, runs):
                result["pending_signature"] = signature
                result["reminder_due"] = config["mode"] == "manual" and self._choices(signature) is None
        except (ValueError, OSError, KeyError, AttributeError) as error:
            result["evolution_blocked_reason"] = str(error)
        return result

    def start(self):
        if self.monitor and self.monitor.is_alive():
            return
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name="luban-auto-monitor")
        self.monitor.start()

    def _monitor(self):
        while not self.stop_event.wait(2):
            try:
                if getattr(self.runtime, "state", "") != "ready":
                    continue
                self.tick()
            except Exception as error:
                self.state, self.reason = "waiting_validation", str(error)

    def tick(self, now=None):
        if getattr(self.runtime, "state", "") != "ready":
            return
        media = getattr(self.teacher_practice, "media_learning", None)
        media_pending = bool(media and media.teacher_pending())
        if not (self._available() or media_pending) or self.closing or self.stop_event.is_set():
            return
        with self.lock:
            if self.worker and self.worker.is_alive():
                return
            if self.busy() or self.runtime.state not in ({"ready", "stopped"} if media_pending else {"ready"}):
                return
            if time.monotonic() - self.last_activity < POLICY["idle_seconds"]:
                self.state, self.reason = "cooldown", "对话空闲后自动学习"
                return
            if media_pending:
                self.teacher_practice.stop_event.clear()
                self.worker = threading.Thread(target=self._work_practice, daemon=True, name="luban-media-teacher")
                self.worker.start()
                return
            runs = training_runs(self.runtime.memory, self.runtime.identity)
            if any(not is_product_run(row) and row["state"] in {"queued", "running", "paused", "committing"} for row in runs):
                self.state, self.reason = "waiting_validation", "研究任务由原实验处理，继续使用已采纳模型"
                return
            pending = self._pending(runs)
            if pending:
                if pending["kind"] == "learning" and not self.settings.snapshot()["automatic"]["learning_enabled"]:
                    self.state, self.reason = "paused", ""
                    return
                if pending["kind"] in {"evolution", "learning"} and self.native_evolution:
                    self.native_evolution("resume", {"run_id": pending["id"]})
                    return
                if is_native_mlx_run(pending):
                    self.state, self.reason = "waiting_validation", "MLX专家训练与恢复必须由原生MLX流程处理"
                    return
                try:
                    self._check_resume_rights(pending, runs)
                    self._launch(pending["kind"], pending["arguments"], pending["records"], run=pending)
                except ValueError as error:
                    pending["reason"] = str(error)
                    with self.runtime.memory.lock, self.runtime.memory.connect() as db:
                        db.execute("UPDATE training_runs SET state='rejected',body=?,updated=? WHERE id=?",
                                   (json.dumps(pending, ensure_ascii=False), time.time(), pending["id"]))
                    self.state, self.reason = "waiting_validation", str(error)
                return
            _, _, blocked = knowledge_keys(self.runtime, runs)
            if blocked:
                self.state, self.reason = "waiting_validation", blocked
                return
            # Ready, verified local material must not starve behind old teacher reviews,
            # each of which resets the idle cooldown. Native admission still owns every gate.
            native_result = None
            if self.native_evolution:
                native_result = self.native_evolution("tick", {"now": (now or datetime.now().astimezone()).isoformat()})
                if native_result.get("state") == "running":
                    self.state, self.reason = "running", native_result.get("reason", "后台正在学习")
                    return
            if self.teacher_practice and self.teacher_practice.pending():
                self.teacher_practice.stop_event.clear()
                self.worker = threading.Thread(target=self._work_practice, daemon=True, name="luban-teacher-practice")
                self.worker.start()
                return
            if self.on_idle and self.on_idle():
                return
            # Native expert learning owns its material signature and continuous schedule.
            for kind in (("router",) if self.native_evolution else ("router", "learning")):
                if kind == "learning" and not self.settings.snapshot()["automatic"]["learning_enabled"]:
                    continue
                try:
                    args, records = self._manifest(kind)
                except ValueError as error:
                    self.reason = str(error)
                    continue
                signature = self._signature(kind, args, records)
                if self._attempted(signature, runs):
                    continue
                self._launch(kind, args, records, signature=signature)
                return
            recheck = getattr(self.runtime, "recheck_external", None)
            binding = self.runtime.external_binding() if callable(recheck) else None
            if binding is not None:
                from .external_brain import ExternalBrain
                if ExternalBrain(self.runtime.memory).recheck_candidates(self.runtime.identity, binding, limit=1):
                    self.worker = threading.Thread(target=self._work_external, daemon=True, name="luban-external-check")
                    self.worker.start()
                    return
            if native_result is not None:
                self.state, self.reason = native_result.get("state", "waiting_materials"), native_result.get("reason", "等待原生进化")
                return
            config, now = self.settings.snapshot()["automatic"], now or datetime.now().astimezone()
            if config["mode"] != "auto" or now.strftime("%H:%M") < config["scheduled_time"]:
                self.state = "waiting_materials"
                return
            day_key = "schedule:" + now.strftime("%Y-%m-%d")
            if self._choices(day_key) or any(row.get("scheduled_occurrence") == day_key for row in runs):
                return
            try:
                args, records = self._manifest("evolution")
            except ValueError as error:
                self.state, self.reason = "waiting_validation", str(error)
                return
            signature = self._signature("evolution", args, records)
            if not self._attempted(signature, runs, now):
                self._launch("evolution", args, records, signature=signature, scheduled_occurrence=day_key)

    def _launch(self, kind, arguments, records, *, signature=None, run=None, scheduled_occurrence=None):
        if kind in {"evolution", "learning"} and self.native_evolution:
            # Do not create a Python TrainingSession or claim the native checkpoint.
            result = self.native_evolution("resume" if run else "learn" if kind == "learning" else "start", {"run_id": run["id"]} if run else {})
            return result["job_id"]
        if not self.command_lock.acquire(blocking=False):
            raise RuntimeError("另一个模型操作正在进行")
        try:
            if self.closing or self.busy() or self.runtime.state != "ready":
                raise RuntimeError("模型忙碌，等待空闲后继续")
            session = TrainingSession(self.runtime, kind, arguments, records, run_id=run["id"] if run else None)
            if signature:
                session.body["automatic_signature"] = signature
                session.body["scheduled_occurrence"] = scheduled_occurrence
                session._write("queued")
            self.runtime.training_session = session
            self.runtime.job = {"id": session.id, "kind": kind, "state": "queued", "step": 0, "steps": arguments["steps"]}
            self.worker = threading.Thread(target=self._work, args=(session,), daemon=True, name="luban-auto-" + kind)
            self.worker.start()
            return session.id
        finally:
            self.command_lock.release()

    def _work_practice(self):
        try:
            if not self.command_lock.acquire(blocking=False):
                return
            try:
                media = getattr(self.teacher_practice, "media_learning", None)
                if (not self.busy() and not self.closing and (self.runtime.state == "ready"
                        or self.runtime.state == "stopped" and media and media.teacher_pending())):
                    self.teacher_practice.run_one()
            finally:
                self.command_lock.release()
        finally:
            self.last_activity = time.monotonic()

    def _work_external(self):
        self.state, self.reason = "running", "正在核对已学内容，外脑继续兜底"
        try:
            if not self.command_lock.acquire(blocking=False):
                return
            try:
                if not self.busy() and not self.closing:
                    self.runtime.recheck_external(maximum=4)
            finally:
                self.command_lock.release()
            self.state, self.reason = "idle", "本地核对完成，未通过内容继续保留参考"
        except Exception as error:
            self.state, self.reason = "waiting_validation", str(error)
        finally:
            self.last_activity = time.monotonic()

    def _work(self, session):
        runtime, kind = self.runtime, session.body["kind"]
        self.state, self.reason = "running", "正在后台" + {"router": "学习决策", "learning": "学习新知识", "evolution": "进化"}[kind]
        try:
            if self.before_training:
                self.before_training(session)
            session.running()
            if kind == "router":
                result = runtime.calibrate_router()
            elif kind == "learning":
                result = runtime.learn(session.body["arguments"]["steps"])
            else:
                result = consolidate(runtime, **session.body["arguments"])
            if session.body["state"] not in {"committing", "committed"}:
                session.rejected(result.get("rejection_reason", "候选未通过检验"))
            self.state, self.reason = "idle", "已完成本轮检查"
        except TrainingPaused:
            session.pause()  # Training method has restored accepted weights and released its contexts.
            runtime.job = {**(runtime.job or {}), "state": "paused"}
            self.state, self.reason = "paused", "学习进度已保存，将在空闲时继续"
        except Exception as error:
            if session.body["state"] not in {"committing", "committed"}:
                session.rejected(str(error))
            runtime.job = {"id": session.id, "kind": kind, "state": "error", "error": str(error)}
            self.state, self.reason = "waiting_validation", str(error)
        finally:
            runtime.training_session = None
            self.last_activity = time.monotonic()
            if session.body["state"] == "committing":
                # Loader recovers the already-validated commit, after all old
                # optimizer/model contexts are gone. Do not overwrite intent.
                runtime.load()
            if kind == "evolution" and session.body["state"] == "committed" and self.on_core_upgrade:
                self.on_core_upgrade()
            if session.body["state"] == "committed" and self.on_neural_upgrade:
                self.on_neural_upgrade()

    def evolve(self, signature):
        with self.lock:
            if self.native_evolution:
                return self.native_evolution("evolve", {"signature": signature})
            args, records = self._manifest("evolution")
            expected = self._signature("evolution", args, records)
            if signature != expected or self._attempted(expected, training_runs(self.runtime.memory, self.runtime.identity)):
                raise ValueError("进化材料已变化或本轮已处理，请刷新后重试")
            job_id = self._launch("evolution", args, records, signature=expected)
            self._choices(expected, "started")
            return {"ok": True, "job_id": job_id}

    def defer(self, signature):
        with self.lock:
            if self.native_evolution:
                return self.native_evolution("defer", {"signature": signature})
            args, records = self._manifest("evolution")
            if signature != self._signature("evolution", args, records):
                raise ValueError("进化材料已变化，请刷新后重试")
            self._choices(signature, "deferred")
        return {"ok": True}

    def start_manual(self, kind, steps=None):
        with self.lock:
            if kind in {"evolution", "learning"} and self.native_evolution:
                if steps is not None:
                    raise ValueError("专家学习预算由验证方法自动决定，学会后提前结束")
                return self.native_evolution("learn" if kind == "learning" else "start", {})
            if kind == "evolution" and steps is not None:
                raise ValueError("产品进化固定450次更新，不接受自定义预算")
            args, records = self._manifest(kind)
            if steps is not None:
                args["steps"] = steps
            signature = self._signature(kind, args, records)
            if self._attempted(signature, training_runs(self.runtime.memory, self.runtime.identity)):
                raise ValueError("这批材料与训练配置已经处理，请等待新的核验资料")
            return {"job_id": self._launch(kind, args, records, signature=signature)}

    def pause(self, closing=False, timeout=55):
        with self.lock:
            self.closing = self.closing or closing
            self.last_activity = time.monotonic()
            if self.teacher_practice:
                self.teacher_practice.pause()
            session = getattr(self.runtime, "training_session", None)
            if getattr(self.runtime, "_external_probe_active", False):
                self.runtime.job["cancel_requested"] = True
            if session:
                session.pause_requested = True
            worker = self.worker
        if self.on_pause:
            self.on_pause()
        if worker and worker is not threading.current_thread():
            worker.join(timeout)
            if worker.is_alive():
                raise RuntimeError("正在保存有效训练进度，请稍后再退出")
        return {"ok": True, "paused": bool(session)}

    def close_status(self):
        worker = self.worker
        return {"running": bool(worker and worker.is_alive()), "resumable": self._available() and bool(
                self._pending(training_runs(self.runtime.memory, self.runtime.identity)))}

    def stop(self):
        self.stop_event.set()
        self.pause(closing=True)
