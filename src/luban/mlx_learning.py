"""MLX expert admission and read-only replay of the existing four-layer router."""
import gc
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .retention import messages, _validate_record, provenance

QUALITY_GATES = ("mlx_training_generation", "mlx_validation_generation",
                 "mlx_core_protection_baseline", "mlx_parent_protection_baseline",
                 "mlx_protection_generation", "mlx_baseline_correct_retained",
                 "mlx_deterministic", "mlx_same_inputs", "mlx_validation_loss_improved",
                 "mlx_replay_loss_retained", "mlx_routes_available")


NET_LEARNING_POLICY = "verified-knowledge-net-gain-v1"
NET_QUALITY_GATES = ("mlx_closed_book_gain", "mlx_net_gain", "mlx_deterministic",
                     "mlx_same_inputs", "mlx_validation_loss_improved", "mlx_routes_available")


def _check_net_learning(report, groups, by_id):
    """Recompute daily-learning gains from actual, routed closed-book observations."""
    from .generation_checks import evaluate_generation, validate_generation_check

    from .retention import normalized_context
    contexts, families = set(), {}
    for kind, rows in groups.items():
        for row in rows:
            context = normalized_context(row)
            if context in contexts:
                raise ValueError("MLX学习材料出现重复上下文或拆分泄漏")
            contexts.add(context)
            family = provenance(row).get("split_family")
            if kind != "protection" and family:
                if not isinstance(family, str) or family in families and families[family] != kind:
                    raise ValueError("MLX学习训练与独立检验模板家族泄漏")
                families[family] = kind
    routes = report.get("route_evidence", {}).get("rows")
    if (not isinstance(routes, list) or len(routes) != len(by_id)
            or any(not isinstance(row, dict) or row.get("id") not in by_id for row in routes)):
        raise ValueError("MLX学习路由材料覆盖不完整")
    route_map = {row["id"]: row for row in routes}
    if set(route_map) != set(by_id) or any(row.get("action") not in {"neural_expert", "local_core"}
                                          for row in routes):
        raise ValueError("MLX学习路由不可用或材料重复")
    observations = report.get("observations", {})

    def check_pair(pair, *, subset=False, candidate=False):
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(part, list) for part in pair)):
            raise ValueError("MLX学习缺少两遍真实闭卷生成")
        checked = []
        for part in pair:
            if any(not isinstance(row, dict) or row.get("id") not in by_id for row in part):
                raise ValueError("MLX学习闭卷材料来源不一致")
            observed = {row["id"]: row for row in part}
            if len(observed) != len(part) or (not subset and set(observed) != set(by_id)):
                raise ValueError("MLX学习闭卷材料覆盖不完整或重复")
            for identifier, row in observed.items():
                sample = by_id[identifier]
                check = validate_generation_check(provenance(sample).get("generation_check"),
                                                  default_expected=sample["answer"])
                if (not isinstance(row.get("answer"), str) or type(row.get("complete")) is not bool
                        or row.get("reference_used") is not False
                        or any(not isinstance(row.get(key), list) or not row[key]
                               or any(type(token) is not int or token < 0 for token in row[key])
                               for key in ("tokens", "input_tokens"))):
                    raise ValueError("MLX学习实际闭卷结果或标记不一致")
                actual = evaluate_generation(row["answer"], check, row["complete"])["passed"]
                route = route_map[identifier]
                if (row.get("passed") is not actual or row.get("route_action") != route["action"]
                        or row["input_tokens"] != route.get("input_tokens")):
                    raise ValueError("MLX学习实际闭卷结果或路由证据不一致")
                expected_source = "core" if route["action"] == "local_core" else "candidate" if candidate else "parent"
                if (not subset and (row.get("source") != expected_source
                                    or row.get("branch_enabled") is not (expected_source != "core"))):
                    raise ValueError("MLX学习观察不是实际回答路由来源")
            if checked and (set(observed) != set(checked[0]) or any(
                    any(row.get(key) != checked[0][identifier].get(key)
                        for key in ("answer", "tokens", "input_tokens", "complete", "passed", "route_action", "source"))
                    for identifier, row in observed.items())):
                raise ValueError("MLX学习两遍闭卷输出不一致")
            checked.append(observed)
        return checked

    baseline = check_pair(observations.get("baseline"))
    candidate = check_pair(observations.get("candidate"), candidate=True)
    check_pair(observations.get("parent", [[], []]), subset=True)
    for identifier, row in candidate[0].items():
        before = baseline[0][identifier]
        if row["input_tokens"] != before["input_tokens"]:
            raise ValueError("MLX学习前后闭卷输入不一致")
        if route_map[identifier]["action"] == "local_core" and any(
                row.get(key) != before.get(key) for key in ("answer", "tokens", "complete", "passed")):
            raise ValueError("MLX本地核心路由不能冒充专家学习增量")
    valid_ids = {row["id"] for row in groups["valid"]}
    protection_ids = {row["id"] for row in groups["protection"]}
    before = _mastered(observations["baseline"])
    after = _mastered(observations["candidate"])
    # TRAIN drives gradients directly; answer routing is assessed on held-out VALID.
    for row in groups["valid"]:
        if row["id"] not in before and route_map[row["id"]]["action"] != "neural_expert":
            raise ValueError("MLX未掌握新知识未被专家路由覆盖")
    gained = (after - before) & valid_ids
    forgotten = protection_ids - after
    progress = {"policy": NET_LEARNING_POLICY, "gained_ids": sorted(gained),
                "forgotten_ids": sorted(forgotten), "retained_ids": sorted(protection_ids & after),
                "mastered_ids": sorted(valid_ids & after), "net_gain": len(gained) - len(forgotten)}
    supplied = report.get("learning_progress")
    if not isinstance(supplied, dict) or set(supplied) != set(progress):
        raise ValueError("MLX学习净增统计缺失或字段不匹配")
    for key, expected in progress.items():
        value = supplied[key]
        if isinstance(expected, list):
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value) or sorted(value) != expected:
                raise ValueError("MLX学习净增统计与真实闭卷结果不一致")
        elif type(value) is not type(expected) or value != expected:
            raise ValueError("MLX学习净增统计与真实闭卷结果不一致")
    loss = report.get("loss", {})
    values = [loss.get(key, {}).get("mean") for key in ("baseline_validation", "validation")]
    if (any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in values)
            or not values[1] < values[0] * .995):
        raise ValueError("MLX学习独立检验损失未改善")
    for key, source in (("baseline_validation", "parent"), ("validation", "candidate")):
        item = loss[key]
        rows = item.get("observations")
        if (not isinstance(rows, list) or len(rows) != len(valid_ids)
                or any(not isinstance(row, dict) or row.get("id") not in valid_ids for row in rows)
                or {row["id"] for row in rows} != valid_ids):
            raise ValueError("MLX学习独立检验损失材料覆盖不完整")
        if any(type(row.get("loss")) not in {int, float} or not math.isfinite(row["loss"])
               or row["loss"] < 0 or row.get("source") != (
                   "core" if route_map[row["id"]]["action"] == "local_core" else source) for row in rows):
            raise ValueError("MLX学习独立检验损失观察无效")
        if not math.isclose(item["mean"], sum(row["loss"] for row in rows) / len(rows), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("MLX学习独立检验平均损失不匹配")
    pool, debt = {}, {}
    for key, destination in (("knowledge_pool", pool), ("relearning_records", debt)):
        rows = report.get(key)
        if not isinstance(rows, list):
            raise ValueError("MLX学习缺少已学知识或复习记录")
        contexts = set()
        for row in rows:
            _validate_record(row, generation_only=provenance(row).get("retention_generation_only") is True)
            context = normalized_context(row)
            if row["purpose"] != "validation" or row["id"] in destination or context in contexts:
                raise ValueError("MLX学习已学知识或复习记录重复或拆分不匹配")
            destination[row["id"]] = row
            contexts.add(context)
    train_ids = {row["id"] for row in groups["train"]}
    if (set(pool) & (train_ids | forgotten | (valid_ids - after))
            or not (valid_ids & after) <= set(pool) or not forgotten <= set(debt)
            or set(debt) & (valid_ids & after)):
        raise ValueError("MLX学习已学知识或复习记录与本轮结果不一致")
    for identifier in (set(pool) | set(debt)) & set(by_id):
        stored = pool.get(identifier, debt.get(identifier))
        sample = by_id[identifier]
        if (normalized_context(stored) != normalized_context(sample)
                or any(stored.get(key) != sample.get(key) for key in ("answer", "purpose"))
                or validate_generation_check(provenance(stored).get("generation_check"), default_expected=stored["answer"])
                != validate_generation_check(provenance(sample).get("generation_check"), default_expected=sample["answer"])):
            raise ValueError("MLX学习已学知识或复习材料内容变化")
    if type(report.get("review_cursor")) is not int or report["review_cursor"] < len(protection_ids):
        raise ValueError("MLX学习抽查进度无效")
    if not gained or progress["net_gain"] <= 0:
        raise ValueError("MLX学习没有独立检验净增益")


def _retention_class(row):
    p = provenance(row)
    if (p.get("original_core_baseline") is True or p.get("retention_scope") == "original_core"
            or p.get("protocol_id") == "explicit-model-identity-v1"
            or p.get("teaching_fact") in {"model_name", "model_fine_tuner", "model_positioning"}
            or isinstance(p.get("retention_from_version"), str)):
        return "hard"
    return "soft"


def _mastered(pair):
    if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(part, list) for part in pair):
        return set()
    first = {row.get("id"): row for row in pair[0] if isinstance(row, dict)}
    second = {row.get("id"): row for row in pair[1] if isinstance(row, dict)}
    return {identifier for identifier, row in first.items()
            if identifier and row.get("passed") is True and row.get("complete") is True
            and row.get("reference_used") is False
            and second.get(identifier, {}).get("passed") is True
            and second.get(identifier, {}).get("complete") is True
            and second.get(identifier, {}).get("reference_used") is False}


def _check_bounded_retention(groups, baseline, parent, candidate):
    before = _mastered(baseline) | _mastered(parent)
    after = _mastered(candidate)
    protection = groups["protection"]
    hard = {row["id"] for row in protection if _retention_class(row) == "hard"}
    soft = {row["id"] for row in protection if _retention_class(row) == "soft"}
    forgotten = before - after
    hard_forgotten = forgotten & hard
    soft_forgotten = forgotten & soft
    soft_count = len(before & soft)
    budget = 0 if soft_count < 5 else max(1, soft_count // 10)
    gained = (after - before) & {row["id"] for row in groups["train"] + groups["valid"]}
    if hard_forgotten or len(soft_forgotten) > budget or (soft_forgotten and len(gained) <= len(soft_forgotten)):
        raise ValueError("MLX专家软保护遗忘超过预算或没有净增量")


def formal_report(runtime, proof):
    """Validate one immutable adoption report; cache by its file identity."""
    import re
    from .generation_checks import evaluate_generation, validate_generation_check
    from .native_q4 import _stamp
    name = proof.get("report", "")
    if (not isinstance(name, str) or not re.fullmatch(r"learning-[a-f0-9]{12}\.json", name)
            or not isinstance(proof.get("report_sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", proof["report_sha256"])):
        raise ValueError("MLX专家缺少正式采纳报告绑定")
    path = Path(runtime.store) / name
    key = (str(path), _stamp(path), json.dumps(proof, sort_keys=True))
    cached = getattr(runtime, "_mlx_adoption_report", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    raw = path.read_bytes()
    if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != proof["report_sha256"]:
        raise ValueError("MLX专家采纳报告内容不匹配")
    report = json.loads(raw)
    daily = report.get("learning_policy") == NET_LEARNING_POLICY
    if report.get("learning_policy") not in {None, NET_LEARNING_POLICY}:
        raise ValueError("MLX专家学习策略未知")
    if (report.get("promoted") is not True or report.get("quality_checked") is not True
            or report.get("inference_engine") != "mlx-q4"
            or report.get("base_identity") != runtime.identity
            or report.get("core_version") != runtime.active_core
            or report.get("active_expert") != proof["expert_id"]
            or report.get("chosen_width") != proof["width"]
            or any(report.get("gates", {}).get(k) is not True for k in (NET_QUALITY_GATES if daily else QUALITY_GATES))):
        raise ValueError("MLX专家未通过完整生成、损失、保护和路由验证")
    groups = report.get("records", {})
    if set(groups) != {"train", "valid", "protection"} or any(not isinstance(x, list) for x in groups.values()):
        raise ValueError("MLX专家报告缺少完整材料")
    if len(groups["train"]) != 16 or not 2 <= len(groups["valid"]) <= 16:
        raise ValueError("MLX专家训练或独立检验材料不完整")
    by_id = {}
    for kind, rows in groups.items():
        for row in rows:
            _validate_record(row, generation_only=provenance(row).get("retention_generation_only") is True)
            if row["purpose"] != ("training" if kind == "train" else "validation"):
                raise ValueError("MLX专家材料拆分不匹配")
            if row["id"] in by_id and (daily or by_id[row["id"]] != row):
                raise ValueError("MLX专家材料ID冲突")
            by_id[row["id"]] = row
    if daily:
        _check_net_learning(report, groups, by_id)
    else:
        observations = report.get("observations", {})
        baseline = observations.get("baseline")
        parent = observations.get("parent", [[], []])
        passes = observations.get("candidate")
        if not isinstance(passes, list) or len(passes) != 2:
            raise ValueError("MLX专家缺少两遍真实闭卷生成")
        previous = None
        for observation in passes:
            if not isinstance(observation, list) or len(observation) != len(by_id):
                raise ValueError("MLX专家闭卷材料覆盖不完整")
            observed = {row["id"]: row for row in observation}
            if set(observed) != set(by_id):
                raise ValueError("MLX专家闭卷材料来源不一致")
            for identifier, row in observed.items():
                sample = by_id[identifier]
                check = validate_generation_check(provenance(sample).get("generation_check"), default_expected=sample["answer"])
                actual = evaluate_generation(row.get("answer", ""), check, row.get("complete") is True)["passed"]
                if (row.get("passed") is not actual or row.get("complete") is not True
                        or row.get("reference_used") is not False
                        or not isinstance(row.get("tokens"), list) or not row["tokens"]
                        or not isinstance(row.get("input_tokens"), list) or not row["input_tokens"]):
                    raise ValueError("MLX专家实际闭卷结果或标记不一致，未通过原标准")
                if not actual and identifier in {item["id"] for item in groups["train"] + groups["valid"]}:
                    raise ValueError("MLX专家训练或独立检验材料未通过原标准")
                if previous is not None and any(row.get(k) != previous[identifier].get(k)
                                                for k in ("tokens", "input_tokens", "complete", "route_action")):
                    raise ValueError("MLX专家两遍闭卷输出不一致")
            previous = observed
        _check_bounded_retention(groups, baseline, parent, passes)
    if _stamp(path) != key[1]:
        raise ValueError("MLX专家采纳报告读取期间变化")
    runtime._mlx_adoption_report = (key, report)
    return report


def accepted_report(runtime, binary=None):
    """None for an unqualified trial; formal observations remain engine-bound."""
    if getattr(runtime, "native_inference_precision", None) != "mlx-q4":
        return None
    storage = getattr(runtime, "loaded_storage", None)
    if not storage:
        return None
    path = Path(storage) / "runtime/mlx/expert-proof.json"
    if not path.is_file():
        return None
    proof = json.loads(path.read_text())
    if proof.get("schema") != "mlx-expert-learning-v1":
        return None
    from .native_mlx import expert_binding
    if binary is None:
        from .product_inference import native_binary
        binary = native_binary(runtime)
    if expert_binding(runtime, binary) is None:
        return None
    return formal_report(runtime, proof)


def check_routes(owner, records):
    """Replay production routing with one resident Q4 prefix under the unloaded lease."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from .native_engine import NativeEngine
    from .native_mlx import model_binding
    from .native_mlx_routing import pooled_features
    import threading
    from .decision import FastDecisionHead
    from .runtime import Runtime

    config = owner.settings.snapshot()
    if config.get("native_inference_precision") != "mlx-q4":
        raise ValueError("MLX学习路由需要当前MLX模型")
    if not isinstance(records, list) or not 1 <= len(records) <= 256:
        raise ValueError("MLX路由材料数量无效")
    seen = set()
    for row in records:
        _validate_record(row, generation_only=provenance(row).get("retention_generation_only") is True)
        if row["id"] in seen:
            raise ValueError("MLX路由材料ID重复")
        seen.add(row["id"])
    store = Path(owner.store)
    pointer_paths = [store / name for name in ("active-core.json", "active.json", "active-router.json")]
    pointers = [path.read_bytes() for path in pointer_paths]
    core, expert, router = [json.loads(value) for value in pointers]
    if (core["identity"] != owner.identity or expert["identity"] != owner.identity
            or router["identity"] != owner.identity or expert.get("core_version") != core["version"]
            or router.get("core_version") != core["version"]):
        raise ValueError("MLX路由来源绑定不一致")
    router_path = store / (router["id"] + ".json")
    router_bytes = router_path.read_bytes()
    meta = json.loads(router_bytes)
    if (meta.get("feature_source") != "mlx-q4-prefix"
            or meta.get("depth") != 4 or meta.get("calibrated") is not True
            or meta.get("expert_id") != expert["id"]
            or hashlib.sha256(router_bytes).hexdigest() != router["metadata_sha256"]):
        raise ValueError("MLX路由来源未校准或不匹配")
    weights_path = store / (router["id"] + ".safetensors")
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("MLX路由权重绑定不一致")
    temporary = Runtime.__new__(Runtime)
    temporary.device, temporary.dtype = "cpu", torch.float32
    temporary.native_inference_precision = "mlx-q4"
    temporary.native_mlx_model_path = config["native_mlx_model_path"]
    temporary.native_engine = temporary.router = None
    temporary._stop = threading.Event()
    result = []
    try:
        model_path = config["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        root = Path(__file__).resolve().parents[2]
        binary = next((path for path in (root / "native/luban-mlx-worker", root / "build/native/luban-mlx-worker")
                       if path.is_file()), None)
        if binary is None:
            raise RuntimeError("MLX 原生推理程序未安装")
        manifest = hashlib.sha256((Path(model_path) / "upstream-file-manifest.json").read_bytes()).hexdigest()
        temporary.native_engine = NativeEngine(binary, model_path, manifest, precision="mlx-q4",
                                               mlx_model=model_binding(temporary))
        temporary.router = FastDecisionHead(2048).eval()
        temporary.router.load_state_dict(load_file(str(weights_path), device="cpu"))
        # This is mathematical head/scope replay, not permission to answer or train.
        # The unchanged production decision function is reused without trial admission.
        temporary.router_meta = {**meta, "execution_scope": "route-replay"}
        temporary.router_meta.pop("inference_engine", None)
        temporary.branch = SimpleNamespace(expert_id=expert["id"], _paths=lambda _: (
            store / "layer-23" / (expert["id"] + ".safetensors"),
            store / "layer-23" / (expert["id"] + ".json")))
        temporary._ready = lambda: None
        temporary.external_hint = lambda _: None
        temporary.shallow = lambda tokens: pooled_features(temporary, tokens)
        for row in records:
            context = messages(row)
            ids = tokenizer.apply_chat_template(context, add_generation_prompt=True,
                                                tokenize=True, enable_thinking=False)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            if not ids or len(ids) > 2048:
                raise ValueError("MLX专家学习路由输入超过本轮2048词元上限")
            temporary._tokens = lambda _m, shallow=False, ids=ids: torch.tensor([ids])
            decision = Runtime.choose_route(temporary, context)
            result.append({"id": row["id"], "action": decision["action"],
                           "source": decision["source"], "input_tokens": ids})
        if any(path.read_bytes() != value for path, value in zip(pointer_paths, pointers)):
            raise ValueError("MLX路由检查期间活动来源变化")
        if router_path.read_bytes() != router_bytes:
            raise ValueError("MLX路由检查期间校准来源变化")
    finally:
        if temporary.native_engine is not None:
            temporary.native_engine.close()
        temporary.native_engine = temporary.router = None
        gc.collect()
    return {"identity": owner.identity, "core_version": core["version"],
            "expert_id": expert["id"], "router_id": router["id"], "router_sha256": meta["sha256"],
            "rows": result, "released": True, "feature_source": "mlx-q4-prefix",
            "scope": "resident MLX Q4 four-layer prefix, production decision head and expert scope"}
