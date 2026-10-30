"""Family-disjoint route evaluation. Test rows never fit temperature or threshold.

Explicit provenance splits are preferred. Legacy training rows stay in train;
legacy validation families are deterministically assigned to held-out splits.
Missing family labels can be audited, but cannot establish family independence.
"""
import hashlib
import json
import math
import unicodedata

import torch
from torch.nn import functional as F
from .retention import verified_source

SPLITS = ("train", "development", "calibration", "test")
# Keep the runtime's conservative abstention boundary when publishing a
# learned expert route.  A finite route set that only passes at threshold 0
# would send unknown prompts to the new branch by default.
# A gated expert must have a high-confidence route before it can replace the
# protected core.  Lower-confidence rows abstain to local_core.
MIN_CALIBRATED_THRESHOLD = 0.85


def _normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_route_samples(records, actions, seed=42, require_family=True):
    """Return copied, de-duplicated rows; reject conflicting labels/splits/families.

    Metadata accepts provenance.family_group and provenance.split. A known
    training purpose is never silently moved to evaluation, nor vice versa.
    Without an old purpose, family hashing assigns 60/15/15/10 percent shares;
    it never resamples families to make a failed class-count gate pass.
    """
    result = {name: [] for name in SPLITS}
    families, prompts = {}, {}
    for row in sorted(records, key=lambda r: str(r.get("id", ""))):
        provenance = row.get("provenance", {})
        provenance = json.loads(provenance) if isinstance(provenance, str) else provenance
        if not isinstance(provenance, dict):
            raise ValueError("路由材料 provenance 必须为对象")
        action = provenance.get("route_action")
        if action not in actions:
            continue
        if not row.get("approved") or not row.get("training_allowed") or not verified_source(row):
            continue
        prompt = _normalize(row.get("prompt", ""))
        if not prompt:
            raise ValueError("路由材料问题不能为空")
        family = provenance.get("family_group")
        known_family = isinstance(family, str) and bool(family.strip())
        if not known_family and require_family:
            raise ValueError("路由材料需要 family_group；问题哈希不能证明任务家族独立")
        family = family.strip() if known_family else "legacy-prompt:" + hashlib.sha256(prompt.encode()).hexdigest()
        split = provenance.get("split")
        purpose = row.get("purpose")
        if split is None:
            score = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(), 16) / 2**256
            if purpose in {"training", "replay"}:
                split = "train"
            elif purpose == "validation":
                split = SPLITS[1 + min(int(score * 3), 2)]
            else:
                split = "train" if score < .6 else "development" if score < .75 else "calibration" if score < .9 else "test"
        if split not in SPLITS:
            raise ValueError("路由 split 必须为 train/development/calibration/test")
        if ((purpose in {"training", "replay"} and split != "train") or
                (purpose == "validation" and split == "train")):
            raise ValueError("不能把已用于训练的材料移到检验，或把保护材料移到训练")
        if family in families and families[family] != split:
            raise ValueError("同一 family_group 不能跨 split")
        families[family] = split
        signature = (action, row.get("answer", "").strip(), family, split,
                     json.dumps(provenance.get("messages"), sort_keys=True, ensure_ascii=False))
        if prompt in prompts:
            if prompts[prompt] != signature:
                raise ValueError("重复问题的标签、答案、family_group 或 split 冲突")
            continue
        prompts[prompt] = signature
        result[split].append({**row, "family_group": family, "split": split,
                              "route_action": action, "family_verified": known_family})
    return result


def validate_route_splits(splits, actions):
    """Validate before fitting: 3 train and 2 independent rows/class/held-out split."""
    supported = [action for action in actions if any(r["route_action"] == action for r in splits["train"])]
    if len(supported) < 2 or "local_core" not in supported:
        raise ValueError("至少标注本地核心与专家/教师两种路由")
    for split in SPLITS:
        rows = splits[split]
        if any(not row["family_verified"] for row in rows):
            raise ValueError("缺少任务家族的历史材料不能用于独立路由晋升")
        if any(row["route_action"] not in supported for row in rows):
            raise ValueError("检验材料包含未训练的路由类别")
        minimum = 3 if split == "train" else 2
        for action in supported:
            if sum(row["route_action"] == action for row in rows) < minimum:
                raise ValueError(f"{split} 每类路由至少 {minimum} 条独立问题，且家族不能跨集合")
    return supported


def _scores(logits, targets):
    logits = torch.as_tensor(logits).detach().to(device="cpu", dtype=torch.float64)
    targets = torch.as_tensor(targets).detach().to(device="cpu")
    if (logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2 or
            targets.shape != (logits.shape[0],) or targets.dtype == torch.bool or
            targets.is_floating_point()):
        raise ValueError("校准需要非空二维 logits 与整数类别标签")
    targets = targets.long()
    if (torch.isnan(logits).any() or torch.isposinf(logits).any() or
            not torch.isfinite(logits).any(dim=1).all() or
            (targets < 0).any() or (targets >= logits.shape[1]).any() or
            not torch.isfinite(logits.gather(1, targets[:, None])).all()):
        raise ValueError("校准 logits 或标签无效")
    return logits, targets


def fit_temperature(logits, targets):
    """Fit one bounded positive temperature to calibration NLL, including T=1 baseline."""
    logits, targets = _scores(logits, targets)
    # Masked unsupported classes have -inf logits; avoid 0 * inf in d/dT.
    logits = torch.where(torch.isneginf(logits), -1e9, logits)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=40, line_search_fn="strong_wolfe")
    candidates = [(float(F.cross_entropy(logits, targets)), 1.0)]

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(.05, 20)
        loss = F.cross_entropy(logits / temperature, targets)
        if not torch.isfinite(loss):
            raise ValueError("温度拟合损失异常")
        candidates.append((float(loss.detach()), float(temperature.detach())))
        loss.backward()
        return loss

    optimizer.step(closure)
    return min(candidates)[1]


def probability_metrics(logits, targets, temperature=1.0, bins=10):
    """Multiclass Brier is sum squared class errors; ECE uses predicted-class confidence."""
    logits, targets = _scores(logits, targets)
    if not math.isfinite(temperature) or temperature <= 0 or not isinstance(bins, int) or bins < 1:
        raise ValueError("温度与可靠性分箱无效")
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, decisions = probabilities.max(dim=-1)
    correct = decisions == targets
    reliability, ece = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        chosen = (confidence >= lower) & ((confidence <= upper) if index == bins - 1 else (confidence < upper))
        count = int(chosen.sum())
        accuracy = float(correct[chosen].double().mean()) if count else None
        mean = float(confidence[chosen].mean()) if count else None
        if count:
            ece += count / len(targets) * abs(accuracy - mean)
        reliability.append({"lower": lower, "upper": upper, "count": count,
                            "accuracy": accuracy, "confidence": mean})
    return {"samples": len(targets), "accuracy": float(correct.double().mean()),
            "nll": float(F.cross_entropy(logits / temperature, targets)),
            "brier": float(((probabilities - F.one_hot(targets, logits.shape[1])) ** 2).sum(dim=1).mean()),
            "ece": ece, "reliability_bins": reliability, "probabilities": probabilities.tolist(),
            "targets": targets.tolist(), "decisions": decisions.tolist(), "confidence": confidence.tolist()}


def _selective(metrics, threshold, actions, supported_actions):
    confidence = torch.tensor(metrics["confidence"], dtype=torch.float64)
    decisions, targets = torch.tensor(metrics["decisions"]), torch.tensor(metrics["targets"])
    selected = confidence >= threshold
    count = int(selected.sum())
    per_action = {}
    for action in supported_actions:
        index = actions.index(action)
        accepted = selected & (decisions == index)
        actual = targets == index
        per_action[action] = {"selected": int(accepted.sum()),
                              "precision": float((targets[accepted] == index).double().mean()) if accepted.any() else None,
                              "recall": float((accepted & actual).sum() / actual.sum()) if actual.any() else None}
    effective = torch.where(selected, decisions, actions.index("local_core"))
    return {"threshold": threshold, "selected": count, "coverage": count / len(targets),
            "accuracy": float((decisions[selected] == targets[selected]).double().mean()) if count else None,
            "fallback_accuracy": float((effective == targets).double().mean()), "action_metrics": per_action}


def _passes(metrics):
    return all(value["selected"] >= 2 and value["precision"] == 1.0 for value in metrics["action_metrics"].values())


def evaluate_router(logits_by_split, targets_by_split, actions, supported_actions):
    """Freeze calibration temperature, choose development threshold, then test once.

    This gate measures labelled route agreement, not answer correctness or a
    permission to call a teacher. Insufficient final evidence refuses promotion.
    """
    if len(set(supported_actions)) < 2 or "local_core" not in supported_actions or any(a not in actions for a in supported_actions):
        raise ValueError("路由类别无效")
    logits, targets = {}, {}
    for split in SPLITS[1:]:
        logits[split], targets[split] = _scores(logits_by_split[split], targets_by_split[split])
        if logits[split].shape[1] != len(actions):
            raise ValueError("路由 logits 类别数不一致")
        for index, action in enumerate(actions):
            if action not in supported_actions:
                if (targets[split] == index).any():
                    raise ValueError("检验材料包含未训练的路由类别")
                logits[split] = logits[split].clone()
                logits[split][:, index] = -torch.inf
    temperature = fit_temperature(logits["calibration"], targets["calibration"])
    evaluated = {split: probability_metrics(logits[split], targets[split], temperature) for split in SPLITS[1:]}
    development = evaluated["development"]
    candidates = [_selective(development, threshold, actions, supported_actions)
                  for threshold in sorted({MIN_CALIBRATED_THRESHOLD,
                                           *[value for value in development["confidence"]
                                             if value >= MIN_CALIBRATED_THRESHOLD]})]
    valid = [candidate for candidate in candidates if _passes(candidate)]
    chosen = max(valid, key=lambda c: (c["coverage"], -c["threshold"])) if valid else _selective(
        development, MIN_CALIBRATED_THRESHOLD, actions, supported_actions)
    test = _selective(evaluated["test"], chosen["threshold"], actions, supported_actions)
    counts_ok = all(int((targets[split] == actions.index(action)).sum()) >= 2
                    for split in SPLITS[1:] for action in supported_actions)
    promoted = bool(valid and chosen["threshold"] >= MIN_CALIBRATED_THRESHOLD and counts_ok
                   and evaluated["test"]["accuracy"] >= .8 and _passes(test))
    return {"calibrated": promoted, "temperature": temperature, "threshold": chosen["threshold"],
            "supported_actions": supported_actions, "development_selection": chosen, "test_selection": test,
            "split_metrics": evaluated, "action_metrics": test["action_metrics"],
            "test_accuracy": evaluated["test"]["accuracy"], "threshold_candidates": candidates,
            "minimum_counts_passed": counts_ok,
            "scope": "family-disjoint labelled route agreement; not general reliability or teacher answer quality"}


def calibrate_expert_scope(features_by_split, targets_by_split, actions, supported_actions):
    """Build a conservative one-class boundary for the gated expert.

    The multiclass head can be confidently wrong outside its finite calibration
    families.  A normalized centroid of expert training representations adds a
    second admission test; development selects a threshold that rejects every
    non-expert row, and the held-out test split must preserve that separation.
    """
    if "neural_expert" not in supported_actions:
        return {"required": False, "passed": True, "scope": "没有专家路由；不启用专家范围门"}
    expert_index = actions.index("neural_expert")
    try:
        train_features = torch.as_tensor(features_by_split["train"]).detach().cpu().float()
        train_targets = torch.as_tensor(targets_by_split["train"]).detach().cpu().long()
    except (KeyError, TypeError, ValueError):
        raise ValueError("专家范围校准缺少训练表征")
    if train_features.ndim != 2 or train_targets.shape != (train_features.shape[0],):
        raise ValueError("专家范围校准表征形状无效")
    expert_train = train_features[train_targets == expert_index]
    if expert_train.shape[0] < 3:
        raise ValueError("专家范围至少需要3条训练表征")
    centroid = F.normalize(expert_train.mean(dim=0), dim=0)
    if not torch.isfinite(centroid).all():
        raise ValueError("专家范围中心非有限")
    similarities, targets = {}, {}
    for split in SPLITS[1:]:
        features = torch.as_tensor(features_by_split[split]).detach().cpu().float()
        split_targets = torch.as_tensor(targets_by_split[split]).detach().cpu().long()
        if features.ndim != 2 or split_targets.shape != (features.shape[0],):
            raise ValueError("专家范围检验表征形状无效")
        values = F.normalize(features, dim=1) @ centroid
        if not torch.isfinite(values).all():
            raise ValueError("专家范围相似度非有限")
        similarities[split], targets[split] = values, split_targets
    development = similarities["development"]
    development_targets = targets["development"]
    candidate_thresholds = sorted({0.0, *[float(value) for value in development]})
    valid = []
    for threshold in candidate_thresholds:
        selected = development >= threshold
        expert_selected = selected & (development_targets == expert_index)
        non_expert_selected = selected & (development_targets != expert_index)
        if int(expert_selected.sum()) >= 2 and not bool(non_expert_selected.any()):
            valid.append((int(expert_selected.sum()), threshold))
    if valid:
        _, threshold = max(valid, key=lambda item: (item[0], -item[1]))
    else:
        threshold = 2.0
    test_values, test_targets = similarities["test"], targets["test"]
    test_selected = test_values >= threshold
    test_expert = test_selected & (test_targets == expert_index)
    test_non_expert = test_selected & (test_targets != expert_index)
    passed = bool(valid and int(test_expert.sum()) >= 2 and not bool(test_non_expert.any()))
    return {
        "required": True,
        "passed": passed,
        "centroid": centroid.tolist(),
        "threshold": float(threshold),
        "development": {"expert_selected": int((development >= threshold).logical_and(development_targets == expert_index).sum()),
                         "non_expert_selected": int((development >= threshold).logical_and(development_targets != expert_index).sum())},
        "test": {"expert_selected": int(test_expert.sum()),
                 "non_expert_selected": int(test_non_expert.sum())},
        "scope": "专家训练表征的一类边界；独立反例未进入专家调用",
    }
