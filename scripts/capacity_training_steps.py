"""Thin capacity scheduling/proof validation, not a training or model entry.

All observations, optimizer counts and checkpoint tensor declarations are
caller-supplied evidence. Separate root binding/review and MODEL GO remain
required; this module cannot authenticate execution or deserialize checkpoints.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import sys

import select_capacity_rows as select
from run_cached_training_audit import safe, sha
from run_training_method_pilot import correct
from luban.generation_checks import validate_generation_check

PROTOCOL = "fictional-capacity-continuous-2-4-6-three-ledgers-check-policy-v2"
EXPOSURES = (2, 4, 6)
RESTORATION = {"original_tensor", "flags_grads", "modes", "rng", "mps_rng", "core_branch",
               "forward_hooks", "backward_hooks", "pointer"}


def _ids(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("Distinct nonempty ordered role IDs required")
    return list(values)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Exact SHA256 evidence binding required")
    return value


def build_capacity_schedule(new_ids, replay_ids, *, method, batch_size, learning_rate, wall_seconds,
                            check_policy="every50"):
    """One continuous original-start arm; explicit method/B/LR/cap, no optimum selection."""
    new_ids, replay_ids = _ids(new_ids), _ids(replay_ids)
    if len(new_ids) not in select.COUNTS or set(new_ids) & set(replay_ids):
        raise ValueError("Capacity N50/100/200/300 and disjoint roles required")
    if (method not in ("native", "cache") or type(batch_size) is not int or batch_size not in (1, 2, 4)
            or method == "native" and batch_size != 1 or type(learning_rate) not in (int, float)
            or learning_rate != 1e-5 or not _positive(wall_seconds)
            or check_policy not in ("every50", "per_new_exposure")):
        raise ValueError("Explicit supported method/B, existing helper LR1e-5 and finite wall cap required")
    n, replay_n = len(new_ids), len(replay_ids)
    total = 2 * n * EXPOSURES[-1]
    markers = {2 * n * exposure: exposure for exposure in EXPOSURES}
    flushes = sorted(set(range(50, total + 1, 50)) | set(markers))
    checks = flushes if check_policy == "every50" else [2 * n * exposure for exposure in range(1, 7)]
    events = [{"ce_event": index + 1, "ce_role": "new" if index % 2 == 0 else "replay",
        "ce_id": (new_ids[(index // 2) % n] if index % 2 == 0 else replay_ids[(index // 2) % replay_n]),
        "replay_kl_id": replay_ids[(index // 2) % replay_n], "pair_index": index // 2}
        for index in range(total)]
    counts = {key: dict.fromkeys(ids, 0) for key, ids in
              (("new_ce", new_ids), ("replay_ce", replay_ids), ("replay_kl", replay_ids))}
    segments, updates, start = [], 0, 0
    for end in flushes:
        by_role = {role: [event for event in events[start:end] if event["ce_role"] == role]
                   for role in ("new", "replay")}
        batches = []
        for offset in range(0, max(map(len, by_role.values())), batch_size):
            for role in ("new", "replay"):
                chosen = by_role[role][offset:offset + batch_size]
                if not chosen:
                    continue
                updates += 1
                batches.append({"optimizer_update": updates, "ce_role": role,
                    "ce_event_ids": [e["ce_event"] for e in chosen], "ce_ids": [e["ce_id"] for e in chosen],
                    "replay_kl_ids": [e["replay_kl_id"] for e in chosen],
                    "actual_batch_size": len(chosen), "short_tail": len(chosen) < batch_size})
        for event in events[start:end]:
            counts[event["ce_role"] + "_ce"][event["ce_id"]] += 1
            counts["replay_kl"][event["replay_kl_id"]] += 1
        segments.append({"ce_event_start": start + 1, "check_ce_event": end, "check_required": end in checks,
            "batches": batches,
            "optimizer_updates_at_check": updates, "double_new_proof": end in markers,
            "new_exposure_marker": markers.get(end),
            "per_id_exposures": {key: dict(values) for key, values in counts.items()},
            "exposure_totals": {key: sum(values.values()) for key, values in counts.items()}})
        start = end
    membership = [event for segment in segments for batch in segment["batches"] for event in batch["ce_event_ids"]]
    if Counter(membership) != Counter(range(1, total + 1)):
        raise RuntimeError("Omitted/duplicated CE exposure events")
    return {"protocol": PROTOCOL, "new_ids": new_ids, "replay_ids": replay_ids, "method": method,
        "tensor_batch": batch_size, "gradient_accumulation": 1, "learning_rate": learning_rate,
        "wall_seconds": wall_seconds, "original_core": None, "original_tensor_sha256": select.ORIGINAL,
        "exposure_markers": list(EXPOSURES), "ce_events": total, "optimizer_updates": updates,
        "check_policy": check_policy, "checks_ce_events": checks, "flush_ce_events": flushes, "segments": segments,
        "ordering": "Zero-based even NEW CE/odd REPLAY CE (1-based odd NEW/even REPLAY) plus one REPLAY KL per CE event; independent NEW/R sample modulo. Within-role order preserved; B>1 grouped Adam order/trajectory changes.",
        "model_started": False, "teacher_calls": 0, "publication": False, "execution_ready": False}


def _qualification(receipt):
    """Final result AND independent post-audit, never merely qualification.json existence."""
    values = {}
    for name in ("qualification", "result", "root_post"):
        path = safe(receipt[name + "_path"])
        if sha(path) != _sha(receipt[name + "_sha256"]):
            raise ValueError("Qualification/final result/root post file SHA drift")
        values[name] = select.read_json(path)
    package, result, root = (values[k] for k in ("qualification", "result", "root_post"))
    protocol_sha = _sha(receipt["protocol_sha256"])
    fixture = package.get("binding", {}).get("scope") == "cpu_fixture"
    if (result.get("protocol") != "fictional-capacity-original400-qualification-v1"
            or result.get("protocol_sha256") != protocol_sha or result.get("passed") is not True
            or result.get("state") != "complete_waiting_root_evidence_review"
            or result.get("qualification_sha256") != receipt["qualification_sha256"]
            or result.get("records") != package.get("records") or result.get("preserved") is not True
            or result.get("model_released") is not True or result.get("actual_os_lock_reacquired_closed") is not True
            or set(result.get("restoration", {})) != RESTORATION
            or any(value is not True for value in result["restoration"].values())
            or not _positive(result.get("wall_seconds")) or result["wall_seconds"] >= 1800
            or type(result.get("wall_cap_seconds")) is not int or result["wall_cap_seconds"] != 1800
            or type(result.get("optimizer_steps")) is not int or result["optimizer_steps"] != 0
            or type(result.get("teacher_calls")) is not int or result["teacher_calls"] != 0 or result.get("publication") is not False
            or any(key.endswith("error") for key in result)
            or root.get("passed") is not True or type(root.get("process_exit_code")) is not int
            or root["process_exit_code"] != 0 or root.get("pid_dead") is not True
            or root.get("result_sha256") != receipt["result_sha256"]
            or root.get("protocol_sha256") != protocol_sha
            or root.get("scope") != ("cpu_fixture" if fixture else "actual_root_post")):
        raise ValueError("Completed/restored/released/exit0/root-post qualification required; late-save failure blocks")
    if package.get("binding", {}).get("scope") not in ("actual_closed_original", "cpu_fixture"):
        raise ValueError("Original actual qualification or explicitly labelled fixture required")
    selection = select.select_capacity_rows(select.CORPUS, package, package["binding"])
    if not fixture and (select.digest(result.get("source_after")) != package["binding"]["source_sha256"]
            or select.digest(result.get("official_after")) != package["binding"]["official_sha256"]):
        raise ValueError("Actual qualification source/official summaries must match full final result")
    train = select._materials(select.read_json(select.CORPUS))
    return package, selection, train, fixture


def _observations(rows, observations, binding, tensor_sha):
    # Reuse existing correctness/finite predicates; add complete causal raw/EOS
    # and immutable sample/state guards for the evidence validator.
    passed = correct(rows, observations)
    for row, observed in zip(rows, observations):
        check = validate_generation_check(select.provenance(row)["generation_check"], default_expected=row["answer"])
        maximum = check.get("max_new_tokens", 192)
        tokens = observed.get("raw_completion_token_ids")
        if (observed.get("sample_sha256") != select.digest(row)
                or observed.get("actual_base_identity") != select.IDENTITY or observed.get("core_version") is not None
                or observed.get("actual_tensor_sha256") != tensor_sha
                or not isinstance(tokens, list) or not tokens or len(tokens) > maximum
                or any(type(value) is not int or value < 0 for value in tokens)
                or any(value in binding["eos_token_ids"] for value in tokens[:-1])
                or observed["complete"] != (tokens[-1] in binding["eos_token_ids"])
                or not observed["complete"] and len(tokens) != maximum
                or type(observed.get("completion_tokens")) is not int or observed["completion_tokens"] != len(tokens)
                or type(observed.get("max_new_tokens")) is not int or observed["max_new_tokens"] != maximum
                or observed.get("thinking") != "off" or observed.get("do_sample") is not False
                or not isinstance(observed.get("decoded_completion_unstripped"), str)
                or observed["decoded_completion_unstripped"].strip() != observed["answer"]):
            raise ValueError("Actual frozen full-context/raw/EOS/sample/original identity proof drift")
    return passed


def observe_capacity_checkpoint(schedule, checkpoint, qualification_receipt, historical_protection):
    """Validate already-collected proofs; does not generate, update, publish or read Torch payloads.

    Historical rows/baseline must be independently bound to their approved full
    13 original-correct NEW +55 REPLAY records. Checkpoint artifact byte SHA is
    verified; its tensor/master/Adam metadata still requires caller/root audit.
    """
    expected = build_capacity_schedule(schedule["new_ids"], schedule["replay_ids"], method=schedule["method"],
        batch_size=schedule["tensor_batch"], learning_rate=schedule["learning_rate"], wall_seconds=schedule["wall_seconds"],
        check_policy=schedule["check_policy"])
    if expected != schedule:
        raise ValueError("Fixed schedule drift")
    package, selection, train, fixture = _qualification(qualification_receipt)
    chosen = selection["selections"][str(len(schedule["new_ids"]))]
    if chosen.get("state") == "waiting" or chosen["ids"] != schedule["new_ids"]:
        raise ValueError("N must be exact frozen TRAIN-only nested unknown selection")
    rows_by_id = {row["id"]: row for row in train}
    news = [rows_by_id[value] for value in schedule["new_ids"]]
    baseline = {value["id"]: value for value in package["records"]}
    selected_baseline = [baseline[row["id"]] for row in news]
    original_correct = [row for row in train if baseline[row["id"]]["passed"]]
    historical, historical_baseline = historical_protection["rows"], historical_protection["baseline"]
    if (select.digest(historical) != _sha(qualification_receipt["historical_rows_sha256"])
            or select.digest(historical_baseline) != _sha(qualification_receipt["historical_baseline_sha256"])
            or len(historical) != 68 or len({r["id"] for r in historical}) != 68
            or Counter(select.provenance(r).get("evolution_role") for r in historical) != {"new": 13, "replay": 55}
            or set(rows_by_id) & {r["id"] for r in historical}
            or _observations(historical, historical_baseline, package["binding"], select.ORIGINAL) != {r["id"] for r in historical}):
        raise ValueError("Exact independently bound original-correct historical13NEW/55REPLAY protection required")
    obligations = original_correct + historical
    if [r["id"] for r in obligations] != checkpoint.get("protected_ids"):
        raise ValueError("Cannot omit original-correct unselected400 or historical protection")
    segments = [segment for segment in schedule["segments"]
                if segment["check_required"] and segment["check_ce_event"] == checkpoint.get("ce_events")]
    if len(segments) != 1:
        raise ValueError("Checks only at exact declared check-policy boundaries")
    segment = segments[0]
    if (checkpoint.get("per_id_exposures") != segment["per_id_exposures"]
            or type(checkpoint.get("optimizer_updates")) is not int
            or checkpoint["optimizer_updates"] != segment["optimizer_updates_at_check"]):
        raise ValueError("Actual NEW CE/REPLAY CE/KL per-ID ledgers or Adam count differ")
    candidate = _sha(checkpoint["candidate_tensor_sha256"])
    frozen = checkpoint.get("frozen_tensor_sha256")
    if frozen != [candidate] * (4 if segment["double_new_proof"] else 3):
        raise ValueError("Actual W before/between/after selected/protection proofs must stay frozen")
    artifact = checkpoint["checkpoint_artifact"]
    if (sha(safe(artifact["path"])) != _sha(artifact["file_sha256"])
            or artifact.get("candidate_tensor_sha256") != candidate
            or artifact.get("optimizer_updates") != checkpoint["optimizer_updates"]
            or artifact.get("payload_scope") != "full_candidate_master_adam_rng_progress"):
        raise ValueError("Actual saved checkpoint byte/tensor/Adam progress binding required")
    first = checkpoint["new_first"]
    _observations(news, first, package["binding"], candidate)
    finite_one = correct(news, first, finite=True)
    second = checkpoint.get("new_second", [])
    if segment["double_new_proof"]:
        _observations(news, second, package["binding"], candidate)
        confirmed = finite_one & correct(news, second, finite=True)
    elif second:
        raise ValueError("Unexpected double proof changes frozen cost protocol")
    else:
        confirmed = set()
    retained = _observations(obligations, checkpoint["protected_observations"], package["binding"], candidate)
    forgotten = {row["id"] for row in obligations} - retained
    before = correct(news, selected_baseline, finite=True)
    elapsed, baseline_loss, replay_loss = (checkpoint[key] for key in ("elapsed_seconds", "baseline_replay_loss", "replay_loss"))
    if (not all(_positive(value) for value in (elapsed, baseline_loss))
            or type(replay_loss) not in (int, float) or not math.isfinite(replay_loss) or replay_loss < 0):
        raise ValueError("Finite positive total time/baseline loss and nonnegative candidate mean loss required")
    safety = not forgotten and replay_loss <= baseline_loss * 1.05
    in_time = elapsed < schedule["wall_seconds"]
    return {"protocol": PROTOCOL, "evidence_scope": "cpu_fixture" if fixture else "caller_evidence_waiting_root_audit",
        "actual_model_evidence_authenticated": False, "qualification_result_sha256": qualification_receipt["result_sha256"],
        "qualification_root_post_sha256": qualification_receipt["root_post_sha256"],
        "ce_events": checkpoint["ce_events"], "check_policy": schedule["check_policy"],
        "optimizer_updates": checkpoint["optimizer_updates"],
        "per_id_exposures": segment["per_id_exposures"], "exposure_totals": segment["exposure_totals"],
        "new_exposure_marker": segment["new_exposure_marker"], "candidate_tensor_sha256": candidate,
        "checkpoint_artifact": artifact, "provisional_correct_ids": sorted(finite_one),
        "provisional_growth_ids": sorted(finite_one - before), "double_proof": segment["double_new_proof"],
        "candidate_confirmed_growth_ids": sorted(confirmed - before),
        "candidate_confirmed_growth": len(confirmed - before),
        "pending_ids": sorted(set(schedule["new_ids"]) - confirmed) if segment["double_new_proof"] else None,
        "unselected_pending_ids": sorted(row["id"] for row in train
            if not baseline[row["id"]]["passed"] and row["id"] not in schedule["new_ids"]),
        "protected_ids": [r["id"] for r in obligations], "forgotten_ids": sorted(forgotten),
        "safety_passed": safety, "within_wall_cap": in_time, "elapsed_seconds": elapsed,
        "candidate_growth_per_minute": len(confirmed - before) * 60 / elapsed if segment["double_new_proof"] else None,
        "continue_allowed": safety and in_time, "production_learned_ids": [], "publication": False,
        "model_called_by_validator": False, "checkpoint_payload_deserialized": False,
        "limitation": "Root must audit actual generation, optimizer ledgers and full candidate/master/Adam checkpoint payload; no product receipt or execution authorization."}


if __name__ == "__main__":
    import json
    print(json.dumps({"protocol": PROTOCOL, "state": "waiting_actual_qualification_root_post_and_method_decision",
        "model_started": False, "torch_imported": "torch" in sys.modules, "teacher_calls": 0,
        "execution_ready": False, "publication": False}))
