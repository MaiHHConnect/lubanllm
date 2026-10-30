"""Pure-stdlib fixed57 G1 same-role B1/B2/B4 research schedules.

CE events are not optimizer updates. No model, training or product settings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROTOCOL = "fixed57-g1-ce-events-same-role-batch-exact-checks-v1"


def build_schedule(new_ids, replay_ids, ce_events, batch_size):
    if type(ce_events) is not int or ce_events not in (150, 352, 704):
        raise ValueError("Fixed study requires150/352/704 CE events")
    if type(batch_size) is not int or batch_size not in (1, 2, 4):
        raise ValueError("Fixed study requires tensor B1/B2/B4")
    for ids in (new_ids, replay_ids):
        if (not isinstance(ids, (list, tuple)) or len(ids) != 57
                or any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != 57):
            raise ValueError("Each fixed role requires57 distinct nonempty ordered IDs")
    if set(new_ids) & set(replay_ids):
        raise ValueError("NEW/REPLAY ID groups cannot overlap")
    original_events = [{"ce_event": index + 1, "ce_role": "new" if index % 2 == 0 else "replay",
        "ce_id": (new_ids if index % 2 == 0 else replay_ids)[(index // 2) % 57],
        "replay_kl_id": replay_ids[(index // 2) % 57], "pair_index": index // 2,
        "sample_index": (index // 2) % 57} for index in range(ce_events)]
    checks = list(range(50, ce_events + 1, 50))
    if checks[-1] != ce_events:
        checks.append(ce_events)
    segments, updates = [], 0
    start = 0
    for end in checks:
        by_role = {role: [event for event in original_events[start:end] if event["ce_role"] == role]
                   for role in ("new", "replay")}
        batches = []
        for offset in range(0, max(map(len, by_role.values())), batch_size):
            for role in ("new", "replay"):
                selected = by_role[role][offset:offset + batch_size]
                if not selected:
                    continue
                updates += 1
                batches.append({"optimizer_update": updates, "ce_role": role,
                    "ce_event_ids": [event["ce_event"] for event in selected],
                    "ce_ids": [event["ce_id"] for event in selected],
                    "replay_kl_ids": [event["replay_kl_id"] for event in selected],
                    "pair_indices": [event["pair_index"] for event in selected],
                    "sample_indices": [event["sample_index"] for event in selected],
                    "actual_batch_size": len(selected), "short_tail": len(selected) < batch_size,
                    "optimizer_steps": 1})
        segments.append({"ce_event_start": start + 1, "check_ce_event": end,
                         "optimizer_updates_at_check": updates, "batches": batches})
        start = end
    counts = {key: {value: 0 for value in ids} for key, ids in
              (("new_ce", new_ids), ("replay_ce", replay_ids), ("replay_kl", replay_ids))}
    for event in original_events:
        counts[event["ce_role"] + "_ce"][event["ce_id"]] += 1
        counts["replay_kl"][event["replay_kl_id"]] += 1
    # Audit generated membership independently of physical batch execution order.
    actual_events = [value for segment in segments for batch in segment["batches"] for value in batch["ce_event_ids"]]
    if Counter(actual_events) != Counter(range(1, ce_events + 1)):
        raise RuntimeError("Schedule omitted or duplicated original CE events")
    payload = {"new_ids": list(new_ids), "replay_ids": list(replay_ids)}
    return {"protocol": PROTOCOL, "ce_events": ce_events, "tensor_batch_requested": batch_size,
        "gradient_accumulation": 1, "field_first_token_weight": 1,
        "ordered_ids_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "original_events": original_events, "segments": segments, "checks_ce_events": checks,
        "optimizer_updates": updates, "per_id_exposures": counts,
        "exposure_totals": {key: sum(value.values()) for key, value in counts.items()},
        "short_tail_batches": sum(batch["short_tail"] for segment in segments for batch in segment["batches"]),
        "ordering": "Original1-based CE event identities/pairindices preserved; within each role order unchanged. B>1 physically executes NEWbatch thenREPLAYbatch, not original sequentialAdam event order.",
        "checkpoint": "After ALL segment batches, exact50-event/final boundary; no later CE event is executed before that check.",
        "scope": "ID scheduling only; caller separately proves actual full57 NEW/57 qualified REPLAY TRAIN/source/gold rights and EOS/checks. No quality/time claim.",
        "model_started": False, "teacher_calls": 0, "generation_calls": 0,
        "product_integration": False, "publication": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-json", type=Path, help="Only ordered new_ids/replay_ids; no model/data mutation")
    parser.add_argument("--ce-events", type=int, choices=(150, 352, 704))
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4))
    args = parser.parse_args()
    if not any(value is not None for value in (args.ids_json, args.ce_events, args.batch_size)):
        result = {"protocol": PROTOCOL, "requires_fixed_ids": True, "model_started": False,
                  "torch_imported": "torch" in sys.modules, "teacher_calls": 0}
    elif any(value is None for value in (args.ids_json, args.ce_events, args.batch_size)):
        parser.error("ids-json, ce-events andbatch-size must be specified together")
    else:
        payload = json.loads(args.ids_json.read_text())
        if not isinstance(payload, dict) or set(payload) != {"new_ids", "replay_ids"}:
            parser.error("ids-json must contain only new_ids/replay_ids")
        result = build_schedule(payload["new_ids"], payload["replay_ids"], args.ce_events, args.batch_size)
        result["ids_file_sha256"] = hashlib.sha256(args.ids_json.read_bytes()).hexdigest()
        result["torch_imported"] = "torch" in sys.modules
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
