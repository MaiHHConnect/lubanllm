"""Draft: one closed goal450 -> bounded1800 episode. CPU validation, no model load.

The old run is evidence, never reopened. This helper does not publish anything.
Actual integration must reuse evolution's TRAIN selection and all eight gates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch

PROTOCOL = "evolution-goal-fixed450-rounds-v2"
PROTOCOL_V3 = "evolution-goal-fixed450-rounds-v3"
SHAPE = (2048, 6144)
BASE_ARGUMENTS = {"steps": 450, "learning_rate": 1e-5,
                  "distillation_weight": 5, "preserve_replay_subspace": True}
HISTORY = list(range(64, 449, 32))
CHECKPOINTS = HISTORY + [900, 1350, 1800]
CHECKPOINTS_V3 = CHECKPOINTS + [2250, 2700, 3150]


def v3(protocol):
    return protocol.get("protocol") == PROTOCOL_V3


def parent_step(protocol):
    return 1800 if v3(protocol) else 450


def final_step(protocol):
    return 3150 if v3(protocol) else 1800


def checkpoints(protocol):
    return (CHECKPOINTS_V3 if v3(protocol) else CHECKPOINTS).copy()


def inherited_observations(report):
    return {key: copy.deepcopy(report["goal_sequence"][key]) for key in
        ("reference_baseline_after_parent_closed", "entry450", "stage_observations")}
FORMAL = {
    "formal_protocol": "7642b03cc4d95fa187320c13d4126765c7baf8761509a2066ae98ac7eb9d50ee",
    "formal_manifest": "5abfc2f656507877a19bca45609288a4f0ab632b6e026b40a2c87dfca134e0c7",
    "materials": "56fe9d75bb4087e6268d52112eff7d260b0f7a16c3cb751728a548b38ce2d270",
    "old_retention": "b0874f236a9cb25d850775ec149ea7e444b691095973bdc6a4925c18f7aa251e",
    "old_training": "b8c61cfc2db9627d1b2f8848ff311d28f7ac3d9bf3e81be62e6aeaee87bc6924",
    "old_protection": "4c93cb78215d698d7dc8733cbfbe247cde8bd0b7b48228529197c2f19487945e",
    "acceptance": "9e75963bbae16928d5d840f1b50857a4cf5cc57776c03b012fc951639bcfb662"}
VALIDATION_FIELDS = {"baseline_new_loss", "baseline_replay_loss", "baseline_new_generation",
                     "baseline_replay_generation", "validation_new_loss", "validation_replay_loss",
                     "new_generation", "replay_generation", "core_retention_generation", "gates"}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def content_sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(path):
    path = Path(path)
    need(path.is_absolute() and ".." not in path.parts, "absolute immutable path required")
    need(not any(item.is_symlink() for item in (path, *path.parents)), "symlink evidence forbidden")
    need(path.is_file(), "evidence file missing")
    return path


def read_json(path):
    from luban.model_packages import _json, MAX_TEXT
    path = plain(path)
    need(path.stat().st_size <= MAX_TEXT, "bounded JSON evidence required")
    value = _json(path.read_bytes())  # Reuse duplicate-key and NaN/Infinity rejection.
    content_sha(value)  # Also rejects overflow float literals such as 1e999.
    return value


def validate_protocol(protocol):
    need(protocol.get("protocol") in {PROTOCOL, PROTOCOL_V3} and protocol.get("status") == "FROZEN_CLOSED_PARENT",
         "not the frozen bounded goal episode")
    need(protocol.get("parent_steps") == parent_step(protocol) and type(protocol["parent_steps"]) is int
         and protocol.get("max_steps") == final_step(protocol) and type(protocol["max_steps"]) is int
         and protocol.get("segment_steps") == 450 and type(protocol["segment_steps"]) is int
         and protocol.get("max_segments") == (7 if v3(protocol) else 4) and type(protocol["max_segments"]) is int,
         "episode budget must be exactly 4x450 including the original segment")
    need(protocol.get("checkpoints") == checkpoints(protocol)
         and all(type(value) is int for value in protocol["checkpoints"]), "episode checkpoint policy changed")
    parent_arguments = protocol.get("parent_arguments")
    need(parent_arguments == ({**BASE_ARGUMENTS, "steps": 1800, "goal_sequence": protocol.get("parent_sequence_metadata")} if v3(protocol) else BASE_ARGUMENTS) and type(parent_arguments["steps"]) is int
         and parent_arguments["preserve_replay_subspace"] is True
         and type(parent_arguments["learning_rate"]) in {int, float}
         and type(parent_arguments["distillation_weight"]) in {int, float}, "original LR/KL/B1 budget changed")
    need(type(protocol.get("seed")) is int and protocol["seed"] == 42
         and protocol.get("batch_size") == 1 and type(protocol["batch_size"]) is int
         and protocol.get("gradient_accumulation") == 1 and type(protocol["gradient_accumulation"]) is int,
         "seed or B1G1 changed")
    need(re.fullmatch(r"[a-f0-9]{12}", protocol.get("parent_run_id", "")) is not None,
         "invalid parent run id")
    need(protocol.get("policy_version") == "goal-fixed450-same-weight-v2"
         and protocol.get("required_consecutive_passes") == 0
         and protocol.get("required_same_weight_passes") == 2
         and type(protocol["required_same_weight_passes"]) is int
         and type(protocol["required_consecutive_passes"]) is int
         and protocol.get("required_new_exposures") == 2
         and type(protocol["required_new_exposures"]) is int,
         "full TRAIN double-pass policy changed")
    need(protocol.get("reference_policy") == "original-accepted-core"
         and protocol.get("basis_policy") == "exact-parent-basis"
         and protocol.get("validation_policy") == "first-complete450-same-W-double-pass-once-then-stop",
         "reference/basis/heldout policy changed")
    return checkpoints(protocol)


def load_verified(path):
    path = plain(path)
    protocol = read_json(path)
    validate_protocol(protocol)
    files = protocol.get("files", {})
    required = {"parent_plan", "parent_result", "parent_journal", "parent_checkpoint",
                "parent_binding", "parent_report", "records", "formal_protocol", "formal_manifest",
                "materials", "old_retention", "old_training", "old_protection", "acceptance",
                "native_manifest", "native_output_manifest", "native_trace", "baseline_old_training"}
    need(set(files) == required | ({"parent_sequence_protocol"} if v3(protocol) else set()), "closed parent/material/native evidence inventory changed")
    need(all(files[key]["sha256"] == value for key, value in FORMAL.items()), "not this exact user-approved goal dataset")
    for entry in files.values():
        need(set(entry) == {"path", "sha256"} and re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]),
             "invalid evidence descriptor")
        need(file_sha(plain(entry["path"])) == entry["sha256"], "evidence SHA mismatch")
    # Every frozen ABI tensor/cache is hashed; no model allocation or Torch load here.
    native_files = protocol.get("native_files", {})
    need(native_files and isinstance(native_files, dict), "native tensor/cache inventory missing")
    for path_name, digest in native_files.items():
        need(file_sha(plain(path_name)) == digest, "native tensor/cache SHA mismatch")
    plan = read_json(files["parent_plan"]["path"])
    need(plan.get("protocol") == (PROTOCOL if v3(protocol) else "evolution-goal-current-core-native-v1")
         and plan.get("arguments") == protocol["parent_arguments"] and plan.get("seed") == 42
         and plan.get("formal_protocol_sha256") == FORMAL["formal_protocol"]
         and plan.get("materials_sha256") == FORMAL["materials"]
         and plan.get("identity") == protocol["identity"] and plan.get("starting_core") == protocol["starting_core"],
         "not the exact original450 goal plan")
    source = protocol.get("parent_source_snapshot", {})
    expected_source = protocol.get("parent_source_sha256", {})
    need(expected_source and expected_source == plan.get("source_sha256") and set(source) == set(expected_source),
         "snapshot must cover every original plan source, not a selected subset")
    for name, entry in source.items():
        need(not Path(name).is_absolute() and ".." not in Path(name).parts and set(entry) == {"path", "sha256"}
             and entry["sha256"] == expected_source[name] and file_sha(plain(entry["path"])) == entry["sha256"],
             "parent source snapshot SHA mismatch")
    implementation = protocol.get("parent_implementation_sha256", {})
    implementation_snapshot = protocol.get("parent_implementation_snapshot", {})
    need(len(implementation) == (20 if v3(protocol) else 19) and set(implementation_snapshot) == set(implementation),
         "all original19 implementation snapshot members required independently of the plan29")
    for name, digest in implementation.items():
        entry = implementation_snapshot[name]
        need(Path(name).name == name and set(entry) == {"path", "sha256"} and entry["sha256"] == digest
             and file_sha(plain(entry["path"])) == digest
             and expected_source.get("src/luban/" + name, digest) == digest,
             "implementation snapshot or overlapping plan source SHA mismatch")
    new_source = protocol.get("new_source_sha256", {})
    need(set(new_source) == set(expected_source) | {"src/luban/" + name for name in implementation} | {"src/luban/goal_sequence.py"},
         "new full source closure must cover original plan29 plus all implementation19 and the helper")
    allowed_changes = {"scripts/train_evolution_goal.py", "src/luban/evolution.py", "src/luban/training_checkpoint.py", "src/luban/goal_sequence.py"}
    need(all(name in allowed_changes or new_source[name] == digest for name, digest in expected_source.items()),
         "unapproved change to original training math/model/native/backend source")
    new_impl = protocol.get("new_implementation_sha256", {})
    need(set(new_impl) == set(implementation) | {"goal_sequence.py"}
         and all(new_source.get("src/luban/" + name) == digest for name, digest in new_impl.items())
         and all(name in {"evolution.py", "training_checkpoint.py", "goal_sequence.py"} or new_impl[name] == digest
                 for name, digest in implementation.items()), "new implementation20 source mapping changed")
    parent_binding = read_json(files["parent_binding"]["path"])
    parent_report = read_json(files["parent_report"]["path"])
    records = read_json(files["records"]["path"])
    journal = read_json(files["parent_journal"]["path"])
    result = read_json(files["parent_result"]["path"])
    need(content_sha(parent_binding) == protocol["parent_binding_sha256"]
         and content_sha(parent_report) == protocol["parent_report_sha256"]
         and content_sha(records) == protocol["records_sha256"], "parent JSON content binding mismatch")
    need(parent_binding["implementation_sha256"] == implementation
         and parent_binding["records_sha256"] == content_sha(records), "parent snapshot and durable binding disagree")
    need(journal.get("id") == protocol["parent_run_id"] and journal.get("state") == "rejected"
         and journal.get("binding") == parent_binding and journal.get("records") == records
         and journal.get("checkpoint", {}).get("sha256") == files["parent_checkpoint"]["sha256"],
         "closed rejected parent journal does not reference this checkpoint")
    need(result.get("released") is True and result.get("passed") is False
         and result.get("adopted") is False and result.get("production_written") is False
         and result.get("plan_sha256") == files["parent_plan"]["sha256"]
         and result.get("training", {}).get("completed_steps") == parent_step(protocol)
         and result.get("training", {}).get("promoted") is False
         and not VALIDATION_FIELDS.intersection(result["training"]), "closed parent result incomplete or heldout visited")
    need(protocol["closure"].get("released") is True
         and protocol["closure"].get("validation_reads") == 0
         and protocol["closure"].get("active_core") == protocol["starting_core"]
         and protocol["closure"].get("error_type") == result.get("error_type")
         and protocol["closure"].get("error") == result.get("error"), "closure/result mismatch")
    parent_elapsed = protocol.get("parent_training_elapsed_seconds")
    need(type(parent_elapsed) in {int, float} and math.isfinite(parent_elapsed) and parent_elapsed >= 0
         and parent_elapsed == result["training"].get("elapsed_seconds"), "parent450 actual elapsed seconds mismatch")
    baseline = read_json(files["baseline_old_training"]["path"])
    need(len(baseline) == 83 and baseline == result.get("baseline_old_training")
         and [row["id"] for row in baseline] == [row["id"] for row in records["groups"]["replay_training"]],
         "original baseline83 observation must remain exactly intact")
    if v3(protocol):
        ancestor = load_verified(files["parent_sequence_protocol"]["path"])
        validate_inheritance(protocol, parent_report, ancestor)
        need(plan["goal_sequence_protocol"] == files["parent_sequence_protocol"]
             and protocol["parent_sequence_metadata"] == metadata(ancestor), "parent1800 plan/protocol chain changed")
    protocol["_protocol_path"] = str(path)
    protocol["_protocol_sha256"] = file_sha(path)
    return protocol


def metadata(protocol):
    validate_protocol(protocol)
    need(re.fullmatch(r"[a-f0-9]{64}", protocol.get("_protocol_sha256", "")), "verified protocol SHA required")
    return {"protocol_sha256": protocol["_protocol_sha256"], "parent_run_id": protocol["parent_run_id"],
            "parent_steps": parent_step(protocol), "total_budget": final_step(protocol), "new_steps_budget": 1350}


def arguments(protocol):
    return {**BASE_ARGUMENTS, "steps": final_step(protocol), "goal_sequence": metadata(protocol)}


def report_metadata(protocol, completed=None):
    completed = parent_step(protocol) if completed is None else completed
    need(type(completed) is int and parent_step(protocol) <= completed <= final_step(protocol), "invalid absolute completed step")
    return {**metadata(protocol), "protocol": protocol["protocol"], "new_steps_executed": completed - parent_step(protocol),
            "cumulative_steps": completed, "selected_segment": (completed + 449) // 450,
            "training_budget_scope": "bounded goal-specific episode, ordinary product512 limit unchanged"}


def observe_training_state(runtime, groups, replay_ids, parameter, step, generation, loss,
                           *, reference_confirmed=(), prior_confirmed=(), label="candidate", expected_new=16):
    """Two observations at ONE immutable W; TRAIN only, independent of selection.

    generation/loss are the existing evolution._generation/_loss functions.
    Row checks are already exact/EOS checks on the frozen materials.
    This function counts evidence; it never edits selected_step or any gate.
    """
    need(step in {0, 450, 900, 1350, 1800, 2250, 2700, 3150, 3600, 4050, 4500, 4950, 5400, 5850, 6300, 6750, 7200, 7650, 8100, 8550, 9000, 9450}, "only registered reference/segment states may be observed")
    new = groups["new_training"]
    replay = {row["id"]: row for row in groups["replay_training"]}
    need(len(new) == expected_new and len(replay_ids) == 13 and set(replay_ids) <= replay.keys(), "TRAIN16/old13 required")
    digest = tensor_sha(parameter)
    expert, started = runtime.branch.expert, time.time()
    proof = {"label": label, "step": step, "tensor_sha256": digest, "observed_at": started,
             "branch_enabled": False, "reference_used": False, "validation_rows_accessed": 0,
             "observations": [], "scope": "same-W double TRAIN proof; not independent generalization or product adoption"}
    runtime.branch.expert = None
    try:
        for _ in range(2):
            observed = {"new": generation(runtime, new),
                        "old13": generation(runtime, [replay[i] for i in replay_ids]),
                        "replay_training_loss": loss(runtime, groups["replay_training"])}
            need(tensor_sha(parameter) == digest, "training observation changed the frozen W")
            for field, ids in (("new", [r["id"] for r in new]), ("old13", replay_ids)):
                need([r["id"] for r in observed[field]] == ids, "observation row inventory changed")
                for row in observed[field]:
                    need(type(row["passed"]) is bool and row.get("branch_enabled") is False
                         and row.get("reference_used") is False
                         and (not row["passed"] or row.get("complete") is True),
                         "observation used a branch/reference or accepted non-EOS output")
            need(math.isfinite(observed["replay_training_loss"]), "nonfinite TRAIN replay loss")
            proof["observations"].append(observed)
        sets = [{r["id"] for r in observed["new"] if r["passed"]} for observed in proof["observations"]]
        confirmed = set.intersection(*sets)
        proof.update(single_observation_correct=[len(ids) for ids in sets], confirmed_ids=sorted(confirmed),
            confirmed_count=len(confirmed), pending_ids=sorted({r["id"] for r in new} - confirmed),
            pending_count=len(new) - len(confirmed), newly_confirmed_from_reference=sorted(confirmed - set(reference_confirmed)),
            newly_confirmed_since_prior=sorted(confirmed - set(prior_confirmed)),
            forgotten_from_reference=sorted(set(reference_confirmed) - confirmed),
            forgotten_since_prior=sorted(set(prior_confirmed) - confirmed),
            old13_both_passed=all(r["passed"] for observed in proof["observations"] for r in observed["old13"]),
            elapsed_seconds=time.time() - started)
        return proof
    finally:
        runtime.branch.expert = expert


def entry_observations(runtime, protocol, payload, parameter, previous, groups, generation, loss):
    """Actual later reference baseline, then actual450; always restores starting W."""
    need(tensor_sha(previous) == protocol["original_tensor_sha256"], "current immutable starting W changed")
    if v3(protocol):
        prior_report = payload["report"]
        proofs = inherited_observations(prior_report)
        prior = set(proofs["entry450"]["confirmed_ids"])
        for observation in proofs["stage_observations"]:
            prior.update(observation["confirmed_ids"])
        current = parameter.detach().clone()
        try:
            with torch.no_grad():
                parameter.copy_(payload["master"].to(parameter.device))
            proofs["entry_parent"] = observe_training_state(runtime, groups, protocol["replay_generation_ids"],
                parameter, 1800, generation, loss,
                reference_confirmed=proofs["reference_baseline_after_parent_closed"]["confirmed_ids"],
                prior_confirmed=prior, label="entry1800_after_parent_closed")
            return proofs
        finally:
            with torch.no_grad():
                parameter.copy_(current)
    need(tensor_sha(previous) == protocol["original_tensor_sha256"], "current immutable b45 differs from the original starting W")
    current, expert = parameter.detach().clone(), runtime.branch.expert
    try:
        with torch.no_grad():
            parameter.copy_(previous.to(parameter.device))
        baseline = observe_training_state(runtime, groups, protocol["replay_generation_ids"], parameter, 0,
            generation, loss, label="reference_baseline_after_parent_closed")
        with torch.no_grad():
            parameter.copy_(payload["master"].to(parameter.device))
        entry = observe_training_state(runtime, groups, protocol["replay_generation_ids"], parameter, 450,
            generation, loss, reference_confirmed=baseline["confirmed_ids"], prior_confirmed=baseline["confirmed_ids"],
            label="entry450_after_parent_closed")
        return {"reference_baseline_after_parent_closed": baseline, "entry450": entry}
    finally:
        with torch.no_grad():
            parameter.copy_(current)
        runtime.branch.expert = expert


def check_call(runtime, path, steps, learning_rate, distillation_weight, preserve,
               field_weight, partial_policy):
    if path is None:
        return None
    protocol = load_verified(path)
    session = getattr(runtime, "training_session", None)
    need(session is not None and session.id == protocol["new_run_id"]
         and session.binding["arguments"] == arguments(protocol) and session.body["arguments"] == arguments(protocol)
         and session.binding["scope"] == "product" and getattr(runtime, "research", False) is not True,
         "bounded episode requires its distinct product session and exact frozen arguments")
    need(steps == final_step(protocol) and type(steps) is int and learning_rate == 1e-5 and distillation_weight == 5
         and preserve is True and field_weight == 1 and partial_policy is None,
         "goal episode cannot change its native mathematical objective")
    need(runtime.identity == protocol["identity"] and runtime.active_core == protocol["starting_core"]
         and session.binding["implementation_sha256"] == protocol["new_implementation_sha256"],
         "episode implementation or current accepted core changed")
    return protocol


def initial_saved(runtime, protocol, session, parameter, previous, groups, generation, loss):
    """Import a verified closed parent into NEW journal; preserve CPU/MPS RNG."""
    from luban import luban_training as engine
    files = protocol["files"]
    closed = torch.load(files["parent_checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    need(closed["binding"] == read_json(files["parent_binding"]["path"]), "actual checkpoint binding differs from snapshot")
    fixture = engine.validate_fixture(Path(files["native_manifest"]["path"]).parent, files["native_manifest"]["sha256"])
    output = Path(files["native_output_manifest"]["path"]).parent
    engine.validate_output(output, fixture, protocol["binary_sha256"])
    native = engine.load_checkpoint(output, fixture)
    payload = derive_payload(protocol, closed, native, session.binding, session.body["records"], protocol["closure"])
    payload["report"]["goal_sequence"].update(entry_observations(runtime, protocol, payload, parameter, previous,
        groups, generation, loss))
    if not v3(protocol):
        payload["report"]["goal_sequence"]["stage_observations"] = []
    entry = payload["report"]["goal_sequence"]["entry_parent" if v3(protocol) else "entry450"]
    classify_round(entry, payload["report"]["baseline_training_replay_loss"] * 1.05 + .01)
    need(not entry["safety_failed"], "closed450 candidate already violated protection; continuation forbidden")
    torch.set_rng_state(closed["rng"])
    if str(runtime.device) == "mps":
        torch.mps.set_rng_state(closed["mps_rng"])
    session.save(payload, {"step": parent_step(protocol), "phase": "derived-parent", "steps": final_step(protocol)})
    return session.load()


def classify_round(proof, budget):
    """Same-W proof defines a completed candidate round, never a production upgrade."""
    proof.update(actual_updates=450, replay_loss_budget=budget,
        replay_loss_both_passed=all(o["replay_training_loss"] <= budget for o in proof["observations"]))
    failed = (not proof["old13_both_passed"] or not proof["replay_loss_both_passed"]
              or bool(proof["forgotten_from_reference"] or proof["forgotten_since_prior"]))
    proof.update(safety_failed=failed, round_result="protection_failed" if failed else
        "satisfied" if not proof["pending_count"] else "partial" if proof["confirmed_count"] else "pending")
    return not failed and not proof["pending_count"]


def observe_boundary(runtime, protocol, report, groups, parameter, step, generation, loss):
    need(step in (CHECKPOINTS_V3 if v3(protocol) else CHECKPOINTS)[len(HISTORY):], "only complete450 boundaries can select a candidate")
    meta = report["goal_sequence"]
    stages = meta["stage_observations"]
    need(step == 900 + 450 * len(stages), "fixed450 rounds must be contiguous")
    need(not any(o["safety_failed"] or o["round_result"] == "satisfied" for o in stages),
         "no optimization after protection failure or satisfaction")
    baseline = meta["reference_baseline_after_parent_closed"]["confirmed_ids"]
    prior = set(meta["entry450"]["confirmed_ids"])
    for observation in stages:
        prior.update(observation["confirmed_ids"])
    if v3(protocol):
        prior.update(meta["entry_parent"]["confirmed_ids"])
    observation = observe_training_state(runtime, groups, protocol["replay_generation_ids"], parameter,
        step, generation, loss, reference_confirmed=baseline, prior_confirmed=prior, label="segment-boundary")
    satisfied = classify_round(observation, report["baseline_training_replay_loss"] * 1.05 + .01)
    stages.append(observation)
    meta.update(report_metadata(protocol, step))
    selection = report["training_selection"]
    first = observation["observations"][0]
    selection["results"].append({"step": step, "new_generation": first["new"],
        "replay_training_generation": first["old13"], "replay_loss": first["replay_training_loss"],
        "accepted": satisfied, "consecutive_passes": 0, "same_weight_passes": 2 if satisfied else 0,
        "minimum_new_exposures": (step // 2) // 16, "tensor_sha256": observation["tensor_sha256"]})
    if satisfied:
        selection["selected_step"] = step
    runtime.job["learning_progress"] = {"last_check_step": step, "candidate_mastered": observation["confirmed_count"],
        "pending": observation["pending_count"], "forgotten_ids": observation["forgotten_since_prior"],
        "scope": "completed450 candidate round, not production adoption"}
    print(json.dumps({"kind": "goal_complete450_round", **{k: observation[k] for k in
        ("step", "confirmed_count", "pending_count", "round_result", "safety_failed", "actual_updates")}},
        ensure_ascii=False), flush=True)
    return observation


def stopped(report):
    return any(o["safety_failed"] for o in report["goal_sequence"]["stage_observations"])


def validate_resume_history(saved, declared, groups):
    """Narrow policy guard; ordinary temporal two-checkpoint selection is untouched."""
    report, step = saved["report"], saved["step"]
    selection, expected = report["training_selection"], declared["training_selection"]
    need(report["goal_sequence_sources"] == declared["goal_sequence_sources"], "closed parent source bindings changed")
    need(all(report["goal_sequence"][k] == declared["goal_sequence"][k] for k in
        ("protocol_sha256", "parent_run_id", "parent_steps", "total_budget", "new_steps_budget")), "episode protocol binding changed")
    need(report["goal_sequence"]["protocol"] in {PROTOCOL, PROTOCOL_V3}
         and report["goal_sequence"]["parent_steps"] <= step <= report["goal_sequence"]["total_budget"],
         "not the fixed450 episode")
    for key in ("checkpoints", "policy_version", "required_consecutive_passes", "required_same_weight_passes",
                "required_new_exposures", "replay_generation_ids", "minimum_replay_generation"):
        need(selection.get(key) == expected.get(key), "fixed450 selection policy changed")
    need(selection["checkpoints"] == (CHECKPOINTS_V3 if meta_v3(report) else CHECKPOINTS) and selection["policy_version"] == "goal-fixed450-same-weight-v2"
         and selection["required_consecutive_passes"] == 0 and selection["required_same_weight_passes"] == 2
         and saved["consecutive_passes"] == 0, "same-W proof cannot fake a temporal streak")
    meta = report["goal_sequence"]
    stages = meta["stage_observations"]
    boundaries = [s for s in (CHECKPOINTS_V3 if meta_v3(report) else CHECKPOINTS)[len(HISTORY):] if s <= step]
    need([o["step"] for o in stages] == boundaries
         and [r["step"] for r in selection["results"]] == HISTORY + boundaries,
         "complete fixed450 observation/history inventory required")
    need(content_sha(selection["results"][:len(HISTORY)]) == report["goal_sequence_sources"]["parent_history_sha256"],
         "immutable original64..448 history changed")
    new_ids, old_ids = [r["id"] for r in groups["new_training"]], selection["replay_generation_ids"]
    prior = set()
    selected = None
    observations = [meta["reference_baseline_after_parent_closed"], meta["entry450"], *stages]
    need([o["step"] for o in observations] == [0, 450, *boundaries], "reference/entry evidence missing")
    reference = set(observations[0]["confirmed_ids"])
    need(observations[0]["tensor_sha256"] == saved["original_sha256"] == report["goal_sequence_sources"]["original_tensor_sha256"]
         and observations[1]["tensor_sha256"] == report["goal_sequence_sources"]["parent_final_tensor_sha256"], "reference/actual450 tensor SHA changed")
    if step in (CHECKPOINTS_V3 if meta_v3(report) else CHECKPOINTS)[len(HISTORY):]:
        need(stages[-1]["tensor_sha256"] == tensor_sha(saved["master"].to(torch.bfloat16)), "boundary proof differs from resumed W")
    for index, proof in enumerate(observations):
        if meta_v3(report) and proof["step"] > 1800:
            prior.update(meta["entry_parent"]["confirmed_ids"])
        need(len(proof["observations"]) == 2 and proof["branch_enabled"] is False
             and proof["reference_used"] is False and proof["validation_rows_accessed"] == 0,
             "same-W double closedbook TRAIN proof required")
        sets = []
        for observation in proof["observations"]:
            for name, ids in (("new", new_ids), ("old13", old_ids)):
                need([r["id"] for r in observation[name]] == ids, "proof TRAIN/old13 inventory changed")
                need(all(type(r["passed"]) is bool and r["branch_enabled"] is False
                    and r["reference_used"] is False and (not r["passed"] or r["complete"] is True)
                    for r in observation[name]), "proof branch/reference/EOS invalid")
            need(math.isfinite(observation["replay_training_loss"]), "nonfinite proof loss")
            sets.append({r["id"] for r in observation["new"] if r["passed"]})
        confirmed = set.intersection(*sets)
        need(proof["confirmed_ids"] == sorted(confirmed) and proof["confirmed_count"] == len(confirmed)
             and proof["single_observation_correct"] == [len(s) for s in sets]
             and proof["pending_ids"] == sorted(set(new_ids) - confirmed) and proof["pending_count"] == 16-len(confirmed)
             and proof["old13_both_passed"] == all(r["passed"] for o in proof["observations"] for r in o["old13"]),
             "proof mastery/pending accounting changed")
        if index:
            need(proof["forgotten_from_reference"] == sorted(reference-confirmed)
                 and proof["forgotten_since_prior"] == sorted(prior-confirmed)
                 and proof["newly_confirmed_from_reference"] == sorted(confirmed-reference)
                 and proof["newly_confirmed_since_prior"] == sorted(confirmed-prior), "proof protection ledger changed")
            checked = copy.deepcopy(proof)
            satisfied = classify_round(checked, report["baseline_training_replay_loss"] * 1.05 + .01)
            need(all(proof[k] == checked[k] for k in ("actual_updates", "replay_loss_budget",
                 "replay_loss_both_passed", "safety_failed", "round_result")), "round safety/result changed")
            if index >= 2:
                row = selection["results"][len(HISTORY)+index-2]
                need(row["new_generation"] == proof["observations"][0]["new"]
                     and row["replay_training_generation"] == proof["observations"][0]["old13"]
                     and row["replay_loss"] == proof["observations"][0]["replay_training_loss"]
                     and row["tensor_sha256"] == proof["tensor_sha256"] and row["accepted"] == satisfied
                     and row["consecutive_passes"] == 0 and row["same_weight_passes"] == (2 if satisfied else 0)
                     and row["minimum_new_exposures"] == (proof["step"] // 2) // 16,
                     "round selection does not match actual double proof")
                if satisfied or proof["safety_failed"]:
                    need(proof["step"] == step and index == len(observations)-1,
                         "optimization continued after satisfaction or protection failure")
                if satisfied:
                    selected = proof["step"]
        prior.update(confirmed)
    if meta_v3(report):
        state = saved["optimizer"]["state"]
        need(len(state) == 1 and float(next(iter(state.values()))["step"]) == step,
             "v3 resume Adam counter does not match actual absolute updates")
        ids = [row["id"] for row in groups["new_training"]]
        exposures = (step + 1) // 2
        need(saved["new_exposures"] == {sample_id: exposures // 16 + int(index < exposures % 16) for index, sample_id in enumerate(ids)},
             "v3 resumed absolute exposure ledger changed")
        need(content_sha(selection["results"][:len(CHECKPOINTS)]) == report["goal_sequence_sources"]["lineage_parent"]["parent_history_sha256"], "immutable parent1800 complete selection history changed")
        validate_parent_entry(meta, report["goal_sequence_sources"], groups, old_ids, report["baseline_training_replay_loss"])
    need(selection["selected_step"] == selected, "only complete450 same-W satisfaction can select")


def finish_metadata(protocol, report):
    report["goal_sequence"].update(report_metadata(protocol, report["completed_steps"]))
    report["new_episode_until_validation_elapsed_seconds"] = time.time() - report["started"]
    report["new_episode_elapsed_scope"] = "derived-state creation through optimization completion, before heldout/final quality checks; runner records full process wall time separately"
    report["parent_training_elapsed_seconds"] = protocol["parent_training_elapsed_seconds"]
    report["cumulative_optimizer_steps"] = report["completed_steps"]


def finite_tensor(value, dtype, shape, name):
    need(isinstance(value, torch.Tensor) and value.device.type == "cpu"
         and value.dtype == dtype and tuple(value.shape) == tuple(shape)
         and torch.isfinite(value).all().item(), "invalid CPU tensor: " + name)


def validate_state(saved, native, protocol, *, expected_step=None):
    """Reject Adam resets, FP32 residual loss, wrong basis/ref, nonfinite data."""
    need(set(native) == {"candidate", "master", "exp_avg", "exp_avg_sq", "adam_step", "reference", "basis"},
         "native terminal state must contain all seven real FINAL tensors")
    for name in ("master", "exp_avg", "exp_avg_sq", "reference"):
        finite_tensor(native[name], torch.float32, SHAPE, name)
    finite_tensor(native["candidate"], torch.bfloat16, SHAPE, "candidate")
    finite_tensor(native["adam_step"], torch.float32, (), "adam_step")
    basis = native["basis"]
    need(isinstance(basis, torch.Tensor) and basis.ndim == 2 and basis.shape[0] == SHAPE[1]
         and 0 < basis.shape[1] < SHAPE[1], "invalid basis shape")
    finite_tensor(basis, torch.float32, tuple(basis.shape), "basis")
    need(float(native["adam_step"]) == (parent_step(protocol) if expected_step is None else expected_step) and torch.equal(native["master"].to(torch.bfloat16), native["candidate"]),
         "native step or BF16 candidate disagrees with the FP32 master")
    need(tensor_sha(native["candidate"]) == protocol["parent_final_tensor_sha256"],
         "actual frozen450 BF16 candidate SHA mismatch")
    finite_tensor(saved["master"], torch.float32, SHAPE, "saved master")
    need(torch.equal(saved["master"], native["master"]) and torch.equal(saved["basis"], basis),
         "durable/native master or basis mismatch")
    need(torch.equal(native["reference"], native["reference"].to(torch.bfloat16).float())
         and tensor_sha(native["reference"].to(torch.bfloat16)) == saved["original_sha256"]
         == protocol["original_tensor_sha256"], "original accepted reference changed")
    need(tensor_sha(basis) == protocol["basis_sha256"] and tensor_sha(native["reference"]) == protocol["reference_sha256"],
         "native reference or exact original basis SHA mismatch")
    optimizer = saved["optimizer"]
    need(set(optimizer) == {"state", "param_groups"} and len(optimizer["state"]) == 1
         and len(optimizer["param_groups"]) == 1, "one complete Adam parameter required")
    group = optimizer["param_groups"][0]
    identifier, state = next(iter(optimizer["state"].items()))
    need(group.get("params") == [identifier] and group.get("lr") == 1e-5
         and tuple(group.get("betas", [])) == (.9, .999) and group.get("eps") == 1e-6
         and group.get("weight_decay") == 0 and group.get("amsgrad") is False
         and group.get("maximize") is False and group.get("differentiable") is False
         and group.get("capturable") is False, "Adam configuration changed")
    need(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Adam moments/counter missing or unexpected")
    for name, native_name in (("step", "adam_step"), ("exp_avg", "exp_avg"), ("exp_avg_sq", "exp_avg_sq")):
        finite_tensor(state[name], torch.float32, () if name == "step" else SHAPE, name)
        need(torch.equal(state[name], native[native_name]), "native/durable Adam mismatch: " + name)
    need(torch.all(native["exp_avg_sq"] >= 0).item(), "invalid negative Adam second moment")


def derive_payload(protocol, closed, native, new_binding, records, closure):
    """Return a NEW payload. The caller saves it in a distinct TrainingSession.

    Before calling: load_verified, engine.load_fixture/load_checkpoint (real FINAL),
    and exact parent snapshot/binding comparison. This never edits old evidence.
    """
    validate_protocol(protocol)
    meta = metadata(protocol)
    need(set(closed) == {"binding", "payload", "rng", "mps_rng"}, "checkpoint envelope schema changed")
    old_binding, old = closed["binding"], closed["payload"]
    need(content_sha(old_binding) == protocol["parent_binding_sha256"]
         and content_sha(old["report"]) == protocol["parent_report_sha256"], "parent binding/report SHA mismatch")
    completed_rejection = (closure.get("exit_code") == 1 and closure.get("error_type") == "ValueError"
                           and closure.get("error") == "隔离候选没有通过正式核心训练门禁")
    need(closure == protocol["closure"] and type(closure.get("exit_code")) is int
         and (closure["exit_code"] == 0 or completed_rejection) and closure.get("released") is True
         and closure.get("adopted") is False and closure.get("production_written") is False
         and closure.get("validation_reads") == 0 and type(closure["validation_reads"]) is int
         and closure.get("final_step") == parent_step(protocol) and type(closure["final_step"]) is int
         and closure.get("parent_journal_state") == "rejected"
         and closure.get("active_core") == old_binding["core_version"], "parent is not closed/rejected/unadopted/heldout-unseen")
    need(set(records) == {"groups", "protection", "retention_regressions"}
         and content_sha(records) == protocol["records_sha256"] == old_binding["records_sha256"]
         == new_binding["records_sha256"], "complete TRAIN/VALID/protection material SHA mismatch")
    groups = records["groups"]
    need({key: len(rows) for key, rows in groups.items()} ==
         {"new_training": 16, "new_validation": 4, "replay_training": 83, "replay_validation": 8}
         and len(records["protection"]) == 89 and len(records["retention_regressions"]) == 8,
         "goal20 / original83 / protection89 / acceptance8 inventory changed")
    differences = {key: [old_binding.get(key), new_binding.get(key)] for key in set(old_binding) | set(new_binding)
                   if old_binding.get(key) != new_binding.get(key)}
    need(set(differences) <= {"arguments", "implementation_sha256"}
         and old_binding["arguments"] == protocol["parent_arguments"] and new_binding["arguments"] == arguments(protocol)
         and old_binding["scope"] == new_binding["scope"] == "product"
         and old_binding["core_version"] == protocol["starting_core"]
         and old_binding["identity"] == protocol["identity"], "unapproved binding/identity/core/device change")
    need(old_binding["implementation_sha256"] == protocol["parent_implementation_sha256"]
         and new_binding["implementation_sha256"] == protocol["new_implementation_sha256"],
         "old/new source revisions must be independently pre-registered")
    need(set(old) == {"format", "id", "parameter", "dataset_sha256", "original_sha256", "step", "master",
                     "optimizer", "basis", "new_exposures", "kl_total", "kl_maximum", "consecutive_passes", "report"},
         "parent payload schema must be complete and exact")
    need(type(old.get("step")) is int and old["step"] == parent_step(protocol) and type(old.get("format")) is int and old["format"] == 1
         and old.get("id") == protocol["parent_run_id"] and old["dataset_sha256"] == protocol["dataset_sha256"]
         and old["parameter"] == protocol["parameter"], "wrong parent terminal step/id/dataset/parameter")
    report, selection = old["report"], old["report"]["training_selection"]
    need(not VALIDATION_FIELDS.intersection(report) and report.get("promoted") is False
         and report.get("steps") == parent_step(protocol) and report.get("version") == "core-" + old["id"]
         and report.get("id") == old["id"] and report.get("previous_version") == protocol["starting_core"]
         and report.get("partial_policy") is None and report["dataset_sha256"] == old["dataset_sha256"],
         "heldout-read/promoted/foreign parent report forbidden")
    need(report["training_engine"]["engine"] == protocol["engine"]
         and report["training_engine"]["binary_sha256"] == protocol["binary_sha256"], "native engine/binary changed")
    new_ids = [row["id"] for row in groups["new_training"]]
    replay_ids = protocol["replay_generation_ids"]
    if v3(protocol):
        ancestor = load_verified(protocol["files"]["parent_sequence_protocol"]["path"])
        validate_inheritance(protocol, report, ancestor)
        validate_resume_history(old, report, groups)
        need(report["training_selection"]["selected_step"] is None and not stopped(report),
             "parent1800 satisfied or failed protection; cannot continue")
        expected_exposure = {sample_id: 900 // 16 + int(index < 900 % 16) for index, sample_id in enumerate(new_ids)}
        need(old["new_exposures"] == expected_exposure and old["consecutive_passes"] == 0
             and math.isfinite(old["kl_total"]) and math.isfinite(old["kl_maximum"]), "parent1800 absolute exposure/KL changed")
    else:
        need(len(replay_ids) == 13 and len(set(replay_ids)) == 13
             and selection["replay_generation_ids"] == replay_ids
             and selection["checkpoints"] == HISTORY and selection["selected_step"] is None
             and selection["required_consecutive_passes"] == selection["required_new_exposures"] == 2,
             "old TRAIN selection inventory or history changed")
        expected_exposure = {sample_id: 225 // 16 + int(index < 225 % 16) for index, sample_id in enumerate(new_ids)}
        need(old["new_exposures"] == expected_exposure, "absolute training exposure ledger changed")
        need([row["step"] for row in selection["results"]] == HISTORY, "complete old64..448 checks required")
        budget, streak = report["baseline_training_replay_loss"] * 1.05 + .01, 0
        need(math.isfinite(budget), "nonfinite replay baseline")
        for row in selection["results"]:
            need([r["id"] for r in row["new_generation"]] == new_ids
                 and [r["id"] for r in row["replay_training_generation"]] == replay_ids,
                 "old checks changed sample inventory")
            for result in row["new_generation"] + row["replay_training_generation"]:
                need(type(result["passed"]) is bool
                     and (not result["passed"] or result["complete"] is True and result["branch_enabled"] is False),
                     "passed check is incomplete or branch-enabled")
            exposures = ((row["step"] + 1) // 2) // 16
            accepted = all(r["passed"] for r in row["new_generation"] + row["replay_training_generation"]) and row["replay_loss"] <= budget and exposures >= 2
            streak = streak + 1 if accepted else 0
            need(type(row["accepted"]) is bool and row["accepted"] == accepted
                 and row["minimum_new_exposures"] == exposures and row["consecutive_passes"] == streak
                 and streak < 2 and math.isfinite(row["replay_loss"]), "old full-pass streak was altered")
        need(old["consecutive_passes"] == streak and math.isfinite(old["kl_total"])
             and math.isfinite(old["kl_maximum"]), "old KL/streak ledger changed")
    validate_state(old, native, protocol)
    for name in ("rng", "mps_rng"):
        value = closed[name]
        need(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.device.type == "cpu"
             and value.ndim == 1 and value.numel() > 0, "complete actual CPU/MPS RNG state required")
    new_id = protocol["new_run_id"]
    need(re.fullmatch(r"[a-f0-9]{12}", new_id) and new_id != old["id"], "new distinct run id required")
    derived = copy.deepcopy(old)
    derived["id"] = new_id
    derived["consecutive_passes"] = 0
    derived["report"].update(id=new_id, version="core-" + new_id, steps=final_step(protocol), goal_sequence=(
        {**report_metadata(protocol), **inherited_observations(report)} if v3(protocol) else report_metadata(protocol)),
        started=time.time(), parent_started=old["report"]["started"],
        elapsed_scope="new episode wall time; parent completed450 duration is separately bound to its immutable report")
    derived["report"]["training_selection"].update(checkpoints=checkpoints(protocol),
        policy_version=protocol["policy_version"], required_consecutive_passes=0, required_same_weight_passes=2,
        scope="original64..448 unchanged; each additional complete450 round ends in two same-W TRAIN16/old13/loss proofs; partial mastery continues, satisfaction selects heldout once")
    derived["report"]["goal_sequence_sources"] = source_metadata(protocol, selection)
    return derived


def source_metadata(protocol, selection=None):
    if selection is None:
        selection = read_json(protocol["files"]["parent_report"]["path"])["training_selection"]
    value = {
        "parent_history_sha256": content_sha(selection["results"]),
        "parent_final_tensor_sha256": protocol["parent_final_tensor_sha256"],
        "original_tensor_sha256": protocol["original_tensor_sha256"],
        "parent_checkpoint_sha256": protocol["files"]["parent_checkpoint"]["sha256"],
        "parent_binding_sha256": protocol["parent_binding_sha256"],
        "parent_report_sha256": protocol["parent_report_sha256"],
        "parent_journal_sha256": protocol["files"]["parent_journal"]["sha256"],
        "native_output_manifest_sha256": protocol["files"]["native_output_manifest"]["sha256"],
        "parent_implementation_sha256": protocol["parent_implementation_sha256"],
        "new_implementation_sha256": protocol["new_implementation_sha256"]}

    if v3(protocol):
        old_report = read_json(protocol["files"]["parent_report"]["path"])
        original = copy.deepcopy(old_report["goal_sequence_sources"])
        value["protocol_sha256"] = protocol["files"]["parent_sequence_protocol"]["sha256"]
        value["parent_goal_sequence_sources_sha256"] = content_sha(original)
        value["inherited_observations_sha256"] = content_sha(inherited_observations(old_report))
        need(value["inherited_observations_sha256"] == protocol["inherited_observations_sha256"]
             and value["parent_goal_sequence_sources_sha256"] == protocol["parent_goal_sequence_sources_sha256"], "immediate parent source ledger changed")
        original["new_implementation_sha256"] = protocol["new_implementation_sha256"]
        original["lineage_parent"] = value
        return original
    return value


def meta_v3(report):
    return report["goal_sequence"]["protocol"] == PROTOCOL_V3


def validate_inheritance(protocol, report, ancestor):
    need(ancestor["protocol"] == PROTOCOL and protocol["files"]["parent_sequence_protocol"]["sha256"] == ancestor["_protocol_sha256"]
         and report["goal_sequence_sources"] == source_metadata(ancestor)
         and report["goal_sequence"]["protocol"] == PROTOCOL
         and all(report["goal_sequence"][key] == value for key, value in metadata(ancestor).items())
         and ancestor["new_run_id"] == protocol["parent_run_id"]
         and ancestor["new_implementation_sha256"] == protocol["parent_implementation_sha256"]
         and ancestor["records_sha256"] == protocol["records_sha256"]
         and ancestor["identity"] == protocol["identity"] and ancestor["starting_core"] == protocol["starting_core"]
         and ancestor["original_tensor_sha256"] == protocol["original_tensor_sha256"]
         and ancestor["basis_sha256"] == protocol["basis_sha256"]
         and ancestor["reference_sha256"] == protocol["reference_sha256"], "immutable1800 ancestry chain mismatch")
    need(content_sha(inherited_observations(report)) == protocol["inherited_observations_sha256"]
         and content_sha(report["goal_sequence_sources"]) == protocol["parent_goal_sequence_sources_sha256"],
         "inherited observations/source ledger changed")


def validate_parent_entry(meta, sources, groups, old_ids, baseline_loss):
    lineage, proof = sources["lineage_parent"], meta["entry_parent"]
    inherited = {key: meta[key] for key in ("reference_baseline_after_parent_closed", "entry450")}
    inherited["stage_observations"] = meta["stage_observations"][:3]
    need(content_sha(inherited) == lineage["inherited_observations_sha256"]
         and proof["step"] == 1800 and proof["label"] == "entry1800_after_parent_closed"
         and proof["tensor_sha256"] == lineage["parent_final_tensor_sha256"]
         and len(proof["observations"]) == 2 and proof["branch_enabled"] is False
         and proof["reference_used"] is False and proof["validation_rows_accessed"] == 0,
         "inherited1800 observations or actual entry W changed")
    new_ids = [row["id"] for row in groups["new_training"]]
    sets = []
    for observed in proof["observations"]:
        for name, ids in (("new", new_ids), ("old13", old_ids)):
            need([row["id"] for row in observed[name]] == ids and all(type(row["passed"]) is bool
                 and row["branch_enabled"] is False and row["reference_used"] is False
                 and (not row["passed"] or row["complete"] is True) for row in observed[name]), "entry1800 inventory/EOS changed")
        need(math.isfinite(observed["replay_training_loss"]), "nonfinite entry1800 loss")
        sets.append({row["id"] for row in observed["new"] if row["passed"]})
    confirmed = set.intersection(*sets)
    reference = set(meta["reference_baseline_after_parent_closed"]["confirmed_ids"])
    prior = set(meta["entry450"]["confirmed_ids"])
    for stage in inherited["stage_observations"]:
        prior.update(stage["confirmed_ids"])
    need(proof["confirmed_ids"] == sorted(confirmed) and proof["confirmed_count"] == len(confirmed)
         and proof["single_observation_correct"] == [len(ids) for ids in sets]
         and proof["pending_ids"] == sorted(set(new_ids)-confirmed) and proof["pending_count"] == 16-len(confirmed)
         and proof["forgotten_from_reference"] == sorted(reference-confirmed)
         and proof["forgotten_since_prior"] == sorted(prior-confirmed)
         and proof["newly_confirmed_from_reference"] == sorted(confirmed-reference)
         and proof["newly_confirmed_since_prior"] == sorted(confirmed-prior)
         and proof["old13_both_passed"] == all(row["passed"] for observed in proof["observations"] for row in observed["old13"]),
         "entry1800 mastery/protection ledger changed")
    checked = copy.deepcopy(proof)
    classify_round(checked, baseline_loss*1.05+.01)
    need(all(proof[key] == checked[key] for key in ("actual_updates", "replay_loss_budget", "replay_loss_both_passed", "safety_failed", "round_result"))
         and not checked["safety_failed"], "entry1800 protection failed or fabricated")
