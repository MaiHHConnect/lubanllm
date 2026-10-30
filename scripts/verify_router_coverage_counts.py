"""Static coverage proposal and TRAIN-only selector; no model execution path.

Qualification must come from a future, separately verified accepted-core run.
This helper checks declared evidence, not a product journal or model ownership.
Fabricated CPU evidence is labelled as a fixture and cannot become real-ready.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
from itertools import product
import json
from pathlib import Path
import re

from luban.generation_checks import evaluate_generation
from luban.retention import normalized_context, provenance, verified_source

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / "research/router_coverage_counts_v3_draft.json"
DATA_SHA256 = "958ed3d2613cf59c0da6b9e79e86fdf9663ef405dfdc46ff144c92d5da8faea8"
COUNTS = (3, 6, 12)
ACTIONS = ("local_core", "teacher")
OPERATIONS = ("add", "subtract", "sum_three", "multiply")
CELLS = tuple(product(OPERATIONS, ("short", "long"), ("brief", "verbose")))


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def descriptor():
    data = DESCRIPTOR.read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("Fixed coverage data changed; review a new proposal instead of updating silently")
    return json.loads(data)


def _train(data):
    rows = [row for row in data["pool"] if provenance(row)["split"] == "train"]
    if (len(rows) != 98 or len({row["id"] for row in rows}) != 98
            or len({normalized_context(row) for row in rows}) != 98
            or any(row["purpose"] != "training" or not verified_source(row)
                   or provenance(row).get("route_action") is not None for row in rows)):
        raise ValueError("Fixed unlabelled TRAIN98 source/context coverage changed")
    return rows


def plan():
    data = descriptor()
    return {"protocol": "router-coverage-counts-v3-selector", "state": "waiting_accepted_core",
        "data_sha256": DATA_SHA256, "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": data["counts"], "train_records_sha256": digest(_train(data)),
        "train_per_actual_class": list(COUNTS), "architecture_fixed": data["architecture_fixed"],
        "selection_policy": data["selection_policy"], "future_core_binding": data["future_core_binding"],
        "heldout_rule": "DEV/CAL/TEST remain fixed; not accepted or inspected by this selector. Later actual class guards still required.",
        "sealed": False, "execution_ready": False, "model_go": False, "teacher_calls": 0,
        "model_allocations": 0, "product_adoption": False,
        "scope": "Static plan plus TRAIN selection only; no actual accepted-core binding/qualification established"}


def cell(row):
    value = provenance(row)
    return (value["operation_family"], value["numeric_digit_band"], value["input_template_length"])


def histogram(rows):
    return dict(sorted(Counter("/".join(cell(row)) for row in rows).items()))


def _waiting(reason, **evidence):
    return {"state": "waiting", "reason": reason, "nested_ids": {}, "execution_ready": False,
            "model_go": False, "product_adoption": False, **evidence}


def select_training(qualification):
    """Choose frozen nested IDs using complete TRAIN-only declared raw evidence.

    No heldout argument exists. The future runner must separately prove the
    accepted core and read this qualification from its once-bound actual raw.
    """
    data = descriptor()
    rows = _train(data)
    if qualification is None:
        return _waiting("No future accepted-core TRAIN qualification; originalNone labels are not current evidence")
    if not isinstance(qualification, dict) or set(qualification) != {"binding", "answers", "evidence_scope"}:
        raise ValueError("Selector accepts only binding/answers/evidence_scope, never heldout labels")
    scope = qualification["evidence_scope"]
    if scope not in {"actual_generation", "fabricated_cpu_fixture"}:
        raise ValueError("Qualification scope must distinguish actual generation from a CPU fixture")
    binding = qualification["binding"]
    if not isinstance(binding, dict):
        raise ValueError("Qualification binding must be an object")
    if binding.get("product_accepted") is not True or binding.get("core_version") is None:
        return _waiting("A formally accepted core and new qualification are required", evidence_scope=scope)
    if (binding.get("identity") != data["base_identity"]
            or not isinstance(binding.get("core_version"), str)
            or not re.fullmatch(r"core-[a-f0-9]{12}", binding["core_version"])
            or binding.get("train_records_sha256") != digest(rows)
            or binding.get("reference_used") is not False or binding.get("teacher_calls") != 0
            or type(binding.get("teacher_calls")) is not int
            or any(not isinstance(binding.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", binding[key])
                   for key in ("core_tensor_sha256", "core_metadata_sha256", "core_weights_sha256", "active_core_pointer_sha256"))):
        raise ValueError("Accepted-core/source SHA, reference or teacher binding changed")
    eos = binding.get("eos_token_ids")
    if not isinstance(eos, list) or not eos or any(type(token) is not int or token < 0 for token in eos):
        raise ValueError("Actual generation EOS IDs required")
    answers = qualification["answers"]
    if not isinstance(answers, list) or any(not isinstance(answer, dict) for answer in answers):
        raise ValueError("TRAIN answers must be a list of records")
    identifiers = [answer.get("id") for answer in answers]
    expected = {row["id"] for row in rows}
    if any(not isinstance(identifier, str) for identifier in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate or invalid TRAIN observation IDs")
    if set(identifiers) - expected:
        raise ValueError("Non-TRAIN/unknown observations cannot influence coverage selection")
    if set(identifiers) != expected:
        return _waiting("Complete fixed TRAIN98 qualification is missing", evidence_scope=scope)
    observed = {answer["id"]: answer for answer in answers}
    labelled = {action: [] for action in ACTIONS}
    for row in rows:
        answer = observed[row["id"]]
        tokens = answer.get("raw_completion_token_ids")
        if (type(answer.get("complete")) is not bool or type(answer.get("passed")) is not bool
                or not isinstance(answer.get("answer"), str) or answer.get("branch_enabled") is not False
                or answer.get("thinking") != "off" or answer.get("do_sample") is not False
                or type(answer.get("max_new_tokens")) is not int or answer["max_new_tokens"] != 512
                or not isinstance(tokens, list) or not 1 <= len(tokens) <= 512
                or any(type(token) is not int or token < 0 for token in tokens)
                or answer["complete"] != (tokens[-1] in eos)):
            raise ValueError("TRAIN qualification must bind full raw EOS, fixed512, greedy/OFF output")
        checked = evaluate_generation(answer["answer"], provenance(row)["generation_check"], answer["complete"])
        if answer["passed"] != checked["passed"]:
            raise ValueError("Declared TRAIN pass contradicts fixed integer gold and EOS")
        labelled["local_core" if checked["passed"] else "teacher"].append(row)
    evidence = {"evidence_scope": scope, "binding": copy.deepcopy(binding),
        "qualification_sha256": digest(qualification), "class_counts": {key: len(value) for key, value in labelled.items()},
        "class_cell_histogram": {key: histogram(value) for key, value in labelled.items()},
        "train_labels": [{"id": row["id"], "route_action": action, "cell": list(cell(row)),
                          "passed": observed[row["id"]]["passed"], "complete": observed[row["id"]]["complete"]}
                         for action in ACTIONS for row in sorted(labelled[action], key=lambda row: row["id"])],
        "teacher_quality_measured": False, "heldout_inspected": False}
    if any(len(labelled[action]) < 12 for action in ACTIONS):
        return _waiting("Each actual TRAIN class needs12 for the common nested3/6/12 comparison", **evidence)
    anchors = [row for row in rows if provenance(row).get("origin_sample_id") in data["historical_evidence"]["anchors"]]
    if len(anchors) != 2 or any(row not in labelled["local_core"] for row in anchors):
        return _waiting("Both old TRAIN anchors must requalify on this accepted core", **evidence)

    def first(action, operations, band, style_order):
        for style in style_order:
            for operation in operations:
                matches = sorted((row for row in labelled[action] if cell(row) == (operation, band, style)
                                  and row not in anchors), key=lambda row: row["id"])
                if matches:
                    return matches[0]
        return None

    chosen = {"local_core": sorted(anchors, key=lambda row: provenance(row)["origin_sample_id"])
        + [first("local_core", OPERATIONS[:3], "short", ("verbose", "brief"))],
        "teacher": [first("teacher", OPERATIONS[:3], "short", ("verbose", "brief")),
            first("teacher", OPERATIONS[:2], "long", ("brief", "verbose")),
            first("teacher", ("multiply",), "long", ("verbose", "brief"))]}
    if any(row is None for selected in chosen.values() for row in selected):
        return _waiting("Mandatory short/long/operation prefix coverage missing; do not substitute heldout", **evidence)
    for action in ACTIONS:
        remaining = {item: sorted((row for row in labelled[action] if cell(row) == item
                                   and row not in chosen[action]), key=lambda row: row["id"]) for item in CELLS}
        while len(chosen[action]) < 12:
            before = len(chosen[action])
            for item in CELLS:
                if remaining[item] and len(chosen[action]) < 12:
                    chosen[action].append(remaining[item].pop(0))
            if len(chosen[action]) == before:
                return _waiting("Coverage expansion exhausted actual TRAIN rows", **evidence)
    nested = {str(count): {action: [row["id"] for row in chosen[action][:count]] for action in ACTIONS} for count in COUNTS}
    distributions = {str(count): {action: histogram(chosen[action][:count]) for action in ACTIONS} for count in COUNTS}
    return {"state": "fixture_ready" if scope == "fabricated_cpu_fixture" else "ready_selection_only",
        "reason": "TRAIN coverage selection complete; accepted journal/heldout gates/execution remain separate",
        "nested_ids": nested, "selected_class_cell_histogram": distributions,
        "distribution_caveat": data["selection_policy"]["count_confound"],
        "execution_ready": False, "model_go": False, "product_adoption": False, **evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, help="CPU selection from separately supplied TRAIN raw; never executes a model")
    args = parser.parse_args()
    result = select_training(json.loads(args.qualification.read_text())) if args.qualification else plan()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
