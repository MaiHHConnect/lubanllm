"""TRAIN-only fixed fictional capacity selection; no model or tokenizer entry.

Qualification JSON is evidence asserted by its caller, not proof of execution.
An independently reviewed binding is required; CPU fixtures never become actual.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luban.generation_checks import evaluate_generation
from luban.retention import eligible_learning_sample, normalized_context, provenance

PROTOCOL = "fictional-capacity-train-only-selection-v1"
CORPUS_SHA256 = "ea7c1da4e4130181d68956adf81836c98921743ab8106a7e3b54f0c1bc4c3876"
IDENTITY = "673b16c69c39f750c6d92a22532d31eafac22bead9c284dc10d465b553b282df"
ORIGINAL = "1706e50833fffb25da45ad958af043a688e5ef495d45f33fa4b1ab623c696719"
CORPUS = ROOT / "research/evolution_capacity_corpus_source_draft.json"
COUNTS = (50, 100, 200, 300)
CELLS = tuple((length, band) for length in ("brief", "extended")
              for band in ("long_code", "short_code"))


def digest(value, *, compact=False):
    # Existing actual generation runners use default JSON separators; the
    # immutable corpus ledger separately uses compact separators.
    options = {"separators": (",", ":")} if compact else {}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, **options).encode()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _materials(data):
    rows, ledger = data.get("rows"), data.get("gold_ledger")
    if not isinstance(rows, list) or len(rows) != 432 or not isinstance(ledger, list) or len(ledger) != 432:
        raise ValueError("Frozen corpus requires400 TRAIN and32 negative-control rows")
    entries = {value["id"]: value for value in ledger}
    if len(entries) != 432:
        raise ValueError("Duplicate ledger identity")
    contexts, families, ids, train = set(), {"training": set(), "validation": set()}, set(), []
    for row in rows:
        p, context = provenance(row), normalized_context(row)
        purpose = row.get("purpose")
        if purpose not in families or row.get("id") in ids or context in contexts:
            raise ValueError("Duplicate/invalid corpus record or normalized context")
        ids.add(row["id"]); contexts.add(context)
        item = entries.get(row["id"])
        cell = (p.get("capacity_cell", {}).get("input_template_length"),
                p.get("capacity_cell", {}).get("answer_character_band"))
        if (not eligible_learning_sample(row) or row.get("source") != "synthetic"
                or row.get("training_allowed") is not True or p.get("program_verified") is not True
                or p.get("human_verified") is not False or p.get("base_identity") != IDENTITY
                or p.get("evolution_role") != "new" or cell not in CELLS
                or not isinstance(item, dict) or item.get("record_sha256") != digest(row, compact=True)
                or item.get("context_sha256") != hashlib.sha256(context.encode()).hexdigest()
                or item.get("cell") != p["capacity_cell"] or item.get("gold") != row.get("answer")
                or item.get("origin_family") != p.get("origin_family")
                or not isinstance(p.get("origin_family"), str) or not p["origin_family"]
                or p.get("split_family") != p["origin_family"]):
            raise ValueError("Corpus record/provenance/ledger drift")
        check = p.get("generation_check")
        if (check != {"type": "exact", "expected": row["answer"], "max_new_tokens": 512}
                or not evaluate_generation(row["answer"], check, True)["passed"]
                or row["answer"] in "\n".join(message["content"] for message in p["messages"])):
            raise ValueError("Gold/check mismatch or answer leaked into input")
        families[purpose].add(p["origin_family"])
        if purpose == "training":
            train.append(row)
    histogram = Counter(_cell(row) for row in train)
    if (set(entries) != ids or len(train) != 400 or set(histogram) != set(CELLS)
            or set(histogram.values()) != {100} or families["training"] & families["validation"]):
        raise ValueError("Missing cell coverage or TRAIN/negative-control family conflict")
    return train


def _cell(row):
    cell = provenance(row)["capacity_cell"]
    return cell["input_template_length"], cell["answer_character_band"]


def _binding(binding):
    required = {"corpus_sha256", "base_identity", "core_version", "actual_tensor_sha256",
                "source_sha256", "official_sha256", "max_new_tokens", "temperature",
                "branch_enabled", "reference_used", "thinking", "do_sample", "eos_token_ids", "scope"}
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("Exact independently reviewed qualification binding required")
    if (binding["corpus_sha256"] != CORPUS_SHA256 or binding["base_identity"] != IDENTITY
            or binding["scope"] not in {"actual_closed_original", "cpu_fixture"}
            or binding["core_version"] is not None and (
                not isinstance(binding["core_version"], str) or not binding["core_version"].strip())
            or type(binding["max_new_tokens"]) is not int or binding["max_new_tokens"] != 512
            or type(binding["temperature"]) not in {int, float} or binding["temperature"] != 0
            or binding["branch_enabled"] is not False or binding["reference_used"] is not False
            or binding["thinking"] != "off" or binding["do_sample"] is not False):
        raise ValueError("Qualification must bind original closed512/T0/OFF/no-reference")
    for key in ("actual_tensor_sha256", "source_sha256", "official_sha256"):
        if not isinstance(binding[key], str) or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None:
            raise ValueError("Missing actual tensor/source/official digest binding")
    if binding["scope"] == "actual_closed_original" and (
            binding["core_version"] is not None or binding["actual_tensor_sha256"] != ORIGINAL):
        raise ValueError("Actual capacity qualification must be originalNone/1706 tensor")
    eos = binding["eos_token_ids"]
    if (not isinstance(eos, list) or not eos or len(set(eos)) != len(eos)
            or any(type(value) is not int or value < 0 for value in eos)):
        raise ValueError("Actual EOS identity binding required")


def validate_observation(row, observed, expected_binding):
    """Shared actual raw/EOS/identity guard; no claim of execution authenticity."""
    if not isinstance(observed, dict) or observed.get("id") != row["id"]:
        raise ValueError("Qualification row identity changed")
    tokens = observed.get("raw_completion_token_ids")
    if (observed.get("sample_sha256") != digest(row)
            or observed.get("actual_base_identity") != expected_binding["base_identity"]
            or observed.get("actual_tensor_sha256") != expected_binding["actual_tensor_sha256"]
            or observed.get("core_version") != expected_binding["core_version"]
            or observed.get("branch_enabled") is not False or observed.get("reference_used") is not False
            or observed.get("thinking") != "off" or observed.get("do_sample") is not False
            or type(observed.get("max_new_tokens")) is not int or observed["max_new_tokens"] != 512
            or type(observed.get("passed")) is not bool or type(observed.get("complete")) is not bool
            or not isinstance(tokens, list) or not tokens or len(tokens) > 512
            or any(type(value) is not int or value < 0 for value in tokens)
            or type(observed.get("completion_tokens")) is not int or observed["completion_tokens"] != len(tokens)
            or any(value in expected_binding["eos_token_ids"] for value in tokens[:-1])
            or observed["complete"] != (tokens[-1] in expected_binding["eos_token_ids"])
            or not observed["complete"] and len(tokens) != 512
            or not isinstance(observed.get("answer"), str)
            or not isinstance(observed.get("decoded_completion_unstripped"), str)
            or observed["decoded_completion_unstripped"].strip() != observed["answer"]):
        raise ValueError("Qualification raw answer/EOS/context/source/original-state proof drift")
    correct = evaluate_generation(observed["answer"], provenance(row)["generation_check"], observed["complete"])
    if observed["passed"] != correct["passed"]:
        raise ValueError("Qualification self-claimed correctness differs from fixed exact gold/EOS")
    return observed["passed"]


def select_capacity_rows(corpus_path=CORPUS, qualification=None, expected_binding=None):
    """Return fixed nested IDs only; absent evidence waits and never fabricates labels."""
    path = Path(corpus_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != CORPUS_SHA256:
        raise ValueError("Immutable400 corpus byte SHA changed")
    train = _materials(read_json(path))
    output = {"protocol": PROTOCOL, "corpus_sha256": CORPUS_SHA256, "counts": list(COUNTS),
              "model_started": False, "tokenizer_used": False, "teacher_calls": 0,
              "execution_ready": False, "production_adoptions": 0, "negative_control_used": False,
              "scope": "Fictional exact-context capacity; no real-world knowledge/generalization proof",
              "qualification": None, "selections": {}}
    if qualification is None or expected_binding is None:
        output.update(state="waiting", reason="Actual TRAIN-only qualification and independent binding not supplied")
        return output
    _binding(expected_binding)
    if (not isinstance(qualification, dict) or set(qualification) != {"binding", "records"}
            or qualification["binding"] != expected_binding):
        raise ValueError("Qualification binding drift")
    records = qualification["records"]
    if (not isinstance(records, list) or len(records) != 400 or any(not isinstance(value, dict) for value in records)
            or [value.get("id") for value in records] != [row["id"] for row in train]):
        raise ValueError("Qualification must cover all400 ordered TRAIN once; no unknown/heldout/duplicate IDs")
    unknown, passed = [], []
    for row, observed in zip(train, records):
        validate_observation(row, observed, expected_binding)
        (passed if observed["passed"] else unknown).append(row)
    by_cell = {cell: sorted((row for row in unknown if _cell(row) == cell),
                           key=lambda row: hashlib.sha256(("42:" + row["id"]).encode()).hexdigest())
               for cell in CELLS}
    output["qualification"] = {"sha256": digest(qualification), "binding": expected_binding,
        "raw_records": 400, "original_correct": len(passed), "original_failed": len(unknown),
        "failed_per_cell": {"/".join(cell): len(rows) for cell, rows in by_cell.items()},
        "evidence_scope": "Caller-asserted actual receipt; must be independently audited before any MODEL GO"}
    for count in COUNTS:
        required = Counter(CELLS[index % 4] for index in range(count))
        missing = {"/".join(cell): n - len(by_cell[cell]) for cell, n in required.items() if len(by_cell[cell]) < n}
        if missing:
            output["selections"][str(count)] = {"state": "waiting", "missing_per_cell": missing, "ids": []}
            continue
        ids = [by_cell[CELLS[index % 4]][index // 4]["id"] for index in range(count)]
        output["selections"][str(count)] = {"state": "fixture_only" if expected_binding["scope"] == "cpu_fixture"
            else "ready_for_root_review", "ids": ids, "ordered_ids_sha256": digest(ids),
            "per_cell": {"/".join(cell): n for cell, n in required.items()}}
    output.update(state="fixture_only" if expected_binding["scope"] == "cpu_fixture" else "qualified_for_root_review",
        selection="TRAIN-only failed rows; per-cell SHA256(42:+id), lexicographic cells, fixed roundrobin",
        distribution_boundary="NestedN changes count and distribution; input-template length and answer-character band are distinct, not token lengths")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    if bool(args.qualification) != bool(args.binding):
        parser.error("Qualification and independently reviewed binding must be supplied together")
    result = select_capacity_rows(args.corpus, read_json(args.qualification) if args.qualification else None,
                                  read_json(args.binding) if args.binding else None)
    result["torch_imported"] = "torch" in sys.modules
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
