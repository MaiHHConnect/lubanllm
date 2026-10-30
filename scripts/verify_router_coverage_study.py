"""Fixed194 accepted-core router comparison. Default is static; never a model GO.

Execution requires a separately approved exact plan SHA and fresh output. Each
generation split is attempted once, all fits are research-only, and raw labels
mean local failure within512 tokens, never verified teacher superiority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import copy
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import tempfile

import verify_router_coverage_counts as selector
from luban.generation_checks import evaluate_generation
from luban.retention import provenance, read_retention_file

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "development", "calibration", "test")
SOURCE_FILES = ["src/luban/" + name for name in (
    "training_checkpoint.py", "runtime.py", "evolution.py", "calibration.py", "retention.py",
    "dynamic.py", "decision.py", "generation_checks.py", "coldpaging.py", "backbone.py",
    "automatic.py", "process_lock.py", "context.py", "core_publication.py", "external_brain.py",
    "memory.py", "settings.py", "backups.py")]
SOURCE_FILES += ["scripts/" + name for name in (
    "verify_router_coverage_study.py", "verify_router_coverage_counts.py", "verify_automatic_batches.py",
    "verify_router_v1.py", "verify_luban0_final.py", "bootstrap_luban0.py")]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def safe(path):
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Research source/output cannot use symlinks")
    return path


def database_sha(path):
    """Read actual SQLite contents, including committed WAL; no production schema writes."""
    with closing(sqlite3.connect(safe(path).as_uri() + "?mode=ro", uri=True)) as db:
        return hashlib.sha256("\n".join(db.iterdump()).encode()).hexdigest()


def artifact_path(store, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Product artifact path is invalid")
    return safe(Path(store) / name)


def module_snapshot(store):
    result = {}
    for path in safe(store).rglob("*"):
        if path.is_symlink():
            raise ValueError("Production module tree contains a symlink")
        if path.is_file():
            result[path.relative_to(store).as_posix()] = sha(path)
    return result


def model_binding(path, identity):
    """Bind the actual official local files, not just their declared manifest SHA."""
    path = safe(path)
    manifest_path = safe(path / "upstream-file-manifest.json")
    files = json.loads(manifest_path.read_text())["Data"]["Files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Official model file manifest is empty")
    hashes = {}
    for record in files:
        name, expected = record.get("Path"), record.get("Sha256")
        if (name in hashes or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)
                or sha(artifact_path(path, name)) != expected):
            raise ValueError("Official model file SHA or unique manifest membership changed")
        hashes[name] = expected
    records = sorted((name, value) for name, value in hashes.items()
                     if name.endswith(".safetensors") or name == "config.json")
    actual_identity = hashlib.sha256(json.dumps(records).encode()).hexdigest()
    if "config.json" not in hashes or not any(name.endswith(".safetensors") for name in hashes) or actual_identity != identity:
        raise ValueError("Official actual model files do not match the accepted base identity")
    return {"path": str(path), "identity": actual_identity, "manifest_sha256": sha(manifest_path), "files": hashes}


def accepted_binding(storage):
    """Require a real committed outer product publication, not a declared SHA string."""
    store = safe(Path(storage) / "modules/luban0-v1")
    pointer_path = store / "active-core.json"
    if not pointer_path.exists():
        return None
    pointer = json.loads(safe(pointer_path).read_text())
    version, identity = pointer.get("version"), pointer.get("identity")
    if (not isinstance(version, str) or not re.fullmatch(r"core-[a-f0-9]{12}", version)
            or identity != selector.descriptor()["base_identity"]):
        raise ValueError("Accepted core identity/version differs from fixed194")
    metadata_path, weights_path = store / (version + ".json"), store / (version + ".safetensors")
    metadata = json.loads(safe(metadata_path).read_text())
    retention_path = safe(store / (version + ".retention.json"))
    retention = read_retention_file(retention_path, identity, version)
    if (metadata.get("identity") != identity or metadata.get("version") != version
            or metadata.get("sha256") != sha(safe(weights_path)) or pointer.get("sha256") != metadata["sha256"]
            or metadata.get("retention_sha256") != sha(retention_path)):
        raise ValueError("Accepted core pointer/weights/metadata/retention SHA differs")
    db_path = safe(Path(storage) / "app/memory.sqlite3")
    with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='training_runs'").fetchone():
            raise ValueError("Formal product journal is required; legacy/core declarations are insufficient")
        if db.execute("SELECT 1 FROM training_runs WHERE identity=? AND state IN "
                      "('queued','running','paused','committing')", (identity,)).fetchone():
            raise ValueError("Pending training job owns this production base")
        rows = db.execute("SELECT id,state,body FROM training_runs WHERE identity=? AND kind='evolution'",
                          (identity,)).fetchall()
    matches = [json.loads(body) for identifier, state, body in rows
               if identifier == version.removeprefix("core-") and state == "committed"]
    if len(matches) != 1:
        raise ValueError("Selected core has no unique committed product journal")
    row = matches[0]
    commit, material = row.get("commit", {}), row.get("commit", {}).get("material_publication", {})
    acceptance = material.get("acceptance", {})
    gates = acceptance.get("gates", {})
    if (row.get("product_accepted") is not True or row.get("kind") != "evolution"
            or row.get("id") != version.removeprefix("core-")
            or row.get("binding", {}).get("identity") != identity
            or commit.get("report", {}).get("checkpoint") != metadata
            or material.get("identity") != identity or material.get("version") != version
            or material.get("groups") != retention["groups"]
            or material.get("retention_regressions") != retention["regression_validation"]
            or set(gates) != {"fresh_development", "known_regression", "engineering", "recovery", "protection_snapshot"}
            or any(value is not True for value in gates.values())
            or commit.get("pointers") is None or len(commit["pointers"]) != 1
            or commit["pointers"][0].get("name") != "active-core.json"
            or commit["pointers"][0].get("after") != pointer):
        raise ValueError("Core is research-only or its complete product acceptance differs")
    artifacts = commit.get("artifacts", {})
    for name, expected in artifacts.items():
        if sha(artifact_path(store, name)) != expected:
            raise ValueError("Committed product artifact SHA changed")
    required = {version + suffix for suffix in (".json", ".safetensors", ".retention.json")}
    if not required <= set(artifacts) or artifacts.get(acceptance.get("report_path")) != acceptance.get("report_sha256"):
        raise ValueError("Complete accepted core and outer report must be journal-bound artifacts")
    raw = json.loads((store / acceptance["report_path"]).read_text())
    tensor_sha = raw.get("recovery", {}).get("core_tensor_sha256_before")
    if (raw.get("outer_gates") != gates or raw.get("recovery", {}).get("passed") is not True
            or raw.get("session_id") != row["id"]
            or raw["recovery"].get("core_version_after") != version
            or raw["recovery"].get("core_tensor_sha256_after") != tensor_sha
            or not isinstance(tensor_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", tensor_sha)):
        raise ValueError("Real product recovery/outer acceptance tensor proof is absent")
    source_rows, implementation = material.get("source_rows"), row.get("binding", {}).get("implementation_sha256")
    if (not isinstance(source_rows, list) or not source_rows
            or hashlib.sha256(json.dumps(source_rows, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                != material.get("source_sha256")
            or not isinstance(implementation, dict) or not implementation
            or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in implementation.values())):
        raise ValueError("Accepted material/source implementation evidence is incomplete")
    return {"identity": identity, "core_version": version, "product_accepted": True,
        "core_tensor_sha256": tensor_sha, "core_metadata_sha256": sha(metadata_path),
        "core_weights_sha256": sha(weights_path), "active_core_pointer_sha256": sha(pointer_path),
        "core_retention_sha256": sha(retention_path), "retention_content_sha256": retention["content_sha256"],
        "graph_verification_sha256": sha(safe(store / "graph-verification.json")),
        "product_journal_sha256": selector.digest(row), "source_material_sha256": material.get("source_sha256"),
        "accepted_training_sources": row.get("binding", {}).get("implementation_sha256"),
        "artifacts": artifacts, "reference_used": False, "teacher_calls": 0}


def plan(storage=None, output=None, model=None):
    data = selector.descriptor()
    binding = accepted_binding(storage) if storage is not None else None
    return {"protocol": "router-coverage-counts-v3-accepted-core", "scope": "research",
        "state": "prepared_unsealed" if binding else "waiting_accepted_core",
        "binding": binding, "data_sha256": selector.DATA_SHA256, "source_sha256": sources(),
        "output": str(safe(output)) if output is not None else None,
        "model": model_binding(model, data["base_identity"]) if model is not None else None,
        "qualification_order": list(SPLITS), "split_counts": data["counts"],
        "train_per_actual_class": list(selector.COUNTS), "architecture": data["architecture_fixed"],
        "qualification_budget": 512, "steps": 160, "seed": 42,
        "selection_policy": data["selection_policy"], "selection_train_only": True,
        "teacher_label": "Actual local budget512 gold/EOS failure only; teacher answer/quality unmeasured",
        "distribution_caveat": data["selection_policy"]["count_confound"],
        "heldout_gate": "Each fixed DEV/CAL/TEST actual class>=2 before any fit; no heldout moves or budget adaptation",
        "qualification_attempts": 1, "fresh_router_per_trial": True,
        "production_publication": False, "teacher_calls": 0, "model_go": False, "sealed": False,
        "execution_ready": False}


def label_split(rows, answers, eos):
    """Validate actual raw+EOS and frozen integer gold before producing route labels."""
    if [row["id"] for row in rows] != [answer.get("id") for answer in answers]:
        raise ValueError("Qualification raw length/order differs from fixed split")
    output = copy.deepcopy(rows)
    for row, answer in zip(output, answers):
        tokens = answer.get("raw_completion_token_ids")
        if (not isinstance(answer.get("answer"), str) or type(answer.get("complete")) is not bool
                or type(answer.get("passed")) is not bool or answer.get("branch_enabled") is not False
                or answer.get("thinking") != "off" or answer.get("do_sample") is not False
                or type(answer.get("max_new_tokens")) is not int or answer["max_new_tokens"] != 512
                or answer.get("reference_used", False) is not False
                or not isinstance(tokens, list) or not 1 <= len(tokens) <= 512
                or any(type(token) is not int or token < 0 for token in tokens)
                or answer["complete"] != (tokens[-1] in eos)):
            raise ValueError("Qualification must be once-only greedy512/OFF raw EOS without references")
        value = provenance(row)
        checked = evaluate_generation(answer["answer"], value["generation_check"], answer["complete"])
        if checked["passed"] != answer["passed"]:
            raise ValueError("Raw pass contradicts frozen math gold/EOS")
        value.update(route_action="local_core" if answer["passed"] else "teacher",
                     teacher_superiority_verified=False)
        row["provenance"] = value
    return output


def fit_records(data, selection, heldout, eos):
    """No TEST input can affect the already-frozen TRAIN selector result."""
    labelled, counts = [], {}
    if set(heldout) != set(SPLITS[1:]):
        raise ValueError("All fixed heldout splits are required; no substitute partitions")
    for split in SPLITS[1:]:
        rows = [row for row in data["pool"] if provenance(row)["split"] == split]
        rows = label_split(rows, heldout[split], eos)
        counts[split] = dict(Counter(provenance(row)["route_action"] for row in rows))
        if any(counts[split].get(action, 0) < 2 for action in selector.ACTIONS):
            return {"state": "waiting", "reason": "Fixed heldout actual class lacks2; no fit or heldout substitution", "class_counts": counts}
        labelled.extend(rows)
    if selection["state"] != "ready_selection_only":
        return {"state": "waiting", "reason": "Actual TRAIN coverage selection is not ready", "class_counts": counts}
    rows = {row["id"]: row for row in selector._train(data)}
    result = {}
    for count in selector.COUNTS:
        train = []
        for action in selector.ACTIONS:
            for identifier in selection["nested_ids"][str(count)][action]:
                row = copy.deepcopy(rows[identifier]);value = provenance(row)
                value.update(route_action=action, teacher_superiority_verified=False);row["provenance"] = value
                train.append(row)
        result[str(count)] = {"train": train, **{split: [row for row in labelled if provenance(row)["split"] == split] for split in SPLITS[1:]}}
    return {"state": "ready_records_only", "records": result, "class_counts": counts}


def clone_accepted(storage, target, binding):
    """Copy committed journal and exact core; never delete the accepted core pointer."""
    target = safe(target)
    target.mkdir(exist_ok=False)
    source_store, store = Path(storage) / "modules/luban0-v1", target / "modules/luban0-v1"
    store.mkdir(parents=True)
    for name in {"active-core.json", "graph-verification.json", *binding["artifacts"]}:
        target_path = artifact_path(store, name);target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact_path(source_store, name), target_path)
    (target / "app").mkdir()
    source_db = safe(Path(storage) / "app/memory.sqlite3")
    with closing(sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True)) as source, closing(
            sqlite3.connect(target / "app/memory.sqlite3")) as destination:
        source.backup(destination)
    if sha(store / "active-core.json") != binding["active_core_pointer_sha256"]:
        raise ValueError("Clone did not preserve accepted core pointer")


def execute(protocol, storage, model, output):
    # Imports/allocation are confined to explicit, later approved execution.
    import torch
    from luban.runtime import Runtime
    from luban.memory import Memory
    from luban.settings import Settings, require_storage
    from luban.training_checkpoint import TrainingSession, training_runs
    from luban.evolution import _generation, _target
    from verify_automatic_batches import measured
    from verify_luban0_final import tensor_digest
    storage = require_storage(storage);output = safe(output)
    if protocol != plan(storage, output, model) or protocol["binding"] is None or protocol["model"] is None:
        raise ValueError("Current product acceptance/data/source differs from preregistered plan")
    if output.parent != storage / "experiments" or output.exists():
        raise ValueError("A fresh direct research experiments directory is required; no retries")
    store = storage / "modules/luban0-v1"
    before = module_snapshot(store)
    db_before = database_sha(storage / "app/memory.sqlite3")
    output.mkdir();baseline = output / "baseline";clone_accepted(storage, baseline, protocol["binding"])
    report = {"protocol": protocol, "state": "qualifying", "qualification": {}, "trials": [],
        "teacher_calls": 0, "product_accepted": False, "production_before": before, "database_before": db_before}
    def save():
        from luban.settings import private_json
        private_json(output / "report.json", report)
    from luban.settings import private_json
    private_json(output / "protocol.json", protocol)
    runtime = None
    def check_runtime():
        binding = protocol["binding"]
        if (runtime.identity != binding["identity"] or runtime.active_core != binding["core_version"]
                or runtime.branch.expert is not None or runtime.router_meta.get("calibrated") is not False
                or sum(p.numel() for p in runtime.router.parameters()) != 6147
                or tensor_digest(_target(runtime)[1]) != binding["core_tensor_sha256"]
                or sha(runtime.store / "active-core.json") != binding["active_core_pointer_sha256"]
                or sha(runtime.store / "graph-verification.json") != binding["graph_verification_sha256"]):
            raise ValueError("Loaded core/tensor/pointer or branch/router differs")
    try:
        with tempfile.TemporaryDirectory(prefix="luban-coverage-settings-") as home:
            settings = Settings(home)
            settings.data.update(storage_root=str(baseline), model_path=str(model), local_thinking="off", temperature=0)
            runtime = Runtime(settings, Memory(baseline), research=True);runtime.load(paging=False);check_runtime()
            report["hardware"] = {"platform": platform.platform(), "device": str(runtime.device),
                "precision": str(next(runtime.model.parameters()).dtype), "torch_version": str(torch.__version__)}
            eos = runtime.model.generation_config.eos_token_id
            eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            data = selector.descriptor();selection = None
            binding = {**protocol["binding"], "eos_token_ids": eos, "train_records_sha256": selector.digest(selector._train(data))}
            for split in SPLITS:
                rows = [row for row in data["pool"] if provenance(row)["split"] == split]
                private_json(output / (split + "-started.json"), {"rows_sha256": selector.digest(rows)})
                with measured(runtime) as measurement:
                    answers = _generation(runtime, rows)
                labelled = label_split(rows, answers, eos)
                report["qualification"][split] = {"answers": answers, "measurement": measurement,
                    "class_counts": dict(Counter(provenance(row)["route_action"] for row in labelled)),
                    "class_cell_histogram": {action: selector.histogram([row for row in labelled
                        if provenance(row)["route_action"] == action]) for action in selector.ACTIONS}};save()
                if split == "train":
                    selection = selector.select_training({"binding": binding, "answers": answers, "evidence_scope": "actual_generation"})
                    report["selection"] = selection;save()
            report["fit_material"] = fit_records(data, selection,
                {split: report["qualification"][split]["answers"] for split in SPLITS[1:]}, eos);save()
            runtime._release_model();runtime = None
            if report["fit_material"]["state"] == "waiting":
                report.update(state="waiting", reason=report["fit_material"]["reason"]);return report
            for count in selector.COUNTS:
                clone = output / ("router-" + str(count));shutil.copytree(baseline, clone)
                settings.data["storage_root"] = str(clone)
                runtime = Runtime(settings, Memory(clone), research=True);runtime.load(paging=False);check_runtime()
                records = {"splits": report["fit_material"]["records"][str(count)]}
                run_id = hashlib.sha256((selector.digest(protocol) + ":" + str(count)).encode()).hexdigest()[:12]
                session = TrainingSession(runtime, "router", {"steps": 160}, records, scope="research", run_id=run_id)
                runtime.training_session = session
                with measured(runtime) as measurement:
                    result = runtime.calibrate_router()
                receipt = next(row for row in training_runs(runtime.memory, runtime.identity) if row["id"] == session.id)
                report["trials"].append({"count_per_class": count, "run_id": session.id, "result": result,
                    "measurement": measurement, "candidate_gate_passed": result["calibrated"], "product_accepted": False,
                    "receipt_state": receipt["state"], "receipt_sha256": selector.digest(receipt),
                    "candidate_checkpoint": receipt.get("checkpoint"),
                    "candidate_checkpoint_directory": str(session.directory),
                    "trainable_parameters": sum(p.numel() for p in runtime.router.parameters()),
                    "actual_answers_source_sha256": selector.digest(report["qualification"]),
                    "exposures_per_selected_row": 160, "split_ids": {k: [r["id"] for r in v] for k,v in records["splits"].items()}})
                runtime.training_session = None;runtime._release_model();runtime = None;save()
            report["state"] = "complete"
    except BaseException as error:
        report.update(state="aborted", error=str(error) or type(error).__name__);raise
    finally:
        if runtime is not None:
            runtime.training_session = None;runtime._release_model()
        report["production_after"] = module_snapshot(store)
        report["database_after"] = database_sha(storage / "app/memory.sqlite3")
        report["production_unchanged"] = before == report["production_after"] and db_before == report["database_after"]
        report["sources_unchanged"] = protocol["source_sha256"] == sources();save()
        report["model_files_unchanged"] = protocol["model"] == model_binding(model, protocol["binding"]["identity"]);save()
        if not report["production_unchanged"] or not report["sources_unchanged"] or not report["model_files_unchanged"]:
            raise RuntimeError("Production, actual model files or preregistered implementation changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args();protocol = plan(args.storage_root, args.output, args.model_path)
    fingerprint = selector.digest(protocol)
    if not args.execute:
        print(selector.encoded({"plan_sha256": fingerprint, "plan": protocol}));return
    if (args.plan_sha256 != fingerprint or protocol["binding"] is None
            or not args.storage_root or not args.model_path or not args.output):
        parser.error("Explicit execution requires accepted product core, approved exact SHA/model/fresh output")
    execute(protocol, args.storage_root, args.model_path, args.output)


if __name__ == "__main__":
    main()
