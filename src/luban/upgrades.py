"""Read verified, persisted Runtime.learn branch upgrades without training content."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
import time
from pathlib import Path

from .settings import require_storage

HEX = r"[0-9a-f]{64}"
PERSONA_GATES = {"two_complete_new_checks", "old_protection", "replay_loss"}
CORE_GATES = {"heldout_new_loss_improved", "protected_replay_loss_retained", "heldout_new_generation",
              "heldout_new_generation_improved", "baseline_protected_replay_generation",
              "protected_replay_generation", "protected_core_retention_generation", "core_tensor_changed"}


def number(value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError("invalid finite number")
    return value


def positive_integer(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("invalid positive integer")
    return value


def _foundation_commit(store, report_path, report, body, event):
    """Verify persona and full ordinary-training releases without counting user growth.

    Proof SHA links are immutable source evidence, not new loss measurements.
    The package contains the four adopted artifacts, not external research files.
    """
    from .retention import read_retention_file
    version, identity = report["version"], report["identity"]
    commit = body.get("commit", {})
    installed = commit.get("report")
    if event != {"kind": "evolution", "body": installed}:
        raise ValueError("foundation durable adoption event mismatch")
    install_fields = {"foundation_install", "source_report_sha256", "training_updates_executed", "installation_scope"}
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if installed != report:
        if (not isinstance(installed, dict) or set(installed) != set(report) | install_fields
                or {key: value for key, value in installed.items() if key not in install_fields} != report
                or installed.get("foundation_install") is not True
                or installed.get("source_report_sha256") != report_sha
                or type(installed.get("training_updates_executed")) is not int
                or installed["training_updates_executed"] != 0
                or installed.get("installation_scope") !=
                "Install the validated release core; steps describe source training, not new training on this device"):
            raise ValueError("foundation installation report mismatch")
    persona = set(report.get("gates", {})) == PERSONA_GATES
    previous = report.get("previous_version", "missing")
    engine = report.get("training_engine", {})
    if (body.get("scope") != "product" or report.get("native_identity_verified") is not True
            or report.get("partial_policy") is not None or install_fields & set(report)
            or set(report.get("gates", {})) != (PERSONA_GATES if persona else CORE_GATES)
            or any(value is not True for value in report["gates"].values())
            or any(not isinstance(report.get(key), str) or not re.fullmatch(HEX, report[key])
                   for key in ("native_identity_sha256", "source_quality_sha256", "source_protocol_sha256"))
            or type(report.get("steps")) is not int or not 1 <= report["steps"] <= 10000
            or type(report.get("completed_steps")) is not int or not 1 <= report["completed_steps"] <= report["steps"]
            or report.get("dtype") != "torch.bfloat16" or report.get("optimizer_dtype") != "torch.float32"
            or type(report.get("trained_parameters")) is not int or report["trained_parameters"] != 12582912
            or type(report.get("added_parameters")) is not int or report["added_parameters"] != 0):
        raise ValueError("incomplete foundation source proof")
    if persona:
        if (previous is not None or report["completed_steps"] != report["steps"]
                or engine != {"name": "Luban 训练引擎", "engine": "luban-training-ce-release-v1"}
                or type(report.get("teacher_api_calls")) is not int or report["teacher_api_calls"] != 0
                or not isinstance(report.get("target"), str) or not report["target"].strip()):
            raise ValueError("incomplete persona foundation source proof")
    else:
        training_version = report.get("source_training_version")
        if (not isinstance(previous, str) or not re.fullmatch(r"core-[a-f0-9]{12}", previous)
                or not isinstance(training_version, str) or not re.fullmatch(r"core-[a-f0-9]{12}", training_version)
                or training_version in {version, previous}
                or report.get("source_previous_version", previous) != previous
                or not isinstance(engine, dict) or engine.get("name") != "Luban 训练引擎"
                or engine.get("engine") != "luban-training-ce-release-v1"
                or type(engine.get("teacher_api_calls")) is not int or engine["teacher_api_calls"] != 0
                or ("teacher_api_calls" in report and (type(report["teacher_api_calls"]) is not int or report["teacher_api_calls"] != 0))
                or report.get("replay_subspace", {}).get("enabled") is not True
                or type(report.get("training_selection", {}).get("selected_step")) is not int
                or report.get("training_selection", {}).get("selected_step") != report["completed_steps"]):
            raise ValueError("incomplete ordinary foundation source proof")
    from .foundation import _distillation_weight
    arguments = {key: report[key] for key in ("steps", "learning_rate")}
    arguments["distillation_weight"] = _distillation_weight(report)
    arguments["preserve_replay_subspace"] = True
    binding = body.get("binding", {})
    if (not 1e-8 <= number(arguments["learning_rate"]) <= .01
            or not 0 <= number(arguments["distillation_weight"]) <= 100
            or (binding.get("arguments") != arguments if "arguments" in binding
                else set(binding) != {"identity", "records_sha256"})):
        raise ValueError("foundation training arguments mismatch")
    if "arguments" in binding:
        bound = binding["arguments"]
        if (type(bound.get("steps")) is not int or type(bound.get("learning_rate")) not in (int, float)
                or type(bound.get("distillation_weight")) not in (int, float)
                or bound.get("preserve_replay_subspace") is not True):
            raise ValueError("foundation training argument types mismatch")
    names = {version + suffix for suffix in (".json", ".safetensors", ".retention.json")} | {report_path.name}
    if any((store / name).is_symlink() for name in names) or store.is_symlink():
        raise ValueError("symlink foundation artifact")
    hashes = {name: hashlib.sha256((store / name).read_bytes()).hexdigest() for name in names}
    if commit.get("artifacts") != hashes or commit.get("event_kind") != "evolution":
        raise ValueError("foundation committed artifacts mismatch")
    metadata = json.loads((store / (version + ".json")).read_text())
    if (metadata.get("identity") != identity or metadata.get("version") != version
            or metadata.get("previous_version", "missing") != previous
            or metadata.get("report") != report_path.name
            or metadata.get("sha256") != hashes[version + ".safetensors"]
            or metadata.get("retention_sha256") != hashes[version + ".retention.json"]
            or (persona and (not isinstance(metadata.get("candidate_tensor_sha256"), str)
                             or not re.fullmatch(HEX, metadata["candidate_tensor_sha256"])))
            or metadata.get("parameter") != report.get("parameter") or not report.get("parameter")
            or metadata.get("shape") != [2048, 6144] or metadata.get("dtype") != report["dtype"]
            or metadata.get("trained_parameters") != report["trained_parameters"]):
        raise ValueError("foundation metadata mismatch")
    snapshot = read_retention_file(store / (version + ".retention.json"), identity, version)
    if not persona:
        groups = snapshot["groups"]
        dataset_sha = hashlib.sha256(json.dumps(groups, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if (report.get("checkpoint") != metadata or report.get("dataset_sha256") != dataset_sha
                or metadata.get("dataset_sha256") != dataset_sha
                or report.get("split_ids") != {key: [row["id"] for row in rows] for key, rows in groups.items()}
                or report.get("sample_sha256") != {row["id"]: hashlib.sha256(json.dumps(row, sort_keys=True,
                    ensure_ascii=False).encode()).hexdigest() for rows in groups.values() for row in rows}
                or report.get("retention", {}).get("file_sha256") != hashes[version + ".retention.json"]
                or report.get("retention", {}).get("content_sha256") != snapshot["content_sha256"]):
            raise ValueError("ordinary foundation checkpoint or material proof mismatch")
    records = {"groups": snapshot["groups"], "protection": snapshot["regression_validation"], "retention_regressions": []}
    records_sha = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (body.get("records") != records or body.get("binding", {}).get("records_sha256") != records_sha
            or not all(snapshot["groups"][key] for key in ("new_training", "new_validation", "replay_training"))
            or not snapshot["regression_validation"]):
        raise ValueError("foundation protection records mismatch")
    pointers = commit.get("pointers", [])
    if (len(pointers) != 1 or pointers[0].get("name") != "active-core.json"
            or pointers[0].get("after") != {"version": version, "identity": identity, "sha256": metadata["sha256"]}):
        raise ValueError("foundation committed pointer mismatch")


def _core_records(root, journal, events=None):
    """Only durable product commits count; experimental promotions are excluded."""
    result = []
    for run_id, entry in journal.items():
        body = entry["body"]
        if (entry["kind"] != "evolution" or entry["state"] != "committed"
                or body.get("scope") == "research" and body.get("product_accepted") is not True):
            continue
        if not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise ValueError("invalid core commit identity")
        paths = [p for name in ("app", "luban0-v1")
                 for p in (root / "modules" / name).glob("evolution-" + run_id + ".json")]
        if len(paths) != 1 or paths[0].is_symlink():
            raise ValueError("missing or ambiguous core commit report")
        report = json.loads(paths[0].read_text())
        version = "core-" + run_id
        if (report.get("id") != run_id
                or report.get("version") != version or report.get("promoted") is not True
                or report.get("publication_state") != "published"
                or not re.fullmatch(HEX, report.get("identity", ""))
                or body.get("binding", {}).get("identity") != report["identity"]):
            raise ValueError("core publication report mismatch")
        foundation_keys = {"native_identity_verified", "native_identity_sha256", "source_quality_sha256", "source_protocol_sha256"}
        installed = body.get("commit", {}).get("report", {})
        foundation = bool(foundation_keys & set(report) or "foundation_install" in installed)
        if foundation:
            _foundation_commit(paths[0].parent, paths[0], report, body, (events or {}).get("training:" + run_id))
            if set(report["gates"]) == PERSONA_GATES:
                continue
        if not foundation and installed != report:
            raise ValueError("core publication report mismatch")
        gates = report.get("gates")
        if not isinstance(gates, dict) or len(gates) < 7 or any(value is not True for value in gates.values()):
            raise ValueError("core protection did not pass")
        metadata = report["checkpoint"]
        store = paths[0].parent
        meta_path = store / (version + ".json")
        weights = store / (version + ".safetensors")
        retention = store / (version + ".retention.json")
        if any(p.is_symlink() for p in (store, meta_path, weights, retention)):
            raise ValueError("symlink core artifact")
        if (json.loads(meta_path.read_text()) != metadata or metadata.get("identity") != report["identity"]
                or metadata.get("version") != version or metadata.get("report") != paths[0].name
                or metadata.get("sha256") != hashlib.sha256(weights.read_bytes()).hexdigest()
                or metadata.get("retention_sha256") != hashlib.sha256(retention.read_bytes()).hexdigest()):
            raise ValueError("core weights or protection checksum mismatch")
        from .evolution_progress import PARTIAL_POLICY, unique_records, validate_partial_progress
        partial = report.get("partial_policy") == PARTIAL_POLICY
        groups = body["records"]["groups"]
        if partial:
            records = unique_records(groups, body["records"]["protection"], body["records"]["retention_regressions"])
            progress = validate_partial_progress(report, body["records"], report["identity"], version, metadata["sha256"])
            snapshot = json.loads(retention.read_text())
            if (report["accepted_retention_groups"] != snapshot["groups"]
                    or report["accepted_retention_regressions"] != snapshot["regression_validation"]):
                raise ValueError("partial accepted materials mismatch")
        if body.get("arguments", {}).get("operation") == "validated-core-import":
            from .automatic import receipt_partial_policy
            if not partial or receipt_partial_policy(body) != PARTIAL_POLICY:
                raise ValueError("validated core import policy mismatch")
            # The original full quality proof is checked above. Installing a
            # validated source with zero new updates is not another evolution.
            continue
        baseline = number(report["baseline_new_loss"]) if report["baseline_new_loss"] is not None else None
        validation = number(report["validation_new_loss"]) if report["validation_new_loss"] is not None else None
        before, replay = number(report["baseline_replay_loss"]), number(report["validation_replay_loss"])
        if (not partial and (not baseline or validation >= baseline * .995)) or replay > before * 1.05 + .01:
            raise ValueError("core validation did not pass")
        groups = body["records"]["groups"]
        training, validation_ids = groups["new_training"], groups["new_validation"]
        train_keys, valid_keys = [row["id"] for row in training], [row["id"] for row in validation_ids]
        if (not 1 <= len(training) <= 10000 or not (0 if partial else 1) <= len(validation_ids) <= 10000
                or len(set(train_keys)) != len(train_keys) or len(set(valid_keys)) != len(valid_keys)
                or set(train_keys) & set(valid_keys)):
            raise ValueError("core validation is not independent")
        completed = number(report["started"]) + number(report["elapsed_seconds"])
        if not 0 < completed <= time.time() + 300:
            raise ValueError("invalid core completion time")
        if foundation or body.get("factory_baseline") is True:
            continue  # Full ordinary release passes the original gates; installation is not user growth.
        result.append({"base_sha256": report["identity"], "weights_sha256": metadata["sha256"],
            "baseline_validation_loss": baseline, "validation_loss": validation,
            "replay_before": before, "replay_loss": replay, "training_samples": len(training),
            "validation_samples": len(validation_ids), "completed_at": completed, "kind": "core",
            **({"newly_mastered": progress["newly_mastered"], "pending": progress["pending"],
                "scope": progress["scope"], "timing": report["timing"]} if partial else {})})
    return result


def read_upgrade_records(storage_root):
    """Fail closed if any production promoted report cannot be verified.

    Legacy learn reports persist accepted=True but omit the replay baseline.
    Their replay values are checked locally; both wire replay fields remain null.
    """
    records = {}
    try:
        root = require_storage(storage_root)
        journal, events = {}, {}
        database = root / "app/memory.sqlite3"
        if database.exists():
            from .bun_bridge import readonly_connection
            with closing(readonly_connection(database)) as db:
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='training_runs'").fetchone():
                    journal = {row[0]: {"kind": row[1], "state": row[2], "body": json.loads(row[3])}
                               for row in db.execute("SELECT id,kind,state,body FROM training_runs")}
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone():
                    events = {row[0]: {"kind": row[1], "body": json.loads(row[2])}
                              for row in db.execute("SELECT id,kind,body FROM events WHERE id LIKE 'training:%'")}
        for run_id, entry in journal.items():
            body = entry["body"]
            event_id = "training:" + run_id
            event = events.get(event_id)
            if "factory_baseline" in body or event and str(event["kind"]).startswith("factory:"):
                if (body.get("factory_baseline") is not True or entry["state"] != "committed"
                        or entry["kind"] not in {"router", "learning", "evolution"}
                        or body.get("commit", {}).get("event_kind") != entry["kind"]
                        or event != {"kind": "factory:" + entry["kind"], "body": body["commit"].get("report")}):
                    raise ValueError("factory baseline adoption event mismatch")
                # Only normalize the in-memory copy for the original validators.
                events[event_id] = {**event, "kind": entry["kind"]}
        paths = sorted(path for name in ("app", "luban0-v1")
                       for path in (root / "modules" / name).glob("learning-*.json"))
        for path in paths:
            store = path.parent
            match = re.fullmatch(r"learning-([0-9a-f]{12})\.json", path.name)
            if not match:
                continue
            entry = journal.get(match[1])
            if entry and (entry["kind"] != "learning" or entry["state"] != "committed" or
                    (entry["body"].get("scope") == "research" and entry["body"].get("product_accepted") is not True)):
                continue
            if path.is_symlink():
                raise ValueError("symlink learning report")
            report = json.loads(path.read_text())
            if report.get("records_sha256") and entry is None:
                continue  # Modern unjournaled artifacts are not legacy adoptions.
            if entry and entry["body"].get("commit", {}).get("report") != report:
                raise ValueError("committed learning report mismatch")
            if report.get("promoted") is False:
                continue
            if report.get("promoted") is not True:
                raise ValueError("missing promotion decision")
            if report.get("inference_engine") == "mlx-q4":
                if entry is None:
                    raise ValueError("MLX learning lacks an immutable adoption journal")
                from .mlx_model_packages import upgrade_record
                record = upgrade_record(store, entry, events.get("training:" + match[1]))
                if entry["body"].get("factory_baseline") is not True:
                    records.setdefault((record["base_sha256"], record["weights_sha256"]), record)
                continue
            run_id = match[1]
            expert_id = "learned-" + run_id
            if report.get("id") != run_id or report.get("active_expert") != expert_id:
                raise ValueError("report identity mismatch")
            width = positive_integer(report["chosen_width"])
            baseline = number(report["baseline_validation_loss"])
            if baseline == 0:
                raise ValueError("zero baseline")
            candidates = report["candidates"]
            if not isinstance(candidates, list):
                raise ValueError("invalid candidates")
            accepted = [c for c in candidates if isinstance(c, dict) and c.get("accepted") is True]
            if not accepted:
                raise ValueError("no accepted candidate")
            chosen = min(accepted, key=lambda c: number(c["objective"]))
            if positive_integer(chosen["width"]) != width:
                raise ValueError("chosen candidate mismatch")
            validation = number(chosen["validation_loss"])
            if not validation < baseline * 0.995:
                raise ValueError("validation did not improve")
            training, validation_ids = report["training_ids"], report["validation_ids"]
            if (not isinstance(training, list) or not isinstance(validation_ids, list)
                    or not 1 <= len(training) <= 16 or not 1 <= len(validation_ids) <= 16):
                raise ValueError("invalid sample counts")
            if (len(set(training)) != len(training) or len(set(validation_ids)) != len(validation_ids)
                    or set(training) & set(validation_ids)):
                raise ValueError("training and validation are not separate")
            replay_before = report.get("replay_before")
            replay_loss = chosen.get("replay_loss")
            if replay_before is None:
                # accepted=True is limited historical evidence, never invent a baseline.
                if len(training) > 1:
                    number(replay_loss)
                elif replay_loss is not None:
                    number(replay_loss)
                replay_loss = None
            else:
                number(replay_before)
                number(replay_loss)
                if replay_loss > replay_before * 1.05 + 0.01:
                    raise ValueError("replay did not retain")
            completed = number(report["started"], minimum=0) + number(report["elapsed_seconds"])
            if completed <= 0 or completed > time.time() + 300 or not math.isfinite(completed):
                raise ValueError("invalid completion time")
            metadata_paths = list(store.glob("layer-*/" + expert_id + ".json"))
            metadata_paths = [p for p in metadata_paths if re.fullmatch(r"layer-[0-9]+", p.parent.name)]
            if len(metadata_paths) != 1:
                raise ValueError("missing or ambiguous persisted expert")
            meta_path = metadata_paths[0]
            weights = meta_path.with_suffix(".safetensors")
            if meta_path.is_symlink() or meta_path.parent.is_symlink() or weights.is_symlink():
                raise ValueError("symlink expert record")
            metadata = json.loads(meta_path.read_text())
            identity = re.fullmatch(r"(" + HEX + r"):" + re.escape(meta_path.parent.name), metadata["identity"])
            if (not identity or positive_integer(metadata["width"]) != width or
                    (entry and entry["body"].get("binding", {}).get("identity") != identity[1])):
                raise ValueError("metadata identity/width mismatch")
            hidden = positive_integer(metadata["hidden"])
            if chosen.get("parameters") != 3 * hidden * width:
                raise ValueError("metadata parameter mismatch")
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if metadata.get("sha256") != digest:
                raise ValueError("persisted weights checksum mismatch")
            record = {"base_sha256": identity[1], "weights_sha256": digest,
                      "baseline_validation_loss": baseline, "validation_loss": validation,
                      "replay_before": replay_before, "replay_loss": replay_loss,
                      "training_samples": len(training), "validation_samples": len(validation_ids),
                      "completed_at": completed}
            if not entry or entry["body"].get("factory_baseline") is not True:
                records.setdefault((identity[1], digest), record)
        for record in _core_records(root, journal, events):
            records.setdefault((record["base_sha256"], record["weights_sha256"]), record)
        return sorted(records.values(), key=lambda record: (record["completed_at"], record["weights_sha256"]))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, sqlite3.Error) as error:
        raise RuntimeError("模型升级次数无法完整核验，请检查学习报告和已保存权重；本次未上报") from error


def read_upgrade_parameter_total(storage_root):
    """Return the verified parameter work represented by each unique upgrade.

    The value is intentionally derived from the same fail-closed records used by
    the ranking count. Older expert records do not carry a parameter size and
    therefore contribute zero until a newer client reports one.
    """
    records = read_upgrade_records(storage_root)
    # The public value represents core parameters changed by a completed
    # evolution. Expert adapter sizes are learning capacity, not core growth.
    keys = {(row["base_sha256"], row["weights_sha256"]) for row in records if row.get("kind") == "core"}
    if not keys:
        return 0
    root = require_storage(storage_root)
    total = 0
    counted = set()
    for folder in ("app", "luban0-v1"):
        store = root / "modules" / folder
        for path in sorted(store.glob("learning-*.json")) + sorted(store.glob("evolution-*.json")):
            try:
                report = json.loads(path.read_text())
                if path.name.startswith("learning-"):
                    expert = report.get("active_expert")
                    metadata_paths = list(store.glob("layer-*/" + str(expert) + ".json"))
                    if len(metadata_paths) != 1:
                        continue
                    metadata = json.loads(metadata_paths[0].read_text())
                    identity = str(metadata.get("identity", "")).split(":", 1)[0]
                    weight_sha = metadata.get("sha256")
                    amount = metadata.get("parameters")
                    if type(amount) is not int:
                        hidden, width = metadata.get("hidden"), metadata.get("width")
                        amount = 3 * hidden * width if type(hidden) is int and type(width) is int else 0
                else:
                    checkpoint = report.get("checkpoint") or {}
                    identity, weight_sha = report.get("identity"), checkpoint.get("sha256")
                    amount = report.get("trained_parameters", checkpoint.get("trained_parameters", 0))
                key = (identity, weight_sha)
                if key in keys and key not in counted and type(amount) is int and 0 <= amount <= 10**15:
                    total += amount
                    counted.add(key)
            except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
                raise RuntimeError("模型升级参数无法完整核验，请检查学习报告和已保存权重；本次未上报") from None
    return total
