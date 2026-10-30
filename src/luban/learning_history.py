"""Read saved learning history without loading model weights."""
import json
import math
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def factory_baseline(run, event):
    """An explicit installed baseline must keep its matching immutable event."""
    kind = run.get("kind")
    marked_event = event is not None and str(event["kind"]).startswith("factory:")
    if "factory_baseline" not in run and not marked_event:
        return False
    report = run.get("commit", {}).get("report")
    if (run.get("factory_baseline") is not True or run.get("state") != "committed"
            or kind not in {"router", "learning", "evolution"}
            or run.get("commit", {}).get("event_kind") != kind or not isinstance(report, dict)
            or event is None or event["kind"] != "factory:" + kind
            or (json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]) != report):
        raise ValueError("出厂学习基线与采纳事件不一致")
    return True


def _factory_runs(db, identity=None):
    if not db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'").fetchone():
        return []
    rows = db.execute("""SELECT t.id,t.kind,t.state,t.body,e.kind AS event_kind,e.body AS event_body FROM training_runs t
        LEFT JOIN events e ON e.id='training:'||t.id
        WHERE (json_type(t.body,'$.factory_baseline') IS NOT NULL OR e.kind LIKE 'factory:%')
        AND (? IS NULL OR t.identity=?)""", (identity, identity))
    result = []
    for identifier, kind, state, body, event_kind, event_body in rows:
        run = {**json.loads(body), "id": identifier, "kind": kind, "state": state}
        if factory_baseline(run, {"kind": event_kind, "body": event_body} if event_kind is not None else None):
            result.append(run)
    return result


def _saved_store(settings):
    config = settings.snapshot()
    root = Path(config["storage_root"])
    if str(root).startswith("/Volumes/") and not (Path("/Volumes") / root.parts[2]).is_mount():
        raise RuntimeError("存储盘未连接，请连接后重试")
    model_config = Path(config["model_path"]) / "config.json"
    if not model_config.is_file():
        return None
    model = json.loads(model_config.read_text())
    kind = model.get("model_type")
    if kind not in {"qwen3", "qwen3_5"}:
        return None
    layers = model.get("text_config", model).get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        return None
    return root / ("modules/luban0-v1" if kind == "qwen3_5" else "modules/app") / f"layer-{layers - 1}"


def saved_learning_versions(settings, store=None):
    store = Path(store) if store is not None else _saved_store(settings)
    if store is None:
        return []
    factory_versions = set()
    database = Path(settings.snapshot()["storage_root"]) / "app/memory.sqlite3"
    if database.exists():
        from .bun_bridge import readonly_connection
        with closing(readonly_connection(database)) as db:
            factory_versions = {run["commit"]["report"].get("active_expert")
                                for run in _factory_runs(db) if run["kind"] == "learning"}
    result = []
    for path in Path(store).glob("learned-*.json"):
        if path.stem in factory_versions:
            continue
        record = json.loads(path.read_text())
        created_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")
        result.append({"id": path.stem, "width": record["width"], "parameters": 3 * record["hidden"] * record["width"],
                       "created_at": created_at})
    return result


def latest_learning_attempt(settings, memory, identity=None):
    """Actual product attempts, including rejection before any optimizer update."""
    if not isinstance(identity, str) or not identity:
        store = _saved_store(settings)
        pointer = store.parent / "active.json" if store else None
        if pointer is None or not pointer.is_file():
            return None
        identity = json.loads(pointer.read_text()).get("identity")
    if not isinstance(identity, str) or not identity:
        return None
    with memory.lock, memory.connect() as db:
        if not db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_runs'").fetchone():
            return None
        _factory_runs(db, identity)
        row = db.execute("""SELECT id,state,json_extract(body,'$.started') AS started
            FROM training_runs WHERE identity=? AND kind IN ('learning','evolution') AND json_valid(body)
            AND COALESCE(json_extract(body,'$.scope'),'product') != 'research'
            AND COALESCE(json_extract(body,'$.binding.scope'),'product') != 'research'
            AND json_type(body,'$.factory_baseline') IS NULL
            AND json_type(body,'$.started') IN ('integer','real') AND json_extract(body,'$.started') > 0
            ORDER BY json_extract(body,'$.started') DESC,updated DESC LIMIT 1""", (identity,)).fetchone()
    if row is None or not math.isfinite(row["started"]):
        return None
    try:
        started = datetime.fromtimestamp(row["started"], timezone.utc).isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError, OSError):
        return None
    return {"id": row["id"], "started_at": started, "state": row["state"]}
