"""Private durable training checkpoints. Candidates never become active here."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

import psutil
import torch


class TrainingPaused(Exception):
    """A safe-step pause, rather than a rejected learning attempt."""


def is_product_run(row):
    return row.get("scope") != "research" or row.get("product_accepted") is True


def is_native_mlx_run(row):
    """These journals belong to the MLX owner, including its interrupted commits."""
    protocol = "native-mlx-expert-preservation-v1"
    return (row.get("arguments", {}).get("protocol") == protocol
            or row.get("binding", {}).get("arguments", {}).get("protocol") == protocol)


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    if value is None:
        return None
    for primitive in (bool, str, int, float):
        if isinstance(value, primitive):
            return primitive(value)
    raise ValueError("训练检查点含不支持的数据类型")


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _artifact(store, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("采纳文件路径无效")
    target = Path(store) / path
    if any((Path(store) / Path(*path.parts[:index])).is_symlink() for index in range(1, len(path.parts) + 1)):
        raise ValueError("采纳文件不能是符号链接")
    return target


def _schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS training_runs (
        id TEXT PRIMARY KEY, identity TEXT NOT NULL, kind TEXT NOT NULL,
        state TEXT NOT NULL, body TEXT NOT NULL, updated REAL NOT NULL)""")
    db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_pending_training_run
        ON training_runs(identity) WHERE state IN ('queued','running','paused','committing')""")


def _owner_alive(row):
    owner = row.get("owner") or {}
    try:
        process = psutil.Process(owner["pid"])
        return abs(process.create_time() - owner["started"]) < .01 and process.is_running()
    except psutil.AccessDenied:
        return True  # Unknown process ownership must not permit a second worker.
    except (psutil.NoSuchProcess, KeyError, TypeError):
        return False


def implementation_hashes(arguments=None):
    directory = Path(__file__).parent
    names = (
        "training_checkpoint.py", "runtime.py", "evolution.py", "evolution_progress.py", "calibration.py", "retention.py",
        "dynamic.py", "decision.py", "generation_checks.py", "coldpaging.py", "backbone.py", "automatic.py", "process_lock.py", "context.py", "core_publication.py", "external_brain.py",
        "luban_training.py", "native_core_training.py", "foundation.py", "goal_sequence.py", "goal_topics.py")
    if arguments and (arguments.get("goal_sequence") or {}).get("parent_steps") in {0, 2700, 4050, 5400, 6750}:
        names += ("goal_fresh.py",)
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def active_hashes(runtime):
    """Bind accepted pointers and their immutable files, including legacy router."""
    store, result = Path(runtime.store), {}
    names = {"active.json", "active-core.json", "active-router.json"}
    for name in tuple(names):
        path = _artifact(store, name)
        if path.exists():
            pointer = json.loads(path.read_text())
            version = pointer.get("version") or pointer.get("id")
            if not isinstance(version, str) or not re.fullmatch(r"[a-z0-9-]+", version):
                raise ValueError("活动训练版本名称无效")
            if name == "active.json" and hasattr(runtime.branch, "_paths"):
                required = {str(path.relative_to(store)) for path in runtime.branch._paths(version)}
            else:
                required = {version + ".json", version + ".safetensors"}
            for filename in required:
                if not _artifact(store, filename).is_file():
                    raise ValueError("活动版本文件缺失，不能创建或恢复训练任务")
            names.update(required)
            if name == "active-core.json":
                names.add(version + ".retention.json")
    if not (store / "active-router.json").exists():
        names.update({"router.json", "router.safetensors"})
    for name in sorted(names):
        path = _artifact(store, name)
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return result


def original_core_protection_hash(runtime):
    """Bind a training run to the immutable ordinary-core baseline file."""
    path = Path(runtime.store) / "original-core-protection.json"
    if path.is_symlink():
        raise ValueError("原始核心保护基线不能是符号链接")
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def training_runs(memory, identity):
    with memory.lock, memory.connect() as db:
        _schema(db)
        rows = db.execute("SELECT * FROM training_runs WHERE identity=? ORDER BY updated DESC", (identity,)).fetchall()
    return [{**json.loads(row["body"]), "state": row["state"]} for row in rows]


class TrainingSession:
    def __init__(self, runtime, kind, arguments, records, *, run_id=None, scope="product"):
        if kind not in {"router", "learning", "evolution"} or scope not in {"product", "research"}:
            raise ValueError("训练任务类型无效")
        if (is_native_mlx_run({"arguments": arguments}) or kind in {"learning", "evolution"}
                and getattr(runtime, "native_inference_precision", None) == "mlx-q4"):
            raise ValueError("MLX专家训练与恢复必须由原生MLX流程处理")
        self.runtime, self.memory = runtime, runtime.memory
        self.id = run_id or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[a-f0-9]{12}", self.id):
            raise ValueError("训练任务ID无效")
        self.directory = Path(runtime.store) / "training-runs" / self.id
        if any(path.is_symlink() for path in (self.directory, self.directory.parent)):
            raise ValueError("训练检查点目录不能是符号链接")
        self.binding = {"protocol": "durable-training-v1", "identity": runtime.identity, "scope": scope,
                        "core_version": runtime.active_core,
                        "expert_id": runtime.branch.expert_id,
                        "device": str(runtime.device), "dtype": str(runtime.dtype),
                        "torch_version": str(torch.__version__), "arguments": arguments,
                        "packages": {name: version(name) for name in ("transformers", "safetensors")},
                        "active_files_sha256": active_hashes(runtime),
                        "original_core_protection_sha256": original_core_protection_hash(runtime),
                        "implementation_sha256": implementation_hashes(arguments),
                        "records_sha256": hashlib.sha256(_encoded(records).encode()).hexdigest()}
        with self.memory.lock, self.memory.connect() as db:
            _schema(db)
            row = db.execute("SELECT * FROM training_runs WHERE id=?", (self.id,)).fetchone()
            if row:
                existing = {**json.loads(row["body"]), "state": row["state"]}
                if _owner_alive(existing) and existing["owner"]["pid"] != os.getpid():
                    raise ValueError("另一个进程仍在执行这个训练任务")
                if existing["binding"] != self.binding or existing["state"] not in {"queued", "running", "paused"}:
                    raise ValueError("检查点与当前模型、材料、训练配置或任务状态不兼容")
                self.body = existing
                if existing["state"] == "paused" and existing.get("paused_at") is not None:
                    paused_at, elapsed = existing["paused_at"], existing.get("pause_elapsed_seconds", 0.0)
                    now = time.time()
                    if (type(paused_at) not in {int, float} or type(elapsed) not in {int, float}
                            or not math.isfinite(paused_at) or not math.isfinite(elapsed)
                            or elapsed < 0 or now < paused_at):
                        raise ValueError("训练暂停计时无效，不能推算暂停耗时")
                    self.body.update(pause_elapsed_seconds=elapsed + now - paused_at, paused_at=None)
            else:
                self.body = {"id": self.id, "kind": kind, "binding": self.binding, "scope": scope,
                             "records": records, "arguments": arguments, "checkpoint": None,
                             "started": time.time(), "progress": {"step": 0, "phase": "preparing"}}
            self.body["owner"] = {"pid": os.getpid(), "started": psutil.Process().create_time()}
            self._write("queued", insert=row is None, db=db)
        self.pause_requested = False

    def _write(self, state, *, insert=False, db=None):
        body = {key: value for key, value in self.body.items() if key != "state"}
        params = (self.id, self.binding["identity"], self.body["kind"], state, _encoded(body), time.time())
        def write(connection):
            _schema(connection)
            if insert:
                connection.execute("INSERT INTO training_runs VALUES (?,?,?,?,?,?)", params)
            else:
                changed = connection.execute("UPDATE training_runs SET state=?,body=?,updated=? WHERE id=? AND identity=?",
                                             (state, params[4], params[5], self.id, self.binding["identity"]))
                if changed.rowcount != 1:
                    raise ValueError("训练任务记录已删除或身份不符")
        if db is not None:
            write(db)
        else:
            with self.memory.lock, self.memory.connect() as connection:
                write(connection)
        self.body["state"] = state

    def running(self):
        self._write("running")

    def save(self, payload, progress):
        """Write a new immutable generation before atomically referencing it in SQLite."""
        self.directory.mkdir(parents=True, exist_ok=True)
        name = uuid.uuid4().hex + ".pt"
        path = self.directory / name
        data = _cpu({"binding": self.binding, "payload": payload,
                     "rng": torch.get_rng_state(),
                     "mps_rng": torch.mps.get_rng_state() if str(self.runtime.device) == "mps" else None})
        if not _finite(data):
            raise ValueError("训练检查点含非有限张量")
        with path.open("xb") as stream:
            os.chmod(path, 0o600)
            torch.save(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checked = torch.load(path, map_location="cpu", weights_only=True)
        if checked["binding"] != self.binding or not _finite(checked):
            raise ValueError("训练检查点回读失败")
        prior = self.body.get("checkpoint")
        self.body.update(checkpoint={"name": name, "sha256": digest, "progress": progress}, progress=progress,
                         previous_checkpoint=prior)
        self._write("running")
        # Only this run's own obsolete generations are removed. Retain current + previous.
        keep = {name, (prior or {}).get("name")}
        for obsolete in self.directory.glob("*.pt"):
            if obsolete.name not in keep and re.fullmatch(r"[a-f0-9]{32}\.pt", obsolete.name) and not obsolete.is_symlink():
                obsolete.unlink()

    def load(self):
        descriptor = self.body.get("checkpoint")
        if descriptor is None:
            return None
        try:
            return self._load_generation(descriptor)
        except (ValueError, OSError):
            previous = self.body.get("previous_checkpoint")
            if previous is None:
                raise
            payload = self._load_generation(previous)
            self.body.update(checkpoint=previous, previous_checkpoint=None,
                             progress=previous.get("progress", {"phase": "resuming"}),
                             recovery_note="最新一代检查点无效，恢复前一代有效断点")
            self._write("running")
            return payload

    def _load_generation(self, descriptor):
        name = descriptor.get("name", "")
        if not re.fullmatch(r"[a-f0-9]{32}\.pt", name):
            raise ValueError("检查点路径无效")
        path = self.directory / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor.get("sha256"):
            raise ValueError("训练检查点SHA校验失败")
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("binding") != self.binding or not _finite(data):
            raise ValueError("训练检查点内容不兼容")
        torch.set_rng_state(data["rng"])
        if data.get("mps_rng") is not None:
            torch.mps.set_rng_state(data["mps_rng"])
        return data["payload"]

    def pause(self):
        self.body["owner"] = None
        if self.body.get("paused_at") is None:
            self.body["paused_at"] = time.time()
        self._write("paused")

    def rejected(self, reason):
        self.body["reason"] = str(reason)
        self._write("rejected")

    def prepare_commit(self, report, pointers, artifacts, *, event_kind, material_publication=None):
        """Durable intent precedes any active-pointer change. Only verified artifacts may commit."""
        if not report.get("promoted", report.get("calibrated", False)):
            raise ValueError("未通过验证的候选不能准备采纳")
        if active_hashes(self.runtime) != self.binding["active_files_sha256"]:
            raise ValueError("活动版本文件已变化，不能采纳基于旧版本训练的候选")
        plans = []
        for name, target in pointers.items():
            if name not in {"active.json", "active-core.json", "active-router.json"}:
                raise ValueError("活动指针路径无效")
            path = Path(self.runtime.store) / name
            if path.is_symlink():
                raise ValueError("活动指针不能是符号链接")
            plans.append({"name": name, "before": json.loads(path.read_text()) if path.exists() else None,
                          "after": target})
        hashes = {}
        for name in artifacts:
            path = _artifact(self.runtime.store, name)
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.body["commit"] = {"report": report, "pointers": plans, "artifacts": hashes,
                               "event_kind": event_kind}
        if material_publication is not None:
            self.body["commit"]["material_publication"] = material_publication
            _validate_publication(self.body)
        self._write("committing")

    def finish_commit(self):
        self.body["state"] = reconcile_commit(self.memory, Path(self.runtime.store), self.body)
        if self.body["state"] != "committed":
            raise ValueError("采纳已按持久回滚意图拒绝")


def ensure_product_state(memory, store, identity):
    """Research's outer protocol alone may finish or retire its unaccepted candidate."""
    for row in training_runs(memory, identity):
        if is_product_run(row):
            continue
        if row["state"] == "committing":
            raise ValueError("研究候选尚未完成外层验收，不能由产品恢复采纳")
        for plan in row.get("commit", {}).get("pointers", []):
            if plan["name"] not in {"active.json", "active-core.json", "active-router.json"}:
                raise ValueError("研究指针名称无效")
            path = Path(store) / plan["name"]
            if path.is_symlink():
                raise ValueError("活动指针不能是符号链接")
            if path.exists() and json.loads(path.read_text()) == plan["after"]:
                raise ValueError("活动研究候选尚未通过产品验收，禁止加载")


def recover_commits(memory, store, identity, *, research=False):
    """Run before Runtime loads pointers; no neural candidate is used for inference."""
    store = Path(store)
    if not research:
        ensure_product_state(memory, store, identity)
    for row in training_runs(memory, identity):
        if is_native_mlx_run(row):
            continue
        if not research and not is_product_run(row):
            continue
        if row["state"] == "committing":
            if _owner_alive(row) and row["owner"]["pid"] != os.getpid():
                raise ValueError("另一个进程仍在提交模型版本")
            try:
                reconcile_commit(memory, store, row)
            except (ValueError, OSError) as error:
                if "rollback_reason" in row:
                    raise  # A conflicting explicit rollback must remain pending, never roll forward.
                # Do not leave corrupt prepared weights selected. Restore only
                # pointers still at this run's own old/new target.
                from .settings import private_json
                for plan in row["commit"]["pointers"]:
                    if plan["name"] not in {"active.json", "active-core.json", "active-router.json"}:
                        raise ValueError("恢复指针名称无效") from error
                    path = store / plan["name"]
                    if path.is_symlink():
                        raise ValueError("恢复指针不能是符号链接") from error
                    current = json.loads(path.read_text()) if path.exists() else None
                    if current not in (plan["before"], plan["after"]):
                        raise ValueError("活动版本已变化，不能回滚损坏提交") from error
                for plan in row["commit"]["pointers"]:
                    path = store / plan["name"]
                    if plan["before"] is None:
                        path.unlink(missing_ok=True)
                    else:
                        private_json(path, plan["before"])
                row["reason"] = "候选采纳恢复失败，已恢复原指针：" + str(error)
                with memory.lock, memory.connect() as db:
                    db.execute("UPDATE training_runs SET state='rejected',body=?,updated=? WHERE id=?",
                               (_encoded(row), time.time(), row["id"]))
        elif row["state"] == "running" and not _owner_alive(row):
            row["owner"] = None
            with memory.lock, memory.connect() as db:
                db.execute("UPDATE training_runs SET state='paused',body=?,updated=? WHERE id=?",
                           (_encoded(row), time.time(), row["id"]))


def _validate_publication(row):
    """Bind the finite material revision to its complete journal and hashed report."""
    from .core_publication import validate_material_publication
    commit = row["commit"]
    plan = commit["material_publication"]
    validate_material_publication(plan, row["binding"]["identity"])
    records = row["records"]
    if (row["kind"] != "evolution" or commit["event_kind"] != "evolution"
            or row.get("product_accepted") is not True or plan["run_id"] != row["id"]
            or plan["groups"] != records.get("groups")
            or plan["retention_regressions"] != records.get("retention_regressions")
            or commit["report"].get("version") != plan["version"]
            or commit["artifacts"].get(plan["acceptance"]["report_path"]) != plan["acceptance"]["report_sha256"]
            or len(commit["pointers"]) != 1 or commit["pointers"][0]["name"] != "active-core.json"
            or commit["pointers"][0]["after"].get("version") != plan["version"]
            or commit["pointers"][0]["after"].get("identity") != plan["identity"]):
        raise ValueError("核心材料发布与完整 journal、活动指针或验收报告不一致")


def reconcile_commit(memory, store, row):
    """Recover the persisted commit or rollback direction, including pointer-write crash windows."""
    if is_native_mlx_run(row):
        raise ValueError("MLX专家训练与恢复必须由原生MLX流程处理")
    from .settings import private_json
    commit = row["commit"]
    with memory.lock, memory.connect() as db:
        _schema(db)
        if not db.in_transaction:
            db.execute("BEGIN IMMEDIATE")
        persisted = db.execute("SELECT state,body FROM training_runs WHERE id=? AND identity=?",
                               (row["id"], row["binding"]["identity"])).fetchone()
        persisted_body = json.loads(persisted["body"]) if persisted else None
        if persisted_body is None or persisted_body.get("commit") != commit:
            raise ValueError("采纳任务缺失或持久提交内容变化")
        event_id = "training:" + row["id"]
        encoded = _encoded(commit["report"])
        previous = db.execute("SELECT kind,body FROM events WHERE id=?", (event_id,)).fetchone()
        if previous and (previous["kind"] != commit["event_kind"] or previous["body"] != encoded):
            raise ValueError("采纳事件ID冲突")
        if persisted["state"] == "committed":
            if not previous:
                raise ValueError("已采纳任务缺少持久回执")
            return "committed"  # A later user restore must survive a repeated acknowledgement.
        if persisted["state"] != "committing":
            raise ValueError("任务未准备采纳")
        rolling_back = "rollback_reason" in persisted_body
        if rolling_back:
            reason = persisted_body["rollback_reason"]
            if (not isinstance(reason, str) or not 0 < len(reason) <= 16384
                    or persisted_body.get("kind") != "evolution" or commit.get("event_kind") != "evolution"
                    or len(commit["pointers"]) != 1 or commit["pointers"][0].get("name") != "active-core.json"
                    or previous):
                raise ValueError("原生核心回滚意图无效或已有采纳事件，等待恢复")
        else:
            for name, digest in commit["artifacts"].items():
                path = _artifact(store, name)
                if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("采纳恢复文件损坏；禁止启用候选")
        for plan in commit["pointers"]:
            if plan["name"] not in {"active.json", "active-core.json", "active-router.json"}:
                raise ValueError("采纳恢复指针路径无效")
            path = store / plan["name"]
            if path.is_symlink():
                raise ValueError("采纳恢复指针不能是符号链接")
            current = json.loads(path.read_text()) if path.exists() else None
            if current not in (plan["before"], plan["after"]):
                raise ValueError("活动版本已被其他任务更改；停止采纳恢复")
        if rolling_back:
            # Startup has no retained native peer: restore disk selection only. Runtime.load
            # subsequently loads the previous source; this is not a fabricated reload ACK.
            plan = commit["pointers"][0]
            if plan["before"] is None:
                (store / plan["name"]).unlink(missing_ok=True)
            else:
                private_json(store / plan["name"], plan["before"])
            directory = os.open(store, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            persisted_body["reason"] = "已按持久回滚意图恢复原指针，等待模型加载：" + reason
            db.execute("UPDATE training_runs SET state='rejected',body=?,updated=? WHERE id=? AND identity=?",
                       (_encoded(persisted_body), time.time(), row["id"], row["binding"]["identity"]))
            return "rejected"
        if "material_publication" in commit:
            _validate_publication(row)
            from .core_publication import apply_material_publication
            apply_material_publication(db, commit["material_publication"], row["binding"]["identity"])
        for plan in commit["pointers"]:
            private_json(store / plan["name"], plan["after"])
        db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?)",
                   (event_id, commit["event_kind"], encoded, row["started"]))
        db.execute("UPDATE training_runs SET state='committed',updated=? WHERE id=?", (time.time(), row["id"]))
        return "committed"
