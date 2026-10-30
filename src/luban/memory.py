"""User-verified samples and conversations, separate from neural weights."""
import json
import hashlib
import sqlite3
import threading
import time
import uuid
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from .settings import require_storage


class Memory:
    def __new__(cls, storage_root):
        from .bun_bridge import enabled, RemoteMemory
        if cls is Memory and enabled():
            return RemoteMemory(storage_root)
        return super().__new__(cls)

    def __init__(self, storage_root):
        self.root = Path(storage_root)
        self.path = self.root / "app/memory.sqlite3"
        self.lock = threading.RLock()
        self.model_identity = None

    def growth_reserved(self, messages):
        path = self.root / "app/growth/rounds.json"
        if not path.exists():
            return False
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("个人成长检验资料不能是符号链接")
        reserved = json.loads(path.read_text()).get("reserved_prompts")
        if not isinstance(reserved, list) or any(not isinstance(p, str) for p in reserved):
            raise ValueError("成长题隔离记录损坏，暂停资料入库")
        normalize = lambda value: "".join(unicodedata.normalize("NFKC", value).split())
        prompts = {normalize(p) for p in reserved}
        return any(normalize(m.get("content", "")) in prompts for m in messages if m.get("role") == "user")

    @contextmanager
    def connect(self):
        require_storage(self.root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        try:
            # ExFAT can change file identity during growth; hold one connection's
            # exclusive lock so SQLite does not mistake this for a moved DB.
            # The service serializes all accesses with Memory.lock.
            db.execute("PRAGMA locking_mode=EXCLUSIVE")
            with db:
                db.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
              id TEXT PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL,
              source TEXT NOT NULL, purpose TEXT NOT NULL, created REAL NOT NULL,
              approved INTEGER NOT NULL, training_allowed INTEGER NOT NULL, provenance TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY, title TEXT NOT NULL, messages TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS usage (
              route TEXT PRIMARY KEY, calls INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL,
              completion_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL, estimated_calls INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS teacher_materials (
              id TEXT PRIMARY KEY, prompt TEXT NOT NULL, answer TEXT NOT NULL, messages TEXT NOT NULL,
              created REAL NOT NULL, provenance TEXT NOT NULL, status TEXT NOT NULL,
              reason TEXT NOT NULL, purpose TEXT NOT NULL, sample_id TEXT);
                """)
                yield db
        finally:
            db.close()

    def add_sample(self, prompt, answer, source="human", purpose="training", approved=False,
                   training_allowed=False, provenance=None):
        if purpose not in {"training", "validation"} or source not in {"human", "teacher", "synthetic"}:
            raise ValueError("学习材料类型无效")
        if not isinstance(prompt, str) or not isinstance(answer, str) or not prompt.strip() or not answer.strip() or len(prompt) + len(answer) > 12000:
            raise ValueError("请填写问题和正确答案，总长度不超过 12000 字符")
        if not approved or not training_allowed:
            raise ValueError("学习材料需要确认正确且允许训练")
        sample_id = uuid.uuid4().hex
        provenance = dict(provenance or {})
        if self.growth_reserved(provenance.get("messages") or [{"role": "user", "content": prompt}]):
            raise ValueError("成长评测题只能检验，不能作为学习材料")
        identity = self.model_identity
        scope = " AND json_extract(provenance, '$.base_identity')=?" if identity else ""
        scope_params = (identity,) if identity else ()
        if identity:
            provenance["base_identity"] = identity
        with self.lock, self.connect() as db:
            duplicate = db.execute("SELECT id FROM samples WHERE prompt=? AND purpose<>?" + scope,
                                   (prompt.strip(), purpose, *scope_params)).fetchone()
            if duplicate:
                raise ValueError("同一问题不能同时用于训练和检验，请使用独立的新问题")
            db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)",
                       (sample_id, prompt.strip(), answer.strip(), source, purpose, time.time(), 1, 1,
                        json.dumps(provenance, ensure_ascii=False)))
        return sample_id

    def samples(self, purpose=None):
        with self.lock, self.connect() as db:
            rows = db.execute("SELECT * FROM samples" + (" WHERE purpose=?" if purpose else "") + " ORDER BY created DESC",
                              (purpose,) if purpose else ()).fetchall()
        records = [dict(r) for r in rows]
        if self.model_identity:
            records = [row for row in records if json.loads(row["provenance"]).get("base_identity") == self.model_identity]
        return [row for row in records if not self.growth_reserved(
            json.loads(row["provenance"]).get("messages") or [{"role": "user", "content": row["prompt"]}])]

    def delete_sample(self, sample_id):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM samples WHERE id=?", (sample_id,))
            db.execute("UPDATE teacher_materials SET status='pending',reason='学习材料已移除',sample_id=NULL WHERE sample_id=?", (sample_id,))

    def record_usage(self, route, usage):
        if route not in {"local", "teacher"}:
            raise ValueError("统计来源无效")
        keys = ("prompt_tokens", "completion_tokens", "total_tokens")
        unknown = usage.get("unknown") is True
        if unknown and any(key not in usage or usage[key] is not None for key in keys):
            raise ValueError("未知词元统计必须保留空值")
        counts = [0, 0, 0] if unknown else [usage.get(k, 0) for k in keys]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
            raise ValueError("词元统计需要非负整数")
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO usage VALUES (?,?,?,?,?,?) ON CONFLICT(route) DO UPDATE SET "
                       "calls=calls+1,prompt_tokens=prompt_tokens+excluded.prompt_tokens,"
                       "completion_tokens=completion_tokens+excluded.completion_tokens,"
                       "total_tokens=total_tokens+excluded.total_tokens,estimated_calls=estimated_calls+excluded.estimated_calls",
                       (route, 1, *counts, int(bool(usage.get("estimated")))))
            if unknown:
                db.execute("INSERT INTO events VALUES (?,?,?,?)", (uuid.uuid4().hex, "usage_unknown",
                           json.dumps({"route": route}), time.time()))

    def usage(self):
        result = {r: dict(calls=0, prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated_calls=0)
                  for r in ("local", "teacher")}
        with self.lock, self.connect() as db:
            for row in db.execute("SELECT * FROM usage"):
                record = dict(row)
                result[record.pop("route")] = record
            for row in db.execute("SELECT json_extract(body,'$.route') AS route,COUNT(*) AS n FROM events "
                                  "WHERE kind='usage_unknown' GROUP BY json_extract(body,'$.route')"):
                if row["route"] in result:
                    result[row["route"]].update(unknown_calls=row["n"], prompt_tokens=None, completion_tokens=None, total_tokens=None)
        return result

    def capture_teacher(self, messages, answer, provenance, eligible, reason=""):
        prompt = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        eligible = teacher_material_eligible(prompt, answer, eligible)
        if self.growth_reserved(messages):
            eligible, reason = False, "成长评测题只用于检验，不进入学习"
        purpose = "validation" if int(hashlib.sha256(prompt.strip().encode()).hexdigest(), 16) % 5 == 0 else "training"
        material_id = uuid.uuid4().hex
        provenance = {**provenance, "messages": messages, "verification_scope": "structural checks only; no factual truth guarantee"}
        identity = self.model_identity
        if identity:
            provenance["base_identity"] = identity
        encoded = json.dumps(provenance, ensure_ascii=False)
        with self.lock, self.connect() as db:
            sample_id = None
            if eligible:
                sample_id, purpose = capture_teacher_sample(db, prompt, answer, purpose, encoded, identity)
            db.execute("INSERT INTO teacher_materials VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (material_id, prompt, answer, json.dumps(messages, ensure_ascii=False), time.time(), encoded,
                        "eligible" if eligible else "pending", reason, purpose, sample_id))
        return material_id

    def teacher_materials(self):
        with self.lock, self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM teacher_materials ORDER BY created DESC LIMIT 100")]

    def material_counts(self):
        with self.lock, self.connect() as db:
            counts = {r["status"]: r["n"] for r in db.execute("SELECT status,COUNT(*) AS n FROM teacher_materials GROUP BY status")}
        return {"total": sum(counts.values()), "pending": counts.get("pending", 0), "eligible": counts.get("eligible", 0)}

    def save_conversation(self, conversation_id, messages):
        conversation_id = conversation_id or uuid.uuid4().hex
        title = next((m["content"][:40] for m in messages if m["role"] == "user"), "新对话")
        with self.lock, self.connect() as db:
            db.execute("INSERT OR REPLACE INTO conversations VALUES (?,?,?,?)",
                       (conversation_id, title, json.dumps(messages, ensure_ascii=False), time.time()))
        return conversation_id

    def conversations(self):
        with self.lock, self.connect() as db:
            return [dict(r) for r in db.execute("SELECT id,title,updated FROM conversations ORDER BY updated DESC LIMIT 50")]

    def conversation(self, conversation_id):
        with self.lock, self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return None if row is None else {**dict(row), "messages": json.loads(row["messages"])}

    def delete_conversation(self, conversation_id):
        with self.lock, self.connect() as db:
            db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def event(self, kind, body):
        event_id = uuid.uuid4().hex
        with self.lock, self.connect() as db:
            db.execute("INSERT INTO events VALUES (?,?,?,?)", (event_id, kind, json.dumps(body, ensure_ascii=False), time.time()))
        return event_id

    def events(self):
        with self.lock, self.connect() as db:
            return [{**dict(r), "body": json.loads(r["body"])} for r in db.execute("SELECT * FROM events ORDER BY created DESC LIMIT 30")]


def teacher_material_eligible(prompt, answer, eligible=True):
    return bool(eligible and prompt.strip() and answer.strip() and len(prompt) + len(answer) <= 12000)


def capture_teacher_sample(db, prompt, answer, purpose, encoded, identity):
    """Shared admission storage; callers have already checked structural eligibility."""
    scope = " AND json_extract(provenance, '$.base_identity')=?" if identity else ""
    scope_params = (identity,) if identity else ()
    previous = db.execute("SELECT id,answer,purpose FROM samples WHERE prompt=?" + scope + " ORDER BY created LIMIT 1",
                          (prompt.strip(), *scope_params)).fetchone()
    if previous:
        purpose = previous["purpose"]
    eligible_scope = " AND approved=1 AND training_allowed=1" if identity else ""
    duplicate = db.execute("SELECT id FROM samples WHERE prompt=? AND answer=?" + scope + eligible_scope,
                           (prompt.strip(), answer.strip(), *scope_params)).fetchone()
    sample_id = duplicate["id"] if duplicate else uuid.uuid4().hex
    if not duplicate:
        db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)",
                   (sample_id, prompt.strip(), answer.strip(), "teacher", purpose, time.time(), 1, 1, encoded))
    return sample_id, purpose
