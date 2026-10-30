"""Private durable teacher-first practice, with no weight updates or automatic paid retries."""
from __future__ import annotations

import json
import threading
import time

from .review import review_candidates, _reference_checks, _locally_verified, _reviews
from .memory import capture_teacher_sample, teacher_material_eligible
from .thinking import ThinkingStream


class TeacherPractice:
    def __init__(self, runtime, settings, teacher):
        self.runtime, self.settings, self.teacher = runtime, settings, teacher
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.initialized = None
        self.media_learning = None

    def _database(self):
        return self.runtime.memory.connect()

    def _initialize(self):
        with self.lock:
            memory = self.runtime.memory
            binding = (memory, str(memory.root), str(memory.path))
            if self.initialized == binding:
                return
            with self.runtime.memory.lock, self._database() as db:
                db.execute("""CREATE TABLE IF NOT EXISTS teacher_practice (
                    request_id TEXT PRIMARY KEY, identity TEXT, messages TEXT NOT NULL,
                    material_id TEXT NOT NULL, state TEXT NOT NULL, decision_id TEXT,
                    created REAL NOT NULL, updated REAL NOT NULL)""")
                # Local interrupted work is repeatable. A paid request with unknown
                # outcome is never replayed merely because the service restarted.
                db.execute("UPDATE teacher_practice SET state='queued' WHERE state='generating'")
                db.execute("UPDATE teacher_practice SET state='interrupted' WHERE state='review_started'")
                db.execute("UPDATE teacher_practice SET state='unbound' WHERE identity IS NULL AND state='answer_ready'")
            self.initialized = binding

    def enqueue(self, request_id, messages, material_id="", *, decision_id=None):
        self._initialize()
        identity = getattr(self.runtime, "identity", None)
        identity = identity if isinstance(identity, str) and identity else None
        state = ("answer_ready" if identity else "unbound") if decision_id else "queued"
        with self.runtime.memory.lock, self._database() as db:
            db.execute("INSERT OR IGNORE INTO teacher_practice VALUES (?,?,?,?,?,?,?,?)",
                       (request_id, identity, json.dumps(messages, ensure_ascii=False),
                        material_id, state, decision_id, time.time(), time.time()))

    def pending(self):
        if self.media_learning and self.media_learning.teacher_pending():
            return True
        if "text" not in self.settings.snapshot()["teacher"].get("teaching_modalities", ["text"]):
            return False
        if not isinstance(getattr(self.runtime, "identity", None), str) or not self.runtime.identity:
            return False
        self._initialize()
        with self.runtime.memory.lock, self._database() as db:
            states = {row["state"] for row in db.execute("SELECT DISTINCT state FROM teacher_practice "
                      "WHERE state IN ('queued','answer_ready') AND (identity=? OR identity IS NULL AND state='queued')",
                      (getattr(self.runtime, "identity", None),))}
        return "queued" in states or "answer_ready" in states and self._can_review()

    def _can_review(self):
        try:
            if not self.settings.agreement()["accepted"]:
                return False
            config, _ = self.teacher._config()
            if "text" not in config.get("teaching_modalities", ["text"]):
                return False
            with self.teacher.lock:
                limit = config["session_call_limit"]
                return bool(config["model"] and (limit == 0 or limit - self.teacher.calls >= 2))
        except (ValueError, RuntimeError):
            return False

    def pause(self):
        self.stop_event.set()

    def _state(self, request_id, state, decision_id=None):
        with self.runtime.memory.lock, self._database() as db:
            db.execute("UPDATE teacher_practice SET state=?,decision_id=COALESCE(?,decision_id),updated=? WHERE request_id=?",
                       (state, decision_id, time.time(), request_id))

    def _recheck_material(self, item, messages):
        if not item["material_id"]:
            return  # Local-first actual answers have no teacher material to promote.
        memory, runtime = self.runtime.memory, self.runtime
        with memory.lock, self._database() as db:
            row = db.execute("SELECT * FROM teacher_materials WHERE id=?", (item["material_id"],)).fetchone()
            if row is None or row["status"] != "pending":
                return
            original = dict(row)
        source = json.loads(original["provenance"])
        if json.loads(original["messages"]) != messages or source.get("base_identity") not in (None, runtime.identity):
            raise ValueError("教师材料与练习上下文或模型不一致")
        # Ordinary RPC methods must run without an active DB lease: Bun waits
        # for that lease before admitting another memory operation.
        agreement = self.settings.agreement()
        reserved = memory.growth_reserved(messages)
        authorized = (source.get("user_responsibility") is True and agreement["accepted"]
                      and source.get("agreement_version") == agreement["version"])
        eligible, reason = runtime.teacher_eligibility(messages, original["answer"], {"agreement_accepted": bool(authorized)})
        if not teacher_material_eligible(original["prompt"], original["answer"]):
            eligible, reason = False, "问题和答案须非空且总长度不超过 12000 字符"
        if reserved:
            eligible, reason = False, "成长评测题只用于检验，不进入学习"
        with memory.lock, self._database() as db:
            current = db.execute("SELECT * FROM teacher_materials WHERE id=?", (item["material_id"],)).fetchone()
            if current is None or dict(current) != original:
                raise ValueError("教师材料已变化，停止当前练习")
            if not eligible:
                db.execute("UPDATE teacher_materials SET reason=? WHERE id=?", (reason, item["material_id"]))
                return
            source.update(base_identity=runtime.identity, messages=messages)
            encoded = json.dumps(source, ensure_ascii=False)
            sample_id, purpose = capture_teacher_sample(db, original["prompt"], original["answer"], original["purpose"], encoded, runtime.identity)
            db.execute("UPDATE teacher_materials SET status='eligible',reason=?,purpose=?,sample_id=?,provenance=? WHERE id=?",
                       (reason, purpose, sample_id, encoded, item["material_id"]))

    def run_one(self):
        """Caller owns the command lock and foreground-busy admission."""
        if self.media_learning and self.media_learning.teacher_pending():
            return self.media_learning.run_teacher_one(self.stop_event)
        if not self.pending() or self.stop_event.is_set():
            return
        runtime = self.runtime
        states = "('queued','answer_ready')" if self._can_review() else "('queued')"
        with runtime.memory.lock, self._database() as db:
            row = db.execute("SELECT * FROM teacher_practice WHERE state IN " + states + " "
                             "AND (identity=? OR identity IS NULL AND state='queued') ORDER BY created,request_id LIMIT 1",
                             (runtime.identity,)).fetchone()
            if row is None:
                return
            item = dict(row)
            # An unassigned question has no old local result or qualification.
            # Bind the first actual attempt once; completed local records never
            # inherit a later model's identity.
            if item["identity"] is None:
                db.execute("UPDATE teacher_practice SET identity=? WHERE request_id=? AND identity IS NULL AND state='queued'",
                           (runtime.identity, item["request_id"]))
        identifier = item["request_id"]
        if item["state"] == "queued":
            self._state(identifier, "generating")
            parts = []
            parser = ThinkingStream(lambda delta: parts.append(delta["content"]) if delta.get("content") else None)
            try:
                # Original pre-answer context only: never show the teacher's
                # current answer, external references, or a teacher's hints.
                messages = json.loads(item["messages"])
                with runtime.lock:
                    if self.stop_event.is_set() or runtime.state != "ready":
                        self._state(identifier, "queued")
                        return
                    self._recheck_material(item, messages)
                    usage = runtime.generate(messages, lambda delta: parser.feed(delta["content"]) if delta.get("content") else None,
                                             self.stop_event, thinking=self.settings.snapshot()["local_thinking"],
                                             allow_reference=False, observe_external=False)
                parser.finish()
                runtime.memory.record_usage("local", usage)
                if self.stop_event.is_set():
                    self._state(identifier, "queued")
                    return
                if (usage.get("finish_reason") != "stop" or usage.get("local_complete") is not True
                        or usage.get("reference_used") is not False or not "".join(parts).strip()):
                    self._state(identifier, "incomplete")
                    return
                decision_id = runtime.memory.event("decision", {
                    "request_id": identifier + ":practice", "messages": messages, "answer": "".join(parts),
                    "decision": {"route": "local", "action": usage.get("local_action", "local_core"), "source": "teacher_first_practice"},
                    "base_identity": runtime.identity,
                    "model_context": {key: usage.get(key) for key in ("core_version", "expert_id", "expert_core_version")},
                    "usage": usage, "complete": True, "teacher_material_id": item["material_id"]})
                self._state(identifier, "answer_ready", decision_id)
                item["decision_id"] = decision_id
            except Exception:
                self._state(identifier, "local_failed")
                return  # Private errors must not leak prompts or credentials to monitor logs.
        if self.stop_event.is_set():
            return
        candidate_id = "route:" + item["decision_id"]
        # Reuse the same independently supported local checks as manual review.
        # A reviewed/locally proven answer needs no second paid request.
        with runtime.lock:
            completed = next((row for row in _reviews(runtime.memory)
                              if row.get("candidate_id") == candidate_id and row.get("status") == "complete"
                              and (row.get("result") or {}).get("correct") is not None), None)
            if completed:
                self._state(identifier, "complete")
                return
            with runtime.memory.lock, self._database() as db:
                event = db.execute("SELECT body FROM events WHERE id=? AND kind='decision'", (item["decision_id"],)).fetchone()
            if event is None:
                self._state(identifier, "missing_answer")
                return
            record = json.loads(event["body"])
            candidate = {"kind": "route", "record": record,
                         "prompt": next((message["content"] for message in reversed(record["messages"])
                                         if message["role"] == "user"), "")}
            if _locally_verified(candidate, _reference_checks(runtime)):
                self._state(identifier, "verified_local")
                return
        # Validate permissions/configuration/remaining budget before crossing the
        # durable paid boundary. review_candidates repeats its own checks.
        if not self._can_review():
            return
        if self.stop_event.is_set():
            return
        self._state(identifier, "review_started")
        try:
            result = review_candidates(runtime, self.settings, self.teacher, [candidate_id], stop_event=self.stop_event)
            verdicts = [row.get("result") for row in result.get("reviews", [])]
            state = ("review_failed" if result.get("stopped_reason") else
                     "unknown" if not verdicts or any(not verdict or verdict.get("correct") is None for verdict in verdicts) else "complete")
            self._state(identifier, state)
        except Exception:
            self._state(identifier, "review_failed")
