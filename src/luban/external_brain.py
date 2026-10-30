"""Bounded exact-context reference hints, separate from learned neural weights."""
from __future__ import annotations

import hashlib
import json
import re
import time

from .generation_checks import evaluate_generation, validate_generation_check, is_finite_generation_check
from .retention import messages, provenance, verified_source

CAPACITY = 4096
MAX_CONTEXT_BYTES = 64 * 1024
MAX_ANSWER_CHARS = 12000
_HEX = re.compile(r"[a-f0-9]{64}")


def _encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _context(context, identity):
    if not isinstance(identity, str) or not _HEX.fullmatch(identity):
        raise ValueError("外脑需要完整基座身份 SHA")
    if (not isinstance(context, list) or not 1 <= len(context) <= 128
            or any(not isinstance(row, dict) or set(row) != {"role", "content"}
                   or row["role"] not in {"system", "user", "assistant"}
                   or not isinstance(row["content"], str) or not row["content"].strip() for row in context)
            or context[-1]["role"] != "user"):
        raise ValueError("外脑上下文必须保留完整角色/正文并以用户问题结束")
    encoded = _encoded(context)
    if len(encoded.encode()) > MAX_CONTEXT_BYTES:
        raise ValueError("外脑上下文超过固定大小限制")
    # No whitespace/case/Unicode normalization: exact role/content only.
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


def _binding(binding):
    if not isinstance(binding, dict) or set(binding) not in ({"core_version", "core_sha256", "expert_id", "expert_sha256"},
                                              {"core_version", "core_sha256", "expert_id", "expert_sha256", "inference"},
                                              {"core_version", "core_sha256", "expert_id", "expert_sha256", "inference", "core_inference"}):
        raise ValueError("本地掌握判定需要真实核心/专家版本与权重 SHA")
    if (not isinstance(binding["core_sha256"], str) or not _HEX.fullmatch(binding["core_sha256"])
            or binding["core_version"] is not None and not re.fullmatch(r"core-[a-f0-9]{12}", str(binding["core_version"]))):
        raise ValueError("核心版本绑定无效")
    expert, digest = binding["expert_id"], binding["expert_sha256"]
    if ((expert is None) != (digest is None) or expert is not None
            and (not isinstance(expert, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", expert)
                 or not isinstance(digest, str) or not _HEX.fullmatch(digest))):
        raise ValueError("专家版本绑定无效")
    for key in ("inference", "core_inference"):
        if key not in binding:
            continue
        engine = binding[key]
        if not isinstance(engine, dict) or not isinstance(engine.get("engine"), str) or len(_encoded(engine)) > 512:
            raise ValueError("推理引擎绑定无效")
    return _encoded(binding)


def _schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS external_brain (
        identity TEXT NOT NULL, context_sha TEXT NOT NULL, context_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('needs_teacher','reference_ready','mastered')),
        sample_id TEXT, sample_sha TEXT, mastered_binding TEXT, teacher_material_id TEXT,
        last_used REAL NOT NULL, observation_json TEXT, checked_binding TEXT, check_json TEXT,
        PRIMARY KEY(identity,context_sha))""")
    columns = {row[1] for row in db.execute("PRAGMA table_info(external_brain)")}
    for name in ("checked_binding", "check_json"):
        if name not in columns:
            db.execute("ALTER TABLE external_brain ADD COLUMN " + name + " TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS external_brain_lru ON external_brain(last_used)")
    db.execute("CREATE INDEX IF NOT EXISTS external_reference_prompt ON samples(prompt,approved,training_allowed,created,id)")


def _sample(db, sample_id, context, identity, expected_sha=None):
    if not sample_id:
        return None
    raw = db.execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone()
    if raw is None:
        return None
    row = dict(raw)
    try:
        value = provenance(row)
        valid = (row.get("approved") == 1 and row.get("training_allowed") == 1 and verified_source(row)
                 and value.get("base_identity") == identity and messages(row) == context
                 and isinstance(row.get("answer"), str) and 0 < len(row["answer"].strip()) <= MAX_ANSWER_CHARS
                 and not value.get("route_only") and not value.get("retention_generation_only"))
    except (TypeError, ValueError, KeyError):
        return None
    if not valid:
        return None
    digest = hashlib.sha256(_encoded(row).encode()).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        return None
    return row, digest


def _mastered_matches(hint, encoded_binding):
    if hint["state"] != "mastered" or not hint["mastered_binding"] or not hint["observation_json"]:
        return False
    try:
        prior, current = json.loads(hint["mastered_binding"]), json.loads(encoded_binding)
        observation = json.loads(hint["observation_json"])
        checks = observation.get("checks", [])
        if (observation.get("complete") is not True or observation.get("reference_used") is not False
                or observation.get("binding") != prior or not isinstance(checks, list) or len(checks) != 2
                or any(not isinstance(check, dict) or check.get("verdict") is not True or check.get("complete") is not True
                       or check.get("reference_used") is not False or check.get("binding") != prior
                       or check.get("context_sha256") != hint["context_sha"]
                       or check.get("sample_sha256") != hint["sample_sha"]
                       or check.get("local_action") != observation.get("local_action") for check in checks)):
            return False
        if observation.get("local_action") == "local_core" and prior["expert_id"] is None:
            # This proof is about branch-OFF execution. The caller must continue
            # to force local_core even if a different expert is now selected.
            return (prior.get("inference") == current.get("core_inference", current.get("inference"))
                    and all(prior[key] == current[key] for key in ("core_version", "core_sha256")))
        return observation.get("local_action") == "neural_expert" and prior == current
    except (ValueError, TypeError, KeyError):
        return False


def _put(db, identity, digest, encoded, *, state, sample_id=None, sample_sha=None, binding=None, teacher_id=None,
         observation=None, checked_binding=None, check=None):
    exists = db.execute("SELECT context_json FROM external_brain WHERE identity=? AND context_sha=?", (identity, digest)).fetchone()
    if exists is not None and exists["context_json"] != encoded:
        raise ValueError("外脑完整上下文与已有摘要冲突")
    evicted = 0
    if exists is None:
        count = db.execute("SELECT COUNT(*) FROM external_brain").fetchone()[0]
        evicted = max(0, count - CAPACITY + 1)
        if evicted:
            db.execute("DELETE FROM external_brain WHERE rowid IN (SELECT rowid FROM external_brain ORDER BY last_used,rowid LIMIT ?)", (evicted,))
    db.execute("""INSERT INTO external_brain VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(identity,context_sha) DO UPDATE SET context_json=excluded.context_json,state=excluded.state,
        sample_id=excluded.sample_id,sample_sha=excluded.sample_sha,mastered_binding=excluded.mastered_binding,
        teacher_material_id=excluded.teacher_material_id,last_used=excluded.last_used,observation_json=excluded.observation_json,
        checked_binding=excluded.checked_binding,check_json=excluded.check_json""",
        (identity, digest, encoded, state, sample_id, sample_sha, binding, teacher_id, time.time(),
         _encoded(observation) if observation else None, checked_binding, _encoded(check) if check else None))
    return evicted


class ExternalBrain:
    def __init__(self, memory):
        self.memory = memory

    def _identity(self, identity):
        current = getattr(self.memory, "model_identity", None)
        if current is not None and current != identity:
            raise ValueError("外脑请求与当前资料库基座不一致")

    def _lookup(self, db, context, identity, digest, encoded, binding):
        hint = db.execute("SELECT * FROM external_brain WHERE identity=? AND context_sha=?", (identity, digest)).fetchone()
        if hint is None:
            return {"status": "needs_teacher", "context_sha256": digest, "reference": None,
                    "has_hint": False, "teacher_needed": False, "external_reference": False, "mastery_scope": "not observed"}
        if hint["context_json"] != encoded:
            raise ValueError("完整上下文与外脑摘要冲突")
        reference = _sample(db, hint["sample_id"], context, identity, hint["sample_sha"])
        state = hint["state"]
        if reference is None:
            state = "needs_teacher"
        elif state == "mastered" and not _mastered_matches(hint, binding):
            state = "reference_ready"
        elif state != "mastered":
            state = "reference_ready"
        if state != hint["state"] or reference is None:
            db.execute("UPDATE external_brain SET state=?,sample_id=?,sample_sha=?,mastered_binding=NULL,observation_json=NULL,checked_binding=?,check_json=? WHERE identity=? AND context_sha=?",
                       (state, reference[0]["id"] if reference else None, reference[1] if reference else None,
                        hint["checked_binding"] if reference else None, hint["check_json"] if reference else None, identity, digest))
        db.execute("UPDATE external_brain SET last_used=? WHERE identity=? AND context_sha=?", (time.time(), identity, digest))
        return {"status": state, "context_sha256": digest, "has_hint": True, "teacher_needed": state != "mastered",
                "reference": None if reference is None else {**reference[0], "sample_id": reference[0]["id"],
                    "provenance": provenance(reference[0]), "sample_sha256": reference[1]},
                "external_reference": reference is not None,
                "mastery_scope": "two complete closed-book checks of one exact context and binding" if state == "mastered" else "not observed",
                "teacher_material_id": hint["teacher_material_id"],
                "local_observation": json.loads(hint["observation_json"]) if state == "mastered" and hint["observation_json"] else None,
                "local_action": json.loads(hint["observation_json"]).get("local_action") if state == "mastered" and hint["observation_json"] else None,
                "checked_binding": json.loads(hint["checked_binding"]) if reference and hint["checked_binding"] else None,
                "last_check": json.loads(hint["check_json"]) if reference and hint["check_json"] else None}

    def lookup(self, context, identity, binding):
        self._identity(identity)
        digest, encoded = _context(context, identity)
        binding = _binding(binding)
        with self.memory.lock, self.memory.connect() as db:
            _schema(db)
            return self._lookup(db, context, identity, digest, encoded, binding)

    def record_teacher(self, context, identity, answer, provenance=None, *, material_id=None):
        """Archive only: not a teacher call, approval, or factual verification."""
        self._identity(identity)
        digest, encoded = _context(context, identity)
        if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_ANSWER_CHARS:
            raise ValueError("教师答复为空或超过固定大小限制")
        supplied = provenance or {}
        if not isinstance(supplied, dict):
            raise ValueError("教师来源必须为对象")
        # Keep only descriptive attribution; ignore all purported verification.
        source = {key: supplied[key][:1000] for key in ("provider", "model", "request_id") if isinstance(supplied.get(key), str)}
        source.update(base_identity=identity, external_brain=True, verification_scope="pending; teacher output alone is not truth")
        with self.memory.lock:
            if material_id is None:
                material_id = self.memory.capture_teacher(context, answer, source, eligible=False,
                    reason="外脑教师答复待独立核验；未授权为正确参考或训练材料")
            else:
                if not isinstance(material_id, str) or not re.fullmatch(r"[a-f0-9]{32}", material_id):
                    raise ValueError("已有教师归档 ID 无效")
                with self.memory.connect() as db:
                    archived = db.execute("SELECT * FROM teacher_materials WHERE id=?", (material_id,)).fetchone()
                    attribution = {} if archived is None else json.loads(archived["provenance"])
                    fields = ("provider", "model", "request_id", "rights_basis", "agreement_version", "user_responsibility", "created")
                    if (archived is None or json.loads(archived["messages"]) != context or archived["answer"] != answer
                            or attribution.get("base_identity") != identity or attribution.get("messages") != context
                            or any(attribution.get(key) != supplied[key] for key in fields if key in supplied)):
                        raise ValueError("教师归档与完整上下文、答案、基座或来源不匹配")
                # Existing archive status/sample approval belongs to its caller.
                # Merely linking it never makes it a verified external reference.
            with self.memory.connect() as db:
                _schema(db)
                old = db.execute("SELECT * FROM external_brain WHERE identity=? AND context_sha=?", (identity, digest)).fetchone()
                reference = _sample(db, old["sample_id"], context, identity, old["sample_sha"]) if old else None
                evicted = _put(db, identity, digest, encoded, state="reference_ready" if reference else "needs_teacher",
                    sample_id=reference[0]["id"] if reference else None, sample_sha=reference[1] if reference else None, teacher_id=material_id,
                    checked_binding=old["checked_binding"] if reference else None,
                    check=json.loads(old["check_json"]) if reference and old["check_json"] else None)
        return {"status": "reference_ready" if reference else "needs_teacher", "has_hint": True, "teacher_needed": True,
                "teacher_material_id": material_id, "verified_source": False, "training_allowed": False,
                "evicted_hints": evicted, "capacity": CAPACITY}

    def register_reference(self, sample_id, context, identity, binding):
        """Link a currently qualified existing sample, never mint verification."""
        self._identity(identity)
        digest, encoded = _context(context, identity)
        binding = _binding(binding)
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("需要已有合格材料的 ID")
        with self.memory.lock, self.memory.connect() as db:
            _schema(db)
            reference = _sample(db, sample_id, context, identity)
            if reference is None:
                raise ValueError("材料未独立核验、未授权、上下文或基座不匹配")
            old = self._lookup(db, context, identity, digest, encoded, binding)
            if old["reference"] is not None and old["reference"]["id"] != sample_id and old["reference"]["answer"] != reference[0]["answer"]:
                raise ValueError("同一完整上下文的已核验参考冲突；请先撤销错误参考")
            # Repeated registration must not erase a matching mastery observation.
            same_reference = (old["reference"] is not None and old["reference"]["sample_id"] == sample_id
                              and old["reference"]["sample_sha256"] == reference[1])
            if same_reference:
                return {**old, "evicted_hints": 0, "capacity": CAPACITY}
            evicted = _put(db, identity, digest, encoded, state="reference_ready",
                sample_id=sample_id, sample_sha=reference[1],
                teacher_id=old.get("teacher_material_id"))
            result = self._lookup(db, context, identity, digest, encoded, binding)
            return {**result, "evicted_hints": evicted, "capacity": CAPACITY}

    def discover_reference(self, context, identity, binding):
        """Bounded indexed search of existing qualified samples for this exact prompt."""
        with self.memory.lock:
            current = self.lookup(context, identity, binding)
            if current["reference"] is not None:
                return current
            selected, gold = None, None
            with self.memory.connect() as db:
                _schema(db)
                # Memory.add_sample stores prompt.strip(), while full messages
                # retain original whitespace; complete context equality below
                # remains mandatory and does not normalize the actual question.
                rows = db.execute("""SELECT id FROM samples WHERE prompt=? AND approved=1 AND training_allowed=1
                    AND CASE WHEN json_valid(provenance) THEN json_extract(provenance,'$.base_identity') END=?
                    ORDER BY created,id LIMIT ?""", (context[-1]["content"].strip(), identity, CAPACITY + 1))
                for index, row in enumerate(rows):
                    if index >= CAPACITY:
                        raise ValueError("同题候选超过固定4096条上限，不能自动选择参考")
                    reference = _sample(db, row["id"], context, identity)
                    if reference is None:
                        continue
                    if selected is None:
                        selected, gold = row["id"], reference[0]["answer"]
                    elif gold != reference[0]["answer"]:
                        raise ValueError("同一完整上下文存在互相冲突的已核验参考，不能自动选择")
            return current if selected is None else self.register_reference(selected, context, identity, binding)

    def observe_local(self, context, identity, binding, answer, *, complete, actual_context, reference_used, local_action):
        """Caller must provide actual generation context/usage, never a judge score."""
        self._identity(identity)
        digest, encoded = _context(context, identity)
        encoded_binding = _binding(binding)
        if not isinstance(answer, str) or len(answer) > MAX_ANSWER_CHARS or type(complete) is not bool or type(reference_used) is not bool:
            raise ValueError("本地回答、EOS 或外部参考标记无效")
        actual_digest, actual_encoded = _context(actual_context, identity)
        actual_local = (local_action in {"local_core", "neural_expert"}
                        and (local_action == "neural_expert") == (binding["expert_id"] is not None))
        independent = not reference_used and actual_digest == digest and actual_encoded == encoded and actual_local
        with self.memory.lock, self.memory.connect() as db:
            _schema(db)
            current = self._lookup(db, context, identity, digest, encoded, encoded_binding)
            reference = current["reference"]
            if not current["has_hint"] and reference is None:
                return {**current, "verdict": None, "method": "no_verified_reference", "evicted_hints": 0, "capacity": CAPACITY}
            verdict, method, observation = None, "no_verified_reference", None
            if reference is not None and independent:
                try:
                    check = validate_generation_check(reference["provenance"].get("generation_check"), default_expected=reference["answer"])
                    finite = is_finite_generation_check(check)
                    if not finite:
                        method = "advisory_only"
                    elif not evaluate_generation(reference["answer"], check, True)["passed"]:
                        method = "reference_rule_conflict"
                    else:
                        verdict = evaluate_generation(answer, check, complete)["passed"]
                        method = "exact_reference" if check["type"] == "exact" else "finite_program_rule"
                except (ValueError, TypeError, KeyError, RecursionError):
                    method = "invalid_reference_rule"
            elif reference is not None:
                method = "reference_used_or_context_or_local_binding_mismatch"
            checked_binding = _encoded(current["checked_binding"]) if current.get("checked_binding") else None
            check_record = current.get("last_check")
            previous_check = check_record
            consecutive = 0
            if reference is not None and independent:
                checked_binding = encoded_binding
                check_record = {"answer_sha256": hashlib.sha256(answer.encode()).hexdigest(), "context_sha256": digest,
                    "sample_sha256": reference["sample_sha256"], "complete": complete, "reference_used": False,
                    "local_action": local_action, "binding": binding, "verdict": verdict, "method": method, "checked_at": time.time()}
                if verdict is True:
                    same_previous = (previous_check is not None and previous_check.get("verdict") is True
                        and previous_check.get("complete") is True and previous_check.get("reference_used") is False
                        and previous_check.get("binding") == binding and previous_check.get("local_action") == local_action
                        and previous_check.get("context_sha256") == digest
                        and previous_check.get("sample_sha256") == reference["sample_sha256"]
                        and type(previous_check.get("consecutive_passes")) is int
                        and previous_check["consecutive_passes"] >= 1)
                    consecutive = 2 if same_previous else 1
                check_record["consecutive_passes"] = consecutive
            elif check_record is not None:
                # Reading a reference or a different context cannot bridge two
                # closed-book observations into a mastery proof.
                check_record = {**check_record, "consecutive_passes": 0}
            if consecutive == 2:
                observation = {"answer_sha256": check_record["answer_sha256"], "context_sha256": digest,
                    "complete": True, "reference_used": False, "local_action": local_action, "method": method,
                    "binding": binding, "observed_at": time.time(), "checks": [previous_check, check_record]}
            state = "mastered" if consecutive == 2 else "reference_ready" if reference else "needs_teacher"
            evicted = _put(db, identity, digest, encoded, state=state,
                sample_id=reference["id"] if reference else None, sample_sha=reference["sample_sha256"] if reference else None,
                binding=encoded_binding if consecutive == 2 else None, teacher_id=current.get("teacher_material_id"), observation=observation,
                checked_binding=checked_binding, check=check_record)
            result = self._lookup(db, context, identity, digest, encoded, encoded_binding)
            return {**result, "verdict": verdict, "method": method, "evicted_hints": evicted, "capacity": CAPACITY,
                    "consecutive_passes": consecutive,
                    "scope": "two closed-book exact-context observations required; not a general learning claim"}

    def stats(self, identity, binding):
        """Bounded streaming hint audit; never load the entire samples/history table."""
        self._identity(identity)
        _context([{"role": "user", "content": "stats"}], identity)
        encoded_binding = _binding(binding)
        with self.memory.lock, self.memory.connect() as db:
            _schema(db)
            cursor = db.execute("SELECT * FROM external_brain WHERE identity=? ORDER BY rowid LIMIT ?", (identity, CAPACITY + 1))
            for index, hint in enumerate(cursor):
                if index >= CAPACITY:
                    raise ValueError("外脑提示表超过固定容量，停止读取")
                context = json.loads(hint["context_json"])
                digest, encoded = _context(context, identity)
                if digest != hint["context_sha"] or encoded != hint["context_json"]:
                    raise ValueError("外脑上下文摘要损坏")
                reference = _sample(db, hint["sample_id"], context, identity, hint["sample_sha"])
                state = ("needs_teacher" if reference is None else "mastered"
                         if _mastered_matches(hint, encoded_binding) else "reference_ready")
                if state != hint["state"] or reference is None:
                    db.execute("UPDATE external_brain SET state=?,sample_id=?,sample_sha=?,mastered_binding=NULL,observation_json=NULL,checked_binding=?,check_json=? WHERE identity=? AND context_sha=?",
                               (state, hint["sample_id"] if reference else None, hint["sample_sha"] if reference else None,
                                hint["checked_binding"] if reference else None, hint["check_json"] if reference else None, identity, digest))
            counts = {"needs_teacher": 0, "reference_ready": 0, "mastered": 0}
            counts.update({row["state"]: row["n"] for row in db.execute("SELECT state,COUNT(*) AS n FROM external_brain WHERE identity=? GROUP BY state", (identity,))})
            return {**counts, "total": sum(counts.values()), "capacity": CAPACITY,
                    "scope": "exact-context hints; separate from model learning/evolution counts"}

    def recheck_candidates(self, identity, binding, limit=4):
        """At most four original-context local probes, once per actual binding."""
        if type(limit) is not int or not 1 <= limit <= 4:
            raise ValueError("外脑每次只能复测1至4条")
        encoded_binding = _binding(binding)
        with self.memory.lock:
            self.stats(identity, binding)
            with self.memory.connect() as db:
                _schema(db)
                rows = db.execute("""SELECT context_json,context_sha FROM external_brain
                    WHERE identity=? AND state='reference_ready' AND (checked_binding IS NULL OR checked_binding<>?
                        OR CASE WHEN json_valid(check_json) THEN json_extract(check_json,'$.consecutive_passes') END=1)
                    ORDER BY last_used,context_sha LIMIT ?""", (identity, encoded_binding, limit)).fetchall()
                result = []
                for row in rows:
                    context = json.loads(row["context_json"])
                    hint = self._lookup(db, context, identity, row["context_sha"], row["context_json"], encoded_binding)
                    if hint["status"] == "reference_ready" and hint["reference"] is not None:
                        result.append({"context": context, "context_sha256": row["context_sha"], "reference": hint["reference"],
                                       "binding": json.loads(encoded_binding)})
                return result
