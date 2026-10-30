"""Read-only learning content projection; actual evidence, never accepted TRAIN counts."""
import copy
import json
import threading
import unittest
from unittest.mock import patch

import test_automatic as fixtures
import test_partial_evolution as partial
from luban.automatic import AutomaticLearning, knowledge_keys, digest
from luban.mlx_learning import QUALITY_GATES
from luban.evolution import consolidate
from luban.retention import daily_learning_source
from luban.training_checkpoint import training_runs


class LearningContentsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AutomaticTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.runtime, self.auto = self.fixture.runtime, self.fixture.auto
        self.runtime.lock = threading.RLock()
        patcher = patch("luban.mlx_learning.accepted_report", return_value=None)
        self.accepted_report = patcher.start()
        self.addCleanup(patcher.stop)

    def _proofs_without_reference(self):
        row = training_runs(self.runtime.memory, self.runtime.identity)[0]
        candidate = row["commit"]["report"]["candidates"][0]
        for proof in candidate["generation_checks"] + candidate["core_retention_generation"]:
            proof["reference_used"] = False
        self._save(row)
        return row

    def _save(self, row):
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE training_runs SET body=? WHERE id=?", (json.dumps(row), row["id"]))

    def test_only_verified_product_teaching_is_listed_without_deleting_research(self):
        rows = [fixtures.sample("manual"),
                {**fixtures.sample("feeding"), "provenance": json.dumps({"feeding_batch_id": "batch",
                    "human_verified": True, "verification_evidence": "user checked QA"})},
                {**fixtures.sample("correction"), "source": "teacher", "provenance": json.dumps({
                    "request_id": "chat", "messages": [{"role": "user", "content": "correction"}],
                    "human_verified": True, "verification_evidence": "user verified correction"})}]
        research = [
            {**fixtures.sample("v4系统1 · 问候"), "provenance": json.dumps({"dataset": "luban0-v4", "verified_by": "local requester"})},
            {**fixtures.sample("请将signal7改为大写"), "source": "synthetic"},
            {**fixtures.sample("old-human-experiment"), "provenance": "{}"},
            {**fixtures.sample("unverified-teacher"), "source": "teacher", "provenance": json.dumps({
                "request_id": "chat", "messages": [{"role": "user", "content": "unverified-teacher"}]})},
            {**fixtures.sample("withdrawn"), "training_allowed": 0},
        ]
        self.fixture.put_materials(rows + research)
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET training_allowed=0 WHERE id='withdrawn'")
        before = self.runtime.memory.samples()
        result = self.auto.learning_contents()
        self.assertEqual({row["id"] for row in result["pending"]["items"]}, {row["id"] for row in rows})
        self.assertEqual(result["learned"]["count"], 0)
        self.assertEqual(self.runtime.memory.samples(), before)

    def test_product_source_does_not_filter_by_question_words(self):
        row = fixtures.sample("请将signal7改为大写")
        self.assertTrue(daily_learning_source(row))  # An explicit future lesson remains allowed.
        for key in ("dataset", "protocol_id", "benchmark_run", "route_only", "retention_generation_only"):
            with self.subTest(key=key):
                self.assertFalse(daily_learning_source({**row, "provenance": {
                    "verified_by": "local requester", key: "historical experiment"}}))

    def test_accepted_training_and_legacy_proof_are_still_pending(self):
        rows = self.fixture.materials("legacy")
        self.fixture.put_materials(rows)
        self.fixture.adopt_fixture(rows)
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"], {"count": 0, "items": []})
        self.assertEqual(result["pending"]["count"], 18)
        self.assertEqual(result["state"], "ready")
        # Historical scheduler receipts keep their existing meaning.
        self.assertEqual(len(knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))[1]), 18)

    def test_only_current_independent_validation_counts_and_training_does_not(self):
        rows = self.fixture.materials("checked")
        self.fixture.put_materials(rows)
        self.fixture.adopt_fixture(rows)
        body = self._proofs_without_reference()
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 2)
        self.assertEqual(result["pending"]["count"], 16)
        self.assertEqual({r["id"] for r in result["learned"]["items"]}, {r["id"] for r in rows if r["purpose"] == "validation"})
        body["commit"]["report"]["candidates"][0]["generation_checks"][0]["reference_used"] = True
        self._save(body)
        self.assertEqual(self.auto.learning_contents()["learned"]["count"], 1)
        self.runtime.active_core = "core-000000000002"
        with patch('luban.automatic.read_core_retention', return_value=None):
            self.assertEqual(self.auto.learning_contents()["learned"]["count"], 0)

    def test_missing_complete_proof_fails_closed_and_does_not_mutate_journal(self):
        rows = self.fixture.materials("damaged")
        self.fixture.put_materials(rows)
        self.fixture.adopt_fixture(rows)
        body = self._proofs_without_reference()
        body["commit"]["report"]["candidates"][0]["generation_checks"][0]["complete"] = False
        self._save(body)
        before = training_runs(self.runtime.memory, self.runtime.identity)
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 0)
        self.assertEqual(result["pending"]["count"], 18)
        self.assertEqual(result["state"], "waiting_validation")
        self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity), before)

    def test_contains_or_nonempty_checks_do_not_claim_mastery(self):
        rows = self.fixture.materials("weak-check")
        rows[-1]["provenance"] = json.dumps({"verified_by": "local requester", "generation_check": {"type": "contains", "expected": "verified"}})
        self.fixture.put_materials(rows)
        self.fixture.adopt_fixture(rows)
        self._proofs_without_reference()
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 1)
        self.assertEqual(result["pending"]["count"], 17)

    def test_deduplication_limit_and_busy_state_keep_counts_honest(self):
        rows = [fixtures.sample(index) for index in range(61)]
        self.fixture.put_materials(rows)
        self.fixture.put_materials([{**rows[0], "id": "same-content-other-id"},
                                   {**rows[1], "id": "route-only", "provenance": json.dumps({"route_only": True})}])
        self.runtime.state = "learning"
        result = self.auto.learning_contents()
        self.assertEqual(result["pending"]["count"], 61)
        self.assertEqual(len(result["pending"]["items"]), result["limit"])
        self.assertEqual(result["limit"], 50)
        self.assertEqual(result["state"], "waiting_model")
        self.assertTrue(all(set(row) == {"id", "prompt"} for row in result["pending"]["items"]))

    def test_actual_expert_cohort_is_pending_without_counting_protection_or_routes(self):
        rows = [fixtures.sample("expert-train"), fixtures.sample("expert-valid", "validation")]
        for row in rows:
            row["provenance"] = json.dumps({"dataset": "authorized-identity-curriculum"})
        self.fixture.put_materials(rows)
        records = {"train": [rows[0]], "valid": [rows[1]]}
        self.runtime.native_inference_precision = "mlx-q4"
        self.auto.native_evolution = lambda method, payload: {"records": records, "state": "ready", "error": None,
            "binding": {"engine": "mlx-q4", "core": self.runtime.active_core, "expert": self.runtime.branch.expert_id}}
        result = self.auto.learning_contents()
        self.assertEqual(result["pending"]["count"], 2)
        self.assertEqual(result["learned"]["count"], 0)
        self.assertEqual({row["id"] for row in result["pending"]["items"]}, {row["id"] for row in rows})
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET training_allowed=0 WHERE id=?", (rows[1]["id"],))
        result = self.auto.learning_contents()
        self.assertEqual(result["pending"]["count"], 0)  # No partial stale projection.
        self.assertEqual(result["state"], "waiting_validation")
        self.assertIn("撤权", result["error"])

    def test_native_source_mismatch_and_blocker_are_visible_without_false_mastery(self):
        self.runtime.native_inference_precision = "mlx-q4"
        response = {"records": None, "binding": None, "state": "waiting_validation", "error": "真实验证未通过"}
        self.auto.native_evolution = lambda method, payload: response
        self.assertEqual(self.auto.learning_contents()["error"], "真实验证未通过")
        response.update(binding={"engine": "mlx-q4", "core": "different-core", "expert": None}, state="ready", error=None)
        result = self.auto.learning_contents()
        self.assertEqual(result["state"], "waiting_validation")
        self.assertIn("来源已变化", result["error"])

    def test_legacy_generator_proofs_do_not_grant_mlx_mastery(self):
        rows = self.fixture.materials("old-engine")
        self.fixture.put_materials(rows)
        self.fixture.adopt_fixture(rows)
        self._proofs_without_reference()
        self.runtime.native_inference_precision = "mlx-q4"
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 0)
        self.assertEqual(result["pending"]["count"], 18)

    def _mlx_accepted_fixture(self):
        rows = [fixtures.sample("formal-train"), fixtures.sample("formal-valid", "validation")]
        for row in rows:
            row["provenance"] = json.dumps({"dataset": "authorized-identity-curriculum"})
        self.fixture.put_materials(rows)
        self.runtime.native_inference_precision = "mlx-q4"
        observation = {"id": rows[1]["id"], "answer": "verified", "passed": True, "complete": True,
                       "reference_used": False, "route_action": "neural_expert", "source": "candidate", "branch_enabled": True,
                       "tokens": [4, 5], "input_tokens": [1, 2, 3]}
        report = {"records": {"train": [rows[0]], "valid": [rows[1]], "protection": []},
                  "observations": {"candidate": [[observation], [dict(observation)]]},
                  "id": "abcabcabcabc", "base_identity": self.runtime.identity,
                  "active_expert": self.runtime.branch.expert_id, "inference_engine": "mlx-q4",
                  "promoted": True, "quality_checked": True, "gates": dict.fromkeys(QUALITY_GATES, True)}
        report["records_sha256"] = digest(report["records"])
        self._save_mlx_report(report)
        self.accepted_report.return_value = report
        return rows, report

    def _save_mlx_report(self, report):
        arguments = {"protocol": "native-mlx-expert-preservation-v1"}
        body = {"id": report["id"], "kind": "learning", "scope": "product", "arguments": arguments,
                "records": report["records"], "binding": {"identity": self.runtime.identity, "scope": "product",
                    "arguments": arguments, "records_sha256": digest(report["records"])}, "commit": {"report": report}}
        training_runs(self.runtime.memory, self.runtime.identity)
        with self.runtime.memory.connect() as db:
            db.execute("INSERT OR REPLACE INTO training_runs VALUES (?,?,?,?,?,?)",
                       (report["id"], self.runtime.identity, "learning", "committed", json.dumps(body), 1))
            db.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?)",
                       ("training:" + report["id"], "learning", json.dumps(report), 1))

    def test_current_mlx_valid_mastery_survives_empty_next_cohort_without_counting_train(self):
        rows, _ = self._mlx_accepted_fixture()
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["items"], [{"id": rows[1]["id"], "prompt": rows[1]["prompt"]}])
        self.assertEqual(result["pending"]["items"], [{"id": rows[0]["id"], "prompt": rows[0]["prompt"]}])
        self.assertEqual(result["state"], "ready")

    def test_mlx_mastery_rechecks_closed_book_full_generation_and_both_passes(self):
        _, original = self._mlx_accepted_fixture()
        for field, value in (("complete", False), ("passed", False), ("reference_used", True),
                             ("route_action", "local_core"), ("answer", "wrong"), ("tokens", []),
                             ("tokens", [True]), ("input_tokens", [7])):
            with self.subTest(field=field, value=value):
                report = copy.deepcopy(original)
                report["observations"]["candidate"][1][0][field] = value
                self._save_mlx_report(report)
                self.accepted_report.return_value = report
                result = self.auto.learning_contents()
                self.assertEqual(result["learned"]["count"], 0)
                self.assertEqual(result["state"], "waiting_validation")

    def test_mlx_mastery_accepts_actual_core_route_with_matching_disabled_branch(self):
        _, report = self._mlx_accepted_fixture()
        for part in report["observations"]["candidate"]:
            part[0].update(route_action="local_core", source="core", branch_enabled=False)
        self._save_mlx_report(report)
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 1)
        self.assertEqual(result["pending"]["count"], 1)

    def test_mlx_status_counts_committed_update_but_only_valid_mastery(self):
        self._mlx_accepted_fixture()
        self.auto.native_evolution = lambda method, payload: {"state": "waiting_materials"}
        result = self.auto.status()
        self.assertEqual((result["learning_count"], result["knowledge_learned"], result["knowledge_pending"]), (1, 1, 1))
        with self.runtime.memory.connect() as db:
            db.execute("DELETE FROM events WHERE id='training:abcabcabcabc'")
        result = self.auto.status()
        self.assertEqual((result["learning_count"], result["knowledge_learned"]), (0, 0))
        self.assertEqual(result["state"], "waiting_validation")

    def test_mlx_prepared_report_does_not_grant_mastery_before_commit(self):
        self._mlx_accepted_fixture()
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE training_runs SET state='committing' WHERE id='abcabcabcabc'")
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 0)
        self.assertEqual(result["state"], "waiting_validation")

    def test_weak_check_and_training_overlap_do_not_grant_independent_mastery(self):
        rows, report = self._mlx_accepted_fixture()
        valid = report["records"]["valid"][0]
        valid["provenance"] = json.dumps({"dataset": "authorized-identity-curriculum",
                                        "generation_check": {"type": "contains", "expected": "verified"}})
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET provenance=? WHERE id=?", (valid["provenance"], valid["id"]))
        report["records_sha256"] = digest(report["records"])
        self._save_mlx_report(report)
        result = self.auto.learning_contents()
        self.assertEqual((result["learned"]["count"], result["pending"]["count"], result["state"]), (0, 2, "ready"))
        valid["prompt"] = rows[0]["prompt"]
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET prompt=? WHERE id=?", (valid["prompt"], valid["id"]))
        report["records_sha256"] = digest(report["records"])
        self._save_mlx_report(report)
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 0)
        self.assertIn("训练资料重合", result["error"])

    def test_mlx_pending_without_native_bridge_stays_owned_by_mlx(self):
        self._mlx_accepted_fixture()
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE training_runs SET state='paused' WHERE id='abcabcabcabc'")
        before = training_runs(self.runtime.memory, self.runtime.identity)
        self.auto.last_activity = 0
        with patch.object(self.auto, "_launch") as launch, patch.object(self.auto, "_check_resume_rights") as rights:
            self.auto.tick()
        launch.assert_not_called()
        rights.assert_not_called()
        self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity), before)
        self.assertEqual(self.auto.state, "waiting_validation")

    def test_mlx_accepted_material_revocation_removes_mastery(self):
        rows, _ = self._mlx_accepted_fixture()
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET training_allowed=0 WHERE id=?", (rows[1]["id"],))
        result = self.auto.learning_contents()
        self.assertEqual(result["learned"]["count"], 0)
        self.assertIn("撤权", result["error"])

    def test_legacy_knowledge_projection_does_not_parse_native_mlx_journal(self):
        rows = self.fixture.materials("still-legacy")
        self.fixture.put_materials(rows)
        self.fixture.adopt_fixture(rows)
        self._proofs_without_reference()
        runs = training_runs(self.runtime.memory, self.runtime.identity)
        native = {"kind": "learning", "state": "committed", "scope": "product",
                  "arguments": {"protocol": "native-mlx-expert-preservation-v1"},
                  "binding": {"identity": self.runtime.identity}, "records": {"native": "opaque"}}
        self.assertEqual(knowledge_keys(self.runtime, runs + [native], mastery_only=True),
                         knowledge_keys(self.runtime, runs, mastery_only=True))

    def test_real_tiny_partial_commit_uses_existing_frozen_mastery_proofs(self):
        fixture = partial.PartialEvolutionTests()
        self.addCleanup(fixture.doCleanups)
        _, runtime, _ = fixture.fixture()
        report = consolidate(runtime, **partial.ARGS)
        runs = training_runs(runtime.memory, runtime.identity)
        runs[0]["product_accepted"] = True  # Isolated CPU fixture only; not product qualification.
        with runtime.memory.connect() as db:
            db.execute("UPDATE training_runs SET body=? WHERE id=?", (json.dumps(runs[0]), runs[0]["id"]))
        runtime.state = "ready"
        auto = AutomaticLearning(runtime, self.fixture.settings, threading.Lock(), lambda: False)
        samples = runtime.memory.samples()
        with runtime.memory.connect() as db:
            for row in samples:
                value = json.loads(row["provenance"])
                value["verified_by"] = "local requester"
                db.execute("UPDATE samples SET provenance=? WHERE id=?", (json.dumps(value), row["id"]))
        result = auto.learning_contents()
        expected = set(report["learning_progress"]["mastered_ids"])
        self.assertGreater(len(expected), 0)
        self.assertEqual({row["id"] for row in result["learned"]["items"]}, expected)
        self.assertEqual(result["learned"]["count"] + result["pending"]["count"], len(runtime.memory.samples()))


if __name__ == "__main__":
    unittest.main()
