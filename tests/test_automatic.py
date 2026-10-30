"""Offline scheduler/receipt/idle/close contracts; no model or teacher HTTP."""
import copy
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from luban.automatic import (AutomaticLearning, POLICY, digest, knowledge_keys, record_keys,
                             mlx_committed_report, mlx_learning_projection)
from luban.generation_checks import evaluate_generation, validate_generation_check
from luban.memory import Memory
from luban.retention import GROUP_KEYS, protected_validation, save_core_retention
from luban.settings import Settings
from luban.training_checkpoint import TrainingPaused, TrainingSession, training_runs


def sample(index, purpose="training"):
    return {"id": str(index), "prompt": str(index), "answer": "verified", "source": "human",
            "purpose": purpose, "approved": 1, "training_allowed": 1,
            "provenance": json.dumps({"verified_by": "local requester"})}


class AutomaticTests(unittest.TestCase):
    def test_learning_switch_only_stops_automatic_learning_and_keeps_manual_start(self):
        self.settings.update({"automatic": {"learning_enabled": False}})
        self.auto.on_idle = Mock(return_value=False)
        records = {"train": [sample("switch")], "valid": [sample("switch-v", "validation")], "protection": []}
        def manifest(kind):
            if kind == "learning":
                return {"steps": 64}, records
            raise ValueError("waiting for router materials")
        with patch.object(self.auto, "_manifest", side_effect=manifest) as prepare, patch.object(self.auto, "_launch", return_value="manual-job") as launch:
            self.auto.tick()
            self.assertEqual([call.args[0] for call in prepare.call_args_list], ["router"])
            self.auto.on_idle.assert_called_once()
            launch.assert_not_called()
            self.assertEqual(self.auto.start_manual("learning"), {"job_id": "manual-job"})
            self.assertEqual(launch.call_args.args[0], "learning")

    def test_learning_switch_does_not_automatically_resume_paused_learning(self):
        self.settings.update({"automatic": {"learning_enabled": False}})
        pending = {"id": "paused-learning", "kind": "learning"}
        self.auto.native_evolution = Mock()
        with patch.object(self.auto, "_pending", return_value=pending):
            self.auto.tick()
        self.auto.native_evolution.assert_not_called()
        self.assertEqual(self.auto.state, "paused")

    def mlx_projection_fixture(self):
        from luban.mlx_learning import NET_LEARNING_POLICY, NET_QUALITY_GATES
        records = {"train": [sample("new-train")],
                   "valid": [sample("new-ok", "validation"), sample("new-pending", "validation")],
                   "protection": [sample("old-ok", "validation"), sample("old-forgotten", "validation")]}
        frozen = sum(records.values(), [])
        observed = []
        for row in frozen:
            passed = row["id"] in {"new-train", "new-ok", "old-ok"}
            observed.append(dict(id=row["id"], answer="verified" if passed else "unfinished", passed=passed,
                complete=passed, reference_used=False, tokens=[1 if passed else 2], input_tokens=[42],
                route_action="neural_expert", source="candidate", branch_enabled=True))
        debt = sample("prior-forgotten", "validation")
        unreviewed = sample("not-reviewed-this-round", "validation")
        report = dict(id="012345abcdef", records=records, records_sha256=digest(records),
            base_identity=self.runtime.identity, inference_engine="mlx-q4", promoted=True, quality_checked=True,
            learning_policy=NET_LEARNING_POLICY, gates=dict.fromkeys(NET_QUALITY_GATES, True),
            observations={"candidate": [observed, copy.deepcopy(observed)]},
            knowledge_pool=[records["valid"][0], records["protection"][0], unreviewed],
            relearning_records=[records["protection"][1], debt])
        arguments = dict(protocol="native-mlx-expert-preservation-v1", learning_policy=NET_LEARNING_POLICY)
        run = dict(id=report["id"], kind="learning", state="committed", scope="product", arguments=arguments,
            binding=dict(scope="product", identity=self.runtime.identity, arguments=arguments,
                         records_sha256=digest(records)), records=records, commit={"report": report})
        self.save_mlx_event(report)
        return report, run, frozen + [debt, unreviewed]

    def save_mlx_event(self, report):
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            db.execute("INSERT OR REPLACE INTO events(id,kind,body,created) VALUES (?,?,?,?)",
                       ("training:" + report["id"], "learning", json.dumps(report), time.time()))

    def test_mlx_net_projection_counts_only_actual_current_closed_book_mastery(self):
        report, run, samples = self.mlx_projection_fixture()
        with patch("luban.mlx_learning.accepted_report", return_value=report):
            rows, mastered = mlx_learning_projection(self.runtime, samples, [run])
        self.assertEqual(mastered, record_keys([samples[1], samples[3]]))
        pending = {row["id"] for key, row in rows.items() if key not in mastered}
        self.assertEqual(pending, {"new-train", "new-pending", "old-forgotten", "prior-forgotten"})
        self.assertNotIn("not-reviewed-this-round", {row["id"] for row in rows.values()})

    def test_factory_mlx_keeps_mastery_but_not_personal_learning_count(self):
        report, run, samples = self.mlx_projection_fixture()
        report['active_expert'] = 'learned-factory'
        run['factory_baseline'] = True
        run['commit']['event_kind'] = 'learning'
        self.save_mlx_event(report)
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE events SET kind='factory:learning'")
        self.runtime.native_inference_precision = 'mlx-q4'
        self.runtime.branch.expert_id = report['active_expert']
        self.auto.native_evolution = lambda method, payload: {}
        with patch('luban.automatic.training_runs', return_value=[run]), \
                patch('luban.automatic.knowledge_keys', return_value=(set(), set(), None)), \
                patch.object(self.runtime.memory, 'samples', return_value=samples), \
                patch('luban.mlx_learning.accepted_report', return_value=report):
            current = self.auto.status()
            self.assertEqual((current['learning_count'], current['knowledge_learned']), (0, 2))
            self.assertNotEqual(current['state'], 'waiting_validation')
            # Historical shipped experts also do not become personal updates.
            self.runtime.branch.expert_id = 'learned-next'
            self.assertEqual(self.auto.status()['learning_count'], 0)
            # A new ordinary adoption still counts after the baseline.
            self.runtime.branch.expert_id = report['active_expert']
            run.pop('factory_baseline')
            self.save_mlx_event(report)
            self.assertEqual(self.auto.status()['learning_count'], 1)

    def test_factory_mlx_marker_cannot_replace_event_or_quality_validation(self):
        report, run, _ = self.mlx_projection_fixture()
        run['commit']['event_kind'] = 'learning'
        for marker, event_kind in ((None, 'factory:learning'), (True, 'learning'), (False, 'factory:learning')):
            run.pop('factory_baseline', None)
            if marker is not None:
                run['factory_baseline'] = marker
            with self.runtime.memory.connect() as db:
                db.execute('UPDATE events SET kind=?', (event_kind,))
            with self.subTest(marker=marker, event=event_kind), self.assertRaises(ValueError):
                mlx_committed_report(self.runtime, report, [run])
        run['factory_baseline'] = True
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE events SET kind='factory:learning'")
        report['gates']['mlx_net_gain'] = False
        with self.assertRaisesRegex(ValueError, '质量报告'):
            mlx_committed_report(self.runtime, report, [run])

    def factory_mlx_projection_fixture(self):
        report, run, _ = self.mlx_projection_fixture()
        report['relearning_records'] = []
        report['active_expert'] = 'learned-factory'
        run.update(factory_baseline=True)
        run['commit']['event_kind'] = 'learning'
        self.save_mlx_event(report)
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE events SET kind='factory:learning' WHERE id=?", ('training:' + run['id'],))
        return report, run

    def test_factory_materials_need_no_personal_queue_and_create_no_personal_items(self):
        report, run = self.factory_mlx_projection_fixture()
        self.runtime.native_inference_precision = 'mlx-q4'
        self.runtime.branch.expert_id = report['active_expert']
        self.auto.native_evolution = lambda method, payload: {}
        with patch('luban.automatic.training_runs', return_value=[run]), \
                patch('luban.automatic.knowledge_keys', return_value=(set(), set(), None)), \
                patch('luban.mlx_learning.accepted_report', return_value=report):
            self.assertEqual(mlx_learning_projection(self.runtime, [], [run]), ({}, set()))
            state = self.auto.status()
            self.assertEqual((state['learning_count'], state['knowledge_learned'], state['knowledge_pending']), (0, 0, 0))
            self.assertNotEqual(state['state'], 'waiting_validation')
        # Inheritance still runs both original closed-book observation checks.
        report['observations']['candidate'][1][1]['tokens'] = [99]
        self.save_mlx_event(report)
        with self.runtime.memory.connect() as db:
            db.execute("UPDATE events SET kind='factory:learning' WHERE id=?", ('training:' + run['id'],))
        with patch('luban.mlx_learning.accepted_report', return_value=report), self.assertRaisesRegex(ValueError, '两遍'):
            mlx_learning_projection(self.runtime, [], [run])

    def personal_after_factory_fixture(self):
        baseline, factory = self.factory_mlx_projection_fixture()
        report, run = copy.deepcopy(baseline), copy.deepcopy(factory)
        report['id'] = run['id'] = 'abcdef012345'
        report['active_expert'] = 'learned-personal'
        run.pop('factory_baseline')
        records = dict(train=[sample('personal-train')], valid=[sample('personal-valid', 'validation')],
                       protection=[copy.deepcopy(baseline['records']['valid'][0])])
        report['records'] = run['records'] = records
        report['records_sha256'] = run['binding']['records_sha256'] = digest(records)
        observations = [dict(id=row['id'], answer='verified', passed=True, complete=True, reference_used=False,
                            tokens=[1], input_tokens=[42], route_action='neural_expert', source='candidate', branch_enabled=True)
                        for row in sum(records.values(), [])]
        report['observations']['candidate'] = [observations, copy.deepcopy(observations)]
        report['knowledge_pool'] = records['valid'] + records['protection']
        run['commit']['report'] = report
        self.save_mlx_event(report)
        return report, [factory, run], records['train'] + records['valid']

    def test_personal_learning_inherits_factory_protection_without_showing_it(self):
        report, runs, samples = self.personal_after_factory_fixture()
        with patch('luban.mlx_learning.accepted_report', return_value=report):
            rows, learned = mlx_learning_projection(self.runtime, samples, runs)
        self.assertEqual({row['id'] for row in rows.values()}, {'personal-train', 'personal-valid'})
        self.assertEqual(learned, record_keys([samples[1]]))
        self.assertIn('new-ok', {row['id'] for row in report['knowledge_pool']})

    def test_inherited_protection_never_overrides_changed_revoked_or_missing_personal_material(self):
        for mutation in ('changed', 'revoked', 'missing-personal', 'changed-factory-record', 'missing-factory', 'bad-purpose'):
            report, runs, samples = self.personal_after_factory_fixture()
            if mutation in {'changed', 'revoked'}:
                actual = copy.deepcopy(report['records']['protection'][0])
                actual.update({'answer': 'changed'} if mutation == 'changed' else {'training_allowed': False})
                samples.append(actual)
            elif mutation == 'missing-personal':
                samples.pop(0)
            elif mutation == 'missing-factory':
                runs.pop(0)
            else:
                report['records']['protection'][0].update({'prompt': 'different'} if mutation == 'changed-factory-record'
                                                       else {'purpose': 'training'})
                report['records_sha256'] = runs[-1]['binding']['records_sha256'] = digest(report['records'])
                self.save_mlx_event(report)
            with self.subTest(mutation=mutation), patch('luban.mlx_learning.accepted_report', return_value=report):
                with self.assertRaisesRegex(ValueError, '资料已变更或撤权'):
                    mlx_learning_projection(self.runtime, samples, runs)

    def test_mlx_net_projection_rejects_forged_results_or_changed_authorization(self):
        for mutation in ("passed", "different-pass", "revoked"):
            report, run, samples = self.mlx_projection_fixture()
            if mutation == "passed":
                for part in report["observations"]["candidate"]:
                    part[2]["passed"] = True
            elif mutation == "different-pass":
                report["observations"]["candidate"][1][1]["tokens"] = [9]
            else:
                samples = copy.deepcopy(samples)
                samples[-2]["training_allowed"] = False
            self.save_mlx_event(report)
            with self.subTest(mutation=mutation), patch("luban.mlx_learning.accepted_report", return_value=report):
                with self.assertRaises(ValueError):
                    mlx_learning_projection(self.runtime, samples, [run])

    def test_mlx_net_receipt_requires_its_frozen_policy_and_all_real_gates(self):
        for mutation in ("missing-gate", "wrong-policy", "missing-event"):
            report, run, _ = self.mlx_projection_fixture()
            if mutation == "missing-gate":
                report["gates"]["mlx_net_gain"] = False
                self.save_mlx_event(report)
            elif mutation == "wrong-policy":
                run["arguments"].pop("learning_policy")
            else:
                with self.runtime.memory.lock, self.runtime.memory.connect() as db:
                    db.execute("DELETE FROM events WHERE id=?", ("training:" + report["id"],))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                mlx_committed_report(self.runtime, report, [run])

    def test_mlx_legacy_projection_does_not_inherit_partial_mastery_policy(self):
        from luban.mlx_learning import QUALITY_GATES
        report, run, samples = self.mlx_projection_fixture()
        report.pop("learning_policy")
        report["gates"] = dict.fromkeys(QUALITY_GATES, True)
        run["arguments"].pop("learning_policy")
        self.save_mlx_event(report)
        with patch("luban.mlx_learning.accepted_report", return_value=report):
            with self.assertRaisesRegex(ValueError, "完整闭卷"):
                mlx_learning_projection(self.runtime, samples, [run])

    def test_manual_evolution_has_fixed_budget_and_requires_ready_model(self):
        with patch.object(self.auto, '_launch') as launch:
            for steps in (352, 704):
                with self.assertRaisesRegex(ValueError, '自定义预算'):
                    self.auto.start_manual('evolution', steps)
            with self.assertRaisesRegex(ValueError, '模型尚未就绪'):
                self.auto.start_manual('evolution')
            launch.assert_not_called()

    def test_material_readiness_does_not_hide_idle_scheduler_failure(self):
        self.runtime.state = "ready"
        self.auto.state, self.auto.reason = "waiting_validation", "idle scheduler failed"
        self.auto.native_evolution = lambda method, payload: {
            "state": "waiting_materials", "reason": "materials ready", "experience": {"current": 16, "target": 16}}
        result = self.auto.status()
        self.assertEqual((result["state"], result["reason"]), ("waiting_validation", "idle scheduler failed"))
        self.assertEqual(result["experience"]["current"], 16)
        self.auto.state = "waiting_materials"
        self.assertEqual(self.auto.status()["reason"], "materials ready")

    def test_ready_native_material_starts_before_old_teacher_reviews(self):
        self.runtime.state = "ready"
        self.auto.last_activity = time.monotonic() - 61
        self.auto.teacher_practice = SimpleNamespace(pending=Mock(return_value=True), pause=Mock())
        self.auto.native_evolution = Mock(return_value={"state": "running", "job_id": "real-native-job"})
        self.auto.tick()
        self.auto.native_evolution.assert_called_once()
        self.assertEqual(self.auto.native_evolution.call_args.args[0], "tick")
        self.auto.teacher_practice.pending.assert_not_called()
        self.assertEqual(self.auto.state, "running")

    def test_insufficient_native_material_keeps_other_idle_work_and_checks_once(self):
        self.runtime.state = "ready"
        self.auto.last_activity = time.monotonic() - 61
        self.auto.teacher_practice = SimpleNamespace(pending=Mock(return_value=False), pause=Mock())
        self.auto.on_idle = Mock(return_value=False)
        self.auto.native_evolution = Mock(return_value={"state": "waiting_materials", "reason": "short 13"})
        with patch.object(self.auto, "_manifest", side_effect=ValueError("router not ready")):
            self.auto.tick()
        self.auto.native_evolution.assert_called_once()
        self.auto.teacher_practice.pending.assert_called_once()
        self.auto.on_idle.assert_called_once()
        self.assertEqual((self.auto.state, self.auto.reason), ("waiting_materials", "short 13"))

    def test_pending_decisions_include_unreviewed_actual_questions_in_current_base(self):
        for identity, prompt in ((self.runtime.identity, "待核验的问题"), ("old-base", "别的基座")):
            self.runtime.memory.event("decision", {"base_identity": identity,
                "messages": [{"role": "user", "content": prompt}], "answer": "未经核验的输出"})
        with patch("luban.automatic.consolidation_samples", return_value={"new_training": []}), \
                patch.object(self.auto, "_manifest", side_effect=ValueError("waiting")):
            result = self.auto.status()
        self.assertEqual(result["decision_pending"], 1)
        self.assertEqual(result["decision_learned"], 0)

    def test_research_queue_is_not_claimed_or_changed_by_product_monitor(self):
        records = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        session = TrainingSession(self.runtime, "learning", {"steps": 64}, records)
        session.body["scope"] = "research"
        session._write("queued")
        before = training_runs(self.runtime.memory, self.runtime.identity)[0]
        with patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            launch.assert_not_called()
        self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity)[0], before)

    def test_research_material_cannot_grant_deleted_material_resume_rights(self):
        records = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        unaccepted = {"kind": "learning", "state": "committed", "scope": "research", "records": records}
        with self.assertRaisesRegex(ValueError, "删除"):
            self.auto._check_resume_rights({"kind": "learning", "records": records}, [unaccepted])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.runtime = SimpleNamespace(memory=Memory(root), store=root / "modules", identity="automatic-base",
            active_core=None, branch=SimpleNamespace(expert_id=None), device="cpu", dtype=torch.float32,
            job=None, state="ready", training_session=None)
        self.settings = Settings(root / "settings")
        self.busy = False
        self.variant = 0
        self.auto = AutomaticLearning(self.runtime, self.settings, threading.Lock(), lambda: self.busy)
        self.auto.last_activity = time.monotonic() - 100
        self.addCleanup(self.auto.stop)

    def freeze(self, kind):
        if kind == "router":
            raise ValueError("missing independent route families")
        if kind == "learning":
            raise ValueError("not enough verified knowledge")
        return {"steps": 450, "learning_rate": 1e-5, "distillation_weight": 5,
                "preserve_replay_subspace": True, "partial_policy": "closed-book-growth-v1"}, {"groups": {"new_training": [sample(self.variant)]}, "protection": [], "retention_regressions": []}

    def test_evolution_default_manual_defer_survives_restart_and_no_paid_calls(self):
        with patch.object(self.auto, "_manifest", side_effect=self.freeze), patch("luban.automatic.consolidation_samples", return_value={key: [sample(i) for i in range(16)] if key == "new_training" else [] for key in GROUP_KEYS}):
            self.auto.tick(datetime(2026, 10, 9, 23, 0))
            self.assertIsNone(self.auto.worker)
            first = self.auto.status()
            self.assertTrue(first["reminder_due"])
            self.auto.defer(first["pending_signature"])
            second = AutomaticLearning(self.runtime, self.settings, threading.Lock(), lambda: False)
            with patch.object(second, "_manifest", side_effect=self.freeze):
                self.assertFalse(second.status()["reminder_due"])
            self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity), [])

    def evolution_materials(self, count=350):
        rows = []
        for index in range(count):
            row = sample(f"experience-{index:04d}")
            row["provenance"] = json.dumps({**json.loads(row["provenance"]), "evolution_role": "new"})
            rows.append(row)
        for role, purpose in (("new", "validation"), ("replay", "training"), ("replay", "validation")):
            row = sample(f"{role}-{purpose}", purpose)
            row["provenance"] = json.dumps({**json.loads(row["provenance"]), "evolution_role": role})
            rows.append(row)
        self.put_materials(rows)

    def test_evolution_350_earns_bounded_450_manifest_and_overflow_is_not_consumed(self):
        self.evolution_materials(360)
        with patch("luban.automatic.core_status", return_value={"ready": True}):
            args, records = self.auto._manifest("evolution")
        self.assertEqual(args, {"steps": 450, "learning_rate": 1e-5, "distillation_weight": 5,
            "preserve_replay_subspace": True, "partial_policy": "closed-book-growth-v1"})
        self.assertEqual(len(records["experience"]), 350)
        self.assertEqual(len(records["groups"]["new_training"]), 57)
        session = TrainingSession(self.runtime, "evolution", args, records)
        signature = self.auto._signature("evolution", args, records)
        session.body["automatic_signature"] = signature
        session.rejected("isolated quality refusal")
        _, pool, unspent = self.auto._evolution_materials()
        self.assertEqual(pool, records["experience"])
        self.assertEqual(unspent, 11)  # 360 TRAIN + 1 VALID, exactly 350 earned contents.
        with patch("luban.automatic.core_status", return_value={"ready": True}), \
                patch.object(self.auto, "_launch") as launch:
            with self.assertRaisesRegex(ValueError, "已经处理"):
                self.auto.start_manual("evolution")
            launch.assert_not_called()

    def test_evolution_349_waits_and_research_model_never_launches(self):
        self.evolution_materials(348)  # one independent NEW VALID also counts as content.
        with patch("luban.automatic.core_status", return_value={"ready": True}):
            self.assertEqual(self.auto._evolution_materials()[2], 349)
            with self.assertRaisesRegex(ValueError, "等待350条"):
                self.auto._manifest("evolution")
        self.runtime.research = True
        with self.assertRaisesRegex(ValueError, "研究模型"):
            self.auto._manifest("evolution")

    def test_evolution_excludes_experiment_new_but_preserves_replay(self):
        self.evolution_materials(2)
        experiments = []
        for role, purpose in (("new", "training"), ("new", "validation"),
                              ("replay", "training"), ("replay", "validation")):
            row = sample("experiment-" + role + "-" + purpose, purpose)
            row["provenance"] = json.dumps({"dataset": "historical-experiment", "evolution_role": role})
            experiments.append(row)
        self.put_materials(experiments)
        groups, pool, unspent = self.auto._evolution_materials()
        self.assertIsNone(pool)
        self.assertEqual(unspent, 3)
        self.assertEqual(len(groups["new_training"]), 2)
        self.assertEqual(len(groups["new_validation"]), 1)
        self.assertIn("experiment-replay-training", {row["id"] for row in groups["replay_training"]})
        self.assertIn("experiment-replay-validation", {row["id"] for row in groups["replay_validation"]})
        self.assertIn("experiment-replay-validation", {
            row["id"] for row in protected_validation(self.runtime, include_learning=True)})

    def test_evolution_same_earned_pool_carries_unmastered_material_after_adoption(self):
        self.evolution_materials()
        with patch("luban.automatic.core_status", return_value={"ready": True}):
            args, records = self.auto._manifest("evolution")
        session = TrainingSession(self.runtime, "evolution", args, records)
        session.rejected("isolated previous round ledger")
        first = records["groups"]["new_training"][0]
        self.current_core_fixture([first])
        groups, pool, unspent = self.auto._evolution_materials()
        self.assertEqual(pool, records["experience"])
        self.assertNotIn(first["id"], {row["id"] for row in groups["new_training"]})
        self.assertEqual(len(groups["new_training"]), 57)
        self.assertEqual(unspent, 1)

    def test_schedule_waits_for_time_and_idle_and_runs_once_per_day(self):
        self.settings.update({"automatic": {"mode": "auto", "scheduled_time": "21:15"}})
        launched = []
        def launch(kind, args, records, **kwargs):
            launched.append(kwargs)
            session = TrainingSession(self.runtime, kind, args, records)
            session.body.update(automatic_signature=kwargs["signature"], scheduled_occurrence=kwargs["scheduled_occurrence"])
            session.rejected("isolated schedule receipt")
        with patch.object(self.auto, "_manifest", side_effect=self.freeze), patch.object(self.auto, "_launch", side_effect=launch):
            self.auto.tick(datetime(2026, 10, 9, 21, 14))
            self.assertEqual(launched, [])
            self.busy = True
            self.auto.tick(datetime(2026, 10, 9, 21, 15))
            self.assertEqual(launched, [])
            self.busy = False
            self.auto.tick(datetime(2026, 10, 9, 21, 16))
            self.auto.tick(datetime(2026, 10, 9, 22, 16))
            self.assertEqual(len(launched), 1)
            self.auto.tick(datetime(2026, 10, 10, 21, 16))
            self.assertEqual(len(launched), 2)  # Pending targets get one bounded new round next day.
            self.variant = 1
            self.auto.tick(datetime(2026, 10, 10, 21, 17))
            self.assertEqual(len(launched), 2)

    def test_earned_experience_cannot_be_earned_again_by_id_or_source_metadata_changes(self):
        self.evolution_materials()
        with patch("luban.automatic.core_status", return_value={"ready": True}):
            args, records = self.auto._manifest("evolution")
        session = TrainingSession(self.runtime, "evolution", args, records)
        session.rejected("isolated unchanged content")
        rows = self.runtime.memory.samples()
        for row in rows:
            row["id"] = "new-id-" + row["id"]
            value = json.loads(row["provenance"])
            value["reviewer_note"] = "metadata revised"
            value["generation_check"] = {"type": "exact", "expected": row["answer"]}
            row["provenance"] = json.dumps(value)
        with patch.object(self.runtime.memory, "samples", return_value=rows):
            _, pool, unspent = self.auto._evolution_materials()
        self.assertIsNone(pool)  # Old allowance cannot be rebound to changed evidence.
        self.assertEqual(unspent, 1)  # Only the original overflow remains unspent.

    def test_actual_product350_manifest_is_scheduled_once_per_day_and_retries_pending(self):
        self.evolution_materials()
        self.settings.update({"automatic": {"mode": "auto", "scheduled_time": "21:15"}})
        original, launched = self.auto._manifest, []
        def manifest(kind):
            if kind != "evolution":
                raise ValueError("isolated preceding stages not eligible")
            return original(kind)
        def launch(kind, args, records, **kwargs):
            session = TrainingSession(self.runtime, kind, args, records)
            session.body.update(automatic_signature=kwargs["signature"], scheduled_occurrence=kwargs["scheduled_occurrence"])
            session.rejected("pending fixed-budget round")
            launched.append(session.body)
        with patch.object(self.auto, "_manifest", side_effect=manifest), \
                patch("luban.automatic.core_status", return_value={"ready": True}), \
                patch.object(self.auto, "_launch", side_effect=launch):
            for day in (15, 15, 16, 16):
                self.auto.tick(datetime(2026, 10, day, 21, 16))
        self.assertEqual(len(launched), 2)
        self.assertEqual(launched[0]["records"]["experience"], launched[1]["records"]["experience"])
        self.assertEqual([row["arguments"]["steps"] for row in launched], [450, 450])

    def test_paused_manifest_resumes_before_reading_changed_materials(self):
        records = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        session = TrainingSession(self.runtime, "learning", {"steps": 64}, records)
        session.save({"step": 8}, {"step": 8})
        session.pause()
        self.runtime.memory.samples = lambda: records["train"] + records["valid"]
        with patch.object(self.auto, "_manifest", side_effect=AssertionError("fresh material must not replace manifest")), patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            launch.assert_called_once()
            self.assertEqual(launch.call_args.args[2], records)
            self.assertEqual(launch.call_args.kwargs["run"]["id"], session.id)

    def test_resume_rejects_experiment_new_material_even_when_still_present(self):
        historical = sample("historical")
        historical["provenance"] = json.dumps({"dataset": "historical-experiment", "evolution_role": "new"})
        self.put_materials([historical])
        for kind in ("learning", "evolution"):
            records = ({"train": [historical], "valid": [], "protection": []} if kind == "learning" else
                       {"groups": {key: [historical] if key == "new_training" else [] for key in GROUP_KEYS},
                        "protection": [], "retention_regressions": []})
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.auto._check_resume_rights({"kind": kind, "records": records}, [])

    def test_resume_preserves_experiment_replay_protection(self):
        historical = sample("historical-replay")
        historical["provenance"] = json.dumps({"dataset": "historical-experiment", "evolution_role": "replay"})
        self.put_materials([historical])
        records = {"groups": {key: [historical] if key == "replay_training" else [] for key in GROUP_KEYS},
                   "protection": [], "retention_regressions": []}
        self.auto._check_resume_rights({"kind": "evolution", "records": records}, [])

    def test_deleted_unaccepted_material_rejects_once_and_keeps_checkpoint(self):
        records = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        session = TrainingSession(self.runtime, "learning", {"steps": 64}, records)
        session.save({"step": 8}, {"step": 8})
        session.pause()
        with patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            launch.assert_not_called()
        row = training_runs(self.runtime.memory, self.runtime.identity)[0]
        self.assertEqual(row["state"], "rejected")
        self.assertTrue((session.directory / row["checkpoint"]["name"]).exists())

    def test_failure_same_material_not_retried_and_changed_answer_gets_new_signature(self):
        records = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        args = {"steps": 64}
        signature = self.auto._signature("learning", args, records)
        session = TrainingSession(self.runtime, "learning", args, records)
        session.body["automatic_signature"] = signature
        session.rejected("old capability failed")
        def freeze(kind):
            if kind != "learning":
                raise ValueError("waiting")
            return args, records
        with patch.object(self.auto, "_manifest", side_effect=freeze), patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            launch.assert_not_called()
            records["train"][0]["answer"] = "new verified answer"
            self.auto.tick()
            launch.assert_called_once()

    def test_close_pause_waits_until_accepted_weights_restored_then_releases_lease(self):
        entered = threading.Event()
        restored = threading.Event()
        self.runtime.accepted_weight = 1
        def learn(steps):
            entered.set()
            self.runtime.accepted_weight = 9
            while not self.runtime.training_session.pause_requested:
                time.sleep(.001)
            self.runtime.accepted_weight = 1
            restored.set()
            raise TrainingPaused()
        self.runtime.learn = learn
        self.auto._launch("learning", {"steps": 64}, {"train": [], "valid": [], "protection": []})
        self.assertTrue(entered.wait(2))
        result = self.auto.pause(closing=True)
        self.assertTrue(result["paused"] and restored.is_set())
        self.assertEqual(self.runtime.accepted_weight, 1)
        row = training_runs(self.runtime.memory, self.runtime.identity)[0]
        self.assertEqual(row["state"], "paused")
        self.assertIsNone(row["owner"])
        self.assertIsNone(self.runtime.training_session)
        self.auto.tick()
        self.assertEqual(len(training_runs(self.runtime.memory, self.runtime.identity)), 1)

    def test_only_committed_receipts_count_and_edited_material_not_learned(self):
        rows = [sample(1), sample(2, "validation")]
        self.runtime.memory.samples = lambda: rows
        records = {"train": [rows[0]], "valid": [rows[1]], "protection": []}
        session = TrainingSession(self.runtime, "learning", {"steps": 64}, records)
        session.pause()
        with patch("luban.automatic.consolidation_samples", side_effect=ValueError("waiting")):
            self.assertEqual(self.auto.status()["learning_count"], 0)
            session.prepare_commit({"promoted": True}, {}, [], event_kind="learning")
            session.finish_commit()
            session.finish_commit()
            state = self.auto.status()
            self.assertEqual((state["learning_count"], state["knowledge_learned"], state["knowledge_pending"]), (1, 2, 0))
            rows[0] = {**rows[0], "answer": "different"}
            state = self.auto.status()
            self.assertEqual((state["knowledge_learned"], state["knowledge_pending"]), (1, 1))

    def test_factory_legacy_receipts_keep_knowledge_without_personal_counts(self):
        rows = [sample('factory'), sample('factory-validation', 'validation')]
        self.runtime.memory.samples = lambda: rows
        records = dict(train=[rows[0]], valid=[rows[1]], protection=[])
        session = TrainingSession(self.runtime, 'learning', {'steps': 64}, records)
        session.prepare_commit({'promoted': True}, {}, [], event_kind='learning')
        session.finish_commit()
        body = training_runs(self.runtime.memory, self.runtime.identity)[0]
        body['factory_baseline'] = True
        with self.runtime.memory.connect() as db:
            db.execute('UPDATE training_runs SET body=? WHERE id=?', (json.dumps(body), session.id))
            db.execute("UPDATE events SET kind='factory:learning' WHERE id=?", ('training:' + session.id,))
        with patch('luban.automatic.consolidation_samples', side_effect=ValueError('waiting')):
            state = self.auto.status()
        self.assertEqual((state['learning_count'], state['knowledge_learned']), (0, 2))

    def test_factory_router_and_evolution_do_not_count_personal_updates(self):
        from luban.calibration import SPLITS
        for kind, records in (
                ('router', {'splits': {key: [] for key in SPLITS}}),
                ('evolution', {'groups': {key: [] for key in GROUP_KEYS}, 'protection': [], 'retention_regressions': []})):
            session = TrainingSession(self.runtime, kind, {'steps': 64}, records)
            session.prepare_commit({'promoted': True, 'calibrated': True}, {}, [], event_kind=kind)
            session.finish_commit()
            body = next(row for row in training_runs(self.runtime.memory, self.runtime.identity) if row['id'] == session.id)
            body['factory_baseline'] = True
            with self.runtime.memory.connect() as db:
                db.execute('UPDATE training_runs SET body=? WHERE id=?', (json.dumps(body), session.id))
                db.execute('UPDATE events SET kind=? WHERE id=?', ('factory:' + kind, 'training:' + session.id))
        with patch('luban.automatic.consolidation_samples', side_effect=ValueError('waiting')):
            state = self.auto.status()
        self.assertEqual((state['fastbrain_updates'], state['evolution_count'], state['learning_count']), (0, 0, 0))
        self.assertNotEqual(state['state'], 'waiting_validation')

    def test_setting_time_validation_and_durable_configuration(self):
        for value in ("24:00", "01:60", "9:00", "02:00:00"):
            with self.assertRaises(ValueError):
                self.settings.update({"automatic": {"mode": "auto", "scheduled_time": value}})
        self.settings.update({"automatic": {"mode": "auto", "scheduled_time": "23:59"}})
        self.assertEqual(Settings(self.settings.home).snapshot()["automatic"], {"mode": "auto", "scheduled_time": "23:59", "learning_enabled": True})

    def materials(self, prefix):
        return ([sample(prefix + str(i)) for i in range(16)]
                + [sample(prefix + "v" + str(i), "validation") for i in range(2)])

    def test_learning_manifest_excludes_experiments_before_slicing_and_keeps_protection(self):
        daily, historical = self.materials("daily"), self.materials("historical")
        for row in historical:
            row["provenance"] = json.dumps({"dataset": "historical-experiment", "evolution_role": "replay"})
        self.put_materials(daily + historical)  # Historical rows are newest in SQLite.
        _, records = self.auto._manifest("learning")
        self.assertEqual(record_keys(records["train"] + records["valid"]), record_keys(daily))
        self.assertEqual({row["id"] for row in records["protection"]},
                         {row["id"] for row in historical if row["purpose"] == "validation"})
        with patch.object(self.runtime.memory, "samples", return_value=historical):
            with self.assertRaisesRegex(ValueError, "等待足够"):
                self.auto._manifest("learning")

    def put_materials(self, rows):
        # Actual SQLite ordering/IDs, isolated CPU fixture, no product data.
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            created = db.execute("SELECT COALESCE(MAX(created), 0) + 1 FROM samples").fetchone()[0]
            db.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", [
                tuple(row[k] for k in ("id", "prompt", "answer", "source", "purpose"))
                + (created + i, 1, 1, row["provenance"]) for i, row in enumerate(rows)])

    def adopt_fixture(self, rows, *, scope="product", product_accepted=False):
        """Real SQLite commit/pointer, explicitly fabricated generation evidence."""
        protection = protected_validation(self.runtime, include_learning=True)
        records = {"train": [r for r in rows if r["purpose"] == "training"],
                   "valid": [r for r in rows if r["purpose"] == "validation"], "protection": protection}
        session = TrainingSession(self.runtime, "learning", {"steps": 64}, records, scope=scope)
        session.body["product_accepted"] = product_accepted
        expert = "learned-" + session.id
        self.runtime.store.mkdir(parents=True, exist_ok=True)
        for suffix in (".json", ".safetensors"):
            (self.runtime.store / (expert + suffix)).write_text("CPU fixture, not model weights")
        def proofs(items):
            return [{"id": r["id"], "answer": r["answer"], "branch_enabled": True,
                     **evaluate_generation(r["answer"], validate_generation_check(None, default_expected=r["answer"]), True)}
                    for r in items]
        candidate = {"accepted": True, "width": 8, "objective": 1., "initialization": "fresh",
                     "generation_checks": proofs(records["valid"]), "core_retention_generation": proofs(protection)}
        report = {"promoted": True, "active_expert": expert, "chosen_width": 8, "candidates": [candidate],
                  "base_identity": self.runtime.identity, "core_version": self.runtime.active_core,
                  "records_sha256": session.binding["records_sha256"], "fixture": True}
        pointer = {"id": expert, "width": 8, "identity": self.runtime.identity, "core_version": self.runtime.active_core}
        session.prepare_commit(report, {"active.json": pointer}, [expert + ".json", expert + ".safetensors"], event_kind="learning")
        session.finish_commit()
        self.runtime.branch.expert_id = expert
        return session

    def current_core_fixture(self, rows, version="core-000000000001"):
        self.runtime.active_core = version
        groups = {key: [] for key in GROUP_KEYS}
        for row in rows:
            row = {**row, "provenance": json.dumps({**json.loads(row["provenance"]), "evolution_role": "new"})}
            groups["new_" + row["purpose"]].append(row)
        save_core_retention(self.runtime, version, groups)

    def test_same_core_adopted_batches_do_not_cycle_after_fresh_branch_and_protection_change(self):
        first, second = self.materials("A"), self.materials("B")
        self.put_materials(first + second)
        self.adopt_fixture(first)
        _, manifest = self.auto._manifest("learning")
        self.assertEqual(record_keys(manifest["train"] + manifest["valid"]), record_keys(second))
        self.assertEqual(len(manifest["protection"]), 2)
        self.adopt_fixture(second)  # Fresh winner replaces old expert; old VALID proof retained.
        history, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertIsNone(blocked)
        self.assertEqual(history, covered)
        self.assertEqual(len(covered), 36)
        with self.assertRaisesRegex(ValueError, "等待足够"):
            self.auto._manifest("learning")
        with patch("luban.automatic.consolidation_samples", side_effect=ValueError("waiting")):
            state = self.auto.status()
        self.assertEqual((state["learning_count"], state["knowledge_learned"], state["knowledge_pending"]), (2, 36, 0))
        with patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            self.auto.tick()
            launch.assert_not_called()
        # A self-consistent new receipt still cannot drop older independent gold.
        current = training_runs(self.runtime.memory, self.runtime.identity)[0]
        current["records"]["protection"] = []
        current["binding"]["records_sha256"] = digest(current["records"])
        current["commit"]["report"]["records_sha256"] = current["binding"]["records_sha256"]
        current["commit"]["report"]["candidates"][0]["core_retention_generation"] = []
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            db.execute("UPDATE training_runs SET body=? WHERE id=?", (json.dumps(current), current["id"]))
        history, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertEqual(len(history), 36)
        self.assertEqual(covered, set())
        self.assertIn("累计独立检验", blocked)

    def test_current_core_retention_excludes_covered_but_old_core_branch_can_rebind_once(self):
        retained, branch = self.materials("core"), self.materials("branch")
        self.put_materials(retained + branch)
        self.adopt_fixture(branch)
        self.current_core_fixture(retained)
        history, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertIsNone(blocked)
        self.assertEqual(history, record_keys(branch))
        self.assertEqual(covered, record_keys(retained))
        args, manifest = self.auto._manifest("learning")
        self.assertEqual(record_keys(manifest["train"] + manifest["valid"]), record_keys(branch))
        # Different core allows the batch despite an old-core same-record receipt.
        with patch.object(self.auto, "_manifest", side_effect=lambda k: (args, manifest) if k == "learning" else (_ for _ in ()).throw(ValueError("waiting"))), \
                patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            launch.assert_called_once()
        failed = TrainingSession(self.runtime, "learning", args, manifest)
        failed.body["automatic_signature"] = self.auto._signature("learning", args, manifest)
        failed.rejected("CPU fixture rejection")
        with patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            launch.assert_not_called()
        # Independent fixture establishes a successful new-core binding, not a model retry.
        self.adopt_fixture(branch)
        _, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertIsNone(blocked)
        self.assertEqual(covered, record_keys(retained + branch))

    def test_bad_current_protection_waits_and_history_survives_missing_or_legacy_branch(self):
        rows = self.materials("old")
        self.put_materials(rows)
        session = self.adopt_fixture(rows)
        body = training_runs(self.runtime.memory, self.runtime.identity)[0]
        body["commit"]["report"]["candidates"][0]["generation_checks"][0]["complete"] = False
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            db.execute("UPDATE training_runs SET body=? WHERE id=?", (json.dumps(body), session.id))
        history, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertEqual(history, record_keys(rows))
        self.assertEqual(covered, set())
        self.assertIn("等待恢复", blocked)
        with patch.object(self.auto, "_launch") as launch:
            self.auto.tick()
            self.auto.tick()
            launch.assert_not_called()
        state = self.auto.status()
        self.assertEqual((state["learning_count"], state["knowledge_learned"], state["state"]), (1, 18, "waiting_validation"))
        (self.runtime.store / "active.json").unlink()
        history, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertEqual(history, record_keys(rows))
        self.assertEqual(covered, set())
        self.assertIsNone(blocked)
        # Legacy artifact has no complete semantic receipt; no invented coverage.
        self.runtime.branch.expert_id = "legacy"
        (self.runtime.store / "active.json").write_text(json.dumps({"id": "legacy", "identity": self.runtime.identity, "core_version": None}))
        self.assertEqual(knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))[1], set())

    def test_completed_latest_rows_are_removed_before_slice_and_keys_preserve_semantics(self):
        older, recent = self.materials("pending"), self.materials("done")
        self.put_materials(older)
        self.put_materials(recent)
        self.adopt_fixture(recent)
        args, manifest = self.auto._manifest("learning")
        self.assertEqual(record_keys(manifest["train"] + manifest["valid"]), record_keys(older))
        alias = {**recent[0], "id": "alias"}
        self.put_materials([alias])
        self.assertEqual(record_keys(self.auto._manifest("learning")[1]["train"]), record_keys(manifest["train"]))
        changed = {**recent[0], "id": "changed", "answer": "new verified answer"}
        changed_check = {**recent[1], "id": "changed-check", "provenance": json.dumps({
            **json.loads(recent[1]["provenance"]), "generation_check": {"type": "contains", "expected": "verified"}})}
        self.put_materials([changed, changed_check])
        keys = record_keys(self.auto._manifest("learning")[1]["train"])
        self.assertTrue(record_keys([changed, changed_check]) <= keys)
        self.assertFalse(record_keys([alias]) <= keys)

    def test_rejected_unaccepted_research_wrong_base_never_grants_coverage(self):
        rows = self.materials("research")
        session = self.adopt_fixture(rows, scope="research")
        history, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertEqual((history, covered, blocked), (set(), set(), None))
        session.body["product_accepted"] = True
        session._write("committed")
        history, covered, blocked = knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))
        self.assertEqual(history, covered)
        self.assertEqual(history, record_keys(rows))
        self.assertIsNone(blocked)
        wrong = training_runs(self.runtime.memory, self.runtime.identity)[0]
        wrong["binding"]["identity"] = "other-base"
        self.assertEqual(knowledge_keys(self.runtime, [wrong])[:2], (set(), set()))
        session._write("rejected")
        self.assertEqual(knowledge_keys(self.runtime, training_runs(self.runtime.memory, self.runtime.identity))[:2], (set(), set()))

    def test_malformed_durable_receipts_wait_without_losing_other_history_or_launching(self):
        rows = self.materials("valid-receipt")
        self.put_materials(rows)
        self.adopt_fixture(rows)
        records = {"train": [sample("bad")], "valid": [], "protection": []}
        bad = TrainingSession(self.runtime, "learning", {"steps": 64}, records)
        bad.prepare_commit({"promoted": True, "fixture": True}, {}, [], event_kind="learning")
        bad.finish_commit()
        original = next(r for r in training_runs(self.runtime.memory, self.runtime.identity) if r["id"] == bad.id)
        cases = [(kind, None) for kind in ("learning", "evolution", "router")] + [
            ("learning", {"valid": [], "protection": []}),
            ("learning", {"train": [], "protection": []}),
            ("learning", {"train": {}, "valid": []}),
            ("learning", {"train": [{"prompt": "missing answer"}], "valid": []}),
            ("evolution", {}), ("evolution", {"groups": {"new_training": []}}),
            ("evolution", {"groups": {key: None for key in GROUP_KEYS}}),
            ("router", {}), ("router", {"splits": {"train": []}}),
            ("router", {"splits": {key: {} for key in ("train", "development", "calibration", "test")}})]
        # A malformed receipt before OR after the good one must not stop its accumulation.
        for kind, malformed in cases:
            for updated in (0, time.time() + 100):
                with self.subTest(kind=kind, records=malformed, updated=updated):
                    body = {**original, "kind": kind}
                    if malformed is None:
                        del body["records"]
                    else:
                        body["records"] = malformed
                    with self.runtime.memory.lock, self.runtime.memory.connect() as db:
                        db.execute("UPDATE training_runs SET body=?,updated=? WHERE id=?", (json.dumps(body), updated, bad.id))
                    runs = training_runs(self.runtime.memory, self.runtime.identity)
                    history, covered, blocked = knowledge_keys(self.runtime, runs)
                    self.assertEqual(history, record_keys(rows))
                    self.assertEqual(covered, set())
                    self.assertIn("等待恢复", blocked)
                    state = self.auto.status()
                    self.assertEqual((state["state"], state["learning_count"], state["knowledge_learned"]),
                                     ("waiting_validation", 1, 18))
                    self.assertEqual((state["fastbrain_updates"], state["evolution_count"]), (0, 0))
                    with patch.object(self.auto, "_launch") as launch:
                        self.auto.tick()
                        self.auto.tick()
                        launch.assert_not_called()
                    self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity), runs)
