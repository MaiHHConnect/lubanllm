"""Pure-stdlib fabricated receipts/proofs, never actual model evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capacity_training_steps as capacity
import select_capacity_rows as select


class CapacityTrainingStepsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = select._materials(select.read_json(select.CORPUS))
        cls.binding = {"corpus_sha256": select.CORPUS_SHA256, "base_identity": select.IDENTITY,
            "core_version": None, "actual_tensor_sha256": select.ORIGINAL, "source_sha256": "2" * 64,
            "official_sha256": "3" * 64, "max_new_tokens": 512, "temperature": 0,
            "branch_enabled": False, "reference_used": False, "thinking": "off", "do_sample": False,
            "eos_token_ids": [9], "scope": "cpu_fixture"}

    def proof(self, row, passed, tensor=select.ORIGINAL):
        answer = row["answer"] if passed else "FABRICATED-CPU-WRONG"
        return {"id": row["id"], "sample_sha256": select.digest(row), "actual_base_identity": select.IDENTITY,
            "actual_tensor_sha256": tensor, "core_version": None, "max_new_tokens": 512,
            "branch_enabled": False, "reference_used": False, "thinking": "off", "do_sample": False,
            "complete": True, "passed": passed, "answer": answer, "decoded_completion_unstripped": answer + "\n",
            "raw_completion_token_ids": [4, 9], "completion_tokens": 2}

    def setup_fixture(self, home):
        # One correct corpus row stays protected even though never selected.
        package = {"binding": self.binding, "records": [self.proof(row, index == 399)
                    for index, row in enumerate(self.train)]}
        historical = [{"id": "CPU-HISTORY-" + str(i), "answer": "H" + str(i),
            "provenance": json.dumps({"evolution_role": "new" if i < 13 else "replay",
                "messages": [{"role": "user", "content": "CPU fixture history " + str(i)}],
                "generation_check": {"type": "exact", "expected": "H" + str(i), "max_new_tokens": 512}})}
            for i in range(68)]
        old = {"rows": historical, "baseline": [self.proof(row, True) for row in historical]}
        qualification_path, result_path, root_path = [home / name for name in ("qualification.json", "result.json", "root.json")]
        self.write(qualification_path, package)
        result = {"protocol": "fictional-capacity-original400-qualification-v1", "protocol_sha256": "5" * 64,
            "passed": True, "state": "complete_waiting_root_evidence_review", "records": package["records"],
            "qualification_sha256": capacity.sha(qualification_path), "preserved": True,
            "restoration": {key: True for key in capacity.RESTORATION}, "model_released": True,
            "actual_os_lock_reacquired_closed": True, "wall_seconds": 20, "wall_cap_seconds": 1800,
            "optimizer_steps": 0, "teacher_calls": 0, "publication": False}
        self.write(result_path, result)
        root = {"scope": "cpu_fixture", "passed": True, "process_exit_code": 0, "pid_dead": True,
                "result_sha256": capacity.sha(result_path), "protocol_sha256": "5" * 64}
        self.write(root_path, root)
        receipt = {name + "_path": str(path) for name, path in
                   (("qualification", qualification_path), ("result", result_path), ("root_post", root_path))}
        receipt.update({name + "_sha256": capacity.sha(path) for name, path in
                        (("qualification", qualification_path), ("result", result_path), ("root_post", root_path))})
        receipt.update(protocol_sha256="5" * 64, historical_rows_sha256=select.digest(historical),
                       historical_baseline_sha256=select.digest(old["baseline"]))
        selected = select.select_capacity_rows(select.CORPUS, package, self.binding)["selections"]["50"]["ids"]
        schedule = capacity.build_capacity_schedule(selected, ["REPLAY-" + str(i) for i in range(57)],
            method="cache", batch_size=4, learning_rate=1e-5, wall_seconds=3600)
        return schedule, receipt, old

    def checkpoint(self, home, schedule, historical, event=200):
        segment = next(s for s in schedule["segments"] if s["check_ce_event"] == event)
        rows = {row["id"]: row for row in self.train}
        selected = [rows[value] for value in schedule["new_ids"]]
        candidate = "6" * 64
        first = [self.proof(row, index < 2, candidate) for index, row in enumerate(selected)]
        protected = [self.train[399]] + historical["rows"]
        artifact = home / "candidate-fixture.json"
        self.write(artifact, {"scope": "CPU fixture, not master/Adam tensor payload"})
        return {"ce_events": event, "optimizer_updates": segment["optimizer_updates_at_check"],
            "per_id_exposures": segment["per_id_exposures"], "candidate_tensor_sha256": candidate,
            "frozen_tensor_sha256": [candidate] * (4 if segment["double_new_proof"] else 3),
            "checkpoint_artifact": {"path": str(artifact), "file_sha256": capacity.sha(artifact),
                "candidate_tensor_sha256": candidate, "optimizer_updates": segment["optimizer_updates_at_check"],
                "payload_scope": "full_candidate_master_adam_rng_progress"},
            "protected_ids": [row["id"] for row in protected],
            "protected_observations": [self.proof(row, True, candidate) for row in protected],
            "new_first": first, "new_second": copy.deepcopy(first) if segment["double_new_proof"] else [],
            "elapsed_seconds": 100, "baseline_replay_loss": 1.0, "replay_loss": 1.01}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))

    def test_schedule_exact_three_ledgers_per_id_and_checkpoint_tails(self):
        for n in select.COUNTS:
            for b in (1, 2, 4):
                plan = capacity.build_capacity_schedule(["N" + str(i) for i in range(n)],
                    ["R" + str(i) for i in range(57)], method="cache", batch_size=b,
                    learning_rate=1e-5, wall_seconds=3600)
                previous_ce, previous_adam = 0, 0
                for segment in plan["segments"]:
                    batches = segment["batches"]
                    event_ids = sorted(e for batch in batches for e in batch["ce_event_ids"])
                    self.assertEqual(event_ids, list(range(previous_ce + 1, segment["check_ce_event"] + 1)))
                    self.assertEqual(segment["optimizer_updates_at_check"], previous_adam + len(batches))
                    ce = segment["check_ce_event"]
                    self.assertEqual(segment["exposure_totals"], {"new_ce": ce // 2, "replay_ce": ce // 2, "replay_kl": ce})
                    if segment["double_new_proof"]:
                        self.assertEqual(set(segment["per_id_exposures"]["new_ce"].values()), {segment["new_exposure_marker"]})
                    previous_ce, previous_adam = ce, segment["optimizer_updates_at_check"]
                self.assertEqual(plan["ce_events"], 12 * n)
                if n == 50:
                    markers = [s["optimizer_updates_at_check"] for s in plan["segments"] if s["double_new_proof"]]
                    self.assertEqual(markers, {1: [200, 400, 600], 2: [104, 208, 312], 4: [56, 112, 168]}[b])

    def test_schedule_invalid_or_unsupported_parameters_rejected(self):
        args = {"new_ids": ["N" + str(i) for i in range(50)], "replay_ids": ["R0", "R1"],
                "method": "cache", "batch_size": 2, "learning_rate": 1e-5, "wall_seconds": 3600}
        for patch in ({"new_ids": args["new_ids"][:-1]}, {"replay_ids": ["R", "R"]},
                      {"method": "native"}, {"batch_size": True}, {"learning_rate": 1e-4},
                      {"wall_seconds": float("nan")}, {"replay_ids": ["N0"]}, {"check_policy": "adaptive"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                capacity.build_capacity_schedule(**{**args, **patch})

    def test_per_exposure_checks_keep_every50_flush_and_full_optimizer_order(self):
        for n in select.COUNTS:
            for b in (1, 2, 4):
                args = {"new_ids": ["N" + str(i) for i in range(n)], "replay_ids": ["R" + str(i) for i in range(57)],
                        "method": "cache", "batch_size": b, "learning_rate": 1e-5, "wall_seconds": 3600}
                old = capacity.build_capacity_schedule(**args)
                plan = capacity.build_capacity_schedule(**args, check_policy="per_new_exposure")
                self.assertEqual(plan["checks_ce_events"], [2 * n * e for e in range(1, 7)])
                self.assertEqual(plan["flush_ce_events"], old["flush_ce_events"])
                self.assertEqual(plan["optimizer_updates"], old["optimizer_updates"])
                self.assertEqual([s["batches"] for s in plan["segments"]], [s["batches"] for s in old["segments"]])
                checks = [s for s in plan["segments"] if s["check_required"]]
                self.assertEqual(len(checks), 6)
                self.assertEqual([s["new_exposure_marker"] for s in checks if s["double_new_proof"]], [2, 4, 6])
                self.assertEqual([s["per_id_exposures"] for s in plan["segments"]],
                                 [s["per_id_exposures"] for s in old["segments"]])

    def test_per_exposure_validator_rejects_flush_only_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); old, receipt, historical = self.setup_fixture(home)
            plan = capacity.build_capacity_schedule(old["new_ids"], old["replay_ids"], method="cache", batch_size=4,
                learning_rate=1e-5, wall_seconds=3600, check_policy="per_new_exposure")
            cp = self.checkpoint(home, plan, historical, event=50)
            with self.assertRaisesRegex(ValueError, "check-policy"):
                capacity.observe_capacity_checkpoint(plan, cp, receipt, historical)
            for event, double in ((100, False), (200, True), (300, False), (400, True), (500, False), (600, True)):
                result = capacity.observe_capacity_checkpoint(plan, self.checkpoint(home, plan, historical, event), receipt, historical)
                self.assertEqual(result["double_proof"], double)
                self.assertEqual(result["candidate_confirmed_growth"], 2 if double else 0)

    def test_double_growth_protection_and_nonpublication_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); plan, receipt, old = self.setup_fixture(home)
            actual = self.checkpoint(home, plan, old)
            report = capacity.observe_capacity_checkpoint(plan, actual, receipt, old)
            self.assertEqual(report["candidate_confirmed_growth"], 2)
            self.assertTrue(report["continue_allowed"])
            self.assertEqual(len(report["protected_ids"]), 69)
            self.assertFalse(report["actual_model_evidence_authenticated"])
            self.assertEqual(report["evidence_scope"], "cpu_fixture")
            self.assertEqual(report["production_learned_ids"], [])
            zero_loss = copy.deepcopy(actual); zero_loss["replay_loss"] = 0.
            self.assertTrue(capacity.observe_capacity_checkpoint(plan, zero_loss, receipt, old)["continue_allowed"])
            single = self.checkpoint(home, plan, old, 50)
            report = capacity.observe_capacity_checkpoint(plan, single, receipt, old)
            self.assertEqual(len(report["provisional_growth_ids"]), 2)
            self.assertEqual(report["candidate_confirmed_growth"], 0)
            self.assertIsNone(report["pending_ids"])

    def test_final_qualification_failure_and_root_receipt_drift_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); plan, receipt, old = self.setup_fixture(home)
            cp = self.checkpoint(home, plan, old)
            for change in ({"passed": False}, {"wall_seconds": 1801}, {"model_released": False},
                           {"restoration": {"x": True}}, {"optimizer_steps": 1}):
                result = select.read_json(receipt["result_path"]); result.update(change)
                self.write(Path(receipt["result_path"]), result)
                changed = {**receipt, "result_sha256": capacity.sha(receipt["result_path"])}
                root = select.read_json(receipt["root_post_path"]); root["result_sha256"] = changed["result_sha256"]
                self.write(Path(receipt["root_post_path"]), root)
                changed["root_post_sha256"] = capacity.sha(receipt["root_post_path"])
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "qualification required"):
                    capacity.observe_capacity_checkpoint(plan, cp, changed, old)
            plan, receipt, old = self.setup_fixture(home)
            root = select.read_json(receipt["root_post_path"]); root["process_exit_code"] = True
            self.write(Path(receipt["root_post_path"]), root)
            receipt["root_post_sha256"] = capacity.sha(receipt["root_post_path"])
            with self.assertRaises(ValueError): capacity.observe_capacity_checkpoint(plan, cp, receipt, old)

    def test_proofs_ledger_omission_and_tensor_drift_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); plan, receipt, old = self.setup_fixture(home)
            cp = self.checkpoint(home, plan, old)
            changes = {"missing_second": lambda c: c.update(new_second=[]),
                "branch": lambda c: c["new_first"][0].update(branch_enabled=True),
                "candidate": lambda c: c["new_second"][0].update(actual_tensor_sha256="7" * 64),
                "raw": lambda c: c["new_first"][0].update(raw_completion_token_ids=[4, 8]),
                "sample": lambda c: c["new_first"][0].update(sample_sha256="0" * 64),
                "frozen": lambda c: c["frozen_tensor_sha256"].__setitem__(1, "7" * 64),
                "omit_unselected_known": lambda c: c["protected_ids"].pop(0),
                "adam": lambda c: c.update(optimizer_updates=50),
                "ledger": lambda c: c["per_id_exposures"]["replay_kl"].update(REPLAY_0=999),
                "artifact": lambda c: c["checkpoint_artifact"].update(file_sha256="0" * 64),
                "boundary": lambda c: c.update(ce_events=199)}
            for name, mutation in changes.items():
                changed = copy.deepcopy(cp); mutation(changed)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    capacity.observe_capacity_checkpoint(plan, changed, receipt, old)

    def test_regression_or_full_wall_overrun_keeps_raw_growth_but_cannot_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); plan, receipt, old = self.setup_fixture(home)
            cp = self.checkpoint(home, plan, old)
            row = self.train[399]
            cp["protected_observations"][0] = self.proof(row, False, cp["candidate_tensor_sha256"])
            result = capacity.observe_capacity_checkpoint(plan, cp, receipt, old)
            self.assertFalse(result["continue_allowed"])
            self.assertEqual(result["forgotten_ids"], [row["id"]])
            cp = self.checkpoint(home, plan, old); cp["elapsed_seconds"] = 3601
            result = capacity.observe_capacity_checkpoint(plan, cp, receipt, old)
            self.assertEqual(result["candidate_confirmed_growth"], 2)
            self.assertFalse(result["within_wall_cap"])
            self.assertFalse(result["continue_allowed"])
            self.assertFalse(result["publication"])

    def test_default_cli_never_imports_torch_or_uses_model(self):
        raw = subprocess.run([sys.executable, "-S", str(select.ROOT / "scripts/capacity_training_steps.py")],
                             text=True, capture_output=True, check=True)
        result = json.loads(raw.stdout)
        self.assertFalse(result["torch_imported"])
        self.assertFalse(result["execution_ready"])
        self.assertFalse(result["model_started"])


if __name__ == "__main__": unittest.main()
