"""Capacity control-flow fixtures; mock answers/step math, real tiny CPU Adam/files/lock."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_capacity_training as runner
import test_capacity_training_steps as fixture


class CapacityTrainingRunnerTests(unittest.TestCase):
    def test_default_and_missing_qualification_prepare_wait_no_torch(self):
        script = ROOT / "scripts/run_capacity_training.py"
        raw = subprocess.run([sys.executable, "-S", str(script)], text=True, capture_output=True, check=True)
        self.assertFalse(json.loads(raw.stdout)["model_started"])
        # Real readonly historical JSON/source scan, not tokenizer or official weight scan.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); plan_path = home / "draft.json"; output = home / "future"
            command = [sys.executable, "-S", "-c", "import sys; import run_capacity_training as r; p=r.prepare(sys.argv[1]); r.write_new(sys.argv[2],p); assert 'torch' not in sys.modules", str(output), str(plan_path)]
            subprocess.run(command, env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
                           text=True, capture_output=True, check=True, cwd=ROOT)
            plan = json.loads(plan_path.read_text())
            self.assertFalse(plan["qualified_source_ready"])
            self.assertIsNone(plan["schedule"])
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "remain waiting"): runner.bind(plan)

    def fixture_arm(self, *, capacity_n=50, check_policy="every50", fail_protection=False, advance=None, mutate=None):
        import torch
        import cached_method_steps
        from verify_cached_training import tensor_sha
        fixture.CapacityTrainingStepsTests.setUpClass()
        f = fixture.CapacityTrainingStepsTests()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); schedule, receipt, historical = f.setup_fixture(home)
            package = runner.select.read_json(receipt["qualification_path"])
            selected = runner.select.select_capacity_rows(runner.select.CORPUS, package, f.binding)["selections"][str(capacity_n)]["ids"]
            schedule = runner.steps.build_capacity_schedule(selected, schedule["replay_ids"], method="cache",
                batch_size=4, learning_rate=1e-5, wall_seconds=3600, check_policy=check_policy)
            news = {r["id"]: r for r in f.train}
            replay = [{"id": value} for value in schedule["replay_ids"]]
            plan = {"capacity_n": capacity_n, "check_policy": check_policy, "schedule": schedule,
                    "method": "cache", "wall_seconds": 3600, "learning_rate": 1e-5,
                    "new_training": [news[value] for value in schedule["new_ids"]],
                    "protected_original_corpus": [f.train[399]], "historical_reference": {**historical, "replay_training": replay},
                    "source_sha256": {}}
            if mutate: mutate(plan)
            model = torch.nn.Linear(3, 2); parameter = model.weight; original = parameter.detach().clone()
            rt = SimpleNamespace(model=model)
            flags = [(p, p.requires_grad, p.grad) for p in model.parameters()]
            rng = torch.get_rng_state().clone(); starts = []; calls = []
            def step(rt, samples, replays, master, opt, reference, basis, **kwargs):
                if not opt.state: starts.append(master.detach().clone())
                master.grad = torch.full_like(master, .1); opt.step()
                if advance: advance()
                return {"cpu_fixture": True, "runtime_restored": True, "optimizer_steps": 1,
                    "ce_role": kwargs["ce_role"], "ce_ids": [r["id"] for r in samples], "replay_kl_ids": [r["id"] for r in replays]}
            def proofs(rt, rows, deadline, candidate, raw_path, phase):
                calls.append((phase, len(rows), candidate))
                new = phase.endswith("new_first") or phase.endswith("new_second")
                return [f.proof(row, (index < 2 if new else not fail_protection or index > 0), candidate)
                        for index, row in enumerate(rows)]
            with patch("luban.evolution._target", return_value=("CPU-W", parameter)), \
                    patch("luban.evolution._loss", return_value=1.01), \
                    patch.object(cached_method_steps, "train_method_step", side_effect=step), \
                    patch.object(runner, "probe_records", side_effect=proofs):
                report = runner.arm(rt, plan, receipt, original, torch.eye(3)[:, :1], {}, 1., home, 0)
            self.assertTrue(torch.equal(parameter, original))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(p.requires_grad == flag and p.grad is grad for p, flag, grad in flags))
            self.assertTrue(torch.equal(starts[0], original.float()))
            payloads = {p.name: torch.load(p, weights_only=True, mmap=True, map_location="cpu") for p in home.glob("checkpoint-*.pt")}
            return report, payloads, calls, tensor_sha(original)

    def test_continuous_real_tiny_adam_three_markers_and_actual_checkpoint_readback(self):
        # Model answers and step gradient are MOCKS; this proves scheduler/file state, not actual learning.
        with patch.object(runner.time, "perf_counter", return_value=1.):
            report, payloads, calls, original_sha = self.fixture_arm()
        self.assertTrue(report["completed"])
        self.assertEqual(report["ledger"]["optimizer_updates"], 168)
        self.assertEqual(report["ledger"]["ce_events"], 600)
        self.assertEqual({k: sum(v.values()) for k, v in report["ledger"]["per_id_exposures"].items()},
                         {"new_ce": 300, "replay_ce": 300, "replay_kl": 600})
        doubles = [c for c in report["checks"] if c["new_second"]]
        self.assertEqual([c["ce_events"] for c in doubles], [200, 400, 600])
        self.assertEqual([c["optimizer_updates"] for c in doubles], [56, 112, 168])
        self.assertEqual([int(float(payloads["checkpoint-" + str(i) + ".pt"]["optimizer"]["state"][0]["step"]))
                          for i in (200, 400, 600)], [56, 112, 168])
        self.assertTrue(all(len(c["protected_observations"]) == 69 for c in report["checks"]))
        self.assertEqual(len([c for c in calls if c[0].endswith("new_second")]), 3)
        self.assertTrue(all(c["verdict"]["production_learned_ids"] == [] for c in report["checks"]))
        self.assertNotEqual(doubles[-1]["candidate_tensor_sha256"], original_sha)
        self.assertEqual(len(payloads), 12)

    def test_first_check_regression_stops_no_more_optimizer_events(self):
        with patch.object(runner.time, "perf_counter", return_value=1.):
            report, payloads, _, _ = self.fixture_arm(fail_protection=True)
        self.assertFalse(report["completed"])
        self.assertEqual(report["ledger"]["ce_events"], 50)
        self.assertEqual(report["ledger"]["optimizer_updates"], 14)
        self.assertEqual(len(payloads), 1)
        self.assertTrue(report["original_tensor_restored"])

    def test_other_N_continuous_tiny_adam_full_exposures_checkpoints_and_fresh_starts(self):
        # Each invocation has a fresh tiny W/master/Adam. Mock answers/gradients
        # do not measure actual model capacity, derivative correctness or speed.
        for n, markers in ((50, [56, 112, 168]), (100, [112, 224, 336]),
                           (200, [224, 448, 672]), (300, [336, 672, 1008])):
            with self.subTest(n=n), patch.object(runner.time, "perf_counter", return_value=1.):
                report, payloads, _, _ = self.fixture_arm(capacity_n=n, check_policy="per_new_exposure")
            self.assertTrue(report["completed"])
            self.assertEqual(report["capacity_n"], n)
            self.assertEqual(report["ledger"]["ce_events"], 12 * n)
            self.assertEqual(report["ledger"]["optimizer_updates"], markers[-1])
            self.assertEqual({k: sum(v.values()) for k, v in report["ledger"]["per_id_exposures"].items()},
                             {"new_ce": 6 * n, "replay_ce": 6 * n, "replay_kl": 12 * n})
            doubles = [c for c in report["checks"] if c["new_second"]]
            self.assertEqual([c["ce_events"] for c in doubles], [4 * n, 8 * n, 12 * n])
            self.assertEqual([c["optimizer_updates"] for c in doubles], markers)
            self.assertTrue(all(len(c["new_first"]) == n and len(c["protected_observations"]) == 69
                                for c in report["checks"]))
            self.assertEqual(len(payloads), 6)
            self.assertEqual([c["ce_events"] for c in report["checks"]], [2 * n * e for e in range(1, 7)])
            self.assertTrue(all(p["progress"]["capacity_n"] == n for p in payloads.values()))
            self.assertTrue(all(p["progress"]["check_policy"] == "per_new_exposure" for p in payloads.values()))
            self.assertEqual(set(report["ledger"]["per_id_exposures"]["new_ce"].values()), {6})
            self.assertTrue(report["original_tensor_restored"])

    def test_N_count_or_selected_order_mismatch_rejects_before_update(self):
        for mutation in (lambda p: p.update(capacity_n=200),
                         lambda p: p["new_training"].pop(),
                         lambda p: p["new_training"].reverse(),
                         lambda p: p["schedule"].update(ce_events=600)):
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "count/order"):
                self.fixture_arm(capacity_n=100, mutate=mutation)

    def test_prepare_all_N_waits_without_defaults_and_rejects_invalid_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve()
            for n in (None, 50, 100, 200, 300):
                plan = runner.prepare(home / ("future-" + str(n)), capacity_n=n)
                self.assertEqual(plan["capacity_n"], n)
                self.assertIsNone(plan["wall_seconds"])
                self.assertFalse(plan["qualified_source_ready"])
                self.assertFalse(Path(plan["output"]).exists())
            for n in (True, "100", 100., 0, 400):
                with self.subTest(n=n), self.assertRaises(ValueError): runner.prepare(home / "invalid", capacity_n=n)

    def test_prepare_and_binding_select_exact_N_and_reject_count_change(self):
        fixture.CapacityTrainingStepsTests.setUpClass(); f = fixture.CapacityTrainingStepsTests()
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); _, receipt, historical = f.setup_fixture(home)
            package, selection, train, _ = runner.steps._qualification(receipt)
            # Explicitly mocked qualification gate and file bindings only for
            # static prepare/bind behavior; these receipts cannot execute.
            official = {"CPU": "MOCK binding, no official weights"}
            history = {**historical, "rows_sha256": receipt["historical_rows_sha256"],
                "baseline_sha256": receipt["historical_baseline_sha256"], "replay_training": [{"id": "R" + str(i)} for i in range(57)],
                "source_sha256": {}, "official_model_binding": official}
            result = runner.select.read_json(receipt["result_path"]); result["source_after"] = {}
            f.write(Path(receipt["result_path"]), result)
            mocked_package = copy.deepcopy(package)
            mocked_package["binding"]["official_sha256"] = runner.content_sha(official)
            receipt_path = home / "fixture-receipt.json"; f.write(receipt_path, receipt)
            with patch.object(runner, "historical_reference", return_value=history), \
                    patch.object(runner.steps, "_qualification", return_value=(mocked_package, selection, train, False)), \
                    patch("verify_router_coverage_study.model_binding", return_value=official), \
                    patch.object(runner, "production_snapshot", return_value={"CPU": True}):
                for n in runner.select.COUNTS:
                    with self.subTest(n=n):
                        plan = runner.prepare(home / ("future-" + str(n)), capacity_n=n,
                            check_policy="per_new_exposure", receipt_path=receipt_path,
                            method="cache", batch_size=4, wall_seconds=1234.)
                        self.assertEqual(len(plan["new_training"]), n)
                        self.assertEqual(plan["schedule"]["new_ids"], selection["selections"][str(n)]["ids"])
                        self.assertEqual(plan["schedule"]["ce_events"], 12 * n)
                        expected_calls = 69 + 6 * (n + 69) + 3 * n
                        self.assertEqual(plan["training_generation_calls_upper_bound"], expected_calls)
                        bound = runner.bind(plan); runner.validate_bound(bound)
                        changed = copy.deepcopy(bound); changed["capacity_n"] = 50 if n != 50 else 100
                        with self.assertRaisesRegex(ValueError, "changed"): runner.validate_bound(changed)
                        changed = copy.deepcopy(bound); changed["check_policy"] = "every50"
                        with self.assertRaisesRegex(ValueError, "changed"): runner.validate_bound(changed)
                        self.assertFalse(Path(plan["output"]).exists())

    def test_update_timeout_keeps_real_count_and_restores_original(self):
        clock = [1.]
        def advance(): clock[0] = 3601.
        with patch.object(runner.time, "perf_counter", side_effect=lambda: clock[0]):
            report, payloads, calls, _ = self.fixture_arm(advance=advance)
        self.assertFalse(report["completed"])
        self.assertEqual(report["ledger"]["optimizer_updates"], 1)
        self.assertEqual(report["ledger"]["ce_events"], 4)
        self.assertEqual(report["error"]["type"], "TimeoutError")
        self.assertFalse(payloads); self.assertFalse(calls)
        self.assertTrue(report["original_tensor_restored"])

    def test_final_arm_save_overrun_cannot_keep_completed_or_quality_true(self):
        clock = [1.]; original_write = Path.write_text
        def save(path, text, *args, **kwargs):
            value = json.loads(text)
            if value.get("completed") is True: clock[0] = 3601.
            return original_write(path, text, *args, **kwargs)
        with patch.object(runner.time, "perf_counter", side_effect=lambda: clock[0]), patch.object(Path, "write_text", save):
            report, payloads, _, _ = self.fixture_arm()
        self.assertEqual(report["ledger"]["ce_events"], 600)
        self.assertEqual(len(payloads), 12)
        self.assertFalse(report["completed"]); self.assertFalse(report["quality_passed"])
        self.assertEqual(report["error"]["type"], "TimeoutError")
        self.assertTrue(report["original_tensor_restored"])

    def test_checkpoint_readback_rejects_altered_adam_state_metadata(self):
        import torch
        model = torch.nn.Linear(3, 2); parameter = model.weight
        master = torch.nn.Parameter(parameter.detach().float().clone())
        opt = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
        master.grad = torch.ones_like(master); opt.step()
        real_load = torch.load
        def wrong(path, **kwargs):
            value = real_load(path, **kwargs)
            value["optimizer"]["param_groups"][0]["lr"] = .1
            return value
        with tempfile.TemporaryDirectory() as tmp, patch("luban.evolution._target", return_value=("CPU-W", parameter)), \
                patch.object(torch, "load", side_effect=wrong), self.assertRaisesRegex(ValueError, "readback mismatch"):
            runner.save_checkpoint(Path(tmp).resolve() / "checkpoint.pt", SimpleNamespace(model=model),
                master, opt, parameter.detach().float(), torch.eye(3)[:, :1], {"optimizer_updates": 1}, {})

    def test_each_completed_raw_is_fsynced_before_later_cancel(self):
        fixture.CapacityTrainingStepsTests.setUpClass(); f = fixture.CapacityTrainingStepsTests()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "raw.jsonl"
            first = f.proof(f.train[0], False)
            with patch.object(runner.prior, "probes", side_effect=[[first], KeyboardInterrupt()]), \
                    patch.object(runner.time, "perf_counter", return_value=1.), self.assertRaises(KeyboardInterrupt):
                runner.probe_records(None, f.train[:2], 100, runner.ORIGINAL, path, "CPU-fixture")
            values = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(values), 1); self.assertEqual(values[0]["observation"], first)

    def test_wrong_preflight_and_consumed_protocol_cannot_allocate_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); path = home / "protocol.json"; output = home / "execution"
            plan = {"protocol": runner.PROTOCOL, "sealed": True, "state": "sealed_waiting_explicit_MODEL_GO", "capacity_n": 50,
                    "output": str(output), "wall_seconds": 3600}
            runner.write_new(path, plan)
            with self.assertRaises(ValueError): runner.execute(path, "0" * 64)
            with patch.object(runner, "validate_bound", side_effect=ValueError("wrong source")), self.assertRaises(ValueError):
                runner.execute(path, runner.sha(path))
            self.assertFalse(output.exists()); self.assertFalse((home / "execution-started.json").exists())
            runner.write_new(home / "execution-started.json", {"CPU-consumed": True})
            with patch.object(runner, "validate_bound"), self.assertRaises(FileExistsError): runner.execute(path, runner.sha(path))
            self.assertFalse(output.exists())

    def test_load_failure_releases_real_cpu_os_lock_and_never_retries(self):
        import torch
        from luban.process_lock import ModelProcessLock
        rt = SimpleNamespace(model=None, device="cpu", dtype=torch.float32, _model_process_lock=None)
        rt.load = lambda **kwargs: (_ for _ in ()).throw(ValueError("CPU fixture model allocation refused"))
        def release():
            if rt._model_process_lock is not None: rt._model_process_lock.close()
            rt._model_process_lock = None
        rt._release_model = release
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve(); output = home / "execution"; path = home / "protocol.json"
            receipt_path = home / "receipt.json"; runner.write_new(receipt_path, {"CPU-only": True})
            history = {"CPU": "fixture"}
            plan = {"protocol": runner.PROTOCOL, "sealed": True, "state": "sealed_waiting_explicit_MODEL_GO", "capacity_n": 50,
                "output": str(output), "wall_seconds": 3600, "qualification_wall_seconds": 10,
                "check_policy": "every50",
                "source_sha256": {}, "production_snapshot": {"CPU": True}, "production_root": str(home),
                "official_model_binding": {"CPU": True}, "model_path": str(home), "device": "cpu", "dtype": "torch.float32",
                "historical_reference": {**history, "replay_training": []}, "qualification_receipt_path": str(receipt_path),
                "qualification_receipt_sha256": runner.sha(receipt_path)}
            runner.write_new(path, plan)
            with patch.object(runner, "validate_bound"), patch("luban.runtime.Runtime", return_value=rt), \
                    patch("verify_router_coverage_study.model_binding", return_value={"CPU": True}), \
                    patch.object(runner, "production_snapshot", return_value={"CPU": True}), \
                    patch.object(runner.steps, "_qualification"), \
                    patch.object(runner, "historical_reference", return_value=plan["historical_reference"]), \
                    self.assertRaises(RuntimeError):
                runner.execute(path, runner.sha(path))
            result = json.loads((output / "result.json").read_text())
            self.assertFalse(result["passed"]); self.assertTrue(result["model_released"])
            self.assertTrue(result["actual_os_lock_reacquired_closed"])
            self.assertEqual(result["error"]["type"], "ValueError")
            self.assertFalse(result["model_started"])
            lock = ModelProcessLock(runner.IDENTITY, "cpu"); lock.close()
            new_plan = {**plan, "output": str(home / "second-output")}; path.write_text(json.dumps(new_plan))
            with patch.object(runner, "validate_bound"), self.assertRaises(FileExistsError): runner.execute(path, runner.sha(path))
            self.assertFalse((home / "second-output").exists())


if __name__ == "__main__": unittest.main()
