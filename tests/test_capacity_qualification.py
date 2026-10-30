"""Small CPU/control fixtures, not400 actual-model labels or execution proof."""
from contextlib import contextmanager
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
import run_capacity_qualification as runner
import test_capacity_selection as fixture_module


class CapacityQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_module.CapacitySelectionTests.setUpClass()
        cls.fixture = fixture_module.CapacitySelectionTests()

    def test_default_stdlib_zero_model_and_cli_refuses_partial_authority(self):
        script = ROOT / "scripts/run_capacity_qualification.py"
        raw = subprocess.run([sys.executable, "-S", str(script)], capture_output=True, text=True, check=True)
        value = json.loads(raw.stdout)
        self.assertFalse(value["torch_imported"]); self.assertFalse(value["model_started"])
        self.assertEqual(value["qualification_generation_calls"], 400)
        self.assertEqual(value["optimizer_steps"], 0)
        for args in (["--run", "missing"], ["--output", "missing"], ["--approved-sha", "0" * 64]):
            self.assertNotEqual(subprocess.run([sys.executable, "-S", str(script), *args], capture_output=True).returncode, 0)

    def test_once_full400_cpu_records_no_negative_control_no_fabricated_actual(self):
        proof = self.fixture.fixture(); values = iter(proof["records"]); durable = []
        with patch.object(runner.prior, "probes", side_effect=lambda *a, **k: [next(values)]):
            actual = runner.qualify(None, self.fixture.train, self.fixture.binding, float("inf"), durable.append, {})
        self.assertEqual(actual, durable); self.assertEqual(len(actual), 400)
        result = runner.select.select_capacity_rows(qualification=proof, expected_binding=self.fixture.binding)
        self.assertEqual(result["state"], "fixture_only")
        self.assertEqual([r["id"] for r in actual], [r["id"] for r in self.fixture.train])

    def test_deadline_after_durable_row_stops_without_refill(self):
        proof = self.fixture.fixture(); clock = [0.]; durable = []
        def record(value): durable.append(value); clock[0] = 1801.
        with patch.object(runner.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(runner.prior, "probes", return_value=[proof["records"][0]]) as calls:
            with self.assertRaises(TimeoutError):
                runner.qualify(None, self.fixture.train, self.fixture.binding, 1800, record, {})
        self.assertEqual(len(durable), 1); self.assertEqual(calls.call_count, 1)

    def test_drift_and_invalid_actual_stop_proof_reject_without_durable_record(self):
        proof = self.fixture.fixture(); record = copy.deepcopy(proof["records"][0])
        record["raw_completion_token_ids"] = [4, 8]
        with patch.object(runner.prior, "probes", return_value=[record]):
            with self.assertRaises(ValueError):
                runner.qualify(None, self.fixture.train[:1], self.fixture.binding, float("inf"), lambda v: self.fail("Bad stop persisted"), {})
        with tempfile.TemporaryDirectory() as home:
            path = Path(home) / "source"; path.write_text("new")
            with patch.object(runner.prior, "probes") as calls, self.assertRaises(ValueError):
                runner.qualify(None, self.fixture.train[:1], self.fixture.binding, float("inf"), lambda v: None, {str(path): "0" * 64})
            calls.assert_not_called()

    def test_cancel_after_durable_row_restores_actual_tiny_state_releases_lock_and_consumes(self):
        import torch
        from luban.process_lock import ModelProcessLock
        from verify_cached_training import tensor_sha
        model = torch.nn.Linear(3, 2); parameter = model.weight
        original = parameter.detach().clone(); original_sha = tensor_sha(original)
        rt = SimpleNamespace(model=model, device="cpu", dtype=torch.float32, identity=runner.IDENTITY,
            active_core=None, branch=SimpleNamespace(expert=None), _model_process_lock=None)
        flags = [(p, p.requires_grad, p.grad) for p in model.parameters()]
        modes = [(m, m.training) for m in model.modules()]; rng = torch.get_rng_state().clone()
        rt.load = lambda **kwargs: None
        rt.model.generation_config = SimpleNamespace(eos_token_id=[9])
        def release():
            rt.model = None
            if rt._model_process_lock is not None: rt._model_process_lock.close()
            rt._model_process_lock = None
        rt._release_model = release
        @contextmanager
        def measured(runtime): yield {"scope": "tiny CPU fixture"}
        with tempfile.TemporaryDirectory() as home:
            home = Path(home).resolve(); output = home / "output"; path = home / "protocol.json"
            rt.store = str(home / "private-model")
            plan = {"protocol": runner.PROTOCOL, "sealed": True, "state": "sealed_waiting_explicit_MODEL_GO",
                "output": str(output), "source_sha256": {}, "production_snapshot": {"fixture": True},
                "production_root": str(home), "official_model_binding": {"fixture": True},
                "model_path": str(home), "device": "cpu", "dtype": "torch.float32",
                "corpus_path": str(runner.select.CORPUS), "corpus_sha256": runner.select.CORPUS_SHA256}
            runner.write_new(path, plan)
            value = self.fixture.fixture()["records"][0]; value["actual_tensor_sha256"] = original_sha
            with patch.object(runner, "validate_bound"), patch.object(runner, "ORIGINAL", original_sha), \
                    patch.object(runner.select, "ORIGINAL", original_sha), \
                    patch("luban.runtime.Runtime", return_value=rt), patch("luban.evolution._target", return_value=("cpu-W", parameter)), \
                    patch("verify_automatic_batches.measured", measured), \
                    patch.object(runner.prior, "probes", side_effect=[[value], KeyboardInterrupt("CPU cancellation fixture")]), \
                    patch.object(runner, "production_snapshot", return_value={"fixture": True}), \
                    patch("verify_router_coverage_study.model_binding", return_value={"fixture": True}):
                with self.assertRaises(RuntimeError): runner.execute(path, runner.sha(path))
            result = json.loads((output / "result.json").read_text())
            self.assertFalse(result["passed"]); self.assertEqual(result["error"]["type"], "KeyboardInterrupt")
            self.assertEqual(len(result["records"]), 1); self.assertFalse((output / "qualification.json").exists())
            self.assertEqual(len((output / "qualification.raw.jsonl").read_text().splitlines()), 1)
            self.assertTrue(all(result["restoration"].values())); self.assertTrue(result["model_released"])
            self.assertTrue(result["actual_os_lock_reacquired_closed"])
            self.assertTrue(torch.equal(parameter, original)); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(p.requires_grad == f and p.grad is g for p, f, g in flags))
            self.assertTrue(all(m.training == v for m, v in modes))
            lock = ModelProcessLock(runner.IDENTITY, "cpu"); lock.close()
            plan["output"] = str(home / "second-output"); path.write_text(json.dumps(plan))
            with patch.object(runner, "validate_bound"), self.assertRaises(FileExistsError):
                runner.execute(path, runner.sha(path))
            self.assertFalse((home / "second-output").exists())


if __name__ == "__main__": unittest.main()
