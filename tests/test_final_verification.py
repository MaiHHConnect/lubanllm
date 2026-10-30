"""Final-evaluation guards with synthetic cases and a real 2x2 CPU tensor."""
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/verify_luban0_final.py"
spec = importlib.util.spec_from_file_location("frozen_final_script", SCRIPT_PATH)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class FinalVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        home, repository, storage = root / "home", root / "repository", root / "storage"
        home.mkdir()
        (repository / "research").mkdir(parents=True)
        storage.mkdir()
        self.expected, self.identity = "core-000000000001", "test-base"
        self.source = root / "final.json"
        rows = [{"id": f"test-final-{i}", "split": "final", "prompt": f"synthetic question {i}",
                 "answer": "fixture", "messages": [{"role": "user", "content": f"synthetic question {i}"}],
                 "generation_check": {"type": "exact", "expected": "fixture"}} for i in range(12)]
        self.source.write_text(json.dumps(rows))
        preparation = root / "dataset.json"
        preparation.write_text(json.dumps({
            "final_test_path": str(self.source), "base_identity": self.identity, "protocol_sha256": "test-protocol",
            "final_scope": "isolated synthetic test", "creative_scope": "none"}))
        (home / "settings.json").write_text(json.dumps({"storage_root": str(storage), "model_path": "no-model"}))
        self.store = storage / "modules"
        self.store.mkdir()
        (self.store / "active-core.json").write_text(json.dumps({"version": self.expected, "identity": self.identity}))
        (self.store / (self.expected + ".json")).write_text(json.dumps({"previous_version": None}))
        self.parameter = nn.Parameter(torch.ones(2, 2), requires_grad=False)
        memory = SimpleNamespace(samples=lambda: [], event=lambda *args: None)
        self.runtime = SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5")),
            identity=self.identity, active_core=self.expected, store=self.store, device="cpu", dtype=torch.float32,
            _base_core_weight=torch.zeros(2, 2), lock=threading.RLock(), _ready=lambda: None, router_meta={},
            status=lambda: {"state": "ready"}, load=lambda: None, memory=memory)
        self.summary = repository / "research/luban0_v1_final.json"
        patches = [
            patch.object(script, "REPOSITORY", repository),
            patch.object(script, "default_home", return_value=home),
            patch.object(script, "Settings", return_value=SimpleNamespace(update=lambda value: None)),
            patch("luban.evolution._target", return_value=("fixture.weight", self.parameter)),
            patch("sys.argv", ["final", "--expected-core", self.expected, "--dataset-report", str(preparation)]),
            patch.dict("sys.modules", {"luban.runtime": SimpleNamespace(Runtime=lambda *args: self.runtime)}),
            patch("luban.memory.Memory", return_value=memory),
            patch("builtins.print"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_refused_repeat_preserves_existing_guard_and_does_not_load_model(self):
        original = json.dumps({"evaluation_started": True, "final_test_sha256": script.digest(self.source),
                               "expected_core": self.expected, "passed": False}).encode()
        self.summary.write_bytes(original)
        with patch.object(self.runtime, "load") as load:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "already evaluated"):
                    script.main()
                self.assertEqual(self.summary.read_bytes(), original)
            load.assert_not_called()
        self.assertTrue(torch.equal(self.parameter, torch.ones(2, 2)))
        self.assertTrue((self.store / "active-core.json").exists())

    def test_keyboard_interrupt_retires_cpu_parameter_and_keeps_started_guard(self):
        def interrupt(runtime, rows):
            self.assertTrue(json.loads(self.summary.read_text())["evaluation_started"])
            raise KeyboardInterrupt()

        with patch.object(script, "branch_off_generation", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                script.main()
        self.assertTrue(torch.equal(self.parameter, torch.zeros(2, 2)))
        self.assertIsNone(self.runtime.active_core)
        self.assertFalse((self.store / "active-core.json").exists())
        report = json.loads(self.summary.read_text())
        self.assertFalse(report["passed"])
        self.assertTrue(report["evaluation_started"])
        self.assertEqual(report["error"], "KeyboardInterrupt")
        self.assertTrue(report["retirement"]["restored"])
        self.assertEqual(self.source.read_bytes(), (Path(report["run_directory"]) / "final-test-original.json").read_bytes())

    def test_system_exit_during_recovery_retires_cpu_parameter(self):
        def answer(runtime, rows):
            return [{"id": rows[0]["id"], "passed": True, "complete": True,
                     "answer": "fixture", "branch_enabled": False}]

        with patch.object(script, "branch_off_generation", side_effect=answer), \
                patch.object(script, "verify_recovery", side_effect=SystemExit(2)):
            with self.assertRaises(SystemExit):
                script.main()
        self.assertTrue(torch.equal(self.parameter, torch.zeros(2, 2)))
        self.assertFalse((self.store / "active-core.json").exists())
        report = json.loads(self.summary.read_text())
        self.assertEqual(len(report["generation"]), 12)
        self.assertTrue(report["final_generation_passed"])
        self.assertFalse(report["passed"])
        self.assertTrue(report["retirement"]["restored"])

    def test_invalid_final_messages_fail_with_value_error(self):
        rows = json.loads(self.source.read_text())
        rows[0]["messages"] = ["invalid"]
        with self.assertRaisesRegex(ValueError, "contexts"):
            script.normalized_final_rows(rows)


if __name__ == "__main__":
    unittest.main()
