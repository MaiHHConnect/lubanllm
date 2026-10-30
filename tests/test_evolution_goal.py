"""CPU/real-Bun boundary checks; never load a model or touch product data."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_evolution_goal.py"
spec = importlib.util.spec_from_file_location("evolution_goal_source", SCRIPT)
goal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(goal)


def sample(identifier, purpose="training", role="new"):
    return {"id": identifier, "prompt": identifier, "answer": "full answer",
        "source": "synthetic", "purpose": purpose, "approved": True, "training_allowed": True,
        "provenance": json.dumps({"evolution_role": role, "program_verified": True,
            "verification_evidence": "CPU fixture", "teacher_generated": False, "teacher_calls": 0,
            "draft": False, "split_family": identifier,
            "generation_check": {"type": "exact", "expected": "full answer", "max_new_tokens": 128}})}


def fixture():
    rows = [sample("new-train-" + str(i)) for i in range(16)]
    rows += [sample("new-valid-" + str(i), "validation") for i in range(4)]
    old = {"version": "core-aaaaaaaaaaaa", "regression_validation": [], "groups": {
        "new_training": [sample("old-new-train")],
        "new_validation": [sample("old-new-valid", "validation")],
        "replay_training": [sample("old-replay-train", role="replay")],
        "replay_validation": [sample("old-replay-valid", "validation", "replay")]}}
    return {"samples": rows}, old


class GoalBoundaryTests(unittest.TestCase):
    def test_cpu_import_and_help_do_not_import_torch(self):
        code = "import runpy,sys;sys.argv=['goal','--help'];\ntry:runpy.run_path(sys.argv[1] if False else " + repr(str(SCRIPT)) + ",run_name='__main__')\nexcept SystemExit:pass\nassert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ordinary450 retains its original gates", result.stdout)

    def test_relabels_all_old_records_without_mutating_originals(self):
        data, old = fixture(); before = copy.deepcopy((data, old))
        result = goal.materials(data, old)
        self.assertEqual((data, old), before)
        self.assertEqual([len(result[key]) for key in result], [16, 4, 2, 2])
        for key in ("replay_training", "replay_validation"):
            for row in result[key]:
                value = json.loads(row["provenance"])
                self.assertEqual(value["evolution_role"], "replay")
                self.assertEqual(value["retention_from_version"], old["version"])

    def test_drafts_and_unverified_or_disallowed_rows_rejected(self):
        for change in ({"approved": False}, {"approved": 1}, {"training_allowed": False}):
            data, old = fixture(); data["samples"][0].update(change)
            with self.assertRaises(ValueError): goal.materials(data, old)
        for change in ({"draft": True}, {"program_verified": False}, {"teacher_generated": True}, {"teacher_calls": 1}):
            data, old = fixture(); row = data["samples"][0]; value = json.loads(row["provenance"])
            value.update(change); row["provenance"] = json.dumps(value)
            with self.assertRaises(ValueError): goal.materials(data, old)

    def test_split_context_and_old_gold_conflicts_rejected(self):
        data, old = fixture(); data["samples"][-1] = sample("new-train-0", "validation")
        with self.assertRaises(ValueError): goal.materials(data, old)
        data, old = fixture(); data["samples"][0]["prompt"] = "old-new-train"
        data["samples"][0]["answer"] = "changed old knowledge"
        with self.assertRaises(ValueError): goal.materials(data, old)

    def test_historical_qualification_is_preserved(self):
        data, old = fixture(); row = old["groups"]["replay_training"][0]
        value = json.loads(row["provenance"])
        value.update(training_generation_gate=True, original_baseline_verified_core=None,
                     original_baseline_generation={"passed": True, "complete": True})
        row["provenance"] = json.dumps(value)
        result = json.loads(goal.materials(data, old)["replay_training"][-1]["provenance"])
        self.assertIsNone(result["original_baseline_verified_core"])
        self.assertEqual(result["original_baseline_generation"], value["original_baseline_generation"])

    def test_eos_and_complete_inventory_required(self):
        expected = [sample("wanted")]
        row = {"id": "wanted", "passed": True, "complete": True, "reference_used": False,
               "branch_enabled": False, "raw_completion_token_ids": [1], "finish_reason": "stop"}
        goal.all_passed([row], expected, native=True)
        for change in ({"complete": False}, {"reference_used": True}, {"branch_enabled": True},
                       {"raw_completion_token_ids": []}, {"finish_reason": "length"}):
            with self.assertRaises(ValueError): goal.all_passed([{**row, **change}], expected, native=True)
        with self.assertRaises(ValueError): goal.all_passed([], expected)

    def test_json_duplicate_nonfinite_and_exclusive_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / "bad.json"
            for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
                path.write_text(value)
                with self.assertRaises(ValueError): goal.read(path)
            path.unlink(); goal.write_new(path, {"ok": True})
            with self.assertRaises(FileExistsError): goal.write_new(path, {"ok": False})

    def test_real_private_bun_rpc_transaction_and_release(self):
        import sqlite3
        from luban.bun_bridge import RemoteMemory, RemoteSettings, readonly_connection
        from contextlib import closing
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); before = os.environ.get("LUBAN_BUN_RPC_TOKEN")
            with goal.data_service(root / "home", root / "service.json") as receipt:
                pid = receipt["pid"]
                settings = RemoteSettings(root / "home")
                with patch("sqlite3.connect", side_effect=AssertionError("Python SQLite forbidden")):
                    settings.update({"storage_root": str(root / "data")})
                    memory = RemoteMemory(root / "data")
                    with memory.connect() as db:
                        db.execute("CREATE TABLE proof(value INTEGER)")
                        db.execute("INSERT INTO proof VALUES(?)", (2**63 - 1,))
                    with self.assertRaises(RuntimeError):
                        with memory.connect() as db:
                            db.execute("INSERT INTO proof VALUES(2)")
                            raise RuntimeError("rollback")
                    with closing(readonly_connection(memory.path)) as db:
                        self.assertEqual(db.execute("SELECT value FROM proof").fetchall()[0][0], 2**63 - 1)
                        with self.assertRaises(sqlite3.OperationalError): db.execute("INSERT INTO proof VALUES(3)")
                    with self.assertRaises(TypeError): RemoteSettings(root / "another-home").snapshot()
                with self.assertRaises(HTTPError) as denied:
                    urlopen(Request(receipt["url"], data=b'{}', method="POST"))
                self.assertEqual(denied.exception.code, 401)
                with self.assertRaises(HTTPError) as hidden:
                    urlopen(receipt["url"].replace("/internal/rpc", "/api/status"))
                self.assertEqual(hidden.exception.code, 404)
            self.assertEqual(os.environ.get("LUBAN_BUN_RPC_TOKEN"), before)
            self.assertTrue(goal.read(root / "service.json")["released"])
            with self.assertRaises(ProcessLookupError): os.kill(pid, 0)

    def test_owner_start_failure_releases_process_and_restores_environment(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); before = os.environ.get("LUBAN_BUN_RPC_TOKEN")
            with patch.object(goal.shutil, "which", return_value="/usr/bin/false"):
                with self.assertRaises(ValueError):
                    with goal.data_service(root / "home", root / "failure.json"):
                        self.fail("failed owner must not yield")
            receipt = goal.read(root / "failure.json")
            self.assertFalse(receipt["started"])
            self.assertTrue(receipt["released"])
            self.assertEqual(os.environ.get("LUBAN_BUN_RPC_TOKEN"), before)
            with self.assertRaises(ProcessLookupError): os.kill(receipt["pid"], 0)


if __name__ == "__main__": unittest.main()
