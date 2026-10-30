"""CPU/mock allocation and real POSIX subprocess locks; no upstream models/HTTP."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import torch
from torch import nn

from luban.memory import Memory
from luban.process_lock import ModelProcessLock
from luban.runtime import Runtime
from luban.training_checkpoint import recover_commits, ensure_product_state, training_runs


@unittest.skipUnless(os.name == "posix", "POSIX lock; Windows explicitly unsupported")
class RuntimeProcessLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "model"
        self.path.mkdir()
        (self.path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        (self.path / "upstream-file-manifest.json").write_text(json.dumps({"Data": {"Files": [
            {"Path": "config.json", "Sha256": hashlib.sha256(uuid.uuid4().bytes).hexdigest()}]}}))
        self.ensure = self.patching("luban.training_checkpoint.ensure_product_state", return_value=None)
        self.recover = self.patching("luban.runtime.recover_commits", side_effect=recover_commits)
        self.allocate = self.patching("luban.runtime.load_backbone", side_effect=self.model)
        self.patching("luban.runtime.text_backbone", side_effect=lambda model: (model, SimpleNamespace(hidden_size=4, num_hidden_layers=1)))
        self.patching("luban.runtime.AutoTokenizer.from_pretrained", return_value=Mock())
        self.patching("luban.runtime.insert_experts", return_value=[SimpleNamespace(expert_id=None)])
        self.patching("luban.evolution.load_active_core", return_value=None)
        self.patching("luban.runtime.Runtime._load_router", return_value=None)

    def patching(self, name, **kwargs):
        patcher = patch(name, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def model(*args):
        model = nn.Linear(4, 4)
        model.config = SimpleNamespace(model_type="qwen3")
        return model

    def runtime(self, rootname, **kwargs):
        root = self.root / rootname
        root.mkdir()
        settings = Mock()
        settings.snapshot.return_value = {"storage_root": str(root), "model_path": str(self.path)}
        runtime = Runtime(settings, Memory(root), **kwargs)
        runtime.device, runtime.dtype = "cpu", torch.float32
        self.addCleanup(runtime._release_model)
        return runtime

    def test_different_roots_same_base_blocks_before_allocation(self):
        first, second = self.runtime("one"), self.runtime("two")
        first.load(paging=False)
        with self.assertRaisesRegex(RuntimeError, "已有模型进程"):
            second.load(paging=False)
        self.assertEqual(self.allocate.call_count, 1)
        self.assertIsNone(second.model)
        first._release_model()
        second.load(paging=False)
        self.assertEqual(self.allocate.call_count, 2)

    def test_same_runtime_reload_keeps_lease(self):
        runtime = self.runtime("one")
        runtime.load(paging=False)
        lease = runtime._model_process_lock
        runtime.load(paging=False)
        self.assertIs(runtime._model_process_lock, lease)
        self.assertEqual(self.allocate.call_count, 2)

    def test_packed_profile_load_and_nonpacked_reload_clears_stale_profile(self):
        runtime = self.runtime("packed")
        profile = self.root / "profile.bin"
        profile.write_bytes(b"fixture")
        config = runtime.settings.snapshot.return_value
        config.update(native_inference_precision="mixed-q4-g16-packed", native_q4_profile=str(profile))
        runtime.load(paging=False)
        self.assertEqual(runtime.native_q4_profile, str(profile))
        config["native_inference_precision"] = "bf16"
        runtime.load(paging=False)
        self.assertIsNone(runtime.native_q4_profile)
        config["native_inference_precision"] = "mixed-q4-g16-packed"
        profile.unlink()
        with self.assertRaisesRegex(ValueError, "已存在文件"):
            runtime.load(paging=False)
        self.assertIsNone(runtime.native_q4_profile)

    def test_native_execution_policy_reloads_and_validates_before_model_allocation(self):
        runtime = self.runtime("policy")
        config = runtime.settings.snapshot.return_value
        policy = {"native_skip_layer_mask": 2, "native_exit_after": 12, "native_layer_paging": True}
        config.update(policy)
        runtime.load(paging=False)
        self.assertTrue(all(getattr(runtime, key) == value for key, value in policy.items()))
        for key in policy:
            config.pop(key)
        runtime.load(paging=False)
        self.assertEqual((runtime.native_skip_layer_mask, runtime.native_exit_after, runtime.native_layer_paging), (0, 24, False))
        config["native_skip_layer_mask"] = True
        self.allocate.reset_mock()
        with self.assertRaisesRegex(ValueError, "跳层掩码"):
            runtime.load(paging=False)
        self.allocate.assert_not_called()

    def test_failed_allocation_releases_lock_and_partial_model(self):
        first, second = self.runtime("one"), self.runtime("two")
        self.allocate.side_effect = RuntimeError("allocation failed")
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            first.load(paging=False)
        self.assertIsNone(first._model_process_lock)
        self.assertIsNone(first.model)
        self.allocate.side_effect = self.model
        second.load(paging=False)

    def test_product_gate_before_recovery_and_allocation(self):
        runtime = self.runtime("one")
        self.ensure.side_effect = ValueError("research pointer forbidden")
        with self.assertRaisesRegex(ValueError, "research pointer"):
            runtime.load(paging=False)
        self.allocate.assert_not_called()
        self.assertEqual(self.recover.call_count, 1)
        self.assertIsNone(runtime._model_process_lock)

    def test_real_research_journal_pointer_rejected_before_allocation(self):
        runtime = self.runtime("one")
        manifest = json.loads((self.path / "upstream-file-manifest.json").read_text())
        records = sorted((row["Path"], row["Sha256"]) for row in manifest["Data"]["Files"])
        identity = hashlib.sha256(json.dumps(records).encode()).hexdigest()
        training_runs(runtime.memory, identity)
        pointer = {"id": "router-0123456789ab", "identity": identity}
        store = self.root / "one/modules/app"
        store.mkdir(parents=True)
        (store / "active-router.json").write_text(json.dumps(pointer))
        body = {"id": "0123456789ab", "scope": "research", "commit": {
            "pointers": [{"name": "active-router.json", "after": pointer}]}}
        with runtime.memory.connect() as db:
            db.execute("INSERT INTO training_runs VALUES (?,?,?,?,?,?)", (
                body["id"], identity, "router", "committed", json.dumps(body), 1.))
        self.ensure.side_effect = ensure_product_state
        with self.assertRaisesRegex(ValueError, "研究候选"):
            runtime.load(paging=False)
        self.allocate.assert_not_called()
        self.assertIsNone(runtime._model_process_lock)
        self.assertEqual(json.loads((store / "active-router.json").read_text()), pointer)

    def test_cancelled_allocation_releases_lock(self):
        first, second = self.runtime("one"), self.runtime("two")
        self.allocate.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            first.load(paging=False)
        self.assertIsNone(first._model_process_lock)
        self.allocate.side_effect = self.model
        second.load(paging=False)

    def test_windows_rejected_explicitly_without_dependency(self):
        with patch("luban.process_lock.os.name", "nt"):
            with self.assertRaisesRegex(RuntimeError, "Windows"):
                ModelProcessLock("0" * 64, "cpu")

    def test_explicit_research_recovery_argument(self):
        runtime = self.runtime("one", research=True)
        runtime.load(paging=False)
        self.ensure.assert_not_called()
        self.recover.assert_called_once_with(runtime.memory, runtime.store, runtime.identity, research=True)

    def test_default_product_recovery_argument(self):
        runtime = self.runtime("one")
        runtime.load(paging=False)
        self.ensure.assert_called_once_with(runtime.memory, runtime.store, runtime.identity)
        self.recover.assert_called_once_with(runtime.memory, runtime.store, runtime.identity, research=False)

    def test_research_declaration_requires_boolean(self):
        with self.assertRaises(ValueError):
            Runtime(Mock(), Mock(), research="research")

    def test_real_subprocess_kill_releases_cross_root_lock(self):
        identity = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        child = """import sys,time
from luban.process_lock import ModelProcessLock
lock=ModelProcessLock(sys.argv[1], 'cpu')
print('locked', flush=True)
time.sleep(30)
"""
        process = subprocess.Popen([sys.executable, "-c", child, identity], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.assertEqual(process.stdout.readline().strip(), b"locked")
        with self.assertRaisesRegex(RuntimeError, "已有模型进程"):
            ModelProcessLock(identity, "cpu")
        process.kill()
        process.communicate(timeout=5)
        self.assertEqual(process.returncode, -9)
        lock = ModelProcessLock(identity, "cpu")
        self.addCleanup(lock.close)

    def test_other_device_is_separate_key(self):
        identity = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        first, second = ModelProcessLock(identity, "cpu"), ModelProcessLock(identity, "mps")
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        self.assertNotEqual(first.key, second.key)


if __name__ == "__main__":
    unittest.main()
