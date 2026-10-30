"""CPU-only carrier admission: never allocate the real model or invoke a teacher."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from luban.native_q4 import binding, FORMAT, CORE_LOAD_WORKSPACE_BYTES, GIB
from luban import product_inference


class NativeQ4Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / "model.safetensors").write_bytes(b"original BF16")
        (self.model / "upstream-file-manifest.json").write_text(json.dumps({"Data": {"Files": [
            {"Path": "config.json"}, {"Path": "model.safetensors"}]}}))
        self.binary = self.root / "worker"
        self.binary.write_bytes(b"fake worker")
        self.receipt = {"binary_sha256": "a" * 64, "direct_q4": {"format": FORMAT,
            "qualified": True, "quantizer_sha256": "a" * 64, "weight_bytes": CORE_LOAD_WORKSPACE_BYTES + 512}}
        self.write_receipt()
        self.runtime = SimpleNamespace(native_inference_precision="mixed-q4-g16", loaded_path=str(self.model),
            loaded_storage=str(self.root), identity="b" * 64, active_core="core-000000000001")
        self.converter = patch("luban.native_q4.NativeEngine.prepare_direct_q4", side_effect=self.prepare)
        self.prepare_mock = self.converter.start()
        self.addCleanup(self.converter.stop)
        memory = patch("luban.native_q4.available_memory_bytes", return_value=4 * GIB)
        memory.start()
        self.addCleanup(memory.stop)

    def write_receipt(self):
        Path(str(self.binary) + ".json").write_text(json.dumps(self.receipt))

    def prepare(self, path, quantizer, stop):
        self.assertFalse(stop.is_set())
        path.write_bytes(b"direct carrier")
        return {"direct_q4_sha256": "d" * 64, "direct_q4_weight_bytes": 512}

    def approve(self):
        return binding(self.runtime, self.binary, prepare=True, stop=threading.Event())

    def test_only_verified_carrier_gets_q4_cold_budget(self):
        self.assertIsNone(binding(self.runtime, self.binary))
        direct = self.approve()
        self.assertEqual(direct["weight_bytes"], 512)
        self.assertEqual(direct["cold_weight_bytes"], CORE_LOAD_WORKSPACE_BYTES + 512)
        self.assertEqual(binding(self.runtime, self.binary), direct)
        self.assertEqual(self.prepare_mock.call_count, 1)
        with patch("luban.product_inference.native_binary", return_value=self.binary):
            self.assertEqual(product_inference.weight_bytes(self.runtime), direct["cold_weight_bytes"])
        # A new process reuses the approved stat and SHA; it does not hash the carrier in Python.
        self.runtime._native_q4_approval = None
        with patch("luban.native_q4.hashlib.sha256", wraps=hashlib.sha256) as digest:
            self.assertEqual(binding(self.runtime, self.binary), direct)
        self.assertEqual(digest.call_count, 1)  # manifest lifecycle binding only
        self.assertEqual(self.prepare_mock.call_count, 1)

    def test_core_changes_keep_base_cache_but_engine_still_gets_current_core(self):
        first = self.approve()
        self.runtime.active_core = "core-000000000002"
        self.assertEqual(binding(self.runtime, self.binary), first)
        self.assertEqual(self.prepare_mock.call_count, 1)

    def test_unqualified_and_other_precisions_keep_full_weight_budget(self):
        self.receipt["direct_q4"]["qualified"] = False
        self.write_receipt()
        self.assertIsNone(self.approve())
        self.prepare_mock.assert_not_called()
        self.runtime.decoder = torch.nn.Linear(4, 4, bias=False, device="meta", dtype=torch.bfloat16)
        with patch("luban.product_inference.native_binary", return_value=self.binary):
            self.assertEqual(product_inference.weight_bytes(self.runtime), 32)
        self.runtime.native_inference_precision = "mixed-q8"
        self.assertIsNone(self.approve())

    def test_capability_implementation_cannot_borrow_another_binary(self):
        self.receipt["direct_q4"]["quantizer_sha256"] = "c" * 64
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "转换实现"):
            self.approve()
        self.prepare_mock.assert_not_called()

    def test_packaged_binary_keeps_original_quantizer_binding(self):
        self.receipt["binary_sha256"] = "c" * 64
        self.receipt["packaging"] = {"source_binary_sha256": "a" * 64}
        self.write_receipt()
        self.assertEqual(self.approve()["quantizer_sha256"], "a" * 64)

    def test_source_change_fails_before_reusing_or_preparing(self):
        self.approve()
        (self.model / "model.safetensors").write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "来源发生变化"):
            self.approve()
        self.assertEqual(self.prepare_mock.call_count, 1)

    def test_carrier_change_or_loss_cannot_reuse_old_approval(self):
        direct = self.approve()
        path = Path(direct["file"])
        path.write_bytes(b"damaged carrier")
        with self.assertRaisesRegex(ValueError, "损坏"):
            binding(self.runtime, self.binary)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "完整批准收据"):
            self.approve()

    def test_symlinks_are_rejected(self):
        direct = self.approve()
        path = Path(direct["file"])
        path.unlink()
        path.symlink_to(self.model / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "普通文件"):
            binding(self.runtime, self.binary)

    def test_prepare_failure_removes_only_new_temporary(self):
        def cancel(path, quantizer, stop):
            path.write_bytes(b"partial")
            raise RuntimeError("cancelled")
        self.prepare_mock.side_effect = cancel
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.approve()
        self.assertFalse(list((self.root / "runtime/direct-q4").iterdir()))
        self.assertEqual((self.model / "model.safetensors").read_bytes(), b"original BF16")

    def test_prepare_retains_system_reserve_and_rejects_oversize_actual_payload(self):
        with patch("luban.native_q4.available_memory_bytes", return_value=GIB):
            with self.assertRaisesRegex(ValueError, "内存不足"):
                self.approve()
        self.prepare_mock.assert_not_called()
        def oversize(path, quantizer, stop):
            path.write_bytes(b"carrier")
            return {"direct_q4_sha256": "d" * 64, "direct_q4_weight_bytes": 513}
        self.prepare_mock.side_effect = oversize
        with self.assertRaisesRegex(ValueError, "内存"):
            self.approve()
        self.assertFalse(list((self.root / "runtime/direct-q4").iterdir()))
