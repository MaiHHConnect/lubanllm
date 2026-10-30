import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from luban.original_protection import protection_rows, read_original_core_protection, _tensor_sha256


class OriginalProtectionTests(unittest.TestCase):
    def make_runtime(self):
        root = Path(tempfile.mkdtemp())
        tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        store = root / "modules" / "luban0-v1"
        store.mkdir(parents=True)
        runtime = SimpleNamespace(store=store, identity="base", _base_core_weight=tensor)
        samples = [{
            "id": f"product-ordinary-{index}", "prompt": f"ordinary-{index}", "answer": "",
            "provenance": {"messages": [{"role": "user", "content": f"ordinary-{index}"}],
                           "generation_check": {"type": "exact", "expected": "4", "max_new_tokens": 8}},
        } for index in range(8)]
        payload = {"schema_version": 1, "protocol": "original-core-protection-v1",
                   "identity": "base", "original_tensor_sha256": _tensor_sha256(tensor),
                   "passed": True, "cases": samples,
                   "generation": [{"id": sample["id"], "answer": "4", "passed": True,
                                   "complete": True, "branch_enabled": False} for sample in samples]}
        (store / "original-core-protection.json").write_text(json.dumps(payload))
        return runtime

    def test_read_and_adapt_baseline_to_read_only_rows(self):
        runtime = self.make_runtime()
        self.assertTrue(read_original_core_protection(runtime)["passed"])
        rows = protection_rows(runtime)
        self.assertEqual(len(rows), 8)
        self.assertTrue(rows[0]["provenance"])
        provenance = json.loads(rows[0]["provenance"])
        self.assertTrue(provenance["retention_generation_only"])
        self.assertFalse(rows[0]["training_allowed"])

    def test_tensor_mismatch_fails_closed(self):
        runtime = self.make_runtime()
        runtime._base_core_weight[0, 0] = 9
        with self.assertRaisesRegex(ValueError, "原始核心保护基线与当前原始张量不匹配"):
            read_original_core_protection(runtime)


if __name__ == "__main__":
    unittest.main()
