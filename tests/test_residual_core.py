"""CPU-only official tiny Qwen graph; no downloaded weights or model hashes."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from test_cached_training import fixture
from luban.core_update import apply_core_residual
from luban.residual_core import (ResidualProjection, get_residual_core,
    install_residual_core, load_residual_core, save_residual_core, uninstall_residual_core)


IDENTITY = "a" * 64  # Declared random tiny fixture, not the production identity.


class ResidualCoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        fork = torch.random.fork_rng(devices=[])
        fork.__enter__()
        self.addCleanup(fork.__exit__, None, None, None)
        torch.manual_seed(42)
        self.runtime, rows, tensors = fixture(self.folder.name, torch.bfloat16)
        self.model = self.runtime.model.eval().requires_grad_(False)
        self.ids, _ = tensors[rows[0]["id"]]
        self.original = self.runtime.branch.base.down_proj
        self.weight = self.original.weight.detach().clone()
        self.delta = torch.randn(self.weight.shape) * .02

    def install(self, delta=None):
        return install_residual_core(self.model, self.delta if delta is None else delta,
                                     base_identity=IDENTITY, base_core=None)

    def logits(self, model=None):
        with torch.no_grad():
            return (model or self.model)(self.ids, use_cache=False).logits

    def test_zero_full_graph_equivalence_and_uninstall_preserves_original_objects(self):
        baseline = self.logits()
        parameters = list(self.model.parameters())
        residual = self.install(torch.zeros_like(self.delta))
        self.assertIs(residual.base, self.original)
        self.assertIs(residual.base.weight, self.original.weight)
        self.assertIs(get_residual_core(self.model), residual)
        self.assertTrue(torch.equal(self.logits(), baseline))
        self.assertEqual([id(p) for p in self.model.parameters()], [id(p) for p in parameters])
        self.assertIn("delta", residual.state_dict())
        self.assertIs(uninstall_residual_core(self.model), self.original)
        self.assertIs(self.runtime.branch.base.down_proj, self.original)
        self.assertTrue(torch.equal(self.logits(), baseline))
        for action in (get_residual_core, uninstall_residual_core):
            with self.assertRaisesRegex(ValueError, "No residual"):
                action(self.model)

    def test_nonzero_matches_research_math_and_keeps_base_and_delta_owned(self):
        baseline = self.logits()
        handle = self.original.register_forward_hook(
            lambda module, inputs, output: apply_core_residual(inputs[0], output, self.delta))
        try:
            expected = self.logits()
        finally:
            handle.remove()
        residual = self.install()
        self.assertTrue(torch.equal(self.logits(), expected))
        self.assertFalse(torch.equal(expected, baseline))
        self.assertTrue(torch.equal(self.original.weight, self.weight))
        self.assertNotEqual(residual.delta.data_ptr(), self.delta.data_ptr())
        saved = residual.delta.clone()
        self.delta.zero_()
        self.assertTrue(torch.equal(residual.delta, saved))
        self.model.bfloat16().cpu()
        self.assertEqual(residual.delta.dtype, torch.float32)
        self.assertTrue(torch.equal(residual.delta, saved))
        self.assertTrue(torch.equal(self.logits(), expected))
        with self.assertRaisesRegex(ValueError, "BF16 base"):
            residual.float()
        self.assertEqual(self.original.weight.dtype, torch.bfloat16)

    def test_saved_delta_restores_direct_forward_and_generate_on_separately_loaded_graph(self):
        fresh = copy.deepcopy(self.model)
        self.install()
        expected = self.logits()
        with torch.no_grad():
            generated = self.model.generate(self.ids, max_new_tokens=3, do_sample=False)
        path = Path(self.folder.name) / "residual.safetensors"
        save_residual_core(self.model, path)
        with safe_open(str(path), framework="pt") as source:
            self.assertEqual(list(source.keys()), ["delta"])
            self.assertEqual(source.get_tensor("delta").dtype, torch.float32)
            metadata = source.metadata()
            self.assertEqual(metadata["base_identity"], IDENTITY)
            self.assertEqual(metadata["base_core"], "None")
            self.assertEqual(metadata["base_dtype"], "BF16")
            self.assertEqual(metadata["delta_dtype"], "F32")
        uninstall_residual_core(self.model)
        restored = load_residual_core(fresh, path, base_identity=IDENTITY, base_core=None)
        self.assertTrue(torch.equal(restored.delta, self.delta))
        self.assertTrue(torch.equal(self.logits(fresh), expected))
        with torch.no_grad():
            self.assertTrue(torch.equal(fresh.generate(self.ids, max_new_tokens=3, do_sample=False), generated))
        self.assertTrue(torch.equal(self.original.weight, self.weight))
        original_file = path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_residual_core(fresh, path)
        self.assertEqual(path.read_bytes(), original_file)
        failed = path.with_name("failed.safetensors")
        with patch("luban.residual_core._publish_new_file", side_effect=OSError("publish failed")):
            with self.assertRaises(OSError):
                save_residual_core(fresh, failed)
        self.assertFalse(failed.exists())
        self.assertEqual(list(path.parent.glob(".residual-*")), [])

    def test_rejects_invalid_delta_and_base_without_installing(self):
        for bad in (torch.zeros(1, 2), self.delta.bfloat16(), self.delta.double(),
                    torch.full_like(self.delta, float("nan")), torch.full_like(self.delta, float("inf"))):
            with self.subTest(dtype=bad.dtype, shape=bad.shape), self.assertRaises(ValueError):
                self.install(bad)
            self.assertIs(self.runtime.branch.base.down_proj, self.original)
        for identity, core in (("unverified", None), (IDENTITY, "core-b45d965728bb"), (IDENTITY, "None")):
            with self.assertRaises(ValueError):
                install_residual_core(self.model, self.delta, base_identity=identity, base_core=core)
        with self.assertRaises(ValueError):
            ResidualProjection(torch.nn.Linear(64, 32), self.delta,
                               base_identity=IDENTITY, base_core=None)
        bad_base = copy.deepcopy(self.original)
        with torch.no_grad():
            bad_base.weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            ResidualProjection(bad_base, self.delta, base_identity=IDENTITY, base_core=None)
        self.install()
        with self.assertRaisesRegex(ValueError, "already installed"):
            self.install()

    def test_publication_reserves_exclusively_and_cleans_failed_commit(self):
        self.install()
        path = Path(self.folder.name) / "publish.safetensors"
        replace = os.replace
        def observe(temporary, target):
            self.assertEqual(target.read_bytes(), b"")
            with self.assertRaises(SafetensorError):
                with safe_open(str(target), framework="pt"):
                    self.fail("Empty reservation must not load")
            with self.assertRaises(FileExistsError):
                save_residual_core(self.model, target)
            replace(temporary, target)
        with patch("luban.residual_core.os.replace", side_effect=observe):
            save_residual_core(self.model, path)
        with safe_open(str(path), framework="pt") as source:
            self.assertTrue(torch.equal(source.get_tensor("delta"), self.delta))
        failed = path.with_name("failed-commit.safetensors")
        with patch("luban.residual_core.os.replace", side_effect=OSError("commit failed")), self.assertRaises(OSError):
            save_residual_core(self.model, failed)
        self.assertFalse(failed.exists())
        self.assertEqual(list(path.parent.glob(".residual-*")), [])

    def test_loading_rejects_wrong_contract_nonfinite_shape_and_duplicate(self):
        residual = self.install()
        binding = dict(residual.binding)
        uninstall_residual_core(self.model)
        path = Path(self.folder.name) / "bad.safetensors"
        for change in ({"base_identity": "b" * 64}, {"base_core": "core-b45d965728bb"},
                       {"base_dtype": "F32"}, {"delta_dtype": "BF16"}, {"math": "merged"},
                       {"format": "unknown"}):
            save_file({"delta": self.delta}, str(path), metadata={**binding, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_residual_core(self.model, path, base_identity=IDENTITY, base_core=None)
            self.assertIs(self.runtime.branch.base.down_proj, self.original)
        for delta in (self.delta.bfloat16(), torch.zeros(1, 2), torch.full_like(self.delta, float("nan"))):
            save_file({"delta": delta}, str(path), metadata=binding)
            with self.assertRaises(ValueError):
                load_residual_core(self.model, path, base_identity=IDENTITY, base_core=None)
        save_file({"delta": self.delta, "other": torch.zeros(1)}, str(path), metadata=binding)
        with self.assertRaises(ValueError):
            load_residual_core(self.model, path, base_identity=IDENTITY, base_core=None)
        self.install()
        with self.assertRaisesRegex(ValueError, "already installed"):
            load_residual_core(self.model, path, base_identity=IDENTITY, base_core=None)


if __name__ == "__main__":
    unittest.main()
