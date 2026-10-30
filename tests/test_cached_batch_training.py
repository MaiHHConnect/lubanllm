"""Actual official tiny Qwen3.5 batch tail CPU guards; no external tensors."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_cached_batch_training as batch
import test_cached_training as existing

REPORTS = []


def fixture(folder, dtype):
    runtime, original_rows, tensors = existing.fixture(folder, dtype)
    new, replay = original_rows[:2], original_rows[2:]
    for role, rows, lengths in (("new", new, (7, 9, 11)), ("replay", replay, (10, 13))):
        for index, length in enumerate(lengths):
            name = f"synthetic-{role}-{index}"
            ids = torch.tensor([[3 + (i * 7 + index) % 54 for i in range(length - 1)] + [2]])
            labels = ids.clone()
            labels[:, :length // 2] = -100
            tensors[name] = (ids, labels)
            rows.append({"id": name, "purpose": "training", "answer": "synthetic controlled TRAIN",
                         "provenance": json.dumps({"evolution_role": role})})
    return runtime, new, replay, tensors


class CachedBatchTrainingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(42)

    def test_b1_b2_b4_actual_fp32_bf16_gradient_means_and_remainder(self):
        for dtype in (torch.float32, torch.bfloat16):
            for size in (1, 2, 4):
                with self.subTest(dtype=dtype, size=size):
                    runtime, new, replay, _ = fixture(self.folder.name, dtype)
                    runtime.model.train()
                    weight = runtime.branch.base.down_proj.weight
                    weight.grad = torch.ones_like(weight)
                    old = weight.detach().clone()
                    flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
                    modes = [module.training for module in runtime.model.modules()]
                    rng = torch.get_rng_state().clone()
                    result = batch.audit_cached_batch(runtime, new, replay, size)
                    REPORTS.append(copy.deepcopy(result))
                    checks = [value for row in result["batches"] for value in list(row["checks"].values()) + row["original_teacher_checks"]]
                    self.assertEqual(result["passed"], all(value["passed"] for value in checks))
                    self.assertTrue(all(value["finite"] for value in checks))
                    for row in result["batches"]:
                        for key in ("mean_ce_gradient", "mean_weighted_kl_gradient", "total_ce_5kl_gradient"):
                            self.assertTrue(row["checks"][key]["standard_allclose_diagnostic"], row)
                    # Real strict/update failures remain failures. Diagnostic
                    # allclose is never used to turn the report into a PASS.
                    self.assertEqual(result["passed"], dtype == torch.bfloat16 and size == 1)
                    self.assertTrue(result["restored"])
                    self.assertEqual(result["retained_cache_entries"], 0)
                    self.assertEqual([item for row in result["batches"] for item in row["new_ids"]], [row["id"] for row in new])
                    self.assertEqual([row["batch_size_actual"] for row in result["batches"]], {1: [1]*5, 2: [2,2,1], 4: [4,1]}[size])
                    self.assertTrue(all(len(row["new_ids"]) == len(row["replay_ids"]) for row in result["batches"]))
                    existing.CachedTrainingTests.assert_restored(self, runtime, weight, old, flags, modes, rng, (0, 0, 0))

    def test_per_question_ce_keeps_question_weight_not_global_token_mean(self):
        labels = torch.tensor([[-100, 1, 2, -100, -100], [-100, 2, 3, 1, 2]])
        logits = torch.tensor([[[1., 4., -2., 0.]]*5, [[2., -1., 3., 0.]]*5], requires_grad=True)
        actual = batch.question_ce(logits, labels)
        expected = torch.stack([batch.cached._ce(logits[index:index+1], labels[index:index+1]) for index in range(2)])
        self.assertTrue(torch.equal(actual, expected))
        self.assertNotAlmostEqual(float(actual.mean().detach()), float(batch.cached._ce(logits, labels).detach()), places=4)
        changed = logits.detach().clone()
        changed[0, 2:] = float("nan")
        # Nonfinite padding is rejected rather than hidden by its -100 labels.
        with self.assertRaises(ValueError): batch.question_ce(changed, labels)

    def test_nonzero_kl_mean_matches_native_independent_gradients_at_same_weight(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                runtime, new, replay, _ = fixture(self.folder.name, dtype)
                runtime.model.eval().requires_grad_(False)
                weight = runtime.branch.base.down_proj.weight
                weight.requires_grad_(True)
                original = weight.detach().clone()
                caches = [batch.cached.capture_tail(runtime, row) for row in replay[:2]]
                candidate = (original.float() + .03 * torch.sin(torch.arange(weight.numel()).reshape(weight.shape))).to(dtype)
                try:
                    with torch.no_grad(): weight.copy_(candidate)
                    _, _, native_losses, native_gradient, teachers = batch._native_means(runtime, new[:1], replay[:2], original, candidate)
                    tail = candidate.clone().requires_grad_(True)
                    logits, labels = batch._pack(runtime, replay[:2], caches, tail)
                    losses = batch.question_kl(logits, labels, teachers)
                    (5 * losses.mean()).backward()
                    tolerance = batch.cached.TOLERANCES[str(dtype)]
                    proof = batch._check(tail.grad.float(), native_gradient, tolerance, "gradient")
                    self.assertGreater(float(native_losses.mean()), 1e-6)
                    self.assertTrue(proof["standard_allclose_diagnostic"], proof)
                    self.assertTrue(batch._check(losses, native_losses, tolerance, "loss")["standard_allclose_diagnostic"])
                    REPORTS.append({"kind": "separate_nonzero_batch_kl_probe", "dtype": str(dtype),
                                    "native_kl": float(native_losses.mean()), "gradient": proof,
                                    "scope": "same-W synthetic perturbation, not learned weights"})
                finally:
                    with torch.no_grad(): weight.copy_(original)
                    for cache in caches: cache.clear()

    def test_malformed_gold_eos_masks_roles_and_batch_sizes_reject(self):
        for change in ("eos", "label-hole", "wrong-gold", "prefix", "purpose", "role", "duplicate", "size", "bool"):
            with self.subTest(change=change):
                runtime, new, replay, tensors = fixture(self.folder.name, torch.float32)
                ids, labels = tensors[new[0]["id"]]
                if change == "eos": ids[0, -1] = labels[0, -1] = 4
                if change == "label-hole": labels[0, -1] = -100
                if change == "wrong-gold": labels[0, -2] = 4
                if change == "prefix": labels[0, 0] = ids[0, 0]
                if change == "purpose": new[0]["purpose"] = "validation"
                if change == "role": new[0]["provenance"] = '{"evolution_role":"replay"}'
                if change == "duplicate": new[1]["id"] = new[0]["id"]
                size = 8 if change == "size" else True if change == "bool" else 2
                old = runtime.branch.base.down_proj.weight.detach().clone()
                with self.assertRaises(ValueError): batch.audit_cached_batch(runtime, new, replay, size)
                self.assertTrue(torch.equal(runtime.branch.base.down_proj.weight, old))
                self.assertFalse(runtime.branch.base.down_proj._forward_pre_hooks)

    def test_pack_rejects_changed_input_source_tensor_and_teacher_mask(self):
        for change in ("input", "source", "x", "teacher"):
            with self.subTest(change=change):
                runtime, new, replay, tensors = fixture(self.folder.name, torch.float32)
                runtime.model.eval().requires_grad_(False)
                weight = runtime.branch.base.down_proj.weight
                weight.requires_grad_(True)
                caches = [batch.cached.capture_tail(runtime, row) for row in new[:2]]
                if change == "input": tensors[new[0]["id"]][0][0, 1] = 8
                if change == "source": caches[0]["binding"]["sources"][str(ROOT / "scripts/verify_cached_training.py")] = "0"*64
                if change == "x": caches[0]["x"][0, 0, 0] = float("nan")
                with self.assertRaises(ValueError):
                    logits, labels = batch._pack(runtime, new[:2], caches, weight)
                    batch.question_kl(logits, labels, [torch.zeros(1, 64)])

    def test_cancel_and_mid_batch_failure_restore_weights_flags_rng_and_hooks(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                runtime, new, replay, _ = fixture(self.folder.name, torch.bfloat16)
                runtime.model.train()
                weight = runtime.branch.base.down_proj.weight
                weight.grad = torch.ones_like(weight)
                flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
                modes = [m.training for m in runtime.model.modules()]
                old, rng = weight.detach().clone(), torch.get_rng_state().clone()
                original_pack = batch._pack
                calls = []
                def fail(*args):
                    calls.append(1)
                    if len(calls) == 2:
                        torch.rand(1)
                        if cancel:
                            runtime.job = {"cancel_requested": True}
                            batch.cached._cancel(runtime)
                        with torch.no_grad(): weight.add_(.01)
                        raise RuntimeError("injected batch failure")
                    return original_pack(*args)
                with patch.object(batch, "_pack", side_effect=fail), self.assertRaises(RuntimeError):
                    batch.audit_cached_batch(runtime, new, replay, 2)
                existing.CachedTrainingTests.assert_restored(self, runtime, weight, old, flags, modes, rng, (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
