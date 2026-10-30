"""Offline real Torch/official tiny hybrid Qwen3.5, no downloaded tensors."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_cached_training as cached
from test_field_training_diagnostic import tiny_runtime

REPORTS = []


def fixture(directory, dtype=torch.float32):
    runtime, _, _, _ = tiny_runtime(directory, dtype)
    runtime.identity = "random-official-qwen35-cache-fixture"
    runtime.lock = threading.RLock()
    runtime.job = None
    rows = []
    tensors = {}
    for name, tokens, prefix, role in (
        ("fixed-short-integer", [3, 7, 13, 19, 23, 2], 4, "new"),
        ("fixed-multi-name", [3, 7, 13, 19, 23, 29, 31, 37, 41, 43, 47, 2], 7, "new"),
        ("fixed-long-replay-identity", [3, 7, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53, 55, 57, 2], 7, "replay"),
    ):
        ids = torch.tensor([tokens])
        labels = ids.clone()
        labels[:, :prefix] = -100
        tensors[name] = (ids, labels)
        rows.append({"id": name, "purpose": "training", "provenance": json.dumps({"evolution_role": role}),
                     "answer": "synthetic fixed TRAIN, not real model knowledge"})
    runtime._sample_tokens = lambda sample: tensors[sample["id"]]
    (Path(directory) / "active-core.json").write_bytes(b'{"fixture":"unchanged"}')
    return runtime, rows, tensors


class CachedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.fork = torch.random.fork_rng(devices=[])
        self.fork.__enter__()
        self.addCleanup(self.fork.__exit__, None, None, None)
        torch.manual_seed(42)

    def assert_restored(self, runtime, weight, old, flags, modes, rng, hooks):
        self.assertTrue(torch.equal(weight, old))
        self.assertEqual([p.requires_grad for p in runtime.model.parameters()], [flag for _, flag, _ in flags])
        for parameter, _, gradient in flags:
            self.assertIs(parameter.grad, gradient)
        self.assertEqual([module.training for module in runtime.model.modules()], modes)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual((len(runtime.branch.base.down_proj._forward_pre_hooks),
                          len(runtime.decoder.layers[-1].post_attention_layernorm._forward_pre_hooks),
                          len(weight._backward_hooks or {})), hooks)
        self.assertEqual((Path(runtime.store) / "active-core.json").read_bytes(), b'{"fixture":"unchanged"}')

    def test_real_official_two_step_ce_kl_adam_projection_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                runtime, rows, _ = fixture(self.folder.name, dtype)
                runtime.model.train()
                runtime.decoder.layers[0].eval()
                weight = runtime.branch.base.down_proj.weight
                weight.grad = torch.ones_like(weight)
                flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
                modes = [module.training for module in runtime.model.modules()]
                old, rng = weight.detach().clone(), torch.get_rng_state().clone()
                report = cached.audit_cached_training(runtime, rows)
                REPORTS.append(copy.deepcopy(report))
                self.assertTrue(report["passed"], report)
                self.assertEqual(len(report["steps"]), 2)
                self.assertEqual(report["backward_order"], ["CE.backward", "(5*KL).backward"])
                self.assertEqual(report["retained_cache_entries"], 0)
                self.assertTrue(report["rng_unchanged"])
                self.assertTrue(report["cached_basis_check"]["exact"])
                for step in report["steps"]:
                    for key in ("logits", "replay_logits", "ce_gradient", "weighted_kl_gradient", "accumulated_gradient",
                                "fp32_master", "cumulative_update", "native_updated_weight", "adam_exp_avg", "adam_exp_avg_sq", "adam_step"):
                        self.assertTrue(step["checks"][key]["exact"], (dtype, step["step"], key))
                self.assertGreater(report["steps"][0]["native_preclip_norm"], 0)
                self.assertGreater(report["cache_bytes"], 0)
                for entry, length in zip(report["cache"], (6, 12, 15)):
                    self.assertEqual(entry["shapes"]["x"], [1, length, 64])
                    self.assertEqual(entry["shapes"]["residual"], [1, length, 32])
                    self.assertEqual(entry["shapes"]["labels"], [1, length])
                    self.assertEqual(entry["bytes"]["x"], length * 64 * torch.tensor([], dtype=dtype).element_size())
                    self.assertGreaterEqual(entry["timing"]["to_cpu_seconds"], 0)
                self.assert_restored(runtime, weight, old, flags, modes, rng, (0, 0, 0))

    def test_cache_rejects_input_labels_sample_tensor_source_core_identity_changes(self):
        changes = ("input", "labels", "sample", "x", "source", "core", "pointer", "identity", "prefix", "graph")
        for change in changes:
            with self.subTest(change=change):
                runtime, rows, tensors = fixture(self.folder.name)
                runtime.model.eval().requires_grad_(False)
                weight = runtime.branch.base.down_proj.weight
                weight.requires_grad_(True)
                cache = cached.capture_tail(runtime, rows[0])
                self.assertEqual(set(cache["bytes"]), {"x", "residual", "ids", "labels"})
                if change == "input": tensors[rows[0]["id"]][0][0, 1] = 9
                if change == "labels": tensors[rows[0]["id"]][1][0, -1] = 3
                if change == "sample": rows[0]["answer"] = "changed"
                if change == "x": cache["x"] = cache["x"][:, :-1]
                if change == "source": cache["binding"]["sources"][str(ROOT / "scripts/verify_cached_training.py")] = "0" * 64
                if change == "core": runtime.active_core = "different-core"
                if change == "pointer": (Path(runtime.store) / "active-core.json").write_text('{"core":"changed"}')
                if change == "identity": runtime.identity = "different-base"
                if change == "prefix":
                    with torch.no_grad(): runtime.decoder.layers[0].mlp.down_proj.weight.add_(.1)
                if change == "graph": runtime.decoder.norm = torch.nn.LayerNorm(32)
                with self.assertRaises(ValueError):
                    cached.cached_logits(runtime, rows[0], cache, weight)

    def test_nonzero_kl_uses_original_native_dtype_teacher_and_separate_backward(self):
        proofs = []
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                runtime, rows, _ = fixture(self.folder.name, dtype)
                runtime.model.eval().requires_grad_(False)
                weight = runtime.branch.base.down_proj.weight
                weight.requires_grad_(True)
                original = weight.detach().clone()
                caches = [cached.capture_tail(runtime, row) for row in (rows[0], rows[-1])]
                # An explicit synthetic rejected-candidate perturbation exercises
                # nonzero KL; it is not a trained tensor or projected update.
                perturbation = torch.sin(torch.arange(weight.numel()).reshape(weight.shape)) * .03
                candidate = (original.float() + perturbation).to(dtype)
                ids, labels = runtime._sample_tokens(rows[0])
                replay_ids, replay_labels = runtime._sample_tokens(rows[-1])
                try:
                    with torch.no_grad(): weight.copy_(candidate)
                    teacher_native = cached._teacher_replay(runtime, weight, original, candidate, replay_ids, replay_labels)
                    weight.grad = None
                    runtime.model(ids, labels=labels, use_cache=False).loss.backward()
                    ce_gradient = weight.grad.detach().clone()
                    kl_native = cached.F.kl_div(cached._answer_log_probs(runtime.model(replay_ids, use_cache=False).logits, replay_labels),
                                                teacher_native, reduction="batchmean", log_target=True)
                    (5 * kl_native).backward()
                    accumulated = weight.grad.detach().clone()
                    tail = candidate.clone().requires_grad_(True)
                    logits, _ = cached.cached_logits(runtime, rows[0], caches[0], tail)
                    cached._ce(logits, labels).backward()
                    self.assertTrue(torch.equal(tail.grad, ce_gradient))
                    with torch.no_grad():
                        logits, _ = cached.cached_logits(runtime, rows[-1], caches[-1], original)
                        teacher_tail = cached._answer_log_probs(logits, replay_labels)
                    self.assertTrue(torch.equal(teacher_tail, teacher_native))
                    logits, _ = cached.cached_logits(runtime, rows[-1], caches[-1], tail)
                    kl_tail = cached.F.kl_div(cached._answer_log_probs(logits, replay_labels), teacher_tail,
                                             reduction="batchmean", log_target=True)
                    (5 * kl_tail).backward()
                    self.assertGreater(float(kl_native.detach()), 1e-5)
                    self.assertTrue(torch.equal(kl_tail.detach(), kl_native.detach()))
                    self.assertTrue(torch.equal(tail.grad, accumulated))
                    self.assertTrue(torch.count_nonzero(accumulated - ce_gradient))
                    proofs.append({"dtype": str(dtype), "native_kl": float(kl_native.detach()),
                                   "teacher_exact": True, "accumulated_gradient_exact": True})
                finally:
                    with torch.no_grad(): weight.copy_(original)
                    caches.clear()
        REPORTS.append({"kind": "synthetic_nonzero_kl_probe", "passed": True, "proofs": proofs,
                        "scope": "deliberately perturbed clone, not an Adam/projected update or real model"})

    def test_branch_training_unfrozen_parameters_validation_or_replaced_graph_reject(self):
        for change in ("branch", "train", "unfrozen", "validation", "forward", "target_dtype"):
            with self.subTest(change=change):
                runtime, rows, _ = fixture(self.folder.name)
                runtime.model.eval().requires_grad_(False)
                weight = runtime.branch.base.down_proj.weight
                weight.requires_grad_(True)
                if change == "branch": runtime.branch.expert = torch.nn.Linear(32, 32)
                if change == "train": runtime.decoder.layers[0].train()
                if change == "unfrozen": runtime.decoder.layers[0].mlp.down_proj.weight.requires_grad_(True)
                if change == "validation": rows[0]["purpose"] = "validation"
                if change == "forward": runtime.model.forward = lambda *args, **kwargs: None
                if change != "target_dtype":
                    with self.assertRaises(ValueError): cached.capture_tail(runtime, rows[0])
                else:
                    cache = cached.capture_tail(runtime, rows[0])
                    with self.assertRaises(ValueError): cached.cached_logits(runtime, rows[0], cache, weight.bfloat16())
                self.assertFalse(runtime.branch.base.down_proj._forward_pre_hooks)
                self.assertFalse(runtime.decoder.layers[-1].post_attention_layernorm._forward_pre_hooks)

    def test_hooks_removed_and_original_state_restored_after_midstep_failure(self):
        runtime, rows, _ = fixture(self.folder.name, torch.bfloat16)
        runtime.model.train()
        weight = runtime.branch.base.down_proj.weight
        weight.grad = torch.ones_like(weight)
        flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
        modes = [module.training for module in runtime.model.modules()]
        old, rng = weight.detach().clone(), torch.get_rng_state().clone()
        original = cached.cached_logits
        calls = []
        def failure(*args):
            calls.append(1)
            if len(calls) == 2:
                with torch.no_grad(): weight.add_(.01)
                torch.rand(1)
                raise RuntimeError("injected after native CE+KL")
            return original(*args)
        with patch.object(cached, "cached_logits", side_effect=failure), self.assertRaisesRegex(RuntimeError, "injected"):
            cached.audit_cached_training(runtime, rows)
        self.assert_restored(runtime, weight, old, flags, modes, rng, (0, 0, 0))

    def test_native_failure_capture_hook_removed_and_cancellation_is_safe(self):
        runtime, rows, _ = fixture(self.folder.name)
        runtime.model.eval().requires_grad_(False)
        weight = runtime.branch.base.down_proj.weight
        weight.requires_grad_(True)
        # Malformed vocabulary token fails inside the actual official forward,
        # after this audit's hooks are installed (no fake graph replacement).
        runtime._sample_tokens(rows[0])[0][0, 0] = 100
        with self.assertRaises(IndexError): cached.capture_tail(runtime, rows[0])
        self.assertFalse(runtime.branch.base.down_proj._forward_pre_hooks)
        self.assertFalse(runtime.decoder.layers[-1].post_attention_layernorm._forward_pre_hooks)
        runtime.job = {"cancel_requested": True}
        original = weight.detach().clone()
        with self.assertRaises(RuntimeError): cached.audit_cached_training(runtime, rows)
        self.assertTrue(torch.equal(weight, original))

    def test_deliberate_native_tail_mismatch_is_not_a_speed_or_equivalence_pass(self):
        runtime, rows, _ = fixture(self.folder.name)
        original = cached.cached_logits
        def wrong(*args):
            logits, seconds = original(*args)
            # Change one actual gold predictor; a uniform shift would preserve CE.
            logits = logits.clone()
            logits[0, -2, 2] += .1
            return logits, seconds
        with patch.object(cached, "cached_logits", side_effect=wrong):
            report = cached.audit_cached_training(runtime, rows)
        self.assertFalse(report["passed"])
        self.assertFalse(report["steps"][0]["checks"]["logits"]["passed"])
        self.assertTrue(report["restored"])


if __name__ == "__main__":
    unittest.main()
