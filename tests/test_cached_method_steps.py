"""Continuous clone-Adam G1 step CPU tests on official random tiny Qwen3.5."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cached_method_steps as method
import test_cached_batch_training as batch_fixture
import test_cached_training as existing

REPORTS = []


def prepared(folder, dtype):
    runtime, new, replay, tensors = batch_fixture.fixture(folder, dtype)
    runtime.model.eval().requires_grad_(False)
    parameter = runtime.branch.base.down_proj.weight
    parameter.requires_grad_(True)
    reference = parameter.detach().float().clone()
    caches = {row["id"]: method.cached.capture_tail(runtime, row) for row in new + replay}
    basis, _ = method.cached._capture_replay_subspace(runtime, parameter, replay)
    return runtime, new, replay, tensors, reference, basis, caches


def uninterrupted_step(runtime, sample, replay, master, optimizer, reference, basis):
    parameter = runtime.branch.base.down_proj.weight
    with torch.no_grad(): parameter.copy_(master)
    ids, labels = runtime._sample_tokens(sample)
    replay_ids, replay_labels = runtime._sample_tokens(replay)
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = None
    teacher = method.cached._teacher_replay(runtime, parameter, reference.to(parameter.dtype), parameter.detach().clone(), replay_ids, replay_labels)
    loss = runtime.model(ids, labels=labels, use_cache=False).loss
    loss.backward()
    logits = runtime.model(replay_ids, use_cache=False).logits
    kl = method.F.kl_div(method.cached._answer_log_probs(logits, replay_labels), teacher, reduction="batchmean", log_target=True)
    (5 * kl).backward()
    master.grad = parameter.grad.detach().float().clone()
    torch.nn.utils.clip_grad_norm_([master], 1, error_if_nonfinite=True)
    optimizer.step()
    method.cached._project_core_delta(master, reference, basis)
    with torch.no_grad(): parameter.copy_(master)
    return float(loss.detach()), float(kl.detach())


class CachedMethodStepTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(42)

    def test_b1_native_and_cached_eight_steps_equal_old_uninterrupted_adam(self):
        for dtype in (torch.float32, torch.bfloat16):
            runtime, new, replay, _, reference, basis, caches = prepared(self.folder.name, dtype)
            original = runtime.branch.base.down_proj.weight.detach().clone()
            paths = {name: torch.nn.Parameter(reference.clone()) for name in ("old", "native", "cache")}
            optimizers = {name: torch.optim.Adam([master], lr=1e-5, eps=1e-6) for name, master in paths.items()}
            trace = []
            for step in range(8):
                role = "new" if step % 2 == 0 else "replay"
                sample = (new if role == "new" else replay)[(step // 2) % (len(new) if role == "new" else len(replay))]
                replay_row = replay[(step // 2) % len(replay)]
                expected = uninterrupted_step(runtime, sample, replay_row, paths["old"], optimizers["old"], reference, basis)
                with torch.no_grad(): runtime.branch.base.down_proj.weight.copy_(original)
                for name in ("native", "cache"):
                    report = method.train_method_step(runtime, [sample], [replay_row], paths[name], optimizers[name], reference, basis,
                                                      method=name, ce_role=role, caches=caches)
                    self.assertEqual((report["ce"], report["kl"]), expected)
                    self.assertTrue(torch.equal(paths[name], paths["old"]), (dtype, step, name))
                    for key in ("exp_avg", "exp_avg_sq", "step"):
                        self.assertTrue(torch.equal(optimizers[name].state[paths[name]][key], optimizers["old"].state[paths["old"]][key]))
                    self.assertTrue(torch.equal(runtime.branch.base.down_proj.weight, original))
                    self.assertTrue(report["runtime_restored"])
                    trace.append(report)
            self.assertEqual(sum(row["new_ce_exposures"] for row in trace), 8)
            self.assertEqual(sum(row["replay_ce_exposures"] for row in trace), 8)
            self.assertEqual(sum(row["replay_kl_exposures"] for row in trace), 16)
            REPORTS.append({"kind": "B1_continuous_eight_step_tiny_exact", "dtype": str(dtype), "native_and_cache_exact": True, "trace": copy.deepcopy(trace),
                            "scope": "Actual master/moments independently persisted across8steps; runtime restored between helper calls"})

    def test_cached_b2_b4_same_role_batches_continuous_state_three_exposure_ledgers(self):
        for dtype in (torch.float32, torch.bfloat16):
            for size in (2, 4):
                runtime, new, replay, _, reference, basis, caches = prepared(self.folder.name, dtype)
                master = torch.nn.Parameter(reference.clone())
                optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
                trace = []
                # CE events within each role stay ordered. This fixture flushes
                # at its explicit end; real checkpoint scheduling is caller-owned.
                for start in range(0, len(new), size):
                    actual = len(new[start:start + size])
                    replay_group = [replay[(start + index) % len(replay)] for index in range(actual)]
                    for role, ce_group in (("new", new[start:start + size]), ("replay", replay_group)):
                        trace.append(method.train_method_step(runtime, ce_group, replay_group, master, optimizer, reference, basis,
                                                              method="cache", ce_role=role, caches=caches))
                self.assertEqual(sum(row["new_ce_exposures"] for row in trace), 5)
                self.assertEqual(sum(row["replay_ce_exposures"] for row in trace), 5)
                self.assertEqual(sum(row["replay_kl_exposures"] for row in trace), 10)
                self.assertEqual(int(optimizer.state[master]["step"]), 6 if size == 2 else 4)
                self.assertTrue(torch.isfinite(master).all())
                self.assertTrue(all(row["timing"]["step_elapsed_seconds"] > 0 and row["timing"]["transfer_seconds"] >= 0 for row in trace))
                self.assertTrue(all(row["checks_run"] is False for row in trace))
                REPORTS.append({"kind": "cached_same_role_method", "dtype": str(dtype), "B": size, "trace": copy.deepcopy(trace), "sequential_adam_equivalence_claimed": False})

    def test_bad_roles_eos_optimizer_inputs_and_native_batches_reject(self):
        for change in ("role", "eos", "nativeB2", "lr", "reference", "basis", "input"):
            with self.subTest(change=change):
                runtime, new, replay, tensors, reference, basis, caches = prepared(self.folder.name, torch.float32)
                master = torch.nn.Parameter(reference.clone())
                optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
                original = runtime.branch.base.down_proj.weight.detach().clone()
                if change == "eos": tensors[new[0]["id"]][0][0, -1] = 4
                if change == "lr": optimizer.param_groups[0]["lr"] = 1e-4
                if change == "reference": reference[0, 0] = float("nan")
                if change == "basis": basis = basis[:, :0]
                if change == "input": tensors[new[0]["id"]][0][0, 1] = 8
                with self.assertRaises(ValueError):
                    method.train_method_step(runtime, new[:2] if change == "nativeB2" else new[:1], replay[:2] if change == "nativeB2" else replay[:1],
                                             master, optimizer, reference, basis, method="native" if change == "nativeB2" else "cache",
                                             ce_role="replay" if change == "role" else "new", caches=caches)
                self.assertTrue(torch.equal(runtime.branch.base.down_proj.weight, original))

    def test_original_reference_teacher_not_runtime_candidate(self):
        runtime, new, replay, _, reference, basis, caches = prepared(self.folder.name, torch.bfloat16)
        parameter = runtime.branch.base.down_proj.weight
        with torch.no_grad(): parameter.add_(.03)
        before = parameter.detach().clone()
        master = torch.nn.Parameter((reference + .03).clone())
        optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
        report = method.train_method_step(runtime, new[:1], replay[:1], master, optimizer, reference, basis, method="cache", ce_role="new", caches=caches)
        self.assertGreater(report["kl"], 1e-6)
        self.assertTrue(torch.equal(parameter, before))
        self.assertFalse(torch.equal(parameter.float(), reference))

    def test_midstep_cancel_and_after_optimizer_failure_restore_runtime(self):
        for phase in ("cancel", "projection"):
            runtime, new, replay, _, reference, basis, caches = prepared(self.folder.name, torch.bfloat16)
            runtime.model.train()
            parameter = runtime.branch.base.down_proj.weight
            parameter.grad = torch.ones_like(parameter)
            original = parameter.detach().clone()
            flags = [(p, p.requires_grad, p.grad) for p in runtime.model.parameters()]
            modes = [m.training for m in runtime.model.modules()]
            rng = torch.get_rng_state().clone()
            master = torch.nn.Parameter(reference.clone())
            optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
            def fail(*args, **kwargs):
                torch.rand(1)
                if phase == "cancel":
                    runtime.job = {"cancel_requested": True}
                    method.cached._cancel(runtime)
                raise RuntimeError("injected clone projection failure")
            target = "_tail" if phase == "cancel" else "cached._project_core_delta"
            with patch("cached_method_steps." + target, side_effect=fail), self.assertRaises(RuntimeError):
                method.train_method_step(runtime, new[:1], replay[:1], master, optimizer, reference, basis, method="cache", ce_role="new", caches=caches)
            existing.CachedTrainingTests.assert_restored(self, runtime, parameter, original, flags, modes, rng, (0, 0, 0))

    def test_nonfinite_or_bad_adam_durable_state_rejects_before_or_after_update(self):
        for change in ("before-moment", "after-moment", "after-step", "before-shape"):
            with self.subTest(change=change):
                runtime, new, replay, _, reference, basis, caches = prepared(self.folder.name, torch.bfloat16)
                master = torch.nn.Parameter(reference.clone())
                optimizer = torch.optim.Adam([master], lr=1e-5, eps=1e-6)
                # Materialize genuine existing Adam state, not a forged dict.
                method.train_method_step(runtime, new[:1], replay[:1], master, optimizer, reference, basis, method="cache", ce_role="new", caches=caches)
                original = runtime.branch.base.down_proj.weight.detach().clone()
                if change == "before-moment": optimizer.state[master]["exp_avg"][0, 0] = float("nan")
                if change == "before-shape": optimizer.state[master]["exp_avg"] = torch.zeros(1)
                actual_step = optimizer.step
                def corrupt(*args, **kwargs):
                    result = actual_step(*args, **kwargs)
                    if change == "after-moment": optimizer.state[master]["exp_avg"][0, 0] = float("nan")
                    if change == "after-step": optimizer.state[master]["step"].add_(1)
                    return result
                with patch.object(optimizer, "step", side_effect=corrupt), self.assertRaises(ValueError):
                    method.train_method_step(runtime, replay[:1], replay[:1], master, optimizer, reference, basis, method="cache", ce_role="replay", caches=caches)
                self.assertTrue(torch.equal(runtime.branch.base.down_proj.weight, original))
                self.assertFalse(runtime.branch.base.down_proj._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
