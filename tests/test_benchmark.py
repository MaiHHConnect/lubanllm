import copy
import hashlib
import json
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from luban.benchmark import (BenchmarkStore, POLICY, POLICY_HASH, SUITE_ID, SUITE_SHA256,
                             model_version, run_benchmark, _StopRequested)
from luban.coldpaging import full_backbone


class TinyGenerator(nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.runtime = runtime
        self.calls = []
        self.mode = "normal"

    def generate(self, tokens, **kwargs):
        self.calls.append(kwargs)
        assert self.runtime.branch.expert is None
        assert self.runtime.state == "evaluating"
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 512
        if self.mode == "error":
            raise RuntimeError("actual generator error")
        if self.mode == "change":
            self.runtime.identity = "b" * 64
        if self.mode == "cancel":
            kwargs["stopping_criteria"][0].event.set()
        generated = ([] if self.mode == "empty" else [5, 9] if self.mode == "secondary_eos"
                     else [5, 2] if self.mode != "length" else [5] * 512)
        return torch.cat((tokens, torch.tensor([generated], dtype=torch.long)), dim=1)


class Tokenizer:
    eos_token_id = 2

    def decode(self, tokens, skip_special_tokens):
        return "A" if tokens.numel() else ""


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime = SimpleNamespace(lock=threading.RLock(), identity="a" * 64,
            active_core=None, store=Path(self.tmp.name), state="ready", training_session=None,
            branch=SimpleNamespace(expert=object()), tokenizer=Tokenizer(), device="cpu",
            dtype=torch.float32, cold_backbone=None)
        self.original_expert = self.runtime.branch.expert
        self.messages = []
        def tokens(messages, thinking, output_reserve=0):
            assert output_reserve == 512
            self.messages.append((messages, thinking))
            return torch.tensor([[1, 3]])
        self.runtime._tokens = tokens
        self.runtime._ready = lambda: None
        self.runtime.model = TinyGenerator(self.runtime)
        self.runtime.inference_identity = lambda: {"engine": "hf"}
        def generate_tokens(tokens, maximum, *, temperature=0.0, emit=None, stop=None):
            with full_backbone(self.runtime):
                return self.runtime.model.generate(tokens, attention_mask=torch.ones_like(tokens),
                    max_new_tokens=maximum, do_sample=temperature > 0,
                    pad_token_id=self.runtime.tokenizer.eos_token_id,
                    stopping_criteria=[_StopRequested(stop)])
        self.runtime.generate_tokens = generate_tokens
        self.challenge = {"challenge_id": "engineering-test", "suite_id": SUITE_ID,
            "suite_sha256": SUITE_SHA256, "policy": copy.deepcopy(POLICY), "policy_hash": POLICY_HASH,
            "model_version": model_version(self.runtime), "questions": [
                {"id": f"question-{i}", "category": "engineering", "prompt": "Choose A."}
                for i in range(48)]}

    def test_core_only_exact_policy_readonly_and_rng_restoration(self):
        rng = torch.get_rng_state().clone()
        before = self.runtime.model.weight.detach().clone()
        progress = []
        result = run_benchmark(self.runtime, self.challenge, progress=progress.append)
        self.assertEqual(len(result["answers"]), 48)
        self.assertTrue(all(row["finish_reason"] == "stop" for row in result["answers"]))
        self.assertEqual(progress[-1], {"completed": 48, "total": 48})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(before, self.runtime.model.weight))
        self.assertEqual(result["metadata"]["teacher_calls"], 0)
        self.assertEqual(result["metadata"]["route"], "local_core")
        self.assertNotIn("benchmark", result)
        self.assertTrue(all(thinking == "off" for _, thinking in self.messages))
        self.assert_restored()

    def test_engine_change_invalidates_previous_challenge(self):
        self.runtime.inference_identity = lambda: {"engine": "luban-native", "binary_sha256": "f" * 64}
        self.assertNotEqual(model_version(self.runtime), self.challenge["model_version"])
        with self.assertRaises(ValueError):
            run_benchmark(self.runtime, self.challenge)
        self.assertEqual(self.runtime.model.calls, [])

    def test_native_helper_does_not_acquire_hf_backbone(self):
        self.runtime.cold_backbone = SimpleNamespace(acquire=lambda: self.fail("HF weights loaded"), release=lambda: None)
        self.runtime.generate_tokens = lambda tokens, maximum, **kwargs: torch.cat((tokens, torch.tensor([[5, 2]])), dim=1)
        result = run_benchmark(self.runtime, self.challenge)
        self.assertEqual(len(result["answers"]), 48)

    def assert_restored(self):
        self.assertEqual(self.runtime.state, "ready")
        self.assertIs(self.runtime.branch.expert, self.original_expert)
        self.assertTrue(self.runtime.model.training)

    def test_truncated_output_is_preserved_not_mislabelled_stop(self):
        self.runtime.model.mode = "length"
        result = run_benchmark(self.runtime, self.challenge)
        self.assertTrue(all(row["finish_reason"] == "length" for row in result["answers"]))
        self.assert_restored()

    def test_reserved_worker_state_is_preserved(self):
        self.runtime.state = "evaluating"
        result = run_benchmark(self.runtime, self.challenge)
        self.assertEqual(len(result["answers"]), 48)
        self.assertEqual(self.runtime.state, "evaluating")
        self.assertIs(self.runtime.branch.expert, self.original_expert)

    def test_model_stop_tokens_override_tokenizer_single_eos(self):
        self.runtime.model.generation_config = SimpleNamespace(eos_token_id=[2, 9])
        self.runtime.model.mode = "secondary_eos"
        result = run_benchmark(self.runtime, self.challenge)
        self.assertTrue(all(row["finish_reason"] == "stop" for row in result["answers"]))

    def test_paged_context_releases_on_generation_error(self):
        class Pager:
            depth_count = 0
            def acquire(self):
                self.depth_count += 1
            def release(self):
                self.depth_count -= 1
        pager = Pager()
        self.runtime.cold_backbone = pager
        self.runtime.model.mode = "error"
        with self.assertRaises(RuntimeError):
            run_benchmark(self.runtime, self.challenge)
        self.assertEqual(pager.depth_count, 0)
        self.assert_restored()

    def test_abort_modes_restore_state_and_branch(self):
        for mode, error in (("error", RuntimeError), ("empty", ValueError),
                            ("cancel", InterruptedError), ("change", ValueError)):
            with self.subTest(mode=mode):
                self.runtime.identity = "a" * 64
                self.runtime.model.mode = mode
                with self.assertRaises(error):
                    run_benchmark(self.runtime, self.challenge)
                self.assert_restored()

    def test_cancel_before_first_generation(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            run_benchmark(self.runtime, self.challenge, stop=stop)
        self.assertEqual(self.runtime.model.calls, [])
        self.assert_restored()

    def test_busy_model_rejected(self):
        self.runtime.state = "learning"
        with self.assertRaises(RuntimeError):
            run_benchmark(self.runtime, self.challenge)
        self.assertEqual(self.runtime.model.calls, [])

    def test_runtime_storage_and_loader_readiness_guard_is_reused(self):
        def moved_storage():
            raise RuntimeError("存储目录已修改，请重新加载模型")
        self.runtime._ready = moved_storage
        with self.assertRaisesRegex(RuntimeError, "存储目录"):
            run_benchmark(self.runtime, self.challenge)
        self.assertEqual(self.runtime.model.calls, [])
        self.assert_restored()

    def test_policy_questions_and_version_fail_closed(self):
        cases = []
        for key, value in (("suite_id", "unknown"), ("policy_hash", "a" * 64),
                           ("suite_sha256", "bad"), ("model_version", "bad"),
                           ("questions", self.challenge["questions"][:-1])):
            cases.append({**self.challenge, key: value})
        cases.append({**self.challenge, "policy": {**POLICY, "teacher_assistance": True}})
        cases.append({**self.challenge, "policy": {**POLICY, "teacher_assistance": 0}})
        duplicate = copy.deepcopy(self.challenge)
        duplicate["questions"][1]["id"] = duplicate["questions"][0]["id"]
        cases.append(duplicate)
        gold = copy.deepcopy(self.challenge)
        gold["questions"][0]["answer"] = "A"
        cases.append(gold)
        for challenge in cases:
            with self.subTest(challenge=challenge.get("suite_id")):
                with self.assertRaises(ValueError):
                    run_benchmark(self.runtime, challenge)
        self.assertEqual(self.runtime.model.calls, [])

    def test_core_fingerprint_binds_actual_weight_and_pointer(self):
        active = "core-0123456789ab"
        self.runtime.active_core = active
        weights = self.runtime.store / (active + ".safetensors")
        weights.write_bytes(b"engineering-test-core-not-production")
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        metadata = {"version": active, "identity": self.runtime.identity, "sha256": sha}
        (self.runtime.store / "active-core.json").write_text(json.dumps(metadata))
        (self.runtime.store / (active + ".json")).write_text(json.dumps(metadata))
        version = model_version(self.runtime)
        self.assertNotEqual(version, self.challenge["model_version"])
        weights.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            model_version(self.runtime)

    def test_in_memory_and_disk_pointer_mismatch_rejected(self):
        (self.runtime.store / "active-core.json").write_text("{}")
        with self.assertRaises(ValueError):
            model_version(self.runtime)

    def test_legacy_placeholder_identity_rejected(self):
        self.runtime.identity = "2.0B"
        with self.assertRaises(ValueError):
            model_version(self.runtime)

    def test_private_store_does_not_persist_answers(self):
        store = BenchmarkStore(self.tmp.name)
        self.assertIsNone(store.read())
        store.save({"id": "server-receipt"}, {"teacher_calls": 0})
        self.assertEqual(store.read()["receipt"]["id"], "server-receipt")
        self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o600)
        self.assertNotIn("answers", store.read())

    def test_store_symlink_rejected(self):
        store = BenchmarkStore(self.tmp.name)
        target = self.runtime.store / "target.json"
        target.write_text("protected")
        store.path.with_name(store.path.name + ".tmp").symlink_to(target)
        with self.assertRaises(ValueError):
            store.save({"id": "server"}, {})
        self.assertEqual(target.read_text(), "protected")

    def test_real_frozen_suite_digest_policy_and_balanced_groups(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "ranking-v1.json"
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SUITE_SHA256)
        suite = json.loads(raw)
        self.assertEqual(suite["policy"], POLICY)
        self.assertEqual(suite["policy_hash"], POLICY_HASH)
        benchmark = [q for q in suite["questions"] if q["kind"] == "benchmark"]
        variants = [q for q in suite["questions"] if q["kind"] == "generalization"]
        self.assertEqual((len(benchmark), len(variants)), (24, 24))
        from collections import Counter
        self.assertEqual(set(Counter(q["category"] for q in benchmark).values()), {3})
        self.assertEqual(set(Counter(q["category"] for q in variants).values()), {8})
        self.assertEqual(len({q["id"] for q in suite["questions"]}), 48)
        self.assertTrue(all(q["family"].startswith("ori_mmlu-") for q in benchmark))
        self.assertTrue(all(q["answer"] in "ABCDEFGHIJ" for q in benchmark))
        self.assertTrue(all("model_outputs" not in q and "cot_content" not in q
                            and "pred" not in q for q in suite["questions"]))

    def test_authored_gold_has_independent_arithmetic_checks(self):
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "ranking-v1.json"
        rows = {q["family"]: q for q in json.loads(path.read_text())["questions"]
                if q["kind"] == "generalization"}
        expected = {"affine-composition": 3 * (5 + 7) - 4,
                    "linear-unknown": 31 - 7, "unit-conversion": 24 * 125 // 10,
                    "weighted-mean": (4 * 70 + 6 * 90) // 10,
                    "modular-transfer": (9 + 29) % 12,
                    "percentage-sequence": 200 * 80 * 125 // 10000,
                    "counting-without-replacement": 6 * 5 // 2,
                    "binary-place-value": int("101101", 2),
                    "filter-and-sum": sum(x for x in [4, 9, 12, 7, 16] if x % 2 == 0 and x > 5),
                    "distinct-count": len(set(["x", "y", "x", "z", "y", "w"])),
                    "matrix-column-sum": sum(row[1] for row in [[2,4,6], [1,3,5], [7,8,9]])}
        for family, answer in expected.items():
            self.assertEqual(int(rows[family]["answer"]), answer, family)
        self.assertEqual(rows["set-intersection"]["answer"], sorted({2,5,8,11} & {1,5,7,11}))
        self.assertEqual(rows["index-transfer"]["answer"], [[11,22,33,44,55][i] for i in [4,1,3]])
        self.assertEqual(rows["keyed-aggregation"]["answer"], {"A": 3 + 7, "B": 5})

    def test_expired_deadline_cannot_start_or_return_a_partial_report(self):
        with patch("luban.benchmark.time.monotonic", return_value=100):
            with self.assertRaises(TimeoutError):
                run_benchmark(self.runtime, self.challenge, deadline_monotonic=99)
        self.assertEqual(self.runtime.model.calls, [])
        self.assert_restored()

    def test_token_criterion_checks_deadline_without_external_timer(self):
        clock = [0.0]
        event = threading.Event()
        rng = torch.get_rng_state().clone()
        original = self.runtime.model.generate
        def token_step(tokens, **kwargs):
            self.assertFalse(kwargs["stopping_criteria"][0](tokens, None))
            clock[0] = 301
            self.assertTrue(kwargs["stopping_criteria"][0](tokens, None))
            return original(tokens, **kwargs)
        with patch("luban.benchmark.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(self.runtime.model, "generate", side_effect=token_step):
            with self.assertRaises(TimeoutError):
                run_benchmark(self.runtime, self.challenge, event, deadline_monotonic=300)
        self.assertEqual(len(self.runtime.model.calls), 1)
        self.assertFalse(event.is_set())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assert_restored()

    def test_timeout_midround_rejects_partial_completed_answers(self):
        clock = [0.0]
        def advance(progress):
            if progress["completed"] == 12:
                clock[0] = 301
        with patch("luban.benchmark.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                run_benchmark(self.runtime, self.challenge, progress=advance, deadline_monotonic=300)
        self.assertEqual(len(self.runtime.model.calls), 12)
        self.assert_restored()

    def test_complete_answers_expiring_during_cold_release_do_not_return(self):
        clock = [0.0]
        class Pager:
            depth = 0
            releases = 0
            def acquire(self): self.depth += 1
            def release(self):
                self.depth -= 1
                self.releases += 1
                if self.releases == 48:
                    clock[0] = 301
        pager = Pager()
        self.runtime.cold_backbone = pager
        with patch("luban.benchmark.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                run_benchmark(self.runtime, self.challenge, deadline_monotonic=300)
        self.assertEqual(len(self.runtime.model.calls), 48)
        self.assertEqual(pager.depth, 0)
        self.assert_restored()

    def test_deadline_cannot_extend_300_seconds_or_change_fixed_protocol(self):
        original = self.runtime.model.generate
        def check_budget(tokens, **kwargs):
            self.assertEqual(kwargs["stopping_criteria"][0].event.deadline, 300)
            self.assertEqual(kwargs["max_new_tokens"], 512)
            return original(tokens, **kwargs)
        with patch("luban.benchmark.time.monotonic", return_value=0), \
                patch.object(self.runtime.model, "generate", side_effect=check_budget):
            report = run_benchmark(self.runtime, self.challenge, deadline_monotonic=3600)
        self.assertEqual(len(report["answers"]), 48)
        self.assertEqual(report["metadata"]["time_budget_seconds"], 300)
        self.assertEqual(report["metadata"]["policy_hash"], POLICY_HASH)
        self.assertEqual(POLICY["max_output_tokens"], 512)

    def test_invalid_deadlines_are_rejected_before_generation(self):
        for invalid in (True, "300", float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    run_benchmark(self.runtime, self.challenge, deadline_monotonic=invalid)
        self.assertEqual(self.runtime.model.calls, [])

    def test_timeout_restores_mixed_module_training_flags(self):
        self.runtime.model.child = nn.Linear(1, 1)
        self.runtime.model.train()
        self.runtime.model.child.eval()
        before = [module.training for module in self.runtime.model.modules()]
        clock = [0.0]
        def expire(_):
            clock[0] = 301
        with patch("luban.benchmark.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                run_benchmark(self.runtime, self.challenge, progress=expire, deadline_monotonic=300)
        self.assertEqual([module.training for module in self.runtime.model.modules()], before)
        self.assert_restored()


if __name__ == "__main__":
    unittest.main()
