import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from luban import product_inference as product
from luban.external_brain import ExternalBrain
from luban.memory import Memory
from luban.runtime import Runtime


class Tokenizer:
    eos_token_id = 9

    def decode(self, tokens, skip_special_tokens=True):
        return "".join({5: "你", 6: "好", 7: "世界"}.get(int(token), "") for token in tokens)


class Native:
    def prompt_cache(self, tokens, *, cache_reuse):
        return None

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = []
        self.closed = False
        self.failure = None

    def generate(self, tokens, **kwargs):
        assert not self.runtime.cold_backbone.loaded
        assert self.runtime.cold_backbone.depth_count == 0
        assert self.runtime.branch.expert is None
        self.calls.append((tokens, kwargs))
        if self.failure is not None:
            raise self.failure
        output = []
        for token in (5, 6, 9):
            if kwargs["stop"].is_set():
                break
            output.append(token)
            kwargs["on_token"](token)
        return {"tokens": output, "finish_reason": "stop"}

    def close(self):
        self.closed = True
        self.runtime.events.append("native_close")


class Pager:
    def __init__(self, runtime):
        self.runtime = runtime
        self.loaded = False
        self.depth_count = 0
        self.resident_loaded = False

    def acquire(self):
        assert self.runtime.native_engine is None
        self.runtime.events.append("hf_acquire")
        self.depth_count += 1
        self.loaded = True

    def release(self):
        self.depth_count -= 1
        if not self.depth_count:
            self._evict()

    def _evict(self):
        self.runtime.events.append("hf_evict")
        self.loaded = False

    def evict_resident(self):
        assert not self.loaded and not self.depth_count
        if self.resident_loaded:
            self.runtime.events.append("hf_resident_evict")
            self.resident_loaded = False


class ProductInferenceTests(unittest.TestCase):
    def test_mlx_chat_end_is_shared_with_generation_completion(self):
        runtime = SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044)),
                                  tokenizer=SimpleNamespace(eos_token_id=248046), native_inference_precision="mlx-q4")
        self.assertEqual(product.eos_token_ids(runtime), [248044, 248046])
        runtime.model.generation_config.eos_token_id = [248044, 248046]
        self.assertEqual(product.eos_token_ids(runtime), [248044, 248046])
        runtime.native_inference_precision = "mixed-q4-g16"
        runtime.model.generation_config.eos_token_id = 248044
        self.assertEqual(product.eos_token_ids(runtime), [248044])

    def test_selected_mlx_context_reaches_each_native_generation(self):
        self.runtime.native_inference_precision = "mlx-q4"
        for limit in (65536, 131072, 262144):
            step = 256 if limit == 65536 else 128
            self.runtime.context_status = lambda limit=limit, step=step: {"current_limit": limit, "prefill_step": step}
            with patch("luban.product_inference.native_binary", return_value=Path("/fixture/luban-worker")):
                output = product.generate_tokens(self.runtime, torch.ones((1, 40000), dtype=torch.long), 4)
            self.assertEqual(output.shape[1], 40003)
            self.assertEqual(self.engine.calls[-1][1]["context_limit"], limit)
            self.assertEqual(self.engine.calls[-1][1]["prefill_step"], step)

    def test_admitted_context_snapshot_is_used_without_reading_a_new_memory_limit(self):
        self.runtime.native_inference_precision = "mlx-q4"
        self.runtime.context_status = Mock(side_effect=AssertionError("must retain the admitted snapshot"))
        context = {"current_limit": 150000, "prefill_step": 32}
        with patch("luban.product_inference.native_binary", return_value=Path("/fixture/luban-worker")):
            product.generate_tokens(self.runtime, torch.ones((1, 140000), dtype=torch.long), 4, context=context)
        self.assertEqual(self.engine.calls[-1][1]["prefill_step"], 32)
        self.assertEqual(self.engine.calls[-1][1]["context_limit"], 150000)
        self.runtime.context_status.assert_not_called()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "luban-worker"
        self.binary.write_bytes(b"mock worker")
        self.binary.with_name("luban-worker.json").write_text(json.dumps({"binary_sha256": "f" * 64}))
        (self.root / "upstream-file-manifest.json").write_text("{}")
        config = {"local_thinking": "off", "max_new_tokens": 64, "temperature": 0.6}
        self.runtime = SimpleNamespace(lock=threading.RLock(), events=[], native_engine=None,
            research=False, device="cpu", dtype=torch.float32, identity="a" * 64, active_core=None,
            store=self.root, loaded_path=str(self.root), state="ready", tokenizer=Tokenizer(),
            branch=SimpleNamespace(expert=None, requires_gate=False, expert_id=None), router_meta={},
            pointer=self.root / "active.json", settings=SimpleNamespace(snapshot=lambda: config))
        runtime = self.runtime
        runtime.text_config = SimpleNamespace(hidden_size=2048, intermediate_size=6144,
            num_hidden_layers=24, vocab_size=248320)
        runtime.cold_backbone = Pager(runtime)
        runtime.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"),
            generation_config=SimpleNamespace(eos_token_id=[9]), generate=Mock(side_effect=self.hf_generate))
        runtime._ready = lambda: None
        runtime._tokens = lambda *args, **kwargs: torch.tensor([[1, 3]])
        runtime.context_status = lambda: {"current_limit": 32768, "memory_estimate_limit": 32768, "prefill_step": 512,
            "native_output_limit": 2048, "linear_bytes_per_token": 1,
            "quadratic_bytes_per_token_squared": 1, "memory_budget_bytes": 1024 ** 3}
        runtime._last_generation_usage_path = self.root / "last-generation-usage.json"
        runtime._load_active_expert = self.load_expert
        runtime._unload_saved_expert = lambda: None
        runtime.external_hint = lambda _: None
        runtime.external_binding = lambda **kwargs: None
        runtime.choose_route = Mock(return_value={"route": "local", "action": "local_core"})
        for name in ("close_native", "generate_tokens", "generate", "_generate_loaded", "inference_identity"):
            setattr(runtime, name, MethodType(getattr(Runtime, name), runtime))
        self.engine = Native(runtime)
        runtime.native_engine = self.engine
        self.patch_binary = patch("luban.product_inference.native_binary", return_value=self.binary)
        self.patch_binary.start()
        self.addCleanup(self.patch_binary.stop)
        self.messages = [{"role": "user", "content": "你好"}]

    def load_expert(self):
        if self.runtime.pointer.exists() and self.runtime.branch.expert is None:
            assert self.runtime.cold_backbone.loaded
            self.runtime.branch.expert = object()

    def hf_generate(self, tokens, **kwargs):
        assert self.runtime.native_engine is None
        assert self.runtime.cold_backbone.loaded
        self.runtime.events.append("hf_generate")
        return torch.cat((tokens, torch.tensor([[7, 9]])), dim=1)

    def answer(self, emit=None, stop=None, **kwargs):
        return self.runtime.generate(self.messages, emit if emit is not None else lambda _: None,
                                     stop if stop is not None else threading.Event(), **kwargs)

    def test_native_core_keeps_hf_cold_and_streams_chinese_without_prompt(self):
        chunks = []
        result = self.answer(chunks.append)
        self.assertEqual("".join(chunk["content"] for chunk in chunks), "你好")
        self.assertEqual(result["local_action"], "local_core")
        self.assertTrue(result["local_complete"])
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["prompt_tokens"], 2)
        self.assertFalse(self.runtime.cold_backbone.loaded)
        self.assertEqual(self.runtime.events, [])
        self.runtime.model.generate.assert_not_called()
        self.assertEqual(self.engine.calls[0][1]["temperature"], 0.6)

    def test_each_request_has_its_own_streamer_and_native_is_reused(self):
        chunks = []
        self.answer(chunks.append)
        self.answer(chunks.append)
        self.assertEqual("".join(row["content"] for row in chunks), "你好你好")
        self.assertEqual(len(self.engine.calls), 2)
        self.assertIs(self.runtime.native_engine, self.engine)

    def test_native_decode_rate_excludes_first_token_and_prefill(self):
        self.native_segments([{"tokens": [5, 6, 9], "finish_reason": "stop",
                              "prefill_ns": 2_000_000_000, "ttft_ns": 2_100_000_000,
                              "decode_ns": 100_000_000}])
        usage = self.answer()
        self.assertEqual(usage["decode_tokens"], 2)
        self.assertEqual(usage["decode_seconds"], 0.1)
        self.assertEqual(usage["decode_tokens_per_second"], 20)
        self.assertEqual(usage["prefill_seconds"], 2)
        self.assertEqual(usage["time_to_first_token_seconds"], 2.1)
        self.assertEqual(usage["tokens_per_second"], usage["end_to_end_tokens_per_second"])

    def test_prefix_cache_requires_explicit_foreground_opt_in(self):
        self.answer()
        self.assertIs(self.engine.calls[-1][1]["cache_reuse"], False)
        self.answer(cache_reuse=True)
        self.assertIs(self.engine.calls[-1][1]["cache_reuse"], True)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.answer(cache_reuse=1)

    def test_continuation_aggregates_only_measured_decode_intervals(self):
        self.native_segments([{"tokens": [5] * 2048, "finish_reason": "length",
                               "decode_ns": 100_000_000_000, "prefill_ns": 2_000_000_000, "ttft_ns": 2_100_000_000},
                              {"tokens": [6, 9], "finish_reason": "stop",
                               "decode_ns": 50_000_000, "prefill_ns": 3_000_000_000, "ttft_ns": 3_100_000_000}])
        usage = self.answer()
        self.assertEqual(usage["completion_tokens"], 2050)
        self.assertEqual(usage["decode_tokens"], 2048)
        self.assertEqual(usage["decode_seconds"], 100.05)
        self.assertAlmostEqual(usage["decode_tokens_per_second"], 2048 / 100.05)
        self.assertEqual(usage["prefill_seconds"], 5)
        self.assertEqual(usage["time_to_first_token_seconds"], 2.1)

    def test_single_token_and_missing_timing_never_reuse_previous_speed(self):
        self.native_segments([{"tokens": [5, 6], "finish_reason": "stop", "decode_ns": 50_000_000},
                              {"tokens": [9], "finish_reason": "stop", "decode_ns": None},
                              {"tokens": [5, 9], "finish_reason": "stop"}])
        self.assertEqual(self.answer()["decode_tokens_per_second"], 20)
        for _ in range(2):
            usage = self.answer()
            self.assertIsNone(usage["decode_tokens_per_second"])
            self.assertIsNone(usage["decode_seconds"])

    def test_cancelled_generation_retains_only_real_decode_measurement(self):
        stop = threading.Event()
        def generate(tokens, **kwargs):
            for token in [5, 6]:
                kwargs["on_token"](token)
            stop.set()
            return {"tokens": [5, 6], "finish_reason": "cancelled", "decode_ns": 50_000_000}
        self.engine.generate = generate
        usage = self.answer(stop=stop)
        self.assertEqual(usage["decode_tokens_per_second"], 20)
        self.assertEqual(usage["finish_reason"], "stop")
        self.assertFalse(usage["local_complete"])

    def test_hf_measures_token_callbacks_instead_of_text_chunks(self):
        clock = Mock(return_value=100.0)
        def generate(tokens, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put(tokens)
            clock.return_value = 110.0
            streamer.put(torch.tensor([5]))
            clock.return_value = 110.25
            streamer.put(torch.tensor([9]))
            streamer.end()
            clock.return_value = 120.0
            return torch.cat((tokens, torch.tensor([[5, 9]])), dim=1)
        self.runtime.model.generate.side_effect = generate
        with patch("luban.product_inference.native_binary", return_value=None), patch("time.perf_counter", clock):
            usage = self.answer()
        self.assertEqual(usage["decode_tokens_per_second"], 4)
        self.assertEqual(usage["decode_tokens"], 1)
        self.assertEqual(usage["time_to_first_token_seconds"], 10)
        self.assertEqual(usage["end_to_end_tokens_per_second"], 0.1)

    def test_hf_observation_without_emit_does_not_decode_stream_text(self):
        def generate(tokens, **kwargs):
            streamer = kwargs["streamer"]
            streamer.put(tokens)
            streamer.put(torch.tensor([5]))
            streamer.put(torch.tensor([9]))
            streamer.end()
            return torch.cat((tokens, torch.tensor([[5, 9]])), dim=1)
        self.runtime.model.generate.side_effect = generate
        self.runtime.tokenizer.decode = Mock(side_effect=AssertionError("unnecessary text decoding"))
        with patch("luban.product_inference.native_binary", return_value=None):
            output = self.runtime.generate_tokens(torch.tensor([[1, 3]]), 2)
        self.assertEqual(output.tolist(), [[1, 3, 5, 9]])
        self.runtime.tokenizer.decode.assert_not_called()
        self.assertGreater(self.runtime._generation_timing["decode_tokens_per_second"], 0)

    def native_segments(self, segments):
        def generate(tokens, **kwargs):
            self.engine.calls.append((tokens, kwargs))
            result = segments[len(self.engine.calls) - 1]
            if isinstance(result, Exception):
                raise result
            for token in result["tokens"]:
                kwargs["on_token"](token)
            return result
        self.engine.generate = generate
        self.runtime.settings.snapshot()["max_new_tokens"] = 0

    def test_automatic_native_answer_continues_raw_tokens_with_one_chinese_streamer(self):
        from luban.runtime import EmitStreamer
        self.native_segments([{"tokens": [5] * 2048, "finish_reason": "length"},
                              {"tokens": [6, 9], "finish_reason": "stop"}])
        chunks = []
        with patch("luban.runtime.EmitStreamer", wraps=EmitStreamer) as streamer:
            result = self.answer(chunks.append)
        self.assertEqual(streamer.call_count, 1)
        self.assertEqual("".join(row["content"] for row in chunks), "你" * 2048 + "好")
        self.assertEqual(self.engine.calls[1][0], [1, 3] + [5] * 2048)
        self.assertEqual([call[1]["max_tokens"] for call in self.engine.calls], [2048, 2048])
        self.assertEqual(result["completion_tokens"], 2050)
        self.assertEqual(result["prompt_tokens"], 2)
        self.assertEqual(result["total_tokens"], 2052)
        self.assertEqual(result["output_limit"], 8192)
        self.assertTrue(result["local_complete"])

    def test_automatic_native_answer_never_exceeds_four_full_segments(self):
        self.native_segments([{"tokens": [5] * 2048, "finish_reason": "length"}] * 4)
        result = self.answer()
        self.assertEqual(len(self.engine.calls), 4)
        self.assertEqual(len(self.engine.calls[-1][0]), 2 + 3 * 2048)
        self.assertEqual(result["completion_tokens"], 8192)
        self.assertEqual(result["finish_reason"], "length")
        self.assertFalse(result["local_complete"])

    def test_native_segments_keep_multibyte_character_buffer_until_complete(self):
        self.runtime.tokenizer.decode = lambda ids, **kw: "".join(
            {5: "你", 8: "<first>", 10: "<second>"}.get(int(token), "") for token in ids
        ).replace("<first><second>", "😀").replace("<first>", "�").replace("<second>", "�")
        self.native_segments([{"tokens": [5] * 2047 + [8], "finish_reason": "length"},
                              {"tokens": [10, 9], "finish_reason": "stop"}])
        chunks = []
        self.answer(chunks.append)
        self.assertEqual("".join(row["content"] for row in chunks), "你" * 2047 + "😀")

    def test_native_last_segment_uses_only_remaining_automatic_budget(self):
        self.native_segments([{"tokens": [5] * 2048, "finish_reason": "length"},
                              {"tokens": [6] * 452, "finish_reason": "length"}])
        status = self.runtime.context_status()
        self.runtime.context_status = lambda: {**status, "memory_estimate_limit": 10000}
        result = self.answer()
        self.assertEqual([call[1]["max_tokens"] for call in self.engine.calls], [2048, 452])
        self.assertEqual(result["completion_tokens"], 2500)
        self.assertEqual(result["output_limit"], 2500)

    def test_native_continuation_stops_for_eos_short_output_or_worker_stop(self):
        for segment in ({"tokens": [5] * 2047 + [9], "finish_reason": "length"},
                        {"tokens": [], "finish_reason": "length"},
                        {"tokens": [5], "finish_reason": "length"},
                        {"tokens": [5] * 2048, "finish_reason": "stop"},
                        {"tokens": [5], "finish_reason": "cancelled"}):
            with self.subTest(reason=segment["finish_reason"], count=len(segment["tokens"])):
                self.engine.calls.clear()
                self.native_segments([segment])
                self.answer()
                self.assertEqual(len(self.engine.calls), 1)

    def test_native_continuation_honors_cancellation_and_closes_after_later_failure(self):
        self.native_segments([{"tokens": [5] * 2048, "finish_reason": "length"}])
        stop = threading.Event()
        self.answer(lambda _: stop.set(), stop)
        self.assertEqual(len(self.engine.calls), 1)
        self.engine.calls.clear()
        self.native_segments([{"tokens": [5] * 2048, "finish_reason": "length"},
                              RuntimeError("later segment failed")])
        chunks = []
        with self.assertRaisesRegex(RuntimeError, "later segment failed"):
            self.answer(chunks.append)
        self.assertEqual("".join(row["content"] for row in chunks), "你" * 2048)
        self.assertTrue(self.engine.closed)
        self.assertIsNone(self.runtime.native_engine)
        self.assertEqual(self.runtime.state, "ready")

    def test_saved_expert_switches_to_hf_after_native_close(self):
        self.runtime.pointer.write_text("{}")
        result = self.answer(temperature=0)
        self.assertEqual(result["local_action"], "neural_expert")
        self.assertEqual(self.runtime.events[:2], ["native_close", "hf_acquire"])
        self.assertIn("hf_generate", self.runtime.events)
        self.assertTrue(self.engine.closed)
        self.assertIsNone(self.runtime.native_engine)
        self.assertFalse(self.runtime.cold_backbone.loaded)

    def saved_expert_with_real_unload(self):
        self.runtime.branch.expert_id = "saved-expert"
        self.runtime.pointer.write_text(json.dumps({"id": "saved-expert"}))
        self.runtime._unload_saved_expert = MethodType(Runtime._unload_saved_expert, self.runtime)
        self.runtime.cold_backbone.resident_loaded = True

    def test_completed_saved_expert_releases_route_only_after_outer_hf_exit(self):
        self.saved_expert_with_real_unload()

        def generate(tokens, **kwargs):
            self.assertEqual(self.runtime.cold_backbone.depth_count, 2)
            self.assertTrue(self.runtime.cold_backbone.resident_loaded)
            return self.hf_generate(tokens, **kwargs)

        self.runtime.model.generate.side_effect = generate
        usage = self.answer(temperature=0)
        self.assertEqual(usage["local_action"], "neural_expert")
        self.assertIsNone(self.runtime.branch.expert)
        self.assertEqual(self.runtime.cold_backbone.depth_count, 0)
        self.assertFalse(self.runtime.cold_backbone.resident_loaded)
        self.assertEqual(self.runtime.events[-2:], ["hf_evict", "hf_resident_evict"])

    def test_completed_expert_refreshes_actual_released_memory_before_return(self):
        self.saved_expert_with_real_unload()
        self.runtime.device = "mps"
        status = self.runtime.context_status()
        self.runtime.context_status = Mock(side_effect=[status, {**status, "memory_estimate_limit": 0}, status])
        stop = Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = False
        with patch("luban.product_inference.torch.mps.driver_allocated_memory", side_effect=[2048, 512]), \
                patch("luban.product_inference.time.monotonic", side_effect=[10, 10.1]):
            usage = self.answer(stop=stop, temperature=0)
        self.assertTrue(usage["local_complete"])
        stop.wait.assert_called_once_with(0.05)

    def test_hf_error_releases_route_without_waiting_or_overwriting_error(self):
        self.saved_expert_with_real_unload()
        failure = RuntimeError("original HF failure")
        self.runtime.model.generate.side_effect = failure
        with patch("luban.product_inference.prepare_native", wraps=product.prepare_native) as prepare:
            with self.assertRaises(RuntimeError) as caught:
                self.answer()
        self.assertIs(caught.exception, failure)
        self.assertFalse(prepare.call_args.kwargs["wait"])
        self.assertFalse(self.runtime.cold_backbone.resident_loaded)
        self.assertIsNone(self.runtime.branch.expert)

    def test_cleanup_failure_keeps_original_hf_exception(self):
        self.saved_expert_with_real_unload()
        failure = RuntimeError("original HF failure")
        self.runtime.model.generate.side_effect = failure
        self.runtime.cold_backbone.evict_resident = Mock(side_effect=RuntimeError("release failure"))
        with self.assertRaises(RuntimeError) as caught:
            self.answer()
        self.assertIs(caught.exception, failure)
        self.assertIn("release failure", caught.exception.__notes__[0])

    def test_cancelled_expert_releases_route_without_memory_wait(self):
        self.saved_expert_with_real_unload()
        stop = threading.Event()

        def generate(tokens, **kwargs):
            stop.set()
            return self.hf_generate(tokens, **kwargs)

        self.runtime.model.generate.side_effect = generate
        with patch("luban.product_inference.prepare_native", wraps=product.prepare_native) as prepare:
            usage = self.answer(stop=stop)
        self.assertFalse(usage["local_complete"])
        self.assertEqual(usage["finish_reason"], "stop")
        self.assertFalse(prepare.call_args.kwargs["wait"])
        self.assertFalse(self.runtime.cold_backbone.resident_loaded)

    def test_enclosing_training_hf_scope_keeps_route_until_its_owner_exits(self):
        self.saved_expert_with_real_unload()
        with product.full_backbone(self.runtime):
            self.answer()
            self.assertEqual(self.runtime.cold_backbone.depth_count, 1)
            self.assertTrue(self.runtime.cold_backbone.resident_loaded)
            self.assertNotIn("hf_resident_evict", self.runtime.events)
            self.runtime.model.generate.side_effect = RuntimeError("nested training observation failed")
            with self.assertRaisesRegex(RuntimeError, "nested training observation failed"):
                self.answer()
            self.assertEqual(self.runtime.cold_backbone.depth_count, 1)
            self.assertTrue(self.runtime.cold_backbone.resident_loaded)
            self.assertNotIn("hf_resident_evict", self.runtime.events)
        self.assertTrue(self.runtime.cold_backbone.resident_loaded)

    def test_non_native_keeps_residency_but_native_cleanup_preserves_attached_expert(self):
        self.saved_expert_with_real_unload()
        with patch("luban.product_inference.native_binary", return_value=None):
            self.answer()
        self.assertTrue(self.runtime.cold_backbone.resident_loaded)
        self.runtime.pointer.unlink()
        expert = self.runtime.branch.expert = object()
        self.answer()
        self.assertIs(self.runtime.branch.expert, expert)
        self.assertFalse(self.runtime.cold_backbone.resident_loaded)

    def test_saved_gated_off_expert_leaves_native_core_selected(self):
        self.runtime.pointer.write_text("{}")
        self.runtime.branch.requires_gate = True
        result = self.answer()
        self.assertEqual(result["local_action"], "local_core")
        self.assertEqual(self.runtime.events, [])
        self.assertIs(self.runtime.native_engine, self.engine)

    def test_stale_local_trial_binding_falls_back_to_native_core(self):
        self.runtime.pointer.write_text("{}")
        self.runtime.native_inference_precision = "mlx-q4"
        self.runtime.router_meta = {"execution_scope": "local-trial"}
        with patch("luban.product_inference.trial_expert_binding", return_value=None):
            result = self.answer()
        self.assertEqual(result["local_action"], "local_core")
        self.assertEqual(self.runtime.events, [])
        self.assertIs(self.runtime.native_engine, self.engine)

    def test_unseen_hint_dictionary_keeps_selected_saved_expert(self):
        self.runtime.pointer.write_text("{}")
        self.runtime.branch.requires_gate = True
        self.runtime.choose_route = lambda _: {"action": "neural_expert"}
        self.runtime.external_hint = lambda _: {"status": "needs_teacher", "has_hint": False,
                                               "reference": None, "local_observation": None}
        result = self.answer(temperature=0)
        self.assertEqual(result["local_action"], "neural_expert")
        self.assertFalse(result["reference_used"])
        self.assertIn("hf_acquire", self.runtime.events)
        self.assertIn("hf_generate", self.runtime.events)

    def test_reference_forces_native_core_even_when_expert_is_selected(self):
        self.runtime.pointer.write_text("{}")
        self.runtime.external_hint = lambda _: {"status": "reference_ready", "has_hint": True,
            "reference": {"sample_id": "sample-1", "answer": "周三"}}
        result = self.answer()
        self.assertTrue(result["reference_used"])
        self.assertEqual(result["reference_sample_id"], "sample-1")
        self.assertEqual(result["local_action"], "local_core")
        self.assertEqual(self.runtime.events, [])

    def test_mastered_core_proof_forces_core_over_existing_expert(self):
        self.runtime.pointer.write_text("{}")
        self.runtime.external_hint = lambda _: {"status": "mastered", "has_hint": True,
            "local_observation": {"local_action": "local_core"}}
        result = self.answer()
        self.assertFalse(result["reference_used"])
        self.assertEqual(result["local_action"], "local_core")
        self.assertEqual(self.runtime.events, [])

    def test_cancel_before_generation_returns_no_tokens_and_restores_state(self):
        stop = threading.Event()
        stop.set()
        result = self.answer(stop=stop)
        self.assertEqual(result["completion_tokens"], 0)
        self.assertFalse(result["local_complete"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(self.runtime.state, "ready")

    def test_cancel_from_chinese_stream_preserves_partial_output(self):
        stop = threading.Event()
        chunks = []
        def emit(value):
            chunks.append(value)
            stop.set()
        result = self.answer(emit, stop)
        self.assertEqual("".join(row["content"] for row in chunks), "你")
        self.assertEqual(result["completion_tokens"], 1)
        self.assertFalse(result["local_complete"])
        self.assertEqual(self.runtime.state, "ready")

    def test_stream_callback_failure_closes_native_and_restores_state(self):
        def emit(_):
            raise RuntimeError("consumer disappeared")
        with self.assertRaisesRegex(RuntimeError, "consumer disappeared"):
            self.answer(emit)
        self.assertTrue(self.engine.closed)
        self.assertIsNone(self.runtime.native_engine)
        self.assertEqual(self.runtime.state, "ready")

    def test_native_failure_closes_engine_and_next_request_starts_fresh(self):
        self.engine.failure = RuntimeError("worker crashed")
        with self.assertRaisesRegex(RuntimeError, "worker crashed"):
            self.answer()
        self.assertIsNone(self.runtime.native_engine)
        fresh = Native(self.runtime)
        with patch("luban.product_inference.start", return_value=fresh) as start:
            result = self.answer()
        start.assert_called_once_with(self.runtime)
        self.assertTrue(result["local_complete"])

    def test_non_native_runtime_uses_existing_hf_generation_policy(self):
        with patch("luban.product_inference.native_binary", return_value=None):
            result = self.answer(temperature=0.7)
        self.assertTrue(result["local_complete"])
        kwargs = self.runtime.model.generate.call_args.kwargs
        self.assertTrue(kwargs["do_sample"])
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertEqual(kwargs["temperature"], 0.7)

    def test_identity_stays_native_while_temporarily_using_hf_for_expert(self):
        self.runtime.cold_backbone.depth_count = 1
        self.assertEqual(product.identity(self.runtime)["engine"], "luban-native-bf16-metal")
        self.assertEqual(product.identity(self.runtime, "neural_expert")["engine"], "torch-hf")
        self.assertEqual(product.identity(self.runtime, "local_core")["engine"], "torch-hf")

    def test_nondefault_execution_graph_changes_identity_without_changing_legacy_default(self):
        baseline = product.identity(self.runtime)
        self.assertEqual(set(baseline), {"engine", "binary_sha256"})
        self.runtime.native_skip_layer_mask = 2
        self.runtime.native_exit_after = 12
        changed = product.identity(self.runtime)
        self.assertEqual(changed["execution_skip_mask"], 2)
        self.assertEqual(changed["execution_exit_after"], 12)
        self.assertNotEqual(changed, baseline)
        self.runtime.native_layer_paging = True
        paged = product.identity(self.runtime)
        self.assertIs(paged["layer_paging_enabled"], True)
        self.assertLess(len(json.dumps(paged).encode()), 512)
        self.assertEqual(product.identity(self.runtime, "neural_expert"), {"engine": "torch-hf", "torch_version": torch.__version__})
        self.runtime.native_skip_layer_mask, self.runtime.native_exit_after, self.runtime.native_layer_paging = 0, 24, False
        self.assertEqual(product.identity(self.runtime), baseline)

    def test_start_binds_exact_active_core_file_and_tensor_key(self):
        self.runtime.active_core = "core-0123456789ab"
        metadata = {"sha256": "b" * 64}
        (self.root / "active-core.json").write_text(json.dumps({"version": self.runtime.active_core,
            "identity": self.runtime.identity, **metadata}))
        key = "model.language_model.layers.23.mlp.base.down_proj.weight"
        with patch("luban.evolution._read_version", return_value=(metadata, torch.zeros(1))) as read, \
                patch("luban.evolution._target", return_value=(key, None)), \
                patch("luban.native_engine.NativeEngine") as constructor:
            product.start(self.runtime)
        read.assert_called_once_with(self.runtime, self.runtime.active_core)
        arguments = constructor.call_args
        self.assertEqual(arguments.kwargs["core"], {"file": str(self.root / (self.runtime.active_core + ".safetensors")),
            "tensor_key": key, "sha256": metadata["sha256"]})
        self.assertEqual(arguments.args[2], hashlib.sha256(b"{}").hexdigest())
        constructor.return_value.start.assert_called_once()

    def test_changed_active_core_pointer_is_rejected_before_worker_allocation(self):
        self.runtime.active_core = "core-0123456789ab"
        (self.root / "active-core.json").write_text(json.dumps({"version": self.runtime.active_core,
            "identity": "c" * 64, "sha256": "b" * 64}))
        with patch("luban.evolution._read_version", return_value=({"sha256": "b" * 64}, torch.zeros(1))), \
                patch("luban.native_engine.NativeEngine") as constructor:
            with self.assertRaisesRegex(ValueError, "活动核心指针"):
                product.start(self.runtime)
        constructor.assert_not_called()

    def test_start_failure_closes_partial_worker(self):
        with patch("luban.native_engine.NativeEngine") as constructor:
            constructor.return_value.start.side_effect = RuntimeError("load failed")
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                product.start(self.runtime)
        constructor.return_value.close.assert_called_once()

    def test_start_passes_policy_and_unique_native_owned_paging_file(self):
        self.runtime.native_skip_layer_mask = 2
        self.runtime.native_exit_after = 12
        self.runtime.native_layer_paging = True
        self.runtime.loaded_storage = str(self.root)
        with patch("luban.native_engine.NativeEngine") as constructor:
            product.start(self.runtime)
            first = constructor.call_args.kwargs
            product.start(self.runtime)
            second = constructor.call_args.kwargs
        self.assertEqual((first['skip_layer_mask'], first['exit_after']), (2, 12))
        self.assertEqual(first['layer_paging_path'].parent, self.root / 'runtime/layer-paging')
        self.assertTrue(first['layer_paging_path'].parent.is_dir())
        self.assertFalse(first['layer_paging_path'].exists())
        self.assertNotEqual(first['layer_paging_path'], second['layer_paging_path'])

    def test_active_core_cannot_be_skipped_or_early_exited(self):
        self.runtime.active_core = "core-0123456789ab"
        for skip_mask, exit_after in ((1 << 23, 24), (0, 23)):
            self.runtime.native_skip_layer_mask = skip_mask
            self.runtime.native_exit_after = exit_after
            with patch("luban.evolution._read_version") as read, patch("luban.native_engine.NativeEngine") as constructor:
                with self.assertRaisesRegex(ValueError, "活动学习核心"):
                    product.start(self.runtime)
                read.assert_not_called()
                constructor.assert_not_called()

    def test_paging_requires_connected_storage_before_worker_start(self):
        self.runtime.native_layer_paging = True
        self.runtime.loaded_storage = '/Volumes/luban-disconnected-pager-fixture/runtime'
        with patch("luban.native_engine.NativeEngine") as constructor:
            with self.assertRaisesRegex(RuntimeError, "存储盘"):
                product.start(self.runtime)
            constructor.assert_not_called()

    def test_native_start_drops_unused_hf_route_before_allocating_worker(self):
        self.runtime.cold_backbone.resident_loaded = True
        with patch("luban.native_engine.NativeEngine") as constructor:
            constructor.return_value.start.side_effect = lambda: self.runtime.events.append("native_start")
            product.start(self.runtime)
        self.assertEqual(self.runtime.events, ["hf_resident_evict", "native_start"])

    def test_native_start_releases_completed_calibrated_hf_route(self):
        self.runtime.router_meta["calibrated"] = True
        self.runtime.cold_backbone.resident_loaded = True
        with patch("luban.native_engine.NativeEngine"):
            product.start(self.runtime)
        self.assertEqual(self.runtime.events, ["hf_resident_evict"])

    def test_native_admission_happens_after_completed_route_is_released(self):
        self.runtime.router_meta["calibrated"] = True
        self.runtime.cold_backbone.resident_loaded = True
        original_tokens = self.runtime._tokens

        def tokens(*args, **kwargs):
            self.assertFalse(self.runtime.cold_backbone.resident_loaded)
            self.runtime.events.append("input_admission")
            return original_tokens(*args, **kwargs)

        self.runtime._tokens = tokens
        self.answer()
        self.assertEqual(self.runtime.events, ["hf_resident_evict", "input_admission"])

    def test_cold_mlx_load_is_admitted_before_measuring_prompt_memory(self):
        self.runtime.native_inference_precision = "mlx-q4"
        status = {**self.runtime.context_status(), "prefill_step": 128}
        self.runtime.context_status = Mock(side_effect=[{**status, "additional_weight_bytes": 1024}, status, status])
        self.runtime.native_engine = None
        original_tokens = self.runtime._tokens
        self.runtime._tokens = lambda *a, **kw: (self.runtime.events.append("prompt_admission") or original_tokens(*a, **kw))
        def start(runtime):
            runtime.events.append("model_load")
            runtime.native_engine = self.engine
            return self.engine
        with patch("luban.product_inference.start", side_effect=start) as load:
            self.answer()
        load.assert_called_once()
        self.assertLess(self.runtime.events.index("model_load"), self.runtime.events.index("prompt_admission"))

    def test_cold_mlx_cannot_load_when_base_allocation_is_not_admitted(self):
        self.runtime.native_inference_precision = "mlx-q4"
        status = {**self.runtime.context_status(), "prefill_step": 128, "additional_weight_bytes": 1024,
                  "memory_estimate_limit": 0, "memory_budget_bytes": 0, "current_limit": 0}
        self.runtime.context_status = Mock(return_value=status)
        self.runtime.native_engine = None
        with patch("luban.product_inference.start") as load:
            with self.assertRaisesRegex(ValueError, "可用内存不足"):
                self.answer()
        load.assert_not_called()

    def test_cold_mlx_loading_does_not_bypass_full_prompt_memory_guard(self):
        self.runtime.native_inference_precision = "mlx-q4"
        status = {**self.runtime.context_status(), "prefill_step": 128}
        self.runtime.context_status = Mock(side_effect=[{**status, "additional_weight_bytes": 1024},
                                                       {**status, "memory_budget_bytes": 1}])
        self.runtime.native_engine = None
        def start(runtime):
            runtime.native_engine = self.engine
            return self.engine
        with patch("luban.product_inference.start", side_effect=start) as load:
            with self.assertRaisesRegex(ValueError, "可用内存"):
                self.answer()
        load.assert_called_once()
        self.assertEqual(self.engine.calls, [])

    def test_cold_native_rechecks_stale_memory_only_after_actual_metal_release(self):
        self.runtime.device = "mps"
        self.runtime.native_engine = None
        self.runtime.cold_backbone.resident_loaded = True
        self.runtime.context_status = Mock(side_effect=[{"memory_estimate_limit": 0},
                                                       {"memory_estimate_limit": 2048}])
        stop = Mock()
        stop.wait.return_value = False
        with patch("luban.product_inference.torch.mps.driver_allocated_memory", side_effect=[2048, 512]), \
                patch("luban.product_inference.time.monotonic", side_effect=[10, 10.1]), \
                patch("luban.product_inference.start") as start:
            product.prepare_native(self.runtime, stop)
        stop.wait.assert_called_once_with(0.05)
        self.assertEqual(self.runtime.context_status.call_count, 2)
        start.assert_not_called()

    def test_real_memory_shortage_still_rejects_after_bounded_release_wait(self):
        self.runtime.device = "mps"
        self.runtime.native_engine = None
        self.runtime.cold_backbone.resident_loaded = True
        status = self.runtime.context_status()
        self.runtime.context_status = Mock(return_value={**status, "memory_estimate_limit": 0,
            "current_limit": 0, "memory_budget_bytes": 0})
        stop = Mock()
        stop.wait.return_value = False
        with patch("luban.product_inference.torch.mps.driver_allocated_memory", side_effect=[2048, 512]), \
                patch("luban.product_inference.time.monotonic", side_effect=[10, 10.1, 12.1]), \
                patch("luban.product_inference.start") as start:
            with self.assertRaisesRegex(ValueError, "当前可用内存不足"):
                self.answer(stop=stop)
        stop.wait.assert_called_once_with(0.05)
        start.assert_not_called()

    def test_release_wait_is_cancellable_without_allocating_native(self):
        self.runtime.device = "mps"
        self.runtime.native_engine = None
        self.runtime.context_status = Mock(return_value={"memory_estimate_limit": 0})
        stop = threading.Event()
        stop.set()
        with patch("luban.product_inference.torch.mps.driver_allocated_memory", side_effect=[2048, 512]), \
                patch("luban.product_inference.start") as start:
            with self.assertRaisesRegex(RuntimeError, "生成已取消"):
                product.prepare_native(self.runtime, stop)
        start.assert_not_called()

    def test_no_release_or_live_worker_does_not_wait_for_system_memory(self):
        self.runtime.device = "mps"
        for memory, process in [([512, 512], None), ([2048, 512], SimpleNamespace(poll=lambda: None))]:
            with self.subTest(memory=memory, process=process):
                self.runtime.native_engine = SimpleNamespace(process=process)
                self.runtime.context_status = Mock(side_effect=AssertionError("not a cold released allocation"))
                stop = Mock()
                with patch("luban.product_inference.torch.mps.driver_allocated_memory", side_effect=memory):
                    product.prepare_native(self.runtime, stop)
                stop.wait.assert_not_called()

    def test_existing_native_worker_releases_route_restored_by_unaccepted_training(self):
        self.runtime.cold_backbone.resident_loaded = True
        with patch("luban.product_inference.start", side_effect=AssertionError("worker must be reused")):
            self.answer()
            self.answer()
        self.assertEqual(self.runtime.events, ["hf_resident_evict"])
        self.assertIs(self.runtime.native_engine, self.engine)
        self.assertEqual(len(self.engine.calls), 2)

    def test_start_cannot_coexist_with_loaded_hf_core(self):
        self.runtime.cold_backbone.loaded = True
        with patch("luban.native_engine.NativeEngine") as constructor:
            with self.assertRaisesRegex(RuntimeError, "释放完整 HF"):
                product.start(self.runtime)
        constructor.assert_not_called()

    def test_rejected_research_core_cannot_allocate_formal_worker(self):
        self.runtime.active_core = "core-0123456789ab"
        with patch("luban.evolution._read_version", side_effect=ValueError("research candidate rejected")), \
                patch("luban.native_engine.NativeEngine") as constructor:
            with self.assertRaisesRegex(ValueError, "research candidate rejected"):
                product.start(self.runtime)
        constructor.assert_not_called()

    def test_research_cpu_and_wrong_dtype_do_not_select_native_binary(self):
        self.patch_binary.stop()
        self.runtime.device = "mps"
        self.runtime.dtype = torch.bfloat16
        self.runtime.research = True
        self.assertIsNone(product.native_binary(self.runtime))
        self.runtime.research = False
        self.runtime.device = "cpu"
        self.assertIsNone(product.native_binary(self.runtime))
        self.runtime.device = "mps"
        self.runtime.dtype = torch.float32
        self.assertIsNone(product.native_binary(self.runtime))

    def test_other_qwen35_sizes_do_not_select_fixed_2b_worker(self):
        self.patch_binary.stop()
        runtime = self.runtime
        runtime.device, runtime.dtype = "mps", torch.bfloat16
        with patch("luban.product_inference.Path.is_file", return_value=True):
            self.assertIsNotNone(product.native_binary(runtime))
            for field, value in (("hidden_size", 1024), ("intermediate_size", 4096),
                                 ("num_hidden_layers", 32), ("vocab_size", 151936)):
                with self.subTest(field=field):
                    original = getattr(runtime.text_config, field)
                    setattr(runtime.text_config, field, value)
                    self.assertIsNone(product.native_binary(runtime))
                    setattr(runtime.text_config, field, original)
            runtime.text_config = None
            self.assertIsNone(product.native_binary(runtime))

    def test_dormant_expert_does_not_invalidate_native_core_mastery(self):
        runtime = self.runtime
        runtime.memory = Memory(self.root / "memory")
        runtime.memory.model_identity = runtime.identity
        runtime.training_session = None
        runtime._base_core_weight = torch.zeros(1, dtype=torch.bfloat16)
        runtime.external_binding = MethodType(Runtime.external_binding, runtime)
        binding = runtime.external_binding(actual_action="local_core")
        brain = ExternalBrain(runtime.memory)
        identifier = runtime.memory.add_sample("你好", "周三", source="human", approved=True,
            training_allowed=True, provenance={"messages": self.messages, "base_identity": runtime.identity})
        brain.register_reference(identifier, self.messages, runtime.identity, binding)
        for _ in range(2):
            result = brain.observe_local(self.messages, runtime.identity, binding, "周三", complete=True,
                actual_context=self.messages, reference_used=False, local_action="local_core")
        self.assertEqual(result["status"], "mastered")
        runtime.branch.expert_id = "learned-0123456789ab"
        metadata = self.root / "expert.json"
        metadata.write_text(json.dumps({"sha256": "b" * 64}))
        runtime.branch._paths = lambda _: (self.root / "expert.safetensors", metadata)
        hint = Runtime.external_hint(runtime, self.messages)
        self.assertEqual(hint["status"], "mastered")
        self.assertEqual(hint["local_observation"]["local_action"], "local_core")

    def test_expert_hf_mastery_is_bound_to_expert_and_actual_engine(self):
        runtime = self.runtime
        runtime.memory = Memory(self.root / "memory")
        runtime.memory.model_identity = runtime.identity
        runtime.training_session = None
        runtime._base_core_weight = torch.zeros(1, dtype=torch.bfloat16)
        runtime.external_binding = MethodType(Runtime.external_binding, runtime)
        runtime.branch.expert_id = "learned-0123456789ab"
        metadata = self.root / "expert.json"
        metadata.write_text(json.dumps({"sha256": "b" * 64}))
        runtime.branch._paths = lambda _: (self.root / "expert.safetensors", metadata)
        binding = runtime.external_binding(actual_action="neural_expert")
        self.assertEqual(binding["inference"]["engine"], "torch-hf")
        self.assertEqual(binding["core_inference"]["engine"], "luban-native-bf16-metal")
        brain = ExternalBrain(runtime.memory)
        identifier = runtime.memory.add_sample("你好", "周三", source="human", approved=True,
            training_allowed=True, provenance={"messages": self.messages, "base_identity": runtime.identity})
        brain.register_reference(identifier, self.messages, runtime.identity, binding)
        for _ in range(2):
            result = brain.observe_local(self.messages, runtime.identity, binding, "周三", complete=True,
                actual_context=self.messages, reference_used=False, local_action="neural_expert")
        self.assertEqual(result["status"], "mastered")
        self.assertEqual(Runtime.external_hint(runtime, self.messages)["status"], "mastered")
        core_only = runtime.external_binding(actual_action="local_core")
        self.assertEqual(brain.lookup(self.messages, runtime.identity, core_only)["status"], "reference_ready")

    def test_hf_core_proof_during_full_load_cannot_become_native_core_mastery(self):
        runtime = self.runtime
        runtime.memory = Memory(self.root / "memory")
        runtime.memory.model_identity = runtime.identity
        runtime.training_session = None
        runtime._base_core_weight = torch.zeros(1, dtype=torch.bfloat16)
        runtime.external_binding = MethodType(Runtime.external_binding, runtime)
        runtime.cold_backbone.depth_count = 1
        binding = runtime.external_binding(actual_action="local_core")
        self.assertEqual(binding["inference"]["engine"], "torch-hf")
        brain = ExternalBrain(runtime.memory)
        identifier = runtime.memory.add_sample("你好", "周三", source="human", approved=True,
            training_allowed=True, provenance={"messages": self.messages, "base_identity": runtime.identity})
        brain.register_reference(identifier, self.messages, runtime.identity, binding)
        for _ in range(2):
            brain.observe_local(self.messages, runtime.identity, binding, "周三", complete=True,
                actual_context=self.messages, reference_used=False, local_action="local_core")
        runtime.cold_backbone.depth_count = 0
        self.assertEqual(Runtime.external_hint(runtime, self.messages)["status"], "reference_ready")


if __name__ == "__main__":
    unittest.main()
