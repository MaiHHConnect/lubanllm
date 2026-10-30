import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from luban.context import GIB, PRESETS, check_input, window, output_budget, fit_context, generation_plan
from luban.settings import Settings
from luban.runtime import Runtime
from unittest.mock import Mock, patch
import torch


MODEL = SimpleNamespace(max_position_embeddings=262144, hidden_size=2048,
    num_attention_heads=8, num_key_value_heads=2, head_dim=256,
    layer_types=["linear_attention"] * 18 + ["full_attention"] * 6)


class ContextTests(unittest.TestCase):
    def test_reserved_mlx_kv_changes_only_storage_estimate_and_keeps_memory_guards(self):
        cfg = {"context_mode": "auto", "context_tokens": 262144}
        legacy = window(cfg, MODEL, 5 * GIB, device="mps", prefill_step=128)
        reserved = window(cfg, MODEL, 5 * GIB, device="mps", prefill_step=128,
                          shared_kv_reservation=True)
        self.assertEqual(legacy["linear_bytes_per_token"], 24576)
        self.assertEqual(reserved["linear_bytes_per_token"], 14336)
        self.assertEqual(legacy["kv_bytes_per_token"], 12288)
        self.assertEqual(reserved["kv_bytes_per_token"], 12288)
        for key in ("memory_budget_bytes", "available_memory_bytes", "additional_weight_bytes",
                    "quadratic_bytes_per_token_squared", "configured_limit", "model_limit"):
            self.assertEqual(reserved[key], legacy[key], key)
        self.assertGreater(reserved["memory_estimate_limit"], legacy["memory_estimate_limit"])
        check_input(reserved["current_limit"] - 16, 16, reserved)
        with self.assertRaises(ValueError):
            check_input(reserved["current_limit"], 16, reserved)
        empty = window(cfg, MODEL, 0, device="mps", prefill_step=32, shared_kv_reservation=True)
        with self.assertRaisesRegex(ValueError, "可用内存不足"):
            check_input(1, 1, empty)
        self.assertEqual(window(cfg, MODEL, 5 * GIB, device="mps", shared_kv_reservation=True),
                         window(cfg, MODEL, 5 * GIB, device="mps"))

    def test_resident_prefix_credit_uses_one_full_kv_not_half_reserved_estimate(self):
        status = window({"context_mode": "auto", "context_tokens": 262144}, MODEL,
                        4679450624, device="mps", prefill_step=128, shared_kv_reservation=True)
        prefix = {"tokens": 139136, "bytes": 1800000000, "prefill_step": 64}
        plan = fit_context(status, 139255, 8192, prefix=prefix)
        self.assertEqual(plan["resident_prefix_bytes"], 139136 * 12288)
        smaller = fit_context(status, 139255, 8192, prefix={**prefix, "bytes": 123456})
        self.assertEqual(smaller["resident_prefix_bytes"], 123456)
        legacy = {**status, "linear_bytes_per_token": 24576}
        legacy.pop("kv_bytes_per_token")
        self.assertEqual(fit_context(legacy, 139255, 8192, prefix=prefix)["resident_prefix_bytes"],
                         139136 * 12288)

    def test_reserved_mlx_kv_requires_live_text_worker_with_explicit_capability(self):
        runtime = Runtime.__new__(Runtime)
        runtime.settings = SimpleNamespace(snapshot=lambda: {"context_mode": "auto", "context_tokens": 262144})
        runtime.text_config = MODEL
        runtime.dtype, runtime.device = torch.bfloat16, "cpu"
        runtime.native_inference_precision = "mlx-q4"
        runtime.cold_backbone = SimpleNamespace(depth_count=0)

        def worker(**changes):
            return SimpleNamespace(**{"process": SimpleNamespace(poll=lambda: None), "precision": "mlx-q4",
                                      "vision": False, "loaded": {"mlx_kv_reservation": True}, **changes})

        with patch("luban.runtime.available_memory_bytes", return_value=5 * GIB), \
                patch("luban.product_inference.native_binary", return_value=Path("/fixture/luban-worker")):
            runtime.native_engine = worker()
            reserved = runtime.context_status()
            self.assertEqual(reserved["linear_bytes_per_token"], 14336)
            for engine in (None, worker(process=None), worker(process=SimpleNamespace(poll=lambda: 1)),
                           worker(loaded=None), worker(loaded={}), worker(loaded={"mlx_kv_reservation": False}),
                           worker(loaded={"mlx_kv_reservation": 1}), worker(vision=True), worker(precision="bf16")):
                with self.subTest(engine=engine):
                    runtime.native_engine = engine
                    legacy = runtime.context_status()
                    self.assertEqual(legacy["linear_bytes_per_token"], 24576)
                    self.assertEqual(legacy["memory_budget_bytes"], reserved["memory_budget_bytes"])
            runtime.native_engine = worker()
            for shallow, depth, precision in ((True, 0, "mlx-q4"), (False, 1, "mlx-q4"), (False, 0, "bf16")):
                runtime.cold_backbone.depth_count = depth
                runtime.native_inference_precision = precision
                status = runtime.context_status(shallow=shallow)
                self.assertIsNone(status["prefill_step"])
                self.assertEqual(status["linear_bytes_per_token"], 77824)

    def test_resident_prefix_is_counted_once_and_keeps_its_qualified_step(self):
        status = window({"context_mode": "auto", "context_tokens": 262144}, MODEL,
                        4679450624, device="mps", prefill_step=128)
        prefix = {"tokens": 139136, "bytes": 1800000000, "prefill_step": 64}
        with self.assertRaises(ValueError):
            check_input(139255, 8192, fit_context(status, 139255, 8192))
        plan = fit_context(status, 139255, 8192, prefix=prefix)
        self.assertEqual(plan["prefill_step"], 64)
        self.assertEqual(plan["resident_prefix_bytes"], 139136 * 12288)
        self.assertEqual(plan["memory_budget_bytes"], status["memory_budget_bytes"])
        self.assertEqual(plan["available_memory_bytes"], status["available_memory_bytes"])
        check_input(139255, 8192, plan)
        # A step change discards the saved state and therefore all credit.
        tight = {**status, "memory_budget_bytes": 2300000000}
        rejected = fit_context(tight, 139255, 8192, prefix=prefix)
        self.assertEqual(rejected["prefill_step"], 32)
        self.assertEqual(rejected["resident_prefix_bytes"], 0)
        with self.assertRaises(ValueError):
            check_input(139255, 8192, rejected)
        manual = {**status, "mode": "manual", "configured_limit": 65536, "current_limit": 65536}
        with self.assertRaises(ValueError):
            check_input(139255, 8192, fit_context(manual, 139255, 8192, prefix=prefix))

    def test_output_admission_can_release_a_prefix_slot_then_recheck_without_invented_memory(self):
        status = {"mode": "auto", "current_limit": 512, "memory_estimate_limit": 48,
                  "configured_limit": 512, "model_limit": 512, "prefill_step": 128,
                  "memory_budget_bytes": 48, "linear_bytes_per_token": 1,
                  "quadratic_bytes_per_token_squared": 0, "native_output_limit": 2048}
        after = {**status, "memory_estimate_limit": 512, "memory_budget_bytes": 512}
        engine = SimpleNamespace(_prefix_tokens=[1], prepare_prompt_cache=Mock())
        runtime = SimpleNamespace(context_status=Mock(side_effect=[status, after]), native_engine=engine)
        plan, maximum = generation_plan(runtime, torch.ones((1, 32), dtype=torch.long), 64)
        self.assertIs(plan, after)
        self.assertEqual(maximum, 64)
        engine.prepare_prompt_cache.assert_called_once_with([1] * 32, cache_reuse=False)
        self.assertEqual(runtime.context_status.call_count, 2)
        runtime.context_status = Mock(return_value=status)
        with self.assertRaises(ValueError):
            generation_plan(runtime, torch.ones((1, 32), dtype=torch.long), 64)
        self.assertEqual(runtime.context_status.call_count, 2)

    def test_adaptive_chunks_admit_actual_failed_lengths_with_the_same_budget(self):
        for original_limit, total in ((119296, 145776), (120832, 147368)):
            status = {**window({"context_mode": "auto", "context_tokens": 262144}, MODEL,
                              6 * GIB, device="mps", prefill_step=128),
                      "current_limit": original_limit, "memory_estimate_limit": original_limit,
                      "memory_budget_bytes": original_limit * 32768}
            with self.assertRaises(ValueError):
                check_input(total - 8192, 8192, status)
            adapted = fit_context(status, total - 8192, 8192)
            self.assertEqual(adapted["prefill_step"], 32)
            self.assertEqual(adapted["memory_budget_bytes"], status["memory_budget_bytes"])
            self.assertEqual(adapted["available_memory_bytes"], status["available_memory_bytes"])
            self.assertEqual(status["prefill_step"], 128)
            check_input(total - 8192, 8192, adapted)
            self.assertIs(fit_context(status, 1000, 64), status)
            self.assertEqual(fit_context(status, 130000, 1000)["prefill_step"], 64)

    def test_adaptive_chunks_keep_settings_and_reject_insufficient_memory(self):
        for mode in ("auto", "manual"):
            status = window({"context_mode": mode, "context_tokens": 131072}, MODEL,
                            6 * GIB, device="mps", prefill_step=128)
            with self.assertRaises(ValueError):
                check_input(145000, 8192, fit_context(status, 145000, 8192))
            insufficient = {**status, "memory_budget_bytes": 1, "memory_estimate_limit": 0}
            with self.assertRaises(ValueError):
                check_input(1000, 16, fit_context(insufficient, 1000, 16))

    def test_default_and_legacy_settings_migrate_to_automatic(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home)
            self.assertEqual(settings.snapshot()["context_mode"], "auto")
            self.assertEqual(settings.snapshot()["context_tokens"], 262144)
            data = settings.snapshot()
            data.pop("context_mode")
            data["context_tokens"] = 4096
            Path(home, "settings.json").write_text(json.dumps(data))
            self.assertEqual(Settings(home).snapshot()["context_tokens"], 262144)

    def test_only_manual_presets_and_automatic_ceiling_are_persisted(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home)
            for tokens in PRESETS:
                settings.update({"context_mode": "manual", "context_tokens": tokens})
                self.assertEqual(Settings(home).snapshot()["context_tokens"], tokens)
            for invalid in (4096, 64000, 300000, True):
                with self.assertRaises(ValueError):
                    settings.update({"context_tokens": invalid})
            settings.update({"context_mode": "auto"})
            self.assertEqual(settings.snapshot()["context_tokens"], 262144)

    def test_automatic_limit_tracks_memory_and_cold_weights(self):
        cfg = {"context_mode": "auto", "context_tokens": 262144}
        small = window(cfg, MODEL, 4 * GIB, device="mps")
        large = window(cfg, MODEL, 12 * GIB, device="mps")
        cold = window(cfg, MODEL, 12 * GIB, device="mps", weight_bytes=3 * GIB)
        self.assertLess(small["current_limit"], large["current_limit"])
        self.assertLess(cold["current_limit"], large["current_limit"])
        self.assertLessEqual(large["current_limit"], 262144)
        self.assertTrue(large["estimated"])
        self.assertEqual(window(cfg, MODEL, 0, device="mps")["current_limit"], 0)

    def test_manual_capacity_does_not_allocate_it_or_bypass_memory_admission(self):
        value = window({"context_mode": "manual", "context_tokens": 262144}, MODEL,
                       8 * GIB, device="mps")
        self.assertEqual(value["current_limit"], 262144)
        check_input(100, 512, value)
        with self.assertRaisesRegex(ValueError, "可用内存"):
            check_input(65536, 512, value)

    def test_chunked_mlx_admits_long_prompts_without_bypassing_memory_or_settings(self):
        cfg = {"context_mode": "auto", "context_tokens": 262144}
        legacy = window(cfg, MODEL, 8 * GIB, device="mps")
        mlx = window(cfg, MODEL, 8 * GIB, device="mps", prefill_step=512)
        self.assertLess(legacy["current_limit"], 24800)
        self.assertGreater(mlx["current_limit"], 65536)
        check_input(40000, 16, mlx)
        for capacity in PRESETS:
            status = window({"context_mode": "manual", "context_tokens": capacity}, MODEL,
                            8 * GIB, device="mps", prefill_step=512)
            self.assertEqual(status["current_limit"], capacity)
            check_input(40000, 16, status)
            with self.assertRaisesRegex(ValueError, "预留输出"):
                check_input(capacity, 16, status)
        with self.assertRaisesRegex(ValueError, "可用内存"):
            check_input(262000, 16, status)
        cold = window(cfg, MODEL, 8 * GIB, device="mps", weight_bytes=2 * GIB, prefill_step=512)
        self.assertLess(cold["current_limit"], mlx["current_limit"])
        self.assertEqual(window(cfg, MODEL, 0, device="mps", prefill_step=512)["current_limit"], 0)

    def test_mlx_runtime_uses_selected_window_and_chunk_estimate_only_for_generation(self):
        runtime = Runtime.__new__(Runtime)
        config = {"context_mode": "manual", "context_tokens": 262144}
        runtime.settings = SimpleNamespace(snapshot=lambda: config)
        runtime.text_config = MODEL
        runtime.dtype, runtime.device = torch.bfloat16, "cpu"
        runtime.native_inference_precision = "mlx-q4"
        runtime.native_engine = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = torch.ones((1, 40000), dtype=torch.long)
        with patch("luban.runtime.available_memory_bytes", return_value=8 * GIB), \
                patch("luban.product_inference.native_binary", return_value=Path("/fixture/luban-worker")):
            for capacity in PRESETS:
                config["context_tokens"] = capacity
                self.assertEqual(runtime.context_status()["native_limit"], capacity)
                self.assertEqual(runtime.context_status()["current_limit"], capacity)
                self.assertEqual(runtime.context_status()["prefill_step"], 256 if capacity == 65536 else 128)
                self.assertEqual(runtime._tokens([], output_reserve=16).shape[1], 40000)
            self.assertIsNone(runtime.context_status(shallow=True)["prefill_step"])
            runtime.cold_backbone = SimpleNamespace(depth_count=1)
            self.assertIsNone(runtime.context_status()["prefill_step"])
            runtime.cold_backbone.depth_count = 0
            config["context_mode"] = "auto"
            self.assertGreater(runtime.context_status()["current_limit"], 32768)
            config["context_tokens"] = 65536
            self.assertLessEqual(runtime.context_status()["current_limit"], 65536)

    def test_smaller_mlx_chunks_admit_actual_agent_length_with_cache_allowance(self):
        cfg = {"context_mode": "manual", "context_tokens": 262144}
        old = window(cfg, MODEL, 7 * GIB, device="mps", prefill_step=512)
        with self.assertRaisesRegex(ValueError, "可用内存"):
            check_input(115625, 8192, old)
        current = window(cfg, MODEL, 7 * GIB, device="mps", prefill_step=128)
        check_input(115625, 8192, current)
        with self.assertRaisesRegex(ValueError, "可用内存"):
            check_input(240000, 8192, current)

    def test_zero_memory_reports_loading_blocker_for_explicit_and_auto_output(self):
        for mode in ("auto", "manual"):
            cfg = {"context_mode": mode, "context_tokens": 262144}
            status = window(cfg, MODEL, 2 * GIB, device="mps", weight_bytes=3 * GIB)
            for requested in (0, 16):
                with self.assertRaisesRegex(ValueError, "可用内存不足.*设置未被改为0"):
                    output_budget(13, requested, status)
            self.assertEqual(status["configured_limit"], 262144)
            self.assertEqual(status["memory_budget_bytes"], 0)

    def test_model_ceiling_and_output_reservation_are_enforced(self):
        model = SimpleNamespace(**{**vars(MODEL), "max_position_embeddings": 1024})
        value = window({"context_mode": "manual", "context_tokens": 65536}, model,
                       16 * GIB, device="mps")
        self.assertEqual(value["current_limit"], 1024)
        with self.assertRaisesRegex(ValueError, "预留输出"):
            check_input(1000, 32, value)
        check_input(900, 32, value)

    def test_automatic_output_tracks_real_admitted_context_and_remaining_space(self):
        status = window({"context_mode": "auto", "context_tokens": 262144}, MODEL,
                        8 * GIB, device="cpu")
        self.assertEqual(output_budget(100, 0, status), status["current_limit"] // 4)
        self.assertEqual(output_budget(status["current_limit"] - 11, None, status), 11)
        with self.assertRaisesRegex(ValueError, "输出空间"):
            output_budget(status["current_limit"], 0, status)

    def test_manual_256k_does_not_make_automatic_output_bypass_memory_admission(self):
        status = window({"context_mode": "manual", "context_tokens": 262144}, MODEL,
                        8 * GIB, device="cpu")
        self.assertLess(output_budget(100, 0, status), 65536)
        self.assertEqual(output_budget(100, 0, status), status["memory_estimate_limit"] // 4)
        with self.assertRaisesRegex(ValueError, "可用内存"):
            output_budget(65536, 16, status)

    def test_explicit_output_is_not_silently_capped_and_engine_limit_is_truthful(self):
        status = window({"context_mode": "manual", "context_tokens": 262144}, MODEL,
                        4000 * GIB, device="cpu")
        self.assertEqual(output_budget(100, 8192, status), 8192)
        self.assertEqual(output_budget(100, 0, status), 32768)
        status["native_output_limit"] = 2048
        self.assertEqual(output_budget(100, 0, status), 8192)
        status["current_limit"] = 32768
        self.assertEqual(output_budget(100, 0, status), 8192)
        self.assertEqual(output_budget(32760, 0, status), 8)
        status["memory_estimate_limit"] = 4096
        self.assertEqual(output_budget(100, 0, status), 1024)
        with self.assertRaisesRegex(ValueError, "执行器最多输出 2048"):
            output_budget(100, 8192, status)
        for invalid in (True, -1, 32769, 1.5, "512"):
            with self.assertRaises(ValueError):
                output_budget(100, invalid, status)

    def test_output_setting_accepts_automatic_and_keeps_saved_manual_choice(self):
        with tempfile.TemporaryDirectory() as home:
            settings = Settings(home)
            self.assertEqual(settings.snapshot()["max_new_tokens"], 0)
            for value in (512, 32768, 0):
                settings.update({"max_new_tokens": value})
                self.assertEqual(Settings(home).snapshot()["max_new_tokens"], value)
            for value in (-1, 15, 32769, True):
                with self.assertRaises(ValueError):
                    settings.update({"max_new_tokens": value})

    def test_runtime_checks_actual_token_count_and_output_before_device_transfer(self):
        runtime = Runtime.__new__(Runtime)
        runtime.settings = SimpleNamespace(snapshot=lambda: {"context_mode": "manual", "context_tokens": 65536})
        runtime.text_config = SimpleNamespace(**{**vars(MODEL), "max_position_embeddings": 1024})
        runtime.decoder = torch.nn.Linear(2, 2, device="meta")
        runtime.model = None
        runtime.dtype, runtime.device = torch.bfloat16, "cpu"
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = torch.ones((1, 1000), dtype=torch.long)
        with patch("luban.runtime.available_memory_bytes", return_value=16 * GIB):
            self.assertGreater(runtime.context_status()["additional_weight_bytes"], 0)
            with self.assertRaisesRegex(ValueError, "预留输出"):
                runtime._tokens([{"role": "user", "content": "fixture"}], output_reserve=32)
            self.assertEqual(runtime._tokens([], output_reserve=16).shape[1], 1000)

    def test_shallow_route_does_not_reserve_cold_weights_but_keeps_memory_guard(self):
        runtime = Runtime.__new__(Runtime)
        runtime.settings = SimpleNamespace(snapshot=lambda: {"context_mode": "auto", "context_tokens": 262144})
        runtime.text_config = MODEL
        # Meta tensor has real shape/bytes without allocating a GiB in this test.
        runtime.decoder = torch.nn.Linear(16384, 16384, bias=False, device="meta")
        runtime.dtype, runtime.device = torch.bfloat16, "cpu"
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = torch.ones((1, 47), dtype=torch.long)
        with patch("luban.runtime.available_memory_bytes", return_value=2 * GIB):
            self.assertEqual(runtime.context_status()["current_limit"], 0)
            with self.assertRaises(ValueError):
                runtime._tokens([])
            self.assertEqual(runtime._tokens([], shallow=True).shape[1], 47)
            self.assertEqual(runtime.context_status(shallow=True)["additional_weight_bytes"], 0)
        with patch("luban.runtime.available_memory_bytes", return_value=0):
            with self.assertRaises(ValueError):
                runtime._tokens([], shallow=True)

    def test_resident_native_child_does_not_reserve_a_second_full_core(self):
        runtime = Runtime.__new__(Runtime)
        runtime.settings = SimpleNamespace(snapshot=lambda: {"context_mode": "auto", "context_tokens": 262144})
        runtime.text_config = MODEL
        runtime.decoder = torch.nn.Linear(32768, 32768, bias=False, device="meta", dtype=torch.bfloat16)
        runtime.dtype, runtime.device = torch.bfloat16, "cpu"
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = torch.ones((1, 47), dtype=torch.long)
        process = SimpleNamespace(poll=lambda: None)
        runtime.native_engine = SimpleNamespace(process=process)
        with patch("luban.runtime.available_memory_bytes", return_value=2 * GIB), \
                patch("luban.product_inference.native_binary", return_value=Path("/fixture/luban-worker")):
            resident = runtime.context_status()
            self.assertEqual(resident["additional_weight_bytes"], 0)
            self.assertGreater(resident["current_limit"], 559)
            self.assertEqual(runtime._tokens([], output_reserve=512).shape[1], 47)
            for engine in (None, SimpleNamespace(process=None),
                           SimpleNamespace(process=SimpleNamespace(poll=lambda: 1))):
                with self.subTest(engine=engine):
                    runtime.native_engine = engine
                    closed = runtime.context_status()
                    self.assertEqual(closed["available_memory_bytes"], resident["available_memory_bytes"])
                    self.assertEqual(closed["additional_weight_bytes"], 2 * GIB)
                    self.assertEqual(closed["current_limit"], 0)
                    with self.assertRaises(ValueError):
                        runtime._tokens([], output_reserve=512)
                    shallow = runtime.context_status(shallow=True)
                    self.assertEqual(shallow["additional_weight_bytes"], 0)
                    self.assertEqual(shallow["current_limit"], resident["current_limit"])
                    self.assertEqual(runtime._tokens([], shallow=True).shape[1], 47)
            runtime.native_engine = SimpleNamespace(process=process)
            with patch("luban.runtime.available_memory_bytes", return_value=0):
                self.assertEqual(runtime.context_status()["current_limit"], 0)
                with self.assertRaises(ValueError):
                    runtime._tokens([], output_reserve=512)

    def test_native_cold_start_budgets_all_weights_including_live_hf_route(self):
        runtime = Runtime.__new__(Runtime)
        runtime.settings = SimpleNamespace(snapshot=lambda: {"context_mode": "auto", "context_tokens": 65536})
        runtime.text_config = MODEL
        runtime.dtype, runtime.device = torch.bfloat16, "cpu"
        runtime.decoder = torch.nn.Module()
        runtime.decoder.add_module("resident", torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16))
        runtime.decoder.add_module("cold", torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device="meta"))
        runtime.cold_backbone = SimpleNamespace(depth_count=0, resident_modules=lambda: (runtime.decoder.resident,))
        runtime.native_engine = None
        with patch("luban.runtime.available_memory_bytes", return_value=4 * GIB), \
                patch("luban.product_inference.native_binary", return_value=Path("/fixture/luban-worker")):
            self.assertEqual(runtime.context_status()["additional_weight_bytes"], (64 + 256) * 2)
            self.assertEqual(runtime.context_status(shallow=True)["additional_weight_bytes"], 0)
            runtime.decoder.resident.to_empty(device="meta")
            self.assertEqual(runtime.context_status(shallow=True)["additional_weight_bytes"], 64 * 2)
            runtime.native_engine = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
            self.assertEqual(runtime.context_status()["additional_weight_bytes"], 0)
            runtime.cold_backbone.depth_count = 1
            self.assertEqual(runtime.context_status()["additional_weight_bytes"], (64 + 256) * 2)


if __name__ == "__main__":
    unittest.main()
