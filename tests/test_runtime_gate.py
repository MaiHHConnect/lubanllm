"""The local API must isolate gated weights and restore them after failures."""
import unittest
import tempfile
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from luban.runtime import Runtime
from luban.context import ContextAdmissionError


class RuntimeGateTests(unittest.TestCase):
    def setUp(self):
        settings = Mock()
        settings.snapshot.return_value = {"local_thinking": "off", "max_new_tokens": 16, "temperature": 0}
        self.runtime = Runtime(settings, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime.pointer = Path(self.temp.name) / "active.json"
        self.runtime._ready = Mock()
        self.runtime._tokens = Mock(return_value=torch.tensor([[2, 3]]))
        self.runtime.context_status = Mock(return_value={"current_limit": 1024,
            "memory_estimate_limit": 1024, "linear_bytes_per_token": 1,
            "quadratic_bytes_per_token_squared": 1, "memory_budget_bytes": 1024 ** 3})
        self.runtime.tokenizer = SimpleNamespace(eos_token_id=1)
        self.original = object()
        self.runtime.branch = SimpleNamespace(expert=self.original, requires_gate=True)
        self.runtime.model = Mock()
        self.runtime.router_meta = {"calibrated": False}

    def test_core_and_unknown_actions_restore_expert(self):
        for action in ("local_core", None, "teacher"):
            self.runtime.choose_route = Mock(return_value={"action": action})
            def generate(tokens, **kwargs):
                self.assertIsNone(self.runtime.branch.expert)
                return torch.tensor([[2, 3, 1]])
            self.runtime.model.generate.side_effect = generate
            self.runtime.generate([], lambda _: None, Mock())
            self.assertIs(self.runtime.branch.expert, self.original)

    def test_calibrated_route_only_admits_the_shallow_model(self):
        self.runtime.router_meta = {"calibrated": True, "threshold": 0.8}
        self.runtime.shallow = Mock(return_value=torch.zeros(1, 2, 4))
        self.runtime.router = Mock(return_value=torch.tensor([[9., -9., -9.]]))
        self.runtime.choose_route([])
        self.runtime._tokens.assert_called_once_with([], shallow=True)

    def test_long_mlx_context_falls_back_to_core_without_truncating_or_hiding_bad_inputs(self):
        self.runtime.native_inference_precision = "mlx-q4"
        self.runtime.router_meta = {"calibrated": True, "feature_source": "mlx-q4-prefix"}
        self.runtime.shallow = Mock()
        messages = [{"role": "user", "content": "full context remains available to generation"}]
        self.runtime._tokens.side_effect = ContextAdmissionError("routing memory estimate")
        self.assertEqual(self.runtime.choose_route(messages)["source"], "router_context_limit")
        self.runtime._tokens.assert_called_once_with(messages, shallow=True)
        self.runtime._tokens.side_effect = None
        self.runtime._tokens.return_value = torch.ones((1, 40000), dtype=torch.long)
        self.assertEqual(self.runtime.choose_route(messages)["action"], "local_core")
        self.runtime.shallow.assert_not_called()
        self.runtime._tokens.side_effect = ValueError("invalid tokenizer input")
        with self.assertRaisesRegex(ValueError, "invalid tokenizer"):
            self.runtime.choose_route(messages)
        self.runtime.native_inference_precision = "bf16"
        self.runtime.router_meta["feature_source"] = "hf-bf16-prefix"
        self.runtime._tokens.side_effect = ContextAdmissionError("routing memory estimate")
        with self.assertRaises(ContextAdmissionError):
            self.runtime.choose_route(messages)

    def test_expert_route_without_scope_proof_falls_back_to_core(self):
        self.runtime.router_meta = {"calibrated": True, "threshold": 0.8,
                                    "gates_expert": True, "expert_id": "expert",
                                    "supported_actions": ["local_core", "neural_expert"]}
        self.runtime.branch.expert_id = "expert"
        self.runtime.branch._paths = Mock(return_value=(Path(self.temp.name) / "weights",
                                                         Path(self.temp.name) / "metadata"))
        self.runtime.branch._paths.return_value[1].write_text(json.dumps({"sha256": "same"}))
        self.runtime.router_meta["expert_sha256"] = "same"
        self.runtime.shallow = Mock(return_value=torch.ones(1, 2, 4))
        self.runtime.router = Mock(return_value=torch.tensor([[-9., 9., -9.]]))
        result = self.runtime.choose_route([])
        self.assertEqual(result["action"], "local_core")
        self.assertEqual(result["source"], "expert_scope_reject")

    def test_uncalibrated_gate_falls_back_to_the_existing_core(self):
        self.assertEqual(self.runtime.choose_route([])["action"], "local_core")

    def test_legacy_persisted_expert_is_forced_behind_gate(self):
        self.runtime.branch.expert_id = "learned-legacy"
        self.runtime.branch.requires_gate = False
        self.runtime._enforce_expert_gate()
        self.assertTrue(self.runtime.branch.requires_gate)
        self.assertFalse(self.runtime.router_meta["calibrated"])
        self.assertFalse(self.runtime.router_meta["gates_expert"])
        self.assertIsNone(self.runtime.router_meta["expert_id"])
        self.assertTrue(self.runtime.router_meta["legacy_expert_gate"])
        self.assertIn("independent route proof", self.runtime.router_meta["legacy_expert_gate_reason"])

    def test_gated_router_requires_direct_expert_generation_proof(self):
        splits = {name: [] for name in ("train", "development", "calibration", "test")}
        for name in splits:
            splits[name].append({"id": name, "route_action": "neural_expert"})
        good = [{"id": name, "branch_enabled": True, "complete": True, "passed": True}
                for name in splits]
        with patch("luban.runtime.full_backbone", return_value=nullcontext()), patch(
                "luban.evolution._generation_loaded", return_value=good):
            proof = self.runtime._prove_expert_routes(splits)
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["count"], 4)
        self.assertIs(self.runtime.branch.expert, self.original)
        failed = [dict(row) for row in good]
        failed[2] = {**failed[2], "passed": False}
        with patch("luban.runtime.full_backbone", return_value=nullcontext()), patch(
                "luban.evolution._generation_loaded", return_value=failed), self.assertRaisesRegex(
                    ValueError, "专家路由材料"):
            self.runtime._prove_expert_routes(splits)

    def test_expert_action_keeps_exact_module(self):
        self.runtime.choose_route = Mock(return_value={"action": "neural_expert"})
        def generate(tokens, **kwargs):
            self.assertIs(self.runtime.branch.expert, self.original)
            return torch.tensor([[2, 3, 1]])
        self.runtime.model.generate.side_effect = generate
        self.runtime.generate([], lambda _: None, Mock())
        self.assertIs(self.runtime.branch.expert, self.original)

    def test_learned_branch_uses_trained_template_and_reports_actual_mode(self):
        self.runtime.choose_route = Mock(return_value={"action": "neural_expert"})
        self.runtime.model.generate.return_value = torch.tensor([[2, 3, 1]])
        usage = self.runtime.generate([], lambda _: None, Mock(), thinking="medium")
        self.runtime._tokens.assert_called_once_with([], "off")
        self.assertEqual(usage["requested_thinking"], "medium")
        self.assertEqual(usage["effective_thinking"], "off")
        self.assertIs(self.runtime.branch.expert, self.original)

    def test_core_keeps_requested_thinking_template(self):
        self.runtime.choose_route = Mock(return_value={"action": "local_core"})
        self.runtime.model.generate.return_value = torch.tensor([[2, 3, 1]])
        usage = self.runtime.generate([], lambda _: None, Mock(), thinking="medium")
        self.runtime._tokens.assert_called_once_with([], "medium")
        self.assertEqual(usage["effective_thinking"], "medium")
        self.assertIs(self.runtime.branch.expert, self.original)

    def test_modified_core_uses_validated_template_without_an_expert(self):
        self.runtime.active_core = "core-0123456789ab"
        self.runtime.branch.expert = None
        self.runtime.choose_route = Mock(return_value={"action": "local_core"})
        self.runtime.model.generate.return_value = torch.tensor([[2, 3, 1]])
        usage = self.runtime.generate([], lambda _: None, Mock(), thinking="medium")
        self.runtime._tokens.assert_called_once_with([], "off")
        self.assertEqual(usage["requested_thinking"], "medium")
        self.assertEqual(usage["effective_thinking"], "off")
        self.assertEqual(usage["local_action"], "local_core")

    def test_compatible_older_expert_can_run_and_is_marked_for_review(self):
        self.runtime.active_core = "core-0123456789ab"
        self.runtime.branch_core_version = "core-old"
        self.runtime.branch.requires_gate = False
        def generate(tokens, **kwargs):
            self.assertIs(self.runtime.branch.expert, self.original)
            return torch.tensor([[2, 3, 1]])
        self.runtime.model.generate.side_effect = generate
        usage = self.runtime.generate([], lambda _: None, Mock())
        self.assertTrue(usage["module_requires_review"])
        self.assertEqual(usage["expert_core_version"], "core-old")
        self.assertIs(self.runtime.branch.expert, self.original)

    def test_generation_and_routing_failures_restore_expert(self):
        self.runtime.choose_route = Mock(return_value={"action": "local_core"})
        self.runtime.model.generate.side_effect = RuntimeError("fixture generation failure")
        with self.assertRaises(RuntimeError):
            self.runtime.generate([], lambda _: None, Mock())
        self.assertIs(self.runtime.branch.expert, self.original)
        self.runtime.choose_route.side_effect = RuntimeError("fixture routing failure")
        with self.assertRaises(RuntimeError):
            self.runtime.generate([], lambda _: None, Mock())
        self.assertIs(self.runtime.branch.expert, self.original)
        self.assertEqual(self.runtime.state, "ready")
