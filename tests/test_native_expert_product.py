"""Published expert bindings and real product route selection, without loading a model."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch
from luban.native_expert import native_expert_binding
from luban import product_inference as product


class NativeExpertProductTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "luban-worker"
        self.identity = "a" * 64
        self.proof = {"schema": "native-expert-q4-v1", "passed": True, "core_version": "core-123456789abc",
                      "core_file_sha256": "b" * 64, "core_tensor_sha256": "c" * 64,
                      "expert_file_sha256": "d" * 64, "worker_binary_sha256": "e" * 64,
                      "graph_binary_sha256": "f" * 64, "report": "learning-abcdef123456.json"}
        self.record = {"identity": self.identity + ":layer-23", "hidden": 2048, "width": 64,
                       "sha256": "d" * 64, "core_version": self.proof["core_version"],
                       "requires_gate": True, "native_q4": self.proof}
        self.pointer = {"id": "learned-abcdef123456", "identity": self.identity, "width": 64,
                        "core_version": self.proof["core_version"]}
        self.runtime = SimpleNamespace(active_core=self.proof["core_version"], identity=self.identity,
            native_inference_precision="mixed-q4-g16", store=self.root, pointer=self.root / "active.json",
            branch=SimpleNamespace(expert_id=self.pointer["id"], expert=None, requires_gate=True,
                _paths=lambda _: (self.root / "expert.safetensors", self.root / "expert.json")))
        self.write("active.json", self.pointer)
        self.write("active-core.json", {"version": self.runtime.active_core, "identity": self.identity, "sha256": "b" * 64})
        self.write("expert.json", self.record)
        self.write("luban-worker.json", {"binary_sha256": "e" * 64})
        self.report = {"identity": self.identity, "quality_checked": True, "gates": {"retention": True}, "native_q4": self.proof}
        self.write(self.proof["report"], self.report)

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_binding_requires_current_core_engine_and_complete_published_proof(self):
        binding = native_expert_binding(self.runtime, self.binary)
        self.assertEqual(binding["sha256"], "d" * 64)
        self.assertEqual(binding["core_sha256"], "b" * 64)
        for key, value in (("passed", False), ("core_version", "other"),
                           ("worker_binary_sha256", "0" * 64), ("expert_file_sha256", "0" * 64),
                           ("report", "../learning-abcdef123456.json")):
            changed = copy.deepcopy(self.record)
            changed["native_q4"][key] = value
            self.write("expert.json", changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                native_expert_binding(self.runtime, self.binary)
        self.write("expert.json", self.record)
        self.write(self.proof["report"], {**self.report, "gates": {"retention": False}})
        with self.assertRaises(ValueError):
            native_expert_binding(self.runtime, self.binary)

    def test_legacy_expert_and_different_precision_do_not_gain_q4_qualification(self):
        self.runtime.native_inference_precision = "bf16"
        self.assertIsNone(native_expert_binding(self.runtime, self.binary))
        self.runtime.native_inference_precision = "mixed-q4-g16"
        self.write("expert.json", {k: v for k, v in self.record.items() if k != "native_q4"})
        self.assertIsNone(native_expert_binding(self.runtime, self.binary))

    def test_qualified_routed_expert_stays_out_of_full_hf_and_core_switch_closes_cache(self):
        runtime = self.runtime
        runtime.router_meta = {"gates_expert": True}
        runtime.external_hint = lambda _: None
        runtime.choose_route = lambda _: {"action": "neural_expert"}
        runtime.branch.activate = Mock(side_effect=lambda *args, **kwargs: setattr(runtime.branch, "expert", object()))
        runtime.cold_backbone = SimpleNamespace(depth_count=0, loaded=False, evict_resident=Mock())
        runtime.device = "cpu"
        runtime.tokenizer = SimpleNamespace(eos_token_id=9)
        runtime.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9))
        expert_binding = native_expert_binding(runtime, self.binary)
        expert_engine = SimpleNamespace(expert=expert_binding, close=Mock(), generate=Mock(return_value={"tokens": [9], "finish_reason": "stop"}))
        core_engine = SimpleNamespace(expert=None, close=Mock(), generate=Mock(return_value={"tokens": [9], "finish_reason": "stop"}))
        runtime.native_engine = expert_engine
        def start(_):
            runtime.native_engine = core_engine
            return core_engine
        with patch.object(product, "native_binary", return_value=self.binary), patch.object(product, "full_backbone") as hf, patch.object(product, "start", side_effect=start):
            with product.answer_backbone(runtime, [{"role": "user", "content": "test"}]):
                product.generate_tokens(runtime, torch.tensor([[1]]), 8)
            runtime.branch.activate.assert_called_once_with(self.pointer["id"], device="cpu")
            expert_engine.generate.assert_called_once()
            hf.assert_not_called()
            runtime.branch.expert = None
            product.generate_tokens(runtime, torch.tensor([[1]]), 8, cache_reuse=True)
            expert_engine.close.assert_called_once()
            core_engine.generate.assert_called_once()
