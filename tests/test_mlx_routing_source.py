import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from luban.decision import FastDecisionHead
from luban.native_mlx_routing import pooled_features, source_matches
from luban.runtime import Runtime


class MLXRoutingSourceTests(unittest.TestCase):
    def test_old_head_cannot_claim_quantized_feature_calibration(self):
        runtime = SimpleNamespace(native_inference_precision="mlx-q4")
        self.assertFalse(source_matches(runtime, {"calibrated": True}))
        self.assertTrue(source_matches(runtime, {"feature_source": "mlx-q4-prefix"}))
        runtime.native_inference_precision = "bf16"
        self.assertTrue(source_matches(runtime, {}))
        self.assertFalse(source_matches(runtime, {"feature_source": "mlx-q4-prefix"}))

    def test_resident_worker_is_reused_without_hf_or_restart(self):
        engine = Mock()
        engine.route_features.return_value = [0.5] * 2048
        runtime = SimpleNamespace(native_engine=engine, device="cpu", native_inference_precision="mlx-q4")
        with patch("luban.product_inference.start", side_effect=AssertionError("restart")):
            actual = Runtime.shallow(runtime, torch.tensor([[12, 34, 56]]))
        self.assertEqual(actual.shape, (1, 1, 2048))
        self.assertEqual(engine.route_features.call_args.args[0], [12, 34, 56])
        self.assertTrue(torch.all(actual == .5))

    def test_source_mismatch_falls_back_before_stale_proof_or_worker(self):
        runtime = SimpleNamespace(native_inference_precision="mlx-q4", _ready=lambda: None,
            external_hint=lambda _: None, router_meta={"calibrated": True, "execution_scope": "local-trial"})
        with patch("luban.product_inference.native_binary", side_effect=AssertionError("worker touched")):
            actual = Runtime.choose_route(runtime, [])
        self.assertEqual(actual["action"], "local_core")
        self.assertEqual(actual["source"], "router_feature_source_changed")

    def test_pooled_vector_mask_does_not_reuse_prompt_length(self):
        head = FastDecisionHead(2048)
        with torch.no_grad():
            head.projection.weight.zero_()
            head.projection.bias.copy_(torch.tensor([10., -10., -10.]))
        runtime = SimpleNamespace(native_inference_precision="mlx-q4", _ready=lambda: None,
            external_hint=lambda _: None, router=head,
            router_meta={"calibrated": True, "feature_source": "mlx-q4-prefix"},
            _tokens=lambda *a, **k: torch.tensor([[12, 34, 56]]),
            shallow=lambda _: torch.ones(1, 1, 2048))
        self.assertEqual(Runtime.choose_route(runtime, [])["action"], "local_core")

    def test_batch_rejected_before_start(self):
        with self.assertRaises(ValueError), patch("luban.product_inference.start") as start:
            pooled_features(SimpleNamespace(), torch.tensor([[1], [2]]))
        start.assert_not_called()
