"""Replay the production route head without constructing the legacy BF16 model."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from luban.decision import FastDecisionHead
from luban.mlx_learning import check_routes


class MLXRouteReplayTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "upstream-file-manifest.json").write_text("{}")
        self.config = {"model_path": str(self.model), "native_mlx_model_path": str(self.model),
                       "native_inference_precision": "mlx-q4"}
        self.owner = SimpleNamespace(store=self.root, identity="identity",
                                     settings=SimpleNamespace(snapshot=lambda: self.config))
        self.meta = {"feature_source": "mlx-q4-prefix", "depth": 4, "calibrated": True,
                     "expert_id": "expert", "threshold": 0.8, "supported_actions": ["local_core", "teacher"]}
        head = FastDecisionHead(2048)
        with torch.no_grad():
            head.projection.weight.zero_()
            head.projection.bias.copy_(torch.tensor([10., -10., -10.]))
        save_file(head.state_dict(), str(self.root / "router.safetensors"))
        self.meta["sha256"] = hashlib.sha256((self.root / "router.safetensors").read_bytes()).hexdigest()
        self.save_pointers()
        self.rows = [dict(id=f"row-{i}", prompt=f"question {i}", answer="answer", purpose="training",
                          approved=True, training_allowed=True, source="human", provenance={}) for i in range(2)]

    def save_pointers(self):
        raw = json.dumps(self.meta).encode()
        (self.root / "router.json").write_bytes(raw)
        for name, data in [("active-core", dict(version="core")),
                           ("active", dict(id="expert", core_version="core")),
                           ("active-router", dict(id="router", core_version="core",
                                                  metadata_sha256=hashlib.sha256(raw).hexdigest()))]:
            (self.root / (name + ".json")).write_text(json.dumps(dict(identity="identity", **data)))

    def run_replay(self, *, mutate=False, fail=False):
        engine = SimpleNamespace(route_features=None, close=lambda: None)
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2])
        from unittest.mock import Mock
        engine.close = Mock()
        def features(*args):
            if mutate:
                (self.root / "active-core.json").write_text("{}")
            if fail:
                raise RuntimeError("worker routing failure")
            return [0.5] * 2048
        engine.route_features = Mock(side_effect=features)
        is_file = Path.is_file
        with patch("luban.native_engine.NativeEngine", return_value=engine) as create, \
             patch("luban.native_mlx.model_binding", return_value={"path": str(self.model)}), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch.object(Path, "is_file", lambda path: path.name == "luban-mlx-worker" or is_file(path)), \
             patch("luban.backbone.load_backbone", side_effect=AssertionError("BF16 must not load")):
            try:
                result = check_routes(self.owner, self.rows)
                self.assertEqual(create.call_count, 1)
                self.assertEqual(engine.route_features.call_count, 2)
                return result
            finally:
                engine.close.assert_called_once()

    def test_reuses_one_native_prefix_and_actual_decision_head(self):
        result = self.run_replay()
        self.assertEqual(result["feature_source"], "mlx-q4-prefix")
        self.assertTrue(result["released"])
        self.assertEqual([row["action"] for row in result["rows"]], ["local_core"] * 2)
        self.assertEqual([row["source"] for row in result["rows"]], ["calibrated_head"] * 2)

    def test_unmarked_and_old_bf16_heads_are_rejected_before_worker(self):
        for source in [None, "hf-bf16-prefix"]:
            self.meta.pop("feature_source", None)
            if source:
                self.meta["feature_source"] = source
            self.save_pointers()
            with patch("luban.native_engine.NativeEngine") as create:
                with self.assertRaisesRegex(ValueError, "来源未校准"):
                    check_routes(self.owner, self.rows)
                create.assert_not_called()

    def test_source_change_and_native_failure_close_resident_worker(self):
        with self.assertRaisesRegex(ValueError, "活动来源变化"):
            self.run_replay(mutate=True)
        self.save_pointers()
        with self.assertRaisesRegex(RuntimeError, "worker routing failure"):
            self.run_replay(fail=True)
