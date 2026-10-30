"""A failed finalization must not leave a countable successful upgrade report."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from luban.dynamic import DynamicMLP
from luban.runtime import Runtime
from luban.settings import private_json


class TinyGreedyModel(torch.nn.Module):
    """Real CPU branch forward/argmax; no substituted generation evaluator."""
    def __init__(self, branch, answer_token=1, complete=True):
        super().__init__()
        self.branch = branch
        self.config = SimpleNamespace(model_type="fixture")
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.complete, self.generated = complete, []
        with torch.no_grad():
            branch.base.weight.zero_()
            branch.base.bias.fill_(-10)
            branch.base.bias[answer_token] = 10

    def forward(self, ids, **kwargs):
        features = torch.nn.functional.one_hot(ids % 4, num_classes=4).float()
        logits = self.branch(features)
        # A real differentiable update, while explicit loss measurements below
        # isolate the publication gate from optimizer tuning in this tiny graph.
        loss = sum(p.square().sum() for p in self.branch.expert.parameters()) * 0.001 + 1
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, ids, **kwargs):
        token = self(ids).logits[:, -1].argmax(-1)[:, None]
        self.generated.append({"token": int(token[0, 0]), "max_new_tokens": kwargs["max_new_tokens"]})
        suffix = torch.cat((token, torch.full_like(token, 3)), dim=1) if self.complete else token
        return torch.cat((ids, suffix), dim=1)


class UpgradeCommitTests(unittest.TestCase):
    def run_learning(self, directory, failed=False, *, answer_token=1, complete=True,
                     previous=False, validation_provenance=None):
        memory = Mock()
        memory.samples.side_effect = lambda split: [
            {"id": split + str(i), "prompt": split + " independent " + str(i), "answer": "expected answer",
             "source": "human", "approved": 1, "training_allowed": 1,
             "provenance": json.dumps((validation_provenance or [{}, {}])[i] if split == "validation" else {})}
            for i in range(2)]
        if failed:
            memory.event.side_effect = OSError("fixture event write failed")
        runtime = Runtime(Mock(), memory)
        runtime.settings.snapshot.return_value = {"context_tokens": 4096, "context_mode": "manual"}
        runtime.device, runtime.dtype = "cpu", torch.float32
        runtime.store, runtime.pointer = Path(directory), Path(directory) / "active.json"
        runtime.identity = "a" * 64
        runtime._ready = Mock()
        runtime.text_config = SimpleNamespace(hidden_size=4)
        runtime.base_count = 20
        runtime.branch = DynamicMLP(torch.nn.Linear(4, 4), 4, 64,
                                    Path(directory) / "layer-0", runtime.identity + ":layer-0", 100000)
        if previous:
            runtime.branch.activate("previous", create=True)
            runtime.branch.hibernate()
            runtime.branch.activate("previous")
            private_json(runtime.pointer, {"id": "previous", "width": 64,
                                           "identity": runtime.identity, "core_version": None})
        runtime.old_pointer = runtime.pointer.read_bytes() if runtime.pointer.exists() else None
        runtime.old_weights = ({key: value.clone() for key, value in runtime.branch.expert.state_dict().items()}
                               if runtime.branch.expert is not None else None)
        runtime._sample_tokens = Mock(return_value=(torch.tensor([[1, 2]]), torch.tensor([[1, 2]])))
        runtime._loss = Mock(side_effect=[2., 2.] + [1., 1.] * 4)
        runtime._tokens = lambda messages, thinking, output_reserve=0: torch.tensor([[0]])
        runtime.tokenizer = SimpleNamespace(eos_token_id=3, decode=lambda tokens, **kwargs:
                                            "expected answer" if int(tokens[0]) == 1 else "wrong answer")
        runtime.model = TinyGreedyModel(runtime.branch, answer_token, complete)
        with patch("luban.runtime.save_training_backup", return_value="fixture-backup"):
            if failed:
                with self.assertRaisesRegex(OSError, "event write failed"):
                    runtime.learn(steps=1)
            else:
                result = runtime.learn(steps=1)
                self.assertEqual(result["promoted"], answer_token == 1 and complete
                                 and not any(p.get("generation_check", {}).get("forbidden")
                                             for p in (validation_provenance or [])))
                self.assertEqual(result["replay_before"], 2)
                self.assertEqual(result["steps"], 1)
        return runtime

    def test_failed_event_publication_rolls_back_without_a_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.run_learning(directory, failed=True)
            self.assertEqual(list(Path(directory).glob("learning-*.json")), [])
            self.assertFalse(runtime.pointer.exists())
            self.assertIsNone(runtime.branch.expert)
            self.assertEqual(runtime.job["state"], "error")

    def test_completed_publication_keeps_report_and_weight_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.run_learning(directory)
            self.assertEqual(len(list(Path(directory).glob("learning-*.json"))), 1)
            self.assertTrue(runtime.pointer.exists())
            self.assertEqual(runtime.job["state"], "complete")
            for candidate in runtime.job["result"]["candidates"]:
                checks = candidate["generation_checks"]
                self.assertEqual({row["id"] for row in checks}, {"validation0", "validation1"})
                self.assertTrue(all(row["passed"] and row["complete"] for row in checks))
                self.assertTrue(all(row["max_new_tokens"] == 512 for row in checks))
            self.assertTrue(runtime.model.generated)

    def test_improved_loss_but_wrong_actual_generation_keeps_old_pointer_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.run_learning(directory, answer_token=2, previous=True)
            report = runtime.job["result"]
            self.assertFalse(report["promoted"])
            self.assertTrue(all(candidate["validation_loss"] < report["baseline_validation_loss"]
                                for candidate in report["candidates"]))
            self.assertTrue(all(not row["passed"] and row["complete"]
                                for candidate in report["candidates"] for row in candidate["generation_checks"]))
            self.assertEqual(runtime.pointer.read_bytes(), runtime.old_pointer)
            self.assertEqual(runtime.branch.expert_id, "previous")
            self.assertTrue(all(torch.equal(value, runtime.branch.expert.state_dict()[key])
                                for key, value in runtime.old_weights.items()))
            self.assertEqual(runtime.state, "ready")

    def test_correct_text_without_eos_cannot_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = self.run_learning(directory, complete=False)
            self.assertFalse(runtime.job["result"]["promoted"])
            self.assertFalse(runtime.pointer.exists())
            self.assertIsNone(runtime.branch.expert)
            for candidate in runtime.job["result"]["candidates"]:
                self.assertTrue(all(row["correct"] and not row["complete"] and not row["passed"]
                                    for row in candidate["generation_checks"]))

    def test_explicit_validator_forbidden_rule_and_budget_remain_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            provenance = [{"generation_check": {"type": "contains", "expected": "expected",
                                                  "forbidden": ["answer"], "max_new_tokens": 8}}, {}]
            runtime = self.run_learning(directory, validation_provenance=provenance)
            for candidate in runtime.job["result"]["candidates"]:
                checks = candidate["generation_checks"]
                self.assertEqual(checks[0]["forbidden_found"], ["answer"])
                self.assertEqual(checks[0]["max_new_tokens"], 8)
                self.assertFalse(checks[0]["passed"])
                self.assertTrue(checks[1]["passed"])
                self.assertEqual(checks[1]["max_new_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
