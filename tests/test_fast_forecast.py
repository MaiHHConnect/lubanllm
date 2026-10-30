import unittest

import torch
from torch import nn

from verify_fast_forecast import AssistedHead, Forecast, budget_outcomes, project_states


class FastForecastTests(unittest.TestCase):
    def test_budget_cannot_turn_unfinished_correct_text_into_success(self):
        actual = {"raw_completion_token_ids": list(range(7)), "passed": True, "complete": True,
                  "branch_enabled": False, "do_sample": False, "thinking": "off", "max_new_tokens": 512}
        self.assertEqual(budget_outcomes(actual), [2, 0, 0])
        actual.update(complete=False, passed=False)
        self.assertEqual(budget_outcomes(actual), [2, 2, 2])
        actual["do_sample"] = True
        with self.assertRaises(ValueError):
            budget_outcomes(actual)

    def test_normalization_never_reads_heldout(self):
        torch.manual_seed(5)
        prompts = {s: torch.randn(3, 2048) for s in ("train", "test")}
        futures = {s: torch.randn(3, 3, 2048) for s in prompts}
        first = project_states(prompts, futures)
        prompts["test"].fill_(1e6); futures["test"].fill_(-1e6)
        second = project_states(prompts, futures)
        for key in first[2]:
            self.assertTrue(torch.equal(first[2][key], second[2][key]))
        self.assertTrue(torch.equal(first[0]["train"], second[0]["train"]))
        self.assertTrue(torch.equal(first[1]["train"], second[1]["train"]))

    def test_rollout_uses_predictions_and_same_parameters(self):
        flat, chain = Forecast(False), Forecast(True)
        for p in flat.parameters():
            nn.init.constant_(p, 0.05)
        chain.load_state_dict(flat.state_dict())
        x = torch.zeros(2, 32)
        a, b = flat(x)[0], chain(x)[0]
        self.assertTrue(torch.equal(a[:, 0], b[:, 0]))
        self.assertFalse(torch.equal(a[:, 1], b[:, 1]))
        self.assertEqual(sum(p.numel() for p in flat.parameters()), sum(p.numel() for p in chain.parameters()))

    def test_zero_branch_preserves_initial_decision_and_learns(self):
        original = nn.Linear(2048, 3)
        assisted = AssistedHead(original.state_dict())
        x = torch.randn(2, 2048)
        self.assertTrue(torch.equal(original(x), assisted(x)))
        assisted(x).sum().backward()
        self.assertGreater(float(assisted.expert.down.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
