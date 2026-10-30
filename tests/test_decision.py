import unittest
import torch
from luban.decision import FastDecisionHead


class DecisionTests(unittest.TestCase):
    def test_masked_pooling_and_training_gradient(self):
        torch.manual_seed(42)
        head = FastDecisionHead(8)
        x = torch.randn(2, 3, 8)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        baseline = head(x, mask)
        changed = x.clone()
        changed[mask == 0] = 999
        torch.testing.assert_close(baseline, head(changed, mask), rtol=0, atol=0)
        torch.nn.functional.cross_entropy(baseline, torch.tensor([0, 2])).backward()
        self.assertGreater(head.projection.weight.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            head(x, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main()
