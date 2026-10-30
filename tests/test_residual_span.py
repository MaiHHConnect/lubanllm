"""Small counterexamples for the shared-proposal span diagnostic."""
import unittest

import torch

from probe_residual_span import proposal_for
from luban.core_update import apply_core_residual
from luban.evolution_progress import extend_protected_basis


class ResidualSpanTests(unittest.TestCase):
    def test_same_proposal_factorial_respects_old_new_span_and_independent_signal(self):
        old = torch.eye(4)[:, :1]
        expanded, _ = extend_protected_basis(old, torch.tensor([[0., 1., 0., 0.]]))
        proposed = torch.tensor([[2., 3., 4., 5.]])
        original = proposed.clone()
        signal = torch.tensor([[0., 0., 1., 0.]])
        for basis, joint, expected in (
                (old, False, [0., 3., 4., 5.]), (expanded, False, [0., 0., 4., 5.]),
                (old, True, [0., 3., 0., 5.]), (expanded, True, [0., 0., 0., 5.])):
            with self.subTest(joint=joint, rank=basis.shape[1]):
                candidate, _ = proposal_for(proposed, basis, [signal], joint)
                torch.testing.assert_close(candidate, torch.tensor([expected]), rtol=0, atol=1e-6)
                self.assertTrue(torch.equal(proposed, original))
                self.assertEqual(float((candidate @ basis).norm()), 0.)
        x = torch.tensor([[0., 1., 0., 0.]], dtype=torch.bfloat16)
        output = torch.ones(1, 1, dtype=torch.bfloat16)
        candidate, _ = proposal_for(proposed, expanded, [signal], False)
        self.assertTrue(torch.equal(apply_core_residual(x, output, candidate), output))
        old_candidate, _ = proposal_for(proposed, old, [signal], False)
        self.assertFalse(torch.equal(apply_core_residual(x, output, old_candidate), output))

    def test_exhausted_or_nonorthogonal_span_is_rejected_not_shrunk_for_learning(self):
        proposed = torch.ones(1, 3)
        for basis in (torch.eye(3), torch.ones(3, 2)):
            for joint in (False, True):
                with self.assertRaises(ValueError):
                    proposal_for(proposed, basis, [proposed], joint)


if __name__ == '__main__':
    unittest.main()
