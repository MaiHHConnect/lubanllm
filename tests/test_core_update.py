"""Small mathematical counterexamples; no model loading or storage."""
import unittest

import torch

from luban.core_update import (apply_core_residual, constrain_residual, constrain_update,
    guard_quantized_step, guard_retention_losses, validate_basis)

from luban.evolution_progress import extend_protected_basis


class CoreUpdateTests(unittest.TestCase):
    def setUp(self):
        self.basis = torch.tensor([[0.], [0.], [1.]])
        self.reference = torch.tensor([[.25, -.5, .75]])
        validate_basis(self.basis)

    def test_extension_adds_orthogonal_direction_and_preserves_old_span(self):
        basis = torch.tensor([[1.], [1.], [0.], [0.]]) / 2 ** .5
        generated = torch.tensor([[2., 2., 3., 0.], [4., 4., 6., 0.]])
        old, inputs = basis.clone(), generated.clone()
        extended, report = extend_protected_basis(basis, generated)
        self.assertEqual(extended.shape, (4, 2))
        self.assertEqual(report['added_rank'], 1)
        self.assertLess(report['old_span_residual'], 1e-6)
        self.assertGreater(report['extra_tolerance'], 0.)
        torch.testing.assert_close(extended.T @ extended, torch.eye(2), rtol=0, atol=2e-6)
        torch.testing.assert_close(extended @ (extended.T @ old), old, rtol=0, atol=2e-6)
        torch.testing.assert_close((generated @ extended) @ extended.T, generated, rtol=0, atol=3e-6)
        self.assertTrue(torch.equal(basis, old))
        self.assertTrue(torch.equal(generated, inputs))
        validate_basis(extended)

    def test_extension_does_not_add_exact_duplicate_or_zero_directions(self):
        for basis in (self.basis, torch.tensor([[1.], [1.], [0.]]) / 2 ** .5):
            for generated in (basis.T.repeat(3, 1), torch.zeros(2, 3)):
                extended, report = extend_protected_basis(basis, generated)
                self.assertEqual(report['added_rank'], 0)
                self.assertTrue(torch.equal(extended, basis))

    def test_extension_supports_empty_old_span_but_full_span_has_no_complement(self):
        empty = torch.empty(3, 0)
        extended, report = extend_protected_basis(empty, torch.tensor([[0., 1., 0.]]))
        self.assertEqual(extended.shape, (3, 1))
        self.assertEqual(report['added_rank'], 1)
        self.assertEqual(report['old_span_residual'], 0.)
        validate_basis(extended)
        unchanged, report = extend_protected_basis(empty, torch.zeros(1, 3))
        self.assertEqual(unchanged.shape, (3, 0))
        self.assertEqual(report['added_rank'], 0)
        with self.assertRaises(ValueError):
            validate_basis(unchanged)
        full, report = extend_protected_basis(self.basis, torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        self.assertEqual(full.shape, (3, 3))
        self.assertEqual(report['added_rank'], 2)
        with self.assertRaises(ValueError):
            validate_basis(full)
        repeated, report = extend_protected_basis(full, torch.ones(1, 3))
        self.assertEqual(report['added_rank'], 0)
        self.assertTrue(torch.equal(repeated, full))

    def test_extension_rejects_invalid_or_nonfinite_cpu_inputs(self):
        invalid = [(self.basis.double(), torch.ones(1, 3)),
            (self.basis, torch.ones(1, 3, dtype=torch.bfloat16)),
            (self.basis, torch.ones(1, 2)), (self.basis, torch.empty(0, 3)),
            (self.basis * 2, torch.ones(1, 3)), (torch.ones(3, 4), torch.ones(1, 3)),
            (torch.full((3, 1), float('nan')), torch.ones(1, 3)),
            (self.basis, torch.full((1, 3), float('inf')))]
        for basis, generated in invalid:
            with self.subTest(shape=tuple(basis.shape)), self.assertRaises(ValueError):
                extend_protected_basis(basis, generated)

    def test_unmerged_residual_preserves_protected_input_where_bf16_merge_does_not(self):
        x = torch.tensor([[1., 1., 0.]], dtype=torch.bfloat16)
        weight = torch.tensor([[.134765625, .051513671875, .09375]], dtype=torch.bfloat16)
        delta = torch.tensor([[.01, -.01, .01]], requires_grad=True)
        original = torch.nn.functional.linear(x, weight)
        residual = apply_core_residual(x, original, delta)
        merged = torch.nn.functional.linear(x, (weight.float() + delta).bfloat16())
        self.assertTrue(torch.equal(residual, original))
        self.assertFalse(torch.equal(merged, original))
        new_x = torch.tensor([[1., 0., 0.]], dtype=torch.bfloat16)
        new_original = torch.nn.functional.linear(new_x, weight)
        new_output = apply_core_residual(new_x, new_original, delta)
        self.assertFalse(torch.equal(new_output, new_original))
        new_output.float().sum().backward()
        self.assertTrue(torch.isfinite(delta.grad).all())
        self.assertEqual(delta.grad.dtype, torch.float32)

    def test_residual_zero_initialization_and_bad_inputs(self):
        x = torch.ones(2, 3, dtype=torch.bfloat16)
        output = torch.tensor([[.2], [-.7]], dtype=torch.bfloat16)
        delta = torch.zeros(1, 3)
        self.assertTrue(torch.equal(apply_core_residual(x, output, delta), output))
        for bad in (torch.zeros(2, 3), torch.zeros(1, 3, dtype=torch.bfloat16),
                    torch.full((1, 3), float("nan"))):
            with self.assertRaises(ValueError):
                apply_core_residual(x, output, bad)

    def test_loss_guard_rejects_individual_regression_even_when_mean_improves(self):
        candidate = self.reference + .01
        accepted, report = guard_retention_losses(candidate, self.reference,
            torch.tensor([1., 1.]), torch.tensor([.1, 1.01]))
        self.assertTrue(report["rejected"])
        self.assertEqual(report["increased_count"], 1)
        self.assertTrue(torch.equal(accepted, self.reference))

    def test_negative_linear_change_does_not_override_actual_curvature_regression(self):
        # L(w)=w² at w=1, delta=-3: g*delta=-6 but L(-2)>L(1).
        candidate = self.reference - 3
        self.assertLess(2 * -3, 0)
        accepted, report = guard_retention_losses(candidate, self.reference,
            torch.tensor([1.]), torch.tensor([4.]))
        self.assertTrue(report["rejected"])
        self.assertTrue(torch.equal(accepted, self.reference))

    def test_loss_guard_accepts_all_nonincreasing_losses_and_rejects_bad_evidence(self):
        candidate = self.reference + .01
        accepted, report = guard_retention_losses(candidate, self.reference,
            torch.tensor([1., 2.]), torch.tensor([1., 1.9]))
        self.assertFalse(report["rejected"])
        self.assertTrue(torch.equal(accepted, candidate))
        for baseline, observed in ((torch.tensor([]), torch.tensor([])),
                (torch.tensor([1.]), torch.tensor([float("nan")])),
                (torch.tensor([1.]), torch.tensor([-1.])),
                (torch.tensor([1.]), torch.tensor([1., 1.])),
                (torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64))):
            with self.assertRaises(ValueError):
                guard_retention_losses(candidate, self.reference, baseline, observed)

    def test_loss_guard_tolerance_is_fixed_per_entry_and_rejects_excess(self):
        candidate = self.reference + .01
        baseline = torch.tensor([1., 2.])
        accepted, report = guard_retention_losses(candidate, self.reference, baseline,
            torch.tensor([1.000001, 2.000010]),
            torch.tensor([0.000002, 0.000005]))
        self.assertTrue(report['rejected'])
        self.assertEqual(report['increased_count'], 1)
        self.assertTrue(report['tolerance_applied'])
        accepted, report = guard_retention_losses(candidate, self.reference, baseline,
            torch.tensor([1.000001, 2.000004]),
            torch.tensor([0.000002, 0.000005]))
        self.assertFalse(report['rejected'])
        self.assertTrue(torch.equal(accepted, candidate))

    def update(self, delta, gradient, current=None):
        current = self.reference if current is None else current
        return constrain_update(current + torch.tensor([delta]), current, self.reference,
                                self.basis, torch.tensor([gradient]))

    def test_joint_residual_active_set_satisfies_kkt_minimum(self):
        basis = torch.tensor([[0.], [0.], [0.], [1.]])
        proposal = torch.tensor([[3., 1., 2., 9.]])
        gradients = torch.tensor([[[1., 0., 0., 7.]], [[1., 1., 0., -8.]]])
        current = torch.zeros_like(proposal)
        candidate, report = constrain_residual(proposal, current, basis, gradients)
        self.assertFalse(report["rejected"])
        torch.testing.assert_close(candidate, torch.tensor([[0., 0., 2., 0.]]))
        torch.testing.assert_close(torch.tensor(report["multipliers"]), torch.tensor([2., 1.]))
        self.assertEqual(report["active_set"], [0, 1])
        projected = proposal - (proposal @ basis) @ basis.T
        stationarity = candidate - projected
        for multiplier, gradient in zip(report["multipliers"], gradients):
            self.assertGreaterEqual(multiplier, 0.)
            stationarity += multiplier * (gradient - (gradient @ basis) @ basis.T)
            self.assertLessEqual(abs(float((gradient * candidate).sum())), 1e-6)
        torch.testing.assert_close(stationarity, torch.zeros_like(current))

    def test_joint_opposite_duplicate_and_zero_signals_do_not_cancel(self):
        gradients = [torch.tensor([[1., 0., 0.]]), torch.tensor([[-1., 0., 0.]]),
                     torch.tensor([[1., 0., 0.]]), torch.zeros(1, 3)]
        candidate, report = constrain_residual(torch.tensor([[1., 2., 9.]]),
                                             torch.zeros(1, 3), self.basis, gradients)
        self.assertFalse(report["rejected"])
        torch.testing.assert_close(candidate, torch.tensor([[0., 2., 0.]]))
        self.assertEqual(report["zero_signals"], [False, False, False, True])

    def test_joint_near_collinear_signals_remain_individually_feasible(self):
        gradients = torch.tensor([[[1., 1e-4, 0.]], [[1., -1e-4, 0.]]])
        current = torch.zeros(1, 3)
        candidate, report = constrain_residual(torch.tensor([[1., 2., 0.]]), current,
                                             self.basis, gradients)
        self.assertFalse(report["rejected"])
        expected = torch.tensor([[1., 2., 0.]])
        expected -= ((expected * gradients[0]).sum() / gradients[0].square().sum()) * gradients[0]
        torch.testing.assert_close(candidate, expected)
        for post, tolerance in zip(report["post_interference"], report["linear_tolerance"]):
            self.assertLessEqual(post, tolerance)

    def test_joint_near_span_feasible_proposal_is_not_destroyed_by_correction(self):
        basis = torch.tensor([[1.], [1.], [0.]]) / 2 ** .5
        proposed = torch.tensor([[.1, -.1, .1]])
        for gradients in (
                [torch.tensor([[1., 1., 1e-8]])],
                [torch.tensor([[1., 1., 1e-8]]), torch.tensor([[2., 2., 2e-8]]),
                 torch.tensor([[-1., -1., -1e-8]]), torch.tensor([[.5, .5, -2e-8]])]):
            candidate, report = constrain_residual(proposed, torch.zeros_like(proposed), basis, gradients)
            self.assertFalse(report['rejected'])
            self.assertTrue(report['already_feasible'])
            self.assertTrue(torch.equal(candidate, proposed))
            self.assertEqual(report['correction_norm'], 0.)
            for actual, bound in zip(report['post_interference'], report['linear_tolerance']):
                self.assertLessEqual(actual, bound)
            self.assertLessEqual(float((candidate @ basis).norm()), report['subspace_tolerance'])

    def test_joint_true_near_span_conflict_still_requires_stable_correction(self):
        basis = torch.tensor([[1.], [1.], [0.]]) / 2 ** .5
        proposed = torch.tensor([[.1, -.1, .1]])
        gradient = torch.tensor([[1., 1., 1e-3]])
        candidate, report = constrain_residual(proposed, torch.zeros_like(proposed), basis, [gradient])
        original_tolerance = 64 * torch.finfo(torch.float32).eps * float(gradient.double().norm()) * float(proposed.double().norm())
        self.assertGreater(report['pre_interference'][0], original_tolerance)
        self.assertFalse(report['already_feasible'])
        self.assertFalse(report['rejected'])
        self.assertFalse(torch.equal(candidate, proposed))
        self.assertLessEqual(report['post_interference'][0], report['linear_tolerance'][0])
        self.assertLessEqual(float((candidate @ basis).norm()), report['subspace_tolerance'])

    def test_joint_fp32_rounding_failure_rejects_without_enlarging_budget(self):
        current = torch.tensor([[14.65451717376709, -3.228864908218384, 0.]])
        proposed = torch.tensor([[14.654515266418457, -3.2288658618927, 0.]])
        gradient = torch.tensor([[-1.492430329322815, 1.7311341762542725, 0.]])
        candidate, report = constrain_residual(proposed, current, self.basis, [gradient])
        self.assertTrue(report["rejected"])
        self.assertEqual(report["rejection_reason"], "fp32_linear_preservation")
        self.assertTrue(torch.equal(candidate, current))
        self.assertGreater(report["candidate_post_interference"][0], report["candidate_linear_tolerance"][0])
        self.assertEqual(report["post_interference"], [0.])

    def test_joint_cumulative_subspace_and_inputs_are_preserved(self):
        basis = torch.tensor([[1.], [1.], [0.]]) / 2 ** .5
        current = torch.tensor([[.125, -.125, .25]])
        gradients = [torch.tensor([[1., 0., .1]]), torch.tensor([[0., 1., -.1]])]
        proposed = current + torch.tensor([[.01, -.03, .1]])
        before = [value.clone() for value in [current, proposed, basis] + gradients]
        candidate, report = constrain_residual(proposed, current, basis, gradients)
        self.assertFalse(report["rejected"])
        self.assertLessEqual(float((candidate @ basis).norm()), report["subspace_tolerance"])
        for old, new in zip(before, [current, proposed, basis] + gradients):
            self.assertTrue(torch.equal(old, new))
        for value, tolerance in zip(report["post_interference"], report["linear_tolerance"]):
            self.assertLessEqual(value, tolerance)

    def test_joint_invalid_and_nonfinite_signals_raise(self):
        current = torch.zeros(1, 3)
        for signals in ([], [current] * 5, current, [current.double()],
                        [torch.ones(2, 3)], [torch.full_like(current, float("nan"))]):
            with self.subTest(signals=signals), self.assertRaises(ValueError):
                constrain_residual(current, current, self.basis, signals)
        with self.assertRaisesRegex(ValueError, "outside"):
            constrain_residual(current, torch.ones_like(current), self.basis, [current])
        with self.assertRaisesRegex(ValueError, "finite"):
            constrain_residual(torch.full_like(current, float("inf")), current, self.basis, [current])

    def test_joint_infeasible_projection_keeps_current_drift_without_mutation(self):
        current = torch.tensor([[0., 0., -1e-6]])
        candidate, report = constrain_residual(current + torch.tensor([[1., 0., 0.]]), current,
                                             self.basis, [torch.tensor([[0., 0., 1.]])])
        self.assertTrue(report["rejected"])
        self.assertEqual(report["rejection_reason"], "no_feasible_preservation_direction")
        self.assertTrue(torch.equal(candidate, current))

    def test_conflict_gets_minimum_halfspace_correction(self):
        candidate, report = self.update([2., 3., 9.], [1., 1., 7.])
        torch.testing.assert_close(candidate - self.reference, torch.tensor([[-.5, .5, 0.]]))
        self.assertTrue(report["corrected"])
        self.assertEqual(report["pre_interference"], 5.)
        self.assertLessEqual(report["post_interference"], report["linear_tolerance"])
        self.assertEqual(report["cumulative_subspace_residual"], 0.)

    def test_nonconflict_preserves_original_cumulative_projection(self):
        current = self.reference + torch.tensor([[.125, -.25, 0.]])
        proposal = current + torch.tensor([[-2., 1., 8.]])
        delta = proposal - self.reference
        expected = self.reference + delta - (delta @ self.basis) @ self.basis.T
        candidate, report = constrain_update(proposal, current, self.reference,
                                             self.basis, torch.tensor([[1., 0., 8.]]))
        self.assertTrue(torch.equal(candidate, expected))
        self.assertFalse(report["corrected"])
        self.assertEqual(report["correction_norm"], 0.)

    def test_actual_adam_momentum_can_reverse_raw_gradient_decision(self):
        master = torch.nn.Parameter(self.reference.clone())
        optimizer = torch.optim.Adam([master], lr=.01, eps=1e-6)
        optimizer.state[master] = {"step": torch.tensor(1.),
            "exp_avg": torch.tensor([[-1., 0., 0.]]), "exp_avg_sq": torch.ones_like(master)}
        master.grad = torch.tensor([[1., 0., 0.]])
        self.assertLess(float((-master.grad * torch.tensor([[1., 0., 0.]])).sum()), 0.)
        optimizer.step()
        self.assertGreater(float(master.detach()[0, 0] - self.reference[0, 0]), 0.)
        candidate, report = constrain_update(master, self.reference, self.reference,
                                             self.basis, torch.tensor([[1., 0., 0.]]))
        self.assertGreater(report["pre_interference"], 0.)
        self.assertTrue(torch.equal(candidate, self.reference))

    def test_zero_signal_is_reported_without_claiming_protection(self):
        candidate, report = self.update([1., 2., 3.], [0., 0., 0.])
        self.assertTrue(report["zero_signal"])
        self.assertFalse(report["corrected"])
        torch.testing.assert_close(candidate - self.reference, torch.tensor([[1., 2., 0.]]))

    def test_basis_only_signal_and_small_current_drift_can_be_infeasible(self):
        current = self.reference + torch.tensor([[0., 0., -1e-6]])
        candidate, report = self.update([1., 0., 0.], [0., 0., 1.], current=current)
        self.assertTrue(torch.equal(candidate, current))
        self.assertTrue(report["rejected"])
        self.assertEqual(report["rejection_reason"], "no_feasible_preservation_direction")
        self.assertFalse(report["corrected"])
        self.assertEqual(report["step_norm"], 0.)
        self.assertEqual(report["post_interference"], 0.)

    def test_fp32_addition_rounding_conflict_rejects_without_relaxing_tolerance(self):
        reference = torch.tensor([[14.65451717376709, -3.228864908218384, -14.287534713745117]])
        proposal = torch.tensor([[14.654515266418457, -3.2288658618927, -14.287535667419434]])
        gradient = torch.tensor([[-1.492430329322815, 1.7311341762542725, -.5033498406410217]])
        candidate, report = constrain_update(proposal, reference, reference, self.basis, gradient)
        self.assertTrue(report["rejected"])
        self.assertEqual(report["rejection_reason"], "fp32_linear_preservation")
        self.assertGreater(report["candidate_post_interference"], report["candidate_linear_tolerance"])
        self.assertTrue(torch.equal(candidate, reference))
        self.assertEqual(report["post_interference"], 0.)
        self.assertEqual(report["step_norm"], 0.)

    def test_fp32_affine_rounding_conflict_also_keeps_current(self):
        reference = torch.tensor([[1000., 0., 0.]])
        basis = torch.tensor([[1.], [2.], [0.]]) / 5 ** .5
        validate_basis(basis)
        candidate, report = constrain_update(reference + torch.tensor([[1., 1., 0.]]),
                                             reference, reference, basis, torch.zeros_like(reference))
        self.assertTrue(report["rejected"])
        self.assertEqual(report["rejection_reason"], "fp32_affine_subspace")
        self.assertGreater(report["candidate_subspace_residual"], report["candidate_subspace_tolerance"])
        self.assertTrue(torch.equal(candidate, reference))
        self.assertEqual(report["cumulative_subspace_residual"], 0.)

    def test_averaged_opposite_signals_cannot_protect_each_individual(self):
        first = torch.tensor([[1., 0., 0.]])
        second = -first
        candidate, report = constrain_update(self.reference + torch.tensor([[1., 0., 0.]]),
            self.reference, self.reference, self.basis, (first + second) / 2)
        self.assertTrue(report["zero_signal"])
        self.assertFalse(report["rejected"])
        self.assertGreater(float(((candidate - self.reference) * first).sum()), 0.)

    def test_signal_inside_protected_subspace_has_no_complement_information(self):
        candidate, report = self.update([1., 0., 0.], [0., 0., 1.])
        self.assertFalse(report["zero_signal"])
        self.assertEqual(report["projected_gradient_norm"], 0.)
        self.assertFalse(report["rejected"])
        torch.testing.assert_close(candidate - self.reference, torch.tensor([[1., 0., 0.]]))

    def test_current_outside_affine_space_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "current is outside"):
            self.update([1., 0., 0.], [1., 0., 0.],
                        current=self.reference + torch.tensor([[0., 0., .01]]))

    def test_nonaxis_subspace_and_multiple_steps(self):
        basis = torch.tensor([[1.], [1.], [0.]]) / 2 ** .5
        validate_basis(basis)
        current = self.reference.clone()
        for index in range(1, 9):
            proposal = current + torch.tensor([[.01 * index, -.02, .03]])
            gradient = torch.tensor([[.2, -.3, float(index % 2)]])
            current, report = constrain_update(proposal, current, self.reference, basis, gradient)
            self.assertLessEqual(report["post_interference"], report["linear_tolerance"])
            self.assertLessEqual(report["cumulative_subspace_residual"], report["subspace_tolerance"])
            self.assertLess(float(((current - self.reference) @ basis).norm()), 1e-6)

    def test_bf16_cast_can_break_fp32_subspace_constraint(self):
        reference = torch.tensor([[.2, -.7, .4]], dtype=torch.bfloat16).float()
        basis = torch.tensor([[1.], [1.], [0.]]) / 2 ** .5
        validate_basis(basis)
        candidate, report = constrain_update(reference + torch.tensor([[.01, -.01, .01]]),
                                             reference, reference, basis, torch.zeros_like(reference))
        self.assertLess(report["cumulative_subspace_residual"], 1e-6)
        quantized = candidate.to(torch.bfloat16).float()
        self.assertGreater(float(((quantized - reference) @ basis).norm()), 1e-4)

    def test_bf16_guard_rejects_rounding_conflict_and_keeps_current_master(self):
        reference = torch.tensor([[.2, -.7, .4]], dtype=torch.bfloat16).float()
        candidate = reference + torch.tensor([[.01, -.01, .01]])
        gradient = torch.tensor([[-1., -1., 0.]])
        self.assertLess(abs(float(((candidate - reference) * gradient).sum())), 1e-7)
        accepted, report = guard_quantized_step(candidate, reference, gradient)
        self.assertTrue(report["quantized_rejected"])
        self.assertGreater(report["quantized_interference"], report["quantized_linear_tolerance"])
        self.assertTrue(torch.equal(accepted, reference))
        self.assertEqual(report["accepted_quantized_interference"], 0.)

    def test_quantized_guard_accepts_nonconflict_and_zero_signal(self):
        candidate = self.reference + torch.tensor([[-.02, .01, 0.]])
        for gradient in (torch.tensor([[1., 0., 0.]]), torch.zeros_like(candidate)):
            accepted, report = guard_quantized_step(candidate, self.reference, gradient)
            self.assertFalse(report["quantized_rejected"])
            self.assertTrue(torch.equal(accepted, candidate))
        self.assertTrue(report["zero_signal"])

    def test_quantized_guard_rejects_invalid_inputs(self):
        for candidate, gradient, dtype in (
                (torch.full_like(self.reference, float("nan")), self.reference, torch.bfloat16),
                (self.reference.double(), self.reference, torch.bfloat16),
                (self.reference, torch.ones(2, 3), torch.bfloat16),
                (self.reference, self.reference, torch.float16)):
            with self.assertRaises(ValueError):
                guard_quantized_step(candidate, self.reference, gradient, dtype=dtype)

    def test_inputs_gradients_and_rng_are_not_mutated(self):
        proposed = (self.reference + 1).requires_grad_()
        proposed.grad = torch.full_like(proposed, 3.)
        gradient = torch.tensor([[1., 0., 0.]], requires_grad=True)
        values = [proposed, self.reference, self.basis, gradient]
        before = [value.detach().clone() for value in values]
        rng = torch.get_rng_state().clone()
        candidate, _ = constrain_update(proposed, self.reference, self.reference, self.basis, gradient)
        self.assertFalse(candidate.requires_grad)
        for value, saved in zip(values, before):
            self.assertTrue(torch.equal(value, saved))
        self.assertTrue(torch.equal(proposed.grad, torch.full_like(proposed, 3.)))
        self.assertIsNone(gradient.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_schema_and_nonfinite_are_rejected(self):
        for index in range(5):
            arguments = [self.reference.clone(), self.reference.clone(), self.reference.clone(),
                         self.basis.clone(), torch.ones_like(self.reference)]
            arguments[index].reshape(-1)[0] = float("nan")
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "finite"):
                constrain_update(*arguments)
        for bad in (torch.ones(1, 2), torch.ones(1, 3, dtype=torch.float64),
                    torch.ones(1, 3, dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):
                constrain_update(bad, self.reference, self.reference, self.basis,
                                 torch.ones_like(self.reference))
        with self.assertRaises(ValueError):
            constrain_update(self.reference, self.reference, self.reference,
                             torch.ones(2, 1), torch.ones_like(self.reference))

    def test_round_basis_validation_rejects_nonorthogonal_and_full_rank(self):
        for bad in (torch.ones(3, 2), torch.eye(3), self.basis * 2,
                    self.basis.to(torch.float64), torch.full((3, 1), float("inf"))):
            with self.subTest(basis=bad), self.assertRaises(ValueError):
                validate_basis(bad)

    def test_finite_inputs_with_overflowing_intermediates_are_rejected(self):
        maximum = torch.finfo(torch.float32).max
        current = torch.tensor([[maximum, 0., 0.]])
        reference = -current
        with self.assertRaisesRegex(ValueError, "finite"):
            constrain_update(current, current, reference, self.basis, torch.ones_like(current))
        with self.assertRaisesRegex(ValueError, "finite"):
            guard_quantized_step(current, -current, torch.ones_like(current))


if __name__ == "__main__":
    unittest.main()
