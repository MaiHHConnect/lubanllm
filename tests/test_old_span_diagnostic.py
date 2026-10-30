"""Run only after the standalone native CPU helper is built; never loads a model."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from luban.evolution_progress import extend_protected_basis


def gptq_scalar_reference(projected, states):
    # Independent full-row scalar propagation, without the native 128-column blocks.
    hessian = (states.T @ states / len(states)).double()
    hessian.diagonal().add_(.01 * float(hessian.diagonal().mean()))
    upper = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True).float().triu_()
    work = projected.clone()
    master = torch.empty_like(work)
    quantized = torch.empty_like(work, dtype=torch.bfloat16)
    for column in range(work.shape[1]):
        value = work[:, column].clone()
        master[:, column] = value
        quantized[:, column] = value.to(torch.bfloat16)
        error = (value - quantized[:, column].float()) / upper[column, column]
        work[:, column:] -= error[:, None] * upper[column, column:][None, :]
    return quantized, master


class OldSpanDiagnosticTests(unittest.TestCase):
    def test_unit_hessian_native_reduces_to_bf16_rne_including_ties(self):
        result = subprocess.run([str(Path("build/old-span-diagnostic/project-old-span").resolve()), "--test-unit-hessian"], capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)["unit_hessian_rne"])

    def _gptq_fixture(self, width=160):
        generator = torch.Generator().manual_seed(142)
        reference = torch.randn((3, width), generator=generator).to(torch.bfloat16).float()
        master = reference + .04 * torch.randn((3, width), generator=generator)
        basis = torch.linalg.qr(torch.randn((width, 1), generator=generator)).Q.contiguous()
        # 8 rows span at most three dimensions: actual rank deficiency requires damping.
        states = torch.randn((8, 3), generator=generator) @ torch.randn((3, width), generator=generator)
        return reference, master, basis, states

    def _run_gptq(self, values, root, mode="gptq-bf16"):
        reference, master, basis, states = values
        paths = []
        for name, value in (("reference", reference), ("master", master), ("candidate", master.to(torch.bfloat16)), ("basis", basis), ("states", states)):
            path = root / (name + ".raw")
            path.write_bytes(value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
            paths.append(str(path))
        return subprocess.run([str(Path("build/old-span-diagnostic/project-old-span").resolve()), *paths,
            str(master.shape[0]), str(master.shape[1]), str(basis.shape[1]), str(len(states)), str(root / "output"), mode, "3"], text=True, capture_output=True)

    def test_gptq_rank_deficient_cross_block_matches_scalar_reference(self):
        torch.set_num_threads(1)
        reference, master, basis, states = self._gptq_fixture()
        _, _, vh = torch.linalg.svd(basis.T.contiguous(), full_matrices=False)
        expanded, _ = extend_protected_basis(vh.T.contiguous(), states)
        delta = master - reference
        projected = reference + delta - (delta @ expanded) @ expanded.T
        expected, expected_master = gptq_scalar_reference(projected, states)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = self._run_gptq((reference, master, basis, states), root)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            actual = torch.frombuffer(bytearray((root / "output/candidate.raw").read_bytes()), dtype=torch.bfloat16).reshape_as(master)
            actual_master = torch.frombuffer(bytearray((root / "output/master.raw").read_bytes()), dtype=torch.float32).reshape_as(master)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.allclose(actual_master, expected_master, atol=3e-6, rtol=3e-6))
            self.assertTrue(torch.equal(actual_master.to(torch.bfloat16), actual))
            self.assertEqual(report["schema"], "known-old-q4-gptq-bf16-v1")
            self.assertEqual((report["selected_iteration"], report["block_size"], report["damping"], report["column_order"]), (1, 128, .01, "natural"))
            self.assertEqual((report["old_positions"], report["q4_train_positions"]), (3, 5))
            rows = report["rounds"]
            for index, value in enumerate((projected.to(torch.bfloat16), actual)):
                drift = (value.float() - reference) @ states.T
                self.assertEqual(rows[index]["iteration"], index)
                self.assertAlmostEqual(rows[index]["reconstruction_frobenius"], float(drift.norm()), places=5)
                self.assertAlmostEqual(rows[index]["known_old_frobenius"], float(drift[:, :3].norm()), places=5)
                self.assertAlmostEqual(rows[index]["q4_train_frobenius"], float(drift[:, 3:].norm()), places=5)
                self.assertAlmostEqual(rows[index]["projected_reconstruction"]["known_old_output_frobenius"], float(((value.float()-projected) @ states.T).norm()), places=5)
            improved = rows[1]["reconstruction_frobenius"] < rows[0]["reconstruction_frobenius"] and all(rows[1][key] <= rows[0][key] for key in ("known_old_frobenius", "q4_train_frobenius"))
            self.assertEqual(report["compensation_improved"], improved)
            self.assertEqual(report["original_span_worsened"], rows[1]["drift"]["original_replay_span_frobenius"] > rows[0]["drift"]["original_replay_span_frobenius"])
            self.assertFalse(report["new_validation_used"])
            self.assertFalse(report["product_admitted"])
            self.assertEqual(report["optimizer_steps"], 0)

    def test_one_refinement_projects_quantized_candidate_then_reuses_gptq(self):
        torch.set_num_threads(1)
        reference, master, basis, states = self._gptq_fixture()
        _, _, vh = torch.linalg.svd(basis.T.contiguous(), full_matrices=False)
        expanded, _ = extend_protected_basis(vh.T.contiguous(), states)
        delta = master - reference
        projected = reference + delta - (delta @ expanded) @ expanded.T
        first, _ = gptq_scalar_reference(projected, states)
        corrected = first.float() - ((first.float() - reference) @ expanded) @ expanded.T
        expected, expected_master = gptq_scalar_reference(corrected, states)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = self._run_gptq((reference, master, basis, states), root, "gptq-bf16-refine")
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(process.stdout)
            actual = torch.frombuffer(bytearray((root / "output/candidate.raw").read_bytes()), dtype=torch.bfloat16).reshape_as(master)
            actual_master = torch.frombuffer(bytearray((root / "output/master.raw").read_bytes()), dtype=torch.float32).reshape_as(master)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.allclose(actual_master, expected_master, atol=4e-6, rtol=4e-6))
            self.assertTrue(torch.equal(actual_master.to(torch.bfloat16), actual))
            self.assertEqual(report["schema"], "known-old-q4-gptq-bf16-refine-v1")
            self.assertEqual((report["selected_iteration"], report["fixed_compensation_passes"]), (2, 2))
            self.assertEqual([v["iteration"] for v in report["rounds"]], [0, 1, 2])
            for index, value in enumerate((projected.to(torch.bfloat16), first, expected)):
                drift = (value.float()-reference) @ states.T
                self.assertAlmostEqual(report["rounds"][index]["reconstruction_frobenius"], float(drift.norm()), places=5)
                self.assertAlmostEqual(report["rounds"][index]["known_old_frobenius"], float(drift[:, :3].norm()), places=5)
                self.assertAlmostEqual(report["rounds"][index]["q4_train_frobenius"], float(drift[:, 3:].norm()), places=5)
            prior, final = report["rounds"][1:]
            self.assertEqual(report["compensation_improved"], final["reconstruction_frobenius"] < prior["reconstruction_frobenius"] and all(final[k] <= prior[k] for k in ("known_old_frobenius", "q4_train_frobenius")))
            self.assertEqual(report["original_span_comparison_iteration"], 1)
            self.assertEqual(report["original_span_worsened"], final["drift"]["original_replay_span_frobenius"] > prior["drift"]["original_replay_span_frobenius"])
            self.assertFalse(report["new_validation_used"])
            self.assertFalse(report["product_admitted"])

    def test_gptq_rejects_nonfinite_and_no_activation_energy(self):
        for invalid in (float("nan"), float("inf"), 0.):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                reference, master, basis, states = self._gptq_fixture(16)
                states.fill_(invalid)
                process = self._run_gptq((reference, master, basis, states), Path(directory))
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse((Path(directory) / "output/candidate.raw").exists())
                self.assertTrue("Nonfinite" in process.stderr or "no activation energy" in process.stderr, process.stderr)

    def test_fixed_two_rounding_corrections_select_only_old_and_q4_errors(self):
        binary = Path("build/old-span-diagnostic/project-old-span").resolve()
        if not binary.is_file():
            self.skipTest("Standalone CPU helper has not been built")
        torch.manual_seed(5)
        reference = torch.randn(2, 6).to(torch.bfloat16).float()
        master = reference + torch.randn(2, 6) * .03
        basis = torch.linalg.qr(torch.randn(6, 1)).Q.contiguous()
        states = torch.randn(3, 6)
        # Reuse the same existing native primitive's initial SVD, then Python's extension.
        _, _, vh = torch.linalg.svd(basis.T.contiguous(), full_matrices=False)
        expanded, _ = extend_protected_basis(vh.T.contiguous(), states)
        delta = master - reference
        pre_round = reference + delta - (delta @ expanded) @ expanded.T
        candidates, scores, groups = [], [], []
        for iteration in range(3):
            quantized = pre_round.to(torch.bfloat16)
            deployed = quantized.float() - reference
            drift = deployed @ states.T
            candidates.append(quantized.clone()); scores.append(float(drift.norm()))
            groups.append((float(drift[:, :2].norm()), float(drift[:, 2:].norm())))
            if iteration < 2:
                pre_round -= (deployed @ expanded) @ expanded.T
        selected = min(range(3), key=scores.__getitem__)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); paths = []
            for name, value in (("reference", reference), ("master", master), ("candidate", master.to(torch.bfloat16)), ("basis", basis), ("states", states)):
                path = root / (name + ".raw"); path.write_bytes(value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()); paths.append(str(path))
            result = subprocess.run([str(binary), *paths, "2", "6", "1", "3", str(root / "selected"), "feedback2", "2"], text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            actual = torch.frombuffer(bytearray((root / "selected/candidate.raw").read_bytes()), dtype=torch.bfloat16).reshape(2, 6)
            self.assertTrue(torch.equal(actual, candidates[selected]))
            self.assertEqual(report["selected_iteration"], selected)
            self.assertEqual([row["iteration"] for row in report["rounds"]], [0, 1, 2])
            for index, row in enumerate(report["rounds"]):
                self.assertAlmostEqual(row["reconstruction_frobenius"], scores[index], places=6)
                self.assertAlmostEqual(row["known_old_frobenius"], groups[index][0], places=6)
                self.assertAlmostEqual(row["q4_train_frobenius"], groups[index][1], places=6)
            improved = scores[selected] < scores[0] and all(a <= b for a, b in zip(groups[selected], groups[0]))
            self.assertEqual(report["feedback_improved"], improved)
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["fixed_feedback_steps"], 2)
            self.assertFalse(report["new_validation_used"])

    def test_native_projection_matches_existing_python_span_and_single_cast(self):
        binary = Path("build/old-span-diagnostic/project-old-span").resolve()
        if not binary.is_file():
            self.skipTest("Standalone CPU helper has not been built")
        reference = torch.tensor([[.2, -.7, .4, .1], [.8, .4, -.3, .2]], dtype=torch.bfloat16).float()
        delta = torch.tensor([[0., .02, -.03, .04], [0., -.03, .04, .02]])
        master = reference + delta
        basis = torch.tensor([[1.], [0.], [0.], [0.]])
        old = torch.tensor([[1., 1., 0., 0.], [0., 2., 0., 0.]])
        expanded, _ = extend_protected_basis(basis, old)
        expected = (reference + delta - (delta @ expanded) @ expanded.T).to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            names = []
            for name, value in (("reference", reference), ("master", master), ("candidate", master.to(torch.bfloat16)), ("basis", basis), ("old", old)):
                path = root / (name + ".raw")
                path.write_bytes(value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()); names.append(str(path))
            result = subprocess.run([str(binary), *names, "2", "4", "1", "2", str(root / "projected")], text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            actual = torch.frombuffer(bytearray((root / "projected/candidate.raw").read_bytes()), dtype=torch.bfloat16).reshape(2, 4)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(report["expanded_rank"], 2)
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertLessEqual(report["original_span_max_abs"], 2e-5)
            self.assertLess(report["projected_master"]["known_old_output_frobenius"], report["master"]["known_old_output_frobenius"])
            self.assertFalse(report["product_admitted"])


if __name__ == "__main__":
    unittest.main()
