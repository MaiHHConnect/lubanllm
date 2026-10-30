"""Pure stdlib ABI fixtures; no Torch, model, optimizer or C++ execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_source", ROOT / "scripts/compare_native_training.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def raw(root, name, shape, dtype="f32", value=0):
    widths = {"f32": ("f", 4), "bf16": ("H", 2), "i64": ("q", 8), "u8": ("B", 1)}
    code, _ = widths[dtype]
    data = struct.pack("<" + code, value) * compare.math.prod(shape)
    file = root / (name + ".raw")
    file.write_bytes(data)
    return {"name": name, "shape": shape, "dtype": dtype, "file": file.name,
            "bytes": len(data), "sha256": compare.sha(file)}


def make_fixture(root):
    root.mkdir()
    shapes = {"candidate": [4, 8], "master": [4, 8], "reference": [4, 8], "basis": [8, 1],
              "exp_avg": [4, 8], "exp_avg_sq": [4, 8], "adam_step": [], "norm": [4], "head": [16, 4]}
    tensors = {key: raw(root, key, shape) for key, shape in shapes.items()}
    rows = {}
    for key, role in (("r000", "new"), ("r001", "replay")):
        rows[key] = {"role": role, "sample_sha256": "b" * 64, "length": 4, "gold": 2}
        for suffix, shape, dtype in (("x", [1, 4, 8], "f32"), ("residual", [1, 4, 4], "f32"),
                                     ("ids", [1, 4], "i64"), ("labels", [1, 4], "i64")):
            tensors[key + "_" + suffix] = raw(root, key + "_" + suffix, shape, dtype)
    config = {"identity": "a" * 64, "device": "cpu", "steps": "2", "norm_eps": "1e-6", "train_eos": "3", "checkpoint_steps": "1,2"}
    steps = [{"ordinal": 1, "role": "new", "ce": "r000", "replay": "r001"},
             {"ordinal": 2, "role": "replay", "ce": "r001", "replay": "r001"}]
    (root / "manifest.tsv").write_text(compare.manifest_text(config, tensors, rows, steps))
    return compare.fixture(root, compare.sha(root / "manifest.tsv"))


def fake_output(root, m, worker="Python-minimal-cached-tail-torch.optim.Adam"):
    """Fabricated byte/schema receipt, explicitly NOT numeric/training evidence."""
    root.mkdir()
    specs = {}
    for index in (1, 2):
        for key in compare.PROBES:
            shape = [] if key == "adam_step" else [2, 16] if key == "ce_logits" or key.endswith("log_probs") else [4, 8]
            specs[f"checkpoint{index:06d}_" + key] = raw(root, f"checkpoint{index:06d}_" + key, shape,
                                                          value=index if key == "adam_step" else 0)
    for key in compare.FINAL:
        original = m["tensors"].get(key)
        shape = original["shape"] if original else [4, 8]
        specs["final_" + key] = raw(root, "final_" + key, shape, value=2 if key == "adam_step" else 0)
    text = [compare.OUTPUT_ABI]
    for v in specs.values():
        text.append("\t".join(("tensor", v["name"], v["dtype"], ",".join(map(str, v["shape"])) or "-",
                               v["file"], str(v["bytes"]), v["sha256"])))
    (root / "tensors.tsv").write_text("\n".join(text) + "\n")
    report = {"abi": compare.ABI, "passed_execution": True, "worker": worker, "manifest_sha256": m["manifest_sha256"],
              "device": "cpu", "steps": 2, "new_ce_exposures": 1, "replay_ce_exposures": 1, "replay_kl_exposures": 2,
              "teacher_calls": 0, "generation_calls": 0, "publication": False, "cpu_threads": 1, "cpu_interop_threads": 1,
              "all_input_tensors_preserved": True, "input_tensor_count": len(m["tensors"]),
              "entry_adam_step": 0, "final_adam_step": 2}
    (root / "result.json").write_text(json.dumps(report))
    lines = ["\t".join(compare.TRACE), "1\tnew\tr000\tr001\t1\t-0.00000001\t1\t0.99999995\t1",
             "2\treplay\tr001\tr001\t1\t0\t1\t1\t1"]
    (root / "trace.tsv").write_text("\n".join(lines) + "\n")
    return specs


class NativeCompareStdlibTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_default_s_mode_does_not_import_torch_or_scan_weights(self):
        code = "import runpy,sys;sys.argv=['compare'];runpy.run_path(" + repr(str(ROOT / "scripts/compare_native_training.py")) + ",run_name='__main__');assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value["model_go"])
        self.assertFalse(value["actual_model_ready"])
        self.assertEqual(value["cpu_tiny"]["final_adam_step"], 4)

    def test_manifest_roundtrip_exact_full_sequence_and_schedule(self):
        m = make_fixture(self.root / "input")
        self.assertEqual(len(m["steps"]), 2)
        self.assertEqual(m["rows"]["r000"]["gold"], 2)
        self.assertEqual(m["tensors"]["r000_x"]["shape"], [1, 4, 8])
        self.assertEqual(m["tensors"]["adam_step"]["shape"], [])

    def test_hash_shape_path_duplicate_and_order_fail_closed(self):
        for change in ("hash", "shape", "path", "duplicate", "order", "unknown"):
            with self.subTest(change=change):
                m = make_fixture(self.root / change)
                path = Path(m["root"]) / "manifest.tsv"
                text = path.read_text()
                if change == "hash":
                    (Path(m["root"]) / "candidate.raw").write_bytes(b"corrupt")
                elif change == "shape":
                    text = text.replace("tensor\tmaster\tf32\t4,8", "tensor\tmaster\tf32\t8,4")
                elif change == "path":
                    text = text.replace("candidate.raw", "../candidate.raw")
                elif change == "duplicate":
                    text += "config\tsteps\t2\n"
                elif change == "order":
                    text = text.replace("step\t1\tnew", "step\t1\treplay")
                else:
                    text += "unknown\tvalue\n"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    compare.fixture(m["root"], compare.sha(path))

    def test_output_receipt_uses_actual_three_ledgers_and_signed_kl(self):
        m = make_fixture(self.root / "input")
        fake_output(self.root / "output", m)
        receipt = compare.output_receipt(self.root / "output", m)
        self.assertEqual(receipt["report"]["final_adam_step"], 2)
        self.assertLess(receipt["trace"][0]["kl"], 0)

    def test_missing_output_state_false_execution_and_wrong_schedule_reject(self):
        m = make_fixture(self.root / "input")
        for change in ("missing", "passed", "counts", "input", "schedule"):
            root = self.root / ("output-" + change)
            fake_output(root, m)
            if change == "missing":
                (root / "final_exp_avg.raw").unlink()
            elif change == "schedule":
                path = root / "trace.tsv"
                path.write_text(path.read_text().replace("1\tnew", "1\treplay"))
            else:
                path = root / "result.json"
                report = json.loads(path.read_text())
                report[{"passed": "passed_execution", "counts": "new_ce_exposures", "input": "manifest_sha256"}[change]] = {"passed": 1, "counts": 2, "input": "f" * 64}[change]
                path.write_text(json.dumps(report))
            with self.assertRaises((ValueError, FileNotFoundError)):
                compare.output_receipt(root, m)

    def test_continue_imports_own_moments_step_and_keeps_reference_basis(self):
        m = make_fixture(self.root / "input")
        output = self.root / "output"
        fake_output(output, m)
        result = compare.continue_fixture(m["root"], output, self.root / "input2", m["manifest_sha256"])
        next_fixture = compare.fixture(result["directory"], result["manifest_sha256"])
        self.assertEqual(struct.unpack("<f", (Path(result["directory"]) / "adam_step.raw").read_bytes())[0], 2)
        for key in ("basis", "reference", "norm", "head", "r000_x", "r001_labels"):
            self.assertEqual(next_fixture["tensors"][key]["sha256"], m["tensors"][key]["sha256"])
        self.assertEqual(json.loads((Path(result["directory"]) / "continuation.json").read_text())["entry_adam_step"], 2)
        with self.assertRaises(ValueError):
            compare.continue_fixture(m["root"], output, self.root / "input2", m["manifest_sha256"])

    def test_fresh_input_overlap_and_symlink_reject(self):
        path = self.root / "input"
        path.mkdir()
        with self.assertRaises(ValueError):
            compare.fresh(path / "nested", (path,))
        alias = self.root / "alias"
        alias.symlink_to(path, target_is_directory=True)
        with self.assertRaises(ValueError):
            compare.safe(alias)

    def test_source_has_lazy_torch_and_fixed_numeric_policy(self):
        tree = ast.parse((ROOT / "scripts/compare_native_training.py").read_text())
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any((getattr(node, "module", "") or "").startswith("torch")
                              or any(a.name.startswith("torch") for a in getattr(node, "names", [])) for node in imports))
        self.assertEqual(compare.TOLERANCES["mps"]["gradient_atol"], 2**-16)
        self.assertEqual(compare.TOLERANCES["mps"]["delta_relative_l2"], 1e-3)
        self.assertEqual(compare.TOLERANCES["mps"]["candidate_agreement"], .9999)
        self.assertEqual(compare.TOLERANCES["cpu"]["candidate_agreement"], 1.)


if __name__ == "__main__":
    unittest.main()
