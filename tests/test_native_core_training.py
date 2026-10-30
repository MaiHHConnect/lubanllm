"""CPU-free tests for engine selection and interruption-ledger validation."""
import ast
from contextlib import nullcontext
import copy
import csv
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import uuid


SOURCE = Path(__file__).resolve().parents[1] / "src/luban/native_core_training.py"
TRACE = ("step", "ce_role", "ce_row", "replay_row", "ce", "kl", "preclip_norm", "loss", "wall_seconds")


def functions(wanted=None):
    tree = ast.parse(SOURCE.read_text())
    wanted = wanted or {"product_graph", "defaults", "select", "_trace"}
    subset = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                             and node.name in wanted], type_ignores=[])
    namespace = {"__file__": str(SOURCE), "__package__": "native_core_test", "Path": Path,
        "csv": csv, "math": math, "copy": copy, "uuid": uuid,
        "torch": SimpleNamespace(bfloat16="bf16"),
        "engine": SimpleNamespace(ENGINE_NAME="Luban 训练引擎", ENGINE_ID="luban-training-ce-release-v1",
                                  TRACE=TRACE, file_sha=lambda _: "a" * 64)}
    exec(compile(subset, str(SOURCE), "exec"), namespace)
    return namespace


def runtime(**changes):
    text = SimpleNamespace(hidden_size=2048, intermediate_size=6144,
                           num_hidden_layers=24, vocab_size=248320)
    config = SimpleNamespace(model_type="qwen3_5", text_config=text)
    values = dict(research=False, device="mps", dtype="bf16", model=SimpleNamespace(config=config))
    values.update(changes)
    return SimpleNamespace(**values)


class Tensor:
    def __init__(self, value, device="mps"):
        self.value, self.device, self.grad = value, device, None

    def detach(self):
        return self

    def cpu(self):
        return self.to("cpu")

    def to(self, device):
        return Tensor(self.value, device)

    def clone(self):
        return self.to(self.device)

    def copy_(self, other):
        self.value = other.value

    @property
    def data(self):
        return self

    @data.setter
    def data(self, other):
        self.value, self.device = other.value, other.device


class NativeCoreTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fn = functions()

    def test_product_defaults_are_native_budget_and_preservation(self):
        self.assertEqual(self.fn["defaults"](runtime(), None, None, None, None), (450, 1e-5, 5, True))
        self.assertEqual(self.fn["defaults"](runtime(), 80, .002, 3, False), (80, .002, 3, False))

    def test_other_graphs_keep_the_explicit_torch_defaults(self):
        for model in (runtime(device="cpu"), runtime(research=True), runtime(dtype="fp32"), runtime(model=None)):
            with self.subTest(model=model):
                self.assertEqual(self.fn["defaults"](model, None, None, None, None), (64, 1e-4, 0, False))

    def test_compatible_product_cannot_silently_fall_back_without_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fn["__file__"] = str(Path(folder) / "src/luban/native_core_training.py")
            with self.assertRaisesRegex(RuntimeError, "不能静默"):
                self.fn["select"](runtime(), preserve_replay_subspace=True, distillation_weight=5,
                                  field_first_token_weight=1, partial_policy=None)

    def test_explicit_non_native_objective_is_labeled(self):
        binary, descriptor = self.fn["select"](runtime(), preserve_replay_subspace=False,
            distillation_weight=0, field_first_token_weight=1, partial_policy=None)
        self.assertIsNone(binary)
        self.assertEqual(descriptor["engine"], "torch-hf")
        self.assertTrue(descriptor["reason"])

    def test_packaged_binary_precedes_workspace_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("native/luban-training", "build/luban-training"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"binary")
            self.fn["__file__"] = str(root / "src/luban/native_core_training.py")
            binary, descriptor = self.fn["select"](runtime(), preserve_replay_subspace=True,
                distillation_weight=5, field_first_token_weight=1, partial_policy=None)
            self.assertEqual(binary, (root / "native/luban-training").resolve())
            self.assertEqual(descriptor["engine"], "luban-training-ce-release-v1")
            partial, partial_descriptor = self.fn["select"](runtime(), preserve_replay_subspace=True,
                distillation_weight=5, field_first_token_weight=1, partial_policy="closed-book-growth-v1")
            self.assertEqual(partial, binary)
            self.assertEqual(partial_descriptor["engine"], "luban-training-ce-release-v1")

    def trace(self, mutation=None, count=1):
        with tempfile.TemporaryDirectory() as folder:
            row = dict(zip(TRACE, ("1", "new", "r000", "r001", "1", ".02", "2", "1.1", ".3")))
            if mutation:
                row.update(mutation)
            with (Path(folder) / "trace.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, delimiter="\t", fieldnames=TRACE)
                writer.writeheader()
                writer.writerow(row)
            fixture = {"config": {"distillation_weight": "5"},
                "steps": [{"ordinal": 1, "role": "new", "ce": "r000", "replay": "r001"}]}
            return self.fn["_trace"](folder, fixture, count)

    def test_checkpoint_trace_is_numeric_and_bound_to_schedule(self):
        self.assertEqual(self.trace()[0]["kl"], .02)
        with self.assertRaisesRegex(ValueError, "排程"):
            self.trace({"ce_row": "r001"})

    def test_corrupt_scalar_and_partial_trace_cannot_be_journaled(self):
        with self.assertRaisesRegex(ValueError, "非有限"):
            self.trace({"kl": "nan"})
        with self.assertRaisesRegex(ValueError, "损失"):
            self.trace({"loss": "1.2"})
        with self.assertRaisesRegex(ValueError, "完整步数"):
            self.trace(count=2)

    def test_native_checkpoint_saves_matching_adam_and_ledger_before_qc(self):
        ns = functions({"steps", "_trace", "_state", "_restore_state", "_move_optimizer", "_move_text"})
        moves, saves = [], []
        model = SimpleNamespace(device="mps")
        decoder = SimpleNamespace(to=lambda device: (moves.append(device), setattr(model, "device", device)),
                                  parameters=lambda: ())
        model.lm_head = SimpleNamespace(to=lambda device: None, parameters=lambda: ())
        master, parameter, reference, basis = (Tensor(0), Tensor(0), Tensor(0), Tensor(7))
        optimizer = SimpleNamespace(state={})
        report = {"id": "abc", "steps": 32, "learning_rate": 1e-5,
            "training_engine": {"segments": []}, "training_selection": {"checkpoints": [32]},
            "distillation": {"weight": 5, "steps": 0, "target_tokens": 0, "kl_mean": None}}
        groups = {"new_training": [{"id": "new"}], "replay_training": [{"id": "old"}]}
        ledger = ({"new": 0}, 0., 0., 0)
        caches = {row["id"]: {"binding": {"prefix": "same"}} for rows in groups.values() for row in rows}
        fixture = {"config": {"distillation_weight": "5"}, "rows": {"r000": {"gold": 2}, "r001": {"gold": 2}},
            "steps": [{"ordinal": i, "role": "new" if i % 2 else "replay",
                       "ce": "r000" if i % 2 else "r001", "replay": "r001"} for i in range(1, 33)],
            "manifest_sha256": "a" * 64}

        def load(output, exported, *, step=None):
            step = step or 32
            return {"candidate": Tensor(step, "cpu"), "master": Tensor(step, "cpu"),
                "exp_avg": Tensor(step + 1, "cpu"), "exp_avg_sq": Tensor(step + 2, "cpu"),
                "adam_step": Tensor(step, "cpu"), "reference": reference.cpu(), "basis": basis.cpu()}

        def run(actual, exported, output, binary, *, on_checkpoint):
            self.assertEqual(model.device, "cpu")
            self.assertEqual(master.device, "cpu")
            Path(output).mkdir()
            rows = [{"step": str(i), "ce_role": s["role"], "ce_row": s["ce"], "replay_row": s["replay"],
                "ce": "1", "kl": ".25", "preclip_norm": "2", "loss": "2.25", "wall_seconds": ".3"}
                for i, s in enumerate(exported["steps"], 1)]
            with (Path(output) / "trace.tsv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, delimiter="\t", fieldnames=TRACE)
                writer.writeheader()
                writer.writerows(rows)
            on_checkpoint(16, {})
            on_checkpoint(32, {})
            return {"receipt_sha256": "b" * 64, "report": {"done": True},
                    "trace": ns["_trace"](output, exported, 32)}

        def save(session, actual, shadow, target, old, value, opt, protected, exposures, total, maximum, streak):
            saves.append({"step": actual.job["step"], "master": value.value,
                "adam": opt.state[value]["step"].value, "report": copy.deepcopy(shadow),
                "exposures": dict(exposures), "total": total, "maximum": maximum, "streak": streak})

        ns["engine"] = SimpleNamespace(TRACE=TRACE, capture_tails=lambda *args: caches,
            export_fixture=lambda *args, **kwargs: fixture, load_checkpoint=load, run_worker=run,
            graph_binding=lambda actual: {"prefix": "same"})
        ns["torch"] = SimpleNamespace(Tensor=Tensor, zeros_like=lambda tensor: Tensor(0, tensor.device),
            tensor=lambda value, **kwargs: Tensor(value, "cpu"), float32="fp32", no_grad=nullcontext,
            mps=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {
                "native_core_test.evolution": SimpleNamespace(_save_core_step=save)}):
            actual = SimpleNamespace(store=folder, model=model, decoder=decoder, device="mps", training_session=object(), job={"step": 0})
            observed, outer = [], []
            for value in ns["steps"](actual, "binary", report, groups, parameter, Tensor(0, "cpu"),
                    master, optimizer, reference, basis, 0, lambda: ledger):
                observed.append(value)
                step = value[0] + 1
                if step % 16 == 0:
                    outer.append((step, master.value, parameter.value, optimizer.state[master]["step"].value))
        self.assertEqual([row[0] for row in observed], list(range(32)))
        self.assertEqual(moves, ["cpu", "mps"])
        self.assertEqual((master.value, parameter.value, master.device), (32, 32, "mps"))
        self.assertEqual(len(saves), 1)  # QC-boundary 32 is deliberately not journaled before its checks.
        self.assertEqual((saves[0]["step"], saves[0]["master"], saves[0]["adam"]), (16, 16, 16))
        self.assertEqual(saves[0]["exposures"], {"new": 8})
        self.assertEqual(saves[0]["total"], 4.)
        self.assertEqual(saves[0]["report"]["distillation"]["target_tokens"], 32)
        self.assertEqual(outer, [(16, 16, 16, 16), (32, 32, 32, 32)])

    def test_text_move_avoids_meta_visual_and_preserves_tied_parameter(self):
        ns = functions({"_move_text"})
        embedding = Tensor(42, "cpu")
        visual = Tensor(0, "meta")
        calls = []

        class Module:
            def __init__(self, name, weights):
                self.name, self.weights = name, weights

            def parameters(self):
                return iter(self.weights)

            def to(self, device):
                calls.append((self.name, device))
                if any(p.device == "meta" for p in self.weights):
                    raise NotImplementedError("Cannot copy out of meta tensor")
                for parameter in self.weights:
                    parameter.data = parameter.to(device)
                return self

        decoder, head = Module("decoder", [embedding]), Module("head", [embedding])
        model = Module("whole-model", [embedding, visual])
        model.lm_head = head
        actual = SimpleNamespace(model=model, decoder=decoder)
        with self.assertRaisesRegex(NotImplementedError, "meta"):
            model.to("cpu")
        calls.clear()
        before = id(embedding)
        ns["_move_text"](actual, "cpu")
        ns["_move_text"](actual, "mps")
        self.assertEqual(calls, [("decoder", "cpu"), ("head", "cpu"), ("decoder", "mps"), ("head", "mps")])
        self.assertIs(next(decoder.parameters()), next(head.parameters()))
        self.assertEqual((id(embedding), embedding.value, embedding.device, visual.device), (before, 42, "mps", "meta"))

    def test_text_move_rejects_replaced_parameters(self):
        ns = functions({"_move_text"})
        weights = [Tensor(42, "cpu")]
        decoder = SimpleNamespace(parameters=lambda: iter(weights), to=lambda device: weights.__setitem__(0, weights[0].to(device)))
        head = SimpleNamespace(parameters=lambda: (), to=lambda device: None)
        with self.assertRaisesRegex(RuntimeError, "Parameter"):
            ns["_move_text"](SimpleNamespace(decoder=decoder, model=SimpleNamespace(lm_head=head)), "cpu")


if __name__ == "__main__":
    unittest.main()
