"""Stdlib IPC, lock inheritance, immutable state and admission counterexamples.

The synthetic worker tests orchestration only; it never trains or loads Torch.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luban import luban_training as training
from luban.process_lock import ModelProcessLock


def tensor(root, name, shape, dtype="f32", data=None):
    count = __import__("math").prod(shape)
    raw = data if data is not None else b"\0" * (count * training.WIDTHS[dtype])
    (root / (name + ".raw")).write_bytes(raw)
    return {"name": name, "dtype": dtype, "shape": shape, "file": name + ".raw",
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def descriptor(value):
    return "\t".join(("tensor", value["name"], value["dtype"], ",".join(map(str, value["shape"])) or "-",
                      value["file"], str(value["bytes"]), value["sha256"]))


def fixture(root, identity, *, steps=3, offset=1):
    root.mkdir()
    values = {key: tensor(root, key, [2, 3], "bf16" if key == "candidate" else "f32")
              for key in ("candidate", "master", "reference", "exp_avg", "exp_avg_sq")}
    values.update(adam_step=tensor(root, "adam_step", [], data=struct.pack("f", offset)),
        basis=tensor(root, "basis", [3, 1]), norm=tensor(root, "norm", [2], "bf16"),
        head=tensor(root, "head", [4, 2], "bf16"))
    for key in ("r000", "r001"):
        for suffix, shape, dtype in (("x", [1, 2, 3], "bf16"), ("residual", [1, 2, 2], "bf16"),
                ("ids", [1, 2], "i64"), ("labels", [1, 2], "i64")):
            values[key + "_" + suffix] = tensor(root, key + "_" + suffix, shape, dtype)
    config = {"identity": identity, "device": "cpu", "steps": str(steps), "norm_eps": "1e-6", "train_eos": "3",
        "checkpoint_steps": ",".join(map(str, training.checkpoints(steps))), "training_offset": str(offset),
        "learning_rate": "1e-5", "distillation_weight": "5"}
    lines = [training.ABI] + ["\t".join(("config", k, v)) for k, v in config.items()]
    lines += [descriptor(v) for v in values.values()]
    lines += ["row\tr000\tnew\t" + "a" * 64 + "\t2\t1", "row\tr001\treplay\t" + "b" * 64 + "\t2\t1"]
    for step in range(1, steps + 1):
        new = (offset + step - 1) % 2 == 0
        lines.append(f"step\t{step}\t{'new' if new else 'replay'}\t{'r000' if new else 'r001'}\tr001")
    (root / "manifest.tsv").write_text("\n".join(lines) + "\n")
    return training.validate_fixture(root, training.file_sha(root / "manifest.tsv"))


# Standalone fake IPC worker: asserts inherited lock and writes the actual raw
# checkpoint schema. Its zero tensors cannot be taken as training evidence.
WORKER = '''
import fcntl, hashlib, json, os, pathlib, struct, sys, time
binary, _, source, _, target, _, approved = sys.argv
root, output = pathlib.Path(source), pathlib.Path(target)
fd = int(os.environ["LUBAN_MODEL_LOCK_FD"])
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
manifest = (root / "manifest.tsv").read_text().splitlines()
config, specs, schedule = {}, {}, []
for line in manifest[1:]:
    p = line.split("\\t")
    if p[0] == "config": config[p[1]] = p[2]
    elif p[0] == "tensor": specs[p[1]] = p
    elif p[0] == "step": schedule.append(p)
named = pathlib.Path("/tmp") / ("luban-model-locks-"+str(os.getuid())) / (config["identity"]+"-cpu.lock")
with named.open("rb") as other:
    try: fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: pass
    else: raise RuntimeError("inherited lock unexpectedly released")
output.mkdir()
entry = int(config.get("training_offset", "0"))
keys = ["ce_logits","teacher_log_probs","student_log_probs","ce_grad","kl_grad","total_grad","clipped_grad","delta","master","candidate","exp_avg","exp_avg_sq","adam_step"]
with (output / "tensors.tsv").open("w") as listing, (output / "trace.tsv").open("w") as trace:
    listing.write("LUBAN_NATIVE_TRAINING_OUTPUT_V1\\n")
    trace.write("step\\tce_role\\tce_row\\treplay_row\\tce\\tkl\\tpreclip_norm\\tloss\\twall_seconds\\n")
    def save(name, schema, data):
        file = name + ".raw"
        (output/file).write_bytes(data)
        listing.write("\\t".join(["tensor",name,schema[2],schema[3],file,str(len(data)),hashlib.sha256(data).hexdigest()])+"\\n")
        listing.flush()
    for row in schedule:
        step = int(row[1])
        if str(step) in config["checkpoint_steps"].split(","):
            for key in keys:
                if key == "ce_logits" or key.endswith("log_probs"):
                    schema = ["tensor",key,"f32","1,4"]; data = b"\\0"*16
                else:
                    base = "candidate" if key in ["ce_grad","kl_grad","total_grad","candidate"] else key if key in specs else "master"
                    schema = specs[base]; data = (root/schema[4]).read_bytes()
                if key == "adam_step": data = struct.pack("f",entry+step)
                save("checkpoint%06d_"%step+key,schema,data)
        trace.write("\\t".join(row[1:])+"\\t1\\t0.1\\t2\\t1.5\\t0.01\\n"); trace.flush()
        if os.environ.get("LUBAN_TEST_STALL") == "1": time.sleep(20)
    for key in ["candidate","master","exp_avg","exp_avg_sq","adam_step","reference","basis"]:
        schema = specs[key]; data = (root/schema[4]).read_bytes()
        if key == "adam_step": data = struct.pack("f",entry+len(schedule))
        save("final_"+key,schema,data)
report = {"abi":"LUBAN_NATIVE_TRAINING_V1","passed_execution":True,"manifest_sha256":approved,
"binary_sha256":hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),"device":"cpu",
"release_ce_graph":True,"reuse_replay_forward":True,"publication":False,"all_input_tensors_preserved":True,
"input_tensor_count":len(specs),"steps":len(schedule),"new_ce_exposures":sum(r[2]=="new" for r in schedule),
"replay_ce_exposures":sum(r[2]=="replay" for r in schedule),"replay_kl_exposures":len(schedule),
"entry_adam_step":entry,"final_adam_step":entry+len(schedule),"teacher_calls":0,"generation_calls":0}
(output/"result.json").write_text(json.dumps(report))
'''


class LubanTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.identity = hashlib.sha256(self.temp.name.encode()).hexdigest()
        self.lock = ModelProcessLock(self.identity, "cpu")
        self.runtime = types.SimpleNamespace(identity=self.identity, device="cpu", _model_process_lock=self.lock,
            training_session=None, job={}, native_engine=None)
        self.fixture = fixture(self.root / "fixture", self.identity)
        self.binary = self.root / "worker"
        self.binary.write_text("#!" + sys.executable + "\n" + WORKER)
        self.binary.chmod(0o700)
        libraries = {}
        for i in range(3):
            path = self.root / (str(i) + ".dylib")
            path.write_bytes(b"synthetic IPC dependency")
            libraries[str(path)] = {"bytes": path.stat().st_size, "sha256": training.file_sha(path)}
        self.build = {"binary_sha256": training.file_sha(self.binary), "release_ce_graph": True,
            "reuse_replay_forward": True, "model_lock_inheritance": True, "torch_library_bindings": libraries}
        self.build_path = Path(str(self.binary) + ".build.json")
        self.build_path.write_text(json.dumps(self.build))

    def tearDown(self):
        self.lock.close()
        self.temp.cleanup()

    def test_no_interpreter_or_model_import_on_helper_admission(self):
        program = "import sys; from luban import luban_training; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        completed = subprocess.run([sys.executable, "-S", "-c", program],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dynamic_steps_checkpoints_and_resume_parity(self):
        self.assertEqual(training.checkpoints(451), [1, 2, *range(50, 451, 50), 451])
        self.assertEqual(self.fixture["steps"][0]["role"], "replay")
        self.assertEqual(training.checkpoints(1), [1])
        for value in (True, 0, 4097):
            with self.assertRaises(ValueError): training.checkpoints(value)
        for selected in ([2, 1, 3], [1, 3, 3], [1, 2], [False, 3]):
            with self.assertRaises(ValueError): training.checkpoints(3, selected)

    def test_same_file_description_lock_is_inherited_without_window(self):
        self.assertEqual(training.inherited_lock(self.runtime), self.lock.fd)
        receipt = training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary)
        self.assertFalse(receipt["quality_checked"])
        self.assertEqual(receipt["report"]["final_adam_step"], 4)
        # Child exit did not unlock the parent's descriptor.
        program = "from luban.process_lock import ModelProcessLock; ModelProcessLock('"+self.identity+"','cpu')"
        completed = subprocess.run([sys.executable, "-S", "-c", program],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, capture_output=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("已有模型进程".encode(), completed.stderr)

    def test_rejects_unrelated_inherited_fd(self):
        with (self.root / "foreign").open("wb") as stream:
            self.runtime._model_process_lock = types.SimpleNamespace(fd=stream.fileno(), key=(self.identity, "cpu"))
            with self.assertRaisesRegex(ValueError, "描述符"): training.inherited_lock(self.runtime)

    def test_rejects_old_binary_and_dependency_changes_before_spawning(self):
        for change in ({"model_lock_inheritance": False}, {"release_ce_graph": False}):
            self.build_path.write_text(json.dumps({**self.build, **change}))
            with self.assertRaises(ValueError), patch.object(training.subprocess, "Popen") as start:
                training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary)
            start.assert_not_called()
        self.build_path.write_text(json.dumps(self.build))
        Path(next(iter(self.build["torch_library_bindings"]))).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "LibTorch"), patch.object(training.subprocess, "Popen") as start:
            training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary)
        start.assert_not_called()

    def test_complete_adam_descriptor_survives_cancel_and_child_is_reaped(self):
        saved = []
        def checkpoint(step, descriptor):
            saved.append((step, descriptor))
            self.runtime.job["cancel_requested"] = True
        with patch.dict(os.environ, {"LUBAN_TEST_STALL": "1"}), self.assertRaisesRegex(RuntimeError, "取消"):
            training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary, on_checkpoint=checkpoint)
        self.assertEqual(saved[0][0], 1)
        self.assertEqual(set(saved[0][1]), set(training.FINAL))
        record = json.loads((self.root / "output.execution.json").read_text())
        self.assertFalse(record["completed"])
        self.assertEqual(record["saved_steps"], [1])
        with self.assertRaises(ProcessLookupError): os.kill(record["pid"], 0)
        self.assertIsNotNone(self.lock.fd)

    def test_callback_error_also_reaps_child_and_preserves_raw(self):
        def checkpoint(step, descriptor): raise OSError("durable checkpoint failed")
        with patch.dict(os.environ, {"LUBAN_TEST_STALL": "1"}), self.assertRaisesRegex(OSError, "checkpoint"):
            training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary, on_checkpoint=checkpoint)
        self.assertTrue((self.root / "output/checkpoint000001_master.raw").exists())
        record = json.loads((self.root / "output.execution.json").read_text())
        self.assertEqual(record["saved_steps"], [])
        with self.assertRaises(ProcessLookupError): os.kill(record["pid"], 0)

    def test_pause_saves_complete_checkpoint_then_raises_existing_pause_type(self):
        class TrainingPaused(Exception): pass
        self.runtime.training_session = types.SimpleNamespace(pause_requested=False)
        saved = []
        def checkpoint(step, descriptor):
            saved.append((step, descriptor))
            self.runtime.training_session.pause_requested = True
        module = types.SimpleNamespace(TrainingPaused=TrainingPaused)
        with patch.dict(sys.modules, {"luban.training_checkpoint": module}), \
                patch.dict(os.environ, {"LUBAN_TEST_STALL": "1"}), self.assertRaises(TrainingPaused):
            training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary, on_checkpoint=checkpoint)
        self.assertEqual(set(saved[0][1]), set(training.FINAL))
        # A partial descriptor for the following step cannot invalidate the
        # already complete previous checkpoint retained for durable recovery.
        with (self.root / "output/tensors.tsv").open("a") as stream:
            stream.write("tensor\tcheckpoint000002_master")
        self.assertEqual(set(training._state_descriptors(self.root / "output", self.fixture, 1)), set(training.FINAL))
        record = json.loads((self.root / "output.execution.json").read_text())
        self.assertEqual(record["saved_steps"], [1])
        with self.assertRaises(ProcessLookupError): os.kill(record["pid"], 0)

    def test_timeout_reaps_child_and_preserves_parent_lock(self):
        with patch.dict(os.environ, {"LUBAN_TEST_STALL": "1"}), self.assertRaisesRegex(ValueError, "超时"):
            training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary, timeout=.2)
        record = json.loads((self.root / "output.execution.json").read_text())
        self.assertFalse(record["completed"])
        with self.assertRaises(ProcessLookupError): os.kill(record["pid"], 0)
        self.assertEqual(training.inherited_lock(self.runtime), self.lock.fd)

    def test_rejects_wrong_adam_progress_and_trace_loss(self):
        receipt = training.run_worker(self.runtime, self.fixture, self.root / "output", self.binary)
        report_path = self.root / "output/result.json"
        report = receipt["report"]
        report_path.write_text(json.dumps({**report, "final_adam_step": 3}))
        with self.assertRaisesRegex(ValueError, "Adam"): training.validate_output(self.root / "output", self.fixture, self.build["binary_sha256"])
        report_path.write_text(json.dumps(report))
        trace_path = self.root / "output/trace.tsv"
        trace_path.write_text(trace_path.read_text().replace("\t1.5\t", "\t1.6\t"))
        with self.assertRaisesRegex(ValueError, "CE/KL"): training.validate_output(self.root / "output", self.fixture, self.build["binary_sha256"])

    def test_input_byte_changes_and_symlink_rejected(self):
        path = self.root / "fixture/master.raw"
        original = path.read_bytes()
        path.write_bytes(b"x" + original[1:])
        with self.assertRaisesRegex(ValueError, "SHA"): training.validate_fixture(self.root / "fixture", self.fixture["manifest_sha256"])
        path.unlink()
        backing = self.root / "backing.raw"
        backing.write_bytes(original)
        path.symlink_to(backing)
        with self.assertRaisesRegex(ValueError, "链接"): training.validate_fixture(self.root / "fixture", self.fixture["manifest_sha256"])


if __name__ == "__main__": unittest.main()
