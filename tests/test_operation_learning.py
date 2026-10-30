"""Private synthetic recorder bytes; never screen capture or model execution."""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid

from luban.operation_learning import prepare_demonstrations, validate_demonstration


def demonstration(index):
    app, task, session = f"com.fixture.app{index}", f"task{index}", f"session{index}"
    steps = []
    for n in range(2):
        steps.append({"observation": {"id": f"frame{n}", "task_id": task, "app_id": app, "session_id": session,
            "captured_at_ms": 1100 + n * 100, "phase": "before_action", "previous_action_id": f"action{n-1}" if n else None,
            "width": 2, "height": 2, "rgb": [n, 2, 3] * 4, "reviewed": True, "sensitive": False},
            "action": {"id": f"action{n}", "task_id": task, "app_id": app, "started_at_ms": 1110 + n * 100,
                "finished_at_ms": 1120 + n * 100, **({"type": "click", "x": 1, "y": 0, "button": "left", "count": 1}
                    if n == 0 else {"type": "type", "text": "cafe\u0301你好", "sensitive": False})}})
    return {"schema": "luban-operation-demonstration-v1", "id": f"demo{index}", "session_id": session,
        "source": "user_demonstration", "source_family": app, "approved": True, "training_allowed": True,
        "rights_basis": "Synthetic CPU fixture, no actual capture", "user_data_scope_confirmed": True,
        "sensitive_reviewed": True, "sensitive": False,
        "recording": {"start": {"explicit": True, "at_ms": 1000}, "stop": {"explicit": True, "at_ms": 2000}},
        "task": {"id": task, "family": app, "goal": "CPU fixture", "app_ids": [app], "allowed_actions": ["click", "type"]},
        "outcome": {"task_id": task, "verified_by": "user", "success": True, "at_ms": 2100, "evidence": "CPU fixture"}, "steps": steps}


class OperationLearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name).resolve()
        self.directory = self.home / "operation-demonstrations"
        self.directory.mkdir(mode=0o700)
        self.ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.selected = [{"file_id": item, "split": split} for item, split in zip(self.ids, ("TRAIN", "HOLDOUT"))]
        self.write(0, demonstration(0)); self.write(1, demonstration(1))

    def write(self, index, value):
        path = self.directory / (self.ids[index] + ".json")
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        path.chmod(0o600)
        return path

    def test_private_records_keep_before_rgb_complete_history_and_json(self):
        records = prepare_demonstrations(self.home, self.selected)
        self.assertEqual([r["split"] for r in records], ["training", "training", "validation", "validation"])
        first, second = records[:2]
        self.assertEqual(first["image"]["rgb"], bytes([0, 2, 3] * 4))
        self.assertEqual(json.loads(first["question"])["history"], [])
        self.assertEqual(json.loads(second["question"])["history"], [json.loads(first["answer"])])
        self.assertNotIn("cafe", second["question"])
        self.assertEqual(json.loads(second["answer"])["text"].encode(), "cafe\u0301你好".encode())
        self.assertNotEqual(json.loads(second["answer"])["text"].encode(), "café你好".encode())
        self.assertEqual(second["provenance"]["file_id"], self.ids[0])
        self.assertTrue(second["provenance"]["training_allowed"])

    def test_saved_rgb_rewrite_changes_resume_binding_without_hashing(self):
        first = prepare_demonstrations(self.home, self.selected)
        updated = demonstration(0)
        updated["steps"][0]["observation"]["rgb"][0] = 99
        self.write(0, updated)
        second = prepare_demonstrations(self.home, self.selected)
        self.assertEqual(first[0]["question"], second[0]["question"])
        self.assertEqual(first[0]["answer"], second[0]["answer"])
        self.assertNotEqual(first[0]["provenance"]["saved_file_stat"], second[0]["provenance"]["saved_file_stat"])

    def test_consent_scope_sensitive_timing_rgb_and_unknown_fields_reject(self):
        changes = [lambda d: d.update(approved=False), lambda d: d.update(training_allowed=False),
            lambda d: d.update(source="synthetic_fixture"), lambda d: d.update(sensitive=True),
            lambda d: d["outcome"].update(success=False), lambda d: d["recording"]["start"].update(explicit=False),
            lambda d: d["steps"][1]["observation"].update(previous_action_id=None),
            lambda d: d["steps"][0]["observation"].update(rgb=[0]),
            lambda d: d["steps"][0]["observation"].update(sensitive=True),
            lambda d: d["steps"][0]["action"].update(x=2),
            lambda d: d["steps"][1]["action"].update(sensitive=True),
            lambda d: d["steps"][0]["action"].update(started_at_ms=1001),
            lambda d: d.update(extra=True), lambda d: d["steps"][0]["action"].update(count=True)]
        for change in changes:
            d = demonstration(0); change(d)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_demonstration(d)

    def test_same_app_family_and_duplicate_session_never_cross_splits(self):
        d = demonstration(0)
        d["id"] = "different-recording"
        self.write(1, d)
        with self.assertRaises(ValueError):
            prepare_demonstrations(self.home, self.selected)
        d["session_id"] = "independent-session"
        for step in d["steps"]:
            step["observation"]["session_id"] = d["session_id"]
        self.write(1, d)
        with self.assertRaisesRegex(ValueError, "家族"):
            prepare_demonstrations(self.home, self.selected)

    def test_file_symlink_hardlink_mode_duplicate_uuid_and_traversal_reject(self):
        original = self.directory / (self.ids[0] + ".json")
        original.chmod(0o644)
        with self.assertRaises(ValueError): prepare_demonstrations(self.home, self.selected)
        original.chmod(0o600)
        link = self.directory / "extra.json"
        os.link(original, link)
        with self.assertRaises(ValueError): prepare_demonstrations(self.home, self.selected)
        link.unlink()
        content = original.read_bytes(); original.unlink(); original.symlink_to(self.directory / (self.ids[1] + ".json"))
        with self.assertRaises(OSError): prepare_demonstrations(self.home, self.selected)
        original.unlink(); original.write_bytes(content); original.chmod(0o600)
        with self.assertRaises(ValueError): prepare_demonstrations(self.home, [self.selected[0]] * 2)
        bad = copy.deepcopy(self.selected); bad[0]["file_id"] = "../other"
        with self.assertRaises(ValueError): prepare_demonstrations(self.home, bad)
        self.directory.chmod(0o755)
        with self.assertRaises(ValueError): prepare_demonstrations(self.home, self.selected)

    def test_duplicate_json_key_and_future_metadata_reject(self):
        original = self.directory / (self.ids[0] + ".json")
        original.write_text(original.read_text().replace('{"schema":', '{"schema":"bad","schema":', 1))
        with self.assertRaisesRegex(ValueError, "重复"): prepare_demonstrations(self.home, self.selected)
        d = demonstration(0); d["steps"][0]["observation"]["future_target"] = "leak"
        with self.assertRaises(ValueError): validate_demonstration(d)


class DynamicOperationKernelTests(unittest.TestCase):
    def test_exact_original_computation_body_and_complete_dynamic_gradients(self):
        import torch
        from luban.native_operation_training import scripted_original_chunk, _scripted_chunk
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
        # Verify the production static adaptation has exactly the accepted
        # original computation body; no generated source or pin hash at runtime.
        old = ast.parse(inspect.getsource(torch_chunk_gated_delta_rule)).body[0]
        new = ast.parse(inspect.getsource(scripted_original_chunk)).body[0]
        expected = copy.deepcopy(old.body)
        expected = [node for node in expected if not (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                    and isinstance(node.test.operand, ast.Name) and node.test.operand.id == "output_final_state")]
        expected[-1].value = ast.Name(id="core_attn_out", ctx=ast.Load())
        self.assertEqual(ast.dump(ast.Module(body=expected, type_ignores=[]), include_attributes=False),
                         ast.dump(ast.Module(body=new.body[2:], type_ignores=[]), include_attributes=False))
        for length in (64, 65, 204, 388):
            values = [torch.arange(length * 4, dtype=torch.float32).reshape(1, length, 1, 4).div(1000) for _ in range(3)]
            values += [torch.full((1, length, 1), -.05), torch.full((1, length, 1), .1)]
            left = [v.clone().requires_grad_() for v in values]
            right = [v.clone().requires_grad_() for v in values]
            with torch.jit.optimized_execution(False):
                original = torch_chunk_gated_delta_rule(*left)[0]
                dynamic = _scripted_chunk(*right)
                self.assertTrue(torch.equal(original, dynamic), length)
                gradients = torch.autograd.grad(original.sum(), left)
                candidates = torch.autograd.grad(dynamic.sum(), right)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(gradients, candidates)), length)
        with self.assertRaises(torch.jit.Error): _scripted_chunk(*right, output_final_state=True)
        with self.assertRaises(torch.jit.Error): _scripted_chunk(*right, cu_seqlens=torch.tensor([0, 388]))


if __name__ == "__main__":
    unittest.main()
