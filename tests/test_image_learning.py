"""Product image learning admission/round persistence without model loading."""
from contextlib import contextmanager
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from luban import image_learning as image
from luban.memory import Memory


class ImageLearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.runtime = SimpleNamespace(memory=Memory(root), store=root / "modules", identity="f" * 64,
            active_core=None, branch=SimpleNamespace(expert_id=None), dtype=torch.bfloat16,
            device="cpu", model=SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5")))
        self.records = [dict(id="train", kind="image", split="training", family="a", question="Q?", answer="yes",
                             image={"width": 1, "height": 1, "rgb": b"\x00\x00\x00"}, approved=True, training_allowed=True),
                        dict(id="holdout", kind="image", split="validation", family="b", question="Q?", answer="no",
                             image={"width": 1, "height": 1, "rgb": b"\xff\xff\xff"}, approved=True, training_allowed=True)]

    def test_capability_does_not_load_or_compile(self):
        with patch.object(image, "library_path", return_value=Path("unused")):
            result = image.capability(self.runtime)
        self.assertTrue(result["available"])
        self.assertEqual(result["round_budget"], 5)

    def test_missing_component_is_unavailable_without_fallback(self):
        with patch.object(image, "library_path", side_effect=RuntimeError("missing")):
            self.assertEqual(image.capability(self.runtime)["reason"], "missing")

    def test_same_family_cannot_cross_split(self):
        self.records[1]["family"] = "a"
        with self.assertRaisesRegex(ValueError, "同源"):
            image._records(self.runtime, self.records)

    def test_permissions_are_independent(self):
        for field in ("approved", "training_allowed"):
            with self.subTest(field=field):
                record = [dict(x) for x in self.records]
                record[0][field] = False
                with self.assertRaises(ValueError):
                    image._records(self.runtime, record)

    def test_question_answer_required(self):
        self.records[0]["answer"] = ""
        with self.assertRaises(ValueError):
            image._records(self.runtime, self.records)

    def test_validation_required(self):
        self.records[1]["split"] = "training"
        with self.assertRaises(ValueError):
            image._records(self.runtime, self.records)

    def test_mixed_modality_and_duplicate_id_rejected(self):
        for field, value in (("kind", "operation"), ("id", "train")):
            with self.subTest(field=field):
                record = [dict(x) for x in self.records]
                record[1][field] = value
                with self.assertRaises(ValueError):
                    image._records(self.runtime, record)

    def test_external_file_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as external:
            path = Path(external) / "image.png"
            path.write_bytes(b"image")
            self.records[0]["asset_path"] = str(path)
            with self.assertRaisesRegex(ValueError, "私有"):
                image._records(self.runtime, self.records)
            linked = Path(self.temporary.name) / "image.png"
            linked.symlink_to(path)
            self.records[0]["asset_path"] = str(linked)
            with self.assertRaises(ValueError):
                image._records(self.runtime, self.records)

    def test_candidate_id_is_opaque_and_identity_bound(self):
        binding = image._binding(self.runtime, self.records)
        body = image._candidate(self.runtime, binding, None)
        self.assertRegex(body["id"], "^[a-f0-9]{12}$")
        self.assertFalse(body["adopted"])
        with self.assertRaises(ValueError):
            image._candidate(self.runtime, binding, "/tmp/other.pt")
        with self.runtime.memory.connect() as db:
            image._schema(db)
            db.execute("INSERT INTO modality_candidates VALUES (?,?,?,?,?,?)",
                       (body["id"], self.runtime.identity, "image", 1, json.dumps(body), 1))
        self.assertEqual(image._candidate(self.runtime, binding, body["id"])["generation"], 1)
        with self.assertRaisesRegex(ValueError, "不一致"):
            image._candidate(self.runtime, {**binding, "core_version": "changed"}, body["id"])

    def test_completed_round_does_not_claim_adoption_or_evolution(self):
        self.assertEqual(image.ROUND_BUDGET, 5)
        self.assertNotIn("evolution_count", image._candidate(self.runtime, image._binding(self.runtime, self.records), None))

    def test_model_lifecycle_restores_device_branch_and_state_on_failure(self):
        held = SimpleNamespace(fd=1, key=(self.runtime.identity, "cpu"))
        self.runtime._model_process_lock = held
        self.runtime._ready = lambda: None
        self.runtime.state = "learning"
        expert = object()
        self.runtime.branch.expert = expert
        changes = []
        self.runtime.model.to = lambda device: changes.append(str(device))
        @contextmanager
        def scope(runtime):
            yield
        lock = SimpleNamespace(close=lambda: changes.append("lock closed"))
        with patch.object(image, "ModelProcessLock", return_value=lock), patch.object(image, "full_backbone", scope), patch.object(image, "load_visual"):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with image.load_for_learning(self.runtime):
                    self.assertIsNone(self.runtime.branch.expert)
                    raise RuntimeError("failure")
        self.assertIs(self.runtime.branch.expert, expert)
        self.assertEqual(self.runtime.state, "learning")
        self.assertEqual(changes, ["cpu", "cpu", "lock closed"])

    def test_restore_device_error_still_restores_branch(self):
        expert = object()
        self.runtime.branch.expert = expert
        self.runtime.state = "learning"
        self.runtime._ready = lambda: None
        self.runtime._model_process_lock = SimpleNamespace(fd=1, key=(self.runtime.identity, "cpu"))
        calls = []
        def move(device):
            calls.append(device)
            if len(calls) == 2:
                raise RuntimeError("restore failed")
        self.runtime.model.to = move
        lock = SimpleNamespace(close=lambda: calls.append("closed"))
        with patch.object(image, "ModelProcessLock", return_value=lock), patch.object(image, "full_backbone", return_value=nullcontext()), patch.object(image, "load_visual"):
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                with image.load_for_learning(self.runtime):
                    self.assertIsNone(self.runtime.branch.expert)
        self.assertIs(self.runtime.branch.expert, expert)
        self.assertEqual(calls[-1], "closed")

    def test_video_binding_includes_order_time_and_private_paths(self):
        for n, record in enumerate(self.records):
            record["kind"] = "video"
            record.pop("image")
            frames = []
            for i in range(2):
                path = Path(self.temporary.name) / f"frame{n}-{i}.jpg"
                path.write_bytes(b"fixture")
                frames.append({"asset_path": str(path), "timestamp_seconds": float(i)})
            record["video_frames"] = frames
        admitted = image._records(self.runtime, self.records)
        binding = image._binding(self.runtime, admitted)
        self.assertEqual(binding["protocol"], "resident-video-learning-v1")
        self.assertEqual(binding["records"][0]["video_frames"], admitted[0]["video_frames"])
        changed = json.loads(json.dumps(admitted))
        changed[0]["video_frames"][1]["timestamp_seconds"] = 2
        self.assertNotEqual(binding, image._binding(self.runtime, changed))
        self.records[0]["video_frames"][1]["asset_path"] = self.records[0]["video_frames"][0]["asset_path"]
        with self.assertRaisesRegex(ValueError, "重复"):
            image._records(self.runtime, self.records)

    def _round_scope(self):
        import threading
        self.runtime.lock = threading.RLock()
        self.runtime._ready = lambda: None
        self.runtime.loaded_path = "unused"
        self.runtime.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "wrong")
        self.steps = []
        self.greedy_calls = 0
        tests = self
        class Engine:
            def __init__(self, *args, **kwargs):
                self.step_count = 0
            def __enter__(self):
                return self
            def __exit__(self, *unused):
                self.close()
            def close(self):
                pass
            def greedy(self, *args):
                tests.greedy_calls += 1
                if getattr(tests, "fail_post_validation", False) and tests.greedy_calls > 1:
                    raise RuntimeError("validation failure")
                return [5, 248046]
            def step(self, *args):
                self.step_count += 1
                tests.steps.append(self.step_count)
                return 1.25
            def save(self, path, identity):
                Path(path).write_text(json.dumps({"step": self.step_count, "identity": identity}))
            def restore(self, path, identity, step):
                data = json.loads(Path(path).read_text())
                assert data == {"step": step, "identity": identity}
                self.step_count = step
        stack = ExitStack()
        stack.enter_context(patch.object(image, "capability", return_value={"available": True}))
        stack.enter_context(patch.object(image.psutil, "virtual_memory", return_value=SimpleNamespace(available=10**12)))
        stack.enter_context(patch.object(image, "load_for_learning", side_effect=lambda r: nullcontext(r.model)))
        stack.enter_context(patch.object(image, "image_processor", return_value=object()))
        stack.enter_context(patch.object(image, "MasterQA", return_value=object()))
        stack.enter_context(patch.object(image, "prepare_sample", return_value=((), torch.tensor([[-100, 248046]]))))
        stack.enter_context(patch.object(image, "lower_graph", return_value=([], [])))
        stack.enter_context(patch.object(image, "NativeEngine", Engine))
        return stack

    def test_fixed_round_finishes_unlearned_and_next_call_continues_candidate(self):
        with self._round_scope():
            first = image.train_round(self.runtime, self.records)
            self.assertEqual(first["state"], "awaiting_quality")
            self.assertEqual(first["completed_steps"], 5)
            self.assertEqual(first["total_adam_step"], 5)
            self.assertEqual(first["learned"], 0)
            self.assertFalse(first["adopted"])
            self.assertFalse(first["protection_checked"])
            self.assertRegex(first["candidate_path"], "^[a-f0-9]{12}$")
            self.assertEqual(self.steps, [1, 2, 3, 4, 5])
            second = image.train_round(self.runtime, self.records, first["candidate_path"])
            self.assertEqual(second["total_adam_step"], 10)
            self.assertEqual(second["completed_steps"], 5)
            self.assertEqual(self.steps, list(range(1, 11)))
            self.assertEqual(second["candidate_path"], first["candidate_path"])

    def test_cancel_saves_completed_step_and_resumes_remainder_not_new_budget(self):
        with self._round_scope():
            first = image.train_round(self.runtime, self.records, cancel=lambda: len(self.steps) >= 2)
            self.assertEqual(first["state"], "paused")
            self.assertEqual(first["completed_steps"], 2)
            self.assertFalse(first["quality_checked"])
            second = image.train_round(self.runtime, self.records, first["candidate_path"])
            self.assertEqual(second["total_adam_step"], 5)
            self.assertEqual(second["actual_updates"], 3)
            self.assertEqual(second["completed_steps"], 5)
            self.assertEqual(self.steps, [1, 2, 3, 4, 5])

    def test_round_relative_progress_and_durable_candidate_before_validation(self):
        with self._round_scope():
            first = image.train_round(self.runtime, self.records)
            events = []
            def progress(event):
                events.append(event)
                if "candidate" in event:
                    candidate = event["candidate"]
                    restored = image._candidate(self.runtime, candidate["binding"], candidate["candidate_path"])
                    self.assertEqual(restored["step"], 10)
                    self.assertEqual(candidate["completed_steps"], 5)
                    self.assertFalse(candidate["adopted"])
            second = image.train_round(self.runtime, self.records, first["candidate_path"], progress=progress)
            self.assertTrue(all(0 <= event["step"] <= 5 and event["steps"] == 5 for event in events))
            self.assertEqual(events[0]["step"], 0)
            self.assertEqual(events[-1]["step"], 5)
            self.assertEqual(second["total_adam_step"], 10)
            self.assertTrue(any("candidate" in event for event in events))

    def test_validation_failure_retains_completed_checkpoint(self):
        with self._round_scope():
            self.fail_post_validation = True
            events = []
            result = image.train_round(self.runtime, self.records, progress=events.append)
            self.assertEqual(result["total_adam_step"], 5)
            self.assertFalse(result["quality_checked"])
            self.assertEqual(result["validation_error"], "validation failure")
            self.assertEqual(image._candidate(self.runtime, result["binding"], result["candidate_path"])["step"], 5)
            self.assertTrue(any(event.get("candidate", {}).get("candidate_path") == result["candidate_path"] for event in events))

    def test_original_uploaded_formats_use_existing_image_decoder(self):
        from PIL import Image
        from luban.native_image_training import prepare_sample
        for suffix, format_name in (("png", "PNG"), ("jpg", "JPEG"), ("webp", "WEBP")):
            with self.subTest(format=format_name):
                path = Path(self.temporary.name) / ("upload." + suffix)
                Image.new("RGB", (128, 96), (25, 75, 125)).save(path, format=format_name)
                def processor(images, **options):
                    self.assertEqual(images[0].size, (128, 96))
                    self.assertEqual(images[0].mode, "RGB")
                    self.assertEqual(options["max_pixels"], 1024)
                    raise RuntimeError("decoder accepted")
                with self.assertRaisesRegex(RuntimeError, "decoder accepted"):
                    prepare_sample(None, None, processor, {"asset_path": str(path)})

    def test_cancel_before_start_has_no_fake_checkpoint(self):
        self.runtime._ready = lambda: None
        import threading
        self.runtime.lock = threading.RLock()
        with patch.object(image, "capability", return_value={"available": True}):
            result = image.train_round(self.runtime, self.records, cancel=lambda: True)
        self.assertIsNone(result["candidate_path"])
        self.assertEqual(result["total_adam_step"], 0)


if __name__ == "__main__":
    unittest.main()
