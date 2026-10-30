"""CPU neural/Adam resume and immutable publication; no pretrained model or HTTP."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from torch import nn

from luban.calibration import SPLITS, split_route_samples
from luban.decision import FastDecisionHead
from luban.dynamic import DynamicMLP, NeuralExpert
from luban.memory import Memory
from luban.runtime import Runtime
from luban.training_checkpoint import TrainingPaused, TrainingSession, recover_commits, training_runs


def sample(index, purpose="training"):
    return {"id": str(index), "prompt": "question " + str(index), "answer": "confirmed",
            "source": "human", "approved": 1, "training_allowed": 1, "purpose": purpose,
            "provenance": json.dumps({"messages": [{"role": "user", "content": str(index)}]})}


class SmallModel(nn.Module):
    def __init__(self, branch):
        super().__init__()
        self.branch = branch
        self.config = SimpleNamespace(model_type="cpu-fixture")

    def forward(self, ids, labels, use_cache):
        inputs = torch.nn.functional.dropout(ids.float(), p=.25, training=True)
        return SimpleNamespace(loss=(self.branch(inputs) - labels.float()).square().mean())


def make_runtime(root):
    runtime = Runtime(Mock(home=Path(root)), Memory(root))
    runtime.store = Path(root) / "modules"
    runtime.store.mkdir(exist_ok=True)
    runtime.identity = "runtime-resume-fixture"
    runtime.device, runtime.dtype = "cpu", torch.float32
    runtime.pointer = runtime.store / "active.json"
    base = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        base.weight.fill_(.1)
    runtime.branch = DynamicMLP(base, 4, 32, runtime.store / "layer-23", runtime.identity + ":layer-23", 1000000)
    runtime.model = SmallModel(runtime.branch)
    runtime.text_config = SimpleNamespace(hidden_size=4)
    runtime.router = FastDecisionHead(4)
    runtime._ready = Mock()
    runtime.status = Mock(return_value={"rss_bytes": 0, "metal_bytes": 0})
    runtime._sample_tokens = Mock(return_value=(torch.tensor([[1., 2., 3., 4.]]), torch.ones(1, 4)))
    runtime._loss = Mock(side_effect=lambda records: (10. if runtime.branch.expert is None else 1.) if records else None)
    runtime._tokens = Mock(side_effect=lambda messages, *, shallow=False: torch.tensor([[int(messages[-1]["content"]) % 2]]))
    runtime.shallow = Mock(side_effect=lambda ids: torch.tensor([[[4., 0., 0., 0.]]] if int(ids[0, 0]) == 0 else [[[0., 4., 0., 0.]]]))
    runtime.state = "ready"
    return runtime


def route_splits():
    rows = []
    index = 0
    for split in SPLITS:
        for action in ("local_core", "teacher"):
            for _ in range(3 if split == "train" else 2):
                label = 0 if action == "local_core" else 1
                row = sample(index, "training" if split == "train" else "validation")
                row["provenance"] = json.dumps({"route_action": action, "family_group": str(index), "split": split,
                    "route_only": True, "messages": [{"role": "user", "content": str(label)}]})
                rows.append(row)
                index += 1
    return split_route_samples(rows, FastDecisionHead.actions)


class RuntimeResumeTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.threads)
        # Freeze source fingerprint during isolated tests while sibling agents edit.
        self.sources = patch("luban.training_checkpoint.implementation_hashes", return_value={"fixture": "fixed-source"})
        self.sources.start()
        self.addCleanup(self.sources.stop)

    def directory(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return temporary.name

    def generation(self, runtime, rows):
        return [{"id": row["id"], "passed": True, "complete": True} for row in rows]

    def events(self, runtime):
        with runtime.memory.lock, runtime.memory.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM events")]

    def test_branch_adam_rng_and_all_candidates_resume_exact(self):
        manifest = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        full = make_runtime(self.directory())
        resumed = make_runtime(self.directory())
        with patch("luban.evolution._generation", side_effect=self.generation):
            full.training_session = TrainingSession(full, "learning", {"steps": 12}, manifest, run_id="111111111111")
            expected = full.learn()
            session = TrainingSession(resumed, "learning", {"steps": 12}, manifest, run_id="222222222222")
            resumed.training_session = session
            original_save = session.save
            def stop_after_eight(payload, progress):
                original_save(payload, progress)
                if progress["step"] == 8:
                    session.pause_requested = True
            session.save = stop_after_eight
            with self.assertRaises(TrainingPaused):
                resumed.learn()
            self.assertIsNone(resumed.branch.expert)
            self.assertIsNone(resumed.branch.expert_id)
            self.assertEqual(resumed.state, "ready")
            self.assertEqual(resumed.job["state"], "paused")
            self.assertFalse(resumed.pointer.exists())
            self.assertEqual(self.events(resumed), [])
            self.assertEqual(training_runs(resumed.memory, resumed.identity)[0]["state"], "running")
            session.pause()  # Worker releases owner only after Runtime/full_backbone exits.
            resumed.training_session = TrainingSession(resumed, "learning", {"steps": 12}, manifest, run_id=session.id)
            result = resumed.learn()
        self.assertTrue(expected["promoted"] and result["promoted"])
        self.assertTrue(resumed.branch.requires_gate)
        metadata = json.loads((resumed.branch._paths(result["active_expert"])[1]).read_text())
        self.assertTrue(metadata["requires_gate"])
        self.assertEqual(result["route_gate"]["fallback"], "local_core")
        self.assertEqual(result["records_sha256"], resumed.training_session.binding["records_sha256"])
        self.assertEqual(expected["chosen_width"], result["chosen_width"])
        self.assertEqual(expected["candidates"], result["candidates"])
        for name, tensor in full.branch.expert.state_dict().items():
            self.assertTrue(torch.equal(tensor, resumed.branch.expert.state_dict()[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in resumed.branch.expert.parameters()))
        self.assertEqual(len(self.events(resumed)), 1)
        recover_commits(resumed.memory, resumed.store, resumed.identity)
        self.assertEqual(len(self.events(resumed)), 1)

    def test_branch_four_candidates_resume_preserves_existing_expert(self):
        manifest = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        full, resumed = make_runtime(self.directory()), make_runtime(self.directory())
        torch.manual_seed(8)
        original = NeuralExpert(4, 64).state_dict()
        for runtime in (full, resumed):
            runtime.branch.width = 64
            runtime.branch.expert = NeuralExpert(4, 64)
            runtime.branch.expert.load_state_dict(original)
            runtime.branch.expert_id = "learned-original"
            runtime._persist_expert("learned-original")
            runtime._loss = Mock(side_effect=[10., None] + [1., None] * 4)
        pointer_before = resumed.pointer.read_bytes()
        with patch("luban.evolution._generation", side_effect=self.generation):
            full.training_session = TrainingSession(full, "learning", {"steps": 12}, manifest)
            expected = full.learn()
            session = TrainingSession(resumed, "learning", {"steps": 12}, manifest)
            resumed.training_session = session
            original_save = session.save
            def pause_second_candidate(payload, progress):
                original_save(payload, progress)
                if progress["candidate"] == 1 and progress["step"] == 8:
                    session.pause_requested = True
            session.save = pause_second_candidate
            with self.assertRaises(TrainingPaused):
                resumed.learn()
            self.assertEqual(resumed.pointer.read_bytes(), pointer_before)
            self.assertEqual(resumed.branch.expert_id, "learned-original")
            self.assertTrue(all(torch.equal(value, resumed.branch.expert.state_dict()[name]) for name, value in original.items()))
            # Completed candidate zero must not be evaluated again after resume.
            resumed._loss = Mock(side_effect=[1., None] * 3)
            session.pause()
            resumed.training_session = TrainingSession(resumed, "learning", {"steps": 12}, manifest, run_id=session.id)
            result = resumed.learn()
        self.assertEqual(len(result["candidates"]), 4)
        self.assertEqual(expected["candidates"], result["candidates"])
        for name, tensor in full.branch.expert.state_dict().items():
            self.assertTrue(torch.equal(tensor, resumed.branch.expert.state_dict()[name]), name)
        expected_candidates = full.training_session.load()["candidates"]
        actual_candidates = resumed.training_session.load()["candidates"]
        for expected_row, actual_row in zip(expected_candidates, actual_candidates):
            for name, tensor in expected_row["weights"].items():
                self.assertTrue(torch.equal(tensor, actual_row["weights"][name]), name)

    def test_branch_real_sigkill_resumes_second_candidate_exactly_once(self):
        manifest = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        full, resumed = make_runtime(self.directory()), make_runtime(self.directory())
        with patch("luban.evolution._generation", side_effect=self.generation):
            full.training_session = TrainingSession(full, "learning", {"steps": 12}, manifest)
            expected = full.learn()
        session = TrainingSession(resumed, "learning", {"steps": 12}, manifest, run_id="abcdef123456")
        session.pause()
        child = r'''
import os, signal, sys, torch
from unittest.mock import patch
from tests.test_runtime_resume import make_runtime, sample
from luban.training_checkpoint import TrainingSession
torch.set_num_threads(1)
runtime = make_runtime(sys.argv[1])
manifest = {"train":[sample(1)],"valid":[sample(2,"validation")],"protection":[]}
with patch("luban.training_checkpoint.implementation_hashes",return_value={"fixture":"fixed-source"}), patch("luban.evolution._generation",side_effect=lambda runtime,rows:[{"id":row["id"],"passed":True,"complete":True} for row in rows]):
    session=TrainingSession(runtime,"learning",{"steps":12},manifest,run_id="abcdef123456")
    runtime.training_session=session
    original=session.save
    def kill(payload,progress):
        original(payload,progress)
        if progress["candidate"]==1 and progress["step"]==8:
            os.kill(os.getpid(),signal.SIGKILL)
    session.save=kill
    runtime.learn()
'''
        process = subprocess.run([sys.executable, "-c", child, str(resumed.store.parent)], capture_output=True, timeout=30)
        self.assertEqual(process.returncode, -9, process.stderr.decode())
        self.assertFalse(resumed.pointer.exists())
        self.assertEqual(self.events(resumed), [])
        recover_commits(resumed.memory, resumed.store, resumed.identity)
        self.assertEqual(training_runs(resumed.memory, resumed.identity)[0]["state"], "paused")
        resumed.training_session = TrainingSession(resumed, "learning", {"steps": 12}, manifest, run_id=session.id)
        with patch("luban.evolution._generation", side_effect=self.generation):
            result = resumed.learn()
        self.assertEqual(expected["candidates"], result["candidates"])
        expected_candidates = full.training_session.load()["candidates"]
        actual_candidates = resumed.training_session.load()["candidates"]
        for expected_row, actual_row in zip(expected_candidates, actual_candidates):
            for name, tensor in expected_row["weights"].items():
                self.assertTrue(torch.equal(tensor, actual_row["weights"][name]), name)
        self.assertEqual(len(self.events(resumed)), 1)
        recover_commits(resumed.memory, resumed.store, resumed.identity)
        self.assertEqual(len(self.events(resumed)), 1)

    def test_branch_unprepared_artifacts_retry_without_overwrite_or_duplicate_event(self):
        runtime = make_runtime(self.directory())
        manifest = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        session = TrainingSession(runtime, "learning", {"steps": 12}, manifest)
        runtime.training_session = session
        with patch("luban.evolution._generation", side_effect=self.generation), patch.object(
                session, "prepare_commit", side_effect=RuntimeError("killed before prepare")):
            with self.assertRaisesRegex(RuntimeError, "before prepare"):
                runtime.learn()
        report_path = runtime.store / ("learning-" + session.id + ".json")
        self.assertTrue(report_path.exists())
        before = report_path.read_bytes()
        self.assertFalse(runtime.pointer.exists())
        self.assertEqual(self.events(runtime), [])
        session.pause()
        runtime.training_session = TrainingSession(runtime, "learning", {"steps": 12}, manifest, run_id=session.id)
        # Simulate death after tensor publication but before paired metadata.
        metadata = runtime.store / "layer-23" / ("learned-" + session.id + ".json")
        metadata.unlink()
        runtime.memory.samples = Mock(side_effect=AssertionError("resume cannot reread live DB"))
        with patch("luban.evolution._generation", side_effect=self.generation):
            report = runtime.learn()
        self.assertTrue(report["promoted"])
        self.assertEqual(report_path.read_bytes(), before)
        self.assertTrue(metadata.exists())
        self.assertEqual(len(self.events(runtime)), 1)

    def test_deleted_accepted_branch_gold_uses_on_baseline_and_rejects_forgetting(self):
        from luban.evolution import _generation
        from luban.retention import protected_validation
        from tests.test_retention import BranchModel
        runtime = make_runtime(self.directory())
        runtime.branch = DynamicMLP(nn.Linear(4, 4), 4, 32, runtime.store / "layer-23",
                                    runtime.identity + ":layer-23", 1000000)
        runtime.model = BranchModel(runtime.branch)
        with torch.no_grad():
            runtime.branch.base.bias.zero_()
            runtime.branch.base.bias[0] = .01
        runtime.branch.expert = NeuralExpert(4, 32)
        with torch.no_grad():
            runtime.branch.expert.gate.weight.fill_(.5)
            runtime.branch.expert.up.weight.fill_(.5)
            runtime.branch.expert.down.weight.zero_()
            runtime.branch.expert.down.weight[1].fill_(.0001)
        runtime.branch.expert_id = "learned-old"
        runtime._persist_expert("learned-old")
        runtime._tokens = lambda messages, thinking, output_reserve=0: torch.tensor([[0]])
        runtime._sample_tokens = lambda row: (torch.tensor([[0, 2]]), torch.tensor([[-100, 2]]))
        runtime.tokenizer = SimpleNamespace(eos_token_id=3, decode=lambda tokens, **kwargs:
                                            {0: "wrong", 1: "identity", 2: "newknowledge"}[int(tokens[0])])
        old = {**sample("old", "validation"), "answer": "identity"}
        old_id = runtime.memory.add_sample(old["prompt"], old["answer"], approved=True, training_allowed=True,
            purpose="validation", provenance=json.loads(old["provenance"]))
        old["id"] = old_id
        self.assertTrue(_generation(runtime, [old])[0]["passed"])
        original_expert = runtime.branch.expert
        runtime.branch.expert = None
        self.assertFalse(_generation(runtime, [old])[0]["passed"])
        runtime.branch.expert = original_expert
        old_session = TrainingSession(runtime, "learning", {"steps": 12}, {"train": [], "valid": [old], "protection": []})
        weights, metadata = runtime.branch._paths(runtime.branch.expert_id)
        old_session.prepare_commit({"promoted": True}, {}, [str(weights.relative_to(runtime.store)),
            str(metadata.relative_to(runtime.store))], event_kind="learning")
        old_session.finish_commit()
        with runtime.memory.lock, runtime.memory.connect() as db:
            db.execute("DELETE FROM samples WHERE id=?", (old_id,))
        self.assertEqual(runtime.memory.samples(), [])
        self.assertEqual(protected_validation(runtime), [])
        protected = protected_validation(runtime, include_learning=True)
        self.assertEqual([row["id"] for row in protected], [old_id])
        self.assertEqual(json.loads(protected[0]["provenance"])["retention_scope"], "accepted_branch")
        new_train = {**sample("new"), "answer": "newknowledge"}
        new_valid = {**sample("new-valid", "validation"), "answer": "newknowledge"}
        runtime.training_session = TrainingSession(runtime, "learning", {"steps": 16},
            {"train": [new_train], "valid": [new_valid], "protection": protected})
        runtime._loss = Mock(side_effect=[2., None] + [1., None] * 3)
        result = runtime.learn()
        self.assertFalse(result["promoted"])
        self.assertTrue(result["core_retention"]["baseline_generation"][0]["passed"])
        self.assertTrue(result["core_retention"]["baseline_generation"][0]["branch_enabled"])
        self.assertTrue(any(not row["core_retention_generation"][0]["passed"] for row in result["candidates"]))
        self.assertTrue(_generation(runtime, protected)[0]["passed"])
        self.assertEqual(runtime.branch.expert_id, "learned-old")
        self.assertEqual(len(self.events(runtime)), 2)  # Accepted old + rejected new, one adoption only.
        runtime.training_session = TrainingSession(runtime, "learning", {"steps": 16},
            {"train": [new_train], "valid": [{**old, "answer": "conflicting gold"}], "protection": protected})
        with self.assertRaisesRegex(ValueError, "冲突"):
            runtime.learn()
        self.assertTrue(_generation(runtime, protected)[0]["passed"])
        self.assertEqual(len(self.events(runtime)), 2)

    def test_router_resume_reuses_features_and_adam_exact(self):
        manifest = {"splits": route_splits()}
        full, resumed = make_runtime(self.directory()), make_runtime(self.directory())
        full.training_session = TrainingSession(full, "router", {"steps": 160}, manifest, run_id="333333333333")
        expected = full.calibrate_router()
        self.assertTrue(full._tokens.call_args_list)
        self.assertTrue(all(call.kwargs == {"shallow": True} for call in full._tokens.call_args_list))
        old = copy.deepcopy(resumed.router.state_dict())
        session = TrainingSession(resumed, "router", {"steps": 160}, manifest, run_id="444444444444")
        resumed.training_session = session
        original_save = session.save
        def stop_after_forty(payload, progress):
            original_save(payload, progress)
            if progress["step"] == 40:
                session.pause_requested = True
        session.save = stop_after_forty
        with self.assertRaises(TrainingPaused):
            resumed.calibrate_router()
        self.assertTrue(all(torch.equal(value, resumed.router.state_dict()[name]) for name, value in old.items()))
        self.assertEqual(self.events(resumed), [])
        shallow_calls = resumed.shallow.call_count
        self.assertEqual(shallow_calls, 18)
        session.pause()
        resumed.training_session = TrainingSession(resumed, "router", {"steps": 160}, manifest, run_id=session.id)
        result = resumed.calibrate_router()
        self.assertEqual(resumed.shallow.call_count, shallow_calls)
        self.assertTrue(expected["calibrated"] and result["calibrated"])
        self.assertEqual(result["records_sha256"], resumed.training_session.binding["records_sha256"])
        for name, tensor in full.router.state_dict().items():
            self.assertTrue(torch.equal(tensor, resumed.router.state_dict()[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in resumed.router.parameters()))
        self.assertEqual(len(self.events(resumed)), 1)
        self.assertTrue((resumed.store / "active-router.json").exists())
        self.assertFalse((resumed.store / "router.safetensors").exists())
        old_projection = copy.deepcopy(resumed.router.state_dict())
        resumed.router = FastDecisionHead(4)
        resumed._load_router()
        self.assertTrue(all(torch.equal(value, resumed.router.state_dict()[name]) for name, value in old_projection.items()))
        recover_commits(resumed.memory, resumed.store, resumed.identity)
        self.assertEqual(len(self.events(resumed)), 1)

    def test_router_feature_pause_before_adam_resumes_same_session_exactly(self):
        manifest = {"splits": route_splits()}
        full = make_runtime(self.directory())
        full.training_session = TrainingSession(full, "router", {"steps": 160}, manifest)
        expected = full.calibrate_router()
        for pause_after in (1, sum(map(len, manifest["splits"].values()))):
            with self.subTest(pause_after=pause_after):
                runtime = make_runtime(self.directory())
                previous = copy.deepcopy(runtime.router.state_dict())
                previous_meta = copy.deepcopy(runtime.router_meta)
                session = TrainingSession(runtime, "router", {"steps": 160}, manifest)
                runtime.training_session = session
                shallow = runtime.shallow.side_effect
                calls = []
                def pause_feature(tokens):
                    calls.append(tokens)
                    value = shallow(tokens)
                    if len(calls) == pause_after:
                        session.pause_requested = True
                    return value
                runtime.shallow.side_effect = pause_feature
                with patch("luban.runtime.torch.optim.Adam") as adam:
                    with self.assertRaises(TrainingPaused):
                        runtime.calibrate_router()
                    adam.assert_not_called()
                self.assertEqual(len(calls), pause_after)
                self.assertIsNone(session.load())
                self.assertEqual(runtime.state, "ready")
                self.assertEqual(runtime.job["state"], "paused")
                self.assertEqual(runtime.router_meta, previous_meta)
                self.assertTrue(all(torch.equal(value, runtime.router.state_dict()[name])
                                    for name, value in previous.items()))
                self.assertFalse((runtime.store / "active-router.json").exists())
                self.assertEqual(self.events(runtime), [])
                session.pause()  # Same worker lifecycle as AutomaticLearning._work.
                runtime.training_session = TrainingSession(runtime, "router", {"steps": 160}, manifest, run_id=session.id)
                runtime.shallow.side_effect = shallow
                result = runtime.calibrate_router()
                self.assertTrue(expected["calibrated"] and result["calibrated"])
                self.assertEqual(runtime.training_session.id, session.id)
                self.assertTrue(all(torch.equal(value, runtime.router.state_dict()[name])
                                    for name, value in full.router.state_dict().items()))
                self.assertEqual(len(self.events(runtime)), 1)

    def test_loss_observation_honors_pause_and_cancel_between_samples(self):
        for interrupted in ("pause", "cancel"):
            for stop_after in (0, 1, 3):
                with self.subTest(interrupted=interrupted, stop_after=stop_after):
                    runtime = make_runtime(self.directory())
                    runtime.training_session = SimpleNamespace(pause_requested=False)
                    runtime.job = {}
                    def interrupt():
                        if interrupted == "pause":
                            runtime.training_session.pause_requested = True
                        else:
                            runtime.job["cancel_requested"] = True
                    calls = []
                    def forward(*args, **kwargs):
                        calls.append(args)
                        if len(calls) == stop_after:
                            interrupt()
                        return SimpleNamespace(loss=torch.tensor(1.))
                    runtime.model = Mock(side_effect=forward)
                    if stop_after == 0:
                        interrupt()
                    with self.assertRaises(TrainingPaused if interrupted == "pause" else RuntimeError):
                        Runtime._loss(runtime, [sample(i) for i in range(3)])
                    self.assertEqual(len(calls), stop_after)
                    self.assertEqual(self.events(runtime), [])

    def test_router_pause_and_rejection_release_unused_hf_route_but_keep_native(self):
        from luban.calibration import evaluate_router
        for outcome in ("pause", "rejected", "accepted"):
            with self.subTest(outcome=outcome):
                runtime = make_runtime(self.directory())
                native = SimpleNamespace(process=Mock(poll=Mock(return_value=None)))
                runtime.native_engine = native
                pager = SimpleNamespace(loaded=False, depth_count=0, evict_resident=Mock())
                runtime.cold_backbone = pager
                session = TrainingSession(runtime, "router", {"steps": 160}, {"splits": route_splits()})
                runtime.training_session = session
                shallow = runtime.shallow.side_effect
                def feature(tokens):
                    if outcome == "pause":
                        session.pause_requested = True
                    return shallow(tokens)
                runtime.shallow.side_effect = feature
                def evaluate(*args):
                    result = evaluate_router(*args)
                    result["calibrated"] = outcome == "accepted"
                    return result
                with patch("luban.calibration.evaluate_router", side_effect=evaluate):
                    if outcome == "pause":
                        with self.assertRaises(TrainingPaused):
                            runtime.calibrate_router()
                    else:
                        runtime.calibrate_router()
                self.assertIs(runtime.native_engine, native)
                self.assertEqual(runtime.state, "ready")
                self.assertTrue(all(parameter.grad is None for parameter in runtime.router.parameters()))
                self.assertEqual(pager.evict_resident.call_count, 0 if outcome == "accepted" else 1)

    def test_corrupt_active_router_never_falls_back_to_legacy(self):
        runtime = make_runtime(self.directory())
        runtime.training_session = TrainingSession(runtime, "router", {"steps": 160}, {"splits": route_splits()})
        runtime.calibrate_router()
        save_file(runtime.router.state_dict(), str(runtime.store / "router.safetensors"))
        (runtime.store / "router.json").write_text(json.dumps({"identity": runtime.identity, "core_version": None,
            "sha256": hashlib.sha256((runtime.store / "router.safetensors").read_bytes()).hexdigest()}))
        pointer = json.loads((runtime.store / "active-router.json").read_text())
        (runtime.store / (pointer["id"] + ".safetensors")).write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "校验"):
            runtime._load_router()

    def test_unverified_frozen_branch_manifest_is_rejected_before_training(self):
        runtime = make_runtime(self.directory())
        unverified = {**sample(1), "source": "teacher"}
        runtime.training_session = TrainingSession(runtime, "learning", {"steps": 12},
            {"train": [unverified], "valid": [sample(2, "validation")], "protection": []})
        with self.assertRaisesRegex(ValueError, "未核验"):
            runtime.learn()
        self.assertFalse(runtime._sample_tokens.called)
        self.assertEqual(self.events(runtime), [])

    def test_post_commit_errors_cannot_rollback_branch_or_router(self):
        for kind in ("learning", "router"):
            with self.subTest(kind=kind):
                runtime = make_runtime(self.directory())
                records = ({"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
                           if kind == "learning" else {"splits": route_splits()})
                session = TrainingSession(runtime, kind, {"steps": 12 if kind == "learning" else 160}, records)
                runtime.training_session = session
                finish = session.finish_commit
                def after_commit():
                    finish()
                    raise OSError("post-commit status failure")
                with patch.object(session, "finish_commit", side_effect=after_commit), patch(
                        "luban.evolution._generation", side_effect=self.generation):
                    result = runtime.learn() if kind == "learning" else runtime.calibrate_router()
                self.assertTrue(result.get("promoted", result.get("calibrated")))
                self.assertEqual(session.body["state"], "committed")
                self.assertEqual(runtime.job["state"], "complete")
                self.assertEqual(len(self.events(runtime)), 1)
                pointer = runtime.pointer if kind == "learning" else runtime.store / "active-router.json"
                self.assertTrue(pointer.exists())
                if kind == "learning":
                    self.assertEqual(runtime.branch.expert_id, result["active_expert"])
                else:
                    self.assertEqual(runtime.router_meta, result)
                recover_commits(runtime.memory, runtime.store, runtime.identity)
                self.assertEqual(len(self.events(runtime)), 1)

    def test_research_commit_gold_requires_outer_product_acceptance(self):
        from luban.retention import protected_validation
        runtime = make_runtime(self.directory())
        records = {"train": [sample(1)], "valid": [sample(2, "validation")], "protection": []}
        session = TrainingSession(runtime, "learning", {"steps": 12}, records)
        session.body["scope"] = "research"
        session._write("queued")
        runtime.training_session = session
        with patch("luban.evolution._generation", side_effect=self.generation):
            self.assertTrue(runtime.learn()["promoted"])
        self.assertEqual(protected_validation(runtime, include_learning=True), [])
        session.body["product_accepted"] = True
        session._write("committed")
        protected = protected_validation(runtime, include_learning=True)
        self.assertEqual([row["id"] for row in protected], ["2"])
        self.assertEqual(session.body["records"], records)

    def test_router_commit_failure_restores_old_live_head_and_can_recover(self):
        runtime = make_runtime(self.directory())
        old = copy.deepcopy(runtime.router.state_dict())
        session = TrainingSession(runtime, "router", {"steps": 160}, {"splits": route_splits()})
        runtime.training_session = session
        with patch.object(session, "finish_commit", side_effect=RuntimeError("crash before pointer")):
            with self.assertRaisesRegex(RuntimeError, "crash"):
                runtime.calibrate_router()
        self.assertFalse((runtime.store / "active-router.json").exists())
        self.assertTrue(all(torch.equal(value, runtime.router.state_dict()[name]) for name, value in old.items()))
        self.assertEqual(self.events(runtime), [])
        recover_commits(runtime.memory, runtime.store, runtime.identity)
        self.assertTrue((runtime.store / "active-router.json").exists())
        self.assertEqual(len(self.events(runtime)), 1)
        runtime._load_router()
        self.assertTrue(runtime.router_meta["calibrated"])


if __name__ == "__main__":
    unittest.main()
