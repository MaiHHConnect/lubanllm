"""Core updates on a tiny real graph; no model downloads or teacher calls."""
import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from luban.dynamic import DynamicMLP
from luban.evolution import (consolidate, consolidation_samples, core_status, load_active_core,
                             restore_core, _answer_log_probs, _teacher_replay, _capture_replay_subspace,
                             _project_core_delta, _constraint_residual, _replay_logit_drift,
                             _retention_loss_guard)
from luban.memory import Memory
from luban.settings import private_json
from luban.retention import read_core_retention


class TinyModel(nn.Module):
    def __init__(self, store):
        super().__init__()
        base = nn.Module()
        base.down_proj = nn.Linear(2, 2, bias=False)
        base.forward = lambda x: base.down_proj(x)
        self.model = nn.Module()
        layer = nn.Module()
        layer.mlp = DynamicMLP(base, 2, 1, store, "fixture-base:layer-0", 6)
        # An active branch would override both desired answers. Training and
        # generation must bypass it without discarding the original module.
        layer.mlp.expert = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            layer.mlp.expert.weight.fill_(100)
            base.down_proj.weight.copy_(torch.tensor([[0., 0.], [.1, 1.]]))
        self.model.layers = nn.ModuleList([layer])
        self.untouched = nn.Parameter(torch.tensor([7.]))
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.generated = []
        self.fail_on_generation = None
        self.cancel_on_generation = None
        self.replay_corruption = False
        self.require_grad_(False)

    def require_grad_(self, value):
        return self.requires_grad_(value)

    def forward(self, ids, labels=None, **kwargs):
        features = F.one_hot(ids, num_classes=2).to(self.model.layers[-1].mlp.base.down_proj.weight.dtype)
        logits = self.model.layers[-1].mlp(features)
        loss = None
        if labels is not None:
            loss = (F.cross_entropy(logits[:, 0], labels[:, 0]) if ids.shape[1] == 1
                    else F.cross_entropy(logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1)))
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, ids, **kwargs):
        self.generated.append(self.model.layers[-1].mlp.expert is None)
        if self.fail_on_generation == len(self.generated):
            raise RuntimeError("actual generation failure")
        if self.cancel_on_generation == len(self.generated):
            self.runtime.job["cancel_requested"] = True
        if self.replay_corruption and len(self.generated) > 2 and int(ids[0, 0]) == 1:
            result = 0
        else:
            result = int(self.forward(ids).logits.argmax(-1)[0, 0])
        return torch.cat((ids, torch.tensor([[result, 3]])), dim=1)


class EvolutionTests(unittest.TestCase):
    def test_retention_loss_guard_is_per_record_and_bounded(self):
        baseline = [{'id': 'old', 'loss': 1.0}, {'id': 'new', 'loss': 2.0}]
        within = [{'id': 'old', 'loss': 1.0}, {'id': 'new', 'loss': 2.0 + 1e-7}]
        result = _retention_loss_guard(baseline, within)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['rows']), 2)
        beyond = [{'id': 'old', 'loss': 1.0}, {'id': 'new', 'loss': 2.001}]
        self.assertFalse(_retention_loss_guard(baseline, beyond)['passed'])

    def setUp(self):
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        store = root / "modules"
        model = TinyModel(store)
        self.runtime = SimpleNamespace(model=model, decoder=model.model, branch=model.model.layers[-1].mlp,
            lock=threading.RLock(), state="ready", job=None, identity="fixture-base", device="cpu", dtype=torch.float32,
            store=store, active_core=None, memory=Memory(root), router_meta={"calibrated": True},
            core_graph_verification={"passed": True, "identity": "fixture-base", "logits_max_abs_diff": 0.0, "scope": "tiny fixture"})
        model.runtime = self.runtime
        # Avoid registering the test Runtime as a child module.
        model.__dict__["runtime"] = self.runtime
        self.runtime._ready = lambda: None
        self.runtime._tokens = lambda messages, thinking, output_reserve=0: torch.tensor([[0 if "new" in messages[-1]["content"] else 1]])
        self.runtime._sample_tokens = lambda sample: (
            torch.tensor([[0 if _role(sample) == "new" else 1]]),
            torch.tensor([[0 if _role(sample) == "new" else 1]]))
        self.runtime.tokenizer = SimpleNamespace(eos_token_id=3,
            decode=lambda tokens, **kwargs: "newknowledge" if int(tokens[0]) == 0 else "oldability")
        self.original_expert = self.runtime.branch.expert
        self.original = self.runtime.branch.base.down_proj.weight.detach().clone()
        load_active_core(self.runtime)
        self.samples = {}
        for role in ("new", "replay"):
            for purpose in ("training", "validation"):
                sample_id = self.runtime.memory.add_sample(f"{role} {purpose}", "newknowledge" if role == "new" else "oldability",
                    source="human", purpose=purpose, approved=True, training_allowed=True,
                    provenance={"evolution_role": role, "split_family": role + "-" + purpose})
                self.samples[role + "_" + purpose] = sample_id

    def run_training(self, steps=16):
        # A toy two-logit model needs an appropriate optimizer scale. Production
        # stays at 1e-4; no fake generation or loss is substituted in this test.
        return consolidate(self.runtime, steps, learning_rate=.01)

    def test_production_qwen_core_update_cannot_disable_protection(self):
        self.runtime.model.config = SimpleNamespace(model_type="qwen3_5")
        with self.assertRaisesRegex(ValueError, "replay 子空间保护"):
            consolidate(self.runtime, 64, distillation_weight=0,
                        preserve_replay_subspace=False)

    def assert_restored(self):
        self.assertTrue(torch.equal(self.runtime.branch.base.down_proj.weight, self.original))
        self.assertIs(self.runtime.branch.expert, self.original_expert)
        self.assertEqual(self.runtime.state, "ready")
        self.assertIsNone(self.runtime.active_core)
        self.assertFalse((self.runtime.store / "active-core.json").exists())
        self.assertTrue(all(not parameter.requires_grad for parameter in self.runtime.model.parameters()))

    def edit_provenance(self, sample_id, update):
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            row = db.execute("SELECT provenance FROM samples WHERE id=?", (sample_id,)).fetchone()
            db.execute("UPDATE samples SET provenance=? WHERE id=?", (json.dumps(json.loads(row[0]) | update), sample_id))

    def test_real_core_update_off_branch_gate_and_reload(self):
        untouched = self.runtime.model.untouched.clone()
        branch_weights = self.original_expert.weight.clone()
        result = self.run_training()
        self.assertTrue(result["promoted"], result)
        self.assertTrue(all(result["gates"].values()))
        self.assertTrue(all(self.runtime.model.generated))
        self.assertFalse(result["baseline_new_generation"][0]["passed"])
        self.assertTrue(result["new_generation"][0]["passed"])
        self.assertTrue(result["replay_generation"][0]["passed"])
        self.assertIs(self.runtime.branch.expert, self.original_expert)
        self.assertTrue(torch.equal(branch_weights, self.original_expert.weight))
        self.assertTrue(torch.equal(untouched, self.runtime.model.untouched))
        self.assertFalse(torch.equal(self.original, self.runtime.branch.base.down_proj.weight))
        self.assertFalse(self.runtime.router_meta["calibrated"])
        weights = self.runtime.branch.base.down_proj.weight.detach().clone()
        with torch.no_grad():
            self.runtime.branch.base.down_proj.weight.copy_(self.original)
        metadata = load_active_core(self.runtime)
        self.assertEqual(metadata["version"], result["version"])
        self.assertTrue(torch.equal(weights, self.runtime.branch.base.down_proj.weight))
        self.assertTrue(torch.equal(self.runtime._base_core_weight, self.original))
        self.assertEqual(result["added_parameters"], 0)
        self.assertEqual(result["trained_parameters"], 4)
        self.assertEqual(set(result["split_ids"]), set(self.samples))
        self.assertIn("rss_bytes", result["sampled_memory_maxima"])

    def test_published_core_binds_complete_retention_before_next_evolution(self):
        expected = consolidation_samples(self.runtime)
        result = self.run_training()
        snapshot = read_core_retention(self.runtime)
        self.assertEqual(snapshot["groups"], expected)
        path = self.runtime.store / (result["version"] + ".retention.json")
        self.assertEqual(result["checkpoint"]["retention_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        for sample_id in self.samples.values():
            self.runtime.memory.delete_sample(sample_id)
        material = consolidation_samples(self.runtime)
        self.assertEqual(len(material["replay_training"]), 2)
        self.assertEqual(len(material["replay_validation"]), 2)
        self.assertFalse(core_status(self.runtime)["ready"])
        self.assertIn("已巩固", core_status(self.runtime)["blocked_reason"])

    def test_missing_bound_retention_rejects_load_legacy_load_remains_compatible(self):
        result = self.run_training()
        path = self.runtime.store / (result["version"] + ".retention.json")
        path.unlink()
        current = self.runtime.branch.base.down_proj.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "缺少完整保护"):
            load_active_core(self.runtime)
        self.assertTrue(torch.equal(current, self.runtime.branch.base.down_proj.weight))
        metadata_path = self.runtime.store / (result["version"] + ".json")
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("retention_sha256")
        metadata_path.write_text(json.dumps(metadata))  # Isolated fixture for the legacy schema.
        self.assertEqual(load_active_core(self.runtime)["version"], result["version"])
        readiness = core_status(self.runtime)
        self.assertFalse(readiness["ready"])
        self.assertIn("缺少完整保护", readiness["blocked_reason"])

    def test_legacy_promoted_core_falls_back_to_original_until_current_proof_exists(self):
        result = self.run_training()
        report_path = self.runtime.store / ("evolution-" + result["id"] + ".json")
        report = json.loads(report_path.read_text())
        report.pop("core_retention_loss", None)
        report["gates"].pop("protected_core_retention_loss", None)
        report_path.write_text(json.dumps(report))
        with torch.no_grad():
            self.runtime.branch.base.down_proj.weight.copy_(self.original)
        loaded = load_active_core(self.runtime)
        self.assertIsNone(loaded)
        self.assertIsNone(self.runtime.active_core)
        self.assertTrue(torch.equal(self.runtime.branch.base.down_proj.weight, self.original))
        self.assertEqual(self.runtime.core_requires_review["version"], result["version"])
        self.assertIn("当前逐条保护", self.runtime.core_requires_review["reason"])

    def test_post_training_replay_failure_rejects_real_weight_update(self):
        self.runtime.model.replay_corruption = True
        result = self.run_training()
        self.assertFalse(result["promoted"])
        self.assertTrue(result["gates"]["heldout_new_generation"])
        self.assertFalse(result["gates"]["protected_replay_generation"])
        self.assert_restored()
        self.assertTrue((self.runtime.store / ("evolution-" + result["id"] + ".json")).exists())

    def test_baseline_replay_failure_prevents_training(self):
        self.runtime.tokenizer.decode = lambda tokens, **kwargs: "wrong"
        result = self.run_training()
        self.assertFalse(result["promoted"])
        self.assertEqual(self.runtime.job["step"], 0)
        self.assertNotIn("gates", result)
        self.assert_restored()

    def test_generation_failure_after_actual_training_rolls_back(self):
        self.runtime.model.fail_on_generation = 3
        with self.assertRaisesRegex(RuntimeError, "actual generation failure"):
            self.run_training()
        self.assertEqual(self.runtime.job["step"], 16)
        self.assert_restored()
        report = self.runtime.job["result"]
        self.assertFalse(report["promoted"])
        self.assertIn("error", report)

    def test_cancel_during_generation_is_never_success(self):
        self.runtime.model.cancel_on_generation = 3
        with self.assertRaisesRegex(RuntimeError, "已取消"):
            self.run_training()
        self.assertEqual(self.runtime.job["step"], 16)
        self.assert_restored()

    def test_admission_requires_verified_nonhuman_not_structural_teacher_pass(self):
        sample_id = self.samples["new_training"]
        with self.runtime.memory.lock, self.runtime.memory.connect() as db:
            db.execute("UPDATE samples SET source='teacher' WHERE id=?", (sample_id,))
        self.edit_provenance(sample_id, {"verification_scope": "structural checks only", "teacher_correct": True})
        self.assertFalse(core_status(self.runtime)["ready"])
        self.edit_provenance(sample_id, {"program_verified": True})
        self.assertFalse(core_status(self.runtime)["ready"])
        self.edit_provenance(sample_id, {"verification_evidence": "fixture program checked target"})
        self.assertTrue(core_status(self.runtime)["ready"])

    def test_failed_original_replay_validation_is_excluded_without_changing_gold(self):
        sample_id = self.samples["replay_validation"]
        self.edit_provenance(sample_id, {"baseline_preservation_eligible": False})
        self.assertEqual(consolidation_samples(self.runtime)["replay_validation"], [])
        self.assertFalse(core_status(self.runtime)["ready"])
        original = next(row for row in self.runtime.memory.samples() if row["id"] == sample_id)
        self.assertEqual(original["answer"], "oldability")
        self.assertEqual(original["approved"], 1)
        self.edit_provenance(sample_id, {"baseline_preservation_eligible": True})
        self.assertEqual([row["id"] for row in consolidation_samples(self.runtime)["replay_validation"]], [sample_id])
        # This field defines protected validation eligibility, not a new-task
        # or training filter. Missing fields remain backward compatible.
        self.edit_provenance(self.samples["new_validation"], {"baseline_preservation_eligible": False})
        self.edit_provenance(self.samples["replay_training"], {"baseline_preservation_eligible": False})
        self.assertTrue(core_status(self.runtime)["ready"])

    def test_context_and_family_leaks_fail_closed(self):
        validation_id = self.samples["new_validation"]
        self.edit_provenance(validation_id, {"messages": [{"role": "user", "content": "new  training"}]})
        with self.assertRaisesRegex(ValueError, "相同上下文"):
            consolidation_samples(self.runtime)
        self.edit_provenance(validation_id, {"messages": [], "split_family": "new-training"})
        with self.assertRaisesRegex(ValueError, "相同模板家族"):
            consolidation_samples(self.runtime)
        self.assert_restored()

    def test_missing_graph_proof_and_invalid_steps_are_rejected_without_mutation(self):
        self.runtime.core_graph_verification = {"passed": False}
        with self.assertRaisesRegex(ValueError, "等价验证"):
            self.run_training()
        for steps in (True, 0, 4097, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                consolidate(self.runtime, steps)
        self.assert_restored()

    def test_atomic_publication_failure_keeps_old_weights_and_active_pointer(self):
        original_json = private_json
        def fail_pointer(path, value):
            if path.name == "active-core.json":
                raise OSError("fixture atomic publication failed")
            original_json(path, value)
        with patch("luban.evolution.private_json", side_effect=fail_pointer):
            with self.assertRaisesRegex(OSError, "atomic publication"):
                self.run_training()
        self.assert_restored()
        self.assertFalse(self.runtime.job["result"]["promoted"])

    def test_record_failure_after_pointer_publication_rolls_back_the_pointer(self):
        with patch.object(self.runtime.memory, "event", side_effect=OSError("evidence record failure")):
            with self.assertRaisesRegex(OSError, "evidence record failure"):
                self.run_training()
        self.assert_restored()
        self.assertFalse(self.runtime.job["result"]["promoted"])

    def test_already_mastered_knowledge_does_not_increment_core_versions(self):
        first = self.run_training()
        before = self.runtime.branch.base.down_proj.weight.detach().clone()
        pointer = (self.runtime.store / "active-core.json").read_bytes()
        readiness = core_status(self.runtime)
        self.assertFalse(readiness["ready"])
        self.assertIn("已巩固", readiness["blocked_reason"])
        with self.assertRaisesRegex(ValueError, "已巩固"):
            self.run_training()
        self.assertEqual(self.runtime.active_core, first["version"])
        self.assertEqual((self.runtime.store / "active-core.json").read_bytes(), pointer)
        self.assertTrue(torch.equal(before, self.runtime.branch.base.down_proj.weight))

    def test_load_rejects_wrong_base_bad_hash_and_tensor_shape_before_mutating(self):
        result = self.run_training()
        version = result["version"]
        weight_file = self.runtime.store / (version + ".safetensors")
        before = self.runtime.branch.base.down_proj.weight.detach().clone()
        self.runtime.identity = "different-base"
        with self.assertRaisesRegex(ValueError, "不兼容"):
            load_active_core(self.runtime)
        self.runtime.identity = "fixture-base"
        original_bytes = weight_file.read_bytes()
        weight_file.write_bytes(original_bytes + b"tampered")
        with self.assertRaisesRegex(ValueError, "权重校验失败"):
            load_active_core(self.runtime)
        weight_file.write_bytes(original_bytes)
        from safetensors.torch import save_file
        name = result["parameter"]
        save_file({name: torch.ones(3, 2)}, str(weight_file))
        metadata_path = self.runtime.store / (version + ".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["sha256"] = hashlib.sha256(weight_file.read_bytes()).hexdigest()
        private_json(metadata_path, metadata)
        private_json(self.runtime.store / "active-core.json", {"identity": "fixture-base", "version": version, "sha256": metadata["sha256"]})
        with self.assertRaisesRegex(ValueError, "尺寸不兼容"):
            load_active_core(self.runtime)
        self.assertTrue(torch.equal(before, self.runtime.branch.base.down_proj.weight))

    def test_restore_core_original_and_saved_version_and_failed_restore(self):
        result = self.run_training()
        trained = self.runtime.branch.base.down_proj.weight.detach().clone()
        restore_core(self.runtime)
        self.assert_restored()
        restore_core(self.runtime, result["version"])
        self.assertTrue(torch.equal(trained, self.runtime.branch.base.down_proj.weight))
        pointer = (self.runtime.store / "active-core.json").read_bytes()
        with patch("luban.evolution.private_json", side_effect=OSError("restore pointer failure")):
            with self.assertRaisesRegex(OSError, "restore pointer failure"):
                restore_core(self.runtime, result["version"])
        self.assertTrue(torch.equal(trained, self.runtime.branch.base.down_proj.weight))
        self.assertEqual((self.runtime.store / "active-core.json").read_bytes(), pointer)
        for invalid in ("../escape", "core-000000000000.json", 1):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                restore_core(self.runtime, invalid)

    def test_checkpoint_symlinks_and_wrong_graph_identity_fail_closed(self):
        self.runtime.core_graph_verification["identity"] = "other-base"
        self.assertFalse(core_status(self.runtime)["ready"])
        self.runtime.core_graph_verification = None
        self.assertFalse(core_status(self.runtime)["ready"])
        outside = self.runtime.store / "outside.json"
        outside.write_text("{}")
        (self.runtime.store / "active-core.json").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            load_active_core(self.runtime)
        self.assertEqual(outside.read_text(), "{}")

    def sequence_training_tokens(self):
        def tokens(sample):
            target = 0 if _role(sample) == "new" else 1
            return torch.tensor([[target, target]]), torch.tensor([[-100, target]])
        self.runtime._sample_tokens = tokens

    def test_distillation_preserves_original_replay_distribution_on_real_core(self):
        self.sequence_training_tokens()
        original_probability = F.softmax(self.original[:, 1], dim=0)
        unregularized = consolidate(self.runtime, 64, learning_rate=.01)
        self.assertTrue(unregularized["promoted"])
        raw_probability = F.softmax(self.runtime.branch.base.down_proj.weight.detach()[:, 1], dim=0)
        raw_drift = (raw_probability - original_probability).abs().sum()
        restore_core(self.runtime)
        preserved = consolidate(self.runtime, 64, learning_rate=.01, distillation_weight=50)
        self.assertTrue(preserved["promoted"], preserved)
        preserved_probability = F.softmax(self.runtime.branch.base.down_proj.weight.detach()[:, 1], dim=0)
        preserved_drift = (preserved_probability - original_probability).abs().sum()
        self.assertLess(preserved_drift, raw_drift / 2)
        self.assertEqual(preserved["distillation"]["weight"], 50)
        self.assertEqual(preserved["distillation"]["target_tokens"], 64)
        self.assertEqual(preserved["distillation"]["steps"], 64)
        self.assertGreater(preserved["distillation"]["kl_mean"], 0)
        self.assertTrue(all(self.runtime.model.generated))
        self.assertTrue(all(preserved["gates"].values()))

    def test_reference_teacher_is_original_detached_and_restores_candidate_before_backward(self):
        parameter = self.runtime.branch.base.down_proj.weight
        self.runtime.branch.expert = None
        parameter.requires_grad_(True)
        candidate = self.original + torch.tensor([[.3, .4], [-.3, -.4]])
        with torch.no_grad():
            parameter.copy_(candidate)
        ids, labels = torch.tensor([[1, 1]]), torch.tensor([[-100, 1]])
        teacher = _teacher_replay(self.runtime, parameter, self.original, candidate, ids, labels)
        expected = F.log_softmax(self.original[:, 1], dim=0).reshape(1, 2)
        torch.testing.assert_close(teacher, expected, atol=0, rtol=0)
        self.assertFalse(teacher.requires_grad)
        torch.testing.assert_close(parameter, candidate, atol=0, rtol=0)
        student = _answer_log_probs(self.runtime.model(ids).logits, labels)
        F.kl_div(student, teacher, log_target=True, reduction="batchmean").backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum(), 0)
        self.assertIsNone(self.runtime.model.untouched.grad)

    def test_teacher_exception_restores_candidate_then_consolidation_restores_old_core(self):
        parameter = self.runtime.branch.base.down_proj.weight
        self.runtime.branch.expert = None
        candidate = self.original + torch.tensor([[.3, .4], [-.3, -.4]])
        with torch.no_grad():
            parameter.copy_(candidate)
        with patch.object(self.runtime.model, "forward", side_effect=RuntimeError("reference failed")):
            with self.assertRaisesRegex(RuntimeError, "reference failed"):
                _teacher_replay(self.runtime, parameter, self.original, candidate,
                                torch.tensor([[1, 1]]), torch.tensor([[-100, 1]]))
        torch.testing.assert_close(parameter, candidate, atol=0, rtol=0)
        with torch.no_grad():
            parameter.copy_(self.original)
        self.runtime.branch.expert = self.original_expert
        self.sequence_training_tokens()
        original_forward = self.runtime.model.forward
        teacher_calls = 0
        def fail_second_reference(ids, labels=None, **kwargs):
            nonlocal teacher_calls
            if not torch.is_grad_enabled() and labels is None and ids.shape[1] == 2:
                teacher_calls += 1
                if teacher_calls == 2:
                    torch.testing.assert_close(parameter, self.original, atol=0, rtol=0)
                    raise RuntimeError("second reference failed")
            return original_forward(ids, labels=labels, **kwargs)
        with patch.object(self.runtime.model, "forward", side_effect=fail_second_reference):
            with self.assertRaisesRegex(RuntimeError, "second reference failed"):
                consolidate(self.runtime, 16, learning_rate=.01, distillation_weight=2)
        self.assertEqual(self.runtime.job["step"], 1)
        self.assertEqual(self.runtime.job["result"]["distillation"]["steps"], 1)
        self.assert_restored()

    def test_distillation_masks_shifted_prompt_positions_and_averages_per_token(self):
        logits = torch.tensor([[[4., 0., -2.], [1., 2., 3.], [3., 2., 1.], [-1., 0., 1.]]], dtype=torch.bfloat16)
        labels = torch.tensor([[-100, -100, 2, 0]])
        result = _answer_log_probs(logits, labels)
        expected = F.log_softmax(logits[0, 1:3].float(), dim=-1)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.shape, (2, 3))
        teacher = F.log_softmax(torch.tensor([[2., 2., 2.], [0., 1., 2.]]), dim=-1)
        kl = F.kl_div(result, teacher, log_target=True, reduction="batchmean")
        manual = (teacher.exp() * (teacher - expected)).sum(dim=-1).mean()
        torch.testing.assert_close(kl, manual, atol=1e-7, rtol=0)
        with self.assertRaisesRegex(RuntimeError, "缺少回答词元"):
            _answer_log_probs(logits, torch.full_like(labels, -100))

    def test_distillation_argument_validation_and_route_only_admission(self):
        for weight in (True, -1, 101, float("nan"), float("inf"), "2"):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                consolidate(self.runtime, distillation_weight=weight)
        self.edit_provenance(self.samples["new_training"], {"route_only": True})
        self.assertFalse(core_status(self.runtime)["ready"])
        self.assert_restored()

    def test_generation_forbidden_fragments_reject_without_rewriting_output(self):
        self.edit_provenance(self.samples["replay_validation"], {
            "generation_check": {"type": "contains", "expected": "oldability", "forbidden": ["BRAND"]}})
        self.runtime.tokenizer.decode = lambda tokens, **kwargs: "newknowledge" if int(tokens[0]) == 0 else "oldability BRAND"
        report = self.run_training()
        self.assertFalse(report["promoted"])
        generated = report["baseline_replay_generation"][0]
        self.assertEqual(generated["answer"], "oldability BRAND")
        self.assertEqual(generated["forbidden_found"], ["BRAND"])
        self.assertFalse(generated["passed"])
        self.assert_restored()
        for forbidden in ("BRAND", [""], [" "], [1], ["x"] * 33, ["x" * 513]):
            self.edit_provenance(self.samples["replay_validation"], {
                "generation_check": {"type": "contains", "expected": "oldability", "forbidden": forbidden}})
            with self.subTest(forbidden=forbidden), self.assertRaisesRegex(ValueError, "禁用片段"):
                consolidation_samples(self.runtime)

    def test_replay_subspace_captures_training_only_and_causal_predictor_inputs(self):
        self.sequence_training_tokens()
        groups = consolidation_samples(self.runtime)
        parameter = self.runtime.branch.base.down_proj.weight
        self.runtime.branch.expert = None
        observed = []
        sample_tokens = self.runtime._sample_tokens
        def observe(sample):
            observed.append((sample["id"], sample["purpose"]))
            return sample_tokens(sample)
        self.runtime._sample_tokens = observe
        hooks = len(self.runtime.branch.base.down_proj._forward_pre_hooks)
        basis, report = _capture_replay_subspace(self.runtime, parameter, groups["replay_training"])
        self.assertEqual(observed, [(self.samples["replay_training"], "training")])
        self.assertEqual(report["rank"], 1)
        self.assertEqual(report["captured_positions"], 1)
        self.assertEqual(report["training_ids"], [self.samples["replay_training"]])
        torch.testing.assert_close(basis @ basis.T, torch.tensor([[0., 0.], [0., 1.]]), rtol=0, atol=0)
        self.assertEqual(len(self.runtime.branch.base.down_proj._forward_pre_hooks), hooks)
        with self.assertRaisesRegex(ValueError, "不能读取检验集"):
            _capture_replay_subspace(self.runtime, parameter, groups["replay_validation"])
        self.assertEqual(len(self.runtime.branch.base.down_proj._forward_pre_hooks), hooks)

    def test_cumulative_projection_constrains_adam_momentum_and_reports_bf16_rounding(self):
        torch.manual_seed(7)
        parameter = torch.tensor([[.2, -.7, .4], [.8, .4, -.3]], dtype=torch.bfloat16)
        reference = parameter.float().clone()
        master = nn.Parameter(reference.clone())
        basis = torch.tensor([[1.], [1.], [0.]]) / 2 ** .5
        optimizer = torch.optim.Adam([master], lr=.02)
        unconstrained_drift = []
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            (master * torch.rand_like(master)).sum().backward()
            optimizer.step()
            unconstrained_drift.append(float(((master.detach() - reference) @ basis).norm()))
            _project_core_delta(master, reference, basis)
            with torch.no_grad():
                parameter.copy_(master)
            self.assertLess(float(((master.detach() - reference) @ basis).abs().max()), 1e-6)
        self.assertTrue(any(value > .01 for value in unconstrained_drift))
        self.assertIn("exp_avg", optimizer.state[master])
        residual = _constraint_residual(parameter, master, reference, basis)
        self.assertLess(residual["master_frobenius"], 1e-6)
        self.assertGreater(residual["quantized_frobenius"], 0)
        self.assertLess(residual["quantized_max_abs"], .02)
        self.assertEqual(residual["weight_dtype"], "torch.bfloat16")

    def test_subspace_requires_two_consecutive_train_passes_before_validation(self):
        self.sequence_training_tokens()
        observed = []
        sample_tokens, prompt_tokens = self.runtime._sample_tokens, self.runtime._tokens
        def samples(sample):
            if sample["purpose"] == "validation":
                observed.append(("loss", sample["id"], self.runtime.job["step"]))
            return sample_tokens(sample)
        def prompts(messages, thinking, output_reserve=0):
            if "validation" in messages[-1]["content"]:
                observed.append(("generation", messages[-1]["content"], self.runtime.job["step"]))
            return prompt_tokens(messages, thinking, output_reserve=output_reserve)
        self.runtime._sample_tokens, self.runtime._tokens = samples, prompts
        result = consolidate(self.runtime, 256, learning_rate=.01, distillation_weight=5, preserve_replay_subspace=True)
        self.assertTrue(result["promoted"], result)
        self.assertEqual(result["completed_steps"], 96)
        self.assertEqual(result["training_selection"]["selected_step"], 96)
        self.assertEqual(result["training_selection"]["checkpoints"], [64, 96, 128, 160, 192, 224, 256])
        self.assertEqual(result["training_selection"]["policy_version"], "v6-bounded-research-512")
        self.assertEqual(result["training_selection"]["required_consecutive_passes"], 2)
        self.assertEqual(result["training_selection"]["required_new_exposures"], 2)
        self.assertTrue(all(row["minimum_new_exposures"] >= 2 for row in result["training_selection"]["results"]))
        self.assertEqual([row["consecutive_passes"] for row in result["training_selection"]["results"]], [1, 2])
        self.assertEqual(len(observed), 8)
        self.assertTrue(all(step == 96 for _, _, step in observed))
        torch.testing.assert_close(self.runtime.branch.base.down_proj.weight[:, 1], self.original[:, 1], rtol=0, atol=0)
        drift = result["replay_subspace"]["replay_training_logit_drift"]
        self.assertTrue(drift["finite"])
        self.assertEqual(drift["max_abs"], 0)
        self.assertEqual(result["replay_subspace"]["constraint_residual"]["quantized_max_abs"], 0)
        self.assertTrue(all(self.runtime.model.generated))

    def test_subspace_failed_train_selection_never_evaluates_validation_and_rolls_back(self):
        self.sequence_training_tokens()
        self.edit_provenance(self.samples["new_training"], {"generation_check": {"type": "exact", "expected": "not generated"}})
        sample_tokens, prompt_tokens = self.runtime._sample_tokens, self.runtime._tokens
        def samples(sample):
            self.assertEqual(sample["purpose"], "training")
            return sample_tokens(sample)
        def prompts(messages, thinking, output_reserve=0):
            self.assertNotIn("validation", messages[-1]["content"])
            return prompt_tokens(messages, thinking, output_reserve=output_reserve)
        self.runtime._sample_tokens, self.runtime._tokens = samples, prompts
        result = consolidate(self.runtime, 96, learning_rate=.01, preserve_replay_subspace=True)
        self.assertFalse(result["promoted"])
        self.assertIsNone(result["training_selection"]["selected_step"])
        self.assertEqual([row["step"] for row in result["training_selection"]["results"]], [64, 96])
        self.assertNotIn("baseline_new_loss", result)
        self.assertNotIn("new_generation", result)
        self.assert_restored()

    def test_subspace_failed_checkpoint_resets_pass_streak_and_progress_omits_answers(self):
        self.sequence_training_tokens()
        generate = self.runtime.model.generate
        def fail_second_checkpoint(ids, **kwargs):
            if self.runtime.job["step"] == 96:
                # An actual generation fault between two passing checkpoints
                # must require two fresh passes, not count separated passes.
                return torch.cat((ids, torch.tensor([[1, 3]])), dim=1)
            return generate(ids, **kwargs)
        progress = io.StringIO()
        with patch.object(self.runtime.model, "generate", side_effect=fail_second_checkpoint), patch("sys.stdout", progress):
            result = consolidate(self.runtime, 256, learning_rate=.01, preserve_replay_subspace=True)
        self.assertTrue(result["promoted"], result)
        self.assertEqual(result["completed_steps"], 160)
        selection = result["training_selection"]
        self.assertEqual([row["step"] for row in selection["results"]], [64, 96, 128, 160])
        self.assertEqual([row["consecutive_passes"] for row in selection["results"]], [1, 0, 1, 2])
        summaries = [json.loads(line) for line in progress.getvalue().splitlines()]
        self.assertEqual([row["accepted"] for row in summaries], [True, False, True, True])
        self.assertNotIn("newknowledge", progress.getvalue())
        self.assertNotIn("oldability", progress.getvalue())
        self.assertTrue(all(row["kind"] == "core_train_checkpoint" for row in summaries))

    def test_subspace_single_passing_checkpoint_cannot_publish_or_use_validation(self):
        self.sequence_training_tokens()
        tokens = self.runtime._sample_tokens
        def training_only(sample):
            self.assertEqual(sample["purpose"], "training")
            return tokens(sample)
        self.runtime._sample_tokens = training_only
        result = consolidate(self.runtime, 64, learning_rate=.01, preserve_replay_subspace=True)
        self.assertTrue(result["training_selection"]["results"][0]["accepted"])
        self.assertIsNone(result["training_selection"]["selected_step"])
        self.assertFalse(result["promoted"])
        self.assertNotIn("baseline_new_loss", result)
        self.assert_restored()

    def test_subspace_full_rank_or_invalid_arguments_fail_closed(self):
        self.sequence_training_tokens()
        for flag, steps in ((1, 64), ("true", 64), (True, 63), (True, 513)):
            with self.subTest(flag=flag, steps=steps), self.assertRaises(ValueError):
                consolidate(self.runtime, steps, preserve_replay_subspace=flag)
        def both_axes(sample):
            return torch.tensor([[0, 1, 1]]), torch.tensor([[-100, 1, 1]])
        self.runtime._sample_tokens = both_axes
        with self.assertRaisesRegex(ValueError, "占满核心输入"):
            consolidate(self.runtime, 64, preserve_replay_subspace=True)
        self.assert_restored()

    def test_research_512_resumes_final_check_with_complete_policy_history(self):
        from luban.training_checkpoint import TrainingPaused, TrainingSession
        self.sequence_training_tokens()
        session, arguments = self.make_session(steps=512, projection=True, scope="research")
        self.assertEqual(session.body["scope"], "research")
        self.assertEqual(session.binding["scope"], "research")
        generate = self.runtime.model.generate
        def late_learning(ids, **kwargs):
            if 0 < self.runtime.job["step"] < 480:
                return torch.cat((ids, torch.tensor([[1, 3]])), dim=1)
            result = generate(ids, **kwargs)
            if self.runtime.job["step"] == 512:
                session.pause_requested = True
            return result
        with patch.object(self.runtime.model, "generate", side_effect=late_learning), patch("sys.stdout", io.StringIO()), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        saved = session.load()
        self.assertEqual(saved["step"], 496)
        self.assertEqual(saved["consecutive_passes"], 1)
        self.assertEqual([row["step"] for row in saved["report"]["training_selection"]["results"]][-3:], [416, 448, 480])
        self.assert_restored()
        session.pause()
        self.runtime.training_session = TrainingSession(self.runtime, "evolution", arguments, session.body["records"], run_id=session.id, scope="research")
        with patch("sys.stdout", io.StringIO()):
            report = consolidate(self.runtime, **arguments)
        self.assertTrue(report["promoted"])
        self.assertEqual(report["completed_steps"], 512)
        self.assertEqual(report["resumed_from_steps"], [496])
        selection = report["training_selection"]
        self.assertEqual(selection["selected_step"], 512)
        self.assertEqual([row["step"] for row in selection["results"]], selection["checkpoints"])
        self.assertEqual([row["consecutive_passes"] for row in selection["results"]][-2:], [1, 2])

    def test_resume_cannot_weaken_declared_exposure_or_streak_policy(self):
        from luban.training_checkpoint import TrainingPaused
        self.sequence_training_tokens()
        session, arguments = self.make_session(steps=512, projection=True)
        save = session.save
        def pause_completed_check(payload, progress):
            save(payload, progress)
            if progress["step"] == 64:
                session.pause_requested = True
        with patch.object(session, "save", side_effect=pause_completed_check), patch("sys.stdout", io.StringIO()), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        saved = session.load()
        saved["report"]["training_selection"]["required_consecutive_passes"] = 1
        session.save(saved, {"step": 64, "phase": "training"})
        session.pause_requested = False
        with self.assertRaisesRegex(ValueError, "选择政策不兼容"):
            consolidate(self.runtime, **arguments)
        self.assert_restored()

    def test_subspace_does_not_select_before_every_new_sample_is_trained_twice(self):
        self.sequence_training_tokens()
        for index in range(32):
            self.runtime.memory.add_sample(f"new training coverage {index}", "newknowledge", source="human",
                purpose="training", approved=True, training_allowed=True,
                provenance={"evolution_role": "new", "split_family": f"new-training-coverage-{index}"})
        result = consolidate(self.runtime, 320, learning_rate=.01, preserve_replay_subspace=True)
        self.assertTrue(result["promoted"], result)
        checkpoints = result["training_selection"]["results"]
        self.assertEqual([row["minimum_new_exposures"] for row in checkpoints], [0, 1, 1, 2, 2])
        self.assertEqual([row["accepted"] for row in checkpoints], [False, False, False, True, True])
        self.assertEqual(result["training_selection"]["selected_step"], 192)
        self.assertEqual(result["training_selection"]["checkpoints"][-2:], [288, 320])

    def test_subspace_capture_failure_removes_hook_and_restores_original_weights(self):
        self.sequence_training_tokens()
        projection = self.runtime.branch.base.down_proj
        hooks = len(projection._forward_pre_hooks)
        with patch.object(self.runtime.model, "forward", side_effect=RuntimeError("capture failure")):
            with self.assertRaisesRegex(RuntimeError, "capture failure"):
                consolidate(self.runtime, 64, preserve_replay_subspace=True)
        self.assertEqual(len(projection._forward_pre_hooks), hooks)
        self.assert_restored()

    def test_semantic_generation_checks_integrate_without_literal_expected(self):
        self.edit_provenance(self.samples["replay_validation"], {
            "generation_check": {"type": "semantic", "validator": "nonempty", "forbidden": ["BRAND"]}})
        result = self.run_training()
        self.assertTrue(result["promoted"])
        row = result["replay_generation"][0]
        self.assertEqual(row["validator"], "nonempty")
        self.assertIsNone(row["expected"])
        self.assertEqual(row["answer"], "oldability")

    def test_official_hybrid_graph_captured_subspace_preserves_replay_logits(self):
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from luban.backbone import text_backbone
        from luban.dynamic import insert_experts
        config = Qwen3_5Config(text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                                "partial_rotary_factor": 1., "mrope_section": [2, 1, 1]},
            "eos_token_id": 2, "pad_token_id": 0},
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                           "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                           "spatial_merge_size": 2, "num_position_embeddings": 16},
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        model = Qwen3_5ForConditionalGeneration(config).eval()
        decoder, _ = text_backbone(model)
        branch = insert_experts(model, [3], 4, self.runtime.store / "official", "official-tiny")[0]
        runtime = SimpleNamespace(model=model, decoder=decoder, branch=branch, job=None,
            _sample_tokens=lambda sample: (torch.tensor([[3, 4, 5, 2]]), torch.tensor([[-100, -100, 5, 2]])))
        replay = consolidation_samples(self.runtime)["replay_training"]
        parameter = branch.base.down_proj.weight
        previous = parameter.detach().clone()
        basis, capture = _capture_replay_subspace(runtime, parameter, replay)
        self.assertEqual(capture["captured_positions"], 2)
        self.assertGreater(capture["rank"], 0)
        self.assertLessEqual(capture["rank"], 2)
        master = nn.Parameter(previous + .01 * torch.randn_like(previous))
        _project_core_delta(master, previous, basis)
        with torch.no_grad():
            parameter.copy_(master)
        drift = _replay_logit_drift(runtime, parameter, previous, master, replay)
        self.assertTrue(drift["finite"])
        self.assertEqual(drift["positions"], 2)
        self.assertLess(drift["max_abs"], 1e-6)
        self.assertFalse(torch.equal(parameter, previous))


    def make_session(self, *, steps=32, projection=False, weight=0, regressions=(), run_id=None, scope="product", learning_rate=.01):
        from luban.training_checkpoint import TrainingSession
        arguments = {"steps": steps, "learning_rate": learning_rate, "distillation_weight": weight,
                     "preserve_replay_subspace": projection}
        records = {"groups": consolidation_samples(self.runtime),
                   "protection": __import__("luban.retention", fromlist=["protected_validation"]).protected_validation(self.runtime),
                   "retention_regressions": list(regressions)}
        session = TrainingSession(self.runtime, "evolution", arguments, records, run_id=run_id, scope=scope)
        self.runtime.training_session = session
        return session, arguments

    def test_640_and832_refuse_legacy_product_or_unregistered_research_before_any_update(self):
        for steps, scope in ((640, "product"), (640, "research"), (832, "product"), (832, "research")):
            session, arguments = self.make_session(steps=steps, projection=True, weight=5,
                scope=scope, learning_rate=1e-5)
            with self.subTest(steps=steps, scope=scope), self.assertRaisesRegex(ValueError, "关系教学|字段教学"):
                consolidate(self.runtime, **arguments)
            self.assert_restored()
            self.assertIsNone(session.load())
            session.rejected("CPU exact-dataset refusal")
        session, arguments = self.make_session(steps=832, projection=False, weight=5,
            scope="research", learning_rate=1e-5)
        with self.assertRaisesRegex(ValueError, "关系教学"):
            consolidate(self.runtime, **arguments)
        self.assertIsNone(session.load())
        session.rejected("CPU projection-disabled832 refusal")
        for steps in (513, 544, 608, 641, 672, 800, 833):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "64–512"):
                consolidate(self.runtime, steps, preserve_replay_subspace=True)
        self.assert_restored()

    def test_relation640_resume_keeps_every_post512_check_and_two_consecutive_passes(self):
        self.extended_relation_resume(640, "RELATION_TRAIN_CHECKPOINTS", "validate_relation_teaching")

    def test_field832_resume_keeps_every_post640_check_and_two_consecutive_passes(self):
        self.extended_relation_resume(832, "FIELD_TRAIN_CHECKPOINTS", "validate_field_teaching")

    def extended_relation_resume(self, steps, checkpoint_name, validator):
        from luban.training_checkpoint import TrainingPaused, TrainingSession
        import luban.evolution as evolution
        checkpoints = getattr(evolution, checkpoint_name)
        self.sequence_training_tokens()
        with torch.no_grad():
            self.runtime.branch.base.down_proj.weight[1, 0] = .0001
        self.original = self.runtime.branch.base.down_proj.weight.detach().clone()
        self.runtime._base_core_weight = self.original.clone()
        session, arguments = self.make_session(steps=steps, projection=True, weight=5,
            scope="research", learning_rate=1e-5)
        generate = self.runtime.model.generate
        def late_learning(ids, **kwargs):
            if 0 < self.runtime.job["step"] < steps - 32 and int(ids[0, 0]) == 0:
                return torch.cat((ids, torch.tensor([[1, 3]])), dim=1)
            result = generate(ids, **kwargs)
            if self.runtime.job["step"] == steps:
                session.pause_requested = True
            return result
        # Exact immutable44/57 authorization is tested separately. This
        # two-logit graph isolates persisted checks, exposures and Adam resume.
        with patch("luban.core_publication." + validator, return_value=True), \
                patch.object(self.runtime.model, "generate", side_effect=late_learning), \
                patch("sys.stdout", io.StringIO()), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        saved = session.load()
        self.assertEqual(saved["step"], steps - 16)
        self.assertEqual(saved["consecutive_passes"], 1)
        self.assertEqual([row["step"] for row in saved["report"]["training_selection"]["results"]], list(checkpoints[:-1]))
        self.assert_restored()
        session.pause()
        self.runtime.training_session = TrainingSession(self.runtime, "evolution", arguments,
            session.body["records"], run_id=session.id, scope="research")
        with patch("luban.core_publication." + validator, return_value=True), patch("sys.stdout", io.StringIO()):
            report = consolidate(self.runtime, **arguments)
        self.assertEqual(report["completed_steps"], steps)
        self.assertEqual(report["resumed_from_steps"], [steps - 16])
        selection = report["training_selection"]
        self.assertEqual(selection["selected_step"], steps)
        self.assertEqual(selection["checkpoints"], list(checkpoints))
        self.assertEqual([row["step"] for row in selection["results"]], list(checkpoints))
        self.assertEqual([row["consecutive_passes"] for row in selection["results"]][-2:], [1, 2])

    def test_durable_pause_resume_matches_adam_master_and_rng_exactly(self):
        from luban.training_checkpoint import TrainingPaused, TrainingSession
        self.sequence_training_tokens()
        forward = self.runtime.model.forward
        def stochastic_gradient(ids, labels=None, **kwargs):
            result = forward(ids, labels=labels, **kwargs)
            if torch.is_grad_enabled():
                result.logits = result.logits * (.75 + .5 * torch.rand(()))
                if labels is not None:
                    result.loss = F.cross_entropy(result.logits[:, :-1].reshape(-1, 2), labels[:, 1:].reshape(-1))
            return result
        self.runtime.model.forward = stochastic_gradient
        control, arguments = self.make_session(weight=2)
        expected_report = consolidate(self.runtime, **arguments)
        self.assertTrue(expected_report["promoted"])
        expected_weight = self.runtime.branch.base.down_proj.weight.detach().clone()
        expected = control.load()
        restore_core(self.runtime)
        session, arguments = self.make_session(weight=2)
        save = session.save
        def pause_after_safe_step(payload, progress):
            save(payload, progress)
            if progress["step"] == 16:
                session.pause_requested = True
        with patch.object(session, "save", side_effect=pause_after_safe_step), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        self.assert_restored()
        with self.runtime.memory.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE kind='evolution_failure'").fetchone()[0], 0)
        self.assertEqual(session.load()["step"], 16)
        session.pause()  # Worker releases ownership only after the core finally restored it.
        resumed = TrainingSession(self.runtime, "evolution", arguments, session.body["records"], run_id=session.id)
        self.runtime.training_session = resumed
        report = consolidate(self.runtime, **arguments)
        self.assertTrue(report["promoted"])
        self.assertEqual(report["id"], session.id)
        self.assertEqual(report["resumed_from_steps"], [16])
        self.assertTrue(torch.equal(expected_weight, self.runtime.branch.base.down_proj.weight))
        actual = resumed.load()
        self.assertTrue(torch.equal(expected["master"], actual["master"]))
        for key, value in expected["optimizer"]["state"][0].items():
            self.assertTrue(torch.equal(value, actual["optimizer"]["state"][0][key]), key)
        self.assertEqual(expected["kl_total"], actual["kl_total"])
        self.assertEqual(expected["new_exposures"], actual["new_exposures"])

    def test_pause_inside_training_check_resumes48_and_rechecks64(self):
        from luban.training_checkpoint import TrainingPaused, TrainingSession
        self.sequence_training_tokens()
        session, arguments = self.make_session(steps=128, projection=True, weight=5)
        generate = self.runtime.model.generate
        def interrupt_check(ids, **kwargs):
            result = generate(ids, **kwargs)
            if self.runtime.job["step"] == 64:
                session.pause_requested = True
            return result
        with patch.object(self.runtime.model, "generate", side_effect=interrupt_check), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        self.assert_restored()
        saved = session.load()
        self.assertEqual(saved["step"], 48)
        self.assertEqual(saved["report"]["training_selection"]["results"], [])
        self.assertEqual(saved["consecutive_passes"], 0)
        session.pause()
        self.runtime.training_session = TrainingSession(self.runtime, "evolution", arguments, session.body["records"], run_id=session.id)
        report = consolidate(self.runtime, **arguments)
        self.assertTrue(report["promoted"])
        self.assertEqual(report["resumed_from_steps"], [48])
        self.assertEqual(report["training_selection"]["selected_step"], 96)
        history = report["training_selection"]["results"]
        self.assertEqual([row["step"] for row in history], [64, 96])
        self.assertEqual([row["consecutive_passes"] for row in history], [1, 2])
        self.assertEqual([row["minimum_new_exposures"] for row in history], [32, 48])

    def test_resumed_exposure_counter_cannot_be_fabricated(self):
        from luban.training_checkpoint import TrainingPaused
        self.sequence_training_tokens()
        session, arguments = self.make_session(steps=128, projection=True)
        save = session.save
        def pause_completed_check(payload, progress):
            save(payload, progress)
            if progress["step"] == 64:
                session.pause_requested = True
        with patch.object(session, "save", side_effect=pause_completed_check), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        self.assert_restored()
        saved = session.load()
        self.assertEqual(saved["consecutive_passes"], 1)
        self.assertEqual(saved["report"]["training_selection"]["results"][0]["step"], 64)
        saved["new_exposures"][self.samples["new_training"]] = 1
        session.save(saved, {"step": 64, "phase": "training"})
        session.pause_requested = False
        with self.assertRaisesRegex(ValueError, "曝光次数"):
            consolidate(self.runtime, **arguments)
        self.assert_restored()

    def test_frozen_manifest_survives_memory_deletion_and_publishes_full_snapshot(self):
        regression = {"id": "extra-old-protection", "prompt": "old protected task", "answer": "oldability",
            "source": "human", "purpose": "validation", "approved": 1, "training_allowed": False,
            "provenance": json.dumps({"evolution_role": "replay", "retention_generation_only": True})}
        session, arguments = self.make_session(regressions=[regression])
        frozen = session.body["records"]["groups"]
        for identifier in self.samples.values():
            self.runtime.memory.delete_sample(identifier)
        self.assertFalse(core_status(self.runtime)["ready"])
        report = consolidate(self.runtime, **arguments)
        self.assertTrue(report["promoted"])
        snapshot = read_core_retention(self.runtime)
        self.assertEqual(snapshot["groups"], frozen)
        self.assertEqual(snapshot["regression_validation"], [regression])
        self.assertEqual(session.body["state"], "committed")
        with self.runtime.memory.connect() as db:
            events = db.execute("SELECT id FROM events WHERE kind='evolution'").fetchall()
        self.assertEqual([row[0] for row in events], ["training:" + session.id])

    def test_prepared_commit_failure_keeps_artifact_hashes_for_idempotent_recovery(self):
        from luban.training_checkpoint import recover_commits, training_runs
        session, arguments = self.make_session()
        with patch.object(session, "finish_commit", side_effect=OSError("simulated commit window")):
            with self.assertRaisesRegex(OSError, "commit window"):
                consolidate(self.runtime, **arguments)
        self.assert_restored()
        self.assertEqual(session.body["state"], "committing")
        planned = session.body["commit"]
        self.assertEqual(planned["report"]["publication_state"], "published")
        self.assertEqual(len(planned["artifacts"]), 4)
        for name, sha in planned["artifacts"].items():
            self.assertEqual(hashlib.sha256((self.runtime.store / name).read_bytes()).hexdigest(), sha)
        with self.runtime.memory.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE kind='evolution_failure'").fetchone()[0], 0)
        recover_commits(self.runtime.memory, self.runtime.store, self.runtime.identity)
        recover_commits(self.runtime.memory, self.runtime.store, self.runtime.identity)
        load_active_core(self.runtime)
        self.assertEqual(self.runtime.active_core, "core-" + session.id)
        with self.runtime.memory.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE kind='evolution'").fetchone()[0], 1)
        self.assertEqual(training_runs(self.runtime.memory, self.runtime.identity)[0]["state"], "committed")

    def test_session_parameter_or_protection_mismatch_cannot_train(self):
        session, arguments = self.make_session()
        with self.assertRaisesRegex(ValueError, "训练参数"):
            consolidate(self.runtime, **(arguments | {"learning_rate": .005}))
        session.body["records"]["protection"] = [{"id": "omitted-active-protection"}]
        with self.assertRaisesRegex(ValueError, "保护材料"):
            consolidate(self.runtime, **arguments)
        self.assert_restored()



    def test_pause_after_selected_checkpoint_never_runs_additional_optimizer_steps(self):
        from luban.training_checkpoint import TrainingPaused, TrainingSession
        self.sequence_training_tokens()
        session, arguments = self.make_session(steps=128, projection=True)
        save = session.save
        def pause_selected(payload, progress):
            save(payload, progress)
            if progress["step"] == 96:
                session.pause_requested = True
        with patch.object(session, "save", side_effect=pause_selected), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        self.assert_restored()
        self.assertEqual(session.load()["report"]["training_selection"]["selected_step"], 96)
        session.pause()
        self.runtime.training_session = TrainingSession(self.runtime, "evolution", arguments, session.body["records"], run_id=session.id)
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("selected candidate must not train again")):
            report = consolidate(self.runtime, **arguments)
        self.assertTrue(report["promoted"])
        self.assertEqual(report["completed_steps"], 96)
        self.assertEqual([row["step"] for row in report["training_selection"]["results"]], [64, 96])

    def test_existing_accepted_core_protection_is_checked_before_frozen_new_training(self):
        from luban.retention import protected_validation
        initial = self.run_training()
        committed = self.runtime.branch.base.down_proj.weight.detach().clone()
        pointer = (self.runtime.store / "active-core.json").read_bytes()
        for purpose in ("training", "validation"):
            self.runtime.memory.add_sample(f"future {purpose}", "newknowledge", source="human", purpose=purpose,
                approved=True, training_allowed=True, provenance={"evolution_role": "new", "split_family": "future-" + purpose})
        session, arguments = self.make_session()
        self.assertEqual(session.body["records"]["protection"], protected_validation(self.runtime))
        self.assertEqual(len(session.body["records"]["protection"]), 2)
        self.runtime.tokenizer.decode = lambda tokens, **kwargs: "newknowledge" if int(tokens[0]) == 0 else "changed old ability"
        report = consolidate(self.runtime, **arguments)
        self.assertFalse(report["promoted"])
        self.assertFalse(all(row["passed"] for row in report["baseline_core_retention_generation"]))
        self.assertEqual(self.runtime.job["step"], 0)
        self.assertIsNone(session.body["checkpoint"])
        self.assertEqual(self.runtime.active_core, initial["version"])
        self.assertTrue(torch.equal(committed, self.runtime.branch.base.down_proj.weight))
        self.assertEqual((self.runtime.store / "active-core.json").read_bytes(), pointer)

    def test_commit_acknowledgement_error_cannot_rollback_already_committed_core(self):
        session, arguments = self.make_session()
        finish = session.finish_commit
        def lost_acknowledgement():
            finish()
            raise OSError("acknowledgement lost after durable commit")
        with patch.object(session, "finish_commit", side_effect=lost_acknowledgement):
            report = consolidate(self.runtime, **arguments)
        self.assertTrue(report["promoted"])
        self.assertEqual(session.body["state"], "committed")
        self.assertEqual(self.runtime.active_core, "core-" + session.id)
        self.assertEqual(json.loads((self.runtime.store / "active-core.json").read_text())["version"], report["version"])
        self.assertFalse(torch.equal(self.original, self.runtime.branch.base.down_proj.weight))
        self.assertIn("acknowledgement", self.runtime.job["commit_acknowledgement_error"])
        for name, sha in session.body["commit"]["artifacts"].items():
            self.assertEqual(hashlib.sha256((self.runtime.store / name).read_bytes()).hexdigest(), sha)
        with self.runtime.memory.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE kind='evolution'").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE kind='evolution_failure'").fetchone()[0], 0)

    def qualified_replay_training(self):
        identifiers = [self.samples["replay_training"]]
        for index in range(3):
            identifiers.append(self.runtime.memory.add_sample(f"replay uppercase {index}", "oldability",
                source="human", purpose="training", approved=True, training_allowed=True,
                provenance={"evolution_role": "replay", "split_family": f"qualified-replay-{index}"}))
        for identifier in identifiers:
            self.edit_provenance(identifier, {"training_generation_gate": True,
                "baseline_preservation_eligible": True, "original_baseline_verified_identity": self.runtime.identity,
                "original_baseline_verified_core": None, "original_baseline_generation": {
                    "id": identifier, "answer": "oldability", "passed": True, "complete": True, "branch_enabled": False}})
        return identifiers

    def test_normalized_exact_replay_is_checked_at_real_training_checkpoints(self):
        self.sequence_training_tokens()
        identifiers = self.qualified_replay_training()
        for identifier in identifiers:
            self.edit_provenance(identifier, {"generation_check": {
                "type": "semantic", "validator": "normalized_exact", "expected": "oldability"}})
        result = consolidate(self.runtime, 128, learning_rate=.01, preserve_replay_subspace=True)
        self.assertTrue(result["promoted"])
        checkpoints = result["training_selection"]["results"]
        self.assertGreaterEqual(len(checkpoints), 2)
        for checkpoint in checkpoints:
            rows = checkpoint["replay_training_generation"]
            self.assertEqual({row["id"] for row in rows}, set(identifiers))
            self.assertTrue(all(row["passed"] and row["complete"] for row in rows))
            self.assertTrue(all(row["validator"] == "normalized_exact" for row in rows))
        self.assertEqual(result["training_selection"]["policy_version"],
                         "v9-whole-string-qualified-replay-generation")

    def test_decoder_replay_failure_rejects_even_when_new_generation_and_loss_pass(self):
        self.sequence_training_tokens()
        self.qualified_replay_training()
        generate = self.runtime.model.generate
        def regress_only_at_training_check(ids, **kwargs):
            output = generate(ids, **kwargs)
            if self.runtime.job["phase"] == "training_check" and int(ids[0, 0]) == 1:
                output[0, -2] = 0
            return output
        with patch.object(self.runtime.model, "generate", side_effect=regress_only_at_training_check):
            result = consolidate(self.runtime, 128, learning_rate=.01, preserve_replay_subspace=True)
        self.assertFalse(result["promoted"])
        self.assertNotIn("new_generation", result)
        for checkpoint in result["training_selection"]["results"]:
            self.assertTrue(all(row["passed"] for row in checkpoint["new_generation"]))
            self.assertLessEqual(checkpoint["replay_loss"], result["baseline_training_replay_loss"] * 1.05 + .01)
            self.assertFalse(any(row["passed"] for row in checkpoint["replay_training_generation"]))
            self.assertFalse(checkpoint["accepted"])
        self.assert_restored()

    def test_qualified_replay_minimum_evidence_and_no_checkpoint_bypass(self):
        self.sequence_training_tokens()
        identifiers = self.qualified_replay_training()
        self.edit_provenance(identifiers[0], {"original_baseline_verified_identity": "another-base"})
        with self.assertRaisesRegex(ValueError, "正确、完整"):
            consolidate(self.runtime, 128, preserve_replay_subspace=True)
        self.edit_provenance(identifiers[0], {"original_baseline_verified_identity": self.runtime.identity})
        with self.assertRaisesRegex(ValueError, "完整训练检查点"):
            consolidate(self.runtime, 16)
        self.edit_provenance(identifiers[0], {"training_generation_gate": False})
        with self.assertRaisesRegex(ValueError, "至少需要四条"):
            consolidate(self.runtime, 128, preserve_replay_subspace=True)
        self.assert_restored()

    def test_resume_cannot_drop_protected_replay_generation_history(self):
        from luban.training_checkpoint import TrainingPaused
        self.sequence_training_tokens()
        self.qualified_replay_training()
        session, arguments = self.make_session(steps=128, projection=True)
        save = session.save
        def pause_checkpoint(payload, progress):
            save(payload, progress)
            if progress["step"] == 64:
                session.pause_requested = True
        with patch.object(session, "save", side_effect=pause_checkpoint), self.assertRaises(TrainingPaused):
            consolidate(self.runtime, **arguments)
        saved = session.load()
        self.assertEqual(saved["report"]["training_selection"]["minimum_replay_generation"], 4)
        saved["report"]["training_selection"]["results"][0]["replay_training_generation"].pop()
        session.save(saved, {"step": 64, "phase": "training"})
        session.pause_requested = False
        with self.assertRaisesRegex(ValueError, "生成或曝光门槛历史"):
            consolidate(self.runtime, **arguments)
        self.assert_restored()

    def test_raw_generation_keeps_ids_and_unstripped_decode_without_changing_gold(self):
        self.runtime.tokenizer.decode = lambda tokens, **kwargs: "\n oldability \n"
        from luban.evolution import _generation_loaded
        self.runtime.branch.expert = None
        row = consolidation_samples(self.runtime)["replay_training"][0]
        generated = _generation_loaded(self.runtime, [row])[0]
        self.assertEqual(generated["raw_completion_token_ids"], [1, 3])
        self.assertEqual(generated["decoded_completion_unstripped"], "\n oldability \n")
        self.assertEqual(generated["answer"], row["answer"])
        self.assertTrue(generated["passed"])
        json.dumps(generated, allow_nan=False)



def _role(sample):
    return json.loads(sample["provenance"])["evolution_role"]


if __name__ == "__main__":
    unittest.main()
