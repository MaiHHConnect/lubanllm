"""Offline accepted-knowledge persistence and real CPU branch generation gates."""
import copy
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F

from luban.dynamic import DynamicMLP
from luban.evolution import consolidation_samples
from luban.memory import Memory
from luban.retention import (GROUP_KEYS, eligible_learning_sample, protected_validation,
                             read_core_retention, read_retention_file, save_core_retention)
from luban.runtime import Runtime


def record(sample_id, role, purpose, *, answer="identity", source="human", **extra):
    value = {"evolution_role": role, "split_family": sample_id,
             "messages": [{"role": "user", "content": "context " + sample_id}], **extra}
    return {"id": sample_id, "prompt": "context " + sample_id, "answer": answer, "purpose": purpose,
            "source": source, "approved": True, "training_allowed": True,
            "provenance": json.dumps(value, ensure_ascii=False)}


def groups():
    return {key: [record(key, *key.split("_"))] for key in GROUP_KEYS}


def regression(index, *, expected="identity", answer=""):
    row = record("regression-" + str(index), "replay", "validation", answer=answer,
                 retention_generation_only=True, generation_check={"type": "exact", "expected": expected})
    row["training_allowed"] = False
    return row


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = SimpleNamespace(store=self.root, identity="fixture-base", active_core="core-111111111111",
                                       lock=threading.RLock(), memory=Memory(self.root))
        self.rows = groups()

    def save(self, regression_samples=()):
        return save_core_retention(self.runtime, self.runtime.active_core, self.rows,
                                   regression_samples=regression_samples)

    def test_full_rows_readback_idempotence_and_generation_only_is_not_training(self):
        extra = [regression(0, answer="", expected="identity")]
        saved = self.save(extra)
        payload = read_core_retention(self.runtime)
        self.assertEqual(payload["groups"], self.rows)
        self.assertEqual(payload["regression_validation"], extra)
        self.assertEqual(saved, self.save(extra))
        self.assertEqual(saved["file_sha256"], hashlib.sha256(Path(saved["path"]).read_bytes()).hexdigest())
        self.assertEqual(len(protected_validation(self.runtime)), 3)
        self.assertFalse(eligible_learning_sample(extra[0]))
        changed = copy.deepcopy(self.rows)
        changed["new_training"][0]["answer"] = "different"
        before = Path(saved["path"]).read_bytes()
        with self.assertRaisesRegex(ValueError, "不可覆盖"):
            save_core_retention(self.runtime, self.runtime.active_core, changed, regression_samples=extra)
        self.assertEqual(Path(saved["path"]).read_bytes(), before)

    def test_legacy_retention_does_not_borrow_mlx_adoption_records(self):
        self.save()
        before = protected_validation(self.runtime, include_learning=True)
        native = {"kind": "learning", "state": "committed", "scope": "product",
                  "arguments": {"protocol": "native-mlx-expert-preservation-v1"}, "records": {"native": "opaque"}}
        with patch("luban.training_checkpoint.training_runs", return_value=[native]):
            self.assertEqual(protected_validation(self.runtime, include_learning=True), before)

    def test_deleted_database_does_not_delete_old_new_replay_or_unlock_evolution(self):
        self.save()
        # All DB rows can be absent; immutable accepted materials remain intact.
        assembled = consolidation_samples(self.runtime)
        self.assertEqual(assembled["new_training"], [])
        self.assertEqual(assembled["new_validation"], [])
        self.assertEqual(len(assembled["replay_training"]), 2)
        self.assertEqual(len(assembled["replay_validation"]), 2)
        self.assertEqual(read_core_retention(self.runtime)["groups"], self.rows)
        repeated = [copy.deepcopy(row) for rows in self.rows.values() for row in rows]
        self.runtime.memory = SimpleNamespace(samples=lambda: repeated)
        self.assertEqual(consolidation_samples(self.runtime), assembled)

    def test_genuinely_new_train_and_valid_merge_without_modifying_original_db(self):
        self.save()
        current = [record("next-train", "new", "training", answer="next"),
                   record("next-valid", "new", "validation", answer="next")]
        before = copy.deepcopy(current)
        self.runtime.memory = SimpleNamespace(samples=lambda: current)
        assembled = consolidation_samples(self.runtime)
        self.assertEqual([row["id"] for row in assembled["new_training"]], ["next-train"])
        self.assertEqual([row["id"] for row in assembled["new_validation"]], ["next-valid"])
        self.assertEqual(len(assembled["replay_training"]), 2)
        self.assertEqual(len(assembled["replay_validation"]), 2)
        self.assertEqual(current, before)

    def test_same_id_context_answer_and_split_conflicts_fail_without_latest_wins(self):
        self.save()
        original = self.rows["new_training"][0]
        for mutation in ({"answer": "correction"}, {"purpose": "validation"},
                         {"provenance": json.dumps({"evolution_role": "new", "messages": [
                             {"role": "user", "content": "another context"}]})}):
            with self.subTest(mutation=mutation):
                conflict = {**original, **mutation}
                self.runtime.memory = SimpleNamespace(samples=lambda: [conflict])
                with self.assertRaisesRegex(ValueError, "冲突.*尚未实现"):
                    consolidation_samples(self.runtime)
        duplicate_context = {**original, "id": "another-id", "answer": "correction"}
        self.runtime.memory = SimpleNamespace(samples=lambda: [duplicate_context])
        with self.assertRaisesRegex(ValueError, "冲突"):
            consolidation_samples(self.runtime)

    def test_generation_only_rule_conflict_rejects_but_never_creates_ce_rows(self):
        self.save([regression(0)])
        row = record("next-train", "new", "training", answer="wrong",
                     messages=[{"role": "user", "content": "context regression-0"}])
        self.runtime.memory = SimpleNamespace(samples=lambda: [row])
        with self.assertRaisesRegex(ValueError, "生成保护规则冲突"):
            consolidation_samples(self.runtime)
        self.runtime.memory = SimpleNamespace(samples=lambda: [])
        assembled = consolidation_samples(self.runtime)
        self.assertNotIn("regression-0", {row["id"] for rows in assembled.values() for row in rows})

    def test_daily_selection_excludes_experiment_conflicts_before_validation(self):
        self.save()
        original = self.rows["new_training"][0]
        conflict = {**original, "answer": "obsolete experimental answer", "provenance": json.dumps({
            **json.loads(original["provenance"]), "dataset": "historical experiment"})}
        lesson = record("daily-lesson", "new", "training", verified_by="local requester")
        self.runtime.memory = SimpleNamespace(samples=lambda: [conflict, lesson])
        with self.assertRaisesRegex(ValueError, "冲突"):
            consolidation_samples(self.runtime)  # Explicit research retains its original checks.
        assembled = consolidation_samples(self.runtime, daily_only=True)
        self.assertEqual(assembled["new_training"], [lesson])
        self.assertEqual({r["id"] for r in assembled["replay_training"]},
                         {r["id"] for key in ("new_training", "replay_training") for r in self.rows[key]})
        self.assertEqual(read_core_retention(self.runtime)["groups"], self.rows)

    def test_nonhuman_needs_true_verification_flag_and_nonempty_evidence(self):
        teacher = record("teacher", "new", "training", source="teacher", program_verified=True)
        self.assertFalse(eligible_learning_sample(teacher))
        value = json.loads(teacher["provenance"])
        value.update(verification_evidence="  ", human_verified=False)
        teacher["provenance"] = json.dumps(value)
        self.assertFalse(eligible_learning_sample(teacher))
        value.update(program_verified=False, verification_evidence="verified by fixture")
        teacher["provenance"] = json.dumps(value)
        self.assertFalse(eligible_learning_sample(teacher))
        value["human_verified"] = True
        teacher["provenance"] = json.dumps(value)
        self.assertTrue(eligible_learning_sample(teacher))

    def test_retired_routing_and_unverified_teacher_cannot_veto_accepted_truth(self):
        self.save()
        accepted = self.rows["new_training"][0]
        retired = {**accepted, "answer": "retired wrong answer", "approved": False}
        routing = copy.deepcopy(accepted)
        routing.update(answer="routing-only wrong answer", provenance=json.dumps({
            **json.loads(routing["provenance"]), "route_only": True}))
        teacher = {**accepted, "answer": "unverified wrong answer", "source": "teacher"}
        current = [retired, routing, teacher, record("next-train", "new", "training", answer="next"),
                   record("next-valid", "new", "validation", answer="next")]
        self.runtime.memory = SimpleNamespace(samples=lambda: current)
        assembled = consolidation_samples(self.runtime)
        self.assertEqual([row["id"] for row in assembled["new_training"]], ["next-train"])
        self.assertEqual(len(assembled["replay_training"]), 2)

    def test_snapshot_missing_tampered_identity_version_and_symlink_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "缺少完整保护"):
            consolidation_samples(self.runtime)
        saved = self.save()
        path = Path(saved["path"])
        original = path.read_bytes()
        payload = json.loads(original)
        payload["groups"]["new_training"][0]["answer"] = "tampered"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "内容校验"):
            read_core_retention(self.runtime)
        path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "不兼容"):
            read_retention_file(path, "another-base", self.runtime.active_core)
        with self.assertRaisesRegex(ValueError, "不兼容"):
            read_retention_file(path, self.runtime.identity, "core-222222222222")
        path.unlink()
        target = self.root / "outside.json"
        target.write_bytes(original)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            read_core_retention(self.runtime)


class BranchModel(nn.Module):
    """Real shared CPU features: learning NEW changes OLD's greedy answer."""
    def __init__(self, branch, *, complete=True):
        super().__init__()
        self.branch = branch
        self.config = SimpleNamespace(model_type="fixture")
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.complete, self.generated = complete, []
        with torch.no_grad():
            branch.base.weight.zero_()
            branch.base.bias.zero_()
            branch.base.bias[1] = 0.01

    def forward(self, ids, labels=None, **kwargs):
        logits = self.branch(torch.ones(*ids.shape, 4))
        loss = F.cross_entropy(logits[:, 0], labels[:, 1]) if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(self, ids, **kwargs):
        token = self(ids).logits[:, -1].argmax(-1)[:, None]
        self.generated.append(int(token[0, 0]))
        suffix = torch.cat((token, torch.full_like(token, 3)), dim=1) if self.complete else token
        return torch.cat((ids, suffix), dim=1)


class LearningRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(42)
        memory = SimpleNamespace(samples=lambda purpose: [record("next-" + purpose, "new", purpose,
                                                                 answer="newknowledge")], event=Mock())
        runtime = Runtime(Mock(), memory)
        runtime.settings.snapshot.return_value = {"context_tokens": 4096, "context_mode": "manual"}
        runtime.device, runtime.dtype = "cpu", torch.float32
        runtime.store, runtime.pointer = self.root, self.root / "active.json"
        runtime.identity, runtime.active_core = "fixture-base", "core-111111111111"
        runtime._ready = Mock()
        runtime.text_config, runtime.base_count = SimpleNamespace(hidden_size=4), 20
        runtime.branch = DynamicMLP(nn.Linear(4, 4), 4, 64, self.root / "layer-0",
                                    runtime.identity + ":layer-0", 100000)
        runtime.model = BranchModel(runtime.branch)
        runtime._tokens = lambda messages, thinking, output_reserve=0: torch.tensor([[0]])
        runtime._sample_tokens = lambda sample: (torch.tensor([[0, 2]]), torch.tensor([[-100, 2]]))
        runtime.tokenizer = SimpleNamespace(eos_token_id=3, decode=lambda tokens, **kwargs:
                                            {0: "wrong", 1: "identity", 2: "newknowledge"}[int(tokens[0])])
        self.runtime = runtime
        save_core_retention(runtime, runtime.active_core, groups(),
                            regression_samples=[regression(index) for index in range(17)])

    def test_real_cpu_generation_rejects_old_knowledge_loss_beyond_latest16(self):
        with patch("luban.runtime.save_training_backup", return_value="fixture-backup"):
            report = self.runtime.learn(steps=16)
        self.assertFalse(report["promoted"])
        self.assertEqual(len(report["core_retention"]["baseline_generation"]), 19)
        self.assertTrue(all(row["passed"] for row in report["core_retention"]["baseline_generation"]))
        self.assertTrue(any(all(row["passed"] for row in candidate["generation_checks"])
                            for candidate in report["candidates"]))
        for candidate in report["candidates"]:
            checks = candidate["core_retention_generation"]
            self.assertEqual(len(checks), 19)
            self.assertTrue(all(row["complete"] and not row["passed"] for row in checks))
        self.assertFalse(self.runtime.pointer.exists())
        self.assertIsNone(self.runtime.branch.expert)
        self.assertEqual(self.runtime.active_core, "core-111111111111")

    def test_core_protection_baseline_requires_eos_before_any_training(self):
        self.runtime.model.complete = False
        with patch("luban.runtime.NeuralExpert", side_effect=AssertionError("must not train")), \
                patch("luban.runtime.save_training_backup") as backup:
            with self.assertRaisesRegex(ValueError, "完整保护检验"):
                self.runtime.learn(steps=16)
            backup.assert_not_called()
        self.assertIsNone(self.runtime.branch.expert)

    def test_deleted_snapshot_blocks_both_learning_and_core_materials(self):
        (self.root / "core-111111111111.retention.json").unlink()
        with self.assertRaisesRegex(ValueError, "缺少完整保护"):
            self.runtime.learn(steps=1)
        with self.assertRaisesRegex(ValueError, "缺少完整保护"):
            consolidation_samples(self.runtime)

    def test_automatic_unverified_teacher_never_enters_learning(self):
        human = record("qualified-human", "new", "training", answer="newknowledge")
        teacher = record("unverified-teacher", "new", "training", source="teacher", answer="wrong",
                         teacher_approved=True)
        self.runtime.memory.samples = lambda purpose: [teacher, human] if purpose == "training" else [
            record("next-validation", "new", "validation", answer="newknowledge")]
        with patch("luban.runtime.save_training_backup", return_value="fixture-backup"):
            report = self.runtime.learn(steps=1)
        self.assertEqual(report["training_ids"], ["qualified-human"])


if __name__ == "__main__":
    unittest.main()
