"""CPU-only exact material-ledger publication and rollback; no model allocation."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import sys

import torch
from safetensors.torch import save_file

from luban.core_publication import (ACCEPTANCE_GATES, SUPERSEDED_DUPLICATES, apply_material_publication,
                                    content_sha, validate_material_publication, publish_isolated_core, file_sha)
from luban.memory import Memory
from luban.retention import save_core_retention
from luban.settings import private_json
from luban.training_checkpoint import TrainingSession, active_hashes
from luban.upgrades import read_upgrade_records


def sample(identifier, role, purpose, *, readonly=False):
    return {"id": identifier, "prompt": "prompt " + identifier, "answer": "gold " + identifier,
        "source": "human", "purpose": purpose, "created": 1., "approved": 1, "training_allowed": not readonly,
        "provenance": json.dumps({"base_identity": "fixture-base", "evolution_role": role,
            "split_family": identifier, "generation_check": {"type": "exact", "expected": "gold " + identifier},
            **({"retention_generation_only": True} if readonly else {})})}


def fixture_plan():
    source = [sample(f"old-new-train-{i}", "new", "training") for i in range(30)]
    source += [sample(identifier, "new", "training") for identifier in sorted(SUPERSEDED_DUPLICATES)]
    source += [sample(f"old-new-valid-{i}", "new", "validation") for i in range(8)]
    source += [sample(f"old-replay-train-{i}", "replay", "training") for i in range(44)]
    source += [sample(f"old-replay-valid-{i}", "replay", "validation") for i in range(17)]
    groups = {key: [copy.deepcopy(row) for row in source
        if row["purpose"] == key.split("_")[1] and json.loads(row["provenance"])["evolution_role"] == key.split("_")[0]]
        for key in ("new_training", "new_validation", "replay_training", "replay_validation")}
    groups["new_training"] = groups["new_training"][:30] + [sample(f"fresh-new-train-{i}", "new", "training") for i in range(4)]
    groups["new_validation"] = [sample(f"fresh-new-valid-{i}", "new", "validation") for i in range(8)]
    groups["replay_training"] += [sample(f"qualified-replay-{i}", "replay", "training") for i in range(4)]
    regressions = [sample(f"old-new-valid-{i}", "replay", "validation", readonly=True) for i in range(8)]
    regressions += [sample(f"engineering-{i}", "replay", "validation", readonly=True) for i in range(16)]
    return {"format": 1, "identity": "fixture-base", "run_id": "123456abcdef", "version": "core-123456abcdef",
        "groups": groups, "retention_regressions": regressions, "source_rows": source, "source_sha256": content_sha(source),
        "superseded_ids": sorted(SUPERSEDED_DUPLICATES | {f"old-new-valid-{i}" for i in range(8)}),
        "acceptance": {"protocol_sha256": "a" * 64, "report_path": "coverage-acceptance-123456abcdef.json",
            "report_sha256": "b" * 64, "gates": dict.fromkeys(ACCEPTANCE_GATES, True), "teacher_calls": 0}}


class CorePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.memory = Memory(Path(self.temp.name))
        self.plan = fixture_plan()
        with self.memory.connect() as db:
            for row in self.plan["source_rows"]:
                db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))

    def rows(self):
        return {row["id"]: row for row in self.memory.samples()}

    def apply(self, plan=None):
        with self.memory.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            apply_material_publication(db, plan or self.plan, "fixture-base")

    def test_publish_keeps_gold_sources_archives_only_fixed_twelve_and_matches_complete_groups(self):
        before = self.rows()
        self.apply()
        after = self.rows()
        for identifier in self.plan["superseded_ids"]:
            expected = before[identifier] | {"approved": 0, "training_allowed": 0}
            self.assertEqual(after[identifier], expected)
        for rows in self.plan["groups"].values():
            for row in rows:
                self.assertEqual(after[row["id"]], row)
        self.assertEqual(len(after), 119)
        with self.memory.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM core_material_archives").fetchone()[0], 12)
        self.assertTrue(all(row["id"] not in after for row in self.plan["retention_regressions"][8:]))

    def test_conflicting_last_anchor_or_new_id_is_detected_before_any_mutation(self):
        for mutation in ("old", "new"):
            with self.subTest(mutation=mutation):
                with self.memory.connect() as db:
                    if mutation == "old":
                        db.execute("UPDATE samples SET answer='changed' WHERE id=?", (self.plan["source_rows"][-1]["id"],))
                    else:
                        row = self.plan["groups"]["new_validation"][-1] | {"answer": "conflict"}
                        db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))
                before = self.rows()
                with self.assertRaisesRegex(ValueError, "材料已变化|稳定 ID"):
                    self.apply()
                self.assertEqual(self.rows(), before)
                with self.memory.connect() as db:
                    self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='core_material_archives'").fetchone())
                    if mutation == "old":
                        db.execute("UPDATE samples SET answer=? WHERE id=?", (self.plan["source_rows"][-1]["answer"], self.plan["source_rows"][-1]["id"]))

    def test_outer_gate_or_omitted_protection_or_arbitrary_retirement_cannot_publish(self):
        before = self.rows()
        for mutation in ("gate", "gold", "retirement", "sha"):
            plan = copy.deepcopy(self.plan)
            if mutation == "gate":
                plan["acceptance"]["gates"]["engineering"] = False
            elif mutation == "gold":
                plan["retention_regressions"][0]["answer"] = "new gold"
            elif mutation == "retirement":
                plan["superseded_ids"][0] = "old-replay-valid-0"
            else:
                plan["source_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.apply(plan)
        self.assertEqual(self.rows(), before)

    def test_late_database_failure_rolls_back_materials_and_archive_transaction(self):
        before = self.rows()
        with self.assertRaisesRegex(RuntimeError, "pointer publication failure"):
            with self.memory.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                apply_material_publication(db, self.plan, "fixture-base")
                raise RuntimeError("pointer publication failure")
        self.assertEqual(self.rows(), before)
        with self.memory.connect() as db:
            self.assertIsNone(db.execute("SELECT name FROM sqlite_master WHERE name='core_material_archives'").fetchone())

    def test_helper_cannot_commit_outside_transaction(self):
        with self.memory.connect() as db, self.assertRaisesRegex(ValueError, "同一事务"):
            apply_material_publication(db, self.plan, "fixture-base")

    def test_actual_publisher_commits_report_exactly_and_upgrade_reader_counts_it(self):
        identity = self.plan["identity"] if len(self.plan["identity"]) == 64 else "a" * 64
        material = copy.deepcopy(self.plan)
        for row in material["source_rows"] + [row for rows in material["groups"].values() for row in rows] + material["retention_regressions"]:
            value = json.loads(row["provenance"])
            if value.get("base_identity") == identity:
                continue
            value["base_identity"] = identity
            if "original_baseline_verified_identity" in value:
                value["original_baseline_verified_identity"] = identity
            row["provenance"] = json.dumps(value)
        production = self.memory
        with production.connect() as db:
            db.execute("DELETE FROM samples")
            for row in material["source_rows"]:
                db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))
        root = Path(self.temp.name)
        source_store, production_store = root / "isolated/modules/luban0-v1", root / "modules/luban0-v1"
        source_store.mkdir(parents=True)
        production_store.mkdir(parents=True)
        source = SimpleNamespace(memory=Memory(root / "isolated"), store=source_store, identity=identity,
            active_core=None, branch=SimpleNamespace(expert_id=None), device="cpu", dtype=torch.float32)
        version, identifier = material["version"], material["run_id"]
        groups, regressions = material["groups"], material["retention_regressions"]
        records = {"groups": groups, "protection": [], "retention_regressions": regressions}
        arguments = {"steps": 832 if len(groups["new_training"]) == 57 else 640 if len(groups["new_training"]) == 44 else 512,
                     "learning_rate": 1e-5, "distillation_weight": 5, "preserve_replay_subspace": True}
        arguments = copy.deepcopy(getattr(self, 'arguments', arguments))
        from luban.evolution import TRAIN_CHECKPOINTS, RELATION_TRAIN_CHECKPOINTS, FIELD_TRAIN_CHECKPOINTS, FIELD_EXPOSURE_CHECKPOINTS
        checkpoints = {512: TRAIN_CHECKPOINTS, 640: RELATION_TRAIN_CHECKPOINTS,
                       832: FIELD_TRAIN_CHECKPOINTS, 1536: FIELD_EXPOSURE_CHECKPOINTS}[arguments['steps']]
        if "field_loss_policy" in material:
            arguments["field_first_token_weight"] = 4
        session = TrainingSession(source, "evolution", arguments, records, run_id=identifier, scope="research")
        save_file({"fixture.weight": torch.eye(2)}, str(source_store / (version + ".safetensors")))
        retention = save_core_retention(source, version, groups, regression_samples=regressions)
        metadata = {"version": version, "identity": identity, "sha256": file_sha(source_store / (version + ".safetensors")),
            "report": "evolution-" + identifier + ".json", "retention_sha256": retention["file_sha256"]}
        private_json(source_store / (version + ".json"), metadata)
        report = {"id": identifier, "version": version, "identity": identity, "promoted": True,
            "steps": arguments['steps'], "training_selection": {"checkpoints": list(checkpoints)},
            "publication_state": "published", "checkpoint": metadata, "gates": {f"gate{i}": True for i in range(8)},
            "baseline_new_loss": 2., "validation_new_loss": 1., "baseline_replay_loss": 1., "validation_replay_loss": 1.,
            "started": time.time() - 10, "elapsed_seconds": 1.}
        if "field_loss_policy" in material:
            report["field_loss_policy"] = copy.deepcopy(material["field_loss_policy"])
        private_json(source_store / metadata["report"], report)
        artifacts = [version + suffix for suffix in (".safetensors", ".retention.json", ".json")] + [metadata["report"]]
        pointer = {"version": version, "identity": identity, "sha256": metadata["sha256"]}
        session.prepare_commit(report, {"active-core.json": pointer}, artifacts, event_kind="evolution")
        session.finish_commit()
        source.active_core = version
        acceptance_path = root / material["acceptance"]["report_path"]
        private_json(acceptance_path, {"outer_gates": dict.fromkeys(ACCEPTANCE_GATES, True), "teacher_calls": 0,
            "protocol_sha256": "a" * 64, "session_id": identifier,
            "retention": {"protected_sha256": content_sha(groups["new_validation"] + groups["replay_validation"] + regressions)}})
        acceptance = material["acceptance"] | {"report_sha256": file_sha(acceptance_path)}
        plan = {"session_id": identifier, "base_identity": identity, "arguments": arguments,
            "four_removed_duplicates": sorted(SUPERSEDED_DUPLICATES),
            "superseded_source_ids": material["superseded_ids"],
            "old_known_development_regressions": regressions[:8],
            "production_active_hashes": active_hashes(SimpleNamespace(store=production_store, branch=None))}
        if "field_loss_policy" in material:
            plan["field_loss_policy"] = copy.deepcopy(material["field_loss_policy"])
        before = read_upgrade_records(root)
        self.assertEqual(before, [])
        result = publish_isolated_core(source, production, production_store, plan, groups, regressions,
                                      material["source_rows"], acceptance_path, acceptance)
        self.assertTrue(result["product_accepted"])
        self.assertEqual(json.loads((production_store / metadata["report"]).read_text()), report)
        after = read_upgrade_records(root)
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0]["kind"], "core")
        self.assertEqual(after[0]["training_samples"], len(groups["new_training"]))
        self.assertEqual(after[0]["validation_samples"], len(groups["new_validation"]))
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from run_identity_coverage import recover_prepared_publication
        resume_report = {"publication_prepared": {"name": acceptance_path.name, "sha256": file_sha(acceptance_path)},
                         "protocol_sha256": "a" * 64, "passed": False}
        self.assertTrue(recover_prepared_publication(plan | {"production_root": str(root)}, root, resume_report))
        self.assertTrue(resume_report["publication_recovered_without_model"])
        self.assertTrue(resume_report["passed"])
        self.assertEqual(read_upgrade_records(root), after)


if __name__ == "__main__":
    unittest.main()
