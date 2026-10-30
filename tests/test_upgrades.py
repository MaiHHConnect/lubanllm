"""Production-history upgrade accounting uses isolated disk fixtures."""
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from luban.upgrades import read_upgrade_records


def upgrade_fixture(root, index=0, weights=None, layer=27):
    run_id = f"{index:012x}"
    store = Path(root) / "modules/app"
    branch = store / f"layer-{layer}"
    branch.mkdir(parents=True, exist_ok=True)
    data = weights if weights is not None else f"weight-{index}".encode()
    (branch / f"learned-{run_id}.safetensors").write_bytes(data)
    metadata = {"identity": "a" * 64 + f":layer-{layer}", "hidden": 8, "width": 32,
                "sha256": hashlib.sha256(data).hexdigest()}
    meta_path = branch / f"learned-{run_id}.json"
    meta_path.write_text(json.dumps(metadata))
    candidate = {"width": 32, "parameters": 768, "validation_loss": 0.8,
                 "replay_loss": 0.9, "accepted": True, "objective": 0.816}
    report = {"id": run_id, "promoted": True, "active_expert": f"learned-{run_id}", "chosen_width": 32,
              "baseline_validation_loss": 1.0, "training_ids": [1, 2], "validation_ids": [3],
              "candidates": [candidate], "started": time.time() - 30, "elapsed_seconds": 10}
    report_path = store / f"learning-{run_id}.json"
    report_path.write_text(json.dumps(report))
    return report_path, meta_path, report, metadata


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_entire_history_deduplicates_weights_without_private_sample_content(self):
        for i in range(65):
            upgrade_fixture(self.root, i)
        upgrade_fixture(self.root, 65, weights=b"weight-0", layer=9)
        records = read_upgrade_records(self.root)
        self.assertEqual(len(records), 65)
        self.assertEqual(set(records[0]), {"base_sha256", "weights_sha256", "baseline_validation_loss", "validation_loss",
                                         "replay_before", "replay_loss", "training_samples", "validation_samples", "completed_at"})
        self.assertIsNone(records[0]["replay_before"])
        self.assertIsNone(records[0]["replay_loss"])
        self.assertEqual(records[0]["training_samples"], 2)

    def test_rejected_and_non_production_records_do_not_count(self):
        path, _, report, _ = upgrade_fixture(self.root)
        report["promoted"] = False
        path.write_text(json.dumps(report))
        for name in ("router.json", "learning-invalid-id.json", "evolution-000000000001.json"):
            (path.parent / name).write_text("broken")
        backup = path.parent / "training-backups/old"
        backup.mkdir(parents=True)
        (backup / "learning-000000000000.json").write_text("broken")
        self.assertEqual(read_upgrade_records(self.root), [])

    def test_explicit_replay_baseline_must_pass(self):
        path, _, report, _ = upgrade_fixture(self.root)
        report["replay_before"] = 1.0
        path.write_text(json.dumps(report))
        self.assertEqual(read_upgrade_records(self.root)[0]["replay_before"], 1.0)
        report["candidates"][0]["replay_loss"] = 1.2
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, "无法完整核验"):
            read_upgrade_records(self.root)

    def test_promoted_corruption_blocks_entire_count(self):
        upgrade_fixture(self.root)
        path, meta_path, report, metadata = upgrade_fixture(self.root, 1)
        mutations = [("promoted", "yes"), ("active_expert", "learned-wrong"), ("baseline_validation_loss", float("nan")),
                     ("chosen_width", 64), ("training_ids", []), ("validation_ids", [1]), ("training_ids", [1, 1]), ("started", time.time() + 1000)]
        original = json.loads(json.dumps(report))
        for key, value in mutations:
            with self.subTest(key=key):
                broken = {**original, key: value}
                path.write_text(json.dumps(broken))
                with self.assertRaises(RuntimeError):
                    read_upgrade_records(self.root)
        path.write_text(json.dumps(original))
        for field, value in (("sha256", "b" * 64), ("identity", "wrong"), ("hidden", True)):
            meta_path.write_text(json.dumps({**metadata, field: value}))
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                read_upgrade_records(self.root)
        meta_path.write_text(json.dumps(metadata))
        meta_path.with_suffix(".safetensors").unlink()
        with self.assertRaises(RuntimeError):
            read_upgrade_records(self.root)

    def test_matching_candidate_requires_acceptance_and_improvement(self):
        path, _, report, _ = upgrade_fixture(self.root)
        for key, value in (("accepted", False), ("validation_loss", 0.999), ("replay_loss", float("inf")), ("parameters", 1)):
            candidate = {**report["candidates"][0], key: value}
            path.write_text(json.dumps({**report, "candidates": [candidate]}))
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                read_upgrade_records(self.root)

    def journal(self, report, state="committed", *, scope=None, accepted=False):
        from luban.memory import Memory
        from luban.training_checkpoint import _schema
        body = {"binding": {"identity": "a" * 64}, "commit": {"report": report}}
        if scope:
            body.update(scope=scope, product_accepted=accepted)
        with Memory(self.root).connect() as db:
            _schema(db)
            db.execute("INSERT OR REPLACE INTO training_runs VALUES(?,?,?,?,?,?)",
                (report["id"], "a" * 64, "learning", state, json.dumps(body), time.time()))

    def test_pending_promoted_artifacts_never_count_as_adoption(self):
        _, _, report, _ = upgrade_fixture(self.root)
        for state in ("queued", "running", "paused", "committing", "rejected"):
            self.journal(report, state)
            with self.subTest(state=state):
                self.assertEqual(read_upgrade_records(self.root), [])
        self.journal(report)
        self.assertEqual(len(read_upgrade_records(self.root)), 1)

    def test_research_trial_not_counted_until_product_accepted(self):
        _, _, report, _ = upgrade_fixture(self.root)
        self.journal(report, scope="research")
        self.assertEqual(read_upgrade_records(self.root), [])
        self.journal(report, scope="research", accepted=True)
        self.assertEqual(len(read_upgrade_records(self.root)), 1)

    def test_qwen_store_counted_and_committed_report_mismatch_rejected(self):
        import shutil
        path, _, report, _ = upgrade_fixture(self.root)
        destination = self.root / "modules/luban0-v1"
        shutil.move(str(path.parent), destination)
        self.journal(report)
        self.assertEqual(len(read_upgrade_records(self.root)), 1)
        path = destination / path.name
        path.write_text(json.dumps({**report, "elapsed_seconds": 11}))
        with self.assertRaisesRegex(RuntimeError, "无法完整核验"):
            read_upgrade_records(self.root)

    def test_unmounted_storage_does_not_create_or_report_zero(self):
        with patch("luban.upgrades.require_storage", side_effect=RuntimeError("存储盘未连接")):
            with self.assertRaisesRegex(RuntimeError, "存储盘未连接"):
                read_upgrade_records(self.root)

    def test_factory_learning_validates_before_excluding_personal_growth(self):
        from luban.memory import Memory
        path, _, report, _ = upgrade_fixture(self.root)
        self.journal(report)
        with Memory(self.root).connect() as db:
            body = json.loads(db.execute("SELECT body FROM training_runs").fetchone()[0])
            body['factory_baseline'] = True
            body['commit']['event_kind'] = 'learning'
            db.execute("UPDATE training_runs SET body=?", (json.dumps(body),))
            db.execute("INSERT INTO events VALUES(?,?,?,?)", ('training:' + report['id'],
                       'factory:learning', json.dumps(report), 1))
        self.assertEqual(read_upgrade_records(self.root), [])
        # Even factory weights must pass the same immutable report verification.
        path.with_name('layer-27').joinpath('learned-' + report['id'] + '.safetensors').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, "无法完整核验"):
            read_upgrade_records(self.root)

    def test_factory_event_requires_matching_marker_and_report(self):
        from luban.memory import Memory
        _, _, report, _ = upgrade_fixture(self.root)
        self.journal(report)
        with Memory(self.root).connect() as db:
            db.execute("INSERT INTO events VALUES(?,?,?,?)", ('training:' + report['id'],
                       'factory:learning', json.dumps(report), 1))
        with self.assertRaises(RuntimeError):
            read_upgrade_records(self.root)
        with Memory(self.root).connect() as db:
            body = json.loads(db.execute("SELECT body FROM training_runs").fetchone()[0])
            body.update(factory_baseline=True)
            body['commit']['event_kind'] = 'learning'
            db.execute("UPDATE training_runs SET body=?", (json.dumps(body),))
            db.execute("UPDATE events SET body='{}'")
        with self.assertRaises(RuntimeError):
            read_upgrade_records(self.root)


if __name__ == "__main__":
    unittest.main()
