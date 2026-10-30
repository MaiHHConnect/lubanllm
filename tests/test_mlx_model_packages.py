"""Private migration retains MLX proof semantics and rejects foreign activation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from luban import model_packages as p, mlx_model_packages as mlx
from luban.settings import private_json, Settings
from test_mlx_net_learning import fixture


class MLXModelPackagesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.store = self.root / "modules/luban0-v1"
        self.store.mkdir(parents=True)
        self.identifier = "123456abcdef"
        expert, router, core = "learned-" + self.identifier, "router-" + self.identifier, "core-abcdef123456"
        flags = dict(execution_scope="product", inference_engine="mlx-q4", quality_admitted=True,
                     training_admitted=True, core_version=core)
        self.active = dict(id=expert, identity=p.TRUSTED_IDENTITY, width=64, **flags)
        self.router = dict(id=router, identity=p.TRUSTED_IDENTITY, sha256="b" * 64, feature_source="mlx-q4-prefix",
                           expert_id=expert, expert_sha256="c" * 64, **flags)
        self.core = dict(version=core, identity=p.TRUSTED_IDENTITY, sha256="d" * 64)
        private_json(self.store / "active.json", self.active)
        private_json(self.store / "active-core.json", self.core)
        private_json(self.store / "active-router.json", self.router)
        private_json(self.store / (router + ".json"), self.router)
        private_json(self.store / "layer-23" / (expert + ".json"), dict(
            identity=p.TRUSTED_IDENTITY + ":layer-23", hidden=2048, width=64, sha256="c" * 64, requires_gate=True, **flags))
        report = fixture()
        report.update(base_identity=p.TRUSTED_IDENTITY, identity=p.TRUSTED_IDENTITY, core_version=core,
                      active_expert=expert, router_id=router, records_sha256=hashlib.sha256(p._encoded(report["records"])).hexdigest())
        self.report_name = "learning-" + self.identifier + ".json"
        private_json(self.store / self.report_name, report)
        self.proof = dict(schema="mlx-expert-learning-v1", scope="product", quality_admitted=True,
                          training_admitted=True, identity=p.TRUSTED_IDENTITY, core_version=core,
                          core_file_sha256="d" * 64, expert_id=expert, expert_file_sha256="c" * 64,
                          router_id=router, router_sha256="b" * 64, width=64, route_checked=True,
                          worker_binary_sha256="e" * 64, mlx_model_sha256=mlx.MLX_FILES["model.safetensors"]["sha256"],
                          report=self.report_name, report_sha256=p._sha(self.store / self.report_name))
        private_json(self.root / mlx.PROOFS[0], self.proof)
        arguments = dict(protocol=mlx.PROTOCOL, learning_policy=report["learning_policy"], steps=64, seed=42)
        binding = {key: self.proof[key] for key in ("identity", "core_version", "core_file_sha256",
                                                  "mlx_model_sha256", "worker_binary_sha256", "router_sha256")}
        binding.update(arguments=arguments, records_sha256=report["records_sha256"])
        artifacts = {"layer-23/" + expert + suffix: "a" * 64 for suffix in (".json", ".safetensors")}
        artifacts.update({router + suffix: "b" * 64 for suffix in (".json", ".safetensors")})
        artifacts[self.report_name] = self.proof["report_sha256"]
        self.run = dict(id=self.identifier, kind="learning", state="committed", scope="product", product_accepted=False,
                        binding=binding, records=report["records"], started=1, arguments=arguments,
                        commit=dict(report=report, artifacts=artifacts, event_kind="learning", pointers=[
                            dict(name=name, before=None, after=value) for name, value in zip(
                                mlx.POINTER_NAMES, (self.active, self.router, self.proof))]))

    def test_formal_proof_keeps_real_net_gain_and_source_binding(self):
        mlx.validate_run(self.run, set())
        self.assertEqual(mlx._proof_runtime(self.store, self.proof)["learning_progress"]["net_gain"], 2)
        for key, value in (("mlx_model_sha256", "f" * 64), ("core_file_sha256", "a" * 64),
                           ("quality_admitted", False), ("route_checked", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                mlx._proof_runtime(self.store, {**self.proof, key: value})

    def test_forged_net_gain_cannot_be_migrated_even_when_report_hash_matches(self):
        report = copy.deepcopy(self.run["commit"]["report"])
        report["learning_progress"]["net_gain"] = 90
        private_json(self.store / self.report_name, report)
        proof = {**self.proof, "report_sha256": p._sha(self.store / self.report_name)}
        with self.assertRaisesRegex(ValueError, "净增统计"):
            mlx._proof_runtime(self.store, proof)

    def test_migration_changes_only_pointer_paths_not_admission(self):
        source = SimpleNamespace(store=self.store, settings=SimpleNamespace(snapshot=lambda: {"storage_root": str(self.root)}))
        installed = mlx.installed_run(self.run, self.root)
        # portable_run receives the body-shaped commit in the real journal path.
        body_shaped = copy.deepcopy(installed)
        self.assertEqual(mlx.portable_run(body_shaped, installed, source), self.run)
        destination = self.root / "new-machine"
        moved = mlx.installed_run(self.run, destination)
        self.assertEqual(moved["binding"], installed["binding"])
        self.assertEqual(moved["commit"]["report"], installed["commit"]["report"])
        self.assertEqual(moved["commit"]["pointers"][2]["after"], self.proof)
        self.assertEqual(moved["commit"]["pointers"][2]["path"], str(destination / mlx.PROOFS[0]))

    def test_journal_rejects_missing_foreign_and_unbound_proofs(self):
        for mutate in (lambda r: r["commit"]["pointers"].pop(),
                       lambda r: r["commit"]["pointers"][2].update(name="../../outside"),
                       lambda r: r["binding"].update(worker_binary_sha256="f" * 64),
                       lambda r: r["commit"]["artifacts"].update({"unrelated.json": "a" * 64})):
            run = copy.deepcopy(self.run)
            mutate(run)
            with self.assertRaises(ValueError):
                mlx.validate_run(run, set())

    def test_factory_journal_keeps_proofs_and_records_across_private_migration(self):
        from luban.memory import Memory
        payload = dict(identity=p.TRUSTED_IDENTITY, runs=[self.run], samples=[], material_archives=[],
                       factory_runs=[self.identifier])
        original = copy.deepcopy(payload)
        p._validate_journal(payload)
        self.assertEqual(p._install_journal(self.root, payload), dict(router=0, learning=0, evolution=0))
        runtime = SimpleNamespace(memory=Memory(self.root), store=self.store,
            settings=SimpleNamespace(snapshot=lambda: dict(storage_root=str(self.root), native_inference_precision="mlx-q4")))
        self.assertEqual(p._journal(runtime), payload)
        with runtime.memory.connect() as db:
            event = db.execute("SELECT kind,body FROM events").fetchone()
            self.assertEqual(event["kind"], "factory:learning")
            self.assertEqual(json.loads(event["body"]), self.run["commit"]["report"])
            body = json.loads(db.execute("SELECT body FROM training_runs").fetchone()[0])
            self.assertIs(body["factory_baseline"], True)
            self.assertEqual(body["records"], self.run["records"])
            self.assertEqual(body["commit"]["pointers"][2]["after"], self.proof)
        self.assertEqual(payload, original)

    def test_factory_ids_must_be_unique_existing_adoptions(self):
        payload = dict(identity=p.TRUSTED_IDENTITY, runs=[self.run], samples=[], material_archives=[])
        for identifiers in ([self.identifier, self.identifier], ["ffffffffffff"], "123456abcdef", [None], {}):
            with self.subTest(identifiers=identifiers), self.assertRaisesRegex(ValueError, "出厂基线"):
                p._validate_journal({**payload, "factory_runs": identifiers})
        p._validate_journal({**payload, "factory_runs": []})

    def test_event_prefix_cannot_convert_normal_or_invalid_adoption_to_factory(self):
        from luban.memory import Memory
        payload = dict(identity=p.TRUSTED_IDENTITY, runs=[self.run], samples=[], material_archives=[])
        self.assertEqual(p._install_journal(self.root, payload)["learning"], 1)
        runtime = SimpleNamespace(memory=Memory(self.root), store=self.store,
            settings=SimpleNamespace(snapshot=lambda: dict(storage_root=str(self.root), native_inference_precision="mlx-q4")))
        self.assertNotIn("factory_runs", p._journal(runtime))
        with runtime.memory.connect() as db:
            db.execute("UPDATE events SET kind='factory:learning'")
        with self.assertRaisesRegex(ValueError, "持久事件"):
            p._journal(runtime)
        with runtime.memory.connect() as db:
            body = json.loads(db.execute("SELECT body FROM training_runs").fetchone()[0])
            body["factory_baseline"] = True
            db.execute("UPDATE training_runs SET body=?", (p._encoded(body).decode(),))
            db.execute("UPDATE events SET body='{}'")
        with self.assertRaisesRegex(ValueError, "持久事件"):
            p._journal(runtime)

    def test_engine_mismatch_rejects_before_changing_old_settings(self):
        binary = self.root / "worker"
        binary.write_bytes(b"synthetic executable identity; never launched")
        private_json(Path(str(binary) + ".json"), {"binary_sha256": "f" * 64})
        staged = dict(native_inference_precision="mlx-q4", native_mlx_model_path=str(self.root / "models/mlx"))
        with patch.object(mlx, "installed_binary", return_value=binary):
            with self.assertRaisesRegex(ValueError, "应用版本不同"):
                mlx.check_activation(self.root, staged)
            private_json(Path(str(binary) + ".json"), {"binary_sha256": self.proof["worker_binary_sha256"]})
            mlx.check_activation(self.root, staged)

    def test_weight_timestamp_preserves_verified_receipt_not_current_time(self):
        path = self.root / "models/mlx/model.safetensors"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
        private_json(path.parent / "download-receipt.json", {"finished": 1700000000.0})
        mlx.restore_weight_time(self.root)
        self.assertEqual(path.stat().st_mtime, 1700000000.0)

    def test_recovery_restores_engine_and_model_paths_together(self):
        settings = Settings(self.root / "settings")
        old = {key: settings.snapshot()[key] for key in (*p._PATH_KEYS, "native_inference_precision", "native_mlx_model_path")}
        new = {**old, "native_inference_precision": "mlx-q4", "native_mlx_model_path": str(self.root / "models/mlx")}
        settings.update(new)
        private_json(settings.home / "package-activation.json", dict(schema=p.SCHEMA, state="prepared", old=old, new=new))
        self.assertTrue(p.recover_activation(settings)["recovered"])
        self.assertEqual({key: settings.snapshot()[key] for key in old}, old)


if __name__ == "__main__":
    unittest.main()
