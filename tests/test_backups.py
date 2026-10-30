"""Offline recovery contracts; no production files or model allocation."""
import hashlib
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from luban.backups import list_training_backups, restore_training_backup, save_training_backup


def write_tensors(path, shapes):
    offset, header = 0, {}
    for name, shape in shapes.items():
        size = 4
        for n in shape:
            size *= n
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Path(self.temp.name)
        self.identity = "base-fixture"

    def state(self, expert="learned-fixture", router=True):
        layer = self.store / "layer-27"
        layer.mkdir(exist_ok=True)
        weights = layer / (expert + ".safetensors")
        write_tensors(weights, {"gate.weight": [32, 4], "up.weight": [32, 4], "down.weight": [4, 32]})
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        (layer / (expert + ".json")).write_text(json.dumps({"identity": self.identity + ":layer-27",
            "hidden": 4, "width": 32, "sha256": digest, "requires_gate": True}))
        (self.store / "active.json").write_text(json.dumps({"id": expert, "width": 32, "identity": self.identity}))
        if router:
            weights = self.store / "router.safetensors"
            write_tensors(weights, {"projection.weight": [3, 4], "projection.bias": [3]})
            (self.store / "router.json").write_text(json.dumps({"identity": self.identity,
                "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "gates_expert": True,
                "expert_id": expert, "expert_sha256": digest}))

    def immutable_router(self, run="111111111111"):
        self.state()
        version = "router-" + run
        weights = self.store / (version + ".safetensors")
        weights.write_bytes((self.store / "router.safetensors").read_bytes())
        metadata = json.loads((self.store / "router.json").read_text())
        metadata.update(id=version, core_version=None, calibrated=True)
        meta = self.store / (version + ".json")
        meta.write_text(json.dumps(metadata))
        (self.store / "active-router.json").write_text(json.dumps({"id": version,
            "identity": self.identity, "core_version": None, "sha256": metadata["sha256"],
            "metadata_sha256": hashlib.sha256(meta.read_bytes()).hexdigest()}))
        return version

    def test_immutable_router_backup_restores_pointer_and_selected_files(self):
        version = self.immutable_router()
        old_pointer = (self.store / "active-router.json").read_bytes()
        backup = save_training_backup(self.store, self.identity, "calibrate")
        manifest = json.loads((self.store / "training-backups" / backup / "manifest.json").read_text())
        self.assertIn(version + ".json", manifest["files"])
        self.assertNotIn("router.json", manifest["files"])
        self.immutable_router("222222222222")
        (self.store / (version + ".safetensors")).write_bytes(b"mutated")
        restore_training_backup(self.store, self.identity, backup)
        self.assertEqual((self.store / "active-router.json").read_bytes(), old_pointer)
        self.assertTrue(list_training_backups(self.store, self.identity)[0]["valid"])
        self.assertFalse((self.store / "router.json").exists())
        self.assertTrue((self.store / "router-222222222222.safetensors").exists())

    def test_immutable_router_metadata_sha_rejected_before_live_changes(self):
        version = self.immutable_router()
        backup = save_training_backup(self.store, self.identity, "calibrate")
        root = self.store / "training-backups" / backup
        metadata = root / (version + ".json")
        metadata.write_text(metadata.read_text() + " ")
        self.refresh_manifest_files(backup, [metadata.name])
        before = self.live_bytes()
        with self.assertRaisesRegex(ValueError, "快脑校验"):
            restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(before, self.live_bytes())

    def test_core_retention_expert_immutable_router_exact_ten_files(self):
        self.immutable_router()
        self.core_state()
        self.retention_state()
        backup = save_training_backup(self.store, self.identity, "calibrate")
        manifest = json.loads((self.store / "training-backups" / backup / "manifest.json").read_text())
        self.assertEqual(len(manifest["files"]), 10)
        self.assertTrue(list_training_backups(self.store, self.identity)[0]["valid"])
        restore_training_backup(self.store, self.identity, backup)

    def live_bytes(self):
        return {p.relative_to(self.store).as_posix(): p.read_bytes() for p in self.store.rglob('*')
                if p.is_file() and 'training-backups' not in p.parts}

    def core_state(self, version="core-111111111111", marker=1, dtype=None):
        import torch
        from safetensors.torch import save_file
        parameter = "model.layers.27.mlp.base.down_proj.weight"
        tensor = torch.arange(8, dtype=dtype or torch.float32).reshape(4, 2) + marker
        weights = self.store / (version + ".safetensors")
        save_file({parameter: tensor}, str(weights))
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        (self.store / (version + ".json")).write_text(json.dumps({"version": version,
            "identity": self.identity, "parameter": parameter, "shape": [4, 2], "dtype": str(tensor.dtype),
            "sha256": digest, "format": "cumulative core tensor replacement", "trained_parameters": 8}))
        (self.store / "active-core.json").write_text(json.dumps({"version": version,
            "identity": self.identity, "sha256": digest}))
        return tensor

    def reload_core(self):
        import torch
        from torch import nn
        from luban.evolution import load_active_core
        model = nn.Module()
        model.model = nn.Module()
        layers = [nn.Identity() for _ in range(28)]
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.base = nn.Module()
        layer.mlp.base.down_proj = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            layer.mlp.base.down_proj.weight.fill_(-10)
        layers[-1] = layer
        model.model.layers = nn.ModuleList(layers)
        runtime = SimpleNamespace(model=model, decoder=model.model, identity=self.identity, store=self.store)
        load_active_core(runtime)
        return runtime, layer.mlp.base.down_proj.weight.detach()

    def retention_state(self, version="core-111111111111"):
        from luban.retention import save_core_retention
        groups = {}
        for role in ("new", "replay"):
            for purpose in ("training", "validation"):
                key = role + "_" + purpose
                groups[key] = [{"id": key, "prompt": key, "answer": "confirmed answer",
                    "source": "human", "purpose": purpose, "approved": 1, "training_allowed": 1,
                    "provenance": json.dumps({"evolution_role": role, "split_family": key})}]
        return save_core_retention(SimpleNamespace(store=self.store, identity=self.identity), version, groups)

    def refresh_manifest_files(self, backup, names):
        root = self.store / "training-backups" / backup
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        for name in names:
            target = root / name
            manifest["files"][name] = {"size": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        path.write_text(json.dumps(manifest))

    def test_retains_latest_three_in_reverse_order(self):
        self.state()
        ids = [save_training_backup(self.store, self.identity, "learn") for _ in range(4)]
        self.assertFalse((self.store / 'training-backups' / ids[0]).exists())
        self.assertEqual([b['id'] for b in list_training_backups(self.store, self.identity)], ids[:0:-1])
        self.assertTrue(all(b['valid'] for b in list_training_backups(self.store, self.identity)))

    def test_pruning_tolerates_disappeared_sidecar_but_not_permission_errors(self):
        self.state()
        for _ in range(3):
            save_training_backup(self.store, self.identity, "learn")
        remove = shutil.rmtree

        def disappeared(path, *args, **kwargs):
            remove(path, *args, **kwargs)
            if "onexc" in kwargs:
                kwargs["onexc"](os.unlink, Path(path) / "._sidecar", FileNotFoundError("already removed"))

        with patch("luban.backups.shutil.rmtree", side_effect=disappeared):
            save_training_backup(self.store, self.identity, "learn")
        self.assertEqual(len(list_training_backups(self.store, self.identity)), 3)

        def denied(path, *args, **kwargs):
            if "onexc" in kwargs:
                kwargs["onexc"](os.unlink, path, PermissionError("cannot delete"))
            else:
                remove(path, *args, **kwargs)

        with patch("luban.backups.shutil.rmtree", side_effect=denied), self.assertRaises(PermissionError):
            save_training_backup(self.store, self.identity, "learn")

    def test_restores_exact_active_expert_and_head_even_when_hibernating(self):
        # A sleeping runtime still has its persisted active pointer and files.
        self.state()
        original = self.live_bytes()
        backup = save_training_backup(self.store, self.identity, "calibrate")
        for name in original:
            (self.store / name).write_bytes(b'changed')
        restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), original)
        self.assertTrue(list_training_backups(self.store, self.identity)[0]['valid'])

    def test_empty_snapshot_removes_active_and_router_but_preserves_unrelated_files(self):
        backup = save_training_backup(self.store, self.identity, "learn")
        self.state()
        (self.store / 'keep.json').write_text('{}')
        restore_training_backup(self.store, self.identity, backup)
        for name in ('active.json', 'router.json', 'router.safetensors'):
            self.assertFalse((self.store / name).exists())
        self.assertTrue((self.store / 'keep.json').exists())
        self.assertTrue((self.store / 'layer-27/learned-fixture.safetensors').exists())

    def test_expert_without_router_restores_missing_head(self):
        self.state(router=False)
        backup = save_training_backup(self.store, self.identity, 'learn')
        self.state()
        restore_training_backup(self.store, self.identity, backup)
        self.assertFalse((self.store / 'router.json').exists())
        self.assertFalse((self.store / 'router.safetensors').exists())
        self.assertTrue((self.store / 'active.json').exists())

    def test_stale_head_preserves_original_binding_with_or_without_active_expert(self):
        self.state()
        router_path = self.store / 'router.json'
        router = json.loads(router_path.read_text())
        router.update(expert_id='learned-previous', expert_sha256='a' * 64)
        router_path.write_text(json.dumps(router))
        original_router = router_path.read_bytes()
        for has_active in (True, False):
            with self.subTest(has_active=has_active):
                if not has_active:
                    (self.store / 'active.json').unlink()
                backup = save_training_backup(self.store, self.identity, 'calibrate')
                router_path.write_bytes(b'changed current head')
                restore_training_backup(self.store, self.identity, backup)
                self.assertEqual(router_path.read_bytes(), original_router)
                self.assertEqual((self.store / 'active.json').exists(), has_active)
                self.assertTrue(list_training_backups(self.store, self.identity)[0]['valid'])

    def test_damage_and_wrong_identity_rejected_before_any_live_write(self):
        self.state()
        backup = save_training_backup(self.store, self.identity, 'learn')
        before = self.live_bytes()
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, 'different-base', backup)
        (self.store / 'training-backups' / backup / 'router.safetensors').write_bytes(b'broken')
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), before)
        self.assertFalse(list_training_backups(self.store, self.identity)[0]['valid'])

    def test_path_traversal_and_manifest_extra_files_are_rejected(self):
        for backup_id in ('../outside', '/absolute', '.', 'x/y', None):
            with self.subTest(backup_id=backup_id), self.assertRaises(ValueError):
                restore_training_backup(self.store, self.identity, backup_id)
        backup = save_training_backup(self.store, self.identity, 'learn')
        manifest_path = self.store / 'training-backups' / backup / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['../settings.json'] = {'size': 0, 'sha256': 'invalid'}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, self.identity, backup)

    def test_invalid_tensor_shape_rejected_even_with_updated_file_checksums(self):
        self.state()
        backup = save_training_backup(self.store, self.identity, 'learn')
        root = self.store / 'training-backups' / backup
        weights = root / 'router.safetensors'
        write_tensors(weights, {'projection.weight': [2, 4], 'projection.bias': [2]})
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        router_path = root / 'router.json'
        router = json.loads(router_path.read_text())
        router['sha256'] = digest
        router_path.write_text(json.dumps(router))
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for path in (weights, router_path):
            manifest['files'][path.name] = {'size': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest))
        before = self.live_bytes()
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), before)

    def test_incomplete_manifest_rejected_before_publication(self):
        self.state()
        backup = save_training_backup(self.store, self.identity, 'learn')
        manifest_path = self.store / 'training-backups' / backup / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        del manifest['kind']
        manifest_path.write_text(json.dumps(manifest))
        (self.store / 'active.json').write_bytes(b'current data')
        before = self.live_bytes()
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), before)

    def test_restore_write_failure_rolls_back_and_does_not_change_backup(self):
        self.state()
        backup = save_training_backup(self.store, self.identity, 'learn')
        for name in self.live_bytes():
            (self.store / name).write_bytes(b'new live data')
        before = self.live_bytes()
        real_replace, count = os.replace, 0
        def fail_second(source, target):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('simulated publication failure')
            return real_replace(source, target)
        with patch('luban.backups.os.replace', side_effect=fail_second), self.assertRaises(OSError):
            restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), before)
        self.assertTrue(list_training_backups(self.store, self.identity)[0]['valid'])

    def test_incomplete_live_head_fails_without_publishing_or_pruning(self):
        self.state()
        backup = save_training_backup(self.store, self.identity, 'learn')
        (self.store / 'router.safetensors').unlink()
        with self.assertRaises(ValueError):
            save_training_backup(self.store, self.identity, 'calibrate')
        self.assertEqual([b['id'] for b in list_training_backups(self.store, self.identity)], [backup])

    def test_restores_core_pointer_tensor_and_actual_reloaded_core(self):
        import torch
        self.state()
        original_tensor = self.core_state()
        original = self.live_bytes()
        backup = save_training_backup(self.store, self.identity, "learn")
        manifest = json.loads((self.store / "training-backups" / backup / "manifest.json").read_text())
        self.assertEqual(len(manifest["files"]), 8)
        self.assertEqual(list_training_backups(self.store, self.identity)[0]["core_version"], "core-111111111111")
        self.core_state("core-222222222222", marker=100)
        for name in original:
            if name != "active-core.json":
                (self.store / name).write_bytes(b"changed live state")
        published = []
        real_replace = os.replace
        def record(source, target):
            published.append(Path(target).name)
            return real_replace(source, target)
        with patch("luban.backups.os.replace", side_effect=record):
            restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(published[-1], "active-core.json")
        for name, data in original.items():
            self.assertEqual((self.store / name).read_bytes(), data)
        self.assertTrue((self.store / "core-222222222222.safetensors").exists())
        runtime, actual_core = self.reload_core()
        torch.testing.assert_close(actual_core, original_tensor, atol=0, rtol=0)
        self.assertEqual(runtime.active_core, "core-111111111111")

    def test_old_empty_backup_removes_core_pointer_and_reloads_original_but_keeps_history(self):
        import torch
        backup = save_training_backup(self.store, self.identity, "learn")
        self.core_state()
        restore_training_backup(self.store, self.identity, backup)
        self.assertFalse((self.store / "active-core.json").exists())
        self.assertTrue((self.store / "core-111111111111.json").exists())
        self.assertTrue((self.store / "core-111111111111.safetensors").exists())
        runtime, actual_core = self.reload_core()
        self.assertIsNone(runtime.active_core)
        torch.testing.assert_close(actual_core, torch.full((4, 2), -10.), atol=0, rtol=0)

    def test_retention_is_backed_up_and_restored_with_nine_bound_files(self):
        self.state()
        self.core_state()
        retention = self.retention_state()
        snapshot = Path(retention["path"])
        original = snapshot.read_bytes()
        metadata_path = self.store / "core-111111111111.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["retention_sha256"] = retention["file_sha256"]
        metadata_path.write_text(json.dumps(metadata))
        backup = save_training_backup(self.store, self.identity, "learn")
        manifest = json.loads((self.store / "training-backups" / backup / "manifest.json").read_text())
        self.assertEqual(len(manifest["files"]), 9)
        snapshot.unlink()
        restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(snapshot.read_bytes(), original)
        self.assertTrue(list_training_backups(self.store, self.identity)[0]["valid"])

    def test_missing_declared_retention_cannot_publish_backup(self):
        self.core_state()
        metadata_path = self.store / "core-111111111111.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["retention_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "保护材料"):
            save_training_backup(self.store, self.identity, "learn")

    def test_corrupt_retention_is_rejected_before_any_live_restore(self):
        self.core_state()
        retention = self.retention_state()
        backup = save_training_backup(self.store, self.identity, "learn")
        name = Path(retention["path"]).name
        path = self.store / "training-backups" / backup / name
        payload = json.loads(path.read_text())
        payload["groups"]["new_validation"][0]["answer"] = "corrupted answer"
        path.write_text(json.dumps(payload))
        self.refresh_manifest_files(backup, [name])
        before = self.live_bytes()
        with self.assertRaisesRegex(ValueError, "校验"):
            restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), before)

    def test_core_missing_corrupt_and_unsafe_versions_rejected_before_live_writes(self):
        self.core_state()
        backup = save_training_backup(self.store, self.identity, "learn")
        root = self.store / "training-backups" / backup
        before = self.live_bytes()
        weights = root / "core-111111111111.safetensors"
        original = weights.read_bytes()
        weights.unlink()
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, self.identity, backup)
        weights.write_bytes(original + b"corrupted")
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, self.identity, backup)
        weights.write_bytes(original)
        pointer = root / "active-core.json"
        saved_pointer = json.loads(pointer.read_text())
        for version in ("../outside", "/absolute", "core-123", "core-111111111111.json"):
            with self.subTest(version=version):
                pointer.write_text(json.dumps(saved_pointer | {"version": version}))
                self.refresh_manifest_files(backup, ["active-core.json"])
                with self.assertRaises(ValueError):
                    restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), before)
        self.assertFalse(list_training_backups(self.store, self.identity)[0]["valid"])

    def test_core_shape_dtype_binding_and_single_tensor_checked_despite_updated_hashes(self):
        import torch
        from safetensors.torch import save_file
        parameter = "model.layers.27.mlp.base.down_proj.weight"
        for tensors in ({parameter: torch.zeros(3, 2)}, {parameter: torch.zeros(4, 2, dtype=torch.bfloat16)},
                        {parameter: torch.zeros(4, 2), "unbound.weight": torch.zeros(1)},
                        {parameter: torch.full((4, 2), float("nan"))}):
            with self.subTest(tensors=list(tensors)):
                self.core_state()
                backup = save_training_backup(self.store, self.identity, "learn")
                root = self.store / "training-backups" / backup
                weights = root / "core-111111111111.safetensors"
                save_file(tensors, str(weights))
                digest = hashlib.sha256(weights.read_bytes()).hexdigest()
                for name in ("core-111111111111.json", "active-core.json"):
                    path = root / name
                    data = json.loads(path.read_text())
                    data["sha256"] = digest
                    path.write_text(json.dumps(data))
                self.refresh_manifest_files(backup, ["core-111111111111.safetensors",
                    "core-111111111111.json", "active-core.json"])
                before = self.live_bytes()
                with self.assertRaisesRegex(ValueError, "真实尺寸或精度|非有限"):
                    restore_training_backup(self.store, self.identity, backup)
                self.assertEqual(self.live_bytes(), before)
        self.core_state(dtype=torch.bfloat16)
        backup = save_training_backup(self.store, self.identity, "learn")
        self.assertTrue(list_training_backups(self.store, self.identity)[0]["valid"])

    def test_core_pointer_publication_failure_rolls_back_all_live_bytes(self):
        self.state()
        self.core_state()
        backup = save_training_backup(self.store, self.identity, "calibrate")
        self.core_state("core-222222222222", marker=200)
        (self.store / "active.json").write_bytes(b"previous live expert pointer")
        before = self.live_bytes()
        real_replace = os.replace
        def fail_core_pointer(source, target):
            if Path(target).name == "active-core.json":
                raise OSError("core pointer publication failed")
            return real_replace(source, target)
        with patch("luban.backups.os.replace", side_effect=fail_core_pointer):
            with self.assertRaisesRegex(OSError, "core pointer publication failed"):
                restore_training_backup(self.store, self.identity, backup)
        self.assertEqual(self.live_bytes(), before)
        self.assertTrue(list_training_backups(self.store, self.identity)[0]["valid"])

    def test_core_metadata_and_pointer_binding_checked_even_with_updated_manifest(self):
        self.core_state()
        for changed in ({"identity": "other-base"}, {"version": "core-222222222222"},
                        {"parameter": "unrelated.weight"}, {"shape": [True, 2]}, {"dtype": []},
                        {"trained_parameters": 9}, {"format": "adapter"}):
            with self.subTest(changed=changed):
                backup = save_training_backup(self.store, self.identity, "learn")
                metadata = self.store / "training-backups" / backup / "core-111111111111.json"
                metadata.write_text(json.dumps(json.loads(metadata.read_text()) | changed))
                self.refresh_manifest_files(backup, [metadata.name])
                before = self.live_bytes()
                with self.assertRaises(ValueError):
                    restore_training_backup(self.store, self.identity, backup)
                self.assertEqual(self.live_bytes(), before)
        backup = save_training_backup(self.store, self.identity, "learn")
        pointer = self.store / "training-backups" / backup / "active-core.json"
        pointer.write_text(json.dumps(json.loads(pointer.read_text()) | {"sha256": "a" * 64}))
        self.refresh_manifest_files(backup, [pointer.name])
        with self.assertRaises(ValueError):
            restore_training_backup(self.store, self.identity, backup)

    def test_missing_live_core_files_and_core_symlinks_cannot_create_backups(self):
        self.core_state()
        weights = self.store / "core-111111111111.safetensors"
        original = weights.read_bytes()
        weights.unlink()
        with self.assertRaisesRegex(ValueError, "完整活动核心"):
            save_training_backup(self.store, self.identity, "learn")
        outside = self.store / "external.safetensors"
        outside.write_bytes(original)
        weights.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            save_training_backup(self.store, self.identity, "learn")
        self.assertEqual(list_training_backups(self.store, self.identity), [])


if __name__ == '__main__':
    unittest.main()
