"""Exercise the shipped reconstruction implementation on real signed, isolated apps."""
import hashlib
import json
from pathlib import Path
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "darwin", "macOS signed-app updater")
class NativeDeltaReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="luban-native-delta-")
        cls.root = Path(cls.temporary.name)
        cls.driver = cls.root / "archive-driver"
        # Compile the actual files in place; no source snapshots or rewritten candidate.
        subprocess.run([
            "/usr/bin/xcrun", "--sdk", "macosx", "swiftc",
            str(ROOT / "native/UpdateArchive.swift"), str(ROOT / "tests/native_update_archive.swift"),
            "-O", "-o", str(cls.driver),
        ], check=True, capture_output=True)
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(cls.driver)], check=True, capture_output=True)
        cls.base = cls.root / "base/Luban.app"
        cls.target = cls.root / "target/Luban.app"
        cls.project = str(cls.root / "runtime")
        Path(cls.project).mkdir()
        for app, version, build in [(cls.base, "0.2.0", 2), (cls.target, "0.3.0", 3)]:
            resources = app / "Contents/Resources"
            binary = app / "Contents/MacOS/Luban"
            binary.parent.mkdir(parents=True)
            (resources / "backend/src/luban").mkdir(parents=True)
            # Use our freshly compiled native-architecture executable: macOS system
            # tools can be arm64e and intentionally fail an arm64 admission check.
            shutil.copyfile(cls.driver, binary)
            binary.chmod(0o755)
            with (app / "Contents/Info.plist").open("wb") as stream:
                plistlib.dump({
                    "CFBundleIdentifier": "net.maihh.luban.native-delta-test",
                    "CFBundleExecutable": "Luban", "CFBundlePackageType": "APPL",
                    "CFBundleShortVersionString": version, "CFBundleVersion": str(build),
                }, stream)
            for relative in ["release.json", "backend/release.json"]:
                (resources / relative).write_text(json.dumps({"version": version, "build": build}))
            (resources / "launcher.json").write_text(json.dumps({"project": cls.project}))
            (resources / "backend/src/luban/server.py").write_text(f"# isolated fixture {version}\n")
            (resources / "unchanged.txt").write_text("unchanged resource\n")
            (resources / "mode-only.txt").write_text("same bytes\n")
            (resources / "mode-only.txt").chmod(0o644 if build == 2 else 0o755)
            if build == 2:
                (resources / "old.py").write_text("# removed in new version\n")
            subprocess.run(["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)], check=True, capture_output=True)
            subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)], check=True, capture_output=True)
        cls.identity = {"bundle_identifier": "net.maihh.luban.native-delta-test", "version": "0.3.0", "build": 3, "project": cls.project}
        cls.baseline_inventory = cls.inventory(cls.base)
        target_inventory = cls.inventory(cls.target)
        cls.changed = sorted(name for name, value in target_inventory.items() if cls.baseline_inventory.get(name) != value)
        cls.patch = cls.root / "valid-delta.zip"
        cls.write_patch(cls.patch)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def inventory(app):
        return {str(path.relative_to(app)): (hashlib.sha256(path.read_bytes()).hexdigest(), bool(path.stat().st_mode & 0o111))
                for path in app.rglob("*") if path.is_file()}

    @classmethod
    def write_patch(cls, destination, *, override=None, link=False):
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in cls.changed:
                path = cls.target / name
                info = zipfile.ZipInfo("Luban.app/" + name)
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | (0o755 if path.stat().st_mode & 0o111 else 0o644)) << 16
                data = override[name] if override and name in override else path.read_bytes()
                archive.writestr(info, data)
            if link:
                info = zipfile.ZipInfo("Luban.app/Contents/Resources/linked.py")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, "unchanged.txt")

    def request(self, *, archive=None, base=None, removed=None, digest=None, identity=None):
        archive = archive or self.patch
        stage = self.root / str(uuid.uuid4())
        request = {"archive": str(archive), "base": str(base or self.base), "stage": str(stage),
                   "sha256": digest or hashlib.sha256(archive.read_bytes()).hexdigest(),
                   "removed": ["Contents/Resources/old.py"] if removed is None else removed,
                   "identity": identity or self.identity}
        completed = subprocess.run([str(self.driver)], input=json.dumps(request) + "\n", text=True,
                                   capture_output=True, check=True, timeout=30)
        result = json.loads(completed.stdout)
        # Every attempt must leave the source app's content and executable bits intact.
        self.assertEqual(self.inventory(self.base), self.baseline_inventory)
        self.assertTrue((self.base / "Contents/Resources/old.py").exists())
        return result, stage

    def test_signed_patch_reconstructs_target_deletes_old_and_keeps_base(self):
        result, stage = self.request()
        self.assertTrue(result["accepted"], result)
        rebuilt = stage / "Luban.app"
        self.assertEqual(self.inventory(rebuilt), self.inventory(self.target))
        self.assertFalse((rebuilt / "Contents/Resources/old.py").exists())
        self.assertTrue((rebuilt / "Contents/Resources/mode-only.txt").stat().st_mode & 0o111)
        self.assertNotIn("Luban.app/Contents/Resources/unchanged.txt", [entry["name"] for entry in result["entries"]])

    def test_wrong_archive_hash_rejected_and_stage_removed(self):
        result, stage = self.request(digest="0" * 64)
        self.assertFalse(result["accepted"])
        self.assertIn("SHA", result["error"])
        self.assertFalse(stage.exists())

    def test_changed_file_cannot_also_be_deleted(self):
        result, stage = self.request(removed=["Contents/Resources/backend/src/luban/server.py"])
        self.assertFalse(result["accepted"])
        self.assertIn("conflicting", result["error"])
        self.assertFalse(stage.exists())

    def test_baseline_symlink_rejected(self):
        linked = self.root / "linked/Luban.app"
        shutil.copytree(self.base, linked)
        (linked / "Contents/Resources/alias.txt").symlink_to("unchanged.txt")
        result, stage = self.request(base=linked)
        self.assertFalse(result["accepted"])
        self.assertIn("Linked", result["error"])
        self.assertFalse(stage.exists())

    def test_patch_symlink_rejected(self):
        patch = self.root / "symlink-delta.zip"
        self.write_patch(patch, link=True)
        result, stage = self.request(archive=patch)
        self.assertFalse(result["accepted"])
        self.assertIn("links", result["error"])
        self.assertFalse(stage.exists())

    def test_valid_zip_with_broken_target_signature_rejected(self):
        patch = self.root / "bad-signature-delta.zip"
        self.write_patch(patch, override={"Contents/Resources/backend/src/luban/server.py": b"# tampered after signing\n"})
        result, _ = self.request(archive=patch)
        self.assertFalse(result["accepted"])
        self.assertIn("validator failed", result["error"])

    def test_identity_project_binding_is_checked_after_reconstruction(self):
        result, _ = self.request(identity={**self.identity, "project": self.project + "-different"})
        self.assertFalse(result["accepted"])
        self.assertIn("environment mismatch", result["error"])


if __name__ == "__main__":
    unittest.main()
