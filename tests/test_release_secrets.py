import importlib.util
import io
import gzip
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("release_secrets", Path(__file__).resolve().parents[1] / "scripts/release-secrets.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ReleaseSecretsTests(unittest.TestCase):
    def test_known_tokens_and_private_key_never_echo_values(self):
        for value in (b"ghp_" + b"a1B2c3D4" * 5, b"sk-proj-" + b"a1B2c3D4" * 5,
                      b"AAC-" + b"1a" * 16, b"-----BEGIN PRIVATE KEY-----\n" + b"Ab12" * 20):
            with self.assertRaises(ValueError) as caught:
                gate.check_content(value)
            self.assertNotIn(value.decode(), str(caught.exception))

    def test_literal_credentials_in_code_or_json(self):
        for label in (b"api_key", b"access_token", b"password", b"RANKING_SUDO_PASSWORD"):
            with self.assertRaises(ValueError):
                gate.check_content(label + b' = "C4qY9b7W2s6F8rK3tP1z"')
        gate.check_content(b'api_key = os.environ["API_KEY"]')
        gate.check_content(b'api_key = "your_api_key_example"')

    def test_documented_url_examples_and_parser_labels_are_safe(self):
        gate.check_content(b'https://user:password@example.com -----BEGIN PRIVATE KEY-----')
        with self.assertRaises(ValueError):
            gate.check_content(b'https://operator:C4qY9b7W2s6F8rK3tP1z@production.host.com')

    def test_stream_boundary_and_binary_embedded_token(self):
        value = b"\0" * (1024 * 1024 - 5) + b"ghp_" + b"a1B2c3D4" * 5 + b"\0"
        with self.assertRaises(ValueError):
            gate.scan_stream(io.BytesIO(value))

    def test_credentials_paths_traversal_and_normalization(self):
        for path in ("Luban.app/.env", "Luban.app/.ssh/config", "../private.txt", "Luban.app/internal/a.html", "Luban.app/._key", "Luban.app/．env"):
            with self.assertRaises(ValueError):
                gate.check_name(path)
        gate.check_name("Luban.app/Contents/Resources/backend/python/lib/python3.12/site-packages/certifi/cacert.pem")
        with self.assertRaises(ValueError):
            gate.check_name("Luban.app/private.pem")

    def test_zip_symlink_duplicate_and_nested_secrets(self):
        for kind in ("link", "duplicate", "nested"):
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as package:
                if kind == "link":
                    info = zipfile.ZipInfo("Luban.app/link")
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    package.writestr(info, "../../secret")
                elif kind == "duplicate":
                    package.writestr("Luban.app/A", "ok")
                    package.writestr("Luban.app/a", "ok")
                else:
                    inner = io.BytesIO()
                    with zipfile.ZipFile(inner, "w") as nested:
                        nested.writestr("secret.py", b"ghp_" + b"a1B2c3D4" * 5)
                    package.writestr("Luban.app/Contents/Resources/backend/python/lib/python312.zip", inner.getvalue())
            data.seek(0)
            with zipfile.ZipFile(data) as package, self.assertRaises(ValueError):
                gate.scan_archive(package)

    def test_only_explicit_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "RELEASE-NOTES.md"
            path.write_text("Public beta notes")
            gate.scan(path)
            link = Path(directory) / "notes.md"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                gate.scan(link)
            with self.assertRaises(ValueError):
                gate.scan(directory)

    def test_dependency_gzip_is_scanned_without_deserializing(self):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr("Luban.app/Contents/Resources/backend/python/lib/python3.12/site-packages/library/data.gz",
                             gzip.compress(b"ghp_" + b"a1B2c3D4" * 5))
        raw.seek(0)
        with zipfile.ZipFile(raw) as archive, self.assertRaises(ValueError):
            gate.scan_archive(archive)


if __name__ == "__main__":
    unittest.main()
