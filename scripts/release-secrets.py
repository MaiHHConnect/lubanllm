"""Fail closed on credential files/values in public release inputs; never print a match."""
from __future__ import annotations

import json
import io
import gzip
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import unicodedata
import zipfile

PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}"),
    re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255})\b"),
    re.compile(rb"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,255}|AKIA[A-Z0-9]{16})\b"),
    re.compile(rb"\bAAC-[a-f0-9]{32}\b"),
    re.compile(rb"(?i)(?:authorization\s*[:=]\s*[\"']?bearer\s+)[A-Za-z0-9._-]{24,255}"),
    re.compile(rb"(?i)(?:__pus|__puus)=[A-Za-z0-9%._-]{16,255}"),
)
URL_CREDENTIAL = re.compile(rb"(?i)https?://[A-Za-z0-9_.%+-]{1,128}:([A-Za-z0-9_.%+!~*-]{1,128})@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
ASSIGNMENT = re.compile(rb'''(?i)["']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password|sudo_password|ranking_sudo_password)["']?\s*[:=]\s*["']([^"'\r\n]{16,512})["']''')
PLACEHOLDER = re.compile(rb"(?i)example|placeholder|your[_ -]|replace[_ -]|dummy|fixture|process\.|os\.environ|\$\{|\$[A-Z_]+|getenv\(|<|>")


def check_name(name):
    normalized = unicodedata.normalize("NFKC", name).replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if normalized != name or name.startswith("/") or ".." in parts:
        raise ValueError("unsafe release path")
    for part in parts:
        lower = part.lower()
        if (lower in {".env", ".ssh", ".aws", ".git", ".codex", ".quarkclouddrive", "internal", "credentials", "credentials.json", "secrets.json", "id_rsa", "id_ed25519", "known_hosts", "authorized_keys"}
                or lower.startswith((".env.", "._")) or lower.endswith((".p12", ".pfx", ".keychain", ".keychain-db"))
                or lower.endswith(".pem") and not re.fullmatch(r"Luban\.app/Contents/Resources/backend/python/lib/python3\.\d+/site-packages/certifi/cacert\.pem", name)):
            raise ValueError("credential/internal filename in release")


def check_content(data):
    if b"LUBAN_INTERNAL_ONLY_MINDMAP" in data or any(pattern.search(data) for pattern in PATTERNS):
        raise ValueError("credential/internal content detected (value redacted)")
    for match in URL_CREDENTIAL.finditer(data):
        example_host = any(match[2].lower() == host or match[2].lower().endswith(b"." + host)
                           for host in (b"example.com", b"example.org", b"example.net", b"test", b"invalid", b"example"))
        if not example_host and match[1].lower() not in {b"password", b"pass", b"pwd"} and not PLACEHOLDER.search(match[1]):
            raise ValueError("credential URL detected (value redacted)")
    for match in ASSIGNMENT.finditer(data):
        value = match[1]
        if (not PLACEHOLDER.search(value) and re.search(rb"[A-Za-z]", value) and re.search(rb"[0-9]", value)
                and len(set(value)) >= 8):
            raise ValueError("literal credential detected (value redacted)")


def scan_stream(stream, limit=None):
    tail, total = b"", 0
    while block := stream.read(1024 * 1024):
        total += len(block)
        if limit is not None and total > limit:
            raise ValueError("compressed member scan limit exceeded")
        data = tail + block
        check_content(data)
        tail = data[-2048:]


def scan_archive(archive, depth=0):
    seen, expanded = set(), 0
    for item in archive.infolist():
        check_name(item.filename)
        folded = item.filename.casefold().rstrip("/")
        if folded in seen:
            raise ValueError("duplicate release member")
        seen.add(folded)
        mode = stat.S_IFMT(item.external_attr >> 16)
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or item.flag_bits & 1:
            raise ValueError("linked, special or encrypted release member")
        expanded += item.file_size
        if expanded > (32 * 1024**2 if depth else 8 * 1024**3) or len(seen) > 100_000:
            raise ValueError("release scan limit exceeded")
        if item.is_dir():
            continue
        if item.filename.endswith(".gz") and re.match(r"Luban\.app/Contents/Resources/backend/python/lib/python3\.\d+/site-packages/", item.filename):
            with archive.open(item) as stream, gzip.GzipFile(fileobj=stream) as content:
                scan_stream(content, 32 * 1024**2)
            continue
        if depth == 0 and item.filename.startswith("Luban.app/Contents/Resources/foundation/"):
            # Fixed inventory and sizes are admitted by release-payload; inspect every
            # non-weight nested member here. The source weight archives were already verified.
            allowed = {"Luban-MLX-Q4-model.zip", "Luban-Speech-model.zip", "Luban-Model-Metadata.zip", "Luban-Factory-Expert.zip"}
            if Path(item.filename).name not in allowed:
                raise ValueError("unknown bundled model archive")
            with archive.open(item) as stream, zipfile.ZipFile(stream) as nested:
                names = set()
                for row in nested.infolist():
                    check_name(row.filename)
                    if row.filename.casefold() in names or stat.S_IFMT(row.external_attr >> 16) not in (0, stat.S_IFREG) or row.flag_bits & 1:
                        raise ValueError("invalid bundled model member")
                    names.add(row.filename.casefold())
                    if row.filename.endswith(".safetensors"):
                        continue
                    if row.file_size > 32 * 1024**2 or Path(row.filename).suffix.lower() in {".zip", ".gz", ".7z", ".tar"}:
                        raise ValueError("invalid bundled model metadata")
                    with nested.open(row) as content:
                        scan_stream(content, 32 * 1024**2)
            continue
        if Path(item.filename).suffix.lower() in {".zip", ".gz", ".7z", ".tar"}:
            if depth or not re.fullmatch(r"Luban\.app/Contents/Resources/backend/python/lib/python\d+\.zip", item.filename) or item.file_size > 32 * 1024**2:
                raise ValueError("uninspected nested archive")
            with archive.open(item) as stream:
                with zipfile.ZipFile(io.BytesIO(stream.read(32 * 1024**2 + 1))) as nested:
                    scan_archive(nested, depth + 1)
        else:
            with archive.open(item) as stream:
                scan_stream(stream)


def scan(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("only explicit regular release files are accepted")
    check_name(path.name)
    if path.suffix.lower() != ".zip":
        with path.open("rb") as stream:
            scan_stream(stream)
        return
    with zipfile.ZipFile(path) as archive:
        scan_archive(archive)


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            raise ValueError("provide explicit release files")
        for name in sys.argv[1:]:
            scan(name)
        print(json.dumps({"gate": "release-secrets", "passed": True, "files": len(sys.argv) - 1}))
    except Exception as error:
        # Zip exceptions can contain filenames supplied by an attacker. Do not echo their text.
        print(json.dumps({"gate": "release-secrets", "passed": False,
                          "reason": str(error) if type(error) is ValueError else type(error).__name__}), file=sys.stderr)
        sys.exit(1)
