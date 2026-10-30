#!/usr/bin/env python3
"""Validate a paper/evidence bundle against worktree, index, or a Git ref.

This is a reproducibility gate, not peer review and not a security boundary.
Local hooks can be bypassed; remote required checks must be configured separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from mindmap_gate import check as check_mindmap

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "papers/0001-bootstrap/bundle.json"
SECTIONS = ("Problem", "Method", "Experiments", "Results", "Limitations", "Reproducibility", "References")


def protected(path):
    parts = PurePosixPath(path).parts
    return (bool(parts) and parts[0] != "papers" and
            not any(p in {".git", ".venv", "__pycache__", "downloads", "storage"} for p in parts) and
            not path.endswith((".pyc", ".safetensors", ".partial", ".DS_Store")))


class Snapshot:
    def __init__(self, source="worktree", root=ROOT):
        self.source, self.root = source, Path(root)

    def read(self, path):
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("unsafe bundle path")
        if self.source == "worktree":
            resolved = (self.root / path).resolve()
            if not resolved.is_relative_to(self.root.resolve()):
                raise ValueError("bundle path escapes root")
            return resolved.read_bytes()
        spec = ":" + path if self.source == "index" else self.source + ":" + path
        return subprocess.check_output(["git", "show", spec], cwd=self.root, stderr=subprocess.DEVNULL)

    def paths(self):
        if self.source == "worktree":
            if (self.root / ".git").exists():
                candidates = subprocess.check_output(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                    cwd=self.root).decode().split("\0")
            else:
                candidates = [str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()]
            return {p for p in candidates if protected(p) and (self.root / p).is_file()}
        cmd = ["git", "ls-files", "-z"] if self.source == "index" else ["git", "ls-tree", "-r", "--name-only", "-z", self.source]
        return {p for p in subprocess.check_output(cmd, cwd=self.root).decode().split("\0") if p and protected(p)}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate(snapshot, release=False):
    errors = []
    try:
        bundle = json.loads(snapshot.read(BUNDLE))
        paper = snapshot.read(bundle["paper"])
        text = paper.decode()
        for section in SECTIONS:
            if "## " + section not in text:
                errors.append("missing paper section: " + section)
        if digest(paper) != bundle["paper_sha256"]:
            errors.append("paper hash mismatch")
        coverage = bundle["files"]
        actual_paths = snapshot.paths()
        errors.extend(check_mindmap(snapshot))
        if actual_paths != set(coverage):
            errors.append("unrecorded or removed files: " + str(sorted(actual_paths ^ set(coverage))))
        for path, expected in coverage.items():
            if digest(snapshot.read(path)) != expected:
                errors.append("file changed without updated paper evidence: " + path)
        evidence = bundle["evidence"]
        if not evidence:
            errors.append("missing reproducible evidence")
        for item in evidence:
            if not item.get("command") or item.get("exit_code") != 0:
                errors.append("missing successful evidence command")
            if digest(snapshot.read(item["path"])) != item["sha256"]:
                errors.append("evidence hash mismatch: " + item["path"])
        if not bundle.get("claims"):
            errors.append("claims must be explicit")
        if release:
            if bundle.get("status") != "final":
                errors.append("release requires final paper; current bundle is a draft")
            if not bundle.get("human_reviewed_by"):
                errors.append("release requires a named human research reviewer")
            publication = urlparse(bundle.get("publication_url", ""))
            if publication.scheme != "https" or publication.hostname not in {"arxiv.org", "doi.org", "openreview.net"}:
                errors.append("release requires public arXiv / DOI / OpenReview paper URL")
            pdf = bundle.get("paper_pdf")
            if not pdf or not snapshot.read(pdf).startswith(b"%PDF-"):
                errors.append("release requires a final paper PDF")
            elif digest(snapshot.read(pdf)) != bundle.get("paper_pdf_sha256"):
                errors.append("final PDF hash mismatch")
            if not bundle.get("scientific_evaluation_complete"):
                errors.append("release requires completed scientific evaluation")
            if not bundle.get("reproduction_command"):
                errors.append("release requires a full scientific reproduction command")
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
        errors.append("invalid or unavailable paper bundle: " + str(error))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="worktree", help="worktree, index, or commit/ref")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    errors = validate(Snapshot(args.source, args.root), args.release)
    if errors:
        print("PAPER GATE BLOCKED\n" + "\n".join("- " + e for e in errors))
        return 1
    print("PAPER GATE PASSED (" + ("release" if args.release else "research draft") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
