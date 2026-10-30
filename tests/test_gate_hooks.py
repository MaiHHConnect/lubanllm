import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GateHookTests(unittest.TestCase):
    def test_hooks_load_dependencies_from_the_checked_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            env = {**os.environ, "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.test",
                   "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.test"}
            env.pop("PYTHONPATH", None)

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, env=env, text=True).strip()

            git("init", "-q")
            (root / "scripts").mkdir()
            for name in ("paper_gate.py", "mindmap_gate.py"):
                shutil.copyfile(ROOT / "scripts" / name, root / "scripts" / name)
            paper = "papers/0001-bootstrap/paper.md"
            evidence = "papers/0001-bootstrap/tests.log"
            (root / paper).parent.mkdir(parents=True)
            (root / paper).write_text("\n".join("## " + title for title in (
                "Problem", "Method", "Experiments", "Results", "Limitations", "Reproducibility", "References")))
            (root / evidence).write_text("fixture evidence\n")
            digest = lambda path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            bundle = {"status": "draft", "paper": paper, "paper_sha256": digest(paper),
                      "claims": ["isolated hook fixture"],
                      "files": {"scripts/" + name: digest("scripts/" + name)
                                for name in ("paper_gate.py", "mindmap_gate.py")},
                      "evidence": [{"path": evidence, "sha256": digest(evidence),
                                    "command": "fixture", "exit_code": 0}]}
            (root / "papers/0001-bootstrap/bundle.json").write_text(json.dumps(bundle))
            git("add", ".")
            commit = git("commit-tree", git("write-tree"), "-m", "isolated fixture")
            # A broken worktree dependency must not replace the staged/ref dependency.
            (root / "scripts/mindmap_gate.py").write_text("raise RuntimeError('unstaged dependency')\n")

            def hook(name, stdin=""):
                return subprocess.run(["sh", str(ROOT / ".githooks" / name)], cwd=root, env=env,
                                      input=stdin, text=True, capture_output=True)

            checked = hook("pre-commit")
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PAPER GATE PASSED", checked.stdout)
            pushed = hook("pre-push", f"refs/heads/fixture {commit} refs/heads/fixture {'0' * 40}\n")
            self.assertEqual(pushed.returncode, 1, pushed.stdout + pushed.stderr)
            self.assertIn("release requires final paper", pushed.stdout)
            self.assertNotIn("ModuleNotFoundError", pushed.stderr)
            git("add", "scripts/mindmap_gate.py")
            rejected = hook("pre-commit")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("unstaged dependency", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
