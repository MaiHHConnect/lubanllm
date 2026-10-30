import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/paper_gate.py"
spec = importlib.util.spec_from_file_location("paper_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class PaperGateTests(unittest.TestCase):
    def test_draft_passes_but_modified_code_and_release_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "src").mkdir()
            (root / "papers/0001-bootstrap").mkdir(parents=True)
            (root / "src/model.py").write_text("initial code")
            (root / "papers/0001-bootstrap/paper.md").write_text(
                "\n".join("## " + s + "\nDocumented evidence." for s in gate.SECTIONS))
            (root / "papers/0001-bootstrap/test.txt").write_text("OK")
            snapshot = gate.Snapshot(root=root)
            paper = "papers/0001-bootstrap/paper.md"
            report = "papers/0001-bootstrap/test.txt"
            bundle = {"paper": paper, "paper_sha256": gate.digest(snapshot.read(paper)),
                      "status": "draft", "claims": ["prototype only"],
                      "files": {p: gate.digest(snapshot.read(p)) for p in snapshot.paths()},
                      "evidence": [{"path": report, "sha256": gate.digest(snapshot.read(report)),
                                    "command": "python -m unittest", "exit_code": 0}]}
            (root / gate.BUNDLE).write_text(json.dumps(bundle))
            self.assertEqual(gate.validate(snapshot), [])
            self.assertTrue(gate.validate(snapshot, release=True))
            (root / "src/model.py").write_text("changed code")
            self.assertTrue(gate.validate(snapshot))
            (root / "src/extra.py").write_text("unrecorded code")
            self.assertTrue(any("unrecorded" in e for e in gate.validate(snapshot)))
            (root / "otherwise-untracked.py").write_text("also requires evidence")
            self.assertTrue(any("otherwise-untracked.py" in e for e in gate.validate(snapshot)))


if __name__ == "__main__":
    unittest.main()
