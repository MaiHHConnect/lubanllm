"""Compile the actual generated current-state consumer; no model or MPS execution."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class NativeExpertStateRestoreTests(unittest.TestCase):
    def test_actual_current_restores_adam_and_keeps_round_teachers_atomically(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="luban-expert-state-") as directory:
            work = Path(directory).resolve()
            call = "import {writeNativeExpertPreservationState} from './scripts/native-expert-preservation-source.ts';await writeNativeExpertPreservationState(process.cwd(),process.argv[1]);"
            subprocess.run(["bun", "-e", call, str(work)], cwd=root, check=True, capture_output=True, text=True)
            build = json.loads((root / "build/native-evolution/tail-build.json").read_text())
            argv = next(row["argv"] for row in build["commands"] if row["argv"][0] == "/usr/bin/clang++")
            argv = [str(root / "native/ExpertStateRestoreCPU.cpp") if value.endswith("/PersistentEvolution.cpp") else value for value in argv]
            argv[argv.index("-o") + 1] = str(work / "state-cpu")
            argv.insert(1, '-DLUBAN_EXPERT_STATE_HEADER="' + str(work / "ExpertPreservedStateCommands.h") + '"')
            compiled = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            run = subprocess.run([str(work / "state-cpu"), str(work / "actual")], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout), dict(passed=True, teacher_count=2, step=1, rejected=2, GPU=False))


if __name__ == "__main__":
    unittest.main()
