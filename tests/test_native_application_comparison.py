"""Supervisor regressions without model/GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/compare_native_application.py"
SPEC = importlib.util.spec_from_file_location("native_application_comparison", SOURCE)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def test_partial_line_keeps_real_deadline(self):
        child = subprocess.Popen(["/bin/sh", "-c", "printf '{'; exec /bin/sleep 2"],
                                 stdout=subprocess.PIPE, start_new_session=True)
        os.set_blocking(child.stdout.fileno(), False)
        began = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                comparison.event(child, bytearray(), timeout=.05)
            self.assertLess(time.monotonic() - began, .5)
        finally:
            os.killpg(child.pid, signal.SIGKILL); child.wait(); child.stdout.close()

    def test_multiple_frames_preserved_in_one_read(self):
        child = subprocess.Popen(["/usr/bin/printf", '{"event":"one"}\n{"event":"two"}\n'], stdout=subprocess.PIPE)
        pending = bytearray()
        try:
            child.wait(timeout=2)
            self.assertEqual(comparison.event(child, pending)["event"], "one")
            self.assertEqual(comparison.event(child, pending)["event"], "two")
        finally:
            child.wait(); child.stdout.close()

    def test_empty_or_short_baseline_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rows in ([], [{"index": 0, "request": {"messages": comparison.CASES[0]}}]):
                (root / "baseline.json").write_text(json.dumps({"passed": True, "rows": rows}))
                with self.assertRaises(ValueError):
                    comparison.candidate(root, Path("/must-not-run"), Path("/must-not-copy"))
                self.assertFalse((root / "swift-candidate").exists())

    def test_incomplete_frame_closed_is_error(self):
        child = subprocess.Popen(["/usr/bin/printf", "{"], stdout=subprocess.PIPE)
        try:
            with self.assertRaises(RuntimeError):
                comparison.event(child, bytearray(), timeout=1)
        finally:
            child.wait(); child.stdout.close()


if __name__ == "__main__":
    unittest.main()
