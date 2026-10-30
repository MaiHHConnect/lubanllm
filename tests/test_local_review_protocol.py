import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_local_review.py"
SPEC = importlib.util.spec_from_file_location("verify_local_review_protocol", SCRIPT)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def perfect_results():
    results = []
    for row in protocol.cases():
        method = ("missing_reference" if row["mode"] == "unverified" else
                  "context_mismatch" if row["mode"] == "mismatch" else "local_reference_advisory")
        results.append({"id": row["id"], "result": {"verdict": row["expected"], "method": method,
                        "raw": json.dumps(row["expected"]), "usage": {"finish_reason": "stop"},
                        "verified_source": False, "gate_enabled": False}})
    return results


class LocalReviewProtocolTests(unittest.TestCase):
    def test_fixed_counts_unique_cases_and_truth_is_literal_fiction_not_training(self):
        rows = protocol.cases()
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({row["id"] for row in rows}), 28)
        self.assertEqual([sum(row["expected"] is expected for row in rows) for expected in (True, False, None)], [10, 14, 4])
        self.assertTrue({"paraphrase", "negation", "numeric", "unit", "condition", "history", "system", "injection", "unknown"}
                        <= {row["category"] for row in rows})
        for row in rows:
            reference = protocol.reference_for(row)
            self.assertFalse(reference["approved"] or reference["training_allowed"])
            self.assertEqual(reference["source"], "synthetic")
            self.assertNotEqual(row["answer"].strip(), reference["answer"].strip())
            self.assertNotIn("human_verified", reference["provenance"])
        changed = copy.deepcopy(rows[0])
        changed["reference_answer"] = "未在来源中出现的答案"
        with self.assertRaises(ValueError):
            protocol.reference_for(changed)
        changed = copy.deepcopy(rows[0])
        changed["context"][-1]["content"] = "问题未提供原始来源"
        with self.assertRaises(ValueError):
            protocol.reference_for(changed)

    def test_positive_coverage_and_one_false_positive_never_pass(self):
        self.assertTrue(protocol.summarize(perfect_results())["protocol_passed"])
        results = perfect_results()
        # One redundant paraphrase may abstain; required categories still pass.
        results[0]["result"].update(verdict=None, raw="null")
        self.assertTrue(protocol.summarize(results)["protocol_passed"])
        results[1]["result"].update(verdict=None, raw="null")
        self.assertFalse(protocol.summarize(results)["protocol_passed"])
        results = perfect_results()
        results[1]["result"].update(verdict=None, raw="null")
        self.assertFalse(protocol.summarize(results)["protocol_passed"])
        for index in range(10, 28):
            results = perfect_results()
            results[index]["result"].update(verdict=True, raw="true")
            summary = protocol.summarize(results)
            self.assertFalse(summary["protocol_passed"])
            self.assertEqual(summary["false_positives"], 1)
            self.assertFalse(summary["gate_enabled"])

    def test_abstention_is_not_rejection_and_parse_failure_is_not_correct_unknown(self):
        results = perfect_results()
        results[10]["result"].update(verdict=None, raw="null")
        self.assertFalse(protocol.summarize(results)["protocol_passed"])
        for changes in ({"raw": "not JSON"}, {"raw": ""}, {"usage": {"finish_reason": "length"}},
                        {"verdict": 0}, {"gate_enabled": True}, {"verified_source": True}):
            results = perfect_results()
            results[-1]["result"].update(changes)
            self.assertFalse(protocol.summarize(results)["protocol_passed"])
        for results in (perfect_results()[:-1], perfect_results()[::-1], perfect_results() + perfect_results()[:1]):
            with self.assertRaises(ValueError):
                protocol.summarize(results)

    def test_default_and_print_plan_work_without_site_packages_or_model_imports(self):
        frozen = json.loads((SCRIPT.parents[1] / "research/local_review_protocol.json").read_text())
        self.assertEqual(frozen, protocol.plan())
        for flags in ([], ["--print-plan"]):
            completed = subprocess.run([sys.executable, "-S", str(SCRIPT), *flags], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(completed.stdout)["case_sha256"], protocol.plan()["case_sha256"])
        missing = subprocess.run([sys.executable, "-S", str(SCRIPT), "--execute"], capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("requires --expected-identity", missing.stderr)

    def test_core_snapshot_binding_and_integrity_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "target"
            source.mkdir()
            self.assertEqual(protocol.snapshot_core(source, target, "original", "base"), {})
            version = "core-123456789abc"
            weights = source / (version + ".safetensors")
            weights.write_bytes(b"CPU fixture only, never loaded as model weights")
            digest = protocol.file_sha(weights)
            active = {"version": version, "identity": "base", "sha256": digest}
            (source / "active-core.json").write_text(json.dumps(active))
            (source / (version + ".json")).write_text(json.dumps({**active}))
            with self.assertRaises(ValueError):
                protocol.snapshot_core(source, target, "original", "base")
            with self.assertRaises(ValueError):
                protocol.snapshot_core(source, target, version, "different base")
            hashes = protocol.snapshot_core(source, target, version, "base")
            self.assertEqual(hashes[weights.name], digest)
            self.assertEqual(json.loads((target / "active-core.json").read_text()), active)
            weights.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                protocol.snapshot_core(source, target, version, "base")


if __name__ == "__main__":
    unittest.main()
