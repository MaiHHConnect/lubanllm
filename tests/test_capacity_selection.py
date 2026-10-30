"""Synthetic CPU observations only; no actual model qualification is claimed."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import select_capacity_rows as select


class CapacitySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = select.read_json(select.CORPUS)
        cls.train = select._materials(cls.data)
        cls.binding = {"corpus_sha256": select.CORPUS_SHA256, "base_identity": select.IDENTITY,
            "core_version": None, "actual_tensor_sha256": "1" * 64, "source_sha256": "2" * 64,
            "official_sha256": "3" * 64, "max_new_tokens": 512, "temperature": 0,
            "branch_enabled": False, "reference_used": False, "thinking": "off", "do_sample": False,
            "eos_token_ids": [9], "scope": "cpu_fixture"}

    def fixture(self):
        return {"binding": copy.deepcopy(self.binding), "records": [
            {"id": row["id"], "sample_sha256": select.digest(row), "actual_base_identity": select.IDENTITY,
             "actual_tensor_sha256": "1" * 64, "core_version": None,
             "max_new_tokens": 512, "branch_enabled": False, "reference_used": False,
             "thinking": "off", "do_sample": False, "complete": True, "passed": False,
             "answer": "CPU-FIXTURE-WRONG", "decoded_completion_unstripped": "CPU-FIXTURE-WRONG\n",
             "raw_completion_token_ids": [4, 9], "completion_tokens": 2} for row in self.train]}

    def run_selection(self, proof):
        return select.select_capacity_rows(select.CORPUS, proof, self.binding)

    def test_no_qualification_default_waits_without_torch_or_tokenizer(self):
        result = select.select_capacity_rows()
        self.assertEqual(result["state"], "waiting")
        self.assertIsNone(result["qualification"])
        self.assertEqual(result["selections"], {})
        command = [sys.executable, "-S", str(select.ROOT / "scripts/select_capacity_rows.py")]
        raw = subprocess.run(command, capture_output=True, text=True, check=True)
        cli = json.loads(raw.stdout)
        self.assertFalse(cli["torch_imported"])
        self.assertFalse(cli["execution_ready"])

    def test_fixture_nesting_cell_balance_is_not_actual_qualification(self):
        result = self.run_selection(self.fixture())
        self.assertEqual(result["state"], "fixture_only")
        self.assertEqual(result["qualification"]["original_failed"], 400)
        previous = []
        for count in select.COUNTS:
            selected = result["selections"][str(count)]
            self.assertEqual(selected["state"], "fixture_only")
            self.assertEqual(selected["ids"][:len(previous)], previous)
            self.assertEqual(len(set(selected["ids"])), count)
            self.assertEqual(sum(selected["per_cell"].values()), count)
            self.assertLessEqual(max(selected["per_cell"].values()) - min(selected["per_cell"].values()), 1)
            previous = selected["ids"]
        self.assertEqual(result, self.run_selection(self.fixture()))

    def test_known_answers_excluded_missing_cell_waits_not_resampled(self):
        proof = self.fixture()
        for row, record in zip(self.train, proof["records"]):
            if select._cell(row) == select.CELLS[0]:
                record.update(answer=row["answer"], decoded_completion_unstripped=row["answer"], passed=True)
        result = self.run_selection(proof)
        self.assertEqual(result["qualification"]["original_correct"], 100)
        self.assertTrue(all(value["state"] == "waiting" and not value["ids"]
                            for value in result["selections"].values()))

    def test_record_and_binding_mutations_fail_closed(self):
        changes = {
            "duplicate": lambda p: p["records"].__setitem__(1, copy.deepcopy(p["records"][0])),
            "heldout": lambda p: p["records"][0].update(id="capacity-v1-validation-0000"),
            "missing": lambda p: p["records"].pop(),
            "record_sha": lambda p: p["records"][0].update(sample_sha256="0" * 64),
            "branch": lambda p: p["records"][0].update(branch_enabled=True),
            "reference": lambda p: p["records"][0].update(reference_used=True),
            "tensor": lambda p: p["records"][0].update(actual_tensor_sha256="0" * 64),
            "self_claim": lambda p: p["records"][0].update(passed=True),
            "bool": lambda p: p["records"][0].update(complete=1),
            "eos": lambda p: p["records"][0].update(raw_completion_token_ids=[4, 8]),
            "truncated": lambda p: p["records"][0].update(complete=False, raw_completion_token_ids=[4, 8]),
            "decode": lambda p: p["records"][0].update(decoded_completion_unstripped="different"),
            "policy": lambda p: p["binding"].update(max_new_tokens=128),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                proof = self.fixture(); change(proof)
                with self.assertRaises(ValueError):
                    self.run_selection(proof)

    def test_fixed_material_bytehash_and_semantic_leakage_guards(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home) / "changed.json"
            path.write_bytes(select.CORPUS.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "byte SHA"):
                select.select_capacity_rows(path)
        # Deliberately recompute the fixture ledger so semantic guards are
        # exercised independently of the immutable production corpus byte SHA.
        for mutation in ("leak", "unverified", "gold"):
            data = copy.deepcopy(self.data)
            row = data["rows"][0]; p = select.provenance(row)
            if mutation == "leak":
                p["messages"][1]["content"] += " " + row["answer"]
            elif mutation == "unverified":
                p["program_verified"] = False
            else:
                p["generation_check"]["expected"] = "CHANGED"
            row["provenance"] = json.dumps(p)
            item = next(v for v in data["gold_ledger"] if v["id"] == row["id"])
            item["record_sha256"] = select.digest(row, compact=True)
            item["context_sha256"] = hashlib.sha256(select.normalized_context(row).encode()).hexdigest()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                select._materials(data)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home) / "duplicate.json"
            path.write_text('{"scope":"actual","scope":"cpu_fixture"}')
            with self.assertRaises(ValueError):
                select.read_json(path)

    def test_actual_scope_rejects_other_core_or_other_valid_tensor_hash(self):
        # Binding guard fixture only, not a fabricated actual qualification.
        binding = {**self.binding, "scope": "actual_closed_original", "actual_tensor_sha256": select.ORIGINAL}
        select._binding(binding)
        for changes in ({"core_version": "different-accepted-core"}, {"actual_tensor_sha256": "4" * 64}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "originalNone/1706"):
                select._binding({**binding, **changes})


if __name__ == "__main__":
    unittest.main()
