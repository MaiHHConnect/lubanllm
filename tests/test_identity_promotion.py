import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from promote_identity_expert import (transactional_replace, validate_original_protection,
                                     validate_parent_preservation_metadata, validate_recursive_proof)


class IdentityPromotionTests(unittest.TestCase):
    def test_recursive_candidate_requires_durable_parent_teacher(self):
        row = {"id": "parent", "provenance": '{"preservation_teacher":"expert"}'}
        self.assertEqual(validate_parent_preservation_metadata({"preservation_rows": [row]}, 2), 1)
        with self.assertRaisesRegex(ValueError, "父专家保护"):
            validate_parent_preservation_metadata({"preservation_rows": []}, 2)

    def test_recursive_proof_requires_all_gate_families_and_matching_versions(self):
        proof = {"protocol": "recursive-admission-v1", "status": "passed",
                 "expert_id": "expert", "router_id": "router",
                 "original_core_protection_sha256": "a" * 64,
                 **{key: True for key in ("original_core_protection", "expert_closed_book",
                                          "route_classification", "expert_scope",
                                          "checkpoint_restore", "transactional_rollback")}}
        result = validate_recursive_proof(proof, expert_id="expert", router_id="router",
                                          original_sha256="a" * 64)
        self.assertTrue(all(result.values()))
        proof["expert_scope"] = False
        with self.assertRaisesRegex(ValueError, "递归发布合同"):
            validate_recursive_proof(proof, expert_id="expert", router_id="router",
                                     original_sha256="a" * 64)

    def test_recursive_round_two_requires_parent_continuity_gate(self):
        proof = {"protocol": "recursive-admission-v2", "status": "passed",
                 "expert_id": "expert", "router_id": "router",
                 "original_core_protection_sha256": "a" * 64,
                 **{key: True for key in ("original_core_protection", "expert_closed_book",
                                          "route_classification", "expert_scope",
                                          "checkpoint_restore", "transactional_rollback",
                                          "parent_continuity")}}
        result = validate_recursive_proof(proof, expert_id="expert", router_id="router",
                                          original_sha256="a" * 64, candidate_round=2)
        self.assertTrue(result["parent_continuity"])
        proof["parent_continuity"] = False
        with self.assertRaisesRegex(ValueError, "递归发布合同"):
            validate_recursive_proof(proof, expert_id="expert", router_id="router",
                                     original_sha256="a" * 64, candidate_round=2)

    def test_original_protection_requires_eight_branch_off_rows(self):
        record = {"protocol": "original-core-protection-v1", "passed": True,
                  "branch_enabled": False, "teacher_calls": 0,
                  "training_performed": False, "identity": "a" * 64,
                  "original_tensor_sha256": "b" * 64,
                  "cases": [{} for _ in range(8)],
                  "generation": [{"passed": True, "complete": True,
                                  "branch_enabled": False, "reference_used": False}
                                 for _ in range(8)]}
        result = validate_original_protection(record, "a" * 64)
        self.assertEqual(result["case_count"], 8)
        record["generation"][0]["branch_enabled"] = True
        with self.assertRaisesRegex(ValueError, "原始核心保护基线"):
            validate_original_protection(record, "a" * 64)

    def test_transaction_restores_existing_bundle_after_mid_commit_failure(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            sources = root / "sources"
            targets = root / "targets"
            sources.mkdir()
            targets.mkdir()
            (sources / "one").write_bytes(b"new-one")
            (sources / "two").write_bytes(b"new-two")
            (targets / "one").write_bytes(b"old-one")
            (targets / "two").write_bytes(b"old-two")
            real_replace = __import__("os").replace
            calls = {"count": 0}

            def fail_once(source, target):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("injected promotion failure")
                return real_replace(source, target)

            with patch("promote_identity_expert.os.replace", side_effect=fail_once):
                with self.assertRaisesRegex(OSError, "injected promotion failure"):
                    transactional_replace(((sources / "one", targets / "one"),
                                           (sources / "two", targets / "two")))
            self.assertEqual((targets / "one").read_bytes(), b"old-one")
            self.assertEqual((targets / "two").read_bytes(), b"old-two")

    def test_transaction_commits_all_targets_together(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            sources, targets = root / "sources", root / "targets"
            sources.mkdir()
            targets.mkdir()
            (sources / "one").write_bytes(b"new-one")
            (sources / "two").write_bytes(b"new-two")
            transactional_replace(((sources / "one", targets / "one"),
                                   (sources / "two", targets / "two")))
            self.assertEqual((targets / "one").read_bytes(), b"new-one")
            self.assertEqual((targets / "two").read_bytes(), b"new-two")


if __name__ == "__main__":
    unittest.main()
