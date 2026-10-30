"""CPU material contracts only; these fixtures do not establish identity learning."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from luban import model_packages as packages
from luban.automatic import AutomaticLearning
from luban.evolution_progress import DIRECTED_TEACHING, PARTIAL_POLICY, validate_product_records
from luban.retention import daily_learning_source


def directed_records(*, positioning=False):
    reference = "local-user-request:directed-identity-contract-fixture"
    facts = {}
    groups = {key: [] for key in ("new_training", "new_validation", "replay_training", "replay_validation")}
    prompts = {"model_name": ("请仅回答你的模型名称。", "这位助手使用什么模型名称？仅给名称。"),
               "model_fine_tuner": ("请仅回答你的微调方名称。", "这位助手由谁微调？仅给名称。")}
    values = {"model_name": "luban0", "model_fine_tuner": "麦好火"}
    if positioning:
        values["model_positioning"] = "无限进化"
        prompts["model_positioning"] = ("你的进化定位是什么？仅给名称。", "这位助手的进化目标叫什么？只给定位名称。")
    for key, value in values.items():
        facts[key] = {"subject": "model", "value": value, "training_ids": [key + "-train"],
                      "validation_ids": [key + "-valid"]}
        for purpose, suffix, prompt in zip(("training", "validation"), ("train", "valid"), prompts[key]):
            groups["new_" + purpose].append({"id": key + "-" + suffix, "prompt": prompt, "answer": value,
                "source": "human", "purpose": purpose, "approved": True, "training_allowed": True,
                "provenance": {"evolution_role": "new", "protocol_id": DIRECTED_TEACHING,
                    "authorization_reference": reference, "teaching_fact": key,
                    "split_family": suffix + "-" + key,
                    "generation_check": {"type": "exact", "expected": value}}})
    for purpose in ("training", "validation"):
        groups["replay_" + purpose].append({"id": "replay-" + purpose, "prompt": "old " + purpose,
            "answer": "old", "source": "human", "purpose": purpose, "approved": True,
            "training_allowed": True, "provenance": {"evolution_role": "replay"}})
    return {"groups": groups, "protection": [], "retention_regressions": [],
        "directed_teaching": {"protocol": DIRECTED_TEACHING,
            "authorization": {"source": "human", "explicit": True, "training_allowed": True, "reference": reference},
            "facts": facts}}


class DirectedTeachingTests(unittest.TestCase):
    def test_current_identity_material_preserves_positioning_and_unknown_user_boundary(self):
        from scripts.current_q4_identity_material import records
        material = records()
        validate_product_records(material)
        self.assertEqual(material["directed_teaching"]["facts"]["model_positioning"]["value"], "无限进化")
        self.assertNotIn("林浩", [row["answer"] for rows in material["groups"].values() for row in rows])
        self.assertIn("不知道", [row["answer"] for row in material["groups"]["replay_validation"]])
        self.assertEqual(len(material["retention_regressions"]), 3)

    def test_authorized_positioning_keeps_independent_train_and_validation(self):
        records = directed_records(positioning=True)
        validate_product_records(records)
        for purpose in ("training", "validation"):
            rows = records["groups"]["new_" + purpose]
            self.assertEqual([r["answer"] for r in rows if r["provenance"]["teaching_fact"] == "model_positioning"], ["无限进化"])

    def test_positioning_does_not_allow_unbounded_claims_or_missing_validation(self):
        for field, value in (("value", "任何问题都能答对"), ("subject", "user"), ("validation_ids", [])):
            records = directed_records(positioning=True)
            records["directed_teaching"]["facts"]["model_positioning"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_product_records(records)

    def test_two_authorized_facts_with_independent_questions_need_no_experience_pool(self):
        records = directed_records()
        validate_product_records(records)
        self.assertNotIn("experience", records)
        self.assertEqual(sum(len(rows) for key, rows in records["groups"].items() if key.startswith("new_")), 4)
        self.assertTrue(all(not daily_learning_source(row) for row in records["groups"]["new_training"]))

    def test_authorization_and_exact_record_schema_fail_closed(self):
        mutations = {
            "both sources": lambda r: r.update(experience=[]),
            "no source": lambda r: r.pop("directed_teaching"),
            "unknown metadata": lambda r: r.update(extra={}),
            "missing authorization": lambda r: r["directed_teaching"].pop("authorization"),
            "not human": lambda r: r["directed_teaching"]["authorization"].update(source="teacher"),
            "implicit": lambda r: r["directed_teaching"]["authorization"].update(explicit=False),
            "truthy not bool": lambda r: r["directed_teaching"]["authorization"].update(explicit=1),
            "revoked": lambda r: r["directed_teaching"]["authorization"].update(training_allowed=False),
            "empty reference": lambda r: r["directed_teaching"]["authorization"].update(reference=" "),
            "unknown authorization": lambda r: r["directed_teaching"]["authorization"].update(extra=True),
            "unknown protocol": lambda r: r["directed_teaching"].update(protocol="anything"),
        }
        for label, mutate in mutations.items():
            records = directed_records()
            mutate(records)
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_product_records(records)

    def test_model_facts_do_not_authorize_declared_user_identity_or_other_facts(self):
        for field, value in (("subject", "user"), ("value", "林浩"), ("value", "Qwen")):
            records = directed_records()
            records["directed_teaching"]["facts"]["model_name"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_product_records(records)
        records = directed_records()
        records["directed_teaching"]["facts"]["user_name"] = {"subject": "user", "value": "林浩"}
        with self.assertRaises(ValueError):
            validate_product_records(records)
        # This is a declared-subject contract, not a natural-language classifier.
        # Actual ambiguous/multi-turn user identity requires later generation tests.

    def test_every_fact_needs_disjoint_complete_train_and_validation_coverage(self):
        mutations = {
            "missing validation": lambda r: r["directed_teaching"]["facts"]["model_name"].update(validation_ids=[]),
            "foreign id": lambda r: r["directed_teaching"]["facts"]["model_name"].update(training_ids=["missing"]),
            "duplicate id": lambda r: r["directed_teaching"]["facts"]["model_name"].update(training_ids=["model_name-train"] * 2),
            "other fact id": lambda r: r["directed_teaching"]["facts"]["model_name"].update(training_ids=["model_fine_tuner-train"]),
            "uncovered row": lambda r: r["groups"]["new_training"].append({**copy.deepcopy(r["groups"]["new_training"][0]), "id": "extra", "prompt": "extra question"}),
            "duplicate context": lambda r: r["groups"]["new_validation"][0].update(prompt=r["groups"]["new_training"][0]["prompt"]),
            "shared family": lambda r: r["groups"]["new_validation"][0]["provenance"].update(split_family="train-model_name"),
            "no family": lambda r: r["groups"]["new_validation"][0]["provenance"].pop("split_family"),
        }
        for label, mutate in mutations.items():
            records = directed_records()
            mutate(records)
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_product_records(records)

    def test_row_licenses_targets_and_authorization_binding_remain_required(self):
        mutations = {
            "not approved": lambda row: row.update(approved=False),
            "not trainable": lambda row: row.update(training_allowed=False),
            "wrong answer": lambda row: row.update(answer="other"),
            "wrong subject reference": lambda row: row["provenance"].update(teaching_fact="user_name"),
            "wrong authorization": lambda row: row["provenance"].update(authorization_reference="another request"),
            "missing protocol": lambda row: row["provenance"].pop("protocol_id"),
            "not human": lambda row: row.update(source="synthetic"),
            "weak check": lambda row: row["provenance"].update(generation_check={"type": "contains", "expected": "luban0"}),
            "wrong check": lambda row: row["provenance"].update(generation_check={"type": "exact", "expected": "other"}),
        }
        for label, mutate in mutations.items():
            records = directed_records()
            mutate(records["groups"]["new_training"][0])
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_product_records(records)

    def test_journal_traversal_preserves_samples_and_validates_metadata_before_skipping(self):
        records = json.loads(json.dumps(directed_records(), ensure_ascii=False))
        rows = list(packages._record_rows(records))
        self.assertEqual(rows, [row for group in records["groups"].values() for row in group])
        for change in ("extra", "authorization", "dual"):
            bad = copy.deepcopy(records)
            if change == "extra":
                bad["directed_teaching"]["unknown"] = "not a sample"
            elif change == "authorization":
                bad["directed_teaching"]["authorization"]["explicit"] = False
            else:
                bad["experience"] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                list(packages._record_rows(bad))

    def test_directed_receipt_cannot_bypass_fixed450_or_actual_saved_proofs(self):
        run = {"kind": "evolution", "scope": "product", "records": directed_records(),
               "arguments": {"steps": 450, "learning_rate": 1e-5, "distillation_weight": 5,
                             "preserve_replay_subspace": True, "partial_policy": PARTIAL_POLICY},
               "commit": {"report": {"steps": 64}}}
        with self.assertRaisesRegex(ValueError, "固定450"):
            packages._validate_product_partial(run)
        run["commit"]["report"] = {"steps": 450, "partial_policy": PARTIAL_POLICY}
        with self.assertRaisesRegex(ValueError, "核心文件绑定"):
            packages._validate_product_partial(run)

    def test_gui_recovery_still_rejects_directed_session(self):
        records = directed_records()
        runtime = SimpleNamespace(memory=SimpleNamespace(samples=lambda: [row for rows in records["groups"].values() for row in rows]))
        auto = SimpleNamespace(runtime=runtime)
        with patch("luban.automatic.read_core_retention", return_value=None), self.assertRaisesRegex(ValueError, "日常新增学习"):
            AutomaticLearning._check_resume_rights(auto, {"kind": "evolution", "records": records}, [])

    def test_daily_experience_still_requires_350_unique_licensed_rows(self):
        records = directed_records()
        records.pop("directed_teaching")
        records["experience"] = [copy.deepcopy(row) for key in ("new_training", "new_validation") for row in records["groups"][key]]
        with self.assertRaisesRegex(ValueError, "350"):
            validate_product_records(records)
        for index in range(346):
            row = copy.deepcopy(records["experience"][0])
            row.update(id=f"daily-{index}", prompt=f"daily fixture {index}")
            records["experience"].append(row)
        validate_product_records(records)
        records["experience"][-1] = records["experience"][0]
        with self.assertRaises(ValueError):
            validate_product_records(records)


if __name__ == "__main__":
    unittest.main()
