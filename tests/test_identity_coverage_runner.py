"""Private seeding and one-use acceptance safety without model loading."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from functools import partial
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_identity_coverage import (qualified_groups, seed_private_groups, require_untouched_engineering,
                                   recover_prepared_publication, verify_training_prefix_alignment)
from test_core_publication import fixture_plan
from luban.evolution import consolidation_samples
from luban.memory import Memory
from luban.runtime import Runtime


class IdentityCoverageRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = fixture_plan()

    def seed_original(self, memory):
        with memory.connect() as db:
            for row in self.fixture["source_rows"]:
                db.execute("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?)", tuple(row.values()))

    def test_private_seeding_matches_complete_fixed_groups_preserving_source_provenance(self):
        memory = Memory(self.root)
        self.seed_original(memory)
        source = copy.deepcopy(self.fixture["source_rows"])
        seed_private_groups(memory, self.fixture["groups"], source)
        runtime = SimpleNamespace(memory=memory, identity="fixture-base", active_core=None, store=self.root / "modules")
        expected = {key: sorted(rows, key=lambda row: row["id"]) for key, rows in self.fixture["groups"].items()}
        self.assertEqual(consolidation_samples(runtime), expected)
        indexed = {row["id"]: row for row in memory.samples()}
        for row in source:
            self.assertEqual(indexed[row["id"]]["answer"], row["answer"])
            self.assertEqual(indexed[row["id"]]["provenance"], row["provenance"])
        self.assertEqual(source, self.fixture["source_rows"])

    def test_conflict_in_last_private_source_does_not_partially_seed(self):
        memory = Memory(self.root)
        self.seed_original(memory)
        with memory.connect() as db:
            db.execute("UPDATE samples SET answer='changed' WHERE id=?", (self.fixture["source_rows"][-1]["id"],))
        before = memory.samples()
        with self.assertRaisesRegex(ValueError, "anchored source"):
            seed_private_groups(memory, self.fixture["groups"], self.fixture["source_rows"])
        self.assertEqual(memory.samples(), before)

    def test_qualification_failure_cannot_create_training_corpus(self):
        groups = copy.deepcopy(self.fixture["groups"])
        qualified = groups["replay_training"][-4:]
        groups["replay_training"] = groups["replay_training"][:-4]
        plan = {"groups_before_original_baseline": groups, "prepared": 100.}
        before = copy.deepcopy(plan)
        with self.assertRaisesRegex(ValueError, "min4"):
            qualified_groups(plan, {"passed": False, "qualified_count": 3, "qualified": qualified[:3]})
        result = qualified_groups(plan, {"passed": True, "qualified_count": 4, "qualified": qualified})
        self.assertEqual(len(result["replay_training"]), 48)
        self.assertEqual(result["new_validation"], sorted(groups["new_validation"], key=lambda row: row["id"]))
        self.assertEqual(plan, before)

    def test_interrupted_engineering_is_rejected_before_question_reuse(self):
        (self.root / "engineering-started.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "interrupted"):
            require_untouched_engineering(self.root, {"engineering": [{"passed": True}] * 15})
        require_untouched_engineering(self.root, {"engineering_complete": True})

    def test_unbound_acceptance_cannot_be_recovered_or_allocate_model(self):
        report = {"publication_prepared": {"name": "wrong", "sha256": "0" * 64}}
        plan = {"session_id": self.fixture["run_id"]}
        path = self.root / ("coverage-acceptance-" + plan["session_id"] + ".json")
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "acceptance changed"):
            recover_prepared_publication(plan, self.root, report)
        self.assertFalse((self.root / "app/memory.sqlite3").exists())

    def prefix_fixture(self, mismatch=False):
        class Tokenizer:
            eos_token_id = 9
            def __init__(self):
                self.contexts = []
            def apply_chat_template(self, context, **kwargs):
                self.contexts.append(copy.deepcopy(context))
                if context[0]['content'] == 'heldout must not be read':
                    raise AssertionError('Heldout prefix leakage')
                return torch.tensor([[1, 3 if mismatch else 2]]) if kwargs['tokenize'] else 'prefix'
            def encode(self, value, **kwargs):
                return [1, 2] if value == 'prefix' else [7, 8]
        runtime = SimpleNamespace(tokenizer=Tokenizer(), device=torch.device('cpu'),
            context_status=lambda *, shallow=False: {'current_limit': 512, 'linear_bytes_per_token': 0,
                'quadratic_bytes_per_token_squared': 0, 'memory_budget_bytes': 1})
        runtime._tokens = partial(Runtime._tokens, runtime)
        runtime._sample_tokens = partial(Runtime._sample_tokens, runtime)
        row = {'id': 'training-prefix', 'prompt': 'uppercase', 'answer': 'UPPERCASE',
               'provenance': json.dumps({'messages': [{'role': 'user', 'content': 'uppercase'}]})}
        heldout = {**row, 'id': 'heldout', 'provenance': json.dumps({'messages': [
                   {'role': 'user', 'content': 'heldout must not be read'}]})}
        plan = {'prefix_alignment_required': True, 'groups_before_original_baseline': {
            'new_training': [row], 'replay_training': [{**row, 'id': 'replay'}],
            'new_validation': [heldout], 'replay_validation': [heldout]},
            'replay_training_candidates': [{**row, 'id': 'candidate'}], 'engineering_cases': [heldout]}
        return runtime, plan

    def test_real_runtime_token_methods_align_gold_eos_without_reading_any_heldout(self):
        runtime, plan = self.prefix_fixture()
        result = verify_training_prefix_alignment(runtime, plan)
        self.assertTrue(result['passed'])
        self.assertEqual(result['count'], 3)
        self.assertEqual([row['id'] for row in result['rows']], ['training-prefix', 'replay', 'candidate'])
        self.assertEqual(len(runtime.tokenizer.contexts), 6)
        self.assertIsNone(verify_training_prefix_alignment(runtime, {'prefix_alignment_required': False}))

    def test_generation_tokenized_prefix_mismatch_rejects_actual_training_labels(self):
        runtime, plan = self.prefix_fixture(mismatch=True)
        with self.assertRaisesRegex(ValueError, 'alignment failed: training-prefix'):
            verify_training_prefix_alignment(runtime, plan)
        self.assertEqual(len(runtime.tokenizer.contexts), 2)

    def test_changed_gold_eos_label_rejects_even_when_inference_prefix_matches(self):
        runtime, plan = self.prefix_fixture()
        original = runtime._sample_tokens
        def missing_eos(row):
            ids, labels = original(row)
            labels[0, -1] = -100
            return ids, labels
        runtime._sample_tokens = missing_eos
        with self.assertRaisesRegex(ValueError, 'gold\\+EOS token alignment failed'):
            verify_training_prefix_alignment(runtime, plan)


if __name__ == "__main__":
    unittest.main()
