"""Real tiny CPU commits cannot regain pending coverage by editing their report."""
import copy
import json
import unittest

from luban.automatic import knowledge_keys, receipt_keys
from luban.training_checkpoint import training_runs
import test_partial_evolution as partial
from luban.evolution import consolidate


class PartialCoverageTests(unittest.TestCase):
    def test_current_report_cannot_downgrade_partial_receipt_to_legacy(self):
        fixture = partial.PartialEvolutionTests()
        self.addCleanup(fixture.doCleanups)
        _, runtime, _ = fixture.fixture()
        report = consolidate(runtime, **partial.ARGS)
        runs = training_runs(runtime.memory, runtime.identity)
        runs[0]['product_accepted'] = True  # Isolated CPU coverage fixture only.
        _, covered, error = knowledge_keys(runtime, runs)
        self.assertIsNone(error)
        self.assertTrue(covered)
        path = runtime.store / ('evolution-' + report['id'] + '.json')
        original = path.read_bytes()
        for mutation in ('policy', 'mastered', 'missing_metadata', 'report_symlink'):
            path.write_bytes(original)
            bad = copy.deepcopy(report)
            metadata_path = runtime.store / (report['version'] + '.json')
            metadata = metadata_path.read_bytes()
            if mutation == 'policy':
                bad.pop('partial_policy')
                path.write_text(json.dumps(bad))
            elif mutation == 'mastered':
                bad['learning_progress']['mastered_ids'] += bad['learning_progress']['pending_ids']
                path.write_text(json.dumps(bad))
            elif mutation == 'missing_metadata':
                metadata_path.unlink()
            else:
                target = path.with_suffix('.original')
                target.write_bytes(original)
                path.unlink()
                path.symlink_to(target.name)
            with self.subTest(mutation=mutation):
                _, current, error = knowledge_keys(runtime, runs)
                self.assertIsNotNone(error)
                self.assertEqual(current, set())
            if path.is_symlink():
                path.unlink()
            metadata_path.write_bytes(metadata)
        path.write_bytes(original)
        self.assertIsNone(knowledge_keys(runtime, runs)[2])

    def test_receipt_policy_must_match_frozen_training_arguments(self):
        fixture = partial.PartialEvolutionTests()
        self.addCleanup(fixture.doCleanups)
        _, runtime, _ = fixture.fixture()
        consolidate(runtime, **partial.ARGS)
        row = training_runs(runtime.memory, runtime.identity)[0]
        receipt_keys(row)
        for location in ('arguments', 'report'):
            changed = copy.deepcopy(row)
            target = changed['arguments'] if location == 'arguments' else changed['commit']['report']
            target.pop('partial_policy')
            with self.subTest(location=location), self.assertRaises(ValueError):
                receipt_keys(changed)
