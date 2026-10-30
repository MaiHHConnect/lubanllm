"""Original zero-Adam import receipts retain full quality and artifact checks."""
import copy
import hashlib
import json
import unittest

from luban.automatic import knowledge_keys, receipt_keys, record_keys
from luban.evolution import consolidate
from luban.training_checkpoint import training_runs
import test_partial_evolution as partial


class ImportReceiptCoverageTests(unittest.TestCase):
    def fixture(self):
        fixture = partial.PartialEvolutionTests()
        self.addCleanup(fixture.doCleanups)
        _, runtime, _, arguments = fixture.product_fixture()
        consolidate(runtime, **arguments)
        row = training_runs(runtime.memory, runtime.identity)[0]
        report = row['commit']['report']
        # Transform only the isolated CPU ledger into the original import schema.
        # This is a compatibility fixture, not evidence of a real native import.
        arguments = {'steps': 450, 'operation': 'validated-core-import',
            'source_training_steps': 450, 'current_updates': 0, 'source_run': 'a' * 12,
            'source_checkpoint_sha256': 'b' * 64, 'candidate_tensor_sha256': 'c' * 64}
        row['arguments'] = arguments
        row['binding']['arguments'] = copy.deepcopy(arguments)
        row['product_accepted'] = True
        report.update(operation='validated-core-import', source_training_steps=450,
            current_updates=0, training_updates_executed=0, source_training_run='a' * 12,
            source_checkpoint_sha256='b' * 64, converted_candidate_tensor_sha256='c' * 64)
        path = runtime.store / ('evolution-' + report['id'] + '.json')
        path.write_text(json.dumps(report))
        row['commit']['artifacts'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return runtime, row, path

    def test_original_import_covers_only_mastered_and_still_requires_immutable_evidence(self):
        runtime, row, path = self.fixture()
        report = row['commit']['report']
        records = [sample for group in row['records']['groups'].values() for sample in group]
        mastered = [sample for sample in records if sample['id'] in report['learning_progress']['mastered_ids']]
        self.assertEqual(receipt_keys(row), record_keys(mastered))
        history, covered, error = knowledge_keys(runtime, [row], mastery_only=True)
        self.assertIsNone(error)
        self.assertEqual(history, covered)
        self.assertEqual(covered, record_keys(mastered))
        pending = [sample for sample in records if sample['id'] in report['learning_progress']['pending_ids']]
        self.assertTrue(pending)
        self.assertFalse(covered & record_keys(pending))
        original = path.read_bytes()
        metadata = runtime.store / (report['version'] + '.json')
        metadata_bytes = metadata.read_bytes()
        for mutation in ('report_missing', 'metadata_missing', 'report_changed', 'bad_proof'):
            path.write_bytes(original)
            metadata.write_bytes(metadata_bytes)
            bad = copy.deepcopy(row)
            if mutation == 'report_missing':
                path.unlink()
            elif mutation == 'metadata_missing':
                metadata.unlink()
            elif mutation == 'report_changed':
                changed = copy.deepcopy(report)
                changed['learning_progress']['mastered_ids'] += changed['learning_progress']['pending_ids']
                path.write_text(json.dumps(changed))
            else:
                bad['commit']['report']['partial_candidate_second'][0]['complete'] = False
            with self.subTest(mutation=mutation):
                _, covered, error = knowledge_keys(runtime, [bad], mastery_only=True)
                self.assertIsNotNone(error)
                self.assertEqual(covered, set())

    def test_compatibility_cannot_be_claimed_by_training_or_changed_import_binding(self):
        _, row, _ = self.fixture()
        receipt_keys(row)
        for mutation in ('ordinary_training', 'report_operation', 'frozen_binding', 'current_updates',
                         'bool_updates', 'report_updates', 'source_steps', 'source_run',
                         'source_checkpoint', 'candidate_tensor', 'explicit_different_policy', 'failed_gate'):
            bad = copy.deepcopy(row)
            arguments, report = bad['arguments'], bad['commit']['report']
            if mutation == 'ordinary_training':
                arguments.pop('operation')
            elif mutation == 'report_operation':
                report['operation'] = 'training'
            elif mutation == 'frozen_binding':
                bad['binding']['arguments']['source_run'] = 'd' * 12
            elif mutation == 'current_updates':
                arguments['current_updates'] = 1
            elif mutation == 'bool_updates':
                arguments['current_updates'] = False
            elif mutation == 'report_updates':
                report['training_updates_executed'] = 1
            elif mutation == 'source_steps':
                arguments['source_training_steps'] = 449
            elif mutation == 'source_run':
                report['source_training_run'] = 'd' * 12
            elif mutation == 'source_checkpoint':
                report['source_checkpoint_sha256'] = 'd' * 64
            elif mutation == 'candidate_tensor':
                report['converted_candidate_tensor_sha256'] = 'd' * 64
            elif mutation == 'explicit_different_policy':
                arguments['partial_policy'] = 'other-policy'
            else:
                report['gates']['original_correct_retained'] = False
            if mutation != 'frozen_binding':
                bad['binding']['arguments'] = copy.deepcopy(arguments)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                receipt_keys(bad)
        # Future importer freezes the quality policy explicitly.
        row['arguments']['partial_policy'] = row['commit']['report']['partial_policy']
        row['binding']['arguments'] = copy.deepcopy(row['arguments'])
        self.assertTrue(receipt_keys(row))
