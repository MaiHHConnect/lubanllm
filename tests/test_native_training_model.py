"""Stdlib admission fixtures only; no actual model/tensor/native execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import native_training_model as model


class NativeTrainingModelTests(unittest.TestCase):
    def test_source_import_does_not_load_torch(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)

    def test_original450_schedule_and50_prefix_match_actual_event_order(self):
        groups = {'new_training': [{'id': 'n' + str(i)} for i in range(57)],
                  'replay_training': [{'id': 'r' + str(i)} for i in range(57)]}
        full = model.schedule(groups, 450)
        self.assertEqual(model.schedule(groups, 50), full[:50])
        self.assertEqual([s['ordinal'] for s in full], list(range(1, 451)))
        for i, step in enumerate(full):
            role = 'new' if i % 2 == 0 else 'replay'; pair = i // 2 % 57
            self.assertEqual(step, {'ordinal': i + 1, 'role': role,
                'ce': ('n' if role == 'new' else 'r') + str(pair), 'replay': 'r' + str(pair)})

    def test_other_budgets_rejected(self):
        for steps in (True, 2, 51, 449, 704):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                model.schedule({}, steps)

    def test_prepare_rejects_product_root_and_allows_fresh_nested_experiment(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home).resolve(); product = root / 'product'; experiments = product / 'experiments'
            private = experiments / 'unique'; private.mkdir(parents=True)
            refs = tuple(root / str(i) for i in range(3))
            for p in refs: p.write_text('{}')
            parent = {'plan': {'protocol': model.end.PROTOCOL, 'production_root': str(product), 'model_path': str(root / 'official')},
                'raw': {'groups_sha256': 'a' * 64}, 'groups': {'new_training': [{'id': 'n' + str(i)} for i in range(57)],
                'replay_training': [{'id': 'r' + str(i)} for i in range(57)]}, 'baseline': {}}
            with patch.object(model.closed, 'closed_parent', return_value=parent), patch.object(model, 'PARENT', refs), \
                    patch.object(model, 'sources', return_value={}):
                for path in (product, product / 'models', experiments / 'execution'):
                    with self.subTest(path=path), self.assertRaises(ValueError): model.prepare(path, 'produce')
                plan = model.prepare(private / 'execution', 'produce')
                self.assertFalse(plan['sealed']); self.assertEqual(plan['generation_upper_bound'], 0)
                self.assertFalse((private / 'execution').exists())

    def test_bind_rejects_tokenizer_changed_even_when_identity_same(self):
        parent = {'plan': {'official_model_binding': {'identity': model.IDENTITY, 'files': {'tokenizer.json': 'old'}}},
                  'raw': {'official_after': {'identity': model.IDENTITY, 'files': {'tokenizer.json': 'old'}}}}
        plan = {'inputs': None, 'output': '/unused', 'action': 'produce', 'model_path': '/unused', 'production_root': '/unused'}
        changed = {'identity': model.IDENTITY, 'files': {'tokenizer.json': 'new'}}
        with patch.object(model, 'prepare', return_value=plan), patch.object(model.closed, 'closed_parent', return_value=parent), \
                patch('verify_router_coverage_study.model_binding', return_value=changed), \
                patch.object(model, 'production_snapshot') as snapshot:
            with self.assertRaisesRegex(ValueError, 'exact old13'): model.bind(plan)
            snapshot.assert_not_called()

    def test_quality_cannot_accept_live_self_report_as_closed_receipt(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home).resolve(); fixture = root / 'fixture50'; fixture.mkdir()
            (fixture / 'manifest.tsv').write_text('synthetic; ABI parser is mocked for this closure guard')
            metadata = {'metadata': {'scope': 'actual_python_frozen_prefix', 'identity': model.IDENTITY,
                'core_version': None, 'original_native_tensor_sha256': model.ORIGINAL}}
            (fixture / 'fixture.json').write_text(json.dumps(metadata))
            (root / 'result.json').write_text(json.dumps({'passed': True, 'pid': 123}))
            # An apparently passed self-report is insufficient. No Root exit/post exists.
            manifest = {'steps': [None] * 50, 'config': {'identity': model.IDENTITY, 'device': 'mps'},
                        'tensors': {'candidate': {'sha256': model.ORIGINAL}}}
            with patch.object(model.abi, 'fixture', return_value=manifest), \
                    patch.object(model.abi, 'output_receipt') as receipt:
                with self.assertRaises(FileNotFoundError): model.external_inputs(fixture, root / 'py', root / 'cpp', root / 'comparison.json')
                receipt.assert_not_called()

    def test_invalid_approved_sha_fails_before_binding(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home).resolve() / 'formal.json'; path.write_text('{}')
            with patch.object(model, 'validate_bound') as validate:
                for value in (None, '', 'a' * 63, 'a' * 64):
                    with self.assertRaises(ValueError): model.execute(path, value)
                validate.assert_not_called()


if __name__ == '__main__': unittest.main()
