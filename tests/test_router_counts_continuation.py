"""Fixed engineering continuation guards with CPU/temporary-file fixtures only."""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_router_counts_continuation as runner
import verify_router_budget_counts as original


def report_fixture():
    protocol = original.plan()
    rows = protocol['descriptor']['pool']
    answers = [{'id': row['id'], 'answer': row['answer'] if '-short-' in row['id'] else 'wrong',
        'complete': True, 'passed': '-short-' in row['id'], 'branch_enabled': False, 'thinking': 'off'}
        for row in rows]
    return {'protocol': protocol, 'production_unchanged': True, 'teacher_calls': 0,
        'qualification': {'answers': answers, **original.summary(rows, answers)},
        'trials': [{'train_per_actual_class': 3, 'state': 'complete', 'candidate_gate_passed': False,
            'product_accepted': False, 'run_id': 'test-old3', 'result': {'test_accuracy': .75}}]}


class RouterContinuationTests(unittest.TestCase):
    def setUp(self):
        self.report = report_fixture()
        old = hashlib.sha256(runner.encoded(self.report['protocol']).encode()).hexdigest()
        self.original_sha = patch.object(runner, 'OLD_PLAN', old)
        self.original_sha.start(); self.addCleanup(self.original_sha.stop)

    def test_fixed6_12_are_original_nested_splits_not_new_labels_or_old3_refit(self):
        before = copy.deepcopy(self.report)
        records = runner.frozen_records(self.report)
        self.assertEqual(self.report, before)
        old = original.nested_routes(original.labelled_rows(self.report['protocol']['descriptor']['pool'],
            self.report['qualification']['answers']), 3)
        self.assertEqual(set(records), {'6', '12'})
        for name in ('development', 'calibration', 'test'):
            self.assertEqual(old[name], records['6']['splits'][name])
            self.assertEqual(old[name], records['12']['splits'][name])
        for action in ('local_core', 'teacher'):
            ids = lambda group: [r['id'] for r in group['train'] if r['route_action'] == action]
            self.assertEqual(ids(old), ids(records['12']['splits'])[:3])
            self.assertEqual(ids(records['6']['splits']), ids(records['12']['splits'])[:6])

    def test_changed_raw_labels_order_eos_and_old3_gate_are_rejected(self):
        for kind in ('order', 'eos', 'label', 'summary', 'old3', 'production', 'teacher', 'gold'):
            value = copy.deepcopy(self.report)
            if kind == 'order': value['qualification']['answers'].reverse()
            if kind == 'eos': value['qualification']['answers'][0]['complete'] = False
            if kind == 'label': value['qualification']['answers'][0]['passed'] = False
            if kind == 'summary': value['qualification']['counts']['passed'] += 1
            if kind == 'old3': value['trials'][0]['candidate_gate_passed'] = True
            if kind == 'production': value['production_unchanged'] = False
            if kind == 'teacher': value['teacher_calls'] = 1
            if kind == 'gold': value['protocol']['descriptor']['pool'][0]['answer'] = 'wrong'
            with self.subTest(kind=kind), self.assertRaises(ValueError): runner.frozen_records(value)

    def make_prior(self, root):
        prior = root / 'experiments' / runner.OLD_RUN; prior.mkdir(parents=True)
        (prior / 'report.json').write_text(runner.encoded(self.report))
        (prior / 'qualification-evidence.json').write_text(runner.encoded(self.report['qualification']))
        store = prior / 'baseline/modules/luban0-v1'; store.mkdir(parents=True)
        (store / 'graph-verification.json').write_text('{}')
        from luban.memory import Memory
        Memory(prior / 'baseline')
        (prior / 'evidence-sha256.json').write_text(runner.encoded({'files':
            {str(p.relative_to(prior)): runner.sha(p) for p in prior.rglob('*') if p.is_file()}}))
        pinned = patch.object(runner, 'OLD_EVIDENCE_SHA', runner.sha(prior / 'evidence-sha256.json'))
        pinned.start(); self.addCleanup(pinned.stop)
        return prior

    def test_static_plan_binds_current_source_and_old_raw_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); prior = self.make_prior(root)
            with patch('luban.runtime.Runtime.load', side_effect=AssertionError('no model')):
                protocol = runner.plan(root)
            self.assertEqual(protocol['train_per_actual_class'], [6, 12])
            self.assertEqual(protocol['qualification_generations'], 0)
            self.assertFalse(protocol['old3_reference']['repeat_fit'])
            self.assertIn('already observed', protocol['heldout_scope'])
            self.assertEqual(protocol['prior_report_sha256'], runner.sha(prior / 'report.json'))
            self.assertEqual(protocol['source_sha256']['src/luban/runtime.py'], runner.sha(ROOT / 'src/luban/runtime.py'))
            self.assertIn('scripts/verify_router_counts_continuation.py', protocol['source_sha256'])

    def test_preparation_rejects_changed_sealed_raw_and_manifest(self):
        for name in ('report.json', 'qualification-evidence.json', 'evidence-sha256.json',
                     'baseline/modules/luban0-v1/graph-verification.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve(); prior = self.make_prior(root)
                path = prior / name; path.write_text(path.read_text() + ' ')
                with self.assertRaisesRegex(ValueError, 'original sealed'):
                    runner.plan(root)

    def test_isolated_failure_records_exact_admission_releases_and_never_requalifies(self):
        import torch
        from luban.memory import Memory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); self.make_prior(root)
            store = root / 'modules/luban0-v1'; store.mkdir(parents=True)
            (store / 'graph-verification.json').write_text('{}')
            memory = Memory(root)
            with memory.connect(): pass
            db_before = runner.sha(memory.path)
            protocol = runner.plan(root); output = root / 'experiments/new-engineering-attempt'
            owners = []
            class FakeRuntime:
                def __init__(self, settings, memory, research):
                    assert research is True
                    self.identity = protocol['identity']; self.active_core = None
                    self.branch = SimpleNamespace(expert=None)
                    self.router = torch.nn.Linear(2, 3); self.released = False
                    self.model = torch.nn.Linear(2, 2); self.device = "mps"
                    owners.append(self)
                def load(self): pass
                def context_status(self, *, shallow=False):
                    return {'current_limit': 0, 'available_memory_bytes': 42, 'additional_weight_bytes': 0, 'shallow': shallow}
                def calibrate_router(self):
                    value = self.context_status(shallow=True)
                    assert value['available_memory_bytes'] == 42
                    raise ValueError('fixture admission failure')
                def _release_model(self): self.released = True
            class FakeSession:
                def __init__(self, runtime, kind, arguments, records, scope):
                    assert scope == 'research' and arguments == {'steps': 160}
                    assert len(records['splits']['train']) == 12
            @contextmanager
            def measured(runtime):
                measurement = {}
                try: yield measurement
                finally: measurement['elapsed_seconds'] = .01
            with patch('luban.runtime.Runtime', FakeRuntime), patch('luban.training_checkpoint.TrainingSession', FakeSession), \
                patch.object(runner, 'measured', measured), patch('torch.mps.driver_allocated_memory', return_value=7), \
                patch('luban.evolution._generation', side_effect=AssertionError('never requalify')), \
                self.assertRaisesRegex(ValueError, 'fixture admission failure'):
                runner.execute(protocol, root / 'unused-model', root, output)
            saved = json.loads((output / 'report.json').read_text())
            self.assertTrue(owners[0].released)
            self.assertTrue(saved['production_unchanged'])
            self.assertEqual(runner.sha(memory.path), db_before)
            self.assertEqual(saved['trials'][0]['state'], 'aborted')
            self.assertEqual(saved['trials'][0]['last_admission']['status']['available_memory_bytes'], 42)
            self.assertEqual(saved['trials'][0]['last_admission']['mps_driver_bytes'], 7)
            self.assertEqual(saved['trials'][0]['last_admission']['arguments'], {'shallow': True})
            self.assertEqual(saved['trials'][0]['measurement']['elapsed_seconds'], .01)
            self.assertEqual(len(saved['trials']), 1)


if __name__ == '__main__': unittest.main()
