"""Offline stdlib admission and real tiny cache lifecycle; no 2B allocation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_cached_training_audit as runner


class CachedTrainingRunnerTests(unittest.TestCase):
    def test_default_is_stdlib_no_torch_and_no_model_allocation(self):
        value = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/run_cached_training_audit.py')],
            capture_output=True, text=True, check=True)
        report = json.loads(value.stdout)
        self.assertIs(report['torch_imported'], False)
        self.assertIs(report['model_started'], False)

    def test_fixed_train_permission_sha_and_unique_output_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder).resolve() / 'unallocated'
            plan = runner.prepare(output)
            self.assertEqual([r['id'] for r in plan['samples']], list(runner.IDS))
            self.assertEqual(plan['optimizer']['steps_per_native_and_cache_clone'], 2)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError): runner.prepare(ROOT / 'unowned-real-output')
            with patch.object(runner, 'PRIOR_SHA', '0' * 64), self.assertRaises(ValueError):
                runner.prepare(output)
            output.mkdir()
            with self.assertRaises(ValueError): runner.prepare(output)

    def test_execute_bad_sha_or_unsealed_refuses_without_allocating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder).resolve() / 'plan.json'
            output = Path(folder).resolve() / 'never-created'
            runner.write_new(path, {'protocol': runner.PROTOCOL, 'sealed': False, 'output': str(output)})
            with self.assertRaises(ValueError): runner.execute(path, '0' * 64)
            with self.assertRaises(ValueError): runner.execute(path, runner.sha(path))
            self.assertFalse(output.exists())

    def test_binding_rejects_source_sample_and_production_changes_before_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = runner.prepare(Path(folder).resolve() / 'new')
            with patch.object(runner, 'prepare', return_value=copy.deepcopy(plan)):
                bad = copy.deepcopy(plan); bad['samples'][0]['answer'] = 'altered'
                with self.assertRaises(ValueError): runner.bind(bad)
                bad = copy.deepcopy(plan); bad['source_sha256'][str(ROOT / 'scripts/run_cached_training_audit.py')] = '0' * 64
                with self.assertRaises(ValueError): runner.bind(bad)
            bound = copy.deepcopy(plan)
            bound.update(official_model_binding={'cpu': True}, production_snapshot={'cpu': True},
                         sealed=True, state='sealed_waiting_explicit_MODEL_GO')
            with patch.object(runner, 'prepare', return_value=plan), patch.object(runner, 'bind', return_value={**bound, 'sealed': False, 'state': 'bound_waiting_root_seal_MODEL_GO'}):
                runner.validate_bound(bound)
            with patch.object(runner, 'prepare', return_value=plan), patch.object(runner, 'bind', return_value={**bound, 'production_snapshot': {'changed': True}}), self.assertRaises(ValueError):
                runner.validate_bound(bound)

    def test_actual_tiny_graph_two_step_audit_and_cpu_lock_release_on_qualification_failure(self):
        import torch
        from luban.process_lock import ModelProcessLock
        import luban.runtime as runtime_module
        import run_identity_coverage
        import verify_router_coverage_study
        import verify_cached_training
        import test_cached_training
        for qualifies, audit_fault in ((True, False), (False, False), (True, True)):
            with self.subTest(qualifies=qualifies, audit_fault=audit_fault), tempfile.TemporaryDirectory() as folder:
                path = Path(folder).resolve()
                rt, rows, _ = test_cached_training.fixture(path)
                rt.research = True; rt.device = 'cpu'; rt.dtype = torch.float32
                rt.active_core = None; rt.identity = runner.IDENTITY
                rt._model_process_lock = None
                original = verify_cached_training.tensor_sha(rt.branch.base.down_proj.weight)
                rt.load = lambda *, paging: None
                def release():
                    rt.model = None
                    if rt._model_process_lock is not None: rt._model_process_lock.close()
                    rt._model_process_lock = None
                rt._release_model = release
                out = path / 'result'
                plan = {'protocol': runner.PROTOCOL, 'sealed': True, 'state': 'sealed_waiting_explicit_MODEL_GO',
                    'output': str(out), 'source_sha256': {}, 'production_snapshot': {'fixture': True},
                    'production_root': str(path), 'official_model_binding': {'fixture': True},
                    'model_path': str(path), 'device': 'cpu', 'dtype': 'torch.float32', 'seed': 42, 'samples': rows}
                protocol = path / 'protocol.json'; runner.write_new(protocol, plan)
                proof = {'id': rows[-1]['id'], 'passed': qualifies, 'complete': True,
                    'answer': 'OLIVE_PATH' if qualifies else 'wrong', 'branch_enabled': False, 'reference_used': False}
                with patch.object(runner, 'validate_bound'), patch.object(runner, 'ORIGINAL', original), \
                        patch.object(runtime_module, 'Runtime', return_value=rt), \
                        patch.object(runner, 'production_snapshot', return_value={'fixture': True}), \
                        patch.object(verify_router_coverage_study, 'model_binding', return_value={'fixture': True}), \
                        patch.object(run_identity_coverage, 'verify_training_prefix_alignment', return_value={'fixture': True}), \
                        patch('luban.evolution._generation', return_value=[proof]):
                    if audit_fault:
                        with patch.object(verify_cached_training, '_ce', side_effect=ValueError('CPU forced failure after native backward')), self.assertRaises(RuntimeError):
                            runner.execute(protocol, runner.sha(protocol))
                    elif qualifies: runner.execute(protocol, runner.sha(protocol))
                    else:
                        with self.assertRaises(RuntimeError): runner.execute(protocol, runner.sha(protocol))
                report = json.loads((out / 'result.json').read_text())
                self.assertEqual(report['passed'], qualifies and not audit_fault)
                self.assertTrue(report['model_released'])
                self.assertTrue(report['actual_process_lock_reacquired_and_closed'])
                self.assertEqual(report['generation_calls'], 1)
                if audit_fault:
                    self.assertTrue(all(report['failure_restoration'].values()))
                elif qualifies:
                    self.assertEqual(len(report['audit']['steps']), 2)
                    self.assertTrue(report['audit']['restored'])
                    self.assertEqual(report['original_tensor_sha256_after'], original)
                lock = ModelProcessLock(runner.IDENTITY, 'cpu'); lock.close()
                # This same output can never be consumed for a second attempt.
                with patch.object(runner, 'validate_bound'), self.assertRaises(FileExistsError):
                    runner.execute(protocol, runner.sha(protocol))


if __name__ == '__main__':
    unittest.main()
