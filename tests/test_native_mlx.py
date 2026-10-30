"""MLX trial bindings cannot inherit ordinary-Q4 expertise or stale sources."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from luban.native_mlx import model_binding, expert_binding, require_training_source
from luban.settings import Settings


class NativeMLXTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.write('model/download-receipt.json', {'complete': True, 'revision': 'fixed', 'files': [
            {'name': 'model.safetensors', 'downloaded': True, 'bytes': 7, 'sha256': 'a' * 64}]})
        self.write('model/config.json', {'model_type': 'qwen3_5', 'quantization': {'bits': 4}, 'text_config': {
            'hidden_size': 2048, 'intermediate_size': 6144, 'num_hidden_layers': 24, 'vocab_size': 248320}})
        (self.model / 'model.safetensors').write_bytes(b'fixture')
        self.runtime = SimpleNamespace(native_inference_precision='mlx-q4', native_mlx_model_path=str(self.model),
            loaded_storage=str(self.root), identity='source', active_core='core-current', store=self.root,
            pointer=self.root / 'active.json', branch=SimpleNamespace(expert_id='learned-trial',
                _paths=lambda _: (self.root / 'expert.safetensors', self.root / 'expert.json')))
        self.runtime.router_meta = {'id': 'router-trial', 'sha256': 'e' * 64, 'execution_scope': 'local-trial',
            'feature_source': 'mlx-q4-prefix',
            'quality_admitted': False, 'trial_engine': 'mlx-q4', 'expert_id': 'learned-trial',
            'expert_sha256': 'd' * 64, 'core_version': 'core-current'}
        self.binary = self.root / 'worker' 
        self.write('worker.json', {'binary_sha256': 'b' * 64})
        self.write('active.json', {'id': 'learned-trial', 'identity': 'source', 'core_version': 'core-current', 'width': 64})
        self.write('active-core.json', {'version': 'core-current', 'identity': 'source', 'sha256': 'c' * 64})
        self.record = {'identity': 'source:layer-23', 'hidden': 2048, 'width': 64, 'requires_gate': True,
                       'sha256': 'd' * 64, 'core_version': 'core-current'}
        self.write('expert.json', self.record)
        self.proof = {'schema': 'mlx-expert-local-trial-v1', 'scope': 'local-trial', 'quality_admitted': False,
            'math_passed': True, 'runtime_checked': True, 'identity': 'source', 'expert_id': 'learned-trial',
            'core_version': 'core-current', 'core_file_sha256': 'c' * 64, 'expert_file_sha256': 'd' * 64,
            'worker_binary_sha256': 'b' * 64, 'mlx_model_sha256': 'a' * 64, 'width': 64, 'route_checked': True,
            'router_id': 'router-trial', 'router_sha256': 'e' * 64}

    def write(self, path, data):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))

    def test_model_binding_reuses_verified_download_and_detects_changed_shape_or_size(self):
        expected = model_binding(self.runtime)
        self.assertEqual(model_binding(self.runtime), expected)
        self.assertEqual(expected['weight_bytes'], 7)
        (self.model / 'model.safetensors').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, '来源发生变化'):
            model_binding(self.runtime)

    def test_missing_proof_does_not_enable_old_expert(self):
        self.assertIsNone(expert_binding(self.runtime, self.binary))

    def test_trial_proof_binds_actual_model_worker_core_expert_and_not_quality(self):
        self.write('runtime/mlx/expert-proof.json', self.proof)
        self.assertEqual(expert_binding(self.runtime, self.binary)['core_sha256'], 'c' * 64)
        for key, bad in [('worker_binary_sha256', 'f' * 64), ('mlx_model_sha256', 'f' * 64),
                         ('scope', 'production'), ('quality_admitted', True), ('math_passed', 1),
                         ('runtime_checked', False), ('expert_id', 'old'), ('width', 32), ('route_checked', False),
                         ('router_sha256', 'f' * 64)]:
            self.write('runtime/mlx/expert-proof.json', {**self.proof, key: bad})
            with self.subTest(key=key), self.assertRaises(ValueError):
                expert_binding(self.runtime, self.binary)
        self.write('runtime/mlx/expert-proof.json', self.proof)
        self.write('expert.json', {**self.record, 'core_version': None})
        with self.assertRaises(ValueError):
            expert_binding(self.runtime, self.binary)

    def test_trial_source_cannot_be_used_as_an_admitted_training_parent(self):
        self.write('layer-23/learned-trial.json', {**self.record, 'quality_admitted': False})
        with self.assertRaisesRegex(ValueError, '本地试用'):
            require_training_source(self.runtime)
        self.write('layer-23/learned-trial.json', self.record)
        require_training_source(self.runtime)
        self.write('active-router.json', {'id': 'router-trial'})
        self.write('router-trial.json', {'execution_scope': 'local-trial'})
        with self.assertRaisesRegex(ValueError, '本地试用'):
            require_training_source(self.runtime)

    def test_trial_cannot_inherit_or_publish_mastery(self):
        from luban.runtime import Runtime
        self.assertIsNone(Runtime.external_binding(self.runtime, actual_action="neural_expert"))
        self.assertIsNone(Runtime.external_binding(self.runtime, actual_action="local_core"))

    def test_formal_admission_requires_actual_complete_report_not_relabelled_trial(self):
        from luban.mlx_learning import accepted_report, QUALITY_GATES
        flags = dict(execution_scope='product', inference_engine='mlx-q4',
                     quality_admitted=True, training_admitted=True)
        self.runtime.router_meta.update(flags)
        self.write('expert.json', {**self.record, **flags})
        self.write('active.json', {'id': 'learned-trial', 'identity': 'source',
                                  'core_version': 'core-current', 'width': 64, **flags})
        proof = {**self.proof, 'schema': 'mlx-expert-learning-v1', 'scope': 'product',
                 'quality_admitted': True, 'training_admitted': True,
                 'report': 'learning-123456abcdef.json', 'report_sha256': '0' * 64}
        self.write('runtime/mlx/expert-proof.json', proof)
        with self.assertRaises((ValueError, OSError)):
            expert_binding(self.runtime, self.binary)
        rows = [{'id': str(i), 'prompt': 'question ' + str(i), 'answer': 'ok',
                 'purpose': 'training' if i < 16 else 'validation', 'source': 'human',
                 'approved': True, 'training_allowed': True, 'provenance': {}}
                for i in range(19)]
        observations = [{'id': row['id'], 'answer': 'ok', 'passed': True, 'complete': True,
                         'reference_used': False, 'tokens': [1], 'input_tokens': [2],
                         'route_action': 'neural_expert'} for row in rows]
        report = dict(promoted=True, quality_checked=True, inference_engine='mlx-q4',
                      base_identity='source', core_version='core-current', active_expert='learned-trial',
                      chosen_width=64, gates=dict.fromkeys(QUALITY_GATES, True),
                      records={'train': rows[:16], 'valid': rows[16:18], 'protection': rows[18:]},
                      observations={'candidate': [observations, copy.deepcopy(observations)]})
        def publish(value):
            self.write(proof['report'], value)
            proof['report_sha256'] = hashlib.sha256((self.root / proof['report']).read_bytes()).hexdigest()
            self.write('runtime/mlx/expert-proof.json', proof)
        publish(report)
        self.assertEqual(accepted_report(self.runtime, self.binary)['active_expert'], 'learned-trial')
        invalid = copy.deepcopy(report)
        invalid['observations']['candidate'][1][-1]['answer'] = 'forgotten'
        publish(invalid)
        with self.assertRaisesRegex(ValueError, '原标准'):
            accepted_report(self.runtime, self.binary)
        invalid = copy.deepcopy(report)
        invalid['gates']['mlx_routes_available'] = False
        publish(invalid)
        with self.assertRaisesRegex(ValueError, '路由验证'):
            accepted_report(self.runtime, self.binary)

    def test_setting_requires_absolute_mlx_model_without_changing_learning_model(self):
        settings = Settings(self.root / 'settings')
        before = settings.snapshot()['model_path']
        data, _ = settings.validated({'native_inference_precision': 'mlx-q4', 'native_mlx_model_path': str(self.model)})
        self.assertEqual(data['model_path'], before)
        for changes in ({'native_inference_precision': 'mlx-q4', 'native_mlx_model_path': ''},
                        {'native_mlx_model_path': 'relative'}):
            with self.assertRaises(ValueError):
                settings.validated(changes)
