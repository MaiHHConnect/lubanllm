"""Offline random tiny Qwen3.5/SQLite fixtures, never the archived2B tensors."""
import copy
import json
from pathlib import Path
import hashlib
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import diagnose_field_training as diagnostic
from luban.backbone import text_backbone
from luban.dynamic import DynamicMLP
from luban.process_lock import ModelProcessLock
from verify_luban0_final import tensor_digest


def tiny_runtime(folder, dtype=torch.float32):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    config = Qwen3_5Config(text_config={
        'vocab_size': 64, 'hidden_size': 32, 'intermediate_size': 64,
        'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 8,
        'linear_key_head_dim': 8, 'linear_value_head_dim': 8,
        'linear_num_key_heads': 2, 'linear_num_value_heads': 4, 'linear_conv_kernel_dim': 4,
        'max_position_embeddings': 128, 'layer_types': ['linear_attention', 'full_attention'],
        'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0,
                            'partial_rotary_factor': 1.0, 'mrope_section': [2, 1, 1]},
        'bos_token_id': 1, 'eos_token_id': 2, 'pad_token_id': 0,
    }, vision_config={'depth': 1, 'hidden_size': 16, 'intermediate_size': 32, 'num_heads': 2,
                      'out_hidden_size': 32, 'patch_size': 2, 'temporal_patch_size': 2,
                      'spatial_merge_size': 2, 'num_position_embeddings': 16},
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
        tie_word_embeddings=True)
    model = Qwen3_5ForConditionalGeneration(config).cpu().to(dtype)
    decoder, _ = text_backbone(model)
    branch = DynamicMLP(decoder.layers[-1].mlp, 32, 8, Path(folder), 'fixture', 100000)
    decoder.layers[-1].mlp = branch
    ids = torch.tensor([[3, 7, 13, 19, 23, 29, 31, 2]])
    labels = ids.clone()
    labels[:, :4] = -100
    runtime = SimpleNamespace(model=model, decoder=decoder, branch=branch, active_core=None,
        store=folder, tokenizer=SimpleNamespace(encode=lambda *a, **k: [23, 29, 31], eos_token_id=2),
        _sample_tokens=lambda sample: (ids, labels), _tokens=lambda *args: ids[:, :4])
    sample = {'id': 'offline-fixture', 'answer': 'fixture gold',
              'provenance': json.dumps({'messages': [{'role': 'user', 'content': 'fixture'}]})}
    basis = torch.linalg.qr(torch.randn(64, 3), mode='reduced').Q
    return runtime, sample, basis, labels


def payload_fixture(folder):
    master = torch.randn(4, 6)
    basis = torch.linalg.qr(torch.randn(6, 2), mode='reduced').Q
    data = {'session_id': 'fixture', 'parameter': 'fixture.weight', 'parameter_shape': [4, 6],
            'basis_shape': [6, 2], 'original_tensor_sha256': 'fixture-original', 'identity': 'a' * 64}
    binding = {'scope': 'research', 'core_version': None, 'fixture': True}
    saved = {'binding': binding, 'payload': {'format': 1, 'id': 'fixture', 'step': 832,
        'parameter': data['parameter'], 'original_sha256': data['original_tensor_sha256'],
        'master': master, 'basis': basis, 'report': {'steps': 832, 'identity': data['identity'],
            'parameter': data['parameter'], 'training_selection': {'selected_step': None}}},
        'rng': torch.get_rng_state(), 'mps_rng': None}
    path = Path(folder) / 'fixture.pt'
    torch.save(saved, path)
    plan = {'data': data, 'candidate_path': str(path), 'candidate_file_sha256': diagnostic.sha(path),
            'checkpoint_binding': binding}
    return plan, saved


class FieldTrainingDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name).resolve()

    def tearDown(self):
        self.folder.cleanup()

    def test_default_and_metadata_prepare_are_stdlib_only_no_tensor_reads(self):
        code = """
import pathlib,sys
sys.path.insert(0,'scripts')
import diagnose_field_training as d
old=pathlib.Path.open
def guarded(path,*args,**kwargs):
    assert path.suffix not in {'.pt','.safetensors'}, 'unapproved tensor read'
    return old(path,*args,**kwargs)
pathlib.Path.open=guarded
p=d.prepare_plan(pathlib.Path(sys.argv[1]))
assert len(p['row_ids'])==9 and p['tensor_binding'] is None
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print(p['state'])
"""
        run = subprocess.run([sys.executable, '-S', '-c', code, str(self.path / 'future-output')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('waiting_explicit_bind', run.stdout)
        default = subprocess.run([sys.executable, '-S', 'scripts/diagnose_field_training.py'],
                                 cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertFalse(json.loads(default.stdout)['torch_imported'])
        self.assertFalse((self.path / 'future-output').exists())

    def test_prepare_allows_experiments_but_rejects_product_archive_model_project(self):
        data, old, *_ = diagnostic.metadata()
        allowed = Path(old['production_root']) / 'experiments/future-diagnostic-NOT-GO'
        self.assertEqual(diagnostic.prepare_plan(allowed)['output'], str(allowed))
        for path in (Path(old['production_root']) / 'app/new', Path(old['production_root']) / 'modules/new',
                     Path(data['source_snapshot']) / 'new', Path(old['model_path']) / 'new', ROOT / 'new'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                diagnostic.prepare_plan(path)

    def test_actual_cpu_weights_only_mmap_payload_guards_and_tensor_binding(self):
        plan, saved = payload_fixture(self.path)
        _, master, basis = diagnostic.tensors(plan)
        bound = diagnostic.tensor_binding(master, basis, 'torch.bfloat16')
        self.assertEqual(bound['master_fp32_sha256'], tensor_digest(saved['payload']['master']))
        self.assertLess(bound['basis_orthogonality_max_abs'], 1e-4)
        for change in ('binding', 'step', 'parameter', 'original', 'shape', 'dtype', 'selected', 'nonfinite'):
            value = copy.deepcopy(saved)
            if change == 'binding': value['binding']['scope'] = 'product'
            if change == 'step': value['payload']['step'] = 816
            if change == 'parameter': value['payload']['parameter'] = 'other'
            if change == 'original': value['payload']['original_sha256'] = 'other'
            if change == 'shape': value['payload']['master'] = torch.zeros(1, 6)
            if change == 'dtype': value['payload']['basis'] = value['payload']['basis'].bfloat16()
            if change == 'selected': value['payload']['report']['training_selection']['selected_step'] = 832
            if change == 'nonfinite': value['payload']['master'][0, 0] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnostic.validate_payload(value, plan)
        changed = copy.deepcopy(plan)
        changed['candidate_file_sha256'] = '0' * 64
        with self.assertRaises(ValueError): diagnostic.tensors(changed)
        with self.assertRaises(ValueError): diagnostic.tensor_binding(master, torch.ones_like(basis), 'torch.float32')

    def test_native_causal_mask_tail_and_fp32_cast_formula_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                runtime, sample, basis, _ = tiny_runtime(self.path, dtype)
                runtime.model.eval().requires_grad_(False)
                runtime.branch.base.down_proj.weight.requires_grad_(True)
                result = diagnostic.diagnose_row(runtime, sample, basis, 7)
                self.assertEqual(result['gold_eos_token_ids'], [23, 29, 31, 2])
                self.assertEqual(result['first_predict_position_zero_based'], 3)
                self.assertEqual(len(result['per_target_token_ce_fp32']), 4)
                self.assertTrue(result['independent_native_gradient']['exact'])
                self.assertTrue(result['independent_native_logits']['exact'])
                self.assertTrue(result['hooks_removed'])
                self.assertGreater(result['native_first_ce_gradient_norm_fp32_cast'], 0)
                self.assertLess(result['native_vs_fp32_outer_relative_difference'], 2 ** -7)
                self.assertAlmostEqual(result['native_mean_ce'], result['computed_mean_ce'], places=6)
                self.assertLessEqual(result['effective_input_ratio_fp32'], 1.000001)
                self.assertNotEqual(result['maximum_non_gold_token'], result['first_gold_token'])
                self.assertFalse(result['observed832_is_gold'])
                same = diagnostic.diagnose_row(runtime, sample, basis, 23)
                self.assertTrue(same['observed832_is_gold'])
                self.assertEqual(same['gold_minus_observed832_logit'], 0)

    def test_restore_original_candidate_all_state_on_success_and_failure(self):
        runtime, sample, basis, _ = tiny_runtime(self.path, torch.bfloat16)
        runtime.branch.expert = torch.nn.Linear(32, 32, bias=False).bfloat16()
        branch = runtime.branch.expert
        weight = runtime.branch.base.down_proj.weight
        original = weight.detach().clone()
        weight.grad = torch.ones_like(weight)
        prior_grad = weight.grad
        prior_flags = [p.requires_grad for p in runtime.model.parameters()]
        runtime.model.train()
        runtime.decoder.layers[0].eval()
        prior_modes = [m.training for m in runtime.model.modules()]
        master = original.float() + .01
        receipt = []
        report = diagnostic.inspect_states(runtime, [sample], master, basis, {sample['id']: {'token_id': 7}},
            tensor_digest(original), lambda value: receipt.append(copy.deepcopy(value)), tensor_digest(master.bfloat16()))
        self.assertEqual(len(report['states']), 2)
        self.assertNotEqual(report['states'][0]['native_weight_sha256'], report['states'][1]['native_weight_sha256'])
        native = diagnostic.diagnose_row
        calls = []
        def failing(*args):
            calls.append(1)
            if len(calls) == 2: raise ValueError('fixture failure after candidate loaded')
            return native(*args)
        with patch.object(diagnostic, 'diagnose_row', side_effect=failing), self.assertRaisesRegex(ValueError, 'fixture failure'):
            diagnostic.inspect_states(runtime, [sample], master, basis, {sample['id']: {'token_id': 7}},
                tensor_digest(original), lambda value: receipt.append(copy.deepcopy(value)))
        self.assertTrue(receipt[-1]['restoration']['passed'])
        self.assertTrue(torch.equal(weight, original))
        self.assertIs(weight.grad, prior_grad)
        self.assertIs(runtime.branch.expert, branch)
        self.assertEqual([p.requires_grad for p in runtime.model.parameters()], prior_flags)
        self.assertEqual([m.training for m in runtime.model.modules()], prior_modes)

    def test_mask_and_native_forward_failure_reject_and_remove_hooks(self):
        runtime, sample, basis, labels = tiny_runtime(self.path)
        runtime.model.eval().requires_grad_(False)
        runtime.branch.base.down_proj.weight.requires_grad_(True)
        labels[0, 2] = 13
        with self.assertRaisesRegex(ValueError, 'mask mismatch'):
            diagnostic.diagnose_row(runtime, sample, basis, 7)
        labels[0, 2] = -100
        modules = [runtime.decoder.layers[-1].post_attention_layernorm, runtime.branch.base.down_proj]
        before = [len(m._forward_pre_hooks) for m in modules]
        with patch.object(runtime.model, 'forward', side_effect=RuntimeError('fixture forward failed')):
            with self.assertRaisesRegex(RuntimeError, 'fixture forward failed'):
                diagnostic.diagnose_row(runtime, sample, basis, 7)
        self.assertEqual([len(m._forward_pre_hooks) for m in modules], before)

    def test_execute_real_cpu_lock_wal_preservation_failure_receipt_and_single_output(self):
        from luban.runtime import Runtime
        from verify_router_coverage_study import model_binding
        product, model_path = self.path / 'product', self.path / 'official-fixture'
        (product / 'app').mkdir(parents=True)
        (product / 'modules/luban0-v1').mkdir(parents=True)
        (product / 'modules/luban0-v1/unchanged.json').write_text('{"fixture":true}')
        db = sqlite3.connect(product / 'app/memory.sqlite3')
        self.addCleanup(db.close)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE fixture (data TEXT)')
        db.execute('INSERT INTO fixture VALUES (?)', ('committed-WAL-fixture',))
        db.commit()
        model_path.mkdir()
        (model_path / 'config.json').write_text('{}')
        (model_path / 'fixture.safetensors').write_bytes(b'not a real model: only hash fixture')
        files = [{'Path': name, 'Sha256': diagnostic.sha(model_path / name)}
                 for name in ('config.json', 'fixture.safetensors')]
        (model_path / 'upstream-file-manifest.json').write_text(json.dumps({'Data': {'Files': files}}))
        identity = hashlib.sha256(json.dumps(sorted((r['Path'], r['Sha256']) for r in files)).encode()).hexdigest()
        torch.manual_seed(42)
        runtime, sample, basis, _ = tiny_runtime(self.path, torch.bfloat16)
        original = runtime.branch.base.down_proj.weight.detach().clone()
        plan, payload = payload_fixture(self.path)
        data = plan['data']
        data.update(identity=identity, original_tensor_sha256=tensor_digest(original),
            parameter='model.language_model.layers.1.mlp.base.down_proj.weight', parameter_shape=[32, 64],
            basis_shape=[64, 3], source_snapshot=str(self.path / 'immutable'), records=[sample],
            observed832_first_token={sample['id']: {'token_id': 7}})
        payload['payload'].update(parameter=data['parameter'], original_sha256=data['original_tensor_sha256'],
                                  master=original.float() + .01, basis=basis)
        payload['payload']['report'].update(identity=identity, parameter=data['parameter'])
        torch.save(payload, plan['candidate_path'])
        data['candidate_file_sha256'] = diagnostic.sha(plan['candidate_path'])
        old = {'production_root': str(product), 'model_path': str(model_path)}
        binding = {**plan['checkpoint_binding'], 'device': 'cpu', 'dtype': 'torch.bfloat16'}
        payload['binding'] = binding
        torch.save(payload, plan['candidate_path'])
        data['candidate_file_sha256'] = diagnostic.sha(plan['candidate_path'])
        source = {str(Path(diagnostic.__file__).resolve()): diagnostic.sha(diagnostic.__file__)}
        del runtime  # No shadow model during execution.

        class FixtureRuntime(Runtime):
            def __init__(self, settings, memory, research):
                self.settings, self.memory, self.research = settings, memory, research
                self.device, self.dtype = 'cpu', torch.bfloat16
                self._model_process_lock = None
                self.model = self.decoder = self.branch = None

            def load(self, *, paging):
                self.store = self.memory.root / 'modules/luban0-v1'
                self.store.mkdir(parents=True)
                fixture, _, _, _ = tiny_runtime(self.store, torch.bfloat16)
                self.__dict__.update(fixture.__dict__)
                self.identity = identity

        native = diagnostic.diagnose_row
        for fail in (False, True):
            calls = []
            def row(*args):
                calls.append(1)
                if fail and len(calls) == 2: raise ValueError('CPU fixture candidate failure')
                return native(*args)
            output = self.path / ('failed' if fail else 'success')
            with patch.object(diagnostic, 'metadata', return_value=(data, old, Path(plan['candidate_path']), source, binding)):
                prepared = diagnostic.prepare_plan(output)
                bound = copy.deepcopy(prepared)
                bound.update(tensor_binding=diagnostic.tensor_binding(payload['payload']['master'], basis, 'torch.bfloat16'),
                    official_model_binding=model_binding(model_path, identity),
                    production_snapshot=diagnostic.production_snapshot(product), state='bound_waiting_root_seal_and_MODEL_GO')
                protocol_path = self.path / ('failure-plan.json' if fail else 'success-plan.json')
                diagnostic.write_new(protocol_path, bound)
                with self.assertRaisesRegex(ValueError, 'approved exact protocol SHA'):
                    diagnostic.execute(protocol_path, '0' * 64)
                with patch('luban.runtime.Runtime', FixtureRuntime), patch.object(diagnostic, 'diagnose_row', side_effect=row):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'saved in result.json'):
                            diagnostic.execute(protocol_path, diagnostic.sha(protocol_path))
                    else:
                        diagnostic.execute(protocol_path, diagnostic.sha(protocol_path))
                result = json.loads((output / 'result.json').read_text())
                self.assertEqual(result['passed'], not fail)
                self.assertTrue(result['diagnostic']['restoration']['passed'])
                self.assertTrue(result['preserved'])
                self.assertTrue(result['model_released'])
                self.assertTrue(result['actual_process_lock_reacquired_and_closed'])
                self.assertIn('sampled_peak_rss_bytes', result['resources'])
                if fail: self.assertIn('candidate failure', result['error']['message'])
                lock = ModelProcessLock(identity, 'cpu')
                lock.close()
                with patch('luban.runtime.Runtime', side_effect=AssertionError('must not load second model')):
                    with self.assertRaises(FileExistsError):
                        diagnostic.execute(protocol_path, diagnostic.sha(protocol_path))


if __name__ == '__main__':
    unittest.main()
