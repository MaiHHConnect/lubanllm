import copy
import json
import unittest
from unittest.mock import patch
from luban import factory_engine as factory


class FactoryEngineTests(unittest.TestCase):
    def setUp(self):
        self.old, self.new = 'a'*64, 'b'*64
        self.proof = dict(schema='mlx-expert-learning-v1', expert_id='learned-bafb4050374f', width=32,
                          quality_admitted=True, training_admitted=True, core_version='core-fdfa0ce5a75c',
                          worker_binary_sha256=self.old, expert_file_sha256='c'*64,
                          core_file_sha256='d'*64, mlx_model_sha256='e'*64)
        self.training = dict(schema='mlx-native-training-qualification-v1', complete=True,
                             worker_binary_sha256=self.new, parent_expert_file_sha256='c'*64,
                             core_file_sha256='d'*64, mlx_model_sha256='e'*64,
                             gates={key:True for key in factory.GATES})
        self.record = dict(schema='luban-factory-engine-qualification-v1', complete=True, teacher_calls=0,
                           sources_unchanged=True, worker_binary_sha256=self.new, source_proof=self.proof,
                           training=self.training,
                           math=dict(passed=True, fp32_pass=True, bf16_pass_assumed=False, bf16_numerical_pass=True,
                                     bf16_relative_l2_limit=.03, expert_file_sha256='c'*64,
                                     stages=[dict(dtype=d, stage=s, finite=True, nonzero=True, fp32_tolerance_pass=True, relative_l2=.001)
                                             for d in ('fp32','bf16') for s in ('gate','up','activation','output')]),
                           quality=dict(complete=True, rows=80, two_passes_equal=True, routes_equal=True,
                                        protected_answers_equal=True, all_previously_correct_still_pass=True))

    def accepted(self, record, proof=None, worker=None):
        with patch.object(factory.Path, 'read_text', return_value=json.dumps([record])):
            return factory.worker_qualified(proof or self.proof, worker or self.new)

    def test_original_exact_worker_still_works_without_any_new_qualification(self):
        with patch.object(factory.Path, 'read_text', side_effect=AssertionError('must not read')):
            self.assertTrue(factory.worker_qualified(self.proof, self.old))
        self.assertFalse(factory.worker_qualified({}, None))

    def test_new_exact_source_and_training_are_separately_qualified(self):
        self.assertTrue(self.accepted(self.record))
        with patch.object(factory.Path, 'read_text', return_value=json.dumps([self.record])):
            self.assertEqual(factory.training_qualification({}, self.new, 'd'*64, 'e'*64), self.training)
            self.assertEqual(factory.training_qualification({}, self.new, 'f'*64, 'e'*64), {})
        for key in ('expert_file_sha256','core_file_sha256','mlx_model_sha256','worker_binary_sha256'):
            proof = {**self.proof,key:'f'*64}
            self.assertFalse(self.accepted(self.record,proof))
        self.assertFalse(self.accepted(self.record,worker='f'*64))

    def test_all_math_lifecycle_and_quality_conditions_fail_closed(self):
        for gate in factory.GATES:
            row=copy.deepcopy(self.record); row['training']['gates'][gate]=False
            self.assertFalse(self.accepted(row),gate)
        for key in ('complete','two_passes_equal','routes_equal','protected_answers_equal','all_previously_correct_still_pass'):
            row=copy.deepcopy(self.record); row['quality'][key]=False
            self.assertFalse(self.accepted(row),key)
        mutations = [lambda r:r['math'].update(fp32_pass=False), lambda r:r['math'].update(bf16_pass_assumed=True),
                     lambda r:r['math']['stages'][4].update(relative_l2=.031),
                     lambda r:r['math']['stages'][0].update(finite=False),
                     lambda r:r['math']['stages'].pop(), lambda r:r.update(worker_binary_sha256='invalid'),
                     lambda r:r['training'].pop('complete'), lambda r:r['source_proof'].update(expert_id='other')]
        for mutation in mutations:
            row=copy.deepcopy(self.record); mutation(row); self.assertFalse(self.accepted(row))
        with patch.object(factory.Path,'read_text',return_value='[]'):
            self.assertFalse(factory.worker_qualified(self.proof,self.new))
