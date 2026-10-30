"""CPU checks for the explicit asymmetric source-fidelity conversion policy."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts')]
from prepare_q4_sequential import (asymmetric_errors, exact_original_baseline, fixed_protection,
                                   quantize_stage, reference_stage, validate_asymmetric_options)
from q4_sequential_protection import establish_baseline, protection_cases


class AsymmetricDriverTests(unittest.TestCase):
    def setUp(self):
        self.cases = protection_cases()
        self.correct = [{'id': c['id'], 'tokens':[1,2], 'complete':True,
            'text': c['check']['expected'] if isinstance(c['check']['expected'],str)
                    else json.dumps(c['check']['expected'])} for c in self.cases]
        self.original = deepcopy(self.correct)
        self.original[1]['text'] = 'original error'
        self.baseline = establish_baseline(self.cases,self.original)

    def test_old_union_still_rejects_lost_previous_gain(self):
        result = fixed_protection(self.cases,self.baseline,self.correct,self.original)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['lost_required_correct_ids'],[self.cases[1]['id']])
        self.assertNotIn('protection_policy',result)

    def test_new_policy_preserves_original_and_reports_lost_prior_gain(self):
        result = fixed_protection(self.cases,self.baseline,self.correct,self.original,
                                  original_bf16_preservation=True)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['protection_policy'],'original_bf16_preservation')
        self.assertEqual(result['prior_q4_only_lost_ids'],[self.cases[1]['id']])
        self.assertFalse(result['prior_union_would_accept'])
        self.assertEqual(result['lost_required_correct_ids'],[])
        changed=deepcopy(self.original);changed[0]['text']='lost original'
        rejected=fixed_protection(self.cases,self.baseline,self.correct,changed,
                                   original_bf16_preservation=True)
        self.assertFalse(rejected['accepted'])
        self.assertEqual(rejected['lost_required_correct_ids'],[self.cases[0]['id']])

    def test_exact_original_baseline_includes_original_errors_and_tokens(self):
        rows=[{**row,'prompt_tokens':[3,4]} for row in self.original]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'baseline-generations.json').write_text(json.dumps(rows))
            exact_original_baseline(rows,root)
            for key,value in [('tokens',[2,1]),('complete',False),('text','fixed original error'),('prompt_tokens',[4,3])]:
                changed=deepcopy(rows);changed[1][key]=value
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'fresh original'):
                    exact_original_baseline(changed,root)

    def options(self):
        return SimpleNamespace(mode='convert',asymmetric_reference=Path('/reference'),
            refit_grid_from=Path('/prior'),task_trajectories_from=Path('/prior'),regularize_from=None)

    def test_asymmetric_mode_requires_absolute_bound_same_task_source(self):
        args=self.options();validate_asymmetric_options(args)
        for key,value in [('mode','control'),('refit_grid_from',None),('task_trajectories_from',None),
                          ('task_trajectories_from',Path('/other')),('regularize_from',Path('/prior')),
                          ('asymmetric_reference',Path('relative'))]:
            bad=self.options();setattr(bad,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_asymmetric_options(bad)
        args.asymmetric_reference=None;args.mode='control';validate_asymmetric_options(args)

    def test_reference_reader_receives_exact_stage_split_and_row_counts(self):
        raw={'calibration':np.zeros((5,32),np.uint16),'observation':np.zeros((3,32),np.uint16)}
        self.assertIsNone(reference_stage(None,2,1,raw))
        with patch('capture_q4_task_reference.read_reference',side_effect=lambda ref,layer,kind,split,n:
                   np.zeros((n,32),np.float32)) as read:
            result=reference_stage({'bound':True},2,1,raw)
            self.assertEqual(read.call_args_list[0].args,({'bound':True},2,1,'calibration',5))
            self.assertEqual(read.call_args_list[1].args,({'bound':True},2,1,'observation',3))
            self.assertEqual(result['observation'].shape,(3,32))
        with patch('capture_q4_task_reference.read_reference',side_effect=ValueError('unbound reference')):
            with self.assertRaisesRegex(ValueError,'unbound'):reference_stage({},2,1,raw)

    def test_wrong_reference_rows_nonfinite_and_first_layer_mismatch_fail_before_fit(self):
        values={s:np.zeros((3,16),np.uint16) for s in ('calibration','observation')}
        good={s:np.zeros((3,16),np.float32) for s in values}
        for candidate,layer in [({**good,'calibration':np.zeros((2,16),np.float32)},1),
                                ({**good,'observation':np.full((3,16),np.nan,np.float32)},1),
                                ({**good,'calibration':np.ones((3,16),np.float32)},0)]:
            with patch('prepare_q4_sequential.inverse_factor') as factor:
                with self.assertRaises(ValueError):
                    quantize_stage(layer,('gate','up'),values,{},Path('/unused'),
                                   refit_grid=True,reference_values=candidate)
                factor.assert_not_called()

    def test_asymmetric_metrics_use_original_input_target_and_separate_new_grid(self):
        weights=np.full((2,16),2,np.float32)
        codes=np.ones((2,16),np.uint8);scales=np.full((2,1),2,np.float32);biases=np.zeros((2,1),np.float32)
        original=np.ones((3,16),np.float32);current=original*.5
        result=asymmetric_errors(weights,codes,scales,biases,current,original)
        self.assertEqual(result['mse'],256)
        self.assertEqual(result['relative_mse'],.25)
        self.assertEqual(result['sampled_output_rows'],[0,1])
        improved=asymmetric_errors(weights,codes,scales*2,biases,current,original)
        self.assertEqual(improved['mse'],0)

    def test_correction_uses_calibration_only_and_gateup_share_single_factor(self):
        values={'calibration':np.full((4,16),0x3f80,np.uint16),
                'observation':np.full((3,16),0x4000,np.uint16)}
        reference={'calibration':np.full((4,16),3,np.float32),
                   'observation':np.full((3,16),7,np.float32)}
        upper=torch.eye(16)
        with tempfile.TemporaryDirectory() as tmp, patch('prepare_q4_sequential.inverse_factor',
                return_value=(upper,{'dead_columns':[]})) as inverse, patch('prepare_q4_packed.asymmetric_factor',
                return_value=torch.zeros((16,16))) as factor:
            # Stop after factor preparation, before any model tensor is read.
            with self.assertRaises(KeyError):
                quantize_stage(1,('gate','up'),values,{},Path(tmp),refit_grid=True,reference_values=reference)
            inverse.assert_called_once()
            factor.assert_called_once()
            np.testing.assert_array_equal(factor.call_args.args[0],reference['calibration'])
            np.testing.assert_array_equal(factor.call_args.args[1],np.ones((4,16),np.float32))
            self.assertIs(factor.call_args.args[2],upper)
            self.assertEqual(factor.call_args.kwargs,{'alpha':.25})


if __name__=='__main__':unittest.main()
