"""Stdlib admission tests for the actual activation script's quality validator."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/activate_luban_foundation.py'
TARGET = 'User-authorized persona fixture'
# Importing the command would import Torch and a model runner. Compile its exact
# stdlib-only admission functions so these tests cannot allocate a model.
tree = ast.parse(SOURCE.read_text())
subset = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in {'sha', 'verified_quality'}], type_ignores=[])
scope = dict(hashlib=hashlib, json=json, math=math, re=re, Path=Path, TARGET=TARGET)
exec(compile(subset, str(SOURCE), 'exec'), scope)


class ActivationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = {'groups': {'new_training': [{'id':'n1'}], 'new_validation':[{'id':'n2'}]}}
        self.baseline = {'replay':[{'id':'r1','passed':True},{'id':'r2','passed':False}], 'replay_loss':.5}
        observed = lambda identifier: {'id':identifier,'passed':True,'complete':True,'reference_used':False,'branch_enabled':False}
        self.quality = {'passed':True,'target':TARGET,'new_first':[observed('n1'),observed('n2')],
            'new_second':[observed('n1'),observed('n2')],'old_protection':[observed('r1')],
            'replay_loss':.52,'baseline_replay_loss':.5,'teacher_api_calls':0}
        self.write()
        (self.root/'native900-candidate.safetensors').write_bytes(b'synthetic path only; no model')

    def tearDown(self): self.temp.cleanup()

    def write(self):
        for name,value in (('protocol.json',self.protocol),('baseline.json',self.baseline),('native900-quality.json',self.quality)):
            (self.root/name).write_text(json.dumps(value))

    def check(self): return scope['verified_quality'](self.root,'native900')

    def test_complete_two_observations_and_original_correct_subset(self):
        checked=self.check()
        self.assertEqual(checked['total_steps'],900)
        self.assertEqual(checked['candidate_file'],'native900-candidate.safetensors')
        self.assertEqual(checked['quality_sha256'],scope['sha'](self.root/'native900-quality.json'))

    def test_missing_duplicate_or_empty_proof_rows_cannot_pass(self):
        original=copy.deepcopy(self.quality)
        for key,rows in (('new_first',[]),('new_second',[original['new_second'][0]]),
                         ('new_first',[original['new_first'][0]]*2),('old_protection',[])):
            self.quality=copy.deepcopy(original);self.quality[key]=rows;self.write()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()

    def test_incomplete_or_reference_or_branch_output_cannot_pass(self):
        original=copy.deepcopy(self.quality)
        for key,value in (('complete',False),('reference_used',True),('branch_enabled',True),('passed',False)):
            self.quality=copy.deepcopy(original);self.quality['new_second'][0][key]=value;self.write()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()

    def test_changed_budget_nan_or_paid_teacher_count_rejected(self):
        original=copy.deepcopy(self.quality)
        for key,value in (('replay_loss',.536),('replay_loss',float('nan')),
                          ('baseline_replay_loss',.49),('teacher_api_calls',1),('target','different')):
            self.quality=copy.deepcopy(original);self.quality[key]=value;self.write()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()

    def test_output_name_cannot_escape_root_or_invent_step_total(self):
        for name in ('../native900','native0','native900/other','unknown'):
            with self.subTest(name=name),self.assertRaises(ValueError):scope['verified_quality'](self.root,name)


if __name__=='__main__':unittest.main()
