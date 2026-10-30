"""Pure stdlib admission/grade fixtures; no model/tokenizer/Torch/PT reads."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import verify_curve_saved_point as runner
from luban.generation_checks import evaluate_generation


def row(role,index):
    rule={'type':'contains' if role=='new' and index==12 else 'exact','expected':'ok','max_new_tokens':4}
    return {'id':role+'-'+str(index).zfill(2),'question':'fixture question '+str(index),'answer':'ok','purpose':'training',
        'source':'synthetic_fixture','approved':1,'training_allowed':1,'created':1,
        'provenance':json.dumps({'messages':[{'role':'user','content':role+' fixture '+str(index)}],
            'evolution_role':role,'generation_check':rule,'scope':'CPU fictional admission fixture; not actual qualification'})}


def observed(sample,correct,candidate):
    answer='ok' if correct else 'wrong';rule=json.loads(sample['provenance'])['generation_check']
    return {'id':sample['id'],'answer':answer,**evaluate_generation(answer,rule,True),
        'branch_enabled':False,'reference_used':False,'sample_sha256':runner.content_sha(sample),
        'actual_tensor_sha256':candidate,'actual_base_identity':runner.IDENTITY,'core_version':None,
        'raw_completion_token_ids':[9,2],'decoded_completion_unstripped':answer,
        'completion_tokens':2,'max_new_tokens':4,'thinking':'off','do_sample':False}


def fixture():
    groups={'new_training':[row('new',i) for i in range(57)],'replay_training':[row('replay',i) for i in range(57)],
            'new_validation':[],'replay_validation':[]}
    baseline={'new':[observed(r,i<13,runner.ORIGINAL) for i,r in enumerate(groups['new_training'])],
        'replay':[observed(r,i<55,runner.ORIGINAL) for i,r in enumerate(groups['replay_training'])],'replay_loss':1.}
    candidate='f'*64
    first=[observed(r,True,candidate) for r in groups['new_training']]
    old=[observed(r,True,candidate) for r in groups['replay_training'][:55]]
    finite=runner.prior.correct(groups['new_training'],first,finite=True)
    finite_before=runner.prior.correct(groups['new_training'],baseline['new'],finite=True)
    saved={'path':'/fictional-fixture-only/lastdown-ce450.pt','file_sha256':'e'*64,'tensor_sha256':candidate,
        'dtype':'torch.bfloat16','shape':[2048,6144],'tensor_bytes':2048*6144*2,'file_bytes':2048*6144*2+1000}
    point={'ce_events':450,'candidate_tensor_sha256':candidate,'saved_weights':saved,'new_first':first,'new_second':[],
        'old_preservation':old,'mastery_confirmed':False,'newly_confirmed_ids':None,'safety_passed':True,'publication':False,
        'forgotten_ids':[],'provisional_correct_ids':sorted(finite),'provisional_growth_ids':sorted(finite-finite_before),
        'baseline_replay_loss':1.,'replay_loss':1.}
    second={'ce_event':450,'saved_weights':saved,'new_second':copy.deepcopy(first),'old_preservation':copy.deepcopy(old),
        'mastery_confirmed':True,'safety_passed':True,'publication':False,'forgotten_ids':[],
        'newly_confirmed_ids':sorted(finite-finite_before),'replay_loss':1.}
    return groups,baseline,point,second


class CurveSavedPointTests(unittest.TestCase):
    def test_default_cli_stdlib_waiting_no_model(self):
        raw=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/verify_curve_saved_point.py')],capture_output=True,text=True,check=True)
        value=json.loads(raw.stdout)
        self.assertFalse(value['model_started']);self.assertFalse(value['torch_imported'])
        self.assertEqual(value['optimizer_steps'],0);self.assertFalse(value['publication'])

    def test_unselected_or_missing_actual_receipt_never_reads_curve(self):
        with patch.object(runner,'closed_receipt',side_effect=AssertionError('Must not read live/absent receipt')):
            for path,point in ((None,None),(None,450),('/never-read',None)):
                value=runner.prepare('/fixture-output',path,point)
                self.assertTrue(value['state'].startswith('waiting'));self.assertEqual(value['wall_seconds'],900)
                with self.assertRaises(ValueError):runner.bind(value)
        for invalid in (True,451,704,352):
            with self.assertRaises(ValueError):runner.prepare('/fixture-output',None,invalid)

    def test_cpu_or_incomplete_post_receipt_rejected_before_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'receipt.json'
            for value in ({'scope':'cpu_fixture','pid_dead':True,'process_exit_code':0},
                          {'scope':'actual_closed_stepcurve_saved_point_receipt','pid_dead':False,'process_exit_code':0},
                          {'scope':'actual_closed_stepcurve_saved_point_receipt','pid_dead':True,'process_exit_code':True}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):runner.closed_receipt(path,450)

    def test_first_proof_binding_and_original_nonfinite_retention(self):
        groups,baseline,point,second=fixture()
        value=runner.selected_summary(groups,baseline,point,450)
        self.assertEqual(len(value['original_correct_ids']['new']),13)
        self.assertEqual(len(value['original_correct_ids']['finite_new']),12)
        self.assertEqual(len(value['original_correct_ids']['replay']),55)
        for changed in ('context','candidate','shape','wrong_point','incomplete_first'):
            altered=copy.deepcopy(point)
            if changed=='context':altered['new_first'][0]['sample_sha256']='0'*64
            elif changed=='candidate':altered['new_first'][0]['actual_tensor_sha256']='0'*64
            elif changed=='shape':altered['saved_weights']['shape']=[2,3]
            elif changed=='wrong_point':altered['ce_events']=500
            else:altered['new_first'].pop()
            with self.assertRaises(ValueError):runner.selected_summary(groups,baseline,altered,450)

    def test_second_intersection_never_counts_original13_or_single_proof(self):
        groups,baseline,point,second=fixture();original=copy.deepcopy((groups,baseline,point))
        row20=groups['new_training'][20]
        second['new_second'][20]=observed(row20,False,point['candidate_tensor_sha256'])
        second['newly_confirmed_ids'].remove(row20['id'])
        result=runner.verify_second(groups,baseline,point,second)
        self.assertEqual(result['candidate_confirmed_growth'],43)
        self.assertNotIn(row20['id'],result['candidate_confirmed_growth_ids'])
        self.assertFalse(set(r['id'] for r in groups['new_training'][:13]) & set(result['candidate_confirmed_growth_ids']))
        self.assertEqual(result['production_learned_ids'],[]);self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual((groups,baseline,point),original)
        second['newly_confirmed_ids'].append(row20['id'])
        with self.assertRaises(ValueError):runner.verify_second(groups,baseline,point,second)

    def test_old_protection_and_loss_fail_closed_without_upgrade(self):
        groups,baseline,point,second=fixture()
        for changed in ('old_wrong','loss','different_savedW','wrong_context'):
            value=copy.deepcopy(second)
            if changed=='old_wrong':value['old_preservation'][0]=observed(groups['replay_training'][0],False,point['candidate_tensor_sha256'])
            elif changed=='loss':value['replay_loss']=1.06
            elif changed=='different_savedW':value['saved_weights']['file_sha256']='0'*64
            else:value['new_second'][0]['sample_sha256']='0'*64
            with self.assertRaises(ValueError):runner.verify_second(groups,baseline,point,value)

    def test_actual_generationEOS_not_trainingEOS_and_incomplete_fullbudget(self):
        groups,baseline,point,second=fixture();sample=groups['new_training'][0];candidate=point['candidate_tensor_sha256']
        value=observed(sample,True,candidate)
        self.assertTrue(runner.raw_proofs([sample],[value],candidate,[2]))
        for changed in ('trainEOS','internalEOS','truncated','decode','boolean_token'):
            altered=copy.deepcopy(value)
            if changed=='trainEOS':altered['raw_completion_token_ids']=[9,3]
            elif changed=='internalEOS':altered['raw_completion_token_ids']=[2,2]
            elif changed=='truncated':altered['complete']=False;altered['passed']=False
            elif changed=='decode':altered['decoded_completion_unstripped']='wrong'
            else:altered['raw_completion_token_ids']=[True,2]
            with self.assertRaises(ValueError):runner.raw_proofs([sample],[altered],candidate,[2])
        incomplete=observed(sample,False,candidate);incomplete.update(raw_completion_token_ids=[9]*4,completion_tokens=4,complete=False,passed=False)
        self.assertTrue(runner.raw_proofs([sample],[incomplete],candidate,[2]))

    def test_unsealed_run_rejected_before_model_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'unsealed.json';path.write_text(json.dumps({'protocol':runner.PROTOCOL,'sealed':False}))
            with patch.object(runner,'validate_bound',side_effect=AssertionError('No model/preflight allowed')):
                with self.assertRaises(ValueError):runner.execute(path,hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual([p.name for p in path.parent.iterdir()],['unsealed.json'])


if __name__=='__main__':unittest.main()
