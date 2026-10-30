import copy
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adopt_mlx_teacher_repair import net_progress, publish, read
from prepare_mlx_expert_trial import save


class RepairAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(id=str(i),purpose='training' if i==0 else 'validation',input_tokens=[i+1],answer='yes') for i in range(4)]

    def pair(self, correct):
        values=[dict(id=r['id'],input_tokens=r['input_tokens'],tokens=[1] if i in correct else [2],
                     answer='yes' if i in correct else 'no',finish_reason='stop',reference_used=False,passed=i in correct)
                for i,r in enumerate(self.rows)]
        return [values,copy.deepcopy(values)]

    def test_one_pending_does_not_block_positive_gain(self):
        progress=net_progress(self.rows,self.pair({0,1}),self.pair({0,1,2}))
        self.assertTrue(progress['passed'])
        self.assertEqual(progress['pending_validation_ids'],['3'])
        self.assertEqual(progress['net_gain'],1)

    def test_gain_must_exceed_loss(self):
        self.assertFalse(net_progress(self.rows,self.pair({0,1}),self.pair({0,2}))['passed'])
        self.assertTrue(net_progress(self.rows,self.pair({0,1}),self.pair({0,2,3}))['passed'])

    def test_train_success_does_not_count(self):
        self.assertFalse(net_progress(self.rows,self.pair({1}),self.pair({0,1}))['passed'])

    def test_forged_score_is_recomputed(self):
        bad=self.pair({0,1,2});bad[0][2]['answer']=bad[1][2]['answer']='no'
        with self.assertRaisesRegex(AssertionError,'Forged'):
            net_progress(self.rows,self.pair({0,1}),bad)

    def test_changed_input_and_nondeterminism_rejected(self):
        bad=self.pair({0,1,2});bad[1][2]['tokens']=[99]
        with self.assertRaisesRegex(AssertionError,'Non-deterministic'):
            net_progress(self.rows,self.pair({0,1}),bad)
        bad=self.pair({0,1,2});bad[0][2]['input_tokens']=bad[1][2]['input_tokens']=[99]
        with self.assertRaisesRegex(AssertionError,'Changed input'):
            net_progress(self.rows,self.pair({0,1}),bad)

    def test_publication_failure_restores_pointers(self):
        from luban.settings import private_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=root/'a.json',root/'b.json';save(a,{'id':'old-a'});save(b,{'id':'old-b'})
            old={str(a):read(a),str(b):read(b)};new={str(a):{'id':'new-a'},str(b):{'id':'new-b'}}
            artifact=root/'new.weights'
            def failing(path,value):
                if path==b and value==new[str(b)]: raise OSError('disk write failed')
                return private_json(path,value)
            with patch('luban.settings.private_json',side_effect=failing):
                with self.assertRaises(OSError):publish(root,old,{str(artifact):b'weights'},new)
            self.assertEqual(read(a),old[str(a)]);self.assertEqual(read(b),old[str(b)])
            self.assertFalse(artifact.exists());self.assertTrue(read(root/'publication.json')['rolled_back'])


if __name__=='__main__':unittest.main()
