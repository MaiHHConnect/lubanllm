"""Guard against misleading intervention selection and omission of layer zero."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from intervene_q4_layers import pair_result, select_layers, request
from intervene_q4_joint import CONFIGS
from verify_q4_joint_candidate import select_candidate


class Q4InterventionTests(unittest.TestCase):
    def test_pair_margin_does_not_claim_full_vocab_win(self):
        result=pair_result({'traces':[{'top1':99,'probe_tokens':[32,33],'probe_logits':[4.,3.]}]})
        self.assertEqual(result['margin'],1.)
        self.assertFalse(result['preferred_is_top1'])

    def test_selection_prefers_full_vocab_winner_and_low_layer_tie(self):
        rows=[{'case_id':'a','layer':i,'probe':{'preferred_is_top1':i!=4,'margin':9 if i==4 else 1}}
              for i in (2,3,4)]
        self.assertEqual(select_layers(rows),[0,2,23])

    def test_layer_zero_and_probe_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            events=iter([{'event':'trace','step':0,'committed_tokens':2,'probe_tokens':[32,33]},
                         {'event':'result'}])
            stdin=io.StringIO()
            worker=SimpleNamespace(out=Path(directory),process=SimpleNamespace(stdin=stdin),receive=lambda:next(events))
            spec={'case':{'tokens':[1,2],'max_tokens':4,'eos_ids':[9]},'forced_tokens':[]}
            request(worker,spec,'test',layer=0,probe=[32,33])
            sent=json.loads(stdin.getvalue())
            self.assertEqual(sent['restore_layer'],0)
            self.assertEqual(sent['max_tokens'],1)
            self.assertFalse(sent['cache_reuse'])
            self.assertEqual(sent['tokens'],[1,2])

    def test_joint_restore_and_positive_control_cover_every_layer(self):
        self.assertEqual(CONFIGS['all_layers'],list(range(24)))
        self.assertEqual(set(CONFIGS['even_layers'])|set(CONFIGS['odd_layers']),set(range(24)))
        with tempfile.TemporaryDirectory() as directory:
            stdin=io.StringIO()
            events=iter([{'event':'result','tokens':[]}])
            worker=SimpleNamespace(out=Path(directory),process=SimpleNamespace(stdin=stdin),receive=lambda:next(events))
            spec={'case':{'tokens':[1,2],'max_tokens':4,'eos_ids':[9]},'forced_tokens':[]}
            request(worker,spec,'joint',layers=[0,8,22])
            self.assertEqual(json.loads(stdin.getvalue())['restore_layers'],[0,8,22])
            for layers in ([],[0,0],[24]):
                with self.assertRaises(ValueError):
                    request(worker,spec,'bad',layers=layers)
            with self.assertRaises(ValueError):
                request(worker,spec,'bad',layer=0,layers=[1])

    def test_full_qualification_excludes_all_bf16_and_prefers_smaller_restore(self):
        passed={'passed':5,'failed':[]}
        with self.assertRaises(ValueError):
            select_candidate({'summary':{'q4':{'all_layers':passed}}})
        self.assertEqual(select_candidate({'summary':{'rank8':{'even_layers':passed},
                          'rank32':{'observed_pair':passed}}}),('rank32','observed_pair',[8,22]))


if __name__=='__main__':
    unittest.main()
