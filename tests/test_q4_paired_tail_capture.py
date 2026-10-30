import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from capture_q4_paired_tails import (TAIL_WIDTHS, VOCAB, check_frame, collect,
                                      read_frame_file, trajectory_layout)


class PairedCaptureTests(unittest.TestCase):
    def fixtures(self):
        records = [dict(id='cal-a', split='calibration', prompt_tokens=[1,2], generated_tokens=[3,9], complete=True, finish_reason='stop'),
                   dict(id='obs-a', split='observation', prompt_tokens=[4], generated_tokens=[5], complete=False, finish_reason='length'),
                   dict(id='cal-b', split='calibration', prompt_tokens=[6], generated_tokens=[7,8], complete=False, finish_reason='length')]
        plan = {'eos_ids':[9], 'cases':[dict(id=r['id'],split=r['split'],tokens=r['prompt_tokens'],forced_tokens=r['generated_tokens'][:-1]) for r in records]}
        return records, plan

    def test_complete_prediction_order_and_unforwarded_last_target(self):
        records, plan = self.fixtures(); layout = trajectory_layout(records, plan, 10)
        self.assertEqual(layout['rows_per_split'], {'calibration':4,'observation':1})
        self.assertEqual([(c['row_start'],c['row_end']) for c in layout['cases']], [(0,2),(0,1),(2,4)])
        self.assertEqual(layout['cases'][0]['prediction_positions'], [1,2])
        self.assertEqual(layout['cases'][0]['forced_tokens'], [3])
        self.assertEqual(layout['cases'][1]['forced_tokens'], [])

    def test_changed_prompt_duplicate_and_forwarded_eos_rejected(self):
        for mutation in ('prompt','duplicate','eos'):
            records, plan = self.fixtures()
            if mutation=='prompt':plan['cases'][0]['tokens']=[7]
            if mutation=='duplicate':records[1]['id']=records[0]['id']
            if mutation=='eos':records[0]['generated_tokens']=[9,9];plan['cases'][0]['forced_tokens']=[9]
            with self.assertRaises(ValueError):trajectory_layout(records,plan,10)

    def test_256_targets_allowed_but_not_257(self):
        records, plan = self.fixtures();r=records[0];p=plan['cases'][0]
        r['generated_tokens']=[3]*255+[9];p['forced_tokens']=[3]*255
        self.assertEqual(len(trajectory_layout(records,plan,10)['cases'][0]['prediction_positions']),256)
        r['generated_tokens'].insert(0,3);p['forced_tokens'].append(3)
        with self.assertRaises(ValueError):trajectory_layout(records,plan,10)

    def frame(self, directory):
        request=dict(id='case',tokens=[1,2],forced_tokens=[],tail_only=True)
        event=dict(event='trace',id='case',step=0,top1=7,vocab=VOCAB,committed_tokens=2,
                   tail_only=True,layer=23,tail_values=sum(TAIL_WIDTHS.values()))
        for kind,width in [('logits',VOCAB),('tail',sum(TAIL_WIDTHS.values()))]:
            path=directory/f'case-step0000.{kind}.bf16'
            raw=np.zeros(width,dtype='<u2')
            if kind=='logits':raw[7]=0x3f80
            raw.tofile(path);event[kind+'_path']=str(path)
        return request,event

    def test_tail_layout_paths_greedy_and_no_full_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);request,event=self.frame(path)
            self.assertEqual(check_frame(event,path,request,0)['tail'].shape,(10240,))
            for key,value in [('top1',8),('layers',24),('tail_path','elsewhere'),('step',1)]:
                bad={**event,key:value}
                with self.assertRaises(ValueError):check_frame(bad,path,request,0)

    def test_nonfinite_trailing_byte_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data';np.array([0x7f80],dtype='<u2').tofile(path)
            with self.assertRaises(ValueError):read_frame_file(path,1)
            path.write_bytes(b'\0\0\0')
            with self.assertRaises(ValueError):read_frame_file(path,1)
            path.write_bytes(b'\0\0');link=path.with_name('link');link.symlink_to(path)
            with self.assertRaises(ValueError):read_frame_file(link,1)

    def test_stream_terminal_must_match_greedy_not_forced_target(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'traces').mkdir();request,event=self.frame(out/'traces')
            terminal=dict(event='result',id='case',tokens=[7],forced_tokens=[],prompt_tokens=2,generated_tokens=1,
                          finish_reason='trace',complete=True,trace_only=True,cached_prompt_tokens=0)
            def worker(ending):
                events=iter([event,ending])
                return SimpleNamespace(out=out,process=SimpleNamespace(stdin=io.StringIO()),receive=lambda:next(events))
            result=collect(worker(terminal),request,[9],'case')
            self.assertEqual(result['result']['tokens'],[7])
            with self.assertRaises(ValueError):collect(worker({**terminal,'tokens':[8]}),request,[9],'case')
            with self.assertRaises(ValueError):collect(worker({**terminal,'cached_prompt_tokens':2}),request,[9],'case')


if __name__=='__main__':unittest.main()
