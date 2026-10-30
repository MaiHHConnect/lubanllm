import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from probe_q4_paired_tail import write_bf16
from validate_q4_paired_tail_proxy import errors, logit_summary, summarize, validate_mapping, LOGIT_COLUMNS


class TailProxyTests(unittest.TestCase):
    def test_full_forward_kl_and_errors_match_direct_probabilities(self):
        a, b = np.array([1., -2., 3., .5]), np.array([2., -.5, 1., 4.])
        p, q = np.exp(a)/np.exp(a).sum(), np.exp(b)/np.exp(b).sum()
        actual = logit_summary(a, b)
        self.assertAlmostEqual(actual[0], np.sum(p*np.log(p/q)), places=14)
        np.testing.assert_allclose(actual[1:4], [np.sqrt(np.mean((a-b)**2)),3.5,4])
        np.testing.assert_array_equal(actual[4:], [2,3,2,2,1,1])

    def test_shift_invariance_and_lowest_id_tie(self):
        a = np.array([7.,7.,2.,-1.])
        equal = logit_summary(a,a)
        np.testing.assert_array_equal(equal,[0,0,0,0,0,0,0,0,2,2])
        shifted = logit_summary(a,a+1000)
        self.assertEqual(shifted[0],0)
        summary = summarize(np.stack([equal,shifted]),LOGIT_COLUMNS)
        self.assertEqual(summary['argmax_mismatches'],0)
        self.assertEqual(summary['exact_rows'],1)
        self.assertEqual(summary['different_values_total'],4)

    def test_metric_inputs_reject_nonfinite_missing_or_wrong_dimensions(self):
        for a,b in [([1],[1]),([1,2],[1]),([1,np.nan],[1,2]),([1,2],[1,np.inf]),([[1,2]],[[1,2]])]:
            for fn in (errors,logit_summary):
                with self.assertRaises(ValueError):
                    fn(a,b)

    def test_complete_mapping_has_separate_split_offsets_and_shared_file_order(self):
        cases=[]; planned=[]
        for i,split in enumerate(('calibration','observation','calibration')):
            start=2 if i==2 else 0
            c=dict(id=f'case-{i}',split=split,tokens=[1,2],targets=[3,9],forced_tokens=[3],
                row_start=start,row_end=start+2,prediction_positions=[1,2],source_complete=True,source_finish_reason='stop')
            cases.append(c)
            planned.append({k:c[k] for k in ('id','split','tokens','forced_tokens')})
        mapping=dict(cases=cases,rows_per_split=dict(calibration=4,observation=2))
        plan=dict(cases=planned,eos_ids=[9])
        rows=validate_mapping(mapping,plan,mapping['rows_per_split'])
        self.assertEqual([r['file_row_start'] for r in rows],[0,2,4])
        self.assertEqual([r['file_row_end'] for r in rows],[2,4,6])
        for field,value in [('row_start',1),('prediction_positions',[0,1]),('targets',[3]),('source_complete',False)]:
            bad=copy.deepcopy(mapping);bad['cases'][0][field]=value
            with self.assertRaises(ValueError):
                validate_mapping(bad,plan,mapping['rows_per_split'])

    def test_existing_bf16_writer_appends_exact_stream_rows(self):
        values=torch.tensor([[1.,-2.,.5],[3.,4.,-5.]],dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stream.bf16'
            with path.open('wb',buffering=0) as stream:
                for row in values:
                    write_bf16(stream,row.reshape(1,-1))
            self.assertEqual(path.stat().st_size,12)
            np.testing.assert_array_equal(np.fromfile(path,dtype='<u2'),values.view(torch.uint16).numpy().ravel())


if __name__ == '__main__':
    unittest.main()
