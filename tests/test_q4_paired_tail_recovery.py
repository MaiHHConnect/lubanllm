import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_q4_paired_tail_recovery import split_rows,schedule,teacher_kl,apply_update


class RecoveryTests(unittest.TestCase):
    def test_noncontiguous_calibration_never_includes_observation(self):
        cases=[dict(split='calibration',file_row_start=0,file_row_end=3750),
               dict(split='observation',file_row_start=3750,file_row_end=4577),
               dict(split='calibration',file_row_start=4577,file_row_end=4724)]
        rows=split_rows(cases,'calibration');order=schedule(rows)
        self.assertEqual(order.shape,(450,9))
        self.assertEqual(set(order[:433].ravel()),set(rows))
        self.assertFalse(set(order.ravel())&set(range(3750,4577)))
        np.testing.assert_array_equal(order,schedule(rows))
        self.assertEqual(len(order.ravel()),4050)
        with self.assertRaises(ValueError):schedule([1,1])

    def test_full_distribution_kl_gradient_and_mean(self):
        logits=torch.tensor([[.2,-.3,1.]],requires_grad=True)
        teacher=torch.tensor([[.7,.1,-.4]])
        loss=teacher_kl(logits,teacher)
        p=teacher.softmax(-1);q=logits.softmax(-1)
        expected=(p*(p.log()-q.log())).sum()
        torch.testing.assert_close(loss,expected)
        (loss/9).backward()
        torch.testing.assert_close(logits.grad,(q.detach()-p)/9)

    def test_master_accumulates_sub_bf16_changes_and_rejects_nan(self):
        p=torch.ones(2,3,dtype=torch.bfloat16,requires_grad=True)
        master=torch.nn.Parameter(p.detach().float())
        opt=torch.optim.Adam([master],lr=1e-4,eps=1e-6)
        for _ in range(30):apply_update(p,master,opt,torch.full_like(master,.1))
        self.assertTrue(torch.equal(p,master.detach().to(torch.bfloat16)))
        self.assertTrue(bool((p<1).all()))
        self.assertEqual(int(opt.state[master]['step']),30)
        with self.assertRaises(ValueError):apply_update(p,master,opt,torch.full_like(master,float('nan')))


if __name__=='__main__':unittest.main()
