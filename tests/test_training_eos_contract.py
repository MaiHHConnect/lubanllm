"""Distinct TRAIN/generation EOS guards and actual official tiny training graphs."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import cached_method_steps as method
import test_cached_batch_training as batch_fixture
from test_cached_method_steps import uninterrupted_step

REPORTS=[]


def different_eos_fixture(folder,dtype):
    runtime,new,replay,tensors=batch_fixture.fixture(folder,dtype)
    generation=runtime.model.generation_config.eos_token_id
    # Same official tiny graph; distinct valid vocabulary tokens, no248k head.
    runtime.tokenizer.eos_token_id=63
    for ids,labels in tensors.values():ids[0,-1]=labels[0,-1]=63
    assert generation==2 and runtime.model.generation_config.eos_token_id==2
    runtime.model.eval().requires_grad_(False)
    parameter=runtime.branch.base.down_proj.weight;parameter.requires_grad_(True)
    reference=parameter.detach().float().clone()
    caches={row['id']:method.cached.capture_tail(runtime,row) for row in new+replay}
    basis,_=method.cached._capture_replay_subspace(runtime,parameter,replay)
    return runtime,new,replay,reference,basis,caches


class TrainingEosContractTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.rng=torch.random.fork_rng(devices=[]);self.rng.__enter__()
        self.addCleanup(self.rng.__exit__,None,None,None);torch.manual_seed(42)

    def test_real_official_id_difference_train_accepts_only_tokenizer_terminal(self):
        runtime=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=248046),
            model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044)))
        eos=method._eos(runtime);self.assertEqual(eos,{248046})
        ids=torch.tensor([[3,5,7,11,248046]]);labels=ids.clone();labels[:,:2]=-100
        method._labels(ids,labels,eos)
        for change in ('generation-terminal','missing','internal-train-terminal','mask-hole','wrong-gold'):
            bad_ids,bad_labels=ids.clone(),labels.clone()
            if change=='generation-terminal':bad_ids[0,-1]=bad_labels[0,-1]=248044
            if change=='missing':bad_ids=bad_ids[:,:-1];bad_labels=bad_labels[:,:-1]
            if change=='internal-train-terminal':bad_ids[0,2]=bad_labels[0,2]=248046
            if change=='mask-hole':bad_labels[0,3]=-100
            if change=='wrong-gold':bad_labels[0,2]=13
            with self.subTest(change=change),self.assertRaises(ValueError):method._labels(bad_ids,bad_labels,eos)
        for invalid in (None,True,False,[248046],-1,'248046'):
            runtime.tokenizer.eos_token_id=invalid
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):method._eos(runtime)
        self.assertEqual(runtime.model.generation_config.eos_token_id,248044)
        REPORTS.append({'kind':'actual_distinct_token_ids_guard_only','generation_eos':248044,'training_eos':248046,
                        'accepted_training_terminal_only':True,'malformed_cases_rejected':5,'no_large_model':True})

    def test_distinct_eos_b1_native_cache_eight_continuous_adam_steps_equal_old(self):
        for dtype in (torch.float32,torch.bfloat16):
            runtime,new,replay,reference,basis,caches=different_eos_fixture(self.folder.name,dtype)
            parameter=runtime.branch.base.down_proj.weight;original=parameter.detach().clone()
            masters={name:torch.nn.Parameter(reference.clone()) for name in ('old','native','cache')}
            optimizers={name:torch.optim.Adam([master],lr=1e-5,eps=1e-6) for name,master in masters.items()}
            traces=[]
            for index in range(8):
                role='new' if index%2==0 else 'replay';ce_rows=new if role=='new' else replay
                ce=ce_rows[(index//2)%len(ce_rows)];kl=replay[(index//2)%len(replay)]
                losses=uninterrupted_step(runtime,ce,kl,masters['old'],optimizers['old'],reference,basis)
                with torch.no_grad():parameter.copy_(original)
                for name in ('native','cache'):
                    actual=method.train_method_step(runtime,[ce],[kl],masters[name],optimizers[name],reference,basis,
                        method=name,ce_role=role,caches=caches)
                    self.assertEqual((actual['ce'],actual['kl']),losses)
                    self.assertTrue(torch.equal(masters[name],masters['old']))
                    self.assertTrue(all(torch.equal(optimizers[name].state[masters[name]][key],
                        optimizers['old'].state[masters['old']][key]) for key in ('step','exp_avg','exp_avg_sq')))
                    self.assertTrue(actual['runtime_restored']);self.assertTrue(torch.equal(parameter,original))
                    traces.append(actual)
            self.assertEqual(runtime.model.generation_config.eos_token_id,2)
            self.assertEqual(runtime.tokenizer.eos_token_id,63)
            REPORTS.append({'kind':'distinct_eos_actual_official_tiny_8steps','dtype':str(dtype),
                'generation_eos':2,'training_eos':63,'native_cache_old_master_moments_exact':True,'traces':copy.deepcopy(traces)})

    def test_distinct_eos_cache_b2_b4_roles_tails_finite_and_runtime_restore(self):
        for dtype in (torch.float32,torch.bfloat16):
            for batch_size in (2,4):
                runtime,new,replay,reference,basis,caches=different_eos_fixture(self.folder.name,dtype)
                parameter=runtime.branch.base.down_proj.weight;original=parameter.detach().clone()
                master=torch.nn.Parameter(reference.clone());optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
                traces=[];rng=torch.get_rng_state().clone()
                for offset in range(0,len(new),batch_size):
                    ce_new=new[offset:offset+batch_size]
                    kl=[replay[(offset+i)%len(replay)] for i in range(len(ce_new))]
                    for role,ce in (('new',ce_new),('replay',kl)):
                        actual=method.train_method_step(runtime,ce,kl,master,optimizer,reference,basis,
                            method='cache',ce_role=role,caches=caches)
                        self.assertTrue(actual['runtime_restored']);self.assertTrue(torch.equal(parameter,original))
                        traces.append(actual)
                self.assertEqual(int(optimizer.state[master]['step']),6 if batch_size==2 else 4)
                self.assertEqual([sum(t[k] for t in traces) for k in ('new_ce_exposures','replay_ce_exposures','replay_kl_exposures')],[5,5,10])
                self.assertTrue(torch.isfinite(master).all());self.assertTrue(torch.equal(torch.get_rng_state(),rng))
                self.assertEqual(runtime.model.generation_config.eos_token_id,2)
                self.assertEqual(runtime.tokenizer.eos_token_id,63)
                REPORTS.append({'kind':'distinct_eos_cache_methods','dtype':str(dtype),'tensor_batch':batch_size,
                    'generation_eos':2,'training_eos':63,'sequential_adam_equivalence_claim':False,'traces':copy.deepcopy(traces)})


if __name__=='__main__':unittest.main()
