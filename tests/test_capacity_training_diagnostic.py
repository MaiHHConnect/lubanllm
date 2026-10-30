"""Admission fixtures and tiny official CPU no-grad math; no actual N50 data."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import diagnose_capacity_training as diagnostic


class AdmissionTests(unittest.TestCase):
    def test_default_waiting_stdlib_and_missing_receipt_never_reads_N(self):
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-S',str(ROOT/'scripts/diagnose_capacity_training.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        value=json.loads(result.stdout);self.assertFalse(value['torch_imported']);self.assertEqual(value['optimizer_steps'],0)
        with patch.object(diagnostic,'closed',side_effect=AssertionError('must not read running data')):
            plan=diagnostic.prepare('/private/tmp/not-an-execution')
            self.assertIn('waiting',plan['state'])
            with self.assertRaises(ValueError):diagnostic.bind(plan)

    def test_actual_scope_and_exit_types_fail_closed(self):
        for bad in ({'scope':'cpu_fixture'}, {'scope':'actual_closed_N50_last_complete_checkpoint','process_exit_code':True,'pid_dead':True},
                    {'scope':'actual_closed_N50_last_complete_checkpoint','process_exit_code':1,'pid_dead':False}):
            with patch.object(diagnostic,'read',return_value=bad),self.assertRaises(ValueError):diagnostic.closed('/private/tmp/fixture-only')

    def test_prepare_allows_fresh_experiment_child_not_protected_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            product=Path(folder).resolve();model=product/'models'/'official';old=product/'experiments'/'closed'/'execution'
            fixture={'plan':{'production_root':str(product),'model_path':str(model),'output':str(old),'source_sha256':{}},
                     'summary':{'CPU_fixture_only':True}}
            with patch.object(diagnostic,'closed',return_value=fixture),patch.object(diagnostic,'sha',return_value='a'*64):
                output=product/'experiments'/'fresh'/'execution'
                plan=diagnostic.prepare(output,product/'fixture-receipt.json')
                self.assertEqual(plan['output'],str(output));self.assertFalse(output.exists())
                for bad in (diagnostic.ROOT,product,product/'experiments',product/'app'/'execution',
                            product/'modules'/'execution',model/'execution',old,old/'new',product/'outside'):
                    with self.subTest(path=bad),self.assertRaises(ValueError):diagnostic.prepare(bad,product/'fixture-receipt.json')
                output.mkdir(parents=True)
                with self.assertRaises(ValueError):diagnostic.prepare(output,product/'fixture-receipt.json')

    def test_partial_source_allows_last_complete_only_not_unwritten_current(self):
        artifact={'candidate_tensor_sha256':'a'*64,'file_sha256':'b'*64}
        post={'execution_passed':False,'quality_passed':False,'completed_points':[
            {'ce_events':100,'payload_verified':True,'checkpoint_artifact':artifact},
            {'ce_events':300,'payload_verified':True,'checkpoint_artifact':artifact}]}
        training={'completed':False,'checks':[{'ce_events':100,'checkpoint_artifact':artifact,'candidate_tensor_sha256':'a'*64},
            {'ce_events':300,'checkpoint_artifact':artifact,'candidate_tensor_sha256':'a'*64},
            {'ce_events':400,'new_first':[{'fixture':'partial'}]}]}
        original=copy.deepcopy((post,training))
        self.assertEqual(diagnostic.last_complete_point(post,training,300)[1],artifact)
        self.assertEqual((post,training),original)
        for bad in (100,400,600,True,None):
            with self.assertRaises(ValueError):diagnostic.last_complete_point(post,training,bad)
        for bad in ('missing-payload','candidate-drift','duplicate-point'):
            a,b=copy.deepcopy((post,training))
            if bad=='missing-payload':a['completed_points'][-1]['payload_verified']=False
            elif bad=='candidate-drift':b['checks'][1]['candidate_tensor_sha256']='c'*64
            else:a['completed_points'].append(copy.deepcopy(a['completed_points'][-1]))
            with self.assertRaises(ValueError):diagnostic.last_complete_point(a,b,300)

    def test_unsealed_rejected_before_execution_and_forbidden_update_calls_absent(self):
        with (patch.object(diagnostic,'sha',return_value='a'*64),patch.object(diagnostic,'read',return_value={'sealed':False}),
              patch.object(diagnostic,'validate_bound',side_effect=AssertionError('must reject before binding'))):
            with self.assertRaises(ValueError):diagnostic.execute('/private/tmp/fixture-only','a'*64)
        tree=ast.parse((ROOT/'scripts/diagnose_capacity_training.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse({'backward','generate','step','zero_grad'} & set(calls))


class TinyMathTests(unittest.TestCase):
    def test_shift_mask_goldEOS_rank_ties_and_gap_no_grad(self):
        import torch
        from torch.nn import functional as F
        labels=torch.tensor([[-100,-100,3,2]])
        logits=torch.zeros((1,4,5));logits[0,1,3]=2;logits[0,1,4]=2;logits[0,2,2]=4
        with torch.no_grad():value=diagnostic.token_statistics(logits,labels)
        self.assertEqual(value['gold_eos_token_ids'],[3,2]);self.assertEqual(value['gold_rank'],1)
        self.assertEqual(value['maximum_non_gold_token'],4);self.assertEqual(value['gold_minus_maximum_non_gold_logit'],0)
        expected=F.cross_entropy(torch.stack([logits[0,1],logits[0,2]]),torch.tensor([3,2]),reduction='none')
        self.assertEqual(value['per_token_ce_fp32'],expected.tolist());self.assertAlmostEqual(value['mean_ce'],float(expected.mean()))
        logits[0,1,4]=3
        with torch.no_grad():changed=diagnostic.token_statistics(logits,labels)
        self.assertEqual(changed['gold_rank'],2);self.assertEqual(changed['gold_minus_maximum_non_gold_logit'],-1)
        logits[0,1,0]=torch.nan
        with torch.no_grad(),self.assertRaises(RuntimeError):diagnostic.token_statistics(logits,labels)

    def test_tiny_official_native_cached_original_candidate_exact_no_grad(self):
        import torch
        from test_field_training_diagnostic import tiny_runtime
        from verify_cached_training import capture_tail
        for dtype in (torch.float32,torch.bfloat16):
            with tempfile.TemporaryDirectory() as folder,torch.random.fork_rng(devices=[]):
                torch.manual_seed(42);runtime,row,_,_=tiny_runtime(folder,dtype)
                runtime.identity='offline-tiny-cpu-not-N50';runtime.job=None;runtime.device='cpu';row['purpose']='training'
                runtime.model.eval().requires_grad_(False);weight=runtime.branch.base.down_proj.weight;weight.requires_grad_(True)
                before=weight.detach().clone();rng=torch.get_rng_state().clone();grads=[p.grad for p in runtime.model.parameters()]
                cache=capture_tail(runtime,row)
                for state in ('original','synthetic-candidate'):
                    if state=='synthetic-candidate':
                        with torch.no_grad():weight.add_(torch.full_like(weight,.001))
                    value=diagnostic.diagnose_row(runtime,row,cache)
                    self.assertTrue(value['native_cached_logits']['passed']);self.assertTrue(value['native_cached_mean_ce']['passed'])
                    self.assertTrue(value['native_cached_logits']['exact']);self.assertEqual(len(value['per_token_ce_fp32']),4)
                    self.assertEqual([p.grad for p in runtime.model.parameters()],grads)
                with torch.no_grad():weight.copy_(before)
                self.assertTrue(torch.equal(before,weight));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertFalse(runtime.branch.base.down_proj._forward_pre_hooks)
                self.assertFalse(runtime.decoder.layers[-1].post_attention_layernorm._forward_pre_hooks)
                bad=copy.deepcopy(row);bad['answer']='different-gold'
                runtime.tokenizer.encode=lambda *a,**k:[23,29]
                with self.assertRaises(ValueError):diagnostic.diagnose_row(runtime,bad,cache)


class MemoryBoundaryTests(unittest.TestCase):
    def test_cpu_release_does_not_call_any_mps_operation(self):
        import torch
        with patch.object(torch.mps,'synchronize',side_effect=AssertionError('CPU must not touch MPS')):
            result=diagnostic.release_unused_mps(SimpleNamespace(device='cpu'),'fixture')
        self.assertFalse(result['enabled']);self.assertEqual(result['seconds'],0)

    def test_mps_release_records_two_samples_without_overriding_zero_guard(self):
        import torch
        import psutil
        events=[]
        def context():
            events.append('context');return {'current_limit':0,'memory_budget_bytes':0}
        with (patch.object(torch.mps,'synchronize',side_effect=lambda:events.append('sync')),
              patch.object(torch.mps,'empty_cache',side_effect=lambda:events.append('empty')),
              patch.object(torch.mps,'current_allocated_memory',side_effect=[100,100]),
              patch.object(torch.mps,'driver_allocated_memory',side_effect=[900,400]),
              patch.object(psutil,'virtual_memory',side_effect=[SimpleNamespace(available=200),SimpleNamespace(available=700)])):
            result=diagnostic.release_unused_mps(SimpleNamespace(device='mps',context_status=context),'fixture')
        self.assertEqual(events,['sync','context','empty','sync','context'])
        self.assertEqual(result['before']['available_bytes'],200);self.assertEqual(result['after']['available_bytes'],700)
        self.assertEqual(result['after']['mps_current_allocated_bytes'],100)
        self.assertEqual(result['after']['mps_driver_allocated_bytes'],400)
        self.assertEqual(result['after']['context']['current_limit'],0);self.assertGreaterEqual(result['seconds'],0)

    def test_actual_tiny_outputs_dropped_after_stats_and_before_next_admission(self):
        import torch
        from test_field_training_diagnostic import tiny_runtime
        import verify_cached_training as cached
        with tempfile.TemporaryDirectory() as folder,torch.random.fork_rng(devices=[]):
            runtime,row,_,_=tiny_runtime(folder,torch.float32);runtime.identity='memory-boundary-CPU-fixture';runtime.job=None;runtime.device='cpu'
            runtime.model.eval().requires_grad_(False);runtime.branch.base.down_proj.weight.requires_grad_(True)
            row['purpose']='training';cache=cached.capture_tail(runtime,row);events=[];refs=[];memory={}
            original_tokens=runtime._tokens;original_cached=cached.cached_logits;original_statistics=diagnostic.token_statistics
            def statistics(logits,labels):
                refs.append(weakref.ref(logits));return original_statistics(logits,labels)
            def tokens(*args,**kwargs):events.append('tokens');return original_tokens(*args,**kwargs)
            def tail(*args,**kwargs):
                result=original_cached(*args,**kwargs);refs.append(weakref.ref(result[0]));return result
            def release(runtime,phase):
                events.append(phase)
                if phase.startswith('after_statistics'):
                    self.assertEqual(len(refs),2);self.assertTrue(all(ref() is None for ref in refs))
                return {'enabled':False,'phase':phase,'seconds':0.0,'CPU_fixture_only':True}
            with patch.object(diagnostic,'release_unused_mps',side_effect=release),patch.object(runtime,'_tokens',side_effect=tokens),patch.object(cached,'cached_logits',side_effect=tail),patch.object(diagnostic,'token_statistics',new=statistics):
                result=diagnostic.diagnose_row(runtime,row,cache,memory=memory)
            self.assertTrue(result['native_cached_logits']['exact'])
            self.assertEqual(events,['after_CPU_capture_before_tokens','tokens','after_statistics_live_outputs_dropped'])
            self.assertGreater(memory['input_tokens'],0);self.assertGreaterEqual(memory['diagnose_seconds'],0)
            self.assertTrue(all(value.device.type=='cpu' for key,value in cache.items() if key in ('x','residual','ids','labels')))
            # A genuine admission refusal is preserved, not retried/bypassed.
            failed={};events=[]
            with patch.object(diagnostic,'release_unused_mps',side_effect=release),patch.object(runtime,'_tokens',side_effect=ValueError('context0')):
                with self.assertRaisesRegex(ValueError,'context0'):diagnostic.diagnose_row(runtime,row,cache,memory=failed)
            self.assertIn('before_admission',failed);self.assertNotIn('after_statistics',failed)
            self.assertEqual(events,['after_CPU_capture_before_tokens'])


if __name__=='__main__':unittest.main()
