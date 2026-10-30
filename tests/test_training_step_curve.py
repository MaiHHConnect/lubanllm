"""704 CE ledger/actual tinyCPU Adam, saved BF16 tensors and admission controls."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_training_step_curve as runner
from test_training_method_pilot import rows,observed
from prepare_tensor_method_schedule import build_schedule


class TrainingStepCurveTests(unittest.TestCase):
    def test_default_zero_model_explicit_method_and_fixed_checks(self):
        raw=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_training_step_curve.py')],capture_output=True,text=True,check=True)
        value=json.loads(raw.stdout)
        self.assertFalse(value['model_started']);self.assertFalse(value['torch_imported'])
        self.assertEqual(value['checks_ce_events'],list(range(50,701,50))+[704])
        for arguments in (['--method','cache'],['--run','missing'],['--prepare','missing','--output','missing']):
            result=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_training_step_curve.py'),*arguments],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
        for method,batch in (('native',2),('native',4),('future',1),('cache',True)):
            with self.assertRaises(ValueError):runner.prepare('missing','missing',method,batch)
        with self.assertRaises(ValueError):runner.prepare('missing','missing','cache',2,51)

    def fixture(self,batch,*,reject=False,select=None,mutation=None,final_save_clock=None):
        import torch
        import cached_method_steps
        from verify_cached_training import tensor_sha
        groups={'new_training':rows('new',57),'replay_training':rows('replay',57)}
        model=torch.nn.Linear(3,2,bias=False).to(torch.bfloat16);parameter=model.weight
        original=parameter.detach().clone();runtime=SimpleNamespace(model=model)
        baseline={'new':[observed(r,i<13) for i,r in enumerate(groups['new_training'])],
                  'replay':[observed(r,i<55) for i,r in enumerate(groups['replay_training'])],'replay_loss':1.}
        old_rows=groups['replay_training'][:55]
        schedule=build_schedule([r['id'] for r in groups['new_training']],[r['id'] for r in groups['replay_training']],704,batch)
        plan={'method':'cache','tensor_batch':batch,'source_sha256':{},'confirm_ce_event':select,
              'schedule_sha256':runner.tensor.content_sha(schedule)}
        flags=[(p,p.requires_grad,p.grad) for p in model.parameters()];modes=[(m,m.training) for m in model.modules()]
        rng=torch.get_rng_state().clone();actual_steps=[]
        def step(rt,samples,replays,master,opt,reference,basis,**kw):
            master.grad=torch.full_like(master,.1);opt.step();actual_steps.append(int(opt.state[master]['step']))
            if mutation:mutation()
            return {'runtime_restored':True,'optimizer_steps':1,'ce_role':kw['ce_role'],
                'ce_ids':[r['id'] for r in samples],'replay_kl_ids':[r['id'] for r in replays]}
        calls=[]
        def probes(rt,samples,deadline,expected):
            self.assertEqual(tensor_sha(parameter),expected);calls.append([r['id'] for r in samples])
            return [observed(r,not(reject and r['id']=='new-0')) for r in samples]
        clock=final_save_clock
        real_write=Path.write_text
        def write(path,text,*args,**kwargs):
            value=json.loads(text)
            if clock is not None and value.get('completed') is True:clock[0]=3601.
            return real_write(path,text,*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('tiny-lastdown',parameter)),\
                patch.object(cached_method_steps,'train_method_step',side_effect=step),patch.object(runner.prior,'probes',side_effect=probes),\
                patch('luban.evolution._loss',return_value=1.),patch.object(Path,'write_text',write):
            output=Path(folder).resolve();result=runner.curve(runtime,groups,baseline,old_rows,original,torch.empty(3,0),{},plan,output,
                                                   3600. if clock is not None else runner.time.perf_counter()+60)
            checkpoints=list(output.glob('*.pt'))
            self.assertEqual(len(checkpoints),len(result['checks']))
            for point in result['checks']:
                saved=point['saved_weights'];weight=torch.load(saved['path'],weights_only=True)
                self.assertEqual(weight.dtype,torch.bfloat16);self.assertEqual(tensor_sha(weight),saved['tensor_sha256'])
                self.assertEqual(runner.sha(saved['path']),saved['file_sha256']);self.assertEqual(saved['tensor_bytes'],12)
                self.assertGreater(saved['file_bytes'],saved['tensor_bytes'])
        self.assertTrue(torch.equal(parameter,original));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.requires_grad==f and p.grad is g for p,f,g in flags));self.assertTrue(all(m.training==v for m,v in modes))
        return result,actual_steps,calls

    def test_actual_cpu_adam704_366_198_all15_saved_points_and_exposures(self):
        for batch,updates in ((1,704),(2,366),(4,198)):
            with self.subTest(batch=batch):
                result,steps,calls=self.fixture(batch)
                self.assertTrue(result['completed']);self.assertTrue(result['quality_passed'])
                self.assertEqual(result['completed_ce_events'],704);self.assertEqual(steps,list(range(1,updates+1)))
                self.assertEqual(result['optimizer_updates'],updates)
                self.assertEqual([p['ce_events'] for p in result['checks']],list(runner.CHECKS))
                for index,point in enumerate(result['checks']):
                    previous=result['checks'][index-1] if index else None
                    self.assertEqual(point['interval_ce_events'],4 if point['ce_events']==704 else 50)
                    self.assertEqual(point['interval_optimizer_updates'],point['optimizer_updates']-(previous['optimizer_updates'] if previous else 0))
                    self.assertEqual(point['cumulative_exposure_totals'],{'new_ce':point['ce_events']//2,'replay_ce':point['ce_events']//2,'replay_kl':point['ce_events']})
                    self.assertEqual({k:sum(v.values()) for k,v in point['cumulative_per_id_exposures'].items()},point['cumulative_exposure_totals'])
                self.assertEqual({k:sum(v.values()) for k,v in result['exposures'].items()},
                                 {'new_ce':352,'replay_ce':352,'replay_kl':704})
                self.assertTrue(all(p['newly_confirmed_ids'] is None and p['mastery_confirmed'] is False for p in result['checks'][:-1]))
                self.assertEqual(len(result['checks'][-1]['newly_confirmed_ids']),44)
                self.assertEqual(len(calls),31)

    def test_registered_middlepoint_second_generation_uses_saved_tensor(self):
        result,steps,calls=self.fixture(4,select=350)
        selected=result['selected_saved_point']
        self.assertEqual(selected['ce_event'],350);self.assertTrue(selected['mastery_confirmed'])
        self.assertTrue(selected['safety_passed']);self.assertEqual(len(selected['newly_confirmed_ids']),44)
        self.assertEqual(len(calls),33);self.assertEqual(len(steps),198)

    def test_safety_failure_at50_and_deadline_after_optimizer_stop_restore(self):
        result,steps,calls=self.fixture(4,reject=True)
        self.assertFalse(result['completed']);self.assertFalse(result['quality_passed'])
        self.assertEqual(result['completed_ce_events'],50);self.assertEqual(len(steps),14)
        self.assertEqual(result['checks'][0]['forgotten_ids'],['new-0'])
        clock=[0.]
        def advance():clock[0]=3601.
        with patch.object(runner.time,'perf_counter',side_effect=lambda:clock[0]):
            failed,steps,_=self.fixture(2,mutation=advance,final_save_clock=clock)
        self.assertFalse(failed['completed']);self.assertEqual(len(steps),1);self.assertEqual(failed['completed_ce_events'],2)
        self.assertTrue(failed['original_tensor_restored'])

    def test_final_curve_raw_save_over_deadline_rejects_after_complete704(self):
        clock=[0.]
        with patch.object(runner.time,'perf_counter',side_effect=lambda:clock[0]):
            result,steps,_=self.fixture(4,final_save_clock=clock)
        self.assertFalse(result['completed']);self.assertFalse(result['quality_passed'])
        self.assertEqual(result['completed_ce_events'],704);self.assertEqual(len(steps),198)
        self.assertEqual(result['error']['type'],'TimeoutError');self.assertTrue(result['original_tensor_restored'])

    def test_saved_checkpoint_tamper_and_nonfinite_rejected(self):
        import torch
        model=torch.nn.Linear(3,2,bias=False).to(torch.bfloat16);parameter=model.weight;before=parameter.detach().clone()
        runtime=SimpleNamespace(model=model)
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('tiny-lastdown',parameter)):
            path=Path(folder).resolve()/'saved.pt';saved=runner.save_weights(parameter,path)
            path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError,'bytes changed'):
                runner.confirm_saved_point(runtime,{}, {},[],{'saved_weights':saved},runner.time.perf_counter()+60)
            with torch.no_grad():parameter.fill_(float('nan'))
            with self.assertRaises(ValueError):runner.save_weights(parameter,Path(folder).resolve()/'nan.pt')
            with torch.no_grad():parameter.copy_(before)

    def test_same_protocol_once_preflight_and_second_output_reject_without_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'protocol.json';output=root/'output'
            value={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO','output':str(output)}
            runner.write_new(path,value)
            with patch.object(runner,'validate_bound',side_effect=ValueError('wrongsource')):
                with self.assertRaises(ValueError):runner.execute(path,runner.sha(path))
            self.assertFalse(output.exists());self.assertFalse((root/'execution-started.json').exists())
            runner.write_new(root/'execution-started.json',{'CPU_consumed':True})
            with patch.object(runner,'validate_bound'):
                with self.assertRaises(FileExistsError):runner.execute(path,runner.sha(path))
            self.assertFalse(output.exists())

    def test_curve_cancel_after_tiny_adam_restores_original_native_weight(self):
        import torch
        import cached_method_steps
        model=torch.nn.Linear(3,2,bias=False).to(torch.bfloat16);parameter=model.weight
        before=parameter.detach().clone();runtime=SimpleNamespace(model=model)
        groups={'new_training':rows('new',57),'replay_training':rows('replay',57)}
        schedule=build_schedule([r['id'] for r in groups['new_training']],[r['id'] for r in groups['replay_training']],704,1)
        plan={'method':'cache','tensor_batch':1,'source_sha256':{},'confirm_ce_event':None,
              'schedule_sha256':runner.tensor.content_sha(schedule)}
        def cancel(rt,samples,replays,master,opt,*args,**kwargs):
            master.grad=torch.ones_like(master);opt.step()
            with torch.no_grad():parameter.add_(1)
            raise KeyboardInterrupt('CPU fixture cancellation after one actual Adam')
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('tiny',parameter)),\
                patch.object(cached_method_steps,'train_method_step',side_effect=cancel):
            with self.assertRaises(KeyboardInterrupt):
                runner.curve(runtime,groups,{},[],before,torch.empty(3,0),{},plan,Path(folder).resolve(),runner.time.perf_counter()+60)
            saved=json.loads((Path(folder)/'curve.json').read_text())
            self.assertEqual(saved['error']['type'],'KeyboardInterrupt');self.assertTrue(saved['original_tensor_restored'])
            self.assertFalse(saved['completed']);self.assertTrue(torch.equal(parameter,before))

    def test_load_failure_releases_actual_cpu_os_lock_and_once_marker(self):
        import torch
        from luban.process_lock import ModelProcessLock
        runtime=SimpleNamespace(model=None,device='cpu',dtype=torch.bfloat16,_model_process_lock=None)
        def fail(*,paging):raise ValueError('CPU fixture no model constructed')
        def release():
            if runtime._model_process_lock is not None:runtime._model_process_lock.close()
            runtime._model_process_lock=None
        runtime.load=fail;runtime._release_model=release
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'protocol.json';output=root/'output';reference=root/'reference.json'
            runner.write_new(reference,{'CPU_fixture_only':True})
            plan={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO',
                  'output':str(output),'model_path':str(root/'no-model'),'device':'cpu','dtype':'torch.bfloat16',
                  'source_sha256':{},'production_root':str(root),'production_snapshot':{'CPU':True},
                  'official_model_binding':{'CPU':True},'b1_actual_reference':{'path':str(reference),'sha256':runner.sha(reference)}}
            runner.write_new(path,plan)
            with patch.object(runner,'validate_bound'),patch('luban.runtime.Runtime',return_value=runtime),\
                    patch('verify_router_coverage_study.model_binding',return_value={'CPU':True}),\
                    patch.object(runner,'production_snapshot',return_value={'CPU':True}):
                with self.assertRaises(RuntimeError):runner.execute(path,runner.sha(path))
            raw=json.loads((output/'result.json').read_text())
            self.assertFalse(raw['model_started']);self.assertFalse(raw['passed'])
            self.assertTrue(raw['model_released']);self.assertTrue(raw['actual_os_lock_reacquired_closed'])
            self.assertTrue(raw['preserved']);self.assertTrue((root/'execution-started.json').exists())
            self.assertGreaterEqual(raw['resources']['sampled_peak_rss_bytes'],0)
            self.assertGreaterEqual(raw['resources']['elapsed_seconds'],0)
            self.assertGreaterEqual(len(raw['resources']['snapshots']),2)
            lock=ModelProcessLock(runner.IDENTITY,'cpu');lock.close()


if __name__=='__main__':unittest.main()
