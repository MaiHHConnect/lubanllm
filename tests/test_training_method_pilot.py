"""Pilot admission/count/deadline/probe guards; no 2B allocation or teacher."""
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
import run_training_method_pilot as runner


def rows(prefix,count):
    return [{'id':f'{prefix}-{i}','answer':'OK','prompt':f'{prefix} {i}','purpose':'training',
        'provenance':json.dumps({'evolution_role':prefix,'generation_check':{'type':'exact','expected':'OK'},
            'messages':[{'role':'user','content':f'{prefix} {i}'}]})} for i in range(count)]


def observed(row,passed=True):
    return {'id':row['id'],'answer':'OK' if passed else 'wrong','passed':passed,'complete':True,
        'branch_enabled':False,'reference_used':False}


class TrainingMethodPilotTests(unittest.TestCase):
    def test_default_stdlib_no_torch_or_model(self):
        raw=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_training_method_pilot.py')],
            capture_output=True,text=True,check=True)
        result=json.loads(raw.stdout)
        self.assertFalse(result['model_started']);self.assertFalse(result['torch_imported'])
        self.assertEqual(result['checks'],[50,100,150])

    def test_exact_old_schedule_and_three_exposure_ledgers(self):
        groups={'new_training':rows('new',57),'replay_training':rows('replay',57)}
        plan=runner.schedule(groups)
        self.assertEqual(len(plan),150)
        self.assertEqual(sum(role=='new' for role,_,_ in plan),75)
        self.assertEqual(sum(role=='replay' for role,_,_ in plan),75)
        self.assertEqual([r['id'] for _,_,r in plan[:4]],['replay-0','replay-0','replay-1','replay-1'])
        self.assertEqual(plan[114][1]['id'],'new-0')
        with self.assertRaises(ValueError):runner.schedule({**groups,'replay_training':groups['replay_training'][:-1]})

    def test_finite_only_strict_closedbook_boolean_and_raw_gold(self):
        material=rows('new',2);proof=[observed(r) for r in material]
        self.assertEqual(runner.correct(material,proof,finite=True),{'new-0','new-1'})
        for changes in ({'passed':1},{'complete':1},{'branch_enabled':True},{'reference_used':True},
                        {'answer':'wrong'},{'id':'another'}):
            bad=copy.deepcopy(proof);bad[0].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):runner.correct(material,bad,finite=True)
        source=json.loads(material[0]['provenance']);source['generation_check']={'type':'contains','expected':'OK'}
        material[0]['provenance']=json.dumps(source)
        self.assertEqual(runner.correct(material,proof,finite=True),{'new-1'})
        self.assertEqual(runner.correct(material,proof),{'new-0','new-1'})

    def test_check50_provisional_and150_frozen_twice_growth_no_forgetting(self):
        news=rows('new',3);old=rows('replay',2);groups={'new_training':news,'replay_training':old}
        baseline={'new':[observed(r,i==0) for i,r in enumerate(news)]}
        import torch
        from verify_cached_training import tensor_sha
        parameter=torch.nn.Parameter(torch.ones(2,3));candidate=tensor_sha(parameter)
        self.addCleanup(patch.stopall)
        patch('luban.evolution._target',return_value=('W',parameter)).start()
        with patch('luban.evolution._loss',return_value=1.),patch.object(runner,'probes',side_effect=lambda rt,rs,d,*rest:[observed(r) for r in rs]):
            first=runner.checkpoint(None,groups,baseline,old,1.,50,candidate,None)
            final=runner.checkpoint(None,groups,baseline,old,1.,150,candidate,None)
        self.assertFalse(first['mastery_confirmed']);self.assertEqual(first['newly_confirmed_ids'],[])
        self.assertEqual(first['provisional_growth_ids'],['new-1','new-2'])
        self.assertTrue(final['mastery_confirmed']);self.assertEqual(final['newly_confirmed_ids'],['new-1','new-2'])
        self.assertTrue(final['safety_passed'])
        def bad(rt,rs,d,*rest):return [observed(r,r['id']!='new-0') for r in rs]
        with patch('luban.evolution._loss',return_value=1.),patch.object(runner,'probes',side_effect=bad):
            failed=runner.checkpoint(None,groups,baseline,old,1.,50,candidate,None)
        self.assertFalse(failed['safety_passed']);self.assertEqual(failed['forgotten_ids'],['new-0'])
        with patch('luban.evolution._loss',return_value=1.050001),patch.object(runner,'probes',side_effect=lambda rt,rs,d,*rest:[observed(r) for r in rs]):
            self.assertFalse(runner.checkpoint(None,groups,baseline,old,1.,50,candidate,None)['safety_passed'])

    def test_probe_deadline_no_generation_and_malformed_execution_no_output(self):
        with patch('luban.evolution._generation') as generate,self.assertRaises(TimeoutError):
            runner.probes(None,rows('new',1),0)
        generate.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'protocol.json';out=Path(folder).resolve()/'result'
            runner.write_new(path,{'protocol':runner.PROTOCOL,'sealed':False,'output':str(out)})
            for digest in ('0'*64,runner.sha(path)):
                with self.assertRaises(ValueError):runner.execute(path,digest)
            self.assertFalse(out.exists())
            self.assertFalse((path.parent/'execution-started.json').exists())

    def test_protocol_directory_consumed_once_even_when_output_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();path=directory/'protocol.json';out=directory/'fresh'
            value={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO',
                'output':str(out)}
            runner.write_new(path,value)
            runner.write_new(directory/'execution-started.json',{'protocol_sha256':'prior-consumed-attempt'})
            with patch.object(runner,'validate_bound'),self.assertRaises(FileExistsError):
                runner.execute(path,runner.sha(path))
            self.assertFalse(out.exists())
            changed=directory/'another-protocol.json';runner.write_new(changed,{**value,'output':str(directory/'other')})
            with patch.object(runner,'validate_bound'),self.assertRaises(FileExistsError):
                runner.execute(changed,runner.sha(changed))
            self.assertFalse((directory/'other').exists())

    def test_cpu_arm150_ledger_and_restore_and_safety_early_stop(self):
        import torch
        import cached_method_steps
        groups={'new_training':rows('new',57),'replay_training':rows('replay',57)}
        original=torch.ones(2,3);parameter=torch.nn.Parameter(original.clone())
        runtime=SimpleNamespace(model=torch.nn.Linear(3,2),branch=SimpleNamespace(base=SimpleNamespace(down_proj=SimpleNamespace(weight=parameter))))
        baseline={'new':[observed(r,False) for r in groups['new_training']], 'replay_loss':1.}
        def step(rt,samples,replays,master,opt,reference,basis,**kwargs):
            master.grad=torch.ones_like(master);opt.step()
            return {'cpu_fixture':True,'runtime_restored':True,'optimizer_steps':1,
                'ce_role':kwargs['ce_role'],'ce_ids':[samples[0]['id']],'replay_kl_ids':[replays[0]['id']]}
        def check(rt,gs,bs,olds,loss,number,sha,deadline):
            return {'step':number,'provisional_new_correct_ids':['new-0'],'newly_confirmed_ids':[],
                'forgotten_ids':[],'safety_passed':True}
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('cpu-W',parameter)),\
                patch.object(cached_method_steps,'train_method_step',side_effect=step),patch.object(runner,'checkpoint',side_effect=check):
            report=runner.arm(runtime,groups,baseline,[],original,torch.empty(3,0),{},'native',0,Path(folder))
        self.assertTrue(report['completed']);self.assertEqual(report['optimizer_steps'],150)
        self.assertEqual([x['step'] for x in report['checks']],[50,100,150])
        self.assertEqual({k:sum(v.values()) for k,v in report['exposures'].items()},
            {'new_ce':75,'replay_ce':75,'replay_kl':150})
        self.assertTrue(torch.equal(parameter,original))
        def reject(*args):return {**check(*args),'safety_passed':False}
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('cpu-W',parameter)),\
                patch.object(cached_method_steps,'train_method_step',side_effect=step),patch.object(runner,'checkpoint',side_effect=reject):
            report=runner.arm(runtime,groups,baseline,[],original,torch.empty(3,0),{},'native',0,Path(folder))
        self.assertFalse(report['completed']);self.assertEqual(report['completed_steps'],50)
        self.assertTrue(torch.equal(parameter,original))

    def test_each_raw_probe_actual_weight_and_material_sha_and_frozen_change_reject(self):
        import torch
        from verify_cached_training import tensor_sha
        parameter=torch.nn.Parameter(torch.ones(2,3));row=rows('new',1)[0]
        with patch('luban.evolution._target',return_value=('W',parameter)), \
                patch('luban.evolution._generation',return_value=[observed(row)]):
            raw=runner.probes(SimpleNamespace(active_core=None),[row],expected_tensor_sha=tensor_sha(parameter))
        self.assertEqual(raw[0]['sample_sha256'],runner.content_sha(row))
        self.assertEqual(raw[0]['actual_tensor_sha256'],tensor_sha(parameter))
        def mutated(*args):
            with torch.no_grad():parameter.add_(1)
            return [observed(row)]
        with patch('luban.evolution._target',return_value=('W',parameter)), \
                patch('luban.evolution._generation',side_effect=mutated),self.assertRaises(ValueError):
            runner.probes(SimpleNamespace(active_core=None),[row],expected_tensor_sha=tensor_sha(parameter))

    def test_deadline_or_weight_change_after_loss_rejects_frozen_final(self):
        import torch
        from verify_cached_training import tensor_sha
        parameter=torch.nn.Parameter(torch.ones(2,3));news=rows('new',1)
        groups={'new_training':news,'replay_training':[]};baseline={'new':[observed(news[0],False)]}
        clock=[0.]
        def loss(*args):clock[0]=2.;return 1.
        with patch('luban.evolution._target',return_value=('W',parameter)), \
                patch('luban.evolution._loss',side_effect=loss),patch.object(runner.time,'perf_counter',side_effect=lambda:clock[0]), \
                patch.object(runner,'probes',side_effect=lambda rt,rs,*args:[observed(r) for r in rs]),self.assertRaises(TimeoutError):
            runner.checkpoint(None,groups,baseline,[],1.,150,tensor_sha(parameter),1.)
        def bad_loss(*args):
            with torch.no_grad():parameter.add_(1)
            return 1.
        with patch('luban.evolution._target',return_value=('W',parameter)), \
                patch('luban.evolution._loss',side_effect=bad_loss), \
                patch.object(runner,'probes',side_effect=lambda rt,rs,*args:[observed(r) for r in rs]),self.assertRaises(ValueError):
            runner.checkpoint(None,groups,baseline,[],1.,150,tensor_sha(parameter),None)

    def test_load_fault_before_model_exists_releases_actual_cpu_os_lock_and_preserves_raw(self):
        import torch
        from luban.process_lock import ModelProcessLock
        rt=SimpleNamespace(model=None,device='cpu',dtype=torch.float32,_model_process_lock=None)
        def fail(*,paging):raise ValueError('CPU fixture load interrupted before model allocated')
        def release():
            if rt._model_process_lock is not None:rt._model_process_lock.close()
            rt._model_process_lock=None
        rt.load=fail;rt._release_model=release
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder).resolve();out=directory/'execution';path=directory/'protocol.json'
            value={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO',
                'output':str(out),'source_sha256':{},'production_snapshot':{'CPU':True},
                'production_root':str(directory),'official_model_binding':{'CPU':True},'model_path':str(directory),
                'device':'cpu','dtype':'torch.float32'}
            runner.write_new(path,value)
            with patch.object(runner,'validate_bound'),patch('luban.runtime.Runtime',return_value=rt), \
                    patch.object(runner,'production_snapshot',return_value={'CPU':True}), \
                    patch('verify_router_coverage_study.model_binding',return_value={'CPU':True}),self.assertRaises(RuntimeError):
                runner.execute(path,runner.sha(path))
            raw=json.loads((out/'result.json').read_text())
            self.assertFalse(raw['passed']);self.assertEqual(raw['error']['type'],'ValueError')
            self.assertTrue(raw['model_released']);self.assertTrue(raw['actual_os_lock_reacquired_closed'])
            lock=ModelProcessLock(runner.IDENTITY,'cpu');lock.close()
            with patch.object(runner,'validate_bound'),self.assertRaises(FileExistsError):runner.execute(path,runner.sha(path))


if __name__=='__main__':unittest.main()
