"""Independent tensor pilot admission/ledgers/failure tests; no native2B model."""
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
import run_tensor_method_pilot as runner
from test_training_method_pilot import rows,observed


class TensorMethodPilotTests(unittest.TestCase):
    def test_default_stdlib_zero_model_and_strict_cli(self):
        script=ROOT/'scripts/run_tensor_method_pilot.py'
        raw=subprocess.run([sys.executable,'-S',str(script)],capture_output=True,text=True,check=True)
        value=json.loads(raw.stdout)
        self.assertFalse(value['model_started']);self.assertFalse(value['torch_imported'])
        self.assertEqual(value['ce_events'],150);self.assertEqual(value['actual_optimizer_updates'],{'cacheB2':78,'cacheB4':42})
        for args in (['--run','missing'],['--b1-result','missing'],['--output','missing'],['--approved-sha','a'*64]):
            result=subprocess.run([sys.executable,'-S',str(script),*args],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)

    def test_actual_failed_partial_b1_reference_allowed_no_fabricated150(self):
        groups={'new_training':rows('new',57),'replay_training':rows('replay',57),
                'new_validation':[],'replay_validation':[]}
        value={'protocol':runner.prior.PROTOCOL,'model_started':True,'passed':False,'preserved':True,
            'model_released':True,'actual_os_lock_reacquired_closed':True,'teacher_calls':0,'publication':False,
            'restoration':{str(i):True for i in range(9)},'methods':[{'method':'native','completed_steps':50}],
            'groups':groups,'groups_sha256':runner.content_sha(groups)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'raw.json';runner.write_new(path,value)
            receipt=runner.reference_receipt(path)
            self.assertFalse(receipt['passed']);self.assertEqual(receipt['available_optimizer_steps'],{'native':50})
            for protocol in ('original57-native-cache-b1-pilot150-v1',
                             'original57-native-cache-b1-pilot150-tokenizer-eos-v2'):
                path.write_text(json.dumps({**value,'protocol':protocol}))
                self.assertEqual(runner.reference_receipt(path)['protocol'],protocol)
            for changes in ({'model_started':False},{'preserved':False},{'teacher_calls':1},{'publication':True},
                            {'restoration':{'x':True}},{'actual_os_lock_reacquired_closed':False},{'protocol':'unknown'},
                            {'groups_sha256':'0'*64}):
                path.write_text(json.dumps({**value,**changes}))
                with self.subTest(changes=changes),self.assertRaises(ValueError):runner.reference_receipt(path)

    def test_complete_material_guard_same_count_different_pool_id_and_raw_mutation(self):
        groups={'new_training':rows('new',57),'replay_training':rows('replay',57),
                'new_validation':[],'replay_validation':[]}
        value={'protocol':runner.prior.PROTOCOL,'model_started':True,'passed':False,'preserved':True,
            'model_released':True,'actual_os_lock_reacquired_closed':True,'teacher_calls':0,'publication':False,
            'restoration':{str(i):True for i in range(9)},'methods':[{'method':'native','completed_steps':0}],
            'groups':groups,'groups_sha256':runner.content_sha(groups)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory).resolve()/'raw.json';runner.write_new(path,value)
            reference=runner.reference_receipt(path)
            runner.verify_training_materials(copy.deepcopy(groups),reference)
            generated=copy.deepcopy(groups)
            provenance=json.loads(generated['replay_training'][0]['provenance'])
            provenance['original_baseline_generation']={'actual_run_proof':'different'}
            generated['replay_training'][0]['provenance']=json.dumps(provenance)
            runner.verify_training_materials(generated,reference)
            with self.assertRaises(ValueError):runner.training_materials({**groups,'unexpected':'bad'})
            with self.assertRaises(ValueError):
                runner.verify_training_materials(groups,{**reference,'training_material_contract':'unknown'})
            for change in ('id','answer','prompt','rights','order','provenance'):
                changed=copy.deepcopy(groups)
                row=changed['replay_training'][0]
                if change=='order':changed['replay_training'][:2]=reversed(changed['replay_training'][:2])
                elif change=='provenance':
                    source=json.loads(row['provenance']);source['generation_check']['expected']='OTHER'
                    row['provenance']=json.dumps(source)
                else:row[change]='changed-fixed-pool-row'
                with self.subTest(change=change),self.assertRaises(ValueError):
                    runner.verify_training_materials(changed,reference)
            path.write_text(json.dumps({**value,'unrelated_raw_field':'tampered'}))
            with self.assertRaisesRegex(ValueError,'raw SHA changed'):
                runner.verify_training_materials(groups,reference)

    def fixture_arm(self,batch,*,reject=False,gain=True,mutation=None):
        import torch
        import cached_method_steps
        groups={'new_training':rows('new',57),'replay_training':rows('replay',57)}
        model=torch.nn.Linear(3,2);parameter=model.weight;original=parameter.detach().clone()
        runtime=SimpleNamespace(model=model)
        baseline={'new':[observed(r,False) for r in groups['new_training']],'replay_loss':1.}
        starts=[]
        def step(rt,samples,replays,master,opt,reference,basis,**kwargs):
            if not opt.state:starts.append(master.detach().clone())
            master.grad=torch.full_like(master,.1);opt.step()
            if mutation:mutation()
            return {'cpu_fixture':True,'runtime_restored':True,'optimizer_steps':1,'ce_role':kwargs['ce_role'],
                    'ce_ids':[r['id'] for r in samples],'replay_kl_ids':[r['id'] for r in replays]}
        def check(rt,gs,bs,old,loss,event,sha,deadline):
            return {'step':event,'provisional_new_correct_ids':['new-0'],'newly_confirmed_ids':['new-0'] if event==150 and gain else [],
                    'forgotten_ids':['old-0'] if reject else [],'safety_passed':not reject}
        flags=[(p,p.requires_grad,p.grad) for p in model.parameters()];modes=[(m,m.training) for m in model.modules()]
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('cpu-W',parameter)),\
                patch.object(cached_method_steps,'train_method_step',side_effect=step),patch.object(runner.prior,'checkpoint',side_effect=check):
            result=runner.arm(runtime,groups,baseline,[],original,torch.empty(3,0),{},batch,0,Path(folder))
        self.assertTrue(torch.equal(parameter,original));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.requires_grad==f and p.grad is g for p,f,g in flags))
        self.assertTrue(all(m.training==mode for m,mode in modes));self.assertTrue(torch.equal(starts[0],original.float()))
        return result

    def test_actual_cpu_adam78_42_full_roles_tail_and_exact_checks(self):
        for batch,updates in ((2,78),(4,42)):
            with self.subTest(batch=batch):
                report=self.fixture_arm(batch)
                self.assertTrue(report['completed']);self.assertTrue(report['quality_passed'])
                self.assertFalse(report['operator_equivalence_claim']);self.assertFalse(report['product_accepted'])
                self.assertEqual(report['optimizer_steps'],updates);self.assertEqual(report['completed_ce_events'],150)
                self.assertEqual([c['ce_events'] for c in report['checks']],[50,100,150])
                self.assertEqual([c['adam_step'] for c in report['checks']],[updates//3,updates*2//3,updates])
                self.assertEqual({k:sum(v.values()) for k,v in report['exposures'].items()},
                                 {'new_ce':75,'replay_ce':75,'replay_kl':150})
                self.assertTrue(any(u['schedule_batch']['short_tail'] for u in report['updates']))
                self.assertTrue(all(report['arm_state_restoration'].values()))

    def test_no_gain_completed_quality_false_and_safety_failure_stops50(self):
        none=self.fixture_arm(2,gain=False)
        self.assertTrue(none['completed']);self.assertFalse(none['quality_passed'])
        rejected=self.fixture_arm(4,reject=True)
        self.assertFalse(rejected['completed']);self.assertFalse(rejected['quality_passed'])
        self.assertEqual(rejected['completed_ce_events'],50);self.assertEqual(rejected['optimizer_steps'],14)
        self.assertEqual(len(rejected['checks']),1)

    def test_deadline_after_actual_update_restores_and_no_later_events(self):
        clock=[0.]
        def advance():clock[0]=3601.
        with patch.object(runner.time,'perf_counter',side_effect=lambda:clock[0]):
            report=self.fixture_arm(2,mutation=advance)
        self.assertFalse(report['completed']);self.assertEqual(report['optimizer_steps'],1)
        self.assertEqual(report['completed_ce_events'],2);self.assertEqual(report['error']['type'],'TimeoutError')
        self.assertTrue(report['original_tensor_restored'])

    def test_bound_requires_reference_and_run_preflight_cannot_consume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();path=root/'protocol.json';output=root/'output'
            value={'protocol':runner.PROTOCOL,'state':'sealed_waiting_explicit_MODEL_GO','sealed':True,'output':str(output)}
            runner.write_new(path,value)
            with self.assertRaises(ValueError):runner.bind({'b1_actual_reference':None})
            with self.assertRaises(ValueError):runner.execute(path,'0'*64)
            with patch.object(runner,'validate_bound',side_effect=ValueError('wrongsource')),self.assertRaises(ValueError):
                runner.execute(path,runner.sha(path))
            self.assertFalse(output.exists());self.assertFalse((root/'execution-started.json').exists())
            runner.write_new(root/'execution-started.json',{'CPU_existing_consumed':True})
            with patch.object(runner,'validate_bound'),self.assertRaises(FileExistsError):runner.execute(path,runner.sha(path))
            self.assertFalse(output.exists())

    def test_final_raw_save_deadline_rejects_completed_arm_and_keeps_restored_weight(self):
        clock=[0.];writes=[];original_write=Path.write_text
        def save(path,text,*args,**kwargs):
            value=json.loads(text)
            writes.append(value)
            if value.get('completed') is True:clock[0]=3601.
            return original_write(path,text,*args,**kwargs)
        with patch.object(runner.time,'perf_counter',side_effect=lambda:clock[0]),patch.object(Path,'write_text',save):
            report=self.fixture_arm(4)
        self.assertEqual(report['completed_ce_events'],150);self.assertEqual(report['optimizer_steps'],42)
        self.assertFalse(report['completed']);self.assertFalse(report['quality_passed'])
        self.assertEqual(report['error']['type'],'TimeoutError')
        self.assertEqual(report['timing']['end_to_end_charged_seconds'],3601.)
        self.assertTrue(report['original_tensor_restored']);self.assertFalse(writes[-1]['completed'])
        self.assertFalse(writes[-1]['quality_passed'])

    def test_loadnone_fault_actual_cpu_os_lock_and_consumed_protocol_blocks_second_output(self):
        import torch
        from luban.process_lock import ModelProcessLock
        rt=SimpleNamespace(model=None,device='cpu',dtype=torch.float32,_model_process_lock=None)
        def fail(*,paging):raise ValueError('CPU fixture no model allocation')
        def release():
            if rt._model_process_lock is not None:rt._model_process_lock.close()
            rt._model_process_lock=None
        rt.load=fail;rt._release_model=release
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();path=root/'protocol.json';out=root/'execution';reference=root/'b1.json'
            runner.write_new(reference,{'CPU_only':True})
            value={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO',
                'output':str(out),'source_sha256':{},'production_snapshot':{'CPU':True},'production_root':str(root),
                'official_model_binding':{'CPU':True},'model_path':str(root),'device':'cpu','dtype':'torch.float32',
                'b1_actual_reference':{'path':str(reference),'sha256':runner.sha(reference)}}
            runner.write_new(path,value)
            with patch.object(runner,'validate_bound'),patch('luban.runtime.Runtime',return_value=rt),\
                    patch.object(runner,'production_snapshot',return_value={'CPU':True}),\
                    patch('verify_router_coverage_study.model_binding',return_value={'CPU':True}),self.assertRaises(RuntimeError):
                runner.execute(path,runner.sha(path))
            result=json.loads((out/'result.json').read_text())
            self.assertFalse(result['model_started']);self.assertFalse(result['passed'])
            self.assertTrue(result['model_released']);self.assertTrue(result['actual_os_lock_reacquired_closed'])
            lock=ModelProcessLock(runner.IDENTITY,'cpu');lock.close()
            value['output']=str(root/'second-output');path.write_text(json.dumps(value))
            with patch.object(runner,'validate_bound'),self.assertRaises(FileExistsError):runner.execute(path,runner.sha(path))
            self.assertFalse((root/'second-output').exists())


if __name__=='__main__':unittest.main()
