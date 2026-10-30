"""SOURCE controls and actual tinyCPU Adam, not 2B generation/quality evidence."""
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
sys.path.insert(0,str(ROOT/'src'))
import run_training_subset450 as runner
from test_training_method_pilot import rows,observed


class Subset450Tests(unittest.TestCase):
    def test_default_and_invalid_cli_never_import_model(self):
        out=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_training_subset450.py')],
                           capture_output=True,text=True,check=True)
        value=json.loads(out.stdout)
        self.assertFalse(value['model_started']);self.assertFalse(value['torch_imported'])
        self.assertEqual((value['ce_events'],value['optimizer_updates'],value['checks']), (450,450,[450]))
        self.assertEqual(value['fullwall_cap_seconds'],2400);self.assertEqual(value['subset_choices'],[16,32,57])
        self.assertFalse(value['CPU_tests_run'])
        for args in (['--run','missing'],['--output','missing'],['--prepare','missing','--output','missing']):
            bad=subprocess.run([sys.executable,'-S',str(ROOT/'scripts/run_training_subset450.py'),*args],
                               capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)

    def groups(self):
        news=rows('new',57)
        for row,id in zip(news,runner.SELECTED_IDS[57]):
            row['id']=id
            source=json.loads(row['provenance'])
            if id in runner.NONFINITE_IDS:source['generation_check']={'type':'contains','expected':'OK'}
            row['provenance']=json.dumps(source)
        return {'new_training':news,'replay_training':rows('replay',57),
                'new_validation':[],'replay_validation':[]}

    def test_nested_exact_schedule_and_three_accounts_all_n(self):
        groups=self.groups();self.assertTrue(set(runner.SELECTED_IDS[16])<set(runner.SELECTED_IDS[32])<set(runner.SELECTED_IDS[57]))
        for n,minimum,maximum in ((16,14,15),(32,7,8),(57,3,4)):
            schedule=runner.subset_schedule(groups,n)
            batches=[b for s in schedule['segments'] for b in s['batches']]
            self.assertEqual([s['check_ce_event'] for s in schedule['segments']],list(range(50,451,50)))
            self.assertEqual([b['optimizer_update'] for b in batches],list(range(1,451)))
            self.assertEqual(schedule['exposure_totals'],{'new_ce':225,'replay_ce':225,'replay_kl':450})
            self.assertEqual(set(schedule['per_id_exposures']['new_ce']),set(runner.SELECTED_IDS[n]))
            self.assertEqual((min(schedule['per_id_exposures']['new_ce'].values()),max(schedule['per_id_exposures']['new_ce'].values())),(minimum,maximum))
            for i,b in enumerate(batches):
                role='new' if i%2==0 else 'replay';pair=i//2
                self.assertEqual(b['ce_role'],role)
                expected=runner.SELECTED_IDS[n][pair%n] if role=='new' else groups['replay_training'][pair%57]['id']
                self.assertEqual(b['ce_ids'],[expected]);self.assertEqual(b['replay_kl_ids'],[groups['replay_training'][pair%57]['id']])
            self.assertEqual(list(schedule['per_id_exposures']['replay_ce'].values()),[4]*54+[3]*3)
            self.assertEqual(list(schedule['per_id_exposures']['replay_kl'].values()),[8]*54+[6]*3)
        for bad in (True,15,33,'16'):
            with self.assertRaises(ValueError):runner.subset_schedule(groups,bad)
        for changed in (groups['new_training'][:-1],list(reversed(groups['new_training']))):
            with self.assertRaises(ValueError):runner.subset_schedule({**groups,'new_training':changed},16)

    def test_fresh_known_finite_counts_and_materials_guard(self):
        groups=self.groups();baseline={'new':[observed(r,r['id'] in runner.KNOWN_IDS) for r in groups['new_training']]}
        for n in (16,32,57):
            selection=runner.selected_classes(groups,baseline,n)
            self.assertEqual(selection['classes'],runner.CLASS_COUNTS[n]);self.assertEqual(len(selection['finite_baseline_correct_ids']),12)
            self.assertEqual([r['id'] for r in selection['selected_training_materials']],list(runner.SELECTED_IDS[n]))
        changed=copy.deepcopy(baseline);known=next(i for i,r in enumerate(groups['new_training']) if r['id'] in runner.KNOWN_IDS)
        changed['new'][known]=observed(groups['new_training'][known],False)
        with self.assertRaises(ValueError):runner.selected_classes(groups,changed,16)
        changed=copy.deepcopy(groups);source=json.loads(changed['new_training'][0]['provenance']);source['generation_check']={'type':'contains','expected':'OK'}
        changed['new_training'][0]['provenance']=json.dumps(source)
        with self.assertRaises(ValueError):runner.selected_classes(changed,baseline,16)

    def fixture(self,*,n=16,reject=False,no_gain=False,untaught_only=False,cancel=False,clock=None):
        import torch
        import cached_method_steps
        from verify_cached_training import tensor_sha
        groups=self.groups()
        # Constant tensor avoids consuming RNG for model initialization.
        parameter=torch.nn.Parameter(torch.full((2,3),.25,dtype=torch.bfloat16))
        original=parameter.detach().clone();runtime=SimpleNamespace()
        baseline={'new':[observed(r,r['id'] in runner.KNOWN_IDS) for r in groups['new_training']],
                  'replay':[observed(r,i<55) for i,r in enumerate(groups['replay_training'])],'replay_loss':1.}
        schedule=runner.subset_schedule(groups,n)
        historical={k:'deliberately_different_CPU_fixture' for k in ('candidate_tensor_sha256',
            'fp32_master_sha256','optimizer_state_sha256','cumulative_delta_sha256','cumulative_per_id_exposures')}
        plan={'protocol':runner.PROTOCOL,'source_sha256':{},'historical450':historical,
              'selected_new_pool_n':n,'selection':runner.selected_classes(groups,baseline,n),
              'schedule_sha256':runner.tensor.content_sha(schedule)}
        calls=[];actual_steps=[];first_proof_at=[];grad=parameter.grad;flag=parameter.requires_grad
        rng=torch.get_rng_state().clone()
        # A genuinely persistent Adam trajectory, independently compared with the same update sequence.
        reference=torch.nn.Parameter(original.float().clone());reference_opt=torch.optim.Adam([reference],lr=1e-5,eps=1e-6)
        for _ in range(450):reference.grad=torch.full_like(reference,.1);reference_opt.step()
        def step(rt,samples,replays,master,opt,*args,**kw):
            master.grad=torch.full_like(master,.1);opt.step();actual_steps.append(int(float(opt.state[master]['step'])))
            if cancel:
                with torch.no_grad():parameter.add_(1)
                raise KeyboardInterrupt('tinyCPU after one actual Adam')
            return {'runtime_restored':True,'optimizer_steps':1,'ce_role':kw['ce_role'],
                    'ce_ids':[r['id'] for r in samples],'replay_kl_ids':[r['id'] for r in replays]}
        def probes(rt,samples,deadline,expected):
            self.assertEqual(tensor_sha(parameter),expected);first_proof_at.append(len(actual_steps))
            calls.append([r['id'] for r in samples])
            return [observed(r,(not no_gain or r['id'] not in runner.SELECTED_IDS[57] or r['id'] in runner.KNOWN_IDS)
                and (not untaught_only or r['id'] not in runner.SELECTED_IDS[n] or r['id'] in runner.KNOWN_IDS)
                and not(reject and r['id']==runner.KNOWN_IDS[0])) for r in samples]
        real_write=Path.write_text
        def write(path,text,*args,**kwargs):
            if clock is not None and json.loads(text).get('completed') is True:clock[0]=2401.
            return real_write(path,text,*args,**kwargs)
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('tiny',parameter)),\
                patch.object(cached_method_steps,'train_method_step',side_effect=step),\
                patch.object(runner.prior,'probes',side_effect=probes),patch('luban.evolution._loss',return_value=1.),\
                patch.object(Path,'write_text',write):
            output=Path(folder).resolve();result=None
            if cancel:
                with self.assertRaises(KeyboardInterrupt):
                    runner.subset(runtime,groups,baseline,groups['replay_training'][:55],original,
                                    torch.empty(3,0),{},plan,output,runner.time.perf_counter()+60,'CPU-fixture')
                result=json.loads((output/'training.json').read_text())
            else:
                result=runner.subset(runtime,groups,baseline,groups['replay_training'][:55],original,
                                      torch.empty(3,0),{},plan,output,2400. if clock is not None else runner.time.perf_counter()+60,'CPU-fixture')
                self.assertEqual(len(list(output.glob('*.pt'))),2)
                payload=torch.load(result['full_training_state']['path'],weights_only=True)
                self.assertEqual(payload['protocol'],runner.PROTOCOL);self.assertEqual(payload['optimizer_updates'],450)
                self.assertEqual(payload['selected_new_pool_n'],n);self.assertEqual(payload['selection'],plan['selection'])
                self.assertEqual(payload['exposures'],schedule['per_id_exposures'])
                self.assertEqual(payload['exposure_totals'],schedule['exposure_totals'])
                self.assertTrue(torch.equal(payload['master'],reference))
                self.assertTrue(torch.equal(payload['candidate'],reference.to(torch.bfloat16)))
                actual_state=next(iter(payload['optimizer_state_dict']['state'].values()))
                for key,value in reference_opt.state[reference].items():self.assertTrue(torch.equal(actual_state[key],value))
                self.assertTrue(torch.equal(payload['cpu_rng'],rng));self.assertIsNone(payload['mps_rng'])
                self.assertEqual(runner.sha(result['full_training_state']['path']),result['full_training_state']['file_sha256'])
                self.assertFalse(result['historical450_comparison_is_quality_gate'])
                self.assertTrue(all(v is False for v in result['historical450_comparison'].values()))
        self.assertTrue(torch.equal(parameter,original));self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        self.assertEqual(parameter.requires_grad,flag);self.assertIs(parameter.grad,grad)
        return result,actual_steps,calls,first_proof_at

    def test_real_cpu450_adam_continuity_only_final169_probes_full_checkpoint(self):
        result,steps,calls,proof_at=self.fixture()
        self.assertTrue(result['completed']);self.assertTrue(result['quality_passed'])
        self.assertEqual(steps,list(range(1,451)));self.assertEqual(proof_at,[450,450,450])
        self.assertEqual([len(c) for c in calls],[57,57,55]);self.assertEqual(sum(map(len,calls)),169)
        self.assertEqual(len(result['checks']),1);self.assertEqual(result['checks'][0]['ce_events'],450)
        self.assertEqual(len(result['checks'][0]['newly_confirmed_ids']),35)  # Finite only, not historical27.
        self.assertEqual(result['gain']['selected_taught_finite_gain'],10)
        self.assertEqual(len(result['gain']['unselected_original_TRAIN_observed_finite_gain_ids']),25)

    def test_final_old_regression_and_no_growth_are_not_success(self):
        result,_,_,_=self.fixture(reject=True)
        self.assertFalse(result['completed']);self.assertFalse(result['quality_passed'])
        self.assertEqual(result['checks'][0]['forgotten_ids'],[runner.KNOWN_IDS[0]])
        result,_,_,_=self.fixture(no_gain=True)
        self.assertTrue(result['completed']);self.assertFalse(result['quality_passed'])
        self.assertEqual(result['checks'][0]['newly_confirmed_ids'],[])

    def test_each_n_full_same_validation_and_untaught_not_primary_gain(self):
        for n,selected,untaught in ((32,21,14),(57,35,0)):
            result,steps,calls,_=self.fixture(n=n)
            self.assertTrue(result['quality_passed']);self.assertEqual(len(steps),450)
            self.assertEqual([len(c) for c in calls],[57,57,55])
            self.assertEqual(result['gain']['selected_taught_finite_gain'],selected)
            self.assertEqual(len(result['gain']['unselected_original_TRAIN_observed_finite_gain_ids']),untaught)
        result,_,_,_=self.fixture(untaught_only=True)
        self.assertTrue(result['completed']);self.assertFalse(result['quality_passed'])
        self.assertEqual(result['gain']['selected_taught_finite_gain'],0)
        self.assertGreater(len(result['gain']['unselected_original_TRAIN_observed_finite_gain_ids']),0)

    def test_last_training_raw_write_timeout_rejects_complete450_and_restores(self):
        clock=[0.]
        with patch.object(runner.time,'perf_counter',side_effect=lambda:clock[0]):result,steps,_,_=self.fixture(clock=clock)
        self.assertFalse(result['completed']);self.assertFalse(result['quality_passed'])
        self.assertEqual(len(steps),450);self.assertEqual(result['error']['type'],'TimeoutError')
        self.assertTrue(result['original_tensor_restored'])

    def test_cancel_after_real_cpu_adam_preserves_error_and_restores(self):
        result,steps,calls,_=self.fixture(cancel=True)
        self.assertEqual(len(steps),1);self.assertFalse(calls)
        self.assertFalse(result['completed']);self.assertEqual(result['error']['type'],'KeyboardInterrupt')
        self.assertTrue(result['original_tensor_restored'])

    def test_once_marker_preflight_mismatch_deadline_and_other_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'protocol.json';output=root/'output'
            plan={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO','output':str(output)}
            runner.write_new(path,plan)
            with self.assertRaises(ValueError):runner.execute(path,'0'*64)
            with patch.object(runner,'validate_bound',side_effect=ValueError('wrongsource')):
                with self.assertRaises(ValueError):runner.execute(path,runner.sha(path))
            clock=[0.]
            def expired(plan):clock[0]=2401.
            with patch.object(runner,'validate_bound',side_effect=expired),patch.object(runner.time,'perf_counter',side_effect=lambda:clock[0]):
                with self.assertRaises(TimeoutError):runner.execute(path,runner.sha(path))
            self.assertFalse(output.exists());self.assertFalse((root/'execution-started.json').exists())
            runner.write_new(root/'execution-started.json',{'CPU_fixture_consumed':True})
            for target in (output,root/'another-output'):
                plan['output']=str(target);path.write_text(json.dumps(plan))
                with patch.object(runner,'validate_bound'):
                    with self.assertRaises(FileExistsError):runner.execute(path,runner.sha(path))
                self.assertFalse(target.exists())

    def test_load_failure_releases_real_cpu_lock_keeps_once_resources_and_error(self):
        import torch
        from luban.process_lock import ModelProcessLock
        runtime=SimpleNamespace(model=None,device='cpu',dtype=torch.bfloat16,_model_process_lock=None)
        def fail(*,paging):raise ValueError('CPU fixture load refused, no model constructed')
        def release():
            if runtime._model_process_lock is not None:runtime._model_process_lock.close()
            runtime._model_process_lock=None
        runtime.load=fail;runtime._release_model=release
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'protocol.json';output=root/'output';reference=root/'reference.json';receipt=root/'receipt.json'
            runner.write_new(reference,{'CPU_fixture_only':True});runner.write_new(receipt,{'CPU_fixture_only':True})
            plan={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO',
                'output':str(output),'model_path':str(root/'no-model'),'device':'cpu','dtype':'torch.bfloat16',
                'source_sha256':{},'production_root':str(root),'production_snapshot':{'CPU':True},
                'official_model_binding':{'CPU':True},'curve_receipt_path':str(receipt),'curve_receipt_sha256':runner.sha(receipt),
                'b1_actual_reference':{'path':str(reference),'sha256':runner.sha(reference)}}
            runner.write_new(path,plan)
            with patch.object(runner,'validate_bound'),patch('luban.runtime.Runtime',return_value=runtime),\
                    patch('verify_router_coverage_study.model_binding',return_value={'CPU':True}),\
                    patch.object(runner,'production_snapshot',return_value={'CPU':True}):
                with self.assertRaises(RuntimeError):runner.execute(path,runner.sha(path))
            raw=json.loads((output/'result.json').read_text())
            self.assertFalse(raw['model_started']);self.assertFalse(raw['passed'])
            self.assertTrue(raw['model_released']);self.assertTrue(raw['actual_os_lock_reacquired_closed'])
            self.assertTrue(raw['preserved']);self.assertTrue((root/'execution-started.json').exists())
            self.assertEqual(raw['error']['type'],'ValueError');self.assertIn('resources',raw)
            self.assertGreaterEqual(len(raw['resources']['snapshots']),2)
            self.assertIn('wall_seconds_after_first_final_result_write',raw)
            lock=ModelProcessLock(runner.IDENTITY,'cpu');lock.close()


if __name__=='__main__':unittest.main()
