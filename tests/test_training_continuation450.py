"""Written SOURCE fixtures; execute only after Root CPU resource-window GO.

Tiny real Adam/native-cache operators are CPU evidence, not 2B continuation.
"""
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
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import run_training_continuation450 as runner
from test_training_method_pilot import observed


def parent_fixture(n=16):
    from test_training_subset450 import Subset450Tests
    groups=Subset450Tests().groups();news=groups['new_training']
    known=set(runner.first.KNOWN_IDS)
    finite_known={r['id'] for r in news if r['id'] in known and r['id'] not in runner.first.NONFINITE_IDS}
    extra=[r['id'] for r in news if r['id'] not in known and r['id'] not in runner.first.NONFINITE_IDS][:5]
    finite_f1=finite_known|set(extra)
    baseline={'new':[observed(r,r['id'] in known) for r in news],
              'replay':[observed(r,i<55) for i,r in enumerate(groups['replay_training'])]}
    point={'new_first':[observed(r,r['id'] in finite_f1|known or r['id'] in runner.first.NONFINITE_IDS) for r in news],
           'new_second':[observed(r,r['id'] in finite_f1|known or r['id'] in runner.first.NONFINITE_IDS) for r in news]}
    schedule=runner.first.subset_schedule(groups,n)
    return {'groups':groups,'baseline':baseline,'point':point,'finite_f1_ids':sorted(finite_f1),
        'confirmed_ids':extra,'candidate_sha256':'CPU_fixture_not_actual_parent','plan':{'selected_new_pool_n':n},
        'training':{'exposures':schedule['per_id_exposures']}}


class Continuation450Tests(unittest.TestCase):
    def test_default_and_invalid_run_are_stdlib_no_torch_no_parent_scan(self):
        script=ROOT/'scripts/run_training_continuation450.py'
        out=subprocess.run([sys.executable,'-S',str(script)],capture_output=True,text=True,check=True)
        report=json.loads(out.stdout)
        self.assertFalse(report['torch_imported']);self.assertFalse(report['model_started']);self.assertFalse(report['CPU_tests_run'])
        self.assertEqual((report['entry_Adam'],report['exit_Adam'],report['source_count']),(450,900,69))
        for args in (['--run','missing'],['--output','missing'],['--prepare','missing','--output','missing']):
            out=subprocess.run([sys.executable,'-S',str(script),*args],capture_output=True,text=True)
            self.assertNotEqual(out.returncode,0)

    def test_pending_all_original_order_parent_cap_and_known13_plus_all_finite_f1_replay(self):
        for n in (16,32,57):
            parent=parent_fixture(n);before=copy.deepcopy(parent);selected=runner.curriculum(parent)
            protect=set(runner.first.KNOWN_IDS)|set(parent['finite_f1_ids'])
            pending=[r['id'] for r in parent['groups']['new_training'] if r['id'] not in protect]
            self.assertEqual(selected['pending_ids'],pending);self.assertEqual(selected['selected_ids'],pending[:n])
            self.assertEqual(len(selected['replay_training']),57+len(protect))
            self.assertEqual(selected['original_known_nonfinite_ids'],sorted(set(runner.first.KNOWN_IDS)-set(parent['finite_f1_ids'])))
            self.assertTrue((set(runner.first.NONFINITE_IDS)-set(runner.first.KNOWN_IDS)).issubset(pending))
            self.assertEqual(parent,before)
            converted={r['id']:r for r in selected['replay_training'][57:]}
            for row in parent['groups']['new_training']:
                if row['id'] in converted:
                    actual=converted[row['id']]
                    self.assertEqual((actual['answer'],actual['prompt'],actual['purpose']),(row['answer'],row['prompt'],row['purpose']))
                    proof=json.loads(actual['provenance'])
                    self.assertEqual(proof['evolution_role'],'replay');self.assertEqual(proof['continuation_source_sha256'],runner.tensor.content_sha(row))
            if n!=57:self.assertNotEqual(selected['selected_ids'],[id for id in runner.first.SELECTED_IDS[n] if id in pending])
            else:self.assertEqual(selected['selected_ids'],pending)

    def test_cumulative_three_accounts_roles_and_parent_to_replay_not_global_step_counts(self):
        parent=parent_fixture();selected=runner.curriculum(parent);schedule=runner.continuation_schedule(parent,selected)
        self.assertEqual([b['cumulative_optimizer_step'] for b in schedule['batches']],list(range(451,901)))
        self.assertEqual(schedule['round_totals'],{'new_ce':225,'replay_ce':225,'replay_kl':450})
        self.assertEqual(schedule['cumulative_totals'],{'new_ce':450,'replay_ce':450,'replay_kl':900})
        firstnew=parent['confirmed_ids'][0]
        self.assertNotIn(firstnew,schedule['round_exposures']['new_ce'])
        self.assertGreater(schedule['round_exposures']['replay_ce'][firstnew],0)
        self.assertEqual(schedule['cumulative_exposures']['new_ce'].get(firstnew,0),parent['training']['exposures']['new_ce'].get(firstnew,0))
        self.assertEqual(len(schedule['batches']),450)
        bad=copy.deepcopy(selected);bad['selected_ids']=[selected['replay_training'][0]['id']]
        with self.assertRaises(ValueError):runner.continuation_schedule(parent,bad)

    def state_fixture(self,folder,parameter):
        import torch
        from verify_cached_training import tensor_sha
        master=torch.nn.Parameter(parameter.detach().float().clone()+.000031)
        optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
        for _ in range(450):master.grad=torch.full_like(master,.1);optimizer.step()
        candidate=master.detach().to(torch.bfloat16)
        saved={'path':str(Path(folder)/'native.pt'),'tensor_sha256':tensor_sha(candidate),'CPU_fixture':True}
        state={key:value.detach().cpu().clone() for key,value in optimizer.state[master].items()}
        state_dict=copy.deepcopy(optimizer.state_dict())
        payload={'protocol':runner.first.PROTOCOL,'protocol_sha256':'CPU_fabricated_parent_protocol',
            'source_sha256':{},'ce_events':450,'optimizer_updates':450,'identity':runner.IDENTITY,
            'original_tensor_sha256':runner.ORIGINAL,'exposures':{'CPU_fixture':True},'native_checkpoint':saved,
            'candidate':candidate,'master':master.detach().clone(),'optimizer_state_dict':state_dict,
            'cpu_rng':torch.get_rng_state().clone(),'mps_rng':None}
        path=Path(folder).resolve()/'state.pt';torch.save(payload,path)
        parent={'plan':{'protocol':runner.first.PROTOCOL,'source_sha256':{}},'descriptor':{'formal':{'sha256':'CPU_fabricated_parent_protocol'}},
            'training':{'exposures':payload['exposures'],'saved_weights':saved,
                'full_training_state':{'path':str(path),'file_sha256':runner.sha(path),'master_sha256':tensor_sha(master)},
                'actual450_state_summary':{'optimizer_state_sha256':{k:tensor_sha(v.reshape(-1)) for k,v in state.items()}}},
            'candidate_sha256':tensor_sha(candidate)}
        return parent,payload,master,optimizer

    def test_actual_tiny_official_adam450_import_451_native_cache_equal_and_fp32_residual_not_original(self):
        import torch
        import cached_method_steps as method
        import test_cached_method_steps as fixture
        for dtype in (torch.bfloat16,):
            with tempfile.TemporaryDirectory() as folder,torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                runtime,new,replays,_,official,basis,caches=fixture.prepared(folder,dtype)
                parameter=runtime.branch.base.down_proj.weight
                parent,payload,_,_=self.state_fixture(folder,parameter)
                loaded,opt,reference,rng,mpsrng=runner.load_parent_state(runtime,parent)
                self.assertTrue(torch.equal(reference,loaded));self.assertFalse(reference.requires_grad)
                self.assertFalse(torch.equal(reference,official));self.assertFalse(torch.equal(reference,parameter.float()))
                self.assertTrue(torch.equal(reference.to(parameter.dtype),parameter))
                self.assertIsNot(opt.state[loaded]['exp_avg'],payload['optimizer_state_dict']['state'][0]['exp_avg'])
                self.assertTrue(torch.equal(rng,payload['cpu_rng']));self.assertIsNone(mpsrng)
                expected=torch.nn.Parameter(loaded.detach().clone());expected_opt=torch.optim.Adam([expected],lr=1e-5,eps=1e-6)
                expected_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
                fixture.uninterrupted_step(runtime,new[0],replays[0],expected,expected_opt,reference,basis)
                with torch.no_grad():parameter.copy_(payload['candidate'])
                before=parameter.detach().clone()
                actual=method.train_method_step(runtime,new[:1],replays[:1],loaded,opt,reference,basis,method='cache',ce_role='new',caches=caches)
                self.assertTrue(actual['runtime_restored']);self.assertTrue(torch.equal(parameter,before))
                self.assertEqual(int(float(opt.state[loaded]['step'])),451);self.assertTrue(torch.equal(loaded,expected))
                for k,v in expected_opt.state[expected].items():self.assertTrue(torch.equal(opt.state[loaded][k],v))

    def test_parent_state_wrong_native_master_moments_rng_sources_or_step_rejects(self):
        import torch
        for change in ('native','masterBF16','round_master','NaNmaster','moment','step','rng','invalid_rng','source','protocol'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                parameter=torch.nn.Parameter(torch.full((2,3),.25,dtype=torch.bfloat16))
                parent,payload,_,_=self.state_fixture(folder,parameter)
                if change=='native':payload['candidate'][0,0]+=1
                if change=='masterBF16':payload['master']=payload['master'].to(torch.bfloat16)
                if change=='round_master':payload['master']=payload['candidate'].float()
                if change=='NaNmaster':payload['master'][0,0]=float('nan')
                if change=='moment':next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg'][0,0]=float('nan')
                if change=='step':next(iter(payload['optimizer_state_dict']['state'].values()))['step'].fill_(0)
                if change=='rng':payload['cpu_rng']=None
                if change=='invalid_rng':payload['cpu_rng']=torch.zeros(1,dtype=torch.uint8)
                if change=='source':payload['source_sha256']={'wrong':'source'}
                if change=='protocol':payload['protocol_sha256']='wrong'
                path=Path(parent['training']['full_training_state']['path']);torch.save(payload,path)
                parent['training']['full_training_state']['file_sha256']=runner.sha(path)
                before=parameter.detach().clone()
                with patch('luban.evolution._target',return_value=('tiny',parameter)):
                    with self.assertRaises(ValueError):runner.load_parent_state(SimpleNamespace(),parent)
                self.assertTrue(torch.equal(parameter,before))

    def test_actual_cpu_parent450_to900_same_moments_ledger_and_no_intermediate_generation(self):
        import torch
        import cached_method_steps
        from verify_cached_training import tensor_sha
        parent=parent_fixture();selected=runner.curriculum(parent);schedule=runner.continuation_schedule(parent,selected)
        parameter=torch.nn.Parameter(torch.full((2,3),.25,dtype=torch.bfloat16))
        master=torch.nn.Parameter(parameter.detach().float().clone()+.000031)
        optimizer=torch.optim.Adam([master],lr=1e-5,eps=1e-6)
        for _ in range(450):master.grad=torch.full_like(master,.1);optimizer.step()
        reference=master.detach().clone();basis=torch.eye(3)[:,:1]
        with torch.no_grad():parameter.copy_(master.to(parameter.dtype))
        entry=parameter.detach().clone();expected=torch.nn.Parameter(master.detach().clone())
        expected_opt=torch.optim.Adam([expected],lr=1e-5,eps=1e-6);expected_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for _ in range(450):
            expected.grad=torch.full_like(expected,.1);expected_opt.step()
            cached_method_steps.cached._project_core_delta(expected,reference,basis)
        steps=[];proof_at=[]
        def step(rt,samples,replays,value,opt,ref,q,**kwargs):
            self.assertTrue(torch.equal(ref,reference));value.grad=torch.full_like(value,.1);opt.step()
            cached_method_steps.cached._project_core_delta(value,ref,q);steps.append(int(float(opt.state[value]['step'])))
            return {'runtime_restored':True,'optimizer_steps':1,'ce_role':kwargs['ce_role'],
                'ce_ids':[r['id'] for r in samples],'replay_kl_ids':[r['id'] for r in replays]}
        def probes(rt,rows,deadline,expected_sha):
            proof_at.append(len(steps));self.assertEqual(tensor_sha(parameter),expected_sha)
            return [observed(row,True) for row in rows]
        plan={'schedule_sha256':runner.tensor.content_sha(schedule),'source_sha256':{},'parent_full_state':{'CPU_fixture':True}}
        with tempfile.TemporaryDirectory() as folder,patch('luban.evolution._target',return_value=('tiny',parameter)),\
                patch('luban.evolution._loss',return_value=1.),patch.object(cached_method_steps,'train_method_step',side_effect=step),\
                patch.object(runner.prior,'probes',side_effect=probes):
            report=runner.train_round(SimpleNamespace(),parent,selected,master,optimizer,reference,basis,{},1.,plan,
                Path(folder).resolve(),runner.time.perf_counter()+60,'CPU_fabricated_protocol')
            self.assertTrue(report['quality_passed']);self.assertEqual(steps,list(range(451,901)));self.assertEqual(proof_at,[450,450,450])
            self.assertTrue(torch.equal(parameter,entry));self.assertTrue(torch.equal(master,expected))
            for key,value in expected_opt.state[expected].items():self.assertTrue(torch.equal(optimizer.state[master][key],value))
            payload=torch.load(report['full_training_state']['path'],weights_only=True)
            self.assertEqual(payload['optimizer_updates'],900);self.assertEqual(payload['round_updates'],450)
            self.assertEqual(payload['exposures'],schedule['cumulative_exposures']);self.assertEqual(payload['round_exposures'],schedule['round_exposures'])
            self.assertTrue(torch.equal(payload['master'],expected));self.assertTrue(torch.equal(payload['candidate'],expected.to(torch.bfloat16)))
            self.assertTrue(torch.equal(payload['reference'],reference));self.assertTrue(torch.equal(payload['basis'],basis))
            self.assertEqual(sum(len(report['checks'][0][key]) for key in ('new_first','new_second','old_preservation')),169)

    def test_live_or_weak_parent_descriptor_rejects_before_artifact_access(self):
        original={'process_exit_code':0,'pid_dead':True,'evidence_preserved':True,'execution_passed':True,
            'quality_passed':True,'positive_confirmed_growth':True,'full_state_payload_verified':True}
        for field in tuple(original):
            broken={**original,field:False if field!='process_exit_code' else 1}
            with patch.object(runner,'read',side_effect=[broken,{},{}]),patch.object(runner,'sha') as hash_call:
                with self.assertRaises(ValueError):runner.closed_parent('CPU_descriptor','CPU_backup','CPU_readback')
                hash_call.assert_not_called()

    def test_no_matching_cpu_receipt_no_parent_access_or_prepare(self):
        for cpu in ({}, {'run':True,'passed':False,'exit_code':0}, {'run':True,'passed':True,'exit_code':0,'source_sha256':{}}):
            with patch.object(runner,'read',return_value=cpu),patch.object(runner,'closed_parent') as parent:
                with self.assertRaises(ValueError):runner.prepare('CPU_output','CPU_descriptor','CPU_backup','CPU_readback')
                parent.assert_not_called()

    def test_prepare_fresh_experiment_path_allowed_product_source_model_modules_parent_rejected(self):
        parent=parent_fixture();parent['descriptor']={'formal':{'sha256':'CPU_fabricated'}}
        parent['training']['full_training_state']={'CPU_fixture':True}
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder).resolve();product=base/'product';experiments=product/'experiments'
            parent['plan'].update(model_path=str(base/'model'),production_root=str(product),output=str(experiments/'old'/'execution'),
                source_sha256={str(ROOT/name):runner.sha(ROOT/name) for name in ('scripts/run_training_subset450.py','tests/test_training_subset450.py')})
            # Fabricated source-only receipt, no parentPT/official/product access.
            old_sources={f'CPU_source_{i}':'fixture' for i in range(65)}
            old_sources.update(parent['plan']['source_sha256']);parent['plan']['source_sha256']=old_sources
            cpu={'run':True,'passed':True,'exit_code':0,'log':'CPU_log','log_sha256':'fixture',
                'source_sha256':{str(ROOT/name):runner.sha(ROOT/name) for name in ('scripts/run_training_continuation450.py','tests/test_training_continuation450.py')}}
            actual_sha=runner.sha
            def digest(path):return actual_sha(path) if str(path).startswith(str(ROOT)) and Path(path).is_file() else 'fixture'
            with patch.object(runner,'read',return_value=cpu),patch.object(runner,'closed_parent',return_value=parent),patch.object(runner,'sha',side_effect=digest):
                plan=runner.prepare(experiments/'fresh'/'execution','CPU_desc','CPU_backup','CPU_readback')
                self.assertEqual(plan['output'],str(experiments/'fresh'/'execution'));self.assertEqual(len(plan['source_sha256']),69)
                for bad in (product,experiments,experiments/'single-level',product/'app'/'execution',product/'modules'/'execution',
                        Path(parent['plan']['model_path'])/'execution',ROOT/'execution',Path(parent['plan']['output']),Path(parent['plan']['output'])/'nested'):
                    with self.assertRaises(ValueError):runner.prepare(bad,'CPU_desc','CPU_backup','CPU_readback')

    def test_actual_saved_rng_restore_is_required_not_just_reported(self):
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123);saved=torch.get_rng_state().clone();torch.rand(4)
            self.assertTrue(runner.restore_parent_rng(saved,None));self.assertTrue(torch.equal(torch.get_rng_state(),saved))
            with patch.object(torch,'get_rng_state',return_value=torch.zeros_like(saved)):
                with self.assertRaises(ValueError):runner.restore_parent_rng(saved,None)

    def test_entry_once_covers_known13_and_all_finite_f1_plus_original55(self):
        import torch
        from verify_cached_training import tensor_sha
        parent=parent_fixture();selected=runner.curriculum(parent)
        parameter=torch.nn.Parameter(torch.full((2,3),.25,dtype=torch.bfloat16))
        parent['candidate_sha256']=tensor_sha(parameter);calls=[]
        def probe(rt,rows,deadline,expected):
            self.assertEqual(len(rows),1);self.assertEqual(expected,tensor_sha(parameter))
            calls.extend(r['id'] for r in rows);return [observed(r,True) for r in rows]
        with patch('luban.evolution._target',return_value=('tiny',parameter)),patch.object(runner.prior,'probes',side_effect=probe):
            report=runner.entry_protection(SimpleNamespace(),parent,selected,runner.time.perf_counter()+60)
        self.assertTrue(report['passed']);self.assertEqual(report['forgotten_ids'],[])
        self.assertEqual(calls,report['protected_new_ids']+report['protected_replay_ids'])
        self.assertEqual(set(report['protected_new_ids']),set(selected['protection_ids']))
        self.assertEqual(len(report['protected_replay_ids']),55);self.assertEqual(len(calls),len(set(calls)))
        self.assertEqual(report['generation_calls'],len(selected['protection_ids'])+55)
        for incomplete in (False,True):
            failed=selected['protection_ids'][0]
            def bad(rt,rows,deadline,expected):
                result=[observed(r,r['id']!=failed) for r in rows]
                if incomplete and rows[0]['id']==failed:
                    result=[observed(rows[0],True)];result[0].update(complete=False,passed=False)
                return result
            with patch('luban.evolution._target',return_value=('tiny',parameter)),patch.object(runner.prior,'probes',side_effect=bad):
                rejected=runner.entry_protection(SimpleNamespace(),parent,selected,runner.time.perf_counter()+60)
            self.assertFalse(rejected['passed']);self.assertIn(failed,rejected['forgotten_ids'])

    def test_entry_mutated_weight_or_expired_deadline_rejects_and_preserves_partial_evidence(self):
        import torch
        from verify_cached_training import tensor_sha
        parent=parent_fixture();selected=runner.curriculum(parent)
        parameter=torch.nn.Parameter(torch.full((2,3),.25,dtype=torch.bfloat16))
        parent['candidate_sha256']=tensor_sha(parameter);record={}
        def changed(rt,rows,deadline,expected):
            with torch.no_grad():parameter.add_(1)
            return [observed(r,True) for r in rows]
        with patch('luban.evolution._target',return_value=('tiny',parameter)),patch.object(runner.prior,'probes',side_effect=changed) as probe:
            with self.assertRaises(ValueError):runner.entry_protection(SimpleNamespace(),parent,selected,runner.time.perf_counter()+60,record)
        self.assertFalse(record['passed']);self.assertEqual(record['generation_calls'],1);self.assertEqual(probe.call_count,1)
        self.assertEqual(len(record['new']),1);self.assertEqual(record['replay'],[])
        record={}
        with patch.object(runner.prior,'probes') as probe:
            with self.assertRaises(ValueError):runner.entry_protection(SimpleNamespace(),parent,selected,0,record)
        self.assertFalse(record['passed']);self.assertEqual(record['generation_calls'],0);probe.assert_not_called()

    def test_final169_all_previous_finite_and_known_contains_protected_expanded_loss(self):
        import torch
        parent=parent_fixture();selected=runner.curriculum(parent);parameter=torch.nn.Parameter(torch.full((2,3),.25,dtype=torch.bfloat16))
        from verify_cached_training import tensor_sha
        calls=[];loss_rows=[]
        def probe(rt,rows,deadline,expected):
            calls.append([r['id'] for r in rows]);return [observed(r,True) for r in rows]
        def loss(rt,rows):loss_rows.append(rows);return 1.
        with patch('luban.evolution._target',return_value=('tiny',parameter)),patch('luban.evolution._loss',side_effect=loss),patch.object(runner.prior,'probes',side_effect=probe):
            result=runner.final_check(SimpleNamespace(),parent,selected,tensor_sha(parameter),1.,runner.time.perf_counter()+60)
        self.assertTrue(result['quality_passed']);self.assertEqual([len(r) for r in calls],[57,57,55])
        self.assertEqual(loss_rows,[selected['replay_training']]);self.assertTrue(result['first_finite_f1_preserved'])
        failed=set(selected['protection_ids'])
        def bad(rt,rows,deadline,expected):return [observed(r,r['id'] not in failed) for r in rows]
        with patch('luban.evolution._target',return_value=('tiny',parameter)),patch('luban.evolution._loss',return_value=1.),patch.object(runner.prior,'probes',side_effect=bad):
            rejected=runner.final_check(SimpleNamespace(),parent,selected,tensor_sha(parameter),1.,runner.time.perf_counter()+60)
        self.assertFalse(rejected['safety_passed']);self.assertEqual(set(rejected['forgotten_ids']),failed)
        with patch('luban.evolution._target',return_value=('tiny',parameter)),patch('luban.evolution._loss',return_value=1.051),patch.object(runner.prior,'probes',side_effect=probe):
            rejected=runner.final_check(SimpleNamespace(),parent,selected,tensor_sha(parameter),1.,runner.time.perf_counter()+60)
        self.assertFalse(rejected['quality_passed'])

    def test_once_preflight_mismatch_and_alternate_output_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();path=root/'protocol.json';output=root/'execution'
            plan={'protocol':runner.PROTOCOL,'sealed':True,'state':'sealed_waiting_explicit_MODEL_GO','output':str(output)}
            runner.write_new(path,plan)
            with self.assertRaises(ValueError):runner.execute(path,'0'*64)
            with patch.object(runner,'validate_bound',side_effect=ValueError('missing closed proof')):
                with self.assertRaises(ValueError):runner.execute(path,runner.sha(path))
            self.assertFalse(output.exists());self.assertFalse((root/'execution-started.json').exists())
            runner.write_new(root/'execution-started.json',{'CPU_consumed':True})
            for destination in (output,root/'other'):
                plan['output']=str(destination);path.write_text(json.dumps(plan))
                with patch.object(runner,'validate_bound'):
                    with self.assertRaises(FileExistsError):runner.execute(path,runner.sha(path))
                self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()
