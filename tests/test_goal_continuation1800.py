"""CPU evidence-chain checks for a distinct closed1800 ->3150 episode."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import train_evolution_goal as goal
import verify_goal_foundation as fresh
import verify_goal_app as app
from luban import goal_sequence as sequence
from tests import test_goal_continuation_entry as entry_tests
save = entry_tests.save


class Closed1800Contracts(unittest.TestCase):
    def setUp(self):
        # Reuse the existing real four-file F/C fixture; extend its immutable ancestry.
        self.base = entry_tests.ContinuedFoundationProofTests('test_std_only_actual_F_C_plan_sequence_formal_chain')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        self.formal = self.base.protocol
        ancestor = goal.read(self.base.sequence_path)
        parent_dir = self.root / 'closed1800'
        parent_dir.mkdir()
        parent_source = dict(ancestor['new_source_sha256'])
        snapshots = {}
        for name in parent_source:
            path = parent_dir / 'source-snapshot' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('frozen v2 CPU source ' + name)
            parent_source[name] = goal.sha(path)
            snapshots[name] = {'path': str(path), 'sha256': goal.sha(path)}
        # Unchanged implementations use original files; only permitted revision members differ.
        changed = {'scripts/train_evolution_goal.py', 'src/luban/evolution.py', 'src/luban/training_checkpoint.py', 'src/luban/goal_sequence.py'}
        for name in parent_source.keys() - changed:
            original = ancestor['parent_source_snapshot'][name]
            parent_source[name] = original['sha256']
            snapshots[name] = original
        ancestor['new_source_sha256'] = parent_source
        ancestor['new_implementation_sha256'] = {name: parent_source['src/luban/' + name] for name in ancestor['new_implementation_sha256']}
        ancestor['checkpoints'] = list(range(64, 449, 32)) + [900, 1350, 1800]
        save(self.base.sequence_path, ancestor)
        old_plan = copy.deepcopy(self.base.plan)
        old_plan.update(source_sha256=parent_source,
            goal_sequence_protocol={'path': str(self.base.sequence_path), 'sha256': goal.sha(self.base.sequence_path)})
        old_meta = {**old_plan['goal_sequence'], 'protocol_sha256': goal.sha(self.base.sequence_path)}
        old_plan['goal_sequence'] = old_meta
        old_plan['arguments']['goal_sequence'] = old_meta
        parent_id = 'b78215ecd9ed'
        ancestor['new_run_id'] = parent_id
        save(self.base.sequence_path, ancestor)
        old_meta['protocol_sha256'] = goal.sha(self.base.sequence_path)
        old_plan['goal_sequence_protocol']['sha256'] = goal.sha(self.base.sequence_path)
        parent = copy.deepcopy(self.base.candidate['training'])
        parent.update(id=parent_id, version='core-' + parent_id, promoted=False, completed_steps=1800)
        parent.pop('gates')
        parent['goal_sequence'].update(old_meta, protocol=goal.CONTINUATION_PROTOCOL,
            new_steps_executed=1350, cumulative_steps=1800, selected_segment=4)
        training = old_plan['groups']['new_training']
        reference = set(parent['goal_sequence']['reference_baseline_after_parent_closed']['confirmed_ids'])
        prior = set(parent['goal_sequence']['entry450']['confirmed_ids'])
        def proof(step, count, prior, label='segment-boundary'):
            result = copy.deepcopy(parent['goal_sequence']['stage_observations'][0])
            ids = {sample['id'] for sample in training[:count]}
            for values in result['observations']:
                for row in values['new']:
                    row['passed'] = row['id'] in ids
            result.update(step=step, label=label, single_observation_correct=[count, count], confirmed_ids=sorted(ids),
                confirmed_count=count, pending_ids=sorted({sample['id'] for sample in training} - ids), pending_count=16-count,
                newly_confirmed_from_reference=sorted(ids-reference), newly_confirmed_since_prior=sorted(ids-prior),
                forgotten_from_reference=sorted(reference-ids), forgotten_since_prior=sorted(prior-ids),
                round_result='satisfied' if count == 16 else 'partial')
            return result
        stages = []
        for step, count in ((900, 5), (1350, 11), (1800, 15)):
            value = proof(step, count, prior)
            stages.append(value)
            prior.update(value['confirmed_ids'])
        parent['goal_sequence']['stage_observations'] = stages
        def selected_row(value):
            first = value['observations'][0]
            satisfied = value['confirmed_count'] == 16
            return {'step': value['step'], 'new_generation': first['new'], 'replay_training_generation': first['old13'],
                'replay_loss': first['replay_training_loss'], 'accepted': satisfied, 'consecutive_passes': 0,
                'same_weight_passes': 2 if satisfied else 0, 'minimum_new_exposures': (value['step']//2)//16,
                'tensor_sha256': value['tensor_sha256']}
        parent['training_selection'].update(selected_step=None, results=parent['training_selection']['results'][:13]+[selected_row(value) for value in stages])
        ancestor_parent_report = goal.read(ancestor['files']['parent_report']['path'])
        parent['goal_sequence_sources'] = fresh.sequence_sources(ancestor, ancestor_parent_report)
        old_plan['old_training_original'] = old_plan['groups']['replay_training']
        source = self.root/'source/modules/luban0-v1'
        source.mkdir(parents=True)
        model = self.root/'model-input'
        model.mkdir()
        old_plan.update(source_store=str(source),source_files={},model_root=str(model), model_config_sha256='a'*64,model_manifest_sha256='b'*64,
            old_training_original=old_plan['groups']['replay_training'],acceptance=old_plan['retention_regressions'])
        old_plan.update(output=str(parent_dir), home=str(parent_dir/'isolated-home'), storage=str(parent_dir/'isolated-storage'))
        self.files = dict(ancestor['files'])
        self.files['parent_plan'] = save(parent_dir/'plan.json', old_plan)
        baseline = goal.read(ancestor['files']['baseline_old_training']['path'])
        old_result = {'protocol':goal.CONTINUATION_PROTOCOL, 'plan_sha256':self.files['parent_plan']['sha256'], 'passed':False,
            'released':True, 'production_written':False, 'adopted':False, 'teacher_api_calls':0,
            'source_core':old_plan['starting_core'], 'baseline_old_training':baseline, 'training':parent}
        self.files['parent_result'] = save(parent_dir/'acceptance.json', old_result)
        self.files['parent_report'] = save(parent_dir/'durable-report.json', parent)
        self.files['parent_binding'] = save(parent_dir/'binding.json', {'implementation_sha256':ancestor['new_implementation_sha256'], 'records_sha256':fresh.digest(goal.read(self.files['records']['path']))})
        self.files['parent_checkpoint'] = save(parent_dir/'checkpoint.json', {'CPU full checkpoint placeholder':1800})
        self.files['parent_journal'] = save(parent_dir/'journal.json', {'id':parent_id, 'state':'rejected',
            'binding':goal.read(self.files['parent_binding']['path']), 'records':goal.read(self.files['records']['path']),
            'checkpoint':{'sha256':self.files['parent_checkpoint']['sha256']}})
        self.files['parent_sequence_protocol'] = old_plan['goal_sequence_protocol']
        new_source = {**parent_source, 'scripts/train_evolution_goal.py':'e'*64,
                      'src/luban/evolution.py':'e'*64, 'src/luban/goal_sequence.py':'f'*64}
        frozen = {**ancestor, 'protocol':goal.CONTINUATION_PROTOCOL_V3, 'parent_steps':1800, 'max_steps':3150, 'max_segments':7,
            'checkpoints':list(range(64,449,32))+[900,1350,1800,2250,2700,3150], 'parent_run_id':parent_id,
            'new_run_id':self.base.source[5:], 'parent_arguments':old_plan['arguments'], 'parent_sequence_metadata':old_plan['goal_sequence'], 'files':self.files,
            'seed':42,'batch_size':1,'gradient_accumulation':1,'policy_version':'goal-fixed450-same-weight-v2',
            'required_consecutive_passes':0,'required_same_weight_passes':2,'required_new_exposures':2,
            'reference_policy':'original-accepted-core','basis_policy':'exact-parent-basis',
            'validation_policy':'first-complete450-same-W-double-pass-once-then-stop',
            'parent_source_sha256':parent_source, 'new_source_sha256':new_source, 'parent_source_snapshot':snapshots,
            'parent_implementation_snapshot':{name:snapshots['src/luban/'+name] for name in ancestor['new_implementation_sha256']},
            'parent_implementation_sha256':ancestor['new_implementation_sha256'],
            'new_implementation_sha256':{name:new_source['src/luban/'+name] for name in ancestor['new_implementation_sha256']},
            'parent_binding_sha256':fresh.digest(goal.read(self.files['parent_binding']['path'])), 'parent_report_sha256':fresh.digest(parent),
            'inherited_observations_sha256':fresh.digest({key:parent['goal_sequence'][key] for key in ('reference_baseline_after_parent_closed','entry450','stage_observations')}),
            'parent_goal_sequence_sources_sha256':fresh.digest(parent['goal_sequence_sources']),
            'closure':{**ancestor['closure'], 'final_step':1800}}
        self.sequence_path = self.root/'v3-sequence.json'
        save(self.sequence_path, frozen)
        checksum = goal.sha(self.sequence_path)
        self.plan = copy.deepcopy(old_plan)
        meta = {'protocol_sha256':checksum,'parent_run_id':parent_id,'parent_steps':1800,'total_budget':3150,'new_steps_budget':1350}
        self.plan.update(protocol=goal.CONTINUATION_PROTOCOL_V3, arguments={**goal.ARGUMENTS,'steps':3150,'goal_sequence':meta},
            goal_sequence=meta, goal_sequence_protocol={'path':str(self.sequence_path),'sha256':checksum}, source_sha256=new_source,
            storage=self.base.plan['storage'],output=str(self.base.acceptance.parent),home=str(self.base.acceptance.parent/'home'))
        report = copy.deepcopy(parent)
        entry = proof(1800,15,prior,'entry1800_after_parent_closed')
        final = proof(2250,16,prior)
        report.update(id=self.base.source[5:], version=self.base.source, promoted=True, completed_steps=2250, steps=3150,
                      gates=self.base.candidate['training']['gates'])
        report['goal_sequence'].update(meta,protocol=goal.CONTINUATION_PROTOCOL_V3,new_steps_executed=450,cumulative_steps=2250,
            selected_segment=5,entry_parent=entry,stage_observations=stages+[final])
        report['goal_sequence_sources'] = fresh.sequence_sources(frozen,parent)
        report['training_selection'].update(checkpoints=frozen['checkpoints'], selected_step=2250,
            results=parent['training_selection']['results']+[selected_row(final)])
        self.report = report
        self.parent = old_plan
        self.frozen = {**frozen,'_protocol_path':str(self.sequence_path),'_protocol_sha256':checksum}
        self.base.plan = self.plan
        save(self.base.acceptance.parent/'plan.json',self.plan)
        name='evolution-'+self.base.source[5:]+'.json'
        save(self.base.store/name,report)
        candidate=self.base.candidate
        candidate.update(protocol=goal.CONTINUATION_PROTOCOL_V3, plan_sha256=goal.sha(self.base.acceptance.parent/'plan.json'),
            training=report, goal_sequence=report['goal_sequence'])
        candidate['artifacts'][name]=goal.sha(self.base.store/name)
        candidate['training_journal'].update(arguments=self.plan['arguments'],binding={'scope':'product','arguments':self.plan['arguments']})
        candidate['training_journal']['commit'].update(report=report,artifacts=candidate['artifacts'])
        candidate['training_event']['body']=report
        candidate['terminal_selected_training_confirmation']=goal.terminal_selected_confirmation(self.plan,report,candidate['rows'],candidate['candidate_tensor_sha256'])
        save(self.base.acceptance,candidate)
        self.base.report.update(report, version=self.base.version,id=self.base.version[5:],
            checkpoint=self.base.report['checkpoint'],retention=self.base.report['retention'],
            source_protocol_sha256=checksum,source_quality_sha256=candidate['artifacts'][name],native_identity_sha256=goal.sha(self.base.acceptance))
        self.base.reseal()

    def test_v3_actual_F_C_and_recursive_closed1800_lineage(self):
        result=fresh.continued_training_binding(self.base.acceptance,self.formal,self.base.report)
        self.assertEqual(result['goal_sequence']['parent_steps'],1800)
        self.assertEqual(result['goal_sequence']['new_steps_executed'],450)
        self.assertEqual(self.base.binding()['continued_training_binding'],result)

    def test_same_weight_pending_parent_and_later_full_round(self):
        goal.goal_observation_binding(self.plan,self.report,self.frozen)
        self.assertEqual(self.report['goal_sequence']['stage_observations'][2]['pending_count'],1)
        self.assertEqual(self.report['goal_sequence']['stage_observations'][3]['pending_count'],0)
        for field in ('tensor_sha256','confirmed_count','pending_count','observations'):
            bad=copy.deepcopy(self.report)
            bad['goal_sequence']['entry_parent'][field] = [] if field=='observations' else 'tampered'
            with self.subTest(field=field),self.assertRaises(ValueError):
                goal.goal_observation_binding(self.plan,bad,self.frozen)

    def test_inherited_real_stage_cannot_be_relabelled_or_rewritten(self):
        for changed in ({'step':1799},{'confirmed_count':16},{'round_result':'satisfied'}):
            bad=copy.deepcopy(self.report)
            bad['goal_sequence']['stage_observations'][2].update(changed)
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                goal.goal_observation_binding(self.plan,bad,self.frozen)

    def test_v3_prepare_and_publish_budget_distinct_from_parent1800(self):
        with patch.object(sequence,'load_verified',return_value=self.frozen),patch.object(goal,'source_hashes',return_value=self.frozen['new_source_sha256']):
            answer=goal.prepared_continuation(SimpleNamespace(goal_sequence_protocol=self.sequence_path,output=self.root/'prepared-v3'))
            plan=goal.read(answer['plan'])
            self.assertEqual(plan['arguments']['steps'],3150)
            self.assertEqual(plan['goal_sequence']['parent_steps'],1800)
            self.assertEqual(plan['groups'],self.parent['groups'])
            import release_goal_foundation as release
            self.assertEqual(release.release_contract(plan)[0]['steps'],3150)

    def test_closed_lineage_mutation_and_VALID_access_rejected(self):
        parent_result=Path(self.files['parent_result']['path'])
        original=goal.read(parent_result)
        bad=copy.deepcopy(original)
        bad['training']['validation_new_loss']=.1
        save(parent_result,bad)
        # Rebinding the hash cannot hide that heldout was actually accessed.
        seq=goal.read(self.sequence_path)
        seq['files']['parent_result']['sha256']=goal.sha(parent_result)
        save(self.sequence_path,seq)
        plan=copy.deepcopy(self.plan)
        plan['goal_sequence_protocol']['sha256']=goal.sha(self.sequence_path)
        with self.assertRaises(ValueError):
            fresh.closed_sequence_binding(plan,self.formal)

    def test_narrow_v3_boundaries_do_not_relax_ordinary_or_accept_early_stop(self):
        for step in (1800,2218,2668,3118,3600):
            report={**self.report,'completed_steps':step}
            with self.subTest(step=step),self.assertRaises(ValueError):
                goal.terminal_selected_confirmation(self.plan,report,self.base.candidate['rows'],'b'*64)

    def test_actual_OS_std_only_F_binding_forbids_torch_and_SQLite(self):
        report_path=self.root/'F-input.json'
        save(report_path,self.base.report)
        code="""import builtins,json,sys,sqlite3
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name=='torch' or name.startswith('torch.'):
  raise AssertionError('Torch/model import forbidden')
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sqlite3.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('SQLite forbidden'))
from pathlib import Path
from verify_goal_foundation import continued_training_binding
x=continued_training_binding(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()))
print(x['goal_sequence']['parent_steps'])
"""
        process=subprocess.run([sys.executable,'-c',code,str(self.base.acceptance),str(self.formal),str(report_path)],capture_output=True,text=True,timeout=20)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertEqual(process.stdout.strip(),'1800')


if __name__=='__main__':
    unittest.main()
