"""CPU-only synthetic phase observations and real frozen material ancestry; no model quality claim."""
import ast,base64,copy,json,shutil,tempfile,unittest,zlib
from pathlib import Path
from unittest.mock import patch
import verify_goal_foundation as foundation
import verify_goal_app as app
import release_goal_foundation as publisher
from luban import goal_fresh as fresh
from luban import goal_sequence as common
import test_verify_app_portable_proof as old_app

ROOT=Path(__file__).resolve().parents[1]
PROTO=Path('/private/tmp/luban-goal-data-phase-preparation-20261011/cpu-prototype/protocol.json')
PHASE='evolution-goal-new-data-fixed450-v1'

def fixture(completed=7200):
    protocol=fresh.load_verified(PROTO)
    groups=protocol['_records']['groups'];old=protocol['_records']['protection'];new=groups['new_training']
    ids={r['id'] for r in new};known={r['id'] for r in new[:24]};replay={r['id']:r for r in groups['replay_training']}
    def row(sample,passed):
        return {'id':sample['id'],'answer':sample['answer'] if passed else '', 'passed':passed,'complete':passed,
            'branch_enabled':False,'reference_used':False,'raw_completion_token_ids':[1,2] if passed else [1]}
    def observe(step,confirmed,prior,reference):
        observation={'new':[row(r,r['id'] in confirmed) for r in new],
            'old13':[row(replay[i],True) for i in protocol['replay_generation_ids']],
            'old89':[row(r,True) for r in old], 'replay_training_loss':1.0}
        value={'step':step,'label':'data-phase-entry6750' if step==6750 else 'data-phase-complete450',
            'tensor_sha256':protocol['parent_final_tensor_sha256'] if step==6750 else 'f'*64,
            'branch_enabled':False,'reference_used':False,'validation_rows_accessed':0,
            'observations':[observation,copy.deepcopy(observation)],'single_observation_correct':[len(confirmed)]*2,
            'confirmed_ids':sorted(confirmed),'confirmed_count':len(confirmed),'pending_ids':sorted(ids-confirmed),'pending_count':len(ids-confirmed),
            'newly_confirmed_from_reference':sorted(confirmed-reference),'newly_confirmed_since_prior':sorted(confirmed-prior),
            'forgotten_from_reference':sorted(reference-confirmed),'forgotten_since_prior':sorted(prior-confirmed),
            'old13_both_passed':True,'old89_both_passed':True}
        if step>6750:common.classify_round(value,1.06)
        return value
    baseline=observe(6750,known,set(),set());stages=[];history=[]
    for step in range(7200,completed+1,450):
        stage=observe(step,ids if step==completed else known,known,known);stages.append(stage)
        first=stage['observations'][0];accepted=stage['round_result']=='satisfied'
        history.append({'step':step,'new_generation':first['new'],'replay_training_generation':first['old13'],
            'replay_loss':1.0,'accepted':accepted,'consecutive_passes':0,'same_weight_passes':2 if accepted else 0,
            'minimum_new_exposures':(step-6750)//2//56,'tensor_sha256':stage['tensor_sha256']})
    source=fresh.phase_sources(protocol);meta=fresh.metadata(protocol)
    report={'completed_steps':completed,'baseline_training_replay_loss':1.0,'goal_sequence_sources':source,
        'goal_sequence':{**fresh.report_metadata(protocol,completed),'parent_archive':fresh.phase_archive(protocol),
            'reference_baseline_before_training':baseline,'stage_observations':stages},
        'training_selection':{'checkpoints':list(range(7200,9451,450)),'selected_step':completed,'results':history,
            'replay_generation_ids':protocol['replay_generation_ids'],'required_consecutive_passes':0,'required_same_weight_passes':2}}
    plan={**fresh.phase_plan_fields(protocol),'protocol':PHASE,'goal_sequence':meta,'arguments':fresh.arguments(protocol),
        'protection':protocol['_records']['protection'], 'goal_sequence_protocol':{'path':str(PROTO),'sha256':foundation.sha(PROTO)}}
    return plan,report,protocol

class DataPhaseReleaseTests(unittest.TestCase):
    def test_only_new_six_complete_rounds_can_select(self):
        for step in range(7200,9451,450):
            plan,report,protocol=fixture(step)
            baseline,confirmed=foundation.fresh_observation_binding(plan,report,protocol)
            self.assertEqual((len(baseline),len(confirmed)),(24,56))
            self.assertEqual(len(report['goal_sequence']['stage_observations']),(step-6750)//450)
    def test_parent_archive_current_sources_budget_and_old_stage_injection_rejected(self):
        for change in ('archive','source','parent','budget','prior24','oldstage','oldhistory','entrytensor','scope','step','exposure'):
            plan,report,protocol=fixture()
            if change=='archive':report['goal_sequence']['parent_archive']['parent_history_sha256']='0'*64
            elif change=='source':report['goal_sequence_sources']['lineage_parent']['parent_result_sha256']='0'*64
            elif change=='parent':report['goal_sequence']['parent_steps']=5400
            elif change=='budget':report['goal_sequence']['total_budget']=6750
            elif change=='prior24':report['goal_sequence_sources']['prior_training_ids'].pop()
            elif change=='oldstage':report['goal_sequence']['stage_observations'].insert(0,protocol['_parent_report']['goal_sequence']['stage_observations'][0])
            elif change=='oldhistory':report['training_selection']['results'].insert(0,protocol['_parent_report']['training_selection']['results'][0])
            elif change=='entrytensor':report['goal_sequence']['reference_baseline_before_training']['tensor_sha256']=protocol['original_tensor_sha256']
            elif change=='scope':report['goal_sequence']['training_budget_scope']='ordinary budget9450'
            elif change=='step':report['completed_steps']=7199
            else:report['training_selection']['results'][0]['minimum_new_exposures']=150
            with self.subTest(change=change):self.assertRaises((ValueError,KeyError),foundation.fresh_observation_binding,plan,report,protocol)
    def test_raw_double_judge_eos_and_old89_protection_not_weakened(self):
        for change in ('old89','answer','raw','reference','double','forgotten'):
            plan,report,protocol=fixture();proof=report['goal_sequence']['stage_observations'][-1]
            if change=='old89':proof['observations'][1]['old89'].pop()
            elif change=='answer':proof['observations'][0]['new'][0]['answer']='假的循环答案'
            elif change=='raw':proof['observations'][0]['new'][0]['raw_completion_token_ids']=[]
            elif change=='reference':proof['observations'][1]['new'][0]['reference_used']=True
            elif change=='double':proof['observations'].pop()
            else:proof['forgotten_from_reference']=['fake']
            with self.subTest(change=change):self.assertRaises((ValueError,KeyError),foundation.fresh_observation_binding,plan,report,protocol)
    def test_original_fourteen_exact_and_twelve_app_cases_appended(self):
        plan,_,protocol=fixture();cases=app.phase_app_cases(plan,PROTO)
        files=protocol['original_formal_files']
        original=app.app_cases(app.read(files['new_material']['path']),app.read(files['old_protection']['path']),app.read(files['acceptance']['path']),fresh=True)
        self.assertEqual(cases[:14],original);self.assertEqual(len(cases),26)
        self.assertEqual([r['id'] for r in cases[14:]],[r['id'] for r in plan['acceptance']])
        self.assertTrue(any(r.get('user_default_temperature') for r in cases[:14]))
        loop=cases[13]['messages'][1]['content'];self.assertIn('14.',loop);self.assertIn('15.',loop)
        for name in ('app_cases','check_response','goal_content','repetition'):
            def node(path):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(node(ROOT/'scripts/verify_goal_app.py'),include_attributes=False),ast.dump(node(ROOT/'original/verify_goal_app.py'),include_attributes=False))
    def test_App_ancestor_current_material_or_missing_current_acceptance_rejected(self):
        for field in ('original_formal_protocol','original_formal_files','formal_files','acceptance'):
            plan,_,_=fixture()
            if field=='acceptance':plan[field].pop()
            elif field=='original_formal_protocol':plan[field]['sha256']='0'*64
            else:plan[field]['new_material']['sha256']='0'*64
            self.assertRaises(ValueError,app.phase_app_cases,plan,PROTO)
    def test_phase_packer18_lossless_separate_parent_history_and_failed_validation(self):
        fixture_app=old_app.PortableAppBindingTests();fixture_app.setUp();self.addCleanup(fixture_app.doCleanups);c=fixture_app.case
        real=fresh.load_verified(PROTO);formal=app.read(c.protocol)
        formal.update(original_formal_protocol=real['original_formal_protocol'],original_formal_files=real['original_formal_files'],files=real['files'])
        c.save(c.protocol,formal);c.plan.update(protocol=PHASE,goal_sequence_protocol={'path':str(c.protocol),'sha256':app.sha(c.protocol)})
        c.save(c.root/'plan.json',c.plan);c.candidate['plan_sha256']=app.sha(c.root/'plan.json');c.save(c.acceptance,c.candidate)
        c.report.update(source_protocol_sha256=app.sha(c.protocol),native_identity_sha256=app.sha(c.acceptance))
        proof=publisher.portable_fresh_proof(c.plan,c.candidate,c.store,c.acceptance,c.root/'plan.json')
        self.assertEqual((proof['policy'],proof['encoding'],len(proof['documents'])),('fresh-b45-data-phase-core8-v1','zlib-base64-utf8-v1',18))
        for name,path in [('parent_report',real['files']['parent_report']['path']),('parent_result',real['files']['parent_result']['path']),('original_protocol',real['original_formal_protocol']['path'])]:
            doc=proof['documents'][name];raw=zlib.decompress(base64.b64decode(doc['data']));self.assertEqual(raw,Path(path).read_bytes());self.assertEqual(doc['raw_sha256'],app.sha(path))
        self.assertNotEqual(proof['documents']['parent_report'],proof['documents']['parent_result'])
        c.report['fresh_source_proof']=proof;c.reseal()
        fixture_app.admission.stop(); fixture_app.admission=patch.object(foundation,'fresh_training_binding',return_value={'training_protocol_sha256':app.sha(c.protocol),'scope':'CPU packing-only isolated admission fixture'}); fixture_app.admission.start();self.addCleanup(fixture_app.admission.stop)
        self.assertEqual(c.binding()['source_protocol_sha256'],app.sha(c.protocol))
        broken=copy.deepcopy(proof);broken['documents']['parent_report']=broken['documents']['parent_result'];c.report['fresh_source_proof']=broken;c.reseal()
        self.assertRaisesRegex(ValueError,'便携证明不等于',c.binding)
    def test_actual_protocol38_snapshot_and_absolute_five_material_binding(self):
        checked=fresh.load_verified(PROTO)
        with tempfile.TemporaryDirectory(dir=ROOT,prefix="protocol38-cpu-") as directory:
            output=Path(directory);snapshot=output/"source-snapshot"
            for name,expected in checked["new_source_sha256"].items():
                paths=[Path('/Users/linhao/Downloads/lubanllm')/name,Path('/private/tmp/luban-goal-data-phase-preparation-20261011')/name]
                source=next(p for p in paths if p.is_file() and foundation.sha(p)==expected)
                target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            parent=foundation.read(checked["files"]["parent_plan"]["path"])
            plan={**parent,**fresh.phase_plan_fields(checked),"protocol":PHASE,"source_sha256":checked["new_source_sha256"],
                "arguments":fresh.arguments(checked),"goal_sequence":fresh.metadata(checked),"output":str(output),
                "goal_sequence_protocol":{"path":str(PROTO),"sha256":foundation.sha(PROTO)}}
            actual,args,meta=foundation.fresh_protocol_binding(plan,PROTO)
            self.assertEqual((len(actual['new_source_sha256']),len(actual['new_implementation_sha256'])),(38,22))
            self.assertEqual((meta['parent_steps'],args['steps']),(6750,9450))
            model=output/'tiny-CPU-model';members=model/'luban-foundation';members.mkdir(parents=True)
            report=members/'fixture.json';report.write_text(json.dumps({'goal_sequence':{}}))
            manifest={'report':'fixture.json','files':{'fixture.json':foundation.sha(report)}}
            with patch.object(foundation,'fresh_training_binding',return_value={'scope':'CPU material binding only; no admitted C/F'}):
                protocol,new,old,hashes=foundation.bound_materials(model,PROTO,manifest,output/'fixture-acceptance.json')
            self.assertEqual((len(new),len(old)),(80,89));self.assertIn('fresh_training_binding',hashes)
            altered=copy.deepcopy(plan);altered['materials']=checked['original_formal_files']['new_material']['path']
            self.assertRaisesRegex(ValueError,'Phase exact formal',foundation.fresh_protocol_binding,altered,PROTO)
            (snapshot/'src/luban/goal_topics.py').write_bytes(b'changed frozen judge')
            self.assertRaisesRegex(ValueError,'source snapshot changed',foundation.fresh_protocol_binding,plan,PROTO)

    def test_all_old_families_remain_registered(self):
        wanted={PHASE,*[f'evolution-goal-prospective-fixed450-v{i}' for i in (1,2,3,4)]}
        self.assertTrue(wanted<=foundation.FRESH_PROTOCOLS);self.assertTrue(wanted<=publisher.FRESH_PROTOCOLS)

if __name__=='__main__':unittest.main(verbosity=2)
