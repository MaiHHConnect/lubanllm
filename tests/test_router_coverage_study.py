"""Private SQLite/artifact and fabricated CPU qualification tests; no model."""
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_router_coverage_study as runner
import verify_router_coverage_counts as selector
from test_router_coverage_counts import fixture_qualification
from luban.retention import provenance, save_core_retention


def accepted_fixture(root):
    """Fake tiny weight bytes and fabricated recovered tensor SHA; never a model proof."""
    root = Path(root);store = root / "modules/luban0-v1";store.mkdir(parents=True)
    identity = selector.descriptor()["base_identity"];run_id = 'abcdef123456';version = 'core-' + run_id
    source = json.loads((ROOT / 'research/core_identity_relation_teaching_draft.json').read_text())
    groups = {key: rows[:1] for key, rows in source['groups'].items()}
    retention = save_core_retention(SimpleNamespace(store=store, identity=identity), version, groups)
    weights = store / (version + '.safetensors');weights.write_bytes(b'CPU fixture only; not actual safetensors')
    metadata = {'identity': identity, 'version': version, 'sha256': runner.sha(weights),
                'retention_sha256': retention['file_sha256']}
    (store / (version + '.json')).write_text(json.dumps(metadata))
    pointer = {'version': version, 'identity': identity, 'sha256': metadata['sha256']}
    (store / 'active-core.json').write_text(json.dumps(pointer))
    (store / 'graph-verification.json').write_text('{"passed":true}')
    gates = dict.fromkeys(('fresh_development','known_regression','engineering','recovery','protection_snapshot'),True)
    report_name = 'coverage-acceptance-' + run_id + '.json'
    raw = {'outer_gates': gates, 'session_id': run_id,
           'recovery': {'passed': True, 'core_tensor_sha256_before': 'a' * 64,
                        'core_tensor_sha256_after':'a'*64,'core_version_after':version}}
    (store / report_name).write_text(json.dumps(raw))
    artifacts = {name: runner.sha(store / name) for name in
                 (version + '.safetensors', version + '.json', version + '.retention.json', report_name)}
    row = {'id':run_id,'kind':'evolution','scope':'research','product_accepted':True,
        'binding':{'identity':identity,'implementation_sha256':{'fixture_source.py':'f' * 64}},
        'commit':{'report':{'checkpoint':metadata},'artifacts':artifacts,
            'pointers':[{'name':'active-core.json','after':pointer}],
            'material_publication':{'identity':identity,'version':version,'groups':groups,
                'retention_regressions':[], 'source_rows':[{'id':'fabricated-source-fixture'}],
                'source_sha256':runner.hashlib.sha256(json.dumps([{'id':'fabricated-source-fixture'}],sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
                'acceptance':{'gates':gates,'report_path':report_name,'report_sha256':artifacts[report_name]}}}}
    app = root / 'app';app.mkdir();db = sqlite3.connect(app / 'memory.sqlite3')
    db.execute('CREATE TABLE training_runs (id TEXT,identity TEXT,kind TEXT,state TEXT,body TEXT,updated REAL)')
    db.execute('INSERT INTO training_runs VALUES (?,?,?,?,?,?)',(run_id,identity,'evolution','committed',json.dumps(row),1.))
    db.commit();db.close()
    return row


def update_receipt(root, row, state='committed'):
    with sqlite3.connect(Path(root) / 'app/memory.sqlite3') as db:
        db.execute('UPDATE training_runs SET body=?,state=?',(json.dumps(row),state))


def heldout_fixture():
    result = {}
    for split in runner.SPLITS[1:]:
        rows = [r for r in selector.descriptor()['pool'] if provenance(r)['split']==split]
        result[split] = [{'id':r['id'],'answer':r['answer'] if i%2==0 else 'wrong',
            'passed':i%2==0,'complete':True,'branch_enabled':False,'thinking':'off',
            'do_sample':False,'max_new_tokens':512,'raw_completion_token_ids':[123,999]} for i,r in enumerate(rows)]
    return result


class RouterCoverageStudyTests(unittest.TestCase):
    def test_default_static_without_site_packages_has_no_model_import_or_binding(self):
        command="import runpy,sys;sys.path.insert(0,'scripts');runpy.run_path('scripts/verify_router_coverage_study.py',run_name='__main__');assert not any(x in sys.modules for x in ['torch','transformers','luban.runtime','luban.teacher','luban.speech'])"
        result=subprocess.run([str(ROOT/'.venv/bin/python'),'-S','-c',command],cwd=ROOT,
                              env={'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout)['plan']
        self.assertEqual(plan['state'],'waiting_accepted_core');self.assertIsNone(plan['binding'])
        self.assertFalse(plan['execution_ready']);self.assertFalse(plan['model_go'])
        self.assertEqual(plan['split_counts'],dict(train=98,development=32,calibration=32,test=32))

    def test_real_sqlite_fixture_binding_readonly_and_clone_keeps_core_disables_router_branch(self):
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
            root=Path(home)/'production';accepted_fixture(root)
            before=runner.database_sha(root/'app/memory.sqlite3');binding=runner.accepted_binding(root)
            self.assertEqual(binding['core_version'],'core-abcdef123456')
            self.assertEqual(before,runner.database_sha(root/'app/memory.sqlite3'))
            store=root/'modules/luban0-v1'
            (store/'active-router.json').write_text('unrelated product router')
            (store/'active.json').write_text('unrelated product branch')
            clone=Path(home)/'clone';runner.clone_accepted(root,clone,binding)
            cloned=clone/'modules/luban0-v1'
            self.assertEqual((cloned/'active-core.json').read_bytes(),(store/'active-core.json').read_bytes())
            self.assertEqual(runner.accepted_binding(clone),binding)
            self.assertFalse((cloned/'active-router.json').exists());self.assertFalse((cloned/'active.json').exists())

    def test_missing_core_waits_and_research_unaccepted_pending_or_gate_failure_cannot_bind(self):
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
            self.assertIsNone(runner.accepted_binding(home))
        for mutation in ('unaccepted','pending','gate','checkpoint','pointer','retention_records','tensor','identity'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
                row=accepted_fixture(home)
                if mutation=='unaccepted':row['product_accepted']=False
                elif mutation=='gate':row['commit']['material_publication']['acceptance']['gates']['engineering']=False
                elif mutation=='checkpoint':row['commit']['report']['checkpoint']['sha256']='f'*64
                elif mutation=='pointer':row['commit']['pointers'][0]['after']['version']='core-ffffffffffff'
                elif mutation=='retention_records':row['commit']['material_publication']['groups']['new_validation']=[]
                elif mutation=='identity':row['binding']['identity']='other-base'
                elif mutation=='tensor':
                    store=Path(home)/'modules/luban0-v1';name=row['commit']['material_publication']['acceptance']['report_path']
                    raw=json.loads((store/name).read_text());raw['recovery']['core_tensor_sha256_before']=None
                    (store/name).write_text(json.dumps(raw))
                update_receipt(home,row,'running' if mutation=='pending' else 'committed')
                with self.assertRaises(ValueError):runner.accepted_binding(home)

    def test_corrupt_weight_and_symlink_refuse_binding(self):
        for link in (False,True):
            with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
                accepted_fixture(home);p=Path(home)/'modules/luban0-v1/core-abcdef123456.safetensors'
                if link:
                    actual=p.with_suffix('.other');p.rename(actual);p.symlink_to(actual)
                else:p.write_bytes(b'changed')
                with self.assertRaises(ValueError):runner.accepted_binding(home)

    def test_snapshot_covers_unselected_module_and_committed_wal_changes(self):
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
            row=accepted_fixture(home);store=Path(home)/'modules/luban0-v1'
            before=runner.module_snapshot(store)
            (store/'unselected-module.bin').write_bytes(b'new private fixture artifact')
            self.assertNotEqual(before,runner.module_snapshot(store))
            db_path=Path(home)/'app/memory.sqlite3'
            with sqlite3.connect(db_path) as db:
                db.execute('PRAGMA journal_mode=WAL')
                logical_before=runner.database_sha(db_path)
                db.execute('UPDATE training_runs SET updated=2');db.commit()
                self.assertNotEqual(logical_before,runner.database_sha(db_path))

    def test_output_namespace_is_bound_in_exact_approved_plan_without_creation(self):
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
            root=Path(home)/'production';accepted_fixture(root)
            first=root/'experiments/first';second=root/'experiments/second'
            a,b=runner.plan(root,first),runner.plan(root,second)
            self.assertNotEqual(selector.digest(a),selector.digest(b))
            self.assertEqual(a['binding'],b['binding'])
            self.assertFalse(first.exists());self.assertFalse(second.exists())
            self.assertFalse(a['sealed']);self.assertFalse(a['execution_ready'])

    def test_model_path_actual_files_and_manifest_not_only_declared_identity_are_bound(self):
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
            path=Path(home)/'model';path.mkdir()
            (path/'config.json').write_text('{"fixture":true}')
            (path/'model.safetensors').write_bytes(b'not actual model weights; CPU fixture')
            (path/'tokenizer.json').write_text('{"fixture":true}')
            hashes={name:runner.sha(path/name) for name in ('config.json','model.safetensors','tokenizer.json')}
            manifest={'Data':{'Files':[{'Path':name,'Sha256':value} for name,value in hashes.items()]}}
            (path/'upstream-file-manifest.json').write_text(json.dumps(manifest))
            identity=runner.hashlib.sha256(json.dumps(sorted((k,v) for k,v in hashes.items()
                if k=='config.json' or k.endswith('.safetensors'))).encode()).hexdigest()
            binding=runner.model_binding(path,identity)
            self.assertEqual(binding['path'],str(path));self.assertEqual(binding['files'],hashes)
            copied=Path(home)/'different-model';runner.shutil.copytree(path,copied)
            self.assertNotEqual(binding,runner.model_binding(copied,identity))
            (path/'tokenizer.json').write_text('changed tokenizer')
            with self.assertRaises(ValueError):runner.model_binding(path,identity)
            (path/'tokenizer.json').write_text('{"fixture":true}')
            (path/'model.safetensors').write_bytes(b'changed weights')
            with self.assertRaises(ValueError):runner.model_binding(path,identity)

    def test_model_wrong_identity_duplicate_or_traversal_manifest_refuses(self):
        with tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
            path=Path(home);(path/'config.json').write_text('{}');(path/'model.safetensors').write_bytes(b'fixture')
            files=[{'Path':name,'Sha256':runner.sha(path/name)} for name in ('config.json','model.safetensors')]
            identity=runner.hashlib.sha256(json.dumps(sorted((r['Path'],r['Sha256']) for r in files)).encode()).hexdigest()
            for change in ('wrong_identity','duplicate','traversal'):
                modified=copy.deepcopy(files)
                if change=='duplicate':modified.append(modified[0])
                if change=='traversal':modified[0]['Path']='../outside'
                (path/'upstream-file-manifest.json').write_text(json.dumps({'Data':{'Files':modified}}))
                with self.subTest(change=change),self.assertRaises(ValueError):
                    runner.model_binding(path,'f'*64 if change=='wrong_identity' else identity)

    def test_artifact_path_traversal_and_missing_source_evidence_are_rejected(self):
        for mutation in ('traversal','source_sha','source_missing','implementation_missing'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(dir=Path(tempfile.gettempdir()).resolve()) as home:
                row=accepted_fixture(home)
                if mutation=='traversal':row['commit']['artifacts']['../outside']='a'*64
                elif mutation=='source_sha':row['commit']['material_publication']['source_sha256']='wrong'
                elif mutation=='source_missing':row['commit']['material_publication'].pop('source_rows')
                else:row['binding']['implementation_sha256']={}
                update_receipt(home,row)
                with self.assertRaises(ValueError):runner.accepted_binding(home)

    def test_train_selection_nested_and_heldout_cannot_change_it(self):
        qualification=fixture_qualification();qualification['evidence_scope']='actual_generation'
        selection=selector.select_training(qualification);before=copy.deepcopy(selection)
        result=runner.fit_records(selector.descriptor(),selection,heldout_fixture(),[999])
        self.assertEqual(selection,before);self.assertEqual(result['state'],'ready_records_only')
        for count in selector.COUNTS:
            records=result['records'][str(count)]
            self.assertEqual(len(records['train']),count*2)
            for split in runner.SPLITS[1:]:self.assertEqual(len(records[split]),32)
        heldout=heldout_fixture()
        for answer in heldout['test']:answer.update(answer='wrong',passed=False)
        blocked=runner.fit_records(selector.descriptor(),selection,heldout,[999])
        self.assertEqual(blocked['state'],'waiting');self.assertEqual(selection,before)
        self.assertNotIn('records',blocked)

    def test_fixture_or_missing_coverage_never_becomes_fit_ready(self):
        selection=selector.select_training(fixture_qualification())
        self.assertEqual(runner.fit_records(selector.descriptor(),selection,heldout_fixture(),[999])['state'],'waiting')
        waiting=selector.select_training(None)
        self.assertEqual(runner.fit_records(selector.descriptor(),waiting,heldout_fixture(),[999])['state'],'waiting')

    def test_raw_gold_eos_budget_reference_and_order_guards(self):
        rows=[r for r in selector.descriptor()['pool'] if provenance(r)['split']=='test']
        original=heldout_fixture()['test']
        for mutation in ('gold','eos','budget','reference','branch','order','missing'):
            answers=copy.deepcopy(original)
            if mutation=='gold':answers[0]['answer']='wrong'
            elif mutation=='eos':answers[0]['raw_completion_token_ids'][-1]=123
            elif mutation=='budget':answers[0]['max_new_tokens']=128
            elif mutation=='reference':answers[0]['reference_used']=True
            elif mutation=='branch':answers[0]['branch_enabled']=True
            elif mutation=='order':answers.reverse()
            else:answers.pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):runner.label_split(rows,answers,[999])


if __name__=='__main__':unittest.main()
