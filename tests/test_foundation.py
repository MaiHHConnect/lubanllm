"""Stdlib admission with a session stub; no model or Torch load."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import contextmanager
import threading

from luban.foundation import install_foundation
from luban.retention import validate_core_retention


class SessionStub:
    def __init__(self, runtime, kind, arguments, records, *, run_id, scope):
        self.runtime = runtime
        self.row = {'id':run_id,'kind':kind,'scope':scope,'arguments':arguments,'records':records,'state':'queued'}
        runtime.memory.runs.append(self.row)

    def prepare_commit(self, report, pointers, artifacts, *, event_kind):
        self.row['state']='committing'
        self.row['commit']={'report':report,'pointers':[{'name':name,'before':None,'after':value} for name,value in pointers.items()],
            'artifacts':{name:hashlib.sha256((self.runtime.store/name).read_bytes()).hexdigest() for name in artifacts},
            'event_kind':event_kind}

    def finish_commit(self):
        for item in self.row['commit']['pointers']:
            (self.runtime.store/item['name']).write_text(json.dumps(item['after']))
        self.runtime.memory.events.append(copy.deepcopy(self.row['commit']))
        self.row['state']='committed'


class MemoryStub:
    def __init__(self):
        self.runs, self.events, self.lock = [], [], threading.RLock()

    @contextmanager
    def connect(self):
        def execute(sql, parameters):
            event = next((value for value in self.events if parameters == ('training:' + self.runs[0]['id'],)), None)
            return SimpleNamespace(fetchone=lambda: None if event is None else
                {'kind':event['event_kind'],'body':json.dumps(event['report'])})
        yield SimpleNamespace(execute=execute)


class FoundationTests(unittest.TestCase):
    def setUp(self):
        module = SimpleNamespace(TrainingSession=SessionStub, training_runs=lambda memory,identity:memory.runs)
        self.stub = patch.dict('sys.modules', {'luban.training_checkpoint':module,
            'luban.evolution':SimpleNamespace(_read_version=self.checked_version)})
        self.stub.start()
        self.model_checks = []

    def tearDown(self): self.stub.stop()

    def checked_version(self, runtime, version):
        self.model_checks.append(version)
        return json.loads((runtime.store/(version+'.json')).read_text()), None

    def fixture(self, root):
        model, store = root/'model', root/'store'
        model.mkdir(); store.mkdir(); (model/'luban-foundation').mkdir()
        version, identity, report = 'core-123456abcdef', 'b'*64, 'evolution-123456abcdef.json'
        groups={key:[] for key in ('new_training','new_validation','replay_training','replay_validation')}
        row={'id':'human-persona','prompt':'Who are you?','answer':'Luban0','source':'human','purpose':'training',
             'approved':True,'training_allowed':True,'provenance':json.dumps({'evolution_role':'new',
                'generation_check':{'type':'exact','expected':'Luban0'}})}
        groups['new_training']=[row]
        retention={'schema_version':1,'version':version,'identity':identity,'groups':groups,'regression_validation':[]}
        encoded=json.dumps(retention,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        retention['content_sha256']=hashlib.sha256(encoded).hexdigest()
        validate_core_retention(retention,identity,version)
        values={version+'.safetensors':b'no model: validated boundary stub only',
            version+'.retention.json':json.dumps(retention).encode(),
            report:json.dumps({'id':version.removeprefix('core-'),'promoted':True,'version':version,'identity':identity,
                'native_identity_verified':True,'publication_state':'published','gates':{'native':True,'retention':True},
                'steps':900,'learning_rate':1e-5,'distillation_weight':5}).encode()}
        values[version+'.json']=json.dumps({'version':version,'identity':identity,'report':report,
            'sha256':hashlib.sha256(values[version+'.safetensors']).hexdigest(),
            'retention_sha256':hashlib.sha256(values[version+'.retention.json']).hexdigest()}).encode()
        for name,data in values.items(): (model/'luban-foundation'/name).write_bytes(data)
        manifest = {'version':version, 'identity':'b'*64, 'quality_passed':True, 'report':report,
            'files':{name:hashlib.sha256(data).hexdigest() for name,data in values.items()}}
        (model/'luban-foundation.json').write_text(json.dumps(manifest))
        return SimpleNamespace(loaded_path=str(model),store=store,identity=identity,
            memory=MemoryStub(),research=False),manifest

    def rewrite(self,runtime,manifest,name,value):
        path=Path(runtime.loaded_path)/'luban-foundation'/name
        path.write_text(json.dumps(value))
        manifest['files'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (Path(runtime.loaded_path)/'luban-foundation.json').write_text(json.dumps(manifest))

    def test_absent_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(install_foundation(SimpleNamespace(loaded_path=t)))

    def test_research_original_does_not_install_or_read_release(self):
        self.assertIsNone(install_foundation(SimpleNamespace(research=True)))

    def test_verified_install_has_journal_pointer_and_no_new_training_claim(self):
        with tempfile.TemporaryDirectory() as t:
            runtime, manifest = self.fixture(Path(t))
            first=install_foundation(runtime)
            self.assertEqual(first, install_foundation(runtime))
            self.assertEqual(first['version'], manifest['version'])
            self.assertEqual(json.loads((runtime.store/'active-core.json').read_text()),first)
            self.assertEqual(len(runtime.memory.runs),1)
            self.assertEqual(len(runtime.memory.events),1)
            event=runtime.memory.events[0]
            self.assertEqual(event['artifacts'],manifest['files'])
            self.assertTrue(event['report']['foundation_install'])
            self.assertEqual(event['report']['training_updates_executed'],0)
            self.assertEqual(len(self.model_checks),2)

    def test_nested_distillation_uses_real_source_value_without_rewriting_report(self):
        with tempfile.TemporaryDirectory() as t:
            runtime, manifest = self.fixture(Path(t))
            report_name = manifest['report']
            path = Path(runtime.loaded_path) / 'luban-foundation' / report_name
            report = json.loads(path.read_text())
            report.pop('distillation_weight')
            report['distillation'] = {'weight': 5, 'teacher_version': 'core-000000000001'}
            self.rewrite(runtime, manifest, report_name, report)
            before = path.read_bytes()
            install_foundation(runtime)
            self.assertEqual(runtime.memory.runs[0]['arguments']['distillation_weight'], 5)
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn('distillation_weight', runtime.memory.events[0]['report'])
            self.assertEqual(runtime.memory.events[0]['report']['distillation'], report['distillation'])

    def test_distillation_missing_invalid_or_conflicting_values_reject_before_copy(self):
        variants = [{}, {'distillation_weight': True}, {'distillation': {'weight': False}},
            {'distillation': {'weight': float('nan')}}, {'distillation': {'weight': float('inf')}},
            {'distillation': {'weight': -1}}, {'distillation': {'weight': 101}},
            {'distillation': {'weight': '5'}}, {'distillation': {}}, {'distillation': []},
            {'distillation_weight': 5, 'distillation': {'weight': 4}}]
        for fields in variants:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as t:
                runtime, manifest = self.fixture(Path(t))
                name = manifest['report']
                report = json.loads((Path(runtime.loaded_path) / 'luban-foundation' / name).read_text())
                report.pop('distillation_weight')
                report.update(fields)
                self.rewrite(runtime, manifest, name, report)
                with self.assertRaisesRegex(ValueError, '蒸馏参数'):
                    install_foundation(runtime)
                self.assertEqual(list(runtime.store.iterdir()), [])

    def test_matching_legacy_and_nested_distillation_are_allowed(self):
        with tempfile.TemporaryDirectory() as t:
            runtime, manifest = self.fixture(Path(t))
            name = manifest['report']
            report = json.loads((Path(runtime.loaded_path) / 'luban-foundation' / name).read_text())
            report['distillation'] = {'weight': 5.0}
            self.rewrite(runtime, manifest, name, report)
            install_foundation(runtime)
            self.assertEqual(runtime.memory.runs[0]['arguments']['distillation_weight'], 5)

    def test_corrupt_member_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as t:
            runtime, manifest=self.fixture(Path(t))
            (Path(runtime.loaded_path)/'luban-foundation'/ (manifest['version']+'.safetensors')).write_bytes(b'bad')
            with self.assertRaises(ValueError):install_foundation(runtime)
            self.assertEqual(list(runtime.store.iterdir()), [])

    def test_conflicting_history_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            runtime, manifest=self.fixture(Path(t))
            path=runtime.store/(manifest['version']+'.json');path.write_bytes(b'previous')
            with self.assertRaises(ValueError):install_foundation(runtime)
            self.assertEqual(path.read_bytes(), b'previous')

    def test_false_quality_and_identity_rejected(self):
        for field,value in [('quality_passed',False),('identity','c'*64)]:
            with tempfile.TemporaryDirectory() as t:
                runtime,manifest=self.fixture(Path(t));manifest[field]=value
                (Path(runtime.loaded_path)/'luban-foundation.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):install_foundation(runtime)

    def test_metadata_version_identity_and_hash_not_just_manifest_sha(self):
        for field,value in (('version','core-aaaaaaaaaaaa'),('identity','c'*64),('sha256','a'*64),
                            ('retention_sha256','a'*64),('report','evolution-aaaaaaaaaaaa.json')):
            with tempfile.TemporaryDirectory() as t:
                runtime,manifest=self.fixture(Path(t));name=manifest['version']+'.json'
                data=json.loads((Path(runtime.loaded_path)/'luban-foundation'/name).read_text());data[field]=value
                self.rewrite(runtime,manifest,name,data)
                with self.subTest(field=field),self.assertRaises(ValueError):install_foundation(runtime)
                self.assertEqual(list(runtime.store.iterdir()),[])

    def test_retention_permission_revoked_rejected_even_with_new_manifest_sha(self):
        with tempfile.TemporaryDirectory() as t:
            runtime,manifest=self.fixture(Path(t));name=manifest['version']+'.retention.json'
            data=json.loads((Path(runtime.loaded_path)/'luban-foundation'/name).read_text())
            data['groups']['new_training'][0]['training_allowed']=False
            payload={key:value for key,value in data.items() if key!='content_sha256'}
            data['content_sha256']=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.rewrite(runtime,manifest,name,data)
            metadata_name=manifest['version']+'.json'
            metadata=json.loads((Path(runtime.loaded_path)/'luban-foundation'/metadata_name).read_text())
            metadata['retention_sha256']=manifest['files'][name]
            self.rewrite(runtime,manifest,metadata_name,metadata)
            with self.assertRaises(ValueError):install_foundation(runtime)
            self.assertEqual(list(runtime.store.iterdir()),[])

    def test_unrelated_committed_journal_cannot_claim_foundation(self):
        with tempfile.TemporaryDirectory() as t:
            runtime,manifest=self.fixture(Path(t));install_foundation(runtime)
            runtime.memory.runs[0]['commit']['report']['native_identity_verified']=False
            with self.assertRaises(ValueError):install_foundation(runtime)

    def test_missing_durable_event_cannot_reinstall_pointer(self):
        with tempfile.TemporaryDirectory() as t:
            runtime,manifest=self.fixture(Path(t));install_foundation(runtime)
            runtime.memory.events.clear()
            (runtime.store/'active-core.json').unlink()
            with self.assertRaisesRegex(ValueError,'持久采纳事件'):install_foundation(runtime)
            self.assertFalse((runtime.store/'active-core.json').exists())


if __name__=='__main__':unittest.main()
