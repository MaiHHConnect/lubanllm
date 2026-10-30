import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch
import zipfile

from luban import model_packages as packages
from luban.memory import Memory
from luban.settings import Settings, private_json


def tensor(path, shapes):
    header, end = {}, 0
    for name, shape in shapes.items():
        count = 1
        for size in shape:
            count *= size
        stop = end + count * 4
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [end, stop]}
        end = stop
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q', len(data)) + data + bytes(end))


class ModelPackagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'production'
        self.source.mkdir()
        trusted = {}
        for key in ('base', 'asr'):
            directory = self.source / 'models' / key
            directory.mkdir(parents=True)
            private_json(directory / 'config.json', {'model_type': 'qwen3_5' if key == 'base' else 'qwen3_asr'})
            tensor(directory / 'model.safetensors', {'fixture.weight': [2, 2]})
            (directory / 'LICENSE').write_text('CPU test fixture, not model distribution')
            trusted[key] = {path.name: packages._record(path) for path in directory.iterdir() if not path.name.startswith('._')}
        self.trust_patch = patch.object(packages, 'TRUSTED', trusted)
        self.trust_patch.start()
        self.addCleanup(self.trust_patch.stop)
        self.settings = Settings(self.root / 'settings')
        self.settings.update({'storage_root': str(self.source), 'model_path': str(self.source / 'models/base'),
                              'speech_model_path': str(self.source / 'models/asr'), 'native_inference_precision': 'bf16'})
        self.memory = Memory(self.source)
        with self.memory.connect() as db:
            db.execute('INSERT INTO conversations VALUES (?,?,?,?)', ('secret-chat', 'DO NOT EXPORT', '[]', 1))
            db.execute('INSERT INTO teacher_materials VALUES (?,?,?,?,?,?,?,?,?,?)', ('raw', 'secret', 'raw secret answer', '[]', 1, '{}', 'pending', '', 'training', None))
        self.store = self.source / 'modules/luban0-v1'
        self.store.mkdir(parents=True)
        self.runtime = SimpleNamespace(lock=threading.RLock(), state='ready', training_session=None,
            identity=packages.TRUSTED_IDENTITY, settings=self.settings, memory=self.memory, store=self.store)
        self.target = self.root / 'model.zip'

    def export(self):
        return packages.export_package(self.runtime, self.target)

    def test_mlx_export_without_configuration_does_not_read_weights_or_create_output(self):
        self.runtime.native_inference_precision = "mlx-q4"
        with patch.object(packages, "_record", side_effect=AssertionError("must not scan weights")):
            with self.assertRaisesRegex(ValueError, "MLX模型路径尚未配置"):
                self.export()
        self.assertFalse(self.target.exists())

    def test_committed_mlx_journal_cannot_be_exported_as_a_legacy_package(self):
        from luban.training_checkpoint import _schema
        body = {"id": "012345abcdef", "kind": "learning", "scope": "product",
                "arguments": {"protocol": "native-mlx-expert-preservation-v1"}}
        with self.memory.connect() as db:
            _schema(db)
            db.execute("INSERT INTO training_runs VALUES (?,?,?,?,?,?)", (body["id"], packages.TRUSTED_IDENTITY,
                       "learning", "committed", json.dumps(body), 1))
        with patch.object(packages, "_record", side_effect=AssertionError("must not scan weights")):
            with self.assertRaisesRegex(ValueError, "MLX采纳任务必须随完整MLX模型包导出"):
                self.export()
        self.assertFalse(self.target.exists())

    def stage(self):
        self.export()
        return packages.stage_import(self.target, self.root / 'imports')

    def rewrite(self, mutate):
        with zipfile.ZipFile(self.target) as archive:
            items = [(info, archive.read(info)) for info in archive.infolist()]
        changed = self.root / 'changed.zip'
        with zipfile.ZipFile(changed, 'w') as archive:
            mutate(archive, items)
        return changed

    def add_router_receipt(self, *, scope='product', accepted=False):
        from luban.training_checkpoint import _schema
        identifier = '123456789abc'
        version = 'router-' + identifier
        weights = self.store / (version + '.safetensors')
        tensor(weights, {'projection.weight': [3, 2048], 'projection.bias': [3]})
        meta = {'id': version, 'identity': packages.TRUSTED_IDENTITY, 'core_version': None,
                'sha256': packages._sha(weights), 'calibrated': True, 'gates_expert': False,
                'supported_actions': ['local_core', 'teacher']}
        private_json(self.store / (version + '.json'), meta)
        active = {'id': version, 'identity': packages.TRUSTED_IDENTITY, 'core_version': None,
                  'sha256': packages._sha(weights), 'metadata_sha256': packages._sha(self.store / (version + '.json'))}
        private_json(self.store / 'active-router.json', active)
        row = {'id': 'sample', 'prompt': 'CPU fixture', 'answer': 'CPU fixture reference', 'source': 'human',
               'purpose': 'training', 'approved': True, 'training_allowed': True, 'provenance': {'messages': [{'role': 'user', 'content': 'CPU fixture'}]}}
        records = {'splits': {'train': [row], 'validation': [], 'calibration': [], 'heldout': []}}
        commit = {'report': meta, 'pointers': [{'name': 'active-router.json', 'before': None, 'after': active}],
                  'artifacts': {name: packages._sha(self.store / name) for name in (version + '.json', version + '.safetensors')}, 'event_kind': 'router'}
        body = {'id': identifier, 'scope': scope, 'product_accepted': accepted, 'kind': 'router',
                'binding': {'identity': packages.TRUSTED_IDENTITY, 'records_sha256': hashlib.sha256(packages._encoded(records)).hexdigest()},
                'records': records, 'started': 1, 'commit': commit, 'owner': {'pid': 999999}, 'checkpoint': '/private/optimizer.pt'}
        with self.memory.connect() as db:
            _schema(db)
            db.execute('INSERT INTO training_runs VALUES (?,?,?,?,?,?)', (identifier, packages.TRUSTED_IDENTITY, 'router', 'committed', packages._encoded(body).decode(), 1))
            db.execute('INSERT INTO events VALUES (?,?,?,?)', ('training:' + identifier, 'router', packages._encoded(meta).decode(), 1))
        return identifier

    def test_roundtrip_excludes_private_data_and_preserves_old_files(self):
        before = packages._sha(self.memory.path)
        exported = self.export()
        inspect = packages.inspect_package(self.target)
        self.assertFalse(inspect['content_verified'])
        staged = packages.stage_import(self.target, self.root / 'imports')
        self.assertEqual(staged['package_id'], exported['package_id'])
        self.assertEqual(staged['counts'], {'router': 0, 'learning': 0, 'evolution': 0})
        self.assertEqual(packages._sha(self.memory.path), before)
        with zipfile.ZipFile(self.target) as archive:
            self.assertNotIn('memory.sqlite3', ' '.join(archive.namelist()))
            raw = archive.read('accepted.json').decode()
            self.assertNotIn('secret', raw)
            for info in archive.infolist():
                if info.filename.endswith('.safetensors'):
                    self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
        with Memory(staged['root']).connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM conversations').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM teacher_materials').fetchone()[0], 0)
        with self.assertRaises(ValueError):
            self.export()

    def test_receipts_survive_import_without_new_count_or_optimizer(self):
        identifier = self.add_router_receipt()
        exported = self.export()
        first = packages.stage_import(self.target, self.root / 'imports')
        second = packages.stage_import(self.target, self.root / 'imports')
        self.assertNotEqual(first['root'], second['root'])
        self.assertEqual(first['package_id'], second['package_id'])
        self.assertEqual(first['counts']['router'], 1)
        with Memory(first['root']).connect() as db:
            rows = db.execute('SELECT id,body FROM training_runs').fetchall()
            self.assertEqual([row['id'] for row in rows], [identifier])
            self.assertNotIn('optimizer', rows[0]['body'])
            self.assertNotIn('owner', json.loads(rows[0]['body']))
            self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 1)
        self.assertEqual(exported['package_id'], first['package_id'])

    def test_factory_baseline_does_not_count_router_or_drop_samples(self):
        identifier = self.add_router_receipt()
        payload = packages._journal(self.runtime)
        sample = payload['runs'][0]['records']['splits']['train'][0]
        payload.update(factory_runs=[identifier], samples=[sample])
        packages._validate_journal(payload)
        destination = self.root / 'factory'
        destination.mkdir()
        self.assertEqual(packages._install_journal(destination, payload), dict(router=0, learning=0, evolution=0))
        with Memory(destination).connect() as db:
            row = dict(db.execute('SELECT * FROM samples').fetchone())
            self.assertEqual(row['prompt'], sample['prompt'])
            self.assertEqual(row['answer'], sample['answer'])
            self.assertEqual(json.loads(row['provenance']), sample['provenance'])
            event = db.execute('SELECT kind,body FROM events').fetchone()
            self.assertEqual(event['kind'], 'factory:router')
            self.assertEqual(json.loads(event['body']), payload['runs'][0]['commit']['report'])
        runtime = SimpleNamespace(memory=Memory(destination), store=self.store, settings=self.settings)
        exported = packages._journal(runtime)
        self.assertEqual(exported['factory_runs'], [identifier])
        self.assertEqual(exported['runs'], payload['runs'])
        self.assertEqual(exported['samples'][0]['answer'], sample['answer'])
        migrated = self.root / 'migrated-factory'
        migrated.mkdir()
        self.assertEqual(packages._install_journal(migrated, exported), dict(router=0, learning=0, evolution=0))

    def test_unaccepted_research_active_weights_rejected(self):
        self.add_router_receipt(scope='research', accepted=False)
        with self.assertRaisesRegex(ValueError, '正式产品采纳'):
            self.export()
        self.assertFalse(self.target.exists())

    def test_pending_job_and_corrupt_receipt_rejected(self):
        identifier = self.add_router_receipt()
        with self.memory.connect() as db:
            db.execute("UPDATE training_runs SET state='paused' WHERE id=?", (identifier,))
        with self.assertRaisesRegex(ValueError, '未完成训练'):
            self.export()
        with self.memory.connect() as db:
            db.execute("UPDATE training_runs SET state='committed' WHERE id=?", (identifier,))
            db.execute('DELETE FROM events')
        with self.assertRaisesRegex(ValueError, '持久事件'):
            self.export()

    def test_path_alias_symlink_duplicate_and_compressed_weight_rejected(self):
        self.export()
        for bad in ('../escape', '/escape', 'models\\base\\config.json', 'models/base/CONFIG.json', 'models/base/ｃonfig.json', 'models/base/NUL', 'models/base/./config.json'):
            def add(archive, items):
                for info, content in items:
                    archive.writestr(info, content)
                archive.writestr(bad, b'x')
            changed = self.rewrite(add)
            with self.assertRaises((ValueError, KeyError)):
                packages.inspect_package(changed)
        def linked(archive, items):
            for info, content in items:
                if info.filename == 'accepted.json':
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, content)
        with self.assertRaises(ValueError):
            packages.inspect_package(self.rewrite(linked))
        def compressed(archive, items):
            for info, content in items:
                if info.filename.endswith('.safetensors'):
                    info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content)
        with self.assertRaises(ValueError):
            packages.inspect_package(self.rewrite(compressed))
        def duplicate(archive, items):
            for info, content in items:
                archive.writestr(info, content)
            archive.writestr('accepted.json', b'{}')
        # assertWarns scans every imported module and triggers optional lazy
        # transformers imports; capture the ZIP warning without that scan.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', UserWarning)
            changed = self.rewrite(duplicate)
        self.assertTrue(any(issubclass(item.category, UserWarning) and
                            'Duplicate name' in str(item.message) for item in caught))
        with self.assertRaises(ValueError):
            packages.inspect_package(changed)

    def test_hash_corruption_and_bomb_do_not_touch_production(self):
        self.export()
        before = packages._sha(self.memory.path)
        def corrupt(archive, items):
            for info, content in items:
                if info.filename.endswith('.safetensors'):
                    content = content[:-1] + b'x'
                archive.writestr(info, content)
        with self.assertRaisesRegex(ValueError, '校验失败'):
            packages.stage_import(self.rewrite(corrupt), self.root / 'imports')
        self.assertEqual(packages._sha(self.memory.path), before)
        def bomb(archive, items):
            for info, content in items:
                if info.filename == 'accepted.json':
                    info.compress_type = zipfile.ZIP_DEFLATED
                    content = b'0' * (2 * 1024 * 1024)
                archive.writestr(info, content)
        with self.assertRaisesRegex(ValueError, '压缩比'):
            packages.inspect_package(self.rewrite(bomb))

    def test_prepared_activation_rolls_back_and_success_requires_token(self):
        staged = self.stage()
        old = {key: self.settings.snapshot()[key] for key in packages._PATH_KEYS}
        token = packages.prepare_activation(self.settings, staged)
        self.assertEqual(self.settings.snapshot()['storage_root'], staged['root'])
        with self.assertRaises(ValueError):
            packages.finish_activation(self.settings, 'wrong')
        restarted = Settings(self.settings.home)
        self.assertTrue(packages.recover_activation(restarted)['recovered'])
        self.assertEqual({key: restarted.snapshot()[key] for key in packages._PATH_KEYS}, old)
        token = packages.prepare_activation(restarted, staged)
        result = packages.finish_activation(restarted, token)
        self.assertTrue(result['committed'])
        self.assertFalse(packages.recover_activation(restarted)['recovered'])
        with self.assertRaises(ValueError):
            packages.cleanup_retired_package(restarted, token)
        self.assertTrue(self.memory.path.exists())

    def test_activation_refuses_tampered_database_or_foreign_settings(self):
        staged = self.stage()
        database = Path(staged['root']) / 'app/memory.sqlite3'
        database.write_bytes(database.read_bytes() + b'tamper')
        with self.assertRaisesRegex(ValueError, '账本'):
            packages.prepare_activation(self.settings, staged)
        self.assertEqual(self.settings.snapshot()['storage_root'], str(self.source))

    def test_reset_unpublished_and_wrong_phrase_make_no_network_or_files(self):
        parent = self.root / 'not-created'
        with patch.object(packages.urllib.request, 'build_opener') as network:
            for phrase in ('wrong', packages.RESET_PHRASE):
                with self.assertRaises(ValueError):
                    packages.stage_reset(phrase, parent)
            network.assert_not_called()
        self.assertFalse(parent.exists())
        self.assertEqual(packages.FOUNDATION_URL, 'https://ai.maihh.net/models/luban0/base.zip')

    def test_disk_failure_never_replaces_existing_model(self):
        self.export()
        before = self.settings.snapshot()
        with patch.object(packages.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, '磁盘空间不足'):
                packages.stage_import(self.target, self.root / 'imports')
        self.assertEqual(before, self.settings.snapshot())
        self.assertTrue((self.source / 'models/base/model.safetensors').exists())

    def test_zip64_metadata_supported_and_central_directory_allocation_bounded(self):
        self.export()
        data = self.target.read_bytes()
        position = data.rfind(b'PK\x05\x06')
        end = struct.unpack('<4s4H2LH', data[position:position + 22])
        header = struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, end[4], end[4], end[5], end[6])
        locator = struct.pack('<4sLQL', b'PK\x06\x07', 0, position, 1)
        path = self.root / 'zip64.zip'
        path.write_bytes(data[:position] + header + locator + data[position:])
        self.assertEqual(packages.inspect_package(path)['package_id'], packages.inspect_package(self.target)['package_id'])
        huge = struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, end[4], end[4], 1024**3, end[6])
        path.write_bytes(data[:position] + huge + locator + data[position:])
        with patch.object(packages.zipfile, 'ZipFile') as never_allocate:
            with self.assertRaisesRegex(ValueError, 'ZIP64'):
                packages.inspect_package(path)
            never_allocate.assert_not_called()

    def test_real_process_kill_after_settings_switch_rolls_back_on_restart(self):
        staged = self.stage()
        binding = self.root / 'child-fixture.json'
        binding.write_text(json.dumps({'trusted': packages.TRUSTED, 'staged': staged}))
        code = '''import json,os,sys
from pathlib import Path
from luban import model_packages as p
from luban.settings import Settings
fixture=json.loads(Path(sys.argv[1]).read_text())
p.TRUSTED=fixture['trusted']
p.prepare_activation(Settings(sys.argv[2]),fixture['staged'])
os.kill(os.getpid(),9)
'''
        child = subprocess.run([sys.executable, '-c', code, str(binding), str(self.settings.home)], capture_output=True, text=True)
        self.assertEqual(child.returncode, -9, child.stderr)
        restarted = Settings(self.settings.home)
        self.assertEqual(restarted.snapshot()['storage_root'], staged['root'])
        self.assertTrue(packages.recover_activation(restarted)['recovered'])
        self.assertEqual(restarted.snapshot()['storage_root'], str(self.source))
        self.assertTrue((self.source / 'models/base/model.safetensors').exists())

    def test_reference_data_secrets_and_binding_tamper_fail_closed(self):
        self.add_router_receipt()
        payload = packages._journal(self.runtime)
        bad = copy.deepcopy(payload)
        bad['runs'][0]['records']['splits']['train'][0]['provenance']['api_key'] = 'secret'
        bad['runs'][0]['binding']['records_sha256'] = hashlib.sha256(packages._encoded(bad['runs'][0]['records'])).hexdigest()
        with self.assertRaisesRegex(ValueError, '秘密'):
            packages._validate_journal(bad)
        bad = copy.deepcopy(payload)
        bad['runs'][0]['records']['splits']['train'][0]['answer'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'SHA'):
            packages._validate_journal(bad)

    def test_product_partial450_journal_roundtrip_keeps_proofs_arguments_and_current_coverage(self):
        from test_partial_evolution import PartialEvolutionTests
        from luban.evolution import consolidate
        from luban.automatic import knowledge_keys
        from luban.training_checkpoint import training_runs
        import shutil
        case = PartialEvolutionTests()
        self.addCleanup(case.doCleanups)
        _, runtime, _, args = case.product_fixture()
        report = consolidate(runtime, **args)
        runtime.memory.path = runtime.memory.path.resolve()
        runtime.store = runtime.store.resolve()
        with patch.object(packages, 'TRUSTED_IDENTITY', runtime.identity):
            payload = packages._journal(runtime)
            run = payload['runs'][0]
            self.assertEqual(run['arguments'], args)
            packages._validate_product_partial(run, runtime.store)
            for mutation in ('steps', 'observations', 'arguments'):
                bad = copy.deepcopy(payload)
                current = bad['runs'][0]
                if mutation == 'steps':
                    current['commit']['report']['completed_steps'] = 449
                elif mutation == 'observations':
                    current['commit']['report']['partial_candidate_second'].pop()
                else:
                    current.pop('arguments')
                with self.assertRaises((ValueError, KeyError)):
                    packages._validate_journal(bad)
            root = self.root / 'partial-restored'
            packages._install_journal(root, payload)
            store = root / 'modules/luban0-v1'
            shutil.copytree(runtime.store, store)
            imported = SimpleNamespace(memory=Memory(root), store=store, identity=runtime.identity,
                active_core=report['version'], branch=SimpleNamespace(expert_id=None))
            restored = training_runs(imported.memory, imported.identity)
            self.assertEqual(restored[0]['arguments'], args)
            historical, covered, error = knowledge_keys(imported, restored)
            self.assertIsNone(error)
            self.assertTrue(historical and covered)
            path = store / run['commit']['report']['checkpoint']['report']
            changed = json.loads(path.read_text())
            changed['completed_steps'] = 449
            private_json(path, changed)
            with self.assertRaisesRegex(ValueError, '实际报告'):
                packages._validate_product_partial(run, store)

    def test_module_graph_limit_precedes_allocation_and_nan_rejected(self):
        filename = 'router-123456789abc.safetensors'
        path = self.store / filename
        tensor(path, {'projection.weight': [3, 10000], 'projection.bias': [3]})
        files = {'modules/luban0-v1/' + filename: packages._record(path)}
        with patch('safetensors.torch.load_file') as no_allocation:
            with self.assertRaisesRegex(ValueError, '快脑尺寸'):
                packages._validate_modules(self.source, {'runs': []}, files)
            no_allocation.assert_not_called()
        tensor(path, {'projection.weight': [3, 2048], 'projection.bias': [3]})
        path.write_bytes(path.read_bytes()[:-4] + struct.pack('<f', float('nan')))
        files = {'modules/luban0-v1/' + filename: packages._record(path)}
        with self.assertRaisesRegex(ValueError, '非有限'):
            packages._validate_modules(self.source, {'runs': []}, files)

    def test_material_publication_archives_preserved_without_reapplying_commit(self):
        from test_core_publication import fixture_plan
        from luban.training_checkpoint import reconcile_commit, training_runs
        plan = json.loads(json.dumps(fixture_plan()).replace('fixture-base', packages.TRUSTED_IDENTITY))
        plan['source_sha256'] = hashlib.sha256(packages._encoded(plan['source_rows'])).hexdigest()
        records = {'groups': plan['groups'], 'protection': [], 'retention_regressions': plan['retention_regressions']}
        report = {'version': plan['version'], 'promoted': True}
        commit = {'report': report, 'pointers': [{'name': 'active-core.json', 'before': None,
                  'after': {'version': plan['version'], 'identity': packages.TRUSTED_IDENTITY}}],
                  'artifacts': {plan['acceptance']['report_path']: plan['acceptance']['report_sha256']},
                  'event_kind': 'evolution', 'material_publication': plan}
        run = {'id': plan['run_id'], 'kind': 'evolution', 'scope': 'research', 'product_accepted': True,
               'state': 'committed', 'started': 1, 'records': records,
               'binding': {'identity': packages.TRUSTED_IDENTITY, 'records_sha256': hashlib.sha256(packages._encoded(records)).hexdigest()},
               'commit': commit}
        sources = {row['id']: row for row in plan['source_rows']}
        payload = {'identity': packages.TRUSTED_IDENTITY, 'runs': [run], 'samples': [row for rows in plan['groups'].values() for row in rows],
                   'material_archives': [{'version': plan['version'], 'sample_id': identifier,
                       'original_sha256': hashlib.sha256(packages._encoded(sources[identifier])).hexdigest(),
                       'reason': 'fixed coverage revision; gold/source/provenance preserved; old validation retained readonly'}
                       for identifier in plan['superseded_ids']]}
        packages._validate_journal(payload)
        self.assertEqual(packages._name('modules/luban0-v1/' + plan['acceptance']['report_path']), 'modules/luban0-v1/' + plan['acceptance']['report_path'])
        root = self.root / 'restored-ledger'
        packages._install_journal(root, payload)
        memory = Memory(root)
        restored = training_runs(memory, packages.TRUSTED_IDENTITY)[0]
        self.assertEqual(restored['commit']['material_publication'], plan)
        with patch('luban.core_publication.apply_material_publication') as no_reapply:
            reconcile_commit(memory, root / 'modules/luban0-v1', restored)
            no_reapply.assert_not_called()
        with memory.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM core_material_archives').fetchone()[0], 12)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM events').fetchone()[0], 1)
        changed = copy.deepcopy(payload)
        changed['material_archives'][0]['original_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, '归档'):
            packages._validate_journal(changed)

    def test_explicit_local_model_alias_allowed_but_linked_members_rejected(self):
        alias = self.source / 'base-alias'
        alias.symlink_to(self.source / 'models/base', target_is_directory=True)
        self.settings.update({'model_path': str(alias)})
        self.export()
        self.assertTrue(self.target.is_file())
        self.target.unlink()
        license_path = self.source / 'models/base/LICENSE'
        other = self.root / 'license'
        license_path.rename(other)
        license_path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            self.export()

    def test_unadopted_verified_external_reference_does_not_enter_model_zip(self):
        from luban.external_brain import ExternalBrain
        self.memory.model_identity = packages.TRUSTED_IDENTITY
        context = [{'role': 'user', 'content': '外脑尚未写入权重的问题'}]
        sample_id = self.memory.add_sample(context[0]['content'], '已人工确认但未采纳的参考', approved=True,
            training_allowed=True, provenance={'base_identity': packages.TRUSTED_IDENTITY, 'messages': context})
        brain = ExternalBrain(self.memory)
        brain.register_reference(sample_id, context, packages.TRUSTED_IDENTITY,
            {'core_version': None, 'core_sha256': 'b' * 64, 'expert_id': None, 'expert_sha256': None})
        self.export()
        with zipfile.ZipFile(self.target) as archive:
            accepted = json.loads(archive.read('accepted.json'))
            self.assertEqual(accepted['samples'], [])
            self.assertNotIn('external_brain', ' '.join(archive.namelist()))

    def test_explicit_private_home_platform_alias_allowed_but_journal_link_rejected(self):
        # On macOS TemporaryDirectory returns /var/... while resolve gives
        # /private/var/...; do not paper over this by changing global TMPDIR.
        platform_home = Path(self.temp.name) / 'settings'
        settings = Settings(platform_home)
        self.assertFalse(packages.recover_activation(settings)['recovered'])
        staged = self.stage()
        token = packages.prepare_activation(settings, staged)
        self.assertTrue(packages.recover_activation(settings)['recovered'])
        canonical = self.settings.home / 'package-activation.json'
        existing = self.root / 'unrelated.json'
        canonical.rename(existing)
        canonical.symlink_to(existing)
        for operation in (lambda: packages.recover_activation(settings),
                          lambda: packages.prepare_activation(settings, staged),
                          lambda: packages.finish_activation(settings, token),
                          lambda: packages.cleanup_retired_package(settings, token)):
            with self.assertRaisesRegex(ValueError, '符号链接'):
                operation()


if __name__ == '__main__':
    unittest.main()
