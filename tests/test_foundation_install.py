"""First-install failures cannot activate partial models or overwrite learned state."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from luban import foundation_install as install
from luban.settings import Settings, private_json


class FoundationInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = Settings(self.root / 'home')
        self.assets = []
        self.archives = {}
        for kind in ('mlx', 'speech', 'metadata', 'expert'):
            content = {'config.json': b'{}'}
            if kind in ('mlx', 'speech'): content['model.safetensors'] = b'fixture-weights'
            if kind == 'expert': content = {'modules/luban0-v1/active.json': b'{}', 'runtime/mlx/expert-proof.json': b'{}'}
            path = self.root / (kind + '.zip')
            with zipfile.ZipFile(path, 'w') as archive:
                for name, data in content.items(): archive.writestr(kind + '/' + name, data)
            self.archives[kind] = path
            self.assets.append({'kind': kind, 'name': path.name, 'prefix': kind, 'files': list(content),
                'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'expanded_bytes': sum(map(len, content.values()))})
        for method in ('validate', 'install_journal'):
            mocked = patch.object(install.factory_expert, method, return_value={})
            mocked.start(); self.addCleanup(mocked.stop)
        manifest = patch.dict(install.factory_expert.MANIFEST, {'asset': self.assets[-1]})
        manifest.start(); self.addCleanup(manifest.stop)
        self.patch = patch.object(install, 'ASSETS', self.assets)
        self.patch.start(); self.addCleanup(self.patch.stop)
        disk = patch.object(install.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(20 << 30, 1 << 30, 19 << 30))
        disk.start(); self.addCleanup(disk.stop)

    def download(self, asset, target, progress):
        shutil.copyfile(self.archives[asset['kind']], target)
        progress(asset['bytes'])

    def test_fresh_home_and_committed_install_stays_stopped(self):
        before = self.settings.snapshot()
        self.assertEqual(before['storage_root'], str(self.root / 'home/data'))
        self.assertEqual(before['native_inference_precision'], 'mlx-q4')
        self.assertTrue(install.installation_status(self.settings)['required'])
        progress = []
        result = install.install_models(self.settings, progress.append, download=self.download)
        self.assertEqual(result, {'installed': True, 'model_started': False})
        after = self.settings.snapshot()
        self.assertFalse(install.installation_status(self.settings)['required'])
        self.assertEqual(after['teacher'], before['teacher'])
        receipt = json.loads((Path(after['native_mlx_model_path']) / 'download-receipt.json').read_text())
        self.assertTrue(receipt['complete'])
        self.assertTrue((Path(after['storage_root']) / 'modules/luban0-v1/active.json').is_file())
        self.assertTrue((Path(after['storage_root']) / 'runtime/mlx/expert-proof.json').is_file())
        self.assertNotEqual(after['storage_root'], before['storage_root'])
        install.factory_expert.validate.assert_called_once()
        self.assertEqual(install.factory_expert.install_journal.call_args.args[0], Path(after['storage_root']))
        self.assertEqual(progress[-1]['bytes'], progress[-1]['total'])
        with self.assertRaisesRegex(ValueError, '已安装'):
            install.install_models(self.settings, progress.append, download=self.download)

    def test_failed_second_download_keeps_settings_and_no_partial_activation(self):
        before = self.settings.snapshot()
        def fail(asset, target, progress):
            if asset['kind'] == 'speech': raise OSError('offline')
            self.download(asset, target, progress)
        with self.assertRaisesRegex(OSError, 'offline'):
            install.install_models(self.settings, lambda _: None, download=fail)
        self.assertEqual(before, self.settings.snapshot())
        self.assertEqual(list((Path(before['storage_root']) / 'app/foundation').iterdir()), [])

    def test_existing_learned_pointer_refuses_fresh_install(self):
        before = self.settings.snapshot()
        private_json(Path(before['storage_root']) / 'modules/luban0-v1/active.json', {'id': 'private'})
        with self.assertRaisesRegex(ValueError, '已有学习模型'):
            install.install_models(self.settings, lambda _: None, download=self.download)
        self.assertEqual(before, self.settings.snapshot())

    def test_settings_write_failure_removes_only_new_install(self):
        before = self.settings.snapshot()
        with patch.object(self.settings, 'update', side_effect=OSError('write failed')):
            with self.assertRaisesRegex(OSError, 'write failed'):
                install.install_models(self.settings, lambda _: None, download=self.download)
        self.assertEqual(before, self.settings.snapshot())
        self.assertEqual(list((Path(before['storage_root']) / 'app/foundation').iterdir()), [])

    def test_unknown_path_duplicate_symlink_and_compressed_weights_rejected(self):
        asset = self.assets[0]
        for unsafe in ('path', 'duplicate', 'symlink', 'compressed'):
            target = self.root / (unsafe + '.zip')
            with zipfile.ZipFile(target, 'w') as archive:
                for name, data in [('config.json', b'{}'), ('model.safetensors', b'fixture-weights')]:
                    row = zipfile.ZipInfo('mlx/' + name)
                    if unsafe == 'symlink': row.external_attr = (stat.S_IFLNK | 0o777) << 16
                    if unsafe == 'compressed': row.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(row, data)
                if unsafe == 'path': archive.writestr('../outside', b'no')
                if unsafe == 'duplicate':
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore'); archive.writestr('mlx/config.json', b'{}')
            with self.assertRaises(ValueError): install._extract(target, self.root / unsafe, asset)
            self.assertFalse((self.root / unsafe).exists())

    def test_seed_validation_or_journal_failure_cannot_change_settings(self):
        before = self.settings.snapshot()
        for method in ('validate', 'install_journal'):
            with patch.object(install.factory_expert, method, side_effect=ValueError('seed invalid')):
                with self.assertRaisesRegex(ValueError, 'seed invalid'):
                    install.install_models(self.settings, lambda _: None, download=self.download)
            self.assertEqual(before, self.settings.snapshot())
            self.assertEqual(list((Path(before['storage_root']) / 'app/foundation').iterdir()), [])

    def test_linked_storage_refused_before_database_access(self):
        before = self.settings.snapshot()
        root = Path(before['storage_root'])
        other = self.root / 'other-data'
        other.mkdir()
        root.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            install.install_models(self.settings, lambda _: None, download=self.download)
        self.assertEqual(list(other.iterdir()), [])
        self.assertEqual(before, self.settings.snapshot())

    def test_existing_user_material_is_not_abandoned(self):
        from luban.memory import Memory
        before = self.settings.snapshot()
        memory = Memory(before['storage_root'])
        memory.add_sample('question', 'answer', approved=True, training_allowed=True)
        with self.assertRaisesRegex(ValueError, '已有使用或学习资料'):
            install.install_models(self.settings, lambda _: None, download=self.download)
        self.assertEqual(before, self.settings.snapshot())
        self.assertEqual(len(memory.samples()), 1)

    def test_nested_whitelist_cannot_escape_new_root(self):
        for name in ('../escape', '/absolute', 'folder/../escape', 'folder\\escape', './escape'):
            with self.assertRaisesRegex(ValueError, '路径无效'):
                install._extract(self.root / 'unused.zip', self.root / 'unsafe', {'files': [name]})
        self.assertFalse((self.root / 'unsafe').exists())

    def test_redirects_restricted_to_github_release_asset_hosts(self):
        for url in ('http://github.com/a', 'https://github.com/another/repo', 'https://127.0.0.1/',
                    'https://github.com@evil.test/a', 'https://release-assets.githubusercontent.com:444/a'):
            with self.assertRaises(ValueError): install._allowed_url(url)
        self.assertEqual(install._allowed_url(install.RELEASE_URL + 'model.zip'), install.RELEASE_URL + 'model.zip')
        install._allowed_url('https://release-assets.githubusercontent.com/file?token=temporary')

    def test_download_hash_and_size_checked_before_extraction(self):
        asset = self.assets[0]
        class Response(io.BytesIO):
            status = 200
            headers = {'Content-Length': str(asset['bytes'])}
            def geturl(self): return install.RELEASE_URL + asset['name']
        raw = self.archives['mlx'].read_bytes()
        for data in (raw[:-1], bytes([raw[0] ^ 1]) + raw[1:]):
            with patch('urllib.request.build_opener') as opener:
                opener.return_value.open.return_value = Response(data)
                out = self.root / 'download.zip'
                with self.assertRaisesRegex(ValueError, '不完整或校验失败'):
                    install._download(asset, out, lambda _: None)
                out.unlink()

    def test_bundled_models_install_with_network_disabled_and_detect_corruption(self):
        folder = self.root / 'app/Contents/Resources/foundation'
        folder.mkdir(parents=True)
        module = self.root / 'app/Contents/Resources/backend/src/luban/foundation_install.py'
        for asset in self.assets:
            shutil.copyfile(self.archives[asset['kind']], folder / asset['name'])
        with patch.object(install, '__file__', str(module)), patch('urllib.request.build_opener', side_effect=AssertionError('network forbidden')):
            self.assertTrue(install.installation_status(self.settings)['bundled'])
            install.install_models(self.settings, lambda _: None)
            self.assertFalse(install.installation_status(self.settings)['required'])
            asset = self.assets[0]
            path = folder / asset['name']
            data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, '校验失败'):
                install._download(asset, self.root / 'corrupt-copy.zip', lambda _: None)
            path.unlink()
            with self.assertRaisesRegex(ValueError, '清单不一致'):
                install.bundled_assets()

    def test_saved_settings_are_not_relocated(self):
        home = self.root / 'existing'
        private_json(home / 'settings.json', {'storage_root': '/custom/root', 'model_path': '/custom/model',
                                           'native_inference_precision': 'bf16'})
        saved = Settings(home).snapshot()
        self.assertEqual(saved['storage_root'], '/custom/root')
        self.assertEqual(saved['model_path'], '/custom/model')
        self.assertEqual(saved['native_inference_precision'], 'bf16')
