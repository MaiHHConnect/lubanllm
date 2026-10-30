"""Real local whitelist ZIP roundtrip, staging only; never loads a backbone."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from types import SimpleNamespace
import uuid
import zipfile

import psutil

from luban import model_packages as packages
from luban.settings import Settings, default_home, private_json
from luban.upgrades import read_upgrade_records

REPO = Path(__file__).resolve().parents[1]
PRODUCTION = Path('/Volumes/移动数据库/lubanllm')
BACKUP = Path('/Volumes/Untitled/lubanllm-backup')
BASE = PRODUCTION / 'models/Qwen3.5-2B-luban0-v1'
ASR = PRODUCTION / 'models/Qwen3-ASR-0.6B-luban0-v1'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def file_state(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Production snapshot refuses symlinks')
    return {'size': path.stat().st_size, 'sha256': sha(path)} if path.exists() else None


def production_state():
    store = PRODUCTION / 'modules/luban0-v1'
    return {
        'database': {name: file_state(PRODUCTION / 'app' / name) for name in
                     ('memory.sqlite3', 'memory.sqlite3-wal', 'memory.sqlite3-shm')},
        'modules': {str(p.relative_to(store)): file_state(p) for p in sorted(store.rglob('*'))
                    if p.is_file() and not p.name.startswith('._')},
        'settings': {name: file_state(default_home().resolve() / name) for name in
                     ('settings.json', 'credentials.json', 'package-activation.json')},
    }


def plan():
    return {'scope': 'actual service ZIP export/stage-import; metadata-only stopped runtime contract, not model load or activation proof',
            'production_root': str(PRODUCTION), 'base': str(BASE), 'asr': str(ASR),
            'whitelist_files': sum(len(group) for group in packages.TRUSTED.values()),
            'whitelist_bytes': sum(item['size'] for group in packages.TRUSTED.values() for item in group.values()),
            'expected_accepted_counts': {'router': 0, 'learning': 0, 'evolution': 0},
            'teacher_calls': 0, 'model_allocations': 0, 'activate': False,
            'rss_measurement': 'psutil current process RSS sampled every 0.5s including hashing and staging; sampled maximum, not OS peak or physical model memory',
            'backup_policy': 'one ZIP plus evidence on second disk; no second copy of staged model weights'}


def run():
    for root in (PRODUCTION, BACKUP):
        if not root.parent.is_mount() or not root.is_dir() or root.is_symlink():
            raise ValueError('Both required external disks must be mounted')
    fixed = plan()
    minimum = fixed['whitelist_bytes'] + 1024**3
    if shutil.disk_usage(PRODUCTION).free < minimum * 2 or shutil.disk_usage(BACKUP).free < minimum:
        raise OSError('Insufficient space for new ZIP/stage and single ZIP backup')
    name = 'model-package-roundtrip-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    directory = PRODUCTION / 'experiments' / name
    directory.mkdir(mode=0o700)
    evidence = directory / 'evidence'
    evidence.mkdir(mode=0o700)
    target = directory / 'luban0-official-local.zip'
    report = {**fixed, 'passed': False, 'pid': os.getpid(), 'directory': str(directory),
              'started_utc': datetime.now(timezone.utc).isoformat(), 'free_bytes_before': {
                  str(root): shutil.disk_usage(root).free for root in (PRODUCTION, BACKUP)},
              'sources': {str(p.relative_to(REPO)): sha(p) for p in
                  (Path(__file__), REPO / 'src/luban/model_packages.py', REPO / 'src/luban/upgrades.py',
                   REPO / 'src/luban/settings.py', REPO / 'src/luban/backups.py',
                   REPO / 'src/luban/retention.py', REPO / 'src/luban/training_checkpoint.py')},
              'progress': [], 'rss_samples': []}
    private_json(evidence / 'protocol.json', report)
    stop = threading.Event()
    started = time.monotonic()
    process = psutil.Process()

    def sample():
        while not stop.wait(0.5):
            report['rss_samples'].append({'elapsed_seconds': time.monotonic() - started,
                                         'rss_bytes': process.memory_info().rss})

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    last_progress = [None, 0.0]

    def progress(item):
        now = time.monotonic()
        if item['phase'] != last_progress[0] or now - last_progress[1] >= 10 or item['bytes'] == item['total']:
            event = {**item, 'elapsed_seconds': now - started, 'rss_bytes': process.memory_info().rss}
            report['progress'].append(event)
            print(json.dumps(event), flush=True)
            last_progress[:] = [item['phase'], now]

    try:
        print(json.dumps({'phase': 'production_before', 'directory': str(directory), 'pid': os.getpid()}), flush=True)
        before = production_state()
        assert not any(before['modules'].get(name) for name in
                       ('active-core.json', 'active-router.json', 'active.json', 'router.json', 'router.safetensors'))
        # Only attributes required by export_package/_journal; no Runtime or
        # production Settings construction and no writable production database.
        settings = Settings(directory / 'private-settings')
        settings.update({'storage_root': str(PRODUCTION), 'model_path': str(BASE), 'speech_model_path': str(ASR)})
        private_settings_before = {n: file_state(settings.home / n) for n in ('settings.json', 'credentials.json')}
        runtime = SimpleNamespace(state='stopped', lock=threading.RLock(), training_session=None,
            identity=packages.TRUSTED_IDENTITY, store=PRODUCTION / 'modules/luban0-v1', settings=settings,
            memory=SimpleNamespace(path=PRODUCTION / 'app/memory.sqlite3', lock=threading.RLock()))
        payload = packages._journal(runtime)
        if payload['runs'] or payload['samples'] or payload['material_archives']:
            raise ValueError('Fixed zero-adopted-modules roundtrip contract changed')
        report['production_before'] = before
        tick = time.monotonic()
        report['export'] = packages.export_package(runtime, target, progress=progress)
        report['export_seconds'] = time.monotonic() - tick
        print(json.dumps({'phase': 'safe_inspection'}), flush=True)
        report['inspection'] = packages.inspect_package(target)
        assert report['inspection']['content_verified'] is False
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            accepted = json.loads(archive.read('accepted.json'))
            assert accepted == payload
            expected_names = {'manifest.json', 'accepted.json'} | {
                'models/' + group + '/' + filename for group, files in packages.TRUSTED.items() for filename in files}
            assert set(archive.namelist()) == expected_names
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()
                       if info.filename.endswith('.safetensors'))
        report['private_exclusions'] = {'exact_whitelist_only': True, 'production_database': True,
            'settings_credentials': True, 'chat_history': True, 'teacher_pending_raw': True,
            'external_brain': True, 'optimizer_checkpoints': True, 'retired_unaccepted_modules': True}
        private_json(evidence / 'manifest.json', manifest)
        tick = time.monotonic()
        report['stage'] = packages.stage_import(target, directory / 'imports', progress=progress)
        report['stage_seconds'] = time.monotonic() - tick
        staged = Path(report['stage']['root'])
        assert report['stage']['content_verified'] is True and report['stage']['package_id'] == report['export']['package_id']
        assert report['stage']['counts'] == fixed['expected_accepted_counts']
        print(json.dumps({'phase': 'independent_staged_sha_readback'}), flush=True)
        readback = {name: file_state(staged / name) for name in manifest['files']}
        assert readback == manifest['files']
        private_json(evidence / 'staged-file-readback.json', readback)
        assert read_upgrade_records(staged) == []
        with sqlite3.connect((staged / 'app/memory.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
            report['staged_database_counts'] = {t: db.execute('SELECT COUNT(*) FROM ' + t).fetchone()[0]
                                               for t in ('samples', 'training_runs', 'events', 'conversations', 'teacher_materials')}
        assert not any(report['staged_database_counts'].values())
        assert {n: file_state(settings.home / n) for n in private_settings_before} == private_settings_before
        report['production_after'] = production_state()
        assert report['production_after'] == before
        report['production_unchanged'] = True
        report['stage_owner_state'] = json.loads((staged / 'package-owner.json').read_text())['state']
        assert report['stage_owner_state'] == 'validated'
        report['per_file_sha_readback'] = True
        report['license_files_preserved'] = ['models/base/LICENSE', 'models/asr/LICENSE-2.0.txt', 'models/asr/license-manifest.json']
        report['zip_sha256'] = sha(target)
        report['passed'] = True
    except BaseException as error:
        report['error'] = str(error) or type(error).__name__
        raise
    finally:
        stop.set()
        sampler.join()
        report['elapsed_seconds_before_backup'] = time.monotonic() - started
        report['sampled_rss_max_bytes'] = max([process.memory_info().rss] + [s['rss_bytes'] for s in report['rss_samples']])
        report['runtime_state_at_exit'] = 'stopped metadata-only contract; no loaded model or production Settings instance'
        private_json(evidence / 'result.json', report)
    print(json.dumps({'phase': 'single_zip_backup', 'bytes': target.stat().st_size}), flush=True)
    tick = time.monotonic()
    backup = BACKUP / name
    backup.mkdir(mode=0o700)
    shutil.copyfile(target, backup / target.name)
    if sha(backup / target.name) != report['zip_sha256']:
        raise ValueError('Second disk ZIP readback SHA mismatch')
    private_json(evidence / 'backup.json', {'root': str(backup), 'zip_sha256': report['zip_sha256'],
        'zip_bytes': target.stat().st_size, 'elapsed_seconds': time.monotonic() - tick, 'stage_weights_not_copied': True})
    shutil.copytree(evidence, backup / 'evidence')
    evidence_sha = {str(p.relative_to(evidence)): sha(p) for p in sorted(evidence.rglob('*'))
                    if p.is_file() and not p.name.startswith('._')}
    for relative, value in evidence_sha.items():
        assert sha(backup / 'evidence' / relative) == value
    summary = {'passed': True, 'result_path': str(evidence / 'result.json'), 'result_sha256': sha(evidence / 'result.json'),
        'zip_path': str(target), 'zip_sha256': report['zip_sha256'], 'zip_bytes': target.stat().st_size,
        'backup': str(backup), 'evidence_sha256': evidence_sha, 'backup_verified': True,
        'model_allocations': 0, 'activation': False, 'elapsed_seconds_including_backup': time.monotonic() - started}
    private_json(evidence / 'completed.json', summary)
    private_json(backup / 'evidence/completed.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='write one new real ZIP/stage/backup set')
    args = parser.parse_args()
    if args.run:
        run()
    else:
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
