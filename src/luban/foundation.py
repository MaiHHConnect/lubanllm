"""Install a validated release core on first use; upstream weights stay intact."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .settings import private_json


def _distillation_weight(report):
    """Read either actual report layout; contradictory source values fail closed."""
    values = [report['distillation_weight']] if 'distillation_weight' in report else []
    if 'distillation' in report:
        nested = report['distillation']
        if not isinstance(nested, dict) or 'weight' not in nested:
            raise ValueError('正式基础核心蒸馏参数无效')
        values.append(nested['weight'])
    if (not values or any(type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 100 for value in values) or any(value != values[0] for value in values)):
        raise ValueError('正式基础核心蒸馏参数缺失、无效或不一致')
    return values[-1]


def install_foundation(runtime):
    # Original-core research must continue to mean the official upstream core,
    # even after the product's validated persona foundation is installed.
    if getattr(runtime, 'research', False):
        return None
    if not getattr(runtime, 'loaded_path', None):
        return None
    source = Path(runtime.loaded_path) / 'luban-foundation.json'
    if not source.exists():
        return None
    if source.is_symlink():
        raise ValueError('正式基础核心清单不能是符号链接')
    manifest = json.loads(source.read_text())
    version = manifest.get('version')
    if (manifest.get('identity') != runtime.identity or not isinstance(version, str)
            or not re.fullmatch(r'core-[a-f0-9]{12}', version)
            or manifest.get('quality_passed') is not True):
        raise ValueError('正式基础核心清单无效或尚未通过检查')
    expected = {version + suffix for suffix in ('.json', '.safetensors', '.retention.json')}
    report = manifest.get('report')
    if report != 'evolution-' + version.removeprefix('core-') + '.json':
        raise ValueError('正式基础核心缺少绑定的训练报告')
    expected.add(report)
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError('正式基础核心文件清单不完整')
    contents = {}
    for name, digest in files.items():
        path = source.parent / 'luban-foundation' / name
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('正式基础核心文件不能是符号链接')
        data = path.read_bytes()
        if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest) or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('正式基础核心文件校验失败')
        contents[name] = data
    checked_report = json.loads(contents[report])
    metadata = json.loads(contents[version + '.json'])
    if (checked_report.get('promoted') is not True or checked_report.get('version') != version
            or checked_report.get('id') != version.removeprefix('core-')
            or checked_report.get('identity') != runtime.identity
            or checked_report.get('publication_state') != 'published'
            or checked_report.get('native_identity_verified') is not True
            or metadata.get('identity') != runtime.identity or metadata.get('version') != version
            or metadata.get('report') != report or metadata.get('sha256') != files[version + '.safetensors']
            or metadata.get('retention_sha256') != files[version + '.retention.json']):
        raise ValueError('正式基础核心训练与正式推理检查未通过')
    from .retention import validate_core_retention
    retention = validate_core_retention(json.loads(contents[version + '.retention.json']), runtime.identity, version)
    gates = checked_report.get('gates')
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        raise ValueError('正式基础核心缺少完整通过的采纳门禁')
    arguments = {key: checked_report.get(key) for key in ('steps', 'learning_rate', 'distillation_weight')}
    arguments['distillation_weight'] = _distillation_weight(checked_report)
    arguments['preserve_replay_subspace'] = True
    if (type(arguments['steps']) is not int or not 1 <= arguments['steps'] <= 10000
            or type(arguments['learning_rate']) not in (int, float)
            or not 1e-8 <= arguments['learning_rate'] <= .01
            or type(arguments['distillation_weight']) not in (int, float)
            or not 0 <= arguments['distillation_weight'] <= 100):
        raise ValueError('正式基础核心训练参数无效')
    for name, data in contents.items():
        target = Path(runtime.store) / name
        if target.is_symlink():
            raise ValueError('正式基础核心目标不能是符号链接')
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError('已有正式基础核心文件不同，禁止覆盖')
        else:
            # A crash while copying must leave only an unreferenced temporary
            # file, rather than a truncated immutable member blocking restart.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.' + target.name + '.', delete=False) as stream:
                    temporary = Path(stream.name)
                    os.chmod(temporary, 0o600)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if target.exists():
                    if target.is_symlink() or target.read_bytes() != data:
                        raise ValueError('基础核心复制期间目标发生变化')
                else:
                    os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    # Validate actual tensor shape/dtype/finite values before creating a durable
    # product adoption journal. Copy+pointer alone cannot be safely exported.
    from .evolution import _read_version
    _read_version(runtime, version)
    from .training_checkpoint import TrainingSession, training_runs
    pointer = {'version': version, 'identity': runtime.identity, 'sha256': metadata['sha256']}
    installed_report = {**checked_report, 'foundation_install': True, 'source_report_sha256': files[report],
        'training_updates_executed': 0,
        'installation_scope': 'Install the validated release core; steps describe source training, not new training on this device'}
    previous = next((row for row in training_runs(runtime.memory, runtime.identity)
                     if row['id'] == version.removeprefix('core-')), None)
    if previous and previous['state'] == 'committed':
        commit = previous.get('commit') or {}
        if (previous.get('kind') != 'evolution' or previous.get('scope', 'product') != 'product'
                or commit.get('report') not in (checked_report, installed_report) or commit.get('artifacts') != files
                or len(commit.get('pointers', [])) != 1
                or commit['pointers'][0].get('name') != 'active-core.json'
                or commit['pointers'][0].get('after') != pointer):
            raise ValueError('已有正式基础核心回执与不可变版本不一致')
        # training_runs closes its connection before this event read; never
        # nest two EXCLUSIVE SQLite connections while holding Memory.lock.
        with runtime.memory.lock, runtime.memory.connect() as db:
            event = db.execute('SELECT kind,body FROM events WHERE id=?',
                ('training:' + previous['id'],)).fetchone()
        if event is None or event['kind'] != 'evolution' or json.loads(event['body']) != commit['report']:
            raise ValueError('已安装正式基础核心缺少匹配的持久采纳事件')
        return pointer
    records = {'groups': retention['groups'], 'protection': retention['regression_validation'], 'retention_regressions': []}
    session = TrainingSession(runtime, 'evolution', arguments, records,
        run_id=version.removeprefix('core-'), scope='product')
    session.prepare_commit(installed_report, {'active-core.json': pointer}, sorted(files), event_kind='evolution')
    session.finish_commit()
    return pointer
