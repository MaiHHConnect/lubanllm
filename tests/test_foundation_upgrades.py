"""Tiny disk/SQLite fixtures; no Torch/model, research proof files or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from luban.upgrades import read_upgrade_records


def digest(value):
    return hashlib.sha256(value).hexdigest()


class FoundationUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / 'modules/luban0-v1'
        self.store.mkdir(parents=True)
        self.version, self.uid, self.identity = 'core-123456abcdef', '123456abcdef', 'a' * 64
        self.report_name = 'evolution-' + self.uid + '.json'
        groups = {key: [] for key in ('new_training', 'new_validation', 'replay_training', 'replay_validation')}
        for key in ('new_training', 'new_validation', 'replay_training'):
            role, purpose = key.split('_')
            groups[key] = [{'id': key, 'prompt': key, 'answer': 'Luban0', 'source': 'human',
                'purpose': purpose, 'approved': True, 'training_allowed': True,
                'provenance': json.dumps({'evolution_role': role, 'generation_check': {'type': 'exact', 'expected': 'Luban0'}})}]
        protected = copy.deepcopy(groups['new_training'][0])
        protected.update(id='protected', purpose='validation', training_allowed=False)
        protected['provenance'] = json.dumps({'retention_generation_only': True, 'evolution_role': 'replay'})
        self.retention = {'schema_version': 1, 'identity': self.identity, 'version': self.version,
            'groups': groups, 'regression_validation': [protected]}
        self.report = {'id': self.uid, 'version': self.version, 'identity': self.identity,
            'previous_version': None, 'promoted': True, 'publication_state': 'published',
            'native_identity_verified': True, 'native_identity_sha256': 'b' * 64,
            'source_quality_sha256': 'c' * 64, 'source_protocol_sha256': 'd' * 64,
            'gates': dict.fromkeys(('two_complete_new_checks', 'old_protection', 'replay_loss'), True),
            'training_engine': {'name': 'Luban 训练引擎', 'engine': 'luban-training-ce-release-v1'},
            'teacher_api_calls': 0, 'target': '我是鲁班零号', 'steps': 900, 'completed_steps': 900,
            'learning_rate': 1e-5, 'distillation_weight': 5, 'dtype': 'torch.bfloat16',
            'optimizer_dtype': 'torch.float32', 'parameter': 'model.language_model.layers.23.mlp.base.down_proj.weight',
            'trained_parameters': 12582912, 'added_parameters': 0}
        self.metadata = {'identity': self.identity, 'version': self.version, 'previous_version': None,
            'report': self.report_name, 'parameter': self.report['parameter'], 'dtype': 'torch.bfloat16',
            'shape': [2048, 6144], 'trained_parameters': 12582912, 'candidate_tensor_sha256': 'e' * 64}
        self.database = self.root / 'app/memory.sqlite3'
        self.database.parent.mkdir()
        with sqlite3.connect(self.database) as db:
            db.execute('CREATE TABLE training_runs(id TEXT,kind TEXT,state TEXT,body TEXT)')
            db.execute('CREATE TABLE events(id TEXT,kind TEXT,body TEXT)')
        self.write()

    def ordinary(self):
        from luban.upgrades import CORE_GATES
        self.report.pop('distillation_weight')
        self.report.pop('target')
        self.report.pop('teacher_api_calls')
        self.report.update(previous_version='core-000000000001', source_training_version='core-000000000002',
            source_previous_version='core-000000000001', gates=dict.fromkeys(CORE_GATES, True),
            steps=450, completed_steps=128, started=time.time() - 30, elapsed_seconds=10,
            replay_subspace={'enabled': True}, training_selection={'selected_step': 128},
            distillation={'weight': 5, 'teacher_version': 'core-000000000001'},
            baseline_new_loss=2, validation_new_loss=1, baseline_replay_loss=1, validation_replay_loss=1)
        self.report['training_engine']['teacher_api_calls'] = 0
        self.metadata['previous_version'] = self.report['previous_version']
        self.metadata.pop('candidate_tensor_sha256')  # Actual ordinary metadata has a file SHA, not this persona-only raw field.

    def write(self, installed=False, mutate=None, sync_proof=True):
        content = {key: value for key, value in self.retention.items() if key != 'content_sha256'}
        self.retention['content_sha256'] = digest(json.dumps(content, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode())
        weights = self.store / (self.version + '.safetensors')
        weights.write_bytes(b'tiny raw boundary fixture, not real model tensors')
        retention = self.store / (self.version + '.retention.json')
        retention.write_text(json.dumps(self.retention))
        self.metadata.update(sha256=digest(weights.read_bytes()), retention_sha256=digest(retention.read_bytes()))
        if 'source_training_version' in self.report and sync_proof:
            groups = self.retention['groups']
            self.metadata['dataset_sha256'] = digest(json.dumps(groups, sort_keys=True, ensure_ascii=False).encode())
            self.report.update(dataset_sha256=self.metadata['dataset_sha256'], checkpoint=copy.deepcopy(self.metadata),
                split_ids={key: [row['id'] for row in rows] for key, rows in groups.items()},
                sample_sha256={row['id']: digest(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
                    for rows in groups.values() for row in rows},
                retention={'file_sha256': self.metadata['retention_sha256'], 'content_sha256': self.retention['content_sha256']})
        (self.store / (self.version + '.json')).write_text(json.dumps(self.metadata))
        report_path = self.store / self.report_name
        report_path.write_text(json.dumps(self.report))
        committed = copy.deepcopy(self.report)
        if installed:
            committed.update(foundation_install=True, source_report_sha256=digest(report_path.read_bytes()),
                training_updates_executed=0,
                installation_scope='Install the validated release core; steps describe source training, not new training on this device')
        records = {'groups': self.retention['groups'], 'protection': self.retention['regression_validation'], 'retention_regressions': []}
        from luban.foundation import _distillation_weight
        self.body = {'scope': 'product', 'binding': {'identity': self.identity,
            'arguments': {key: self.report[key] for key in ('steps', 'learning_rate')} | {
                'distillation_weight': _distillation_weight(self.report), 'preserve_replay_subspace': True},
            'records_sha256': digest(json.dumps(records, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())},
            'records': copy.deepcopy(records), 'commit': {'report': committed, 'event_kind': 'evolution',
                'artifacts': {path.name: digest(path.read_bytes()) for path in self.store.iterdir()},
                'pointers': [{'name': 'active-core.json', 'before': None, 'after': {'version': self.version,
                    'identity': self.identity, 'sha256': self.metadata['sha256']}}]}}
        if mutate:
            mutate(self.body)
        self.persist()

    def persist(self):
        with sqlite3.connect(self.database) as db:
            db.execute('DELETE FROM training_runs')
            db.execute('DELETE FROM events')
            db.execute('INSERT INTO training_runs VALUES(?,?,?,?)', (self.uid, 'evolution', 'committed', json.dumps(self.body)))
            db.execute('INSERT INTO events VALUES(?,?,?)', ('training:' + self.uid, 'evolution', json.dumps(self.body['commit']['report'])))

    def rejected(self):
        with self.assertRaisesRegex(RuntimeError, '无法完整核验'):
            read_upgrade_records(self.root)

    def test_original_adoption_and_fresh_install_are_not_user_evolution(self):
        self.assertEqual(read_upgrade_records(self.root), [])
        self.write(installed=True)
        self.assertEqual(read_upgrade_records(self.root), [])

    def test_factory_core_uses_original_validator_without_personal_growth(self):
        self.body['factory_baseline'] = True
        self.persist()
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE events SET kind='factory:evolution'")
        self.assertEqual(read_upgrade_records(self.root), [])
        # The factory label cannot hide damaged original model artifacts.
        (self.store / (self.version + '.safetensors')).write_bytes(b'changed')
        self.rejected()

    def test_export_binding_without_live_arguments_is_verified_without_fabrication(self):
        self.body['binding'].pop('arguments')
        self.persist()
        self.assertEqual(read_upgrade_records(self.root), [])
        self.body['binding']['unverified_extra'] = True
        self.persist()
        self.rejected()
        self.write(installed=True)
        self.body['binding'].pop('arguments')
        self.persist()
        self.assertEqual(read_upgrade_records(self.root), [])

    def test_install_allows_exactly_four_fields_zero_updates_and_bound_source(self):
        for key, value in [('extra', True), ('training_updates_executed', 1), ('source_report_sha256', 'f' * 64),
                           ('foundation_install', False), ('installation_scope', 'other')]:
            with self.subTest(key=key):
                self.write(installed=True, mutate=lambda body: body['commit']['report'].update({key: value}))
                self.rejected()
        self.write(installed=True, mutate=lambda body: body['commit']['report'].pop('installation_scope'))
        self.rejected()

    def test_partial_foundation_proof_never_silently_skipped(self):
        original = copy.deepcopy(self.report)
        for key in ('native_identity_verified', 'native_identity_sha256', 'source_quality_sha256', 'source_protocol_sha256', 'previous_version'):
            with self.subTest(key=key):
                self.report = copy.deepcopy(original)
                self.report.pop(key)
                self.write()
                self.rejected()
        self.report = original | {'partial_policy': 'closed-book-growth-v1'}
        self.write()
        self.rejected()

    def test_metadata_retention_permission_and_commit_tampering_rejected(self):
        original = copy.deepcopy(self.metadata)
        for key, value in [('identity', 'f' * 64), ('version', 'core-000000000000'), ('candidate_tensor_sha256', 'bad'),
                           ('parameter', 'other'), ('shape', [2, 2])]:
            with self.subTest(key=key):
                self.metadata = original | {key: value}
                self.write()
                self.rejected()
        self.metadata = original
        self.retention['groups']['new_training'][0]['training_allowed'] = False
        self.write()  # Rehash everything: content/permission semantics must still reject.
        self.rejected()

    def test_artifact_records_pointer_arguments_and_event_must_match(self):
        mutations = [lambda body: body['commit']['artifacts'].update({self.report_name: 'f' * 64}),
            lambda body: body['records']['groups'].update(new_training=[]),
            lambda body: body['binding'].update(records_sha256='f' * 64),
            lambda body: body['binding']['arguments'].update(steps=450),
            lambda body: body['commit']['pointers'][0]['after'].update(identity='f' * 64),
            lambda body: body.update(scope='research', product_accepted=True)]
        for mutation in mutations:
            self.write(mutate=mutation)
            self.rejected()
        self.write()
        with sqlite3.connect(self.database) as db:
            db.execute('DELETE FROM events')
        self.rejected()

    def test_ordinary_core_still_requires_seven_gates(self):
        for key in ('native_identity_verified', 'native_identity_sha256', 'source_quality_sha256', 'source_protocol_sha256'):
            self.report.pop(key)
        self.write()
        self.rejected()

    def test_real_ordinary_release_and_zero_update_install_verify_without_persona_guard(self):
        self.ordinary()
        self.write()
        self.assertEqual(read_upgrade_records(self.root), [])
        self.write(installed=True)
        self.assertEqual(read_upgrade_records(self.root), [])
        source = json.loads((self.store / self.report_name).read_text())
        self.assertEqual(source['previous_version'], 'core-000000000001')
        self.assertEqual(source['distillation']['weight'], 5)
        self.assertNotIn('distillation_weight', source)
        self.assertEqual(self.body['commit']['report']['training_updates_executed'], 0)

    def test_ordinary_release_requires_every_original_gate_and_no_partial_policy(self):
        from luban.upgrades import CORE_GATES
        for gate in CORE_GATES:
            for mutation in ('missing', 'false'):
                with self.subTest(gate=gate, mutation=mutation):
                    self.setUpOrdinary()
                    if mutation == 'missing':
                        self.report['gates'].pop(gate)
                    else:
                        self.report['gates'][gate] = False
                    self.write(installed=True)
                    self.rejected()
        self.setUpOrdinary()
        self.report['partial_policy'] = 'closed-book-growth-v1'
        self.write(installed=True)
        self.rejected()

    def setUpOrdinary(self):
        # A fresh in-memory source restores the same fixture without creating another TemporaryDirectory.
        original = getattr(self, '_ordinary_source', None)
        if original is None:
            self.ordinary()
            self._ordinary_source = copy.deepcopy((self.report, self.metadata))
        else:
            self.report, self.metadata = copy.deepcopy(original)

    def test_ordinary_release_keeps_original_loss_gate_checks(self):
        for field, value in [('validation_new_loss', 2), ('validation_replay_loss', 2),
                             ('baseline_new_loss', float('nan')), ('started', time.time() + 1000)]:
            with self.subTest(field=field):
                self.setUpOrdinary()
                self.report[field] = value
                self.write(installed=True)
                self.rejected()

    def test_ordinary_source_lineage_parameters_and_native_proof_must_match(self):
        for field, value in [('source_training_version', self.version), ('source_training_version', 'invalid'),
                ('source_training_version', 'core-000000000001'), ('previous_version', None),
                ('source_previous_version', 'core-000000000009'), ('native_identity_sha256', 'bad'),
                ('source_quality_sha256', 'bad'), ('source_protocol_sha256', 'bad'), ('native_identity_verified', False),
                ('parameter', 'other'), ('trained_parameters', 1), ('dtype', 'torch.float32'), ('completed_steps', True)]:
            with self.subTest(field=field):
                self.setUpOrdinary()
                self.report[field] = value
                self.write(installed=True)
                self.rejected()

    def test_ordinary_release_checkpoint_and_material_links_are_not_just_rehashed_claims(self):
        self.setUpOrdinary()
        for mutation in (lambda: self.report['checkpoint'].update(sha256='f' * 64),
                lambda: self.report['retention'].update(content_sha256='f' * 64),
                lambda: self.report['retention'].update(file_sha256='f' * 64),
                lambda: self.report.update(dataset_sha256='f' * 64),
                lambda: self.report['split_ids'].update(new_training=[]),
                lambda: self.report['sample_sha256'].update(new_training='f' * 64)):
            self.setUpOrdinary()
            self.write(installed=True)
            mutation()
            self.write(installed=True, sync_proof=False)  # Rebind journal/event/artifacts to prove semantic checking.
            self.rejected()

    def test_ordinary_install_still_requires_exact_source_and_zero_updates(self):
        for mutation in (lambda body: body['commit']['report'].update(training_updates_executed=1),
                lambda body: body['commit']['report'].update(source_report_sha256='f' * 64),
                lambda body: body['commit']['report'].update(extra=True),
                lambda body: body['commit']['artifacts'].update({self.report_name: 'f' * 64}),
                lambda body: body['binding']['arguments'].update(distillation_weight=4),
                lambda body: body['binding']['arguments'].update(preserve_replay_subspace=1),
                lambda body: body['commit']['pointers'][0]['after'].update(identity='f' * 64)):
            self.setUpOrdinary()
            self.write(installed=True, mutate=mutation)
            self.rejected()
        self.setUpOrdinary()
        self.write(installed=True)
        self.body['binding'].pop('arguments')  # The fixed ZIP binding remains valid.
        self.persist()
        self.assertEqual(read_upgrade_records(self.root), [])
        with sqlite3.connect(self.database) as db:
            db.execute('DELETE FROM events')
        self.rejected()

    def test_ordinary_install_revoked_training_permission_is_rejected_after_all_hashes_refresh(self):
        self.ordinary()
        self.retention['groups']['new_training'][0]['training_allowed'] = False
        self.write(installed=True)
        self.rejected()

    def test_existing_ordinary_api_reports_still_count_with_legacy_seven_gate_contract(self):
        self.ordinary()
        self.write()
        for field in ('native_identity_verified', 'native_identity_sha256', 'source_quality_sha256',
                      'source_protocol_sha256', 'source_training_version', 'source_previous_version'):
            self.report.pop(field)
        self.write(sync_proof=False)
        self.assertEqual(len(read_upgrade_records(self.root)), 1)
        self.report['gates'].pop('core_tensor_changed')
        self.write(sync_proof=False)
        self.assertEqual(len(read_upgrade_records(self.root)), 1)


if __name__ == '__main__':
    unittest.main()
