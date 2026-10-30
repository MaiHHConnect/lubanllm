"""CPU-only publication fixtures: no production model, journal or GPU writes."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from luban.core_publication import content_sha
from luban.generation_checks import evaluate_generation
from luban.retention import _encoded, provenance
from luban.upgrades import CORE_GATES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('goal_foundation_release', ROOT / 'scripts/release_goal_foundation.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
from tests.test_evolution_goal import sample


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def protected(identifier):
    row = sample(identifier, 'validation', 'replay')
    row['training_allowed'] = False
    value = provenance(row)
    value['retention_generation_only'] = True
    row['provenance'] = json.dumps(value)
    return row


def seal(snapshot):
    snapshot['content_sha256'] = release.hashlib.sha256(_encoded({key: value for key, value in snapshot.items()
                                                               if key != 'content_sha256'})).hexdigest()


class GoalFoundationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.identity = 'a' * 64
        self.previous, self.version = 'core-000000000045', 'core-000000000046'
        self.model, self.source, self.output = (self.root / name for name in ('base', 'source', 'run'))
        for directory in (self.model, self.source, self.output):
            directory.mkdir()
        self.store = self.output / 'isolated-storage/modules/luban0-v1'
        self.store.mkdir(parents=True)
        self.trusted = {}
        for name in release.TRUSTED['base']:
            path = self.model / name
            path.write_bytes(b'cpu pinned base fixture:' + name.encode())
            if name == 'config.json':
                save(path, {'text_config': {'eos_token_id': [99]}})
            self.trusted[name] = release._record(path)
        (self.model / 'luban-foundation').mkdir()
        (self.model / 'luban-foundation/old.json').write_text('old foundation excluded')
        (self.model / 'luban-foundation.json').write_text('old manifest excluded')
        self.addCleanup(patch.stopall)
        patch.object(release, 'TRUSTED', {'base': self.trusted}).start()
        patch.object(release, 'TRUSTED_IDENTITY', self.identity).start()
        self.source_hashes = {'scripts/release_goal_foundation.py': release.sha(Path(release.__file__))}
        patch.object(release, 'source_hashes', return_value=self.source_hashes).start()
        old_groups = {}
        for key, count in [('new_training', 40), ('new_validation', 40), ('replay_training', 43), ('replay_validation', 40)]:
            role, purpose = key.split('_')
            old_groups[key] = [sample('old-' + key + '-' + str(i), purpose, role) for i in range(count)]
        old = {'schema_version': 1, 'version': self.previous, 'identity': self.identity, 'groups': old_groups,
               'regression_validation': [protected('old-regression-' + str(i)) for i in range(9)]}
        seal(old)
        save(self.source / (self.previous + '.retention.json'), old)
        self.pointer = {'version': self.previous, 'identity': self.identity, 'sha256': 'b' * 64}
        for name in ('active-core.json', 'graph-verification.json', self.previous + '.json',
                     'evolution-' + self.previous.removeprefix('core-') + '.json'):
            save(self.source / name, self.pointer if name == 'active-core.json' else {})
        (self.source / (self.previous + '.safetensors')).write_bytes(b'prior CPU fixture')
        material = {'TRAIN': [sample('new-train-' + str(i)) for i in range(16)],
                    'VALID': [sample('new-valid-' + str(i), 'validation') for i in range(4)]}
        from train_evolution_goal import materials
        groups = materials({'samples': material['TRAIN'] + material['VALID']}, old)
        acceptance = [protected('acceptance-' + str(i)) for i in range(8)]
        protection = old_groups['new_validation'] + old_groups['replay_validation'] + old['regression_validation']
        old_train = [row for rows in old_groups.values() for row in rows if row['purpose'] == 'training']
        self.formal = self.root / 'protocol.json'
        formal_files = {'new_material': material, 'old_retention_complete': old,
                        'old_training_original': old_train, 'old_protection': protection, 'acceptance': acceptance}
        names = {}
        for key, value in formal_files.items():
            name = key + '.json'
            save(self.root / name, value)
            names[key] = name
        save(self.formal, {**names, 'source_core': self.pointer, 'status': 'PRE_REGISTERED_NOT_TRAINED',
                          'weights_updated': False, 'production_written': False, 'gpu_used': False, 'teacher_calls': 0,
                          'source_retention_sha256': release.sha(self.source / (self.previous + '.retention.json'))})
        self.plan_path = self.output / 'plan.json'
        self.plan = {'protocol': release.PROTOCOL, 'arguments': copy.deepcopy(release.ARGUMENTS), 'seed': 42,
            'source_sha256': self.source_hashes, 'source_store': str(self.source), 'model_root': str(self.model),
            'model_config_sha256': self.trusted['config.json']['sha256'],
            'model_manifest_sha256': self.trusted['upstream-file-manifest.json']['sha256'],
            'starting_core': self.previous, 'identity': self.identity, 'output': str(self.output),
            'storage': str(self.output / 'isolated-storage'), 'formal_protocol': str(self.formal),
            'formal_protocol_sha256': release.sha(self.formal), 'formal_files': {
                key: {'path': str(self.root / name), 'sha256': release.sha(self.root / name)} for key, name in names.items()},
            'source_files': {path.name: release.sha(path) for path in self.source.iterdir()},
            'materials': str(self.root / names['new_material']), 'materials_sha256': release.sha(self.root / names['new_material']),
            'groups': groups, 'protection': protection, 'old_training_original': old_train,
            'acceptance': acceptance, 'retention_regressions': acceptance}
        save(self.plan_path, self.plan)
        snapshot = copy.deepcopy(old)
        snapshot.update(version=self.version, groups=groups, regression_validation=old['regression_validation'] + acceptance)
        seal(snapshot)
        save(self.store / (self.version + '.retention.json'), snapshot)
        self.report_name = 'evolution-' + self.version.removeprefix('core-') + '.json'
        parameter = 'model.language_model.layers.23.mlp.base.down_proj.weight'
        raw = bytes(25165824)
        header = json.dumps({parameter: {'dtype': 'BF16', 'shape': [2048, 6144], 'data_offsets': [0, len(raw)]}}).encode()
        weights = self.store / (self.version + '.safetensors')
        weights.write_bytes(struct.pack('<Q', len(header)) + header + raw)
        self.raw_sha = release.hashlib.sha256(raw).hexdigest()
        metadata = {'identity': self.identity, 'version': self.version, 'previous_version': self.previous,
            'report': self.report_name, 'parameter': parameter, 'dtype': 'torch.bfloat16', 'shape': [2048, 6144],
            'trained_parameters': 12582912, 'sha256': release.sha(weights),
            'dataset_sha256': content_sha(groups), 'retention_sha256': release.sha(self.store / (self.version + '.retention.json'))}
        save(self.store / (self.version + '.json'), metadata)
        self.report = {'id': self.version.removeprefix('core-'), 'version': self.version, 'identity': self.identity,
            'previous_version': self.previous, 'promoted': True, 'publication_state': 'published',
            'gates': dict.fromkeys(CORE_GATES, True), 'steps': 450, 'completed_steps': 128,
            'learning_rate': 1e-5, 'distillation': {'weight': 5, 'teacher_version': self.previous},
            'training_engine': {'name': 'Luban 训练引擎', 'engine': 'luban-training-ce-release-v1', 'teacher_api_calls': 0},
            'dtype': 'torch.bfloat16', 'optimizer_dtype': 'torch.float32', 'trained_parameters': 12582912,
            'added_parameters': 0, 'parameter': parameter, 'replay_subspace': {'enabled': True},
            'training_selection': {'selected_step': 128}, 'started': time.time() - 20, 'elapsed_seconds': 5,
            'baseline_new_loss': 2, 'validation_new_loss': 1, 'baseline_replay_loss': 1, 'validation_replay_loss': 1,
            'dataset_sha256': content_sha(groups), 'checkpoint': metadata,
            'split_ids': {key: [row['id'] for row in rows] for key, rows in groups.items()},
            'sample_sha256': {row['id']: content_sha(row) for rows in groups.values() for row in rows},
            'retention': {'file_sha256': metadata['retention_sha256'], 'content_sha256': snapshot['content_sha256'],
                          'path': str(self.store / (self.version + '.retention.json'))}}
        save(self.store / self.report_name, self.report)
        rows = {}
        wanted = groups['new_training'] + groups['new_validation'] + acceptance
        for phase in ('hf_new_first', 'hf_new_second', 'hf_old', 'native_new_first', 'native_new_second', 'native_old'):
            rows[phase] = []
            for row in protection if phase.endswith('old') else wanted:
                check = provenance(row)['generation_check']
                observed = {'id': row['id'], 'answer': row['answer'],
                    **evaluate_generation(row['answer'], check, True), 'raw_completion_token_ids': [1, 99],
                    'decoded_completion_unstripped': row['answer'], 'completion_tokens': 2,
                    'max_new_tokens': 128, 'branch_enabled': False, 'reference_used': False,
                    'thinking': 'off', 'do_sample': False}
                if phase.startswith('native'):
                    observed['finish_reason'] = 'stop'
                rows[phase].append(observed)
        self.accepted = {'protocol': release.PROTOCOL, 'plan_sha256': release.sha(self.plan_path), 'passed': True,
            'released': True, 'production_written': False, 'adopted': False, 'teacher_api_calls': 0,
            'source_core': self.previous, 'candidate_version': self.version, 'training': self.report,
            'completed_steps': 128, 'candidate_tensor_sha256': self.raw_sha,
            'native_loaded': {'core_loaded': True, 'base_identity': self.identity,
                             'core_tensor_sha256': self.raw_sha, 'core_file_sha256': metadata['sha256']},
            'rows': rows, 'artifacts': {path.name: release.sha(path) for path in self.store.iterdir()}}
        self.acceptance_path = self.output / 'acceptance.json'
        save(self.acceptance_path, self.accepted)
        self.args = SimpleNamespace(plan=self.plan_path, acceptance=self.acceptance_path, native_proof=self.acceptance_path,
                                    target=self.root / 'luban0-v2', output_plan=self.root / 'release-plan.json')

    def prepare(self):
        return release.prepare(self.args)

    def test_cpu_cli_help_no_torch_or_network(self):
        code = "import runpy,sys;sys.path.insert(0," + repr(str(ROOT / 'scripts')) + ");sys.argv=['release','--help'];\ntry:runpy.run_path(" + repr(str(Path(release.__file__))) + ",run_name='__main__')\nexcept SystemExit:pass\nassert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_preserves_C_and_all13_base_files_reseals_only_F_links(self):
        before = {path.name: path.read_bytes() for path in self.store.iterdir()}
        frozen = self.prepare()
        result = release.build(self.args.output_plan, frozen['sha256'])
        self.assertTrue(result['passed'])
        self.assertFalse(result['uploaded'])
        self.assertFalse(result['production_written'])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.iterdir()})
        for name, expected in self.trusted.items():
            self.assertEqual(release._record(self.args.target / name), expected)
        manifest = release.read(self.args.target / 'luban-foundation.json')
        foundation = self.args.target / 'luban-foundation'
        self.assertEqual(len(list(foundation.iterdir())), 4)
        self.assertEqual(set(manifest['files']), {path.name for path in foundation.iterdir()})
        report = release.read(foundation / manifest['report'])
        self.assertNotEqual(report['version'], self.version)
        self.assertEqual(report['previous_version'], self.previous)
        self.assertEqual(report['source_training_version'], self.version)
        self.assertEqual(report['source_quality_sha256'], release.sha(self.store / self.report_name))
        self.assertEqual(report['native_identity_sha256'], release.sha(self.acceptance_path))
        self.assertEqual(report['source_protocol_sha256'], release.sha(self.formal))
        changed = {'id', 'version', 'checkpoint', 'retention', 'native_identity_verified', 'native_identity_sha256',
                   'source_quality_sha256', 'source_protocol_sha256', 'source_training_version', 'source_previous_version'}
        self.assertEqual({key: value for key, value in report.items() if key not in changed},
                         {key: value for key, value in self.report.items() if key not in changed})
        self.assertEqual((foundation / (report['version'] + '.safetensors')).read_bytes(), before[self.version + '.safetensors'])
        snapshot = release.read(foundation / (report['version'] + '.retention.json'))
        old = release.read(self.store / (self.version + '.retention.json'))
        self.assertEqual({key: value for key, value in snapshot.items() if key not in {'version', 'content_sha256'}},
                         {key: value for key, value in old.items() if key not in {'version', 'content_sha256'}})
        for file in foundation.iterdir():
            self.assertEqual(file.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.args.target.stat().st_mode & 0o777, 0o700)
        self.assertFalse((foundation / 'old.json').exists())
        # Feed the actual new output into the existing first-install boundary.
        # Only model readback and durable session are stubs; all file/SHA checks are real.
        from tests.test_foundation import MemoryStub, SessionStub
        from luban.foundation import install_foundation
        destination = self.root / 'fresh-device/modules/luban0-v1'
        destination.mkdir(parents=True)
        runtime = SimpleNamespace(loaded_path=str(self.args.target), store=destination, identity=self.identity,
                                  memory=MemoryStub(), research=False)
        checked = []
        def read_version(runtime, version):
            checked.append(version)
            return release.read(runtime.store / (version + '.json')), None
        with patch.dict('sys.modules', {
                'luban.training_checkpoint': SimpleNamespace(TrainingSession=SessionStub,
                    training_runs=lambda memory, identity: memory.runs),
                'luban.evolution': SimpleNamespace(_read_version=read_version)}):
            pointer = install_foundation(runtime)
            self.assertEqual(install_foundation(runtime), pointer)
        self.assertEqual(len(runtime.memory.runs), 1)
        installed = runtime.memory.runs[0]['commit']['report']
        self.assertIs(installed['foundation_install'], True)
        self.assertEqual(installed['training_updates_executed'], 0)
        self.assertEqual(runtime.memory.runs[0]['arguments']['distillation_weight'], 5)
        self.assertEqual(checked, [manifest['version'], manifest['version']])
        journal, events = release.receipt(report, snapshot, manifest['files'])
        journal[report['id']]['body']['commit']['report'] = installed
        events['training:' + report['id']]['body'] = installed
        self.assertEqual(release._core_records(self.root / 'fresh-device', journal, events), [])
        with self.assertRaises(ValueError): release.build(self.args.output_plan, frozen['sha256'])

    def test_unapproved_or_changed_acceptance_cannot_build(self):
        frozen = self.prepare()
        with self.assertRaises(ValueError): release.build(self.args.output_plan, '0' * 64)
        self.accepted['passed'] = False
        save(self.acceptance_path, self.accepted)
        with self.assertRaises(ValueError): release.build(self.args.output_plan, frozen['sha256'])
        self.assertFalse(self.args.target.exists())

    def test_incomplete_false_or_nonEOS_closedbook_rows_rejected(self):
        original = copy.deepcopy(self.accepted)
        for change in ({'passed': False}, {'raw_completion_token_ids': [1, 2]}, {'raw_completion_token_ids': [True, 99]},
                       {'reference_used': True}, {'answer': 'wrong answer'}, {'finish_reason': 'length'}):
            self.accepted = copy.deepcopy(original)
            self.accepted['rows']['native_new_first'][0].update(change)
            save(self.acceptance_path, self.accepted)
            with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.args.target.exists())

    def test_missing_old_protection_or_native_loaded_sha_rejected(self):
        self.accepted['rows']['hf_old'].pop()
        save(self.acceptance_path, self.accepted)
        with self.assertRaises(ValueError): self.prepare()
        self.accepted['native_loaded']['core_tensor_sha256'] = 'f' * 64
        save(self.acceptance_path, self.accepted)
        with self.assertRaises(ValueError): self.prepare()

    def test_independent_native_proof_must_match_raw_rows(self):
        native = copy.deepcopy(self.accepted)
        native['rows']['native_old'][0]['answer'] = 'changed raw answer'
        path = self.output / 'native-proof.json'
        save(path, native)
        self.args.native_proof = path
        with self.assertRaises(ValueError): self.prepare()

    def test_source_C_changed_and_untrusted_upstream_rejected(self):
        path = self.store / self.report_name
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with self.assertRaises(ValueError): self.prepare()
        path.write_bytes(original)
        (self.model / 'README.md').write_text('tampered')
        with self.assertRaises(ValueError): self.prepare()

    def test_failed_full_loss_gate_leaves_no_published_target(self):
        self.report['validation_new_loss'] = 3
        save(self.store / self.report_name, self.report)
        self.accepted['training'] = self.report
        self.accepted['artifacts'][self.report_name] = release.sha(self.store / self.report_name)
        save(self.acceptance_path, self.accepted)
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.args.target.exists())
        self.assertEqual(list(self.root.glob('.goal-foundation-*')), [])

    def test_existing_or_symlink_target_never_overwritten(self):
        self.args.target.mkdir()
        marker = self.args.target / 'keep'
        marker.write_text('must keep')
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(marker.read_text(), 'must keep')
        marker.unlink(); self.args.target.rmdir()
        self.args.target.symlink_to(self.model, target_is_directory=True)
        with self.assertRaises(ValueError): self.prepare()

    def test_copy_failure_cleans_staging_and_never_writes_manifest(self):
        frozen = self.prepare()
        with patch.object(release, 'copy_new', side_effect=OSError('injected disk failure')):
            with self.assertRaises(OSError): release.build(self.args.output_plan, frozen['sha256'])
        self.assertFalse(self.args.target.exists())
        self.assertEqual(list(self.root.glob('.goal-foundation-*')), [])

    def test_tensor_nonfinite_or_wrong_header_rejected_without_torch(self):
        path = self.store / (self.version + '.safetensors')
        metadata = release.read(self.store / (self.version + '.json'))
        with path.open('r+b') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            stream.seek(8 + length)
            stream.write(struct.pack('<H', 0x7f80))
        with self.assertRaises(ValueError): release.tensor_sha(path, metadata)
        header = json.dumps({metadata['parameter']: {'dtype': 'F32', 'shape': [2048, 6144],
                                                    'data_offsets': [0, 25165824]}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + bytes(25165824))
        with self.assertRaises(ValueError): release.tensor_sha(path, metadata)

    def test_final_manifest_move_failure_is_incomplete_and_cannot_overwrite_on_retry(self):
        frozen = self.prepare()
        rename = Path.rename
        def failing_rename(source, target):
            if Path(target) == self.args.target / 'luban-foundation.json':
                raise OSError('injected final manifest move failure')
            return rename(source, target)
        with patch.object(Path, 'rename', failing_rename):
            with self.assertRaises(OSError): release.build(self.args.output_plan, frozen['sha256'])
        self.assertTrue(self.args.target.is_dir())
        self.assertFalse((self.args.target / 'luban-foundation.json').exists())
        with self.assertRaises(ValueError): release.build(self.args.output_plan, frozen['sha256'])

    def test_symlink_upstream_and_unmounted_target_rejected(self):
        file = self.model / 'LICENSE'
        moved = self.root / 'original-LICENSE'
        file.rename(moved)
        file.symlink_to(moved)
        with self.assertRaises(ValueError): release.check_base(self.model, self.identity)
        with patch.object(Path, 'is_dir', return_value=True), patch.object(Path, 'is_mount', return_value=False):
            with self.assertRaises(ValueError): release.mounted('/Volumes/unmounted-fixture/models')


if __name__ == '__main__':
    unittest.main()
